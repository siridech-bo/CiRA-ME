"""
CiRA ME - Data Loader Service
Handles loading and parsing of CSV, Edge Impulse JSON, Edge Impulse CBOR, and CiRA CBOR formats
"""

import os
import json
import uuid
import time
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Global session storage for loaded data
# NOTE: This is in-process memory — the WSGI server MUST run with a single
# worker process (threads are fine) so all requests share this dict.
_data_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()

# Session limits
_SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours
_MAX_SESSIONS = 50


class DataLoader:
    """Service for loading data from various formats."""

    def __init__(self):
        self.supported_formats = ['csv', 'json', 'cbor']

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for tracking loaded data."""
        return str(uuid.uuid4())

    def _store_session(self, session_id: str, data: pd.DataFrame, metadata: Dict) -> None:
        """Store loaded data in session."""
        # Compute min sample length for windowing guidance
        if 'sample_id' in data.columns:
            metadata['min_sample_length'] = int(data.groupby('sample_id').size().min())
        else:
            metadata['min_sample_length'] = len(data)

        with _sessions_lock:
            # Lazy cleanup: evict expired sessions
            self._evict_expired()

            _data_sessions[session_id] = {
                'data': data,
                'metadata': metadata,
                'created_at': time.monotonic(),
            }

            # Cap enforcement: if still over limit, evict oldest
            if len(_data_sessions) > _MAX_SESSIONS:
                oldest_id = min(
                    _data_sessions,
                    key=lambda k: _data_sessions[k].get('created_at', 0),
                )
                if oldest_id != session_id:
                    del _data_sessions[oldest_id]

    def _get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data. Returns None for expired or missing sessions."""
        with _sessions_lock:
            session = _data_sessions.get(session_id)
            if session is None:
                return None
            if time.monotonic() - session.get('created_at', 0) > _SESSION_TTL_SECONDS:
                del _data_sessions[session_id]
                return None
            return session

    # Patterns to detect time/timestamp columns (case-insensitive)
    TIME_COLUMN_PATTERNS = [
        'timestamp', 'time', 'time (s)', 'time(s)', 'time_s',
        'time (ms)', 'time(ms)', 'time_ms', 'elapsed', 'elapsed_time',
        't (s)', 't(s)', 't_s', 'datetime', 'date_time'
    ]

    def _detect_time_column(self, columns: List[str]) -> Optional[str]:
        """Detect time/timestamp column by matching common naming patterns."""
        columns_lower = {col.lower().strip(): col for col in columns}

        # Exact match first
        for pattern in self.TIME_COLUMN_PATTERNS:
            if pattern in columns_lower:
                return columns_lower[pattern]

        # Prefix match (e.g. "time" matches "Time (seconds)")
        for col_lower, col_original in columns_lower.items():
            if col_lower.startswith('time') or col_lower.startswith('timestamp'):
                return col_original

        return None

    def _detect_label_column(self, columns: List[str]) -> Optional[str]:
        """Detect label column by matching common naming patterns."""
        columns_lower = {col.lower().strip(): col for col in columns}
        label_patterns = ['label', 'labels', 'class', 'class_name', 'target', 'category']

        for pattern in label_patterns:
            if pattern in columns_lower:
                return columns_lower[pattern]

        return None

    def load_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a CSV file.

        Expected format:
        - Headers in first row
        - Numeric sensor columns
        - Optional label column (e.g. 'label', 'class', 'target')
        - Optional time column (e.g. 'timestamp', 'Time (s)', 'time', 'elapsed')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # Identify columns
        columns = df.columns.tolist()
        label_col = self._detect_label_column(columns)
        timestamp_col = self._detect_time_column(columns)

        # Convention: first column is always time if not detected by name
        if not timestamp_col and len(columns) > 0 and pd.api.types.is_numeric_dtype(df[columns[0]]):
            timestamp_col = columns[0]

        # Columns to exclude from sensor data
        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if timestamp_col:
            exclude_cols.add(timestamp_col)

        # Get sensor columns (numeric, excluding label and time)
        sensor_cols = [
            col for col in columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        # Generate session ID and store data
        session_id = self._generate_session_id()
        metadata = {
            'format': 'csv',
            'file_path': file_path,
            'total_rows': len(df),
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': label_col,
            'timestamp_column': timestamp_col,
            'labels': df[label_col].unique().tolist() if label_col else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def load_csv_multiple(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Load data from multiple CSV files as one dataset.

        All files must have identical columns. Each file becomes a separate
        sample_id so windowing respects file boundaries.
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        if len(file_paths) == 1:
            return self.load_csv(file_paths[0])

        # Validate all files exist
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"File not found: {fp}")

        # Read headers from all files and check column compatibility
        reference_cols = None
        reference_file = None
        for fp in file_paths:
            cols = pd.read_csv(fp, nrows=0).columns.tolist()
            if reference_cols is None:
                reference_cols = cols
                reference_file = os.path.basename(fp)
            elif cols != reference_cols:
                raise ValueError(
                    f"Column mismatch: '{os.path.basename(fp)}' has columns {cols}, "
                    f"but '{reference_file}' has columns {reference_cols}"
                )

        # Load and concatenate all files with sample_id
        dfs = []
        for idx, fp in enumerate(file_paths):
            df = pd.read_csv(fp)
            df['sample_id'] = idx
            df['source_file'] = os.path.basename(fp)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Detect columns (identical across files, so use once)
        columns = reference_cols
        label_col = self._detect_label_column(columns)
        timestamp_col = self._detect_time_column(columns)

        if not timestamp_col and len(columns) > 0 and pd.api.types.is_numeric_dtype(combined[columns[0]]):
            timestamp_col = columns[0]

        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if timestamp_col:
            exclude_cols.add(timestamp_col)

        sensor_cols = [
            col for col in columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(combined[col])
        ]

        # Store session
        session_id = self._generate_session_id()
        all_columns = combined.columns.tolist()
        metadata = {
            'format': 'csv',
            'file_path': os.path.dirname(file_paths[0]),
            'file_paths': file_paths,
            'is_multi_csv': True,
            'total_rows': len(combined),
            'total_samples': len(file_paths),
            'columns': all_columns,
            'sensor_columns': sensor_cols,
            'label_column': label_col,
            'timestamp_column': timestamp_col,
            'labels': combined[label_col].unique().tolist() if label_col else None,
            'source_files': [os.path.basename(fp) for fp in file_paths]
        }

        self._store_session(session_id, combined, metadata)

        # Stratified preview: sample from each file
        preview_dfs = []
        rows_per_file = max(10 // len(file_paths), 2)
        for idx in range(len(file_paths)):
            file_df = combined[combined['sample_id'] == idx]
            preview_dfs.append(file_df.head(rows_per_file))
        preview = pd.concat(preview_dfs, ignore_index=True)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': preview.to_dict(orient='records')
        }

    def load_edge_impulse_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from Edge Impulse JSON format.

        Expected format:
        {
            "protected": {...},
            "payload": {
                "device_name": "...",
                "device_type": "...",
                "interval_ms": 10,
                "sensors": [{"name": "accX", "units": "m/s2"}, ...],
                "values": [[ax1, ay1, az1], [ax2, ay2, az2], ...]
            }
        }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle both single sample and multi-sample formats
        samples = []

        if 'payload' in data:
            # Single sample format
            samples.append(data)
        elif isinstance(data, list):
            # Multi-sample format
            samples = data
        else:
            raise ValueError("Unsupported Edge Impulse JSON structure")

        # Convert to DataFrame
        all_rows = []
        for sample in samples:
            payload = sample.get('payload', sample)
            sensors = payload.get('sensors', [])
            values = payload.get('values', [])
            interval_ms = payload.get('interval_ms', 1)
            label = sample.get('label', payload.get('label', 'unknown'))

            sensor_names = [s['name'] for s in sensors]

            for i, row in enumerate(values):
                row_dict = {
                    'timestamp': i * interval_ms / 1000.0,
                    'label': label
                }
                for j, sensor_name in enumerate(sensor_names):
                    if j < len(row):
                        row_dict[sensor_name] = row[j]
                all_rows.append(row_dict)

        df = pd.DataFrame(all_rows)

        # Generate session ID and store data
        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = [col for col in columns if col not in ['label', 'timestamp']]

        metadata = {
            'format': 'edge_impulse_json',
            'file_path': file_path,
            'total_rows': len(df),
            'total_samples': len(samples),
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': df['label'].unique().tolist() if 'label' in df.columns else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def load_edge_impulse_cbor(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from Edge Impulse CBOR format.
        Supports both single file and folder-based loading.
        """
        # Check if it's a folder (dataset root)
        if os.path.isdir(file_path):
            return self.load_edge_impulse_cbor_folder(file_path)

        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 library required. Install with: pip install cbor2")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = cbor2.load(f)

        # Process same as JSON format
        samples = []

        if 'payload' in data:
            samples.append(data)
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("Unsupported Edge Impulse CBOR structure")

        all_rows = []
        for sample in samples:
            payload = sample.get('payload', sample)
            sensors = payload.get('sensors', [])
            values = payload.get('values', [])
            interval_ms = payload.get('interval_ms', 1)
            label = sample.get('label', payload.get('label', 'unknown'))

            sensor_names = [s['name'] for s in sensors]

            for i, row in enumerate(values):
                row_dict = {
                    'timestamp': i * interval_ms / 1000.0,
                    'label': label
                }
                for j, sensor_name in enumerate(sensor_names):
                    if j < len(row):
                        row_dict[sensor_name] = row[j]
                all_rows.append(row_dict)

        df = pd.DataFrame(all_rows)

        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = [col for col in columns if col not in ['label', 'timestamp']]

        metadata = {
            'format': 'edge_impulse_cbor',
            'file_path': file_path,
            'total_rows': len(df),
            'total_samples': len(samples),
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': df['label'].unique().tolist() if 'label' in df.columns else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def load_edge_impulse_cbor_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Load entire Edge Impulse CBOR dataset from a folder.

        Expected structure:
        - info.labels (JSON file with file metadata and labels)
        - training/ (folder with .cbor files)
        - testing/ (folder with .cbor files)
        """
        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 library required. Install with: pip install cbor2")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        # Try to load info.labels for label mapping
        label_map = {}  # filename -> label
        label_map_by_path = {}  # relative path (category/filename) -> label
        info_labels_labels = set()  # Track labels from info.labels
        info_labels_path = os.path.join(folder_path, 'info.labels')
        has_info_labels = os.path.exists(info_labels_path)
        if has_info_labels:
            with open(info_labels_path, 'r') as f:
                info_data = json.load(f)
                for file_info in info_data.get('files', []):
                    file_path_rel = file_info.get('path', '')
                    label_info = file_info.get('label', {})
                    if isinstance(label_info, dict):
                        label = label_info.get('label', 'unknown')
                    else:
                        label = str(label_info)
                    info_labels_labels.add(label)
                    # Store by both full relative path and filename for flexible matching
                    # Normalize path separators for consistent matching
                    normalized_path = file_path_rel.replace('\\', '/')
                    label_map_by_path[normalized_path] = label
                    filename = os.path.basename(file_path_rel)
                    label_map[filename] = label

        # Find all CBOR files in training and testing folders
        all_rows = []
        total_samples = 0
        sensor_names = None
        categories = {'training': 0, 'testing': 0}
        label_match_stats = {'by_path': 0, 'by_filename': 0, 'by_prefix': 0}
        prefix_labels = set()  # Track labels that came from filename prefix

        for category in ['training', 'testing']:
            category_path = os.path.join(folder_path, category)
            if not os.path.exists(category_path):
                continue

            for filename in os.listdir(category_path):
                if not filename.endswith('.cbor'):
                    continue

                cbor_file_path = os.path.join(category_path, filename)

                # Determine label - try multiple lookup strategies
                # 1. Try full relative path (category/filename)
                rel_path = f"{category}/{filename}"
                if rel_path in label_map_by_path:
                    label = label_map_by_path[rel_path]
                    label_match_stats['by_path'] += 1
                # 2. Try just filename
                elif filename in label_map:
                    label = label_map[filename]
                    label_match_stats['by_filename'] += 1
                # 3. Extract from filename prefix (e.g., "idle.1.cbor" -> "idle")
                else:
                    label = filename.split('.')[0]
                    label_match_stats['by_prefix'] += 1
                    prefix_labels.add(label)

                try:
                    with open(cbor_file_path, 'rb') as f:
                        data = cbor2.load(f)

                    payload = data.get('payload', data)
                    sensors = payload.get('sensors', [])
                    values = payload.get('values', [])
                    interval_ms = payload.get('interval_ms', 1)

                    if sensor_names is None and sensors:
                        sensor_names = [s['name'] for s in sensors]

                    for i, row in enumerate(values):
                        row_dict = {
                            'timestamp': i * interval_ms / 1000.0,
                            'label': label,
                            'sample_id': total_samples,
                            'category': category
                        }
                        for j, sensor_name in enumerate(sensor_names or []):
                            if j < len(row):
                                row_dict[sensor_name] = row[j]
                        all_rows.append(row_dict)

                    total_samples += 1
                    categories[category] += 1

                except Exception as e:
                    print(f"Warning: Failed to load {cbor_file_path}: {e}")
                    continue

        if not all_rows:
            raise ValueError("No valid CBOR files found in the dataset folder")

        df = pd.DataFrame(all_rows)

        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = sensor_names or [col for col in columns if col not in ['label', 'timestamp', 'sample_id', 'category']]

        metadata = {
            'format': 'edge_impulse_cbor',
            'file_path': folder_path,
            'is_folder': True,
            'total_rows': len(df),
            'total_samples': total_samples,
            'training_samples': categories['training'],
            'testing_samples': categories['testing'],
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': df['label'].unique().tolist(),
            # Diagnostic info for label detection
            'label_debug': {
                'has_info_labels': has_info_labels,
                'info_labels_classes': list(info_labels_labels) if info_labels_labels else [],
                'match_stats': label_match_stats,
                'prefix_detected_labels': list(prefix_labels)
            }
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def scan_dataset_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Scan a dataset folder to get its structure without loading any CBOR data.
        Reads info.labels if available, otherwise scans filenames.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        info_labels_path = os.path.join(folder_path, 'info.labels')
        has_info_labels = os.path.exists(info_labels_path)

        categories = {}
        all_labels = set()

        if has_info_labels:
            # Edge Impulse format: parse info.labels
            with open(info_labels_path, 'r') as f:
                info_data = json.load(f)

            for file_info in info_data.get('files', []):
                category = file_info.get('category', 'unknown')
                label_info = file_info.get('label', {})
                if isinstance(label_info, dict):
                    label = label_info.get('label', 'unknown')
                else:
                    label = str(label_info)

                if category not in categories:
                    categories[category] = {'file_count': 0, 'labels': {}}
                categories[category]['file_count'] += 1

                if label not in categories[category]['labels']:
                    categories[category]['labels'][label] = {'file_count': 0}
                categories[category]['labels'][label]['file_count'] += 1
                all_labels.add(label)

            detected_format = 'edge_impulse_cbor'
        else:
            # CiRA or Edge Impulse without info.labels: scan folder structure
            dataset_path = os.path.join(folder_path, 'dataset')
            scan_root = dataset_path if os.path.exists(dataset_path) else folder_path

            detected_format = 'cira_cbor' if os.path.exists(dataset_path) else 'edge_impulse_cbor'

            for cat_name in ['training', 'testing', 'train', 'test']:
                cat_path = os.path.join(scan_root, cat_name)
                if not os.path.exists(cat_path):
                    continue

                categories[cat_name] = {'file_count': 0, 'labels': {}}

                for filename in os.listdir(cat_path):
                    if not filename.endswith('.cbor'):
                        continue

                    label = filename.split('.')[0]
                    categories[cat_name]['file_count'] += 1

                    if label not in categories[cat_name]['labels']:
                        categories[cat_name]['labels'][label] = {'file_count': 0}
                    categories[cat_name]['labels'][label]['file_count'] += 1
                    all_labels.add(label)

        total_files = sum(c['file_count'] for c in categories.values())

        return {
            'folder_path': folder_path,
            'format': detected_format,
            'has_info_labels': has_info_labels,
            'categories': categories,
            'all_labels': sorted(list(all_labels)),
            'total_files': total_files
        }

    def preview_partition(
        self,
        folder_path: str,
        category: str = None,
        label: str = None,
        rows: int = 100,
        format_hint: str = None
    ) -> Dict[str, Any]:
        """
        Load and preview only a filtered partition of a dataset folder.
        Only loads CBOR files matching the category/label filter.
        """
        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 library required. Install with: pip install cbor2")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        # Build label map from info.labels if available
        label_map = {}
        label_map_by_path = {}
        info_labels_path = os.path.join(folder_path, 'info.labels')
        has_info_labels = os.path.exists(info_labels_path)

        if has_info_labels:
            with open(info_labels_path, 'r') as f:
                info_data = json.load(f)
                for file_info in info_data.get('files', []):
                    file_path_rel = file_info.get('path', '')
                    label_info = file_info.get('label', {})
                    if isinstance(label_info, dict):
                        file_label = label_info.get('label', 'unknown')
                    else:
                        file_label = str(label_info)
                    normalized_path = file_path_rel.replace('\\', '/')
                    label_map_by_path[normalized_path] = file_label
                    filename = os.path.basename(file_path_rel)
                    label_map[filename] = file_label

        # Detect format
        info_labels_exists = os.path.exists(info_labels_path)
        training_folder = os.path.join(folder_path, 'training')
        dataset_folder = os.path.join(folder_path, 'dataset')
        train_folder = os.path.join(folder_path, 'train')

        is_ei = format_hint == 'ei-cbor' or info_labels_exists or os.path.exists(training_folder)
        is_cira = format_hint == 'cira-cbor' or os.path.exists(dataset_folder) or os.path.exists(train_folder)

        # Determine which categories to scan
        if is_ei:
            all_categories = ['training', 'testing']
            scan_root = folder_path
            detected_format = 'edge_impulse_cbor'
        elif is_cira:
            all_categories = ['train', 'test', 'training', 'testing']
            scan_root = dataset_folder if os.path.exists(dataset_folder) else folder_path
            detected_format = 'cira_cbor'
        else:
            all_categories = ['training', 'testing', 'train', 'test']
            scan_root = folder_path
            detected_format = 'edge_impulse_cbor'

        # Filter categories
        if category:
            categories_to_scan = [category]
        else:
            categories_to_scan = all_categories

        all_rows = []
        total_samples = 0
        sensor_names = None
        sampling_rate = None

        for cat in categories_to_scan:
            cat_path = os.path.join(scan_root, cat)
            if not os.path.exists(cat_path):
                continue

            for filename in os.listdir(cat_path):
                if not filename.endswith('.cbor'):
                    continue

                # Determine label for this file
                rel_path = f"{cat}/{filename}"
                if rel_path in label_map_by_path:
                    file_label = label_map_by_path[rel_path]
                elif filename in label_map:
                    file_label = label_map[filename]
                else:
                    file_label = filename.split('.')[0]

                # Apply label filter
                if label and file_label != label:
                    continue

                cbor_file_path = os.path.join(cat_path, filename)

                try:
                    with open(cbor_file_path, 'rb') as f:
                        data = cbor2.load(f)

                    if 'samples' in data:
                        # CiRA sample-based format
                        parsed_rows, n_samples = self._parse_cira_samples(
                            data, file_label,
                            sample_id_offset=total_samples,
                            category=cat
                        )
                        all_rows.extend(parsed_rows)
                        total_samples += n_samples
                        if sensor_names is None and rows:
                            sensor_names = ['value']
                    elif 'payload' in data or 'sensors' in data or 'values' in data:
                        # Edge Impulse format
                        payload = data.get('payload', data)
                        sensors = payload.get('sensors', [])
                        values = payload.get('values', [])
                        interval_ms = payload.get('interval_ms', 1)
                        if sensor_names is None and sensors:
                            sensor_names = [s['name'] for s in sensors]
                        for i, row in enumerate(values):
                            row_dict = {
                                'timestamp': i * interval_ms / 1000.0,
                                'label': file_label,
                                'sample_id': total_samples,
                                'category': cat
                            }
                            for j, sn in enumerate(sensor_names or []):
                                if j < len(row):
                                    row_dict[sn] = row[j]
                            all_rows.append(row_dict)
                        total_samples += 1
                    else:
                        # Legacy CiRA format: {'metadata': {...}, 'data': {'ch': [...]}}
                        meta = data.get('metadata', {})
                        raw_data = data.get('data', {})
                        if sensor_names is None:
                            sensor_names = meta.get('channels', list(raw_data.keys()))
                            sampling_rate = meta.get('sampling_rate', 100)
                        if meta.get('label'):
                            file_label = meta['label']
                        max_len = max(len(raw_data.get(ch, [])) for ch in sensor_names) if sensor_names else 0
                        for i in range(max_len):
                            row_dict = {
                                'timestamp': i / (sampling_rate or 100),
                                'label': file_label,
                                'sample_id': total_samples,
                                'category': cat
                            }
                            for ch in sensor_names:
                                ch_data = raw_data.get(ch, [])
                                row_dict[ch] = ch_data[i] if i < len(ch_data) else np.nan
                            all_rows.append(row_dict)
                        total_samples += 1
                except Exception as e:
                    print(f"Warning: Failed to load {cbor_file_path}: {e}")
                    continue

        if not all_rows:
            raise ValueError(f"No data found for filter: category={category}, label={label}")

        df = pd.DataFrame(all_rows)

        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = sensor_names or [col for col in columns if col not in ['label', 'timestamp', 'sample_id', 'category']]

        metadata = {
            'format': detected_format,
            'file_path': folder_path,
            'is_folder': True,
            'is_partition_preview': True,
            'filter': {'category': category, 'label': label},
            'total_rows': len(df),
            'total_samples': total_samples,
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': df['label'].unique().tolist()
        }

        self._store_session(session_id, df, metadata)

        # Sort and return preview
        sort_cols = []
        if 'sample_id' in df.columns:
            sort_cols.append('sample_id')
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.iloc[:rows].to_dict(orient='records')
        }

    def load_full_dataset(self, folder_path: str, format_hint: str = None, preview_session_id: str = None) -> Dict[str, Any]:
        """
        Load the complete dataset into a session for windowing.
        Cleans up preview session if provided.
        """
        if preview_session_id:
            self.cleanup_session(preview_session_id)

        # Detect format and delegate to existing loaders
        info_labels = os.path.join(folder_path, 'info.labels')
        dataset_folder = os.path.join(folder_path, 'dataset')
        training_folder = os.path.join(folder_path, 'training')
        train_folder = os.path.join(folder_path, 'train')

        if format_hint == 'ei-cbor' or os.path.exists(info_labels) or os.path.exists(training_folder):
            result = self.load_edge_impulse_cbor_folder(folder_path)
        elif format_hint == 'cira-cbor' or os.path.exists(dataset_folder) or os.path.exists(train_folder):
            result = self.load_cira_cbor_folder(folder_path)
        else:
            try:
                result = self.load_edge_impulse_cbor_folder(folder_path)
            except (ValueError, KeyError):
                result = self.load_cira_cbor_folder(folder_path)

        # Return session_id and metadata only (no preview array to save bandwidth)
        return {
            'session_id': result['session_id'],
            'metadata': result['metadata']
        }

    def cleanup_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        with _sessions_lock:
            _data_sessions.pop(session_id, None)

    @staticmethod
    def _evict_expired() -> None:
        """Remove all expired sessions. Caller must hold _sessions_lock."""
        now = time.monotonic()
        expired = [
            sid for sid, s in _data_sessions.items()
            if now - s.get('created_at', 0) > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del _data_sessions[sid]

    def _parse_cira_samples(self, cbor_data: Dict, file_label: str,
                             sample_id_offset: int = 0,
                             category: str = None) -> tuple:
        """
        Parse CiRA CBOR sample-based format.
        Returns (rows_list, num_samples).
        """
        samples = cbor_data.get('samples', [])
        if not samples:
            return [], 0

        rows = []
        for idx, sample in enumerate(samples):
            sample_data = sample.get('data', [])
            label = sample.get('class_name', file_label)
            sid = sample_id_offset + idx

            for i, val in enumerate(sample_data):
                row = {
                    'timestamp': i,
                    'value': val,
                    'label': label,
                    'sample_id': sid
                }
                if category is not None:
                    row['category'] = category
                rows.append(row)

        return rows, len(samples)

    def load_cira_cbor(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from CiRA CBOR format.
        Supports both single file and folder-based loading.

        CiRA CBOR format: {'samples': [{'class_id', 'class_name', 'data': [...], 'timestamp'}, ...]}
        """
        # Check if it's a folder (dataset root)
        if os.path.isdir(file_path):
            return self.load_cira_cbor_folder(file_path)

        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 library required. Install with: pip install cbor2")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = cbor2.load(f)

        if 'samples' in data:
            # CiRA CBOR format: {'samples': [{'class_id', 'class_name', 'data', 'timestamp'}, ...]}
            label = os.path.splitext(os.path.basename(file_path))[0].split('.')[0]
            parsed_rows, num_samples = self._parse_cira_samples(data, label)

            if not parsed_rows:
                raise ValueError("No sample data found in CiRA CBOR file")

            df = pd.DataFrame(parsed_rows)
            sensor_cols = ['value']
            labels = df['label'].unique().tolist()
        else:
            # Legacy format: {'metadata': {...}, 'data': {'ch1': [...], ...}}
            meta = data.get('metadata', {})
            raw_data = data.get('data', {})

            sampling_rate = meta.get('sampling_rate', 100)
            channels = meta.get('channels', list(raw_data.keys()))
            label = meta.get('label', 'unknown')

            max_len = max(len(raw_data.get(ch, [])) for ch in channels) if channels else 0

            df_dict = {'timestamp': [i / sampling_rate for i in range(max_len)]}
            for ch in channels:
                ch_data = raw_data.get(ch, [])
                if len(ch_data) < max_len:
                    ch_data = list(ch_data) + [np.nan] * (max_len - len(ch_data))
                df_dict[ch] = ch_data
            df_dict['label'] = [label] * max_len

            df = pd.DataFrame(df_dict)
            sensor_cols = channels
            labels = [label]

        session_id = self._generate_session_id()
        columns = df.columns.tolist()

        metadata = {
            'format': 'cira_cbor',
            'file_path': file_path,
            'total_rows': len(df),
            'total_samples': num_samples if 'samples' in data else 1,
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': labels,
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def load_cira_cbor_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Load entire CiRA CBOR dataset from a folder.

        Expected structure:
        - dataset/train/ (folder with .cbor files)
        - dataset/test/ (folder with .cbor files)
        OR
        - train/ (folder with .cbor files)
        - test/ (folder with .cbor files)

        CiRA CBOR format: {'samples': [{'class_id', 'class_name', 'data': [...], 'timestamp'}, ...]}
        """
        try:
            import cbor2
        except ImportError:
            raise ImportError("cbor2 library required. Install with: pip install cbor2")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        # Check for dataset subfolder structure
        dataset_path = os.path.join(folder_path, 'dataset')
        if os.path.exists(dataset_path):
            folder_path = dataset_path

        all_rows = []
        total_samples = 0
        categories = {'train': 0, 'test': 0}

        for category in ['train', 'test', 'training', 'testing']:
            category_path = os.path.join(folder_path, category)
            if not os.path.exists(category_path):
                continue

            cat_key = 'train' if category in ['train', 'training'] else 'test'

            for filename in os.listdir(category_path):
                if not filename.endswith('.cbor'):
                    continue

                cbor_file_path = os.path.join(category_path, filename)
                file_label = filename.split('.')[0]

                try:
                    with open(cbor_file_path, 'rb') as f:
                        data = cbor2.load(f)

                    if 'samples' in data:
                        # CiRA sample-based format
                        parsed_rows, n_samples = self._parse_cira_samples(
                            data, file_label,
                            sample_id_offset=total_samples,
                            category=cat_key
                        )
                        all_rows.extend(parsed_rows)
                        total_samples += n_samples
                        categories[cat_key] += n_samples
                    else:
                        # Legacy format: {'metadata': {...}, 'data': {'ch': [...]}}
                        meta = data.get('metadata', {})
                        raw_data = data.get('data', {})
                        channels = meta.get('channels', list(raw_data.keys()))
                        sampling_rate = meta.get('sampling_rate', 100)

                        if meta.get('label'):
                            file_label = meta['label']

                        max_len = max(len(raw_data.get(ch, [])) for ch in channels) if channels else 0
                        for i in range(max_len):
                            row_dict = {
                                'timestamp': i / sampling_rate,
                                'label': file_label,
                                'sample_id': total_samples,
                                'category': cat_key
                            }
                            for ch in channels:
                                ch_data = raw_data.get(ch, [])
                                row_dict[ch] = ch_data[i] if i < len(ch_data) else np.nan
                            all_rows.append(row_dict)
                        total_samples += 1
                        categories[cat_key] += 1

                except Exception as e:
                    print(f"Warning: Failed to load {cbor_file_path}: {e}")
                    continue

        if not all_rows:
            raise ValueError("No valid CBOR files found in the dataset folder")

        df = pd.DataFrame(all_rows)

        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = [col for col in columns if col not in ['label', 'timestamp', 'sample_id', 'category']]

        metadata = {
            'format': 'cira_cbor',
            'file_path': folder_path,
            'is_folder': True,
            'total_rows': len(df),
            'total_samples': total_samples,
            'training_samples': categories['train'],
            'testing_samples': categories['test'],
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': df['label'].unique().tolist(),
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(10).to_dict(orient='records')
        }

    def preview(self, file_path: str, rows: int = 10, format_hint: str = None) -> Dict[str, Any]:
        """Preview data from any supported format."""
        # Handle folder-based datasets
        if os.path.isdir(file_path):
            return self.preview_folder(file_path, rows, format_hint)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            result = self.load_csv(file_path)
        elif ext == '.json':
            result = self.load_edge_impulse_json(file_path)
        elif ext == '.cbor':
            # Try CiRA CBOR first, then Edge Impulse CBOR
            try:
                result = self.load_cira_cbor(file_path)
            except (ValueError, KeyError):
                result = self.load_edge_impulse_cbor(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Limit preview rows
        session = self._get_session(result['session_id'])
        if session:
            result['preview'] = session['data'].head(rows).to_dict(orient='records')

        return result

    def preview_folder(self, folder_path: str, rows: int = 10, format_hint: str = None) -> Dict[str, Any]:
        """Preview data from a folder-based dataset."""
        # Detect format based on folder structure
        info_labels = os.path.join(folder_path, 'info.labels')
        dataset_folder = os.path.join(folder_path, 'dataset')
        training_folder = os.path.join(folder_path, 'training')
        train_folder = os.path.join(folder_path, 'train')

        # Determine format
        if format_hint == 'ei-cbor' or os.path.exists(info_labels) or os.path.exists(training_folder):
            # Edge Impulse CBOR format
            result = self.load_edge_impulse_cbor_folder(folder_path)
        elif format_hint == 'cira-cbor' or os.path.exists(dataset_folder) or os.path.exists(train_folder):
            # CiRA CBOR format
            result = self.load_cira_cbor_folder(folder_path)
        else:
            # Try Edge Impulse first, then CiRA
            try:
                result = self.load_edge_impulse_cbor_folder(folder_path)
            except (ValueError, KeyError):
                result = self.load_cira_cbor_folder(folder_path)

        # Create stratified preview - sample from each label AND category
        session = self._get_session(result['session_id'])
        if session:
            df = session['data']
            label_col = result['metadata'].get('label_column', 'label')

            if label_col in df.columns:
                has_category = 'category' in df.columns
                labels = df[label_col].unique()

                if has_category:
                    # Sample from each (label, category) combination for representative preview
                    categories = df['category'].unique()
                    num_groups = len(labels) * len(categories)
                    rows_per_group = max(rows // num_groups, 5)

                    preview_dfs = []
                    for label in labels:
                        for category in categories:
                            group_df = df[(df[label_col] == label) & (df['category'] == category)]
                            if not group_df.empty:
                                preview_dfs.append(group_df.head(rows_per_group))
                else:
                    # No category column - sample per label only
                    rows_per_label = max(rows // len(labels), 10)
                    preview_dfs = []
                    for label in labels:
                        label_df = df[df[label_col] == label]
                        preview_dfs.append(label_df.head(rows_per_label))

                preview_df = pd.concat(preview_dfs, ignore_index=True)
                # Sort by sample_id and timestamp for consistent ordering
                if 'sample_id' in preview_df.columns:
                    sort_cols = ['category', 'sample_id', 'timestamp'] if has_category and 'timestamp' in preview_df.columns else \
                                ['category', 'sample_id'] if has_category else \
                                ['sample_id', 'timestamp'] if 'timestamp' in preview_df.columns else ['sample_id']
                    preview_df = preview_df.sort_values(sort_cols)
                result['preview'] = preview_df.head(rows).to_dict(orient='records')
            else:
                result['preview'] = df.head(rows).to_dict(orient='records')

        return result

    def _get_window_label(self, window_labels: np.ndarray, label_method: str) -> str:
        """Determine a single label for a window from its constituent labels."""
        if label_method == 'majority':
            unique, counts = np.unique(window_labels, return_counts=True)
            return unique[np.argmax(counts)]
        elif label_method == 'first':
            return window_labels[0]
        elif label_method == 'last':
            return window_labels[-1]
        elif label_method == 'threshold':
            unique, counts = np.unique(window_labels, return_counts=True)
            max_count = np.max(counts)
            if max_count / len(window_labels) > 0.5:
                return unique[np.argmax(counts)]
            else:
                return 'ambiguous'
        return window_labels[0]

    def apply_windowing(
        self,
        session_id: str,
        window_size: int = 128,
        stride: int = 64,
        label_method: str = 'majority',
        test_ratio: float = 0.2,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Apply windowing to loaded data, respecting sample boundaries.

        Windows are created WITHIN each sample_id to prevent data leakage
        across different recordings. The 'category' column (training/testing)
        is preserved per window for proper train/test splitting.

        For CSV data without a pre-existing category column, a train/test
        split is applied automatically:
        - sample_id present: whole samples assigned to train or test
        - no sample_id: temporal split with a gap of window_size

        Args:
            session_id: Session ID from data loading
            window_size: Number of samples per window
            stride: Step size between windows
            label_method: How to assign labels ('majority', 'first', 'last', 'threshold')
            test_ratio: Fraction of data reserved for testing (0.1–0.5)

        Returns:
            Windowed data information
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        df = session['data']
        metadata = session['metadata']

        sensor_cols = list(metadata['sensor_columns'])
        label_col = metadata.get('label_column')
        has_sample_id = 'sample_id' in df.columns
        has_category = 'category' in df.columns

        # Regression mode: target_column is a sensor column to predict
        # Remove it from input features and use it to generate continuous targets
        regression_target = None
        if target_column and target_column in sensor_cols:
            regression_target = target_column
            sensor_cols = [c for c in sensor_cols if c != target_column]
            label_col = None  # Override — use target_column values, not categorical labels

        # --- Auto train/test split for CSV data without pre-existing categories ---
        split_method = None
        if has_category:
            split_method = 'preset'  # CBOR folders already have training/testing
        elif test_ratio and 0 < test_ratio < 1:
            if has_sample_id:
                sample_ids = df['sample_id'].unique()
                n_samples = len(sample_ids)
                n_test_files = max(1, round(n_samples * test_ratio))
                actual_file_ratio = n_test_files / n_samples

                # Use file-level split only when enough files to honor the ratio
                # (tolerance: actual ratio within 10% of requested ratio)
                use_file_split = (n_samples >= 4 and
                                  abs(actual_file_ratio - test_ratio) <= 0.10)

                if use_file_split:
                    # File-level split: assign whole files to train or test
                    split_method = 'sample'
                    n_train = n_samples - n_test_files

                    if label_col:
                        sample_labels = df.groupby('sample_id')[label_col].agg(
                            lambda x: x.value_counts().index[0]
                        )
                        from sklearn.model_selection import train_test_split as sk_split
                        try:
                            train_ids, test_ids = sk_split(
                                sample_ids, test_size=n_test_files,
                                random_state=42,
                                stratify=sample_labels[sample_ids].values
                            )
                        except ValueError:
                            train_ids = sample_ids[:n_train]
                            test_ids = sample_ids[n_train:]
                    else:
                        train_ids = sample_ids[:n_train]
                        test_ids = sample_ids[n_train:]

                    test_set = set(test_ids)
                    df = df.copy()
                    df['category'] = df['sample_id'].apply(
                        lambda sid: 'testing' if sid in test_set else 'training'
                    )
                    has_category = True
                else:
                    # Too few files for file-level split to honor the ratio.
                    # Use stratified window-level split: create all windows first,
                    # then assign train/test by stratified random split on labels.
                    # This ensures all classes appear in both train and test.
                    split_method = 'stratified'
                    # Don't assign categories now — will be done after windowing
            else:
                # Temporal split with gap for single CSV
                split_method = 'temporal'
                total_rows = len(df)
                gap = window_size  # gap equal to window_size prevents overlap
                test_rows = max(window_size, round(total_rows * test_ratio))
                train_rows = total_rows - test_rows - gap

                if train_rows < window_size:
                    # Not enough data for gap, skip gap
                    gap = 0
                    train_rows = total_rows - test_rows

                # Ensure both splits can produce at least 2 windows each
                min_rows_for_2_windows = window_size + stride
                if train_rows < min_rows_for_2_windows or test_rows < min_rows_for_2_windows:
                    # Reduce gap to half, redistribute
                    gap = max(0, gap // 2)
                    desired_test = max(min_rows_for_2_windows, round(total_rows * test_ratio))
                    train_rows = total_rows - desired_test - gap
                    test_rows = desired_test
                    if train_rows < min_rows_for_2_windows:
                        # Last resort: no gap, split proportionally
                        gap = 0
                        test_rows = max(min_rows_for_2_windows, round(total_rows * test_ratio))
                        train_rows = total_rows - test_rows
                    split_method = 'temporal (with gap)' if gap > 0 else 'temporal (no gap)'

                df = df.copy()
                df['category'] = 'gap'
                df.iloc[:train_rows, df.columns.get_loc('category')] = 'training'
                df.iloc[train_rows + gap:, df.columns.get_loc('category')] = 'testing'
                # Remove gap rows so no window can span train/test
                df = df[df['category'] != 'gap'].reset_index(drop=True)
                has_category = True

        windows = []
        labels = []
        categories = []

        if has_sample_id:
            # Window within each sample to prevent cross-sample leakage
            for sample_id, sample_df in df.groupby('sample_id', sort=False):
                sample_data = sample_df[sensor_cols].values
                n_rows = len(sample_data)

                if n_rows < window_size:
                    continue

                # Get category for this sample (for file-level or preset splits)
                sample_category = None
                if has_category:
                    sample_category = sample_df['category'].iloc[0]

                # Pre-extract target column data for regression
                sample_target = sample_df[regression_target].values if regression_target else None

                n_windows = (n_rows - window_size) // stride + 1

                for i in range(n_windows):
                    start = i * stride
                    end = start + window_size

                    windows.append(sample_data[start:end])

                    if regression_target:
                        # Regression: use mean of target column in window
                        labels.append(float(np.mean(sample_target[start:end])))
                    elif label_col:
                        window_labels = sample_df.iloc[start:end][label_col].values
                        labels.append(self._get_window_label(window_labels, label_method))

                    if sample_category is not None:
                        categories.append(sample_category)
        else:
            # No sample_id — window within each category to prevent leakage
            if has_category:
                for cat, cat_df in df.groupby('category', sort=False):
                    cat_data = cat_df[sensor_cols].values
                    n_rows = len(cat_data)
                    if n_rows < window_size:
                        continue

                    cat_target = cat_df[regression_target].values if regression_target else None

                    n_windows = (n_rows - window_size) // stride + 1
                    for i in range(n_windows):
                        start = i * stride
                        end = start + window_size
                        windows.append(cat_data[start:end])
                        categories.append(cat)
                        if regression_target:
                            labels.append(float(np.mean(cat_target[start:end])))
                        elif label_col:
                            window_labels = cat_df.iloc[start:end][label_col].values
                            labels.append(self._get_window_label(window_labels, label_method))
            else:
                # No category, no sample_id — plain sequential windowing
                num_samples = len(df)
                n_windows = max(0, (num_samples - window_size) // stride + 1)

                # Pre-extract target column for regression
                plain_target = df[regression_target].values if regression_target else None

                for i in range(n_windows):
                    start = i * stride
                    end = start + window_size

                    window_data = df.iloc[start:end][sensor_cols].values
                    windows.append(window_data)

                    if regression_target:
                        labels.append(float(np.mean(plain_target[start:end])))
                    elif label_col:
                        window_labels = df.iloc[start:end][label_col].values
                        labels.append(self._get_window_label(window_labels, label_method))

        num_windows = len(windows)
        if num_windows == 0:
            raise ValueError(
                f"No windows could be created. Window size ({window_size}) may be "
                f"larger than individual samples. Try a smaller window size."
            )

        # --- Post-windowing random split for regression (can't stratify continuous targets) ---
        if split_method == 'stratified' and regression_target and labels and num_windows > 1:
            from sklearn.model_selection import train_test_split as sk_split
            indices = np.arange(num_windows)
            train_idx, test_idx = sk_split(indices, test_size=test_ratio, random_state=42)
            categories = ['training'] * num_windows
            for idx in test_idx:
                categories[idx] = 'testing'
            split_method = 'random (regression)'

        # --- Post-windowing stratified split (for few-file multi-CSV) ---
        # Skip stratified split for regression (continuous labels can't be stratified)
        if split_method == 'stratified' and labels and num_windows > 1 and not regression_target:
            from sklearn.model_selection import train_test_split as sk_split
            label_arr = np.array(labels)
            indices = np.arange(num_windows)
            try:
                train_idx, test_idx = sk_split(
                    indices, test_size=test_ratio,
                    random_state=42, stratify=label_arr
                )
            except ValueError:
                # Stratification failed (too few samples per class), use random
                train_idx, test_idx = sk_split(
                    indices, test_size=test_ratio, random_state=42
                )
            categories = ['training'] * num_windows
            for idx in test_idx:
                categories[idx] = 'testing'

        # --- Validate split: ensure test set has >= 2 classes ---
        # If category-based split left test with < 2 classes, re-split with stratification
        # Skip for regression (continuous targets, not categorical classes)
        if labels and categories and num_windows > 1 and not regression_target:
            from sklearn.model_selection import train_test_split as sk_split
            label_arr = np.array(labels)
            cat_arr_check = np.array(categories)
            test_mask = cat_arr_check == 'testing'
            train_mask = cat_arr_check == 'training'

            test_classes = set(label_arr[test_mask]) if test_mask.any() else set()
            train_classes = set(label_arr[train_mask]) if train_mask.any() else set()
            all_classes = set(label_arr)

            if len(test_classes) < 2 and len(all_classes) >= 2:
                print(f"[DataLoader] Split validation: test has only {len(test_classes)} class(es) "
                      f"({test_classes}), need >= 2. Re-splitting with stratification...")
                indices = np.arange(num_windows)
                try:
                    train_idx, test_idx = sk_split(
                        indices, test_size=max(test_ratio or 0.2, 0.2),
                        random_state=42, stratify=label_arr
                    )
                    categories = ['training'] * num_windows
                    for idx in test_idx:
                        categories[idx] = 'testing'
                    split_method = 'stratified (auto-corrected)'
                    print(f"[DataLoader] Re-split: train={len(train_idx)}, test={len(test_idx)}, "
                          f"test classes={set(label_arr[test_idx])}")
                except ValueError as e:
                    # Still can't stratify — too few samples per class
                    print(f"[DataLoader] Stratified re-split failed: {e}. "
                          f"Need more data (record longer).")

        # --- Per-channel min-max normalization (0-1) ---
        # Compute stats from training data only to prevent leakage
        all_windows = np.array(windows)  # (num_windows, window_size, num_channels)
        cat_arr_temp = np.array(categories) if categories else None

        if cat_arr_temp is not None and 'training' in cat_arr_temp:
            train_mask = cat_arr_temp == 'training'
            train_windows = all_windows[train_mask]
        else:
            train_windows = all_windows

        # Flatten to (total_train_samples, num_channels) for stats
        train_flat = train_windows.reshape(-1, all_windows.shape[2])
        ch_min = train_flat.min(axis=0)
        ch_max = train_flat.max(axis=0)
        ch_range = ch_max - ch_min

        # Identify and drop constant columns (range == 0)
        active_mask = ch_range > 1e-10
        dropped_cols = [sensor_cols[i] for i in range(len(sensor_cols)) if not active_mask[i]]
        kept_cols = [sensor_cols[i] for i in range(len(sensor_cols)) if active_mask[i]]

        if dropped_cols:
            print(f"[DataLoader] Dropped {len(dropped_cols)} constant column(s): {dropped_cols}")

        # Keep only non-constant channels
        all_windows = all_windows[:, :, active_mask]
        ch_min = ch_min[active_mask]
        ch_max = ch_max[active_mask]
        ch_range = ch_range[active_mask]

        # Apply min-max normalization: (x - min) / (max - min) → [0, 1]
        all_windows = (all_windows - ch_min) / ch_range

        # Update sensor_cols in metadata
        sensor_cols = kept_cols
        windows = list(all_windows)

        # Save normalization params for deployment
        norm_params = {
            'method': 'min_max',
            'channel_min': ch_min.tolist(),
            'channel_max': ch_max.tolist(),
            'sensor_columns': kept_cols,
            'dropped_columns': dropped_cols,
        }

        # Store windowed data
        windowed_session_id = self._generate_session_id()
        overlap_pct = float((window_size - stride) / window_size * 100)

        # Build category distribution
        category_dist = None
        if categories:
            cat_arr = np.array(categories)
            unique_cats, cat_counts = np.unique(cat_arr, return_counts=True)
            category_dist = {str(c): int(cnt) for c, cnt in zip(unique_cats, cat_counts)}

        windowed_metadata = {
            **metadata,
            'windowed': True,
            'window_size': int(window_size),
            'stride': int(stride),
            'overlap': overlap_pct,
            'num_windows': int(num_windows),
            'label_method': label_method,
            'original_session_id': session_id,
            'per_sample_windowing': has_sample_id,
            'has_category_split': bool(categories),
            'category_distribution': category_dist,
            'split_method': split_method,
            'test_ratio': float(test_ratio) if test_ratio else None,
            'normalization': norm_params,
            'sensor_columns': kept_cols,
            'target_column': regression_target,
        }

        with _sessions_lock:
            self._evict_expired()
            _data_sessions[windowed_session_id] = {
                'windows': all_windows,
                'labels': np.array(labels) if labels else None,
                'categories': np.array(categories) if categories else None,
                'metadata': windowed_metadata,
                'created_at': time.monotonic(),
            }

        # Build label distribution with native Python types
        label_dist = None
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = {str(label): int(count) for label, count in zip(unique_labels, counts)}

        # Build per-split class distributions
        test_label_dist = None
        train_label_dist = None
        if labels and categories:
            label_arr = np.array(labels)
            cat_arr = np.array(categories)
            for split_name, split_dist_ref in [('testing', 'test'), ('training', 'train')]:
                mask = cat_arr == split_name
                if mask.any():
                    split_labels = label_arr[mask]
                    ul, uc = np.unique(split_labels, return_counts=True)
                    dist = {str(l): int(c) for l, c in zip(ul, uc)}
                    if split_name == 'testing':
                        test_label_dist = dist
                    else:
                        train_label_dist = dist

        return {
            'session_id': windowed_session_id,
            'metadata': windowed_metadata,
            'summary': {
                'num_windows': int(num_windows),
                'window_shape': list(all_windows[0].shape) if len(all_windows) > 0 else None,
                'label_distribution': label_dist,
                'category_distribution': category_dist,
                'train_label_distribution': train_label_dist,
                'test_label_distribution': test_label_dist,
                'normalization': 'min_max (0-1)',
                'dropped_columns': dropped_cols,
                'active_channels': len(kept_cols),
            }
        }

    def get_windowed_data(self, session_id: str) -> tuple:
        """Get windowed data arrays for ML training."""
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if 'windows' not in session:
            raise ValueError("Session does not contain windowed data")

        return session['windows'], session['labels']

    def get_window_sample(self, session_id: str, window_index: int) -> Dict[str, Any]:
        """
        Get a single window sample for visualization.

        Args:
            session_id: Session ID for windowed data
            window_index: Index of the window to retrieve (0-based)

        Returns:
            Dictionary containing:
            - data: List of lists, one per channel [[ch1_values], [ch2_values], ...]
            - channels: List of channel names
            - label: Label for this window
            - total_windows: Total number of windows
            - window_index: Current window index
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if 'windows' not in session:
            raise ValueError("Session does not contain windowed data")

        windows = session['windows']
        labels = session['labels']
        metadata = session['metadata']

        total_windows = len(windows)

        if window_index < 0 or window_index >= total_windows:
            raise ValueError(f"Window index {window_index} out of range (0-{total_windows - 1})")

        # Get the window data: shape is (window_size, num_channels)
        window_data = windows[window_index]

        # Transpose to get (num_channels, window_size) for easier chart plotting
        # Each channel becomes a list of values
        channel_data = []
        for ch_idx in range(window_data.shape[1]):
            channel_data.append(window_data[:, ch_idx].tolist())

        # Get channel names from metadata
        channels = metadata.get('sensor_columns', [f'ch{i}' for i in range(window_data.shape[1])])

        # Get label for this window
        label = labels[window_index] if labels is not None and len(labels) > window_index else None

        return {
            'data': channel_data,
            'channels': channels,
            'label': str(label) if label is not None else None,
            'total_windows': total_windows,
            'window_index': window_index,
            'window_size': window_data.shape[0]
        }

    def get_windows_by_label(self, session_id: str, label: str = None) -> Dict[str, Any]:
        """
        Get window indices filtered by label.

        Args:
            session_id: Session ID for windowed data
            label: Label to filter by (None returns all labels with their indices)

        Returns:
            Dictionary containing:
            - labels: Dict mapping label -> list of window indices
            - total_windows: Total number of windows
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if 'windows' not in session:
            raise ValueError("Session does not contain windowed data")

        labels = session['labels']
        total_windows = len(session['windows'])

        if labels is None:
            return {
                'labels': {},
                'total_windows': total_windows
            }

        # Build index mapping for each label
        label_indices: Dict[str, List[int]] = {}
        for idx, lbl in enumerate(labels):
            lbl_str = str(lbl)
            if lbl_str not in label_indices:
                label_indices[lbl_str] = []
            label_indices[lbl_str].append(idx)

        # If specific label requested, return only that
        if label is not None:
            return {
                'labels': {label: label_indices.get(label, [])},
                'total_windows': total_windows
            }

        return {
            'labels': label_indices,
            'total_windows': total_windows
        }
