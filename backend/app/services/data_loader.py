"""
CiRA ME - Data Loader Service
Handles loading and parsing of CSV, Edge Impulse JSON, Edge Impulse CBOR, and CiRA CBOR formats
"""

import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Global session storage for loaded data
_data_sessions: Dict[str, Dict] = {}


class DataLoader:
    """Service for loading data from various formats."""

    def __init__(self):
        self.supported_formats = ['csv', 'json', 'cbor']

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for tracking loaded data."""
        return str(uuid.uuid4())

    def _store_session(self, session_id: str, data: pd.DataFrame, metadata: Dict) -> None:
        """Store loaded data in session."""
        _data_sessions[session_id] = {
            'data': data,
            'metadata': metadata
        }

    def _get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data."""
        return _data_sessions.get(session_id)

    def load_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a CSV file.

        Expected format:
        - Headers in first row
        - Numeric sensor columns
        - Optional 'label' column
        - Optional 'timestamp' column
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # Identify columns
        columns = df.columns.tolist()
        label_col = 'label' if 'label' in columns else None
        timestamp_col = 'timestamp' if 'timestamp' in columns else None

        # Get sensor columns (numeric, excluding label and timestamp)
        sensor_cols = [
            col for col in columns
            if col not in ['label', 'timestamp'] and pd.api.types.is_numeric_dtype(df[col])
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

                    if is_cira and 'metadata' in data and 'data' in data:
                        # CiRA format
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
                    else:
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
            df = df.sort_values(sort_cols)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': df.head(rows).to_dict(orient='records')
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
        if session_id in _data_sessions:
            del _data_sessions[session_id]

    def load_cira_cbor(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from CiRA CBOR format.
        Supports both single file and folder-based loading.
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

        # Handle CiRA CBOR structure
        meta = data.get('metadata', {})
        raw_data = data.get('data', {})

        sampling_rate = meta.get('sampling_rate', 100)
        channels = meta.get('channels', list(raw_data.keys()))
        label = meta.get('label', 'unknown')

        # Build DataFrame
        df_dict = {}

        # Find max length
        max_len = max(len(raw_data.get(ch, [])) for ch in channels) if channels else 0

        # Generate timestamps
        df_dict['timestamp'] = [i / sampling_rate for i in range(max_len)]

        # Add channel data
        for ch in channels:
            ch_data = raw_data.get(ch, [])
            # Pad with NaN if needed
            if len(ch_data) < max_len:
                ch_data = list(ch_data) + [np.nan] * (max_len - len(ch_data))
            df_dict[ch] = ch_data

        # Add label
        df_dict['label'] = [label] * max_len

        df = pd.DataFrame(df_dict)

        session_id = self._generate_session_id()
        columns = df.columns.tolist()
        sensor_cols = channels

        metadata = {
            'format': 'cira_cbor',
            'file_path': file_path,
            'total_rows': len(df),
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label',
            'timestamp_column': 'timestamp',
            'labels': [label],
            'sampling_rate': sampling_rate,
            'device_id': meta.get('device_id'),
            'recording_date': meta.get('recording_date')
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

        # Find all CBOR files in train and test folders
        all_rows = []
        total_samples = 0
        channels = None
        sampling_rate = 100
        categories = {'train': 0, 'test': 0}

        for category in ['train', 'test', 'training', 'testing']:
            category_path = os.path.join(folder_path, category)
            if not os.path.exists(category_path):
                continue

            # Normalize category name
            cat_key = 'train' if category in ['train', 'training'] else 'test'

            for filename in os.listdir(category_path):
                if not filename.endswith('.cbor'):
                    continue

                cbor_file_path = os.path.join(category_path, filename)

                # Extract label from filename prefix (e.g., "sine.1.cbor..." -> "sine")
                label = filename.split('.')[0]

                try:
                    with open(cbor_file_path, 'rb') as f:
                        data = cbor2.load(f)

                    meta = data.get('metadata', {})
                    raw_data = data.get('data', {})

                    # Get channels from first file
                    if channels is None:
                        channels = meta.get('channels', list(raw_data.keys()))
                        sampling_rate = meta.get('sampling_rate', 100)

                    # Use label from metadata if available, otherwise from filename
                    if meta.get('label'):
                        label = meta['label']

                    # Build rows for this sample
                    max_len = max(len(raw_data.get(ch, [])) for ch in channels) if channels else 0

                    for i in range(max_len):
                        row_dict = {
                            'timestamp': i / sampling_rate,
                            'label': label,
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
        sensor_cols = channels or [col for col in columns if col not in ['label', 'timestamp', 'sample_id', 'category']]

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
            'sampling_rate': sampling_rate
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
        label_method: str = 'majority'
    ) -> Dict[str, Any]:
        """
        Apply windowing to loaded data, respecting sample boundaries.

        Windows are created WITHIN each sample_id to prevent data leakage
        across different recordings. The 'category' column (training/testing)
        is preserved per window for proper train/test splitting.

        Args:
            session_id: Session ID from data loading
            window_size: Number of samples per window
            stride: Step size between windows
            label_method: How to assign labels ('majority', 'first', 'last', 'threshold')

        Returns:
            Windowed data information
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        df = session['data']
        metadata = session['metadata']

        sensor_cols = metadata['sensor_columns']
        label_col = metadata.get('label_column')
        has_sample_id = 'sample_id' in df.columns
        has_category = 'category' in df.columns

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

                # Get category for this sample (all rows in a sample share the same category)
                sample_category = None
                if has_category:
                    sample_category = sample_df['category'].iloc[0]

                n_windows = (n_rows - window_size) // stride + 1

                for i in range(n_windows):
                    start = i * stride
                    end = start + window_size

                    windows.append(sample_data[start:end])

                    if label_col:
                        window_labels = sample_df.iloc[start:end][label_col].values
                        labels.append(self._get_window_label(window_labels, label_method))

                    if sample_category is not None:
                        categories.append(sample_category)
        else:
            # No sample_id - fall back to sequential windowing
            num_samples = len(df)
            n_windows = (num_samples - window_size) // stride + 1

            for i in range(n_windows):
                start = i * stride
                end = start + window_size

                window_data = df.iloc[start:end][sensor_cols].values
                windows.append(window_data)

                if label_col:
                    window_labels = df.iloc[start:end][label_col].values
                    labels.append(self._get_window_label(window_labels, label_method))

        num_windows = len(windows)
        if num_windows == 0:
            raise ValueError(
                f"No windows could be created. Window size ({window_size}) may be "
                f"larger than individual samples. Try a smaller window size."
            )

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
            'category_distribution': category_dist
        }

        _data_sessions[windowed_session_id] = {
            'windows': np.array(windows),
            'labels': np.array(labels) if labels else None,
            'categories': np.array(categories) if categories else None,
            'metadata': windowed_metadata
        }

        # Build label distribution with native Python types
        label_dist = None
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = {str(label): int(count) for label, count in zip(unique_labels, counts)}

        return {
            'session_id': windowed_session_id,
            'metadata': windowed_metadata,
            'summary': {
                'num_windows': int(num_windows),
                'window_shape': list(windows[0].shape) if windows else None,
                'label_distribution': label_dist,
                'category_distribution': category_dist
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
