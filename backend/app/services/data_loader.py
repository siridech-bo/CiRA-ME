"""
CiRA ME - Data Loader Service
Handles loading and parsing of CSV, Edge Impulse JSON, Edge Impulse CBOR, and CiRA CBOR formats
"""

import csv
import os
import json
import uuid
import time
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class DataValidationError(Exception):
    """Raised for user-facing dataset validation problems (bad headers, empty
    files, non-numeric sensor columns, etc.). The route handler surfaces
    ``code``/``message``/``hint`` to the frontend so it can render a friendly
    dialog instead of a generic error toast.
    """

    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def _safe_labels(series):
    """Convert pandas unique values to JSON-safe native Python types."""
    return [v.item() if hasattr(v, 'item') else v for v in series.unique()]


def _safe_records(df):
    """Return df as JSON-safe dict-records with NaN / +/-Inf swapped for None.

    Standard JSON does not permit NaN or Infinity tokens; Flask's default
    jsonify writes literal NaN and the browser then fails to parse the
    response body. First swap ±inf → NaN, then object-cast + where(notna,
    None) yields Python None for every non-finite float.
    """
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    return cleaned.astype(object).where(cleaned.notna(), None).to_dict(orient='records')


def _safe_preview_records(df):
    """Convenience wrapper: safe_records(df.head(10))."""
    return _safe_records(df.head(10))


# Global session storage for loaded data
# NOTE: This is in-process memory — the WSGI server MUST run with a single
# worker process (threads are fine) so all requests share this dict.
_data_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()

# Session limits
_SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours
_MAX_SESSIONS = 50


def _build_multi_csv_selection_dir(session_id: str, file_paths: List[str]) -> str:
    """Create a per-session directory containing copies of just the CSVs the
    user selected, and return its absolute path.

    Lives under DATASETS_ROOT_PATH/.multi_csv_selections/<session_id>/ so
    every container that bind-mounts the datasets root can see it (backend,
    TI ModelMaker). We copy instead of symlinking because the backend and TI
    containers mount the datasets folder at DIFFERENT container paths
    (/app/datasets vs /app/data/datasets), and absolute-path symlinks written
    from the backend's perspective resolve to non-existent paths inside TI.
    Relative symlinks would also work but the per-session data is already
    capped by the load_csv_multiple upstream guardrails, so a plain copy is
    the simplest robust option.
    """
    import shutil
    datasets_root = os.environ.get(
        'DATASETS_ROOT_PATH',
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets')
    )
    selection_dir = os.path.join(datasets_root, '.multi_csv_selections', session_id)
    os.makedirs(selection_dir, exist_ok=True)
    for fp in file_paths:
        dest = os.path.join(selection_dir, os.path.basename(fp))
        if os.path.lexists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        shutil.copy2(fp, dest)
    return selection_dir


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

    # ── Phase G — Labels sidecar integration ─────────────────────────────
    #
    # After load_csv() (and any other loader that returns a single-CSV
    # DataFrame), look for `<csv_stem>.labels.json` colocated with the file.
    # When present, add a `label` column populated by mapping each row's
    # timestamp into the label ranges.
    #
    # Failure policy matches the sidecar route's: malformed/missing sidecar
    # never crashes the loader — the DataFrame comes back unlabeled.

    def _labels_sidecar_path(self, csv_path: str) -> str:
        """Sibling sidecar path — same directory, `<stem>.labels.json`."""
        directory = os.path.dirname(csv_path)
        stem, _ext = os.path.splitext(os.path.basename(csv_path))
        return os.path.join(directory, f'{stem}.labels.json')

    def _read_labels_sidecar(self, csv_path: str) -> Optional[Dict[str, Any]]:
        """Return the parsed sidecar dict, or None on missing/malformed.

        Logs a warning on malformed JSON but never raises — a broken sidecar
        must not knock the data pipeline over.
        """
        sidecar = self._labels_sidecar_path(csv_path)
        if not os.path.isfile(sidecar):
            return None
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            import logging
            logging.getLogger(__name__).warning(
                '[data_loader] sidecar %s malformed: %s — ignoring',
                sidecar, e,
            )
            return None
        if not isinstance(data, dict):
            return None
        return data

    # PHASE-G-FOLLOWUP: labels are only applied by load_csv() (single-CSV
    # path). load_csv_multiple() concat + JOIN loaders skip the sidecar —
    # for v1, cross-file labels don't have a canonical anchor. Add per-file
    # sidecar merging when users start asking for it.

    def _apply_labels_sidecar(
        self,
        df: pd.DataFrame,
        csv_path: str,
        timestamp_col: Optional[str],
    ) -> int:
        """Add a `label` column to `df` from the CSV's sibling sidecar.

        Ranges are half-open `[from, to)` matched against the timestamp
        column's numeric values. Rows outside any range get None. Returns
        the count of labels applied (0 when no sidecar or empty).
        """
        if not timestamp_col or timestamp_col not in df.columns:
            return 0
        sidecar = self._read_labels_sidecar(csv_path)
        if not sidecar:
            return 0
        raw_labels = sidecar.get('labels')
        if not isinstance(raw_labels, list) or not raw_labels:
            return 0

        # Build label column initialized to None. Iterate ranges and stamp
        # the class where the timestamp falls in [from, to).
        ts_series = df[timestamp_col]
        if not pd.api.types.is_numeric_dtype(ts_series):
            # After load_csv's _coerce_timestamp_to_seconds() the column is
            # numeric. If a caller reaches here with a non-numeric ts, skip
            # rather than guess — labels are numeric-x by contract.
            import logging
            logging.getLogger(__name__).warning(
                '[data_loader] cannot apply labels — timestamp column %r is '
                'not numeric (dtype=%s)',
                timestamp_col, ts_series.dtype,
            )
            return 0

        # Start with the existing label column if the CSV had one — sidecar
        # labels take precedence but rows outside sidecar ranges keep the
        # CSV-native label. This matches operator intent: "add labels
        # post-hoc to my continuous recording."
        if 'label' in df.columns:
            label_col = df['label'].copy()
        else:
            label_col = pd.Series([None] * len(df), index=df.index, dtype=object)

        applied = 0
        for entry in raw_labels:
            try:
                frm = float(entry['from'])
                to = float(entry['to'])
                cls = str(entry['class']).strip()
            except (KeyError, TypeError, ValueError):
                continue  # silently skip malformed entries
            if not cls or not (frm < to):
                continue
            mask = (ts_series >= frm) & (ts_series < to)
            n = int(mask.sum())
            if n > 0:
                label_col.loc[mask] = cls
                applied += n

        df['label'] = label_col

        import logging
        logging.getLogger(__name__).info(
            '[data_loader] applied %d labels from sidecar to %s',
            applied, csv_path,
        )
        return applied

    # Patterns to detect time/timestamp columns (case-insensitive).
    # ``index`` is accepted so a CSV with a plain row-counter column named
    # ``index`` (or ``Index``, ``INDEX``…) can serve as the timestamp axis.
    TIME_COLUMN_PATTERNS = [
        'timestamp', 'time', 'time (s)', 'time(s)', 'time_s',
        'time (ms)', 'time(ms)', 'time_ms', 'elapsed', 'elapsed_time',
        't (s)', 't(s)', 't_s', 'datetime', 'date_time', 'index'
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

    def _coerce_timestamp_to_seconds(self, df, timestamp_col: str) -> Optional[str]:
        """Convert a non-numeric timestamp column to seconds-since-first-sample.

        Returns the ISO string of the anchor (first parsed datetime) for
        downstream display; returns ``None`` when the column was already
        numeric. Raises DataValidationError(NON_NUMERIC_TIMESTAMP) when the
        column is non-numeric AND cannot be parsed as datetime.
        """
        series = df[timestamp_col]
        if pd.api.types.is_numeric_dtype(series):
            return None

        # Try lenient parse to find the first offender for the error hint;
        # then require every value to parse.
        try:
            parsed_coerce = pd.to_datetime(series, errors='coerce', format='mixed')
        except Exception:
            parsed_coerce = pd.to_datetime(series, errors='coerce')

        bad_mask = parsed_coerce.isna() & series.notna()
        if bad_mask.any():
            first_bad_pos = int(bad_mask.to_numpy().nonzero()[0][0])
            first_bad_val = series.iloc[first_bad_pos]
            file_row = first_bad_pos + 2  # +1 header, +1 1-based
            raise DataValidationError(
                code='NON_NUMERIC_TIMESTAMP',
                message=(
                    f"Timestamp column `{timestamp_col}` has an unparseable "
                    f"value on row {file_row}: `{first_bad_val}`."
                ),
                hint=(
                    "Use a numeric time (seconds since start), an integer index, "
                    "or a standard datetime like `2025-04-01 08:00:00`."
                ),
            )

        # All values parsed. Anchor at first sample, convert to seconds.
        anchor = parsed_coerce.iloc[0]
        seconds = (parsed_coerce - anchor).dt.total_seconds()
        df[timestamp_col] = seconds.astype(float)
        return anchor.isoformat() if pd.notna(anchor) else None

    def _check_mixed_type_sensor(self, df, columns, exclude_cols) -> None:
        """Raise NON_NUMERIC_SENSOR when a column looks like a corrupted sensor.

        A non-excluded, object-dtype column is treated as a corrupted sensor
        if it is >=80% numerically coercible AND has at least one value that
        fails to coerce. The error names the column, the first offending row
        (1-based, accounting for the header row), and the bad value itself.
        """
        for col in columns:
            if col in exclude_cols:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                continue
            n_total = len(series)
            if n_total == 0:
                continue
            coerced = pd.to_numeric(series, errors='coerce')
            n_numeric = int(coerced.notna().sum())
            if n_numeric == 0:
                continue  # Fully non-numeric → treat as intentional string column.
            ratio_numeric = n_numeric / n_total
            if ratio_numeric < 0.8:
                continue
            bad_mask = coerced.isna() & series.notna()
            if not bad_mask.any():
                # All NaN failures are blanks — different problem, let downstream handle.
                continue
            first_bad_pos = int(bad_mask.to_numpy().nonzero()[0][0])
            first_bad_val = series.iloc[first_bad_pos]
            file_row = first_bad_pos + 2  # +1 for header, +1 for 1-based
            raise DataValidationError(
                code='NON_NUMERIC_SENSOR',
                message=f"Column `{col}` has a non-numeric value on row {file_row}: `{first_bad_val}`.",
                hint=(
                    f"Column `{col}` is {ratio_numeric:.0%} numeric — it looks like a sensor "
                    f"column with dirty data. Replace `{first_bad_val}` in row {file_row} "
                    f"with a number (or blank) and re-upload."
                ),
            )

    def _detect_label_column(self, columns: List[str]) -> Optional[str]:
        """Detect label column by matching common naming patterns."""
        columns_lower = {col.lower().strip(): col for col in columns}
        label_patterns = ['label', 'labels', 'class', 'class_name', 'target', 'category']

        for pattern in label_patterns:
            if pattern in columns_lower:
                return columns_lower[pattern]

        return None

    def load_csv(self, file_path: str, apply_labels: bool = True) -> Dict[str, Any]:
        """
        Load data from a CSV file.

        Expected format:
        - Headers in first row
        - Numeric sensor columns
        - Optional label column (e.g. 'label', 'class', 'target')
        - Optional time column (e.g. 'timestamp', 'Time (s)', 'time', 'elapsed', 'index')

        Phase G: when `apply_labels=True` (default) and a sibling
        `<csv_stem>.labels.json` sidecar exists, its label ranges are
        stamped onto a `label` column of the returned DataFrame. Rows
        outside every range keep whatever label the CSV already had (or
        None). Windowing/features/training already consume `label` when
        present — no downstream changes needed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise DataValidationError(
                code='EMPTY_FILE',
                message="The CSV file is empty or has no data rows.",
                hint="Verify the file has a header row plus at least one data row.",
            )

        # Empty (header-only or zero-column) file
        if df.empty or len(df.columns) == 0:
            raise DataValidationError(
                code='EMPTY_FILE',
                message="The CSV file is empty or has no data rows.",
                hint="Verify the file has a header row plus at least one data row.",
            )

        # Identify columns
        columns = df.columns.tolist()
        label_col = self._detect_label_column(columns)
        timestamp_col = self._detect_time_column(columns)

        # Convention: first column is always time if not detected by name
        if not timestamp_col and len(columns) > 0 and pd.api.types.is_numeric_dtype(df[columns[0]]):
            timestamp_col = columns[0]

        if not timestamp_col:
            raise DataValidationError(
                code='NO_TIME_OR_INDEX',
                message="The file has no `time` or `index` column.",
                hint="Add a column named `time` (seconds/ms) or `index` (row counter), then re-upload.",
            )

        # If the timestamp column holds ISO / mixed datetime strings, convert
        # in-place to seconds-since-first-sample so downstream numeric
        # operations (sample-rate detection, windowing, features) work.
        time_epoch_start = self._coerce_timestamp_to_seconds(df, timestamp_col)

        # Columns to exclude from sensor data
        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if timestamp_col:
            exclude_cols.add(timestamp_col)

        # Catch columns that look like corrupted sensors (mostly numeric but a
        # few stray text values). Runs BEFORE the empty check so it fires even
        # when other, clean numeric columns exist.
        self._check_mixed_type_sensor(df, columns, exclude_cols)

        # Get sensor columns (numeric, excluding label and time)
        sensor_cols = [
            col for col in columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not sensor_cols:
            # Try to point at a specific offending column: any non-numeric
            # column that isn't the detected label/time.
            offending = next(
                (c for c in columns
                 if c not in exclude_cols and not pd.api.types.is_numeric_dtype(df[c])),
                None,
            )
            if offending is not None:
                raise DataValidationError(
                    code='NON_NUMERIC_SENSOR',
                    message=f"Column `{offending}` has non-numeric values.",
                    hint="Check for stray text, blank cells, or wrong delimiters in the CSV.",
                )
            raise DataValidationError(
                code='NO_SENSOR_COLUMNS',
                message="No numeric sensor columns detected.",
                hint="Ensure at least one column contains numeric values. Text columns are treated as labels.",
            )

        # Phase G — apply sidecar labels BEFORE metadata / preview so the
        # `labels` metadata reflects sidecar-added classes and downstream
        # windowing sees them without reload.
        sidecar_applied = 0
        if apply_labels:
            try:
                sidecar_applied = self._apply_labels_sidecar(
                    df, file_path, timestamp_col,
                )
                # After applying, treat `label` as the label column even if
                # the CSV didn't originally have one.
                if sidecar_applied > 0 and 'label' in df.columns and not label_col:
                    label_col = 'label'
            except Exception:
                # HARD REQ: never crash on sidecar. Log + carry on.
                import logging
                logging.getLogger(__name__).exception(
                    '[data_loader] apply_labels_sidecar failed for %s '
                    '— continuing without labels',
                    file_path,
                )

        # Phase H — expose channels_by_sensor so downstream nodes (chart,
        # normalize, feature-extract) can render multi-axis columns as one
        # named group per source sensor. Single-value CSVs get the sensor
        # name → [sensor_name] so consumers can iterate uniformly.
        channels_by_sensor = self._resolve_channels_by_sensor(
            file_path, sensor_cols,
        )

        # Generate session ID and store data
        session_id = self._generate_session_id()
        metadata = {
            'format': 'csv',
            'file_path': file_path,
            'total_rows': len(df),
            'columns': list(df.columns),  # includes label if sidecar added
            'sensor_columns': sensor_cols,
            'label_column': label_col,
            'timestamp_column': timestamp_col,
            'time_epoch_start': time_epoch_start,
            'labels': _safe_labels(df[label_col]) if label_col else None,
            'labels_sidecar_applied': sidecar_applied,
            'channels_by_sensor': channels_by_sensor,
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_preview_records(df)
        }

    def _resolve_channels_by_sensor(
        self, file_path: str, sensor_cols: List[str],
    ) -> Optional[Dict[str, List[str]]]:
        """Look up the source sensor's channels config from the asset tree.

        Ingest router writes each sensor's CSV under
        ``<datasets_root>/<root>/<plant>/<machine>/<sensor>/<date>.csv`` —
        so the parent directory of ``file_path`` relative to the datasets
        root is the sensor's ``topic_path``. Look up ``AssetSensorMeta``:
        if the sensor has ``channels`` configured (multi-axis), return
        ``{sensor_name: [ch1, ch2, ...]}``. If not, fall back to
        ``{sensor_name: [sensor_name]}`` so downstream code always sees a
        uniform dict shape. Returns None on any lookup failure — callers
        treat missing as "unknown / assume flat".
        """
        try:
            datasets_root = os.environ.get(
                'DATASETS_ROOT_PATH',
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'datasets',
                ),
            )
            abs_path = os.path.abspath(file_path)
            abs_root = os.path.abspath(datasets_root)
            if not abs_path.startswith(abs_root):
                return None
            # Sensor directory relative to root, using forward-slash so it
            # matches AssetNode.topic_path regardless of OS.
            parent_dir = os.path.dirname(abs_path)
            rel = os.path.relpath(parent_dir, abs_root).replace(os.sep, '/')
            if rel.startswith('.'):
                return None
            sensor_name = rel.rsplit('/', 1)[-1]
            # Deferred imports so the pure-data path stays testable in
            # isolation without needing the Flask app / models module.
            try:
                from ..models import AssetNode, AssetSensorMeta
            except Exception:
                return None
            node = AssetNode.get_by_topic_path(rel)
            if not node:
                # Router CSV location, but no tree node — could be a stale
                # file from a retired sensor. Fall back to a flat map.
                return {sensor_name: list(sensor_cols) if sensor_cols else [sensor_name]}
            meta = AssetSensorMeta.get(node['id'])
            channels = (meta or {}).get('channels')
            if channels:
                # Preserve only channels actually present in the CSV. If the
                # tree config was updated mid-day the file may still have
                # the old columns; expose the intersection so downstream
                # nodes only enumerate real columns.
                return {sensor_name: [c for c in channels if c in sensor_cols] or list(channels)}
            return {sensor_name: [sensor_name]}
        except Exception:
            # HARD REQ — never crash the loader on metadata lookup.
            import logging
            logging.getLogger(__name__).warning(
                '[data_loader] channels_by_sensor lookup failed for %s',
                file_path, exc_info=True,
            )
            return None

    def _sniff_text_delimiter(self, file_path: str, sample_bytes: int = 4096) -> str:
        """Sniff the delimiter of a text file. Falls back to ',' on failure."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(sample_bytes)
            if not sample.strip():
                return ','
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
                return dialect.delimiter
            except csv.Error:
                return ','
        except OSError:
            return ','

    def load_text(
        self,
        file_path: str,
        delimiter: Optional[str] = None,
        header_row: int = 1,
        skip_rows: int = 0,
        column_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Load data from a delimited text file (.txt, .tsv, .dat, .log).

        Downstream metadata/columns/labels detection is identical to CSV.

        Args:
            file_path: Absolute path to the text file.
            delimiter: Explicit delimiter. If ``None``, sniff via ``csv.Sniffer``
                on the first ~4 KB and fall back to ``','`` on failure.
            header_row: 1-based row containing headers. ``0`` means headerless —
                columns are auto-generated as ``col_1..col_N``.
            skip_rows: Number of rows to skip from the top BEFORE the header row.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Coerce inputs
        try:
            header_row = int(header_row) if header_row is not None else 1
        except (TypeError, ValueError):
            header_row = 1
        try:
            skip_rows = int(skip_rows) if skip_rows is not None else 0
        except (TypeError, ValueError):
            skip_rows = 0
        if skip_rows < 0:
            skip_rows = 0

        # Sniff delimiter if not provided
        if not delimiter:
            delimiter = self._sniff_text_delimiter(file_path)

        # pandas' header kwarg: None = headerless, integer = row index of header
        # AFTER skiprows have been consumed.
        if header_row <= 0:
            header_kw: Optional[int] = None
        else:
            header_kw = header_row - 1

        try:
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                header=header_kw,
                skiprows=skip_rows if skip_rows > 0 else None,
                encoding='utf-8',
                engine='python',
            )
        except pd.errors.EmptyDataError:
            raise DataValidationError(
                code='EMPTY_FILE',
                message="The text file is empty or has no data rows.",
                hint="Verify the file has a header row plus at least one data row.",
            )
        except UnicodeDecodeError:
            raise DataValidationError(
                code='ENCODING_ERROR',
                message="The text file is not valid UTF-8.",
                hint="Re-save the file with UTF-8 encoding and try again.",
            )
        except Exception as e:
            raise DataValidationError(
                code='PARSE_ERROR',
                message=f"Failed to parse text file: {e}",
                hint="Check that the delimiter and header row match the file's layout.",
            )

        if df.empty or len(df.columns) == 0:
            raise DataValidationError(
                code='EMPTY_FILE',
                message="The text file is empty or has no data rows.",
                hint="Verify the file has a header row plus at least one data row.",
            )

        # Fix A: whitespace-delimited files often have trailing spaces on each
        # row, which pandas materialises as fully-empty trailing columns. Drop
        # all-NaN columns so downstream sees only real signals.
        df = df.dropna(axis=1, how='all')
        if len(df.columns) == 0:
            raise DataValidationError(
                code='EMPTY_FILE',
                message="The text file has no data columns after cleanup.",
                hint="Check the delimiter — the file may not contain any real columns.",
            )

        # Headerless — pandas will use 0..N integers as column names. Rename to
        # human-friendly col_1..col_N so downstream code (which assumes string
        # column names) is happy.
        if header_kw is None:
            df.columns = [f'col_{i + 1}' for i in range(len(df.columns))]

        # Fix B: if caller passed explicit column_names AND count matches the
        # actual column count, apply them (typically only used with headerless
        # files loaded via a curated catalog entry).
        if column_names and len(column_names) == len(df.columns):
            df.columns = list(column_names)

        # Downstream detection is identical to CSV.
        columns = df.columns.tolist()
        label_col = self._detect_label_column(columns)
        timestamp_col = self._detect_time_column(columns)

        if not timestamp_col and len(columns) > 0 and pd.api.types.is_numeric_dtype(df[columns[0]]):
            timestamp_col = columns[0]

        if not timestamp_col:
            raise DataValidationError(
                code='NO_TIME_OR_INDEX',
                message="The file has no `time` or `index` column.",
                hint="Add a column named `time` (seconds/ms) or `index` (row counter), then re-upload.",
            )

        # ISO / mixed datetime strings → seconds-since-first-sample.
        time_epoch_start = self._coerce_timestamp_to_seconds(df, timestamp_col)

        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if timestamp_col:
            exclude_cols.add(timestamp_col)

        # Catch corrupted-sensor columns (mostly numeric with stray text).
        self._check_mixed_type_sensor(df, columns, exclude_cols)

        sensor_cols = [
            col for col in columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not sensor_cols:
            offending = next(
                (c for c in columns
                 if c not in exclude_cols and not pd.api.types.is_numeric_dtype(df[c])),
                None,
            )
            if offending is not None:
                raise DataValidationError(
                    code='NON_NUMERIC_SENSOR',
                    message=f"Column `{offending}` has non-numeric values.",
                    hint="Check for stray text, blank cells, or wrong delimiters in the file.",
                )
            raise DataValidationError(
                code='NO_SENSOR_COLUMNS',
                message="No numeric sensor columns detected.",
                hint="Ensure at least one column contains numeric values. Text columns are treated as labels.",
            )

        session_id = self._generate_session_id()
        metadata = {
            # Reported as 'csv' so downstream (windowing, features, TI export)
            # treats it identically — the shape of the DataFrame is the same.
            'format': 'csv',
            'source_format': 'text',
            'file_path': file_path,
            'delimiter': delimiter,
            'header_row': header_row,
            'skip_rows': skip_rows,
            'total_rows': len(df),
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': label_col,
            'timestamp_column': timestamp_col,
            'time_epoch_start': time_epoch_start,
            'labels': _safe_labels(df[label_col]) if label_col else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_preview_records(df)
        }

    def load_csv_multiple(
        self,
        file_paths: List[str],
        merge_mode: Optional[str] = None,
        alignment: str = 'exact',
        tolerance_ms: Optional[float] = None,
        resample_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Load data from multiple CSV files as one dataset.

        merge_mode:
            - None (default): auto-detect. All files under the same parent
              folder → concat rows (existing "many-days-of-same-sensor"
              behavior). Files span multiple parent folders → JOIN by
              timestamp with one column per sensor folder.
            - 'concat': force row concatenation (existing behavior, requires
              identical column names across files).
            - 'join': force cross-sensor JOIN even if files share a folder.

        alignment (only used when merge_mode resolves to 'join'):
            - 'exact': inner join on identical timestamp values.
            - 'nearest': pd.merge_asof with tolerance_ms tolerance
              (per-file timestamps may drift slightly).
            - 'resample': resample each file to resample_hz then align on
              the common grid (interpolates gaps).
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        if len(file_paths) == 1:
            return self.load_csv(file_paths[0])

        # Validate all files exist
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"File not found: {fp}")

        # Auto-detect merge mode from folder layout when caller didn't ask.
        # Files under one parent → user is picking multiple days of the same
        # sensor → row-concat. Files under different parents → user is picking
        # different sensors for the same machine → JOIN by timestamp.
        auto_detected_join = False
        if merge_mode is None:
            parent_dirs = {os.path.dirname(fp) for fp in file_paths}
            if len(parent_dirs) > 1:
                merge_mode = 'join'
                auto_detected_join = True
            else:
                merge_mode = 'concat'

        if merge_mode == 'join':
            # When we auto-detected JOIN (caller didn't specify alignment
            # either), default to 'nearest' with a 200 ms tolerance. Real
            # multi-sensor data almost never has synchronised timestamps
            # (each sensor publishes at its own cadence + network jitter),
            # so the 'exact' default the parameter signature carries produces
            # a 1-row join with nothing matching. This matches what the
            # frontend merge-alignment dialog offers as its default.
            if auto_detected_join and alignment == 'exact':
                alignment = 'nearest'
                if tolerance_ms is None:
                    tolerance_ms = 200
            return self._load_csv_multi_join(
                file_paths,
                alignment=alignment,
                tolerance_ms=tolerance_ms,
                resample_hz=resample_hz,
            )

        # ── merge_mode == 'concat' path (existing behavior) ──────────────
        # Read headers from all files and check column compatibility
        reference_cols = None
        reference_file = None
        for fp in file_paths:
            try:
                cols = pd.read_csv(fp, nrows=0).columns.tolist()
            except pd.errors.EmptyDataError:
                raise DataValidationError(
                    code='EMPTY_FILE',
                    message="The CSV file is empty or has no data rows.",
                    hint="Verify the file has a header row plus at least one data row.",
                )
            if reference_cols is None:
                reference_cols = cols
                reference_file = os.path.basename(fp)
            elif cols != reference_cols:
                raise DataValidationError(
                    code='COLUMN_MISMATCH',
                    message="Selected CSV files have different columns.",
                    hint="Ensure all selected files have identical headers, then re-upload.",
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

        if not timestamp_col:
            raise DataValidationError(
                code='NO_TIME_OR_INDEX',
                message="The file has no `time` or `index` column.",
                hint="Add a column named `time` (seconds/ms) or `index` (row counter), then re-upload.",
            )

        # Global datetime → seconds conversion. Anchor is the first row of the
        # combined dataset so multi-day recordings preserve inter-file gaps.
        time_epoch_start = self._coerce_timestamp_to_seconds(combined, timestamp_col)

        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if timestamp_col:
            exclude_cols.add(timestamp_col)

        self._check_mixed_type_sensor(combined, columns, exclude_cols)

        sensor_cols = [
            col for col in columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(combined[col])
        ]

        if not sensor_cols:
            offending = next(
                (c for c in columns
                 if c not in exclude_cols and not pd.api.types.is_numeric_dtype(combined[c])),
                None,
            )
            if offending is not None:
                raise DataValidationError(
                    code='NON_NUMERIC_SENSOR',
                    message=f"Column `{offending}` has non-numeric values.",
                    hint="Check for stray text, blank cells, or wrong delimiters in the CSV.",
                )
            raise DataValidationError(
                code='NO_SENSOR_COLUMNS',
                message="No numeric sensor columns detected.",
                hint="Ensure at least one column contains numeric values. Text columns are treated as labels.",
            )

        # Store session
        session_id = self._generate_session_id()
        all_columns = combined.columns.tolist()

        # Build a session-scoped directory containing symlinks (or copies as
        # fallback) to only the selected files. Downstream consumers — TI
        # training, feature extraction, pipeline replay — read `file_path` as
        # a directory and glob for CSVs. Pointing them at the parent folder
        # (as we used to) meant a "Select 2 files" click actually shipped the
        # whole shared/ directory to TI. This dir contains exactly the user's
        # selection, and lives under the datasets root so it's visible to
        # every container that mounts it.
        selection_dir = _build_multi_csv_selection_dir(session_id, file_paths)

        metadata = {
            'format': 'csv',
            'file_path': selection_dir,
            'file_paths': file_paths,
            'is_multi_csv': True,
            'total_rows': len(combined),
            'total_samples': len(file_paths),
            'columns': all_columns,
            'sensor_columns': sensor_cols,
            'label_column': label_col,
            'timestamp_column': timestamp_col,
            'time_epoch_start': time_epoch_start,
            'labels': _safe_labels(combined[label_col]) if label_col else None,
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
            'preview': _safe_records(preview)
        }

    def _load_csv_multi_join(
        self,
        file_paths: List[str],
        alignment: str = 'exact',
        tolerance_ms: Optional[float] = None,
        resample_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Cross-sensor JOIN of multiple single-sensor CSVs by timestamp.

        Each file contributes columns named after its parent folder (the
        sensor name). Files may live in different folders but must each
        have a detectable timestamp column.
        """
        frames = []
        sensor_names: List[str] = []
        # Phase H — channels_by_sensor for the joined output. Single-value
        # sensors map to their own name; multi-axis sensors expose each
        # declared axis. Filled in as each file is processed.
        channels_by_sensor: Dict[str, List[str]] = {}

        for fp in file_paths:
            df = pd.read_csv(fp)
            cols = df.columns.tolist()

            ts_col = self._detect_time_column(cols)
            if not ts_col and cols and pd.api.types.is_numeric_dtype(df[cols[0]]):
                ts_col = cols[0]
            if not ts_col:
                raise DataValidationError(
                    code='NO_TIME_OR_INDEX',
                    message=f"File {os.path.basename(fp)} has no timestamp column for JOIN.",
                    hint="Cross-sensor JOIN needs a `time` / `timestamp` / `index` column in every file.",
                )

            # Normalise timestamp to seconds in-place (adds a computed column
            # when the source was datetime strings).
            self._coerce_timestamp_to_seconds(df, ts_col)

            label_col = self._detect_label_column(cols)
            sensor = os.path.basename(os.path.dirname(fp)) or 'sensor'
            sensor_names.append(sensor)

            # Numeric sensor columns from THIS file (excludes timestamp + label).
            value_cols = [
                c for c in cols
                if c != ts_col and c != label_col and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not value_cols:
                raise DataValidationError(
                    code='NO_SENSOR_COLUMNS',
                    message=f"File {os.path.basename(fp)} has no numeric sensor columns to JOIN.",
                    hint="Ensure the file contains at least one numeric non-timestamp column.",
                )

            # Phase H — look up this sensor's channels from the asset tree
            # so multi-axis files get the spec's `sensor.axis` naming.
            declared_channels = self._resolve_channels_by_sensor(fp, value_cols)
            declared_list = None
            if declared_channels:
                declared_list = declared_channels.get(sensor)

            # Rename: single value column takes the sensor name directly;
            # declared multi-axis columns use `sensor.axis` (spec §H.5);
            # anything else falls back to the legacy `sensor__origColName`
            # prefix so existing non-multi-axis multi-column files still work.
            rename = {}
            output_names = []
            if len(value_cols) == 1:
                rename[value_cols[0]] = sensor
                output_names.append(sensor)
                channels_by_sensor[sensor] = [sensor]
            elif declared_list and set(declared_list) & set(value_cols):
                per_axis = []
                for c in value_cols:
                    if c in declared_list:
                        renamed = f'{sensor}.{c}'
                    else:
                        # A stray non-axis column on a multi-axis CSV — keep
                        # the legacy prefix so we never silently drop data.
                        renamed = f'{sensor}__{c}'
                    rename[c] = renamed
                    output_names.append(renamed)
                    if c in declared_list:
                        per_axis.append(c)
                channels_by_sensor[sensor] = per_axis
            else:
                for c in value_cols:
                    renamed = f"{sensor}__{c}"
                    rename[c] = renamed
                    output_names.append(renamed)
                channels_by_sensor[sensor] = [rename[c] for c in value_cols]
            rename[ts_col] = '__ts'
            df = df.rename(columns=rename)
            df = df[['__ts'] + output_names]
            df = df.sort_values('__ts').reset_index(drop=True)
            frames.append(df)

        # ── Merge ────────────────────────────────────────────────────────
        if alignment == 'exact':
            combined = frames[0]
            for f in frames[1:]:
                combined = combined.merge(f, on='__ts', how='inner')
            if combined.empty:
                raise DataValidationError(
                    code='NO_MATCHING_TIMESTAMPS',
                    message="No exact-timestamp matches across the selected sensor files.",
                    hint="Try alignment mode 'nearest' with a tolerance, or 'resample' to a common rate.",
                )

        elif alignment == 'nearest':
            if not tolerance_ms or tolerance_ms <= 0:
                raise ValueError("tolerance_ms > 0 required for 'nearest' alignment")
            tol = pd.Timedelta(milliseconds=float(tolerance_ms))
            # merge_asof needs a datetime-like column
            for f in frames:
                f['__ts_td'] = pd.to_timedelta(f['__ts'], unit='s')
            combined = frames[0]
            for f in frames[1:]:
                combined = pd.merge_asof(
                    combined.sort_values('__ts_td'),
                    f.sort_values('__ts_td'),
                    on='__ts_td', tolerance=tol, direction='nearest',
                    suffixes=('', '__dup'),
                )
                # Drop duplicated __ts columns from right frames.
                for dup in [c for c in combined.columns if c.endswith('__dup') or (c == '__ts' and '__ts_td' in combined.columns and combined.columns.tolist().count('__ts') > 1)]:
                    if dup in combined.columns:
                        combined = combined.drop(columns=[dup])
            combined = combined.dropna()
            if combined.empty:
                raise DataValidationError(
                    code='NO_NEAREST_MATCHES',
                    message=f"No timestamp matches within {tolerance_ms} ms tolerance.",
                    hint="Increase tolerance_ms, or switch to 'resample' alignment.",
                )
            combined = combined.drop(columns=['__ts_td'])

        elif alignment == 'resample':
            if not resample_hz or resample_hz <= 0:
                raise ValueError("resample_hz > 0 required for 'resample' alignment")
            period_ms = int(round(1000.0 / float(resample_hz)))
            resampled = []
            for f in frames:
                idx = pd.to_datetime(f['__ts'], unit='s')
                f2 = f.drop(columns=['__ts']).set_index(idx)
                f2 = f2.resample(f"{period_ms}ms").mean()
                resampled.append(f2)
            combined = resampled[0]
            for f in resampled[1:]:
                combined = combined.join(f, how='outer')
            combined = combined.interpolate(method='linear').dropna()
            if combined.empty:
                raise DataValidationError(
                    code='RESAMPLE_EMPTY',
                    message="Resample-aligned dataset is empty after interpolation.",
                    hint="Check timestamp ranges overlap and rate is reasonable for the data.",
                )
            combined = combined.reset_index().rename(columns={'index': '__ts'})
            combined['__ts'] = (combined['__ts'] - combined['__ts'].iloc[0]).dt.total_seconds()

        else:
            raise ValueError(f"Unknown alignment mode: {alignment!r}")

        # Rename __ts to a friendlier name downstream consumers understand.
        combined = combined.rename(columns={'__ts': 'timestamp_s'})

        # Apply labels sidecar for cross-sensor JOIN: labels attach to time
        # ranges (not individual sensors), so any source CSV's sidecar is a
        # valid anchor. Canonically use file_paths[0] — writer + reader agree.
        # Data flow: user writes labels via PUT with csv_path=file_paths[0];
        # this loader reads that same sidecar and stamps the joined DataFrame
        # with a `label` column so downstream training gets supervised data.
        labels_applied = 0
        if file_paths:
            try:
                labels_applied = self._apply_labels_sidecar(
                    df=combined,
                    csv_path=file_paths[0],
                    timestamp_col='timestamp_s',
                )
            except Exception:
                logger.warning('[data_loader] JOIN sidecar apply failed',
                               exc_info=True)

        session_id = self._generate_session_id()
        columns = combined.columns.tolist()
        sensor_cols = [c for c in columns if c not in ('timestamp_s', 'label')]

        selection_dir = _build_multi_csv_selection_dir(session_id, file_paths)

        metadata = {
            'format': 'csv',
            'file_path': selection_dir,
            'file_paths': file_paths,
            'is_multi_csv': True,
            'is_cross_sensor_join': True,
            'total_rows': len(combined),
            'total_samples': 1,  # combined is one continuous joined dataset
            'columns': columns,
            'sensor_columns': sensor_cols,
            'label_column': 'label' if 'label' in combined.columns else None,
            'timestamp_column': 'timestamp_s',
            'time_epoch_start': 0.0,
            'labels': None,
            'labels_sidecar_applied': labels_applied,
            'source_files': [os.path.basename(fp) for fp in file_paths],
            'source_sensors': sensor_names,
            'channels_by_sensor': channels_by_sensor,
            'merge_mode': 'join',
            'alignment': alignment,
            'tolerance_ms': tolerance_ms,
            'resample_hz': resample_hz,
        }

        self._store_session(session_id, combined, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_records(combined.head(20)),
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
            'labels': _safe_labels(df['label']) if 'label' in df.columns else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_preview_records(df)
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
            'labels': _safe_labels(df['label']) if 'label' in df.columns else None
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_preview_records(df)
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
            'labels': _safe_labels(df['label']),
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
            'preview': _safe_preview_records(df)
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
            'labels': _safe_labels(df['label'])
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
            'preview': _safe_records(df.iloc[:rows])
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
            labels = _safe_labels(df['label'])
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
            'preview': _safe_preview_records(df)
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
            'labels': _safe_labels(df['label']),
        }

        self._store_session(session_id, df, metadata)

        return {
            'session_id': session_id,
            'metadata': metadata,
            'preview': _safe_preview_records(df)
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
        elif ext in ('.txt', '.tsv', '.dat', '.log') or format_hint == 'text':
            # Text file — sniff delimiter, keep default header/skip settings.
            # The Text Import wizard calls load_text directly with user-picked
            # settings; this path handles the "just preview it" case.
            result = self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Limit preview rows
        session = self._get_session(result['session_id'])
        if session:
            result['preview'] = _safe_records(session['data'].head(rows))

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
                result['preview'] = _safe_records(preview_df.head(rows))
            else:
                result['preview'] = _safe_records(df.head(rows))

        return result

    def _get_window_label(self, window_labels: np.ndarray, label_method: str) -> str:
        """Determine a single label for a window from its constituent labels.

        Sidecar labels (Phase G) legitimately produce None for rows outside
        any labelled range. Left as-is, that mixes str and None in the
        object array, and `np.unique` sorts internally → TypeError
        `'<' not supported between instances of 'str' and 'NoneType'`.
        Coerce None → 'unlabeled' first so the aggregator is None-safe;
        downstream code (train/test split, stratification) treats
        'unlabeled' as its own class.
        """
        # Fast path — coerce object-dtype None → 'unlabeled' so np.unique
        # doesn't have to sort mixed types.
        if window_labels.dtype == object:
            window_labels = np.array(
                ['unlabeled' if v is None else str(v) for v in window_labels],
                dtype=object,
            )

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
        target_column: str = None,
        selected_columns: list = None,
        split_strategy: str = 'temporal_end',
        no_windowing: bool = False,
        normalization_method: str = 'min_max'
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

        # Filter to user-selected columns if provided
        if selected_columns:
            sensor_cols = [c for c in sensor_cols if c in selected_columns]
            if not sensor_cols:
                raise ValueError("No valid sensor columns selected")
            print(f"[DataLoader] Using {len(sensor_cols)} of {len(metadata['sensor_columns'])} columns: {sensor_cols}")

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
                # Single CSV — apply split_strategy
                total_rows = len(df)
                gap = window_size  # gap prevents train/test window overlap
                df = df.copy()

                if split_strategy == 'temporal_blocks':
                    # Interleaved blocks: divide signal into N blocks, distribute test blocks evenly
                    # Adaptive: each test block should have at least 3 windows worth of data
                    min_block_rows = window_size * 3  # minimum rows per block for meaningful windows
                    max_possible_blocks = max(3, total_rows // min_block_rows)

                    # Scale test blocks with data size: 2 for small, up to 5 for large datasets
                    if total_rows < 2000:
                        n_test_blocks = 2
                    elif total_rows < 10000:
                        n_test_blocks = 3
                    elif total_rows < 50000:
                        n_test_blocks = 4
                    else:
                        n_test_blocks = 5

                    n_blocks = min(max_possible_blocks, round(n_test_blocks / test_ratio))
                    n_blocks = max(n_test_blocks + 2, n_blocks)  # at least 2 train blocks between tests
                    n_test_blocks = min(n_test_blocks, max(2, round(n_blocks * test_ratio)))
                    # Ensure test ratio doesn't exceed requested by too much
                    while n_test_blocks / n_blocks > test_ratio * 1.5 and n_test_blocks > 2:
                        n_test_blocks -= 1
                    block_size = total_rows // n_blocks

                    # Evenly space test blocks throughout the signal
                    test_block_indices = set()
                    step = n_blocks / n_test_blocks
                    for i in range(n_test_blocks):
                        idx = int(round(step * (i + 0.5)))
                        idx = min(idx, n_blocks - 1)
                        test_block_indices.add(idx)

                    half_gap = gap // 2
                    df['category'] = 'training'
                    for block_idx in range(n_blocks):
                        start = block_idx * block_size
                        end = start + block_size if block_idx < n_blocks - 1 else total_rows
                        if block_idx in test_block_indices:
                            # Gap at start of test block
                            gap_end_pos = min(start + half_gap, end)
                            df.iloc[start:gap_end_pos, df.columns.get_loc('category')] = 'gap'
                            # Test region
                            test_end = end - half_gap if end < total_rows else end
                            df.iloc[gap_end_pos:test_end, df.columns.get_loc('category')] = 'testing'
                            # Gap at end of test block
                            if test_end < end:
                                df.iloc[test_end:end, df.columns.get_loc('category')] = 'gap'

                    print(f"[DataLoader] Interleaved: {n_blocks} blocks, {n_test_blocks} test blocks at indices {sorted(test_block_indices)}, block_size={block_size}")
                    df = df[df['category'] != 'gap'].reset_index(drop=True)
                    has_category = True
                    split_method = 'interleaved blocks'

                elif split_strategy == 'random':
                    # Random split — assign categories after windowing (at window level)
                    split_method = 'random'
                    # Don't assign row-level categories; will be done after windowing below

                else:
                    # Default: temporal_end — test at the end
                    split_method = 'temporal'
                    test_rows = max(window_size, round(total_rows * test_ratio))
                    train_rows = total_rows - test_rows - gap

                    if train_rows < window_size:
                        gap = 0
                        train_rows = total_rows - test_rows

                    min_rows_for_2_windows = window_size + stride
                    if train_rows < min_rows_for_2_windows or test_rows < min_rows_for_2_windows:
                        gap = max(0, gap // 2)
                        desired_test = max(min_rows_for_2_windows, round(total_rows * test_ratio))
                        train_rows = total_rows - desired_test - gap
                        test_rows = desired_test
                        if train_rows < min_rows_for_2_windows:
                            gap = 0
                            test_rows = max(min_rows_for_2_windows, round(total_rows * test_ratio))
                            train_rows = total_rows - test_rows
                        split_method = 'temporal (with gap)' if gap > 0 else 'temporal (no gap)'

                    df['category'] = 'gap'
                    df.iloc[:train_rows, df.columns.get_loc('category')] = 'training'
                    df.iloc[train_rows + gap:, df.columns.get_loc('category')] = 'testing'
                    df = df[df['category'] != 'gap'].reset_index(drop=True)
                    has_category = True

        windows = []
        labels = []
        categories = []

        if no_windowing:
            # Raw mode: each row is one sample, no windowing or feature extraction
            split_method = split_method or 'random'
            print(f"[DataLoader] Raw mode: label_col={label_col}, regression_target={regression_target}, columns={list(df.columns)}")

            # For classification: ensure label_col is found
            # Note: 'category' is excluded — it's used internally for train/test splitting
            if not label_col and not regression_target:
                for candidate in ['label', 'labels', 'class', 'class_name', 'target']:
                    if candidate in df.columns:
                        label_col = candidate
                        break

            # Ensure label_col is not also in sensor_cols (avoid using label as input feature)
            if label_col and label_col in sensor_cols:
                sensor_cols = [c for c in sensor_cols if c != label_col]

            print(f"[DataLoader] Raw mode: {len(df)} rows, label_col={label_col}, "
                  f"regression_target={regression_target}, sensor_cols={sensor_cols[:5]}...")

            for idx in range(len(df)):
                row_data = df.iloc[idx][sensor_cols].values
                windows.append(row_data.reshape(1, -1))  # shape (1, n_features)

                if regression_target:
                    labels.append(float(df.iloc[idx][regression_target]))
                elif label_col and label_col in df.columns:
                    labels.append(df.iloc[idx][label_col])

                if has_category:
                    categories.append(df.iloc[idx]['category'])

            # If no categories assigned yet, do random split
            if not categories and test_ratio and 0 < test_ratio < 1:
                from sklearn.model_selection import train_test_split as sk_split
                num_rows = len(windows)
                indices = np.arange(num_rows)
                if labels and not regression_target:
                    try:
                        train_idx, test_idx = sk_split(
                            indices, test_size=test_ratio,
                            random_state=42, stratify=np.array(labels)
                        )
                    except ValueError:
                        train_idx, test_idx = sk_split(indices, test_size=test_ratio, random_state=42)
                else:
                    train_idx, test_idx = sk_split(indices, test_size=test_ratio, random_state=42)
                categories = ['training'] * num_rows
                for idx in test_idx:
                    categories[idx] = 'testing'
                split_method = 'random (raw mode)'

        elif has_sample_id:
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
                is_interleaved = split_method in ('interleaved blocks', 'random')
                if is_interleaved:
                    # Interleaved/random: window the ENTIRE signal sequentially,
                    # assign category based on the center of each window
                    all_data = df[sensor_cols].values
                    all_cats = df['category'].values
                    all_target = df[regression_target].values if regression_target else None
                    n_rows = len(all_data)
                    n_windows = (n_rows - window_size) // stride + 1
                    for i in range(n_windows):
                        start = i * stride
                        end = start + window_size
                        center = start + window_size // 2
                        win_cat = all_cats[center]
                        windows.append(all_data[start:end])
                        categories.append(win_cat)
                        if regression_target:
                            labels.append(float(np.mean(all_target[start:end])))
                        elif label_col:
                            window_labels = df.iloc[start:end][label_col].values
                            labels.append(self._get_window_label(window_labels, label_method))
                else:
                    # Temporal end block: group by category to prevent cross-boundary windows
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

        # --- Post-windowing random split (user-selected random strategy) ---
        if split_method == 'random' and num_windows > 1:
            from sklearn.model_selection import train_test_split as sk_split
            indices = np.arange(num_windows)
            if labels and not regression_target:
                # Classification: stratified random to ensure all classes in both sets
                try:
                    train_idx, test_idx = sk_split(
                        indices, test_size=test_ratio,
                        random_state=42, stratify=np.array(labels)
                    )
                except ValueError:
                    train_idx, test_idx = sk_split(indices, test_size=test_ratio, random_state=42)
            else:
                train_idx, test_idx = sk_split(indices, test_size=test_ratio, random_state=42)
            categories = ['training'] * num_windows
            for idx in test_idx:
                categories[idx] = 'testing'
            split_method = 'random'

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

        # --- Per-channel normalization ---
        # Compute stats from training data only to prevent leakage.
        # Method is user-selectable: 'min_max' (default, historical behavior),
        # 'z_score', 'robust', or 'none' (identity).
        all_windows = np.array(windows)  # (num_windows, window_size, num_channels)
        cat_arr_temp = np.array(categories) if categories else None

        if cat_arr_temp is not None and 'training' in cat_arr_temp:
            train_mask = cat_arr_temp == 'training'
            train_windows = all_windows[train_mask]
        else:
            train_windows = all_windows

        # Flatten to (total_train_samples, num_channels) for stats
        train_flat = train_windows.reshape(-1, all_windows.shape[2])

        # Normalize the method string against known values (default to min_max)
        norm_method = (normalization_method or 'min_max').lower()
        if norm_method not in ('min_max', 'z_score', 'robust', 'none'):
            print(f"[DataLoader] Unknown normalization_method '{normalization_method}', falling back to 'min_max'")
            norm_method = 'min_max'

        if norm_method == 'min_max':
            ch_min = train_flat.min(axis=0)
            ch_max = train_flat.max(axis=0)
            ch_range = ch_max - ch_min

            # Constant-column dropping is only meaningful for min-max:
            # a channel with zero range provides no signal AND would divide-by-zero here.
            # For other methods we keep all channels (z_score guards std==0, robust guards iqr==0).
            active_mask = ch_range > 1e-10
            dropped_cols = [sensor_cols[i] for i in range(len(sensor_cols)) if not active_mask[i]]
            kept_cols = [sensor_cols[i] for i in range(len(sensor_cols)) if active_mask[i]]

            if dropped_cols:
                print(f"[DataLoader] Dropped {len(dropped_cols)} constant column(s): {dropped_cols}")

            all_windows = all_windows[:, :, active_mask]
            ch_min = ch_min[active_mask]
            ch_max = ch_max[active_mask]
            ch_range = ch_range[active_mask]

            # (x - min) / (max - min) → [0, 1]
            all_windows = (all_windows - ch_min) / ch_range

            sensor_cols = kept_cols
            norm_params = {
                'method': 'min_max',
                'channel_min': [float(v) for v in ch_min],
                'channel_max': [float(v) for v in ch_max],
                'sensor_columns': kept_cols,
                'dropped_columns': dropped_cols,
            }
        elif norm_method == 'z_score':
            ch_mean = train_flat.mean(axis=0)
            ch_std = train_flat.std(axis=0)
            # Guard against zero-std channels (constant columns): replace with 1.0
            # so the transform reduces to (x - mean), i.e. leaves the channel centered at 0.
            ch_std_safe = np.where(ch_std > 1e-10, ch_std, 1.0)

            dropped_cols = []
            kept_cols = list(sensor_cols)

            all_windows = (all_windows - ch_mean) / ch_std_safe

            norm_params = {
                'method': 'z_score',
                'channel_mean': [float(v) for v in ch_mean],
                'channel_std': [float(v) for v in ch_std_safe],
                'sensor_columns': kept_cols,
                'dropped_columns': dropped_cols,
            }
        elif norm_method == 'robust':
            # np.percentile returns array shape (3, num_channels)
            pct = np.percentile(train_flat, [25, 50, 75], axis=0)
            ch_q25 = pct[0]
            ch_median = pct[1]
            ch_q75 = pct[2]
            ch_iqr = ch_q75 - ch_q25
            ch_iqr_safe = np.where(ch_iqr > 1e-10, ch_iqr, 1.0)

            dropped_cols = []
            kept_cols = list(sensor_cols)

            all_windows = (all_windows - ch_median) / ch_iqr_safe

            norm_params = {
                'method': 'robust',
                'channel_median': [float(v) for v in ch_median],
                'channel_iqr': [float(v) for v in ch_iqr_safe],
                'sensor_columns': kept_cols,
                'dropped_columns': dropped_cols,
            }
        else:  # 'none' — identity, no scaling
            dropped_cols = []
            kept_cols = list(sensor_cols)
            # No transform applied to all_windows.
            norm_params = {
                'method': 'none',
                'sensor_columns': kept_cols,
                'dropped_columns': dropped_cols,
            }

        windows = list(all_windows)

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
            'no_windowing': bool(no_windowing),
            'window_size': int(window_size) if not no_windowing else 1,
            'stride': int(stride) if not no_windowing else 1,
            'overlap': overlap_pct if not no_windowing else 0,
            'num_windows': int(num_windows),
            'label_method': label_method if not no_windowing else 'direct',
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
