"""Folder Watcher service.

Per-watcher worker thread polls its input folder every N seconds, reads
each file row-by-row, runs each row through the associated ME-LAB
endpoint's model, writes an output CSV alongside, deletes the input.

The parse layer supports three modes:
  csv     — one row per comma-separated line (legacy default).
  regex   — user-supplied Python regex with named capture groups.
            Each matching line becomes a row; non-matching lines skipped.
  json    — each line is a JSON object; keys become columns.

Optional prediction sinks (per-watcher):
  MQTT    — one message per row, published to a configurable topic.
  Daily   — an aggregated CSV per day at
            <SHARED_FOLDER_PATH>/log_watcher/<safe_name>/<YYYY-MM-DD>.csv.
Both sink failures are swallowed so a broker outage / disk hiccup can't
brick the worker.
"""

import threading
import time
import os
import csv
import glob
import json
import re
import logging
from datetime import datetime
from typing import Dict, List

from flask import current_app

from ..models import FolderWatcher, MeLabEndpoint, SavedModel
from .melab_service import ModelManager

logger = logging.getLogger(__name__)

# One worker per watcher. Keyed by watcher_id.
_workers: Dict[int, "_WatcherWorker"] = {}
_workers_lock = threading.Lock()

# Skip files whose mtime is within this window — probably still being written.
MTIME_QUIET_SECONDS = 5.0

# Per-file locks for the daily-aggregated-CSV sink so concurrent worker ticks
# (across watchers writing to overlapping files) don't interleave rows.
_daily_csv_locks: Dict[str, threading.Lock] = {}
_daily_csv_locks_guard = threading.Lock()


def _get_daily_csv_lock(path: str) -> threading.Lock:
    with _daily_csv_locks_guard:
        lock = _daily_csv_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _daily_csv_locks[path] = lock
        return lock


class _WatcherWorker(threading.Thread):
    def __init__(self, watcher_id: int, flask_app=None):
        super().__init__(daemon=True, name=f"folder-watcher-{watcher_id}")
        self.watcher_id = watcher_id
        self._stop_flag = threading.Event()
        # Capture the Flask app object so the worker thread can push an
        # app context — required for current_app.config reads inside the
        # MQTT / daily-CSV sink helpers.
        self._flask_app = flask_app

    def stop(self):
        self._stop_flag.set()

    def run(self):
        # Main loop: check stop flag, poll folder, sleep for poll_interval_s.
        # An app context is pushed so current_app.config['SHARED_FOLDER_PATH']
        # etc. work inside sink helpers, matching how the Flask request layer
        # would provide it during a normal HTTP call.
        if self._flask_app is not None:
            self._flask_app.app_context().push()
        while not self._stop_flag.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.exception(f"[FolderWatcher {self.watcher_id}] tick failed: {e}")
                # Don't clobber a 'stopped' status flip that stop_watcher may
                # have set concurrently while we were in _tick(). Only record
                # the error if the DB still thinks we're running.
                current = FolderWatcher.get_by_id(self.watcher_id)
                if current and current.get('status') == 'running':
                    FolderWatcher.update(
                        self.watcher_id,
                        status='error',
                        last_error=str(e)[:500],
                    )
            # Reload state — watcher may have been stopped or edited during tick
            watcher = FolderWatcher.get_by_id(self.watcher_id)
            if not watcher or watcher.get('status') != 'running':
                break
            interval = int(watcher.get('poll_interval_s', 60) or 60)
            # Sleep in short slices so stop is responsive
            for _ in range(interval * 2):
                if self._stop_flag.is_set():
                    break
                time.sleep(0.5)

    def _tick(self):
        # Reload state each tick — watcher config may have changed
        watcher = FolderWatcher.get_by_id(self.watcher_id)
        if not watcher:
            self._stop_flag.set()
            return
        input_dir = watcher['input_folder']
        output_dir = watcher['output_folder']
        if not os.path.isdir(input_dir):
            os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        error_dir = os.path.join(input_dir, '_error')

        # Endpoint-level health check — done once per tick, before any files.
        # A paused/deleted endpoint would cause every single file to fail with
        # the same error and pile up in _error/. Raising here bubbles to the
        # tick-level except handler in run(), which flips the watcher status
        # to 'error' with an actionable message so the user sees the problem
        # instead of just watching their input folder empty itself into _error/.
        endpoint = MeLabEndpoint.get_by_id(watcher['endpoint_id'])
        if not endpoint:
            raise RuntimeError(
                f"endpoint {watcher['endpoint_id']} was deleted — stop this watcher"
            )
        if endpoint.get('status') != 'active':
            raise RuntimeError(
                f"endpoint {watcher['endpoint_id']} is {endpoint.get('status')}, "
                f"not active — cannot predict. Reactivate it in ME-LAB, or "
                f"point this watcher at a different endpoint."
            )

        # Gather candidate files
        pattern = os.path.join(input_dir, watcher.get('file_glob') or '*.txt')
        candidates = sorted(glob.glob(pattern))
        now = time.time()
        for path in candidates:
            if self._stop_flag.is_set():
                break
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now - mtime < MTIME_QUIET_SECONDS:
                continue  # still being written, try next tick
            try:
                self._process_file(path, output_dir, watcher)
                try:
                    os.remove(path)
                except OSError as re_:
                    logger.warning(
                        f"[FolderWatcher {self.watcher_id}] could not delete {path}: {re_}"
                    )
            except Exception as e:
                logger.exception(
                    f"[FolderWatcher {self.watcher_id}] file failed: {path}"
                )
                os.makedirs(error_dir, exist_ok=True)
                try:
                    os.rename(
                        path, os.path.join(error_dir, os.path.basename(path))
                    )
                except OSError:
                    pass
        FolderWatcher.update(
            self.watcher_id, last_run_at=datetime.utcnow().isoformat()
        )

    def _process_file(self, path: str, output_dir: str, watcher: dict):
        # Load the ME-LAB model. Refuse to run against a paused / deleted
        # endpoint — otherwise the worker would silently keep predicting or
        # move every file to _error/ on each tick.
        endpoint = MeLabEndpoint.get_by_id(watcher['endpoint_id'])
        if not endpoint:
            raise RuntimeError(
                f"endpoint {watcher['endpoint_id']} was deleted — stop this watcher"
            )
        if endpoint.get('status') != 'active':
            raise RuntimeError(
                f"endpoint {watcher['endpoint_id']} is {endpoint.get('status')}, "
                f"not active — cannot predict"
            )
        # Pre-flight: confirm the saved model file still exists so we can raise
        # a clean, actionable error before we spend time parsing the CSV.
        saved = SavedModel.get_by_id(endpoint['saved_model_id'])
        if not saved or not saved.get('model_path'):
            raise RuntimeError(
                f"model file missing for endpoint {watcher['endpoint_id']}"
            )

        # ── Dispatch on parse_mode ─────────────────────────────────────────
        # rows_raw is List[Dict[col_name, value]] — column_names is the header
        # for the output CSV in insertion order.
        parse_mode = (watcher.get('parse_mode') or 'csv').lower()
        if parse_mode == 'regex':
            column_names, rows_raw = _parse_regex(path, watcher.get('parse_regex') or '')
        elif parse_mode == 'json':
            column_names, rows_raw = _parse_json(path)
        elif parse_mode == 'key_value':
            columns_split = _split_columns(watcher.get('parse_columns') or '')
            column_names, rows_raw = _parse_key_value(path, columns_split)
        else:
            column_names, rows_raw = _parse_csv(path, watcher.get('header_mode', 'auto'))

        if not rows_raw:
            return  # nothing to predict

        # Build the numeric feature matrix in the ORDER of column_names. Any
        # non-numeric value is dropped from that row (replaced with 0) so the
        # matrix is rectangular, matching the model's expected feature width.
        # In practice regex/json filter to numeric columns upstream so this
        # only matters for CSV rows that hit `_parse_csv` with mixed content.
        import numpy as np
        features = np.array(
            [[_to_float(r.get(c, '')) for c in column_names] for r in rows_raw],
            dtype=np.float64,
        )
        # Use the canonical endpoint-scoped inference so label decoding +
        # counter bookkeeping stay in one place (see ModelManager.predict_by_endpoint).
        preds = ModelManager.predict_by_endpoint(watcher['endpoint_id'], features)

        # Write output CSV with the columns the customer's diagram specified.
        # If output already exists (customer re-uploaded same filename), rotate.
        # Cap counter so a runaway loop can't O(N) itself; after that, fall
        # back to a timestamped name that's guaranteed unique.
        out_name = os.path.basename(path)
        out_path = os.path.join(output_dir, out_name)
        base_stem, ext = os.path.splitext(out_name)
        MAX_COLLISION_TRIES = 1000
        counter = 1
        while os.path.exists(out_path) and counter <= MAX_COLLISION_TRIES:
            out_path = os.path.join(output_dir, f"{base_stem}_{counter}{ext}")
            counter += 1
        if os.path.exists(out_path):
            # Fallback: timestamp millis. Effectively guaranteed unique.
            ts_ms = int(time.time() * 1000)
            out_path = os.path.join(output_dir, f"{base_stem}_{ts_ms}{ext}")

        # Write to a .tmp file first, then atomic-rename into place. This
        # avoids leaving a partially-written CSV if we crash or the process
        # gets SIGKILL'd mid-write. Input is NOT deleted here — the caller
        # (_tick) deletes only after this method returns cleanly.
        tmp_path = out_path + '.tmp'
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        source_basename = os.path.basename(path)
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'source_file', 'record_index', 'sensor_values',
                'prediction', 'confidence', 'predicted_at',
            ])
            for i, (row, p) in enumerate(zip(rows_raw, preds), start=1):
                sensor_str = '|'.join(f"{c}={row.get(c, '')}" for c in column_names)
                label, conf = _extract_label_and_conf(p)
                w.writerow([
                    source_basename, i, sensor_str, label, conf, ts,
                ])

        # Atomic-rename tmp → final. os.replace is portable and overwrites
        # cleanly on all platforms.
        os.replace(tmp_path, out_path)

        # ── Prediction sinks ────────────────────────────────────────────
        # Both are best-effort — failures logged, never raised, so a broker
        # outage / disk hiccup can't derail the pipeline.
        try:
            self._emit_sinks(watcher, source_basename, column_names, rows_raw, preds)
        except Exception as e:
            logger.warning(
                f"[FolderWatcher {self.watcher_id}] sink dispatch failed: {e}"
            )

        # Increment counters atomically
        FolderWatcher.increment_counters(
            self.watcher_id, files_delta=1, rows_delta=len(rows_raw)
        )

    # ------------------------------------------------------------------
    # Sinks: MQTT publish + daily aggregated CSV
    # ------------------------------------------------------------------
    def _emit_sinks(
        self,
        watcher: dict,
        source_basename: str,
        column_names: List[str],
        rows_raw: List[dict],
        preds: list,
    ):
        mqtt_enabled = bool(watcher.get('mqtt_enabled'))
        daily_enabled = bool(watcher.get('daily_csv_enabled'))
        if not mqtt_enabled and not daily_enabled:
            return

        watcher_name = watcher.get('name') or f"watcher_{watcher['id']}"
        raw_topic = watcher.get('mqtt_topic') or f'alerts/{watcher_name}'
        topic = str(raw_topic).replace('{name}', _slug(watcher_name))

        # Resolve the daily-CSV directory once. current_app.config reads
        # require the app context which run() pushes at thread start.
        daily_dir = None
        if daily_enabled:
            try:
                datasets_root = current_app.config['DATASETS_ROOT_PATH']
                shared_folder = current_app.config['SHARED_FOLDER_PATH']
                daily_dir = os.path.join(
                    datasets_root, shared_folder, 'log_watcher', _slug(watcher_name)
                )
                os.makedirs(daily_dir, exist_ok=True)
            except Exception as e:
                logger.warning(
                    f"[FolderWatcher {self.watcher_id}] daily-CSV dir setup failed: {e}"
                )
                daily_dir = None

        # One dedicated MQTT client for the whole file: cheaper than one
        # ephemeral connect per row, still short-lived per file.
        mqtt_client = None
        if mqtt_enabled:
            try:
                mqtt_client = _mqtt_connect()
            except Exception as e:
                logger.warning(
                    f"[FolderWatcher {self.watcher_id}] MQTT connect failed: {e}"
                )
                mqtt_client = None

        try:
            for idx, (row, p) in enumerate(zip(rows_raw, preds), start=1):
                label, conf = _extract_label_and_conf(p)
                iso_ts = datetime.utcnow().isoformat() + 'Z'
                # Publish payload built once and reused by both sinks.
                payload = {
                    'timestamp': iso_ts,
                    'watcher_name': watcher_name,
                    'source_file': source_basename,
                    'record_index': idx,
                    'prediction': label,
                    'confidence': conf if conf != '' else None,
                    'row': {c: row.get(c) for c in column_names},
                }

                if mqtt_client is not None:
                    try:
                        mqtt_client.publish(
                            topic, json.dumps(payload, default=str), qos=0
                        )
                    except Exception as e:
                        logger.warning(
                            f"[FolderWatcher {self.watcher_id}] MQTT publish row {idx} failed: {e}"
                        )

                if daily_dir is not None:
                    try:
                        _append_daily_row(daily_dir, watcher_name, payload, column_names)
                    except Exception as e:
                        logger.warning(
                            f"[FolderWatcher {self.watcher_id}] daily-CSV append row {idx} failed: {e}"
                        )
        finally:
            if mqtt_client is not None:
                try:
                    mqtt_client.loop_stop()
                except Exception:
                    pass
                try:
                    mqtt_client.disconnect()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Parsers — each returns (column_names, rows) where rows is List[Dict].
# ---------------------------------------------------------------------------

def _parse_csv(path: str, header_mode: str):
    """Legacy CSV parser. Non-numeric rows are skipped.
    Column names are either the first-row header or auto-generated (col_0..).
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline().rstrip('\r\n')
    parts = first_line.split(',')
    if header_mode == 'auto':
        is_headered = bool(parts) and not all(_is_float(p) for p in parts)
    elif header_mode == 'headered':
        is_headered = True
    else:
        is_headered = False

    if is_headered:
        column_names = [p.strip() or f'col_{i}' for i, p in enumerate(parts)]
    else:
        column_names = [f'col_{i}' for i in range(len(parts))]

    rows: List[dict] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        if is_headered:
            f.readline()  # skip header
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            vals = line.split(',')
            try:
                floats = [float(v) for v in vals]
            except ValueError:
                continue  # skip non-numeric rows
            row = {}
            for i, v in enumerate(floats):
                name = column_names[i] if i < len(column_names) else f'col_{i}'
                row[name] = v
            rows.append(row)
    return column_names, rows


def _parse_regex(path: str, pattern: str):
    """Regex parser. Each matching line's named groups become a row.
    Non-matching lines are skipped. Non-numeric captures are dropped.

    Column order is the order named groups first appear in the pattern; a row
    is emitted only if it retains at least one numeric column.
    """
    if not pattern:
        return [], []
    compiled = re.compile(pattern)
    # Named groups in definition order — used as the canonical column list so
    # the output CSV has a stable header even if some rows omit groups.
    ordered_names = [
        n for n, i in sorted(compiled.groupindex.items(), key=lambda x: x[1])
    ]
    if not ordered_names:
        # Regex with no named groups can't yield columns; treat as empty.
        return [], []
    rows: List[dict] = []
    seen_columns: List[str] = []
    seen_set = set()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            m = compiled.search(line)
            if not m:
                continue
            row = {}
            for name in ordered_names:
                v = m.group(name) if name in m.groupdict() else None
                if v is None:
                    continue
                fv = _try_float(v)
                if fv is None:
                    continue  # drop non-numeric
                row[name] = fv
                if name not in seen_set:
                    seen_set.add(name)
                    seen_columns.append(name)
            if row:
                rows.append(row)
    return seen_columns, rows


def _parse_json(path: str):
    """JSON-lines parser. Each line is one JSON object; keys become columns.
    Non-numeric values are dropped. Column order follows first-seen key order.
    """
    rows: List[dict] = []
    seen_columns: List[str] = []
    seen_set = set()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            row = {}
            for k, v in obj.items():
                fv = _try_float(v)
                if fv is None:
                    continue
                key = str(k)
                row[key] = fv
                if key not in seen_set:
                    seen_set.add(key)
                    seen_columns.append(key)
            if row:
                rows.append(row)
    return seen_columns, rows


def _split_columns(raw: str) -> List[str]:
    """Split a comma-separated `parse_columns` string into a list of names,
    stripping whitespace and dropping empty entries."""
    if not raw:
        return []
    return [c.strip() for c in str(raw).split(',') if c.strip()]


def _parse_key_value(path: str, columns: List[str]):
    """Key = Value parser (file variant). See _parse_key_value_content for
    the actual matching logic; this wrapper just reads the file."""
    if not columns:
        return [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    col_names, rows = _parse_key_value_content(content, columns)
    # Convert numeric-row-of-floats format into a List[Dict] to match the
    # signature of the other file parsers, which _process_file consumes.
    dict_rows: List[dict] = []
    import math
    for row in rows:
        d = {}
        for i, name in enumerate(col_names):
            if i < len(row):
                v = row[i]
                # Drop NaN so downstream _to_float defaults them to 0.0
                if not (isinstance(v, float) and math.isnan(v)):
                    d[name] = v
        if d:
            dict_rows.append(d)
    return col_names, dict_rows


# ---------------------------------------------------------------------------
# Content-based parsers (used by the preview endpoint) — take raw text
# instead of a file path so we can test parse configs before saving.
# Return (columns: List[str], rows: List[List[float]]) — matrix form, not
# dict-of-rows — since the preview UI wants a flat table with NaNs preserved.
# ---------------------------------------------------------------------------

def _parse_key_value_content(content: str, columns: List[str]):
    """Extract key=value / key:value pairs from each line.

    Skips lines where NO column matched. Missing columns for a row get NaN.
    Common separators supported: `=`, `:`, `: `. Column matching is
    case-insensitive; unit suffixes after the number (e.g. `45.32°C`,
    `0.87g`) are tolerated because we only anchor on the leading number.
    """
    if not columns:
        return [], []
    patterns = {
        col: re.compile(
            rf'\b{re.escape(col)}\s*[=:]\s*(-?\d+\.?\d*)',
            re.IGNORECASE,
        )
        for col in columns
    }
    rows: List[List[float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        row: List[float] = []
        matched = 0
        for col in columns:
            m = patterns[col].search(line)
            if m:
                try:
                    row.append(float(m.group(1)))
                    matched += 1
                    continue
                except ValueError:
                    pass
            row.append(float('nan'))
        if matched > 0:
            rows.append(row)
    return list(columns), rows


def _parse_regex_content(content: str, pattern: str):
    """Regex parser, string-in variant. Same semantics as _parse_regex."""
    if not pattern:
        return [], []
    compiled = re.compile(pattern)
    ordered_names = [
        n for n, i in sorted(compiled.groupindex.items(), key=lambda x: x[1])
    ]
    if not ordered_names:
        return [], []
    rows: List[List[float]] = []
    seen_columns: List[str] = []
    seen_set = set()
    for line in content.splitlines():
        line = line.rstrip('\r\n')
        if not line:
            continue
        m = compiled.search(line)
        if not m:
            continue
        row_dict = {}
        for name in ordered_names:
            v = m.group(name) if name in m.groupdict() else None
            if v is None:
                continue
            fv = _try_float(v)
            if fv is None:
                continue
            row_dict[name] = fv
            if name not in seen_set:
                seen_set.add(name)
                seen_columns.append(name)
        if row_dict:
            # Materialize with the columns we've seen so far — later rows may
            # widen the set, so pad missing columns with NaN at return time.
            rows.append(row_dict)
    # Materialize into a rectangular matrix using the union of seen columns.
    matrix: List[List[float]] = []
    for row_dict in rows:
        matrix.append([
            row_dict.get(c, float('nan')) for c in seen_columns
        ])
    return seen_columns, matrix


def _parse_json_content(content: str):
    """JSON-lines parser, string-in variant. Same semantics as _parse_json."""
    row_dicts: List[dict] = []
    seen_columns: List[str] = []
    seen_set = set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        row = {}
        for k, v in obj.items():
            fv = _try_float(v)
            if fv is None:
                continue
            key = str(k)
            row[key] = fv
            if key not in seen_set:
                seen_set.add(key)
                seen_columns.append(key)
        if row:
            row_dicts.append(row)
    matrix: List[List[float]] = []
    for row in row_dicts:
        matrix.append([row.get(c, float('nan')) for c in seen_columns])
    return seen_columns, matrix


def _parse_csv_content(content: str, header_mode: str):
    """CSV parser, string-in variant. Same semantics as _parse_csv."""
    lines = content.splitlines()
    if not lines:
        return [], []
    first_line = lines[0].rstrip('\r\n')
    parts = first_line.split(',')
    if header_mode == 'auto':
        is_headered = bool(parts) and not all(_is_float(p) for p in parts)
    elif header_mode == 'headered':
        is_headered = True
    else:
        is_headered = False

    if is_headered:
        column_names = [p.strip() or f'col_{i}' for i, p in enumerate(parts)]
        data_lines = lines[1:]
    else:
        column_names = [f'col_{i}' for i in range(len(parts))]
        data_lines = lines

    rows: List[List[float]] = []
    for line in data_lines:
        line = line.rstrip('\r\n')
        if not line:
            continue
        vals = line.split(',')
        try:
            floats = [float(v) for v in vals]
        except ValueError:
            continue
        # Pad / truncate to header width so the matrix stays rectangular
        if len(floats) < len(column_names):
            floats = floats + [float('nan')] * (len(column_names) - len(floats))
        elif len(floats) > len(column_names):
            floats = floats[:len(column_names)]
        rows.append(floats)
    return column_names, rows


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _try_float(v):
    """Return float(v) or None if it doesn't parse. Booleans are NOT coerced
    (Python `True`→1.0 would silently poison feature vectors)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    if isinstance(v, str):
        try:
            return float(v.strip())
        except (ValueError, TypeError):
            return None
    return None


def _to_float(v):
    """Feature-matrix coercion: unknown → 0.0. Used only when a column is
    present in the header but missing on a specific row."""
    f = _try_float(v)
    return 0.0 if f is None else f


def _extract_label_and_conf(p):
    """ModelManager returns dicts (label/value + optional confidence/score)
    for structured outputs and raw scalars/strings for simple algorithms.
    Normalize to (label_str, confidence_str) for CSV + MQTT payloads."""
    if isinstance(p, dict):
        if 'label' in p:
            label = p.get('label')
        elif 'value' in p:
            label = p.get('value')
        else:
            label = ''
        conf = p.get('confidence', '')
        if conf == '' and 'score' in p:
            conf = p.get('score', '')
        return label, conf
    return str(p), ''


_SLUG_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _slug(name: str) -> str:
    """Filesystem-safe watcher slug for daily-CSV subdirs. Also safe as MQTT
    topic segment."""
    if not name:
        return 'watcher'
    return _SLUG_RE.sub('_', name).strip('_') or 'watcher'


def _mqtt_connect():
    """Open a live paho.mqtt client. Caller must call loop_stop + disconnect.
    Reuses the broker host/port env vars matching Batch B's app_builder sink."""
    import paho.mqtt.client as paho_mqtt

    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))
    client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker_host, broker_port, keepalive=5)
    client.loop_start()
    return client


def _append_daily_row(
    daily_dir: str, watcher_name: str, payload: dict, column_names: List[str]
):
    """Append one row to <daily_dir>/<YYYY-MM-DD>.csv.
    Header is written on first-touch of that day's file. Uses a per-path
    lock so concurrent worker ticks don't interleave writes."""
    day = datetime.utcnow().strftime('%Y-%m-%d')
    file_path = os.path.join(daily_dir, f'{day}.csv')

    row = {
        'timestamp': payload.get('timestamp', ''),
        'watcher_name': payload.get('watcher_name', watcher_name),
        'source_file': payload.get('source_file', ''),
        'record_index': payload.get('record_index', ''),
        'prediction': payload.get('prediction', ''),
        'confidence': payload.get('confidence', ''),
    }
    input_row = payload.get('row') or {}
    for c in column_names:
        row[f'in_{c}'] = input_row.get(c, '')

    lock = _get_daily_csv_lock(file_path)
    with lock:
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or os.path.getsize(file_path) == 0:
                writer.writeheader()
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

def start_watcher(watcher_id: int, flask_app=None):
    """Start (or restart) a watcher. Idempotent within one process.

    `flask_app` may be passed explicitly (e.g. from the app factory's
    rehydration call, which runs outside a request/app context). If omitted,
    we fall back to `current_app` — which is available during any HTTP
    request that triggers /start.
    """
    if flask_app is None:
        try:
            flask_app = current_app._get_current_object()  # type: ignore[attr-defined]
        except RuntimeError:
            flask_app = None

    with _workers_lock:
        # Garbage-collect any stale dead entry from a previous run.
        existing = _workers.get(watcher_id)
        if existing and existing.is_alive():
            return  # already running
        if existing:
            _workers.pop(watcher_id, None)
        FolderWatcher.update(watcher_id, status='running', last_error=None)
        worker = _WatcherWorker(watcher_id, flask_app=flask_app)
        _workers[watcher_id] = worker
        worker.start()


# Join timeout accepts that we may SIGKILL a worker mid-file on Docker
# shutdown. The design is fault-tolerant: input files are only deleted after
# the output CSV is atomically committed (see _process_file), so a killed
# worker leaves the input file in place and next boot re-processes it. 30s
# is enough for most mid-file predicts to finish cleanly on typical models.
_JOIN_TIMEOUT_SECONDS = 30.0


def stop_watcher(watcher_id: int):
    with _workers_lock:
        worker = _workers.pop(watcher_id, None)
    if worker:
        worker.stop()
        worker.join(timeout=_JOIN_TIMEOUT_SECONDS)
    FolderWatcher.update(watcher_id, status='stopped')


def rehydrate_running_watchers(flask_app=None):
    """Called from app.__init__.py at startup. Any watcher whose persisted
    status is 'running' gets its worker thread respawned.

    `flask_app` should be the Flask app object; it's threaded through so each
    worker's context-push at run() can succeed (rehydration itself doesn't
    execute inside an app context)."""
    try:
        for w in FolderWatcher.get_all_running():
            try:
                start_watcher(w['id'], flask_app=flask_app)
                logger.info(
                    f"[FolderWatcher] Rehydrated watcher {w['id']} ({w['name']})"
                )
            except Exception as e:
                logger.exception(
                    f"Failed to rehydrate watcher {w['id']}: {e}"
                )
    except Exception as e:
        logger.exception(f"rehydrate_running_watchers failed: {e}")
