"""Folder Watcher service.

Per-watcher worker thread polls its input folder every N seconds, reads
each file row-by-row, runs each row through the associated ME-LAB
endpoint's model, writes an output CSV alongside, deletes the input.
"""

import threading
import time
import os
import csv
import glob
import logging
from datetime import datetime
from typing import Dict

from ..models import FolderWatcher, MeLabEndpoint, SavedModel
from .melab_service import ModelManager

logger = logging.getLogger(__name__)

# One worker per watcher. Keyed by watcher_id.
_workers: Dict[int, "_WatcherWorker"] = {}
_workers_lock = threading.Lock()

# Skip files whose mtime is within this window — probably still being written.
MTIME_QUIET_SECONDS = 5.0


class _WatcherWorker(threading.Thread):
    def __init__(self, watcher_id: int):
        super().__init__(daemon=True, name=f"folder-watcher-{watcher_id}")
        self.watcher_id = watcher_id
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        # Main loop: check stop flag, poll folder, sleep for poll_interval_s.
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
                except OSError as re:
                    logger.warning(
                        f"[FolderWatcher {self.watcher_id}] could not delete {path}: {re}"
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
        saved = SavedModel.get_by_id(endpoint['saved_model_id'])
        if not saved or not saved.get('model_path'):
            raise RuntimeError(
                f"model file missing for endpoint {watcher['endpoint_id']}"
            )
        model_data = ModelManager.load_model(saved['model_path'])
        mode = endpoint.get('mode', 'classification')

        # Detect / respect header mode
        header_mode = watcher.get('header_mode', 'auto')
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline().rstrip('\r\n')
        parts = first_line.split(',')
        if header_mode == 'auto':
            is_headered = bool(parts) and not all(_is_float(p) for p in parts)
        elif header_mode == 'headered':
            is_headered = True
        else:
            is_headered = False

        # Read all rows
        import numpy as np
        rows = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            if is_headered:
                f.readline()  # skip header
            for line in f:
                line = line.rstrip('\r\n')
                if not line:
                    continue
                vals = line.split(',')
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    continue  # skip non-numeric rows

        if not rows:
            return  # nothing to predict

        features = np.array(rows, dtype=np.float64)
        preds = ModelManager.predict(model_data, features, mode)

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
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'source_file', 'record_index', 'sensor_values',
                'prediction', 'confidence', 'predicted_at',
            ])
            ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            for i, (row, p) in enumerate(zip(rows, preds), start=1):
                sensor_str = '|'.join(str(v) for v in row)
                if isinstance(p, dict):
                    # ModelManager.predict returns dicts with 'label' (cls/anom)
                    # or 'value' (regression).
                    if 'label' in p:
                        label = p.get('label')
                    elif 'value' in p:
                        label = p.get('value')
                    else:
                        label = ''
                    conf = p.get('confidence', '')
                    if conf == '' and 'score' in p:
                        conf = p.get('score', '')
                else:
                    label = str(p)
                    conf = ''
                w.writerow([
                    os.path.basename(path), i, sensor_str, label, conf, ts,
                ])

        # Atomic-rename tmp → final. os.replace is portable and overwrites
        # cleanly on all platforms.
        os.replace(tmp_path, out_path)

        # Increment counters atomically
        FolderWatcher.increment_counters(
            self.watcher_id, files_delta=1, rows_delta=len(rows)
        )


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def start_watcher(watcher_id: int):
    """Start (or restart) a watcher. Idempotent within one process."""
    with _workers_lock:
        # Garbage-collect any stale dead entry from a previous run.
        existing = _workers.get(watcher_id)
        if existing and existing.is_alive():
            return  # already running
        if existing:
            _workers.pop(watcher_id, None)
        FolderWatcher.update(watcher_id, status='running', last_error=None)
        worker = _WatcherWorker(watcher_id)
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


def rehydrate_running_watchers():
    """Called from app.__init__.py at startup. Any watcher whose persisted
    status is 'running' gets its worker thread respawned."""
    try:
        for w in FolderWatcher.get_all_running():
            try:
                start_watcher(w['id'])
                logger.info(
                    f"[FolderWatcher] Rehydrated watcher {w['id']} ({w['name']})"
                )
            except Exception as e:
                logger.exception(
                    f"Failed to rehydrate watcher {w['id']}: {e}"
                )
    except Exception as e:
        logger.exception(f"rehydrate_running_watchers failed: {e}")
