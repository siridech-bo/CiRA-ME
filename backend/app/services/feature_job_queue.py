"""
CiRA ME - Feature Extraction Job Queue

Bounded ThreadPoolExecutor (5 slots) that runs tsfresh feature extraction
asynchronously so the Flask process never has more than 5 heavy CPU jobs in
flight. Keeps unrelated endpoints (health checks, MQTT publisher heartbeats,
Log Watcher polling, etc.) responsive under workshop-scale concurrent load.

Phase 1 of the Problem 2 fix from
docs/PLAN_2026-07-11_mqtt-robustness_feature-hybrid.md.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------------
MAX_WORKERS = 5
DEFAULT_AVG_SECONDS = 60          # used until we have completed jobs to average
ROLLING_WINDOW_JOBS = 10          # rolling average of last N completed jobs
JANITOR_INTERVAL_SECONDS = 300    # 5 min
TTL_SECONDS = 30 * 60             # 30 min after completion

# Statuses
STATUS_QUEUED = 'queued'
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_ERROR = 'error'
STATUS_CANCELLED = 'cancelled'

TERMINAL_STATUSES = {STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED}


# ---- Worker callable --------------------------------------------------------
# The worker imports the extractor lazily to avoid circular imports at module
# import time (features.py imports this module).
def _run_extraction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the synchronous tsfresh + DSP extraction and return the result dict.

    Runs inside a worker thread. Reused body of the old sync /extract handler.
    Raises on failure — the caller stores the message.
    """
    from ..routes.features import _do_extract  # deferred to avoid circular
    return _do_extract(payload)


# ---- Registry entry ---------------------------------------------------------
@dataclass
class _Job:
    job_id: str
    user_id: Optional[int]
    payload: Dict[str, Any]
    status: str = STATUS_QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Future returned by the executor.submit call. Kept so we can check
    # future.done() / future.cancel(). Not exposed externally.
    future: Optional[Future] = None
    cancel_requested: bool = False


# ---- Job queue --------------------------------------------------------------
class FeatureJobQueue:
    """In-memory bounded async job queue for feature extraction.

    Not multi-process safe — assumes a single Flask worker (project runs
    gunicorn --workers 1). See docs/PLAN_2026-07-11 follow-up ideas for the
    Redis-backed variant.
    """

    def __init__(
        self,
        max_workers: int = MAX_WORKERS,
        worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = _run_extraction,
    ):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='feature-job',
        )
        self._worker_fn = worker_fn
        self._max_workers = max_workers
        self._jobs: Dict[str, _Job] = {}
        self._lock = threading.Lock()
        # Rolling window of completion durations (seconds).
        self._completion_history: List[float] = []
        self._janitor_thread = threading.Thread(
            target=self._janitor_loop,
            name='feature-job-janitor',
            daemon=True,
        )
        self._janitor_thread.start()
        logger.info(
            f'[feature_job_queue] initialized max_workers={max_workers} ttl={TTL_SECONDS}s'
        )

    # ---- Public API ---------------------------------------------------------
    def submit(self, payload: Dict[str, Any], user_id: Optional[int]) -> str:
        """Enqueue a job and return its id."""
        job_id = str(uuid.uuid4())
        job = _Job(job_id=job_id, user_id=user_id, payload=payload)
        with self._lock:
            self._jobs[job_id] = job
        # Submit outside the lock (submit itself is cheap but keep lock scope tight)
        future = self._executor.submit(self._worker_wrapper, job_id)
        with self._lock:
            job.future = future
        logger.info(
            f'[feature_job_queue] submitted job_id={job_id} user_id={user_id} '
            f'queued={self._count_status(STATUS_QUEUED)} running={self._count_status(STATUS_RUNNING)}'
        )
        return job_id

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a serializable status dict, or None for unknown ids."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            out: Dict[str, Any] = {
                'status': job.status,
                'submitted_at': job.submitted_at,
            }
            if job.started_at is not None:
                out['started_at'] = job.started_at
            if job.completed_at is not None:
                out['completed_at'] = job.completed_at
            if job.status == STATUS_QUEUED:
                out['queue_position'] = self._queue_position_locked(job_id)
            if job.status == STATUS_DONE:
                out['features'] = job.result
            if job.status == STATUS_ERROR:
                out['error'] = job.error
            return out

    def cancel(self, job_id: str) -> Tuple[bool, str]:
        """Attempt to cancel a job.

        Returns:
          (True,  'cancelled')             — successfully cancelled a queued job
          (True,  'noop_already_done')     — job already terminal, no-op
          (False, 'unknown_job')           — id not in registry
          (False, 'cannot_cancel_running') — running jobs cannot be interrupted
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return (False, 'unknown_job')
            if job.status in TERMINAL_STATUSES:
                return (True, 'noop_already_done')
            if job.status == STATUS_RUNNING:
                # Best-effort: the tsfresh call is CPU-bound and doesn't check
                # for interruption. We flag the intent so the wrapper can drop
                # the result on completion, but the thread will keep running.
                job.cancel_requested = True
                return (False, 'cannot_cancel_running')
            # STATUS_QUEUED — try to cancel the future.
            future = job.future
            if future is not None and future.cancel():
                job.status = STATUS_CANCELLED
                job.completed_at = time.time()
                logger.info(f'[feature_job_queue] cancelled job_id={job_id}')
                return (True, 'cancelled')
            # Future couldn't be cancelled — worker probably just picked it up.
            job.cancel_requested = True
            return (False, 'cannot_cancel_running')

    def estimate_wait_seconds(self, job_id: str) -> int:
        """Return an estimated wait time in seconds for a queued job.

        For non-queued jobs, returns 0.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != STATUS_QUEUED:
                return 0
            position = self._queue_position_locked(job_id)
            avg = self._avg_completion_seconds_locked()
            return int(position * avg)

    def queue_position(self, job_id: str) -> int:
        """Return this job's position in the queue (0-based)."""
        with self._lock:
            return self._queue_position_locked(job_id)

    def avg_completion_seconds(self) -> float:
        with self._lock:
            return self._avg_completion_seconds_locked()

    def snapshot(self) -> Dict[str, Any]:
        """Small dict suitable for /api/health."""
        with self._lock:
            return {
                'workers_max': self._max_workers,
                'queued': self._count_status(STATUS_QUEUED),
                'running': self._count_status(STATUS_RUNNING),
            }

    # ---- Internals ----------------------------------------------------------
    def _worker_wrapper(self, job_id: str) -> None:
        """Executor entrypoint: runs on a worker thread."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            # If cancelled between submit and start (rare — future.cancel raced),
            # bail out cleanly.
            if job.status == STATUS_CANCELLED or job.cancel_requested:
                if job.status != STATUS_CANCELLED:
                    job.status = STATUS_CANCELLED
                    job.completed_at = time.time()
                return
            job.status = STATUS_RUNNING
            job.started_at = time.time()
            payload = job.payload

        try:
            result = self._worker_fn(payload)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job.cancel_requested:
                    # User cancelled mid-run — discard result silently.
                    job.status = STATUS_CANCELLED
                    job.completed_at = time.time()
                    return
                job.status = STATUS_DONE
                job.result = result
                job.completed_at = time.time()
                if job.started_at is not None:
                    self._record_completion_locked(job.completed_at - job.started_at)
        except Exception as e:  # noqa: BLE001 — worker catch-all
            logger.exception(f'[feature_job_queue] job_id={job_id} raised: {e}')
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.status = STATUS_ERROR
                job.error = str(e)
                job.completed_at = time.time()

    def _queue_position_locked(self, job_id: str) -> int:
        """0 if next in line (or already running-slot-available), else count of
        earlier queued jobs.

        Must be called with self._lock held.
        """
        target = self._jobs.get(job_id)
        if target is None or target.status != STATUS_QUEUED:
            return 0
        ahead = 0
        for j in self._jobs.values():
            if j.job_id == job_id:
                continue
            if j.status == STATUS_QUEUED and j.submitted_at < target.submitted_at:
                ahead += 1
        return ahead

    def _count_status(self, status: str) -> int:
        # Caller must hold lock.
        return sum(1 for j in self._jobs.values() if j.status == status)

    def _avg_completion_seconds_locked(self) -> float:
        if not self._completion_history:
            return float(DEFAULT_AVG_SECONDS)
        return sum(self._completion_history) / len(self._completion_history)

    def _record_completion_locked(self, seconds: float) -> None:
        self._completion_history.append(seconds)
        if len(self._completion_history) > ROLLING_WINDOW_JOBS:
            # Drop oldest.
            self._completion_history = self._completion_history[-ROLLING_WINDOW_JOBS:]

    def _janitor_loop(self) -> None:
        """Daemon: evict terminal jobs older than TTL_SECONDS."""
        while True:
            try:
                time.sleep(JANITOR_INTERVAL_SECONDS)
                self._janitor_sweep()
            except Exception as e:  # noqa: BLE001 — never let the thread die
                logger.exception(f'[feature_job_queue] janitor error: {e}')

    def _janitor_sweep(self) -> None:
        cutoff = time.time() - TTL_SECONDS
        evicted = 0
        with self._lock:
            stale_ids = [
                jid
                for jid, j in self._jobs.items()
                if j.status in TERMINAL_STATUSES
                and j.completed_at is not None
                and j.completed_at < cutoff
            ]
            for jid in stale_ids:
                self._jobs.pop(jid, None)
                evicted += 1
        if evicted:
            logger.info(f'[feature_job_queue] janitor evicted {evicted} stale jobs')


# Module-level singleton.
feature_job_queue = FeatureJobQueue()
