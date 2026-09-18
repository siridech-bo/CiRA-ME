"""CiRA ME — Solutions Analysis Job Queue (F8 Phase 1, 2026-09-18).

Solutions/*/run endpoints used to compute synchronously inside the gunicorn
request thread. With --workers 1 --threads 8, 40 concurrent workshop clicks
either (a) blocked all HTTP thread slots so /api/health went silent, or
(b) tripped gunicorn's --timeout kill and 502'd the whole worker.

This module decouples submission from computation:

  1. POST /api/solutions/<kind>/run  →  create_job() returns immediately with
     a UUID job_id. A daemon thread runs the actual heavy work behind a
     module-level Semaphore(_MAX_CONCURRENT), so at most N analyses use
     CPU at once. The other 40 - N sit in Python's queue, NOT in the HTTP
     thread pool.

  2. GET /api/solutions/jobs/<job_id>  →  get_job() returns the current
     snapshot: {status, queue_position, elapsed_s, result?, error?}.

State is per-process because gunicorn runs --workers 1 — same lifetime and
sharing model as _data_sessions, _publishers, _recording_jobs elsewhere in
this codebase. If you migrate to multi-worker gunicorn, this dict has to
move to Redis first (mirroring the memory note in the .103 runbook).

A janitor thread evicts jobs older than _JOB_TTL_SECONDS past their
finished_at, so the dict never grows unbounded during a workshop.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Concurrent analyses cap. 4 is deliberately conservative: it leaves at
# least half the 8 gunicorn threads free for cheap requests (/api/health,
# asset-tree browse, App Builder edits) so the app stays responsive even
# under a workshop storm. Bump via env var if the box has more headroom.
_MAX_CONCURRENT = 4

# How long a finished job (done or failed) is retained before the janitor
# evicts it. 30 min is comfortably longer than a Solutions flow: the user
# hits Analyze, waits for the poll to complete, reads results, saves the
# model. If they walk away for coffee, they can still fetch the job later.
_JOB_TTL_SECONDS = 30 * 60

# How often the janitor sweeps for expired jobs.
_JANITOR_INTERVAL_SECONDS = 60


_semaphore = threading.Semaphore(_MAX_CONCURRENT)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_janitor_started = False
_janitor_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _queue_position_locked(job_id: str) -> int:
    """Return this job's 1-based position in the queue.

    Position 0 means "not queued anymore" (running / done / failed).
    Positions are computed from submitted_at order — a job submitted
    later sees a higher number.

    MUST be called with _jobs_lock held.
    """
    target = _jobs.get(job_id)
    if not target or target.get('status') != 'queued':
        return 0
    submitted_at = target.get('submitted_at') or 0
    ahead = sum(
        1 for j in _jobs.values()
        if j.get('status') == 'queued'
        and (j.get('submitted_at') or 0) < submitted_at
    )
    return ahead + 1


def _janitor_loop() -> None:
    """Evict finished jobs whose TTL has expired.

    Runs forever, sleeps _JANITOR_INTERVAL_SECONDS between passes. Errors
    are swallowed with a log — the janitor thread must never die.
    """
    while True:
        try:
            now = _now()
            with _jobs_lock:
                expired = [
                    jid for jid, j in _jobs.items()
                    if j.get('status') in ('done', 'failed')
                    and j.get('finished_at') is not None
                    and now - j['finished_at'] > _JOB_TTL_SECONDS
                ]
                for jid in expired:
                    _jobs.pop(jid, None)
            if expired:
                logger.info(
                    '[solutions_jobs] evicted %d expired job(s)', len(expired),
                )
        except Exception:
            logger.exception('[solutions_jobs] janitor pass failed')
        time.sleep(_JANITOR_INTERVAL_SECONDS)


def _ensure_janitor_running() -> None:
    """Start the janitor on first submission (lazy init).

    Doing it here — not at import — avoids spinning a thread in test
    subprocesses or CLI scripts that import the module for a moment and
    then exit.
    """
    global _janitor_started
    if _janitor_started:
        return
    with _janitor_lock:
        if _janitor_started:
            return
        threading.Thread(
            target=_janitor_loop,
            name='solutions-jobs-janitor',
            daemon=True,
        ).start()
        _janitor_started = True


def _run_worker(job_id: str, target: Callable[[], Any]) -> None:
    """Execute one job.

    Blocks on the semaphore before starting the actual compute — this is
    how the queue mechanism works. The other N jobs sit here waiting for
    an active slot to free.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return  # cancelled / evicted before we got the slot
    with _semaphore:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job['status'] = 'running'
            job['started_at'] = _now()
        try:
            result = target()
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job:
                    job['status'] = 'done'
                    job['result'] = result
                    job['finished_at'] = _now()
        except Exception as e:
            logger.exception('[solutions_jobs] job %s failed: %s', job_id, e)
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job:
                    job['status'] = 'failed'
                    job['error'] = str(e)[:1000]
                    job['finished_at'] = _now()


def create_job(kind: str, user_id: Optional[int],
               target: Callable[[], Any]) -> str:
    """Register a new analysis job and hand it to a daemon thread.

    Returns the job_id (UUID hex). The caller returns this in the HTTP
    response with a 202; the client then polls get_job().
    """
    _ensure_janitor_running()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            'job_id': job_id,
            'kind': kind,
            'user_id': user_id,
            'status': 'queued',
            'submitted_at': _now(),
            'started_at': None,
            'finished_at': None,
            'result': None,
            'error': None,
        }
    threading.Thread(
        target=_run_worker,
        args=(job_id, target),
        name=f'solutions-job-{job_id[:8]}',
        daemon=True,
    ).start()
    return job_id


def get_job(job_id: str, user_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """Return a snapshot of a job's state, or None if not found / not the
    owner.

    The owner check is per-request: a job's user_id is set at create time
    and matched here. Anonymous jobs (user_id=None) are only visible to
    other anonymous callers, which shouldn't happen in practice because
    every submit path goes through @login_required.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        # Auth gate: caller must be the submitter.
        if job.get('user_id') != user_id:
            return None
        snapshot = {
            'job_id': job['job_id'],
            'kind': job['kind'],
            'status': job['status'],
            'submitted_at': job['submitted_at'],
            'started_at': job['started_at'],
            'finished_at': job['finished_at'],
            'queue_position': _queue_position_locked(job_id),
        }
        if job['started_at']:
            end = job['finished_at'] or _now()
            snapshot['elapsed_s'] = round(end - job['started_at'], 2)
        if job['status'] == 'done':
            snapshot['result'] = job['result']
        elif job['status'] == 'failed':
            snapshot['error'] = job['error']
        return snapshot


def stats() -> Dict[str, Any]:
    """Debug / observability helper. Returns cache-wide counters.

    Not user-facing but wired to /api/solutions/jobs/_stats for admin
    smoke tests during load spikes.
    """
    with _jobs_lock:
        counts = {'queued': 0, 'running': 0, 'done': 0, 'failed': 0}
        for j in _jobs.values():
            counts[j.get('status', 'queued')] = counts.get(
                j.get('status', 'queued'), 0
            ) + 1
        return {
            'total': len(_jobs),
            'by_status': counts,
            'max_concurrent': _MAX_CONCURRENT,
            'available_slots': _semaphore._value,  # type: ignore[attr-defined]
        }
