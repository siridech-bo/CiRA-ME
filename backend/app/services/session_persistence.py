"""
CiRA ME - Session Persistence Service (Approach 2b)

Persists in-memory pipeline session dicts (_data_sessions from data_loader,
_feature_sessions from feature_extractor) to per-project pickle files, so the
data itself survives backend restarts, session expiry, and container rebuilds.

Motivation:
    The DB tracks pipeline stage completion (data_sessions / windowed_sessions
    / feature_sessions rows) but only stores metadata. Actual DataFrames /
    ndarrays live in in-process dicts. When the backend restarts (workshop
    container rebuild, session timeout re-login, crash), the dicts wipe but the
    DB still says "features complete", so the Projects list shows a green
    Features chip while every downstream route 404s with
    "Feature session not found: features_..." .

    Under a 65-attendee workshop, forcing every user to redo Windowing +
    Extract Features on re-login is unacceptable. Pickling the session dict
    entry after each stage — and reloading it on hydrate — costs ~16 MB per
    project on disk (~1 GB max for 65 workshop projects) and near-zero CPU on
    hydrate. Same pattern as the existing SQLite metadata rows, extended to the
    payload.

Storage layout:
    <base>/<project_id>/
        session_data_<data_session_id>.pkl        # raw loaded CSV
        session_windowed_<windowed_session_id>.pkl # windowed arrays + labels
        session_features_<feature_session_id>.pkl  # feature matrix + labels

Base directory:
    SESSION_STORE_DIR env var, default ``data/projects`` (matches the
    existing DATABASE_PATH ./data/cirame.db convention in Config).

File format:
    pickle protocol 4 — handles pd.DataFrame + np.ndarray efficiently and is
    the same format sklearn model artifacts already use elsewhere in the
    codebase. Not JSON (couldn't round-trip DataFrames/ndarrays).

All writes are atomic (write to .tmp then os.replace) — a container kill
mid-write cannot leave a half-written pickle that would poison hydrate.
Load functions never raise; they log a warning on corrupt / missing files and
return None so the route can fall back to "user re-runs the stage".
"""

import logging
import os
import pickle
import shutil
import tempfile
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Pickle protocol 4 — supported since Python 3.4, handles numpy arrays and
# large DataFrames without the 2 GB limit of protocol 2.
_PICKLE_PROTOCOL = 4

# Base directory for per-project pickle files. Overridable so tests can point
# it at a scratch dir.
_DEFAULT_BASE = os.path.join('data', 'projects')


def _base_dir() -> str:
    """Root under which per-project session pickles live.

    Read the env var on every call rather than at import time — tests patch
    SESSION_STORE_DIR after the module is loaded and the persistence layer
    should honor it.
    """
    return os.environ.get('SESSION_STORE_DIR', _DEFAULT_BASE)


def _project_dir(project_id: int) -> str:
    """Directory holding all pickles for a single project."""
    return os.path.join(_base_dir(), str(int(project_id)))


def _ensure_project_dir(project_id: int) -> str:
    """Create the project's pickle directory if missing; return its path."""
    d = _project_dir(project_id)
    os.makedirs(d, exist_ok=True)
    return d


def _session_path(project_id: int, kind: str, session_id: str) -> str:
    """Absolute path for a persisted session pickle.

    kind is 'data' | 'windowed' | 'features'.
    """
    fname = f'session_{kind}_{session_id}.pkl'
    return os.path.join(_project_dir(project_id), fname)


def _atomic_dump(path: str, obj) -> None:
    """Pickle obj to path atomically (write-then-replace).

    Written to a NamedTemporaryFile in the SAME directory as the target so
    ``os.replace`` is atomic on all platforms (cross-device rename is not).
    """
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    # delete=False so we can close, then os.replace. Same-dir tempfile ensures
    # the rename stays within one filesystem.
    fd, tmp_path = tempfile.mkstemp(
        prefix='.session_', suffix='.tmp', dir=dirname
    )
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=_PICKLE_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tempfile on any error path.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _safe_load(path: str) -> Optional[Dict]:
    """Load a pickle from path, returning None (never raising) on any
    error. Corrupt / missing files log a warning so the operator can spot
    disk-corruption issues without the request failing outright.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError,
            ModuleNotFoundError, ValueError, OSError) as e:
        logger.warning(
            'Corrupt / unreadable session pickle at %s: %s. Ignoring.',
            path, e,
        )
        return None


# ────────────────────────────────────────────────────────────────────────────
# Public API — persist_*
# ────────────────────────────────────────────────────────────────────────────

def persist_data_session(project_id: int, session_id: str, session: Dict) -> None:
    """Snapshot a `_data_sessions` entry to disk. Silent no-op on error.

    Call from the ROUTE layer right after DataSession.create() succeeds; the
    service layer creates the in-memory entry before project_id is known.
    """
    if project_id is None or not session_id or session is None:
        return
    try:
        path = _session_path(project_id, 'data', session_id)
        _atomic_dump(path, session)
    except Exception as e:
        logger.warning(
            '[session_persistence] persist_data_session(pid=%s, sid=%s) '
            'failed: %s', project_id, session_id, e,
        )


def persist_windowed_session(project_id: int, session_id: str, session: Dict) -> None:
    """Snapshot a windowed `_data_sessions` entry (windows + labels + meta)
    to disk. Called from routes right after WindowedSession.create().
    """
    if project_id is None or not session_id or session is None:
        return
    try:
        path = _session_path(project_id, 'windowed', session_id)
        _atomic_dump(path, session)
    except Exception as e:
        logger.warning(
            '[session_persistence] persist_windowed_session(pid=%s, sid=%s) '
            'failed: %s', project_id, session_id, e,
        )


def persist_feature_session(project_id: int, session_id: str, session: Dict) -> None:
    """Snapshot a `_feature_sessions` entry to disk. Called from routes
    right after FeatureSession.create() (extract / register-fast / apply-
    selection paths).
    """
    if project_id is None or not session_id or session is None:
        return
    try:
        path = _session_path(project_id, 'features', session_id)
        _atomic_dump(path, session)
    except Exception as e:
        logger.warning(
            '[session_persistence] persist_feature_session(pid=%s, sid=%s) '
            'failed: %s', project_id, session_id, e,
        )


# ────────────────────────────────────────────────────────────────────────────
# Public API — load_*
# ────────────────────────────────────────────────────────────────────────────

def load_data_session(project_id: int, session_id: str) -> Optional[Dict]:
    """Return persisted data session dict or None if absent/corrupt."""
    if project_id is None or not session_id:
        return None
    return _safe_load(_session_path(project_id, 'data', session_id))


def load_windowed_session(project_id: int, session_id: str) -> Optional[Dict]:
    """Return persisted windowed session dict or None if absent/corrupt."""
    if project_id is None or not session_id:
        return None
    return _safe_load(_session_path(project_id, 'windowed', session_id))


def load_feature_session(project_id: int, session_id: str) -> Optional[Dict]:
    """Return persisted feature session dict or None if absent/corrupt."""
    if project_id is None or not session_id:
        return None
    return _safe_load(_session_path(project_id, 'features', session_id))


# ────────────────────────────────────────────────────────────────────────────
# Public API — delete_project_sessions
# ────────────────────────────────────────────────────────────────────────────

def delete_project_sessions(project_id: int) -> int:
    """Evict all persisted session files for a project. Returns number of
    files removed. Called from the project DELETE route so orphan pickles
    don't accumulate on disk.

    Removes the entire per-project directory (not just individual files) to
    also clean up any stray tempfiles left by a killed atomic-write.
    """
    if project_id is None:
        return 0
    d = _project_dir(project_id)
    if not os.path.isdir(d):
        return 0
    try:
        count = 0
        for name in os.listdir(d):
            if name.endswith('.pkl') or name.startswith('.session_'):
                count += 1
        shutil.rmtree(d, ignore_errors=True)
        return count
    except Exception as e:
        logger.warning(
            '[session_persistence] delete_project_sessions(pid=%s) '
            'failed: %s', project_id, e,
        )
        return 0
