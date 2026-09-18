"""CiRA ME — Solutions Catalog Routes (F8 Phase 0, 2026-09-17).

Read-only REST facade over the Solutions catalog (`constants/solutions.py`)
and the feature-extractor registry (`services/extractors/`). All read
endpoints allow any logged-in user; there are no write endpoints in Phase 0
(the catalog is curated in code, not user-editable — see the plan's
non-goals).

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md §Phase 0.
"""

import logging

from flask import Blueprint, jsonify, request

from ..auth import login_required
from ..constants.solutions import get_all_solutions, get_solution
from ..services.extractors import registry as extractor_registry
from ..services import solutions_jobs

logger = logging.getLogger(__name__)
solutions_bp = Blueprint('solutions', __name__)


# ── Common: parse /run bodies and queue as a job ───────────────────────────


def _validate_run_body(data):
    """Extract the common /run request body shape once.

    Every solution's runner takes the same base shape; returns (kwargs,
    error_json, status_code). On error the caller returns the tuple's
    (error_json, status_code).
    """
    data_session_id = data.get('data_session_id')
    sampling_rate = data.get('sampling_rate')
    if not data_session_id or not sampling_rate:
        return None, (
            jsonify({'error': 'data_session_id and sampling_rate are required'}),
            400,
        )
    return {
        'data_session_id': data_session_id,
        'sampling_rate': float(sampling_rate),
        'params': data.get('params') or {},
        'window_s': float(data.get('window_s', 1.0)),
        'overlap': float(data.get('overlap', 0.5)),
        'approach': data.get('approach', 'auto'),
        'algorithm': data.get('algorithm'),
        'selected_columns': data.get('selected_columns'),
        'project_id': data.get('project_id'),
        'user_id': getattr(request, 'current_user', {}).get('id'),
    }, None


def _submit_run_job(kind, runner_fn, extra_kwargs_from_body=None):
    """Turn a POST /<kind>/run into a queued job.

    Returns 202 + {job_id, status, queue_position, poll_url}. The runner
    executes in a daemon thread behind the Solutions job semaphore.
    """
    data = request.get_json() or {}
    kwargs, err = _validate_run_body(data)
    if err is not None:
        return err
    if extra_kwargs_from_body:
        kwargs.update(extra_kwargs_from_body(data))
    user_id = getattr(request, 'current_user', {}).get('id')
    job_id = solutions_jobs.create_job(
        kind=kind,
        user_id=user_id,
        target=lambda: runner_fn(**kwargs),
    )
    snapshot = solutions_jobs.get_job(job_id, user_id) or {}
    return jsonify({
        'job_id': job_id,
        'status': snapshot.get('status', 'queued'),
        'queue_position': snapshot.get('queue_position', 1),
        'poll_url': f'/api/solutions/jobs/{job_id}',
    }), 202


# ── Solutions catalog ──────────────────────────────────────────────────────


@solutions_bp.route('', methods=['GET'])
@solutions_bp.route('/', methods=['GET'])
@login_required
def list_solutions():
    """List every solution template, with derived availability + param form."""
    return jsonify({
        'solutions': [s.to_dict(extractor_registry) for s in get_all_solutions()],
    })


@solutions_bp.route('/<solution_id>', methods=['GET'])
@login_required
def get_one_solution(solution_id):
    sol = get_solution(solution_id)
    if not sol:
        return jsonify({'error': 'Solution not found'}), 404
    return jsonify({'solution': sol.to_dict(extractor_registry)})


# ── Feature-extractor registry ─────────────────────────────────────────────


@solutions_bp.route('/extractors', methods=['GET'])
@login_required
def list_extractors():
    """List registered feature extractors (id, param_schema, signal reqs).

    The Features step uses this to render the extractor picker (the new
    third extraction mode alongside Fast/DSP and TSFresh).
    """
    return jsonify({
        'extractors': [e.to_dict() for e in extractor_registry.all()],
    })


# ── Self-contained Solution App runners ────────────────────────────────────


@solutions_bp.route('/mcsa/run', methods=['POST'])
@login_required
def run_mcsa_solution():
    """Queue an MCSA analysis job. Returns 202 + {job_id, poll_url}.

    Since 2026-09-18 the actual compute happens on a background thread
    behind Semaphore(4) — 40 concurrent workshop clicks stack up in
    Python instead of blocking the gunicorn thread pool. Client polls
    GET /api/solutions/jobs/<job_id> for status + result.
    """
    from ..services.solutions_runner import run_mcsa

    def _mcsa_extras(data):
        # MCSA has one extra param (skip_startup_s) + a different default
        # window (10 s, not the base 1 s). Apply on top of the common body.
        return {
            'skip_startup_s': float(data.get('skip_startup_s', 0.0)),
            'window_s': float(data.get('window_s', 10.0)),
        }
    return _submit_run_job('mcsa', run_mcsa, _mcsa_extras)


@solutions_bp.route('/mcsa/save', methods=['POST'])
@login_required
def save_mcsa():
    """Save a trained MCSA model as a deployable SavedModel (registry
    feature_extraction pipeline_config). It then appears in the saved-models
    list and deploys via the existing Deploy flow — inference reproduces
    windowing → MCSA features → predict."""
    from ..services.solutions_runner import save_mcsa_model

    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    tsid = data.get('training_session_id')
    win = data.get('windowed_session_id')
    feat = data.get('feature_session_id')
    settings = data.get('settings') or {}

    if not (name and tsid and win and feat):
        return jsonify({'error': 'name, training_session_id, windowed_session_id, '
                                 'feature_session_id are required'}), 400
    try:
        model_id = save_mcsa_model(
            name, tsid, win, feat, settings, request.current_user['id'])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"MCSA save failed: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({'saved_model_id': model_id, 'name': name})


@solutions_bp.route('/vibration/run', methods=['POST'])
@login_required
def run_vibration_solution():
    """Queue a Machine Vibration analysis job. See run_mcsa_solution for the
    queue contract."""
    from ..services.solutions_runner import run_vibration
    return _submit_run_job('vibration', run_vibration)


@solutions_bp.route('/vibration/save', methods=['POST'])
@login_required
def save_vibration():
    """Save a trained bearing model as a deployable SavedModel."""
    from ..services.solutions_runner import save_solution_model

    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    tsid = data.get('training_session_id')
    win = data.get('windowed_session_id')
    feat = data.get('feature_session_id')
    settings = data.get('settings') or {}
    if not (name and tsid and win and feat):
        return jsonify({'error': 'name, training_session_id, windowed_session_id, '
                                 'feature_session_id are required'}), 400
    try:
        model_id = save_solution_model(
            name, 'machine_vibration', tsid, win, feat, settings,
            request.current_user['id'])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Vibration save failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify({'saved_model_id': model_id, 'name': name})


@solutions_bp.route('/pump/run', methods=['POST'])
@login_required
def run_pump_solution():
    """Queue a Pump analysis job."""
    from ..services.solutions_runner import run_pump
    return _submit_run_job('pump', run_pump)


@solutions_bp.route('/pump/save', methods=['POST'])
@login_required
def save_pump():
    """Save a trained pump model as a deployable SavedModel."""
    from ..services.solutions_runner import save_solution_model

    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    tsid = data.get('training_session_id')
    win = data.get('windowed_session_id')
    feat = data.get('feature_session_id')
    settings = data.get('settings') or {}
    if not (name and tsid and win and feat):
        return jsonify({'error': 'name, training_session_id, windowed_session_id, '
                                 'feature_session_id are required'}), 400
    try:
        model_id = save_solution_model(
            name, 'pump_analysis', tsid, win, feat, settings,
            request.current_user['id'])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pump save failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify({'saved_model_id': model_id, 'name': name})


@solutions_bp.route('/pump-fusion/run', methods=['POST'])
@login_required
def run_pump_fusion_solution():
    """Queue a Pump Fusion (current + vibration) analysis job."""
    from ..services.solutions_runner import run_pump_fusion
    return _submit_run_job('pump_fusion', run_pump_fusion)


@solutions_bp.route('/pump-fusion/save', methods=['POST'])
@login_required
def save_pump_fusion():
    """Save a trained pump-fusion model as a deployable SavedModel."""
    from ..services.solutions_runner import save_solution_model

    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    tsid = data.get('training_session_id')
    win = data.get('windowed_session_id')
    feat = data.get('feature_session_id')
    settings = data.get('settings') or {}
    if not (name and tsid and win and feat):
        return jsonify({'error': 'name, training_session_id, windowed_session_id, '
                                 'feature_session_id are required'}), 400
    try:
        model_id = save_solution_model(
            name, 'pump_fusion', tsid, win, feat, settings,
            request.current_user['id'])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pump fusion save failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify({'saved_model_id': model_id, 'name': name})


# ── Async job status polling ───────────────────────────────────────────────


@solutions_bp.route('/jobs/<job_id>', methods=['GET'])
@login_required
def get_solutions_job(job_id):
    """Poll a queued/running Solutions analysis job.

    Returned shape:
      status=queued     → {queue_position}
      status=running    → {elapsed_s, queue_position=0}
      status=done       → {result, elapsed_s}
      status=failed     → {error, elapsed_s}

    Auth-scoped: caller must be the submitter. Non-owner or unknown-id
    both return 404 so we don't leak whether a job_id exists.
    """
    user_id = getattr(request, 'current_user', {}).get('id')
    snapshot = solutions_jobs.get_job(job_id, user_id)
    if snapshot is None:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(snapshot)


@solutions_bp.route('/jobs/_stats', methods=['GET'])
@login_required
def get_solutions_job_stats():
    """Debug endpoint — admin-usable during load spikes to see the queue
    depth. Not user-facing; safe to expose since it returns aggregate
    counters only."""
    return jsonify(solutions_jobs.stats())
