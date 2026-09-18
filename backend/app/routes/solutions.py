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

logger = logging.getLogger(__name__)
solutions_bp = Blueprint('solutions', __name__)


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
    """Run the Motor Current (MCSA) Solution App end-to-end from a loaded data
    session: window (steady-state) → MCSA features → auto-selected model →
    domain results. Self-contained — does not touch the generic pipeline state.

    Body:
      data_session_id  — id from the data-load step (required)
      sampling_rate    — DAQ rate in Hz (required; MCSA is frequency-based)
      params           — MCSA nameplate params (pole_pairs, line_freq_hz, ...)
      window_s         — window length in seconds (default 10)
      skip_startup_s   — drop leading transient windows (default 0)
      project_id       — optional
    """
    from ..services.solutions_runner import run_mcsa

    data = request.get_json() or {}
    data_session_id = data.get('data_session_id')
    sampling_rate = data.get('sampling_rate')
    params = data.get('params') or {}
    window_s = data.get('window_s', 10.0)
    overlap = data.get('overlap', 0.5)
    approach = data.get('approach', 'auto')
    algorithm = data.get('algorithm')
    skip_startup_s = data.get('skip_startup_s', 0.0)
    selected_columns = data.get('selected_columns')
    project_id = data.get('project_id')

    if not data_session_id or not sampling_rate:
        return jsonify({'error': 'data_session_id and sampling_rate are required'}), 400

    user_id = getattr(request, 'current_user', {}).get('id')
    try:
        result = run_mcsa(
            data_session_id=data_session_id,
            sampling_rate=float(sampling_rate),
            params=params,
            window_s=float(window_s),
            overlap=float(overlap),
            approach=approach,
            algorithm=algorithm,
            skip_startup_s=float(skip_startup_s),
            selected_columns=selected_columns,
            project_id=project_id,
            user_id=user_id,
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"MCSA solution run failed: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify(result)


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
    """Run the Machine Vibration (bearing) Solution App end-to-end: window →
    bearing-envelope features → auto-selected model → domain results."""
    from ..services.solutions_runner import run_vibration

    data = request.get_json() or {}
    data_session_id = data.get('data_session_id')
    sampling_rate = data.get('sampling_rate')
    if not data_session_id or not sampling_rate:
        return jsonify({'error': 'data_session_id and sampling_rate are required'}), 400
    try:
        result = run_vibration(
            data_session_id=data_session_id,
            sampling_rate=float(sampling_rate),
            params=data.get('params') or {},
            window_s=float(data.get('window_s', 1.0)),
            overlap=float(data.get('overlap', 0.5)),
            approach=data.get('approach', 'auto'),
            algorithm=data.get('algorithm'),
            selected_columns=data.get('selected_columns'),
            project_id=data.get('project_id'),
            user_id=getattr(request, 'current_user', {}).get('id'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Vibration solution run failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify(result)


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
    """Run the Pump (cavitation/impeller) Solution App end-to-end."""
    from ..services.solutions_runner import run_pump

    data = request.get_json() or {}
    data_session_id = data.get('data_session_id')
    sampling_rate = data.get('sampling_rate')
    if not data_session_id or not sampling_rate:
        return jsonify({'error': 'data_session_id and sampling_rate are required'}), 400
    try:
        result = run_pump(
            data_session_id=data_session_id,
            sampling_rate=float(sampling_rate),
            params=data.get('params') or {},
            window_s=float(data.get('window_s', 1.0)),
            overlap=float(data.get('overlap', 0.5)),
            approach=data.get('approach', 'auto'),
            algorithm=data.get('algorithm'),
            selected_columns=data.get('selected_columns'),
            project_id=data.get('project_id'),
            user_id=getattr(request, 'current_user', {}).get('id'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pump solution run failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify(result)


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
    """Run the Pump Fusion (current + vibration) Solution App end-to-end."""
    from ..services.solutions_runner import run_pump_fusion

    data = request.get_json() or {}
    data_session_id = data.get('data_session_id')
    sampling_rate = data.get('sampling_rate')
    if not data_session_id or not sampling_rate:
        return jsonify({'error': 'data_session_id and sampling_rate are required'}), 400
    try:
        result = run_pump_fusion(
            data_session_id=data_session_id,
            sampling_rate=float(sampling_rate),
            params=data.get('params') or {},
            window_s=float(data.get('window_s', 1.0)),
            overlap=float(data.get('overlap', 0.5)),
            approach=data.get('approach', 'auto'),
            algorithm=data.get('algorithm'),
            selected_columns=data.get('selected_columns'),
            project_id=data.get('project_id'),
            user_id=getattr(request, 'current_user', {}).get('id'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pump fusion run failed: {e}")
        return jsonify({'error': str(e)}), 500
    return jsonify(result)


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
