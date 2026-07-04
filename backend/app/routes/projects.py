"""CiRA ME - Projects Routes (F4).

Per-user Projects view: pipeline stage progress + deploy breakdown.
Admin sees all projects via ?all=1 (F4.2).
One project = one dataset (F4.3). Dataset swap = clone.
"""

import logging
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..models import (
    Project, FeatureTemplate,
    DataSession, WindowedSession, FeatureSession,
    get_db,
)

logger = logging.getLogger(__name__)
projects_bp = Blueprint('projects', __name__)


def _owned_or_none(project_id: int, user):
    """Return project dict if owned by user (or admin). None otherwise."""
    p = Project.get_by_id(project_id)
    if not p:
        return None
    if p.get('user_id') != user['id'] and user.get('role') != 'admin':
        return None
    return p


@projects_bp.route('', methods=['GET'])
@projects_bp.route('/', methods=['GET'])
@login_required
def list_projects():
    """List projects. Admin can pass ?all=1 to see everyone's."""
    user = request.current_user
    see_all = request.args.get('all', '').strip() in ('1', 'true', 'yes')
    if see_all and user.get('role') == 'admin':
        rows = Project.get_all_with_status(user_id=None)
    else:
        rows = Project.get_all_with_status(user_id=user['id'])
    return jsonify({'projects': rows})


@projects_bp.route('', methods=['POST'])
@projects_bp.route('/', methods=['POST'])
@login_required
def create_project():
    """Create a new project."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400
    mode = data.get('mode') or 'anomaly'
    description = data.get('description')
    config = data.get('config')

    pid = Project.create(
        name=name,
        user_id=request.current_user['id'],
        mode=mode,
        description=description,
        config=config,
    )
    return jsonify({'id': pid, 'name': name, 'mode': mode}), 201


@projects_bp.route('/<int:project_id>', methods=['GET'])
@login_required
def get_project(project_id):
    """Full detail including all stage rows for the project."""
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404

    p['data_sessions'] = DataSession.get_by_project(project_id)
    p['windowed_sessions'] = WindowedSession.get_by_project(project_id)
    p['feature_sessions'] = FeatureSession.get_by_project(project_id)

    with get_db() as conn:
        c = conn.cursor()
        p['saved_models'] = [dict(r) for r in c.execute(
            'SELECT id, name, algorithm, created_at FROM saved_models '
            'WHERE project_id = ? ORDER BY created_at DESC',
            (project_id,)
        ).fetchall()]
        p['melab_endpoints'] = [dict(r) for r in c.execute(
            'SELECT id, name, status, created_at FROM melab_endpoints '
            'WHERE project_id = ? ORDER BY created_at DESC',
            (project_id,)
        ).fetchall()]
        p['app_builder_apps'] = [dict(r) for r in c.execute(
            'SELECT id, name, status, slug FROM app_builder_apps '
            'WHERE project_id = ? ORDER BY created_at DESC',
            (project_id,)
        ).fetchall()]
        p['deploy_records'] = [dict(r) for r in c.execute(
            'SELECT * FROM deploy_records '
            'WHERE project_id = ? ORDER BY created_at DESC',
            (project_id,)
        ).fetchall()]

    p['feature_template'] = FeatureTemplate.get(project_id)
    return jsonify(p)


@projects_bp.route('/<int:project_id>', methods=['PATCH'])
@login_required
def update_project(project_id):
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404
    data = request.get_json(silent=True) or {}
    updates = {k: v for k, v in data.items()
               if k in ('name', 'description', 'mode', 'config', 'current_stage')}
    if not updates:
        return jsonify({'error': 'No valid fields to update'}), 400
    if 'name' in updates and not str(updates['name']).strip():
        return jsonify({'error': 'name cannot be empty'}), 400
    Project.update(project_id, **updates)
    return jsonify(Project.get_by_id(project_id))


@projects_bp.route('/<int:project_id>', methods=['DELETE'])
@login_required
def delete_project(project_id):
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404
    Project.delete(project_id)
    return jsonify({'message': 'Project deleted'})


@projects_bp.route('/<int:project_id>/clone', methods=['POST'])
@login_required
def clone_project(project_id):
    """Q1: copy projects.config + feature_templates only. Do NOT copy
    saved_models / training_sessions. Force retrain on the clone.
    """
    import json
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json(silent=True) or {}
    new_name = (data.get('name') or f"{p['name']} (clone)").strip()
    new_config = data.get('config')

    src_config = p.get('config')
    if isinstance(src_config, str):
        try:
            src_config = json.loads(src_config)
        except Exception:
            src_config = None
    config_to_use = new_config if new_config is not None else src_config

    new_id = Project.create(
        name=new_name,
        user_id=request.current_user['id'],
        mode=p.get('mode') or 'anomaly',
        description=p.get('description'),
        config=config_to_use if isinstance(config_to_use, dict) else None,
    )

    # Copy feature template if present
    src_tpl = FeatureTemplate.get(project_id)
    if src_tpl and src_tpl.get('ordered_feature_names'):
        FeatureTemplate.upsert(new_id, src_tpl['ordered_feature_names'])

    return jsonify({'id': new_id, 'name': new_name,
                    'message': 'Project cloned'}), 201


@projects_bp.route('/<int:project_id>/hydrate', methods=['POST'])
@login_required
def hydrate_project(project_id):
    """Re-materialize the persisted data session into memory so pipeline
    views can pick up where the user left off.

    Persisted rows survive backend restarts, but the loader's in-memory
    _data_sessions dict does not. Without this endpoint, clicking a stage
    chip on the Projects list drops the user on an empty Data Source page
    even though the row shows "csv · 623 rows".

    Also returns the persisted windowing config + feature session so the
    downstream views can pre-fill their forms — user just clicks Apply
    again rather than reconstructing settings from memory.
    """
    import json
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404

    result = {
        'project_id': project_id,
        'data_session': None,
        'windowing_config': None,
        'feature_session': None,
    }

    ds_rows = DataSession.get_by_project(project_id)
    if not ds_rows:
        return jsonify(result)
    ds = ds_rows[0]

    fmt = ds.get('format') or 'csv'
    file_path = ds.get('file_path')
    if not file_path:
        return jsonify({'error': 'DataSession has no file_path'}), 500

    # Re-load into the same LRU-capped in-memory dict the ingest endpoints
    # populate. This gives the frontend a fresh session_id it can use as a
    # normal handle for windowing / features downstream.
    try:
        from ..services.data_loader import DataLoader
        loader = DataLoader()
        if fmt == 'csv':
            loaded = loader.load_csv(file_path)
        elif fmt == 'csv_multi':
            # file_path is the multi-csv selection directory in this case
            import os, glob as _g
            csv_files = sorted(_g.glob(os.path.join(file_path, '*.csv')))
            loaded = loader.load_csv_multiple(csv_files) if csv_files else None
        elif fmt in ('ei_json', 'edge_impulse_json'):
            loaded = loader.load_edge_impulse_json(file_path)
        elif fmt in ('ei_cbor', 'edge_impulse_cbor'):
            loaded = loader.load_edge_impulse_cbor(file_path)
        elif fmt in ('cira_cbor', 'cira'):
            loaded = loader.load_cira_cbor(file_path)
        else:
            return jsonify({
                'error': f'Unsupported persisted format: {fmt}'
            }), 400
        if not loaded:
            return jsonify({
                'error': 'Persisted file could not be re-loaded (deleted or moved)'
            }), 410
        result['data_session'] = loaded
    except Exception as e:
        logger.exception(f'[projects] hydrate failed for project {project_id}: {e}')
        return jsonify({'error': f'Reload failed: {e}'}), 500

    # Persisted windowing / feature configs — return for form pre-fill,
    # but do NOT auto-replay them (user should click Apply, matching Q4).
    ws_rows = WindowedSession.get_by_project(project_id)
    if ws_rows:
        ws = ws_rows[0]
        cfg = ws.get('config')
        try:
            result['windowing_config'] = json.loads(cfg) if isinstance(cfg, str) else cfg
        except (TypeError, json.JSONDecodeError):
            result['windowing_config'] = None

    fs_rows = FeatureSession.get_by_project(project_id)
    if fs_rows:
        fs = fs_rows[0]
        names = fs.get('feature_names')
        try:
            names = json.loads(names) if isinstance(names, str) else names
        except (TypeError, json.JSONDecodeError):
            names = None
        result['feature_session'] = {
            'method': fs.get('method'),
            'feature_names': names or [],
            'num_features': fs.get('num_features'),
        }

    return jsonify(result)


@projects_bp.route('/<int:project_id>/feature-template', methods=['GET'])
@login_required
def get_feature_template(project_id):
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404
    tpl = FeatureTemplate.get(project_id)
    return jsonify(tpl or {'project_id': project_id,
                            'ordered_feature_names': [],
                            'version': 0})


@projects_bp.route('/<int:project_id>/feature-template', methods=['PUT'])
@login_required
def put_feature_template(project_id):
    p = _owned_or_none(project_id, request.current_user)
    if not p:
        return jsonify({'error': 'Project not found'}), 404
    data = request.get_json(silent=True) or {}
    features = data.get('ordered_feature_names')
    if not isinstance(features, list):
        return jsonify({'error': 'ordered_feature_names (list) required'}), 400
    features = [str(f) for f in features]
    tpl = FeatureTemplate.upsert(project_id, features)
    return jsonify(tpl)
