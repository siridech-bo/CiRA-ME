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
