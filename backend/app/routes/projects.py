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
    # Approach 2b: also evict pickled session blobs so /data/projects/
    # doesn't accumulate orphans. Runs after the DB delete succeeds so a
    # partial failure keeps the pickles paired with their metadata.
    try:
        from ..services.session_persistence import delete_project_sessions
        evicted = delete_project_sessions(project_id)
        if evicted:
            logger.info(
                '[projects] deleted %d persisted session pickle(s) for pid=%s',
                evicted, project_id,
            )
    except Exception as e:
        logger.warning(
            '[projects] delete_project_sessions failed for pid=%s: %s',
            project_id, e,
        )
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

    # `rehydrated` flags let the frontend show a targeted toast when only
    # some stages could be restored from disk (Approach 2b — persist to
    # disk). data:false means the pickle was missing AND we had to re-parse
    # the CSV. windowed/features flags are true only when the pickle was
    # loaded successfully into the in-memory session dicts.
    result = {
        'project_id': project_id,
        'data_session': None,
        'windowing_config': None,
        'feature_session': None,
        'rehydrated': {
            'data': False,
            'windowed': False,
            'features': False,
        },
    }

    ds_rows = DataSession.get_by_project(project_id)
    if not ds_rows:
        return jsonify(result)
    ds = ds_rows[0]

    fmt = ds.get('format') or 'csv'
    file_path = ds.get('file_path')
    if not file_path:
        return jsonify({'error': 'DataSession has no file_path'}), 500

    persisted_sid = ds.get('session_id')

    # Re-load into the same LRU-capped in-memory dict the ingest endpoints
    # populate. This gives the frontend a fresh session_id it can use as a
    # normal handle for windowing / features downstream.
    #
    # `format` in the DataSession row is what DataLoader recorded on ingest,
    # not what the user picked in the picker. Multi-CSV selections come
    # through as format='csv' with a DIRECTORY file_path (see
    # load_csv_multiple: it records 'csv' + selection_dir, not 'csv_multi').
    # So we dispatch on path-is-directory first, then on format.
    try:
        import os
        from ..services.data_loader import DataLoader, _data_sessions, _sessions_lock
        from ..services.session_persistence import (
            load_data_session, load_windowed_session, load_feature_session,
        )
        loader = DataLoader()

        # Approach 2b: try the pickle first — it's O(1MB) reads instead of
        # re-parsing potentially large CSVs. Also handles the case where
        # the source CSV has been moved / deleted since ingest.
        restored = None
        if persisted_sid:
            restored = load_data_session(project_id, persisted_sid)
        if restored is not None:
            with _sessions_lock:
                _data_sessions[persisted_sid] = restored
            meta = restored.get('metadata', {}) or {}
            loaded = {
                'session_id': persisted_sid,
                'metadata': meta,
            }
            # Provide a small preview so the frontend Data Source view has
            # rows to render without another round-trip. Guard against
            # non-DataFrame sessions (windowed dicts don't carry 'data').
            df = restored.get('data')
            if df is not None:
                try:
                    loaded['preview'] = df.head(10).to_dict(orient='records')
                    loaded['total_rows'] = len(df)
                except Exception:
                    pass
            result['data_session'] = loaded
            result['rehydrated']['data'] = True
        else:
            if not os.path.exists(file_path):
                return jsonify({
                    'error': f'Persisted file no longer exists: {file_path}'
                }), 410

            if os.path.isdir(file_path):
                # Multi-CSV selection dir (T3 pattern). Glob for CSVs and
                # reload as a batch.
                import glob as _g
                csv_files = sorted(_g.glob(os.path.join(file_path, '*.csv')))
                if not csv_files:
                    return jsonify({
                        'error': f'Multi-CSV directory contains no .csv files: {file_path}'
                    }), 410
                loaded = loader.load_csv_multiple(csv_files)
            elif fmt == 'csv':
                loaded = loader.load_csv(file_path)
            elif fmt in ('edge_impulse_json', 'ei_json'):
                loaded = loader.load_edge_impulse_json(file_path)
            elif fmt in ('edge_impulse_cbor', 'ei_cbor'):
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

    # Persisted windowing / feature configs — return for form pre-fill AND
    # try to rehydrate the in-memory session dict from disk so the user
    # doesn't have to re-click Apply on Windowing (Approach 2b).
    ws_rows = WindowedSession.get_by_project(project_id)
    if ws_rows:
        ws = ws_rows[0]
        cfg = ws.get('config')
        try:
            result['windowing_config'] = json.loads(cfg) if isinstance(cfg, str) else cfg
        except (TypeError, json.JSONDecodeError):
            result['windowing_config'] = None

        ws_sid = ws.get('session_id')
        if ws_sid:
            try:
                win_entry = load_windowed_session(project_id, ws_sid)
                if win_entry is not None:
                    with _sessions_lock:
                        _data_sessions[ws_sid] = win_entry
                    result['rehydrated']['windowed'] = True
                    # Also expose the restored session_id so the frontend
                    # can use it directly for downstream calls (features,
                    # window-sample, etc.) without re-applying windowing.
                    result['windowed_session'] = {
                        'session_id': ws_sid,
                        'num_windows': ws.get('num_windows'),
                    }
                else:
                    logger.warning(
                        '[hydrate] windowed pickle missing for pid=%s sid=%s',
                        project_id, ws_sid,
                    )
            except Exception as e:
                logger.warning(
                    '[hydrate] windowed rehydrate failed pid=%s: %s',
                    project_id, e,
                )

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
            'session_id': fs.get('session_id'),
        }

        fs_sid = fs.get('session_id')
        if fs_sid:
            try:
                from ..services.feature_extractor import _feature_sessions
                feat_entry = load_feature_session(project_id, fs_sid)
                if feat_entry is not None:
                    _feature_sessions[fs_sid] = feat_entry
                    result['rehydrated']['features'] = True
                else:
                    logger.warning(
                        '[hydrate] feature pickle missing for pid=%s sid=%s',
                        project_id, fs_sid,
                    )
            except Exception as e:
                logger.warning(
                    '[hydrate] feature rehydrate failed pid=%s: %s',
                    project_id, e,
                )

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
