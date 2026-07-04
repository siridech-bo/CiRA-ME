"""CiRA ME - Folder Watcher Routes.

Per-user watchers that poll a container-visible folder, run each row of
each file through a ME-LAB endpoint, and write results to an output CSV.
"""

import logging
from flask import Blueprint, request, jsonify

from ..auth import login_required
from ..models import FolderWatcher, MeLabEndpoint
from ..services import folder_watcher_service

logger = logging.getLogger(__name__)
folder_watchers_bp = Blueprint('folder_watchers', __name__)


_MUTABLE_FIELDS = (
    'name', 'input_folder', 'output_folder',
    'poll_interval_s', 'file_glob', 'header_mode',
)
_VALID_HEADER_MODES = ('auto', 'headered', 'headerless')


def _serialize(watcher: dict) -> dict:
    """Enrich a watcher row with the endpoint's display name / algorithm."""
    if not watcher:
        return watcher
    result = dict(watcher)
    try:
        ep = MeLabEndpoint.get_by_id(watcher['endpoint_id'])
        if ep:
            result['endpoint_name'] = ep.get('name')
            result['endpoint_algorithm'] = ep.get('algorithm')
            result['endpoint_mode'] = ep.get('mode')
            result['endpoint_status'] = ep.get('status')
        else:
            result['endpoint_name'] = None
            result['endpoint_algorithm'] = None
            result['endpoint_mode'] = None
            result['endpoint_status'] = None
    except Exception:
        pass
    return result


def _owned_or_404(watcher_id: int, user_id: int):
    watcher = FolderWatcher.get_by_id(watcher_id)
    if not watcher or watcher.get('user_id') != user_id:
        return None
    return watcher


@folder_watchers_bp.route('/', methods=['GET'])
@folder_watchers_bp.route('', methods=['GET'])
@login_required
def list_watchers():
    """List current user's watchers."""
    watchers = FolderWatcher.get_by_user(request.current_user['id'])
    return jsonify([_serialize(w) for w in watchers])


@folder_watchers_bp.route('/', methods=['POST'])
@folder_watchers_bp.route('', methods=['POST'])
@login_required
def create_watcher():
    """Create a new watcher. The endpoint must belong to the caller."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    endpoint_id = (data.get('endpoint_id') or '').strip()
    input_folder = (data.get('input_folder') or '').strip()
    output_folder = (data.get('output_folder') or '').strip()

    if not name:
        return jsonify({'error': 'name is required'}), 400
    if not endpoint_id:
        return jsonify({'error': 'endpoint_id is required'}), 400
    if not input_folder:
        return jsonify({'error': 'input_folder is required'}), 400
    if not output_folder:
        return jsonify({'error': 'output_folder is required'}), 400

    ep = MeLabEndpoint.get_by_id(endpoint_id)
    if not ep or ep.get('user_id') != request.current_user['id']:
        return jsonify({'error': 'Endpoint not found or not owned by user'}), 400

    header_mode = data.get('header_mode') or 'auto'
    if header_mode not in _VALID_HEADER_MODES:
        return jsonify({'error': f'header_mode must be one of {_VALID_HEADER_MODES}'}), 400

    poll_interval_s = data.get('poll_interval_s') or 60
    try:
        poll_interval_s = int(poll_interval_s)
    except (TypeError, ValueError):
        return jsonify({'error': 'poll_interval_s must be an integer'}), 400
    if poll_interval_s < 10 or poll_interval_s > 3600:
        return jsonify({'error': 'poll_interval_s must be between 10 and 3600'}), 400

    file_glob = (data.get('file_glob') or '*.txt').strip() or '*.txt'

    watcher_id = FolderWatcher.create(
        user_id=request.current_user['id'],
        name=name,
        endpoint_id=endpoint_id,
        input_folder=input_folder,
        output_folder=output_folder,
        poll_interval_s=poll_interval_s,
        file_glob=file_glob,
        header_mode=header_mode,
    )
    watcher = FolderWatcher.get_by_id(watcher_id)
    return jsonify(_serialize(watcher)), 201


@folder_watchers_bp.route('/<int:watcher_id>', methods=['GET'])
@login_required
def get_watcher(watcher_id):
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    return jsonify(_serialize(watcher))


@folder_watchers_bp.route('/<int:watcher_id>', methods=['PATCH'])
@login_required
def update_watcher(watcher_id):
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    data = request.get_json(silent=True) or {}

    # endpoint_id is immutable on PATCH: changing it would require reloading
    # the model, purging counters, and re-validating ownership. Reject with a
    # clear error rather than silently ignoring — the frontend used to include
    # endpoint_id in the payload, which made it look like the change stuck.
    if 'endpoint_id' in data and data['endpoint_id'] != watcher.get('endpoint_id'):
        return jsonify({
            'error': 'endpoint_id cannot be changed. Delete this watcher and '
                     'create a new one for the different endpoint.'
        }), 400

    updates = {}
    for k in _MUTABLE_FIELDS:
        if k in data:
            updates[k] = data[k]

    # Validate poll_interval_s
    if 'poll_interval_s' in updates:
        try:
            updates['poll_interval_s'] = int(updates['poll_interval_s'])
        except (TypeError, ValueError):
            return jsonify({'error': 'poll_interval_s must be an integer'}), 400
        if updates['poll_interval_s'] < 10 or updates['poll_interval_s'] > 3600:
            return jsonify({'error': 'poll_interval_s must be between 10 and 3600'}), 400

    # Validate header_mode
    if 'header_mode' in updates and updates['header_mode'] not in _VALID_HEADER_MODES:
        return jsonify({'error': f'header_mode must be one of {_VALID_HEADER_MODES}'}), 400

    # Trim strings
    for k in ('name', 'input_folder', 'output_folder', 'file_glob'):
        if k in updates and isinstance(updates[k], str):
            updates[k] = updates[k].strip()
            if k in ('name', 'input_folder', 'output_folder') and not updates[k]:
                return jsonify({'error': f'{k} cannot be empty'}), 400
    if 'file_glob' in updates and not updates['file_glob']:
        updates['file_glob'] = '*.txt'

    if updates:
        FolderWatcher.update(watcher_id, **updates)

    watcher = FolderWatcher.get_by_id(watcher_id)
    return jsonify(_serialize(watcher))


@folder_watchers_bp.route('/<int:watcher_id>', methods=['DELETE'])
@login_required
def delete_watcher(watcher_id):
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    # Stop worker first, then delete row
    try:
        folder_watcher_service.stop_watcher(watcher_id)
    except Exception as e:
        logger.warning(f"[FolderWatcher] stop before delete failed: {e}")
    FolderWatcher.delete(watcher_id, request.current_user['id'])
    return jsonify({'message': 'Watcher deleted'})


@folder_watchers_bp.route('/<int:watcher_id>/start', methods=['POST'])
@login_required
def start_watcher(watcher_id):
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    folder_watcher_service.start_watcher(watcher_id)
    watcher = FolderWatcher.get_by_id(watcher_id)
    return jsonify({'status': watcher.get('status'), 'watcher': _serialize(watcher)})


@folder_watchers_bp.route('/<int:watcher_id>/stop', methods=['POST'])
@login_required
def stop_watcher(watcher_id):
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    folder_watcher_service.stop_watcher(watcher_id)
    watcher = FolderWatcher.get_by_id(watcher_id)
    return jsonify({'status': watcher.get('status'), 'watcher': _serialize(watcher)})
