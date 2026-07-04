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


# ---------------------------------------------------------------------------
# File browsing — spare the user from having to shell in and ls the folders.
# ---------------------------------------------------------------------------

# Cap responses so a mis-configured watcher pointing at a huge folder can't
# OOM the API. Files are listed newest-first; if there are more than the cap
# the response includes total count so the UI can hint.
_MAX_FILES_PER_FOLDER = 50
# Preview cap: 200 KB is enough for a few thousand rows of a CSV; beyond that
# we return only the head + a note.
_PREVIEW_MAX_BYTES = 200 * 1024

_VALID_KINDS = ('input', 'output', 'error')


def _list_folder(dir_path: str, glob_pattern: str | None = None) -> tuple[list[dict], int]:
    """List files in a folder as {name, size, mtime}. Returns (list, total_count)."""
    import os
    import glob as _glob
    if not os.path.isdir(dir_path):
        return [], 0
    if glob_pattern:
        matches = _glob.glob(os.path.join(dir_path, glob_pattern))
    else:
        matches = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    files = []
    for p in matches:
        if not os.path.isfile(p):
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        files.append({
            'name': os.path.basename(p),
            'size': st.st_size,
            'mtime': st.st_mtime,
        })
    total = len(files)
    files.sort(key=lambda f: f['mtime'], reverse=True)
    return files[:_MAX_FILES_PER_FOLDER], total


@folder_watchers_bp.route('/<int:watcher_id>/files', methods=['GET'])
@login_required
def list_files(watcher_id):
    """List files in this watcher's input / output / error folders.

    Response: {input: {files: [...], total: N},
               output: {...},
               error:  {...}}
    Each file: {name, size, mtime}. Sorted newest-first, capped at
    _MAX_FILES_PER_FOLDER.
    """
    import os
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    input_dir = watcher['input_folder']
    output_dir = watcher['output_folder']
    error_dir = os.path.join(input_dir, '_error')

    # Input pass through the watcher's glob so we don't advertise unrelated
    # files sitting in the folder — matches the runtime's actual pick-up set.
    input_files, input_total = _list_folder(input_dir, watcher.get('file_glob') or '*.txt')
    output_files, output_total = _list_folder(output_dir)
    error_files, error_total = _list_folder(error_dir)

    return jsonify({
        'input':  {'files': input_files,  'total': input_total,  'folder': input_dir},
        'output': {'files': output_files, 'total': output_total, 'folder': output_dir},
        'error':  {'files': error_files,  'total': error_total,  'folder': error_dir},
    })


@folder_watchers_bp.route('/<int:watcher_id>/files/<kind>/<path:filename>', methods=['GET'])
@login_required
def get_file_content(watcher_id, kind, filename):
    """Read a specific file's content for preview. Response: {content, size,
    truncated: bool}.

    `kind` is one of `input`, `output`, `error`. `filename` is validated to
    be a bare basename (no path traversal). Content is capped at
    _PREVIEW_MAX_BYTES; larger files return only the head + `truncated: true`
    so the UI can offer a download link instead.
    """
    import os
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    if kind not in _VALID_KINDS:
        return jsonify({'error': f'kind must be one of {_VALID_KINDS}'}), 400

    # Reject any path-traversal attempt. os.path.basename strips separators.
    safe_name = os.path.basename(filename)
    if safe_name != filename or safe_name in ('', '.', '..'):
        return jsonify({'error': 'invalid filename'}), 400

    input_dir = watcher['input_folder']
    if kind == 'input':
        folder = input_dir
    elif kind == 'output':
        folder = watcher['output_folder']
    else:  # 'error'
        folder = os.path.join(input_dir, '_error')

    full_path = os.path.join(folder, safe_name)
    if not os.path.isfile(full_path):
        return jsonify({'error': 'file not found'}), 404

    try:
        size = os.path.getsize(full_path)
    except OSError:
        return jsonify({'error': 'file not readable'}), 500

    truncated = size > _PREVIEW_MAX_BYTES
    read_bytes = _PREVIEW_MAX_BYTES if truncated else size
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(read_bytes)
    except OSError as e:
        return jsonify({'error': f'could not read: {e}'}), 500

    return jsonify({
        'name': safe_name,
        'kind': kind,
        'size': size,
        'content': content,
        'truncated': truncated,
    })
