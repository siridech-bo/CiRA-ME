"""CiRA ME - Folder Watcher Routes.

Per-user watchers that poll a container-visible folder, run each row of
each file through a ME-LAB endpoint, and write results to an output CSV.
"""

import os
import re
import shutil
import logging
import tempfile
import zipfile
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from ..auth import login_required
from ..models import FolderWatcher, MeLabEndpoint
from ..services import folder_watcher_service

logger = logging.getLogger(__name__)
folder_watchers_bp = Blueprint('folder_watchers', __name__)


_MUTABLE_FIELDS = (
    'name', 'input_folder', 'output_folder',
    'poll_interval_s', 'file_glob', 'header_mode',
    'parse_mode', 'parse_regex', 'parse_columns',
    'mqtt_enabled', 'mqtt_topic', 'daily_csv_enabled',
)
_VALID_HEADER_MODES = ('auto', 'headered', 'headerless')
_VALID_PARSE_MODES = ('csv', 'regex', 'json', 'key_value')

# Preview endpoint caps — small enough that a rogue caller can't chew through
# CPU or memory, big enough that a real log file's first hundred lines fit.
_PREVIEW_SAMPLE_MAX_BYTES = 4096
_PREVIEW_MAX_ROWS = 100


def _validate_parse_columns(raw) -> tuple[str | None, str | None]:
    """Return (normalized_string, error_message).
    Empty / all-whitespace / no valid column names → error."""
    if raw is None:
        return None, 'List at least one column name (comma-separated) for key_value parse mode.'
    if not isinstance(raw, str):
        return None, 'parse_columns must be a comma-separated string'
    cols = [c.strip() for c in raw.split(',') if c.strip()]
    if not cols:
        return None, 'List at least one column name (comma-separated) for key_value parse mode.'
    return ', '.join(cols), None


def _serialize(watcher: dict) -> dict:
    """Enrich a watcher row with the endpoint's display name / algorithm."""
    if not watcher:
        return watcher
    result = dict(watcher)
    # Normalize INTEGER flag columns to booleans for the frontend.
    for _flag in ('mqtt_enabled', 'daily_csv_enabled'):
        if _flag in result:
            result[_flag] = bool(result.get(_flag))
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

    # ── Log Watcher parse mode + regex validation ─────────────────────────
    parse_mode = (data.get('parse_mode') or 'csv').strip().lower()
    if parse_mode not in _VALID_PARSE_MODES:
        return jsonify({'error': f'parse_mode must be one of {_VALID_PARSE_MODES}'}), 400
    parse_regex = data.get('parse_regex')
    if parse_mode == 'regex':
        if not parse_regex or not str(parse_regex).strip():
            return jsonify({'error': 'parse_regex is required when parse_mode="regex"'}), 400
        try:
            re.compile(parse_regex)
        except re.error as e:
            return jsonify({'error': f'parse_regex is not a valid Python regex: {e}'}), 400
    else:
        parse_regex = None  # ignore any incoming regex for non-regex modes

    # ── Log Watcher parse_columns (key_value mode) ────────────────────────
    parse_columns = data.get('parse_columns')
    if parse_mode == 'key_value':
        normalized, err = _validate_parse_columns(parse_columns)
        if err:
            return jsonify({
                'error': err,
                'error_code': 'PARSE_COLUMNS_REQUIRED',
                'hint': 'List at least one column name (comma-separated) for key_value parse mode.',
            }), 400
        parse_columns = normalized
    else:
        parse_columns = None  # ignore for non-key_value modes

    # ── MQTT publish sink ─────────────────────────────────────────────────
    mqtt_enabled = bool(data.get('mqtt_enabled'))
    mqtt_topic = (data.get('mqtt_topic') or '').strip() or None
    if mqtt_enabled and not mqtt_topic:
        return jsonify({'error': 'mqtt_topic is required when mqtt_enabled=true'}), 400

    # ── Daily aggregated CSV sink ─────────────────────────────────────────
    daily_csv_enabled = bool(data.get('daily_csv_enabled'))

    watcher_id = FolderWatcher.create(
        user_id=request.current_user['id'],
        name=name,
        endpoint_id=endpoint_id,
        input_folder=input_folder,
        output_folder=output_folder,
        poll_interval_s=poll_interval_s,
        file_glob=file_glob,
        header_mode=header_mode,
        parse_mode=parse_mode,
        parse_regex=parse_regex,
        parse_columns=parse_columns,
        mqtt_enabled=mqtt_enabled,
        mqtt_topic=mqtt_topic,
        daily_csv_enabled=daily_csv_enabled,
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

    # ── Log Watcher fields ────────────────────────────────────────────────
    # We validate the *effective* parse_mode (post-patch) against parse_regex
    # so a two-step edit (regex first, then swap to csv) can succeed without
    # regex validation failing on an empty pattern.
    effective_parse_mode = updates.get('parse_mode', watcher.get('parse_mode') or 'csv')
    if 'parse_mode' in updates:
        pm = str(updates['parse_mode']).strip().lower()
        if pm not in _VALID_PARSE_MODES:
            return jsonify({'error': f'parse_mode must be one of {_VALID_PARSE_MODES}'}), 400
        updates['parse_mode'] = pm
        effective_parse_mode = pm

    if 'parse_regex' in updates:
        pr = updates['parse_regex']
        if pr is not None and not isinstance(pr, str):
            return jsonify({'error': 'parse_regex must be a string'}), 400
        pr = (pr or '').strip() or None
        updates['parse_regex'] = pr

    if effective_parse_mode == 'regex':
        # Resolve the pattern that will actually be stored (updated value if
        # present, else the existing one). Require it to compile.
        eff_regex = updates.get('parse_regex', watcher.get('parse_regex'))
        if not eff_regex:
            return jsonify({'error': 'parse_regex is required when parse_mode="regex"'}), 400
        try:
            re.compile(eff_regex)
        except re.error as e:
            return jsonify({'error': f'parse_regex is not a valid Python regex: {e}'}), 400

    # parse_columns — normalize + validate against effective parse_mode.
    if 'parse_columns' in updates:
        pc = updates['parse_columns']
        if pc is not None and not isinstance(pc, str):
            return jsonify({'error': 'parse_columns must be a string'}), 400
        # Normalize to trimmed comma-joined form (or None if empty)
        if pc is None:
            updates['parse_columns'] = None
        else:
            cols = [c.strip() for c in pc.split(',') if c.strip()]
            updates['parse_columns'] = ', '.join(cols) if cols else None

    if effective_parse_mode == 'key_value':
        eff_cols = updates.get('parse_columns', watcher.get('parse_columns'))
        _, err = _validate_parse_columns(eff_cols)
        if err:
            return jsonify({
                'error': err,
                'error_code': 'PARSE_COLUMNS_REQUIRED',
                'hint': 'List at least one column name (comma-separated) for key_value parse mode.',
            }), 400

    # MQTT enable / topic — same "effective post-patch" gate as regex.
    if 'mqtt_enabled' in updates:
        updates['mqtt_enabled'] = bool(updates['mqtt_enabled'])
    if 'mqtt_topic' in updates:
        mt = updates['mqtt_topic']
        if mt is not None and not isinstance(mt, str):
            return jsonify({'error': 'mqtt_topic must be a string'}), 400
        updates['mqtt_topic'] = (mt or '').strip() or None
    effective_mqtt_enabled = updates.get(
        'mqtt_enabled', bool(watcher.get('mqtt_enabled'))
    )
    if effective_mqtt_enabled:
        eff_topic = updates.get('mqtt_topic', watcher.get('mqtt_topic'))
        if not eff_topic:
            return jsonify({'error': 'mqtt_topic is required when mqtt_enabled=true'}), 400

    if 'daily_csv_enabled' in updates:
        updates['daily_csv_enabled'] = bool(updates['daily_csv_enabled'])

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

# `processed` is the post-success archive folder created by
# folder_watcher_service._move_to_processed. It's a sibling of the input
# folder (technically nested at <input>/_processed/) so it needs its own kind
# so the frontend can list, preview, download, and delete from it too.
_VALID_KINDS = ('input', 'output', 'error', 'processed')

# Per-file upload cap for the multipart upload endpoint. Matches the
# data_sources.py upload cap so operators aren't surprised by different limits
# across the app.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _folder_for_kind(watcher: dict, kind: str) -> str:
    """Return the on-disk folder for one of `_VALID_KINDS`. Callers must
    validate `kind` themselves first (this is a lookup, not a gatekeeper)."""
    input_dir = watcher['input_folder']
    if kind == 'input':
        return input_dir
    if kind == 'output':
        return watcher['output_folder']
    if kind == 'error':
        return os.path.join(input_dir, '_error')
    if kind == 'processed':
        return os.path.join(input_dir, '_processed')
    # Should never reach here — caller validated kind first.
    raise ValueError(f'unknown kind: {kind}')


def _safe_basename(filename: str) -> str | None:
    """Reject any path-traversal attempt. Returns the safe basename or None
    if the input is invalid. All file-manipulation endpoints must use this."""
    if not isinstance(filename, str):
        return None
    safe_name = os.path.basename(filename)
    if safe_name != filename or safe_name in ('', '.', '..'):
        return None
    return safe_name


def _list_folder(dir_path: str, glob_pattern: str | None = None, include_error_reason: bool = False) -> tuple[list[dict], int]:
    """List files in a folder as {name, size, mtime, matches_glob, error_reason?}.

    Returns (list, total_count). If ``glob_pattern`` is supplied, every file
    is still listed (so uploads that don't match the watcher's file_glob
    remain visible), but each carries a ``matches_glob`` boolean the frontend
    can use to badge / grey out files the runtime will ignore. Callers that
    don't pass a pattern get ``matches_glob=True`` on every entry (nothing to
    filter on).

    When ``include_error_reason`` is True (used for the ``_error/`` folder),
    each file entry is enriched with the first line of its sidecar
    ``<name>.error`` file so the UI can render WHY it failed inline. Sidecar
    files themselves are excluded from the listing.
    """
    import os
    import fnmatch
    if not os.path.isdir(dir_path):
        return [], 0
    # Always list everything under the folder. Filtering is now a per-entry
    # flag rather than a hide-vs-show decision so users can find and delete
    # files they uploaded to the wrong watcher.
    entries = os.listdir(dir_path)
    files = []
    for name in entries:
        # Hide sidecar `.error` files from the listing — they're metadata for
        # the actual failed input, not user-visible files.
        if include_error_reason and name.endswith('.error'):
            continue
        p = os.path.join(dir_path, name)
        if not os.path.isfile(p):
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        matches_glob = True
        if glob_pattern:
            matches_glob = fnmatch.fnmatch(name, glob_pattern)
        entry = {
            'name': name,
            'size': st.st_size,
            'mtime': st.st_mtime,
            'matches_glob': matches_glob,
        }
        if include_error_reason:
            # First line of the sidecar is the exception summary; keep only
            # that so the table row stays short. Preview endpoint can show
            # the full traceback on click.
            sidecar = os.path.join(dir_path, f"{name}.error")
            try:
                if os.path.isfile(sidecar):
                    with open(sidecar, 'r', encoding='utf-8', errors='replace') as ef:
                        first_line = ef.readline().rstrip('\r\n')
                        if first_line:
                            entry['error_reason'] = first_line[:300]
            except OSError:
                pass
        files.append(entry)
    total = len(files)
    files.sort(key=lambda f: f['mtime'], reverse=True)
    return files[:_MAX_FILES_PER_FOLDER], total


@folder_watchers_bp.route('/<int:watcher_id>/files', methods=['GET'])
@login_required
def list_files(watcher_id):
    """List files in this watcher's input / output / error / processed folders.

    Response: {input:     {files: [...], total: N, folder: ...},
               output:    {...},
               error:     {...},
               processed: {...}}
    Each file: {name, size, mtime}. Sorted newest-first, capped at
    _MAX_FILES_PER_FOLDER.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    input_dir = watcher['input_folder']
    output_dir = watcher['output_folder']
    error_dir = os.path.join(input_dir, '_error')
    processed_dir = os.path.join(input_dir, '_processed')

    # List every file in the input folder; the glob just decides which
    # entries carry matches_glob=True. Hiding non-matches would leave
    # accidentally-uploaded files invisible and un-deletable in the UI.
    input_files, input_total = _list_folder(input_dir, watcher.get('file_glob') or '*.txt')
    output_files, output_total = _list_folder(output_dir)
    # Enrich error entries with the first line of their sidecar `.error`
    # file so the UI can show WHY each file failed instead of a bare
    # "Failed at HH:MM:SS".
    error_files, error_total = _list_folder(error_dir, include_error_reason=True)
    processed_files, processed_total = _list_folder(processed_dir)

    return jsonify({
        'input':     {'files': input_files,     'total': input_total,     'folder': input_dir},
        'output':    {'files': output_files,    'total': output_total,    'folder': output_dir},
        'error':     {'files': error_files,     'total': error_total,     'folder': error_dir},
        'processed': {'files': processed_files, 'total': processed_total, 'folder': processed_dir},
    })


@folder_watchers_bp.route('/<int:watcher_id>/files/<kind>/<path:filename>', methods=['GET'])
@login_required
def get_file_content(watcher_id, kind, filename):
    """Read a specific file's content for preview. Response: {content, size,
    truncated: bool}.

    `kind` is one of `input`, `output`, `error`, `processed`. `filename` is
    validated to be a bare basename (no path traversal). Content is capped at
    _PREVIEW_MAX_BYTES; larger files return only the head + `truncated: true`
    so the UI can offer a download link instead.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    if kind not in _VALID_KINDS:
        return jsonify({'error': f'kind must be one of {_VALID_KINDS}'}), 400

    safe_name = _safe_basename(filename)
    if safe_name is None:
        return jsonify({'error': 'invalid filename'}), 400

    folder = _folder_for_kind(watcher, kind)
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


# ---------------------------------------------------------------------------
# File manager — upload / download / delete / bulk zip / retry / paths / history.
# Everything below is scoped to a single watcher, ownership-checked, and
# path-traversal-safe via _safe_basename.
# ---------------------------------------------------------------------------

@folder_watchers_bp.route('/<int:watcher_id>/upload', methods=['POST'])
@login_required
def upload_files(watcher_id):
    """Multipart upload one-or-more files into the watcher's input folder.

    Uses the same basename-strip pattern as data_sources.py because Firefox
    and Linux drag-drop send the full relative path in `file.filename`.
    Duplicate names are suffixed `_1`, `_2`, matching the Data Source upload.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    files = request.files.getlist('files')
    if not files or all((f.filename or '') == '' for f in files):
        return jsonify({'error': 'No files provided'}), 400

    input_dir = watcher['input_folder']
    try:
        os.makedirs(input_dir, exist_ok=True)
    except OSError as e:
        return jsonify({'error': f'Could not create input folder: {e}'}), 500

    uploaded: list = []
    errors: list = []

    for f in files:
        raw_name = f.filename or ''
        if not raw_name:
            continue

        # Firefox / Linux drag-drop bug: filename may be a nested path. Strip
        # to the leaf so secure_filename doesn't collapse the slashes into
        # underscores.
        fname_raw = raw_name.replace('\\', '/')
        fname_base = os.path.basename(fname_raw)
        safe_name = secure_filename(fname_base)
        if not safe_name:
            errors.append({'name': raw_name, 'reason': 'invalid filename'})
            continue

        # Size gate — check before saving, without loading the whole thing.
        try:
            f.stream.seek(0, os.SEEK_END)
            size = f.stream.tell()
            f.stream.seek(0)
        except Exception:
            size = None
        if size is not None and size > _MAX_UPLOAD_BYTES:
            return jsonify({
                'error': (
                    f"{safe_name} is larger than the "
                    f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit"
                )
            }), 413

        # De-duplicate the target name if a file with the same basename already
        # exists in the input folder (mirrors data_sources.py behaviour).
        stem, ext = os.path.splitext(safe_name)
        final_name = safe_name
        counter = 1
        while os.path.exists(os.path.join(input_dir, final_name)):
            final_name = f"{stem}_{counter}{ext}"
            counter += 1
            if counter > 1000:
                # Pathological collision — fall back to timestamp for guaranteed uniqueness.
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                final_name = f"{stem}_{ts}{ext}"
                break

        dest = os.path.join(input_dir, final_name)
        try:
            f.save(dest)
        except Exception as e:
            errors.append({'name': raw_name, 'reason': f'save failed: {e}'})
            continue

        try:
            saved_size = os.path.getsize(dest)
        except OSError:
            saved_size = size or 0

        uploaded.append({'name': final_name, 'size': saved_size})

    return jsonify({
        'ok': True,
        'uploaded': uploaded,
        'errors': errors,
    }), 200 if uploaded or not errors else 400


@folder_watchers_bp.route(
    '/<int:watcher_id>/files/<kind>/<path:filename>/download', methods=['GET']
)
@login_required
def download_file(watcher_id, kind, filename):
    """Binary download of a file from the watcher's input/output/error/processed
    folder. Uses send_file with `as_attachment=True` so the browser prompts
    for save rather than trying to render binary CSVs inline.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    if kind not in _VALID_KINDS:
        return jsonify({'error': f'kind must be one of {_VALID_KINDS}'}), 400

    safe_name = _safe_basename(filename)
    if safe_name is None:
        return jsonify({'error': 'invalid filename'}), 400

    full_path = os.path.join(_folder_for_kind(watcher, kind), safe_name)
    if not os.path.isfile(full_path):
        return jsonify({'error': 'file not found'}), 404

    # send_file wants an absolute path so behaviour is stable across CWDs.
    return send_file(
        os.path.abspath(full_path),
        as_attachment=True,
        download_name=safe_name,
    )


@folder_watchers_bp.route(
    '/<int:watcher_id>/files/<kind>/<path:filename>', methods=['DELETE']
)
@login_required
def delete_file(watcher_id, kind, filename):
    """Delete a single file from the watcher's input/output/error/processed
    folder. Refuses subdirectories — we only manage flat files here.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    if kind not in _VALID_KINDS:
        return jsonify({'error': f'kind must be one of {_VALID_KINDS}'}), 400

    safe_name = _safe_basename(filename)
    if safe_name is None:
        return jsonify({'error': 'invalid filename'}), 400

    full_path = os.path.join(_folder_for_kind(watcher, kind), safe_name)
    if os.path.isdir(full_path):
        return jsonify({'error': 'cannot delete a subdirectory'}), 400
    if not os.path.isfile(full_path):
        return jsonify({'error': 'file not found'}), 404

    try:
        os.remove(full_path)
    except OSError as e:
        return jsonify({'error': f'delete failed: {e}'}), 500

    # When deleting an error file, also remove its sidecar `.error` metadata
    # so we don't leak stale reason-of-failure files across retries.
    if kind == 'error':
        sidecar = full_path + '.error'
        if os.path.isfile(sidecar):
            try:
                os.remove(sidecar)
            except OSError:
                pass

    return jsonify({'ok': True})


@folder_watchers_bp.route(
    '/<int:watcher_id>/files/output/zip', methods=['GET']
)
@login_required
def download_output_zip(watcher_id):
    """Zip up every regular file in the top level of the watcher's output
    folder and stream it back as a single attachment.

    Skips subdirectories (output is typically flat) so a stray _error/ or
    other admin folder doesn't get bundled. If the output folder is empty
    we return 404 — the frontend disables the button in that case, but this
    catches the race.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    output_dir = watcher['output_folder']
    if not os.path.isdir(output_dir):
        return jsonify({'error': 'output folder does not exist yet'}), 404

    # Collect files first so we can 404 before allocating a tempfile.
    entries: list = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isfile(full):
            entries.append((name, full))
    if not entries:
        return jsonify({'error': 'output folder is empty'}), 404

    watcher_slug = folder_watcher_service._slug(watcher.get('name') or f"watcher_{watcher_id}")
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    zip_name = f"{watcher_slug}_outputs_{ts}.zip"

    # NamedTemporaryFile with delete=False so Flask/send_file can stream the
    # file back after we close it. We rely on the OS temp cleanup here rather
    # than deleting manually — deleting mid-stream races with Flask's send.
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"watcher_{watcher_id}_zip_", suffix='.zip', delete=False
    )
    tmp_path = tmp.name
    tmp.close()
    try:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, full in entries:
                zf.write(full, arcname=name)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return jsonify({'error': f'could not build zip: {e}'}), 500

    logger.info(
        f"[FolderWatcher {watcher_id}] built output zip with "
        f"{len(entries)} file(s) → {zip_name}"
    )
    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=zip_name,
        mimetype='application/zip',
    )


@folder_watchers_bp.route(
    '/<int:watcher_id>/files/error/<path:filename>/retry', methods=['POST']
)
@login_required
def retry_error_file(watcher_id, filename):
    """Move an errored file back to the input folder so the watcher picks it
    up again on the next tick. On name-collision with something already in
    input, suffix with a timestamp so both are preserved.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    safe_name = _safe_basename(filename)
    if safe_name is None:
        return jsonify({'error': 'invalid filename'}), 400

    input_dir = watcher['input_folder']
    error_dir = os.path.join(input_dir, '_error')
    error_path = os.path.join(error_dir, safe_name)
    if not os.path.isfile(error_path):
        return jsonify({'error': 'error file not found'}), 404

    try:
        os.makedirs(input_dir, exist_ok=True)
    except OSError as e:
        return jsonify({'error': f'could not ensure input folder: {e}'}), 500

    target_name = safe_name
    target_path = os.path.join(input_dir, target_name)
    if os.path.exists(target_path):
        stem, ext = os.path.splitext(safe_name)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        target_name = f"{stem}_retry_{ts}{ext}"
        target_path = os.path.join(input_dir, target_name)

    try:
        shutil.move(error_path, target_path)
    except OSError as e:
        return jsonify({'error': f'retry move failed: {e}'}), 500

    # Retry means "have another go" — the previous failure reason no longer
    # applies. Delete the sidecar so the next run's error (if any) is what
    # the operator sees instead of the stale one.
    sidecar = error_path + '.error'
    if os.path.isfile(sidecar):
        try:
            os.remove(sidecar)
        except OSError:
            pass

    logger.info(
        f"[FolderWatcher {watcher_id}] retry {safe_name} → {target_path}"
    )
    return jsonify({'ok': True, 'new_path': target_path, 'new_name': target_name})


# ---------------------------------------------------------------------------
# Path introspection (Tier 2) — helps customers rsync / scp / automate against
# the container's folder set without having to shell in and figure out the
# host-side mount points themselves.
# ---------------------------------------------------------------------------

# The container mount root — folder_watcher_service creates watcher subtrees
# under /app/watcher-data by default. If the deployment overrides this we
# still want to expose whatever the watcher was configured with, so we split
# on this prefix rather than assume it.
_CONTAINER_WATCHER_ROOT = '/app/watcher-data'


def _to_host_path(container_path: str, host_base: str | None) -> str | None:
    """Map a container-side path to its host-side equivalent by rebasing off
    _CONTAINER_WATCHER_ROOT. Returns None if the container path isn't under
    that root, or if the operator hasn't set WATCHER_HOST_BASE_PATH.
    """
    if not host_base:
        return None
    if not container_path:
        return None
    root = _CONTAINER_WATCHER_ROOT.rstrip('/')
    # Normalise the leading slash + trailing whitespace.
    p = container_path.strip()
    if not (p == root or p.startswith(root + '/')):
        return None
    remainder = p[len(root):].lstrip('/')
    host_base_clean = host_base.rstrip('/')
    if not remainder:
        return host_base_clean
    return f"{host_base_clean}/{remainder}"


@folder_watchers_bp.route('/<int:watcher_id>/paths', methods=['GET'])
@login_required
def watcher_paths(watcher_id):
    """Return container + host paths + API URLs for each of this watcher's
    input / output / processed / error folders. Read-only introspection —
    doesn't touch any file.
    """
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404

    input_dir = watcher['input_folder']
    output_dir = watcher['output_folder']
    processed_dir = os.path.join(input_dir, '_processed')
    error_dir = os.path.join(input_dir, '_error')

    host_base = os.environ.get('WATCHER_HOST_BASE_PATH') or None

    # request.host_url includes scheme + host + trailing slash, e.g.
    # "https://me.cira-core.com/". Strip the trailing slash so we don't emit
    # double slashes when concatenating with the API path.
    base_url = (request.host_url or '').rstrip('/')

    def _entry(container_path: str, api_extra: dict) -> dict:
        entry = {
            'container': container_path,
            'host': _to_host_path(container_path, host_base),
        }
        entry.update(api_extra)
        return entry

    return jsonify({
        'input': _entry(input_dir, {
            'api_upload': f"{base_url}/api/folder-watchers/{watcher_id}/upload",
        }),
        'output': _entry(output_dir, {
            'api_download_zip':
                f"{base_url}/api/folder-watchers/{watcher_id}/files/output/zip",
        }),
        'processed': _entry(processed_dir, {}),
        'error': _entry(error_dir, {}),
        'watcher_host_base_configured': bool(host_base),
    })


# ---------------------------------------------------------------------------
# History inference — no new DB table. We reconstruct pairs of input →
# output by matching basenames (stem, then substring fallback) between the
# processed archive and the output folder.
# ---------------------------------------------------------------------------

_HISTORY_MAX_ENTRIES = 200


def _infer_history(watcher: dict) -> list[dict]:
    """Return a list of {input_name, output_name, processed_at, output_size,
    status} entries, newest-first, capped at _HISTORY_MAX_ENTRIES.

    Matching heuristic:
      1. Same stem (foo.csv ↔ foo.csv)
      2. Output-basename starts with input-basename's stem
         (foo.csv ↔ foo_predictions.csv) — handles the collision-suffix path.
    Any leftover outputs that never matched are reported as status="output_only"
    so the operator can still see them in history.
    """
    input_dir = watcher['input_folder']
    output_dir = watcher['output_folder']
    processed_dir = os.path.join(input_dir, '_processed')

    def _scan(dir_path: str) -> list[dict]:
        if not os.path.isdir(dir_path):
            return []
        out = []
        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if not os.path.isfile(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append({'name': name, 'mtime': st.st_mtime, 'size': st.st_size})
        return out

    processed_files = _scan(processed_dir)
    output_files = _scan(output_dir)

    # Index outputs by name so we can pop-match into them.
    outputs_by_name: dict[str, dict] = {o['name']: o for o in output_files}
    outputs_used: set[str] = set()

    entries: list[dict] = []
    for pf in processed_files:
        pf_name = pf['name']
        # Strip the "processed_YYYYMMDD_HHMMSS" collision suffix if present,
        # so we can match on the original basename.
        original = re.sub(r'\.processed_\d{8}_\d{6}$', '', pf_name)
        stem, _ = os.path.splitext(original)

        # 1. Exact-name match.
        match = outputs_by_name.get(original)
        # 2. Same stem + any extension (e.g. output CSV of a JSON input).
        if match is None:
            for oname, ometa in outputs_by_name.items():
                if oname in outputs_used:
                    continue
                o_stem, _ = os.path.splitext(oname)
                if o_stem == stem:
                    match = ometa
                    break
        # 3. Output basename starts with input stem (foo → foo_predictions.csv).
        if match is None and stem:
            for oname, ometa in outputs_by_name.items():
                if oname in outputs_used:
                    continue
                if oname.startswith(stem + '_') or oname.startswith(stem + '.'):
                    match = ometa
                    break

        if match is not None:
            outputs_used.add(match['name'])
            entries.append({
                'input_name': original,
                'archive_name': pf_name,
                'output_name': match['name'],
                'processed_at': match['mtime'],
                'output_size': match['size'],
                'status': 'success',
            })
        else:
            entries.append({
                'input_name': original,
                'archive_name': pf_name,
                'output_name': None,
                'processed_at': pf['mtime'],
                'output_size': None,
                'status': 'archived_no_output',
            })

    # Any outputs the matcher didn't consume — surfaced so operators can still
    # see them in the timeline (e.g. output produced before we started
    # archiving inputs, or after a manual re-run).
    for oname, ometa in outputs_by_name.items():
        if oname in outputs_used:
            continue
        entries.append({
            'input_name': None,
            'archive_name': None,
            'output_name': oname,
            'processed_at': ometa['mtime'],
            'output_size': ometa['size'],
            'status': 'output_only',
        })

    entries.sort(key=lambda e: e['processed_at'] or 0, reverse=True)
    return entries[:_HISTORY_MAX_ENTRIES]


@folder_watchers_bp.route('/<int:watcher_id>/history', methods=['GET'])
@login_required
def watcher_history(watcher_id):
    """List processing history inferred from disk. Newest-first, capped at
    _HISTORY_MAX_ENTRIES."""
    watcher = _owned_or_404(watcher_id, request.current_user['id'])
    if not watcher:
        return jsonify({'error': 'Watcher not found'}), 404
    entries = _infer_history(watcher)
    return jsonify({'entries': entries, 'total': len(entries)})


# ---------------------------------------------------------------------------
# Preview parse — no side effects. Lets the user paste a sample line and see
# how the current parse config would extract rows before saving the watcher.
# ---------------------------------------------------------------------------

@folder_watchers_bp.route('/detect-columns', methods=['POST'])
@login_required
def detect_columns():
    """Scan a sample log snippet for `key=value` / `key:value` patterns and
    return unique key names in first-seen order. Used by the edit form's
    "Auto-detect columns" button so operators don't have to know their log
    schema upfront.
    """
    import re
    data = request.get_json(silent=True) or {}
    sample = str(data.get('sample_content', ''))[:4096]
    if not sample.strip():
        return jsonify({'columns': []})

    # Word chars = key; then =/:/: followed by an optional sign and digits.
    # Anchors on \b so we don't grab prefixes of longer tokens like `pid=123`
    # embedded in URLs. Ignores keys with no numeric value.
    pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{1,63})\s*[=:]\s*-?\d')
    seen: set = set()
    columns: list = []
    for m in pattern.finditer(sample):
        name = m.group(1)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        columns.append(name)
    return jsonify({'columns': columns})


@folder_watchers_bp.route('/preview-parse', methods=['POST'])
@login_required
def preview_parse():
    """Dry-run the parse layer against a sample string. No file I/O, no DB
    writes. Returns detected columns + up to _PREVIEW_MAX_ROWS parsed rows
    with NaN replaced by null.
    """
    import math
    data = request.get_json(silent=True) or {}
    parse_mode = (data.get('parse_mode') or 'key_value').strip().lower()
    if parse_mode not in _VALID_PARSE_MODES:
        return jsonify({
            'error': f'parse_mode must be one of {_VALID_PARSE_MODES}',
            'error_code': 'INVALID_PARSE_MODE',
        }), 400

    sample_content = data.get('sample_content') or ''
    if not isinstance(sample_content, str):
        return jsonify({'error': 'sample_content must be a string'}), 400
    if len(sample_content) > _PREVIEW_SAMPLE_MAX_BYTES:
        sample_content = sample_content[:_PREVIEW_SAMPLE_MAX_BYTES]

    warnings: list[str] = []
    columns: list[str] = []
    rows: list[list] = []

    total_lines = sum(1 for ln in sample_content.splitlines() if ln.strip())

    try:
        if parse_mode == 'regex':
            pattern = data.get('parse_regex') or ''
            if not str(pattern).strip():
                return jsonify({
                    'error': 'parse_regex is required for regex mode',
                    'error_code': 'INVALID_REGEX',
                    'hint': 'Provide a Python regex with named capture groups.',
                }), 400
            try:
                re.compile(pattern)
            except re.error:
                return jsonify({
                    'error': 'Invalid regex',
                    'error_code': 'INVALID_REGEX',
                    'hint': 'Check your named-capture groups syntax.',
                }), 400
            columns, rows = folder_watcher_service._parse_regex_content(
                sample_content, pattern
            )
        elif parse_mode == 'json':
            columns, rows = folder_watcher_service._parse_json_content(sample_content)
        elif parse_mode == 'key_value':
            raw_cols = data.get('parse_columns') or ''
            cols_list = [c.strip() for c in str(raw_cols).split(',') if c.strip()]
            if not cols_list:
                return jsonify({
                    'error': 'List at least one column name (comma-separated) for key_value parse mode.',
                    'error_code': 'PARSE_COLUMNS_REQUIRED',
                    'hint': 'List at least one column name (comma-separated) for key_value parse mode.',
                }), 400
            columns, rows = folder_watcher_service._parse_key_value_content(
                sample_content, cols_list
            )
        else:  # csv
            header_mode = data.get('header_mode') or 'auto'
            if header_mode not in _VALID_HEADER_MODES:
                header_mode = 'auto'
            columns, rows = folder_watcher_service._parse_csv_content(
                sample_content, header_mode
            )
    except Exception as e:
        logger.exception('preview_parse failed')
        return jsonify({
            'error': f'Preview failed: {e}',
            'error_code': 'PREVIEW_FAILED',
        }), 500

    # Cap the output row count so a huge sample can't blow up the response.
    if len(rows) > _PREVIEW_MAX_ROWS:
        rows = rows[:_PREVIEW_MAX_ROWS]

    # Replace NaN with None (→ JSON null) so the frontend renders a clean "—"
    # instead of the browser turning NaN into a JSON parse error.
    def _safe(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    safe_rows = [[_safe(v) for v in row] for row in rows]

    skipped_lines = max(0, total_lines - len(rows))
    if skipped_lines > 0:
        warnings.append(
            f'Skipped {skipped_lines} line{"s" if skipped_lines != 1 else ""} '
            f'that did not parse'
        )

    return jsonify({
        'columns': list(columns),
        'rows': safe_rows,
        'row_count': len(safe_rows),
        'skipped_lines': skipped_lines,
        'warnings': warnings,
    })
