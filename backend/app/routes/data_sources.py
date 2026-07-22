"""
CiRA ME - Data Sources Routes
Handles CSV, Edge Impulse JSON, Edge Impulse CBOR, and CiRA CBOR formats
"""

import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from ..auth import login_required, validate_path, get_user_folders, _is_path_within
from ..services.data_loader import DataLoader, DataValidationError
from ..models import Project, DataSession, WindowedSession

logger = logging.getLogger(__name__)

data_sources_bp = Blueprint('data_sources', __name__)

# Allowed file extensions for upload
# Note: txt/tsv/dat/log are handled by the Text File format wizard.
ALLOWED_EXTENSIONS = {'csv', 'json', 'cbor', 'txt', 'tsv', 'dat', 'log'}
TEXT_EXTENSIONS = {'txt', 'tsv', 'dat', 'log'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@data_sources_bp.route('/datasets-root', methods=['GET'])
@login_required
def get_datasets_root():
    """Get the datasets root path."""
    return jsonify({
        'path': current_app.config['DATASETS_ROOT_PATH']
    })


@data_sources_bp.route('/user-folders', methods=['GET'])
@login_required
def get_user_accessible_folders():
    """Get folders accessible to the current user."""
    user = request.current_user
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    folders = get_user_folders(user, datasets_root, shared_folder)

    return jsonify({'folders': folders})


@data_sources_bp.route('/browse', methods=['POST'])
@login_required
def browse_directory():
    """Browse a directory for files and subdirectories."""
    data = request.get_json() or {}

    path = data.get('path')
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    user = request.current_user

    # Handle root/empty path - show user's accessible folders
    if not path or os.path.normpath(os.path.abspath(path)) == os.path.normpath(os.path.abspath(datasets_root)):
        # For annotators, show their accessible folders as "virtual root"
        if user.get('role') != 'admin':
            folders = get_user_folders(user, datasets_root, shared_folder)
            items = []
            for folder in folders:
                items.append({
                    'name': folder['name'],
                    'path': folder['path'],
                    'is_dir': True,
                    'extension': None,
                    'size': None,
                    'file_type': None
                })
            return jsonify({
                'current_path': datasets_root,
                'items': items
            })
        # For admins, default to datasets root
        path = datasets_root

    # Validate path access
    if not validate_path(path, user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403

    if not os.path.exists(path):
        return jsonify({'error': 'Path not found'}), 404

    if not os.path.isdir(path):
        return jsonify({'error': 'Path is not a directory'}), 400

    items = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        is_dir = os.path.isdir(item_path)

        # Get file extension and size
        ext = os.path.splitext(item)[1].lower() if not is_dir else None
        size = os.path.getsize(item_path) if not is_dir else None

        # Modified time (unix seconds) — used by the File Manager UI's
        # "Modified" column and sort. `getmtime` is safe on missing entries
        # here because we just os.listdir()'d the parent.
        try:
            modified = os.path.getmtime(item_path)
        except OSError:
            modified = None

        # Determine file type
        file_type = None
        if ext == '.csv':
            file_type = 'csv'
        elif ext == '.json':
            file_type = 'json'
        elif ext == '.cbor':
            file_type = 'cbor'
        elif ext in ('.txt', '.tsv', '.dat', '.log'):
            file_type = 'text'

        items.append({
            'name': item,
            'path': item_path,
            'is_dir': is_dir,
            'extension': ext,
            'size': size,
            'file_type': file_type,
            'modified': modified,
        })

    # Sort: directories first, then files
    items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))

    return jsonify({
        'current_path': path,
        'items': items
    })


@data_sources_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """
    Upload a dataset file from the user's local machine.

    Files are saved to the shared folder for processing.
    Supports CSV, JSON (Edge Impulse), and CBOR formats.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Check file size (read content length from headers if available)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        return jsonify({
            'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB'
        }), 400

    # Get target folder from form data (optional)
    target_folder = request.form.get('folder', '')
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    datasets_root = current_app.config['DATASETS_ROOT_PATH']

    # Determine upload directory
    if target_folder:
        # Validate target folder access
        if not validate_path(target_folder, request.current_user, datasets_root, shared_folder):
            return jsonify({'error': 'Access denied to target folder'}), 403
        upload_dir = target_folder
    else:
        # Default to shared folder with user's uploads subdirectory
        user = request.current_user
        user_upload_dir = os.path.join(shared_folder, 'uploads', f"user_{user['id']}")
        os.makedirs(user_upload_dir, exist_ok=True)
        upload_dir = user_upload_dir

    # Secure the filename and handle duplicates.
    # Some browsers (Firefox, Chromium on Linux via drag-drop) put the full
    # relative path in file.filename. secure_filename() would then replace
    # every '/' and '\' with '_', producing flat names like
    # "USED_for_time_series_dataset_train_idle.csv". Strip to basename first
    # so the leaf name stays clean; nested folders are re-created below from
    # the separately-sent `relative_path` form field.
    fname_raw = (file.filename or '').replace('\\', '/')
    fname_base = os.path.basename(fname_raw)
    original_filename = secure_filename(fname_base)
    if not original_filename:
        original_filename = f"upload_{uuid.uuid4().hex[:8]}.csv"

    # Optional: folder-upload relative path. Preserves nested directory
    # structure (e.g. ``dataset/train/idle.csv``). We sanitize the incoming
    # path to reject absolute paths, drive letters, and parent traversal so a
    # malicious client cannot escape ``upload_dir``.
    relative_path = (request.form.get('relative_path') or '').strip()
    save_dir = upload_dir
    if relative_path:
        # Normalize separators, strip leading slashes, split into parts.
        rp = relative_path.replace('\\', '/').lstrip('/')
        # Reject drive letters like "C:/foo"
        if len(rp) >= 2 and rp[1] == ':':
            return jsonify({'error': 'Invalid relative_path (drive letter not allowed)'}), 400
        parts = [p for p in rp.split('/') if p not in ('', '.')]
        if any(p == '..' for p in parts):
            return jsonify({'error': 'Invalid relative_path (parent traversal not allowed)'}), 400
        # Last part is the filename — replace with the secured version.
        if parts:
            dir_parts = [secure_filename(p) for p in parts[:-1]]
            # Drop any parts that secure_filename reduced to empty
            dir_parts = [p for p in dir_parts if p]
            if dir_parts:
                save_dir = os.path.join(upload_dir, *dir_parts)
                # Final containment check
                save_dir_norm = os.path.normpath(os.path.abspath(save_dir))
                upload_dir_norm = os.path.normpath(os.path.abspath(upload_dir))
                if not save_dir_norm.startswith(upload_dir_norm + os.sep) and save_dir_norm != upload_dir_norm:
                    return jsonify({'error': 'Invalid relative_path (escapes upload directory)'}), 400
                os.makedirs(save_dir, exist_ok=True)

    # Check for existing file and add suffix if needed
    base_name, extension = os.path.splitext(original_filename)
    final_filename = original_filename
    counter = 1

    while os.path.exists(os.path.join(save_dir, final_filename)):
        final_filename = f"{base_name}_{counter}{extension}"
        counter += 1

    # Save the file
    file_path = os.path.join(save_dir, final_filename)
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    # Determine file type
    ext = extension.lower().lstrip('.')
    file_type = {
        'csv': 'csv',
        'json': 'json',
        'cbor': 'cbor',
        'txt': 'text',
        'tsv': 'text',
        'dat': 'text',
        'log': 'text',
    }.get(ext, 'unknown')

    return jsonify({
        'success': True,
        'filename': final_filename,
        'path': file_path,
        'size': file_size,
        'file_type': file_type,
        'upload_dir': upload_dir
    })


@data_sources_bp.route('/upload-multiple', methods=['POST'])
@login_required
def upload_multiple_files():
    """
    Upload multiple dataset files at once.

    Files are saved to a new folder in shared/uploads.
    Useful for Edge Impulse format which has training/testing folders.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    # Get folder name from form data
    folder_name = request.form.get('folder_name', '')
    if not folder_name:
        folder_name = f"upload_{uuid.uuid4().hex[:8]}"
    folder_name = secure_filename(folder_name)

    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    user = request.current_user

    # Create upload directory
    upload_dir = os.path.join(shared_folder, 'uploads', f"user_{user['id']}", folder_name)
    os.makedirs(upload_dir, exist_ok=True)

    uploaded_files = []
    errors = []
    total_size = 0

    for file in files:
        if file.filename == '':
            continue

        if not allowed_file(file.filename):
            errors.append(f'{file.filename}: File type not allowed')
            continue

        # Check file size
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            errors.append(f'{file.filename}: File too large')
            continue

        # Preserve relative path structure if provided (for folder uploads)
        relative_path = request.form.get(f'path_{file.filename}', '')
        if relative_path:
            # Create subdirectories if needed (e.g., training/testing)
            subdir = os.path.dirname(secure_filename(relative_path.replace('\\', '/')))
            if subdir:
                target_dir = os.path.join(upload_dir, subdir)
                os.makedirs(target_dir, exist_ok=True)
            else:
                target_dir = upload_dir
        else:
            target_dir = upload_dir

        # Strip path from file.filename before securing — see comment on the
        # single-file /upload endpoint for why this matters (Firefox / Linux
        # drag-drop send full relative path in filename).
        fname_raw = (file.filename or '').replace('\\', '/')
        filename = secure_filename(os.path.basename(fname_raw))
        file_path = os.path.join(target_dir, filename)

        try:
            file.save(file_path)
            total_size += file_size
            uploaded_files.append({
                'filename': filename,
                'path': file_path,
                'size': file_size
            })
        except Exception as e:
            errors.append(f'{file.filename}: {str(e)}')

    return jsonify({
        'success': len(uploaded_files) > 0,
        'uploaded_count': len(uploaded_files),
        'uploaded_files': uploaded_files,
        'errors': errors,
        'total_size': total_size,
        'upload_dir': upload_dir
    })


@data_sources_bp.route('/delete-upload', methods=['POST'])
@login_required
def delete_uploaded_file():
    """
    Delete a file or folder from user's accessible locations.

    Users can delete files from:
    - Their own uploads folder (shared/uploads/user_{id}/)
    - Their private folder
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    user = request.current_user

    # Normalize paths
    file_path_norm = os.path.normpath(os.path.abspath(file_path))
    datasets_root_norm = os.path.normpath(os.path.abspath(datasets_root))

    # File must be within datasets root
    if not _is_path_within(file_path_norm, datasets_root_norm):
        return jsonify({'error': 'Access denied'}), 403

    # Check if user can delete this file
    can_delete = False
    delete_reason = ''

    # Check 1: User's own uploads folder
    user_uploads_path = os.path.normpath(
        os.path.join(datasets_root, shared_folder, 'uploads', f"user_{user['id']}")
    )
    if _is_path_within(file_path_norm, user_uploads_path):
        can_delete = True
        delete_reason = 'user_uploads'

    # Check 2: User's private folder
    private_folder = user.get('private_folder')
    if private_folder:
        private_path = os.path.normpath(os.path.join(datasets_root, private_folder))
        if _is_path_within(file_path_norm, private_path):
            can_delete = True
            delete_reason = 'private_folder'

    # Check 3: Admin can delete from anywhere (except protected folders)
    if user.get('role') == 'admin':
        can_delete = True
        delete_reason = 'admin'

    if not can_delete:
        return jsonify({'error': 'You can only delete files from your own folders'}), 403

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        import shutil

        is_dir = os.path.isdir(file_path)
        file_name = os.path.basename(file_path)

        if is_dir:
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

        return jsonify({
            'success': True,
            'deleted': file_path,
            'type': 'directory' if is_dir else 'file',
            'name': file_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_sources_bp.route('/admin/delete', methods=['POST'])
@login_required
def admin_delete_file():
    """
    Admin-only endpoint to delete any file or folder in the datasets directory.

    This is a powerful operation - use with caution!
    Only admins can access this endpoint.
    """
    user = request.current_user

    # Check admin role
    if user.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    # Normalize paths for comparison
    file_path = os.path.normpath(os.path.abspath(file_path))
    datasets_root_norm = os.path.normpath(os.path.abspath(datasets_root))

    # Safety check: file must be within datasets root
    if not _is_path_within(file_path, datasets_root_norm):
        return jsonify({'error': 'File must be within datasets directory'}), 403

    # Prevent deleting the datasets root itself
    if file_path == datasets_root_norm:
        return jsonify({'error': 'Cannot delete the datasets root directory'}), 403

    # Prevent deleting the shared folder root
    shared_folder_norm = os.path.normpath(os.path.abspath(shared_folder))
    if file_path == shared_folder_norm:
        return jsonify({'error': 'Cannot delete the shared folder root'}), 403

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        import shutil

        is_dir = os.path.isdir(file_path)
        file_name = os.path.basename(file_path)

        if is_dir:
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

        return jsonify({
            'success': True,
            'deleted': file_path,
            'type': 'directory' if is_dir else 'file',
            'name': file_name
        })
    except PermissionError:
        return jsonify({'error': 'Permission denied - file may be in use'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_sources_bp.route('/sensor-files-for-date', methods=['POST'])
@login_required
def sensor_files_for_date():
    """List one CSV per child sensor folder for a given date.

    Used by the file-browser "Load all sensors" button. Body:
        folder_path: absolute path to a machine folder whose direct children
                     are sensor folders (each containing daily CSVs).
        date:        YYYY-MM-DD (default: today).

    Returns:
        {
          "folder_path": "...",
          "date": "2026-07-21",
          "sensor_files": [
            {"sensor": "pressure", "file": "2026-07-21.csv",
             "path": "<abs path>", "exists": true},
            {"sensor": "temperature", "file": "2026-07-21.csv",
             "path": "<abs path>", "exists": false},
            ...
          ]
        }

    A sensor entry with `exists: false` is returned so the UI can tell the
    user which sensors have no file for that date rather than silently
    dropping them from the basket.
    """
    from datetime import date as _date
    data = request.get_json() or {}
    folder_path = data.get('folder_path')
    date_str = data.get('date') or _date.today().isoformat()

    if not folder_path:
        return jsonify({'error': 'folder_path required'}), 400
    if not isinstance(date_str, str) or len(date_str) != 10 or date_str.count('-') != 2:
        return jsonify({'error': 'date must be YYYY-MM-DD'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    if not validate_path(folder_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403
    if not os.path.isdir(folder_path):
        return jsonify({'error': 'Not a directory'}), 404

    filename = f"{date_str}.csv"
    sensor_files = []
    try:
        for entry in sorted(os.listdir(folder_path)):
            if entry.startswith('.'):
                continue  # hide .multi_csv_selections / other internal dirs
            child = os.path.join(folder_path, entry)
            if not os.path.isdir(child):
                continue
            candidate = os.path.join(child, filename)
            sensor_files.append({
                'sensor': entry,
                'file': filename,
                'path': candidate,
                'exists': os.path.isfile(candidate),
            })
    except OSError as e:
        return jsonify({'error': f'listdir failed: {e}'}), 500

    return jsonify({
        'folder_path': folder_path,
        'date': date_str,
        'sensor_files': sensor_files,
    })


@data_sources_bp.route('/scan', methods=['POST'])
@login_required
def scan_dataset():
    """Scan a dataset folder to get its structure without loading data."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    folder_path = data.get('folder_path')
    if not folder_path:
        return jsonify({'error': 'Folder path required'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(folder_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403

    try:
        loader = DataLoader()
        result = loader.scan_dataset_folder(folder_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/ingest/csv', methods=['POST'])
@login_required
def ingest_csv():
    """Ingest data from a CSV file."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this file'}), 403

    try:
        loader = DataLoader()
        result = loader.load_csv(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/ingest/csv-multiple', methods=['POST'])
@login_required
def ingest_csv_multiple():
    """Ingest data from multiple CSV files as one dataset."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_paths = data.get('file_paths')
    if not file_paths or not isinstance(file_paths, list):
        return jsonify({'error': 'file_paths (list) required'}), 400

    # Validate all paths
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    for fp in file_paths:
        if not validate_path(fp, request.current_user, datasets_root, shared_folder):
            return jsonify({'error': f'Access denied to: {fp}'}), 403

    # Cross-sensor JOIN parameters (all optional). When merge_mode is
    # omitted, load_csv_multiple auto-detects: same folder → row concat,
    # different folders → JOIN by timestamp.
    merge_mode = data.get('merge_mode')  # None | 'concat' | 'join'
    alignment = data.get('alignment', 'exact')  # 'exact' | 'nearest' | 'resample'
    tolerance_ms = data.get('tolerance_ms')  # required if alignment='nearest'
    resample_hz = data.get('resample_hz')  # required if alignment='resample'

    try:
        loader = DataLoader()
        result = loader.load_csv_multiple(
            file_paths,
            merge_mode=merge_mode,
            alignment=alignment,
            tolerance_ms=tolerance_ms,
            resample_hz=resample_hz,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/ingest/ei-json', methods=['POST'])
@login_required
def ingest_edge_impulse_json():
    """Ingest data from Edge Impulse JSON format."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this file'}), 403

    try:
        loader = DataLoader()
        result = loader.load_edge_impulse_json(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/ingest/ei-cbor', methods=['POST'])
@login_required
def ingest_edge_impulse_cbor():
    """Ingest data from Edge Impulse CBOR format."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this file'}), 403

    try:
        loader = DataLoader()
        result = loader.load_edge_impulse_cbor(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/ingest/cira-cbor', methods=['POST'])
@login_required
def ingest_cira_cbor():
    """Ingest data from CiRA CBOR format."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this file'}), 403

    try:
        loader = DataLoader()
        result = loader.load_cira_cbor(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/preview', methods=['POST'])
@login_required
def preview_data():
    """Preview data from any supported format. Supports partition filtering for folders."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    file_paths = data.get('file_paths')  # Multi-CSV support
    rows = data.get('rows', 10)
    format_hint = data.get('format')
    category = data.get('category')  # Optional: partition filter
    label = data.get('label')        # Optional: label filter
    # Text-format overrides — the Text Import wizard passes these after the
    # user confirms delimiter / header row / skip rows.
    delimiter = data.get('delimiter')
    header_row = data.get('header_row')
    skip_rows = data.get('skip_rows')

    if not file_path and not file_paths:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    try:
        loader = DataLoader()

        # Multi-CSV preview — same cross-sensor JOIN params as /ingest/csv-multiple.
        if file_paths and isinstance(file_paths, list) and len(file_paths) > 1:
            for fp in file_paths:
                if not validate_path(fp, request.current_user, datasets_root, shared_folder):
                    return jsonify({'error': f'Access denied to: {fp}'}), 403
            result = loader.load_csv_multiple(
                file_paths,
                merge_mode=data.get('merge_mode'),
                alignment=data.get('alignment', 'exact'),
                tolerance_ms=data.get('tolerance_ms'),
                resample_hz=data.get('resample_hz'),
            )
            # Limit preview rows
            session = loader._get_session(result['session_id'])
            if session:
                result['preview'] = session['data'].head(rows).to_dict(orient='records')
            return jsonify(result)

        if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
            return jsonify({'error': 'Access denied to this path'}), 403

        # Text format — always go through load_text with wizard settings.
        if format_hint == 'text' and not os.path.isdir(file_path):
            result = loader.load_text(
                file_path,
                delimiter=delimiter,
                header_row=header_row if header_row is not None else 1,
                skip_rows=skip_rows if skip_rows is not None else 0,
            )
            session = loader._get_session(result['session_id'])
            if session:
                result['preview'] = session['data'].head(rows).to_dict(orient='records')
            return jsonify(result)

        # Use partition preview if category/label filters provided on a directory
        if os.path.isdir(file_path) and category is not None:
            result = loader.preview_partition(file_path, category=category, label=label, rows=rows, format_hint=format_hint)
        else:
            result = loader.preview(file_path, rows, format_hint)

        return jsonify(result)
    except DataValidationError as e:
        return jsonify({
            'error': e.message,
            'error_code': e.code,
            'hint': e.hint,
        }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/load-from-url', methods=['POST'])
@login_required
def load_from_url():
    """
    Fetch a remote CSV / text file over HTTPS and load it directly into a session.

    Streams to a temp file (100 MB hard cap), dispatches to load_csv or load_text,
    and always deletes the temp file. No file is persisted to server storage —
    the caller sees a normal Data Preview response with ``metadata.source_url`` set.
    """
    data = request.get_json() or {}

    url = (data.get('url') or '').strip()
    fmt = (data.get('format') or 'csv').lower()
    delimiter = data.get('delimiter')
    header_row = data.get('header_row', 1)
    skip_rows = data.get('skip_rows', 0)
    column_names = data.get('column_names')
    if column_names is not None:
        if not isinstance(column_names, list) or not all(isinstance(x, str) for x in column_names):
            column_names = None

    # --- URL validation --------------------------------------------------
    if not url:
        return jsonify({
            'error': 'URL is required.',
            'error_code': 'INVALID_URL',
            'hint': 'Paste a direct https:// link to a CSV or text file.',
        }), 400

    if not url.lower().startswith('https://'):
        return jsonify({
            'error': 'Only https:// URLs are allowed.',
            'error_code': 'INVALID_URL',
            'hint': 'Use a secure https:// link. http:// and other schemes are blocked for safety.',
        }), 400

    if fmt not in ('csv', 'text'):
        return jsonify({
            'error': f"Unsupported format '{fmt}'.",
            'error_code': 'INVALID_URL',
            'hint': "Format must be 'csv' or 'text'.",
        }), 400

    max_bytes = 100 * 1024 * 1024  # 100 MB

    # Derive a suffix from the URL basename so pandas' auto-parsers see the
    # right extension when they sniff the temp file.
    try:
        url_basename = os.path.basename(urlsplit(url).path) or ''
    except Exception:
        url_basename = ''
    suffix = os.path.splitext(url_basename)[1].lower()
    if suffix not in ('.csv', '.txt', '.tsv', '.dat', '.log'):
        suffix = '.csv' if fmt == 'csv' else '.txt'

    temp_path = None
    resp = None
    try:
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            return jsonify({
                'error': f'Failed to fetch URL: {e}',
                'error_code': 'URL_FETCH_FAILED',
                'hint': 'Check the URL is reachable and returns a direct file (not an HTML page).',
            }), 400

        # Pre-flight size check via Content-Length (server may lie / omit).
        content_length = resp.headers.get('Content-Length')
        if content_length:
            try:
                declared = int(content_length)
                if declared > max_bytes:
                    return jsonify({
                        'error': (
                            f'Remote file is {declared // (1024*1024)} MB, '
                            f'which exceeds the {max_bytes // (1024*1024)} MB limit.'
                        ),
                        'error_code': 'URL_TOO_LARGE',
                        'hint': 'Download the file manually and upload it via the Upload dialog.',
                    }), 400
            except (TypeError, ValueError):
                pass  # Ignore non-integer Content-Length; enforce during stream.

        # Stream to temp file with running-total enforcement.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    return jsonify({
                        'error': (
                            f'Download exceeded the {max_bytes // (1024*1024)} MB limit '
                            'during streaming.'
                        ),
                        'error_code': 'URL_TOO_LARGE',
                        'hint': 'Download the file manually and upload it via the Upload dialog.',
                    }), 400
                tmp.write(chunk)

        # Dispatch to the existing loader — no changes to data_loader.py.
        loader = DataLoader()
        try:
            if fmt == 'text':
                result = loader.load_text(
                    temp_path,
                    delimiter=delimiter,
                    header_row=header_row if header_row is not None else 1,
                    skip_rows=skip_rows if skip_rows is not None else 0,
                    column_names=column_names,
                )
            else:
                result = loader.load_csv(temp_path)
        except DataValidationError as e:
            return jsonify({
                'error': e.message,
                'error_code': e.code,
                'hint': e.hint,
            }), 400

        # Overwrite the temp path in metadata so it doesn't leak, and add
        # source_url so the frontend can display provenance.
        meta = result.get('metadata') or {}
        meta['source_url'] = url
        meta['file_path'] = url  # hide the temp path from downstream UI
        result['metadata'] = meta

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@data_sources_bp.route('/text-sniff', methods=['POST'])
@login_required
def text_sniff():
    """Sniff the delimiter of a text file and return a head of raw lines
    for the Text Import wizard's client-side preview.

    Request body:
        file_path (str): absolute path to the text file
        sample_bytes (int, optional): bytes to read for sniffing (default 4096)

    Response:
        detected_delimiter (str): the sniffed delimiter (falls back to ',')
        raw_lines (list[str]): up to 20 head lines (already stripped of \\r\\n)
        encoding (str): always 'utf-8' — the wizard has no encoding picker
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    sample_bytes = data.get('sample_bytes', 4096)
    try:
        sample_bytes = int(sample_bytes)
    except (TypeError, ValueError):
        sample_bytes = 4096
    # Clamp to a sensible range so a malicious client can't ask for a huge read
    sample_bytes = max(256, min(sample_bytes, 65536))

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        loader = DataLoader()
        detected = loader._sniff_text_delimiter(file_path, sample_bytes=sample_bytes)

        # Read the head of the file as raw lines for the wizard preview.
        raw_lines = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                # Strip trailing newline but keep in-line whitespace so the
                # preview matches what the user sees in a text editor.
                raw_lines.append(line.rstrip('\r\n'))

        # Heuristic: detect a preamble of comment / short lines so the wizard
        # can pre-populate "Skip N rows from top". Split each line by the
        # detected delimiter, find the most common field count, and count
        # leading lines with a different (usually smaller) count.
        suggested_skip = 0
        if raw_lines:
            counts = [len(line.split(detected)) for line in raw_lines]
            # Most common non-1 count = expected column count. Skip leading
            # lines that differ from it (they're preamble / comments).
            from collections import Counter
            freq = Counter(c for c in counts if c > 1)
            if freq:
                expected = freq.most_common(1)[0][0]
                for cnt in counts:
                    if cnt == expected:
                        break
                    suggested_skip += 1
                # Safety: don't skip everything.
                if suggested_skip >= len(raw_lines):
                    suggested_skip = 0

        return jsonify({
            'detected_delimiter': detected,
            'raw_lines': raw_lines,
            'suggested_skip_rows': suggested_skip,
            'encoding': 'utf-8',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/load-full', methods=['POST'])
@login_required
def load_full_dataset():
    """Load the complete dataset into a session for windowing."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    folder_path = data.get('folder_path')
    format_hint = data.get('format')
    preview_session_id = data.get('preview_session_id')

    if not folder_path:
        return jsonify({'error': 'Folder path required'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(folder_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403

    try:
        loader = DataLoader()
        result = loader.load_full_dataset(folder_path, format_hint, preview_session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/windowing', methods=['POST'])
@login_required
def apply_windowing():
    """Apply windowing to loaded data.

    F4 Q4: Persist data_sessions + windowed_sessions on this apply boundary
    if a project_id was provided. Latest-apply wins for current_stage.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    window_size = data.get('window_size', 128)
    stride = data.get('stride', 64)
    label_method = data.get('label_method', 'majority')
    test_ratio = data.get('test_ratio', 0.2)
    target_column = data.get('target_column')
    selected_columns = data.get('selected_columns')
    split_strategy = data.get('split_strategy', 'temporal_end')
    no_windowing = data.get('no_windowing', False)
    # F3: user-selectable normalization. Default 'min_max' preserves prior behavior.
    normalization_method = data.get('normalization_method', 'min_max')
    project_id = data.get('project_id')

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        loader = DataLoader()
        result = loader.apply_windowing(
            session_id,
            window_size=window_size,
            stride=stride,
            label_method=label_method,
            test_ratio=test_ratio,
            target_column=target_column,
            selected_columns=selected_columns,
            split_strategy=split_strategy,
            no_windowing=no_windowing,
            normalization_method=normalization_method
        )
        # F4: persist DataSession + WindowedSession rows on Apply (Q4)
        if project_id:
            try:
                pid = int(project_id)
                proj = Project.get_by_id(pid)
                if proj and proj.get('user_id') == request.current_user['id']:
                    # Load original data session to snapshot format/rows
                    src = loader._get_session(session_id) or {}
                    src_meta = (src.get('metadata') or {})
                    ds_id = DataSession.create(
                        project_id=pid,
                        file_path=src_meta.get('file_path', ''),
                        format=src_meta.get('format', 'unknown'),
                        session_id=session_id,
                        sensor_columns=src_meta.get('sensor_columns'),
                        label_column=src_meta.get('label_column'),
                        labels=src_meta.get('labels'),
                        total_rows=src_meta.get('total_rows'),
                    )
                    win_meta = result.get('metadata') or {}
                    windowed_sid = result.get('session_id')
                    WindowedSession.create(
                        project_id=pid,
                        data_session_id=ds_id,
                        config={
                            'window_size': window_size,
                            'stride': stride,
                            'label_method': label_method,
                            'test_ratio': test_ratio,
                            'split_strategy': split_strategy,
                            'no_windowing': no_windowing,
                            'normalization_method': normalization_method,
                            'target_column': target_column,
                            'selected_columns': selected_columns,
                        },
                        num_windows=result.get('num_windows'),
                        window_shape=result.get('window_shape'),
                        normalization=win_meta.get('normalization'),
                        session_id=windowed_sid,
                    )
                    Project.touch(pid, 'windowing')

                    # Approach 2b: pickle the raw + windowed session dicts
                    # from the loader's in-memory store so we can restore
                    # them on hydrate without re-parsing the CSV or
                    # re-windowing. Runs BELOW the DB writes so we never
                    # persist a pickle without a matching DB row.
                    try:
                        from ..services.data_loader import _data_sessions
                        from ..services.session_persistence import (
                            persist_data_session, persist_windowed_session,
                        )
                        raw = _data_sessions.get(session_id)
                        if raw is not None:
                            persist_data_session(pid, session_id, raw)
                        win = _data_sessions.get(windowed_sid) if windowed_sid else None
                        if win is not None:
                            persist_windowed_session(pid, windowed_sid, win)
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"[persist] Windowing pickle failed: {e}"
                        )
            except Exception as e:
                # Don't let persistence break the windowing response
                import logging
                logging.getLogger(__name__).warning(
                    f"[F4] Persisting windowing state failed: {e}"
                )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/window-sample', methods=['POST'])
@login_required
def get_window_sample():
    """Get a single window sample for visualization."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    window_index = data.get('window_index', 0)

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        loader = DataLoader()
        result = loader.get_window_sample(session_id, window_index)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/windows-by-label', methods=['POST'])
@login_required
def get_windows_by_label():
    """Get window indices grouped by label for filtering."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    label = data.get('label')  # Optional: filter by specific label

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        loader = DataLoader()
        result = loader.get_windows_by_label(session_id, label)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@data_sources_bp.route('/download', methods=['GET'])
@login_required
def download_file():
    """Download a file from the datasets directory."""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this file'}), 403

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(file_path)
    )


# ---------------------------------------------------------------------------
# File Manager endpoints — full folder/file management for the Manage Files UI
# ---------------------------------------------------------------------------
# All four endpoints:
#   * gated by @login_required
#   * validate every path with validate_path (user must be inside their
#     allowed area — private folder + shared folder + admin-any-datasets)
#   * reject targets that already exist (409)
#   * reject sources equal to destination (400)
#   * reject anything that would leave the datasets root


def _norm(p: str) -> str:
    """Absolute + normpath. Used everywhere for containment / equality checks."""
    return os.path.normpath(os.path.abspath(p))


@data_sources_bp.route('/mkdir', methods=['POST'])
@login_required
def create_folder():
    """
    Create a new (sub-)folder inside an allowed area.

    Body: { folder: <parent-abs-path>, name: <new-folder-name> }
    """
    data = request.get_json() or {}
    parent = data.get('folder')
    raw_name = (data.get('name') or '').strip()

    if not parent or not raw_name:
        return jsonify({'error': 'folder and name are required'}), 400

    name = secure_filename(raw_name)
    if not name:
        return jsonify({'error': 'Invalid folder name'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    user = request.current_user

    if not validate_path(parent, user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to parent folder'}), 403

    if not os.path.isdir(parent):
        return jsonify({'error': 'Parent folder not found'}), 404

    target = _norm(os.path.join(parent, name))

    # Re-validate the target — belt & suspenders against a name that somehow
    # dodges secure_filename (shouldn't happen, but this is cheap insurance).
    if not validate_path(target, user, datasets_root, shared_folder):
        return jsonify({'error': 'Target path outside allowed area'}), 403

    if os.path.exists(target):
        return jsonify({'error': 'A file or folder with that name already exists'}), 409

    try:
        os.makedirs(target)
    except OSError as e:
        return jsonify({'error': f'Failed to create folder: {e}'}), 500

    return jsonify({'ok': True, 'path': target})


@data_sources_bp.route('/rename', methods=['POST'])
@login_required
def rename_item():
    """
    Rename a single file or folder in place.

    Body: { path: <abs-src-path>, new_name: <new-basename> }
    """
    data = request.get_json() or {}
    src = data.get('path')
    raw_new = (data.get('new_name') or '').strip()

    if not src or not raw_new:
        return jsonify({'error': 'path and new_name are required'}), 400

    new_name = secure_filename(raw_new)
    if not new_name:
        return jsonify({'error': 'Invalid new_name'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    user = request.current_user

    if not validate_path(src, user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to source path'}), 403

    if not os.path.exists(src):
        return jsonify({'error': 'Source path not found'}), 404

    parent = os.path.dirname(_norm(src))
    target = _norm(os.path.join(parent, new_name))

    if target == _norm(src):
        return jsonify({'error': 'New name is the same as current name'}), 400

    if not validate_path(target, user, datasets_root, shared_folder):
        return jsonify({'error': 'Target path outside allowed area'}), 403

    if os.path.exists(target):
        return jsonify({'error': 'A file or folder with that name already exists'}), 409

    try:
        os.rename(src, target)
    except OSError as e:
        return jsonify({'error': f'Rename failed: {e}'}), 500

    return jsonify({'ok': True, 'path': target})


def _bulk_transfer(op_name: str):
    """
    Shared body for /move and /copy — both take the same request shape,
    only differ in the per-item action.
    """
    data = request.get_json() or {}
    sources = data.get('sources')
    destination = data.get('destination')

    if not isinstance(sources, list) or not sources:
        return jsonify({'error': 'sources (non-empty list) is required'}), 400

    if not destination or not isinstance(destination, str):
        return jsonify({'error': 'destination is required'}), 400

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    user = request.current_user

    if not validate_path(destination, user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to destination'}), 403

    if not os.path.isdir(destination):
        return jsonify({'error': 'Destination is not a folder'}), 400

    dest_norm = _norm(destination)

    ok_paths = []
    errors = []

    for src in sources:
        try:
            if not isinstance(src, str) or not src:
                errors.append({'path': str(src), 'reason': 'Invalid source path'})
                continue

            if not validate_path(src, user, datasets_root, shared_folder):
                errors.append({'path': src, 'reason': 'Access denied'})
                continue

            if not os.path.exists(src):
                errors.append({'path': src, 'reason': 'Not found'})
                continue

            src_norm = _norm(src)
            name = os.path.basename(src_norm)
            if not name:
                errors.append({'path': src, 'reason': 'Cannot resolve basename'})
                continue

            target = _norm(os.path.join(dest_norm, name))

            # Refuse no-op operations
            if src_norm == target:
                errors.append({'path': src, 'reason': 'Source and destination are the same'})
                continue

            # Refuse moving/copying a folder into itself or a child of itself
            if os.path.isdir(src_norm) and _is_path_within(dest_norm, src_norm):
                errors.append({'path': src, 'reason': 'Cannot place folder into itself'})
                continue

            if not validate_path(target, user, datasets_root, shared_folder):
                errors.append({'path': src, 'reason': 'Target outside allowed area'})
                continue

            if os.path.exists(target):
                errors.append({'path': src, 'reason': 'Target already exists'})
                continue

            if op_name == 'move':
                shutil.move(src_norm, target)
            else:  # copy
                if os.path.isdir(src_norm):
                    shutil.copytree(src_norm, target, dirs_exist_ok=False)
                else:
                    shutil.copy2(src_norm, target)

            ok_paths.append(target)
        except Exception as e:
            errors.append({'path': src, 'reason': str(e)})

    key = 'moved' if op_name == 'move' else 'copied'
    return jsonify({'ok': True, key: ok_paths, 'errors': errors})


@data_sources_bp.route('/move', methods=['POST'])
@login_required
def move_items():
    """
    Bulk move files/folders.

    Body: { sources: [<abs>...], destination: <abs-folder> }
    Per-item failures are collected; the request only 400s on shape errors.
    """
    return _bulk_transfer('move')


@data_sources_bp.route('/copy', methods=['POST'])
@login_required
def copy_items():
    """
    Bulk copy files/folders.

    Body: { sources: [<abs>...], destination: <abs-folder> }
    Files use shutil.copy2 (preserves metadata); folders use shutil.copytree
    with dirs_exist_ok=False (existing destination is an error).
    """
    return _bulk_transfer('copy')


# ── Phase G — Labels sidecar (in-app timeline labeler) ───────────────────
#
# GET/PUT /api/data/labels — sidecar labels colocated with the source CSV.
# See docs/PLAN_2026-07-22_labeler-and-profile-swap.md for the JSON format.
#
# Design note: the plan doc uses `/api/pipeline/data/labels` in a spec-side
# table; the actual mount path is `/api/data/labels` because that's where
# the rest of the data-source pipeline lives (see backend/app/__init__.py
# `register_blueprint(data_sources_bp, url_prefix='/api/data')`). The
# frontend already talks to `/api/data/*` for preview + ingest, so
# consistency wins over the plan's aspirational prefix.


def _labels_sidecar_path(csv_path: str) -> str:
    """Return the sidecar path colocated with the given CSV: same directory,
    same stem, `.labels.json` extension. Case-preserving on Windows and Linux.
    """
    directory = os.path.dirname(csv_path)
    base = os.path.basename(csv_path)
    stem, _ext = os.path.splitext(base)
    return os.path.join(directory, f'{stem}.labels.json')


def _resolve_labels_csv_path(raw_path):
    """Resolve a caller-provided csv_path to an absolute, validated path.

    Returns (abs_path, error_response_or_none). On failure error_response is
    a Flask (jsonify, status) tuple ready to return; on success it's None.
    """
    if not raw_path or not isinstance(raw_path, str):
        return None, (jsonify({'error': "'csv_path' (string) required"}), 400)

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    # Reject empty / null bytes early — os.path.abspath silently normalizes
    # `\x00` on some platforms which would slip through validate_path.
    if '\x00' in raw_path:
        return None, (jsonify({'error': 'csv_path contains null byte'}), 400)

    # Resolve to absolute. abspath() normalizes `..` sequences; the subsequent
    # validate_path enforces the datasets-root/shared-folder prefix so any
    # remaining traversal ends up outside the allowed prefix and gets rejected.
    abs_path = os.path.abspath(raw_path)

    if not validate_path(
        abs_path, request.current_user, datasets_root, shared_folder,
    ):
        return None, (jsonify({'error': 'Access denied to this path'}), 403)

    # Only allow CSV targets — labels don't apply to CBOR/JSON here and the
    # sidecar filename convention is `<csv_stem>.labels.json`.
    if not abs_path.lower().endswith('.csv'):
        return None, (jsonify({
            'error': 'csv_path must end with .csv'
        }), 400)

    return abs_path, None


def _validate_labels_payload(payload):
    """Validate the labels array. Returns error message on failure, else None.

    Rules:
    - each label has numeric `from` < `to`
    - each label has non-empty `class` string
    - no two labels overlap (ranges are half-open `[from, to)` for the check)
    """
    if not isinstance(payload, list):
        return "'labels' must be an array"

    normalised = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            return f'labels[{i}] must be an object'
        try:
            frm = float(item['from'])
            to = float(item['to'])
        except (KeyError, TypeError, ValueError):
            return f'labels[{i}] requires numeric "from" and "to"'
        cls = item.get('class')
        if not isinstance(cls, str) or not cls.strip():
            return f'labels[{i}] requires non-empty "class" string'
        if not (frm < to):
            return f'labels[{i}] requires from ({frm}) < to ({to})'
        normalised.append((i, frm, to, cls.strip()))

    # Overlap check — sort by `from`, walk once.
    sorted_labels = sorted(normalised, key=lambda t: t[1])
    for a, b in zip(sorted_labels, sorted_labels[1:]):
        _ia, _fa, ta, ca = a
        ib, fb, _tb, cb = b
        if fb < ta:
            return (
                f'labels[{ib}] range overlaps existing label {ca!r} '
                f'at {a[1]}–{ta}'
            )
    return None


@data_sources_bp.route('/labels', methods=['GET'])
@login_required
def get_labels():
    """Read the sidecar labels for a CSV.

    Query: csv_path=<absolute-or-relative path>

    Returns the sidecar's parsed JSON when it exists, else an empty envelope
    `{"csv": <basename>, "x_column": null, "labels": []}`. Malformed sidecars
    are surfaced with `warning` in the response so the UI can flag them
    without the label mode falling over.
    """
    raw_path = request.args.get('csv_path', '', type=str)
    abs_path, err = _resolve_labels_csv_path(raw_path)
    if err is not None:
        return err

    sidecar_path = _labels_sidecar_path(abs_path)
    empty = {'csv': os.path.basename(abs_path), 'x_column': None, 'labels': []}
    if not os.path.isfile(sidecar_path):
        return jsonify(empty)

    try:
        with open(sidecar_path, 'r', encoding='utf-8') as f:
            parsed = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # HARD REQ: never crash on malformed sidecar. Log + return empty
        # with a warning so the UI can render it as "unparseable — start
        # fresh?" without breaking the label mode.
        logger.warning('[labels] sidecar %s malformed: %s', sidecar_path, e)
        out = dict(empty)
        out['warning'] = f'sidecar malformed: {e}'
        return jsonify(out)

    if not isinstance(parsed, dict):
        logger.warning('[labels] sidecar %s not an object', sidecar_path)
        out = dict(empty)
        out['warning'] = 'sidecar not a JSON object'
        return jsonify(out)

    # Coerce shape defensively — old sidecars may lack fields.
    labels = parsed.get('labels')
    if not isinstance(labels, list):
        labels = []
    return jsonify({
        'csv': parsed.get('csv') or os.path.basename(abs_path),
        'x_column': parsed.get('x_column'),
        'labels': labels,
        'updated_at': parsed.get('updated_at'),
        'updated_by': parsed.get('updated_by'),
    })


@data_sources_bp.route('/labels', methods=['PUT'])
@login_required
def put_labels():
    """Write the sidecar labels for a CSV. Any authed user allowed.

    Body:
        {
          "csv_path": "...",
          "x_column": "timestamp_iso" | null,
          "labels": [{"from": 0.0, "to": 45.12, "class": "idle"}, ...]
        }

    Writes atomically (temp file + rename) and stamps `updated_at` (UTC ISO)
    + `updated_by` (username). Returns the JSON that was written.
    """
    data = request.get_json(silent=True) or {}
    abs_path, err = _resolve_labels_csv_path(data.get('csv_path'))
    if err is not None:
        return err

    # Refuse to write a sidecar for a CSV that doesn't exist. Otherwise
    # any authed user could spray orphan .labels.json files (and their
    # parent directories) throughout the datasets root — no security
    # boundary crossed, but noisy garbage (QA G polish #6).
    if not os.path.isfile(abs_path):
        return jsonify({
            'error': f'csv_path does not point to an existing file: '
                     f'{data.get("csv_path")!r}',
        }), 400

    labels = data.get('labels', [])
    validation_error = _validate_labels_payload(labels)
    if validation_error is not None:
        return jsonify({'error': validation_error}), 400

    x_column = data.get('x_column')
    if x_column is not None and not isinstance(x_column, str):
        return jsonify({'error': "'x_column' must be a string or null"}), 400

    # Rebuild the labels array in a canonical shape so downstream consumers
    # can trust from/to are floats and class is stripped.
    canonical_labels = [
        {
            'from': float(l['from']),
            'to': float(l['to']),
            'class': str(l['class']).strip(),
        }
        for l in labels
    ]

    sidecar_path = _labels_sidecar_path(abs_path)
    envelope = {
        'csv': os.path.basename(abs_path),
        'x_column': x_column,
        'labels': canonical_labels,
        'updated_at': datetime.now(timezone.utc).isoformat().replace(
            '+00:00', 'Z'),
        'updated_by': (request.current_user or {}).get('username') or 'unknown',
    }

    # Atomic write: tempfile in the same directory (so os.replace stays on the
    # same filesystem), then os.replace over the target.
    parent_dir = os.path.dirname(sidecar_path)
    try:
        os.makedirs(parent_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(sidecar_path) + '.',
            suffix='.tmp',
            dir=parent_dir,
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(envelope, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass  # fsync is best-effort; not fatal on some FS
            os.replace(tmp_path, sidecar_path)
            # tempfile.mkstemp creates the file with 0600 (owner-only). The
            # sidecar is public-by-intent data; downstream consumers (data
            # loader, training, external scripts) may run under different
            # UIDs on customer installs. Widen to 0644 post-rename.
            # (QA G polish #7 — no-op on Windows where chmod is a stub.)
            try:
                os.chmod(sidecar_path, 0o644)
            except OSError:
                pass
        except Exception:
            # Clean up the temp file so failed writes don't accumulate.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.exception('[labels] atomic write failed for %s', sidecar_path)
        return jsonify({'error': f'Failed to write labels sidecar: {e}'}), 500

    logger.info(
        '[labels] wrote %d labels to %s (by %s)',
        len(canonical_labels), sidecar_path, envelope['updated_by'],
    )
    return jsonify(envelope)
