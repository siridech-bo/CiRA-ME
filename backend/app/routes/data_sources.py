"""
CiRA ME - Data Sources Routes
Handles CSV, Edge Impulse JSON, Edge Impulse CBOR, and CiRA CBOR formats
"""

import os
from flask import Blueprint, request, jsonify, current_app
from ..auth import login_required, validate_path, get_user_folders
from ..services.data_loader import DataLoader

data_sources_bp = Blueprint('data_sources', __name__)


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

        # Determine file type
        file_type = None
        if ext == '.csv':
            file_type = 'csv'
        elif ext == '.json':
            file_type = 'json'
        elif ext == '.cbor':
            file_type = 'cbor'

        items.append({
            'name': item,
            'path': item_path,
            'is_dir': is_dir,
            'extension': ext,
            'size': size,
            'file_type': file_type
        })

    # Sort: directories first, then files
    items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))

    return jsonify({
        'current_path': path,
        'items': items
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
    rows = data.get('rows', 10)
    format_hint = data.get('format')
    category = data.get('category')  # Optional: partition filter
    label = data.get('label')        # Optional: label filter

    if not file_path:
        return jsonify({'error': 'File path required'}), 400

    # Validate path access
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']

    if not validate_path(file_path, request.current_user, datasets_root, shared_folder):
        return jsonify({'error': 'Access denied to this path'}), 403

    try:
        loader = DataLoader()

        # Use partition preview if category/label filters provided on a directory
        if os.path.isdir(file_path) and category is not None:
            result = loader.preview_partition(file_path, category=category, label=label, rows=rows, format_hint=format_hint)
        else:
            result = loader.preview(file_path, rows, format_hint)

        return jsonify(result)
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
    """Apply windowing to loaded data."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    window_size = data.get('window_size', 128)
    stride = data.get('stride', 64)
    label_method = data.get('label_method', 'majority')

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        loader = DataLoader()
        result = loader.apply_windowing(
            session_id,
            window_size=window_size,
            stride=stride,
            label_method=label_method
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
