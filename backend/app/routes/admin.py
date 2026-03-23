"""
CiRA ME - Admin Routes
"""

import os
import sys
import time
import json
from flask import Blueprint, request, jsonify, current_app
from ..models import User, SavedModel, get_db
from ..auth import admin_required, login_required

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/users', methods=['GET'])
@admin_required
def list_users():
    """List all users (admin only)."""
    users = User.get_all()
    return jsonify({'users': users})


@admin_bp.route('/users', methods=['POST'])
@admin_required
def create_user():
    """Create a new user (admin only)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    username = data.get('username')
    password = data.get('password')
    display_name = data.get('display_name', username)
    role = data.get('role', 'annotator')
    private_folder = data.get('private_folder')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    if role not in ['admin', 'annotator']:
        return jsonify({'error': 'Invalid role'}), 400

    # Check if username exists
    if User.get_by_username(username):
        return jsonify({'error': 'Username already exists'}), 400

    # Create private folder if specified
    if private_folder:
        datasets_root = current_app.config['DATASETS_ROOT_PATH']
        folder_path = os.path.join(datasets_root, private_folder)
        os.makedirs(folder_path, exist_ok=True)

    user_id = User.create(username, password, display_name, role, private_folder)

    return jsonify({
        'message': 'User created successfully',
        'user_id': user_id
    }), 201


@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id: int):
    """Update a user (admin only)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    user = User.get_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Prevent modifying the main admin
    if user['username'] == 'admin' and request.current_user['id'] != user_id:
        return jsonify({'error': 'Cannot modify main admin user'}), 403

    # Create private folder if specified and different from current
    private_folder = data.get('private_folder')
    if private_folder and private_folder != user.get('private_folder'):
        datasets_root = current_app.config['DATASETS_ROOT_PATH']
        folder_path = os.path.join(datasets_root, private_folder)
        os.makedirs(folder_path, exist_ok=True)

    User.update(user_id, **data)

    return jsonify({'message': 'User updated successfully'})


@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id: int):
    """Delete a user (admin only)."""
    user = User.get_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Prevent deleting main admin
    if user['username'] == 'admin':
        return jsonify({'error': 'Cannot delete main admin user'}), 403

    # Prevent self-deletion
    if request.current_user['id'] == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 403

    User.delete(user_id)

    return jsonify({'message': 'User deleted successfully'})


@admin_bp.route('/users/<int:user_id>/password', methods=['PUT'])
@admin_required
def reset_user_password(user_id: int):
    """Reset a user's password (admin only)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    new_password = data.get('new_password')
    if not new_password or len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    user = User.get_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    User.change_password(user_id, new_password)

    return jsonify({'message': 'Password reset successfully'})


@admin_bp.route('/create-folder', methods=['POST'])
@admin_required
def create_folder():
    """Create a new dataset folder (admin only)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    folder_name = data.get('folder_name')
    if not folder_name:
        return jsonify({'error': 'Folder name required'}), 400

    # Sanitize folder name
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    folder_path = os.path.join(datasets_root, folder_name)

    if os.path.exists(folder_path):
        return jsonify({'error': 'Folder already exists'}), 400

    os.makedirs(folder_path)

    return jsonify({
        'message': 'Folder created successfully',
        'path': folder_path
    }), 201


@admin_bp.route('/delete-folder', methods=['POST'])
@admin_required
def delete_folder():
    """Delete a dataset folder (admin only)."""
    import shutil

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    folder_name = data.get('folder_name')
    if not folder_name:
        return jsonify({'error': 'Folder name required'}), 400

    # Prevent deleting shared folder
    if folder_name == current_app.config['SHARED_FOLDER_PATH']:
        return jsonify({'error': 'Cannot delete shared folder'}), 403

    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    folder_path = os.path.join(datasets_root, folder_name)

    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404

    shutil.rmtree(folder_path)

    return jsonify({'message': 'Folder deleted successfully'})


# ─── Dashboard Stats ──────────────────────────────────────────────

# Track server start time
_server_start_time = time.time()


@admin_bp.route('/dashboard-stats', methods=['GET'])
@login_required
def dashboard_stats():
    """Get system overview stats for the dashboard."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()

            # User stats
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            active_users = cursor.fetchone()[0]

            # Saved models stats
            cursor.execute('SELECT COUNT(*) FROM saved_models')
            total_models = cursor.fetchone()[0]

            # Models by mode
            cursor.execute('''
                SELECT mode, COUNT(*) as count
                FROM saved_models
                GROUP BY mode
            ''')
            models_by_mode = {row['mode']: row['count'] for row in cursor.fetchall()}

            # Models by algorithm
            cursor.execute('''
                SELECT algorithm, COUNT(*) as count
                FROM saved_models
                GROUP BY algorithm
                ORDER BY count DESC
                LIMIT 10
            ''')
            models_by_algorithm = {row['algorithm']: row['count'] for row in cursor.fetchall()}

            # Recent models (last 10)
            cursor.execute('''
                SELECT sm.id, sm.name, sm.algorithm, sm.mode, sm.metrics,
                       sm.created_at, u.display_name as user_name
                FROM saved_models sm
                LEFT JOIN users u ON sm.user_id = u.id
                ORDER BY sm.created_at DESC
                LIMIT 10
            ''')
            recent_models = []
            for row in cursor.fetchall():
                model = dict(row)
                # Parse metrics JSON
                if model.get('metrics') and isinstance(model['metrics'], str):
                    try:
                        model['metrics'] = json.loads(model['metrics'])
                    except (json.JSONDecodeError, TypeError):
                        model['metrics'] = {}
                recent_models.append(model)

        # System info
        uptime_seconds = time.time() - _server_start_time
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        if days > 0:
            uptime_str = f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            uptime_str = f"{hours}h {minutes}m"
        else:
            uptime_str = f"{minutes}m"

        # CPU info
        import platform
        import psutil
        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'usage_percent': psutil.cpu_percent(interval=None),  # non-blocking
            'processor': platform.processor() or platform.machine() or 'Unknown',
        }

        # Memory info
        mem = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(mem.total / (1024**3), 1),
            'used_gb': round(mem.used / (1024**3), 1),
            'available_gb': round(mem.available / (1024**3), 1),
            'usage_percent': mem.percent,
        }

        # GPU/torch info
        torch_available = False
        cuda_available = False
        gpu_info = None
        torch_version = None
        try:
            import torch
            torch_available = True
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                try:
                    gpu_info = {
                        'name': torch.cuda.get_device_name(0),
                        'count': torch.cuda.device_count(),
                        'cuda_version': getattr(torch.version, 'cuda', None) or 'N/A',
                    }
                    try:
                        props = torch.cuda.get_device_properties(0)
                        total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
                        gpu_info['memory_total_gb'] = round(total_mem / (1024**3), 1)
                    except Exception:
                        gpu_info['memory_total_gb'] = None
                    try:
                        gpu_info['memory_used_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
                        gpu_info['memory_reserved_gb'] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
                    except Exception:
                        pass
                except Exception as gpu_err:
                    gpu_info = {'error': str(gpu_err)}
        except (ImportError, OSError):
            pass

        # Disk info
        try:
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_gb': round(disk.total / (1024**3), 1),
                'used_gb': round(disk.used / (1024**3), 1),
                'free_gb': round(disk.free / (1024**3), 1),
                'usage_percent': disk.percent,
            }
        except Exception:
            disk_info = None

        # Models directory size
        models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'models')
        models_size_mb = 0
        if os.path.exists(models_path):
            for f in os.listdir(models_path):
                fp = os.path.join(models_path, f)
                if os.path.isfile(fp):
                    models_size_mb += os.path.getsize(fp)
            models_size_mb = round(models_size_mb / (1024 * 1024), 1)

        return jsonify({
            'users': {
                'total': total_users,
                'active': active_users,
            },
            'models': {
                'total': total_models,
                'by_mode': models_by_mode,
                'by_algorithm': models_by_algorithm,
            },
            'recent_models': recent_models,
            'system': {
                'uptime': uptime_str,
                'platform': platform.system() + ' ' + platform.release(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'torch_available': torch_available,
                'torch_version': torch_version,
                'cuda_available': cuda_available,
                'gpu': gpu_info,
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'models_disk_mb': models_size_mb,
                'version': '1.0.0',
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
