"""
CiRA ME - Admin Routes
"""

import os
from flask import Blueprint, request, jsonify, current_app
from ..models import User
from ..auth import admin_required

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
