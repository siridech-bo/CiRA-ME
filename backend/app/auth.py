"""
CiRA ME - Authentication Utilities and Decorators
"""

from functools import wraps
from flask import session, jsonify, request, current_app
from datetime import datetime, timedelta
from .models import User


def login_required(f):
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        # Check session expiration
        login_time = session.get('login_time')
        if login_time:
            login_dt = datetime.fromisoformat(login_time)
            lifetime = current_app.config.get('SESSION_LIFETIME_HOURS', 8)
            if datetime.utcnow() - login_dt > timedelta(hours=lifetime):
                session.clear()
                return jsonify({'error': 'Session expired'}), 401

        # Get user and check if active
        user = User.get_by_id(user_id)
        if not user or not user.get('is_active'):
            session.clear()
            return jsonify({'error': 'User not found or inactive'}), 401

        # Attach user to request context
        request.current_user = user
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator to require admin role for a route."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        user = request.current_user
        if user.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)

    return decorated_function


def validate_path(path: str, user: dict, datasets_root: str, shared_folder: str) -> bool:
    """
    Validate that a user has access to the given path.
    Prevents directory traversal attacks.
    """
    import os

    # Normalize paths
    path = os.path.normpath(os.path.abspath(path))
    datasets_root = os.path.normpath(os.path.abspath(datasets_root))
    shared_path = os.path.normpath(os.path.join(datasets_root, shared_folder))

    # Path must be within datasets root
    if not path.startswith(datasets_root):
        return False

    # Admins can access anything within datasets root
    if user.get('role') == 'admin':
        return True

    # Annotators can access shared folder
    if path.startswith(shared_path):
        return True

    # Annotators can access their private folder
    private_folder = user.get('private_folder')
    if private_folder:
        private_path = os.path.normpath(os.path.join(datasets_root, private_folder))
        if path.startswith(private_path):
            return True

    return False


def get_user_folders(user: dict, datasets_root: str, shared_folder: str) -> list:
    """Get list of folders accessible to a user."""
    import os

    folders = []

    # Shared folder is accessible to all
    shared_path = os.path.join(datasets_root, shared_folder)
    if os.path.exists(shared_path):
        folders.append({
            'name': shared_folder,
            'path': shared_path,
            'type': 'shared'
        })

    # Admins get all folders
    if user.get('role') == 'admin':
        for item in os.listdir(datasets_root):
            item_path = os.path.join(datasets_root, item)
            if os.path.isdir(item_path) and item != shared_folder:
                folders.append({
                    'name': item,
                    'path': item_path,
                    'type': 'private'
                })
    else:
        # Annotators get their private folder
        private_folder = user.get('private_folder')
        if private_folder:
            private_path = os.path.join(datasets_root, private_folder)
            if os.path.exists(private_path):
                folders.append({
                    'name': private_folder,
                    'path': private_path,
                    'type': 'private'
                })

    return folders
