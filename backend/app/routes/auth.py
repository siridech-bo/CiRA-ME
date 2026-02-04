"""
CiRA ME - Authentication Routes
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime
from ..models import User
from ..auth import login_required

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = User.get_by_username(username)

    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    if not user.get('is_active'):
        return jsonify({'error': 'Account is disabled'}), 401

    if not User.verify_password(user, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Update last login
    User.update_last_login(user['id'])

    # Create session
    session['user_id'] = user['id']
    session['login_time'] = datetime.utcnow().isoformat()

    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user['id'],
            'username': user['username'],
            'display_name': user['display_name'],
            'role': user['role']
        }
    })


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout endpoint."""
    session.clear()
    return jsonify({'message': 'Logout successful'})


@auth_bp.route('/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user info."""
    user = request.current_user
    return jsonify({
        'id': user['id'],
        'username': user['username'],
        'display_name': user['display_name'],
        'role': user['role'],
        'private_folder': user.get('private_folder')
    })


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({'error': 'Current and new password required'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    user = request.current_user

    # Verify current password
    full_user = User.get_by_id(user['id'])
    if not User.verify_password(full_user, current_password):
        return jsonify({'error': 'Current password is incorrect'}), 401

    # Update password
    User.change_password(user['id'], new_password)

    return jsonify({'message': 'Password changed successfully'})
