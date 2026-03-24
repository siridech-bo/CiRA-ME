"""
CiRA ME - ME-LAB Routes
AI-as-a-Service inference endpoints.
"""

import time
import uuid
import logging
import numpy as np
from functools import wraps
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..models import MeLabEndpoint, MeLabApiKey, SavedModel, get_db
from ..services.melab_service import ModelManager

logger = logging.getLogger(__name__)
melab_bp = Blueprint('melab', __name__)

# Quota limits per role
ENDPOINT_LIMITS = {'admin': 20, 'annotator': 3}
RATE_LIMITS = {'admin': 300, 'annotator': 60}  # requests per minute

# Simple rate limiter (in-memory)
_rate_buckets = {}


def api_key_required(f):
    """Decorator for API key authentication (for inference endpoints)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'X-API-Key header required'}), 401

        auth = MeLabApiKey.validate(api_key)
        if not auth:
            return jsonify({'error': 'Invalid or expired API key'}), 401

        request.melab_auth = auth
        return f(*args, **kwargs)
    return decorated


# ─── Endpoint Management (Session Auth) ──────────────────────────

@melab_bp.route('/endpoints', methods=['GET'])
@login_required
def list_endpoints():
    """List all ME-LAB endpoints for current user."""
    endpoints = MeLabEndpoint.get_all(request.current_user['id'])
    return jsonify(endpoints)


@melab_bp.route('/endpoints', methods=['POST'])
@login_required
def create_endpoint():
    """Create a new ME-LAB inference endpoint from a saved model."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    saved_model_id = data.get('saved_model_id')
    name = data.get('name', '').strip()

    if not saved_model_id:
        return jsonify({'error': 'saved_model_id required'}), 400

    # Check quota
    user = request.current_user
    role = user.get('role', 'annotator')
    limit = ENDPOINT_LIMITS.get(role, 3)
    current_count = MeLabEndpoint.count_active(user['id'])
    if current_count >= limit:
        return jsonify({
            'error': f'Endpoint limit reached ({current_count}/{limit}). '
                     f'Delete or pause existing endpoints to create new ones.'
        }), 403

    # Get saved model info
    saved = SavedModel.get_by_id(saved_model_id)
    if not saved:
        return jsonify({'error': 'Saved model not found'}), 404

    import json
    pipeline_config = saved.get('pipeline_config', {})
    if isinstance(pipeline_config, str):
        try: pipeline_config = json.loads(pipeline_config)
        except: pipeline_config = {}

    # Extract feature info
    feature_names = []
    n_features = 0
    feat_sel = pipeline_config.get('feature_selection', {})
    feat_ext = pipeline_config.get('feature_extraction', {})
    if feat_sel and feat_sel.get('selected_features'):
        feature_names = feat_sel['selected_features']
    elif feat_ext and feat_ext.get('feature_names'):
        feature_names = feat_ext['feature_names']
    n_features = len(feature_names)

    if not name:
        name = f"{saved['algorithm']} - {saved['mode']}"

    endpoint_id = uuid.uuid4().hex[:12]

    MeLabEndpoint.create(
        endpoint_id=endpoint_id,
        user_id=user['id'],
        saved_model_id=saved_model_id,
        name=name,
        mode=saved['mode'],
        algorithm=saved['algorithm'],
        feature_names=feature_names,
        n_features=n_features,
        description=data.get('description', ''),
    )

    return jsonify({
        'endpoint_id': endpoint_id,
        'name': name,
        'url': f'/api/melab/v1/{endpoint_id}/predict',
        'message': 'Endpoint created. Generate an API key to start using it.',
    }), 201


@melab_bp.route('/endpoints/<endpoint_id>', methods=['PUT'])
@login_required
def update_endpoint(endpoint_id):
    """Update an endpoint (change model, name, status)."""
    data = request.get_json()
    endpoint = MeLabEndpoint.get_by_id(endpoint_id)
    if not endpoint:
        return jsonify({'error': 'Endpoint not found'}), 404
    if endpoint['user_id'] != request.current_user['id']:
        return jsonify({'error': 'Access denied'}), 403

    if 'saved_model_id' in data:
        saved = SavedModel.get_by_id(data['saved_model_id'])
        if not saved:
            return jsonify({'error': 'Saved model not found'}), 404
        MeLabEndpoint.update_model(endpoint_id, data['saved_model_id'],
                                    saved.get('algorithm'))
        # Unload old model from cache
        old_saved = SavedModel.get_by_id(endpoint['saved_model_id'])
        if old_saved and old_saved.get('model_path'):
            ModelManager.unload_model(old_saved['model_path'])

    if 'status' in data:
        MeLabEndpoint.update_status(endpoint_id, data['status'])

    return jsonify({'message': 'Endpoint updated'})


@melab_bp.route('/endpoints/<endpoint_id>', methods=['DELETE'])
@login_required
def delete_endpoint(endpoint_id):
    """Delete an endpoint."""
    deleted = MeLabEndpoint.delete(endpoint_id, request.current_user['id'])
    if deleted:
        return jsonify({'message': 'Endpoint deleted'})
    return jsonify({'error': 'Endpoint not found or access denied'}), 404


# ─── API Key Management (Session Auth) ───────────────────────────

@melab_bp.route('/keys', methods=['GET'])
@login_required
def list_keys():
    """List API keys for current user."""
    keys = MeLabApiKey.get_all(request.current_user['id'])
    return jsonify(keys)


@melab_bp.route('/keys', methods=['POST'])
@login_required
def create_key():
    """Generate a new API key."""
    data = request.get_json() or {}
    name = data.get('name', 'default')

    result = MeLabApiKey.create(request.current_user['id'], name)

    return jsonify({
        'id': result['id'],
        'key': result['key'],  # Only shown once!
        'prefix': result['prefix'],
        'message': 'Save this key — it will not be shown again.',
    }), 201


@melab_bp.route('/keys/<int:key_id>', methods=['DELETE'])
@login_required
def revoke_key(key_id):
    """Revoke an API key."""
    revoked = MeLabApiKey.revoke(key_id, request.current_user['id'])
    if revoked:
        return jsonify({'message': 'API key revoked'})
    return jsonify({'error': 'Key not found or access denied'}), 404


# ─── Inference Endpoint (API Key Auth) ───────────────────────────

@melab_bp.route('/v1/<endpoint_id>/predict', methods=['POST'])
@api_key_required
def predict(endpoint_id):
    """Run inference on an ME-LAB endpoint.

    Request:
        X-API-Key: melab_xxxx...
        Content-Type: application/json
        {
            "data": [[0.12, 0.45, ...], [0.11, 0.44, ...]]
        }

    Response:
        {
            "endpoint_id": "abc123",
            "model_name": "RF Regressor",
            "mode": "regression",
            "predictions": [{"value": 42.7}, {"value": 43.1}],
            "latency_ms": 12,
            "timestamp": "2026-03-24T10:30:00Z"
        }
    """
    start_time = time.time()

    # Get endpoint
    endpoint = MeLabEndpoint.get_by_id(endpoint_id)
    if not endpoint:
        return jsonify({'error': 'Endpoint not found'}), 404

    if endpoint['status'] != 'active':
        return jsonify({'error': f'Endpoint is {endpoint["status"]}'}), 503

    # Check user owns endpoint
    auth = request.melab_auth
    if endpoint['user_id'] != auth['user_id']:
        return jsonify({'error': 'Access denied'}), 403

    # Parse input
    body = request.get_json()
    if not body or 'data' not in body:
        return jsonify({'error': 'Request body must contain "data" field'}), 400

    data = body['data']
    try:
        if isinstance(data[0], (list, tuple)):
            features = np.array(data, dtype=np.float64)
        else:
            features = np.array([data], dtype=np.float64)
    except (ValueError, TypeError, IndexError) as e:
        return jsonify({'error': f'Invalid data format: {e}'}), 400

    # Validate feature count
    expected = endpoint.get('n_features', 0)
    if expected > 0 and features.shape[1] != expected:
        return jsonify({
            'error': f'Expected {expected} features, got {features.shape[1]}',
            'expected_features': endpoint.get('feature_names', []),
        }), 400

    # Load model
    saved = SavedModel.get_by_id(endpoint['saved_model_id'])
    if not saved:
        return jsonify({'error': 'Model not found'}), 500

    model_path = saved.get('model_path', '')
    if not model_path:
        return jsonify({'error': 'Model file not available'}), 500

    try:
        model_data = ModelManager.load_model(model_path)
    except FileNotFoundError:
        return jsonify({'error': 'Model file not found on disk'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {e}'}), 500

    # Run inference
    try:
        predictions = ModelManager.predict(model_data, features, endpoint['mode'])
    except Exception as e:
        logger.error(f"[ME-LAB] Inference error on {endpoint_id}: {e}")
        return jsonify({'error': f'Inference failed: {e}'}), 500

    latency_ms = (time.time() - start_time) * 1000

    # Record usage
    MeLabEndpoint.record_inference(endpoint_id)
    try:
        from datetime import datetime
        with get_db() as conn:
            conn.cursor().execute(
                'INSERT INTO melab_usage_log (endpoint_id, api_key_id, request_size, latency_ms, status_code, created_at) VALUES (?,?,?,?,?,?)',
                (endpoint_id, auth['key_id'], len(features), latency_ms, 200, datetime.utcnow().isoformat())
            )
            conn.commit()
    except Exception:
        pass  # Usage logging failure should not break inference

    from datetime import datetime
    return jsonify({
        'endpoint_id': endpoint_id,
        'model_name': endpoint.get('name', ''),
        'mode': endpoint['mode'],
        'predictions': predictions,
        'count': len(predictions),
        'latency_ms': round(latency_ms, 1),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    })


@melab_bp.route('/v1/<endpoint_id>/health', methods=['GET'])
def endpoint_health(endpoint_id):
    """Check if an endpoint is ready."""
    endpoint = MeLabEndpoint.get_by_id(endpoint_id)
    if not endpoint:
        return jsonify({'status': 'not_found'}), 404

    saved = SavedModel.get_by_id(endpoint['saved_model_id'])
    import os
    model_exists = saved and saved.get('model_path') and os.path.exists(saved['model_path'])

    return jsonify({
        'endpoint_id': endpoint_id,
        'status': endpoint['status'],
        'model_ready': model_exists,
        'mode': endpoint['mode'],
        'algorithm': endpoint['algorithm'],
        'inference_count': endpoint.get('inference_count', 0),
        'last_inference': endpoint.get('last_inference_at'),
    })


@melab_bp.route('/health', methods=['GET'])
def melab_health():
    """ME-LAB service health check."""
    cache_info = ModelManager.get_cache_info()
    return jsonify({
        'status': 'healthy',
        'service': 'ME-LAB',
        'cached_models': cache_info['cached_models'],
    })


# ─── Usage (Session Auth) ────────────────────────────────────────

@melab_bp.route('/usage', methods=['GET'])
@login_required
def get_usage():
    """Get usage statistics for current user's endpoints."""
    user_id = request.current_user['id']
    endpoints = MeLabEndpoint.get_all(user_id)

    with get_db() as conn:
        cursor = conn.cursor()
        # Total inferences
        endpoint_ids = [e['id'] for e in endpoints]
        if endpoint_ids:
            placeholders = ','.join('?' * len(endpoint_ids))
            cursor.execute(f'''
                SELECT endpoint_id, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM melab_usage_log
                WHERE endpoint_id IN ({placeholders})
                GROUP BY endpoint_id
            ''', endpoint_ids)
            usage = {r['endpoint_id']: {'count': r['count'], 'avg_latency': r['avg_latency']}
                     for r in cursor.fetchall()}
        else:
            usage = {}

    return jsonify({
        'endpoints': len(endpoints),
        'usage': usage,
    })
