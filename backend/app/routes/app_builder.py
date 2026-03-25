"""
CiRA ME - App Builder Routes
Visual pipeline builder for composing ML inference apps.
"""

import io
import csv
import json
import time
import logging
import secrets
import re
import numpy as np
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..models import AppBuilderApp, MeLabEndpoint, MeLabApiKey, SavedModel, get_db
from ..services.melab_service import ModelManager

logger = logging.getLogger(__name__)
app_builder_bp = Blueprint('app_builder', __name__)


# ─── Static node catalog ─────────────────────────────────────────

NODE_CATALOG = [
    {
        'type': 'input',
        'label': 'Data Input',
        'description': 'Accepts JSON array or CSV upload',
        'category': 'io',
    },
    {
        'type': 'transform.window',
        'label': 'Windowing',
        'description': 'Sliding window over time-series data',
        'category': 'transform',
        'params': {'window_size': 100, 'step_size': 50},
    },
    {
        'type': 'transform.normalize',
        'label': 'Normalize',
        'description': 'Min-max or z-score normalization',
        'category': 'transform',
        'params': {'method': 'zscore'},
    },
    {
        'type': 'transform.feature_extract',
        'label': 'Feature Extraction',
        'description': 'Extract statistical features from windowed data',
        'category': 'transform',
        'params': {'features': ['mean', 'std', 'rms', 'max', 'min']},
    },
    {
        'type': 'output',
        'label': 'Output',
        'description': 'Format and return results',
        'category': 'io',
    },
]

# Supported statistical features
FEATURE_FUNCTIONS = {
    'mean': lambda x: np.mean(x, axis=0),
    'std': lambda x: np.std(x, axis=0),
    'var': lambda x: np.var(x, axis=0),
    'rms': lambda x: np.sqrt(np.mean(x ** 2, axis=0)),
    'max': lambda x: np.max(x, axis=0),
    'min': lambda x: np.min(x, axis=0),
    'median': lambda x: np.median(x, axis=0),
    'skew': lambda x: _safe_skew(x),
    'kurtosis': lambda x: _safe_kurtosis(x),
    'peak_to_peak': lambda x: np.ptp(x, axis=0),
}


def _safe_skew(x):
    """Compute skewness without importing scipy."""
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    s[s == 0] = 1.0
    return np.mean(((x - m) / s) ** 3, axis=0)


def _safe_kurtosis(x):
    """Compute kurtosis without importing scipy."""
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    s[s == 0] = 1.0
    return np.mean(((x - m) / s) ** 4, axis=0) - 3.0


def _slugify(text):
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = text.strip('-')
    return text or 'app'


def _auth_from_session_or_apikey():
    """Authenticate via session or API key. Returns user_id or None."""
    # Try session first
    from flask import session
    user_id = session.get('user_id')
    if user_id:
        return user_id

    # Try API key
    api_key = request.headers.get('X-API-Key')
    if api_key:
        auth = MeLabApiKey.validate(api_key)
        if auth:
            return auth['user_id']

    return None


# ─── CRUD Routes (Session Auth) ──────────────────────────────────

@app_builder_bp.route('/apps', methods=['GET'])
@login_required
def list_apps():
    """List all apps for current user."""
    apps = AppBuilderApp.get_all(request.current_user['id'])
    result = []
    for app in apps:
        # Derive mode from nodes
        mode = None
        nodes = app.get('nodes', [])
        for node in nodes:
            ntype = node.get('type', '') if isinstance(node, dict) else ''
            if ntype.startswith('model.endpoint.'):
                endpoint_id = ntype.replace('model.endpoint.', '')
                ep = MeLabEndpoint.get_by_id(endpoint_id)
                if ep:
                    mode = ep.get('mode')
                break

        result.append({
            'id': app['id'],
            'name': app['name'],
            'status': app['status'],
            'mode': mode,
            'created_at': app['created_at'],
            'updated_at': app.get('updated_at'),
            'calls': app['calls'],
            'slug': app.get('slug'),
            'access': app.get('access', 'private'),
        })
    return jsonify(result)


@app_builder_bp.route('/apps', methods=['POST'])
@login_required
def create_app():
    """Create a new app."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    name = data.get('name', 'Untitled App').strip()
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    access = data.get('access', 'private')

    app_id = AppBuilderApp.create(
        user_id=request.current_user['id'],
        name=name,
        nodes=nodes,
        edges=edges,
        access=access,
    )

    app = AppBuilderApp.get_by_id(app_id)
    return jsonify(app), 201


@app_builder_bp.route('/apps/<int:app_id>', methods=['GET'])
@login_required
def get_app(app_id):
    """Get full app schema."""
    app = AppBuilderApp.get_by_id(app_id)
    if not app:
        return jsonify({'error': 'App not found'}), 404
    if app['user_id'] != request.current_user['id']:
        return jsonify({'error': 'Access denied'}), 403
    return jsonify(app)


@app_builder_bp.route('/apps/<int:app_id>', methods=['PUT'])
@login_required
def update_app(app_id):
    """Update an app."""
    app = AppBuilderApp.get_by_id(app_id)
    if not app:
        return jsonify({'error': 'App not found'}), 404
    if app['user_id'] != request.current_user['id']:
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    allowed = {}
    if 'name' in data:
        allowed['name'] = data['name'].strip()
    if 'nodes' in data:
        allowed['nodes'] = data['nodes']
    if 'edges' in data:
        allowed['edges'] = data['edges']
    if 'access' in data:
        allowed['access'] = data['access']

    AppBuilderApp.update(app_id, **allowed)

    updated = AppBuilderApp.get_by_id(app_id)
    return jsonify(updated)


@app_builder_bp.route('/apps/<int:app_id>', methods=['DELETE'])
@login_required
def delete_app(app_id):
    """Delete an app."""
    deleted = AppBuilderApp.delete(app_id, request.current_user['id'])
    if deleted:
        return jsonify({'message': 'App deleted'})
    return jsonify({'error': 'App not found or access denied'}), 404


# ─── Publish ─────────────────────────────────────────────────────

@app_builder_bp.route('/apps/<int:app_id>/publish', methods=['POST'])
@login_required
def publish_app(app_id):
    """Validate pipeline and publish the app."""
    app = AppBuilderApp.get_by_id(app_id)
    if not app:
        return jsonify({'error': 'App not found'}), 404
    if app['user_id'] != request.current_user['id']:
        return jsonify({'error': 'Access denied'}), 403

    nodes = app.get('nodes', [])

    # Validate: must have at least an input and output node
    node_types = [n.get('type', '') for n in nodes if isinstance(n, dict)]
    if 'input' not in node_types:
        return jsonify({'error': 'Pipeline must have an input node'}), 400
    if 'output' not in node_types:
        return jsonify({'error': 'Pipeline must have an output node'}), 400

    # Validate: model endpoint nodes must reference active endpoints
    for nt in node_types:
        if nt.startswith('model.endpoint.'):
            endpoint_id = nt.replace('model.endpoint.', '')
            ep = MeLabEndpoint.get_by_id(endpoint_id)
            if not ep:
                return jsonify({'error': f'Endpoint {endpoint_id} not found'}), 400
            if ep['status'] != 'active':
                return jsonify({'error': f'Endpoint {endpoint_id} is not active'}), 400

    # Generate slug
    base_slug = _slugify(app['name'])
    slug = f"{base_slug}-{secrets.token_hex(3)}"

    # Ensure slug uniqueness
    while AppBuilderApp.get_by_slug(slug):
        slug = f"{base_slug}-{secrets.token_hex(3)}"

    AppBuilderApp.publish(app_id, slug)

    return jsonify({
        'slug': slug,
        'web_url': f'/app/{slug}',
        'api_endpoint': f'/api/app-builder/run/{slug}',
        'status': 'published',
    })


# ─── Runtime (API Key or Session Auth) ───────────────────────────

@app_builder_bp.route('/run/<slug>', methods=['POST'])
def run_app(slug):
    """Execute a published app pipeline."""
    # Authenticate via session or API key
    user_id = _auth_from_session_or_apikey()
    if not user_id:
        return jsonify({'error': 'Authentication required (session or X-API-Key)'}), 401

    # Load app
    app = AppBuilderApp.get_by_slug(slug)
    if not app:
        return jsonify({'error': 'App not found'}), 404

    if app['status'] != 'published':
        return jsonify({'error': 'App is not published'}), 400

    # Access check: private apps only accessible by owner
    if app['access'] == 'private' and app['user_id'] != user_id:
        return jsonify({'error': 'Access denied'}), 403

    start_time = time.time()

    nodes = app.get('nodes', [])
    edges = app.get('edges', [])

    # Sort nodes by pipeline order using edges (topological-ish: follow edge order)
    ordered_nodes = _order_nodes(nodes, edges)

    # Parse input data from request
    input_data = _parse_input(request)
    if isinstance(input_data, tuple):
        # Error response
        return input_data

    # Execute pipeline
    try:
        current_data = input_data
        for node in ordered_nodes:
            ntype = node.get('type', '')
            params = node.get('params', node.get('data', {}).get('params', {})) or {}

            if ntype == 'input':
                # Input node: data already parsed
                continue

            elif ntype == 'transform.window':
                current_data = _apply_windowing(current_data, params)

            elif ntype == 'transform.normalize':
                current_data = _apply_normalization(current_data, params)

            elif ntype == 'transform.feature_extract':
                current_data = _apply_feature_extraction(current_data, params)

            elif ntype.startswith('model.endpoint.'):
                endpoint_id = ntype.replace('model.endpoint.', '')
                current_data = _run_model_inference(endpoint_id, current_data)

            elif ntype == 'output':
                # Output node: finalize
                continue

    except Exception as e:
        logger.error(f"[AppBuilder] Pipeline error in app {slug}: {e}")
        return jsonify({'error': f'Pipeline execution failed: {str(e)}'}), 500

    # Increment calls
    AppBuilderApp.increment_calls(app['id'])

    latency_ms = (time.time() - start_time) * 1000
    return jsonify({
        'app': app['name'],
        'slug': slug,
        'result': current_data if not isinstance(current_data, np.ndarray) else current_data.tolist(),
        'latency_ms': round(latency_ms, 1),
    })


# ─── Capabilities ────────────────────────────────────────────────

@app_builder_bp.route('/capabilities', methods=['GET'])
@login_required
def get_capabilities():
    """Return node catalog merged with active ME-LAB endpoints."""
    catalog = list(NODE_CATALOG)

    # Add active ME-LAB endpoints as model nodes
    endpoints = MeLabEndpoint.get_all(request.current_user['id'])
    for ep in endpoints:
        if ep.get('status') == 'active':
            catalog.append({
                'type': f"model.endpoint.{ep['id']}",
                'label': ep.get('name', ep['algorithm']),
                'description': ep.get('description', f"{ep['algorithm']} ({ep['mode']})"),
                'category': 'model',
                'mode': ep['mode'],
                'algorithm': ep['algorithm'],
                'endpoint_id': ep['id'],
                'n_features': ep.get('n_features', 0),
                'feature_names': ep.get('feature_names', []),
            })

    return jsonify(catalog)


# ─── Pipeline Helpers ─────────────────────────────────────────────

def _order_nodes(nodes, edges):
    """Order nodes topologically using edges."""
    if not edges:
        return nodes

    # Build adjacency and in-degree
    node_map = {}
    for n in nodes:
        nid = n.get('id')
        if nid is not None:
            node_map[nid] = n

    if not node_map:
        return nodes

    in_degree = {nid: 0 for nid in node_map}
    adj = {nid: [] for nid in node_map}

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in node_map and tgt in node_map:
            adj[src].append(tgt)
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Kahn's algorithm
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    ordered = []
    while queue:
        nid = queue.pop(0)
        ordered.append(node_map[nid])
        for neighbor in adj.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Append any remaining nodes not in edges
    ordered_ids = {n.get('id') for n in ordered}
    for n in nodes:
        if n.get('id') not in ordered_ids:
            ordered.append(n)

    return ordered


def _parse_input(req):
    """Parse input data from request body (JSON or CSV file)."""
    # Check for file upload
    if req.files and 'file' in req.files:
        file = req.files['file']
        try:
            text = file.read().decode('utf-8')
            reader = csv.reader(io.StringIO(text))
            rows = []
            for row in reader:
                try:
                    rows.append([float(v) for v in row])
                except ValueError:
                    continue  # Skip header or non-numeric rows
            if not rows:
                return jsonify({'error': 'CSV file contains no numeric data'}), 400
            return np.array(rows, dtype=np.float64)
        except Exception as e:
            return jsonify({'error': f'Failed to parse CSV: {e}'}), 400

    # JSON body
    body = req.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request must contain JSON body with "data" or a CSV file upload'}), 400

    data = body.get('data')
    if data is None:
        return jsonify({'error': 'JSON body must contain "data" field'}), 400

    try:
        arr = np.array(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid data format: {e}'}), 400


def _apply_windowing(data, params):
    """Apply sliding window to data."""
    window_size = params.get('window_size', 100)
    step_size = params.get('step_size', 50)

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples = data.shape[0]
    if n_samples < window_size:
        # Return as single window
        return data.reshape(1, *data.shape)

    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        windows.append(data[start:start + window_size])

    return np.array(windows)


def _apply_normalization(data, params):
    """Apply normalization to data."""
    method = params.get('method', 'zscore')

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    if method == 'minmax':
        d_min = data.min(axis=0) if data.ndim <= 2 else data.min(axis=-2)
        d_max = data.max(axis=0) if data.ndim <= 2 else data.max(axis=-2)
        denom = d_max - d_min
        denom[denom == 0] = 1.0
        if data.ndim <= 2:
            return (data - d_min) / denom
        else:
            return (data - d_min[:, np.newaxis, :]) / denom[:, np.newaxis, :]
    else:  # zscore
        mean = data.mean(axis=0) if data.ndim <= 2 else data.mean(axis=-2)
        std = data.std(axis=0) if data.ndim <= 2 else data.std(axis=-2)
        std[std == 0] = 1.0
        if data.ndim <= 2:
            return (data - mean) / std
        else:
            return (data - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]


def _apply_feature_extraction(data, params):
    """Extract statistical features from windowed data."""
    feature_names = params.get('features', ['mean', 'std', 'rms', 'max', 'min'])

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    # If data is 3D (windows x window_size x channels), extract per window
    if data.ndim == 3:
        all_features = []
        for window in data:
            feats = []
            for fname in feature_names:
                fn = FEATURE_FUNCTIONS.get(fname)
                if fn:
                    result = fn(window)
                    if np.isscalar(result):
                        feats.append(result)
                    else:
                        feats.extend(result.tolist())
            all_features.append(feats)
        return np.array(all_features, dtype=np.float64)

    # If 2D, treat as single window
    if data.ndim == 2:
        feats = []
        for fname in feature_names:
            fn = FEATURE_FUNCTIONS.get(fname)
            if fn:
                result = fn(data)
                if np.isscalar(result):
                    feats.append(result)
                else:
                    feats.extend(result.tolist())
        return np.array([feats], dtype=np.float64)

    return data


def _run_model_inference(endpoint_id, data):
    """Run ME-LAB model inference on data."""
    endpoint = MeLabEndpoint.get_by_id(endpoint_id)
    if not endpoint:
        raise ValueError(f"Endpoint {endpoint_id} not found")
    if endpoint['status'] != 'active':
        raise ValueError(f"Endpoint {endpoint_id} is not active")

    saved = SavedModel.get_by_id(endpoint['saved_model_id'])
    if not saved or not saved.get('model_path'):
        raise ValueError(f"Model for endpoint {endpoint_id} not available")

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    model_data = ModelManager.load_model(saved['model_path'])
    predictions = ModelManager.predict(model_data, data, endpoint['mode'])

    # Record the inference
    MeLabEndpoint.record_inference(endpoint_id)

    return predictions
