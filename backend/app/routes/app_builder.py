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


@app_builder_bp.route('/apps/by-slug/<slug>', methods=['GET'])
def get_app_by_slug(slug):
    """Get published app by slug (no auth for public apps)."""
    app = AppBuilderApp.get_by_slug(slug)
    if not app:
        return jsonify({'error': 'App not found'}), 404
    if app['status'] != 'published':
        return jsonify({'error': 'App is not published'}), 404
    # Derive mode from model nodes
    import json as json_mod
    nodes = app.get('nodes', [])
    if isinstance(nodes, str):
        nodes = json_mod.loads(nodes)
    mode = None
    algorithm = None
    sensor_columns = []
    for node in nodes:
        if node.get('type', '').startswith('model.endpoint.'):
            endpoint_id = node['type'].split('.')[-1]
            endpoint = MeLabEndpoint.get_by_id(endpoint_id)
            if endpoint:
                mode = endpoint.get('mode')
                algorithm = endpoint.get('algorithm')
                # Get sensor columns from model's pipeline_config
                saved = SavedModel.get_by_id(endpoint.get('saved_model_id'))
                if saved:
                    pc = saved.get('pipeline_config', {})
                    if isinstance(pc, str):
                        pc = json_mod.loads(pc) if pc else {}
                    sensor_columns = pc.get('normalization', {}).get('sensor_columns', [])
            break
    result = dict(app)
    result['mode'] = mode
    result['algorithm'] = algorithm
    result['sensor_columns'] = sensor_columns
    return jsonify(result)


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
    if not any(nt.startswith('input.') for nt in node_types):
        return jsonify({'error': 'Pipeline must have an input node'}), 400
    if not any(nt.startswith('output.') for nt in node_types):
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
        column_names = None  # Track column names through pipeline
        model_mode = None
        actual_values = None  # Ground truth for comparison

        # If input is a DataFrame, remember column names and extract target
        import pandas as pd
        if isinstance(current_data, pd.DataFrame):
            column_names = list(current_data.columns)
            logger.info(f"[AppBuilder] Input: DataFrame {current_data.shape}, columns={column_names}")

            # Try to get target column from output node config
            target_col = None
            for node in ordered_nodes:
                if node.get('type', '').startswith('output.'):
                    target_col = node.get('config', {}).get('target_column')
                    if target_col:
                        break
            # Also check model's pipeline_config for target_column
            if not target_col:
                for node in ordered_nodes:
                    if node.get('type', '').startswith('model.endpoint.'):
                        eid = node['type'].replace('model.endpoint.', '')
                        ep = MeLabEndpoint.get_by_id(eid)
                        if ep:
                            saved = SavedModel.get_by_id(ep.get('saved_model_id'))
                            if saved:
                                pc = saved.get('pipeline_config', {})
                                if isinstance(pc, str):
                                    pc = json.loads(pc) if pc else {}
                                target_col = pc.get('target_column')
                        break

            if target_col and target_col in current_data.columns:
                raw_target = current_data[target_col].values
                logger.info(f"[AppBuilder] Target column '{target_col}': {len(raw_target)} values")
            else:
                raw_target = None

        for node in ordered_nodes:
            ntype = node.get('type', '')
            params = node.get('config', {}) or {}

            if ntype.startswith('input.'):
                continue

            elif ntype == 'transform.window':
                if isinstance(current_data, pd.DataFrame):
                    current_data, column_names = _extract_sensor_data(current_data, ordered_nodes)

                # Compute per-window actual values from raw_target (saved before normalize)
                if raw_target is not None and actual_values is None:
                    ws = params.get('window_size', 32)
                    st = params.get('step', params.get('stride', 16))
                    actual_values = []
                    for start in range(0, len(raw_target) - ws + 1, st):
                        actual_values.append(float(np.mean(raw_target[start:start + ws])))

                current_data = _apply_windowing(current_data, params)

            elif ntype == 'transform.normalize':
                # Convert DataFrame to numpy, keeping only sensor columns
                if isinstance(current_data, pd.DataFrame):
                    current_data, column_names = _extract_sensor_data(current_data, ordered_nodes)

                # Try to get normalization params from the model's pipeline_config
                norm_params = dict(params)
                for n2 in ordered_nodes:
                    if n2.get('type', '').startswith('model.endpoint.'):
                        eid = n2['type'].replace('model.endpoint.', '')
                        ep = MeLabEndpoint.get_by_id(eid)
                        if ep:
                            saved = SavedModel.get_by_id(ep.get('saved_model_id'))
                            if saved:
                                pc = saved.get('pipeline_config', {})
                                if isinstance(pc, str):
                                    pc = json.loads(pc) if pc else {}
                                model_norm = pc.get('normalization', {})
                                if model_norm and model_norm.get('channel_min'):
                                    norm_params['_model_norm'] = model_norm
                                    norm_params['_sensor_columns'] = column_names
                        break
                current_data = _apply_normalization(current_data, norm_params)

            elif ntype == 'transform.fill_missing':
                current_data = _apply_fill_missing(current_data, params)

            elif ntype == 'transform.feature_extract':
                if column_names:
                    params = dict(params)
                    params['_column_names'] = column_names
                current_data = _apply_feature_extraction(current_data, params)
                logger.info(f"[AppBuilder] After feature_extract: shape={current_data.shape}")

            elif ntype.startswith('model.endpoint.'):
                endpoint_id = ntype.replace('model.endpoint.', '')
                # Get mode for response formatting
                ep = MeLabEndpoint.get_by_id(endpoint_id)
                if ep:
                    model_mode = ep.get('mode')
                current_data = _run_model_inference(endpoint_id, current_data)
                logger.info(f"[AppBuilder] After model inference: {len(current_data)} predictions")

            elif ntype.startswith('output.'):
                continue

    except Exception as e:
        import traceback
        logger.error(f"[AppBuilder] Pipeline error in app {slug}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Pipeline execution failed: {str(e)}'}), 500

    # Increment calls
    AppBuilderApp.increment_calls(app['id'])

    latency_ms = (time.time() - start_time) * 1000

    # Format response based on model predictions
    predictions = []
    if isinstance(current_data, list):
        predictions = current_data  # List[Dict] from ModelManager.predict()
    elif isinstance(current_data, np.ndarray):
        predictions = current_data.tolist()

    # Extract prediction values for the frontend
    pred_values = []
    if predictions and isinstance(predictions[0], dict):
        for p in predictions:
            if 'value' in p:
                pred_values.append(p['value'])
            elif 'label' in p:
                pred_values.append(p['label'])
            else:
                pred_values.append(p)
    else:
        pred_values = predictions

    # Compute summary stats for regression
    response = {
        'app': app['name'],
        'slug': slug,
        'mode': model_mode,
        'predictions': pred_values,
        'actual': actual_values if actual_values else None,
        'count': len(pred_values),
        'latency_ms': round(latency_ms, 1),
    }

    if model_mode == 'regression' and pred_values:
        vals = [v for v in pred_values if isinstance(v, (int, float))]
        if vals:
            response['mean'] = float(np.mean(vals))
            response['std'] = float(np.std(vals))
            response['min'] = float(np.min(vals))
            response['max'] = float(np.max(vals))
            # Compute R² if actual values available
            if actual_values and len(actual_values) == len(vals):
                actuals = np.array(actual_values)
                preds = np.array(vals)
                ss_res = np.sum((actuals - preds) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                response['r2'] = float(r2)
                response['rmse'] = float(np.sqrt(np.mean((actuals - preds) ** 2)))
                response['mae'] = float(np.mean(np.abs(actuals - preds)))

    return jsonify(response)


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
            # Get target_column from saved model's pipeline_config
            target_col = None
            try:
                saved = SavedModel.get_by_id(ep['saved_model_id'])
                if saved:
                    pc = saved.get('pipeline_config', {})
                    if isinstance(pc, str):
                        pc = json.loads(pc)
                    target_col = pc.get('target_column')
            except Exception:
                pass

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
                'target_column': target_col,
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
    """Parse input data from request body (JSON or CSV file).
    Returns a pandas DataFrame to preserve column names.
    """
    import pandas as pd

    # Check for file upload
    if req.files and 'file' in req.files:
        file = req.files['file']
        try:
            text = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(text))
            # Drop non-numeric columns except timestamp
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return jsonify({'error': 'CSV file contains no numeric data'}), 400
            # Keep timestamp if it exists, plus all numeric columns
            keep_cols = []
            for col in df.columns:
                if col in numeric_cols:
                    keep_cols.append(col)
            df = df[keep_cols].dropna()
            return df
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
    step_size = params.get('step', params.get('step_size', 50))

    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).values

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


def _extract_sensor_data(df, ordered_nodes):
    """Extract sensor-only numeric data from DataFrame.
    Drops timestamp, label, category, and any columns not in the model's sensor_columns.
    Returns (numpy_array, column_names_list).
    """
    import pandas as pd
    sensor_df = df.select_dtypes(include=[np.number])

    # Find model's sensor columns from pipeline_config
    model_sensors = None
    for n in ordered_nodes:
        if n.get('type', '').startswith('model.endpoint.'):
            eid = n['type'].replace('model.endpoint.', '')
            ep = MeLabEndpoint.get_by_id(eid)
            if ep:
                saved = SavedModel.get_by_id(ep.get('saved_model_id'))
                if saved:
                    pc = saved.get('pipeline_config', {})
                    if isinstance(pc, str):
                        pc = json.loads(pc) if pc else {}
                    model_sensors = pc.get('normalization', {}).get('sensor_columns', [])
            break

    if model_sensors:
        # Keep only model's sensor columns
        keep = [c for c in model_sensors if c in sensor_df.columns]
        if keep:
            sensor_df = sensor_df[keep]
    else:
        # Fallback: drop known non-sensor columns
        non_sensor = ('timestamp', 'time', 'time_sec', 'index',
                      'label', 'class', 'target', 'category', 'sample_id')
        drop = [c for c in sensor_df.columns if c.lower() in non_sensor]
        if drop:
            sensor_df = sensor_df.drop(columns=drop)

    return sensor_df.values.astype(np.float64), list(sensor_df.columns)


def _apply_fill_missing(data, params):
    """Fill missing values in data."""
    method = params.get('method', 'interpolate')
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    import pandas as pd
    if data.ndim == 2:
        df = pd.DataFrame(data)
        if method == 'ffill':
            df = df.ffill().bfill()
        elif method == 'bfill':
            df = df.bfill().ffill()
        elif method == 'zero':
            df = df.fillna(0)
        else:  # interpolate
            df = df.interpolate().ffill().bfill()
        return df.values.astype(np.float64)
    return np.nan_to_num(data, nan=0.0)


def _apply_normalization(data, params):
    """Apply normalization to data.
    If _model_norm is provided, use the training normalization params
    (channel_min/channel_max) for consistency with how the model was trained.
    """
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).values

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    # Use model's training normalization if available
    model_norm = params.get('_model_norm')
    if model_norm:
        channel_min = np.array(model_norm.get('channel_min', []), dtype=np.float64)
        channel_max = np.array(model_norm.get('channel_max', []), dtype=np.float64)
        sensor_cols = model_norm.get('sensor_columns', [])
        input_cols = params.get('_sensor_columns', [])

        if len(channel_min) > 0 and len(channel_max) > 0:
            # Map model's normalization to input data columns
            # The model's sensor_columns may be a subset of input columns
            n_cols = data.shape[-1] if data.ndim >= 2 else 1
            d_min = np.zeros(n_cols)
            d_max = np.ones(n_cols)

            for i, col in enumerate(input_cols):
                if col in sensor_cols:
                    idx = sensor_cols.index(col)
                    if idx < len(channel_min):
                        d_min[i] = channel_min[idx]
                        d_max[i] = channel_max[idx]

            denom = d_max - d_min
            denom[denom == 0] = 1.0
            logger.info(f"[AppBuilder] Applying model normalization: min={d_min[:3]}..., max={d_max[:3]}...")
            return (data - d_min) / denom

    # Fallback: compute normalization from data
    method = params.get('method', 'zscore')
    if method == 'minmax':
        d_min = data.min(axis=0)
        d_max = data.max(axis=0)
        denom = d_max - d_min
        denom[denom == 0] = 1.0
        return (data - d_min) / denom
    else:  # zscore
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        return (data - mean) / std


def _apply_feature_extraction(data, params):
    """Extract features from windowed data using CiRA ME's feature extractor."""
    feature_names = params.get('features', ['mean', 'std', 'rms', 'max', 'min'])

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    # Use column names if available (passed from pipeline context)
    column_names = params.get('_column_names')
    if not column_names:
        column_names = _derive_sensor_columns(feature_names)

    # Try using CiRA ME's feature extractor for DSP features
    try:
        from ..services.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()

        if data.ndim == 3 and column_names:
            n_channels = data.shape[2]
            sensor_cols = [c for c in column_names if c not in ('timestamp', 'label', 'class', 'target')]
            if len(sensor_cols) > n_channels:
                sensor_cols = sensor_cols[:n_channels]

            if len(sensor_cols) == n_channels:
                # Use extract_from_windows_direct — produces ALL DSP features
                result_df = extractor.extract_from_windows_direct(data, sensor_cols)
                logger.info(f"[AppBuilder] Extracted {result_df.shape[1]} features from {data.shape[0]} windows")

                # Select only the features the model needs (in order)
                selected = []
                for fname in feature_names:
                    if fname in result_df.columns:
                        selected.append(result_df[fname].values)
                    else:
                        logger.warning(f"[AppBuilder] Feature '{fname}' not found in extracted features")
                        selected.append(np.zeros(len(result_df)))

                return np.column_stack(selected) if selected else result_df.values
    except Exception as e:
        logger.warning(f"CiRA ME feature extractor failed: {e}, falling back to generic")

    # Fallback: generic statistical features
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
                else:
                    feats.append(0.0)  # unknown feature
            all_features.append(feats)
        return np.array(all_features, dtype=np.float64)

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
            else:
                feats.append(0.0)
        return np.array([feats], dtype=np.float64)

    return data


def _derive_sensor_columns(feature_names):
    """Derive sensor column names from CiRA ME feature names.
    e.g., 'abs_energy_RPM' -> column 'RPM', 'spectral_bandwidth_Vibration_Y' -> column 'Vibration_Y'
    """
    # Known CiRA ME feature prefixes
    known_prefixes = [
        'abs_energy', 'abs_sum_of_changes', 'spectral_bandwidth',
        'margin_factor', 'peak_to_peak', 'rms', 'mean', 'std',
        'max', 'min', 'median', 'variance', 'kurtosis', 'skewness',
        'crest_factor', 'shape_factor', 'impulse_factor',
        'spectral_centroid', 'spectral_rolloff', 'spectral_flatness',
        'zero_crossing_rate', 'band_power',
    ]
    columns = set()
    for fname in feature_names:
        for prefix in sorted(known_prefixes, key=len, reverse=True):
            if fname.startswith(prefix + '_'):
                col = fname[len(prefix) + 1:]
                columns.add(col)
                break
    return sorted(columns) if columns else None


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

    # Decode integer labels for classification/anomaly
    if endpoint['mode'] != 'regression' and predictions:
        model_obj = model_data.get('model')

        # Build prediction decoder: index → class name
        decode_map = None

        # Priority 1: label_inverse_map (exact encoder mapping from training)
        inv_map = model_data.get('label_inverse_map')
        if inv_map:
            decode_map = {int(k): str(v) for k, v in inv_map.items()}
        # Priority 2: model.classes_ with string values (sklearn native)
        elif hasattr(model_obj, 'classes_') and len(model_obj.classes_) > 0:
            if isinstance(model_obj.classes_[0], (str, np.str_)):
                decode_map = None  # Predictions are already strings
        # Priority 3: class_names (may be wrong order for old models, but best we have)
        if not decode_map and not inv_map and model_data.get('class_names'):
            cn = model_data['class_names']
            decode_map = {i: cn[i] for i in range(len(cn))}

        if decode_map:
            for p in predictions:
                label = p.get('label', '')
                try:
                    idx = int(float(label))
                    if idx in decode_map:
                        p['label'] = decode_map[idx]
                        p['prediction'] = decode_map[idx]
                except (ValueError, TypeError):
                    pass

    # Record the inference
    MeLabEndpoint.record_inference(endpoint_id)

    return predictions
