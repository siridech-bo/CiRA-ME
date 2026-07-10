"""
CiRA ME - App Builder Routes
Visual pipeline builder for composing ML inference apps.
"""

import io
import os
import csv
import json
import time
import logging
import secrets
import re
import threading
from datetime import datetime
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from ..auth import login_required
from ..models import AppBuilderApp, MeLabEndpoint, MeLabApiKey, SavedModel, get_db
from ..services.melab_service import ModelManager

logger = logging.getLogger(__name__)
app_builder_bp = Blueprint('app_builder', __name__)

# ─── Prediction sink helpers ─────────────────────────────────────
# One lock per CSV file path so concurrent writes within the same day
# are serialized (append-safe). Locks are keyed by the resolved absolute
# path so different apps / days don't block each other.
_prediction_sink_locks = {}
_prediction_sink_locks_guard = threading.Lock()


def _get_prediction_sink_lock(path):
    with _prediction_sink_locks_guard:
        lock = _prediction_sink_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _prediction_sink_locks[path] = lock
        return lock


def _persist_prediction(app, response, request_json):
    """Persist a prediction to CSV and/or MQTT if the app has an
    output.prediction_sink node. Never raises — a sink failure must
    not break the inference response."""
    try:
        nodes = app.get('nodes', []) or []
        sink = next((n for n in nodes if n.get('type') == 'output.prediction_sink'), None)
        if not sink:
            return

        config = sink.get('config', {}) or {}
        mode = str(config.get('mode', 'csv')).lower()
        if mode not in ('csv', 'mqtt', 'both'):
            mode = 'csv'
        mqtt_topic_tpl = config.get('mqtt_topic') or 'predictions/{slug}'

        slug = app.get('slug') or ''
        request_json = request_json or {}

        # Extract channel names + last-row values from incoming JSON payload.
        channels_meta = request_json.get('channels') or []
        data_rows = request_json.get('data') or []
        channel_values = {}
        if channels_meta and data_rows:
            try:
                last_row = data_rows[-1]
                if isinstance(last_row, dict):
                    for ch in channels_meta:
                        if ch in last_row:
                            channel_values[str(ch)] = last_row[ch]
                elif isinstance(last_row, (list, tuple)):
                    for i, ch in enumerate(channels_meta):
                        if i < len(last_row):
                            channel_values[str(ch)] = last_row[i]
            except Exception:
                channel_values = {}

        timestamp = datetime.utcnow().isoformat() + 'Z'
        is_multi = bool(response.get('multi_model'))

        payload = {
            'timestamp': timestamp,
            'app_slug': slug,
        }
        if channel_values:
            payload['channels'] = channel_values

        # Build per-mode payload details
        if is_multi:
            models_map = {}
            for eid, mentry in (response.get('models') or {}).items():
                preds = mentry.get('predictions') or []
                if preds:
                    models_map[mentry.get('name', eid)] = preds[-1]
            payload['models'] = models_map
            actual_list = response.get('actual')
            if actual_list:
                try:
                    payload['actual'] = actual_list[-1]
                except Exception:
                    pass
        else:
            preds = response.get('predictions') or []
            if preds:
                payload['prediction'] = preds[-1]
            preds_full = response.get('predictions_full') or []
            if preds_full and isinstance(preds_full[-1], dict):
                last = preds_full[-1]
                if 'confidence' in last:
                    payload['confidence'] = last['confidence']
                if 'anomaly_score' in last:
                    payload['anomaly_score'] = last['anomaly_score']

        # ── CSV persistence ─────────────────────────────────────
        if mode in ('csv', 'both'):
            try:
                _write_prediction_csv(slug, payload, is_multi)
            except Exception as e:
                logger.warning(f"[PredictionSink] CSV write failed for slug={slug}: {e}")

        # ── MQTT publish ────────────────────────────────────────
        if mode in ('mqtt', 'both'):
            try:
                topic = str(mqtt_topic_tpl).replace('{slug}', slug)
                _publish_prediction_mqtt(topic, payload)
            except Exception as e:
                logger.warning(f"[PredictionSink] MQTT publish failed for slug={slug}: {e}")

    except Exception as e:
        # Blanket guard — never let sink errors propagate.
        logger.warning(f"[PredictionSink] unexpected error: {e}")


def _write_prediction_csv(slug, payload, is_multi):
    """Append a single row to shared/predictions/<slug>/<YYYY-MM-DD>.csv.
    Writes a header when the file is first created."""
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared_folder = current_app.config['SHARED_FOLDER_PATH']
    day = datetime.utcnow().strftime('%Y-%m-%d')
    dir_path = os.path.join(datasets_root, shared_folder, 'predictions', slug)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f'{day}.csv')

    # Flatten payload into a single row.
    channels = payload.get('channels') or {}
    row = {
        'timestamp': payload.get('timestamp', ''),
        'app_slug': payload.get('app_slug', ''),
    }
    for ch, val in channels.items():
        row[f'ch_{ch}'] = val

    if is_multi:
        for model_name, pred_val in (payload.get('models') or {}).items():
            row[f'model_{model_name}'] = pred_val
        if 'actual' in payload:
            row['actual'] = payload['actual']
    else:
        if 'prediction' in payload:
            row['prediction'] = payload['prediction']
        if 'confidence' in payload:
            row['confidence'] = payload['confidence']
        if 'anomaly_score' in payload:
            row['anomaly_score'] = payload['anomaly_score']

    lock = _get_prediction_sink_lock(file_path)
    with lock:
        file_exists = os.path.exists(file_path)
        # First-write locks the schema for the day (header written once).
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or os.path.getsize(file_path) == 0:
                writer.writeheader()
            writer.writerow(row)


def _publish_prediction_mqtt(topic, payload):
    """Publish payload JSON to the internal Mosquitto broker.
    Uses a short-lived client to keep this fire-and-forget."""
    import paho.mqtt.client as paho_mqtt

    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

    client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker_host, broker_port, keepalive=5)
    try:
        client.loop_start()
        client.publish(topic, json.dumps(payload, default=str), qos=0)
    finally:
        try:
            client.loop_stop()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass


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
        # Derive mode from nodes (single model or multi-model)
        mode = _derive_app_mode(app.get('nodes', []))

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

    # Check app quota
    user = request.current_user
    max_apps = user.get('max_apps') or 10
    current_apps = len(AppBuilderApp.get_all(user['id']))
    if current_apps >= max_apps:
        return jsonify({
            'error': f'App limit reached ({current_apps}/{max_apps}). '
                     f'Delete unused apps or contact admin to increase limit.'
        }), 403

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
    import json as json_mod
    nodes = app.get('nodes', [])
    if isinstance(nodes, str):
        nodes = json_mod.loads(nodes)

    mode, algorithm, sensor_columns = _derive_app_details(nodes)
    is_multi = any(n.get('type') == 'output.multi_model_compare' for n in nodes)

    result = dict(app)
    result['mode'] = mode
    result['algorithm'] = 'multi-model' if is_multi else algorithm
    result['sensor_columns'] = sensor_columns
    result['multi_model'] = is_multi
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

    # F4: link app to project via any endpoint referenced by its nodes
    try:
        from ..models import get_db as _get_db, Project as _Project
        _pid_found = None
        for nt in node_types:
            if nt.startswith('model.endpoint.'):
                eid = nt.replace('model.endpoint.', '')
                ep = MeLabEndpoint.get_by_id(eid)
                if ep and ep.get('project_id'):
                    _pid_found = ep['project_id']
                    break
        if _pid_found:
            with _get_db() as _c:
                _cur = _c.cursor()
                _cur.execute(
                    'UPDATE app_builder_apps SET project_id = ? WHERE id = ?',
                    (_pid_found, app_id)
                )
                _c.commit()
            _Project.touch(_pid_found, 'deploy')
    except Exception:
        pass

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
    # Load app first to check access policy
    app = AppBuilderApp.get_by_slug(slug)
    if not app:
        return jsonify({'error': 'App not found'}), 404

    if app['status'] != 'published':
        return jsonify({'error': 'App is not published'}), 400

    # Authenticate based on access policy
    access = app.get('access', 'private')
    user_id = _auth_from_session_or_apikey()

    if access == 'public':
        # No auth needed — anyone can use
        pass
    elif access == 'team':
        # Requires login (session auth)
        if not user_id:
            return jsonify({'error': 'Login required to use this app'}), 401
    else:  # private
        if not user_id:
            return jsonify({'error': 'Authentication required (session or X-API-Key)'}), 401
        if app['user_id'] != user_id:
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

    # Check for live target values and channels (from MQTT with ground truth column)
    live_target_values = None
    request_body = request.get_json(silent=True)
    if request_body and request_body.get('target_values'):
        live_target_values = request_body['target_values']

    # Execute pipeline
    try:
        current_data = input_data
        column_names = None  # Track column names through pipeline
        model_mode = None
        actual_values = None  # Ground truth for comparison
        dataset_labels = None  # For integer label decoding
        is_raw_mode_model = False  # Whether model was trained without windowing
        is_dl_model = False  # Whether model is deep learning (TimesNet) — skip feature extraction

        # Pre-fetch dataset_labels, raw mode flag, and DL flag from any endpoint in the pipeline
        for node in ordered_nodes:
            _nt = node.get('type', '')
            _eids = []
            if _nt.startswith('model.endpoint.'):
                _eids.append(_nt.replace('model.endpoint.', ''))
            if _nt == 'output.multi_model_compare':
                for _es in node.get('config', {}).get('endpoint_ids', []):
                    _eids.append(_es.split(':')[0])
            for _eid in _eids:
                _ep = MeLabEndpoint.get_by_id(_eid)
                if _ep:
                    _saved = SavedModel.get_by_id(_ep.get('saved_model_id'))
                    if _saved:
                        _pc = _saved.get('pipeline_config', {})
                        if isinstance(_pc, str):
                            _pc = json.loads(_pc) if _pc else {}
                        if _pc.get('no_windowing'):
                            is_raw_mode_model = True
                            print(f"[AppBuilder] Detected no_windowing=True for endpoint {_eid}", flush=True)
                        _alg = (_saved.get('algorithm') or '').lower()
                        _approach = (_pc.get('training_approach') or '').lower()
                        if _approach == 'dl' or _alg.startswith('timesnet'):
                            is_dl_model = True
                            print(f"[AppBuilder] Detected DL model for endpoint {_eid} (algorithm={_alg}, approach={_approach})", flush=True)
                        _di = _saved.get('dataset_info', {})
                        if isinstance(_di, str):
                            _di = json.loads(_di) if _di else {}
                        if _di.get('labels'):
                            dataset_labels = sorted([str(l) for l in _di['labels']])
                        break
            if dataset_labels or is_raw_mode_model or is_dl_model:
                break

        # If input is a DataFrame, remember column names and extract target
        import pandas as pd
        # Save raw input signal for multi-model timeline visualization
        raw_signal_preview = None
        try:
            if isinstance(current_data, pd.DataFrame):
                # Pick first numeric column (most representative) and downsample to max 500 points
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
                non_sensor = {'timestamp', 'time', 'index', 'label', 'class', 'target', 'category', 'sample_id'}
                sensor_only = [c for c in numeric_cols if c.lower() not in non_sensor]
                if sensor_only:
                    sig = current_data[sensor_only[0]].values
                    if len(sig) > 500:
                        idx = np.linspace(0, len(sig) - 1, 500, dtype=int)
                        sig = sig[idx]
                    raw_signal_preview = [float(v) for v in sig if not np.isnan(v)]
            elif isinstance(current_data, np.ndarray):
                # MQTT live: flatten and take first channel
                arr = current_data
                if arr.ndim == 2 and arr.shape[1] > 0:
                    sig = arr[:, 0]
                    if len(sig) > 500:
                        idx = np.linspace(0, len(sig) - 1, 500, dtype=int)
                        sig = sig[idx]
                    raw_signal_preview = [float(v) for v in sig if not np.isnan(v)]
        except Exception as _e:
            raw_signal_preview = None

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
            # Check model's pipeline_config for target_column and dataset_labels.
            # Collect from single model nodes AND multi-model compare endpoints.
            # NOTE: dataset_labels may already be set by the pre-fetch loop above;
            # only overwrite if we find a value here. Iterate ALL endpoints and only
            # break once we have both target_col and labels — otherwise a multi-model
            # app whose first endpoint lacks dataset_info.labels would silently skip
            # label decoding and produce 0% accuracy against integer-encoded targets.
            all_endpoint_ids = []
            for node in ordered_nodes:
                ntype = node.get('type', '')
                if ntype.startswith('model.endpoint.'):
                    all_endpoint_ids.append(ntype.replace('model.endpoint.', ''))
                if ntype == 'output.multi_model_compare':
                    for eidStr in node.get('config', {}).get('endpoint_ids', []):
                        all_endpoint_ids.append(eidStr.split(':')[0])
            for eid in all_endpoint_ids:
                ep = MeLabEndpoint.get_by_id(eid)
                if not ep:
                    continue
                saved = SavedModel.get_by_id(ep.get('saved_model_id'))
                if not saved:
                    continue
                pc = saved.get('pipeline_config', {})
                if isinstance(pc, str):
                    pc = json.loads(pc) if pc else {}
                if not target_col:
                    target_col = pc.get('target_column')
                if not dataset_labels:
                    di = saved.get('dataset_info', {})
                    if isinstance(di, str):
                        di = json.loads(di) if di else {}
                    if di.get('labels'):
                        dataset_labels = sorted([str(l) for l in di['labels']])
                if target_col and dataset_labels:
                    break

            if target_col and target_col in current_data.columns:
                raw_target = current_data[target_col].values
                # Decode integer labels using dataset_labels
                if dataset_labels and pd.api.types.is_numeric_dtype(current_data[target_col]):
                    max_idx = int(current_data[target_col].max())
                    if max_idx < len(dataset_labels):
                        raw_target = np.array([dataset_labels[int(v)] for v in raw_target])
                logger.info(f"[AppBuilder] Target column '{target_col}': {len(raw_target)} values")
            else:
                raw_target = None

        # Raw mode: skip windowing/normalize/feature_extract — pass raw values to model
        if is_raw_mode_model:
            if isinstance(current_data, pd.DataFrame):
                current_data, column_names = _extract_sensor_data(current_data, ordered_nodes)
            elif isinstance(current_data, np.ndarray):
                # MQTT JSON input: select only model's expected sensor columns from channels
                mqtt_channels = request_body.get('channels', []) if request_body else []
                # Get model's expected sensor columns from pipeline_config
                model_sensors = None
                for _n in ordered_nodes:
                    _nt = _n.get('type', '')
                    _eid = None
                    if _nt.startswith('model.endpoint.'):
                        _eid = _nt.replace('model.endpoint.', '')
                    if _eid:
                        _ep = MeLabEndpoint.get_by_id(_eid)
                        if _ep:
                            _saved = SavedModel.get_by_id(_ep.get('saved_model_id'))
                            if _saved:
                                _pc = _saved.get('pipeline_config', {})
                                if isinstance(_pc, str):
                                    _pc = json.loads(_pc) if _pc else {}
                                model_sensors = _pc.get('normalization', {}).get('sensor_columns', [])
                        break
                if model_sensors and mqtt_channels and len(mqtt_channels) == current_data.shape[1]:
                    # Map channel names to column indices, select only model's columns
                    ch_to_idx = {ch: i for i, ch in enumerate(mqtt_channels)}
                    col_indices = [ch_to_idx[s] for s in model_sensors if s in ch_to_idx]
                    if col_indices:
                        current_data = current_data[:, col_indices]
                        print(f"[AppBuilder] Raw mode MQTT: selected {len(col_indices)}/{len(mqtt_channels)} columns for model", flush=True)
                    else:
                        print(f"[AppBuilder] Raw mode MQTT: no matching columns! model expects {model_sensors[:5]}, got {mqtt_channels[:5]}", flush=True)
                else:
                    print(f"[AppBuilder] Raw mode MQTT: model_sensors={model_sensors is not None}, channels={len(mqtt_channels)}, data_cols={current_data.shape[1]}", flush=True)
            # current_data is now (n_rows, n_model_features) numpy array

        for node in ordered_nodes:
            ntype = node.get('type', '')
            params = node.get('config', {}) or {}

            if ntype.startswith('input.'):
                continue

            elif is_raw_mode_model and ntype in ('transform.window', 'transform.feature_extract'):
                # Skip windowing and feature extraction for raw mode models
                # (normalization is still applied — model was trained on normalized data)
                continue

            elif is_dl_model and ntype == 'transform.feature_extract':
                # DL models (TimesNet) consume the raw windowed tensor and learn features
                # internally. Skip statistical feature extraction so the model receives the
                # window shape it was trained on.
                logger.info(f"[AppBuilder] DL model detected — skipping feature_extract")
                continue

            elif ntype == 'transform.window':
                if isinstance(current_data, pd.DataFrame):
                    current_data, column_names = _extract_sensor_data(current_data, ordered_nodes)

                # Compute per-window actual values from raw_target (saved before normalize)
                if raw_target is not None and actual_values is None:
                    ws = params.get('window_size', 32)
                    st = params.get('step', params.get('stride', 16))
                    actual_values = []
                    # Determine mode from multi-model or single-model endpoint
                    _app_mode = None
                    for _n in ordered_nodes:
                        _nt = _n.get('type', '')
                        if _nt.startswith('model.endpoint.'):
                            _ep = MeLabEndpoint.get_by_id(_nt.replace('model.endpoint.', ''))
                            if _ep: _app_mode = _ep.get('mode')
                            break
                        if _nt == 'output.multi_model_compare':
                            _app_mode = _n.get('config', {}).get('mode', 'regression')
                            break
                    for start in range(0, len(raw_target) - ws + 1, st):
                        window = raw_target[start:start + ws]
                        if _app_mode in ('classification', 'anomaly'):
                            # Use mode (most frequent label) for classification/anomaly
                            from collections import Counter
                            actual_values.append(Counter(window).most_common(1)[0][0])
                        else:
                            actual_values.append(float(np.mean(window)))

                current_data = _apply_windowing(current_data, params)

            elif ntype == 'transform.normalize':
                # Convert DataFrame to numpy, keeping only sensor columns
                if isinstance(current_data, pd.DataFrame):
                    current_data, column_names = _extract_sensor_data(current_data, ordered_nodes)

                # Try to get normalization params from the model's pipeline_config
                norm_params = dict(params)
                _norm_eids = []
                for n2 in ordered_nodes:
                    _nt2 = n2.get('type', '')
                    if _nt2.startswith('model.endpoint.'):
                        _norm_eids.append(_nt2.replace('model.endpoint.', ''))
                    if _nt2 == 'output.multi_model_compare':
                        for _es in n2.get('config', {}).get('endpoint_ids', []):
                            _norm_eids.append(_es.split(':')[0])
                for _neid in _norm_eids:
                    ep = MeLabEndpoint.get_by_id(_neid)
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

            elif ntype == 'output.multi_model_compare':
                # Run multiple models on the same feature data
                endpoint_ids_raw = params.get('endpoint_ids', [])
                # Parse "id:name (algo)" format
                endpoint_ids = [eid.split(':')[0] for eid in endpoint_ids_raw if eid]
                # Set model_mode from node config
                model_mode = model_mode or params.get('mode')
                print(f"[AppBuilder] Multi-model: {len(endpoint_ids)} endpoints: {endpoint_ids}, data type={type(current_data).__name__}, shape={current_data.shape if hasattr(current_data, 'shape') else 'N/A'}", flush=True)
                if endpoint_ids and isinstance(current_data, np.ndarray):
                    # Filter to active endpoints first, then apply max 5 limit
                    valid_endpoints = []
                    for eid in endpoint_ids:
                        ep = MeLabEndpoint.get_by_id(eid)
                        if ep and ep['status'] == 'active':
                            valid_endpoints.append((eid, ep))
                        else:
                            print(f"[AppBuilder] Endpoint {eid}: not found or inactive (skipped)", flush=True)
                    valid_endpoints = valid_endpoints[:5]  # Max 5 active models
                    print(f"[AppBuilder] Running {len(valid_endpoints)} active endpoints", flush=True)

                    multi_results = {}
                    for eid, ep in valid_endpoints:
                        try:
                            model_mode = model_mode or ep.get('mode')
                            preds = _run_model_inference(eid, current_data.copy())
                            multi_results[eid] = {
                                'name': ep.get('name', eid),
                                'algorithm': ep.get('algorithm', ''),
                                'mode': ep.get('mode', ''),
                                'predictions': preds,
                            }
                            print(f"[AppBuilder] Endpoint {eid} ({ep.get('name')}): {len(preds)} predictions OK", flush=True)
                        except Exception as me:
                            print(f"[AppBuilder] Endpoint {eid} FAILED: {me}", flush=True)
                            multi_results[eid] = {
                                'name': eid,
                                'error': str(me),
                            }
                    print(f"[AppBuilder] Multi-model complete: {len(multi_results)} models ran", flush=True)
                    # Store multi results — will be used in response building
                    current_data = multi_results
                continue

            elif ntype.startswith('output.'):
                continue

    except Exception as e:
        import traceback
        logger.error(f"[AppBuilder] Pipeline error in app {slug}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Pipeline execution failed: {str(e)}'}), 500

    # Increment calls
    AppBuilderApp.increment_calls(app['id'])

    latency_ms = (time.time() - start_time) * 1000

    # Check if this is a multi-model comparison result
    is_multi = isinstance(current_data, dict) and all(
        isinstance(v, dict) and 'predictions' in v or 'error' in v
        for v in current_data.values()
    ) if isinstance(current_data, dict) else False

    if is_multi:
        # If live target values provided (MQTT with ground truth), compute actual
        if live_target_values and actual_values is None:
            try:
                vals = live_target_values
                if model_mode == 'regression':
                    actual_values = [float(np.mean([float(v) for v in vals]))]
                elif model_mode in ('classification', 'anomaly'):
                    # Decode integer labels using dataset_labels
                    from collections import Counter
                    most_common = Counter(vals).most_common(1)[0][0]
                    # Try to decode integer label
                    if dataset_labels and isinstance(most_common, (int, float)):
                        idx = int(most_common)
                        if 0 <= idx < len(dataset_labels):
                            most_common = dataset_labels[idx]
                    elif dataset_labels and str(most_common).isdigit():
                        idx = int(most_common)
                        if 0 <= idx < len(dataset_labels):
                            most_common = dataset_labels[idx]
                    actual_values = [most_common]
            except Exception as e:
                print(f"[AppBuilder] Live target decode error: {e}", flush=True)

        # Multi-model response
        # Count windows from first model's predictions
        first_model_count = 0
        for mresult in current_data.values():
            if 'predictions' in mresult:
                first_model_count = len(mresult['predictions'])
                break

        multi_response = {
            'app': app['name'],
            'slug': slug,
            'mode': model_mode,
            'multi_model': True,
            'actual': actual_values if actual_values else None,
            'num_windows': first_model_count,
            'models': {},
            'latency_ms': round(latency_ms, 1),
            'signal_preview': raw_signal_preview,
        }

        for eid, mresult in current_data.items():
            if 'error' in mresult:
                multi_response['models'][eid] = {'name': mresult.get('name', eid), 'error': mresult['error']}
                continue

            preds = mresult['predictions']
            pred_vals = []
            for p in preds:
                if isinstance(p, dict):
                    pred_vals.append(p.get('value', p.get('label', p)))
                else:
                    pred_vals.append(p)

            model_entry = {
                'name': mresult.get('name', eid),
                'algorithm': mresult.get('algorithm', ''),
                'mode': mresult.get('mode', model_mode),
                'predictions': pred_vals,
                'count': len(pred_vals),
            }

            # Compute metrics if actual values available
            if actual_values and len(actual_values) == len(pred_vals):
                if model_mode == 'regression':
                    vals = [v for v in pred_vals if isinstance(v, (int, float))]
                    acts = actual_values[:len(vals)]
                    if vals:
                        acts_arr = np.array(acts, dtype=np.float64)
                        preds_arr = np.array(vals, dtype=np.float64)
                        ss_res = np.sum((acts_arr - preds_arr) ** 2)
                        ss_tot = np.sum((acts_arr - np.mean(acts_arr)) ** 2)
                        model_entry['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
                        model_entry['rmse'] = float(np.sqrt(np.mean((acts_arr - preds_arr) ** 2)))
                        model_entry['mae'] = float(np.mean(np.abs(acts_arr - preds_arr)))
                elif model_mode == 'classification':
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_true = [str(a).strip().lower() for a in actual_values]
                    y_pred = [str(p).strip().lower() for p in pred_vals]
                    try:
                        model_entry['accuracy'] = float(accuracy_score(y_true, y_pred))
                        model_entry['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                        model_entry['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                        model_entry['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                    except Exception:
                        pass

            multi_response['models'][eid] = model_entry

        _persist_prediction(app, multi_response, request.get_json(silent=True) or {})
        return jsonify(multi_response)

    # Format response based on model predictions (single model)
    predictions = []
    if isinstance(current_data, list):
        predictions = current_data
    elif isinstance(current_data, np.ndarray):
        predictions = current_data.tolist()

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

    predictions_full = []
    if predictions and isinstance(predictions[0], dict):
        predictions_full = predictions

    response = {
        'app': app['name'],
        'slug': slug,
        'mode': model_mode,
        'predictions': pred_values,
        'predictions_full': predictions_full,
        'actual': actual_values if actual_values else None,
        'count': len(pred_values),
        'latency_ms': round(latency_ms, 1),
    }

    # Anomaly summary
    if model_mode == 'anomaly' and pred_values:
        n_anomaly = sum(1 for p in pred_values if str(p).lower() in ('anomaly', '1', '-1'))
        response['anomaly_count'] = n_anomaly
        response['normal_count'] = len(pred_values) - n_anomaly

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

    _persist_prediction(app, response, request.get_json(silent=True) or {})
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

        # If channel names provided (from live stream), return DataFrame
        channels = body.get('channels')
        if channels and len(channels) == arr.shape[1]:
            return pd.DataFrame(arr, columns=channels)

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


def _derive_app_mode(nodes):
    """Derive the mode from pipeline nodes (single or multi-model)."""
    if isinstance(nodes, str):
        nodes = json.loads(nodes)

    # Check single model node
    for node in nodes:
        ntype = node.get('type', '') if isinstance(node, dict) else ''
        if ntype.startswith('model.endpoint.'):
            eid = ntype.replace('model.endpoint.', '')
            ep = MeLabEndpoint.get_by_id(eid)
            if ep:
                return ep.get('mode')

    # Check multi-model compare endpoints
    for node in nodes:
        if isinstance(node, dict) and node.get('type') == 'output.multi_model_compare':
            endpoint_ids = node.get('config', {}).get('endpoint_ids', [])
            for eidStr in endpoint_ids:
                eid = eidStr.split(':')[0]
                ep = MeLabEndpoint.get_by_id(eid)
                if ep:
                    return ep.get('mode')

    # Check for signal recorder
    for node in nodes:
        if isinstance(node, dict) and node.get('type') == 'output.signal_recorder':
            return 'recorder'

    return None


def _derive_app_details(nodes):
    """Derive mode, algorithm, and sensor_columns from pipeline nodes."""
    mode = None
    algorithm = None
    sensor_columns = []

    # Collect endpoint IDs from all sources
    endpoint_ids = []
    for node in nodes:
        ntype = node.get('type', '') if isinstance(node, dict) else ''
        if ntype.startswith('model.endpoint.'):
            endpoint_ids.append(ntype.replace('model.endpoint.', ''))
        if ntype == 'output.multi_model_compare':
            for eidStr in node.get('config', {}).get('endpoint_ids', []):
                endpoint_ids.append(eidStr.split(':')[0])

    # Get details from first valid endpoint
    for eid in endpoint_ids:
        ep = MeLabEndpoint.get_by_id(eid)
        if ep:
            mode = ep.get('mode')
            algorithm = ep.get('algorithm')
            saved = SavedModel.get_by_id(ep.get('saved_model_id'))
            if saved:
                pc = saved.get('pipeline_config', {})
                if isinstance(pc, str):
                    pc = json.loads(pc) if pc else {}
                sensor_columns = pc.get('normalization', {}).get('sensor_columns', [])
            break

    # Check signal recorder
    if not mode:
        for node in nodes:
            if isinstance(node, dict) and node.get('type') == 'output.signal_recorder':
                mode = 'recorder'

    return mode, algorithm, sensor_columns


def _extract_sensor_data(df, ordered_nodes):
    """Extract sensor-only numeric data from DataFrame.
    Drops timestamp, label, category, and any columns not in the model's sensor_columns.
    Returns (numpy_array, column_names_list).
    """
    import pandas as pd
    sensor_df = df.select_dtypes(include=[np.number])

    # Find model's sensor columns from pipeline_config (single-model or multi-model)
    model_sensors = None
    all_eids = []
    for n in ordered_nodes:
        ntype = n.get('type', '')
        if ntype.startswith('model.endpoint.'):
            all_eids.append(ntype.replace('model.endpoint.', ''))
        if ntype == 'output.multi_model_compare':
            for eidStr in n.get('config', {}).get('endpoint_ids', []):
                all_eids.append(eidStr.split(':')[0])
    for eid in all_eids:
        ep = MeLabEndpoint.get_by_id(eid)
        if ep:
            saved = SavedModel.get_by_id(ep.get('saved_model_id'))
            if saved:
                pc = saved.get('pipeline_config', {})
                if isinstance(pc, str):
                    pc = json.loads(pc) if pc else {}
                model_sensors = pc.get('normalization', {}).get('sensor_columns', [])
        if model_sensors:
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

    If _model_norm is provided, use the training normalization params for
    consistency with how the model was trained. Supported training methods
    (from norm_params.method): 'min_max', 'z_score', 'robust', 'none'.
    Missing method defaults to 'min_max' (legacy models saved before F3).

    Fallback (no _model_norm): compute normalization from the input data
    using params.method — supports 'minmax', 'zscore', 'robust', 'none'.
    """
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).values

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)

    # Use model's training normalization if available
    model_norm = params.get('_model_norm')
    if model_norm:
        # Legacy models predate F3 and have no 'method' key — treat as min_max.
        train_method = model_norm.get('method', 'min_max')
        sensor_cols = model_norm.get('sensor_columns', [])
        input_cols = params.get('_sensor_columns', [])
        n_cols = data.shape[-1] if data.ndim >= 2 else 1

        if train_method == 'none':
            logger.info("[AppBuilder] Model trained with normalization='none' — passing through")
            return data

        if train_method == 'z_score':
            channel_mean = np.array(model_norm.get('channel_mean', []), dtype=np.float64)
            channel_std = np.array(model_norm.get('channel_std', []), dtype=np.float64)
            if len(channel_mean) > 0 and len(channel_std) > 0:
                d_mean = np.zeros(n_cols)
                d_std = np.ones(n_cols)
                for i, col in enumerate(input_cols):
                    if col in sensor_cols:
                        idx = sensor_cols.index(col)
                        if idx < len(channel_mean):
                            d_mean[i] = channel_mean[idx]
                            d_std[i] = channel_std[idx] if channel_std[idx] != 0 else 1.0
                logger.info(f"[AppBuilder] Applying model z_score normalization: mean={d_mean[:3]}..., std={d_std[:3]}...")
                return (data - d_mean) / d_std

        elif train_method == 'robust':
            channel_median = np.array(model_norm.get('channel_median', []), dtype=np.float64)
            channel_iqr = np.array(model_norm.get('channel_iqr', []), dtype=np.float64)
            if len(channel_median) > 0 and len(channel_iqr) > 0:
                d_median = np.zeros(n_cols)
                d_iqr = np.ones(n_cols)
                for i, col in enumerate(input_cols):
                    if col in sensor_cols:
                        idx = sensor_cols.index(col)
                        if idx < len(channel_median):
                            d_median[i] = channel_median[idx]
                            d_iqr[i] = channel_iqr[idx] if channel_iqr[idx] != 0 else 1.0
                logger.info(f"[AppBuilder] Applying model robust normalization: median={d_median[:3]}..., iqr={d_iqr[:3]}...")
                return (data - d_median) / d_iqr

        else:  # 'min_max' (or legacy no-method)
            channel_min = np.array(model_norm.get('channel_min', []), dtype=np.float64)
            channel_max = np.array(model_norm.get('channel_max', []), dtype=np.float64)
            if len(channel_min) > 0 and len(channel_max) > 0:
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
    elif method == 'robust':
        # Compute per-channel median and IQR from the input data
        d_median = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr > 0, iqr, 1.0)
        return (data - d_median) / iqr
    elif method == 'none':
        return data
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

        if data.ndim != 3:
            logger.warning(f"[AppBuilder] Skipping CiRA extractor: data is {data.ndim}D, expected 3D windowed. Falling back to generic.")
        elif not column_names:
            logger.warning(f"[AppBuilder] Skipping CiRA extractor: no column_names available and none derivable from feature names. Falling back to generic.")
        else:
            n_channels = data.shape[2]
            sensor_cols = [c for c in column_names if c not in ('timestamp', 'label', 'class', 'target')]
            if len(sensor_cols) > n_channels:
                sensor_cols = sensor_cols[:n_channels]

            if len(sensor_cols) != n_channels:
                logger.warning(
                    f"[AppBuilder] Skipping CiRA extractor: sensor column count {len(sensor_cols)} "
                    f"({sensor_cols}) does not match window channel count {n_channels}. "
                    f"Falling back to generic — predictions likely wrong."
                )
            else:
                # Use extract_from_windows_direct — produces ALL DSP features
                result_df = extractor.extract_from_windows_direct(data, sensor_cols)
                logger.info(f"[AppBuilder] Extracted {result_df.shape[1]} features from {data.shape[0]} windows")

                # Select only the features the model needs (in order)
                selected = []
                missing = []
                for fname in feature_names:
                    if fname in result_df.columns:
                        selected.append(result_df[fname].values)
                    else:
                        missing.append(fname)
                        selected.append(np.zeros(len(result_df)))

                if missing:
                    # Loud warning + surface the mismatch to the caller so the model
                    # doesn't silently receive zeros for expected features.
                    logger.warning(
                        f"[AppBuilder] {len(missing)}/{len(feature_names)} requested features "
                        f"not produced by extractor. Missing (first 5): {missing[:5]}. "
                        f"Available (first 5): {list(result_df.columns)[:5]}. "
                        f"Model will receive zeros for missing features — predictions may be wrong."
                    )
                    if len(missing) == len(feature_names):
                        raise ValueError(
                            f"None of the {len(feature_names)} requested features were produced by "
                            f"the extractor. Check that the Feature Extract config matches the model's "
                            f"trained feature names (available in the endpoint's 'Expects' panel). "
                            f"Sample requested: {feature_names[:3]}. "
                            f"Sample produced: {list(result_df.columns)[:3]}"
                        )

                return np.column_stack(selected) if selected else result_df.values
    except ValueError:
        # Preserve the clear feature-mismatch error we raise above; don't fall back silently.
        raise
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
    """Run ME-LAB model inference on data.

    Thin wrapper over ModelManager.predict_by_endpoint for backward
    compatibility with pipeline replay + multi-model compare paths.
    """
    try:
        return ModelManager.predict_by_endpoint(endpoint_id, data)
    except RuntimeError as e:
        # Preserve the ValueError contract older callers may expect.
        raise ValueError(str(e))
