"""
CiRA ME - MQTT Test Publisher
Publishes CSV data to MQTT broker for testing live stream apps.
"""

import os
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from ..auth import login_required

mqtt_bp = Blueprint('mqtt', __name__)
logger = logging.getLogger(__name__)

# Active publishers (keyed by session_id)
_publishers = {}


@mqtt_bp.route('/status', methods=['GET'])
@login_required
def mqtt_status():
    """Check MQTT broker connectivity and list active publishers."""
    broker_ok = False
    broker_url = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

    try:
        import paho.mqtt.client as paho_mqtt
        client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker_url, broker_port, keepalive=5)
        client.disconnect()
        broker_ok = True
    except Exception as e:
        logger.warning(f"MQTT broker check failed: {e}")

    active = []
    for sid, pub in _publishers.items():
        active.append({
            'session_id': sid,
            'topic': pub.get('topic'),
            'file': pub.get('filename'),
            'rate': pub.get('rate'),
            'published': pub.get('published', 0),
            'total': pub.get('total', 0),
            'running': pub.get('running', False),
        })

    return jsonify({
        'broker_connected': broker_ok,
        'broker_host': broker_url,
        'broker_port': broker_port,
        'active_publishers': active,
    })


@mqtt_bp.route('/publish', methods=['POST'])
@login_required
def start_publish():
    """Start publishing CSV data to MQTT topic.

    Body (JSON):
      file_path: path to CSV file in datasets
      topic: MQTT topic to publish to
      rate: messages per second (default 10)
      loop: whether to loop the data (default false)
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    file_path = data.get('file_path', '')
    topic = data.get('topic', 'sensors/test')
    rate = float(data.get('rate', 10))
    loop = data.get('loop', False)

    if not file_path:
        return jsonify({'error': 'file_path required'}), 400

    # Resolve path
    datasets_root = os.environ.get('DATASETS_ROOT_PATH',
                                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets'))
    full_path = os.path.join(datasets_root, file_path)
    if not os.path.exists(full_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    # Load CSV
    try:
        df = pd.read_csv(full_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude timestamp-like columns
        # Any column name that lowercases into this set is treated as
        # metadata, not a sensor channel. Widened so publishers don't
        # accidentally leak timestamps/indices as sensor values, which
        # then bump downstream subscribers off-by-one during positional
        # channel fallback. Cover common time-column aliases.
        non_sensor = (
            'timestamp', 'time', 'ts', 'datetime', 'date', 'epoch', 'unix_time',
            'elapsed', 'sec', 'seconds', 'ms', 'milliseconds', 't',
            'index', 'idx', 'i', 'row', 'row_id', 'sample_id',
            'label', 'labels', 'class', 'class_name', 'target', 'category',
        )
        sensor_cols = [c for c in numeric_cols if c.lower() not in non_sensor]
        if not sensor_cols:
            return jsonify({'error': 'CSV has no numeric sensor columns'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

    session_id = f"pub_{int(time.time())}_{os.path.basename(file_path)[:20]}"

    # Stop existing publisher with same topic
    for sid, pub in list(_publishers.items()):
        if pub.get('topic') == topic and pub.get('running'):
            pub['running'] = False

    # Start publisher thread
    pub_state = {
        'topic': topic,
        'filename': os.path.basename(file_path),
        'rate': rate,
        'loop': loop,
        'published': 0,
        'total': len(df),
        'running': True,
        'sensor_cols': sensor_cols,
        'error': None,
    }
    _publishers[session_id] = pub_state

    # Detect label column for ground truth inclusion
    label_col = None
    for candidate in ['label', 'labels', 'class', 'class_name', 'target', 'category']:
        if candidate in df.columns:
            label_col = candidate
            break

    # For CBOR-converted CSVs, the filename contains the true label (e.g., idle.1.cbor...)
    # Build a mapping from integer labels to string labels using filename
    label_decode_map = None
    if label_col:
        fname = os.path.basename(file_path).lower()
        # CBOR format: {label}.{index}.cbor...
        parts = fname.split('.')
        if len(parts) >= 3 and 'cbor' in fname:
            string_label = parts[0]  # e.g., "idle", "wave", "updown"
            # All rows in this file have the same label
            int_label = df[label_col].iloc[0]
            label_decode_map = {int_label: string_label}
            # Try to build full map from the dataset folder
            dataset_dir = os.path.dirname(full_path)
            try:
                for f2 in os.listdir(dataset_dir):
                    if f2.endswith('.csv') and 'cbor' in f2.lower():
                        p2 = f2.split('.')
                        if len(p2) >= 3:
                            lbl_str = p2[0]
                            try:
                                df2 = pd.read_csv(os.path.join(dataset_dir, f2), nrows=1)
                                if label_col in df2.columns:
                                    lbl_int = df2[label_col].iloc[0]
                                    label_decode_map[lbl_int] = lbl_str
                            except Exception:
                                pass
            except Exception:
                pass

    thread = threading.Thread(
        target=_publish_worker,
        args=(session_id, df, sensor_cols, topic, rate, loop, label_col, label_decode_map),
        daemon=True,
    )
    thread.start()

    return jsonify({
        'session_id': session_id,
        'topic': topic,
        'rate': rate,
        'total_rows': len(df),
        'sensor_columns': sensor_cols,
        'message': f'Publishing {len(df)} rows to {topic} at {rate}/s',
    })


@mqtt_bp.route('/publish/<session_id>/stop', methods=['POST'])
@login_required
def stop_publish(session_id):
    """Stop an active publisher."""
    pub = _publishers.get(session_id)
    if not pub:
        return jsonify({'error': 'Publisher not found'}), 404
    pub['running'] = False
    return jsonify({'message': 'Publisher stopped', 'published': pub['published']})


@mqtt_bp.route('/datasets', methods=['GET'])
@login_required
def list_datasets():
    """List available CSV files for publishing.

    Skips internal / hidden folders (names starting with `.`) so operators
    don't have to wade through `.multi_csv_selections/` artifacts. No count
    cap — the frontend renders a browser dialog with folder navigation
    rather than a flat dropdown.
    """
    datasets_root = os.environ.get('DATASETS_ROOT_PATH',
                                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets'))
    files = []
    for root, dirs, filenames in os.walk(datasets_root):
        # Prune hidden / internal folders in-place so os.walk skips them.
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in filenames:
            if not f.endswith('.csv'):
                continue
            if f.startswith('.'):
                continue
            rel = os.path.relpath(os.path.join(root, f), datasets_root)
            try:
                size = os.path.getsize(os.path.join(root, f))
            except OSError:
                size = 0
            files.append({
                'path': rel.replace('\\', '/'),
                'name': f,
                'size_kb': round(size / 1024, 1),
            })
    return jsonify(files)


@mqtt_bp.route('/ws-info', methods=['GET'])
@login_required
def ws_info():
    """Lightweight endpoint: return MQTT WebSocket host + port for the browser.

    Unlike /broker-info this does NOT connect to the broker (no 2s sleep for
    $SYS scrape). Used by the browser-side MQTT Test Publisher so ~65 workshop
    attendees don't slam Flask.
    """
    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    ws_port = int(os.environ.get('MQTT_BROKER_WS_PORT', '9001'))
    return jsonify({
        'host': broker_host,
        'ws_port': ws_port,
    })


@mqtt_bp.route('/csv-rows', methods=['GET'])
@login_required
def csv_rows():
    """Return CSV file as parsed JSON rows for browser-side MQTT publishing.

    Query params:
      path: relative path under DATASETS_ROOT_PATH

    Response matches what _publish_worker would produce, so the browser can
    build the same payload shape (sensor cols + optional label decoding).
    """
    file_path = request.args.get('path', '')
    if not file_path:
        return jsonify({'error': 'path required'}), 400

    datasets_root = os.environ.get('DATASETS_ROOT_PATH',
                                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets'))
    full_path = os.path.join(datasets_root, file_path)
    if not os.path.exists(full_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    # Load CSV — verbatim from start_publish sensor-column detection.
    try:
        df = pd.read_csv(full_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Any column name that lowercases into this set is treated as
        # metadata, not a sensor channel. Widened so publishers don't
        # accidentally leak timestamps/indices as sensor values, which
        # then bump downstream subscribers off-by-one during positional
        # channel fallback. Cover common time-column aliases.
        non_sensor = (
            'timestamp', 'time', 'ts', 'datetime', 'date', 'epoch', 'unix_time',
            'elapsed', 'sec', 'seconds', 'ms', 'milliseconds', 't',
            'index', 'idx', 'i', 'row', 'row_id', 'sample_id',
            'label', 'labels', 'class', 'class_name', 'target', 'category',
        )
        sensor_cols = [c for c in numeric_cols if c.lower() not in non_sensor]
        if not sensor_cols:
            return jsonify({'error': 'CSV has no numeric sensor columns'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

    # Detect label column — verbatim from start_publish.
    label_col = None
    for candidate in ['label', 'labels', 'class', 'class_name', 'target', 'category']:
        if candidate in df.columns:
            label_col = candidate
            break

    # Build integer→string label map for CBOR-converted CSVs — verbatim from start_publish.
    label_decode_map = None
    if label_col:
        fname = os.path.basename(file_path).lower()
        parts = fname.split('.')
        if len(parts) >= 3 and 'cbor' in fname:
            string_label = parts[0]
            int_label = df[label_col].iloc[0]
            label_decode_map = {int_label: string_label}
            dataset_dir = os.path.dirname(full_path)
            try:
                for f2 in os.listdir(dataset_dir):
                    if f2.endswith('.csv') and 'cbor' in f2.lower():
                        p2 = f2.split('.')
                        if len(p2) >= 3:
                            lbl_str = p2[0]
                            try:
                                df2 = pd.read_csv(os.path.join(dataset_dir, f2), nrows=1)
                                if label_col in df2.columns:
                                    lbl_int = df2[label_col].iloc[0]
                                    label_decode_map[lbl_int] = lbl_str
                            except Exception:
                                pass
            except Exception:
                pass

    # Build row list: sensor columns as floats + label column (raw, unicoded).
    # Frontend applies label_decode_map when constructing the MQTT payload.
    cols_to_emit = list(sensor_cols)
    if label_col and label_col in df.columns:
        cols_to_emit.append(label_col)

    rows = []
    sub = df[cols_to_emit]
    for rec in sub.to_dict(orient='records'):
        clean = {}
        for c in sensor_cols:
            v = rec.get(c)
            try:
                clean[c] = float(v)
            except (TypeError, ValueError):
                clean[c] = None
        if label_col:
            raw = rec.get(label_col)
            # Match _publish_worker: unwrap numpy scalars.
            if hasattr(raw, 'item'):
                raw = raw.item()
            clean[label_col] = raw
        rows.append(clean)

    # JSON keys must be strings — stringify int keys in label_decode_map.
    decode_out = None
    if label_decode_map:
        decode_out = {}
        for k, v in label_decode_map.items():
            if hasattr(k, 'item'):
                k = k.item()
            decode_out[str(k)] = v

    return jsonify({
        'rows': rows,
        'sensor_columns': sensor_cols,
        'label_column': label_col,
        'label_decode_map': decode_out,
        'total_rows': len(df),
    })


@mqtt_bp.route('/broker-info', methods=['GET'])
@login_required
def broker_info():
    """Return broker host, port, WebSocket port, connection status, and $SYS stats."""
    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))
    ws_port = int(os.environ.get('MQTT_BROKER_WS_PORT', '9001'))

    result = {
        'host': broker_host,
        'port': broker_port,
        'ws_port': ws_port,
        'connected': False,
        'version': None,
        'clients_connected': None,
        'messages_received': None,
        'messages_sent': None,
        'uptime': None,
    }

    try:
        import paho.mqtt.client as paho_mqtt

        sys_data = {}
        got_data = threading.Event()

        def on_connect(client, userdata, flags, rc, properties=None):
            client.subscribe('$SYS/#')

        def on_message(client, userdata, msg):
            try:
                sys_data[msg.topic] = msg.payload.decode('utf-8', errors='replace')
            except Exception:
                pass

        client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker_host, broker_port, keepalive=10)
        client.loop_start()

        # Wait up to 2 seconds for $SYS messages
        time.sleep(2)

        client.loop_stop()
        client.disconnect()

        result['connected'] = True
        result['version'] = sys_data.get('$SYS/broker/version')
        result['clients_connected'] = sys_data.get('$SYS/broker/clients/connected')
        result['messages_received'] = sys_data.get('$SYS/broker/messages/received')
        result['messages_sent'] = sys_data.get('$SYS/broker/messages/sent')
        result['uptime'] = sys_data.get('$SYS/broker/uptime')

    except Exception as e:
        logger.warning(f"Broker info fetch failed: {e}")

    return jsonify(result)


@mqtt_bp.route('/topics', methods=['GET'])
@login_required
def list_topics():
    """Subscribe to # for several seconds and return discovered topics."""
    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))
    duration = min(int(request.args.get('duration', 5)), 15)  # max 15 seconds

    topics = {}

    try:
        import paho.mqtt.client as paho_mqtt

        def on_connect(client, userdata, flags, rc, properties=None):
            client.subscribe('#')

        def on_message(client, userdata, msg):
            topic = msg.topic
            if not topic.startswith('$SYS'):
                topics[topic] = time.time()

        client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker_host, broker_port, keepalive=10)
        client.loop_start()

        time.sleep(duration)

        client.loop_stop()
        client.disconnect()

    except Exception as e:
        logger.warning(f"Topic discovery failed: {e}")

    result = []
    for topic, ts in sorted(topics.items()):
        result.append({
            'topic': topic,
            'last_seen': ts,
        })

    return jsonify(result)


@mqtt_bp.route('/topics/subscribe-test', methods=['POST'])
@login_required
def subscribe_test():
    """Test subscribe to a topic and return last N messages."""
    data = request.get_json()
    if not data or not data.get('topic'):
        return jsonify({'error': 'topic is required'}), 400

    topic = data['topic']
    count = int(data.get('count', 5))
    timeout = min(float(data.get('timeout', 5)), 10)  # max 10 seconds

    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

    messages = []

    try:
        import paho.mqtt.client as paho_mqtt

        def on_connect(client, userdata, flags, rc, properties=None):
            client.subscribe(topic)

        def on_message(client, userdata, msg):
            try:
                payload = msg.payload.decode('utf-8', errors='replace')
            except Exception:
                payload = str(msg.payload)
            messages.append({
                'topic': msg.topic,
                'payload': payload,
                'timestamp': time.time(),
                'qos': msg.qos,
            })

        client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker_host, broker_port, keepalive=10)
        client.loop_start()

        # Wait until we have enough messages or timeout
        start = time.time()
        while len(messages) < count and (time.time() - start) < timeout:
            time.sleep(0.1)

        client.loop_stop()
        client.disconnect()

    except Exception as e:
        logger.warning(f"Subscribe test failed: {e}")
        return jsonify({'error': str(e), 'messages': []}), 500

    return jsonify({
        'topic': topic,
        'count': len(messages),
        'messages': messages[:count],
    })


def _publish_worker(session_id, df, sensor_cols, topic, rate, loop, label_col=None, label_decode_map=None):
    """Background thread that publishes CSV rows to MQTT."""
    import paho.mqtt.client as paho_mqtt

    pub = _publishers.get(session_id)
    if not pub:
        return

    broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
    broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

    try:
        client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker_host, broker_port, keepalive=60)
        client.loop_start()
    except Exception as e:
        pub['error'] = str(e)
        pub['running'] = False
        logger.error(f"MQTT publisher failed to connect: {e}")
        return

    interval = 1.0 / rate if rate > 0 else 0.1
    row_idx = 0

    try:
        while pub['running']:
            row = df.iloc[row_idx]
            payload = {col: float(row[col]) for col in sensor_cols}
            payload['_timestamp'] = time.time()
            payload['_index'] = row_idx
            # Include label column for ground truth (decoded to string if possible)
            if label_col and label_col in df.columns:
                raw_label = row[label_col]
                if hasattr(raw_label, 'item'):
                    raw_label = raw_label.item()
                # Decode integer label to string using filename-derived map
                if label_decode_map and raw_label in label_decode_map:
                    payload[label_col] = label_decode_map[raw_label]
                else:
                    payload[label_col] = raw_label

            client.publish(topic, json.dumps(payload), qos=0)
            pub['published'] += 1

            row_idx += 1
            if row_idx >= len(df):
                if loop:
                    row_idx = 0
                else:
                    break

            time.sleep(interval)
    except Exception as e:
        pub['error'] = str(e)
        logger.error(f"MQTT publisher error: {e}")
    finally:
        pub['running'] = False
        client.loop_stop()
        client.disconnect()
        logger.info(f"MQTT publisher {session_id} stopped after {pub['published']} messages")
