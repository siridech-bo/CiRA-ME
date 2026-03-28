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
        sensor_cols = [c for c in numeric_cols if c.lower() not in ('timestamp', 'time', 'index')]
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

    thread = threading.Thread(
        target=_publish_worker,
        args=(session_id, df, sensor_cols, topic, rate, loop),
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
    """List available CSV files for publishing."""
    datasets_root = os.environ.get('DATASETS_ROOT_PATH',
                                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets'))
    files = []
    for root, dirs, filenames in os.walk(datasets_root):
        for f in filenames:
            if f.endswith('.csv'):
                rel = os.path.relpath(os.path.join(root, f), datasets_root)
                size = os.path.getsize(os.path.join(root, f))
                files.append({
                    'path': rel.replace('\\', '/'),
                    'name': f,
                    'size_kb': round(size / 1024, 1),
                })
    return jsonify(files[:100])  # Limit to 100 files


def _publish_worker(session_id, df, sensor_cols, topic, rate, loop):
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
