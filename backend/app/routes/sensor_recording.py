"""
CiRA ME - Sensor Recording Routes
API endpoints for recording system sensor data (CPU, RAM, disk, network, GPU)
"""

import os
from flask import Blueprint, request, jsonify, current_app
from ..auth import admin_required
from ..services.sensor_recorder import start_recording, get_recording_status, stop_recording

sensor_bp = Blueprint('sensors', __name__)


@sensor_bp.route('/start', methods=['POST'])
@admin_required
def start_sensor_recording():
    """Start a sensor recording session."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    mode = data.get('mode', 'network_traffic')
    if mode not in ('manual', 'network_traffic', 'disk_io'):
        return jsonify({'error': f'Invalid mode: {mode}. Use manual, network_traffic, or disk_io'}), 400

    duration = data.get('duration', 120)
    rate = data.get('rate', 2)
    label = data.get('label', 'Normal')
    filename = data.get('filename', '').strip() or None

    if duration < 5 or duration > 600:
        return jsonify({'error': 'Duration must be between 5 and 600 seconds'}), 400
    if rate < 1 or rate > 10:
        return jsonify({'error': 'Rate must be between 1 and 10 Hz'}), 400

    # Output to shared/sensor_recordings/
    datasets_root = current_app.config['DATASETS_ROOT_PATH']
    shared = current_app.config['SHARED_FOLDER_PATH']
    output_dir = os.path.join(datasets_root, shared, 'sensor_recordings')

    try:
        job = start_recording(
            mode=mode,
            duration=duration,
            rate=rate,
            label=label,
            output_dir=output_dir,
            filename=filename,
        )
        return jsonify(job)
    except ImportError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@sensor_bp.route('/status/<job_id>', methods=['GET'])
@admin_required
def sensor_recording_status(job_id: str):
    """Get the status and progress of a sensor recording job."""
    job = get_recording_status(job_id)
    if not job:
        return jsonify({'error': f'Recording job not found: {job_id}'}), 404
    return jsonify(job)


@sensor_bp.route('/stop/<job_id>', methods=['POST'])
@admin_required
def stop_sensor_recording(job_id: str):
    """Stop a recording early and save partial data."""
    job = stop_recording(job_id)
    if not job:
        return jsonify({'error': f'Recording job not found: {job_id}'}), 404
    return jsonify(job)
