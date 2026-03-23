"""
CiRA ME - TI TinyML Routes
API endpoints for TI ModelMaker integration.
"""

import logging
from flask import Blueprint, request, jsonify, Response
from ..auth import login_required
from ..services.ti_integration import TIIntegration

logger = logging.getLogger(__name__)
ti_bp = Blueprint('ti_tinyml', __name__)


@ti_bp.route('/status', methods=['GET'])
@login_required
def ti_status():
    """Check if TI ModelMaker service is available."""
    ti = TIIntegration()
    return jsonify(ti.get_health())


@ti_bp.route('/devices', methods=['GET'])
@login_required
def ti_devices():
    """Get supported TI MCU devices."""
    try:
        ti = TIIntegration()
        devices = ti.get_devices()
        return jsonify(devices)
    except Exception as e:
        return jsonify({'error': f'TI service unavailable: {e}'}), 503


@ti_bp.route('/models', methods=['GET'])
@login_required
def ti_models():
    """Get available models from TI model zoo."""
    task = request.args.get('task', 'timeseries_regression')
    device = request.args.get('device')
    source = request.args.get('source', 'all')

    try:
        ti = TIIntegration()
        models = ti.get_models(task=task, device=device, source=source)
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': f'TI service unavailable: {e}'}), 503


@ti_bp.route('/train', methods=['POST'])
@login_required
def ti_train():
    """Train a model using TI ModelMaker."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    mode = data.get('mode', 'regression')
    model_names = data.get('model_names', [])
    model_name = data.get('model_name')  # backward compat
    target_device = data.get('target_device', 'F2837')
    dataset_path = data.get('dataset_path')
    config = data.get('config', {})

    if model_name and not model_names:
        model_names = [model_name]

    if not model_names:
        return jsonify({'error': 'model_names required'}), 400
    if not dataset_path:
        return jsonify({'error': 'dataset_path required'}), 400

    # Map dataset path for TI container
    ti_dataset_path = dataset_path.replace(
        '/app/datasets/shared', '/app/data/datasets/shared'
    )

    try:
        ti = TIIntegration()
        task_type = ti.map_cira_mode_to_ti_task(mode)

        result = ti.train(
            task_type=task_type,
            model_names=model_names,
            target_device=target_device,
            dataset_path=ti_dataset_path,
            config=config,
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"TI training error: {e}")
        return jsonify({'error': str(e)}), 500


@ti_bp.route('/train-stream', methods=['POST'])
@login_required
def ti_train_stream():
    """Stream training progress via SSE."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    mode = data.get('mode', 'regression')
    model_name = data.get('model_name')
    target_device = data.get('target_device', 'F2837')
    dataset_path = data.get('dataset_path')
    config = data.get('config', {})

    if not model_name or not dataset_path:
        return jsonify({'error': 'model_name and dataset_path required'}), 400

    ti_dataset_path = dataset_path.replace(
        '/app/datasets/shared', '/app/data/datasets/shared'
    )

    ti = TIIntegration()
    task_type = ti.map_cira_mode_to_ti_task(mode)

    import requests as req

    def generate():
        try:
            resp = req.post(
                f'{ti.base_url}/train-stream',
                json={
                    'task_type': task_type,
                    'model_name': model_name,
                    'target_device': target_device,
                    'dataset_path': ti_dataset_path,
                    'config': config,
                },
                stream=True,
                timeout=660,
            )
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line + '\n\n'
        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@ti_bp.route('/download/<run_id>', methods=['GET'])
@login_required
def ti_download(run_id):
    """Download compiled model artifacts."""
    try:
        ti = TIIntegration()
        content = ti.download_artifacts(run_id)
        return Response(
            content,
            mimetype='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename=ti_model_{run_id}.zip'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
