"""
CiRA ME - TI TinyML Routes
API endpoints for TI ModelMaker integration.
"""

import math
import logging
from flask import Blueprint, request, jsonify, Response
from ..auth import login_required
from ..services.ti_integration import TIIntegration


def _sanitize_nan(obj):
    """Replace NaN/Inf float values with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj

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


@ti_bp.route('/train-ml', methods=['POST'])
@login_required
def ti_train_ml_with_features():
    """Train Traditional ML model using CiRA ME's feature pipeline, export via emlearn.

    This route trains using CiRA ME's windowed+extracted features (same as Traditional ML tab),
    then sends the trained model to TI container for emlearn C code export.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    model_name = data.get('model_name')  # ML_DT_REG, ML_RF_REG, etc.
    target_device = data.get('target_device', 'F2837')
    test_size = data.get('test_size', 0.2)
    hyperparameters = data.get('hyperparameters', {})
    mode = data.get('mode', 'regression')

    if not feature_session_id:
        return jsonify({'error': 'feature_session_id required (extract features first)'}), 400
    if not model_name:
        return jsonify({'error': 'model_name required'}), 400

    try:
        from ..services.ml_trainer import MLTrainer
        from ..config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS

        trainer = MLTrainer()

        # Map TI model names to CiRA ME algorithm names
        ti_to_cira = {
            'ML_DT_REG': 'dt_reg', 'ML_RF_REG': 'rf_reg',
            'ML_XGB_REG': 'xgb_reg', 'ML_LGBM_REG': 'lgbm_reg',
            'ML_DT_CLF': 'dt', 'ML_RF_CLF': 'rf',
            'ML_XGB_CLF': 'gb', 'ML_IFOREST': 'iforest',
        }
        cira_algo = ti_to_cira.get(model_name)
        if not cira_algo:
            return jsonify({'error': f'Unknown ML model: {model_name}'}), 400

        # Train using CiRA ME's feature pipeline
        if mode == 'regression':
            result = trainer.train_regression(
                feature_session_id, cira_algo, hyperparameters,
                test_size=test_size,
                user_id=request.current_user['id']
            )
        elif mode == 'anomaly':
            result = trainer.train_anomaly(
                feature_session_id, cira_algo, hyperparameters,
                user_id=request.current_user['id']
            )
        else:
            result = trainer.train_classification(
                feature_session_id, cira_algo, hyperparameters,
                test_size=test_size,
                user_id=request.current_user['id']
            )

        # Add model size estimate
        import os
        model_path = result.get('model_path', '')
        if model_path and os.path.exists(model_path):
            size_kb = os.path.getsize(model_path) / 1024
            result['metrics']['model_size_kb'] = round(size_kb, 1)
            # emlearn C code is much smaller than pickle
            result['metrics']['model_size_int8_kb'] = round(size_kb * 0.1, 1)

        # Add pipeline info
        result['pipeline'] = 'cira_features'

        return jsonify(_sanitize_nan(result))

    except Exception as e:
        logger.error(f"TI ML training error: {e}")
        return jsonify({'error': str(e)}), 400


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
