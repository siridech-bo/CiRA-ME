"""
CiRA ME - ML Training Routes
Handles Anomaly Detection (PyOD), Classification (Scikit-learn), and Deep Learning (TimesNet)
"""

import logging
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..services.ml_trainer import MLTrainer
from ..services.timesnet_trainer import TimesNetTrainer, TimesNetConfig
from ..services.data_loader import _data_sessions
from ..config import ANOMALY_ALGORITHMS, CLASSIFICATION_ALGORITHMS

logger = logging.getLogger(__name__)
training_bp = Blueprint('training', __name__)


@training_bp.route('/algorithms', methods=['GET'])
@login_required
def get_algorithms():
    """Get available ML and DL algorithms for both modes."""
    return jsonify({
        'anomaly_detection': ANOMALY_ALGORITHMS,
        'classification': CLASSIFICATION_ALGORITHMS,
        'deep_learning': {
            'timesnet': {
                'name': 'TimesNet',
                'description': 'State-of-the-art time-series deep learning model'
            }
        }
    })


@training_bp.route('/gpu-status', methods=['GET'])
@login_required
def get_gpu_status():
    """
    Get GPU availability and status for deep learning training.
    Returns whether CUDA is available, GPU info, and memory usage.

    Uses subprocess to check GPU status to avoid DLL conflicts.
    """
    import os
    import sys
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    status = {
        'available': False,
        'cuda_available': False,
        'device_name': None,
        'device_count': 0,
        'memory_total': None,
        'memory_used': None,
        'memory_free': None,
        'error': None,
        'recommendation': 'cpu',
        'torch_available': False
    }

    # Use subprocess approach to avoid DLL conflicts with other CUDA apps
    config_path = None
    output_path = None

    try:
        subprocess_script = Path(__file__).parent.parent / 'services' / 'torch_subprocess.py'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            job = {'task': 'check_gpu', 'config': {}, 'data': {}}
            json.dump(job, config_file)
            config_path = config_file.name

        output_path = config_path.replace('.json', '_output.json')

        result = subprocess.run(
            [sys.executable, str(subprocess_script), config_path, output_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                subprocess_status = json.load(f)

            status.update(subprocess_status)

            if status.get('cuda_available'):
                status['recommendation'] = 'cuda'
            else:
                status['recommendation'] = 'cpu'
                if not status.get('error'):
                    status['info'] = 'CUDA not available. Training will use CPU.'

            return jsonify(status)

        else:
            # Subprocess failed but didn't throw exception
            if result.stderr:
                logger.warning(f"GPU check subprocess stderr: {result.stderr}")
            status['torch_available'] = True  # Assume available if subprocess ran
            status['recommendation'] = 'cpu'
            status['info'] = 'GPU check completed. Subprocess training available.'

    except subprocess.TimeoutExpired:
        status['error'] = 'GPU check timed out'
        status['torch_available'] = True
        status['info'] = 'GPU check timed out but training should work.'
    except Exception as e:
        logger.warning(f"Subprocess GPU check failed: {e}")
        status['error'] = str(e)
        status['torch_available'] = True  # Subprocess failed but torch may still work
        status['info'] = 'Subprocess training available.'
    finally:
        # Cleanup temp files
        for path in [config_path, output_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass

    return jsonify(status)


@training_bp.route('/train/anomaly', methods=['POST'])
@login_required
def train_anomaly_model():
    """Train an anomaly detection model using PyOD."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithm = data.get('algorithm', 'iforest')
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if algorithm not in ANOMALY_ALGORITHMS:
        return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_anomaly(
            feature_session_id,
            algorithm,
            hyperparameters,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/train/classification', methods=['POST'])
@login_required
def train_classification_model():
    """Train a classification model using Scikit-learn."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithm = data.get('algorithm', 'rf')
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')
    test_size = data.get('test_size', 0.2)

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if algorithm not in CLASSIFICATION_ALGORITHMS:
        return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_classification(
            feature_session_id,
            algorithm,
            hyperparameters,
            test_size=test_size,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/train/anomaly/compare', methods=['POST'])
@login_required
def train_anomaly_compare():
    """
    Train multiple anomaly detection algorithms and compare their performance.

    Request body:
    {
        "feature_session_id": "session_id",
        "algorithms": ["iforest", "lof", "hbos", "knn"],
        "hyperparameters": {"contamination": 0.1, "n_estimators": 100}
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithms = data.get('algorithms', [])
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if not algorithms or not isinstance(algorithms, list):
        return jsonify({'error': 'algorithms must be a non-empty list'}), 400

    # Validate algorithms
    invalid_algos = [a for a in algorithms if a not in ANOMALY_ALGORITHMS]
    if invalid_algos:
        return jsonify({'error': f'Unknown algorithms: {invalid_algos}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_anomaly_compare(
            feature_session_id,
            algorithms,
            hyperparameters,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Anomaly comparison training error: {e}")
        return jsonify({'error': str(e)}), 400


@training_bp.route('/train/classification/compare', methods=['POST'])
@login_required
def train_classification_compare():
    """
    Train multiple classification algorithms and compare their performance.

    Request body:
    {
        "feature_session_id": "session_id",
        "algorithms": ["rf", "gb", "svm", "knn"],
        "hyperparameters": {"n_estimators": 100},
        "test_size": 0.2
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithms = data.get('algorithms', [])
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')
    test_size = data.get('test_size', 0.2)

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if not algorithms or not isinstance(algorithms, list):
        return jsonify({'error': 'algorithms must be a non-empty list'}), 400

    # Validate algorithms
    invalid_algos = [a for a in algorithms if a not in CLASSIFICATION_ALGORITHMS]
    if invalid_algos:
        return jsonify({'error': f'Unknown algorithms: {invalid_algos}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_classification_compare(
            feature_session_id,
            algorithms,
            hyperparameters,
            test_size=test_size,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Classification comparison training error: {e}")
        return jsonify({'error': str(e)}), 400


@training_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    """Make predictions using a trained model."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    training_session_id = data.get('training_session_id')
    feature_session_id = data.get('feature_session_id')

    if not training_session_id or not feature_session_id:
        return jsonify({'error': 'Training session ID and feature session ID required'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.predict(training_session_id, feature_session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/metrics/<training_session_id>', methods=['GET'])
@login_required
def get_metrics(training_session_id: str):
    """Get metrics for a training session."""
    try:
        trainer = MLTrainer()
        result = trainer.get_metrics(training_session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/export/<training_session_id>', methods=['POST'])
@login_required
def export_model(training_session_id: str):
    """Export a trained model to various formats."""
    data = request.get_json() or {}

    export_format = data.get('format', 'pickle')  # pickle, onnx, joblib

    try:
        trainer = MLTrainer()
        result = trainer.export_model(training_session_id, export_format)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============== TimesNet Deep Learning Routes ==============

@training_bp.route('/timesnet/config', methods=['GET'])
@login_required
def get_timesnet_config():
    """Get default TimesNet configuration."""
    mode = request.args.get('mode', 'anomaly')
    num_channels = int(request.args.get('num_channels', 3))
    num_classes = int(request.args.get('num_classes', 2))

    trainer = TimesNetTrainer()
    config = trainer.get_default_config(mode, num_channels, num_classes)

    return jsonify({
        'config': config.to_dict(),
        'period_options': [4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        'description': {
            'seq_len': 'Input sequence length (window size)',
            'd_model': 'Model dimension (embedding size)',
            'd_ff': 'Feed-forward dimension',
            'num_kernels': 'Number of inception kernels',
            'top_k': 'Number of top periods to use',
            'e_layers': 'Number of encoder layers',
            'dropout': 'Dropout rate',
            'period_list': 'List of periods for multi-periodic analysis'
        }
    })


@training_bp.route('/timesnet/train/anomaly', methods=['POST'])
@login_required
def train_timesnet_anomaly():
    """Train TimesNet for anomaly detection (end-to-end, no feature extraction needed)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    windowed_session_id = data.get('windowed_session_id')
    config_data = data.get('config', {})
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 32)
    learning_rate = data.get('learning_rate', 0.001)
    device = data.get('device', 'cpu')  # 'cpu' or 'cuda'

    if not windowed_session_id:
        return jsonify({'error': 'Windowed session ID required'}), 400

    # Get windowed data
    session = _data_sessions.get(windowed_session_id)
    if not session or 'windows' not in session:
        return jsonify({'error': 'Windowed session not found'}), 400

    windows = session['windows']
    labels = session.get('labels')
    categories = session.get('categories')  # For proper train/test split

    try:
        trainer = TimesNetTrainer(device=device)

        # Build config
        num_channels = windows.shape[2] if len(windows.shape) == 3 else 1
        config = TimesNetConfig(
            seq_len=config_data.get('seq_len', windows.shape[1]),
            enc_in=num_channels,
            d_model=config_data.get('d_model', 64),
            d_ff=config_data.get('d_ff', 128),
            num_kernels=config_data.get('num_kernels', 6),
            top_k=config_data.get('top_k', 3),
            e_layers=config_data.get('e_layers', 2),
            dropout=config_data.get('dropout', 0.1),
            period_list=config_data.get('period_list', [8, 16, 32, 64]),
            task_name='anomaly_detection'
        )

        result = trainer.train_anomaly(
            windows=windows,
            labels=labels,
            categories=categories,  # Pass categories for proper evaluation
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            user_id=request.current_user['id']
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/timesnet/train/classification', methods=['POST'])
@login_required
def train_timesnet_classification():
    """Train TimesNet for classification (end-to-end, no feature extraction needed)."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    windowed_session_id = data.get('windowed_session_id')
    config_data = data.get('config', {})
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 32)
    learning_rate = data.get('learning_rate', 0.001)
    test_size = data.get('test_size', 0.2)
    device = data.get('device', 'cpu')  # 'cpu' or 'cuda'

    if not windowed_session_id:
        return jsonify({'error': 'Windowed session ID required'}), 400

    # Get windowed data
    session = _data_sessions.get(windowed_session_id)
    if not session or 'windows' not in session:
        return jsonify({'error': 'Windowed session not found'}), 400

    windows = session['windows']
    labels = session.get('labels')
    categories = session.get('categories')  # For proper train/test split

    if labels is None:
        return jsonify({'error': 'Labels required for classification'}), 400

    try:
        trainer = TimesNetTrainer(device=device)

        # Build config
        num_channels = windows.shape[2] if len(windows.shape) == 3 else 1
        import numpy as np
        num_classes = len(np.unique(labels))

        config = TimesNetConfig(
            seq_len=config_data.get('seq_len', windows.shape[1]),
            enc_in=num_channels,
            c_out=num_classes,
            d_model=config_data.get('d_model', 64),
            d_ff=config_data.get('d_ff', 128),
            num_kernels=config_data.get('num_kernels', 6),
            top_k=config_data.get('top_k', 5),
            e_layers=config_data.get('e_layers', 3),
            dropout=config_data.get('dropout', 0.1),
            period_list=config_data.get('period_list', [8, 16, 32, 64]),
            task_name='classification',
            num_class=num_classes
        )

        result = trainer.train_classification(
            windows=windows,
            labels=labels,
            categories=categories,  # Pass categories for proper splitting
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_size=test_size,
            user_id=request.current_user['id']
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/timesnet/predict', methods=['POST'])
@login_required
def timesnet_predict():
    """Make predictions using a trained TimesNet model."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    training_session_id = data.get('training_session_id')
    windowed_session_id = data.get('windowed_session_id')

    if not training_session_id or not windowed_session_id:
        return jsonify({'error': 'Training and windowed session IDs required'}), 400

    # Get windowed data
    session = _data_sessions.get(windowed_session_id)
    if not session or 'windows' not in session:
        return jsonify({'error': 'Windowed session not found'}), 400

    windows = session['windows']

    try:
        trainer = TimesNetTrainer()
        result = trainer.predict(training_session_id, windows)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/timesnet/metrics/<training_session_id>', methods=['GET'])
@login_required
def get_timesnet_metrics(training_session_id: str):
    """Get metrics for a TimesNet training session."""
    try:
        trainer = TimesNetTrainer()
        result = trainer.get_metrics(training_session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
