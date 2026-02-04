"""
CiRA ME - ML Training Routes
Handles Anomaly Detection (PyOD), Classification (Scikit-learn), and Deep Learning (TimesNet)
"""

from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..services.ml_trainer import MLTrainer
from ..services.timesnet_trainer import TimesNetTrainer, TimesNetConfig
from ..services.data_loader import _data_sessions
from ..config import ANOMALY_ALGORITHMS, CLASSIFICATION_ALGORITHMS

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

    if not windowed_session_id:
        return jsonify({'error': 'Windowed session ID required'}), 400

    # Get windowed data
    session = _data_sessions.get(windowed_session_id)
    if not session or 'windows' not in session:
        return jsonify({'error': 'Windowed session not found'}), 400

    windows = session['windows']
    labels = session.get('labels')

    try:
        trainer = TimesNetTrainer()

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

    if not windowed_session_id:
        return jsonify({'error': 'Windowed session ID required'}), 400

    # Get windowed data
    session = _data_sessions.get(windowed_session_id)
    if not session or 'windows' not in session:
        return jsonify({'error': 'Windowed session not found'}), 400

    windows = session['windows']
    labels = session.get('labels')

    if labels is None:
        return jsonify({'error': 'Labels required for classification'}), 400

    try:
        trainer = TimesNetTrainer()

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
