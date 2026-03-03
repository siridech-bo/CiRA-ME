"""
CiRA ME - ML Training Routes
Handles Anomaly Detection (PyOD), Classification (Scikit-learn), and Deep Learning (TimesNet)
"""

import math
import logging
import pickle
import numpy as np
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..services.ml_trainer import MLTrainer, _model_sessions
from ..services.timesnet_trainer import TimesNetTrainer, TimesNetConfig
from ..services.feature_extractor import FeatureExtractor
from ..services.data_loader import _data_sessions
from ..services.deployer import load_saved_model_session
from ..config import ANOMALY_ALGORITHMS, CLASSIFICATION_ALGORITHMS
from ..models import SavedModel

logger = logging.getLogger(__name__)


def _sanitize_nan(obj):
    """Replace NaN/Inf float values with None so JSON serialization is valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj
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
        return jsonify(_sanitize_nan(result))
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
        return jsonify(_sanitize_nan(result))
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
        return jsonify(_sanitize_nan(result))
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
        return jsonify(_sanitize_nan(result))
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


@training_bp.route('/export-saved/<int:model_id>', methods=['POST'])
@login_required
def export_saved_model(model_id: int):
    """Export a saved benchmark model."""
    data = request.get_json() or {}
    export_format = data.get('format', 'pickle')

    try:
        saved_model = SavedModel.get_by_id(model_id)
        if not saved_model:
            return jsonify({'error': f'Saved model not found: {model_id}'}), 404

        # Load model from disk into a temporary session
        session = load_saved_model_session(
            saved_model['model_path'],
            algorithm=saved_model['algorithm'],
            mode=saved_model['mode']
        )
        temp_session_id = f"saved_{model_id}"
        _model_sessions[temp_session_id] = session

        # Use the existing export flow
        trainer = MLTrainer()
        result = trainer.export_model(temp_session_id, export_format)
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

        if 'error' in result:
            return jsonify(result), 400

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

        if 'error' in result:
            return jsonify(result), 400

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


# ─── Saved Models (Benchmarks) ─────────────────────────────────────

@training_bp.route('/saved-models', methods=['GET'])
@login_required
def list_saved_models():
    """List all saved benchmark models for the current user."""
    models = SavedModel.get_all(request.current_user['id'])
    return jsonify(models)


@training_bp.route('/save-benchmark', methods=['POST'])
@login_required
def save_benchmark():
    """Save the current best model as a named benchmark."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    training_session_id = data.get('training_session_id')
    name = data.get('name', '').strip()

    if not training_session_id:
        return jsonify({'error': 'training_session_id required'}), 400

    # Get the in-memory session
    session = _model_sessions.get(training_session_id)
    if not session:
        return jsonify({'error': 'Training session not found (may have expired)'}), 404

    if not name:
        algo_name = session.get('algorithm', 'model')
        name = f"{algo_name} - {session.get('created_at', '')[:16]}"

    model_id = SavedModel.save(
        name=name,
        algorithm=session.get('algorithm', ''),
        mode=session.get('mode', ''),
        metrics=session.get('metrics', {}),
        model_path=session.get('model_path', ''),
        training_session_id=training_session_id,
        pipeline_config=data.get('pipeline_config', {}),
        dataset_info=data.get('dataset_info', {}),
        user_id=request.current_user['id']
    )

    return jsonify({'id': model_id, 'name': name, 'message': 'Model saved as benchmark'})


@training_bp.route('/saved-models/<int:model_id>', methods=['DELETE'])
@login_required
def delete_saved_model(model_id):
    """Delete a saved benchmark model."""
    deleted = SavedModel.delete(model_id, request.current_user['id'])
    if deleted:
        return jsonify({'message': 'Model deleted'})
    return jsonify({'error': 'Model not found or access denied'}), 404


@training_bp.route('/saved-models/compare', methods=['POST'])
@login_required
def compare_saved_models():
    """Compare metrics of two saved models side-by-side."""
    data = request.get_json()
    id1 = data.get('model_id_1')
    id2 = data.get('model_id_2')

    if not id1 or not id2:
        return jsonify({'error': 'Two model IDs required'}), 400

    m1 = SavedModel.get_by_id(id1)
    m2 = SavedModel.get_by_id(id2)

    if not m1 or not m2:
        return jsonify({'error': 'One or both models not found'}), 404

    # Build comparison
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    comparison = []
    for key in metric_keys:
        v1 = (m1.get('metrics') or {}).get(key)
        v2 = (m2.get('metrics') or {}).get(key)
        diff = None
        if v1 is not None and v2 is not None:
            diff = round(v2 - v1, 4)
        comparison.append({
            'metric': key,
            'model_1': round(v1, 4) if v1 is not None else None,
            'model_2': round(v2, 4) if v2 is not None else None,
            'diff': diff,
            'better': 'model_2' if diff and diff > 0 else ('model_1' if diff and diff < 0 else 'equal')
        })

    return jsonify({
        'model_1': {'id': m1['id'], 'name': m1['name'], 'algorithm': m1['algorithm'], 'created_at': m1['created_at']},
        'model_2': {'id': m2['id'], 'name': m2['name'], 'algorithm': m2['algorithm'], 'created_at': m2['created_at']},
        'comparison': comparison
    })


# ─── Independent Test Data Evaluation ──────────────────────────────

@training_bp.route('/evaluate', methods=['POST'])
@login_required
def evaluate_on_new_data():
    """Evaluate a saved model on new test data features."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    model_id = data.get('saved_model_id')
    feature_session_id = data.get('feature_session_id')

    if not model_id or not feature_session_id:
        return jsonify({'error': 'saved_model_id and feature_session_id required'}), 400

    saved = SavedModel.get_by_id(model_id)
    if not saved:
        return jsonify({'error': 'Saved model not found'}), 404

    model_path = saved.get('model_path')
    if not model_path or not __import__('os').path.exists(model_path):
        return jsonify({'error': 'Model file not found on disk'}), 404

    try:
        # Load the saved model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        scaler = model_data.get('scaler')

        # Get new test features
        X_new, y_new, _ = FeatureExtractor.get_features_for_training(feature_session_id)

        if y_new is None:
            return jsonify({'error': 'New dataset must have labels for evaluation'}), 400

        # Scale with the original scaler
        if scaler is not None:
            X_new_scaled = scaler.transform(X_new)
        else:
            X_new_scaled = X_new

        # Predict
        y_pred = model.predict(X_new_scaled)

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        new_metrics = {
            'accuracy': float(accuracy_score(y_new, y_pred)),
            'precision': float(precision_score(y_new, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_new, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_new, y_pred, average='weighted', zero_division=0)),
            'test_samples': len(X_new)
        }

        # Build comparison with original metrics
        original_metrics = saved.get('metrics', {})
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        comparison = []
        for key in metric_keys:
            orig = original_metrics.get(key)
            new = new_metrics.get(key)
            diff = None
            if orig is not None and new is not None:
                diff = round(new - orig, 4)
            comparison.append({
                'metric': key,
                'original': round(orig, 4) if orig is not None else None,
                'new_data': round(new, 4) if new is not None else None,
                'diff': diff
            })

        return jsonify(_sanitize_nan({
            'saved_model': {'id': saved['id'], 'name': saved['name'], 'algorithm': saved['algorithm']},
            'original_metrics': original_metrics,
            'new_metrics': new_metrics,
            'comparison': comparison
        }))

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({'error': str(e)}), 400
