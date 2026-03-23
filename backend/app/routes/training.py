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
from ..config import ANOMALY_ALGORITHMS, CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS
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
        'regression': REGRESSION_ALGORITHMS,
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


@training_bp.route('/train/regression', methods=['POST'])
@login_required
def train_regression_model():
    """Train a regression model."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithm = data.get('algorithm', 'rf_reg')
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')
    test_size = data.get('test_size', 0.2)
    target_column = data.get('target_column')

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if algorithm not in REGRESSION_ALGORITHMS:
        return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_regression(
            feature_session_id,
            algorithm,
            hyperparameters,
            test_size=test_size,
            target_column=target_column,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(_sanitize_nan(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@training_bp.route('/train/regression/compare', methods=['POST'])
@login_required
def train_regression_compare():
    """Train multiple regression algorithms and compare."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    algorithms = data.get('algorithms', [])
    hyperparameters = data.get('hyperparameters', {})
    project_id = data.get('project_id')
    test_size = data.get('test_size', 0.2)
    target_column = data.get('target_column')

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    if not algorithms or not isinstance(algorithms, list):
        return jsonify({'error': 'algorithms must be a non-empty list'}), 400

    invalid_algos = [a for a in algorithms if a not in REGRESSION_ALGORITHMS]
    if invalid_algos:
        return jsonify({'error': f'Unknown algorithms: {invalid_algos}'}), 400

    try:
        trainer = MLTrainer()
        result = trainer.train_regression_compare(
            feature_session_id,
            algorithms,
            hyperparameters,
            test_size=test_size,
            target_column=target_column,
            project_id=project_id,
            user_id=request.current_user['id']
        )
        return jsonify(_sanitize_nan(result))
    except Exception as e:
        logger.error(f"Regression comparison training error: {e}")
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

    # Get the in-memory session (check both ML and TimesNet sessions)
    from ..services.timesnet_trainer import _timesnet_sessions
    session = _model_sessions.get(training_session_id) or _timesnet_sessions.get(training_session_id)
    if not session:
        return jsonify({'error': 'Training session not found (may have expired)'}), 404

    if not name:
        algo_name = session.get('algorithm', 'model')
        name = f"{algo_name} - {session.get('created_at', '')[:16]}"

    # Build complete pipeline_config: start from frontend data, augment from backend sessions
    pipeline_config = data.get('pipeline_config', {})

    # Augment normalization from windowed session if not already provided
    windowed_session_id = pipeline_config.get('windowed_session_id')
    if windowed_session_id and not pipeline_config.get('normalization'):
        windowed_session = _data_sessions.get(windowed_session_id)
        if windowed_session and 'metadata' in windowed_session:
            wm = windowed_session['metadata']
            pipeline_config['normalization'] = wm.get('normalization')
            if 'windowing' not in pipeline_config:
                pipeline_config['windowing'] = {
                    'window_size': wm.get('window_size'),
                    'stride': wm.get('stride'),
                    'label_method': wm.get('label_method'),
                    'test_ratio': wm.get('test_ratio'),
                }

    # Augment feature info from feature session if not already provided
    feature_session_id = pipeline_config.get('feature_session_id')
    if feature_session_id and not pipeline_config.get('feature_extraction'):
        from ..services.feature_extractor import _feature_sessions
        feat_session = _feature_sessions.get(feature_session_id)
        if feat_session:
            pipeline_config['feature_extraction'] = {
                'method': feat_session.get('metadata', {}).get('extraction_method', 'lightweight'),
                'feature_set': feat_session.get('metadata', {}).get('feature_set'),
                'feature_names': feat_session.get('feature_names', []),
                'num_features': len(feat_session.get('feature_names', [])),
            }

    # Augment feature selection from selection session if not already provided
    selection_session_id = pipeline_config.get('selection_session_id')
    if selection_session_id and not pipeline_config.get('feature_selection'):
        from ..services.feature_extractor import _selection_sessions
        sel_session = _selection_sessions.get(selection_session_id)
        if sel_session:
            pipeline_config['feature_selection'] = {
                'method': sel_session.get('method'),
                'fdr_level': sel_session.get('fdr_level'),
                'selected_features': sel_session.get('selected_features', []),
                'num_selected': len(sel_session.get('selected_features', [])),
            }

    # Add training info
    pipeline_config['training'] = {
        'algorithm': session.get('algorithm'),
        'hyperparameters': session.get('hyperparameters', {}),
    }

    model_id = SavedModel.save(
        name=name,
        algorithm=session.get('algorithm', ''),
        mode=session.get('mode', ''),
        metrics=session.get('metrics', {}),
        model_path=session.get('model_path', ''),
        training_session_id=training_session_id,
        pipeline_config=pipeline_config,
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


@training_bp.route('/evaluate-raw', methods=['POST'])
@login_required
def evaluate_raw_csv():
    """Evaluate a saved model on a new raw CSV file.
    Automatically replays the full saved pipeline (window → normalize → features → predict).
    Accepts multipart form data: file (CSV) + saved_model_id.
    """
    import os
    import tempfile

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    model_id = request.form.get('saved_model_id')
    if not model_id:
        return jsonify({'error': 'saved_model_id required'}), 400

    saved = SavedModel.get_by_id(int(model_id))
    if not saved:
        return jsonify({'error': 'Saved model not found'}), 404

    pipeline_config = saved.get('pipeline_config', {})
    if not pipeline_config or not pipeline_config.get('normalization'):
        return jsonify({
            'error': 'This model was saved without full pipeline config. '
                     'Re-save the model from a training session to enable raw CSV evaluation.'
        }), 400

    model_path = saved.get('model_path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found on disk'}), 404

    # Save uploaded file to temp location
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        from ..services.pipeline_replay import replay_ml_pipeline, replay_dl_pipeline

        approach = pipeline_config.get('training_approach', 'ml')

        if approach == 'dl':
            result = replay_dl_pipeline(tmp_path, pipeline_config, model_data)
        else:
            result = replay_ml_pipeline(tmp_path, pipeline_config, model_data)

        # Build comparison with original metrics
        original_metrics = saved.get('metrics', {})
        if result.get('has_labels') and result.get('new_metrics'):
            metric_keys = ['accuracy', 'precision', 'recall', 'f1']
            comparison = []
            for key in metric_keys:
                orig = original_metrics.get(key)
                new = result['new_metrics'].get(key)
                diff = round(new - orig, 4) if orig is not None and new is not None else None
                comparison.append({
                    'metric': key,
                    'original': round(orig, 4) if orig is not None else None,
                    'new_data': round(new, 4) if new is not None else None,
                    'diff': diff,
                })
            result['comparison'] = comparison
            result['original_metrics'] = original_metrics

        result['saved_model'] = {
            'id': saved['id'],
            'name': saved['name'],
            'algorithm': saved['algorithm'],
        }

        return jsonify(_sanitize_nan(result))

    except Exception as e:
        logger.error(f"Raw CSV evaluation error: {e}")
        return jsonify({'error': str(e)}), 400
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Custom Model Editor ──────────────────────────────────────────

@training_bp.route('/custom-model/execute', methods=['POST'])
@login_required
def execute_custom_model():
    """Execute user-submitted custom model code."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    code = data.get('code', '')
    feature_session_id = data.get('feature_session_id')
    task = data.get('task', 'classification')
    test_size = data.get('test_size', 0.2)
    user_config = data.get('config', {})
    timeout = min(data.get('timeout', 300), 600)  # Max 10 minutes

    if not code.strip():
        return jsonify({'error': 'No code provided'}), 400

    if not feature_session_id:
        return jsonify({'error': 'Feature session ID required'}), 400

    try:
        from ..services.custom_model_runner import CustomModelRunner
        runner = CustomModelRunner()
        result = runner.execute(
            code=code,
            feature_session_id=feature_session_id,
            task=task,
            test_size=test_size,
            user_config=user_config,
            timeout=timeout,
            user_id=request.current_user['id']
        )

        if result.get('status') == 'success':
            # Store in pipeline for downstream deploy
            if result.get('training_session_id'):
                training_session = {
                    'training_session_id': result['training_session_id'],
                    'algorithm': 'custom',
                    'mode': task,
                    'metrics': result.get('metrics', {}),
                }

            return jsonify(_sanitize_nan(result))
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Custom model execution error: {e}")
        return jsonify({'error': str(e)}), 400


@training_bp.route('/custom-model/templates', methods=['GET'])
@login_required
def get_custom_model_templates():
    """Get available custom model templates."""
    return jsonify(CUSTOM_MODEL_TEMPLATES)


# Template library
CUSTOM_MODEL_TEMPLATES = [
    {
        'id': 'sklearn_classifier',
        'name': 'Sklearn Classifier',
        'description': 'Classification with scikit-learn (Random Forest, SVM, etc.)',
        'task': 'classification',
        'code': '''from cira_base import CiraModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MyClassifier(CiraModel):
    def build(self, config):
        self.model = GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 5),
            learning_rate=config.get('learning_rate', 0.05),
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
'''
    },
    {
        'id': 'sklearn_regressor',
        'name': 'Sklearn Regressor',
        'description': 'Regression with scikit-learn (SVR, ElasticNet, etc.)',
        'task': 'regression',
        'code': '''from cira_base import CiraModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

class MyRegressor(CiraModel):
    def build(self, config):
        self.model = GradientBoostingRegressor(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 5),
            learning_rate=config.get('learning_rate', 0.05),
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        return {
            "r2": r2_score(y_val, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        }

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
'''
    },
    {
        'id': 'pytorch_mlp',
        'name': 'PyTorch MLP',
        'description': 'Custom neural network with PyTorch',
        'task': 'classification',
        'code': '''from cira_base import CiraModel
import numpy as np

class MyNeuralNet(CiraModel):
    def build(self, config):
        import torch
        import torch.nn as nn

        self.epochs = config.get('epochs', 100)
        self.lr = config.get('learning_rate', 1e-3)

        self.net = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.n_classes)
        )

    def train(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        X_t = torch.FloatTensor(X_train)
        y_t = torch.LongTensor(y_train.astype(int))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.net(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        # Validation
        with torch.no_grad():
            preds = self.net(torch.FloatTensor(X_val)).argmax(dim=1).numpy()

        acc = (preds == y_val.astype(int)).mean()
        return {"accuracy": float(acc)}

    def predict(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.FloatTensor(X)).argmax(dim=1).numpy()

    def get_model(self):
        return self.net
'''
    },
    {
        'id': 'xgboost_model',
        'name': 'XGBoost',
        'description': 'XGBoost gradient boosting (classification or regression)',
        'task': 'classification',
        'code': '''from cira_base import CiraModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MyXGBoost(CiraModel):
    def build(self, config):
        self.model = XGBClassifier(
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
        y_pred = self.model.predict(X_val)
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
'''
    },
    {
        'id': 'ensemble_model',
        'name': 'Voting Ensemble',
        'description': 'Combine multiple models via voting/averaging',
        'task': 'classification',
        'code': '''from cira_base import CiraModel
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MyEnsemble(CiraModel):
    def build(self, config):
        self.model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
            ],
            voting='soft'
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
'''
    },
    {
        'id': 'anomaly_autoencoder',
        'name': 'Custom AutoEncoder',
        'description': 'PyTorch autoencoder for anomaly detection via reconstruction error',
        'task': 'anomaly',
        'code': '''from cira_base import CiraModel
import numpy as np

class MyAutoEncoder(CiraModel):
    def build(self, config):
        import torch
        import torch.nn as nn

        self.epochs = config.get('epochs', 100)
        self.lr = config.get('learning_rate', 1e-3)
        self.threshold_percentile = config.get('threshold_percentile', 95)

        hidden = config.get('hidden_dim', 32)
        latent = config.get('latent_dim', 8)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_features),
        )
        self.threshold = 0.0

    def train(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        X_t = torch.FloatTensor(X_train)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            encoded = self.encoder(X_t)
            decoded = self.decoder(encoded)
            loss = loss_fn(decoded, X_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Reconstruction Loss: {loss.item():.6f}")

        # Set threshold from training data reconstruction errors
        with torch.no_grad():
            recon = self.decoder(self.encoder(X_t))
            errors = ((recon - X_t) ** 2).mean(dim=1).numpy()
            self.threshold = float(np.percentile(errors, self.threshold_percentile))

        # Evaluate on validation
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val)
            recon_v = self.decoder(self.encoder(X_v))
            val_errors = ((recon_v - X_v) ** 2).mean(dim=1).numpy()
            anomalies = (val_errors > self.threshold).sum()

        return {
            "reconstruction_loss": float(loss.item()),
            "threshold": self.threshold,
            "val_anomalies": int(anomalies),
            "val_total": len(X_val),
        }

    def predict(self, X):
        import torch
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            recon = self.decoder(self.encoder(X_t))
            errors = ((recon - X_t) ** 2).mean(dim=1).numpy()
            return (errors > self.threshold).astype(int)

    def get_model(self):
        return {"encoder": self.encoder, "decoder": self.decoder, "threshold": self.threshold}
'''
    },
]
