"""
CiRA ME - Feature Engineering Routes
Handles TSFresh and Custom DSP feature extraction
"""

from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..services.feature_extractor import FeatureExtractor
from ..services.llm_service import get_llm_service
from ..services.data_loader import _data_sessions
from ..config import Config

features_bp = Blueprint('features', __name__)


@features_bp.route('/available', methods=['GET'])
@login_required
def get_available_features():
    """Get list of all available features."""
    config = Config()

    return jsonify({
        'tsfresh_features': config.TSFRESH_FEATURES,
        'dsp_features': config.CUSTOM_DSP_FEATURES,
        'total_features': len(config.TSFRESH_FEATURES) + len(config.CUSTOM_DSP_FEATURES)
    })


@features_bp.route('/llm-status', methods=['GET'])
@login_required
def get_llm_status():
    """Get LLM service status including GPU information."""
    try:
        llm_service = get_llm_service()
        status = llm_service.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })


@features_bp.route('/extract', methods=['POST'])
@login_required
def extract_features():
    """Extract features from windowed data."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    selected_features = data.get('features', None)  # None means all features
    include_tsfresh = data.get('include_tsfresh', True)
    include_dsp = data.get('include_dsp', True)

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.extract(
            session_id,
            selected_features=selected_features,
            include_tsfresh=include_tsfresh,
            include_dsp=include_dsp
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/recommend', methods=['POST'])
@login_required
def recommend_features():
    """
    Get LLM-powered feature recommendations based on data characteristics.
    Uses local Llama 3.2 if available, falls back to rule-based recommendations.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    mode = data.get('mode', 'anomaly')  # 'anomaly' or 'classification'
    use_llm = data.get('use_llm', True)  # Can disable LLM if needed

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        # Get session data for context
        session = _data_sessions.get(session_id)
        if not session:
            return jsonify({'error': f'Session not found: {session_id}'}), 400

        # Prepare data statistics for LLM
        metadata = session.get('metadata', {})
        data_stats = {
            'num_windows': len(session.get('windows', [])),
            'window_size': metadata.get('window_size', 128),
            'num_channels': len(metadata.get('sensor_columns', [])),
            'sampling_rate': metadata.get('sampling_rate', 100),
        }

        # Get label distribution if available
        labels = session.get('labels')
        if labels is not None:
            import numpy as np
            unique, counts = np.unique(labels, return_counts=True)
            data_stats['label_distribution'] = dict(zip([str(l) for l in unique], counts.tolist()))

        # Build available features list
        config = Config()
        available_features = config.TSFRESH_FEATURES + config.CUSTOM_DSP_FEATURES

        # Try LLM-powered recommendations
        if use_llm:
            llm_service = get_llm_service()
            status = llm_service.get_status()

            if status.get('available') and status.get('model_installed'):
                result = llm_service.recommend_features(
                    data_stats=data_stats,
                    available_features=available_features,
                    mode=mode,
                    sensor_info=metadata.get('sensor_info')
                )
                result['llm_status'] = {
                    'model': status.get('model'),
                    'gpu_loaded': status.get('gpu_loaded', False),
                    'gpu_info': status.get('gpu_info')
                }
                return jsonify(result)

        # Fallback to rule-based recommendations
        extractor = FeatureExtractor()
        result = extractor.recommend_features(session_id, mode)
        result['llm_used'] = False
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/importance', methods=['POST'])
@login_required
def get_feature_importance():
    """Get feature importance scores after training."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    training_session_id = data.get('training_session_id')

    if not session_id or not training_session_id:
        return jsonify({'error': 'Session ID and training session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.get_importance(session_id, training_session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/preview', methods=['POST'])
@login_required
def get_feature_preview():
    """Get feature preview data for visualization."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    num_rows = data.get('num_rows', 100)

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.get_feature_preview(session_id, num_rows)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/distribution', methods=['POST'])
@login_required
def get_feature_distribution():
    """Get distribution data for a specific feature."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    feature_name = data.get('feature_name')
    bins = data.get('bins', 20)

    if not session_id or not feature_name:
        return jsonify({'error': 'Session ID and feature name required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.get_feature_distribution(session_id, feature_name, bins)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
