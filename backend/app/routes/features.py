"""
CiRA ME - Feature Engineering Routes
Handles TSFresh and Custom DSP feature extraction
"""

import logging
from flask import Blueprint, request, jsonify
from ..auth import login_required
from ..services.feature_extractor import FeatureExtractor
from ..services.llm_service import get_llm_service
from ..services.data_loader import _data_sessions
from ..config import Config

logger = logging.getLogger(__name__)
features_bp = Blueprint('features', __name__)


@features_bp.route('/available', methods=['GET'])
@login_required
def get_available_features():
    """Get list of all available features and extraction methods."""
    config = Config()

    # Get available feature sets info
    extractor = FeatureExtractor()
    feature_sets = extractor.get_available_feature_sets()

    return jsonify({
        'tsfresh_features': config.TSFRESH_FEATURES,
        'dsp_features': config.CUSTOM_DSP_FEATURES,
        'total_lightweight_features': len(config.TSFRESH_FEATURES) + len(config.CUSTOM_DSP_FEATURES),
        **feature_sets
    })


@features_bp.route('/extract-tsfresh', methods=['POST'])
@login_required
def extract_tsfresh_features():
    """
    Extract features using the REAL tsfresh library.

    Provides comprehensive feature extraction with 800+ features including:
    - Statistical features
    - Autocorrelation at multiple lags
    - FFT coefficients
    - AR model coefficients
    - Wavelet coefficients
    - And many more...

    Request body:
    {
        "session_id": "windowed_session_id",
        "feature_set": "efficient",  // "minimal", "efficient", "comprehensive"
        "n_jobs": 1  // Number of parallel jobs
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    feature_set = data.get('feature_set', 'efficient')
    n_jobs = data.get('n_jobs', 1)

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    if feature_set not in ['minimal', 'efficient', 'comprehensive']:
        return jsonify({'error': 'feature_set must be: minimal, efficient, or comprehensive'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.extract_tsfresh(
            session_id,
            feature_set=feature_set,
            n_jobs=n_jobs
        )
        return jsonify(result)
    except ImportError as e:
        return jsonify({
            'error': str(e),
            'suggestion': 'Use /extract endpoint for lightweight features instead'
        }), 400
    except Exception as e:
        logger.error(f"tsfresh extraction error: {e}")
        return jsonify({'error': str(e)}), 400


@features_bp.route('/select-fresh', methods=['POST'])
@login_required
def select_features_fresh():
    """
    Select features using tsfresh's FRESH algorithm.

    FRESH = FeatuRe Extraction based on Scalable Hypothesis tests

    Uses hypothesis testing with Benjamini-Hochberg FDR correction to select
    statistically significant features.

    Request body:
    {
        "session_id": "feature_session_id",
        "fdr_level": 0.05,  // False Discovery Rate threshold
        "multiclass": true,
        "ml_task": "classification"  // or "regression"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    fdr_level = data.get('fdr_level', 0.05)
    multiclass = data.get('multiclass', True)
    n_significant = data.get('n_significant', 1)
    ml_task = data.get('ml_task', 'classification')

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.select_features_fresh(
            session_id,
            fdr_level=fdr_level,
            multiclass=multiclass,
            n_significant=n_significant,
            ml_task=ml_task
        )
        return jsonify(result)
    except ImportError as e:
        return jsonify({
            'error': str(e),
            'suggestion': 'Use /select endpoint for sklearn-based selection instead'
        }), 400
    except Exception as e:
        logger.error(f"FRESH selection error: {e}")
        return jsonify({'error': str(e)}), 400


@features_bp.route('/select-fresh-combined', methods=['POST'])
@login_required
def select_features_fresh_combined():
    """
    Chained feature selection: FRESH + target count.

    First applies FRESH algorithm to get statistically significant features,
    then reduces to target count using mutual information ranking.

    Request body:
    {
        "session_id": "feature_session_id",
        "fdr_level": 0.05,
        "n_features": 20,
        "ml_task": "classification"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    fdr_level = data.get('fdr_level', 0.05)
    n_features = data.get('n_features', 20)
    ml_task = data.get('ml_task', 'classification')

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.select_features_fresh_combined(
            session_id,
            fdr_level=fdr_level,
            n_features=n_features,
            ml_task=ml_task
        )
        return jsonify(result)
    except ImportError as e:
        return jsonify({
            'error': str(e),
            'suggestion': 'Use /select endpoint for sklearn-based selection instead'
        }), 400
    except Exception as e:
        logger.error(f"FRESH combined selection error: {e}")
        return jsonify({'error': str(e)}), 400


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


@features_bp.route('/select', methods=['POST'])
@login_required
def select_features():
    """
    Intelligent feature selection from extracted features.

    Methods:
    - 'variance': Remove low-variance features
    - 'correlation': Remove highly correlated features
    - 'mutual_info': Select by mutual information with labels
    - 'anova': Select by ANOVA F-score
    - 'combined': Apply all methods in sequence (recommended)
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    method = data.get('method', 'combined')
    n_features = data.get('n_features', 15)
    variance_threshold = data.get('variance_threshold', 0.01)
    correlation_threshold = data.get('correlation_threshold', 0.95)

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.select_features(
            session_id,
            method=method,
            n_features=n_features,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/correlations', methods=['POST'])
@login_required
def get_correlations():
    """Get feature correlation matrix for visualization."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    features = data.get('features')  # Optional: specific features to include

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.get_feature_correlations(session_id, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/apply-selection', methods=['POST'])
@login_required
def apply_feature_selection():
    """Apply feature selection and create a reduced feature session."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    session_id = data.get('session_id')
    selected_features = data.get('selected_features', [])

    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400

    if not selected_features:
        return jsonify({'error': 'Selected features list required'}), 400

    try:
        extractor = FeatureExtractor()
        result = extractor.apply_selection(session_id, selected_features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@features_bp.route('/llm-select', methods=['POST'])
@login_required
def llm_feature_selection():
    """
    LLM-powered intelligent feature selection.

    Analyzes extracted features and uses Llama 3.2 to recommend the optimal
    subset based on data characteristics and task type.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('session_id')
    mode = data.get('mode', 'classification')  # 'anomaly' or 'classification'
    n_features = data.get('n_features', 15)

    if not feature_session_id:
        return jsonify({'error': 'Session ID required'}), 400

    try:
        extractor = FeatureExtractor()

        # First, run statistical selection
        stat_result = extractor.select_features(
            feature_session_id,
            method='combined',
            n_features=n_features * 2  # Get more candidates for LLM to filter
        )

        # Get feature statistics for LLM context
        preview = extractor.get_feature_preview(feature_session_id, 50)

        # Try LLM-powered refinement
        llm_service = get_llm_service()
        status = llm_service.get_status()

        logger.info(f"LLM Status: {status}")
        print(f"[DEBUG] LLM Status: {status}")  # Console output for debugging

        if status.get('available') and status.get('model_installed'):
            # Build context for LLM
            feature_info = []
            for feat in stat_result['selected_features'][:20]:  # Top 20 candidates
                score = stat_result['importance_scores'].get(feat, 0)
                feat_stats = preview.get('feature_stats', {}).get(feat, {})
                feature_info.append({
                    'name': feat,
                    'importance': round(score, 4),
                    'mean': round(feat_stats.get('mean', 0), 4),
                    'std': round(feat_stats.get('std', 0), 4)
                })

            # Get LLM recommendation
            llm_result = _llm_select_features(
                llm_service,
                feature_info,
                mode,
                n_features,
                preview.get('label_counts', {})
            )

            if llm_result['success']:
                return jsonify({
                    'session_id': stat_result['session_id'],
                    'selected_features': llm_result['selected_features'],
                    'importance_scores': {
                        f: stat_result['importance_scores'].get(f, 0)
                        for f in llm_result['selected_features']
                    },
                    'reasoning': llm_result['reasoning'],
                    'llm_used': True,
                    'llm_model': status.get('model'),
                    'statistical_candidates': stat_result['selected_features'],
                    'selection_log': stat_result['selection_log'] + ['LLM refinement applied'],
                    'original_count': stat_result['original_count'],
                    'final_count': len(llm_result['selected_features'])
                })

        # Fallback to statistical selection
        logger.info("LLM not available, falling back to statistical selection")
        print(f"[DEBUG] LLM not available or model not installed. available={status.get('available')}, model_installed={status.get('model_installed')}")
        return jsonify({
            **stat_result,
            'llm_used': False,
            'reasoning': [
                'Statistical feature selection applied',
                f"Removed {stat_result['original_count'] - stat_result['final_count']} redundant features",
                'Features ranked by combined importance score'
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


def _llm_select_features(
    llm_service,
    feature_info: list,
    mode: str,
    n_features: int,
    label_counts: dict
) -> dict:
    """Helper function for LLM feature selection."""
    import json

    system_prompt = """You are an expert in signal processing and machine learning feature selection.
Your task is to select the optimal subset of features for the given ML task.

Consider:
1. Feature importance scores (higher is better)
2. Feature diversity (avoid selecting similar features)
3. Task requirements (anomaly detection vs classification)
4. Computational efficiency

Respond ONLY with valid JSON in this exact format:
{
  "selected_features": ["feature1", "feature2", ...],
  "reasoning": ["reason1", "reason2", "reason3"]
}"""

    features_text = "\n".join([
        f"- {f['name']}: importance={f['importance']}, mean={f['mean']}, std={f['std']}"
        for f in feature_info
    ])

    user_prompt = f"""Task: {mode.upper()}
Label distribution: {label_counts}
Target features: {n_features}

Candidate features (ranked by importance):
{features_text}

Select exactly {n_features} features that provide the best coverage for {mode}.
Prioritize:
- High importance scores
- Diverse feature types (mix of time-domain and frequency-domain)
- Features that complement each other

Respond with JSON only."""

    result = llm_service.generate(user_prompt, system_prompt)

    if not result['success']:
        return {'success': False}

    try:
        response_text = result['response']
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        parsed = json.loads(response_text.strip())
        selected = parsed.get('selected_features', [])

        # Validate features exist
        valid_features = [f['name'] for f in feature_info]
        selected = [f for f in selected if f in valid_features][:n_features]

        if not selected:
            return {'success': False}

        return {
            'success': True,
            'selected_features': selected,
            'reasoning': parsed.get('reasoning', [])
        }

    except Exception:
        return {'success': False}
