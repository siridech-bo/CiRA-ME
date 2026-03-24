"""
CiRA ME - ME-LAB Model Manager Service
Handles model loading, caching, and inference for ME-LAB endpoints.
"""

import os
import time
import pickle
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional, List
from collections import OrderedDict

logger = logging.getLogger(__name__)

# LRU model cache
_model_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
_MAX_CACHED_MODELS = 50


class ModelManager:
    """Manages model loading and inference for ME-LAB endpoints."""

    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """Load a model from disk, with LRU caching."""
        with _cache_lock:
            if model_path in _model_cache:
                # Move to end (most recently used)
                _model_cache.move_to_end(model_path)
                return _model_cache[model_path]

        # Load from disk
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Warm up: run a dummy prediction to trigger any lazy init
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        if model and hasattr(model, 'predict'):
            try:
                n_features = len(model_data.get('feature_names', []))
                if n_features > 0:
                    dummy = np.zeros((1, n_features))
                    if scaler:
                        dummy = scaler.transform(dummy)
                    model.predict(dummy)
            except Exception:
                pass  # Warm-up failure is not critical

        with _cache_lock:
            _model_cache[model_path] = model_data
            _model_cache.move_to_end(model_path)
            # Evict oldest if cache full
            while len(_model_cache) > _MAX_CACHED_MODELS:
                evicted_path, _ = _model_cache.popitem(last=False)
                logger.info(f"[ME-LAB] Evicted model from cache: {evicted_path}")

        logger.info(f"[ME-LAB] Loaded model: {model_path} (cache size: {len(_model_cache)})")
        return model_data

    @staticmethod
    def unload_model(model_path: str):
        """Remove a model from cache."""
        with _cache_lock:
            if model_path in _model_cache:
                del _model_cache[model_path]

    @staticmethod
    def predict(model_data: Dict, features: np.ndarray, mode: str) -> List[Dict]:
        """Run inference on feature vectors.

        Args:
            model_data: Loaded model dict (model, scaler, etc.)
            features: numpy array of shape (n_samples, n_features)
            mode: 'anomaly', 'classification', or 'regression'

        Returns:
            List of prediction dicts
        """
        model = model_data['model']
        scaler = model_data.get('scaler')

        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        predictions = []

        if mode == 'regression':
            y_pred = model.predict(features_scaled)
            for val in y_pred:
                predictions.append({'value': float(val)})

        elif mode == 'anomaly':
            y_pred = model.predict(features_scaled)
            scores = None
            if hasattr(model, 'decision_function'):
                try:
                    scores = model.decision_function(features_scaled)
                except Exception:
                    pass

            for i, pred in enumerate(y_pred):
                result = {
                    'label': 'anomaly' if pred == 1 else 'normal',
                    'prediction': int(pred),
                }
                if scores is not None:
                    result['score'] = float(scores[i])
                predictions.append(result)

        else:  # classification
            y_pred = model.predict(features_scaled)
            probas = None
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(features_scaled)
                except Exception:
                    pass

            classes = model_data.get('classes', [])
            for i, pred in enumerate(y_pred):
                result = {
                    'label': str(pred),
                    'prediction': str(pred),
                }
                if probas is not None:
                    result['confidence'] = float(np.max(probas[i]))
                    if classes:
                        result['probabilities'] = {
                            str(c): float(p) for c, p in zip(classes, probas[i])
                        }
                predictions.append(result)

        return predictions

    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """Get cache statistics."""
        with _cache_lock:
            return {
                'cached_models': len(_model_cache),
                'max_cache_size': _MAX_CACHED_MODELS,
                'cached_paths': list(_model_cache.keys()),
            }
