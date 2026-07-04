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
    def _is_timesnet(model_data: Dict) -> bool:
        """A TimesNet pickle has a torch state_dict, not a fitted estimator."""
        return 'model_state' in model_data and 'model' not in model_data

    @staticmethod
    def _predict_timesnet(model_data: Dict, windows: np.ndarray, mode: str) -> List[Dict]:
        """Inference for TimesNet (DL) models via the torch subprocess.

        The subprocess pattern is reused from pipeline_replay so we get
        consistent behavior across App Builder / ME-LAB / Wizard for
        classification and anomaly. `windows` is a 3D tensor of shape
        (n_windows, window_size, n_channels) — already windowed and
        normalized by the pipeline runner upstream.

        Note: subprocess spawn adds ~1-3s of cold-start latency per call.
        For live MQTT streams that arrive faster than the subprocess can
        return, the caller should batch windows before invoking.
        """
        import tempfile
        from .pipeline_replay import _run_timesnet_subprocess

        if not isinstance(windows, np.ndarray):
            windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:
            # Single window arriving as (window_size, n_channels) — batch it.
            windows = windows[np.newaxis, ...]

        # Dump model_data to a temp pickle so the subprocess reads torch
        # tensors directly (avoids JSON serialization of state_dict).
        tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='_tn.pkl', delete=False)
        try:
            pickle.dump(model_data, tmp)
            tmp_path = tmp.name
        finally:
            tmp.close()

        config_dict = model_data.get('config', {}) or {}
        if mode == 'anomaly':
            task = 'predict_anomaly'
            data_payload = {
                'windows': windows.tolist(),
                'labels': None,
                'threshold': float(model_data.get('threshold', 0)),
                'model_data_path': tmp_path,
            }
        else:
            task = 'predict_classification'
            data_payload = {
                'windows': windows.tolist(),
                'labels': None,
                'label_classes': model_data.get('label_encoder_classes', []),
                'model_data_path': tmp_path,
            }

        try:
            result = _run_timesnet_subprocess(task, config_dict, data_payload)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not result.get('success'):
            raise RuntimeError(result.get('error') or 'TimesNet subprocess failed')

        preds = result.get('predictions') or []
        confs = result.get('confidences') or []
        probs_all = result.get('probabilities') or []
        label_classes = model_data.get('label_encoder_classes') or []

        out: List[Dict] = []
        for i, p in enumerate(preds):
            entry: Dict[str, Any] = {
                'label': str(p),
                'prediction': str(p),
            }
            if i < len(confs):
                entry['confidence'] = float(confs[i])
            if i < len(probs_all) and label_classes:
                entry['probabilities'] = {
                    str(label_classes[j]): float(probs_all[i][j])
                    for j in range(min(len(label_classes), len(probs_all[i])))
                }
            if mode == 'anomaly':
                # Anomaly subprocess emits 0/1; normalize to normal/anomaly label.
                is_anom = int(p) == 1
                entry['label'] = 'anomaly' if is_anom else 'normal'
                entry['prediction'] = int(p)
                if i < len(confs):
                    # Reconstruction-error-derived confidence isn't calibrated
                    # the same way as softmax — expose it as `score` too.
                    entry['score'] = float(confs[i])
            out.append(entry)
        return out

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
        # TimesNet (DL) models don't fit the sklearn interface — dispatch
        # to the subprocess-based inference path.
        if ModelManager._is_timesnet(model_data):
            return ModelManager._predict_timesnet(model_data, features, mode)

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
    def predict_by_endpoint(endpoint_id, features: np.ndarray) -> List[Dict]:
        """Canonical endpoint-based inference.

        Loads endpoint + saved model, runs ModelManager.predict, applies label
        decoding (label_inverse_map -> classes_ -> class_names), and increments
        the endpoint's inference counter.

        Args:
            endpoint_id: str or int endpoint id.
            features: numpy array of shape (n_samples, n_features). 1D arrays
                are reshaped to (1, n).

        Returns:
            List of prediction dicts (same shape as ModelManager.predict).

        Raises:
            RuntimeError with an actionable message on any failure.
        """
        # Local imports keep melab_service free of Flask/app-level cycles.
        from ..models import MeLabEndpoint, SavedModel

        endpoint = MeLabEndpoint.get_by_id(endpoint_id)
        if not endpoint:
            raise RuntimeError(f"Endpoint {endpoint_id} not found")
        if endpoint.get('status') != 'active':
            raise RuntimeError(
                f"Endpoint {endpoint_id} is {endpoint.get('status')}, not active"
            )

        saved = SavedModel.get_by_id(endpoint['saved_model_id'])
        if not saved or not saved.get('model_path'):
            raise RuntimeError(
                f"Model file missing for endpoint {endpoint_id}"
            )

        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            model_data = ModelManager.load_model(saved['model_path'])
        except FileNotFoundError as e:
            raise RuntimeError(str(e))

        mode = endpoint.get('mode', 'classification')
        predictions = ModelManager.predict(model_data, features, mode)

        # Decode integer labels for classification/anomaly (byte-identical to
        # the previous _run_model_inference implementation in app_builder.py).
        if mode != 'regression' and predictions:
            model_obj = model_data.get('model')

            decode_map = None

            # Priority 1: label_inverse_map (exact encoder mapping from training)
            inv_map = model_data.get('label_inverse_map')
            if inv_map:
                decode_map = {int(k): str(v) for k, v in inv_map.items()}
            # Priority 2: model.classes_ with string values (sklearn native)
            elif hasattr(model_obj, 'classes_') and len(model_obj.classes_) > 0:
                if isinstance(model_obj.classes_[0], (str, np.str_)):
                    decode_map = None  # Predictions are already strings
            # Priority 3: class_names (best-effort for old models)
            if not decode_map and not inv_map and model_data.get('class_names'):
                cn = model_data['class_names']
                decode_map = {i: cn[i] for i in range(len(cn))}

            if decode_map:
                for p in predictions:
                    label = p.get('label', '')
                    try:
                        idx = int(float(label))
                        if idx in decode_map:
                            p['label'] = decode_map[idx]
                            p['prediction'] = decode_map[idx]
                    except (ValueError, TypeError):
                        pass

        # Record the inference (matches previous behavior).
        try:
            MeLabEndpoint.record_inference(endpoint_id)
        except Exception:
            # Counter update is best-effort; don't fail the prediction.
            pass

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
