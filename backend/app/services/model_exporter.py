"""
CiRA ME — Model Exporter (Phase I Q4)
=====================================

Convert a trained sklearn / XGBoost / LightGBM model to ONNX so the browser
can run inference client-side via `onnxruntime-web`.

The `export_to_onnx()` entry point is designed to be called at DEPLOY TIME
(i.e. when a benchmark model is being packaged for a published App Builder
app). It NEVER RAISES — a None return means "client-side inference is not
supported for this model" and the deploy pipeline proceeds unchanged (server
round-trip inference still works).

Design rules (locked in the Phase I plan):
  * Dispatch by isinstance — one branch per supported model family
  * Parity check: predict the first 100 rows via BOTH the original model and
    the ONNX runtime, and require exact match (classifiers) or 1e-3 max abs
    error (regressors) before the ONNX is considered valid.
  * On any failure — unsupported type, converter exception, parity mismatch,
    file-write error — delete any half-written .onnx, log at WARNING, and
    return None.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Parity thresholds ────────────────────────────────────────────────
# Classifier: outputs are class labels — MUST match exactly on the 100 test
# rows or we treat the ONNX as broken. Small numerical drift in a classifier
# would silently flip a prediction near the decision boundary.
_CLASSIFIER_TOL = 0  # exact match required

# Regressor: max abs error of 1e-3 in normalized feature space is the ceiling
# allowed by Phase I plan. Anything looser risks user-visible discrepancy.
_REGRESSOR_TOL = 1e-3


# ── Public API ──────────────────────────────────────────────────────

def export_to_onnx(
    model: Any,
    feature_shape: int,
    output_path: str,
    x_sample: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Convert `model` to ONNX and write it to `output_path`.

    Args:
        model:         Trained sklearn / XGBoost / LightGBM estimator.
        feature_shape: Number of input features the model expects (int).
        output_path:   Where to write the .onnx bytes.
        x_sample:      Optional (N, feature_shape) float array used for the
                       parity check. If omitted, we synthesize a random matrix
                       from a fixed seed — the check then only verifies the
                       ONNX runtime can execute without error, not that
                       predictions match on real data. Prefer passing real
                       training rows when available.

    Returns:
        On success: metadata dict with keys
          {path, model_type, opset, feature_shape, parity_rows, parity_kind}
        On any failure: None. Deploy pipeline treats this as
        "client inference not available for this model".
    """
    # Never raise — every branch below must either return a dict or None.
    try:
        model_type = _classify_model(model)
        if model_type is None:
            logger.info(
                "[model_exporter] Skipping ONNX export — model type "
                "%r not in the client-inference dispatch list.",
                type(model).__name__,
            )
            return None

        # Delegate to family-specific converter. Each converter returns the
        # serialized onnx.ModelProto (or None on unsupported).
        onnx_model, opset = _convert(model, model_type, int(feature_shape))
        if onnx_model is None:
            logger.info(
                "[model_exporter] Converter for %r returned no model — "
                "unsupported subvariant. Skipping.", model_type,
            )
            return None

        # Write .onnx bytes to disk first — parity check LOADS it back via
        # onnxruntime to exercise the same code path the browser will use.
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        # Parity check — synthesize sample if the caller didn't provide one.
        if x_sample is None:
            rng = np.random.default_rng(42)
            x_sample = rng.standard_normal((100, int(feature_shape))).astype(np.float32)
        else:
            x_sample = np.asarray(x_sample, dtype=np.float32)
            if x_sample.ndim == 1:
                x_sample = x_sample.reshape(1, -1)
            # Cap at 100 rows — Phase I plan's contract.
            if x_sample.shape[0] > 100:
                x_sample = x_sample[:100]
            if x_sample.shape[1] != int(feature_shape):
                logger.warning(
                    "[model_exporter] Sample feature count %d does not match "
                    "declared feature_shape %d — trimming/padding sample "
                    "would produce garbage predictions. Falling back to "
                    "random sample for parity check.",
                    x_sample.shape[1], int(feature_shape),
                )
                rng = np.random.default_rng(42)
                x_sample = rng.standard_normal((100, int(feature_shape))).astype(np.float32)

        parity_kind = _parity_kind_for(model, model_type)
        parity_ok, parity_diag = _parity_check(
            model, output_path, x_sample, parity_kind
        )
        if not parity_ok:
            # Delete the half-broken .onnx so the deploy pipeline doesn't
            # advertise a supported flag for a file that will silently
            # produce wrong predictions in the browser.
            try:
                os.remove(output_path)
            except OSError:
                pass
            logger.warning(
                "[model_exporter] Parity check FAILED for %r (%s): %s — "
                "deleted %s. Client inference will not be offered.",
                model_type, parity_kind, parity_diag, output_path,
            )
            return None

        logger.info(
            "[model_exporter] ONNX export OK — %s (%s parity on %d rows) → %s",
            model_type, parity_kind, x_sample.shape[0], output_path,
        )
        return {
            'path': output_path,
            'model_type': model_type,
            'opset': opset,
            'feature_shape': int(feature_shape),
            'parity_rows': int(x_sample.shape[0]),
            'parity_kind': parity_kind,
        }

    except Exception as exc:
        # Blanket guard — spec says export_to_onnx MUST NEVER raise. Log with
        # exception info so the failure is diagnosable from the backend log.
        logger.warning(
            "[model_exporter] ONNX export failed for %r: %s",
            type(model).__name__, exc, exc_info=True,
        )
        # Best-effort cleanup — don't leave a partial file behind.
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass
        return None


# ── Dispatch ────────────────────────────────────────────────────────

def _classify_model(model: Any) -> Optional[str]:
    """Return a short model-type key or None if unsupported.

    Order matters: XGBoost's Booster inherits from BaseEstimator when using
    the sklearn API, but we want to dispatch it through onnxmltools, not
    skl2onnx.
    """
    # XGBoost — checked first because XGBClassifier/XGBRegressor also expose
    # sklearn's BaseEstimator interface.
    try:
        import xgboost as _xgb  # type: ignore
        if isinstance(model, (_xgb.XGBClassifier, _xgb.XGBRegressor)):
            return 'xgboost'
    except Exception:
        pass

    # LightGBM — same story (subclasses of sklearn base).
    try:
        import lightgbm as _lgb  # type: ignore
        if isinstance(model, (_lgb.LGBMClassifier, _lgb.LGBMRegressor)):
            return 'lightgbm'
    except Exception:
        pass

    # sklearn families the Phase I plan calls out explicitly. Everything else
    # returns None so the deploy pipeline records
    # `client_inference_supported=False`.
    try:
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    except Exception:
        return None

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return 'sklearn_rf'
    if isinstance(model, (GradientBoostingClassifier, GradientBoostingRegressor)):
        return 'sklearn_gb'
    if isinstance(model, LogisticRegression):
        return 'sklearn_logreg'
    if isinstance(model, LinearRegression):
        return 'sklearn_linreg'
    if isinstance(model, (SVC, SVR)):
        return 'sklearn_svm'
    if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        return 'sklearn_knn'
    return None


def _parity_kind_for(model: Any, model_type: str) -> str:
    """'classifier' → exact match required; 'regressor' → 1e-3 tolerance."""
    # LogisticRegression is a classifier despite the name — special-case it
    # BEFORE the substring check below (which would match 'regression').
    if model_type == 'sklearn_logreg':
        return 'classifier'
    name = type(model).__name__.lower()
    # Regression classes never end with 'Classifier'; both XGB/LGBM and
    # sklearn follow that convention. Check 'classifier' first so a class
    # named 'RandomForestClassifier' doesn't accidentally hit 'regression'.
    if 'classifier' in name:
        return 'classifier'
    if 'regressor' in name or 'regression' in name:
        return 'regressor'
    # Default to classifier (stricter) if we can't tell.
    return 'classifier'


# ── Converters ──────────────────────────────────────────────────────

def _convert(model: Any, model_type: str, feature_shape: int):
    """Return (onnx.ModelProto, opset_int) or (None, None) on unsupported."""
    if model_type in (
        'sklearn_rf', 'sklearn_gb', 'sklearn_logreg',
        'sklearn_linreg', 'sklearn_svm', 'sklearn_knn',
    ):
        return _convert_sklearn(model, feature_shape)
    if model_type == 'xgboost':
        return _convert_xgboost(model, feature_shape)
    if model_type == 'lightgbm':
        return _convert_lightgbm(model, feature_shape)
    return None, None


def _convert_sklearn(model: Any, feature_shape: int):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    # opset 15 is the highest most sklearn ops support without a warning under
    # skl2onnx>=1.16 — using the library's default keeps forward-compat.
    initial_type = [('float_input', FloatTensorType([None, int(feature_shape)]))]
    # Force ZipMap OFF for classifiers so the output is a plain probability
    # tensor (browser doesn't have to unpack a map<int,float>).
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {'zipmap': False}} if hasattr(model, 'predict_proba') else None,
    )
    return onx, 15


def _convert_xgboost(model: Any, feature_shape: int):
    # onnxmltools converter for XGBoost — needs skl2onnx registered as a
    # dependency for the boolean tensor ops it emits.
    from onnxmltools import convert_xgboost  # type: ignore
    from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore
    initial_type = [('float_input', FloatTensorType([None, int(feature_shape)]))]
    onx = convert_xgboost(model, initial_types=initial_type, target_opset=15)
    return onx, 15


def _convert_lightgbm(model: Any, feature_shape: int):
    from onnxmltools import convert_lightgbm  # type: ignore
    from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore
    initial_type = [('float_input', FloatTensorType([None, int(feature_shape)]))]
    onx = convert_lightgbm(model, initial_types=initial_type, target_opset=15)
    return onx, 15


# ── Parity check ────────────────────────────────────────────────────

def _parity_check(model: Any, onnx_path: str, x: np.ndarray, kind: str):
    """Predict `x` with both `model` and the ONNX runtime, compare outputs.

    Returns (ok, diag_string). `diag_string` is short enough for a log line.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f'onnxruntime import failed: {e}'

    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        return False, f'onnxruntime.InferenceSession failed: {e}'

    input_name = sess.get_inputs()[0].name
    try:
        y_onnx_out = sess.run(None, {input_name: x.astype(np.float32)})
    except Exception as e:
        return False, f'session.run failed: {e}'

    try:
        y_orig = model.predict(x)
    except Exception as e:
        return False, f'original model.predict failed: {e}'

    # ONNX output shape: [labels_tensor, probabilities_tensor, ...] for
    # classifiers with predict_proba, or just [predictions_tensor] otherwise.
    y_onnx = np.asarray(y_onnx_out[0]).ravel()
    y_orig_arr = np.asarray(y_orig).ravel()

    if y_onnx.shape != y_orig_arr.shape:
        return False, (
            f'shape mismatch: onnx={y_onnx.shape}, orig={y_orig_arr.shape}'
        )

    if kind == 'classifier':
        # Coerce both sides to strings so int-vs-str label representations
        # don't cause a false negative (skl2onnx sometimes emits int64 for
        # numeric class labels even when the sklearn model returned str).
        try:
            eq = np.array([str(a) == str(b) for a, b in zip(y_orig_arr, y_onnx)])
        except Exception as e:
            return False, f'classifier comparison failed: {e}'
        if not eq.all():
            n_mismatch = int((~eq).sum())
            return False, (
                f'classifier parity: {n_mismatch}/{len(eq)} predictions differ'
            )
        return True, f'classifier parity OK ({len(eq)} rows)'

    # Regression — 1e-3 max-abs-error ceiling from spec.
    try:
        y_onnx_f = y_onnx.astype(np.float64)
        y_orig_f = y_orig_arr.astype(np.float64)
    except Exception as e:
        return False, f'regression cast to float failed: {e}'
    max_err = float(np.max(np.abs(y_orig_f - y_onnx_f)))
    if not np.isfinite(max_err) or max_err > _REGRESSOR_TOL:
        return False, f'regression parity: max_abs_err={max_err:.6g} > {_REGRESSOR_TOL}'
    return True, f'regression parity OK (max_abs_err={max_err:.3e})'


# ── Convenience wrapper for the deploy pipeline ──────────────────────

def export_saved_model_to_onnx(
    model_pkl_path: str,
    x_sample: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Load a pickle saved by MLTrainer, export sibling .onnx, return metadata.

    The pickle format is `{'model': ..., 'scaler': ..., 'feature_names': [...], ...}`
    — see `MLTrainer.train_classification` / `train_regression`. We infer
    feature count from `feature_names` when present, else from
    `model.n_features_in_`.

    Returns None on any failure — deploy pipeline treats this as
    "client inference not available for this model".
    """
    if not model_pkl_path or not os.path.exists(model_pkl_path):
        return None
    try:
        import pickle
        with open(model_pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning(
            "[model_exporter] Could not load pickle %s: %s",
            model_pkl_path, e,
        )
        return None

    model = data.get('model')
    if model is None:
        return None

    feat_names = data.get('feature_names') or []
    feature_shape = len(feat_names)
    if not feature_shape:
        feature_shape = getattr(model, 'n_features_in_', 0)
    if not feature_shape:
        logger.info(
            "[model_exporter] Cannot infer feature count for %s — skipping",
            model_pkl_path,
        )
        return None

    onnx_path = os.path.splitext(model_pkl_path)[0] + '.onnx'
    return export_to_onnx(model, feature_shape, onnx_path, x_sample=x_sample)
