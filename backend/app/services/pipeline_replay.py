"""
CiRA ME - Pipeline Replay Service
Replays a saved preprocessing pipeline on new raw data for evaluation.
Supports both ML (feature-based) and DL (TimesNet) pipelines.
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

SUBPROCESS_SCRIPT = Path(__file__).parent / 'torch_subprocess.py'

logger = logging.getLogger(__name__)


def _run_timesnet_subprocess(task: str, config: dict, data: dict, timeout: int = 120) -> dict:
    """Run a TimesNet inference task in an isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'task': task, 'config': config, 'data': data}, f)
        config_path = f.name

    output_path = config_path.replace('.json', '_output.json')

    try:
        cmd = [sys.executable, str(SUBPROCESS_SCRIPT), config_path, output_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SUBPROCESS_SCRIPT.parent.parent.parent)
        )
        if result.returncode != 0:
            return {'success': False, 'error': result.stderr or 'Subprocess failed'}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
        return {'success': False, 'error': 'No output from subprocess'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': f'Inference timed out after {timeout}s'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        for p in [config_path, output_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


def _get_window_label(window_labels, method='majority'):
    """Assign a single label to a window from its constituent labels."""
    if method == 'first':
        return window_labels[0]
    elif method == 'last':
        return window_labels[-1]
    else:  # majority
        values, counts = np.unique(window_labels, return_counts=True)
        return values[np.argmax(counts)]


def _window_data(sensor_data: np.ndarray, window_size: int, stride: int,
                 labels: Optional[np.ndarray] = None,
                 label_method: str = 'majority'):
    """Window raw sensor data into fixed-size segments."""
    n_rows = len(sensor_data)
    if n_rows < window_size:
        raise ValueError(
            f"CSV has {n_rows} rows but pipeline requires window_size={window_size}"
        )

    windows = []
    window_labels = []
    n_windows = (n_rows - window_size) // stride + 1

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows.append(sensor_data[start:end])
        if labels is not None:
            wl = labels[start:end]
            window_labels.append(_get_window_label(wl, label_method))

    return np.array(windows), window_labels if labels is not None else None


def _normalize_windows(windows: np.ndarray, ch_min: np.ndarray,
                       ch_max: np.ndarray) -> np.ndarray:
    """Apply saved min-max normalization to windows."""
    ch_range = ch_max - ch_min
    ch_range[ch_range == 0] = 1.0
    return (windows - ch_min) / ch_range


def replay_ml_pipeline(csv_path: str, pipeline_config: dict,
                       model_data: dict) -> Dict[str, Any]:
    """
    Full ML pipeline replay: CSV -> window -> normalize -> features -> scale -> predict.

    Args:
        csv_path: Path to the uploaded CSV file
        pipeline_config: The saved pipeline_config JSON
        model_data: The loaded model pickle dict

    Returns:
        Dict with predictions, metrics (if labels present), pipeline info
    """
    from .feature_extractor import FeatureExtractor

    norm = pipeline_config.get('normalization', {})
    sensor_columns = norm.get('sensor_columns', [])
    if not sensor_columns:
        raise ValueError("Pipeline config missing normalization.sensor_columns")

    # Step 1: Load CSV
    df = pd.read_csv(csv_path)
    missing = [c for c in sensor_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required sensor columns: {missing}")

    # Detect label column
    label_col = None
    for candidate in ['label', 'class', 'target', 'category']:
        if candidate in df.columns:
            label_col = candidate
            break

    labels_raw = df[label_col].values if label_col else None

    # Step 2: Window
    wc = pipeline_config.get('windowing', {})
    window_size = wc.get('window_size', 128)
    stride = wc.get('stride', 64)
    label_method = wc.get('label_method', 'majority')

    sensor_data = df[sensor_columns].values
    all_windows, window_labels = _window_data(
        sensor_data, window_size, stride, labels_raw, label_method
    )

    logger.info(f"Pipeline replay: {len(all_windows)} windows from {len(df)} rows")

    # Step 3: Normalize
    ch_min = np.array(norm.get('channel_min', []))
    ch_max = np.array(norm.get('channel_max', []))
    if len(ch_min) > 0 and len(ch_max) > 0:
        all_windows = _normalize_windows(all_windows, ch_min, ch_max)

    # Step 4: Extract features
    feat_config = pipeline_config.get('feature_extraction', {})
    extractor = FeatureExtractor()
    feature_matrix = extractor.extract_from_windows_direct(
        windows=all_windows,
        sensor_columns=sensor_columns,
        method=feat_config.get('method', 'lightweight'),
        feature_set=feat_config.get('feature_set', 'efficient'),
    )

    # Step 5: Apply feature selection filter
    sel_config = pipeline_config.get('feature_selection')
    expected_features = None
    if sel_config and sel_config.get('selected_features'):
        expected_features = sel_config['selected_features']
    elif feat_config.get('feature_names'):
        expected_features = feat_config['feature_names']
    # Also check model pickle for feature names
    elif model_data.get('feature_names'):
        expected_features = model_data['feature_names']

    if expected_features:
        # Add missing columns as 0, reorder to match training
        for col in expected_features:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
        feature_matrix = feature_matrix[expected_features]

    X_new = feature_matrix.values

    # Step 6: Scale and predict
    model = model_data['model']
    scaler = model_data.get('scaler')

    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
    else:
        X_new_scaled = X_new

    y_pred = model.predict(X_new_scaled)

    result: Dict[str, Any] = {
        'predictions': y_pred.tolist(),
        'num_windows': len(y_pred),
        'pipeline_steps': ['load_csv', 'windowing', 'normalize',
                           'feature_extract', 'scale', 'predict'],
    }

    # Prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    result['prediction_distribution'] = dict(zip(
        [str(u) for u in unique.tolist()], counts.tolist()
    ))

    # Probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_new_scaled)
            result['probabilities'] = y_proba.tolist()
        except Exception:
            pass

    # Step 7: Compute metrics if labels present
    if window_labels:
        y_true = np.array(window_labels)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        result['new_metrics'] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'test_samples': len(y_true),
        }
        result['has_labels'] = True
    else:
        result['has_labels'] = False

    return result


def replay_dl_pipeline(csv_path: str, pipeline_config: dict,
                       model_data: dict) -> Dict[str, Any]:
    """
    Full TimesNet pipeline replay: CSV -> window -> normalize -> predict.
    No feature extraction step.

    Args:
        csv_path: Path to the uploaded CSV file
        pipeline_config: The saved pipeline_config JSON
        model_data: The loaded model pickle dict

    Returns:
        Dict with predictions, metrics (if labels present)
    """
    norm = pipeline_config.get('normalization', {})
    sensor_columns = norm.get('sensor_columns', [])
    if not sensor_columns:
        raise ValueError("Pipeline config missing normalization.sensor_columns")

    # Step 1: Load CSV
    df = pd.read_csv(csv_path)
    missing = [c for c in sensor_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required sensor columns: {missing}")

    label_col = None
    for candidate in ['label', 'class', 'target', 'category']:
        if candidate in df.columns:
            label_col = candidate
            break

    labels_raw = df[label_col].values if label_col else None

    # Step 2: Window
    wc = pipeline_config.get('windowing', {})
    window_size = wc.get('window_size', 128)
    stride = wc.get('stride', 64)
    label_method = wc.get('label_method', 'majority')

    sensor_data = df[sensor_columns].values
    all_windows, window_labels = _window_data(
        sensor_data, window_size, stride, labels_raw, label_method
    )

    logger.info(f"DL Pipeline replay: {len(all_windows)} windows from {len(df)} rows")

    # Step 3: Normalize
    ch_min = np.array(norm.get('channel_min', []))
    ch_max = np.array(norm.get('channel_max', []))
    if len(ch_min) > 0 and len(ch_max) > 0:
        all_windows = _normalize_windows(all_windows, ch_min, ch_max)

    # Step 4: Predict using TimesNet via subprocess (same pattern as timesnet_trainer)
    config_dict = model_data.get('config', {})
    mode = model_data.get('mode', 'classification')

    # Save model_data to a temp pickle so the subprocess can load tensors directly.
    # This avoids JSON serialization of PyTorch tensors entirely.
    import pickle
    model_data_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_model.pkl', delete=False)
    try:
        pickle.dump(model_data, model_data_file)
        model_data_path = model_data_file.name
    finally:
        model_data_file.close()

    if mode == 'anomaly':
        task = 'predict_anomaly'
        data_payload = {
            'windows': all_windows.tolist(),
            'labels': np.array(window_labels).tolist() if window_labels else None,
            'threshold': float(model_data.get('threshold', 0)),
            'model_data_path': model_data_path,
        }
    else:
        task = 'predict_classification'
        data_payload = {
            'windows': all_windows.tolist(),
            'labels': window_labels if window_labels else None,
            'label_classes': model_data.get('label_encoder_classes', []),
            'model_data_path': model_data_path,
        }

    try:
        subprocess_result = _run_timesnet_subprocess(task, config_dict, data_payload)
    finally:
        if os.path.exists(model_data_path):
            try:
                os.unlink(model_data_path)
            except Exception:
                pass

    if not subprocess_result.get('success'):
        raise RuntimeError(subprocess_result.get('error', 'TimesNet prediction failed'))

    y_pred = np.array(subprocess_result.get('predictions', []))
    metrics = subprocess_result.get('metrics', {})

    result: Dict[str, Any] = {
        'predictions': y_pred.tolist(),
        'num_windows': len(y_pred),
        'pipeline_steps': ['load_csv', 'windowing', 'normalize', 'timesnet_predict'],
    }

    unique, counts = np.unique(y_pred, return_counts=True)
    result['prediction_distribution'] = dict(zip(
        [str(u) for u in unique.tolist()], counts.tolist()
    ))

    if window_labels and len(metrics) > 0:
        result['new_metrics'] = {
            'accuracy': float(metrics.get('accuracy', 0)),
            'precision': float(metrics.get('precision', 0)),
            'recall': float(metrics.get('recall', 0)),
            'f1': float(metrics.get('f1', 0)),
            'test_samples': len(window_labels),
        }
        result['has_labels'] = True
    elif window_labels:
        y_true = np.array(window_labels)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        result['new_metrics'] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(
                y_true, y_pred, average='weighted', zero_division=0)),
            'test_samples': len(y_true),
        }
        result['has_labels'] = True
    else:
        result['has_labels'] = False

    return result
