"""
CiRA ME - CiRA CLAW Export Service

Converts a saved CiRA ME model to a CiRA CLAW deployment package:
  model.onnx       -- model + scaler as a unified ONNX graph
  cira_model.json  -- CiRA CLAW manifest
  labels.txt       -- class names (one per line)

Supported algorithms:
  ML classification: rf, gb, svm, mlp, knn, dt, nb, lr  -> label_prob output
  ML anomaly:        iforest, ocsvm                      -> anomaly_score output
  Deep learning:     timesnet (classification)           -> softmax output
                     timesnet (anomaly)                  -> reconstruction output

Blocked (not ONNX-convertible):
  lof, hbos, copod, ecod, knn (PyOD anomaly)
  Any model trained with 'tsfresh' feature method
"""

import os
import json
import pickle
import zipfile
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Algorithms that can be converted to ONNX via skl2onnx
ONNX_SUPPORTED_ML_CLASS = ['rf', 'gb', 'svm', 'mlp', 'knn', 'dt', 'nb', 'lr']
ONNX_SUPPORTED_ANOMALY = ['iforest', 'ocsvm']
ONNX_BLOCKED_ANOMALY = ['lof', 'hbos', 'copod', 'ecod']

# Canonical DSP feature names in exact extraction order
# Must match feature_extractor.py _compute_dsp_features() output dict key order
DSP_FEATURE_NAMES = [
    'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
    'impulse_factor', 'margin_factor', 'zero_crossing_rate',
    'autocorr_lag1', 'autocorr_lag5', 'binned_entropy',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'spectral_flatness', 'spectral_entropy', 'peak_frequency',
    'spectral_skewness', 'spectral_kurtosis',
    'band_power_low', 'band_power_mid', 'band_power_high',
]

# Canonical statistical feature names in exact extraction order
# Must match feature_extractor.py TSFRESH_FEATURES dict key order
STATISTICAL_FEATURE_NAMES = [
    'mean', 'std', 'min', 'max', 'median', 'sum', 'variance',
    'skewness', 'kurtosis', 'abs_energy', 'root_mean_square',
    'mean_abs_change', 'mean_change', 'count_above_mean', 'count_below_mean',
    'first_location_of_maximum', 'first_location_of_minimum',
    'last_location_of_maximum', 'last_location_of_minimum',
    'percentage_of_reoccurring_values', 'sum_of_reoccurring_values',
    'abs_sum_of_changes', 'range', 'interquartile_range',
    'mean_second_derivative',
]


class CiraCLAWExporter:
    """Converts CiRA ME saved models to CiRA CLAW deployment packages."""

    def export(self, model_id: int) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            model_id: SavedModel database ID

        Returns:
            {'path': str, 'filename': str, 'size': int}

        Raises:
            ValueError: For unsupported algorithms or incomplete pipeline configs
            ImportError: If required packages (skl2onnx, onnx) are not installed
        """
        from ..models import SavedModel

        saved = SavedModel.get_by_id(model_id)
        if not saved:
            raise ValueError(f"Saved model not found: {model_id}")

        pipeline_config = saved.get('pipeline_config', {})
        if not pipeline_config or not pipeline_config.get('normalization'):
            raise ValueError(
                "Model is missing pipeline configuration. "
                "Re-save the model from an active training session to enable CiRA CLAW export."
            )

        self._validate_pipeline_config(pipeline_config, saved.get('algorithm', ''))

        # Load model data from disk
        model_path = saved.get('model_path', '')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model file not found on disk: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Determine export path based on approach
        approach = pipeline_config.get('training_approach', 'ml')
        algorithm = saved.get('algorithm', '').lower()
        mode = saved.get('mode', 'classification')

        if approach == 'dl' or algorithm == 'timesnet':
            onnx_bytes = self._export_timesnet_model(model_data, pipeline_config)
            output_format = 'softmax' if mode == 'classification' else 'reconstruction'
        else:
            onnx_bytes = self._export_ml_model(model_data, pipeline_config, algorithm, mode)
            output_format = self._determine_output_format(algorithm, mode)

        # Assemble manifest
        cira_model = self._build_cira_model_json(
            saved, model_data, pipeline_config, output_format
        )

        # Build class label list
        classes = cira_model['output']['classes']

        # Write zip
        safe_name = saved.get('name', 'model').replace(' ', '_').replace('/', '_')
        zip_filename = f"cira_claw_{safe_name}.zip"
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('model.onnx', onnx_bytes)
            zf.writestr('cira_model.json', json.dumps(cira_model, indent=2))
            zf.writestr('labels.txt', '\n'.join(str(c) for c in classes) + '\n')

        return {
            'path': zip_path,
            'filename': zip_filename,
            'size': os.path.getsize(zip_path),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pipeline_config(self, pipeline_config: dict, algorithm: str):
        """
        Raise ValueError with clear message if config is incomplete or
        the algorithm is not supported for CiRA CLAW export.
        """
        algo = algorithm.lower()

        # Block unsupported anomaly algorithms
        if algo in ONNX_BLOCKED_ANOMALY:
            raise ValueError(
                f"Algorithm '{algo}' cannot be converted to ONNX and is not supported "
                f"for CiRA CLAW export.\n"
                f"Supported anomaly algorithms: {ONNX_SUPPORTED_ANOMALY}\n"
                f"Suggestion: Retrain with Isolation Forest (iforest) or One-Class SVM (ocsvm)."
            )

        # Block tsfresh feature method
        feat = pipeline_config.get('feature_extraction', {})
        if feat.get('method') == 'tsfresh':
            raise ValueError(
                "This model was trained using the real tsfresh library which produces "
                "hundreds of features. CiRA CLAW's C runtime can only reproduce the "
                "46-feature lightweight DSP pipeline. "
                "Retrain the model using the 'Lightweight' feature extraction method."
            )

        # Check for required pipeline fields
        errors = []
        norm = pipeline_config.get('normalization', {})
        if not norm.get('sensor_columns'):
            errors.append("normalization.sensor_columns is missing or empty")
        if not norm.get('channel_min') or not norm.get('channel_max'):
            errors.append("normalization.channel_min / channel_max are missing")

        wc = pipeline_config.get('windowing', {})
        if not wc.get('window_size'):
            errors.append("windowing.window_size is missing")

        sel = pipeline_config.get('feature_selection', {})
        feat_names = feat.get('feature_names', [])
        selected = sel.get('selected_features', [])
        if not feat_names and not selected:
            errors.append(
                "feature_extraction.feature_names and feature_selection.selected_features "
                "are both missing — cannot determine input feature count for ONNX export"
            )

        if errors:
            raise ValueError(
                "Pipeline config is incomplete for CiRA CLAW export:\n" +
                "\n".join(f"  - {e}" for e in errors) +
                "\nRe-save the model from an active training session to capture the full pipeline."
            )

    # ------------------------------------------------------------------
    # ML / PyOD ONNX export
    # ------------------------------------------------------------------

    def _export_ml_model(self, model_data: dict, pipeline_config: dict,
                          algorithm: str, mode: str) -> bytes:
        """
        Convert an sklearn or PyOD model (+ StandardScaler) to ONNX bytes.
        The scaler is baked into the ONNX pipeline so CiRA CLAW never needs
        to apply normalization separately.

        Supports:
          - All sklearn classifiers: Pipeline([scaler, model])
          - PyOD iforest/ocsvm: Pipeline([scaler, model.detector_])
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            from sklearn.pipeline import Pipeline as SKPipeline
        except ImportError:
            raise ImportError(
                "skl2onnx is required for CiRA CLAW ML export. "
                "Install with: pip install skl2onnx onnx"
            )

        model = model_data['model']
        scaler = model_data.get('scaler')

        # Determine number of input features
        sel = pipeline_config.get('feature_selection', {})
        feat = pipeline_config.get('feature_extraction', {})
        selected = sel.get('selected_features') or feat.get('feature_names', [])
        n_features = len(selected)

        if n_features == 0:
            raise ValueError("Cannot determine input feature count from pipeline_config.")

        # For PyOD models, extract the inner sklearn estimator
        inner_model = model
        if mode == 'anomaly' and hasattr(model, 'detector_'):
            inner_model = model.detector_

        # Build sklearn pipeline with scaler baked in
        if scaler is not None:
            sk_pipeline = SKPipeline([('scaler', scaler), ('model', inner_model)])
        else:
            sk_pipeline = inner_model

        initial_types = [('float_input', FloatTensorType([None, n_features]))]

        try:
            onnx_model = convert_sklearn(sk_pipeline, initial_types=initial_types)
        except Exception as e:
            raise ValueError(
                f"ONNX conversion failed for algorithm '{algorithm}': {e}\n"
                "This may indicate an unsupported sklearn estimator configuration."
            )

        return onnx_model.SerializeToString()

    # ------------------------------------------------------------------
    # TimesNet ONNX export
    # ------------------------------------------------------------------

    def _export_timesnet_model(self, model_data: dict, pipeline_config: dict) -> bytes:
        """
        Convert a TimesNet PyTorch model to ONNX bytes.

        Handles two storage formats:
          Format A (subprocess path): model_data['model_state']['model_state_dict']
                                      stores weights as nested Python lists
          Format B (in-process path): model_data['model_state'] is OrderedDict of tensors
        """
        try:
            import torch
            import torch.nn as nn
            import io
        except ImportError:
            raise ImportError(
                "PyTorch is required for TimesNet ONNX export. "
                "Install with: pip install torch"
            )

        # --- Unpack model state and config ---
        inner = model_data.get('model_state', {})

        if isinstance(inner, dict) and 'model_state_dict' in inner:
            # Format A: subprocess-trained model
            cfg = inner.get('config', model_data.get('config', {}))
            mode = inner.get('mode', model_data.get('mode', 'anomaly'))
            raw_state = inner['model_state_dict']
            state_dict = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in raw_state.items()
            }
            class_names = (inner.get('label_encoder_classes')
                           or model_data.get('label_encoder_classes', []))
            threshold = float(inner.get('threshold', 0.5))
        else:
            # Format B: in-process trained model
            cfg = model_data.get('config', {})
            mode = model_data.get('mode', 'anomaly')
            state_dict = inner  # already an OrderedDict of tensors
            class_names = model_data.get('label_encoder_classes', [])
            threshold = float(model_data.get('threshold', 0.5))

        # Convert config object to dict if needed (TimesNetConfig dataclass)
        if hasattr(cfg, '__dict__'):
            cfg = vars(cfg)

        # Get dimensions from pipeline_config (authoritative)
        norm = pipeline_config.get('normalization', {})
        num_channels = len(norm.get('sensor_columns', []))
        if num_channels == 0:
            num_channels = cfg.get('enc_in', 1)

        window_size = pipeline_config.get('windowing', {}).get('window_size', 128)

        # Build architecture — must exactly match torch_subprocess.py
        if mode == 'classification':
            num_classes = len(class_names) if class_names else cfg.get('num_class', 2)
            model = self._build_timesnet_classifier(cfg, num_channels, num_classes)
        else:
            model = self._build_timesnet_encoder(cfg, num_channels)

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        dummy = torch.zeros(1, window_size, num_channels)

        buffer = io.BytesIO()
        try:
            torch.onnx.export(
                model,
                dummy,
                buffer,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=12,
            )
        except Exception as e:
            raise ValueError(f"TimesNet ONNX export failed: {e}")

        return buffer.getvalue()

    def _build_timesnet_encoder(self, cfg: dict, num_channels: int):
        """
        Anomaly detection encoder.
        Architecture mirrors build_timesnet_encoder() in torch_subprocess.py exactly.
        """
        import torch.nn as nn

        d_model = cfg.get('d_model', 64)
        d_ff = cfg.get('d_ff', 128)
        dropout = cfg.get('dropout', 0.1)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(num_channels, d_model)
                self.encoder = nn.Sequential(
                    nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1),
                )
                self.projection = nn.Linear(d_model, num_channels)

            def forward(self, x):
                x = self.embed(x).transpose(1, 2)
                x = self.decoder(self.encoder(x)).transpose(1, 2)
                return self.projection(x)

        return Encoder()

    def _build_timesnet_classifier(self, cfg: dict, num_channels: int, num_classes: int):
        """
        Classification model.
        Architecture mirrors build_timesnet_classifier() in torch_subprocess.py exactly.
        """
        import torch.nn as nn

        d_model = cfg.get('d_model', 64)
        d_ff = cfg.get('d_ff', 128)
        dropout = cfg.get('dropout', 0.1)

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(num_channels, d_model)
                self.encoder = nn.Sequential(
                    nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, num_classes),
                )

            def forward(self, x):
                x = self.embed(x).transpose(1, 2)
                x = self.encoder(x)
                x = self.pool(x).squeeze(-1)
                return self.classifier(x)

        return Classifier()

    # ------------------------------------------------------------------
    # cira_model.json assembly
    # ------------------------------------------------------------------

    def _determine_output_format(self, algorithm: str, mode: str) -> str:
        """Return the output.format string for cira_model.json."""
        algo = algorithm.lower()
        if algo in ONNX_SUPPORTED_ANOMALY or mode == 'anomaly':
            return 'anomaly_score'
        return 'label_prob'

    def _extract_class_names(self, model_data: dict) -> List[str]:
        """Extract class names from model_data in a format-agnostic way."""
        inner = model_data.get('model_state', {})
        if isinstance(inner, dict):
            names = (inner.get('label_encoder_classes')
                     or model_data.get('label_encoder_classes')
                     or model_data.get('classes', []))
        else:
            names = model_data.get('classes', [])
        return [str(n) for n in names] if names else []

    def _extract_anomaly_threshold(self, model_data: dict) -> Optional[float]:
        """Extract threshold value for anomaly models."""
        inner = model_data.get('model_state', {})
        if isinstance(inner, dict):
            t = inner.get('threshold', model_data.get('threshold'))
        else:
            t = model_data.get('threshold')
        return float(t) if t is not None else None

    def _build_cira_model_json(self, saved_model: dict, model_data: dict,
                                pipeline_config: dict, output_format: str) -> dict:
        """
        Assemble the complete cira_model.json manifest.
        Reads all pipeline parameters from pipeline_config (already stored in DB).
        """
        norm = pipeline_config.get('normalization', {})
        wc = pipeline_config.get('windowing', {})
        feat = pipeline_config.get('feature_extraction', {})
        sel = pipeline_config.get('feature_selection', {})

        selected_features = sel.get('selected_features') or feat.get('feature_names', [])

        mode = saved_model.get('mode', 'classification')
        algorithm = saved_model.get('algorithm', 'unknown')

        classes = self._extract_class_names(model_data)
        threshold = None
        if mode == 'anomaly' or output_format in ('reconstruction', 'anomaly_score'):
            threshold = self._extract_anomaly_threshold(model_data)

        return {
            "version": "1.0",
            "generated_by": "cira_me",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_name": saved_model.get('name', 'unnamed'),
            "training_approach": pipeline_config.get('training_approach', 'ml'),
            "algorithm": algorithm,
            "mode": mode,
            "input_type": "signal",
            "normalization": {
                "sensor_columns": norm.get('sensor_columns', []),
                "channel_min": norm.get('channel_min', []),
                "channel_max": norm.get('channel_max', []),
            },
            "windowing": {
                "window_size": wc.get('window_size', 128),
                "stride": wc.get('stride', 64),
            },
            "feature_extraction": {
                "method": feat.get('method', 'lightweight'),
                "features_per_channel": 46,
                "sampling_rate_hz": feat.get('sampling_rate', 100.0),
                "statistical_features": STATISTICAL_FEATURE_NAMES,
                "dsp_features": DSP_FEATURE_NAMES,
            },
            "feature_selection": {
                "selected_features": selected_features,
                "num_selected": len(selected_features),
            },
            "output": {
                "format": output_format,
                "classes": classes,
                "num_classes": len(classes),
                "anomaly_threshold": threshold,
            },
            "model_file": "model.onnx",
        }
