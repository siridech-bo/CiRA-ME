"""
CiRA ME - Deployment Service
Handles model export and SSH deployment to edge devices
"""

import os
import uuid
import pickle
from typing import Dict, Any, Optional
from datetime import datetime

from .ml_trainer import _model_sessions


def load_saved_model_session(model_path: str, algorithm: str = 'unknown',
                              mode: str = 'classification') -> Dict[str, Any]:
    """Load a saved model from disk and create a session-compatible dict.

    This allows saved benchmark models (stored as pickle on disk) to be
    used with the deployer and exporter without needing an in-memory session.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    session = {
        'model': data.get('model'),
        'scaler': data.get('scaler'),
        'algorithm': data.get('algorithm', algorithm),
        'mode': data.get('mode', mode),
        'model_path': model_path,
        'hyperparameters': data.get('hyperparameters', {}),
        'metrics': data.get('metrics', {}),
    }
    return session

# Global storage for deployments
_deployments: Dict[str, Dict] = {}


class Deployer:
    """Service for deploying models to edge devices."""

    def __init__(self):
        pass

    def test_connection(
        self,
        host: str,
        username: str,
        password: Optional[str] = None,
        port: int = 22,
        key_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test SSH connection to a remote host."""
        try:
            import paramiko
        except ImportError:
            raise ImportError("paramiko library required. Install with: pip install paramiko")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if key_path:
                key = paramiko.RSAKey.from_private_key_file(key_path)
                client.connect(host, port=port, username=username, pkey=key, timeout=10)
            else:
                client.connect(host, port=port, username=username, password=password, timeout=10)

            # Get system info
            stdin, stdout, stderr = client.exec_command('uname -a && cat /etc/os-release | head -5')
            system_info = stdout.read().decode('utf-8').strip()

            # Check for CUDA/Jetson
            stdin, stdout, stderr = client.exec_command('nvcc --version 2>/dev/null || echo "No CUDA"')
            cuda_info = stdout.read().decode('utf-8').strip()

            client.close()

            return {
                'status': 'connected',
                'host': host,
                'system_info': system_info,
                'cuda_info': cuda_info,
                'message': 'Connection successful'
            }

        except Exception as e:
            return {
                'status': 'failed',
                'host': host,
                'error': str(e),
                'message': 'Connection failed'
            }
        finally:
            client.close()

    def deploy(
        self,
        training_session_id: str,
        target_type: str,
        export_format: str,
        ssh_config: Dict,
        options: Dict,
        saved_model_session: Dict = None
    ) -> Dict[str, Any]:
        """Deploy a trained model to an edge device via SSH.

        Args:
            training_session_id: In-memory session ID (from current training)
            target_type: Target device type
            export_format: Export format (pickle, joblib, onnx)
            ssh_config: SSH connection configuration
            options: Deployment options
            saved_model_session: Pre-loaded session dict from a saved model on disk.
                                 If provided, this is used instead of looking up training_session_id.
        """
        try:
            import paramiko
            from scp import SCPClient
        except ImportError:
            raise ImportError("paramiko and scp libraries required. Install with: pip install paramiko scp")

        # Get model session — either from saved model or in-memory
        if saved_model_session:
            session = saved_model_session
            # Temporarily register so export_model() can find it
            _model_sessions[training_session_id] = session
        else:
            session = _model_sessions.get(training_session_id)
            if not session:
                raise ValueError(f"Training session not found: {training_session_id}")

        deployment_id = str(uuid.uuid4())
        deployment_status = {
            'id': deployment_id,
            'training_session_id': training_session_id,
            'target_type': target_type,
            'export_format': export_format,
            'steps': [],
            'status': 'in_progress',
            'started_at': datetime.utcnow().isoformat()
        }

        _deployments[deployment_id] = deployment_status

        try:
            # Step 1: Export model
            deployment_status['steps'].append({'step': 'export', 'status': 'in_progress'})

            from .ml_trainer import MLTrainer
            trainer = MLTrainer()
            export_result = trainer.export_model(training_session_id, export_format)
            model_path = export_result['path']

            deployment_status['steps'][-1]['status'] = 'completed'
            deployment_status['steps'][-1]['result'] = export_result

            # Step 2: Generate inference script
            if options.get('include_inference_script', True):
                deployment_status['steps'].append({'step': 'generate_script', 'status': 'in_progress'})

                script_result = self.generate_inference_script(training_session_id, 'python')
                script_path = script_result['path']

                deployment_status['steps'][-1]['status'] = 'completed'

            # Step 3: Connect via SSH
            deployment_status['steps'].append({'step': 'ssh_connect', 'status': 'in_progress'})

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if ssh_config.get('key_path'):
                key = paramiko.RSAKey.from_private_key_file(ssh_config['key_path'])
                client.connect(
                    ssh_config['host'],
                    port=ssh_config.get('port', 22),
                    username=ssh_config['username'],
                    pkey=key,
                    timeout=30
                )
            else:
                client.connect(
                    ssh_config['host'],
                    port=ssh_config.get('port', 22),
                    username=ssh_config['username'],
                    password=ssh_config.get('password'),
                    timeout=30
                )

            deployment_status['steps'][-1]['status'] = 'completed'

            # Step 4: Create remote directory
            deployment_status['steps'].append({'step': 'create_directory', 'status': 'in_progress'})

            remote_path = ssh_config.get('remote_path', '/home/user/models')
            client.exec_command(f'mkdir -p {remote_path}')

            deployment_status['steps'][-1]['status'] = 'completed'

            # Step 5: Transfer files
            deployment_status['steps'].append({'step': 'transfer', 'status': 'in_progress'})

            scp = SCPClient(client.get_transport())

            # Transfer model
            remote_model_path = os.path.join(remote_path, os.path.basename(model_path))
            scp.put(model_path, remote_model_path)

            # Transfer inference script
            if options.get('include_inference_script', True):
                remote_script_path = os.path.join(remote_path, 'inference.py')
                scp.put(script_path, remote_script_path)

            # Transfer scaler if included
            if options.get('include_scaler', True):
                scaler_path = model_path.replace('.onnx', '_scaler.pkl').replace('.joblib', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    remote_scaler_path = os.path.join(remote_path, os.path.basename(scaler_path))
                    scp.put(scaler_path, remote_scaler_path)

            scp.close()
            deployment_status['steps'][-1]['status'] = 'completed'

            # Step 6: Validate deployment
            deployment_status['steps'].append({'step': 'validate', 'status': 'in_progress'})

            stdin, stdout, stderr = client.exec_command(f'ls -la {remote_path}')
            file_list = stdout.read().decode('utf-8')

            deployment_status['steps'][-1]['status'] = 'completed'
            deployment_status['steps'][-1]['result'] = {'files': file_list}

            client.close()

            # Update final status
            deployment_status['status'] = 'completed'
            deployment_status['completed_at'] = datetime.utcnow().isoformat()
            deployment_status['remote_path'] = remote_path

            return {
                'deployment_id': deployment_id,
                'status': 'completed',
                'remote_path': remote_path,
                'steps': deployment_status['steps'],
                'message': 'Deployment successful'
            }

        except Exception as e:
            deployment_status['status'] = 'failed'
            deployment_status['error'] = str(e)
            deployment_status['completed_at'] = datetime.utcnow().isoformat()

            raise

    def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get the status of a deployment."""
        deployment = _deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")

        return deployment

    def generate_inference_script(self, training_session_id: str, language: str = 'python',
                                    saved_model_session: Dict = None) -> Dict[str, Any]:
        """Generate an inference script for a trained model."""
        session = saved_model_session or _model_sessions.get(training_session_id)
        if not session:
            raise ValueError(f"Training session not found: {training_session_id}")

        algorithm = session['algorithm']
        mode = session['mode']

        if language == 'python':
            script = self._generate_python_inference_script(session)
        elif language == 'cpp':
            script = self._generate_cpp_inference_script(session)
        else:
            raise ValueError(f"Unsupported language: {language}")

        # Save script
        script_path = session['model_path'].replace('.pkl', f'_inference.{language if language == "cpp" else "py"}')
        with open(script_path, 'w') as f:
            f.write(script)

        return {
            'path': script_path,
            'language': language,
            'script': script
        }

    def _generate_python_inference_script(self, session: Dict) -> str:
        """Generate Python inference script."""
        mode = session['mode']
        algorithm = session['algorithm']

        if mode == 'anomaly':
            return f'''#!/usr/bin/env python3
"""
CiRA ME - Anomaly Detection Inference Script
Model: {algorithm}
Generated: {datetime.utcnow().isoformat()}
"""

import pickle
import numpy as np
import sys

def load_model(model_path):
    """Load the trained model and scaler."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

def predict(model, scaler, features):
    """
    Make anomaly predictions.

    Args:
        model: Trained PyOD model
        scaler: StandardScaler for feature normalization
        features: numpy array of shape (n_samples, n_features)

    Returns:
        predictions: 0 for normal, 1 for anomaly
        scores: Anomaly decision scores
    """
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    scores = model.decision_function(features_scaled)
    return predictions, scores

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load model
    model, scaler = load_model(model_path)

    # Load data (CSV format expected)
    import pandas as pd
    data = pd.read_csv(data_path)
    features = data.select_dtypes(include=[np.number]).values

    # Predict
    predictions, scores = predict(model, scaler, features)

    # Output results
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        status = "ANOMALY" if pred == 1 else "NORMAL"
        print(f"Sample {{i}}: {{status}} (score: {{score:.4f}})")

    print(f"\\nTotal anomalies detected: {{np.sum(predictions)}} / {{len(predictions)}}")

if __name__ == "__main__":
    main()
'''
        else:
            return f'''#!/usr/bin/env python3
"""
CiRA ME - Classification Inference Script
Model: {algorithm}
Generated: {datetime.utcnow().isoformat()}
"""

import pickle
import numpy as np
import sys

def load_model(model_path):
    """Load the trained model and scaler."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data.get('classes', [])

def predict(model, scaler, features):
    """
    Make classification predictions.

    Args:
        model: Trained scikit-learn classifier
        scaler: StandardScaler for feature normalization
        features: numpy array of shape (n_samples, n_features)

    Returns:
        predictions: Class labels
        probabilities: Class probabilities (if available)
    """
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)

    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)

    return predictions, probabilities

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load model
    model, scaler, classes = load_model(model_path)

    # Load data (CSV format expected)
    import pandas as pd
    data = pd.read_csv(data_path)
    features = data.select_dtypes(include=[np.number]).values

    # Predict
    predictions, probabilities = predict(model, scaler, features)

    # Output results
    for i, pred in enumerate(predictions):
        if probabilities is not None:
            conf = np.max(probabilities[i]) * 100
            print(f"Sample {{i}}: {{pred}} (confidence: {{conf:.1f}}%)")
        else:
            print(f"Sample {{i}}: {{pred}}")

    # Summary
    unique, counts = np.unique(predictions, return_counts=True)
    print("\\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {{cls}}: {{count}}")

if __name__ == "__main__":
    main()
'''

    def _generate_cpp_inference_script(self, session: Dict) -> str:
        """Generate C++ inference script (ONNX Runtime)."""
        return '''/*
 * CiRA ME - C++ Inference Script (ONNX Runtime)
 *
 * Compile with:
 * g++ -o inference inference.cpp -lonnxruntime
 *
 * Usage:
 * ./inference model.onnx input_data.csv
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <onnxruntime_cxx_api.h>

std::vector<std::vector<float>> load_csv(const std::string& filepath) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filepath);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (...) {
                // Skip non-numeric columns
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <data.csv>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* data_path = argv[2];

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CiRAME");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load model
    Ort::Session session(env, model_path, session_options);

    // Load data
    auto data = load_csv(data_path);

    std::cout << "Loaded " << data.size() << " samples" << std::endl;

    // Run inference (implementation depends on model structure)
    // ...

    return 0;
}
'''

    def generate_full_inference_script(self, pipeline_config: dict,
                                       algorithm: str = 'unknown',
                                       mode: str = 'classification') -> str:
        """Generate a self-contained inference script with full DSP pipeline.

        Includes: CSV loading, windowing, normalization, feature extraction,
        feature selection, scaling, and model prediction.
        """
        norm = pipeline_config.get('normalization', {})
        wc = pipeline_config.get('windowing', {})
        feat_config = pipeline_config.get('feature_extraction', {})
        sel_config = pipeline_config.get('feature_selection')
        approach = pipeline_config.get('training_approach', 'ml')

        sensor_columns = norm.get('sensor_columns', [])
        channel_min = norm.get('channel_min', [])
        channel_max = norm.get('channel_max', [])
        window_size = wc.get('window_size', 128)
        stride = wc.get('stride', 64)

        expected_features = []
        if sel_config and sel_config.get('selected_features'):
            expected_features = sel_config['selected_features']
        elif feat_config.get('feature_names'):
            expected_features = feat_config['feature_names']

        tsfresh_warning = ""
        if feat_config.get('method') == 'tsfresh':
            tsfresh_warning = (
                "# NOTE: Original model used tsfresh features.\n"
                "# This script uses lightweight features for edge deployment.\n"
                "# Prediction accuracy may differ from training.\n"
            )

        return f'''#!/usr/bin/env python3
"""
CiRA ME - Full Pipeline Inference Script
Algorithm: {algorithm} ({mode})
Approach: {approach}
Generated: {datetime.utcnow().isoformat()}

Usage: python inference.py <model.pkl> <data.csv>
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq

{tsfresh_warning}
# ===== Pipeline Configuration (embedded from training) =====
SENSOR_COLUMNS = {sensor_columns}
WINDOW_SIZE = {window_size}
STRIDE = {stride}
CHANNEL_MIN = np.array({channel_min})
CHANNEL_MAX = np.array({channel_max})
EXPECTED_FEATURES = {expected_features}
MODE = "{mode}"


# ===== Step 1: Load CSV =====
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {{missing}}")
    return df


# ===== Step 2: Windowing =====
def apply_windowing(data):
    n_rows = len(data)
    windows = []
    for i in range((n_rows - WINDOW_SIZE) // STRIDE + 1):
        start = i * STRIDE
        windows.append(data[start:start + WINDOW_SIZE])
    return np.array(windows)


# ===== Step 3: Min-Max Normalization =====
def normalize(windows):
    ch_range = CHANNEL_MAX - CHANNEL_MIN
    ch_range[ch_range == 0] = 1.0
    return (windows - CHANNEL_MIN) / ch_range


# ===== Step 4: Feature Extraction (lightweight DSP) =====
def _autocorr(x, lag):
    n = len(x)
    if lag >= n:
        return 0.0
    m = np.mean(x)
    v = np.var(x)
    if v < 1e-10:
        return 0.0
    return np.mean((x[:n-lag] - m) * (x[lag:] - m)) / v


def _binned_entropy(x, bins=10):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-10))


def extract_features(windows):
    """Extract lightweight DSP features per window."""
    all_features = []

    for window in windows:
        row = {{}}
        n_ch = window.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # --- Statistical features (25 per channel) ---
            for ch_idx in range(n_ch):
                col = SENSOR_COLUMNS[ch_idx]
                ch = window[:, ch_idx]

                row[f"mean_{{col}}"] = float(np.mean(ch))
                row[f"std_{{col}}"] = float(np.std(ch))
                row[f"min_{{col}}"] = float(np.min(ch))
                row[f"max_{{col}}"] = float(np.max(ch))
                row[f"median_{{col}}"] = float(np.median(ch))
                row[f"sum_{{col}}"] = float(np.sum(ch))
                row[f"variance_{{col}}"] = float(np.var(ch))
                row[f"skewness_{{col}}"] = float(stats.skew(ch))
                row[f"kurtosis_{{col}}"] = float(stats.kurtosis(ch))
                row[f"abs_energy_{{col}}"] = float(np.sum(ch ** 2))
                row[f"root_mean_square_{{col}}"] = float(np.sqrt(np.mean(ch ** 2)))
                row[f"mean_abs_change_{{col}}"] = float(np.mean(np.abs(np.diff(ch))))
                row[f"mean_change_{{col}}"] = float(np.mean(np.diff(ch)))
                row[f"count_above_mean_{{col}}"] = float(np.sum(ch > np.mean(ch)))
                row[f"count_below_mean_{{col}}"] = float(np.sum(ch < np.mean(ch)))
                row[f"first_location_of_maximum_{{col}}"] = float(np.argmax(ch) / len(ch))
                row[f"first_location_of_minimum_{{col}}"] = float(np.argmin(ch) / len(ch))
                row[f"last_location_of_maximum_{{col}}"] = float((len(ch) - 1 - np.argmax(ch[::-1])) / len(ch))
                row[f"last_location_of_minimum_{{col}}"] = float((len(ch) - 1 - np.argmin(ch[::-1])) / len(ch))
                row[f"percentage_of_reoccurring_values_{{col}}"] = float(len(np.unique(ch)) / len(ch))
                row[f"sum_of_reoccurring_values_{{col}}"] = float(np.sum([v for v in ch if np.sum(ch == v) > 1]))
                row[f"abs_sum_of_changes_{{col}}"] = float(np.sum(np.abs(np.diff(ch))))
                row[f"range_{{col}}"] = float(np.max(ch) - np.min(ch))
                row[f"interquartile_range_{{col}}"] = float(np.percentile(ch, 75) - np.percentile(ch, 25))
                msd = float(np.mean(np.diff(np.diff(ch)))) if len(ch) > 2 else 0.0
                row[f"mean_second_derivative_{{col}}"] = msd

            # --- DSP features (19 per channel) ---
            rms_vals = np.sqrt(np.mean(window ** 2, axis=0))
            peak_vals = np.max(np.abs(window), axis=0)
            mean_abs = np.mean(np.abs(window), axis=0)
            rms_safe = np.where(rms_vals == 0, 1e-10, rms_vals)
            mean_abs_safe = np.where(mean_abs == 0, 1e-10, mean_abs)
            mean_sqrt = np.mean(np.sqrt(np.abs(window)), axis=0) ** 2
            mean_sqrt_safe = np.where(mean_sqrt == 0, 1e-10, mean_sqrt)

            for ch_idx in range(n_ch):
                col = SENSOR_COLUMNS[ch_idx]
                ch = window[:, ch_idx]

                row[f"rms_{{col}}"] = float(rms_vals[ch_idx])
                row[f"peak_to_peak_{{col}}"] = float(np.max(ch) - np.min(ch))
                row[f"crest_factor_{{col}}"] = float(peak_vals[ch_idx] / rms_safe[ch_idx])
                row[f"shape_factor_{{col}}"] = float(rms_safe[ch_idx] / mean_abs_safe[ch_idx])
                row[f"impulse_factor_{{col}}"] = float(peak_vals[ch_idx] / mean_abs_safe[ch_idx])
                row[f"margin_factor_{{col}}"] = float(peak_vals[ch_idx] / mean_sqrt_safe[ch_idx])
                zc = np.sum(np.diff(np.sign(ch)) != 0)
                row[f"zero_crossing_rate_{{col}}"] = float(zc / (len(ch) - 1))
                row[f"autocorr_lag1_{{col}}"] = float(_autocorr(ch, 1))
                row[f"autocorr_lag5_{{col}}"] = float(_autocorr(ch, 5))
                row[f"binned_entropy_{{col}}"] = float(_binned_entropy(ch))

                # FFT features
                n_samples = len(ch)
                freqs = fftfreq(n_samples, 1.0 / 100.0)
                pos_mask = freqs >= 0
                fft_mag = np.abs(fft(ch))[pos_mask]
                fft_f = freqs[pos_mask]
                total_power = np.sum(fft_mag ** 2)
                tp_safe = total_power if total_power > 0 else 1e-10
                norm_power = fft_mag ** 2 / tp_safe

                row[f"spectral_centroid_{{col}}"] = float(np.sum(fft_f * norm_power))
                sc = row[f"spectral_centroid_{{col}}"]
                row[f"spectral_bandwidth_{{col}}"] = float(np.sqrt(np.sum(((fft_f - sc) ** 2) * norm_power)))
                cumsum = np.cumsum(norm_power)
                ri = np.searchsorted(cumsum, 0.95 * cumsum[-1])
                row[f"spectral_rolloff_{{col}}"] = float(fft_f[min(ri, len(fft_f) - 1)])
                gm = np.exp(np.mean(np.log(fft_mag + 1e-10)))
                am = np.mean(fft_mag)
                row[f"spectral_flatness_{{col}}"] = float(gm / (am + 1e-10))
                np_safe = norm_power + 1e-10
                row[f"spectral_entropy_{{col}}"] = float(-np.sum(np_safe * np.log2(np_safe)))
                row[f"peak_frequency_{{col}}"] = float(fft_f[np.argmax(fft_mag)])
                s_skew = stats.skew(fft_mag)
                s_kurt = stats.kurtosis(fft_mag)
                row[f"spectral_skewness_{{col}}"] = float(s_skew if np.isfinite(s_skew) else 0.0)
                row[f"spectral_kurtosis_{{col}}"] = float(s_kurt if np.isfinite(s_kurt) else 0.0)

        # Sanitize NaN/Inf
        for k, v in row.items():
            if not np.isfinite(v):
                row[k] = 0.0

        all_features.append(row)

    df = pd.DataFrame(all_features)

    # Select only expected features in correct order
    if EXPECTED_FEATURES:
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        df = df[EXPECTED_FEATURES]

    return df.values


# ===== Step 5: Load Model and Predict =====
def load_model(model_path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("scaler"), data.get("classes", [])


def predict(model, scaler, features):
    if scaler is not None:
        features = scaler.transform(features)
    predictions = model.predict(features)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)
        except Exception:
            pass
    return predictions, probabilities


# ===== Main =====
def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.pkl> <data.csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]

    print("[1/5] Loading CSV...")
    df = load_csv(csv_path)
    raw_data = df[SENSOR_COLUMNS].values
    print(f"      Loaded {{len(raw_data)}} rows, {{len(SENSOR_COLUMNS)}} channels")

    print("[2/5] Windowing...")
    windows = apply_windowing(raw_data)
    print(f"      Created {{len(windows)}} windows (size={{WINDOW_SIZE}}, stride={{STRIDE}})")

    print("[3/5] Normalizing...")
    windows = normalize(windows)

    print("[4/5] Extracting features...")
    features = extract_features(windows)
    print(f"      Extracted {{features.shape[1]}} features per window")

    print("[5/5] Predicting...")
    model, scaler, classes = load_model(model_path)
    predictions, probabilities = predict(model, scaler, features)

    # Output results
    for i, pred in enumerate(predictions):
        line = f"Window {{i}}: {{pred}}"
        if probabilities is not None:
            conf = np.max(probabilities[i]) * 100
            line += f" ({{conf:.1f}}%)"
        print(line)

    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\\nSummary: {{dict(zip(unique.tolist(), counts.tolist()))}}")


if __name__ == "__main__":
    main()
'''

    def generate_deployment_package(self, saved_model: dict) -> Dict[str, Any]:
        """Generate a complete deployment package as a zip file.

        Args:
            saved_model: Dict from SavedModel.get_by_id()

        Returns:
            Dict with 'path' (zip file path) and 'filename'
        """
        import json
        import zipfile
        import tempfile

        pipeline_config = saved_model.get('pipeline_config', {})
        algorithm = saved_model.get('algorithm', 'unknown')
        mode = saved_model.get('mode', 'classification')
        model_path = saved_model.get('model_path', '')
        approach = pipeline_config.get('training_approach', 'ml')

        # Create temp zip
        safe_name = saved_model.get('name', 'model').replace(' ', '_')
        zip_filename = f"cira_deploy_{safe_name}.zip"
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Model file
            if model_path and os.path.exists(model_path):
                zf.write(model_path, 'model.pkl')

            # Pipeline config JSON
            zf.writestr('pipeline_config.json',
                         json.dumps(pipeline_config, indent=2, default=str))

            # Inference script
            if approach == 'dl':
                # For TimesNet, generate a simpler script (window + normalize + note about torch)
                script = self._generate_dl_inference_script(pipeline_config, algorithm)
            else:
                script = self.generate_full_inference_script(
                    pipeline_config, algorithm, mode)
            zf.writestr('inference.py', script)

            # Requirements
            if approach == 'dl':
                reqs = "numpy>=1.21\nscipy>=1.7\npandas>=1.3\ntorch>=2.0\n"
            else:
                reqs = "numpy>=1.21\nscipy>=1.7\npandas>=1.3\nscikit-learn>=1.0\n"
                if algorithm in ('iforest', 'lof', 'ocsvm', 'hbos', 'knn', 'copod', 'ecod'):
                    reqs += "pyod>=1.0\n"
            zf.writestr('requirements.txt', reqs)

        return {
            'path': zip_path,
            'filename': zip_filename,
            'size': os.path.getsize(zip_path),
        }

    def _generate_dl_inference_script(self, pipeline_config: dict,
                                       algorithm: str) -> str:
        """Generate inference script for TimesNet deployment."""
        norm = pipeline_config.get('normalization', {})
        wc = pipeline_config.get('windowing', {})
        sensor_columns = norm.get('sensor_columns', [])
        channel_min = norm.get('channel_min', [])
        channel_max = norm.get('channel_max', [])
        window_size = wc.get('window_size', 128)
        stride = wc.get('stride', 64)

        return f'''#!/usr/bin/env python3
"""
CiRA ME - TimesNet Inference Script
Algorithm: {algorithm}
Generated: {datetime.utcnow().isoformat()}

Usage: python inference.py <model.pkl> <data.csv>
NOTE: Requires PyTorch. Install with: pip install torch
"""

import sys
import pickle
import numpy as np
import pandas as pd

SENSOR_COLUMNS = {sensor_columns}
WINDOW_SIZE = {window_size}
STRIDE = {stride}
CHANNEL_MIN = np.array({channel_min})
CHANNEL_MAX = np.array({channel_max})


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {{missing}}")
    return df[SENSOR_COLUMNS].values


def apply_windowing(data):
    windows = []
    for i in range((len(data) - WINDOW_SIZE) // STRIDE + 1):
        start = i * STRIDE
        windows.append(data[start:start + WINDOW_SIZE])
    return np.array(windows)


def normalize(windows):
    ch_range = CHANNEL_MAX - CHANNEL_MIN
    ch_range[ch_range == 0] = 1.0
    return (windows - CHANNEL_MIN) / ch_range


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.pkl> <data.csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]

    print("[1/4] Loading CSV...")
    raw_data = load_csv(csv_path)
    print(f"      Loaded {{len(raw_data)}} rows, {{len(SENSOR_COLUMNS)}} channels")

    print("[2/4] Windowing...")
    windows = apply_windowing(raw_data)
    print(f"      Created {{len(windows)}} windows")

    print("[3/4] Normalizing...")
    windows = normalize(windows)

    print("[4/4] Predicting with TimesNet...")
    # Load model state and run inference
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    try:
        import torch
        # TimesNet inference requires the model architecture class
        # See the CiRA ME documentation for the full TimesNet model definition
        print("TimesNet model loaded. Implement model architecture for edge inference.")
        print(f"Config: {{model_data.get('config', {{}})}}")
        print(f"Windows shape: {{windows.shape}}")
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
