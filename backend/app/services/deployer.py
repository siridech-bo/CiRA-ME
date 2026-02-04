"""
CiRA ME - Deployment Service
Handles model export and SSH deployment to edge devices
"""

import os
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from .ml_trainer import _model_sessions

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
        options: Dict
    ) -> Dict[str, Any]:
        """Deploy a trained model to an edge device via SSH."""
        try:
            import paramiko
            from scp import SCPClient
        except ImportError:
            raise ImportError("paramiko and scp libraries required. Install with: pip install paramiko scp")

        # Get model session
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

    def generate_inference_script(self, training_session_id: str, language: str = 'python') -> Dict[str, Any]:
        """Generate an inference script for a trained model."""
        session = _model_sessions.get(training_session_id)
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
