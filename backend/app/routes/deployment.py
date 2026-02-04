"""
CiRA ME - Deployment Routes
Handles model export and SSH deployment to edge devices (NVIDIA Jetson)
"""

from flask import Blueprint, request, jsonify
from ..auth import login_required, admin_required
from ..services.deployer import Deployer

deployment_bp = Blueprint('deployment', __name__)


@deployment_bp.route('/targets', methods=['GET'])
@login_required
def get_deployment_targets():
    """Get available deployment targets."""
    return jsonify({
        'targets': [
            {
                'id': 'jetson_nano',
                'name': 'NVIDIA Jetson Nano',
                'description': '4GB RAM, Maxwell GPU',
                'formats': ['onnx', 'tensorrt', 'pickle']
            },
            {
                'id': 'jetson_xavier',
                'name': 'NVIDIA Jetson Xavier NX',
                'description': '8GB RAM, Volta GPU',
                'formats': ['onnx', 'tensorrt', 'pickle']
            },
            {
                'id': 'raspberry_pi',
                'name': 'Raspberry Pi 4',
                'description': '4/8GB RAM, ARM Cortex-A72',
                'formats': ['onnx', 'pickle']
            },
            {
                'id': 'custom_ssh',
                'name': 'Custom SSH Target',
                'description': 'Any Linux device with SSH access',
                'formats': ['onnx', 'pickle', 'joblib']
            }
        ]
    })


@deployment_bp.route('/test-connection', methods=['POST'])
@login_required
def test_ssh_connection():
    """Test SSH connection to a deployment target."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    host = data.get('host')
    username = data.get('username')
    password = data.get('password')
    port = data.get('port', 22)
    key_path = data.get('key_path')

    if not host or not username:
        return jsonify({'error': 'Host and username required'}), 400

    try:
        deployer = Deployer()
        result = deployer.test_connection(
            host=host,
            username=username,
            password=password,
            port=port,
            key_path=key_path
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/deploy', methods=['POST'])
@login_required
def deploy_model():
    """Deploy a trained model to an edge device."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    training_session_id = data.get('training_session_id')
    target_type = data.get('target_type', 'custom_ssh')
    export_format = data.get('export_format', 'onnx')

    # SSH configuration
    ssh_config = {
        'host': data.get('host'),
        'username': data.get('username'),
        'password': data.get('password'),
        'port': data.get('port', 22),
        'key_path': data.get('key_path'),
        'remote_path': data.get('remote_path', '/home/user/models')
    }

    # Deployment options
    options = {
        'include_scaler': data.get('include_scaler', True),
        'include_inference_script': data.get('include_inference_script', True),
        'include_requirements': data.get('include_requirements', True)
    }

    if not training_session_id:
        return jsonify({'error': 'Training session ID required'}), 400

    if not ssh_config['host'] or not ssh_config['username']:
        return jsonify({'error': 'SSH host and username required'}), 400

    try:
        deployer = Deployer()
        result = deployer.deploy(
            training_session_id=training_session_id,
            target_type=target_type,
            export_format=export_format,
            ssh_config=ssh_config,
            options=options
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/status/<deployment_id>', methods=['GET'])
@login_required
def get_deployment_status(deployment_id: str):
    """Get the status of a deployment."""
    try:
        deployer = Deployer()
        result = deployer.get_status(deployment_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/generate-inference-script', methods=['POST'])
@login_required
def generate_inference_script():
    """Generate an inference script for a trained model."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    training_session_id = data.get('training_session_id')
    language = data.get('language', 'python')  # python, cpp

    if not training_session_id:
        return jsonify({'error': 'Training session ID required'}), 400

    try:
        deployer = Deployer()
        result = deployer.generate_inference_script(training_session_id, language)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
