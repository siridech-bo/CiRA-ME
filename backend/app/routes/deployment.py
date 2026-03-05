"""
CiRA ME - Deployment Routes
Handles model export and SSH deployment to edge devices (NVIDIA Jetson)
"""

from flask import Blueprint, request, jsonify
from ..auth import login_required, admin_required
from ..services.deployer import Deployer, load_saved_model_session
from ..models import SavedModel

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
                'id': 'rdk_x5',
                'name': 'Horizon RDK X5',
                'description': '4GB RAM, BPU 10 TOPS',
                'formats': ['onnx', 'pickle']
            },
            {
                'id': 'ubuntu_x86',
                'name': 'Ubuntu x86 PC',
                'description': 'Any Linux x86_64 machine',
                'formats': ['onnx', 'pickle', 'joblib']
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
    saved_model_id = data.get('saved_model_id')
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
        'deploy_mode': data.get('deploy_mode', 'files'),  # 'files' | 'docker'
        'enable_gpu': data.get('enable_gpu', False),
        'jetpack_version': data.get('jetpack_version'),
    }

    if not training_session_id and not saved_model_id:
        return jsonify({'error': 'Training session ID or saved model ID required'}), 400

    if not ssh_config['host'] or not ssh_config['username']:
        return jsonify({'error': 'SSH host and username required'}), 400

    try:
        deployer = Deployer()
        saved_model_session = None

        # Load saved model from disk if saved_model_id provided
        if saved_model_id:
            saved_model = SavedModel.get_by_id(int(saved_model_id))
            if not saved_model:
                return jsonify({'error': f'Saved model not found: {saved_model_id}'}), 404
            saved_model_session = load_saved_model_session(
                saved_model['model_path'],
                algorithm=saved_model['algorithm'],
                mode=saved_model['mode']
            )
            # Use a synthetic session ID for the deployer
            training_session_id = f"saved_{saved_model_id}"

        pipeline_config = saved_model['pipeline_config'] if saved_model_id else {}
        result = deployer.deploy(
            training_session_id=training_session_id,
            target_type=target_type,
            export_format=export_format,
            ssh_config=ssh_config,
            options=options,
            saved_model_session=saved_model_session,
            pipeline_config=pipeline_config
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
    saved_model_id = data.get('saved_model_id')
    language = data.get('language', 'python')  # python, cpp

    if not training_session_id and not saved_model_id:
        return jsonify({'error': 'Training session ID or saved model ID required'}), 400

    try:
        deployer = Deployer()
        saved_model_session = None

        if saved_model_id:
            saved_model = SavedModel.get_by_id(int(saved_model_id))
            if not saved_model:
                return jsonify({'error': f'Saved model not found: {saved_model_id}'}), 404
            saved_model_session = load_saved_model_session(
                saved_model['model_path'],
                algorithm=saved_model['algorithm'],
                mode=saved_model['mode']
            )

        result = deployer.generate_inference_script(
            training_session_id or f"saved_{saved_model_id}",
            language,
            saved_model_session=saved_model_session
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/package/<int:model_id>', methods=['POST'])
@login_required
def generate_package(model_id):
    """Generate and download a deployment package for a saved model.
    Returns a zip file containing model, inference script, pipeline config, and requirements.
    """
    from flask import send_file

    saved = SavedModel.get_by_id(model_id)
    if not saved:
        return jsonify({'error': 'Model not found'}), 404

    pipeline_config = saved.get('pipeline_config', {})
    if not pipeline_config.get('normalization'):
        return jsonify({
            'error': 'Model missing pipeline config. '
                     'Re-save the model from a training session to enable package generation.'
        }), 400

    try:
        deployer = Deployer()
        result = deployer.generate_deployment_package(saved)

        return send_file(
            result['path'],
            as_attachment=True,
            download_name=result['filename'],
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/remote-files', methods=['POST'])
@login_required
def list_remote_files():
    """List files and containers on the remote deployment host."""
    data = request.get_json()
    ssh_config = {
        'host': data.get('host'),
        'username': data.get('username'),
        'password': data.get('password'),
        'port': data.get('port', 22),
    }
    remote_path = data.get('remote_path', '~/cira_models')
    try:
        deployer = Deployer()
        result = deployer.list_remote_files(ssh_config, remote_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/build-log', methods=['POST'])
@login_required
def get_build_log():
    """Fetch the Docker build log from the remote device."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    ssh_config = {
        'host': data.get('host'),
        'username': data.get('username'),
        'password': data.get('password'),
        'port': data.get('port', 22),
    }
    log_file = data.get('log_file')
    container_name = data.get('container_name')

    if not ssh_config['host'] or not log_file or not container_name:
        return jsonify({'error': 'host, log_file, and container_name required'}), 400

    try:
        deployer = Deployer()
        result = deployer.get_build_log(ssh_config, log_file, container_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@deployment_bp.route('/run-inference', methods=['POST'])
@login_required
def run_inference_remote():
    """Upload a CSV and run inference on the remote device."""
    ssh_host = request.form.get('host')
    ssh_username = request.form.get('username')
    ssh_password = request.form.get('password')
    ssh_port = int(request.form.get('port', 22))
    remote_path = request.form.get('remote_path', '~/cira_models')
    deploy_mode = request.form.get('deploy_mode', 'docker')
    service_name = request.form.get('service_name', 'inference')
    container_name_override = request.form.get('container_name', '').strip()

    if 'file' not in request.files:
        return jsonify({'error': 'No CSV file provided'}), 400
    f = request.files['file']
    csv_bytes = f.read()

    ssh_config = {
        'host': ssh_host, 'username': ssh_username,
        'password': ssh_password, 'port': ssh_port,
    }
    try:
        deployer = Deployer()
        result = deployer.run_inference_remote(
            ssh_config, remote_path, deploy_mode, service_name,
            csv_bytes, f.filename,
            container_name_override=container_name_override
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
