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
    Supports both pickle (.pkl) and ONNX (.onnx) formats.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    # Detect ONNX files by extension or magic bytes
    if model_path.endswith('.onnx'):
        return _load_onnx_session(model_path, algorithm, mode)

    # Try pickle first
    try:
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
    except Exception:
        # Maybe it's an ONNX file without .onnx extension
        return _load_onnx_session(model_path, algorithm, mode)


def _load_onnx_session(model_path: str, algorithm: str, mode: str) -> Dict[str, Any]:
    """Load an ONNX model file and wrap it in a session-compatible dict."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ValueError("onnxruntime not installed — cannot load ONNX model")

    ort_session = ort.InferenceSession(model_path)

    # Create a wrapper that matches sklearn's predict() interface
    class OnnxModelWrapper:
        def __init__(self, session):
            self.session = session
            self.input_name = session.get_inputs()[0].name
            self.input_shape = session.get_inputs()[0].shape

        def predict(self, X):
            import numpy as np
            X = np.array(X, dtype=np.float32)
            # ONNX models may expect specific shapes
            if len(self.input_shape) == 4 and X.ndim == 2:
                # TI NN model expects [batch, channels, window, 1]
                # but we have [batch, features] — need to reshape
                pass  # Use as-is, let ONNX handle it
            result = self.session.run(None, {self.input_name: X})
            return result[0].flatten()

    return {
        'model': OnnxModelWrapper(ort_session),
        'scaler': None,
        'algorithm': algorithm,
        'mode': mode,
        'model_path': model_path,
        'hyperparameters': {},
        'metrics': {},
        'is_onnx': True,
    }

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
        """Test SSH connection and auto-detect JetPack version, Python, nvidia runtime."""
        import re
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

            def _run(cmd):
                _, o, e = client.exec_command(cmd, timeout=10)
                o.channel.recv_exit_status()
                return o.read().decode('utf-8', errors='replace').strip()

            # Basic system info
            system_info = _run('uname -m && cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d \'"\'')

            # Detect Jetson / L4T
            l4t_release = _run('cat /etc/nv_tegra_release 2>/dev/null || echo ""')
            is_jetson = bool(l4t_release)
            jetpack_version = None
            l4t_revision = None
            if is_jetson:
                m = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', l4t_release)
                if m:
                    l4t_major = int(m.group(1))
                    l4t_revision = f'R{l4t_major}.{m.group(2)}'
                    if l4t_major == 32:
                        jetpack_version = '4.6'
                    elif l4t_major == 35:
                        jetpack_version = '5.x'
                    elif l4t_major == 36:
                        jetpack_version = '6.x'
                    else:
                        jetpack_version = f'L4T-R{l4t_major}'

            # Python version
            python_version = _run('python3 --version 2>&1 | head -1')

            # nvidia Docker runtime
            docker_runtimes = _run('docker info 2>/dev/null | grep -i "runtimes:"')
            nvidia_runtime = 'nvidia' in docker_runtimes.lower()

            # CUDA
            cuda_version = _run('nvcc --version 2>/dev/null | grep release | awk \'{print $5}\' | tr -d , || echo ""')

            # Disk + RAM
            disk_free = _run('df -h / | tail -1 | awk \'{print $4}\'')
            ram_free  = _run('free -h | grep Mem | awk \'{print $7}\'')

            return {
                'status': 'connected',
                'host': host,
                'message': 'Connection successful',
                'system_info': system_info,
                # Jetson-specific
                'is_jetson': is_jetson,
                'jetpack_version': jetpack_version,
                'l4t_revision': l4t_revision,
                # Runtime
                'python_version': python_version,
                'cuda_version': cuda_version if cuda_version else None,
                'nvidia_runtime': nvidia_runtime,
                # Resources
                'disk_free': disk_free,
                'ram_free': ram_free,
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

    def generate_dockerfile(self, approach: str, algorithm: str,
                            target_type: str = '', jetpack_version: str = None,
                            gpu: bool = False) -> str:
        """Generate a Dockerfile for the inference container."""
        is_dl = approach == 'dl' or algorithm.lower() == 'timesnet'
        is_jetson = 'jetson' in target_type.lower()

        # Jetson + DL + GPU → L4T PyTorch base image (torch already installed)
        if is_dl and is_jetson and gpu and jetpack_version:
            l4t_tags = {
                '4.6': 'r32.7.1-pth1.10-py3',
                '5.x': 'r35.2.1-pth2.0-py3',
                '6.x': 'r36.2.0-pth2.3.0-py3',
            }
            tag = l4t_tags.get(jetpack_version, 'r32.7.1-pth1.10-py3')
            return f'''FROM nvcr.io/nvidia/l4t-pytorch:{tag}

WORKDIR /app

# PyTorch already in L4T base — install only lightweight deps
RUN pip install --no-cache-dir numpy scipy pandas

# Copy model artifacts
COPY model.pkl inference.py pipeline_config.json ./

CMD ["tail", "-f", "/dev/null"]
'''

        # Standard Python slim image
        if is_dl:
            if is_jetson:
                # ARM64 (Jetson CPU) — pin torch<2.0 AND numpy<2.0.
                # torch 2.x uses ARMv8.2-A instructions (SDOT/UDOT) → SIGILL on
                # Cortex-A57/A53 (ARMv8.0-A, Jetson Nano).
                # torch 1.13.x was compiled against NumPy 1.x; NumPy 2.x breaks its
                # C ABI (_ARRAY_API not found → RuntimeError: "Could not infer dtype").
                pip_install = "RUN pip install --no-cache-dir 'numpy<2.0' scipy pandas 'torch<2.0'"
            else:
                # x86 — two separate RUN commands is REQUIRED here.
                # --index-url overrides ALL pip sources (replaces PyPI entirely), so
                # scipy/pandas/numpy cannot be found at download.pytorch.org/whl/cpu.
                # Solution: install PyPI packages first, then torch with its own index.
                pip_install = (
                    "RUN pip install --no-cache-dir numpy scipy pandas && \\\n"
                    "    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
                )
        else:
            pyod_algorithms = {'iforest', 'lof', 'ocsvm', 'hbos', 'knn', 'copod', 'ecod', 'suod'}
            base_pkgs = "numpy scipy pandas scikit-learn"
            if algorithm.lower() in pyod_algorithms:
                base_pkgs += " pyod"
            pip_install = f"RUN pip install --no-cache-dir {base_pkgs}"

        return f'''FROM python:3.10-slim

WORKDIR /app

# Install dependencies
{pip_install}

# Copy model artifacts
COPY model.pkl inference.py pipeline_config.json ./

CMD ["tail", "-f", "/dev/null"]
'''

    def generate_docker_compose(self, service_name: str, gpu: bool = False) -> str:
        """Generate a docker-compose.yml for the inference service."""
        safe_name = service_name.lower().replace(' ', '_').replace('-', '_')[:20]
        container_name = f'cira-{safe_name}'
        gpu_lines = (
            '    runtime: nvidia\n'
            '    environment:\n'
            '      - NVIDIA_VISIBLE_DEVICES=all\n'
        ) if gpu else ''
        return f'''services:
  {safe_name}:
    build: .
    container_name: {container_name}
    restart: unless-stopped
    volumes:
      - ./data:/data
    command: ["tail", "-f", "/dev/null"]
{gpu_lines}'''

    def deploy(
        self,
        training_session_id: str,
        target_type: str,
        export_format: str,
        ssh_config: Dict,
        options: Dict,
        saved_model_session: Dict = None,
        pipeline_config: Dict = None
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

        deploy_mode = options.get('deploy_mode', 'files')
        enable_gpu = options.get('enable_gpu', False)
        jetpack_version = options.get('jetpack_version')
        pipeline_config = pipeline_config or {}

        try:
            # Step 1: Locate model file
            deployment_status['steps'].append({'step': 'export', 'status': 'in_progress'})

            # For saved models use model_path directly (already a .pkl on disk)
            # For in-memory sessions, export via MLTrainer
            if saved_model_session and saved_model_session.get('model_path'):
                model_path = saved_model_session['model_path']
            else:
                from .ml_trainer import MLTrainer
                trainer = MLTrainer()
                export_result = trainer.export_model(training_session_id, export_format)
                model_path = export_result['path']

            deployment_status['steps'][-1]['status'] = 'completed'
            deployment_status['steps'][-1]['result'] = {'path': model_path}

            # Step 2: Generate inference script (+ Dockerfile if docker mode)
            deployment_status['steps'].append({'step': 'generate_script', 'status': 'in_progress'})

            algorithm = session.get('algorithm', 'unknown')
            mode = session.get('mode', 'classification')
            approach = pipeline_config.get('training_approach', 'ml')

            if pipeline_config.get('normalization'):
                # Full pipeline script (preferred — uses saved normalization/features)
                if approach == 'dl':
                    script_content = self._generate_dl_inference_script(pipeline_config, algorithm)
                else:
                    script_content = self.generate_full_inference_script(
                        pipeline_config, algorithm, mode)
            else:
                script_result = self.generate_inference_script(training_session_id, 'python',
                                                                saved_model_session=session)
                script_content = script_result['script']

            import tempfile, json as _json
            tmp_dir = tempfile.mkdtemp()
            script_path = os.path.join(tmp_dir, 'inference.py')
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Write pipeline_config.json
            config_path = os.path.join(tmp_dir, 'pipeline_config.json')
            with open(config_path, 'w') as f:
                _json.dump(pipeline_config, f, indent=2, default=str)

            if deploy_mode == 'docker':
                dockerfile_content = self.generate_dockerfile(
                    approach, algorithm, target_type, jetpack_version, enable_gpu)
                dockerfile_path = os.path.join(tmp_dir, 'Dockerfile')
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                # build_script_path written after remote_path is known (Step 4)

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

            remote_path = ssh_config.get('remote_path', '~/cira_models')
            # Expand ~ to absolute path on the remote
            _, out, _ = client.exec_command('echo $HOME')
            out.channel.recv_exit_status()
            home_dir = out.read().decode().strip()
            if home_dir:
                remote_path = remote_path.replace('~', home_dir)
            # Create directory and block until done
            _, stdout, stderr = client.exec_command(f'mkdir -p {remote_path}')
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err = stderr.read().decode().strip()
                raise RuntimeError(f"Failed to create remote directory {remote_path}: {err}")

            deployment_status['steps'][-1]['status'] = 'completed'

            # Compute safe_name here so it's available for build script + Step 6
            safe_name = algorithm.lower().replace(' ', '_').replace('-', '_')[:20] or 'model'

            if deploy_mode == 'docker':
                # Generate build script now that remote_path is resolved.
                # Uses plain docker build + docker run — no docker-compose dependency.
                container_name = f'cira-{safe_name}'
                log_file = f'/tmp/cira_build_{safe_name}.log'
                docker_cfg_path = f'/tmp/cira_dcfg_{safe_name}'
                gpu_run_flags = '--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all' if enable_gpu else ''
                build_script = (
                    '#!/bin/sh\n'
                    f'LOG="{log_file}"\n'
                    'echo "[CiRA] Build started at $(date)" >> "$LOG"\n'
                    'export DOCKER_BUILDKIT=0\n'
                    f'export DOCKER_CONFIG="{docker_cfg_path}"\n'
                    f'mkdir -p "{docker_cfg_path}"\n'
                    f'printf \'{{"auths":{{}}}}\' > "{docker_cfg_path}/config.json"\n'
                    f'cd "{remote_path}"\n'
                    f'echo "[CiRA] Building image cira-{safe_name}..." >> "$LOG"\n'
                    f'docker build -t cira-{safe_name} . >> "$LOG" 2>&1\n'
                    'BUILD_EXIT=$?\n'
                    'if [ $BUILD_EXIT -ne 0 ]; then\n'
                    '    echo "[CiRA] Build FAILED (exit $BUILD_EXIT)" >> "$LOG"\n'
                    '    exit 1\n'
                    'fi\n'
                    f'echo "[CiRA] Removing old container {container_name}..." >> "$LOG"\n'
                    f'docker stop {container_name} >> "$LOG" 2>&1 || true\n'
                    f'docker rm   {container_name} >> "$LOG" 2>&1 || true\n'
                    f'echo "[CiRA] Starting container {container_name}..." >> "$LOG"\n'
                    f'docker run -d \\\n'
                    f'    --name {container_name} \\\n'
                    f'    --restart unless-stopped \\\n'
                    f'    -v "{remote_path}/data:/data" \\\n'
                    f'    {gpu_run_flags} \\\n'
                    f'    cira-{safe_name} >> "$LOG" 2>&1\n'
                    'RUN_EXIT=$?\n'
                    'if [ $RUN_EXIT -eq 0 ]; then\n'
                    f'    echo "[CiRA] Container {container_name} started OK" >> "$LOG"\n'
                    f'    echo "{container_name}" > "{remote_path}/cira_container_name.txt"\n'
                    'else\n'
                    '    echo "[CiRA] docker run FAILED (exit $RUN_EXIT)" >> "$LOG"\n'
                    'fi\n'
                )
                build_script_path = os.path.join(tmp_dir, 'cira_build.sh')
                with open(build_script_path, 'w', newline='\n') as f:
                    f.write(build_script)

            # Step 5: Transfer files
            deployment_status['steps'].append({'step': 'transfer', 'status': 'in_progress'})

            scp = SCPClient(client.get_transport())

            # Always transfer model and inference script
            scp.put(model_path, os.path.join(remote_path, 'model.pkl'))
            scp.put(script_path, os.path.join(remote_path, 'inference.py'))
            scp.put(config_path, os.path.join(remote_path, 'pipeline_config.json'))

            if deploy_mode == 'docker':
                scp.put(dockerfile_path, os.path.join(remote_path, 'Dockerfile'))
                scp.put(build_script_path, os.path.join(remote_path, 'cira_build.sh'))
                # Create data directory on remote for input/output CSVs
                client.exec_command(f'mkdir -p {remote_path}/data')

            scp.close()
            deployment_status['steps'][-1]['status'] = 'completed'

            # Step 6: Validate / start container
            deployment_status['steps'].append({'step': 'validate', 'status': 'in_progress'})

            if deploy_mode == 'docker':
                # Run build script in background (nohup survives SSH disconnect)
                bg_cmd = (
                    f'chmod +x {remote_path}/cira_build.sh && '
                    f'nohup sh {remote_path}/cira_build.sh > {log_file} 2>&1 &'
                )
                _, bgout, _ = client.exec_command(bg_cmd, timeout=30)
                bgout.channel.recv_exit_status()

                # Poll for up to 60 s — covers re-deploys where image is already cached
                import time as _time
                container_started = False
                ps_cmd = (
                    f'docker ps --filter "name={container_name}" '
                    f'--filter "status=running" --format "{{{{.Names}}}}" 2>/dev/null'
                )
                for _ in range(12):
                    _time.sleep(5)
                    _, psout, _ = client.exec_command(ps_cmd, timeout=10)
                    psout.channel.recv_exit_status()
                    if container_name in psout.read().decode().strip():
                        container_started = True
                        break

                # Capture recent build log for UI
                _, logout, _ = client.exec_command(
                    f'tail -40 {log_file} 2>/dev/null || echo ""', timeout=10)
                logout.channel.recv_exit_status()
                build_log = logout.read().decode().strip()

                deployment_status['steps'][-1]['result'] = {
                    'container_started': container_started,
                    'build_log': build_log[-500:],
                    'log_file': log_file,
                }
            else:
                container_name = None
                container_started = True  # n/a for files mode
                log_file = None
                _, lstdout, _ = client.exec_command(f'ls -la {remote_path}')
                file_list = lstdout.read().decode('utf-8')
                deployment_status['steps'][-1]['result'] = {'files': file_list}

            deployment_status['steps'][-1]['status'] = 'completed'

            client.close()

            # Update final status
            deployment_status['status'] = 'completed'
            deployment_status['completed_at'] = datetime.utcnow().isoformat()
            deployment_status['remote_path'] = remote_path

            return {
                'deployment_id': deployment_id,
                'status': 'completed',
                'remote_path': remote_path,
                'deploy_mode': deploy_mode,
                'service_name': safe_name,
                'container_name': container_name,
                'container_started': container_started,
                'build_log_file': log_file,
                'steps': deployment_status['steps'],
                'message': 'Deployment successful'
            }

        except Exception as e:
            deployment_status['status'] = 'failed'
            deployment_status['error'] = str(e)
            deployment_status['completed_at'] = datetime.utcnow().isoformat()

            raise

    def get_build_log(self, ssh_config: Dict, log_file: str,
                      container_name: str) -> Dict[str, Any]:
        """Fetch the Docker build log from the remote device and check container status."""
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(ssh_config['host'], port=ssh_config.get('port', 22),
                           username=ssh_config['username'],
                           password=ssh_config.get('password'), timeout=10)

            def _run(cmd):
                _, o, _ = client.exec_command(cmd, timeout=15)
                o.channel.recv_exit_status()
                return o.read().decode('utf-8', errors='replace').strip()

            log_text = _run(f'tail -150 {log_file} 2>/dev/null || echo "(log file not found)"')
            ps_out = _run(
                f'docker ps --filter "name={container_name}" '
                f'--filter "status=running" --format "{{{{.Names}}}}" 2>/dev/null'
            )
            container_started = container_name in ps_out

            return {
                'log': log_text,
                'container_started': container_started,
                'container_name': container_name,
            }
        finally:
            client.close()

    def list_remote_files(self, ssh_config: Dict, remote_path: str) -> Dict[str, Any]:
        """List files in the remote deployment directory."""
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(ssh_config['host'], port=ssh_config.get('port', 22),
                           username=ssh_config['username'],
                           password=ssh_config.get('password'), timeout=10)
            _, out, _ = client.exec_command(f'ls -lh {remote_path} 2>&1')
            out.channel.recv_exit_status()
            files = out.read().decode()
            _, out2, _ = client.exec_command(
                f'docker ps --format "table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Image}}}}" 2>&1')
            out2.channel.recv_exit_status()
            containers = out2.read().decode()
            return {'files': files, 'containers': containers}
        finally:
            client.close()

    def run_inference_remote(self, ssh_config: Dict, remote_path: str,
                              deploy_mode: str, service_name: str,
                              csv_bytes: bytes, csv_filename: str,
                              container_name_override: str = '') -> Dict[str, Any]:
        """Upload a CSV to the remote device and run inference."""
        import paramiko
        from scp import SCPClient
        import tempfile

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(ssh_config['host'], port=ssh_config.get('port', 22),
                           username=ssh_config['username'],
                           password=ssh_config.get('password'), timeout=10)

            # Keep the SSH connection alive during slow operations (e.g. torch load on ARM).
            # Without keepalive, the transport silently drops when no SSH packets arrive
            # for > socket-timeout seconds (common on Jetson where torch import takes 30-60 s).
            client.get_transport().set_keepalive(30)

            # Write CSV to temp file and SCP to remote data/input.csv
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                tmp.write(csv_bytes)
                tmp_path = tmp.name

            scp = SCPClient(client.get_transport())
            remote_csv = os.path.join(remote_path, 'data', 'input.csv')
            # Ensure data dir exists
            _, out, _ = client.exec_command(f'mkdir -p {remote_path}/data')
            out.channel.recv_exit_status()
            scp.put(tmp_path, remote_csv)
            scp.close()
            os.unlink(tmp_path)

            # Run a command over SSH and return (stdout, stderr).
            # Uses threads so both streams are drained concurrently — prevents buffer
            # deadlock when stderr fills up while we are waiting for exit status.
            # No channel timeout is set; the thread join timeout acts as the wall-clock limit.
            def _ssh_run(cmd, timeout=120):
                import threading
                _, o, e = client.exec_command(cmd)   # no per-channel timeout
                out_buf, err_buf = [''], ['']

                def _read(fh, buf):
                    try:
                        buf[0] = fh.read().decode('utf-8', errors='replace')
                    except Exception:
                        pass

                t_out = threading.Thread(target=_read, args=(o, out_buf), daemon=True)
                t_err = threading.Thread(target=_read, args=(e, err_buf), daemon=True)
                t_out.start(); t_err.start()
                t_out.join(timeout); t_err.join(timeout)
                return out_buf[0], err_buf[0]

            # Names that are CiRA infrastructure (not inference containers)
            _INFRA_NAMES = {'cira-runtime', 'cira-backend', 'cira-frontend',
                            'cira-nginx', 'cira-proxy', 'cira-db', 'cira-redis'}

            if deploy_mode == 'docker':
                # Resolve container name: override > saved file > service_name fallback
                if (container_name_override
                        and container_name_override not in ('cira-', 'cira')
                        and container_name_override not in _INFRA_NAMES):
                    container_name = container_name_override
                else:
                    # Primary: read the name saved by cira_build.sh
                    _, fnout, _ = client.exec_command(
                        f'cat "{remote_path}/cira_container_name.txt" 2>/dev/null', timeout=10)
                    fnout.channel.recv_exit_status()
                    saved_name = fnout.read().decode().strip()
                    if (saved_name and saved_name.startswith('cira-')
                            and len(saved_name) > 5
                            and saved_name not in _INFRA_NAMES):
                        container_name = saved_name
                    else:
                        # Fallback: derive from service_name
                        safe_sn = service_name.lower().replace(' ', '_').replace('-', '_')[:20]
                        container_name = f'cira-{safe_sn}' if safe_sn else 'cira-inference'

                # Check if container is running
                ps_cmd = (
                    f'docker ps --filter "name={container_name}" '
                    f'--filter "status=running" --format "{{{{.Names}}}}" 2>/dev/null'
                )
                _, csout, _ = client.exec_command(ps_cmd, timeout=10)
                csout.channel.recv_exit_status()
                is_running = container_name in csout.read().decode().strip()

                if not is_running:
                    # Image may be cached — try docker start (no rebuild needed)
                    _, stout, _ = client.exec_command(
                        f'docker start {container_name} 2>&1', timeout=30)
                    stout.channel.recv_exit_status()
                    # Re-check
                    _, csout2, _ = client.exec_command(ps_cmd, timeout=10)
                    csout2.channel.recv_exit_status()
                    if container_name not in csout2.read().decode().strip():
                        raise RuntimeError(
                            f"Container '{container_name}' is not running on the remote device. "
                            f"The Docker image may still be building — please use "
                            f"'Watch build log' to monitor progress, then retry inference."
                        )

                # Verify /data/input.csv is visible inside the container
                # (comes from the volume mount; if the mount failed it won't exist).
                data_check, _ = _ssh_run(
                    f'docker exec {container_name} ls /data/input.csv 2>&1', timeout=15)
                if 'No such file' in data_check or not data_check.strip():
                    raise RuntimeError(
                        f'/data/input.csv not found inside container {container_name}. '
                        f'The volume mount may have failed — please redeploy the model.'
                    )

                # Quick sanity-check: verify docker exec can capture python3 output.
                ping_out, ping_err = _ssh_run(
                    f"docker exec {container_name} python3 -c \"print('CiRA_OK')\"",
                    timeout=15)
                if 'CiRA_OK' not in ping_out and 'CiRA_OK' not in ping_err:
                    raise RuntimeError(
                        f"docker exec output capture not working on this device.\n"
                        f"python3 echo test: stdout={ping_out!r} stderr={ping_err!r}"
                    )

                # Use sh -c '... 2>&1' to merge Python stderr into stdout INSIDE
                # the container before it hits the docker exec pipe — ensures
                # tracebacks are always captured.  timeout=300 covers torch
                # cold-start on ARM (30-60 s with no output).
                infer_cmd = (
                    f"docker exec {container_name} "
                    f"sh -c 'python3 -u /app/inference.py /app/model.pkl /data/input.csv 2>&1'"
                )
                stdout_data, stderr_data = _ssh_run(infer_cmd, timeout=300)

                if 'No such container' in stderr_data or 'Error response from daemon' in stderr_data:
                    raise RuntimeError(
                        f"Docker error on remote: {stderr_data.strip()}\n"
                        f"Please redeploy the model and wait for the container to be ready."
                    )
            else:
                infer_cmd = f'cd {remote_path} && python3 -u inference.py model.pkl data/input.csv'
                stdout_data, stderr_data = _ssh_run(infer_cmd)

                # Detect numpy/sklearn version mismatch (model saved on newer numpy 2.x)
                needs_upgrade = (
                    "numpy._core" in stderr_data or
                    "No module named 'numpy._core'" in stderr_data or
                    "binary incompatibility" in stderr_data or
                    "ModuleNotFoundError" in stderr_data
                )
                if needs_upgrade:
                    # Upgrade to numpy-2.x-compatible stack (works on Python 3.10+)
                    upgrade_cmd = (
                        "pip3 install -q 'numpy>=2.0' 'pandas>=2.2' "
                        "'scikit-learn>=1.5,<1.8' 'scipy>=1.13' && "
                        f"cd {remote_path} && python3 -u inference.py model.pkl data/input.csv"
                    )
                    stdout_data, stderr_data = _ssh_run(upgrade_cmd, timeout=300)

            output = stdout_data
            if stderr_data:
                output = (stdout_data + '\n[stderr]\n' + stderr_data).strip()
            if not output:
                output = 'No output from inference'

            parsed = self._parse_inference_output(output)
            result = {'success': True, 'output': output, 'csv': csv_filename, **parsed}
            # Return resolved container_name so frontend can update its state
            if deploy_mode == 'docker':
                result['container_name'] = container_name
            return result
        finally:
            client.close()

    def _parse_inference_output(self, output: str) -> dict:
        """Parse raw inference.py stdout into structured metrics."""
        import re, ast
        result = {
            'num_windows': None,
            'prediction_distribution': None,
            'avg_confidence': None,
            'num_features': None,
        }

        # Windows created
        m = re.search(r'Created (\d+) windows', output)
        if m:
            result['num_windows'] = int(m.group(1))

        # Features extracted
        m = re.search(r'Extracted (\d+) features', output)
        if m:
            result['num_features'] = int(m.group(1))

        # Summary dict e.g. Summary: {'Normal': 45, 'Anomaly': 12}
        m = re.search(r'Summary:\s*(\{[^\}]+\})', output)
        if m:
            try:
                result['prediction_distribution'] = ast.literal_eval(m.group(1))
            except Exception:
                pass

        # Confidence values from "Window N: Label (XX.X%)"
        confidences = [float(x) for x in re.findall(r'\((\d+\.\d+)%\)', output)]
        if confidences:
            result['avg_confidence'] = round(sum(confidences) / len(confidences), 1)

        return result

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

        if mode == 'regression':
            return f'''#!/usr/bin/env python3
"""
CiRA ME - Regression Inference Script
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
    Make regression predictions.

    Args:
        model: Trained regression model
        scaler: StandardScaler for feature normalization
        features: numpy array of shape (n_samples, n_features)

    Returns:
        predictions: Predicted continuous values
    """
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    return predictions

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
    predictions = predict(model, scaler, features)

    # Output results
    for i, pred in enumerate(predictions):
        print(f"Sample {{i}}: {{pred:.4f}}")

    print(f"\\nPrediction statistics:")
    print(f"  Mean:  {{np.mean(predictions):.4f}}")
    print(f"  Std:   {{np.std(predictions):.4f}}")
    print(f"  Min:   {{np.min(predictions):.4f}}")
    print(f"  Max:   {{np.max(predictions):.4f}}")

if __name__ == "__main__":
    main()
'''
        elif mode == 'anomaly':
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
    if MODE == "regression":
        for i, pred in enumerate(predictions):
            print(f"Window {{i}}: {{pred:.4f}}")
        print(f"\\nPrediction statistics:")
        print(f"  Mean:  {{np.mean(predictions):.4f}}")
        print(f"  Std:   {{np.std(predictions):.4f}}")
        print(f"  Min:   {{np.min(predictions):.4f}}")
        print(f"  Max:   {{np.max(predictions):.4f}}")
    else:
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
        """Generate a self-contained TimesNet inference script with embedded architecture.

        Architectures and state-dict format exactly match torch_subprocess.py:
          build_timesnet_encoder()     — anomaly detection
          build_timesnet_classifier()  — classification
        Weights are stored as plain Python lists (via .tolist()) nested one level deep
        under model_data["model_state"]["model_state_dict"], and must be converted back
        to tensors before load_state_dict().
        """
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
Requires: torch, numpy, pandas  (pip install torch numpy pandas)
"""

import sys
import pickle
import numpy as np
import pandas as pd

SENSOR_COLUMNS = {sensor_columns}
WINDOW_SIZE = {window_size}
STRIDE = {stride}
CHANNEL_MIN = np.array({channel_min}, dtype=np.float32)
CHANNEL_MAX = np.array({channel_max}, dtype=np.float32)


# ===== Exact TimesNet architectures (mirrors torch_subprocess.py) =====

def _build_encoder(cfg):
    """Anomaly encoder — matches build_timesnet_encoder() in torch_subprocess.py."""
    import torch.nn as nn
    enc_in  = cfg.get("enc_in",  3)
    d_model = cfg.get("d_model", 64)
    d_ff    = cfg.get("d_ff",    128)
    dropout = cfg.get("dropout", 0.1)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed   = nn.Linear(enc_in, d_model)
            self.encoder = nn.Sequential(
                nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1),
            )
            self.projection = nn.Linear(d_model, enc_in)

        def forward(self, x):
            x = self.embed(x).transpose(1, 2)
            x = self.decoder(self.encoder(x)).transpose(1, 2)
            return self.projection(x)

    return Encoder()


def _build_classifier(cfg, num_classes):
    """Classifier — matches build_timesnet_classifier() in torch_subprocess.py."""
    import torch.nn as nn
    enc_in  = cfg.get("enc_in",  3)
    d_model = cfg.get("d_model", 64)
    d_ff    = cfg.get("d_ff",    128)
    dropout = cfg.get("dropout", 0.1)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Linear(enc_in, d_model)
            self.encoder    = nn.Sequential(
                nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.pool       = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, num_classes),
            )

        def forward(self, x):
            x = self.embed(x).transpose(1, 2)
            x = self.pool(self.encoder(x)).squeeze(-1)
            return self.classifier(x)

    return Classifier()


def _load_state_dict(model_data):
    """
    Weights are stored as Python lists (via .tolist()) one level deep:
      model_data["model_state"]["model_state_dict"] -> dict of lists
    Convert to float32 tensors before load_state_dict().
    """
    import torch
    inner = model_data.get("model_state", {{}})
    raw   = inner.get("model_state_dict", {{}})
    return {{k: torch.tensor(np.array(v, dtype=np.float32)) for k, v in raw.items()}}


# ===== Pipeline =====

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {{missing}}")
    return df[SENSOR_COLUMNS].values.astype(np.float32)


def apply_windowing(data):
    windows = []
    for i in range((len(data) - WINDOW_SIZE) // STRIDE + 1):
        start = i * STRIDE
        windows.append(data[start:start + WINDOW_SIZE])
    return np.array(windows, dtype=np.float32)


def normalize(windows):
    ch_range = CHANNEL_MAX - CHANNEL_MIN
    ch_range[ch_range == 0] = 1.0
    return (windows - CHANNEL_MIN) / ch_range


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.pkl> <data.csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path   = sys.argv[2]

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not found. Install with: pip install torch")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {{device}}")

    print("[1/4] Loading CSV...")
    raw_data = load_csv(csv_path)
    print(f"      Loaded {{len(raw_data)}} rows, {{len(SENSOR_COLUMNS)}} channels")

    print("[2/4] Windowing...")
    windows = apply_windowing(raw_data)
    print(f"      Created {{len(windows)}} windows (size={{WINDOW_SIZE}}, stride={{STRIDE}})")

    print("[3/4] Normalizing...")
    windows = normalize(windows)

    print("[4/4] Predicting with TimesNet...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    inner      = model_data.get("model_state", {{}})
    cfg        = inner.get("config",  model_data.get("config",  {{}}))
    mode       = inner.get("mode",    model_data.get("mode",    "anomaly"))
    state_dict = _load_state_dict(model_data)
    x          = torch.FloatTensor(windows).to(device)

    with torch.no_grad():
        if mode == "classification":
            class_names = inner.get("label_encoder_classes",
                                    model_data.get("label_encoder_classes", []))
            num_cls = len(class_names) if class_names else cfg.get("num_classes", cfg.get("num_class", 2))
            model = _build_classifier(cfg, num_cls)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            probs  = torch.softmax(model(x), dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            labels = [class_names[p] if p < len(class_names) else str(p) for p in preds]

            for i, (label, prob) in enumerate(zip(labels, probs)):
                print(f"Window {{i}}: {{label}} ({{float(np.max(prob))*100:.1f}}%)")

            unique, counts = np.unique(labels, return_counts=True)
            print(f"\\nSummary: {{dict(zip(unique.tolist(), counts.tolist()))}}")

        else:  # anomaly
            threshold = float(inner.get("threshold", model_data.get("threshold", 0.5)))
            model = _build_encoder(cfg)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            recon  = model(x).cpu().numpy()
            errors = np.mean((windows - recon) ** 2, axis=(1, 2))
            labels = ["Anomaly" if e > threshold else "Normal" for e in errors]

            for i, (label, err) in enumerate(zip(labels, errors)):
                conf = min(abs(float(err) - threshold) / (threshold + 1e-9) * 50 + 50, 99.9)
                print(f"Window {{i}}: {{label}} ({{conf:.1f}}%)")

            unique, counts = np.unique(labels, return_counts=True)
            print(f"\\nSummary: {{dict(zip(unique.tolist(), counts.tolist()))}}")


if __name__ == "__main__":
    main()
'''
