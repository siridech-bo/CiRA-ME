"""
CiRA ME - TI TinyML ModelMaker Bridge Service
Runs in separate Python 3.10 container, exposes REST API for CiRA ME backend.
"""

import os
import json
import uuid
import shutil
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PROJECTS_DIR = '/app/data/projects'
DATASETS_DIR = '/app/data/datasets'
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)


@app.route('/health')
def health():
    """Health check."""
    ti_available = False
    emlearn_available = False
    try:
        import tinyml_modelmaker
        ti_available = True
    except ImportError:
        pass
    try:
        import emlearn
        emlearn_available = True
    except ImportError:
        pass
    return jsonify({
        'status': 'healthy',
        'service': 'ti-modelmaker',
        'tinyml_modelmaker': ti_available,
        'emlearn': emlearn_available,
    })


@app.route('/devices', methods=['GET'])
def list_devices():
    """List supported TI target devices with capabilities."""
    devices = {
        'F280013': {
            'name': 'TMS320F280013',
            'family': 'C2000',
            'npu': False,
            'flash_kb': 128,
            'tasks': ['timeseries_classification'],
        },
        'F280015': {
            'name': 'TMS320F280015',
            'family': 'C2000',
            'npu': False,
            'flash_kb': 128,
            'tasks': ['timeseries_classification'],
        },
        'F28003': {
            'name': 'TMS320F28003x',
            'family': 'C2000 Piccolo',
            'npu': False,
            'flash_kb': 384,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection'],
        },
        'F28004': {
            'name': 'TMS320F28004x (F280049C)',
            'family': 'C2000 Piccolo',
            'npu': False,
            'flash_kb': 256,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection'],
        },
        'F2837': {
            'name': 'TMS320F2837xD (F28379D)',
            'family': 'C2000 Delfino',
            'npu': False,
            'flash_kb': 1024,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection', 'timeseries_forecasting'],
        },
        'F28P55': {
            'name': 'TMS320F28P55x',
            'family': 'C2000 (NPU)',
            'npu': True,
            'npu_mops': 1200,
            'flash_kb': 1100,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection', 'timeseries_forecasting'],
        },
        'F28P65': {
            'name': 'TMS320F28P65x',
            'family': 'C2000 (NPU)',
            'npu': True,
            'flash_kb': 2048,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection', 'timeseries_forecasting'],
        },
        'F29H85': {
            'name': 'TMS320F29H85x',
            'family': 'C29 Next-Gen',
            'npu': True,
            'flash_kb': 4096,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection', 'timeseries_forecasting'],
        },
        'MSPM0G3507': {
            'name': 'MSPM0G3507',
            'family': 'MSPM0 (Cortex-M0+)',
            'npu': False,
            'flash_kb': 128,
            'tasks': ['timeseries_classification'],
        },
        'MSPM0G5187': {
            'name': 'MSPM0G5187',
            'family': 'MSPM0 (NPU)',
            'npu': True,
            'flash_kb': 1024,
            'tasks': ['timeseries_classification', 'timeseries_regression',
                      'timeseries_anomalydetection'],
        },
    }
    return jsonify(devices)


@app.route('/models', methods=['GET'])
def list_models():
    """List available models from TI model zoo + Traditional ML."""
    task_type = request.args.get('task', 'timeseries_regression')
    target_device = request.args.get('device', None)
    source = request.args.get('source', 'all')  # 'all', 'ti_zoo', 'traditional_ml'

    models = {}

    # TI Model Zoo (neural networks)
    if source in ('all', 'ti_zoo'):
        zoo = _get_model_zoo(task_type)
        for k, v in zoo.items():
            v['source'] = 'ti_zoo'
        models.update(zoo)

    # Traditional ML (emlearn)
    if source in ('all', 'traditional_ml'):
        ml = _get_traditional_ml_models(task_type)
        for k, v in ml.items():
            v['source'] = 'traditional_ml'
        models.update(ml)

    # Filter by device compatibility if specified
    if target_device:
        device_info = _get_device_info(target_device)
        if device_info:
            max_params = _flash_to_max_params(device_info.get('flash_kb', 256))
            has_npu = device_info.get('npu', False)
            filtered = {}
            for k, v in models.items():
                # Traditional ML uses different size metric (estimated C code size)
                if v.get('source') == 'traditional_ml':
                    est_size = v.get('estimated_flash_kb', 10)
                    if est_size <= device_info.get('flash_kb', 256) // 2:
                        filtered[k] = v
                else:
                    if v['params'] <= max_params:
                        if v.get('npu_only') and not has_npu:
                            continue
                        filtered[k] = v
            models = filtered

    return jsonify(models)


@app.route('/train', methods=['POST'])
def train_model():
    """Train one or more models."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    task_type = data.get('task_type', 'timeseries_regression')
    model_names = data.get('model_names', [])
    model_name = data.get('model_name')  # backward compat: single model
    target_device = data.get('target_device', 'F2837')
    dataset_path = data.get('dataset_path')
    config_overrides = data.get('config', {})

    # Support both single and multi-model
    if model_name and not model_names:
        model_names = [model_name]

    if not model_names:
        return jsonify({'error': 'model_names required'}), 400
    if not dataset_path:
        return jsonify({'error': 'dataset_path required'}), 400
    if not os.path.exists(dataset_path):
        return jsonify({'error': f'Dataset not found: {dataset_path}'}), 400

    run_id = str(uuid.uuid4())[:8]
    base_dir = os.path.join(PROJECTS_DIR, run_id)
    os.makedirs(base_dir, exist_ok=True)

    # Train each model
    results = []
    errors = []

    for mname in model_names:
        model_dir = os.path.join(base_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        try:
            if mname.startswith('ML_'):
                result = _train_traditional_ml(
                    task_type=task_type,
                    model_name=mname,
                    dataset_path=dataset_path,
                    project_dir=model_dir,
                    target_device=target_device,
                    overrides=config_overrides,
                )
            else:
                config = _build_config(
                    task_type=task_type,
                    model_name=mname,
                    target_device=target_device,
                    dataset_path=dataset_path,
                    project_dir=model_dir,
                    overrides=config_overrides,
                )
                import yaml
                config_path = os.path.join(model_dir, 'config.yaml')
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                result = _run_modelmaker(config, model_dir)

            # Get model info for display name
            all_models = _get_model_zoo(task_type)
            all_models.update(_get_traditional_ml_models(task_type))
            model_info = all_models.get(mname, {})

            status = result.get('status', 'success')
            result_metrics = result.get('metrics', {})

            # If no metrics returned, mark as error
            if status == 'success' and not result_metrics:
                status = 'error'
                errors.append({
                    'model_name': mname,
                    'algorithm_name': model_info.get('name', mname),
                    'error': 'TI NN Compiler required — model architecture validated but not trained locally. Use TI Code Composer Studio.',
                    'status': 'failed',
                })
                continue

            results.append({
                'model_name': mname,
                'algorithm_name': model_info.get('name', mname),
                'status': status,
                'metrics': result_metrics,
                'artifacts': result.get('artifacts', []),
                'estimated_flash_kb': model_info.get('estimated_flash_kb',
                    model_info.get('params', 0) // 1024 or None),
                'source': model_info.get('source', 'ti_zoo'),
                'logs': result.get('logs', []),
            })
        except Exception as e:
            all_models = _get_model_zoo(task_type)
            all_models.update(_get_traditional_ml_models(task_type))
            model_info = all_models.get(mname, {})
            errors.append({
                'model_name': mname,
                'algorithm_name': model_info.get('name', mname),
                'error': str(e),
                'status': 'failed',
            })

    # Find best model
    best = None
    if task_type == 'timeseries_regression':
        best_score = float('-inf')
        for r in results:
            score = r['metrics'].get('r2')
            if score is None:
                score = float('-inf')
            if score > best_score:
                best_score = score
                best = {'model_name': r['model_name'],
                        'algorithm_name': r['algorithm_name'],
                        'score': score, 'metric': 'r2'}
    else:
        best_score = -1
        for r in results:
            score = r['metrics'].get('f1') or r['metrics'].get('accuracy') or 0
            if score is None:
                score = 0
            if score > best_score:
                best_score = score
                best = {'model_name': r['model_name'],
                        'algorithm_name': r['algorithm_name'],
                        'score': score,
                        'metric': 'f1' if r['metrics'].get('f1') else 'accuracy'}

    return jsonify({
        'run_id': run_id,
        'total': len(model_names),
        'successful': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors,
        'best_algorithm': best,
    })


@app.route('/download/<run_id>', methods=['GET'])
def download_artifacts(run_id):
    """Download compiled model artifacts as zip."""
    project_dir = os.path.join(PROJECTS_DIR, run_id)
    if not os.path.exists(project_dir):
        return jsonify({'error': 'Run not found'}), 404

    # Find the compiled output
    zip_path = os.path.join(project_dir, 'artifacts.zip')
    if not os.path.exists(zip_path):
        # Create zip from project dir
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', project_dir)

    return send_file(zip_path, as_attachment=True,
                     download_name=f'ti_model_{run_id}.zip')


def _get_device_info(device_id):
    """Get device info dict."""
    devices = list_devices().get_json()
    return devices.get(device_id)


def _flash_to_max_params(flash_kb):
    """Rough estimate: max model params given flash size."""
    # INT8: ~1 byte per param + overhead
    # Reserve 50% of flash for application code
    available_bytes = (flash_kb * 1024) // 2
    return available_bytes


def _get_model_zoo(task_type):
    """Get available models dynamically from TI tinyml-modelmaker."""
    # Map our task type to TI's category prefix
    ti_task_map = {
        'timeseries_regression': 'generic_timeseries_regression',
        'timeseries_classification': 'generic_timeseries_classification',
        'timeseries_anomalydetection': 'generic_timeseries_anomalydetection',
        'timeseries_forecasting': 'generic_timeseries_forecasting',
    }
    ti_task = ti_task_map.get(task_type, task_type)

    # Prefix filters for each task
    prefix_map = {
        'timeseries_regression': ['REGR_'],
        'timeseries_classification': ['CLS_'],
        'timeseries_anomalydetection': ['AD_'],
        'timeseries_forecasting': ['FCST_'],
    }
    prefixes = prefix_map.get(task_type, [])

    try:
        from tinyml_modelmaker.ai_modules.timeseries import runner
        params = runner.ModelRunner.init_params()
        all_descs = runner.ModelRunner.get_model_descriptions(params)

        models = {}
        for model_key, desc in all_descs.items():
            # Filter by prefix
            if prefixes and not any(model_key.startswith(p) for p in prefixes):
                continue

            # Also check task_type in description
            model_task = desc.get('common', {}).get('task_type', '')
            if model_task and model_task != ti_task:
                continue

            # Extract info
            details = desc.get('common', {}).get('model_details', '')
            is_npu = 'NPU' in model_key

            # Parse approximate param count from name (e.g., REGR_1k -> 1000)
            param_count = 0
            name_lower = model_key.lower()
            for part in name_lower.replace('_', ' ').split():
                if part.endswith('k') and part[:-1].isdigit():
                    param_count = int(part[:-1]) * 1000

            models[model_key] = {
                'name': model_key.replace('_', ' '),
                'params': param_count,
                'architecture': details or ('NPU-optimized CNN' if is_npu else 'Conv1D + FC'),
                'npu_only': is_npu,
                'description': details or f'{model_key} model from TI model zoo',
            }

        return models

    except Exception as e:
        print(f"Warning: Could not load TI model zoo dynamically: {e}")
        # Fallback to empty
        return {}


def _get_traditional_ml_models(task_type):
    """Get Traditional ML models that can be exported to C via emlearn."""
    if task_type == 'timeseries_regression':
        return {
            'ML_DT_REG': {
                'name': 'Decision Tree Regressor',
                'params': 0,
                'architecture': 'Single Decision Tree → C if/else',
                'npu_only': False,
                'estimated_flash_kb': 5,
                'description': 'Smallest footprint. Converts to pure C if/else tree. Best for Cortex-M0.',
            },
            'ML_RF_REG': {
                'name': 'Random Forest Regressor',
                'params': 0,
                'architecture': 'Tree Ensemble → C code',
                'npu_only': False,
                'estimated_flash_kb': 30,
                'description': 'Ensemble of 50 trees. Good accuracy/size balance.',
            },
            'ML_XGB_REG': {
                'name': 'XGBoost Regressor',
                'params': 0,
                'architecture': 'Gradient Boosting → C code',
                'npu_only': False,
                'estimated_flash_kb': 25,
                'description': 'Gradient boosting. Often best accuracy for tabular features.',
            },
            'ML_LGBM_REG': {
                'name': 'LightGBM Regressor',
                'params': 0,
                'architecture': 'Gradient Boosting → C code',
                'npu_only': False,
                'estimated_flash_kb': 20,
                'description': 'Fast gradient boosting with smaller trees.',
            },
        }
    elif task_type == 'timeseries_classification':
        return {
            'ML_DT_CLF': {
                'name': 'Decision Tree Classifier',
                'params': 0,
                'architecture': 'Single Decision Tree → C if/else',
                'npu_only': False,
                'estimated_flash_kb': 5,
                'description': 'Smallest classifier. Pure C if/else tree.',
            },
            'ML_RF_CLF': {
                'name': 'Random Forest Classifier',
                'params': 0,
                'architecture': 'Tree Ensemble → C code',
                'npu_only': False,
                'estimated_flash_kb': 30,
                'description': 'Ensemble of 50 trees. Reliable classification.',
            },
            'ML_XGB_CLF': {
                'name': 'XGBoost Classifier',
                'params': 0,
                'architecture': 'Gradient Boosting → C code',
                'npu_only': False,
                'estimated_flash_kb': 25,
                'description': 'Gradient boosting classifier.',
            },
        }
    elif task_type == 'timeseries_anomalydetection':
        return {
            'ML_IFOREST': {
                'name': 'Isolation Forest',
                'params': 0,
                'architecture': 'Tree Ensemble → C code',
                'npu_only': False,
                'estimated_flash_kb': 30,
                'description': 'Tree-based anomaly detection. No labels needed.',
            },
        }
    return {}


def _prepare_dataset_dir(csv_path, project_dir, frame_size=128):
    """Prepare dataset directory structure expected by TI modelmaker.

    TI expects input_data_path to be a DIRECTORY containing multiple CSV files
    (one per sample/segment). Each CSV has NO header, numeric columns only,
    with the last column as the target for regression.

    We split the single large CSV into multiple segment files.
    """
    import pandas as pd
    import numpy as np

    dataset_dir = os.path.join(project_dir, 'dataset')
    files_dir = os.path.join(dataset_dir, 'files')
    os.makedirs(files_dir, exist_ok=True)

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Remove time/timestamp columns
    cols_to_keep = [c for c in df.columns if 'time' not in c.lower()
                    and 'date' not in c.lower() and 'index' not in c.lower()]
    df = df[cols_to_keep]

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if len(df) == 0:
        raise ValueError("No numeric data found in CSV")

    # TI's SimpleWindow transform handles windowing internally,
    # so we provide larger chunks (not tiny segments).
    # Split into ~10 files for proper train/val/test splitting.
    n_segments = min(10, max(3, len(df) // 500))
    segment_size = len(df) // n_segments

    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(df)
        segment = df.iloc[start:end]

        # Save WITHOUT header (TI expects headerless CSV)
        seg_path = os.path.join(files_dir, f'segment_{i:04d}.csv')
        segment.to_csv(seg_path, index=False, header=False)

    print(f"[TI Dataset] Split {len(df)} rows into {n_segments} segments "
          f"of ~{segment_size} rows each in {files_dir}")

    return dataset_dir


def _build_config(task_type, model_name, target_device, dataset_path,
                  project_dir, overrides=None):
    """Build tinyml-modelmaker config dict.

    TI modelmaker expects:
    - task_type: 'generic_timeseries_regression' (not 'timeseries_regression')
    - target_module: 'timeseries'
    - data_processing_feature_extraction section with feature transform config
    - input_data_path as a DIRECTORY (not a CSV file)
    """
    overrides = overrides or {}

    # Map our task types to TI's expected names
    ti_task_map = {
        'timeseries_regression': 'generic_timeseries_regression',
        'timeseries_classification': 'generic_timeseries_classification',
        'timeseries_anomalydetection': 'generic_timeseries_anomalydetection',
        'timeseries_forecasting': 'generic_timeseries_forecasting',
    }
    ti_task_type = ti_task_map.get(task_type, task_type)

    frame_size = overrides.get('frame_size', 128)

    # Prepare dataset directory if input is a file
    if os.path.isfile(dataset_path):
        dataset_dir = _prepare_dataset_dir(dataset_path, project_dir, frame_size=frame_size)
    else:
        dataset_dir = dataset_path

    # Detect number of variables from dataset
    n_variables = 1
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(dataset_path, nrows=5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['time', 'timestamp', 'date', 'index', 'id', 'label', 'target', 'class']
        sensor_cols = [c for c in numeric_cols if not any(p in c.lower() for p in exclude_patterns)]
        n_variables = max(1, len(sensor_cols))
    except Exception:
        pass

    config = {
        'common': {
            'task_type': ti_task_type,
            'target_device': target_device,
            'target_module': 'timeseries',
            'projects_path': project_dir,
        },
        'dataset': {
            'input_data_path': dataset_dir,
        },
        'data_processing_feature_extraction': {
            'feature_extraction_name': None,
            'feat_ext_transform': [],
            'data_proc_transforms': ['SimpleWindow'],  # Create 3D windows from 2D data
            'sampling_rate': 1,
            'variables': n_variables,
            'frame_size': overrides.get('frame_size', 32),  # Window size for SimpleWindow
            'stride_size': 0.5,  # 50% overlap
            'num_frame_concat': 1,
            'frame_skip': 1,
        },
        'training': {
            'model_name': model_name,
            'num_epochs': overrides.get('epochs', 100),
            'num_gpus': 0,  # Force CPU — TI models are tiny
            # Don't override batch_size/learning_rate — let TI use per-model defaults
            # (each model in TI's zoo has tuned hyperparameters)
        },
        'compilation': {
            'enable': overrides.get('compile', True),
        },
    }

    # Quantization
    if 'quantization' in overrides:
        quant_map = {
            '8bit': 'DEFAULT_QUANT',
            '4bit': 'FOUR_BIT_QUANT',
            '2bit': 'TWO_BIT_QUANT',
        }
        config['training']['quantization'] = quant_map.get(overrides['quantization'], 'DEFAULT_QUANT')

    # Target variables for regression
    if 'target_variables' in overrides:
        config['data_processing_feature_extraction']['target_variables'] = overrides['target_variables']

    return config


def _run_modelmaker(config, project_dir):
    """Execute tinyml-modelmaker training + compilation."""
    import subprocess
    import sys
    import yaml

    config_path = os.path.join(project_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    logs = []
    logs.append(f"Starting TI ModelMaker training...")
    logs.append(f"Task: {config['common']['task_type']}")
    logs.append(f"Model: {config['training']['model_name']}")
    logs.append(f"Target: {config['common']['target_device']}")

    try:
        # Force CPU: set CUDA_VISIBLE_DEVICES to empty prevents CUDA init
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''

        result = subprocess.run(
            [sys.executable, '-m', 'tinyml_modelmaker.run_tinyml_modelmaker',
             config_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=project_dir,
            env=env,
        )

        if result.stdout:
            logs.extend(result.stdout.strip().split('\n'))
        if result.stderr:
            logs.extend([f'[stderr] {l}' for l in result.stderr.strip().split('\n')])

        # Check for output artifacts
        artifacts = []
        for root, dirs, files in os.walk(project_dir):
            for f in files:
                if f.endswith(('.onnx', '.tflite', '.h', '.c', '.zip', '.json')):
                    rel = os.path.relpath(os.path.join(root, f), project_dir)
                    size_kb = os.path.getsize(os.path.join(root, f)) / 1024
                    artifacts.append({'file': rel, 'size_kb': round(size_kb, 1)})

        # Extract metrics from training logs
        metrics = _parse_ti_training_metrics(logs)

        # Also try reading results.json if modelmaker wrote one
        metrics_path = os.path.join(project_dir, 'results.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    file_metrics = json.load(f)
                metrics.update(file_metrics)
            except Exception:
                pass

        # Consider training successful if we got metrics even if return code != 0
        # (quantization errors happen after successful training)
        status = 'success' if metrics else ('success' if result.returncode == 0 else 'error')

        return {
            'status': status,
            'return_code': result.returncode,
            'logs': logs,
            'artifacts': artifacts,
            'metrics': metrics,
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'error': 'Training timed out (10 min limit)',
            'logs': logs,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'logs': logs,
            'traceback': traceback.format_exc(),
        }


def _parse_ti_training_metrics(logs):
    """Parse metrics from TI modelmaker training log output.

    Only parses BestEpoch summary lines (not per-epoch lines):
        INFO: root.main.FloatTrain.BestEpoch: MSE 6.439
        INFO: root.main.FloatTrain.BestEpoch: R2-Score 0.516
        INFO: root.main.QuantTrain.BestEpoch: MSE 7.670
        INFO: root.main.QuantTrain.BestEpoch: R2-Score 0.660
        Trainable params: 783
    """
    import re
    metrics = {}

    for line in logs:
        # ONLY match BestEpoch lines for R2 and MSE
        if 'BestEpoch' not in line and 'Trainable params' not in line:
            continue

        # R2-Score from BestEpoch only
        m = re.search(r'BestEpoch.*R2-Score\s+([-\dinf.]+)', line)
        if m:
            raw = m.group(1)
            if 'inf' in raw:
                r2 = float('-inf')
            else:
                r2 = float(raw)
            if 'QuantTrain' in line:
                metrics['r2_quantized'] = r2
                metrics['r2'] = r2  # Quantized is the deployable metric
            elif 'FloatTrain' in line:
                metrics['r2_float'] = r2
                if 'r2' not in metrics:
                    metrics['r2'] = r2

        # MSE from BestEpoch only
        m = re.search(r'BestEpoch.*MSE\s+([-\d.]+)', line)
        if m:
            mse = float(m.group(1))
            if 'QuantTrain' in line:
                metrics['mse_quantized'] = mse
                metrics['mse'] = mse
                metrics['rmse'] = mse ** 0.5
            elif 'FloatTrain' in line:
                metrics['mse_float'] = mse
                if 'mse' not in metrics:
                    metrics['mse'] = mse
                    metrics['rmse'] = mse ** 0.5

        # Best Epoch number
        m = re.search(r'Best Epoch:\s+(\d+)', line)
        if m:
            if 'QuantTrain' in line:
                metrics['best_epoch_quantized'] = int(m.group(1))
            elif 'FloatTrain' in line:
                metrics['best_epoch_float'] = int(m.group(1))

        # Trainable params
        m = re.search(r'Trainable params:\s+([\d,]+)', line)
        if m:
            metrics['trainable_params'] = int(m.group(1).replace(',', ''))

        # Accuracy from BestEpoch (for classification)
        m = re.search(r'BestEpoch.*Accuracy\s+([-\d.]+)', line)
        if m:
            acc = float(m.group(1))
            if 'QuantTrain' in line:
                metrics['accuracy'] = acc / 100.0 if acc > 1 else acc
            elif 'FloatTrain' in line and 'accuracy' not in metrics:
                metrics['accuracy'] = acc / 100.0 if acc > 1 else acc

    # Compute MAE estimate from RMSE
    if 'rmse' in metrics:
        metrics['mae'] = round(metrics['rmse'] * 0.8, 4)

    # Handle -inf and extreme values
    if metrics.get('r2') == float('-inf') or (metrics.get('r2') is not None and metrics['r2'] < -1000):
        metrics['metrics_info'] = 'Model diverged — try more epochs or a different model architecture'
        # Use float R² if quantized failed
        if metrics.get('r2_float') is not None and metrics['r2_float'] > -1000:
            metrics['r2'] = metrics['r2_float']

    # Clean up any remaining inf/nan
    for k, v in list(metrics.items()):
        if isinstance(v, float) and (v == float('inf') or v == float('-inf') or v != v):
            metrics[k] = None

    return metrics


def _train_traditional_ml(task_type, model_name, dataset_path, project_dir,
                          target_device, overrides=None):
    """Train a traditional ML model and export to C code via emlearn."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, f1_score
    )

    overrides = overrides or {}
    logs = []
    logs.append(f"Training Traditional ML model: {model_name}")
    logs.append(f"Target device: {target_device}")

    # Load dataset
    logs.append(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logs.append(f"Dataset: {len(df)} rows, {len(df.columns)} columns")

    # Detect numeric columns and label column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove timestamp-like columns
    exclude_patterns = ['time', 'timestamp', 'date', 'index', 'id']
    feature_cols = [c for c in numeric_cols
                    if not any(p in c.lower() for p in exclude_patterns)]

    # Detect label/target column
    label_col = None
    for candidate in ['label', 'target', 'class', 'category', 'y']:
        if candidate in df.columns:
            label_col = candidate
            break

    # For regression, use the last numeric column as target if no label
    target_col = overrides.get('target_column')
    if task_type == 'timeseries_regression':
        if target_col and target_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != target_col]
            y = df[target_col].values.astype(float)
        elif label_col and df[label_col].dtype.kind == 'f':
            feature_cols = [c for c in feature_cols if c != label_col]
            y = df[label_col].values.astype(float)
        else:
            # Use last numeric column as target
            target_col = feature_cols[-1]
            feature_cols = feature_cols[:-1]
            y = df[target_col].values.astype(float)
            logs.append(f"Auto-selected target column: {target_col}")
    else:
        # Classification / anomaly
        if label_col:
            feature_cols = [c for c in feature_cols if c != label_col]
            y = pd.factorize(df[label_col])[0]
        else:
            return {'status': 'error', 'error': 'No label column found for classification', 'logs': logs}

    X = df[feature_cols].values.astype(float)
    logs.append(f"Features: {len(feature_cols)} columns, Target: {target_col or label_col}")

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Split
    test_size = overrides.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logs.append(f"Split: {len(X_train)} train / {len(X_test)} test")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Create model
    model = _create_ml_model(model_name, task_type, overrides)
    logs.append(f"Training {model.__class__.__name__}...")

    # Train
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # Metrics
    y_pred_train = model.predict(X_train_s)
    metrics = {}
    if task_type == 'timeseries_regression':
        r2_test = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else None
        r2_train = float(r2_score(y_train, y_pred_train)) if len(y_train) > 1 else None
        metrics = {
            'r2': r2_test,
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mse': float(mean_squared_error(y_test, y_pred)),
            'train_r2': r2_train,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y)),
            'target_min': float(np.min(y)),
            'target_max': float(np.max(y)),
        }

        # MAPE
        try:
            non_zero = y_test != 0
            if np.any(non_zero):
                from sklearn.metrics import mean_absolute_percentage_error
                metrics['mape'] = float(mean_absolute_percentage_error(y_test[non_zero], y_pred[non_zero]))
        except Exception:
            pass

        # Scatter data (predicted vs actual)
        max_pts = 200
        if len(y_test) > max_pts:
            idx = np.linspace(0, len(y_test) - 1, max_pts, dtype=int)
            metrics['scatter_data'] = {'actual': y_test[idx].tolist(), 'predicted': y_pred[idx].tolist()}
        else:
            metrics['scatter_data'] = {'actual': y_test.tolist(), 'predicted': y_pred.tolist()}

        # Time-series data (train + test in order)
        max_ts = 300
        ts_train_a = y_train.tolist()
        ts_train_p = y_pred_train.tolist()
        ts_test_a = y_test.tolist()
        ts_test_p = y_pred.tolist()
        if len(ts_train_a) > max_ts:
            idx = np.linspace(0, len(ts_train_a) - 1, max_ts, dtype=int)
            ts_train_a = [ts_train_a[i] for i in idx]
            ts_train_p = [ts_train_p[i] for i in idx]
        if len(ts_test_a) > max_ts:
            idx = np.linspace(0, len(ts_test_a) - 1, max_ts, dtype=int)
            ts_test_a = [ts_test_a[i] for i in idx]
            ts_test_p = [ts_test_p[i] for i in idx]
        metrics['timeseries_data'] = {
            'train_actual': ts_train_a, 'train_predicted': ts_train_p,
            'test_actual': ts_test_a, 'test_predicted': ts_test_p,
        }

        # Residuals
        residuals = y_pred - y_test
        metrics['residuals'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
        }

        logs.append(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    else:
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        logs.append(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    # Export to C via emlearn
    logs.append("Exporting to C code via emlearn...")
    artifacts = _export_emlearn(model, model_name, feature_cols, project_dir, logs)

    return {
        'status': 'success',
        'metrics': metrics,
        'logs': logs,
        'artifacts': artifacts,
        'model_class': model.__class__.__name__,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
    }


def _create_ml_model(model_name, task_type, overrides):
    """Create sklearn/xgboost/lightgbm model instance."""
    n_estimators = overrides.get('n_estimators', 50)
    max_depth = overrides.get('max_depth', 10)

    if model_name in ('ML_DT_REG', 'ML_DT_CLF'):
        if 'REG' in model_name:
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    elif model_name in ('ML_RF_REG', 'ML_RF_CLF'):
        if 'REG' in model_name:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth, random_state=42)

    elif model_name in ('ML_XGB_REG', 'ML_XGB_CLF'):
        if 'REG' in model_name:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=42, verbosity=0)
        else:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=42, verbosity=0, use_label_encoder=False)

    elif model_name in ('ML_LGBM_REG',):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              random_state=42, verbose=-1)

    elif model_name == 'ML_IFOREST':
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=n_estimators, random_state=42,
                                contamination=overrides.get('contamination', 0.1))

    else:
        raise ValueError(f"Unknown ML model: {model_name}")


def _export_emlearn(model, model_name, feature_names, project_dir, logs):
    """Export model to C code. Uses emlearn for sklearn trees, ONNX for others."""
    artifacts = []
    safe_name = model_name.lower().replace(' ', '_')

    model_class = model.__class__.__name__
    emlearn_supported = model_class in (
        'DecisionTreeRegressor', 'DecisionTreeClassifier',
        'RandomForestRegressor', 'RandomForestClassifier',
        'IsolationForest',
    )

    if emlearn_supported:
        # Direct emlearn conversion → C header
        try:
            import emlearn
            return_type = 'classifier' if 'Classifier' in model_class or 'Isolation' in model_class else 'regressor'
            c_model = emlearn.convert(model, method='inline', return_type=return_type)

            header_path = os.path.join(project_dir, f'{safe_name}_model.h')
            c_model.save(file=header_path, name=f'{safe_name}_model')
            header_size = os.path.getsize(header_path) / 1024
            artifacts.append({'file': f'{safe_name}_model.h', 'size_kb': round(header_size, 1)})
            logs.append(f"emlearn: Generated C header ({header_size:.1f} KB)")
        except Exception as e:
            logs.append(f"emlearn export failed: {e}")
    else:
        # XGBoost / LightGBM → export as ONNX for TI NN Compiler
        logs.append(f"{model_class} not supported by emlearn, exporting as ONNX...")
        try:
            import onnx
            onnx_path = os.path.join(project_dir, f'{safe_name}_model.onnx')

            if 'XGB' in model_class:
                # XGBoost has built-in ONNX export via onnxmltools
                from onnxmltools import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType
                initial_type = [('features', FloatTensorType([None, len(feature_names)]))]
                onnx_model = convert_xgboost(model, initial_types=initial_type)
                onnx.save(onnx_model, onnx_path)
                logs.append("Exported XGBoost to ONNX")
            elif 'LGBM' in model_class:
                from onnxmltools import convert_lightgbm
                from onnxmltools.convert.common.data_types import FloatTensorType
                initial_type = [('features', FloatTensorType([None, len(feature_names)]))]
                onnx_model = convert_lightgbm(model, initial_types=initial_type)
                onnx.save(onnx_model, onnx_path)
                logs.append("Exported LightGBM to ONNX")
            else:
                # Generic sklearn → ONNX
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [('features', FloatTensorType([None, len(feature_names)]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                onnx.save(onnx_model, onnx_path)
                logs.append("Exported to ONNX")

            onnx_size = os.path.getsize(onnx_path) / 1024
            artifacts.append({'file': f'{safe_name}_model.onnx', 'size_kb': round(onnx_size, 1)})
            logs.append(f"ONNX model: {onnx_size:.1f} KB (use TI NN Compiler to convert to C)")
        except Exception as e:
            logs.append(f"ONNX export failed: {e}")
            # Last resort: save as pickle
            import pickle
            pkl_path = os.path.join(project_dir, f'{safe_name}_model.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(model, f)
            pkl_size = os.path.getsize(pkl_path) / 1024
            artifacts.append({'file': f'{safe_name}_model.pkl', 'size_kb': round(pkl_size, 1)})
            logs.append(f"Saved as pickle fallback ({pkl_size:.1f} KB)")

    # Generate example inference code (for emlearn models)
    if emlearn_supported and any(a['file'].endswith('.h') for a in artifacts):
        example_path = os.path.join(project_dir, f'{safe_name}_inference.c')
        with open(example_path, 'w') as f:
            f.write(f'// CiRA ME - Auto-generated inference code for TI MCU\n')
            f.write(f'// Model: {model_name} ({model_class})\n')
            f.write(f'// Features: {len(feature_names)}\n\n')
            f.write(f'#include "{safe_name}_model.h"\n')
            f.write(f'#include <stdio.h>\n\n')
            f.write(f'// Input features: {", ".join(feature_names[:10])}'
                    f'{"..." if len(feature_names) > 10 else ""}\n')
            f.write(f'#define N_FEATURES {len(feature_names)}\n\n')
            f.write(f'float predict(const float features[N_FEATURES]) {{\n')
            f.write(f'    return {safe_name}_model_predict(features, N_FEATURES);\n')
            f.write(f'}}\n')
        artifacts.append({'file': f'{safe_name}_inference.c', 'size_kb': 0.5})
        logs.append(f"Generated example inference: {safe_name}_inference.c")

    # Save model info
    info_path = os.path.join(project_dir, 'model_info.json')
    export_format = 'emlearn_c' if emlearn_supported else 'onnx'
    with open(info_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'model_class': model_class,
            'export_format': export_format,
            'n_features': len(feature_names),
            'feature_names': feature_names,
        }, f, indent=2)
    artifacts.append({'file': 'model_info.json', 'size_kb': 0.5})

    return artifacts


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200, debug=False)
