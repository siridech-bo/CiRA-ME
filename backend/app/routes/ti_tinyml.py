"""
CiRA ME - TI TinyML Routes
API endpoints for TI ModelMaker integration.
"""

import math
import logging
import requests
from flask import Blueprint, request, jsonify, Response
from ..auth import login_required
from ..services.ti_integration import TIIntegration


def _sanitize_nan(obj):
    """Replace NaN/Inf float values with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj

logger = logging.getLogger(__name__)
ti_bp = Blueprint('ti_tinyml', __name__)


@ti_bp.route('/status', methods=['GET'])
@login_required
def ti_status():
    """Check if TI ModelMaker service is available."""
    ti = TIIntegration()
    return jsonify(ti.get_health())


@ti_bp.route('/devices', methods=['GET'])
@login_required
def ti_devices():
    """Get supported TI MCU devices."""
    try:
        ti = TIIntegration()
        devices = ti.get_devices()
        return jsonify(devices)
    except Exception as e:
        return jsonify({'error': f'TI service unavailable: {e}'}), 503


@ti_bp.route('/models', methods=['GET'])
@login_required
def ti_models():
    """Get available models from TI model zoo."""
    task = request.args.get('task', 'timeseries_regression')
    device = request.args.get('device')
    source = request.args.get('source', 'all')

    try:
        ti = TIIntegration()
        models = ti.get_models(task=task, device=device, source=source)
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': f'TI service unavailable: {e}'}), 503


@ti_bp.route('/train', methods=['POST'])
@login_required
def ti_train():
    """Train a model using TI ModelMaker."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    mode = data.get('mode', 'regression')
    model_names = data.get('model_names', [])
    model_name = data.get('model_name')  # backward compat
    target_device = data.get('target_device', 'F2837')
    dataset_path = data.get('dataset_path')
    config = data.get('config', {})

    if model_name and not model_names:
        model_names = [model_name]

    if not model_names:
        return jsonify({'error': 'model_names required'}), 400
    if not dataset_path:
        return jsonify({'error': 'dataset_path required'}), 400

    # Map dataset path for TI container
    ti_dataset_path = dataset_path.replace(
        '/app/datasets', '/app/data/datasets'
    )

    try:
        ti = TIIntegration()
        task_type = ti.map_cira_mode_to_ti_task(mode)

        result = ti.train(
            task_type=task_type,
            model_names=model_names,
            target_device=target_device,
            dataset_path=ti_dataset_path,
            config=config,
        )

        return jsonify(result)

    except requests.HTTPError as e:
        # Surface the TI container's actual error message
        ti_error = None
        if e.response is not None:
            try:
                ti_error = e.response.json().get('error')
            except Exception:
                ti_error = e.response.text
        msg = ti_error or str(e)
        logger.error(f"TI training failed: {msg}")
        return jsonify({'error': f'TI training failed: {msg}'}), 500
    except Exception as e:
        logger.error(f"TI training error: {e}")
        return jsonify({'error': str(e)}), 500


@ti_bp.route('/train-stream', methods=['POST'])
@login_required
def ti_train_stream():
    """Stream training progress via SSE."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    mode = data.get('mode', 'regression')
    model_name = data.get('model_name')
    target_device = data.get('target_device', 'F2837')
    dataset_path = data.get('dataset_path')
    config = data.get('config', {})

    if not model_name or not dataset_path:
        return jsonify({'error': 'model_name and dataset_path required'}), 400

    ti_dataset_path = dataset_path.replace(
        '/app/datasets', '/app/data/datasets'
    )

    ti = TIIntegration()
    task_type = ti.map_cira_mode_to_ti_task(mode)

    import requests as req

    def generate():
        import json
        try:
            resp = req.post(
                f'{ti.base_url}/train-stream',
                json={
                    'task_type': task_type,
                    'model_name': model_name,
                    'target_device': target_device,
                    'dataset_path': ti_dataset_path,
                    'config': config,
                },
                stream=True,
                timeout=660,
            )
            if not resp.ok:
                try:
                    ti_error = resp.json().get('error', resp.text)
                except Exception:
                    ti_error = resp.text
                yield f"data: {json.dumps({'type': 'error', 'message': f'TI training failed: {ti_error}'})}\n\n"
                return
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line + '\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@ti_bp.route('/train-ml', methods=['POST'])
@login_required
def ti_train_ml_with_features():
    """Train Traditional ML model using CiRA ME's feature pipeline, export via emlearn.

    This route trains using CiRA ME's windowed+extracted features (same as Traditional ML tab),
    then sends the trained model to TI container for emlearn C code export.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    feature_session_id = data.get('feature_session_id')
    model_name = data.get('model_name')  # ML_DT_REG, ML_RF_REG, etc.
    target_device = data.get('target_device', 'F2837')
    test_size = data.get('test_size', 0.2)
    hyperparameters = data.get('hyperparameters', {})
    mode = data.get('mode', 'regression')

    if not feature_session_id:
        return jsonify({'error': 'feature_session_id required (extract features first)'}), 400
    if not model_name:
        return jsonify({'error': 'model_name required'}), 400

    try:
        from ..services.ml_trainer import MLTrainer
        from ..config import REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS

        trainer = MLTrainer()

        # Map TI model names to CiRA ME algorithm names
        ti_to_cira = {
            'ML_DT_REG': 'dt_reg', 'ML_RF_REG': 'rf_reg',
            'ML_XGB_REG': 'xgb_reg', 'ML_LGBM_REG': 'lgbm_reg',
            'ML_DT_CLF': 'dt', 'ML_RF_CLF': 'rf',
            'ML_XGB_CLF': 'gb', 'ML_IFOREST': 'iforest',
        }
        cira_algo = ti_to_cira.get(model_name)
        if not cira_algo:
            return jsonify({'error': f'Unknown ML model: {model_name}'}), 400

        # Train using CiRA ME's feature pipeline
        if mode == 'regression':
            result = trainer.train_regression(
                feature_session_id, cira_algo, hyperparameters,
                test_size=test_size,
                user_id=request.current_user['id']
            )
        elif mode == 'anomaly':
            result = trainer.train_anomaly(
                feature_session_id, cira_algo, hyperparameters,
                user_id=request.current_user['id']
            )
        else:
            result = trainer.train_classification(
                feature_session_id, cira_algo, hyperparameters,
                test_size=test_size,
                user_id=request.current_user['id']
            )

        # Add model size estimate
        import os
        model_path = result.get('model_path', '')
        if model_path and os.path.exists(model_path):
            size_kb = os.path.getsize(model_path) / 1024
            result['metrics']['model_size_kb'] = round(size_kb, 1)
            # emlearn C code is much smaller than pickle
            result['metrics']['model_size_int8_kb'] = round(size_kb * 0.1, 1)

        # Add pipeline info
        result['pipeline'] = 'cira_features'

        return jsonify(_sanitize_nan(result))

    except Exception as e:
        logger.error(f"TI ML training error: {e}")
        return jsonify({'error': str(e)}), 400


def _export_ti_nn_onnx_package(onnx_path, algorithm, mode, metrics, saved):
    """Package a native ONNX model (TI NN training output) into a TI MCU zip.

    TI NN models (REGR_*, CLS_*, AE_* from tinyml-modelmaker) are saved
    directly as PyTorch-exported ONNX, so no sklearn-to-ONNX conversion or
    emlearn C generation applies. The customer takes the packaged ONNX into
    Code Composer Studio and runs the TI MCU NN Compiler over it to produce
    the target-specific C code.
    """
    import os
    import tempfile
    import shutil
    import zipfile
    import json as _json_mod
    from flask import send_file

    tmp_dir = tempfile.mkdtemp()
    safe_name = algorithm.lower().replace(' ', '_').replace('-', '_') or 'ti_nn_model'
    dest_onnx = os.path.join(tmp_dir, 'model.onnx')
    shutil.copy2(onnx_path, dest_onnx)
    onnx_size_kb = round(os.path.getsize(dest_onnx) / 1024, 1)

    # Pull the CCS project templates (firmware skeleton, serial test tool) from
    # the TI container so the zip is a complete CCS-ready package, not just
    # the raw ONNX. If TI is unavailable we ship without them — the ONNX and
    # README are still useful on their own.
    ccs_templates = {}
    try:
        ti = TIIntegration()
        _resp = requests.get(f'{ti.base_url}/ccs-templates', timeout=5)
        if _resp.ok:
            ccs_templates = (_resp.json() or {}).get('files', {}) or {}
    except Exception as _exc:
        logger.warning(f"Could not fetch CCS templates from TI container: {_exc}")

    zip_path = os.path.join(tmp_dir, f'ti_mcu_{safe_name}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(dest_onnx, 'model.onnx')
        for _name, _content in ccs_templates.items():
            zf.writestr(_name, _content)
        zf.writestr('model_info.json', _json_mod.dumps({
            'model_name': algorithm,
            'model_type': 'ti_nn_onnx',
            'mode': mode,
            'metrics': metrics,
            'onnx_size_kb': onnx_size_kb,
            'export_note': 'Native ONNX from TI tinyml-modelmaker. Use the '
                           'TI MCU NN Compiler in Code Composer Studio to '
                           'convert model.onnx into C for the target MCU.',
        }, indent=2))
        _template_list = '\n'.join(
            f'  {n:20s} CCS project template'
            for n in sorted(ccs_templates)
        )
        if _template_list:
            _template_section = f'\nCCS templates included:\n{_template_list}\n'
        else:
            _template_section = ''
        zf.writestr('README.txt',
            f'CiRA ME - TI MCU Package (TI NN Model)\n'
            f'======================================\n\n'
            f'Model:     {algorithm}\n'
            f'Mode:      {mode}\n'
            f'ONNX size: {onnx_size_kb} KB\n\n'
            f'Files in this zip:\n'
            f'  model.onnx       ONNX model exported by tinyml-modelmaker\n'
            f'  model_info.json  Metrics + metadata from the training run\n'
            f'{_template_section}\n'
            f'Deployment to TMS320:\n'
            f'  1. Open TI Code Composer Studio.\n'
            f'  2. Use TI MCU NN Compiler to convert model.onnx into C code\n'
            f'     targeting your specific device (F28003, F28379D, etc.).\n'
            f'     This produces model_weights.h, model_config.h, and model.h.\n'
            f'  3. Add cira_main.c (if included above) as your project entry.\n'
            f'  4. Wire the generated inference call in cira_main.c\'s loop.\n'
            f'  5. Build and flash. Use cira_serial_test.py from your PC to\n'
            f'     stream test frames over UART and verify predictions.\n')

    return send_file(zip_path, as_attachment=True,
                     download_name=f'ti_mcu_{safe_name}.zip')


@ti_bp.route('/export-saved/<int:model_id>', methods=['POST'])
@login_required
def ti_export_saved_model(model_id):
    """Export a saved model to TI MCU C code via emlearn."""
    import pickle
    import os
    import tempfile
    import shutil

    from ..models import SavedModel, DeployRecord as _DeployRecord, Project as _Project

    saved = SavedModel.get_by_id(model_id)
    if not saved:
        return jsonify({'error': 'Model not found'}), 404

    # F4: record a ti_mcu deploy row up-front (Watch-out 6: never break the
    # zip download because of a DB blip). We do this at the entry point so
    # any of the multi-path returns below still credit the target.
    try:
        _pid = saved.get('project_id')
        if _pid:
            _DeployRecord.create(
                project_id=_pid,
                target='ti_mcu',
                saved_model_id=model_id,
                ref_id=f"ti_mcu_{saved.get('algorithm', 'model')}",
                metadata={'algorithm': saved.get('algorithm'),
                          'mode': saved.get('mode')},
            )
            _Project.touch(_pid, 'deploy')
    except Exception:
        pass

    model_path = saved.get('model_path', '')
    algorithm = saved.get('algorithm', 'model')
    mode = saved.get('mode', 'regression')
    import json as json_mod
    metrics = saved.get('metrics', {})
    if isinstance(metrics, str):
        try:
            metrics = json_mod.loads(metrics)
        except Exception:
            metrics = {}

    # If model file exists, load and convert to ONNX
    if model_path and os.path.exists(model_path):
        # Detect the file format. TI NN models (CLS_ResAdd_3k, REGR_*, etc.)
        # are saved as PyTorch-exported ONNX by tinyml-modelmaker; traditional
        # ML models (RF, XGBoost, DT) are saved as sklearn pickles. The two
        # export flows are incompatible — /convert-to-c only speaks pickle
        # and pickle.load blows up on ONNX bytes with an UnpicklingError,
        # which surfaces as an opaque "TI MCU export failed" toast.
        try:
            with open(model_path, 'rb') as _f:
                _magic = _f.read(4)
        except OSError:
            _magic = b''
        # ONNX protobuf starts with 0x08 (varint tag, field 1 = ir_version).
        # sklearn pickles start with 0x80 (PROTO opcode).
        is_onnx = _magic.startswith(b'\x08')

        if is_onnx:
            return _export_ti_nn_onnx_package(
                model_path, algorithm, mode, metrics, saved
            )

        # Send pickle to TI container for emlearn C export
        try:
            ti = TIIntegration()
            import requests as req

            with open(model_path, 'rb') as f:
                resp = req.post(
                    f'{ti.base_url}/convert-to-c',
                    files={'model': ('model.pkl', f, 'application/octet-stream')},
                    data={'mode': mode},
                    timeout=60,
                )

            if resp.status_code == 200:
                # Return the zip from TI container directly
                tmp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(tmp_dir, f'ti_mcu_{algorithm.replace(" ", "_")}.zip')
                with open(zip_path, 'wb') as f:
                    f.write(resp.content)

                from flask import send_file
                return send_file(zip_path, as_attachment=True,
                    download_name=f'ti_mcu_{algorithm.replace(" ", "_")}.zip')
            else:
                # TI container failed — fall back to ONNX export
                error_msg = resp.json().get('error', 'emlearn conversion failed') if resp.headers.get('content-type', '').startswith('application/json') else 'emlearn conversion failed'
                logger.warning(f"TI emlearn failed: {error_msg}, falling back to ONNX")
                # Fall through to ONNX export below

        except Exception as e:
            logger.warning(f"TI container unavailable for C export: {e}, falling back to ONNX")

        # Fallback: ONNX export from backend
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
    else:
        # No model file (TI-trained models) — export metrics + info only
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, 'ti_mcu_package.zip')
        import zipfile

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('model_info.json', json_mod.dumps({
                'model_name': algorithm,
                'mode': mode,
                'metrics': metrics,
                'note': 'This model was trained in the TI container. '
                        'Re-train using TI TinyML tab to generate C code artifacts.',
            }, indent=2))
            zf.writestr('README.txt',
                f'CiRA ME - TI MCU Export\n'
                f'=======================\n\n'
                f'Model: {algorithm}\n'
                f'Mode: {mode}\n'
                f'R2: {metrics.get("r2", "N/A")}\n\n'
                f'This model was trained in the TI TinyML container.\n'
                f'To get C code for MCU deployment:\n'
                f'  1. Go to Training > TI TinyML tab\n'
                f'  2. Select the same model and train\n'
                f'  3. Download the artifacts from there\n')

        from flask import send_file
        return send_file(zip_path, as_attachment=True,
            download_name=f'ti_mcu_{algorithm.replace(" ", "_")}.zip')

    try:

        if model is None:
            return jsonify({'error': 'No model object in pickle file'}), 400

        # Export as ONNX for TI MCU NN Compiler
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        import zipfile
        import json as json_mod

        model_class = model.__class__.__name__
        n_features = len(feature_names) if feature_names else 10
        safe_name = algorithm.lower().replace(' ', '_').replace('-', '_')

        tmp_dir = tempfile.mkdtemp()

        # Convert to ONNX
        initial_type = [('features', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path = os.path.join(tmp_dir, 'model.onnx')
        onnx.save(onnx_model, onnx_path)

        # Save pickle too (for Python-based deployment)
        pkl_copy = os.path.join(tmp_dir, 'model.pkl')
        shutil.copy2(model_path, pkl_copy)

        # Create zip package
        zip_path = os.path.join(tmp_dir, 'ti_mcu_package.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(onnx_path, 'model.onnx')
            zf.write(pkl_copy, 'model.pkl')
            zf.writestr('model_info.json', json_mod.dumps({
                'model_name': algorithm,
                'model_class': model_class,
                'mode': saved.get('mode', 'regression'),
                'n_features': n_features,
                'feature_names': feature_names,
                'onnx_size_kb': round(os.path.getsize(onnx_path) / 1024, 1),
                'export_note': 'Use TI MCU NN Compiler to convert model.onnx to C code for TMS320',
            }, indent=2))
            zf.writestr('README.txt',
                f'CiRA ME - TI MCU Export Package\n'
                f'================================\n\n'
                f'Model: {algorithm} ({model_class})\n'
                f'Mode: {saved.get("mode", "regression")}\n'
                f'Features: {n_features}\n'
                f'ONNX size: {round(os.path.getsize(onnx_path)/1024, 1)} KB\n\n'
                f'Files:\n'
                f'  model.onnx      - ONNX model for TI NN Compiler\n'
                f'  model.pkl       - Python pickle (for testing)\n'
                f'  model_info.json - Model metadata\n\n'
                f'Deployment:\n'
                f'  1. Open TI Code Composer Studio\n'
                f'  2. Use TI MCU NN Compiler to convert model.onnx\n'
                f'  3. Include generated C code in your project\n'
                f'  4. Call the inference function with {n_features} float features\n')

        from flask import send_file
        return send_file(zip_path, as_attachment=True,
            download_name=f'ti_mcu_{safe_name}.zip')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@ti_bp.route('/download/<run_id>', methods=['GET'])
@login_required
def ti_download(run_id):
    """Download compiled model artifacts."""
    try:
        ti = TIIntegration()
        content = ti.download_artifacts(run_id)
        return Response(
            content,
            mimetype='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename=ti_model_{run_id}.zip'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
