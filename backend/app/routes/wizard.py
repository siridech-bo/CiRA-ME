"""
CiRA ME - Multi-Dataset Wizard (F2)

Runs N saved ME-LAB endpoints against M CSV datasets in one shot and returns
a matrix of aggregated per-(dataset, model) predictions so the customer can
pick the smallest model meeting their confidence bar.

Design constraints locked with the customer (2026-07):
  Q1 Single-mode only     — mixed-mode endpoint selection is rejected.
  Q2 Anomaly included     — cell shows label + score, no probabilities.
  Q3 Row-count cap 100k   — reject datasets > 100k rows (fail-fast).
  Q4 Raw-mode only in v1  — windowed / feature-extracted models rejected.
  Q5 Model size in bytes  — auto-scaled KB/MB in the frontend hover only.

No new DB tables — runs live under DATASETS_ROOT/.wizard_runs/<user_id>/<run_id>/
and are torn down by DELETE /api/wizard/runs/<run_id>.
"""

import io
import csv
import json
import os
import shutil
import time
import uuid
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request, send_file

from ..auth import login_required, validate_path
from ..models import MeLabEndpoint, SavedModel
from ..services.data_loader import DataLoader
from ..services.melab_service import ModelManager

logger = logging.getLogger(__name__)
wizard_bp = Blueprint('wizard', __name__)

# Locked customer decisions live as module constants so they're greppable.
ROW_CAP = 100_000               # Q3
MAX_MODELS = 5
MAX_DATASETS = 5
MAX_MATRIX_CELLS = 25           # N x M cap; matches app_builder multi-model cap


# ─── Path helpers ────────────────────────────────────────────────────────────

def _wizard_root() -> str:
    """Base directory that holds every user's wizard runs.

    Mirrors the T3 .multi_csv_selections pattern (copies, not symlinks) so any
    container bind-mounting DATASETS_ROOT_PATH can see the files at a stable
    path.
    """
    datasets_root = current_app.config.get('DATASETS_ROOT_PATH') or os.environ.get(
        'DATASETS_ROOT_PATH', 'datasets'
    )
    return os.path.join(datasets_root, '.wizard_runs')


def _run_dir(user_id: int, run_id: str) -> str:
    # basename() blocks path-traversal attempts via run_id.
    safe_run_id = os.path.basename(str(run_id))
    if not safe_run_id or safe_run_id != str(run_id):
        raise ValueError('Invalid run_id')
    return os.path.join(_wizard_root(), str(int(user_id)), safe_run_id)


def _uploads_dir(user_id: int, run_id: str) -> str:
    return os.path.join(_run_dir(user_id, run_id), 'uploads')


def _load_run_meta(user_id: int, run_id: str) -> Optional[Dict]:
    meta_path = os.path.join(_run_dir(user_id, run_id), 'meta.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_run_meta(user_id: int, run_id: str, meta: Dict) -> None:
    d = _run_dir(user_id, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)


# ─── Dataset validation ──────────────────────────────────────────────────────

def _extract_sensor_columns(file_path: str) -> Tuple[List[str], int]:
    """Return (sensor_column_names, row_count).

    Uses DataLoader.load_csv so the column-drop convention (label + timestamp)
    stays in one place. Reads a small head first just to enforce the row cap
    without loading 100k+ rows into RAM twice.
    """
    # Fail-fast row-cap check: count lines cheaply before pandas parses everything.
    row_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Skip header (first line) — matches DataLoader.load_csv contract.
        f.readline()
        for _ in f:
            row_count += 1
            if row_count > ROW_CAP:
                # Bail early — the loader would otherwise chew through it.
                break
    if row_count > ROW_CAP:
        raise ValueError(
            f"Dataset exceeds {ROW_CAP:,}-row cap (has more than "
            f"{ROW_CAP:,} data rows). Trim the CSV and retry."
        )

    loader = DataLoader()
    info = loader.load_csv(file_path)
    # DataLoader.load_csv returns {'session_id', 'metadata': {...}, 'preview'}.
    meta = info.get('metadata') or {}
    return (
        list(meta.get('sensor_columns') or []),
        int(meta.get('total_rows') or 0),
    )


@wizard_bp.route('/validate-datasets', methods=['POST'])
@login_required
def validate_datasets():
    """Two input shapes: multipart upload OR JSON dataset_paths."""
    user = request.current_user
    user_id = user['id']
    run_id = uuid.uuid4().hex
    uploads = _uploads_dir(user_id, run_id)
    os.makedirs(uploads, exist_ok=True)

    saved_files: List[Tuple[str, str]] = []  # (display_name, absolute_path)

    try:
        # ── Multipart upload path ────────────────────────────────────────
        if request.files:
            files = request.files.getlist('files[]') or request.files.getlist('files')
            if not files:
                # Some clients send single-field form uploads.
                files = list(request.files.values())
            if not files:
                return jsonify({'error': 'No files uploaded'}), 400
            if len(files) > MAX_DATASETS:
                return jsonify({
                    'error': f'Max {MAX_DATASETS} datasets per run '
                             f'(got {len(files)}).'
                }), 400
            for fs in files:
                if not fs.filename:
                    continue
                # basename() blocks any leading path components in the upload name.
                safe_name = os.path.basename(fs.filename)
                if not safe_name.lower().endswith('.csv'):
                    return jsonify({
                        'error': f"{safe_name}: only .csv files accepted."
                    }), 400
                dest = os.path.join(uploads, safe_name)
                fs.save(dest)
                saved_files.append((safe_name, dest))

        # ── JSON reference path ──────────────────────────────────────────
        else:
            payload = request.get_json(silent=True) or {}
            paths = payload.get('dataset_paths') or []
            if not isinstance(paths, list) or not paths:
                return jsonify({'error': 'dataset_paths required'}), 400
            if len(paths) > MAX_DATASETS:
                return jsonify({
                    'error': f'Max {MAX_DATASETS} datasets per run '
                             f'(got {len(paths)}).'
                }), 400
            datasets_root = current_app.config['DATASETS_ROOT_PATH']
            shared_folder = current_app.config['SHARED_FOLDER_PATH']
            for p in paths:
                if not isinstance(p, str):
                    return jsonify({'error': 'dataset_paths entries must be strings'}), 400
                abs_p = os.path.abspath(p)
                if not validate_path(abs_p, user, datasets_root, shared_folder):
                    return jsonify({'error': f'Access denied for path: {p}'}), 403
                if not os.path.isfile(abs_p):
                    return jsonify({'error': f'Path not found: {p}'}), 404
                # Copy into the run's uploads dir so downstream logic doesn't have
                # to distinguish uploaded vs referenced datasets.
                safe_name = os.path.basename(abs_p)
                dest = os.path.join(uploads, safe_name)
                try:
                    shutil.copy2(abs_p, dest)
                except OSError as e:
                    return jsonify({'error': f'Copy failed for {safe_name}: {e}'}), 500
                saved_files.append((safe_name, dest))

        if not saved_files:
            return jsonify({'error': 'No usable CSVs supplied'}), 400

        # ── Schema strictness (F2.1) + Row cap (Q3) ──────────────────────
        per_dataset: List[Dict] = []
        for idx, (name, path) in enumerate(saved_files):
            try:
                cols, rows = _extract_sensor_columns(path)
            except ValueError as ve:
                # Row-cap hit → 413 Payload Too Large.
                _cleanup(user_id, run_id)
                return jsonify({
                    'error': 'ROW_CAP_EXCEEDED',
                    'dataset': name,
                    'message': str(ve),
                }), 413
            except Exception as e:
                _cleanup(user_id, run_id)
                return jsonify({
                    'error': f'Failed to parse {name}: {e}'
                }), 400
            per_dataset.append({
                'id': f'd{idx}',
                'name': name,
                'path': path,
                'row_count': rows,
                'sensor_columns': cols,
            })

        expected = per_dataset[0]['sensor_columns']
        expected_set = set(expected)
        mismatches = []
        for ds in per_dataset[1:]:
            ds_set = set(ds['sensor_columns'])
            if ds_set != expected_set:
                mismatches.append({
                    'dataset': ds['name'],
                    'extra': sorted(ds_set - expected_set),
                    'missing': sorted(expected_set - ds_set),
                })
        if mismatches:
            _cleanup(user_id, run_id)
            return jsonify({
                'error': 'SCHEMA_MISMATCH',
                'expected': expected,
                'mismatches': mismatches,
            }), 409

        # ── Persist run metadata ─────────────────────────────────────────
        meta = {
            'run_id': run_id,
            'user_id': user_id,
            'columns': expected,
            'datasets': [
                {
                    'id': ds['id'],
                    'name': ds['name'],
                    'path': ds['path'],
                    'row_count': ds['row_count'],
                }
                for ds in per_dataset
            ],
            'created_at': time.time(),
        }
        _save_run_meta(user_id, run_id, meta)

        return jsonify({
            'run_id': run_id,
            'columns': expected,
            'datasets': [
                {'id': ds['id'], 'name': ds['name'], 'row_count': ds['row_count']}
                for ds in per_dataset
            ],
        })
    except Exception as e:
        logger.exception(f"[Wizard] validate-datasets failed: {e}")
        _cleanup(user_id, run_id)
        return jsonify({'error': f'validate-datasets failed: {e}'}), 500


# ─── Run ────────────────────────────────────────────────────────────────────

def _load_endpoint_or_reject(endpoint_id, user) -> Tuple[Optional[Dict], Optional[Tuple[Dict, int]]]:
    """Fetch endpoint enforcing ownership + active status. Returns (endpoint,
    error_pair). error_pair is (response_dict, http_status) or None."""
    ep = MeLabEndpoint.get_by_id(endpoint_id)
    if not ep:
        return None, ({'error': f'Endpoint {endpoint_id} not found'}, 404)
    if user.get('role') != 'admin' and ep.get('user_id') != user['id']:
        return None, ({'error': f'Access denied for endpoint {endpoint_id}'}, 403)
    if ep.get('status') != 'active':
        return None, ({'error': f'Endpoint {endpoint_id} is not active'}, 400)
    return ep, None


def _is_raw_mode(saved: Dict) -> bool:
    """Q4 gate: reject anything that isn't strictly one-row-per-feature-vector.

    Rejects if any of these are set on the saved model's pipeline_config:
      - windowing.enabled == True and no_windowing not truthy
      - feature_extraction present (non-empty)
      - training_approach == 'dl'
    """
    pc = saved.get('pipeline_config', {}) or {}
    if isinstance(pc, str):
        try:
            pc = json.loads(pc) if pc else {}
        except json.JSONDecodeError:
            pc = {}

    # DL always requires windowed input.
    if str(pc.get('training_approach') or '').lower() == 'dl':
        return False

    # Feature extraction implies windowed / non-raw.
    fe = pc.get('feature_extraction') or {}
    if isinstance(fe, dict) and fe:
        # An empty {} is fine; anything with content means feature-extracted.
        if fe.get('feature_names') or fe.get('features') or fe.get('domain'):
            return False

    # Explicit windowing block.
    win = pc.get('windowing') or {}
    if isinstance(win, dict) and win.get('enabled') and not pc.get('no_windowing'):
        return False

    return True


@wizard_bp.route('/run', methods=['POST'])
@login_required
def run():
    user = request.current_user
    user_id = user['id']
    data = request.get_json(silent=True) or {}
    run_id = str(data.get('run_id') or '')
    endpoint_ids = data.get('endpoint_ids') or []

    if not run_id:
        return jsonify({'error': 'run_id required'}), 400
    if not isinstance(endpoint_ids, list) or not endpoint_ids:
        return jsonify({'error': 'endpoint_ids required'}), 400
    if len(endpoint_ids) > MAX_MODELS:
        return jsonify({
            'error': f'Max {MAX_MODELS} endpoints per run (got {len(endpoint_ids)})'
        }), 400

    meta = _load_run_meta(user_id, run_id)
    if not meta:
        return jsonify({'error': 'run_id not found or expired'}), 404

    datasets = meta.get('datasets') or []
    if len(datasets) * len(endpoint_ids) > MAX_MATRIX_CELLS:
        return jsonify({
            'error': f'Matrix cap exceeded: {len(datasets)} × {len(endpoint_ids)} '
                     f'> {MAX_MATRIX_CELLS}. Reduce datasets or models.'
        }), 400

    # ── Validate endpoints (ownership + active + Q1 single-mode + Q4 raw-only)
    resolved: List[Tuple[str, Dict, Dict, int]] = []  # (eid, endpoint, saved, size_bytes)
    modes_seen = set()
    for eid in endpoint_ids:
        ep, err = _load_endpoint_or_reject(eid, user)
        if err:
            resp, code = err
            return jsonify(resp), code
        saved = SavedModel.get_by_id(ep['saved_model_id'])
        if not saved or not saved.get('model_path'):
            return jsonify({
                'error': f"Endpoint {eid}: model file missing."
            }), 400
        try:
            size_bytes = int(os.path.getsize(saved['model_path']))
        except OSError:
            size_bytes = 0
        # Q4 gate.
        if not _is_raw_mode(saved):
            return jsonify({
                'error': f"Endpoint {eid} ({ep.get('name')}) is a "
                         "windowed / feature-extracted model. The wizard "
                         "supports raw-mode models only in v1 — pick a model "
                         "trained without windowing / feature extraction."
            }), 400
        modes_seen.add(ep.get('mode'))
        resolved.append((str(eid), ep, saved, size_bytes))

    if len(modes_seen) > 1:
        return jsonify({
            'error': f"Mixed endpoint modes selected ({sorted(modes_seen)}). "
                     "All endpoints must share the same mode "
                     "(classification / regression / anomaly)."
        }), 400
    mode = next(iter(modes_seen))

    # ── Load each dataset once, then run every endpoint against it ───
    expected_cols: List[str] = list(meta.get('columns') or [])
    matrix: List[List[Dict]] = []
    aggregated: List[Dict] = []
    predictions_root = _run_dir(user_id, run_id)

    for ds in datasets:
        ds_row: List[Dict] = []
        ds_path = ds['path']
        try:
            df = pd.read_csv(ds_path)
            # Restrict to expected sensor columns in schema order so every
            # model sees the same feature layout across datasets.
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                # Schema was checked at validate time; a mismatch here is an
                # invariant violation, not user error.
                raise RuntimeError(f"expected columns missing from {ds['name']}: {missing}")
            features = df[expected_cols].to_numpy(dtype=np.float64)
        except Exception as e:
            # Whole dataset failed → error for every cell in this row.
            err_cell = {'error': f"dataset load failed: {e}"}
            for _ in resolved:
                ds_row.append(err_cell)
            matrix.append(ds_row)
            continue

        for eid, ep, saved, size_bytes in resolved:
            t0 = time.perf_counter()
            try:
                preds = ModelManager.predict_by_endpoint(eid, features)
                latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                logger.exception(f"[Wizard] endpoint {eid} on dataset {ds['name']} failed: {e}")
                ds_row.append({'error': str(e)})
                continue

            # Persist per-row predictions + latency for the export endpoint.
            # Latency must live alongside preds so /export can render real
            # timings instead of 0.0 (the export path has no live perf counter).
            try:
                pred_file = os.path.join(
                    predictions_root,
                    f"predictions_{eid}_{ds['id']}.json"
                )
                with open(pred_file, 'w', encoding='utf-8') as pf:
                    json.dump({
                        'preds': preds,
                        'latency_ms': latency_ms,
                    }, pf)
            except OSError:
                # Export is best-effort; matrix is the primary output.
                pass

            cell = _aggregate_cell(preds, mode, latency_ms)
            ds_row.append(cell)

            # Aggregated summary row.
            avg_conf = None
            if mode == 'classification':
                confs = [p.get('confidence') for p in preds if isinstance(p, dict)]
                confs = [c for c in confs if c is not None]
                if confs:
                    avg_conf = float(np.mean(confs))
            elif mode == 'anomaly':
                scores = [p.get('score') for p in preds if isinstance(p, dict)]
                scores = [s for s in scores if s is not None]
                if scores:
                    avg_conf = float(np.mean(scores))
            latencies = [latency_ms / max(1, len(preds))]  # per-row estimate
            aggregated.append({
                'dataset_id': ds['id'],
                'dataset_name': ds['name'],
                'model_id': eid,
                'model_name': ep.get('name'),
                'avg_confidence': avg_conf,
                'avg_latency_ms': float(np.mean(latencies)),
                'correct_pct': None,   # no ground-truth channel in v1
            })

        matrix.append(ds_row)

    response = {
        'run_id': run_id,
        'mode': mode,
        'models': [
            {
                'id': eid,
                'name': ep.get('name'),
                'algorithm': ep.get('algorithm'),
                'size_bytes': size_bytes,
            }
            for eid, ep, saved, size_bytes in resolved
        ],
        'datasets': [
            {'id': ds['id'], 'name': ds['name'], 'row_count': ds['row_count']}
            for ds in datasets
        ],
        'matrix': matrix,
        'aggregated': aggregated,
    }
    return jsonify(response)


def _aggregate_cell(preds: List[Dict], mode: str, total_latency_ms: float) -> Dict:
    """Collapse per-row predictions into one matrix cell.

    - classification/anomaly: modal label; confidence = mean confidence of the
      rows whose label matched the modal label.
    - regression: mean predicted value.
    """
    n_rows = max(1, len(preds))
    per_row_latency = total_latency_ms / n_rows

    if not preds:
        return {'error': 'no predictions'}

    if mode == 'regression':
        vals = [p.get('value') for p in preds if isinstance(p, dict) and p.get('value') is not None]
        if not vals:
            return {'error': 'no numeric predictions'}
        return {
            'predicted_label': float(np.mean(vals)),
            'confidence': None,
            'probabilities': None,
            'latency_ms': per_row_latency,
        }

    if mode == 'anomaly':
        labels = [str(p.get('label', '')) for p in preds if isinstance(p, dict)]
        modal_label, _ = Counter(labels).most_common(1)[0]
        # Aggregate score = mean over rows tagged with the modal label.
        matching_scores = [
            p.get('score') for p in preds
            if isinstance(p, dict) and str(p.get('label', '')) == modal_label
            and p.get('score') is not None
        ]
        agg_score = float(np.mean(matching_scores)) if matching_scores else None
        return {
            'predicted_label': modal_label,
            'confidence': None,
            'score': agg_score,
            'probabilities': None,
            'latency_ms': per_row_latency,
        }

    # classification
    labels = [str(p.get('label', '')) for p in preds if isinstance(p, dict)]
    if not labels:
        return {'error': 'no labels'}
    modal_label, _ = Counter(labels).most_common(1)[0]
    matching_confs = [
        p.get('confidence') for p in preds
        if isinstance(p, dict) and str(p.get('label', '')) == modal_label
        and p.get('confidence') is not None
    ]
    modal_conf = float(np.mean(matching_confs)) if matching_confs else None

    # Aggregate per-class probability = mean across all rows.
    prob_accum = defaultdict(list)
    for p in preds:
        if not isinstance(p, dict):
            continue
        probs = p.get('probabilities') or {}
        for cls, v in probs.items():
            try:
                prob_accum[str(cls)].append(float(v))
            except (TypeError, ValueError):
                pass
    aggregated_probs = None
    if prob_accum:
        aggregated_probs = {
            cls: float(np.mean(vs)) for cls, vs in prob_accum.items()
        }

    return {
        'predicted_label': modal_label,
        'confidence': modal_conf,
        'probabilities': aggregated_probs,
        'latency_ms': per_row_latency,
    }


# ─── Export ─────────────────────────────────────────────────────────────────

@wizard_bp.route('/export', methods=['POST', 'GET'])
@login_required
def export():
    """CSV export in two levels: aggregated or per_row."""
    user = request.current_user
    user_id = user['id']

    # Accept both JSON POST and query-string GET (frontend uses window.location).
    if request.method == 'GET':
        run_id = request.args.get('run_id', '')
        level = request.args.get('level', 'aggregated')
    else:
        body = request.get_json(silent=True) or {}
        run_id = str(body.get('run_id') or '')
        level = str(body.get('level') or 'aggregated')

    if not run_id:
        return jsonify({'error': 'run_id required'}), 400
    if level not in ('aggregated', 'per_row'):
        return jsonify({'error': "level must be 'aggregated' or 'per_row'"}), 400

    meta = _load_run_meta(user_id, run_id)
    if not meta:
        return jsonify({'error': 'run_id not found or expired'}), 404

    run_dir = _run_dir(user_id, run_id)
    # Scan predictions_<eid>_<did>.json files to know what ran.
    # Value shape: (preds_list, latency_ms). Old runs (list-only) get latency=0.
    pred_files: Dict[Tuple[str, str], Tuple[List[Dict], float]] = {}
    for entry in os.listdir(run_dir):
        if not entry.startswith('predictions_') or not entry.endswith('.json'):
            continue
        stem = entry[len('predictions_'):-len('.json')]
        # Split on the LAST underscore (dataset id is 'd0'/'d1'/…, safe).
        try:
            eid, did = stem.rsplit('_', 1)
        except ValueError:
            continue
        try:
            with open(os.path.join(run_dir, entry), 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                preds = raw.get('preds') or []
                latency = float(raw.get('latency_ms') or 0.0)
            else:
                # Back-compat: earlier runs persisted a bare list.
                preds = raw if isinstance(raw, list) else []
                latency = 0.0
            pred_files[(eid, did)] = (preds, latency)
        except (OSError, json.JSONDecodeError):
            continue

    dataset_by_id = {ds['id']: ds for ds in meta.get('datasets', [])}

    buf = io.StringIO()
    writer = csv.writer(buf)

    if level == 'aggregated':
        writer.writerow([
            'dataset', 'model', 'mode', 'predicted_label',
            'avg_confidence', 'avg_latency_ms', 'model_size_bytes', 'error',
        ])
        for (eid, did), (preds, latency_ms) in pred_files.items():
            ds = dataset_by_id.get(did) or {'name': did}
            ep = MeLabEndpoint.get_by_id(eid) or {}
            saved = SavedModel.get_by_id(ep.get('saved_model_id')) if ep else None
            size_bytes = 0
            if saved and saved.get('model_path') and os.path.exists(saved['model_path']):
                try:
                    size_bytes = os.path.getsize(saved['model_path'])
                except OSError:
                    size_bytes = 0
            mode = ep.get('mode', '')
            cell = _aggregate_cell(preds, mode, latency_ms)
            writer.writerow([
                ds.get('name', did),
                ep.get('name', eid),
                mode,
                cell.get('predicted_label', ''),
                cell.get('confidence', '') if cell.get('confidence') is not None else '',
                cell.get('latency_ms', ''),
                size_bytes,
                cell.get('error', ''),
            ])
    else:
        # per_row — expand every prediction row, discover all probability classes.
        all_prob_classes: set = set()
        for preds, _lat in pred_files.values():
            for p in preds:
                if isinstance(p, dict) and isinstance(p.get('probabilities'), dict):
                    all_prob_classes.update(p['probabilities'].keys())
        prob_cols = sorted(all_prob_classes)
        writer.writerow(
            ['dataset', 'model', 'row_idx', 'predicted_label', 'confidence', 'score']
            + [f'prob_{c}' for c in prob_cols]
        )
        for (eid, did), (preds, _latency_ms) in pred_files.items():
            ds = dataset_by_id.get(did) or {'name': did}
            ep = MeLabEndpoint.get_by_id(eid) or {}
            for idx, p in enumerate(preds):
                if not isinstance(p, dict):
                    continue
                probs = p.get('probabilities') or {}
                row = [
                    ds.get('name', did),
                    ep.get('name', eid),
                    idx,
                    p.get('label', p.get('value', '')),
                    p.get('confidence', ''),
                    p.get('score', ''),
                ]
                for c in prob_cols:
                    row.append(probs.get(c, ''))
                writer.writerow(row)

    data = buf.getvalue().encode('utf-8')
    fname = f'wizard_{run_id}_{level}.csv'
    return send_file(
        io.BytesIO(data),
        mimetype='text/csv',
        as_attachment=True,
        download_name=fname,
    )


# ─── Cleanup ────────────────────────────────────────────────────────────────

def _cleanup(user_id: int, run_id: str) -> None:
    try:
        d = _run_dir(user_id, run_id)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        # Cleanup is best-effort — don't mask a caller's original error.
        pass


@wizard_bp.route('/runs/<run_id>', methods=['DELETE'])
@login_required
def delete_run(run_id):
    user = request.current_user
    try:
        _cleanup(user['id'], run_id)
    except ValueError:
        return jsonify({'error': 'Invalid run_id'}), 400
    return jsonify({'deleted': True})
