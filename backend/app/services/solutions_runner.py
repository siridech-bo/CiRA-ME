"""CiRA ME — Solution App orchestrators (F8, 2026-09-18).

Self-contained, task-specific runners that chain the existing engines
(windowing → registry feature extraction → model training) into ONE call, so
a Solution App's frontend collects domain inputs and gets domain results —
without threading the generic 5-step pipeline.

Currently: Motor Current (MCSA). Vibration/pump land here later.

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md (Solution Apps).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .data_loader import DataLoader, _data_sessions
from .feature_extractor import FeatureExtractor, _feature_sessions
from .ml_trainer import MLTrainer

logger = logging.getLogger(__name__)

# Physically-meaningful features surfaced as human-readable evidence.
_MCSA_EVIDENCE_FEATURES = [
    'mcsa_brb_strongest_db',
    'mcsa_brb_lsb_db',
    'mcsa_brb_usb_db',
    'mcsa_ecc_strongest_db',
    'mcsa_stator_strongest_db',
    'mcsa_slip',
    'mcsa_slip_confident',
    'mcsa_supported',
]


_DEFAULT_ALGO = {'anomaly': 'iforest', 'classification': 'rf'}
_ALGO_NAMES = {
    'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM', 'lof': 'Local Outlier Factor',
    'ecod': 'ECOD', 'copod': 'COPOD', 'hbos': 'HBOS',
    'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'svm': 'SVM',
    'knn': 'k-NN', 'dt': 'Decision Tree', 'lr': 'Logistic Regression',
}


def run_mcsa(
    data_session_id: str,
    sampling_rate: float,
    params: Dict[str, Any],
    window_s: float = 10.0,
    overlap: float = 0.5,
    approach: str = 'auto',
    algorithm: Optional[str] = None,
    skip_startup_s: float = 0.0,
    selected_columns: Optional[List[str]] = None,
    project_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the whole Motor Current (MCSA) diagnosis from a loaded data session.

    window → MCSA features → model → domain results, returning the fully
    resolved settings and per-window signal previews for transparency.

    approach: 'auto' (labels>1class → classification, else anomaly),
              'anomaly', or 'classification'. `algorithm` overrides the default.
    """
    if data_session_id not in _data_sessions:
        raise ValueError('Data session not found or expired — please reload your data.')

    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError('Sampling rate must be a positive number (Hz).')

    overlap = min(0.95, max(0.0, float(overlap)))
    window_size = max(16, int(round(window_s * fs)))
    stride = max(16, int(round(window_size * (1.0 - overlap))))

    # 1) Windowing (reuses the generic windower). selected_columns pins the
    # analysis to exactly the current channels (MCSA needs the 3 phases).
    loader = DataLoader()
    win = loader.apply_windowing(
        data_session_id, window_size=window_size, stride=stride,
        selected_columns=selected_columns or None, normalization_method='none',
    )
    win_sid = win['session_id']

    # The generic windower is sample-based and fs-agnostic; the MCSA extractor
    # needs the real sampling rate, so stamp it onto the windowed session.
    _data_sessions[win_sid].setdefault('metadata', {})['sampling_rate'] = fs

    # Optional: drop the leading startup-transient windows (smeared carrier).
    if skip_startup_s > 0:
        drop = int(round(skip_startup_s * fs / stride))
        _drop_leading_windows(win_sid, drop)

    # 2) MCSA feature extraction (physics-aware registry extractor).
    from .extractors import registry as _ext_registry
    resolved_params = _ext_registry.get('mcsa').validate_params(params)
    fe = FeatureExtractor()
    feat = fe.extract_with_registry(win_sid, 'mcsa', params)
    feat_sid = feat['session_id']

    # 3) Choose approach + train.
    wsess = _data_sessions[win_sid]
    labels = wsess.get('labels')
    classes: List[str] = sorted({str(x) for x in labels}) if labels is not None else []
    has_labeled_faults = len(classes) > 1

    chosen = approach if approach in ('anomaly', 'classification') else (
        'classification' if has_labeled_faults else 'anomaly')
    algo = algorithm or _DEFAULT_ALGO[chosen]

    trainer = MLTrainer()
    if chosen == 'classification':
        if not has_labeled_faults:
            raise ValueError(
                'Classification needs labeled data with at least 2 classes (add a '
                'label column), or choose the Anomaly approach.')
        training = trainer.train_classification(
            feat_sid, algo, project_id=project_id, user_id=user_id)
    else:
        training = trainer.train_anomaly(
            feat_sid, algo, project_id=project_id, user_id=user_id)

    evidence = _mcsa_evidence(feat_sid, labels, classes)

    settings = {
        'approach': chosen,
        'algorithm': algo,
        'algorithm_name': _ALGO_NAMES.get(algo, algo),
        'window_s': window_s,
        'window_size': window_size,
        'stride': stride,
        'overlap': round(overlap, 3),
        'sampling_rate': fs,
        'extractor': 'mcsa',
        'extractor_params': resolved_params,
        'phase_columns': selected_columns,
    }

    return {
        'solution': 'motor_current_mcsa',
        'mode': chosen,
        'algorithm': algo,
        'algorithm_name': _ALGO_NAMES.get(algo, algo),
        'classes': classes,
        'sampling_rate': fs,
        'window_s': window_s,
        'window_size': window_size,
        'num_windows': feat.get('num_windows'),
        'num_features': feat.get('num_features'),
        'windows_failed': feat.get('windows_failed', 0),
        'windowed_session_id': win_sid,
        'feature_session_id': feat_sid,
        'training': training,
        'evidence': evidence,
        'settings': settings,
        'window_previews': _window_previews(
            win_sid,
            fs=fs,
            line_freq=float(resolved_params['line_freq_hz']),
            phase=int(resolved_params['phase']),
            slips=_window_slips(feat_sid),
        ),
    }


def save_solution_model(
    name: str,
    solution_id: str,
    training_session_id: str,
    windowed_session_id: str,
    feature_session_id: str,
    settings: Dict[str, Any],
    user_id: int,
) -> int:
    """Persist a trained Solution-App model as a deployable SavedModel with a
    `pipeline_config` whose feature_extraction block points at the registry
    extractor used (from `settings['extractor']`) — so the existing deploy/replay
    path reproduces the exact windowing → registry features → predict chain at
    inference. Extractor-agnostic (works for mcsa, bearing_envelope, ...).
    """
    from .ml_trainer import _model_sessions
    from ..models import SavedModel

    session = _model_sessions.get(training_session_id)
    if not session:
        raise ValueError('Training session expired — re-run the analysis before saving.')

    extractor_id = settings.get('extractor')
    feat_entry = _feature_sessions.get(feature_session_id)
    feature_names = (feat_entry or {}).get('feature_names', [])
    wm = (_data_sessions.get(windowed_session_id) or {}).get('metadata', {})

    pipeline_config = {
        'solution': solution_id,
        'mode': session.get('mode'),
        'windowed_session_id': windowed_session_id,
        'feature_session_id': feature_session_id,
        'windowing': {
            'window_size': settings.get('window_size'),
            'stride': settings.get('stride'),
        },
        'normalization': wm.get('normalization'),
        'feature_extraction': {
            'method': 'registry',
            'extractor_id': extractor_id,
            'extractor_params': settings.get('extractor_params', {}),
            'sampling_rate': settings.get('sampling_rate'),
            'feature_names': feature_names,
            'num_features': len(feature_names),
        },
        'feature_selection': {
            'selected_features': feature_names,
            'num_selected': len(feature_names),
        },
        'training': {
            'algorithm': session.get('algorithm'),
            'hyperparameters': session.get('hyperparameters', {}),
        },
    }

    return SavedModel.save(
        name=name,
        algorithm=session.get('algorithm', ''),
        mode=session.get('mode', ''),
        metrics=session.get('metrics', {}),
        model_path=session.get('model_path', ''),
        training_session_id=training_session_id,
        pipeline_config=pipeline_config,
        dataset_info={'solution': solution_id},
        user_id=user_id,
    )


# Back-compat alias — MCSA save route calls this.
def save_mcsa_model(name, training_session_id, windowed_session_id,
                    feature_session_id, settings, user_id):
    return save_solution_model(name, 'motor_current_mcsa', training_session_id,
                               windowed_session_id, feature_session_id, settings, user_id)


def _window_slips(feat_sid: str) -> Optional[np.ndarray]:
    """Per-window estimated slip (aligned 1:1 with windows), for sideband marks."""
    entry = _feature_sessions.get(feat_sid)
    if not entry or 'mcsa_slip' not in entry['features'].columns:
        return None
    return entry['features']['mcsa_slip'].values


def _spectrum_dbc(sig: np.ndarray, fs: float, fmax: float = 200.0):
    """Single-sided amplitude spectrum in dBc (relative to the fundamental),
    from DC to fmax. Hann-windowed to suppress leakage. Native resolution."""
    x = np.asarray(sig, dtype=float)
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w))
    freq = np.fft.rfftfreq(len(x), 1.0 / fs)
    ref = float(np.max(mag)) if np.max(mag) > 0 else 1.0   # the fundamental
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    mask = freq <= fmax
    return freq[mask], db[mask]


def _window_previews(
    win_sid: str,
    fs: float = 5000.0,
    line_freq: float = 50.0,
    phase: int = 0,
    slips: Optional[np.ndarray] = None,
    max_windows: int = 6,
    preview_samples: int = 5000,
    fmax: float = 200.0,
) -> List[dict]:
    """Per-window previews: a native-resolution time slice (NOT decimated — that
    aliases a mains sinusoid) AND a frequency-domain MCSA spectrum (dBc vs Hz)
    with fundamental + broken-bar sideband markers. Both are zoomable client-side.
    """
    s = _data_sessions.get(win_sid)
    if not s or 'windows' not in s or len(s['windows']) == 0:
        return []
    windows = s['windows']
    labels = s.get('labels')
    cols = (s.get('metadata') or {}).get('sensor_columns') or \
        [f'ch{i}' for i in range(np.asarray(windows[0]).shape[1])]
    n = len(windows)
    idxs = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(max_windows, n))})
    out = []
    for i in idxs:
        w = np.asarray(windows[i], dtype=float)          # (wsize, nch)
        take = min(int(preview_samples), w.shape[0])
        seg = w[:take]
        channels = {cols[c]: [round(float(v), 5) for v in seg[:, c]]
                    for c in range(min(len(cols), w.shape[1]))}

        # Spectrum on the analysis phase channel (full window for resolution).
        pch = phase if 0 <= phase < w.shape[1] else 0
        freq, db = _spectrum_dbc(w[:, pch], fs, fmax=fmax)

        # Broken-bar sidebands (1 ± 2ks)·f0 at this window's estimated slip.
        sidebands = []
        if slips is not None and i < len(slips):
            slip = float(slips[i])
            if 0.0 < slip < 0.2:
                for k in (1, 2):
                    for sign in (-1, 1):
                        fhz = (1.0 + sign * 2.0 * k * slip) * line_freq
                        if 0 < fhz <= fmax:
                            sidebands.append(round(fhz, 3))

        out.append({
            'index': i,
            'label': (str(labels[i]) if labels is not None and i < len(labels) else None),
            'n_samples': int(w.shape[0]),
            'preview_samples': int(take),
            'channels': channels,
            'spectrum': {
                'freq': [round(float(f), 3) for f in freq],
                'db': [round(float(d), 2) for d in db],
                'fundamental_hz': line_freq,
                'sidebands': sidebands,
            },
        })
    return out


def _drop_leading_windows(win_sid: str, n: int) -> None:
    """Remove the first `n` windows (and aligned labels/categories) in place."""
    if n <= 0:
        return
    s = _data_sessions.get(win_sid)
    if not s or 'windows' not in s:
        return
    w = s['windows']
    if len(w) <= n:
        return
    s['windows'] = w[n:]
    for key in ('labels', 'categories'):
        v = s.get(key)
        if isinstance(v, (list, np.ndarray)) and len(v) > n:
            s[key] = v[n:]


def _solution_evidence(feat_sid: str, labels, classes: List[str],
                       feature_list: List[str]) -> Dict[str, Any]:
    """Per-group means of the key physics features — the plain-language evidence
    that makes a verdict explainable (e.g. sideband/envelope level by class)."""
    entry = _feature_sessions.get(feat_sid)
    if not entry:
        return {}
    df = entry['features']
    feats = [c for c in feature_list if c in df.columns]
    if not feats:
        return {}

    groups: Dict[str, Any] = {}
    if labels is not None and len(classes) > 1:
        lab = np.asarray([str(x) for x in labels])
        for c in classes:
            m = lab == c
            if m.sum():
                groups[c] = {f: round(float(np.mean(df[f].values[m])), 3) for f in feats}
    else:
        groups['all'] = {f: round(float(np.mean(df[f].values)), 3) for f in feats}

    return {'features': feats, 'by_group': groups}


def _mcsa_evidence(feat_sid: str, labels, classes: List[str]) -> Dict[str, Any]:
    return _solution_evidence(feat_sid, labels, classes, _MCSA_EVIDENCE_FEATURES)


# ── Machine Vibration (bearing envelope) ───────────────────────────────────

_BEARING_EVIDENCE_FEATURES = [
    'bearing_bpfo_env_db', 'bearing_bpfi_env_db', 'bearing_bsf_env_db',
    'bearing_ftf_env_db', 'bearing_kurtosis', 'bearing_crest_factor',
    'bearing_rms',
]


def _envelope_spectrum(sig: np.ndarray, fs: float, band_lo: float,
                       band_hi: float, fmax: float):
    """Envelope spectrum (amplitude vs Hz): bandpass → Hilbert envelope → FFT.
    dB relative to the envelope-spectrum peak. Native resolution up to fmax."""
    from scipy import signal as _sig
    x = np.asarray(sig, dtype=float)
    x = x - np.mean(x)
    nyq = fs / 2.0
    lo = max(1.0, band_lo)
    hi = min(band_hi, nyq * 0.98)
    if hi <= lo + 1.0 or len(x) < 32:
        return np.array([0.0]), np.array([-120.0])
    try:
        sos = _sig.butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
        env = np.abs(_sig.hilbert(_sig.sosfiltfilt(sos, x)))
        env = env - np.mean(env)
        w = np.hanning(len(env))
        mag = np.abs(np.fft.rfft(env * w))
        freq = np.fft.rfftfreq(len(env), 1.0 / fs)
        ref = float(np.max(mag)) if np.max(mag) > 0 else 1.0
        db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
        mask = freq <= fmax
        return freq[mask], db[mask]
    except Exception:
        return np.array([0.0]), np.array([-120.0])


def _bearing_fault_freqs(params: Dict[str, Any]) -> Dict[str, float]:
    fr = float(params['shaft_speed_hz'])
    n = int(params['n_balls'])
    bd = params.get('ball_diameter')
    pd_ = params.get('pitch_diameter')
    if bd and pd_ and pd_ > 0:
        ratio = (bd / pd_) * math.cos(math.radians(float(params['contact_angle_deg'])))
        return {'bpfo': (n / 2) * fr * (1 - ratio), 'bpfi': (n / 2) * fr * (1 + ratio),
                'bsf': (pd_ / (2 * bd)) * fr * (1 - ratio ** 2), 'ftf': (fr / 2) * (1 - ratio)}
    return {'bpfo': 0.4 * n * fr, 'bpfi': 0.6 * n * fr, 'bsf': 0.2 * n * fr, 'ftf': 0.4 * fr}


def _vibration_previews(win_sid: str, fs: float, rparams: Dict[str, Any],
                        max_windows: int = 6, preview_samples: int = 5000) -> List[dict]:
    """Per-window previews: native time slice + ENVELOPE spectrum with the
    bearing fault-frequency markers (BPFO/BPFI/BSF/FTF)."""
    s = _data_sessions.get(win_sid)
    if not s or 'windows' not in s or len(s['windows']) == 0:
        return []
    windows = s['windows']
    labels = s.get('labels')
    cols = (s.get('metadata') or {}).get('sensor_columns') or \
        [f'ch{i}' for i in range(np.asarray(windows[0]).shape[1])]
    axis = int(rparams.get('axis', 0))
    band_lo = float(rparams.get('band_low_hz', 1000.0))
    band_hi = float(rparams.get('band_high_hz', 5000.0))
    faults = _bearing_fault_freqs(rparams)
    fmax = max(faults.values()) * 3.0
    n = len(windows)
    idxs = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(max_windows, n))})
    out = []
    for i in idxs:
        w = np.asarray(windows[i], dtype=float)
        take = min(int(preview_samples), w.shape[0])
        seg = w[:take]
        channels = {cols[c]: [round(float(v), 5) for v in seg[:, c]]
                    for c in range(min(len(cols), w.shape[1]))}
        pch = axis if 0 <= axis < w.shape[1] else 0
        freq, db = _envelope_spectrum(w[:, pch], fs, band_lo, band_hi, fmax)
        out.append({
            'index': i,
            'label': (str(labels[i]) if labels is not None and i < len(labels) else None),
            'n_samples': int(w.shape[0]),
            'preview_samples': int(take),
            'channels': channels,
            'spectrum': {
                'freq': [round(float(f), 3) for f in freq],
                'db': [round(float(d), 2) for d in db],
                'fundamental_hz': round(float(rparams.get('shaft_speed_hz', 0)), 3),
                'sidebands': [round(float(v), 2) for v in faults.values()],
                'is_envelope': True,
            },
        })
    return out


_PUMP_EVIDENCE_FEATURES = [
    'pump_hf_energy_ratio', 'pump_spectral_entropy', 'pump_spectral_centroid_hz',
    'pump_bpf_db', 'pump_kurtosis', 'pump_crest_factor', 'pump_rms',
]


def _pump_previews(win_sid: str, fs: float, rparams: Dict[str, Any],
                   max_windows: int = 6, preview_samples: int = 5000) -> List[dict]:
    """Per-window previews: time slice + spectrum with shaft + blade-pass markers."""
    s = _data_sessions.get(win_sid)
    if not s or 'windows' not in s or len(s['windows']) == 0:
        return []
    windows = s['windows']
    labels = s.get('labels')
    cols = (s.get('metadata') or {}).get('sensor_columns') or \
        [f'ch{i}' for i in range(np.asarray(windows[0]).shape[1])]
    axis = int(rparams.get('axis', 0))
    fr = float(rparams.get('shaft_speed_hz', 25.0))
    bpf = int(rparams.get('n_blades', 6)) * fr
    fmax = min(float(rparams.get('cav_band_high_hz', 10000.0)), fs / 2 * 0.98)
    markers = [round(fr, 2), round(bpf, 2), round(2 * bpf, 2), round(3 * bpf, 2)]
    n = len(windows)
    idxs = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(max_windows, n))})
    out = []
    for i in idxs:
        w = np.asarray(windows[i], dtype=float)
        take = min(int(preview_samples), w.shape[0])
        seg = w[:take]
        channels = {cols[c]: [round(float(v), 5) for v in seg[:, c]]
                    for c in range(min(len(cols), w.shape[1]))}
        pch = axis if 0 <= axis < w.shape[1] else 0
        freq, db = _spectrum_dbc(w[:, pch], fs, fmax=fmax)
        out.append({
            'index': i,
            'label': (str(labels[i]) if labels is not None and i < len(labels) else None),
            'n_samples': int(w.shape[0]),
            'preview_samples': int(take),
            'channels': channels,
            'spectrum': {
                'freq': [round(float(f), 2) for f in freq],
                'db': [round(float(d), 2) for d in db],
                'fundamental_hz': round(fr, 2),
                'sidebands': [m for m in markers if 0 < m <= fmax],
            },
        })
    return out


def run_pump(
    data_session_id: str,
    sampling_rate: float,
    params: Dict[str, Any],
    window_s: float = 1.0,
    overlap: float = 0.5,
    approach: str = 'auto',
    algorithm: Optional[str] = None,
    selected_columns: Optional[List[str]] = None,
    project_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Centrifugal-pump (cavitation/impeller) diagnosis. Parallel to run_vibration;
    extractor 'pump_cavitation'."""
    if data_session_id not in _data_sessions:
        raise ValueError('Data session not found or expired — please reload your data.')
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError('Sampling rate must be a positive number (Hz).')
    overlap = min(0.95, max(0.0, float(overlap)))
    window_size = max(16, int(round(window_s * fs)))
    stride = max(16, int(round(window_size * (1.0 - overlap))))

    loader = DataLoader()
    win = loader.apply_windowing(data_session_id, window_size=window_size,
                                 stride=stride, selected_columns=selected_columns or None, normalization_method='none')
    win_sid = win['session_id']
    _data_sessions[win_sid].setdefault('metadata', {})['sampling_rate'] = fs

    from .extractors import registry as _ext_registry
    resolved_params = _ext_registry.get('pump_cavitation').validate_params(params)
    fe = FeatureExtractor()
    feat = fe.extract_with_registry(win_sid, 'pump_cavitation', params)
    feat_sid = feat['session_id']

    wsess = _data_sessions[win_sid]
    labels = wsess.get('labels')
    classes = sorted({str(x) for x in labels}) if labels is not None else []
    has_labeled_faults = len(classes) > 1
    chosen = approach if approach in ('anomaly', 'classification') else (
        'classification' if has_labeled_faults else 'anomaly')
    algo = algorithm or _DEFAULT_ALGO[chosen]

    trainer = MLTrainer()
    if chosen == 'classification':
        if not has_labeled_faults:
            raise ValueError('Classification needs labeled data with ≥2 classes, '
                             'or choose the Anomaly approach.')
        training = trainer.train_classification(feat_sid, algo, project_id=project_id, user_id=user_id)
    else:
        training = trainer.train_anomaly(feat_sid, algo, project_id=project_id, user_id=user_id)

    settings = {
        'approach': chosen, 'algorithm': algo, 'algorithm_name': _ALGO_NAMES.get(algo, algo),
        'window_s': window_s, 'window_size': window_size, 'stride': stride,
        'overlap': round(overlap, 3), 'sampling_rate': fs,
        'extractor': 'pump_cavitation', 'extractor_params': resolved_params,
        'phase_columns': selected_columns,
    }
    return {
        'solution': 'pump_analysis', 'mode': chosen, 'algorithm': algo,
        'algorithm_name': _ALGO_NAMES.get(algo, algo), 'classes': classes,
        'sampling_rate': fs, 'window_s': window_s, 'window_size': window_size,
        'num_windows': feat.get('num_windows'), 'num_features': feat.get('num_features'),
        'windows_failed': feat.get('windows_failed', 0),
        'windowed_session_id': win_sid, 'feature_session_id': feat_sid,
        'training': training,
        'evidence': _solution_evidence(feat_sid, labels, classes, _PUMP_EVIDENCE_FEATURES),
        'settings': settings,
        'window_previews': _pump_previews(win_sid, fs, resolved_params),
    }


_PUMP_FUSION_EVIDENCE_FEATURES = [
    'pfus_vib_hf_ratio', 'pfus_vib_entropy', 'pfus_vib_bpf_db',
    'pfus_park_ripple', 'pfus_park_ac_rms', 'pfus_park_lf_energy',
    'pfus_vib_kurtosis',
]


def run_pump_fusion(
    data_session_id: str,
    sampling_rate: float,
    params: Dict[str, Any],
    window_s: float = 1.0,
    overlap: float = 0.5,
    approach: str = 'auto',
    algorithm: Optional[str] = None,
    selected_columns: Optional[List[str]] = None,
    project_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Pump current+vibration fusion diagnosis. selected_columns must be
    [vibration, Ia, Ib, Ic] in that order. Extractor 'pump_fusion'."""
    if data_session_id not in _data_sessions:
        raise ValueError('Data session not found or expired — please reload your data.')
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError('Sampling rate must be a positive number (Hz).')
    overlap = min(0.95, max(0.0, float(overlap)))
    window_size = max(16, int(round(window_s * fs)))
    stride = max(16, int(round(window_size * (1.0 - overlap))))

    loader = DataLoader()
    win = loader.apply_windowing(data_session_id, window_size=window_size,
                                 stride=stride, selected_columns=selected_columns or None, normalization_method='none')
    win_sid = win['session_id']
    _data_sessions[win_sid].setdefault('metadata', {})['sampling_rate'] = fs

    from .extractors import registry as _ext_registry
    resolved_params = _ext_registry.get('pump_fusion').validate_params(params)
    fe = FeatureExtractor()
    feat = fe.extract_with_registry(win_sid, 'pump_fusion', params)
    feat_sid = feat['session_id']

    wsess = _data_sessions[win_sid]
    labels = wsess.get('labels')
    classes = sorted({str(x) for x in labels}) if labels is not None else []
    has_labeled_faults = len(classes) > 1
    chosen = approach if approach in ('anomaly', 'classification') else (
        'classification' if has_labeled_faults else 'anomaly')
    algo = algorithm or _DEFAULT_ALGO[chosen]

    trainer = MLTrainer()
    if chosen == 'classification':
        if not has_labeled_faults:
            raise ValueError('Classification needs labeled data with ≥2 classes, '
                             'or choose the Anomaly approach.')
        training = trainer.train_classification(feat_sid, algo, project_id=project_id, user_id=user_id)
    else:
        training = trainer.train_anomaly(feat_sid, algo, project_id=project_id, user_id=user_id)

    settings = {
        'approach': chosen, 'algorithm': algo, 'algorithm_name': _ALGO_NAMES.get(algo, algo),
        'window_s': window_s, 'window_size': window_size, 'stride': stride,
        'overlap': round(overlap, 3), 'sampling_rate': fs,
        'extractor': 'pump_fusion', 'extractor_params': resolved_params,
        'phase_columns': selected_columns,
    }
    return {
        'solution': 'pump_fusion', 'mode': chosen, 'algorithm': algo,
        'algorithm_name': _ALGO_NAMES.get(algo, algo), 'classes': classes,
        'sampling_rate': fs, 'window_s': window_s, 'window_size': window_size,
        'num_windows': feat.get('num_windows'), 'num_features': feat.get('num_features'),
        'windows_failed': feat.get('windows_failed', 0),
        'windowed_session_id': win_sid, 'feature_session_id': feat_sid,
        'training': training,
        'evidence': _solution_evidence(feat_sid, labels, classes, _PUMP_FUSION_EVIDENCE_FEATURES),
        'settings': settings,
        'window_previews': _pump_previews(win_sid, fs, resolved_params),
    }


def run_vibration(
    data_session_id: str,
    sampling_rate: float,
    params: Dict[str, Any],
    window_s: float = 1.0,
    overlap: float = 0.5,
    approach: str = 'auto',
    algorithm: Optional[str] = None,
    selected_columns: Optional[List[str]] = None,
    project_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Machine Vibration (bearing) diagnosis: window → bearing-envelope features
    → model → domain results. Parallel to run_mcsa; extractor 'bearing_envelope'."""
    if data_session_id not in _data_sessions:
        raise ValueError('Data session not found or expired — please reload your data.')
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError('Sampling rate must be a positive number (Hz).')

    overlap = min(0.95, max(0.0, float(overlap)))
    window_size = max(16, int(round(window_s * fs)))
    stride = max(16, int(round(window_size * (1.0 - overlap))))

    loader = DataLoader()
    win = loader.apply_windowing(
        data_session_id, window_size=window_size, stride=stride,
        selected_columns=selected_columns or None, normalization_method='none')
    win_sid = win['session_id']
    _data_sessions[win_sid].setdefault('metadata', {})['sampling_rate'] = fs

    from .extractors import registry as _ext_registry
    resolved_params = _ext_registry.get('bearing_envelope').validate_params(params)
    fe = FeatureExtractor()
    feat = fe.extract_with_registry(win_sid, 'bearing_envelope', params)
    feat_sid = feat['session_id']

    wsess = _data_sessions[win_sid]
    labels = wsess.get('labels')
    classes = sorted({str(x) for x in labels}) if labels is not None else []
    has_labeled_faults = len(classes) > 1
    chosen = approach if approach in ('anomaly', 'classification') else (
        'classification' if has_labeled_faults else 'anomaly')
    algo = algorithm or _DEFAULT_ALGO[chosen]

    trainer = MLTrainer()
    if chosen == 'classification':
        if not has_labeled_faults:
            raise ValueError('Classification needs labeled data with ≥2 classes, '
                             'or choose the Anomaly approach.')
        training = trainer.train_classification(feat_sid, algo, project_id=project_id, user_id=user_id)
    else:
        training = trainer.train_anomaly(feat_sid, algo, project_id=project_id, user_id=user_id)

    settings = {
        'approach': chosen, 'algorithm': algo, 'algorithm_name': _ALGO_NAMES.get(algo, algo),
        'window_s': window_s, 'window_size': window_size, 'stride': stride,
        'overlap': round(overlap, 3), 'sampling_rate': fs,
        'extractor': 'bearing_envelope', 'extractor_params': resolved_params,
        'phase_columns': selected_columns,
    }
    return {
        'solution': 'machine_vibration', 'mode': chosen, 'algorithm': algo,
        'algorithm_name': _ALGO_NAMES.get(algo, algo), 'classes': classes,
        'sampling_rate': fs, 'window_s': window_s, 'window_size': window_size,
        'num_windows': feat.get('num_windows'), 'num_features': feat.get('num_features'),
        'windows_failed': feat.get('windows_failed', 0),
        'windowed_session_id': win_sid, 'feature_session_id': feat_sid,
        'training': training,
        'evidence': _solution_evidence(feat_sid, labels, classes, _BEARING_EVIDENCE_FEATURES),
        'settings': settings,
        'window_previews': _vibration_previews(win_sid, fs, resolved_params),
    }
