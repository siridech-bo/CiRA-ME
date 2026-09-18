"""CiRA ME — Bearing envelope-analysis extractor (F8 Machine Vibration, 2026-09-18).

Rolling-element bearing fault features from accelerometer vibration, via the
standard envelope-analysis method:
  bandpass around a high-frequency resonance → Hilbert envelope →
  envelope spectrum → amplitude at the bearing characteristic frequencies
  (BPFO / BPFI / BSF / FTF, derived from bearing geometry + shaft speed).

Clean scipy implementation (textbook envelope analysis — Randall & Antoni).
Feature names are stable and self-describing (e.g. `bearing_bpfo_env_db`).

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md · research recommended
tzarcrept/Kurtogram/VibPy; this is an equivalent focused implementation.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import signal, stats

from .base import FeatureExtractor, ParamDef

_ABSENT_DB = -120.0


class BearingEnvelopeExtractor(FeatureExtractor):
    id = 'bearing_envelope'
    display_name = 'Bearing Envelope Analysis'
    description = (
        'Rolling-element bearing fault features from accelerometer vibration: '
        'BPFO/BPFI/BSF/FTF amplitudes in the envelope spectrum (dB over the '
        'local floor) plus time-domain kurtosis/crest. Bandpass → Hilbert '
        'envelope → envelope spectrum at geometry-derived fault frequencies.'
    )
    param_schema = [
        ParamDef(
            name='shaft_speed_hz', label='Shaft speed', type='float',
            default=30.0, min=0.1, unit='Hz',
            help='Rotation frequency of the shaft (rpm / 60).',
        ),
        ParamDef(
            name='n_balls', label='Rolling elements', type='int',
            default=8, min=3,
            help='Number of balls/rollers in the bearing.',
        ),
        ParamDef(
            name='ball_diameter', label='Ball diameter', type='float',
            default=None, required=False, min=0.1, unit='mm',
            help='Rolling-element diameter. Optional — with pitch diameter, '
                 'gives exact fault frequencies; else standard approximations.',
        ),
        ParamDef(
            name='pitch_diameter', label='Pitch diameter', type='float',
            default=None, required=False, min=0.1, unit='mm',
            help='Bearing pitch diameter. Optional.',
        ),
        ParamDef(
            name='contact_angle_deg', label='Contact angle', type='float',
            default=0.0, min=0.0, max=45.0, unit='deg',
        ),
        ParamDef(
            name='band_low_hz', label='Resonance band low', type='float',
            default=1000.0, min=0.0, unit='Hz',
            help='Low edge of the demodulation band (bearing resonance).',
        ),
        ParamDef(
            name='band_high_hz', label='Resonance band high', type='float',
            default=5000.0, min=1.0, unit='Hz',
            help='High edge of the demodulation band. Clamped below Nyquist.',
        ),
        ParamDef(
            name='axis', label='Accelerometer axis', type='int',
            default=0, min=0, max=2,
            help='Which channel (column) to analyze for multi-axis sensors.',
        ),
    ]
    required_channels = 0            # 1-axis or multi-axis
    min_sample_rate_hz = 5000.0      # bearing resonances live at kHz

    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        x = np.asarray(window, dtype=float)
        axis = int(params['axis'])
        if axis >= x.shape[1]:
            axis = 0
        sig = x[:, axis]
        sig = sig - np.mean(sig)

        fr = float(params['shaft_speed_hz'])
        n = int(params['n_balls'])
        bd = params.get('ball_diameter')
        pd_ = params.get('pitch_diameter')
        angle = float(params['contact_angle_deg'])

        # ── Bearing characteristic frequencies ──
        if bd and pd_ and pd_ > 0:
            ratio = (bd / pd_) * math.cos(math.radians(angle))
        else:
            ratio = None
        if ratio is not None:
            bpfo = (n / 2.0) * fr * (1.0 - ratio)
            bpfi = (n / 2.0) * fr * (1.0 + ratio)
            bsf = (pd_ / (2.0 * bd)) * fr * (1.0 - ratio ** 2)
            ftf = (fr / 2.0) * (1.0 - ratio)
        else:
            # Standard approximations when geometry is unknown.
            bpfo = 0.40 * n * fr
            bpfi = 0.60 * n * fr
            bsf = 0.20 * n * fr
            ftf = 0.40 * fr
        faults = {'bpfo': bpfo, 'bpfi': bpfi, 'bsf': bsf, 'ftf': ftf}

        # ── Time-domain features (impulsiveness is the bearing-fault tell) ──
        rms = float(np.sqrt(np.mean(sig ** 2)))
        rms_safe = rms if rms > 0 else 1e-12
        feats: Dict[str, float] = {
            'bearing_rms': rms,
            'bearing_kurtosis': float(stats.kurtosis(sig)),
            'bearing_crest_factor': float(np.max(np.abs(sig)) / rms_safe),
            'bearing_peak_to_peak': float(np.max(sig) - np.min(sig)),
            'bearing_shaft_hz': fr,
        }

        # ── Envelope spectrum ──
        nyq = fs / 2.0
        lo = max(1.0, float(params['band_low_hz']))
        hi = min(float(params['band_high_hz']), nyq * 0.98)
        env_freq, env_mag = None, None
        if hi > lo + 1.0 and len(sig) > 32:
            try:
                sos = signal.butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
                filt = signal.sosfiltfilt(sos, sig)
                env = np.abs(signal.hilbert(filt))
                env = env - np.mean(env)
                w = np.hanning(len(env))
                env_mag = np.abs(np.fft.rfft(env * w))
                env_freq = np.fft.rfftfreq(len(env), 1.0 / fs)
            except Exception:
                env_freq = env_mag = None

        # spectral kurtosis of the demod band (band-selection quality signal)
        feats['bearing_band_kurtosis'] = (
            float(stats.kurtosis(filt)) if env_mag is not None else 0.0)

        # amplitude (dB over local floor) at each fault frequency + 2nd harmonic
        for name, f0 in faults.items():
            feats[f'bearing_{name}_hz'] = float(f0)
            feats[f'bearing_{name}_env_db'] = self._env_db(env_freq, env_mag, f0, fs)
            feats[f'bearing_{name}_h2_db'] = self._env_db(env_freq, env_mag, 2 * f0, fs)

        return feats

    @staticmethod
    def _env_db(freq, mag, target_hz: float, fs: float) -> float:
        """Peak envelope amplitude near target_hz, in dB over the spectrum
        median (prominence). Returns the absent floor if unusable."""
        if freq is None or mag is None or target_hz <= 0 or target_hz >= freq[-1]:
            return _ABSENT_DB
        tol = max(1.0, 0.02 * target_hz)          # ±2% search window
        band = (freq >= target_hz - tol) & (freq <= target_hz + tol)
        if not np.any(band):
            return _ABSENT_DB
        peak = float(np.max(mag[band]))
        floor = float(np.median(mag)) or 1e-12
        if peak <= 0:
            return _ABSENT_DB
        return 20.0 * math.log10(peak / floor)
