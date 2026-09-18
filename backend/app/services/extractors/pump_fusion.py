"""CiRA ME — Pump current+vibration FUSION extractor (F8, 2026-09-18).

Multi-modal centrifugal-pump fault features fusing:
  - 3-phase MOTOR CURRENT via the Park vector modulus (Clarke transform →
    |i| = sqrt(Id²+Iq²)); cavitation / hydraulic instability modulates it, so
    the modulus ripple + its spectrum are current-side fault features.
  - VIBRATION cavitation/blade-pass features (broadband HF energy, spectral
    entropy, blade-pass amplitude).

Input window must carry 4 channels in order: [vibration, Ia, Ib, Ic].
Clean numpy/scipy implementation. (Wavelet-denoise + VMD from the Han recipe
are documented future refinements; not implemented here.)

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import signal, stats

from .base import FeatureExtractor, ParamDef

_ABSENT_DB = -120.0
_SQRT23 = math.sqrt(2.0 / 3.0)
_INV_SQRT6 = 1.0 / math.sqrt(6.0)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class PumpFusionExtractor(FeatureExtractor):
    id = 'pump_fusion'
    display_name = 'Pump Fusion (Current + Vibration)'
    description = (
        'Multi-modal centrifugal-pump fault features: 3-phase current Park '
        'vector modulus (ripple + spectrum) fused with vibration cavitation / '
        'blade-pass features. Input channels: [vibration, Ia, Ib, Ic].'
    )
    param_schema = [
        ParamDef(name='shaft_speed_hz', label='Shaft speed', type='float',
                 default=25.0, min=0.1, unit='Hz', help='Pump shaft rpm ÷ 60.'),
        ParamDef(name='n_blades', label='Impeller blades', type='int',
                 default=6, min=1, help='Blade-pass = n_blades × shaft.'),
        ParamDef(name='line_freq_hz', label='Line frequency', type='float',
                 default=50.0, choices=[50.0, 60.0], unit='Hz',
                 help='Mains frequency of the motor current.'),
        ParamDef(name='cav_band_low_hz', label='Cavitation band low', type='float',
                 default=1000.0, min=100.0, unit='Hz',
                 help='Low edge of the vibration cavitation band (pump-specific).'),
        ParamDef(name='cav_band_high_hz', label='Cavitation band high', type='float',
                 default=9500.0, min=200.0, unit='Hz',
                 help='High edge (clamped below Nyquist).'),
    ]
    required_channels = 4        # [vibration, Ia, Ib, Ic]
    min_sample_rate_hz = 8000.0

    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        x = np.asarray(window, dtype=float)
        if x.shape[1] < 4:
            raise ValueError(
                'Pump fusion needs 4 channels: [vibration, Ia, Ib, Ic]. '
                f'Got {x.shape[1]}. Select the vibration column then the 3 current phases.')
        vib = x[:, 0] - np.mean(x[:, 0])
        ia, ib, ic = x[:, 1], x[:, 2], x[:, 3]

        fr = float(params['shaft_speed_hz'])
        n_blades = int(params['n_blades'])
        bpf = n_blades * fr
        feats: Dict[str, float] = {'pfus_shaft_hz': fr, 'pfus_bpf_hz': float(bpf)}

        # ── Current: Park vector modulus (Clarke transform) ──
        id_ = _SQRT23 * ia - _INV_SQRT6 * ib - _INV_SQRT6 * ic
        iq = _INV_SQRT2 * ib - _INV_SQRT2 * ic
        modulus = np.sqrt(id_ ** 2 + iq ** 2)
        m_mean = float(np.mean(modulus)) or 1e-12
        m_ac = modulus - m_mean
        feats['pfus_park_mean'] = m_mean
        feats['pfus_park_ripple'] = float(np.std(modulus) / abs(m_mean))   # coeff of variation
        feats['pfus_park_ac_rms'] = float(np.sqrt(np.mean(m_ac ** 2)))
        # spectrum of the modulus AC — cavitation adds low-frequency fluctuation.
        mfreq, mmag = self._spectrum(m_ac, fs, fmax=min(500.0, fs / 2 * 0.9))
        if mmag is not None and mmag.size:
            mp = mmag ** 2
            feats['pfus_park_lf_energy'] = float(np.sum(mp[mfreq <= 100]) / (np.sum(mp) or 1e-12))
            p = mp / (np.sum(mp) or 1e-12)
            p = p[p > 0]
            feats['pfus_park_spec_entropy'] = float(-np.sum(p * np.log2(p))) if p.size else 0.0
        else:
            feats['pfus_park_lf_energy'] = 0.0
            feats['pfus_park_spec_entropy'] = 0.0

        # ── Vibration: cavitation + blade-pass ──
        rms = float(np.sqrt(np.mean(vib ** 2)))
        feats['pfus_vib_rms'] = rms
        feats['pfus_vib_kurtosis'] = float(stats.kurtosis(vib))
        vfreq, vmag = self._spectrum(vib, fs, fmax=fs / 2 * 0.99)
        if vmag is not None and vmag.size:
            vp = vmag ** 2
            vtot = float(np.sum(vp)) or 1e-12
            nyq = fs / 2.0
            lo = min(float(params['cav_band_low_hz']), nyq * 0.95)
            hi = min(float(params['cav_band_high_hz']), nyq * 0.99)
            band = (vfreq >= lo) & (vfreq <= hi)
            feats['pfus_vib_hf_ratio'] = float(np.sum(vp[band]) / vtot)
            feats['pfus_vib_centroid_hz'] = float(np.sum(vfreq * vp) / vtot)
            p = vp / vtot
            p = p[p > 0]
            feats['pfus_vib_entropy'] = float(-np.sum(p * np.log2(p))) if p.size else 0.0
            feats['pfus_vib_bpf_db'] = self._peak_db(vfreq, vmag, bpf)
        else:
            for k in ('pfus_vib_hf_ratio', 'pfus_vib_centroid_hz', 'pfus_vib_entropy'):
                feats[k] = 0.0
            feats['pfus_vib_bpf_db'] = _ABSENT_DB
        return feats

    @staticmethod
    def _spectrum(sig: np.ndarray, fs: float, fmax: float):
        x = np.asarray(sig, dtype=float)
        if len(x) < 32:
            return None, None
        w = np.hanning(len(x))
        mag = np.abs(np.fft.rfft((x - np.mean(x)) * w))
        freq = np.fft.rfftfreq(len(x), 1.0 / fs)
        mask = freq <= fmax
        return freq[mask], mag[mask]

    @staticmethod
    def _peak_db(freq, mag, target_hz: float) -> float:
        if target_hz <= 0 or target_hz >= freq[-1]:
            return _ABSENT_DB
        tol = max(1.0, 0.03 * target_hz)
        band = (freq >= target_hz - tol) & (freq <= target_hz + tol)
        if not np.any(band):
            return _ABSENT_DB
        peak = float(np.max(mag[band]))
        floor = float(np.median(mag)) or 1e-12
        return 20.0 * math.log10(peak / floor) if peak > 0 else _ABSENT_DB
