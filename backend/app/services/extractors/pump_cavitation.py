"""CiRA ME — Centrifugal pump cavitation/impeller extractor (F8 Pump, 2026-09-18).

Physics-aware features for centrifugal-pump fault detection from vibration (or
pressure) signals:
  - Blade-pass frequency (BPF = n_blades × shaft) amplitude + harmonics →
    impeller / hydraulic-imbalance faults.
  - Cavitation indicators: high-frequency broadband energy ratio, spectral
    entropy, centroid, kurtosis (bubble-collapse raises high-freq broadband
    noise and impulsiveness).

Clean scipy implementation (dependency-light). The research Han-recipe
(wavelet+VMD+Park-vector+Hilbert marginal spectrum) is the richer multi-modal
approach; this is a focused single-signal v1.

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import stats

from .base import FeatureExtractor, ParamDef

_ABSENT_DB = -120.0


class PumpCavitationExtractor(FeatureExtractor):
    id = 'pump_cavitation'
    display_name = 'Pump Cavitation / Impeller Analysis'
    description = (
        'Centrifugal-pump fault features from vibration/pressure: blade-pass '
        'frequency amplitude + harmonics (impeller faults) and cavitation '
        'broadband indicators (high-frequency energy ratio, spectral entropy, '
        'centroid, kurtosis).'
    )
    param_schema = [
        ParamDef(
            name='shaft_speed_hz', label='Shaft speed', type='float',
            default=25.0, min=0.1, unit='Hz',
            help='Pump shaft rotation frequency (rpm / 60).',
        ),
        ParamDef(
            name='n_blades', label='Impeller blades', type='int',
            default=6, min=1,
            help='Number of impeller vanes (blade-pass = n_blades × shaft).',
        ),
        ParamDef(
            name='cav_band_low_hz', label='Cavitation band low', type='float',
            default=1000.0, min=100.0, unit='Hz',
            help='Low edge of the band where cavitation broadband energy sits. '
                 'Pump-specific — validated on 4TU data it was ~1-5 kHz. Widen/'
                 'narrow to the band where your cavitation signal rises.',
        ),
        ParamDef(
            name='cav_band_high_hz', label='Cavitation band high', type='float',
            default=9500.0, min=200.0, unit='Hz',
            help='High edge of the cavitation band. Clamped below Nyquist.',
        ),
        ParamDef(
            name='axis', label='Signal channel', type='int',
            default=0, min=0, max=3,
            help='Which channel (column) to analyze (vibration/pressure).',
        ),
    ]
    required_channels = 0
    min_sample_rate_hz = 8000.0      # cavitation broadband needs a high rate

    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        x = np.asarray(window, dtype=float)
        axis = int(params['axis'])
        if axis >= x.shape[1]:
            axis = 0
        sig = x[:, axis]
        sig = sig - np.mean(sig)

        fr = float(params['shaft_speed_hz'])
        n_blades = int(params['n_blades'])
        bpf = n_blades * fr

        rms = float(np.sqrt(np.mean(sig ** 2)))
        rms_safe = rms if rms > 0 else 1e-12
        feats: Dict[str, float] = {
            'pump_rms': rms,
            'pump_kurtosis': float(stats.kurtosis(sig)),
            'pump_crest_factor': float(np.max(np.abs(sig)) / rms_safe),
            'pump_peak_to_peak': float(np.max(sig) - np.min(sig)),
            'pump_shaft_hz': fr,
            'pump_bpf_hz': float(bpf),
        }

        # ── Spectrum ──
        w = np.hanning(len(sig))
        mag = np.abs(np.fft.rfft(sig * w))
        freq = np.fft.rfftfreq(len(sig), 1.0 / fs)
        power = mag ** 2
        total = float(np.sum(power)) or 1e-12
        ref = float(np.max(mag)) or 1e-12

        # Blade-pass amplitude + harmonics (dB over local floor).
        feats['pump_shaft_1x_db'] = self._peak_db(freq, mag, fr)
        feats['pump_bpf_db'] = self._peak_db(freq, mag, bpf)
        feats['pump_bpf_h2_db'] = self._peak_db(freq, mag, 2 * bpf)
        feats['pump_bpf_h3_db'] = self._peak_db(freq, mag, 3 * bpf)

        # Cavitation indicators.
        nyq = fs / 2.0
        lo = min(float(params['cav_band_low_hz']), nyq * 0.95)
        hi = min(float(params['cav_band_high_hz']), nyq * 0.99)
        band = (freq >= lo) & (freq <= hi)
        feats['pump_hf_energy_ratio'] = float(np.sum(power[band]) / total)

        p = power / total
        p = p[p > 0]
        feats['pump_spectral_entropy'] = float(-np.sum(p * np.log2(p))) if p.size else 0.0
        feats['pump_spectral_centroid_hz'] = float(np.sum(freq * power) / total)
        feats['pump_spectral_kurtosis'] = float(stats.kurtosis(mag))
        # broadband flatness: geometric/arithmetic mean of the spectrum
        gm = float(np.exp(np.mean(np.log(mag + 1e-12))))
        am = float(np.mean(mag)) or 1e-12
        feats['pump_spectral_flatness'] = gm / am

        return feats

    @staticmethod
    def _peak_db(freq, mag, target_hz: float) -> float:
        """Peak amplitude near target_hz in dB over the spectrum median."""
        if target_hz <= 0 or target_hz >= freq[-1]:
            return _ABSENT_DB
        tol = max(1.0, 0.03 * target_hz)
        band = (freq >= target_hz - tol) & (freq <= target_hz + tol)
        if not np.any(band):
            return _ABSENT_DB
        peak = float(np.max(mag[band]))
        floor = float(np.median(mag)) or 1e-12
        if peak <= 0:
            return _ABSENT_DB
        return 20.0 * math.log10(peak / floor)
