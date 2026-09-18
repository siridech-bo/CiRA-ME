"""CiRA ME — Reference extractor: raw per-channel statistics (F8 Phase 0).

Not a physics-aware vertical extractor — it exists to (1) prove the
FeatureExtractor contract end-to-end and (2) give the Features step a
working third mode before `mcsa` / `bearing_envelope` / `pump_cavitation`
are vendored in later phases. Depends on numpy + scipy only.

Feature names follow the stable, self-describing convention:
`ch{index}_{stat}` (e.g. 'ch0_rms', 'ch2_kurtosis').
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from .base import FeatureExtractor, ParamDef


class RawStatsExtractor(FeatureExtractor):
    id = 'raw_stats'
    display_name = 'Raw Statistics (reference)'
    description = (
        'Per-channel time-domain statistics (mean, std, rms, peak-to-peak, '
        'skewness, kurtosis, crest factor). Reference extractor proving the '
        'registry contract; not tuned for any specific fault physics.'
    )
    param_schema = [
        ParamDef(
            name='detrend', label='Remove DC offset', type='bool',
            default=True, help='Subtract per-channel mean before stats.',
        ),
    ]
    required_channels = 0        # any
    min_sample_rate_hz = 0.0

    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        x = np.asarray(window, dtype=float)
        if params.get('detrend'):
            x = x - np.mean(x, axis=0, keepdims=True)

        feats: Dict[str, float] = {}
        n_ch = x.shape[1]
        for c in range(n_ch):
            col = x[:, c]
            rms = float(np.sqrt(np.mean(col ** 2)))
            rms_safe = rms if rms > 0 else 1e-10
            feats[f'ch{c}_mean'] = float(np.mean(col))
            feats[f'ch{c}_std'] = float(np.std(col))
            feats[f'ch{c}_rms'] = rms
            feats[f'ch{c}_peak_to_peak'] = float(np.max(col) - np.min(col))
            feats[f'ch{c}_skewness'] = float(stats.skew(col))
            feats[f'ch{c}_kurtosis'] = float(stats.kurtosis(col))
            feats[f'ch{c}_crest_factor'] = float(np.max(np.abs(col)) / rms_safe)
        return feats
