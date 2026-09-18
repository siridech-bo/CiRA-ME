"""CiRA ME — MCSA extractor (F8 Phase 1, 2026-09-17).

Motor Current Signature Analysis: turns a window of 3-phase (or single-phase)
stator current into physically-meaningful fault features — broken-rotor-bar,
stator inter-turn, and eccentricity sideband levels plus a sensorless slip
estimate — by adapting our FeatureExtractor contract to the vendored
`statorscope` library (see vendor/statorscope/PROVENANCE.md).

Feature names are stable and self-describing so tree-model importances stay
explainable, e.g. `mcsa_brb_lsb_db` = broken-rotor-bar lower-sideband level
in dBc relative to the fundamental.

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md §Phase 1.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .base import FeatureExtractor, ParamDef
from .vendor.statorscope import Motor, Recording, diagnose

#: dBc value assigned to a fault component that was absent / unmeasurable.
#: NOT 0.0 — 0 dBc means "at carrier level" (catastrophic), the opposite of
#: absent. Real fault sidebands live at -25..-70 dBc, so a floor well below
#: anything physical reads to the model as "nothing there".
ABSENT_DBC = -120.0


class MCSAExtractor(FeatureExtractor):
    id = 'mcsa'
    display_name = 'Motor Current Signature Analysis'
    description = (
        'Induction-motor fault features from stator current: broken-rotor-bar '
        '(1±2ks)·f, stator inter-turn, and eccentricity sideband levels/prominence, '
        'plus sensorless per-window slip. Levels are dBc relative to the '
        'fundamental. Powered by the vendored statorscope library.'
    )
    param_schema = [
        ParamDef(
            name='pole_pairs', label='Pole pairs', type='int',
            default=2, min=1,
            help='Pole PAIRS, not poles. A 4-pole motor has pole_pairs=2.',
        ),
        ParamDef(
            name='line_freq_hz', label='Line frequency', type='float',
            default=50.0, choices=[50.0, 60.0], unit='Hz',
            help='Supply frequency (mains).',
        ),
        ParamDef(
            name='rotor_bars', label='Rotor bars', type='int',
            default=None, required=False, min=1,
            help='Number of rotor bars/slots. Optional — improves eccentricity '
                 'and rotor-slot models.',
        ),
        ParamDef(
            name='rated_rpm', label='Rated speed', type='float',
            default=None, required=False, min=1.0, unit='rpm',
            help='Nameplate full-load speed. Optional — bounds the slip search.',
        ),
        ParamDef(
            name='phase', label='Phase to analyze', type='int',
            default=0, min=0, max=2,
            help='Which current phase (column) to analyze.',
        ),
        ParamDef(
            name='calibrate', label='Grid-lock sample rate', type='bool',
            default=True,
            help='Correct the sample rate from the supply before analysis. '
                 'Leave on unless the acquisition has a verified hardware clock.',
        ),
        ParamDef(
            name='absent_dbc', label='Absent-component floor', type='float',
            default=ABSENT_DBC, min=-200.0, max=-40.0, unit='dBc',
            help='dBc value assigned to a fault component that was not '
                 'measurable. The default (-120) is deliberately extreme so a '
                 'tree model reads it as "nothing there"; anomaly models '
                 '(IsolationForest) work better with a less extreme floor '
                 '(e.g. -90) since -120 creates outlier points. See '
                 'docs/VALIDATION-2026-09-18_mcsa-unesp.md.',
        ),
    ]
    required_channels = 0            # works on 1- or 3-phase current
    min_sample_rate_hz = 1000.0      # need Nyquist clear of the stator harmonics

    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        x = np.asarray(window, dtype=float)
        n_ch = x.shape[1]
        phase = int(params['phase'])
        if phase >= n_ch:
            phase = 0

        rec = Recording.from_array(x, float(fs), name='mcsa_window')
        motor = Motor(
            pole_pairs=int(params['pole_pairs']),
            rotor_bars=(int(params['rotor_bars']) if params.get('rotor_bars') else None),
            line_hz=float(params['line_freq_hz']),
            rated_rpm=(float(params['rated_rpm']) if params.get('rated_rpm') else None),
        )
        diag = diagnose(rec, motor, phase=phase, calibrate=bool(params['calibrate']))
        floor = float(params.get('absent_dbc', ABSENT_DBC))
        return self._features_from_diagnosis(diag, floor)

    # ── projection: Diagnosis → flat feature dict ──
    def _features_from_diagnosis(self, diag, floor: float = ABSENT_DBC) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        lvl = lambda v: (float(v) if math.isfinite(v) else floor)  # noqa: E731

        # Slip + acquisition-trust block
        slip = diag.slip
        feats['mcsa_slip'] = float(slip.slip)
        feats['mcsa_slip_rpm'] = float(slip.rpm)
        feats['mcsa_slip_score_db'] = lvl(slip.score_db)
        feats['mcsa_slip_confident'] = 1.0 if slip.confident else 0.0
        feats['mcsa_clock_trustworthy'] = 1.0 if diag.clock.trustworthy else 0.0
        feats['mcsa_supported'] = 1.0 if diag.supported else 0.0

        by_kind = {f.kind: f for f in diag.faults}

        # Broken rotor bar — the classic signature. Pull the k=1 lower/upper
        # sidebands specifically (evidence is ordered (1-2s)f, (1+2s)f, ...).
        brb = by_kind.get('broken_rotor_bar')
        if brb is not None:
            feats['mcsa_brb_strongest_db'] = lvl(brb.strongest_dbc)
            feats['mcsa_brb_detected'] = 1.0 if brb.detected else 0.0
            ev = brb.evidence
            lsb = ev[0] if len(ev) >= 1 else None
            usb = ev[1] if len(ev) >= 2 else None
            feats['mcsa_brb_lsb_db'] = lvl(lsb.level_dbc) if lsb else floor
            feats['mcsa_brb_lsb_prom_db'] = lvl(lsb.prominence_db) if lsb else 0.0
            feats['mcsa_brb_usb_db'] = lvl(usb.level_dbc) if usb else floor
            feats['mcsa_brb_usb_prom_db'] = lvl(usb.prominence_db) if usb else 0.0

        # Stator inter-turn
        stator = by_kind.get('stator_interturn')
        if stator is not None:
            feats['mcsa_stator_strongest_db'] = lvl(stator.strongest_dbc)
            feats['mcsa_stator_detected'] = 1.0 if stator.detected else 0.0

        # Eccentricity
        ecc = by_kind.get('eccentricity')
        if ecc is not None:
            feats['mcsa_ecc_strongest_db'] = lvl(ecc.strongest_dbc)
            feats['mcsa_ecc_detected'] = 1.0 if ecc.detected else 0.0

        return feats
