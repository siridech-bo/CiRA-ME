"""CiRA ME — Solutions Catalog (F8 Phase 0, 2026-09-17).

Declarative catalog of vertical predictive-maintenance TEMPLATES. Each
Solution is DATA (not code): a preset that fills in every pipeline step —
data-source profile → windowing → physics-aware feature extractor → model →
deploy format — so a user picks a use case and never touches the DSP details.

A Solution references its feature extractor by `id` (see
`services/extractors/`). It does NOT duplicate the extractor's parameter
form — `to_dict()` pulls `param_schema` from the referenced extractor so
there is one source of truth for nameplate fields.

`status` is DERIVED at projection time from whether the referenced extractor
is registered yet: 'ready' if the extractor exists, else 'planned'. This lets
the catalog list all planned verticals now while extractors land per phase.

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md §Phased rollout.
Adding a vertical = one Solution here + one extractor in services/extractors/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Solution:
    id: str
    display_name: str
    icon: str                 # mdi- name
    description: str
    data_profile: dict        # {'channels': int, 'sample_rate_hz': float, 'unit': str}
    windowing: dict           # {'window_s': float, ...}
    feature_extractor: str    # extractor id in services/extractors/
    models: dict              # {'default': algo_id, 'advanced': algo_id}
    deploy: dict              # {'targets': [...], 'format': str}
    source: str = 'builtin'   # 'builtin' | 'partner:<id>'  (F6 seam)
    tags: List[str] = field(default_factory=list)

    def to_dict(self, extractor_registry=None) -> dict:
        """JSON-safe projection for the /solutions endpoint.

        When `extractor_registry` is supplied, derive availability + the
        parameter form from the referenced extractor (single source of
        truth). Without it, params is empty and status is 'unknown'.
        """
        available = bool(extractor_registry and extractor_registry.has(self.feature_extractor))
        param_schema: List[dict] = []
        if available:
            ext = extractor_registry.get(self.feature_extractor)
            param_schema = [p.to_dict() for p in ext.param_schema]
        return {
            'id': self.id,
            'display_name': self.display_name,
            'icon': self.icon,
            'description': self.description,
            'data_profile': dict(self.data_profile),
            'windowing': dict(self.windowing),
            'feature_extractor': self.feature_extractor,
            'param_schema': param_schema,
            'models': dict(self.models),
            'deploy': dict(self.deploy),
            'source': self.source,
            'tags': list(self.tags),
            'extractor_available': available,
            'status': ('ready' if available else 'planned')
            if extractor_registry else 'unknown',
        }


# ── Catalog ────────────────────────────────────────────────────────────────
# All three verticals are listed now; their extractors land per phase, so
# 'planned' entries render in the catalog as "coming soon" until the
# extractor is registered.


_MOTOR_CURRENT_MCSA = Solution(
    id='motor_current_mcsa',
    display_name='Motor Current (MCSA)',
    icon='mdi-flash',
    description=(
        'Induction-motor fault diagnosis from 3-phase stator current. '
        'Detects broken rotor bars, eccentricity and unbalance via current '
        'sideband analysis — anomaly-first (healthy data only), classification '
        'when labeled faults exist.'
    ),
    data_profile={'channels': 3, 'sample_rate_hz': 5000.0, 'unit': 'a'},
    windowing={'window_s': 10.0},                       # the non-obvious one
    feature_extractor='mcsa',                           # Phase 1 (statorscope)
    models={'default': 'iforest', 'advanced': 'xgb'},
    deploy={'targets': ['linux_api'], 'format': 'onnx'},
    tags=['motor', 'current', 'mcsa', 'anomaly'],
)

_MACHINE_VIBRATION = Solution(
    id='machine_vibration',
    display_name='Machine Vibration',
    icon='mdi-vibrate',
    description=(
        'Rolling-element bearing fault diagnosis from accelerometer vibration. '
        'Bearing characteristic frequencies (BPFO/BPFI/BSF/FTF) via envelope '
        'analysis and kurtogram-selected demodulation.'
    ),
    data_profile={'channels': 1, 'sample_rate_hz': 25600.0, 'unit': 'g'},
    windowing={'window_s': 1.0},
    feature_extractor='bearing_envelope',               # Phase 2
    models={'default': 'iforest', 'advanced': 'xgb'},
    deploy={'targets': ['linux_api'], 'format': 'onnx'},
    tags=['vibration', 'bearing', 'envelope', 'anomaly'],
)

_PUMP_ANALYSIS = Solution(
    id='pump_analysis',
    display_name='Pump Analysis',
    icon='mdi-pump',
    description=(
        'Centrifugal-pump cavitation and impeller-fault detection from a '
        'vibration (or pressure) signal: blade-pass frequency analysis plus '
        'cavitation broadband indicators (high-frequency energy ratio, spectral '
        'entropy/centroid). Motor-current fusion (Park vector modulus + VMD) is '
        'a planned enhancement.'
    ),
    data_profile={'channels': 1, 'sample_rate_hz': 20000.0, 'unit': 'g'},
    windowing={'window_s': 1.0},   # 1 s @ 20 kHz = 20k samples — plenty for cavitation
    feature_extractor='pump_cavitation',                # Phase 3
    models={'default': 'iforest', 'advanced': 'xgb'},
    deploy={'targets': ['linux_api'], 'format': 'onnx'},
    tags=['pump', 'cavitation', 'fusion', 'anomaly'],
)


_PUMP_FUSION = Solution(
    id='pump_fusion',
    display_name='Pump Fusion (Current + Vibration)',
    icon='mdi-pump',
    description=(
        'Multi-modal centrifugal-pump diagnosis fusing 3-phase motor current '
        '(Park vector modulus + spectrum) with pump vibration (cavitation / '
        'blade-pass). Load a recording with a vibration column + the 3 current '
        'phases (Ia, Ib, Ic).'
    ),
    data_profile={'channels': 4, 'sample_rate_hz': 20000.0, 'unit': 'mixed'},
    windowing={'window_s': 1.0},
    feature_extractor='pump_fusion',
    models={'default': 'iforest', 'advanced': 'xgb'},
    deploy={'targets': ['linux_api'], 'format': 'onnx'},
    tags=['pump', 'cavitation', 'current', 'vibration', 'fusion'],
)


_SOLUTIONS: Dict[str, Solution] = {
    s.id: s for s in [
        _MOTOR_CURRENT_MCSA,
        _MACHINE_VIBRATION,
        _PUMP_ANALYSIS,
        _PUMP_FUSION,
    ]
}


def get_all_solutions() -> List[Solution]:
    """List every solution in the catalog. Ordered by insertion above."""
    return list(_SOLUTIONS.values())


def get_solution(solution_id: str) -> Optional[Solution]:
    return _SOLUTIONS.get(solution_id)
