"""CiRA ME — Feature-extractor registry package (F8 Phase 0).

Import `registry` to look up extractors by id. Built-in extractors register
themselves here at import time. Physics-aware vertical extractors land in
later phases and are added to `_BUILTINS`:
  - Phase 1: `mcsa`             (vendored statorscope, Apache-2.0)
  - Phase 2: `bearing_envelope` (vendored tzarcrept + Kurtogram)
  - Phase 3: `pump_cavitation`  (Han-recipe assembly)

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md
"""

from __future__ import annotations

from .base import FeatureExtractor, ParamDef, ExtractorRegistry
from .raw_stats import RawStatsExtractor
from .mcsa import MCSAExtractor
from .bearing_envelope import BearingEnvelopeExtractor
from .pump_cavitation import PumpCavitationExtractor
from .pump_fusion import PumpFusionExtractor

registry = ExtractorRegistry()

# Built-in extractors, registered in order. Add vertical extractors here as
# each phase lands.
_BUILTINS = [
    RawStatsExtractor,
    MCSAExtractor,             # Motor Current — vendored statorscope
    BearingEnvelopeExtractor,  # Machine Vibration — envelope analysis
    PumpCavitationExtractor,   # Pump — cavitation / blade-pass (vibration)
    PumpFusionExtractor,       # Pump — current + vibration fusion
]

for _cls in _BUILTINS:
    registry.register(_cls())


__all__ = ['registry', 'FeatureExtractor', 'ParamDef', 'ExtractorRegistry']
