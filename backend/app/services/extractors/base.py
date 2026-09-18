"""CiRA ME — Feature-Extractor Registry: contract & registry (F8 Phase 0).

A pluggable registry of physics-aware feature extractors. Each Solution
template (`constants/solutions.py`) references an extractor by `id`; the
Features pipeline step invokes it to turn a window into named,
physically-meaningful features.

Design ref: docs/PLAN_2026-09-17_solutions-catalog.md §Architecture.

Contract rules — STABLE across versions (Partner-SDK extractors depend on
this; see docs/PLAN_2026-08-12_partner-sdk.md):
- Feature NAMES must be stable and self-describing (e.g. 'mcsa_brb_lsb_db',
  'bpfo_env_peak_db') so tree-model feature importances stay explainable.
- extract() returns a flat {name: float} dict — no NaN/Inf (sanitize to 0.0).
- Extractors are STATELESS: all tuning arrives via `params` (validated
  against `param_schema`) so one instance is safe to reuse and serialize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

import numpy as np


# ── Parameter schema ──────────────────────────────────────────────────────


@dataclass
class ParamDef:
    """One nameplate/config field an extractor needs (e.g. motor poles).

    Rendered as a form field by the Solution template UI and validated
    (defaults filled, types coerced, ranges checked) before `extract()`.
    """
    name: str                       # key in the params dict, e.g. 'poles'
    label: str                      # UI label, e.g. 'Number of poles'
    type: str                       # 'int' | 'float' | 'enum' | 'bool'
    default: Any = None
    required: bool = True           # if False, a None value stays None
    min: Optional[float] = None     # for int/float
    max: Optional[float] = None
    choices: Optional[List[Any]] = None   # for enum
    unit: Optional[str] = None      # UI suffix, e.g. 'Hz'
    help: Optional[str] = None      # tooltip

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'label': self.label,
            'type': self.type,
            'default': self.default,
            'required': self.required,
            'min': self.min,
            'max': self.max,
            'choices': list(self.choices) if self.choices else None,
            'unit': self.unit,
            'help': self.help,
        }

    def coerce(self, value: Any) -> Any:
        """Coerce+validate one value against this def. Raises ValueError."""
        if value is None:
            value = self.default
        if value is None:
            if not self.required:
                return None       # optional field left unset — pass through
            raise ValueError(f"param '{self.name}' is required")
        try:
            if self.type == 'int':
                value = int(value)
            elif self.type == 'float':
                value = float(value)
            elif self.type == 'bool':
                value = bool(value)
        except (TypeError, ValueError):
            raise ValueError(f"param '{self.name}' must be {self.type}")
        if self.type in ('int', 'float'):
            if self.min is not None and value < self.min:
                raise ValueError(f"param '{self.name}' < min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"param '{self.name}' > max {self.max}")
        if self.type == 'enum' and self.choices and value not in self.choices:
            raise ValueError(f"param '{self.name}' not in {self.choices}")
        return value


# ── Extractor contract ────────────────────────────────────────────────────


class FeatureExtractor:
    """Base class every extractor conforms to — first-party, vendored, or
    Partner-SDK. Subclass, set the class attributes, implement `extract()`.

    Keep the class importable with only numpy/scipy at module load; pull
    heavier / vendored deps inside `extract()` so registry construction
    never fails because an optional dependency is missing.
    """

    id: str = ''
    display_name: str = ''
    description: str = ''
    param_schema: List[ParamDef] = []
    required_channels: int = 0      # 0 = any channel count
    min_sample_rate_hz: float = 0.0

    # ── override this ──
    def extract(self, window: np.ndarray, fs: float, params: dict) -> Dict[str, float]:
        """Turn one window into named features.

        window: (n_samples, n_channels) float array.
        fs:     sampling rate in Hz.
        params: already validated via `validate_params()`.
        Returns a flat {feature_name: float} dict (finite values only).
        """
        raise NotImplementedError

    # ── shared helpers ──
    def validate_params(self, params: Optional[dict]) -> dict:
        """Fill defaults, coerce types, range-check. Raises ValueError."""
        params = dict(params or {})
        out = {}
        for pdef in self.param_schema:
            out[pdef.name] = pdef.coerce(params.get(pdef.name))
        return out

    def check_signal(self, window: np.ndarray, fs: float) -> None:
        """Guard channel count / sample rate before extraction."""
        if window.ndim != 2:
            raise ValueError("window must be 2-D (n_samples, n_channels)")
        n_ch = window.shape[1]
        if self.required_channels and n_ch != self.required_channels:
            raise ValueError(
                f"extractor '{self.id}' needs {self.required_channels} "
                f"channels, got {n_ch}"
            )
        if self.min_sample_rate_hz and fs < self.min_sample_rate_hz:
            raise ValueError(
                f"extractor '{self.id}' needs fs >= {self.min_sample_rate_hz} "
                f"Hz, got {fs}"
            )

    @staticmethod
    def sanitize(features: Dict[str, float]) -> Dict[str, float]:
        """Replace NaN/Inf with 0.0 so downstream models never see them."""
        clean = {}
        for k, v in features.items():
            fv = float(v)
            clean[k] = fv if math.isfinite(fv) else 0.0
        return clean

    def run(self, window: np.ndarray, fs: float, params: Optional[dict] = None) -> Dict[str, float]:
        """Full guarded path: validate → check → extract → sanitize.
        Prefer this over calling `extract()` directly."""
        p = self.validate_params(params)
        self.check_signal(window, fs)
        return self.sanitize(self.extract(window, fs, p))

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'display_name': self.display_name,
            'description': self.description,
            'param_schema': [p.to_dict() for p in self.param_schema],
            'required_channels': self.required_channels,
            'min_sample_rate_hz': self.min_sample_rate_hz,
        }


# ── Registry ──────────────────────────────────────────────────────────────


class ExtractorRegistry:
    """Holds one instance per extractor id. Populated at import time by the
    built-in extractors (see `extractors/__init__.py`); Partner-SDK
    extractors register through the same `register()` at load."""

    def __init__(self) -> None:
        self._items: Dict[str, FeatureExtractor] = {}

    def register(self, extractor: FeatureExtractor) -> None:
        if not extractor.id:
            raise ValueError("extractor must define a non-empty id")
        if extractor.id in self._items:
            raise ValueError(f"extractor id '{extractor.id}' already registered")
        self._items[extractor.id] = extractor

    def get(self, extractor_id: str) -> Optional[FeatureExtractor]:
        return self._items.get(extractor_id)

    def has(self, extractor_id: str) -> bool:
        return extractor_id in self._items

    def all(self) -> List[FeatureExtractor]:
        return list(self._items.values())
