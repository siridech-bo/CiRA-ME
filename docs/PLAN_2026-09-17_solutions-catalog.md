# Solutions Catalog — Vertical PdM Templates over a Pluggable Extractor Registry

**Status:** 🔧 Active — start with MCSA template
**Date:** 2026-09-17
**Owner:** siridech.bo@kmitl.ac.th
**Related tracker item:** to be added under OPEN ITEMS as `F8`
**Research backing:** [RESEARCH-2026-09-17_solutions-catalog-reuse.md](RESEARCH-2026-09-17_solutions-catalog-reuse.md) — verified reuse targets, licenses, datasets
**Related plans:** [PLAN_2026-08-12_partner-sdk.md](PLAN_2026-08-12_partner-sdk.md) (F6 — the catalog is the surface partner solutions plug into)

---

## Why this exists

Customer wants a **catalog of vertical solution templates** — pick a use case
("Motor Current", "Pump Analysis", "Machine Vibration"), get a pre-wired
end-to-end pipeline (data-source profile → windowing → physics-aware features →
model → deploy-in-a-specific-format), and never have to know the DSP details.

Today a user assembles every pipeline step by hand and must already know the
non-obvious settings (10 s windows for MCSA, sideband feature families, bearing
fault frequencies). A "Motor Current Diagnosis" preset fills all of that in.
The first vertical (MCSA) was previously scoped as a from-scratch build; the
[research pass](RESEARCH-2026-09-17_solutions-catalog-reuse.md) found the hard
part (`statorscope`, Apache-2.0) already exists, turning it into integration.

## Locked decisions (from the 2026-09-17 discussion)

- **Naming: "Solutions", NOT "Applications".** "App Builder" already exists and
  builds *dashboards*; a second "Applications" collides in both UI and code.
  These are different layers: App Builder = presentation on top of a model;
  Solutions = a pre-built ML *pipeline recipe* for a use case.
- **Nav placement: a new top-level group, above GLOBAL TOOLS.** It's the front
  door — where a user starts a new project. Selecting a Solution launches the
  existing Windowing→Features→Training→Deploy pipeline pre-filled, NOT a new
  bespoke screen.
- **NOT the Multi-Dataset Wizard.** That is a dataset merge/combine tool, not a
  use-case gallery. Putting Solutions there conflates "combine my CSVs" with
  "start a motor project."
- **Templates are DATA, not code.** Each Solution is a declarative spec in a
  registry (same pattern as `constants/machine_profiles.py`). Adding a vertical
  = one spec + one extractor, NOT a new page. The pipeline UI renders from spec.
- **Physics-aware feature extractors are a separate pluggable registry.** A
  Solution spec references an extractor by `id`. This is the real reuse win.
- **Vendor-and-freeze policy for reused extractors.** The best per-vertical
  extractors are small single-author repos. Fork them into our repo, freeze,
  own the copy, test against our contract — NEVER track them as live pip deps.
- **No external pretrained weights.** Ship reused extractors feeding our
  existing PyOD / XGBoost / TimesNet models. Weights trained on other
  motors/bearings transfer poorly to customer hardware.
- **Anomaly-first.** Each Solution defaults to the anomaly / per-motor-baseline
  path (healthy data only — the realistic factory case). Classification is the
  opt-in "advanced, once you have labeled faults" path.
- **Design the spec schema so a Partner-SDK (F6) solution is just another
  entry** in the same catalog from day one.

---

## Architecture

### Two registries, one seam

```
SOLUTIONS registry  (constants/solutions.py)      ← declarative specs, "data"
        │  references extractor by id
        ▼
EXTRACTOR registry  (services/extractors/…)       ← pluggable physics-aware code
        │  conforms to one contract
        ▼
Existing pipeline: Windowing → Features → Training (PyOD/XGB/TimesNet) → Deploy
```

The **seam** between a Solution and its extractor is the nameplate/param object
(e.g. poles, line-frequency, rated-slip for MCSA; bearing geometry for
vibration). The Solution spec collects it once; the extractor consumes it.

### Extractor contract (the one interface everything conforms to)

Every extractor — first-party, vendored, or Partner-SDK — implements:

```python
class FeatureExtractor(Protocol):
    id: str                       # 'mcsa', 'bearing_envelope', 'pump_cavitation'
    display_name: str
    param_schema: list[ParamDef]  # nameplate fields the template form renders
    required_channels: int        # e.g. 3 for 3-phase current
    min_sample_rate_hz: float

    def extract(
        self,
        window: np.ndarray,       # (n_samples, n_channels)
        fs: float,                # sampling rate Hz
        params: dict,             # validated against param_schema
    ) -> dict[str, float]:        # named physically-meaningful features
        ...
```

Feature **names** must be stable and self-describing (`mcsa_brb_lsb_db`,
`bpfo_env_peak_db`) so tree-model feature-importance stays explainable — the
core of the pitch. The output dict flows into the existing feature step exactly
like the current DSP/tsfresh extractors.

### Solution spec shape

```python
Solution(
    id='motor_current_mcsa',
    display_name='Motor Current (MCSA)',
    icon='mdi-flash',
    data_profile={'channels': 3, 'sample_rate_hz': 5000, 'unit': 'a'},
    windowing={'window_s': 10},                       # the non-obvious one
    feature_extractor='mcsa',                          # ← registry id
    param_schema=['poles', 'line_freq_hz', 'rated_slip'],
    models={'default': 'iforest', 'advanced': 'xgboost'},
    deploy={'targets': ['linux_api'], 'format': 'onnx'},
    source='builtin',                                  # or 'partner:<id>' (F6)
)
```

### Shared preprocessing: order-tracking / speed-normalization

VFD-driven current (Utwente pump) and post-VFD / 2-phase current (Paderborn)
break classic fixed-supply MCSA sideband math. Design **one** speed-normalization
/ order-tracking stage that extractors can opt into via a spec flag. Build it
once during the MCSA phase; vibration and pump reuse it.

---

## Phased rollout

### Phase 0 — Registries + one seam (foundation)

- `services/extractors/` package + the `FeatureExtractor` contract above.
- `constants/solutions.py` registry + `to_dict()` for a `/solutions` endpoint
  (mirror `machine_profiles.py`).
- Wire the extractor registry into [FeaturesView.vue](../frontend/src/views/pipeline/FeaturesView.vue)
  as a **third extraction mode** next to Fast (DSP) / TSFresh.
- Persist `param_schema` values on the project/data-source object so the feature
  step can read them back.
- **Effort:** ~1 wk. **Ship gate:** an empty registry renders in Features with
  no regression to existing DSP/TSFresh modes.

### Phase 1 — MCSA template (first vertical, highest value)

- **Vendor `statorscope`** (Apache-2.0) behind the contract as the `mcsa`
  extractor. Fork + freeze + own the copy. Record provenance/license.
- Nameplate form: poles, line-frequency, rated-slip; per-window slip estimation
  from the vendored code.
- Rule-based threshold layer (sideband dB thresholds ~−35 to −40 dB) as an
  explainability add-on alongside the anomaly score.
- Solution spec `motor_current_mcsa`; default IForest, advanced XGBoost.
- **Validate on Paderborn** (real accelerated-life faults) — do NOT trust
  statorscope's synthetic-only accuracy figures.
- Deploy: Linux/API (ONNX). MCU deferred.
- **Effort:** ~2 wks. **Ship gate:** healthy→fault detection demonstrated on
  Paderborn, feature importances render, one-click from catalog to deploy.

### Phase 2 — Machine Vibration template (bearings)

- Vendor `tzarcrept/bearing-fault-detection` (MIT, geometry-aware
  BPFO/BPFI/BSF/FTF + Hilbert envelope) and the `danielnewman09/Kurtogram`
  snippet (BSD-2, demod-band selection) behind the contract; borrow VibPy
  functions as needed.
- Nameplate form: bearing geometry (balls, ball dia, pitch dia, contact angle).
- **License-check and add** the canonical datasets not verified in research
  (CWRU / MFPT / IMS-NASA / XJTU-SY); MAFAULDA is UFRJ citation-notice, not OSI.
- **Effort:** ~2 wks.

### Phase 3 — Pump Analysis template (only genuine build)

- No drop-in repo exists. Assemble the **Han et al. 2024 recipe**: wavelet
  denoise (PyWavelets) + VMD (vmdpy) + Park vector modulus (trivial) + Hilbert
  marginal spectrum (scipy). Reuse the Phase-1 order-tracking stage (post-VFD).
- Datasets: Utwente/4TU (CC BY, all-fault) + ESPset (MIT, **spectra only**).
- **Effort:** ~3 wks (highest — lowest priority).

### Phase 4 — Partner-SDK seam (after F6 lands)

- Accept `source='partner:<id>'` Solutions and partner-provided extractors that
  conform to the same contract → first-party and OEM solutions share one catalog.
- **Effort:** folds into F6; schema already partner-ready from Phase 0.

---

## Add TSFEL alongside tsfresh (cross-cutting, any phase)

`TSFEL` (BSD-3, Fraunhofer, actively maintained) is the one reliably-maintained
code asset found. Its config-driven `get_features_by_domain()` maps cleanly onto
the extractor registry and is edge-friendly for the TI/MCU path. Add as a
general extractor option independent of the three verticals.

---

## Reuse-vs-build summary

| Template | Reuse (vendor-and-freeze) | Build | Priority |
|---|---|---|---|
| Motor Current (MCSA) | `statorscope` (Apache-2.0) | registry adapter, rule layer, Paderborn validation | 1st |
| Machine Vibration | `tzarcrept` (MIT) + `Kurtogram` (BSD-2) + VibPy (MIT) | registry adapter, dataset license-check | 2nd |
| Pump Analysis | Utwente + ESPset datasets; Han recipe from PyWavelets/vmdpy/scipy | the pump extractor itself | 3rd |
| Cross-cutting | TSFEL (BSD-3) | registry option | any |

## What to trust (reliability ranking)

- **Most reliable code:** TSFEL (institutional, maintained, permissive). The
  three per-vertical extractors are reliable only as *frozen, in-repo code we
  own and test* — not as dependencies.
- **Most reliable dataset:** Paderborn (primary-verified, real faults,
  current+vibration). Validate MCSA and vibration here first.
- **Reliability comes from us owning the frozen copy + validating on Paderborn**,
  not from upstream repos staying alive.

## Open questions to resolve before/inside each phase

1. Vendor-and-freeze vs. reimplement each single-author extractor against the
   contract? (Default: vendor-and-freeze, reimplement only if the contract fit
   is poor.)
2. Does the VFD/post-VFD/2-phase current need the order-tracking stage before
   MCSA extractors work? (Assume yes; build it in Phase 1.)
3. CWRU / MFPT / IMS-NASA / XJTU-SY license checks (Phase 2).
4. Confirmed: **no external pretrained weights** — ship extractors + our models.

---

## Non-goals (v1)

- MCU/edge deploy for the templates (Linux/API first; MCU is a later add).
- 2D-CNN spectrogram / ResNet path (research tier, not a template default).
- Tier 2/3 low-code customization of Solutions (specs are curated, not
  user-editable in v1).
