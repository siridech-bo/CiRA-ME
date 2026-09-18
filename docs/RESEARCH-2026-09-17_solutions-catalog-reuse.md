# Research — Reusable Methods, Repos & Datasets for the Solutions Catalog

**Date:** 2026-09-17
**Author:** Deep-research pass (101 agents, 19 sources fetched, 25 claims adversarially verified — 25/25 confirmed, 0 refuted)
**Context:** Bootstrapping three industrial predictive-maintenance "solution templates" (Motor Current / MCSA, Pump Analysis, Machine Vibration) on the existing CiRA ME Python stack, feeding a **pluggable feature-extractor registry**. See the companion architecture discussion for the Solutions-catalog UI/registry design.

---

## Research question

Find existing open-source methods, models, libraries, reference repositories, and public labeled datasets we can reuse to bootstrap three end-to-end templates (raw signal → windowing → physics-aware feature extraction → ML/DL model → deploy) **without building from scratch**, that integrate into our stack:

- Python backend: scikit-learn, PyOD (IForest, OCSVM, LOF, ECOD, COPOD, HBOS), XGBoost, LightGBM, tsfresh, TimesNet (PyTorch), ONNX export, SciPy FFT.
- Deploy targets: Linux/HTTP API (primary), TI MCU/edge (secondary) → lightweight, exportable extractors matter.
- Pluggable feature-extractor registry: each template references a physics-aware extractor by id. **Biggest reuse win is feature-extraction code, not model code.**
- License: prefer permissive (MIT/BSD/Apache) for commercial integration; flag GPL/AGPL.

---

## Executive summary

Across all three templates, the biggest reuse wins are in **physics-aware feature-extraction code** and **public labeled datasets** — all under permissive licenses that drop into the existing numpy/scipy/scikit-learn/PyOD stack.

- **Motor Current (MCSA):** `statorscope` (Apache-2.0, numpy+scipy only) already implements the exact sideband families needed — broken-rotor-bar `(1±2ks)·f`, stator-fault, and eccentricity — plus sensorless per-window slip estimation. Datasets: UFES synthetic broken-rotor-bar (Treml/Carletti), IEEE 1 HP stator-fault set, Paderborn (synchronous current + vibration).
- **Machine Vibration (bearings) — most mature:** a BSD Fast-Kurtogram Python port (Antoni #48912), MIT-licensed VibPy bearing toolbox, and an MIT geometry-aware BPFO/BPFI/BSF/FTF + envelope-analysis reference. Canonical datasets: CWRU / Paderborn / MFPT / MAFAULDA / XJTU-SY.
- **Pump Analysis:** cavitation/impeller work covered by the CC-BY Utwente 4TU multi-fault pump dataset, MIT ESPset spectra, and a concrete published methods stack (wavelet-threshold denoise + VMD + Park vector modulus + Hilbert marginal spectrum).
- **Cross-cutting:** TSFEL (BSD-3) adds a config-driven feature pipeline that maps cleanly onto a pluggable extractor registry alongside tsfresh. External **pretrained model zoos are not recommended** — ship reused extractors + existing PyOD/XGBoost/TimesNet models.

**Maturity warning:** the three best per-vertical extractors (`statorscope`, `Kurtogram-Analysis`, `tzarcrept/bearing-fault-detection`) are small single-author repos (2–7 commits, one dormant since 2019). Treat them as **vendored-and-frozen reference code forked into our registry**, not as tracked pip dependencies.

---

## 1. Motor Current (MCSA)

### Feature-extraction code

**[statorscope](https://github.com/ali-kin4/statorscope)** — **the single best drop-in MCSA extractor.**

- **License:** Apache-2.0 (permissive, commercially reusable).
- **Deps:** Python 3.11+, `numpy>=1.26` / `scipy>=1.11` only.
- **Implements (verbatim in README):**
  - Broken-rotor-bar sidebands at `(1 ± 2ks)·f`
  - Stator inter-turn fault signatures at `f·[n(1−s)/p ± k]`
  - Eccentricity signatures at `f ± k·f_r`
  - **Sensorless per-window slip estimation** from current-spectrum sideband geometry (mean slip error 0.0002 self-reported)
- **Verdict:** vote 3-0 on capability/license; 2-1 on the slip-accuracy metric.
- **Caveat:** the 0.0002 slip figure is against a **synthetic** simulator (BBIM2023, 8 test cases, single-author self-report) — treat as illustrative, not validated hardware accuracy. This is the module previously flagged as "net-new"; it already exists — the work becomes integration, not invention.

### Datasets

| Dataset | Source | Specs | Notes |
|---|---|---|---|
| **IEEE 1 HP 3-phase stator-fault** | [ieee-dataport](https://ieee-dataport.org/documents/three-phase-induction-motor-stator-fault-data) | 10 kHz, Ia/Ib/Ic | **Stator inter-turn only** (not rotor-bar/eccentricity); 10 kHz adequate for 60 Hz sidebands |
| **UFES synthetic broken-rotor-bar** (Carletti/Encarnação, Treml) | [ieee-dataport](https://ieee-dataport.org/documents/synthetic-dataset-induction-motor-broken-rotor-bar-analysis) | 8 kHz .mat, 0–4 bars × 50/75/100% load = 390 scenarios | **Synthetic/simulated**, calibrated with 26 physical motors |
| **Paderborn** | [uni-paderborn](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter) | Synchronous current + vibration @ 64 kHz, 32 bearings (6 healthy, 12 artificial, 14 real accelerated-life) | Records 2 of 3 phases (3rd = −(i1+i2), recoverable); ~0.25 Hz resolution; enables MCSA+vibration fusion and *realistic* validation |

**Verdict:** unanimous (3-0) on all three dataset spec claims.

---

## 2. Machine Vibration — rolling-element bearings (most mature area)

### Feature-extraction code

| Repo | License | What it provides | Maturity |
|---|---|---|---|
| **[tzarcrept/bearing-fault-detection](https://github.com/tzarcrept/bearing-fault-detection)** | MIT | Geometry-aware **BPFO/BPFI/BSF/FTF** re-derived from bearing geometry (8 balls, d=7.145 mm, D=28.519 mm → FTF 0.375, BSF 1.871, BPFO 2.998, BPFI 5.002 orders) + 1–10 kHz band-pass + Hilbert envelope extracting peaks at FTF/BSF/2×BSF/BPFO/BPFI vs noise floor. **111 fixed per-window features**, exportable into a registry. | Small (~7 commits, MAFAULDA-only) — reference code, not a library |
| **[VibPy](https://github.com/andrek10/bearing-vibration-diagnostics-toolbox)** | MIT | Purpose-built rolling-element bearing diagnostics toolbox (envelope / high-frequency-resonance demod). Univ. of Agder PhD research. | Modest (~41 stars, PhD-era) |
| **[danielnewman09/Kurtogram-Analysis](https://github.com/danielnewman09/Kurtogram-Analysis)** | BSD-2-Clause | Python port of **Jerome Antoni's Fast Kurtogram** (#48912): spectral kurtosis across binary-ternary wavelet-packet levels, optimal-band location (max-kurtosis level+freq), envelope demod at that band — the standard demod-band-selection method. | Small (~19 stars, ~2 commits, last active 2019) — vendor the snippet |

**Verdicts:** VibPy 3-0; Kurtogram 3-0; tzarcrept 3-0 on features/license, 2-1 on geometry-derivation (independent physics check confirms orders are geometrically correct, d/D=0.2505).

### Datasets

Canonical set referenced: **CWRU, Paderborn, MFPT, MAFAULDA, XJTU-SY**. In this pass only **Paderborn** and **MAFAULDA** were verified with primary sources.

- **MAFAULDA** — © UFRJ (citation/copyright notice, cite Ribeiro et al.); **not a formal OSI license** — data-use license distinct from the MIT code that uses it.
- **CWRU / MFPT / IMS-NASA / XJTU-SY** — named but **not individually verified here**; each needs a separate license check before shipping (see Open Questions).
- Aggregator reference: **[awesome-bearing-dataset](https://github.com/VictorBauler/awesome-bearing-dataset)** (secondary source, useful index).

---

## 3. Pump Analysis (least mature — dataset + method, not a drop-in repo)

### Datasets

| Dataset | License | Specs | Notes |
|---|---|---|---|
| **Utwente / 4TU "NLN-EMP"** (Bruinsma/Geertsma/Loendersloot/Tinga 2024) | CC BY | Simultaneous **vibration + 3-phase current + voltage** on induction-motor-driven centrifugal pumps; **11 fault types** incl. cavitation, impeller damage, bearing defects, broken rotor bar, stator winding short, unbalance; 3-phase current @ 20 kHz | Single dataset relevant to **all three** templates. DOI: data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0. **Caveat:** current is post-VFD → classic fixed-supply MCSA sidebands need adaptation |
| **[ESPset](https://github.com/NINFA-UFES/ESPset)** (NINFA-UFES) | MIT | 6,032 labeled accelerometer signals from Electrical Submersible Centrifugal Pumps (offshore oil) | **Public portion = frequency-domain spectra normalized by rotating frequency, NOT raw waveforms** → raw-signal windowing unavailable. Mirror: Mendeley m268jsw339 |

Sources: [Utwente pub](https://research.utwente.nl/en/publications/motor-current-and-vibration-monitoring-dataset-for-various-faults/) · [Data in Brief DOI](https://doi.org/10.1016/j.dib.2023.109741) · [ESPset KBS 2024 DOI](https://doi.org/10.1016/j.knosys.2024.111452)

### Methods stack (reusable recipe, not a repo)

**Han et al., Sensors 2024, 24(11):3410 (CC BY)** — [DOI](https://doi.org/10.3390/s24113410) — two-modality fusion:

1. Improved **wavelet-threshold denoising** (→ PyWavelets)
2. **Variational Mode Decomposition (VMD)** with energy-entropy on vibration (→ vmdpy)
3. **Park vector modulus** transform on 3-phase current (trivial)
4. **Hilbert marginal-spectrum** analysis (→ PyEMD / scipy Hilbert)
5. t-SNE fusion → 98.55% reported accuracy

**Verdict:** 3-0. Supplies **methods, not a library** — reusability rests on existing Python impls listed above. No single drop-in pump repo exists; this vertical is the only genuine build (and the lowest-priority one).

---

## Cross-cutting

### Feature libraries beyond tsfresh

- **[TSFEL](https://github.com/fraunhoferportugal/tsfel)** — **BSD-3-Clause, actively maintained.** `get_features_by_domain()` returns a JSON-serializable statistical/temporal/spectral config (~390 features) that **maps cleanly onto a pluggable extractor registry** and persists config for reproducibility. Explicitly designed for computationally-restricted/edge targets → fits the TI-MCU deploy path. Verdict 3-0.
- Feature-count context (for registry sizing): catch22 = 22, TSFEL = 390, tsfresh = up to 1,558, hctsa = 7,730.
- **[enDAQ](https://endaq.com/pages/endaq-open-source-python-library-for-shock-vibration-analysis)** (blog source) — PSD/SRS/FFT + integration to velocity/displacement; candidate general vibration DSP library.

### Pretrained model zoos / benchmarks

- **[ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark)** (MIT) — canonical reproducible benchmark (TIE 2020).
- **[ZhaoZhibin/UDTL](https://github.com/ZhaoZhibin/UDTL)** (MIT) — unsupervised deep transfer learning for diagnosis.
- **Verdict/recommendation:** useful as *benchmarks*, but **do not adopt external pretrained weights** — ship reused extractors + existing PyOD/XGBoost/TimesNet models. Weights trained on other motors/bearings transfer poorly to customer hardware.

---

## Recommended reuse-vs-build per template

| Template | Reuse | Build | Priority |
|---|---|---|---|
| **Motor Current (MCSA)** | Vendor `statorscope` as the `mcsa` extractor | Registry adapter + slip validation on real data | 1st (was scariest, now integration) |
| **Machine Vibration** | Vendor `tzarcrept` + `Kurtogram` snippet; borrow VibPy functions | Registry adapter; add CWRU/MFPT license-checked datasets | 2nd |
| **Pump Analysis** | Utwente + ESPset datasets; assemble Han recipe from PyWavelets/vmdpy/scipy | The pump extractor itself (only genuine build) | 3rd (lowest) |

**Shared preprocessing to design once:** an **order-tracking / speed-normalization** stage for VFD-driven current (Utwente pump) and post-VFD / 2-phase current (Paderborn), so fixed-supply MCSA sideband extractors work on variable-speed supplies.

---

## Caveats (verbatim from the verified report)

- Maturity varies sharply: `statorscope`, `Kurtogram-Analysis`, `tzarcrept/bearing-fault-detection` are small single-author repos (some ~2–7 commits, one unmaintained since 2019) — best treated as **vendored reference snippets, not maintained dependencies**.
- Several headline accuracy numbers are self-reported against **synthetic** data (statorscope slip 0.0002 = 8 simulator cases; UFES broken-rotor-bar set = simulated, not physical).
- **Dataset licenses are distinct from code licenses** (e.g. MAFAULDA © UFRJ citation notice, not OSI; ESPset public portion is spectra only, no raw waveforms).
- Utwente pump and Paderborn currents are measured **after a VFD / on only 2 of 3 phases** respectively → classic fixed-supply MCSA sideband analysis needs adaptation.
- WebSearch was unavailable during several verifications, so cross-checks lean on primary source pages; independent code-level exercise of statorscope's three signature implementations was not performed.
- Two canonical vibration datasets named (CWRU, MFPT, IMS/NASA, XJTU-SY) are referenced but **not individually verified** — only Paderborn and MAFAULDA were confirmed with primary sources.

---

## Open questions

1. Which small single-author extractors (`statorscope`, Kurtogram port, `tzarcrept`) should be **vendored-and-frozen vs. reimplemented in-house** against the registry contract, given low maintenance activity and commercial-support risk?
2. Does the **VFD-driven / post-VFD / 2-phase** current in Utwente and Paderborn require an **order-tracking / speed-normalization** preprocessing stage before fixed-supply MCSA extractors work?
3. For the bearing template, do the canonical **CWRU / MFPT / IMS-NASA / XJTU-SY** datasets need to be added and license-checked (only Paderborn and MAFAULDA verified here)?
4. Is a permissive **pretrained model zoo / benchmark** (UDTL / DL-Benchmark style) worth adopting, or should templates ship feature-extractors + existing PyOD/XGBoost/TimesNet models rather than external weights? *(Research recommendation: the latter.)*

---

## Sources

**Primary**

- statorscope — https://github.com/ali-kin4/statorscope
- IEEE 1 HP stator-fault dataset — https://ieee-dataport.org/documents/three-phase-induction-motor-stator-fault-data
- UFES synthetic broken-rotor-bar dataset — https://ieee-dataport.org/documents/synthetic-dataset-induction-motor-broken-rotor-bar-analysis
- Paderborn bearing datacenter — https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter
- Kurtogram-Analysis — https://github.com/danielnewman09/Kurtogram-Analysis
- VibPy (bearing-vibration-diagnostics-toolbox) — https://github.com/andrek10/bearing-vibration-diagnostics-toolbox
- tzarcrept/bearing-fault-detection — https://github.com/tzarcrept/bearing-fault-detection
- Utwente/4TU pump dataset — https://research.utwente.nl/en/publications/motor-current-and-vibration-monitoring-dataset-for-various-faults/ · https://doi.org/10.1016/j.dib.2023.109741
- ESPset — https://github.com/NINFA-UFES/ESPset · https://doi.org/10.1016/j.knosys.2024.111452 · https://data.mendeley.com/datasets/m268jsw339/1
- Han et al. pump cavitation methods — https://doi.org/10.3390/s24113410
- TSFEL — https://github.com/fraunhoferportugal/tsfel
- DL-based-Intelligent-Diagnosis-Benchmark — https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark
- UDTL — https://github.com/ZhaoZhibin/UDTL
- MCSA FFT methods (arXiv) — https://arxiv.org/html/2401.15417v1
- Time-series feature-lib comparison (arXiv) — https://arxiv.org/abs/2110.10914

**Secondary / index / blog**

- awesome-bearing-dataset — https://github.com/VictorBauler/awesome-bearing-dataset
- Predictive-Maintenance-Motor-Fault-Classification — https://github.com/himansh1257/Predictive-Maintenance-Motor-Fault-Classification
- enDAQ vibration library — https://endaq.com/pages/endaq-open-source-python-library-for-shock-vibration-analysis
- 18 libraries for time-series feature extraction — https://medium.com/data-science-collective/18-libraries-for-time-series-feature-extraction-f12fd1bae738

---

*Stats: 5 angles · 19 sources fetched · 91 claims extracted · 25 verified · 25 confirmed · 0 refuted · 9 findings after synthesis · 101 agent calls.*
