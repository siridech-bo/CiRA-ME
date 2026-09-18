# MCSA Extractor — Real-Data Validation (UNESP + ITSC)

**Date:** 2026-09-18
**Scope:** F8 Solutions Catalog, Phase 1 — the `mcsa` feature extractor (vendored statorscope)
**Status:** ✅ Extractor validated on real hardware data. Phase 1 core proven, with documented boundaries.
**Environment:** dev box (`.167`) only — nothing deployed to `.103`.
**Related:** [PLAN_2026-09-17_solutions-catalog.md](PLAN_2026-09-17_solutions-catalog.md) · [RESEARCH-2026-09-17_solutions-catalog-reuse.md](RESEARCH-2026-09-17_solutions-catalog-reuse.md)

---

## Verdict

The `mcsa` extractor produces **textbook-correct, explainable MCSA features on real induction-motor current**, and CiRA ME's own models (XGBoost / IsolationForest) detect developed rotor-bar faults with strong accuracy:

- **Healthy vs faulty: 92.2%** (balanced 90.5%), load-grouped CV
- **Healthy vs severe (3–4 broken bars): 96.7%** (balanced 95.9%)

The limits we found match MCSA physics exactly and **confirm** the plan's anomaly-first, per-motor-baseline design rather than contradict it. Two boundaries are important and stated plainly below: (1) cross-*motor* transfer is **untested** here, and (2) the global anomaly model is weak and needs per-load baselining.

---

## What was tested

| Dataset | Role | Real? | Rate / regime |
|---|---|---|---|
| statorscope `synthesize` | positive control | synthetic | 5 kHz, 10 s, fault injected at −42 dBc |
| **ITSC** (ibarram/ITSC) | specificity / robustness | real | 60 Hz, 1 kHz, 5 s, no-load, stator faults + healthy |
| **UNESP** "banco de dados experimental" | **positive BRB detection** | **real** | 60 Hz, **50 kHz**, 20 s, 0/1/2/3/4 broken bars × 8 loads × 10 reps |

UNESP motor: 1 hp, 4-pole (→ `pole_pairs=2`), 60 Hz, 34 rotor bars, rated 1715 rpm. Five files `struct_{rs,r1b,r2b,r3b,r4b}_R1.mat` (~1.5 GB each), MATLAB **v7.3 (HDF5)**.

---

## Method note (a real gotcha)

**The backend image runs `PYTHONOPTIMIZE=1`, which strips `assert` statements.** Early validation "PASS" lines were gated on asserts that never executed — they validated nothing. **All validation here uses explicit `if`-checks**, never `assert`. Anyone writing in-container checks must do the same.

UNESP `.mat` files are **v7.3/HDF5**, which `scipy.io.loadmat` cannot read — `h5py` is required (not in the image; `pip install h5py` at runtime). Structure: top-level key = rotor tag → `torque05…torque40` load groups → `Ia/Ib/Ic` each a `(10,1)` object-reference array → 10 reps of `(1, 1001000)` @ 50 kHz.

---

## Results

### 1. Synthetic positive control (sanity)

Fault injected at −42 dBc, slip 0.03: extractor detected it, sidebands measured at −42.0 dBc, **slip recovered 0.0300 (exact)**, healthy not detected, **37.5 dB** separation. Extractor logic correct.

### 2. ITSC — specificity / robustness (real, but low-fs / no-load)

65 real recordings ran with **0 errors** (robust on real current). Findings:

- All 12 stator-fault classes → 0 detections. This is **correct-by-design**: statorscope estimates slip *from broken-bar sidebands* and **refuses when it can't** — at no-load there are none.
- Healthy: broken-bar "detected" on 2/5, **1/5 even survived the clock audit** (`mcsa_supported=1`) → a false positive.
- **Cause:** ITSC's **1 kHz / no-load** regime is below MCSA's reliable operating point. **Conclusion:** downstream rules/models must require `mcsa_supported`, and sub-few-kHz data is out-of-regime.

### 3. UNESP — the critical discovery: window on steady-state

First attempt sliced the **first 10 s** of each record → **0% detection on everything**. Root cause was **not** the extractor: the first ~5 s is the **motor startup transient**, which smears the 60 Hz carrier across ±8 Hz, raising the local noise floor and burying the sidebands. statorscope's clock audit correctly **refused** it.

| slice | carrier smear | 4-bar detected | strongest sideband |
|---|---|---|---|
| 0–10 s (startup) | 8.0 Hz | ❌ | −47 dBc, ~0 dB prom |
| **5–15 s (steady)** | 0.4 Hz | ✅ | **−28.9 dBc, +56 dB prom** |
| 10–20 s (steady) | 0.4 Hz | ✅ | −29.0 dBc, +55 dB prom |

→ **Windowing must land on steady-state signal.** This is a real product requirement, not a tuning detail. (calibrate on/off was irrelevant — it was purely the transient.)

### 4. UNESP — continuous feature separation (steady-state)

`mcsa_brb_strongest_db` (dBc), 24 windows/condition across 8 loads:

| condition | mean | std | gap vs healthy |
|---|---|---|---|
| healthy | −49.6 | 23.4 | — |
| 1-bar | −49.1 | 22.0 | +0.5 dB |
| 2-bar | −42.6 | 16.5 | +7.0 dB |
| 3-bar | −35.7 | 3.6 | +13.9 dB |
| 4-bar | −33.7 | 3.8 | +16.0 dB |

Developed faults (3–4 bar) separate cleanly and tightly; a **single broken bar is ≈ healthy (+0.5 dB)** — an **inherent MCSA physics limit**, not an implementation flaw. The large variance on healthy/1-bar is **load-dependent** (low slip → sidebands collapse into the carrier → floored to −120).

### 5. UNESP — model-level results (the ship-gate number)

400 windows (16 features each), CiRA ME's own stack, **load-grouped 4-fold CV** (model must generalize to unseen load levels):

**XGBoost (classification)**

| Task | Accuracy | Balanced |
|---|---|---|
| Healthy vs faulty | **92.2%** | 90.5% |
| Healthy vs severe (3–4 bar) | **96.7%** | 95.9% |
| Severity (0–4 bars) | 71.5% | 71.5% |

Per-severity recall: 1-bar 88%, 2-bar 89%, 3-bar 100%, 4-bar 98%; healthy specificity 88%. Severity confusion matrix is near-diagonal (errors almost all adjacent, 3↔4).

**Top features** (XGBoost importance — the explainability payoff): `mcsa_brb_usb_db` (0.24), `mcsa_ecc_strongest_db` (0.22), `mcsa_stator_strongest_db` (0.18), then slip features. Physically sensible and reportable ("flagged because the upper broken-bar sideband is elevated").

**IsolationForest (anomaly, trained on healthy only):** ROC-AUC **0.74** — weak. Undermined by load-driven healthy variance and the **−120 dBc "absent" encoding** creating artificial extreme points in the anomaly manifold.

---

## Caveats (must be read alongside the numbers)

1. **Cross-motor transfer is NOT tested.** UNESP has **one physical rotor per fault level**, so train and test always share the same physical rotor (different loads). This validates *"detect faults on this motor across operating conditions"* — the per-motor-baseline case — but not generalization to a **different** motor, which the research flagged as MCSA/ML's known weak point. The strong 88% 1-bar recall is likely **same-rotor memorization**; do **not** read it as "incipient single-bar detection solved."
2. **The global anomaly model is weak (AUC 0.74)** — evidence for **per-load / per-motor baselines**, not one global IsolationForest.
3. **Single broken bar ≈ healthy** on absolute sideband level — inherent MCSA physics; catching incipient faults requires per-motor baseline drift tracking, not an absolute threshold.

---

## Actionable findings (feed back into the plan)

1. **Steady-state windowing is mandatory** — the windowing/ingestion step must skip transients (startup, load steps). Ties directly to the plan's order-tracking/steady-state preprocessing stage.
2. **Downstream must gate on `mcsa_supported`** (statorscope's clock audit) — it cleanly separated "untrustworthy window" from "real fault" in both the transient and the low-fs cases.
3. **Reconsider the `−120` absent-value encoding** in [mcsa.py](../backend/app/services/extractors/mcsa.py) for the anomaly path — the extreme floor distorts IsolationForest. Consider a per-feature "missing" indicator or a less extreme floor.
4. **Anomaly path should be per-load / per-motor baselined**, not a single global model.
5. **Do not use statorscope's binary verdict** as the classifier — it over-fires on real healthy motors (92%). Use the continuous 16-feature vector in our models (already the design).

---

## Reproduction

Harness (dev box, `D:\tmp\mcsaval\`), run against the backend image with datasets mounted:

```bash
# feature extraction (writes unesp_features.csv; ~11 min for 400 windows)
docker run --rm -w /app -e PYTHONUNBUFFERED=1 \
  -v "D:/dataset/banco de dados experimental:/ds:ro" \
  -v "D:/tmp/mcsaval:/data" cirame-backend \
  sh -c 'pip install -q h5py; python /data/feature_extract_unesp.py'

# train + report (IForest + XGBoost, load-grouped CV)
docker run --rm -w /app -v "D:/tmp/mcsaval:/data" cirame-backend \
  python /data/train_unesp.py
```

Scripts: `feature_extract_unesp.py`, `train_unesp.py`, `report_levels.py`, `validate_unesp.py`, `validate_mcsa.py`, `diag.py`, `exp.py`. Datasets: UNESP at `D:\dataset\banco de dados experimental\`; ITSC via `github.com/ibarram/ITSC`.
