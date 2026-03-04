# MCU Deployment Plan (Group 2) — Revisit When Hardware Ready

## Target Devices
- TI F28P55X (C2000 DSP)
- STM32 (Cortex-M with CMSIS-DSP)
- Any bare-metal MCU with C compiler

## Core Constraint
No Linux, no Python, no Docker. C/C++ only. Models must be compiled into firmware.

## What's Feasible
| Model Type | Feasible? | Notes |
|---|---|---|
| Random Forest / Decision Tree | ✅ | Convert to C if/else chains via m2cgen |
| SVM / Logistic Regression | ✅ | Matrix multiply in pure C |
| Threshold anomaly detection | ✅ | Trivial |
| Large ensembles (>100 trees) | ⚠️ | May exceed Flash |
| TimesNet / any DL | ❌ | Too large, no PyTorch |

## Deployment Package Contents
```
deployment_package/
├── model.c              ← decision tree weights as C arrays (via m2cgen/micromlgen)
├── model.h
├── features.c           ← DSP feature extraction in pure C (FFT via TI DSPLib or CMSIS-DSP)
├── features.h
├── inference.c          ← main inference loop
├── inference.h
└── ccs_project/         ← Code Composer Studio project (for F28P55X)
    ├── .ccsproject
    └── targetConfigs/
```

## Memory Budget (F28P55X example)
| Component | RAM | Flash |
|---|---|---|
| Sensor window buffer (128×6ch×float32) | ~3 KB | — |
| Feature vector (50 features×float32) | ~200 B | — |
| Random Forest (50 trees, depth 5) | ~5 KB | ~30–80 KB |
| Feature extraction code | ~2 KB | ~10–20 KB |
| **Total** | **~10 KB** | **~50–100 KB** |
Well within 192KB RAM / 1.5MB Flash.

## Required Libraries
- **m2cgen** (`pip install m2cgen`) — converts sklearn → pure C with zero deps
- **micromlgen** — targets Arduino/MCU, Random Forest → C
- **emlearn** — ML for embedded, good RF → C support
- **TI DSPLib** — hardware FFT on F28P55X (already available in CCS)
- **CMSIS-DSP** — ARM DSP library for STM32

## Feature Extraction in C
Features that need re-implementation in C:
- **Trivial**: mean, std, min, max, median, variance, RMS, peak-to-peak, zero_crossing_rate
- **Moderate**: skewness, kurtosis, autocorrelation, binned_entropy
- **Requires FFT**: spectral_centroid, spectral_bandwidth, spectral_rolloff, peak_frequency, spectral_entropy
  → Use TI DSPLib `DSP_fft32x32()` on F28P55X or CMSIS-DSP `arm_rfft_fast_f32()` on STM32

## Implementation Steps (when ready)
1. Add MCU as a target group in DeployView.vue (separate from Docker group)
2. Backend: add `generate_c_code(saved_model, pipeline_config)` in deployer.py
   - Call m2cgen to convert sklearn model → C
   - Generate features.c with all 44 DSP features as pure C functions
   - Generate inference.c with windowing + normalize + feature_extract + predict
   - Generate CCS project files for F28P55X target
3. Add `/api/deployment/mcu-package/<model_id>` endpoint
4. Frontend: "Download CCS Project" button (no SSH — JTAG flash required)
5. Note: Flash deployment itself is done via JTAG (XDS110 debugger) in Code Composer Studio — not automated via CiRA ME

## Deployment Flow (MCU)
```
CiRA ME (Download .zip)
    ↓
Unzip → Open in Code Composer Studio
    ↓
Build → Flash via JTAG (XDS110 for F28P55X)
    ↓
Real-time inference on MCU
```
No SSH involved — JTAG is the programming interface.

## Notes
- Only ML models supported (TimesNet requires PyTorch)
- tsfresh features must be replaced with lightweight equivalents in C
- F28P55X has hardware FFT via CLA (Control Law Accelerator) — spectral features are fast
- Consider fixed-point arithmetic for performance (F28P55X has FPU but 32-bit float is slower than Q15)
