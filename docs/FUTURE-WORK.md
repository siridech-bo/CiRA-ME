# Future Work

Larger initiatives that aren't customer-driven yet but are known
long-term directions. Not blocked, not queued — parked here so we don't
lose the strategic context between sessions.

Reassess at every planning cycle. Move an item into
[CUSTOMER-TRACKER.md](./CUSTOMER-TRACKER.md) as an active feature when a
customer asks for it or when leadership prioritises it.

## Items

### FW1. TCN (Temporal Convolutional Network) for edge deployment
**Type:** new algorithm | **Size:** Large

**Why we'd do it:** smallest MCU-friendly DL architecture we know of.
Target: 5-15 KB INT8 quantized. Fits alongside `REGR_*` / `CLS_*` in
the TI NN model zoo but with a simpler compute graph.

**Where it fits:** new algorithm in `ml_trainer.py` alongside the
existing sklearn/XGBoost lineup, with export path through the same TI
container the current TI NN models use.

**Open questions:**
- Which TCN variant — dilated (WaveNet-style), causal, or bidirectional?
- Reuse tinyml-modelmaker's training loop or bring our own?
- Same hyperparameter surface as TI NN, or expose kernel/dilation?

### FW2. Web Serial API for MCU flashing
**Type:** DX improvement | **Size:** Large

**Why we'd do it:** eliminate the "install CCS to flash your model"
step for the customer. Browser talks to the C2000 board directly over
USB via the Web Serial API. No IDE download, no toolchain install.

**Where it fits:** new page in the frontend, calls navigator.serial.*,
streams the compiled `.hex` from `/api/ti/export-saved/<id>` (or a new
`/api/ti/flash-payload/<id>` endpoint) to the board.

**Blockers / risks:**
- Web Serial is Chromium-only (no Firefox, no Safari)
- Needs HTTPS on the frontend (currently HTTP)
- Bootloader protocol per TMS320 device varies; we'd need one implementation per target family
- User needs to hold BOOT/reset in a specific sequence — probably an
  onboarding walkthrough with screenshots

### FW3. ONNX Runtime Web (WASM) for browser-side inference
**Type:** new deploy target | **Size:** Medium

**Why we'd do it:** run inference in the browser tab that opens a
published App Builder app. No backend round-trip, no MQTT, works
offline. Existing infra already exports ONNX for TI NN and CLAW models;
the runtime is the only thing missing.

**Where it fits:** App Builder Model Endpoint node gets a new "runtime"
config option: `server` (current default) or `browser`. Browser runtime
loads `onnxruntime-web` via CDN or bundle, downloads the ONNX from
`/api/ti/export-saved/<id>` or `/api/deployment/cira-claw-package/<id>`,
runs inference in a WASM sandbox.

**Blockers / risks:**
- Model download size on cellular
- Feature extraction still needs to happen — either port the DSP
  extractor to JS/WASM (hard) or precompute features server-side
  then run only the model in browser (defeats the point)
- Not all models exportable to ONNX cleanly (custom Python models,
  some sklearn variants)

---

## When to promote something out of here

An item moves from **Future Work** to **Active** (CUSTOMER-TRACKER.md
OPEN ITEMS) when at least one of:

- A customer asks for it explicitly
- Leadership sets a release date it needs to land by
- We hit an engineering forcing-function (e.g. we start needing it as a
  building block for something already committed)

When you promote something, write a design doc alongside the tracker
entry — the notes here are just seeds.
