# Plan — MQTT Robustness (P1) + Feature-Extraction Hybrid (P2)

**Recorded:** 2026-07-11
**Trigger:** Workshop pain points reported after a session with ~65 attendees.
**Owner:** Claude Opus 4.7 (this session)

---

## Background — the two workshop failures

### Problem 1 — Sensor board publishes `{"X": …, "Y": …, "Z": …}` but Signal Recorder logs all zeros

The board emits raw uppercase-keyed payloads:

```
Published to Broker: {"X": -9.73, "Y": 0.9, "Z": -1.61}
Published to Broker: {"X": -9.73, "Y": 0.9, "Z": -1.57}
Container is active. Waiting for sensor stream...
```

The Signal Recorder records `X=0, Y=0, Z=0` because `parseSensorPayload()` in `frontend/src/views/PublishedAppView.vue` does exact-case key lookup and the App Builder MQTT node is typically configured with lowercase channel names (`x, y, z`). Uppercase `X` doesn't match lowercase `x`, values resolve to `undefined`, then `?? 0` fallback → all zeros written to the CSV.

**Root cause:** parser is fragile. Any future sensor with a slightly different payload shape (nested, prefixed keys, semantic names, etc.) will silently fail the same way.

**Constraint:** once a workshop is running, we can't ssh in and fix the parser. It has to be robust and self-healing before it ships.

### Problem 2 — Server meltdown + MQTT broker stops receiving publishes under 65 concurrent users

When ~65 attendees all click "Extract Features" at approximately the same time, the Flask backend saturates. `tsfresh` is CPU-heavy (700+ features per window using NumPy/SciPy/statsmodels). With a small worker pool everything else on the process — including the backend's paho-mqtt publisher connections used by Prediction Sink and Log Watcher — starves. From the operator's view, "the MQTT broker stopped receiving publishes."

**Root causes:**
1. Feature extraction is a synchronous HTTP call. No queue, no backpressure.
2. Flask workers block on tsfresh for many seconds. Other requests time out.
3. Backend's MQTT client heartbeat drops during the block. Client disconnects silently.

---

## Full plan — 6 commits across ~18-20 hours

Ship order + dependencies:

```
Problem 1 — MQTT robust parsing (~3 hr, 1 commit)
    ├─ Layer 1 · runtime auto-adapt
    ├─ Layer 2 · App Builder detect-channels helper
    └─ Layer 3 · raw payload preview toggle

Problem 2 — Feature-extraction hybrid
    Phase 1 · Backend queue with 5-slot cap  (~4 hr, 1 commit)
    Phase 2 · Web Worker + JS feature library  (~7 hr, 1 commit)
    Phase 3 · App Builder MQTT client-side features  (~4 hr, 1 commit)
```

Each commit leaves the app in a fully working state — no half-baked phases between sessions.

---

## Problem 1 — Layer-by-layer detail

### Layer 1 · Runtime auto-adaptation in `parseSensorPayload()`

**File:** `frontend/src/views/PublishedAppView.vue`

Rewrite the parser to defensively handle any JSON shape:

1. **Recursive flatten to numeric leaves.** Walk the JSON tree, collect every leaf that's a number. Use dotted key-paths (`X`, `data.x`, `values[0]`).
2. **Case-insensitive channel matching.** When the app has channels configured (e.g. `x, y, z`) and the payload delivers `X, Y, Z`, match via `.toLowerCase()`.
3. **Auto-fallback.** If configured channels match ZERO payload keys, use the first N discovered numeric leaves as values and record them under the configured channel names.
4. **Warning banner.** When auto-fallback triggers, show a non-blocking banner on the recorder UI: *"Configured channels `x, y, z` didn't match payload keys `X, Y, Z`. Auto-matched by position. Update your App Builder MQTT node to future-proof."*

**Support matrix after this change:**

| Payload | Behavior |
|---|---|
| `{"X": -9.73, "Y": 0.9, "Z": -1.61}` | ✅ Case-insensitive match to configured `x, y, z` |
| `{"x": -9.73, "y": 0.9, "z": -1.61}` | ✅ Existing match |
| `{"accX": …, "accY": …, "accZ": …}` | ✅ Auto-fallback to first 3 numeric keys |
| `{"data": {"x": 1.2, "y": 3.4}}` | ✅ Flattened to `data.x`, `data.y`; auto-fallback matches |
| `{"timestamp": 12345, "temp": 45.3, "vib": 0.87}` | ✅ Only numeric leaves picked; `timestamp` treated as a channel too (operator can exclude via config) |
| `{"values": [1.2, 3.4, 5.6]}` | ✅ Existing behavior preserved |
| `[1.2, 3.4, 5.6]` (bare array) | ✅ Fallback to indexed `[0]`, `[1]`, `[2]` |
| Anything not covered above | ✅ Warning banner + operator can diagnose via Layer 3 |

**Tests to pass before commit:**
- Case-insensitive: `{"X":1, "Y":2, "Z":3}` records `x=1, y=2, z=3` when channels configured as `x, y, z`.
- Nested: `{"a":{"b":5}}` records `a.b=5` in auto-fallback mode.
- Auto-fallback warning banner appears exactly once per session, dismissible.
- Existing configured lowercase payloads work unchanged (regression).

### Layer 2 · "Detect channels from sample MQTT message" helper

**Files:**
- `backend/app/routes/app_builder.py` — new endpoint `POST /api/app-builder/detect-channels`
- `frontend/src/views/AppBuilderEditorView.vue` — MQTT node config panel

**Backend endpoint:**
- Body: `{sample: <any JSON>}`.
- Uses the same recursive-flatten logic from Layer 1 (extracted into a shared util). Returns `{channels: ["X", "Y", "Z", "timestamp"]}`.
- Cap sample size at 4 KB. Auth: `@login_required`.

**Frontend UX (mirrors Log Watcher's Auto-detect columns button):**
- On the `input.live_stream` node config panel, add **"Detect channels from sample MQTT message"** button.
- Opens a small dialog with a textarea. Paste one JSON payload → click Detect → the `channels` field auto-populates.
- Operator can trim the list (e.g. remove `timestamp`) before saving.

**Tests to pass before commit:**
- Click Detect with `{"X":1, "Y":2, "Z":3}` → channels become `X, Y, Z`.
- Click Detect with `{"data":{"x":1.2, "y":3.4}, "device":"abc"}` → channels become `data.x, data.y` (device excluded — not numeric).
- Empty / malformed sample → friendly error.
- Cancel button restores previous channel values.

### Layer 3 · "Show raw MQTT message" toggle in Signal Recorder

**File:** `frontend/src/views/PublishedAppView.vue`

Add a small toggle in the Signal Recorder panel while MQTT is connected: **"Show raw MQTT"**.

- When on, display the last 3 messages pretty-printed as JSON in a monospace pre block below the live preview panel.
- Ring buffer keeps only the last 3 to avoid unbounded memory.
- Persist toggle state in `localStorage` keyed by slug (matches the Wall-monitor display prefs pattern).

**Tests to pass before commit:**
- Toggle on → next 3 published messages appear as JSON.
- Newest message at top; ring buffer trims to 3.
- Toggle off → panel hides.

**Milestone at end of Problem 1:** any workshop attendee with an unknown sensor board can plug in → see raw payload via Layer 3 → click Layer 2's Detect button → save. Zero developer intervention.

---

## Problem 2 — Phase-by-phase detail

### Phase 1 · Backend feature-extraction queue with 5-slot cap

Prevents server meltdown. Ships as its own commit; app fully working with only this.

**Backend files:**
- New service: `backend/app/services/feature_job_queue.py`
- Route: `backend/app/routes/features.py` — convert `/api/features/extract` from sync to async

**New service:**
- `ThreadPoolExecutor(max_workers=5)` — hard cap on concurrent tsfresh jobs.
- In-memory job registry: `Dict[job_id -> {status, submitted_at, started_at?, completed_at?, result?, error?}]`.
- TTL 30 minutes after completion; janitor thread evicts stale entries every 5 min.
- Job statuses: `queued → running → done | error | cancelled`.
- Provides:
  - `submit(payload, user_id) -> job_id`
  - `get_status(job_id) -> {status, queue_position?, features?, error?}`
  - `cancel(job_id) -> bool`

**Route changes:**
- `POST /api/features/extract` — accepts the same body as before but returns `{job_id, status: 'queued', queue_position, estimated_wait_seconds}` immediately (status 202).
- New `GET /api/features/extract/<job_id>` — returns current status. When `done`, includes the features payload.
- New `DELETE /api/features/extract/<job_id>` — cancels if not yet running; noop with 200 if already done.

**Frontend changes:**
- `FeaturesView.vue` — polls `GET /api/features/extract/<job_id>` every 2 s.
- UI states:
  - `queued`: **"Position N in queue · ~X seconds wait"** with a spinner.
  - `running`: **"Extracting features..."** with an indeterminate progress.
  - `done`: features appear as before.
  - `error`: existing error handling.
- **Cancel button** available during queued/running.

**Metrics for backpressure UX:**
- Estimated wait: `queue_position × avg_completion_time` (default avg 60 s until we have data; refine as jobs complete).

**Tests to pass before commit:**
- Submit 20 jobs simultaneously; verify only 5 run at a time, others queue.
- Cancel a queued job → status becomes `cancelled`, another queued job promotes to `running`.
- Cancel a running job → best-effort cancel, verify subsequent jobs run.
- Server stays responsive on other endpoints (`GET /api/health` returns < 100 ms) throughout.
- Job status persists across polls; TTL evicts after 30 min.

**Milestone:** server survives arbitrary concurrent load. Queue is visible + honest with users.

### Phase 2 · Web Worker + JavaScript feature library ("Fast Mode")

Introduces the client-side path. Requires Phase 1 to have shipped (Fast Mode is offered from the queue-full dialog).

**New frontend files:**
- `frontend/src/workers/feature-worker.ts` — Web Worker that receives windows and posts back feature vectors.
- `frontend/src/lib/js-features.ts` — pure-JS feature library.

**JS feature library — targeted feature set (~30-35 features per channel):**

**Statistical:**
- mean, std, min, max, range, RMS, median, IQR, skewness, kurtosis (10)

**Spectral (via FFT):**
- dominant frequency, spectral centroid, spectral entropy, peak amplitude, low/mid/high band energy (7)

**Temporal:**
- zero crossings, mean crossings, autocorrelation@lag1, autocorrelation@lag5, sample entropy (5)

**Change-based:**
- first-diff mean, first-diff std, second-diff mean, second-diff std (4)

**Amplitude:**
- 25th percentile, 75th percentile, absolute energy, mean absolute deviation (4)

Total: **~30 per channel**. FFT via a tiny 200-line radix-2 implementation (no external dep).

**Backend changes:**
- New endpoint: `POST /api/features/from-vector` — accepts pre-computed features + labels, stores in the pipeline session identically to how `/api/features/extract` would have.
- Body: `{windowing_session_id, feature_names, feature_matrix, labels?}`
- Pipeline session gains `feature_mode: 'server_full' | 'client_fast'`.

**Frontend changes:**
- Extend Phase 1's queue-full dialog:
  ```
  Server is busy (5/5 running, 12 waiting).
  Wait ~10 min for full-quality features (700+ features via tsfresh)
  OR
  Use Fast Mode — compute 30 features in your browser now.
  Note: your model will use these 30 features from now on.
  
  [ Wait in queue ]  [ Use Fast Mode ]
  ```
- Fast Mode path: iterate windows → post to Web Worker → collect feature vectors → post to `/api/features/from-vector`.
- Progress bar based on window count.
- Small permanent banner on the Features stage after Fast Mode is used: **"This pipeline used Fast Mode (30 features). Deployed model will require Fast Mode too."**
- Pipeline session mode persists to Training + Deploy stages.

**Tests to pass before commit:**
- Trigger Fast Mode manually (skip the queue dialog) — verify Web Worker processes windows and posts feature vectors.
- Feature values match server-side output on a known window (spot-check mean/std to within a small epsilon).
- Model trained on `client_fast` features gives reasonable accuracy on a known dataset (~within 5% of `server_full` for factory-sensor tasks).
- Pipeline session persists `feature_mode` and it flows to Training stage.
- Fallback: if Web Worker throws (should be very rare), user sees a friendly error, session isn't corrupted.

**Milestone:** users choose Fast Mode → get instant results → server unaffected.

### Phase 3 · App Builder MQTT runtime supports client-side features

Without this, Fast Mode works for training but MQTT inference in published apps would still call server-side tsfresh with the wrong feature set. This makes Fast Mode a first-class citizen through the entire pipeline.

**Backend changes:**
- App definition storage gets `feature_mode` (`server_full` | `client_fast`), inherited from the training pipeline.
- `POST /api/app-builder/run/<slug>` accepts either:
  - Raw window data (existing) — server computes features server-side using the app's `feature_mode`.
  - Pre-computed feature vector — server skips feature extraction, runs the model directly.
- Dispatches based on payload shape (`data` array vs `feature_vector` array).

**Frontend changes:**
- `PublishedAppView.vue` reads the app's `feature_mode` on load.
- Live Stream card shows: **"Feature mode: Fast (client-side)"** or **"Feature mode: Full (server-side)"**.
- If `client_fast`:
  - Window collection → Web Worker → feature vector → POST to `/api/app-builder/run/<slug>` with `{feature_vector: [...]}`.
  - Backend runs only the model prediction.
- If `server_full`: existing path (send raw window).

**Tests to pass before commit:**
- End-to-end: train in Fast Mode → publish app → connect MQTT → inference matches offline prediction.
- Backend load: MQTT inference load is minimal (no tsfresh calls) even with many concurrent published apps.
- Latency: client-side feature extraction + model prediction round-trip is faster than server-side full path.

**Milestone:** Fast Mode is production-quality; App Builder respects it throughout.

---

## Test data — reuse what we have

- **Factory sensor CSV:** `datasets/shared/log-watcher-test/train_factory.csv` — 75 rows × 3 channels + labels. Perfect for training + validating both feature modes.
- **Log samples for Layer 2/3 diagnostics:** `datasets/shared/log-watcher-test/sample_apache.log`.
- **Test payloads for Layer 1:** synthetic, defined per-test-case above.

---

## Commit checklist (each commit)

- All tests listed above pass.
- Docker rebuild + container health check.
- Backend + frontend logs clean (no exceptions).
- Session .md file updated with the commit hash + what shipped.

---

## Running notes

*(Updated at each phase completion.)*

- [ ] **P1 all layers** — coder + QA dispatched · commit hash · verified: ___
- [ ] **P2 Phase 1** — coder + QA dispatched · commit hash · verified: ___
- [ ] **P2 Phase 2** — coder + QA dispatched · commit hash · verified: ___
- [ ] **P2 Phase 3** — coder + QA dispatched · commit hash · verified: ___

---

## Follow-up ideas (out of scope for this plan)

- Redis-backed job registry (for multi-worker Flask + horizontal scale).
- Feature-set comparison dashboard so operators can see trained-model accuracy across modes.
- Export the JS feature library as a standalone package so factories can build offline inference tools.
