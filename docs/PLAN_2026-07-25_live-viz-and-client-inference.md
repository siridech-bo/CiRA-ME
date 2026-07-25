# Phase I — Live signal viz + record controls + client-side inference

**Date:** 2026-07-25
**Effort:** ~3.5 days (~1.5d Q2/Q3, ~2d Q4)
**Depends on:** Phase D (ingest router), Phase F (simulator), Phase H (multi-axis)

---

## Motivation — 2026-07-25 customer visit

Four issues collected on-site. Q1 shipped already (Machine Simulators moved to SERVICES group). Q2 dissolves into Q3 (see below). Q3 and Q4 are the substantive work.

### Q2 → Q3 (same root cause)

Customer reported "only admin can see signal values on Machine Simulators; regular users cannot." Investigation:

- Backend `/api/simulators/snapshot` returns full data to any authed user (verified with fresh annotator account)
- Frontend `SimulatorCard.vue` has NO admin gate on the sparkline or value display
- BUT `MachineWorkspaceView.vue` has ZERO live MQTT viz — no sparklines, no live signal, nothing

Customer clicked their real machine (Machine_4 with adxl345 sensor) in the tree → landed on the machine workspace → saw no live signal there → assumed non-admin can't see values.

The real fix isn't a permission tweak — it's **putting live viz where users naturally navigate**: on the machine workspace, not on the admin-oriented `/global/simulators` page.

### Q3 as stated

Real MQTT sensors (Machine_4 / adxl345) reach CSV storage but have no live graph like the simulator page. Users also want per-sensor **recording toggles** and **sampling throttle** to manage DB size. Same controls should apply to simulator instances (currently always-record via the router).

### Q4 as stated

App Builder inference has Fast Mode Web Worker feature extraction (P2 Phase 2) but the actual model prediction still round-trips to the backend — ~50-200 ms per prediction, painful for real-time dashboards. Want client-side inference for traditional ML (sklearn / XGBoost), toggle-selectable like Fast Mode.

## Decisions (locked)

### Q3

| Decision | Value |
|---|---|
| Live viz location | New **"Live" tab** on the machine workspace (7 tabs: Overview / Data / Live / History / Labels / Configure / Models) |
| Transport | Browser mqtt.js WebSocket to `ws://<host>:9001/mqtt`, subscribes to `<machine>/+` |
| Sparkline rendering | Same shape as Machine Simulators card — one cell per sensor, multi-axis renders 3 lines (blue/green/amber) |
| Rolling window | 60 s default, configurable per-machine (`live_window_s` in workspace UI state, localStorage-persist) |
| Recording toggle | Per-**sensor** field on `AssetSensorMeta` — `ingest_enabled: BOOLEAN NOT NULL DEFAULT 1`. Router checks this per message. |
| Sample throttle | Per-**sensor** field — `min_write_interval_ms: INTEGER` (nullable, default NULL = write every message). Router tracks last-write timestamp per sensor. |
| Auto-stop | Per-**sensor** field — `record_until: TEXT` (nullable ISO timestamp). Router checks per message; auto-flips `ingest_enabled=0` at expiry. |
| Recording controls UX | Above the sparkline grid on the Live tab. Admin-only mutations (matches existing tree admin gating). |
| Also applies to simulators | Yes — simulator instances flow through the same router, so the same per-sensor controls apply. Nothing sim-specific needed. |
| Storage impact display | Live tab shows today's data size (bytes) + row count per sensor, so users see the impact of their throttle choice |

### Q4

| Decision | Value |
|---|---|
| Approach | **Option A — ONNX runtime in browser** (`onnxruntime-web` ~2 MB gzipped) |
| Deploy step converter | `skl2onnx` / `onnxmltools` in a new `backend/app/services/model_exporter.py` — dispatch by model type |
| Model types supported at v1 | RandomForest · GradientBoosting · LogisticRegression · LinearRegression · SVC (RBF, linear) · KNN · XGBoost |
| Fallback when conversion fails | Toggle disabled with tooltip: "This model type isn't yet supported client-side — falls back to server inference." Model still works, just via server. |
| Toggle location | App Builder canvas top bar, adjacent to Fast Mode toggle |
| Toggle label | "⚡ Client Inference" with icon `mdi-flash` (Fast Mode uses `mdi-lightning-bolt-outline`) |
| Bundle format | Published app carries `<model_id>.onnx` alongside `<model_id>.pkl`; browser prefers .onnx if toggle on |
| Parity check | On deploy, convert model → predict same 100 test rows both ways → compare outputs. If max error > 1e-3 (classification: predictions must match exactly; regression: within 1e-3), reject the ONNX export and log a warning. |

## Backend

### I.1 — Recording controls schema + router integration

**Migration**: add three columns to `asset_sensor_meta`:
- `ingest_enabled BOOLEAN NOT NULL DEFAULT 1`
- `min_write_interval_ms INTEGER` (nullable)
- `record_until TEXT` (nullable ISO timestamp)

**Models**: `AssetSensorMeta.upsert` and `.get` include the new fields. Default when creating: all defaults (record every message, no throttle, no auto-stop).

**Router** (`mqtt_ingest_router.py`):
- Cache carries `ingest_enabled`, `min_write_interval_ms`, `record_until` per sensor
- Router adds a per-topic `_last_write_ts` map (monotonic seconds)
- Per message flow AFTER parse + BEFORE `_append_row`/`_append_multi_row`:
  1. If `ingest_enabled=false` → skip (no CSV write, no rejection log, just drop for storage purposes)
  2. If `record_until` set and now > record_until → flip `ingest_enabled=false` (DB update via async safe queue) + skip
  3. If `min_write_interval_ms` set and (now - last_write_ts) < interval → skip
  4. Else write + update last_write_ts
- Live viz is unaffected — browser subscribes directly to broker, sees every message regardless of recording state

**Endpoints** (`asset_tree.py`):
- New `PATCH /api/asset-tree/nodes/<id>/recording` — admin-only, body `{enabled, min_write_interval_ms, auto_stop_at}`
- Any field absent = leave unchanged (partial-PATCH-friendly, per Phase H polish #2 pattern)
- Auto-stop timestamp accepted as ISO 8601 or as `null` to clear
- Fires `_reload_ingest_router` so change takes immediate effect

**Storage stats endpoint** — new `GET /api/asset-tree/nodes/<id>/data-stats?date=YYYY-MM-DD`:
- Returns `{bytes, row_count, first_ts, last_ts, per_hour_counts}` for a sensor's CSV on that date
- Used by the Live tab's "today's storage" display
- Open to any authed user (read-only)

### I.2 — Client-side inference (ONNX conversion)

**New service**: `backend/app/services/model_exporter.py`
- `export_to_onnx(model, feature_shape, output_path) -> Optional[str]` returns .onnx path or None on failure
- Dispatch by isinstance: `sklearn.RandomForestClassifier` → `skl2onnx.convert_sklearn(...)`, XGBoost → `onnxmltools.convert_xgboost(...)`, etc.
- Parity validation: convert → load ONNX → predict same test batch → compare with sklearn/XGBoost predictions → return None if mismatch beyond tolerance

**Deploy step hook**: existing deploy pipeline calls `export_to_onnx` after saving the .pkl. Writes .onnx sibling file. Deploy metadata records `client_inference_supported: bool`.

**Published app manifest**: gains `client_inference_supported`. Frontend reads this to enable/disable the toggle.

**Dependencies**: add `skl2onnx>=1.16`, `onnxmltools>=1.12`, `onnxruntime>=1.15` (backend for parity check) to `requirements.txt`. Container rebuild required.

## Frontend

### I.3 — Machine workspace Live tab

New tab in `MachineWorkspaceView.vue` between Data and History (tab order: Overview / Data / **Live** / History / Labels / Configure / Models).

Component: `frontend/src/components/MachineLivePanel.vue`

- On mount: subscribe via mqtt.js to `<machine_topic>/+`
- Per-sensor buffer keyed by sensor name (from asset tree children of this machine)
- For each sensor:
  - Header: name + unit + (multi-axis chip if channels) + last value(s)
  - Sparkline: reuse `SimulatorCard`'s sparkline logic (extract to `SensorSparkline.vue` shared component)
- **Recording controls panel** (admin-only) below sparklines:
  - Per-sensor rows: [name] · toggle switch · sample-every input · auto-stop dropdown · today's storage stat
  - Global "Stop recording all" and "Resume all" buttons
- **Rolling window** selector (5s / 30s / 60s / 5min / 30min) — top of panel
- Non-admin sees the controls read-only with a disabled overlay + "Read-only — admins can edit" chip

### I.4 — Shared sparkline component

Extract `SensorSparkline.vue` from `SimulatorCard.vue`'s inline chart logic. Used by:
- `SimulatorCard.vue` (existing)
- `MachineLivePanel.vue` (new)
- Any future viz that needs a compact multi-axis-aware sparkline

Props: `{ name, channels?, values: Record<string, number[]>, isChaos?, height? }`. Same rendering (blue/green/amber, IMU convention, left-padded nulls, spanGaps).

### I.5 — App Builder client-side inference toggle

`AppBuilderEditorView.vue`:
- Top bar toggle button `⚡ Client Inference` next to Fast Mode toggle
- Disabled with tooltip when active flow's Classification/Regression model doesn't support client inference (per deploy metadata `client_inference_supported`)
- Persists per-flow in localStorage key `cira.appbuilder.<flow_id>.clientInference`

`PublishedAppView.vue`:
- If toggle is ON at publish time (stored in the app's config):
  - Load `<model_id>.onnx` via `onnxruntime-web`
  - Inference node's `predict(features)` calls `onnx_session.run(features)` locally
  - Latency dropped from ~50-200 ms to single-digit ms
- If OFF or model doesn't support: existing server round-trip

Bundle size impact: `onnxruntime-web` is lazy-loaded only when at least one published app has client inference enabled.

## Deliverables

| ID | Owner | Description |
|---|---|---|
| I.1 | backend | Recording controls: schema migration + router integration + PATCH endpoint + data-stats endpoint |
| I.2 | backend | Model exporter service + deploy step hook + parity validation |
| I.3 | frontend | MachineWorkspaceView Live tab with sparklines + admin-gated recording controls |
| I.4 | frontend | SensorSparkline shared component (refactor SimulatorCard to use it) |
| I.5 | frontend | App Builder Client Inference toggle + ONNX runtime integration in PublishedAppView |
| I.QA | agent | Adversarial QA — router throttle correctness, ONNX parity, live-tab thread safety, admin gating |
| I.T | user | Personal browser test |

## Edge cases

- Recording toggle race: PATCH ingest_enabled=false while a message is mid-flush → router's per-message check catches it; at most one row leaks
- min_write_interval + auto_stop combined → both checked in order; auto_stop wins if both trip
- Broker connection drops during live viz → mqtt.js auto-reconnects; sparkline shows a gap
- Non-admin visits Live tab → sees viz + read-only controls
- Machine has zero sensors → Live tab shows placeholder "No sensors registered on this machine yet"
- Machine simulator publishing when recording disabled → sim keeps publishing (broker sees traffic), viz updates, but router skips CSV writes (correct — user chose not to store)
- Sensor auto-stop timestamp in the past at PATCH time → auto-flip `ingest_enabled=0` immediately
- ONNX conversion fails for a model → deploy still succeeds, client inference toggle disabled with tooltip explaining why
- Client inference on published app with browser that doesn't support WebAssembly → onnxruntime-web fails to load → fall back to server inference automatically
- Model changes after client-inference toggle enabled → PublishedAppView reloads the .onnx file on model version change
- Very large model (>50 MB) → warn on deploy, don't auto-enable client inference

## Not doing (v1)

- Client inference for deep learning (TCN etc.) — ONNX supports it but skip until asked
- Server-side pre-aggregation of live viz data (e.g. downsampled preview) — browser subscribes directly, no need
- Per-user recording preferences (all controls are machine-wide)
- Recording history / audit UI beyond the existing `AssetTreeAudit` events

## Definition of done

- Non-admin user opens a machine workspace → Live tab shows sparklines updating in real time
- Admin toggles a sensor's recording OFF via Live tab → router immediately stops writing that sensor's CSV
- Admin sets min_write_interval_ms=1000 on a 100 Hz sensor → CSV grows at ~1 row/sec instead of ~100
- Auto-stop expiry works: sensor auto-flips to disabled at the configured ISO timestamp
- App Builder Client Inference toggle: enabled for supported models, disabled for unsupported (tooltip explains)
- Published app with Client Inference ON runs inference in <5 ms per row (vs 50-200 ms server)
- Zero blockers from QA
- Personal browser test passes end-to-end
