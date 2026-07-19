# Session 2026-07-19 — Phase D + Phase F shipped

Two atomic commits on `master`, both pushed to `origin/master`:

- `4a95770` — Phase D: MQTT ingest router + workshop defenses
- `dd92102` — Phase F: Machine Simulator + sidebar reorg

Overall Asset Tree refactor progress: **94%** (6 of 7 phases done). Only Phase E (OculusT labels integration) remains, and it's explicitly deferred as not urgent.

## Phase D — MQTT ingest router + workshop defenses

Delivered a background service that owns a paho-mqtt client subscribed to `#`, routes messages by topic path to daily CSVs under `datasets/<topic_path>/`, and enforces strict/learn-mode topic rules against the asset tree.

### Architecture
- Three daemon threads:
  - **connect**: paho client with exponential-backoff reconnect (now correctly resets on successful connect — was staying high forever after any initial storm)
  - **writer**: buffered flush every 200 ms or 100 messages; UTC-midnight rotation resets `_headers_written` per file
  - **janitor**: sweeps every 6 h against `ingest_retention_days`; admin-only `POST /ingest-janitor/run-now` triggers synchronous sweep
- In-memory tree cache with fire-and-forget `reload_tree()` hook wired to all 8 tree-mutation routes
- Meta-prefix short-circuit uses exact segment match (not `startswith` — `_metastasized` correctly rejected)

### Config surface (all admin-gated)
- `PATCH /api/asset-tree/config` — `ingest_enabled` · `ingest_retention_days` · `meta_prefixes` · `topic_mode`
- `PUT /api/asset-tree/config` — `level_names` · `root_name` (schema-affecting)
- `GET /api/asset-tree/ingest-stats` — writer state, thread aliveness, reconnect counters, cache size (open to all)
- `GET /api/asset-tree/rejected-topics?date=YYYY-MM-DD&limit=N` — path traversal blocked, malformed dates return empty

### Routing (strict mode)
- Root check first, then meta-prefix exact match
- `{"value": N}`, `{"v": N}`, `{"val": N}`, or bare number → CSV row
- `{"value": null}` → **rejected as `unparseable payload`** (was writing literal string 'None' to CSV — QA polish #1)
- Everything else → rejection log (in-memory tail + on-disk TSV)

### Frontend
- Settings → MQTT Rules view with Config / Rejected / Stats tabs, 3 s auto-poll
- `MqttTestPublisher` clamps browser publish rate to 5 msg/s (workshop-safety hint text; server-side publishers unaffected)

### QA
0 blockers, 5 polish. **3 fixed**:
- `{"value": null}` writing string 'None' to CSV
- Reconnect backoff never resetting after successful connect
- `import re` in hot path

**2 skipped** (cosmetic — TSV escape for tab/newline; PATCH `?date` echo).

## Phase F — Machine Simulator + sidebar reorg

Delivered server-side signal generators so demos and workshops can show the full ingestion pipeline without real hardware. Six built-in profiles publish to `cirame-mosquitto` at each sensor's sample rate, flow through the Phase D ingest router, and land in daily CSVs — structurally identical to how a real IoT device would connect.

### Profile library
- **Air Compressor** — pressure / temperature / vibration / current
- **Industrial Boiler** — flame / feedwater_temp / steam_pressure / flue_gas_temp / o2_percent
- **Centrifugal Pump** — inlet_p / outlet_p / flow_rate / motor_current / vibration
- **Conveyor** — speed_rpm / motor_current / belt_tension / vibration / product_count
- **CNC Spindle** — spindle_rpm / spindle_load / temperature / vibration_x/y/z
- **Chiller / HVAC** — refrigerant_p / evap_temp / cond_temp / compressor_current

Each profile has multiple states (idle / running / fault / maintenance / off / chaos / …) with per-sensor `(mean, std, sinusoid_amplitude, period_s)` tuples. `sample()` returns `None` for silent sensors (off / maintenance) — no fake "None" strings emitted.

### Chaos state
Every ~30 s, rotate through 4 poison flavours to exercise Phase D reject paths:
- `{"value": null}` (parse error)
- Garbage bytes (parse error)
- `wrong_root/<name>/<sensor>` (root mismatch)
- `<root>/plant_XX_nonexistent/<name>/<sensor>` (unknown plant, strict-mode reject)

Sensor name is drawn randomly from `profile.sensors` — 5 of 6 profiles don't have a sensor called "temperature" so the original hard-coded `/temperature` variant meant only Compressor + CNC actually exercised the reject paths (QA polish #3 fix).

### Backend
- `MachineSimulator` singleton mirroring `MqttIngestRouter` structure
- One shared paho client (`cira-simulator-<pid>`), one daemon thread per `_SimulatedMachine`
- Per-sensor `_next_due` cadence; sensors publish at their own rate
- Ring buffer (deque maxlen=60) per sensor for UI sparklines
- All threads swallow exceptions + log — never crashes the backend
- **In-memory only** — restart wipes; matches `_publishers` / `_recording_jobs` pattern

### REST endpoints (`/api/simulators/*`)
| Method | Path | Auth |
|---|---|---|
| GET | `/profiles` | any |
| GET | `/` | any |
| GET | `/snapshot` | any |
| POST | `/` | admin |
| PATCH | `/<id>` | admin |
| DELETE | `/<id>` | admin |
| POST | `/publish-raw` | admin |

Every mutation writes an `AssetTreeAudit` event.

### Autoprovision safety (QA polish #1 + #2 fixes)
`POST /simulators/` with `autoprovision_tree: true` walks the `topic_base` segments and creates any missing asset nodes. **Now validates topic_base is at exactly the machine level** (one above sensor leaf). Previously:
- Too-short path silently mounted sensors at the plant/machine level, polluting the taxonomy
- Too-deep path silently created no sensor children, leaving the sim publishing forever to unroutable paths

Both now return a clear 400 with the expected segment count.

### Frontend
- **`/global/simulators`** — `SimulatorsView.vue` (grid of cards, global msg/s ticker, add / stop-all, raw-publish widget)
- `SimulatorCard.vue` — profile icon + name + topic_base (monospace) + state dropdown + start/stop + msg/s + uptime + per-sensor sparklines (Chart.js) + delete
- `SimulatorNewDialog.vue` — profile picker → name + topic_base + initial_state + autoprovision checkbox → preview

### Sidebar reorganization
- **Dropped** the redundant "ASSET TREE" section header (tree itself starts with the factory icon)
- **Renamed** Settings menu `Asset Tree` → dynamic **`<Root> Setup`**, reading `assetTreeStore.config.root_name` (Factory / Stores / Site / Hospital, fallback "Structure Setup"). Same source drives the admin page's H1.
- **Collapsed** Dashboard + Projects into a new `SidebarLegacyGroup.vue` at the bottom, muted grey `[legacy]` chip, `cira.sidebar.legacyExpanded` localStorage persist (default false)
- **Added** "Machine Simulators" entry under GLOBAL TOOLS
- Section dividers for visual rhythm

### QA
0 blockers, 7 polish. **5 fixed**:
- Autoprovision level validation (topic_base too-short → sensors at wrong level; topic_base too-deep → unroutable forever)
- Chaos state rotates through `profile.sensors` instead of hardcoded `/temperature`
- Refuse simulator create when `root_name` is unset
- UNIT_PRESETS extended with `lpm` / `kn` / `count` / `boolean` (Pump / Conveyor sensors were falling back to `percent`)
- Raw-publish rejects empty payload up-front

**2 skipped** — `connected` field lag under reconnect race (unverifiable, benign under gunicorn `--workers 1`); 10 Hz tick drift under load (only affects 100 Hz which isn't a use case).

## Hand-off failures worth calling out

Two things I got wrong in the browser-test hand-off, both caught by the user:

1. **Told user port 5175 for frontend** — it's 3030. Ports are documented right there in `docker-compose.yml`; I guessed instead of checking.
2. **Told user to hard-refresh browser without rebuilding the frontend container** — frontend is baked into an image (no HMR), same as backend. `docker compose up -d --build frontend` is required after Vue/TS changes.

Both saved to memory as feedback + reference entries:
- `feedback_verify_before_asserting.md` — hard rule about verifying ports / endpoints / DB state before asserting them
- `reference_cirame_ports.md` — canonical port list (frontend 3030, backend 5100)
- MEMORY.md updated with "Verify before asserting" section + frontend-rebuild-required note

## Files touched — Phase D (10 files, +2039/−17)

New:
- `backend/app/services/mqtt_ingest_router.py`
- `docs/MQTT-TOPIC-NAMESPACING.md`
- `frontend/src/views/MqttRulesView.vue`

Modified:
- `backend/app/__init__.py` — router boot behind `WERKZEUG_RUN_MAIN` gate
- `backend/app/models.py` — `AssetTreeConfig.patch(...)`, `AssetNode.get_by_topic_path(...)`
- `backend/app/routes/asset_tree.py` — MQTT Rules endpoints
- `frontend/src/App.vue`
- `frontend/src/components/MqttTestPublisher.vue` — rate clamp
- `frontend/src/router/index.ts` — `/settings/mqtt-rules` route
- `docs/asset-tree-traces.html`

## Files touched — Phase F (14 files, +3053/−28)

New:
- `backend/app/constants/machine_profiles.py`
- `backend/app/routes/simulators.py`
- `backend/app/services/machine_simulator.py`
- `docs/PLAN_2026-07-19_machine-simulator.md`
- `frontend/src/components/SidebarLegacyGroup.vue`
- `frontend/src/components/SimulatorCard.vue`
- `frontend/src/components/SimulatorNewDialog.vue`
- `frontend/src/views/SimulatorsView.vue`

Modified:
- `backend/app/__init__.py` — sim boot + blueprint register
- `backend/app/constants/sensor_presets.py` — added `lpm` / `kn` / `count` / `boolean` units
- `frontend/src/App.vue` — sidebar reorg
- `frontend/src/router/index.ts` — `/global/simulators` route
- `frontend/src/views/AssetTreeAdminView.vue` — dynamic H1
- `docs/asset-tree-traces.html`

## What's next

- **Phase E (OculusT labels integration)** — deferred, not urgent
- **`.103` production deploy** — 2 commits stacked on origin/master; runbook in `docs/DEPLOY-me-cira-core-com.md`
- **Multi-machine data pooling** — still TODO in `ml_trainer.py` from Phase C plan
- **MOMENT foundation model integration** — parked plan doc, restart when Phase E lands
