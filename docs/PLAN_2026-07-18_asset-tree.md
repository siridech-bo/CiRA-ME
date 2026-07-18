# PLAN — Asset Tree, Machine Groups, and Continuous Ingestion

**Status**: draft, awaiting implementation kickoff
**Session**: 2026-07-18 post-workshop discussion with siridech-bo
**Related plans**: [PLAN_2026-07-XX_moment-foundation-model.md](PLAN_2026-07-XX_moment-foundation-model.md) — MOMENT sits well alongside this (foundation models eat raw windows, which the tree ingests naturally)
**Related deferred**: [DEFERRED-2026-07-17.md](DEFERRED-2026-07-17.md) — Points 1 (continuous ingestion) and 2 (OculusT labeling) from this plan replace the earlier deferred items

---

## 1. Motivation

CiRA ME is currently organized around **abstract Projects**. That works for POC / demo but not for factory customers, who think about their operation as:

> "I have plant_A, plant_B. Each plant has machines. Each machine has sensors. Each machine can have several models trained for it. Data streams in 24/7 from the sensors."

This plan replaces the Project abstraction with a **physical asset hierarchy** that mirrors how customers actually reason about their fleet. Ingestion, labeling, training, and deployment all snap onto real machines instead of floating in Project-space.

**Related customer feedback captured during the 2026-07-18 workshop:**
1. **Continuous ingestion** — Signal Recorder is session-based (start/stop); customers want always-on MQTT-to-disk ingestion that data scientists can mine later.
2. **Labeling gap** — no interactive labeling UI between "raw sensor data on disk" and "windowed classifier training". OculusT (`D:\CiRA OculusT`) already exists as a standalone tool; integration is Phase E.
3. **65-user MQTT hang** — separate concern, address via topic namespacing (naturally enforced by the tree) + publisher rate limiting. Small companion task, not blocking this plan.
4. **Dark/light mode toggle** — small quality-of-life request, folded in as Phase 0.

## 2. Concepts

Two orthogonal concepts, kept clean:

### 2.1 Asset Tree (physical)

A customizable hierarchy that reflects real hardware. Default preset: 4 levels — `factory / plant / machine / sensor`. Level labels are renameable per-install. Depth is 2-6 levels (enforced).

Every node in the tree corresponds to a folder on disk:

```
data/factory/plant_A/machine_1/temperature/2026-07-18.csv
                    /machine_1/vibration/2026-07-18.csv
                    /machine_2/...
        plant_B/...
```

MQTT topic path == on-disk path. Devices publish to `factory/plant_A/machine_1/temperature` and the ingestion service writes to the matching folder.

### 2.2 Machine Groups (logical)

Lightweight admin-created tags: **name + list of machines**. Groups are orthogonal to the tree — they do not appear as tree nodes. Purpose: cross-machine training + fleet-wide model deployment.

Examples:
- `"All extruders"` → machine_1, machine_5, machine_9
- `"Plant A baseline"` → plant_A's first three machines
- `"New commissioning"` → recently added machines

A machine can belong to unlimited groups. Retired machines stay in groups but are flagged.

### 2.3 Confirmed product decisions (from the 2026-07-18 discussion)

| # | Decision |
|---|---|
| 1 | Customizable hierarchy with 4+1 presets (Factory / Hospital / Fleet / Farm / Custom) |
| 2 | Model attachment at **machine level**, many models per machine allowed |
| 3 | **Strict** MQTT topic enforcement by default, with **Learn** mode toggle for R&D + `_meta` topic exceptions |
| 4 | Empty tree on first login → **force wizard**, no skip |
| 5 | Sidebar tree: **auto-collapse past plant + search box** at top |
| 6 | Sensor level: **click-to-filter** the parent machine workspace, no dedicated sensor page in MVP |
| 7 | Groups are logical, not tree nodes; lightweight (name + machine list) |
| 8 | Training button lives in the **machine workspace** with scope toggle (Just this / Group / Ad-hoc) |
| 9 | Retired machines **stay in groups** but flagged; historical training runs remain valid |
| 10 | **Multiple group membership** per machine, no limit |
| 11 | Wizard offers **"Copy sensors from another machine"** for new machines |
| 12 | Wizard offers **preset sensor templates** (vibration monitor / thermal / rotating machinery) |
| 13 | **Compatibility validation** before cross-machine training (sensor names, units, sample rates match) |

---

## 3. Data model

### 3.1 New tables

```sql
-- The hierarchy itself.
CREATE TABLE asset_tree_config (
  id                INTEGER PRIMARY KEY,
  level_names       TEXT NOT NULL,        -- JSON array: ["factory","plant","machine","sensor"]
  root_name         TEXT NOT NULL,        -- e.g. "factory" (the actual root node's name)
  topic_mode        TEXT NOT NULL,        -- 'strict' | 'learn'
  meta_prefixes     TEXT NOT NULL,        -- JSON array: ["_meta","_health","_config","_cmd"]
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL
);
-- Exactly one row. Enforced at the app layer.

-- Every node in the tree — plants, machines, sensors, and anything in between.
CREATE TABLE asset_nodes (
  id                INTEGER PRIMARY KEY,
  parent_id         INTEGER NULL REFERENCES asset_nodes(id) ON DELETE CASCADE,
  level             INTEGER NOT NULL,     -- 0 = root, 1 = plant, 2 = machine, 3 = sensor
  name              TEXT NOT NULL,        -- topic-safe segment (matches [a-zA-Z0-9_-]+)
  display_name      TEXT NULL,            -- optional human label
  description       TEXT NULL,
  location_tag      TEXT NULL,            -- free text: "Building 3, Bay 2"
  topic_path        TEXT NOT NULL UNIQUE, -- materialized "factory/plant_A/machine_1"
  status            TEXT NOT NULL,        -- 'active' | 'retired'
  retired_at        TEXT NULL,
  created_at        TEXT NOT NULL,
  UNIQUE(parent_id, name)
);

-- Sensor-leaf metadata (only leaves carry these).
CREATE TABLE asset_sensor_meta (
  asset_id          INTEGER PRIMARY KEY REFERENCES asset_nodes(id) ON DELETE CASCADE,
  unit              TEXT NULL,            -- "°C", "mm/s²", "Hz", ...
  sample_rate_hz    REAL NULL,
  expected_min      REAL NULL,
  expected_max      REAL NULL,
  data_type         TEXT NULL             -- 'float' | 'int' | 'string'
);

-- Ad-hoc machine groups.
CREATE TABLE machine_groups (
  id                INTEGER PRIMARY KEY,
  name              TEXT NOT NULL UNIQUE,
  description       TEXT NULL,
  created_by        INTEGER NOT NULL REFERENCES users(id),
  created_at        TEXT NOT NULL
);

CREATE TABLE machine_group_members (
  group_id          INTEGER NOT NULL REFERENCES machine_groups(id) ON DELETE CASCADE,
  machine_asset_id  INTEGER NOT NULL REFERENCES asset_nodes(id) ON DELETE CASCADE,
  added_at          TEXT NOT NULL,
  PRIMARY KEY(group_id, machine_asset_id)
);

-- Model ↔ machine binding. Many-to-many.
-- `trained_on_machines` is a JSON array snapshot at training time so we
-- can audit even after machines are retired.
CREATE TABLE model_machine_bindings (
  saved_model_id    INTEGER NOT NULL REFERENCES saved_models(id) ON DELETE CASCADE,
  machine_asset_id  INTEGER NOT NULL REFERENCES asset_nodes(id) ON DELETE CASCADE,
  role              TEXT NOT NULL,        -- 'trained_on' | 'deployed_to'
  trained_via_group TEXT NULL,            -- group name at train time (nullable)
  bound_at          TEXT NOT NULL,
  PRIMARY KEY(saved_model_id, machine_asset_id, role)
);
```

### 3.2 Audit log

```sql
CREATE TABLE asset_tree_audit (
  id                INTEGER PRIMARY KEY,
  actor_user_id     INTEGER NOT NULL REFERENCES users(id),
  event_type        TEXT NOT NULL,   -- 'node_created', 'node_renamed', 'node_retired',
                                     -- 'node_moved', 'group_created', 'group_updated', ...
  target_type       TEXT NOT NULL,   -- 'node' | 'group' | 'config'
  target_id         INTEGER NULL,
  payload           TEXT NOT NULL,   -- JSON snapshot of the change
  created_at        TEXT NOT NULL
);
```

### 3.3 Legacy compatibility

Existing `projects` table is preserved verbatim. On first tree setup, a synthetic legacy node is created:

```
{root}/_legacy/
```

Every existing project appears as a synthetic machine under `_legacy`. Their DataSessions, WindowedSessions, and models remain attached to the legacy project row. Users can either leave them or use a migration wizard (Phase B optional).

## 4. Backend service — MQTT ingest router

New service: `backend/app/services/mqtt_ingest_router.py`.

- Subscribes to `#` on Mosquitto at boot (respecting `asset_tree_config.root_name` as the required top segment).
- For each incoming message:
  1. Split topic by `/`.
  2. Check first segment matches the tree's root; if not, drop and log to `_rejected_topics.log`.
  3. If any segment matches a `meta_prefixes` entry, route to a per-machine `_meta/` folder (not into the training tree).
  4. Look up the topic in the `asset_nodes` table.
     - **Strict mode**: if not found, log to rejection log, do nothing.
     - **Learn mode**: create the missing tree nodes with default metadata; write to the new leaf's folder.
  5. Append the payload (with `_timestamp` and `_source_topic`) to today's rolling CSV in the sensor leaf's folder.
- Files rotate daily (`YYYY-MM-DD.csv`). Rotation cutoff at midnight UTC.
- Files are auto-registered as datasets pointed at by the machine node (queryable via existing Data Source browse).

## 5. First-run wizard (5 steps)

Full-screen stepper. Runs when `asset_tree_config` row is missing.

### Step 1 — Preset

Card grid: Factory (default) / Hospital / Fleet / Farm / Custom.

Each card shows a small tree preview. Selecting a preset pre-fills Step 2.

### Step 2 — Configure level names

- List of levels with inline-rename inputs.
- **Add level below** button (up to 6 total).
- **Remove level** button (min 2 total).
- Live topic-pattern preview: `{root}/{level2}/{level3}/{level4}`.
- Live example: `factory/plant_A/machine_1/temperature`.

### Step 3 — Build initial tree

Two-pane layout — tree on left, node detail form on right.

**Left pane**:
- Recursive tree with expand/collapse.
- Right-click context menu: Add child / Rename / Delete / Duplicate subtree.
- Drag-drop to move a subtree within the same depth.
- **"Import from YAML"** button — paste a nested spec and auto-build.
- Buttons: `+ Add plant`, `+ Add machine`, `+ Add sensor` (context-aware).

**Right pane** — depends on selected node type:
- **Any node**: name, description, location tag, computed topic path (copyable).
- **Machine node**: additional **"Copy sensors from another machine"** button (opens a picker of existing machines, clones their sensor children).
- **Sensor leaf**: unit, sample rate, expected min/max, data type. Preset templates panel below (Vibration monitor / Thermal / Rotating machinery / Blank).

**Skip option**: users can skip building the tree entirely and add machines later. Wizard still saves the level config.

### Step 4 — MQTT publisher rules

- Show the topic pattern devices must follow.
- Radio: **Strict** (default, recommended) / **Learn** (R&D, workshops).
- Editable meta-topic exceptions list (default `_meta, _health, _config, _cmd`).
- **"Test a topic"** widget: paste any topic → tell the user if it matches, which leaf it routes to, and any warning.

### Step 5 — Confirm & finish

Summary card of what was configured. Button: **Start using CiRA ME →** drops into the new main navigation.

## 6. Ongoing admin (`Settings → Asset Tree`)

Same two-pane layout as wizard Step 3, plus:

- **Audit log tab**: who added / renamed / retired / moved what and when.
- **Retire button** on any node — moves the subtree's folders to `_archive/`, hides from live views, preserves historical data.
- **Move subtree** — drag-drop across depth-matched slots.
- **Rename** with warning modal — devices must be reconfigured to publish to the new topic; existing data on the old path stays where it is.
- **Bulk edit** for sensor metadata — change sample rate on all 24 temperature sensors at once.

## 7. Sidebar restructure

**Before** (roughly):
```
Dashboard · Projects · Data Source · Windowing · Features · Training
· Deploy · ME-LAB · MQTT Broker · App Builder · Folder Watcher
```

**After**:
```
Asset Tree                                    [🔍 search]
  ▸ factory
    ▾ plant_A                                 (auto-collapsed past plant unless explicit)
      ▸ machine_1                             ← primary navigation target
      ▸ machine_2
    ▸ plant_B
Live Broker                                   ← Mosquitto status card
Global tools                                  ← operate across the whole tree
  ME-LAB · App Builder · Folder Watcher · MQTT Broker
Settings
  Asset Tree · Machine Groups · Users · MQTT Rules · Preferences
```

Old top-level pipeline pages (Data Source, Windowing, Features, Training, Deploy) move **inside** the machine workspace.

## 8. Machine workspace

Selecting a machine in the tree opens its workspace. Breadcrumb + 6 tabs:

```
factory / plant_A / machine_1                    [Retire] [Rename]

[Overview] [Data] [Models] [Deploy] [Labels] [History]
```

### 8.1 Overview
- Live sensor tile row: current value + connection status per sensor.
- Recent activity feed (dataset auto-ingested, model deployed, anomaly detected, ...).
- Quick action buttons (Upload data, Train a model, Deploy an app).

### 8.2 Data
- Data Source view scoped to this machine's folder.
- Table of datasets (auto-ingested rolling daily CSVs + manual uploads).
- Existing File Manager available (reused from earlier work).

### 8.3 Models
- List of models attached to this machine, grouped into:
  - **Trained on this machine only** — solo models.
  - **Group models this machine participates in** — cross-machine models from a group.
- Each row: name, training approach (ML / DL / Foundation / TI), size, accuracy, deployed status.
- **"Train new model"** button → drops into Training view with scope toggle.

### 8.4 Deploy
- App Builder apps scoped to this machine.
- Existing App Builder flow, filtered.

### 8.5 Labels
- Placeholder in MVP. Phase E adds OculusT integration here.
- For now: shows a link "Coming soon — open OculusT (separate app) to label historical data."

### 8.6 History
- Long-timeline chart of any sensor (dropdown to switch).
- Model predictions overlaid.
- Zoom / pan / export.

## 9. Machine Groups

New settings page `Settings → Machine Groups`. Simple table:

```
Name                 Members   Description                     Actions
━━━━━━━━━━━━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━
All extruders           12    Baseline for extruder fleet     [Edit] [Delete]
Plant A baseline         3    First-quarter reference          [Edit] [Delete]
New commissioning        4    Machines still in ramp-up        [Edit] [Delete]
```

Edit modal:
- Name + description.
- Mini tree picker with checkboxes on machine-level nodes (sensors greyed out).
- Retired machines show a "(retired)" tag next to the name.

## 10. Cross-machine training

### 10.1 Training view — scope selector at the top

```
Training data source
  ( ) Just this machine     — machine_1 only              (500 samples)
  (●) A group of machines   — [ All extruders  ▾ ]        (12 machines, 6,247 samples)
  ( ) Ad-hoc selection      — [ pick machines from tree… ]

Selected: 12 machines, 6,247 samples, 3 classes
✓  Sensor sets compatible
```

Everything downstream (Windowing config, Features, Training algorithm) operates on the union of the selected machines' data.

### 10.2 Compatibility validation

Before training kicks off, backend validates:

1. All selected machines have identical sensor **names**.
2. Sensor **units** match across all machines.
3. Sensor **sample rates** match (or user opts into resampling).
4. Sensor **data types** match.

If any mismatch:
```
⚠  Cannot train — sensor mismatch across selected machines:

   machine_1  →  temperature, vibration, pressure
   machine_5  →  temperature, vibration                ← missing pressure
   machine_9  →  temp,        vibration, pressure     ← named differently

   Fix: rename or add missing sensors so all match, then retry.
```

Training is blocked until resolved.

### 10.3 Model artifact metadata

Every saved model records:

```json
{
  "trained_on_machines": [1, 5, 9, 11, 12, 15, ...],
  "trained_via_group":   "All extruders",
  "trained_via_group_snapshot_at": "2026-07-18T14:22:00Z",
  "deploy_targets":      [1, 5, 9]
}
```

`deploy_targets` is editable post-training via **"Rebind machines"** on the model detail view.

## 11. Legacy Projects migration

Existing `projects` rows are surfaced under a synthetic `_legacy` machine per project. All existing DataSessions / WindowedSessions / SavedModels remain reachable exactly as they were. No breaking change to APIs.

**Optional migration wizard (Phase B extension, not required):**
- Table of existing projects.
- For each project, admin picks: (a) leave in legacy, (b) reassign to a machine (dropdown of tree machines).
- Reassignment moves DB references + copies dataset files into the target machine's folder.

## 12. MQTT ingestion router (Phase D detail)

Backing point 1 from the workshop debrief. New service subscribes to Mosquitto and writes to the tree.

**Config UI** (`Settings → MQTT Rules`):
- Enable / disable the router.
- Rotation policy (daily / hourly / N-samples).
- Storage format (CSV default, Parquet optional if we add pyarrow).
- Retention policy (keep 30 days / 90 days / forever).
- Rejected-topic log viewer.

**File layout**:
```
data/factory/plant_A/machine_1/temperature/2026-07-18.csv
                             /vibration/2026-07-18.csv
                             /_meta/2026-07-18.log
data/_rejected_topics/2026-07-18.log
```

Data Source browse in the machine workspace shows these files as regular datasets.

## 13. OculusT integration (Phase E detail)

Existing `D:\CiRA OculusT` docker app at localhost:3010 provides Plotly.js box-selection labeling on time-series CSVs.

**Integration approach (Phase E, Option A from the discussion):**
- Add a Docker network bridge so OculusT can read `data/factory/...` from CiRA ME's disk mount.
- The **Labels tab** in a machine workspace opens OculusT in a new browser tab, prefilling the dataset path.
- OculusT saves labeled CSVs back to `data/factory/{plant}/{machine}/labels/{campaign}/`.
- Windowing view picks up the labeled version when present.

Auth alignment (shared session cookie) is a separate small task.

**Option B (merge OculusT's frontend into CiRA ME) is a later possibility** — Option A ships fast.

## 14. Phase breakdown

Total: ~6-7 weeks calendar time. Each phase is shippable on its own.

### Phase 0 — Dark/Light mode toggle (~2 hours)
Small, unrelated customer request. Slot in first because it's tiny and adds visible value early.

- Use existing Vuetify `useTheme()` API. Add a toggle icon in the top header bar.
- Persist in `localStorage` under `cira.theme` (`'dark' | 'light'`, default `'dark'`).
- Ship as its own commit before anything else.

### Phase A — Data model + first-run wizard + basic tree admin (~2 weeks)
- New DB tables + migrations.
- Backend endpoints for tree CRUD, groups CRUD, compatibility validation.
- 5-step wizard.
- `Settings → Asset Tree` admin view (two-pane, no drag-drop yet).
- Legacy compat: existing projects appear under `_legacy` machine.

**Milestone**: An admin can define a factory / plant / machine / sensor tree and see it persist. Existing pipeline still works via legacy nodes.

### Phase B — Sidebar restructure + machine workspace + scoped pipeline (~1 week)
- Sidebar becomes the tree.
- Machine workspace with 6 tabs (Overview / Data / Models / Deploy / Labels / History).
- Data Source / Windowing / Features / Training / Deploy pages become tab content, always aware of the currently-selected machine.
- Retire / rename / move subtree.

**Milestone**: End-to-end pipeline works when scoped to a real tree machine. Old top-level pipeline pages redirect into the tree.

### Phase C — Machine Groups + cross-machine training (~1 week)
- `Settings → Machine Groups` page.
- Training view scope toggle (single / group / ad-hoc).
- Compatibility validation.
- Model bindings.

**Milestone**: A model can be trained on 12 machines and deployed to a subset.

### Phase D — MQTT ingest router (~1 week)
- Background service subscribes to Mosquitto.
- Rolling daily CSVs land in tree folders.
- Rejected-topic log viewer.
- Retention policy job.
- Companion small fixes: publisher rate limit clamp + topic namespace docs (addresses workshop point 3).

**Milestone**: Devices publish 24/7, files accumulate under the tree, data scientists browse and train from history.

### Phase E — OculusT labels integration (~1-2 weeks)
- Docker network bridge or shared disk mount so OculusT sees CiRA ME's tree.
- Labels tab in machine workspace opens OculusT prefilled.
- Auth cookie alignment.
- Labeled CSVs flow back into Windowing.

**Milestone**: Full loop — ingest → label → train — usable end-to-end from a machine workspace.

## 15. Open questions to resolve at Phase A kickoff

1. **Migration UX**: how loud should the "you have unmigrated legacy projects" indicator be? Nagging vs quietly-there.
2. **Sensor unit taxonomy**: predefined list (dropdown) or free text? My default: dropdown with common presets + "custom" option.
3. **Retire vs delete for nodes**: I've assumed retire-only (never destructive). Confirm before Phase A ships.
4. **Group visibility**: are all groups visible to all users, or per-user? My default: all-visible for MVP, per-user access control in a later phase.

## 16. Not doing (explicit non-goals)

- **Model federation** (each machine trains locally, results averaged) — real concept, revisit when a customer asks for data-privacy between machines.
- **Automatic group discovery** ("these 12 machines look similar, group them?") — v2 idea.
- **Weighted training samples** (contribute machine_1 2× because better labels) — ML weeds, defer.
- **Sensor-level workspace** — click-to-filter for MVP; separate page only if customers ask.
- **Real time-series database** (TimescaleDB / InfluxDB container) — the file-based ingest covers 90% of customer queries. Bolt on a TSDB later against the same source of truth if range queries become common.
- **Nested groups** (groups of groups) — flat list only for MVP.

## 17. Related / follow-on

- **Point 3 (65-user MQTT hang)** — addressed opportunistically in Phase D via topic namespacing (`factory/{plant}/{machine}/{sensor}` is user-scopeable) + publisher rate limiting. Not a separate plan.
- **MOMENT foundation model** — sits perfectly on top: foundation models eat raw windows, and the tree ingests them. Bind MOMENT to a machine or a group; training approach = 'foundation' as planned. Consider running MOMENT Phase A concurrently with Asset Tree Phase C.
- **Fast Mode 5-feature parity gap** — unrelated, still worth ~1hr fix per DEFERRED-2026-07-17.md item 6.
