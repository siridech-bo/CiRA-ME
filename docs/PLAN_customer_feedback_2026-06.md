# Customer Feedback Plan — June 2026

**Source:** customer email feedback (chayapatp deployment), received 2026-06-19
**Owner:** siridech.bo@kmitl.ac.th
**Status:** planning — no code changes yet
**Last updated:** 2026-06-23

---

## Why this document exists

A production customer (chayapatp) sent three items of feedback. Two are feature requests, one is a bug. Before touching code I traced each through the codebase to ground the plan in what actually exists. This document captures what I found, what was decided, and what is still open, so the next session can pick up cold without re-doing the discovery.

This is **production software** — CiRA ME v1.1+ shipped with a polished user manual ([docs/USER_MANUAL.md](USER_MANUAL.md)). Treat every change as a production change.

---

## The three items, ranked

| # | Item | Type | Effort | Priority |
|---|---|---|---|---|
| 1 | TI REGR training returns 400 BAD REQUEST | Bug (production regression) | ~30 min | **P0 — blocks customer demo** |
| 2 | Folder Watcher + ML Prediction | New feature | ~1.5 days | P1 — fills real industrial gap |
| 3 | End-to-end project status view (not just App Builder) | New feature | ~1 week | P1 — needed for multi-project users |

Recommended order: **1 → 2 → 3**. Item 1 unblocks the customer immediately; item 2 has small scope; item 3 is the biggest and benefits from doing item 2 first (so we know what "deploy" includes).

---

## Item 1 — TI REGR 400 BAD REQUEST (production regression)

### Symptom

Customer trains REGR models on the TI TinyML tab. **All 10 REGR variants** (REGR_1k, REGR_2k, REGR_3k, ..., REGR_8k NPU, REGR_500 NPU, etc.) fail with the same surfaced error:

> `400 Client Error: BAD REQUEST for url: http://cirame-ti-modelmaker:5200/train`

Traditional ML and TimesNet (Deep Learning) training succeed on the same dataset. Only TI training fails.

### Root cause — found

Commit [1bce223](../commit) (2026-05-06, "Persist user data across deployment updates") widened the production volume mounts in [deployment/docker-compose.yml:28,102](../deployment/docker-compose.yml):

| Container | Old mount (≤ May 6) | New mount (current production) |
|---|---|---|
| backend | `./shared:/app/datasets/shared` | `./datasets:/app/datasets` |
| ti-modelmaker | `./shared:/app/data/datasets/shared` | `./datasets:/app/data/datasets` |

But the backend's path-remap in [backend/app/routes/ti_tinyml.py:89-91](../backend/app/routes/ti_tinyml.py#L89-L91) and [ti_tinyml.py:129-131](../backend/app/routes/ti_tinyml.py#L129-L131) **was not updated** when the mount changed. It still says:

```python
ti_dataset_path = dataset_path.replace(
    '/app/datasets/shared', '/app/data/datasets/shared'
)
```

For any user-private file (e.g. `/app/datasets/<user_id>/learn27.csv`), the substring doesn't match → `.replace()` is a no-op → backend sends `dataset_path=/app/datasets/<user_id>/learn27.csv` to the TI container → TI container's mount only covers `/app/data/datasets/` → [server.py:217-218](../ti-modelmaker/server.py#L217-L218) returns:

```python
return jsonify({'error': f'Dataset not found: {dataset_path}'}), 400
```

The backend's `resp.raise_for_status()` then bubbles the HTTPError string up to the frontend, which is what the customer sees in the screenshot. **Why ML/DL works:** those trainers run inside the backend container reading from its own mount — they never cross the container boundary, so the broken remap doesn't touch them. TI is the only path that crosses containers.

### Why all 10 REGR fail identically

The 400 fires at request validation, **before** the per-model training loop ([server.py:217-218](../ti-modelmaker/server.py#L217-L218)). If any one model could be reached, it would be reported in the `errors[]` array of a 200 response, not a top-level 400. The uniform 400 is the smoking gun for a request-level validation failure.

### The fix (two parts)

**Part A — correct the path remap** (both copies):

```python
# backend/app/routes/ti_tinyml.py:89-91 and :129-131
ti_dataset_path = dataset_path.replace('/app/datasets', '/app/data/datasets')
```

**Part B — surface the real error**

Currently `resp.raise_for_status()` discards the TI container's JSON `{error: ...}` body. The customer only sees `"400 BAD REQUEST for url: ..."` — opaque. Wrap the POST:

```python
resp = requests.post(f'{ti.base_url}/train', json=payload, timeout=660)
if not resp.ok:
    try:
        ti_error = resp.json().get('error', resp.text)
    except Exception:
        ti_error = resp.text
    return jsonify({'error': f'TI training failed: {ti_error}'}), 500
return jsonify(resp.json())
```

This applies to **both** `/api/ti/train` and `/api/ti/train-stream` routes.

### Also check: dev compose vs prod compose mismatch

[docker-compose.yml](../docker-compose.yml) (dev/local) still uses the **old** mount pattern (`./shared:/app/datasets/shared`). After the fix, dev would break because the remap (`/app/datasets` → `/app/data/datasets`) would also need the dev mount to be `./datasets:/app/datasets`.

**Decision needed:** either
- (a) Update dev compose to match prod layout (clean, but means local devs need to recreate `./shared` → `./datasets/`), or
- (b) Make the remap config-driven via an env var (e.g. `TI_DATASET_PATH_PREFIX`)

Recommendation: **(a)** — consistency between dev and prod is worth more than the one-time local migration. Matches how production is shipping.

### Verification plan

1. Apply Part A + Part B in dev (with updated dev compose)
2. Train one REGR model on a `shared/` file → expect success
3. Train one REGR model on a per-user private file → expect success (this is the case that's broken)
4. Train one TI classification model → expect success (was also broken silently)
5. Trigger a real TI error (e.g. wrong task type) → expect the actual error message surfaced, not a 400 string

### Files to touch

- [backend/app/routes/ti_tinyml.py](../backend/app/routes/ti_tinyml.py) — path remap + error surfacing (both routes)
- [docker-compose.yml](../docker-compose.yml) — update dev mount if going with option (a)

---

## Item 2 — Folder Watcher + ML Prediction (new feature)

### The customer's use case

A factory machine or PLC writes sensor data into text files in a folder (e.g. `\\fileserver\sensors\machine_001\input\`). CiRA ME watches that folder, every 60 seconds picks up any new file, runs each row through a trained model, drops predictions into an output folder using the same filename, and deletes the input file.

See [Feature_Watcher-ML-Prediction.png](../shared/Feature_Watcher-ML-Prediction.png) — the customer's flow diagram.

### Why this matters

PLCs and legacy machines often cannot speak HTTP or MQTT, but **every PLC can write a file**. This pattern (file in, file out) is how industrial customers commonly integrate with new systems. Currently CiRA ME supports:

| Pattern | Trigger | Transport | Latency | Status |
|---|---|---|---|---|
| App Builder published app | User upload | HTTP POST + CSV | seconds | ✅ exists |
| MQTT live streaming | Producer publishes | MQTT topic | sub-second | ✅ exists |
| **Folder Watcher** | **File appears** | **Filesystem** | **~60s** | ❌ this feature |

### Key architectural insight from the user manual

From [USER_MANUAL.md §4.2](USER_MANUAL.md):

> **Raw Mode (no windowing) toggle:** For pre-processed tabular data where each row is already a complete feature vector. Skips windowing AND feature extraction.

The customer's diagram shows:
- Input row: `71.49,1.47,31.30,1576.15,218.83` — 5 values
- Output: `prediction=NORMAL, confidence=0.9512` per row, one prediction per row

**This is literally Raw Mode classification.** Each row = one sample = one prediction. No windowing needed. No feature extraction needed. The Folder Watcher does not need its own pipeline — it can call an existing **ME-LAB endpoint** (with API key auth) and let the saved model do its job.

That collapses the implementation:

```
Watcher daemon → for each file → for each row → POST to ME-LAB endpoint → write CSV row → delete input
```

### Architecture decision

**Where the watcher daemon lives:** inside the backend container, with a worker pool similar to the existing [mqtt_publisher.py](../backend/app/routes/mqtt_publisher.py) pattern, but with **DB-backed state** so watchers survive backend restarts.

**Why not a separate container:** complicates the deployment story (one more `.tar` to ship, one more compose entry). The backend already does long-running work (MQTT publisher threads, ME-LAB inference), so it's the right home. Use threading.Thread with daemon=True and rehydrate from DB on app startup.

**Why not just call ME-LAB internally:** we can. The watcher uses `ModelManager.predict()` directly ([backend/app/services/melab_service.py](../backend/app/services/melab_service.py)) without HTTP round-trips. API key auth is a fallback if the watcher ever lives in a different process.

### Required DB schema

```sql
CREATE TABLE IF NOT EXISTS folder_watchers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    input_folder TEXT NOT NULL,            -- container-side path
    output_folder TEXT NOT NULL,           -- container-side path
    endpoint_id TEXT NOT NULL,             -- FK to melab_endpoints.id
    poll_interval_s INTEGER DEFAULT 60,
    file_glob TEXT DEFAULT '*.txt',        -- which files to pick up
    file_has_header INTEGER DEFAULT 0,     -- 0/1
    status TEXT DEFAULT 'stopped',         -- stopped|running|error
    last_run_at TEXT,
    last_error TEXT,
    files_processed INTEGER DEFAULT 0,
    rows_processed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (endpoint_id) REFERENCES melab_endpoints(id)
);
```

### Required volume mount

[docker-compose.yml](../docker-compose.yml) and [deployment/docker-compose.yml](../deployment/docker-compose.yml):

```yaml
volumes:
  - ./watcher-data:/app/watcher-data
```

Subfolder convention: `./watcher-data/<user_id>/<watcher_name>/input/` and `.../output/`. Customers who want network shares can replace `./watcher-data/` with a bind-mount to a UNC path / NFS mount.

### Output file format (per the diagram)

CSV, same filename as input, columns:

```
source_file, record_index, sensor_values, prediction, confidence, predicted_at
machine_001.txt, 1, 71.49|1.47|31.30|1576.15|218.83, NORMAL, 0.9512, 2026-06-19 14:31:05
```

Note `sensor_values` uses pipe `|` delimiter (per diagram).

### Error handling (the diagram is silent — proposed)

- Mid-write detection: skip files whose mtime is within last 5 seconds; pick up next cycle
- Predict failure (NaN, wrong feature count, model error): write `prediction=ERROR, confidence=0, predicted_at=<timestamp>` for the failing row; continue processing other rows
- File-level catastrophic failure: move input file to `/error/` instead of `/output/`, log to DB

### UI plan

New sidebar entry **"Folder Watcher"** under SERVICES (peer of App Builder). Three pages:

1. **List view** — table of watchers with status, last run, files processed, start/stop buttons
2. **Create/edit form** — pick endpoint, input folder, output folder, poll interval, file glob, header toggle
3. **Detail view** — recent activity log, error history, throughput chart

### Effort estimate (revised down from 3 days → 1.5 days)

- Backend: blueprint + DB migration + worker pool + ModelManager reuse → ~0.5 day
- Frontend: list + create form + detail → ~0.5 day
- Compose + volume + smoke tests → ~0.5 day

### Open questions for the customer

| # | Question | Why it matters |
|---|---|---|
| 1 | One watcher = one model? Or can a watcher run multiple models per file (multi-model output)? | DB schema (one endpoint_id vs many) |
| 2 | Does the input file ever have a header row, or always headerless? | Default value of `file_has_header` |
| 3 | Network share or local folder? If network share, will Windows host mount the SMB share into Docker? | Deployment story — biggest risk to delivery |
| 4 | Should each prediction also be persisted to DB (auditable history) or output file only? | DB size vs auditability tradeoff |
| 5 | Multi-tenant: per-user folders, or one shared system-wide folder? | Default folder layout under `./watcher-data/` |
| 6 | Does the watcher need to support time-series files (multi-row signal = one prediction via windowing), or only Raw Mode (one row = one prediction)? | Whether we reuse pipeline_replay vs just ME-LAB |

### Files to touch (anticipated)

- New: `backend/app/routes/folder_watcher.py` (CRUD + start/stop endpoints)
- New: `backend/app/services/folder_watcher_service.py` (worker pool + DB rehydration)
- Edit: [backend/app/models.py](../backend/app/models.py) (add `folder_watchers` table + class)
- Edit: [backend/app/__init__.py](../backend/app/__init__.py) (register blueprint + start workers on app boot)
- New: `frontend/src/views/FolderWatcherListView.vue`, `FolderWatcherEditView.vue`
- Edit: frontend sidebar / router
- Edit: both `docker-compose.yml` files (volume mount)
- Edit: [docs/USER_MANUAL.md](USER_MANUAL.md) — new section "Folder Watcher"

---

## Item 3 — End-to-end project status view

### The customer's ask

> "Tell me end-to-end status from data to deploy in every project, not just App Builder."

Across all of a user's bodies of work, show pipeline stage progress: Data → Windowing → Features → Training → Deploy.

### Current gap (what I found)

A `projects` table already exists ([backend/app/models.py:42-53](../backend/app/models.py#L42-L53)) with `id, name, description, mode, user_id, config(JSON), created_at, updated_at`. A `Project` class with CRUD is at [models.py:293-367](../backend/app/models.py#L293-L367). Training routes accept optional `project_id` ([training.py:161,195,240,290,332,370](../backend/app/routes/training.py#L161)).

**But:**
- No `/api/projects` blueprint, no UI
- Only `training_sessions` has a `project_id` FK. `saved_models`, `melab_endpoints`, `app_builder_apps` are owned by `user_id` only (no project link)
- Pre-training stages (data session, windowing config, feature session) live only in **backend memory + frontend Pinia store** ([pipeline.ts:106-121](../frontend/src/stores/pipeline.ts)). Lost on container restart or browser refresh.

So today CiRA ME is technically a project-based product (the table exists) but practically a session-based product (state lives in browsers and process memory). The customer's ask exposes this gap.

### Naming decision — ACCEPTED

**The actual UI string is "Folder Management"** ([AdminView.vue:68,71](../frontend/src/views/AdminView.vue#L68)), not "Project Folders". The phrase "project folder" only appears in the user manual ([USER_MANUAL.md:129](USER_MANUAL.md#L129)) as a description, not in the UI.

So the rename is **doc-only** (one line in the user manual: "project folder" → "dataset folder"). The new "Projects" sidebar concept can be introduced cleanly without UI churn.

### Plan

**Phase A — persist pipeline stages (the meaningful work)**

Add three tables that mirror the Pinia store:

```sql
CREATE TABLE data_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    format TEXT NOT NULL,                  -- csv|ei_json|ei_cbor|cira_cbor
    sensor_columns JSON,
    label_column TEXT,
    labels JSON,
    total_rows INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE windowed_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    data_session_id INTEGER NOT NULL,
    config JSON,                           -- window_size, stride, label_method, test_ratio, split_strategy, no_windowing
    num_windows INTEGER,
    window_shape JSON,
    normalization JSON,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (data_session_id) REFERENCES data_sessions(id)
);

CREATE TABLE feature_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    windowed_session_id INTEGER NOT NULL,
    method TEXT,                           -- lightweight|tsfresh|both
    feature_names JSON,
    num_features INTEGER,
    selection JSON,                        -- optional selection state
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (windowed_session_id) REFERENCES windowed_sessions(id)
);
```

Wire each pipeline "Apply" button to write to DB, not just memory.

**Phase B — wire `project_id` through everything**

Add `project_id` FK (nullable for backwards-compat) to:
- `saved_models`
- `melab_endpoints`
- `app_builder_apps`

Add `/api/projects` blueprint: `GET /api/projects` (list mine), `POST /api/projects` (create), `GET /api/projects/<id>` (detail with all stage rows), `PATCH /api/projects/<id>` (rename), `DELETE /api/projects/<id>`.

**Phase C — project header in pipeline pages**

Add a project selector to the top of the pipeline pages:
- Default behaviour: auto-create a draft project on first data upload (lowest friction, preserves current UX)
- User can rename anytime
- "New Project" button in the header always available

**Phase D — Projects view**

New sidebar entry **"Projects"** (top of the list, above PIPELINE).

**List view (table per project):**

| Column | Source |
|---|---|
| Name | `projects.name` |
| Mode | `projects.mode` (chip color) |
| Stage status | Data ✅/❌ from `data_sessions`, Windowing from `windowed_sessions`, Features from `feature_sessions`, Training from `training_sessions.status`, Deploy from count(`melab_endpoints` + `app_builder_apps`) |
| Best metric | Best `saved_models.metrics` row for this project (R² for regression, F1 for classification) |
| Updated | `projects.updated_at` |
| Actions | Open · Duplicate · Delete |

**Detail view (per project):**

Per-stage cards showing config used, when, by which user, with quick links to the underlying pipeline pages pre-loaded with that project's state. Includes deploy targets (ME-LAB endpoints + App Builder apps + TI MCU exports if tracked).

**Phase E — backwards compatibility**

Auto-create one **"Default Project"** per user on first login after upgrade. Adopt all orphan `saved_models`, `melab_endpoints`, `app_builder_apps` into it. Zero data loss, clean view from day one. Migration script in [backend/app/models.py](../backend/app/models.py) init block.

**Phase F — TI run tracking** (sub-task to decide separately)

TI runs today don't write `saved_models` rows — artifacts live in the TI container. For TI runs to show up in project status, either:
- (i) Add a `ti_training_runs` table tracking run_id, project_id, model_names, metrics, artifacts_path
- (ii) Make TI runs write `saved_models` rows like ML/DL do

Recommendation: **(ii)** — uniform handling everywhere downstream (dashboard, project status, deploy). Requires changes to [routes/ti_tinyml.py](../backend/app/routes/ti_tinyml.py) post-training to persist.

### Effort estimate

- Phase A (DB migrations + persist stage state): ~1 day
- Phase B (project routes + FK migration): ~1 day
- Phase C (pipeline header + save-on-apply wiring): ~1 day
- Phase D (project list + detail UI): ~2 days
- Phase E (orphan adoption migration): ~0.5 day
- Phase F (TI run tracking): ~0.5 day
- **Total: ~6 days for v1**

### Open questions for the customer

| # | Question | Why it matters |
|---|---|---|
| 1 | One project = one dataset? Or can a project swap datasets mid-life? | DB schema (1:1 vs 1:N data_sessions per project) |
| 2 | Should "Deploy" status include all 4 targets (ME-LAB, App Builder, TI MCU, Jetson SSH) as separate badges, or aggregated as one ✅? | UI density |
| 3 | Should other users on the same server see each other's projects (read-only) or only admin sees all? | Permissions on `GET /api/projects` |

### Files to touch (anticipated)

- New: `backend/app/routes/projects.py`
- Edit: [backend/app/models.py](../backend/app/models.py) — 3 new tables + Migration: `project_id` FK on saved_models/endpoints/apps
- Edit: [backend/app/routes/data_sources.py](../backend/app/routes/data_sources.py) — persist data_session
- Edit: [backend/app/routes/training.py](../backend/app/routes/training.py) — already takes `project_id`, just wire it through frontend
- Edit: [backend/app/routes/features.py](../backend/app/routes/features.py) — persist feature_session
- Edit: [backend/app/routes/ti_tinyml.py](../backend/app/routes/ti_tinyml.py) — persist TI runs as saved_models
- Edit: [frontend/src/stores/pipeline.ts](../frontend/src/stores/pipeline.ts) — add projectId, project actions
- Edit: each pipeline view to bind to project header
- New: `frontend/src/views/ProjectListView.vue`, `ProjectDetailView.vue`
- Edit: sidebar + router
- Edit: [docs/USER_MANUAL.md](USER_MANUAL.md) — rename "project folder" → "dataset folder"; add Projects section

---

## Accepted decisions

1. **Rename "project folder" → "dataset folder" in user-facing docs.** UI doesn't need changes (it's already "Folder Management"). Only [docs/USER_MANUAL.md:129](USER_MANUAL.md#L129) needs updating.
2. **TI fix approach (path remap correction + error surfacing) approved in principle**, pending dev/prod compose alignment decision (option a vs b in §Item 1).
3. **Folder Watcher will be Raw Mode classification calling existing ME-LAB endpoints**, not its own pipeline.
4. **Projects v1 = one project = one dataset.** Swap dataset = clone project. Simpler.
5. **Orphan adoption via Default Project per user** on the upgrade migration.

---

## Open questions waiting on customer

Forwarded as a single list to ask the customer in one go:

### Folder Watcher
1. One watcher = one model, or multiple models per watcher?
2. Input files headered or headerless?
3. Network share or local folder for input? If network share, who handles the Windows SMB mount?
4. Should predictions be persisted to DB too (audit trail), or only the output file?
5. Per-user folders or one shared system folder?
6. Time-series files (multi-row signal → 1 prediction) supported, or only Raw Mode (1 row → 1 prediction)?

### Project status view
7. Can one project swap datasets, or is project locked to one dataset?
8. Deploy status: 4 separate badges (ME-LAB, App Builder, TI, Jetson) or one aggregated ✅?
9. Cross-user project visibility (admin only? everyone read-only? strict per-user?)

---

## Stale memory entries to update (housekeeping)

While exploring I noticed memory contains some outdated facts. Update at end of this work:

| Memory claim | Reality (per code / manual) | Action |
|---|---|---|
| Default password `admin123` | Manual says `cira123` | Verify in code, then update memory |
| `max_folder_mb` default `500` | Manual says `1000` | Verify in `models.py:161`, then update memory |
| "Pre-commercial / POC" | Polished user manual exists, customers in production | Update memory: v1.1 shipping, paying customers |
| "8 templates" in App Builder | Manual lists ~11 (3 CSV + 3 MQTT + Recorder + 4 multi-model + Blank) | Update memory |

---

## Key file references (for next session)

| Concern | File:line |
|---|---|
| TI bug — broken path remap | [backend/app/routes/ti_tinyml.py:89-91](../backend/app/routes/ti_tinyml.py#L89-L91), [:129-131](../backend/app/routes/ti_tinyml.py#L129-L131) |
| TI container train validation | [ti-modelmaker/server.py:195-218](../ti-modelmaker/server.py#L195-L218) |
| Production compose mount layout | [deployment/docker-compose.yml:28,102](../deployment/docker-compose.yml#L28) |
| Dev compose mount layout (mismatched!) | [docker-compose.yml](../docker-compose.yml) |
| `projects` table (half-built) | [backend/app/models.py:42-53](../backend/app/models.py#L42-L53), [:293-367](../backend/app/models.py#L293-L367) |
| Pipeline state (ephemeral) | [frontend/src/stores/pipeline.ts:106-121](../frontend/src/stores/pipeline.ts#L106-L121) |
| ME-LAB ModelManager (Folder Watcher will reuse) | [backend/app/services/melab_service.py](../backend/app/services/melab_service.py) |
| MQTT publisher (worker pool pattern reference) | [backend/app/routes/mqtt_publisher.py:20](../backend/app/routes/mqtt_publisher.py#L20) |
| App Builder pipeline runner (reference for Raw Mode handling) | [backend/app/routes/app_builder.py](../backend/app/routes/app_builder.py) |
| User Manual (source of truth for user-facing behavior) | [docs/USER_MANUAL.md](USER_MANUAL.md) |

---

## Next actions

1. **Get answers** to the 9 open customer questions (forward as one email)
2. **Decide dev/prod compose alignment** (Item 1 fix Part C): option (a) align dev to prod, or (b) env-var the remap
3. **Implement Item 1** (~30 min once decided) — unblocks the customer
4. **Implement Item 2** (~1.5 days, after questions 1-6 answered)
5. **Implement Item 3** (~6 days, after questions 7-9 answered)

When picking this back up: read this document, then check `git log --since=<this doc's date>` to see what's already been done.
