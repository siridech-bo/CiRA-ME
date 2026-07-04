# Customer Feedback Tracker

Running log of customer feature requests, bugs, and their fixes. Read this
file **first** at the start of any customer-touching session — context
compression can drop everything else in the conversation, but this file
survives. Update it at the end of every session that changed customer-facing
state.

## Legend

- ✅ Done + verified end-to-end (browser, curl, or unit)
- 🔧 In progress — commits landed but not fully verified
- ⏸  Blocked — waiting on customer answer or external decision
- 📋 Open — queued, not started
- ❌ Rejected — deliberately not doing (reason recorded)

Each item lists commit SHAs where relevant. `git show <sha>` for details.

---

## OPEN ITEMS

**None.** All three big customer requests from the 2026-06 / 2026-07
round shipped 2026-07-04: F1 Folder Watcher (T15), F2 Multi-Dataset
Wizard (T16), F4 Project Status view (T17). Small housekeeping items
in [FOLLOW-UPS.md](./FOLLOW-UPS.md) (~40 min total) and larger parked
initiatives in [FUTURE-WORK.md](./FUTURE-WORK.md) remain.

---

## DONE — this round (July 2026)

Ordered newest first. Every SHA is on the `master` branch.

### T17. F4 End-to-end Project Status view (v1)
**Status:** ✅ Done | **Shipped:** 2026-07-04
Projects sidebar entry showing pipeline stage progress (Data →
Windowing → Features → Training → Deploy) for every body of work.
Three customer-facing decisions shipped verbatim: F4.1 aggregated
Deploy badge with hover-tooltip breakdown (ME-LAB / App Builder /
TI MCU / Jetson), F4.2 strict per-user visibility with admin
`?all=1`, F4.3 one project = one dataset (swap = clone).

Feature selection reordering shipped as a per-project **feature
template** (stable ordered contract across retrainings) so the ME-LAB
/ App Builder payload column order stays fixed even after retraining
with different feature-extraction runs.

Internal architecture decisions locked at implementation time (all 5
user-approved 2026-07-04):
- **Clone force-retrains** — only `projects.config` + `feature_templates`
  copy across; `saved_models` / `training_sessions` do not. Rationale:
  the whole point of a clone is a new dataset — copying stale metrics
  would mislead.
- **`current_stage` = latest-apply-wins.** If user re-windows a trained
  project, list shows "windowing" not "training" — reflects reality.
- **Legacy project: one per user, mixed-mode.** Adopts all pre-F4
  orphan resources (saved_models / melab_endpoints / app_builder_apps /
  training_sessions) — except folder_watchers, which per Q5 stay
  detached (`project_id IS NULL`).
- **Persist Data/Windowing/Features on Apply**, not on ingest — user who
  browses a file and doesn't apply doesn't spam the DB.
- **Folder Watchers left detached** — pre-F4 watchers never appear in
  Deploy breakdown even after upgrade.

Schema: 5 new tables (`data_sessions`, `windowed_sessions`,
`feature_sessions`, `feature_templates`, `deploy_records`), 4 ALTERs
adding `project_id` to `saved_models` / `melab_endpoints` /
`app_builder_apps` / `folder_watchers`, and 1 ALTER adding
`current_stage` to `projects`. All `try/except`-wrapped for idempotent
boot. Explicit cascade in `Project.delete()` — child rows hard-deleted,
external refs (saved_models etc.) detach via `project_id = NULL`
(SQLite `PRAGMA foreign_keys` is off).

Backend: new `/api/projects` blueprint with 10 routes (list, CRUD,
clone, feature-template GET/PUT). `project_id` threaded through
windowing, features, save-benchmark, ME-LAB endpoint create, App
Builder publish, TI MCU export, deployment/package, cira-claw-package,
and Jetson SSH deploy. TI + Jetson deploys now write `deploy_records`
rows (wrapped in try/except so a DB blip doesn't sink the export).

Frontend: `ProjectsListView.vue` with auto-refresh (15s, tab-visibility
guard, mirrors F1 Folder Watcher pattern), 5-state stage chips
(complete / in_progress / not_started / skipped / failed), aggregated
Deploy badge with hover-tooltip breakdown. `ProjectDetailView.vue`
with per-stage cards + inline feature-template up/down reorder editor
(no new deps). Sidebar entry between Dashboard and PIPELINE subheader.
Pinia `pipeline.ts` extended with `projectId`, `setActiveProject`,
`createProjectAndAdopt`. Auto-create wired into `applyWindowing` —
first Windowing apply of a fresh session materializes a project named
`{dataset_stem} {timestamp}`. `project_id` sent on training, TimesNet,
and save-benchmark calls so trained models attach to the active
project.

User manual: `docs/USER_MANUAL.md` "project folder" → "dataset folder"
per plan §6.

Verified via docker-exec suite:
- All 5 new tables + 5 ALTERs present after fresh boot.
- Legacy adoption idempotent — exactly 1 Legacy project per orphan-
  owning user (34 total on the existing DB).
- All 10 project routes registered; auth-gate returns 401.
- Cascade delete: child rows (data/windowing/feature sessions +
  deploy_records) hard-deleted; saved_model `project_id → NULL`, row
  preserved.
- `GET /api/projects` response shape matches plan §2b (stages dict,
  deploy_breakdown per target, best_metric).
- Frontend bundle contains `ProjectsListView-*.js` +
  `ProjectDetailView-*.js` with the wire-in fixes applied.

**Known gap deliberately not fixed in v1:**
`/api/deployment/package/<id>` inserts `deploy_records` as
`target='ti_mcu'` unconditionally — CCS packages are TI-family so this
matches intent, but if Jetson-only packages ever flow through this
endpoint they'll be miscounted. Triage-worthy but not a v1 blocker.

### T16. F2 Multi-Dataset Wizard (v1)
**Status:** ✅ Done | **Shipped:** 2026-07-04
3-step wizard that runs N saved ME-LAB endpoints against M CSV datasets
and returns an aggregated matrix so the customer can pick the smallest
model meeting their confidence bar. Both customer-facing decisions
shipped verbatim: F2.1 schema strict (reject mismatched columns with a
clear per-dataset diff), F2.2 cell shows modal label + mean confidence
with per-class probability breakdown on hover.

Internal architecture decisions locked at implementation time:
- **Single-mode only** — mixing classification / regression / anomaly
  in one run rejected.
- **Anomaly supported** — cell shows label + score; no per-class probs.
- **100k-row cap** per dataset (fail-fast, actionable message).
- **Raw Mode only in v1** — windowed / feature-extracted models rejected
  at Step 2 with a clear "pick a raw-mode model" message.
- **Model size in bytes in CSV export, auto-scaled KB/MB in hover.**

Zero-behavior refactor also shipped: `_run_model_inference` extracted
out of `app_builder.py` into `ModelManager.predict_by_endpoint` in
`melab_service.py`. Folder Watcher (T15) also migrated to the new
canonical helper — no behavior change, single source of truth for
endpoint-based inference.

Endpoints:
- `POST /api/wizard/validate-datasets` — multipart upload OR
  `dataset_paths[]` referencing files under `datasets/`.
- `POST /api/wizard/run`
- `POST|GET /api/wizard/export?level=aggregated|per_row`
- `DELETE /api/wizard/runs/<run_id>`

Verified via docker-exec suite: predict_by_endpoint present on
ModelManager; module imports clean; all 4 routes registered; 401 on
unauth POST; aggregation logic returns real per-row latency from
persisted-prediction files (both new dict format + legacy list format
handled). Frontend needs manual browser QA (Step 1 mode-disable, Step 2
"From Datasets" tab file browser, matrix cell hover-menu rendering).

Coder-flagged bug **fixed inline before merge:** aggregated CSV export
was writing `avg_latency_ms: 0.0` because latencies weren't persisted
alongside predictions. Fix: persistence now writes
`{preds, latency_ms}` dict; export path unpacks the tuple and passes
real latency into `_aggregate_cell`. Old bare-list format kept working
as a back-compat degradation to 0 latency.

### T15. F1 Folder Watcher + ML Prediction (v1)
**Status:** ✅ Done | **Commits:** `0a35d50` `576da36` + File Browser | **Shipped:** 2026-07-04
Backend daemon polls a user-specified folder every N seconds, runs each
file's rows through a ME-LAB endpoint, writes results to an output
folder as CSV, deletes the input. Matches the customer's flow diagram
exactly (columns: `source_file, record_index, sensor_values, prediction,
confidence, predicted_at`; sensor values pipe-separated; input filename
preserved on output).

Failure-tolerant by design: output is written to `.tmp` first and
atomic-renamed into place; input is only deleted after successful
commit. Killing the worker mid-file leaves the input alone for the next
boot. Files whose mtime is within 5s are skipped (probably still being
written). Endpoint status is checked at every tick — a paused/deleted
endpoint flips watcher status to `error` with an actionable message.
Watchers survive backend restarts via DB rehydration (guarded against
Flask dev-reloader double-start).

Verified via a 3-part docker-exec suite:
- Happy path — 3-row input → 3 predictions → correct output CSV with
  exact customer-spec header, sensor values pipe-separated, input
  deleted after commit.
- No `.tmp` files left behind (atomic rename working).
- Endpoint-status guard — paused endpoint flips watcher to `error`,
  message actionable, input file preserved.

Shipped the design decisions verbatim (F1.1–F1.6). All 3 QA blockers +
5 important findings addressed before merge. Frontend list view has
auto-refresh, status chip, start/stop/delete; edit form disables the
endpoint select on edit (matches the backend's immutable rule).

**Post-ship UX addition (2026-07-04):** In-browser File Browser dialog
so users don't have to shell into `./watcher-data/<user_id>/<name>/`
to inspect what's flowing through. New button
(`mdi-folder-eye-outline`) on each row opens a dialog with Output /
Input / Errors tabs, listing files (name / size / mtime) with a preview
panel (200 KB cap, truncation flag). Backend adds two auth-gated
endpoints: `GET /api/folder-watchers/<id>/files` (folder listing) and
`GET /api/folder-watchers/<id>/files/<kind>/<filename>` (file preview),
with `os.path.basename` path-traversal defense and a `_VALID_KINDS`
whitelist.

### T14. F3 Normalization method choice
**Status:** ✅ Done | **Commits:** `b110fe3` | **Shipped:** 2026-07-04
User can now pick `min_max` / `z_score` / `robust` / `none` at the
Windowing step. Method + fitted params travel with the SavedModel and
apply identically at every inference/export site: App Builder runner,
ME-LAB endpoint, pipeline replay (ML+DL), deployer's generated Python
inference scripts, CLAW exporter validation and manifest. Legacy
`min_max` models keep predicting byte-identically.

Verified via 10-subtest docker-exec suite — all green. Wired through
8 files across 4 layers (data_loader, app_builder, pipeline_replay,
deployer, cira_claw_exporter, plus store + Windowing UI). Method
literals `min_max`, `z_score`, `robust`, `none` consistent across every
layer.

**In-scope follow-ups NOT taken (see Discovered follow-ups below):**
App Builder Normalize NODE UI literal naming inconsistency (`minmax`
vs `min_max`), and a pre-existing `NoneType.get` in the CLAW exporter.

### T1. TI MCU export failed for ONNX-native (TI NN) saved models
**Status:** ✅ Done | **Commits:** `d3c39b8` `709915d` `3113966`
The export-saved endpoint assumed every saved model was a sklearn pickle;
it pickle-loaded ONNX bytes and crashed. Added format detection (0x08 =
ONNX protobuf tag, 0x80 = pickle PROTO opcode) and a dedicated ONNX
packager that emits the same CCS-ready zip. Both "Download Package" (on
model card, `/api/deployment/package/<id>`) and "TI MCU Package" (Step
1/Step 2 flow, `/api/ti/export-saved/<id>`) now converge on the same 5-file
zip: `model.onnx`, `cira_main.c`, `cira_serial_test.py`, `model_info.json`,
`README.txt`. CCS templates are fetched at package-build time from a new
`/ccs-templates` endpoint on the TI container.

Also fixed a related `NoneType.get` crash — `dict.get(key, default)`
returns None (not default) when the value stored at key is None, and TI NN
models persist `pipeline_config.feature_extraction = null` explicitly.

### T2. Deploy view actionable error for session model
**Status:** ✅ Done | **Commits:** `1fd2b9c`
The TI MCU / CLAW export paths need `selectedSavedModelId`, but the session
model has no ID yet. Was surfacing an opaque "TI MCU export requires a
saved model" toast. Replaced with an actionable message pointing at the
"Save Model as Benchmark" button on the Training page + an inline amber
hint on the TI MCU Package radio.

### T3. Multi-CSV selection sent the whole parent folder to TI
**Status:** ✅ Done | **Commits:** `4d79785` `6dde2af`
`load_csv_multiple` stashed `os.path.dirname(file_paths[0])` as
`metadata.file_path`. Selecting 2 files sent the entire parent folder to
downstream consumers. Now creates a per-session directory of file **copies**
at `datasets/.multi_csv_selections/<session_id>/`. Copies not symlinks
because backend and TI containers mount `datasets/` at different container
paths (`/app/datasets` vs `/app/data/datasets`), so absolute-path symlinks
written from backend break inside TI.

### T4. TI /train guardrails against huge folders + concurrency lock
**Status:** ✅ Done | **Commits:** `16ef17a`
Selecting a broadly-scoped folder like `datasets/shared/` swept up 70+
unrelated CSVs (some 600k+ rows), allocated multi-GB of memory, blocked
Flask's dev server, and got the container restarted by Docker mid-request.
Added: (a) fail-fast on >30 CSVs or >200 MB total, (b) non-blocking
threading.Lock on `/train` so a UI retry gets an immediate 429 instead of
starting a competing subprocess.

### T5. TI CSV directory glob priority
**Status:** ✅ Done | **Commits:** `0a59fd4`
Recursive `**/*.csv` was always tried first, so `datasets/shared/` swept
in nested subfolders too. Now: top-level `*.csv` first, recurse only as
fallback.

### T6. TI classification training — full pipeline
**Status:** ✅ Done | **Commits:** `7abea7f` `4a57ce2` `4a274bf` `e120dac`
`7e50c5d` `86a8762`
Seven iterations. Final state:
- Per-class `train/val/test/<class>/` layout under `input_data_path/classes/`
- Annotation index files `annotations/instances_{train,val,test}_list.txt`
- Handles multi-CSV directory input (Edge Impulse convention: filename stem
  becomes label if no label column present)
- Ensures test split ≥ 1 file per class when n_chunks ≥ 3
- Errors clearly on single-class datasets
- Metric parser recognises TI's actual log format (`Acc@1`, `F1-Score`,
  `AUC ROC Score`) — was looking for `Accuracy N` which TI never emits.

Verified end-to-end on Edge Impulse motion dataset (4 classes): training
completes with real accuracy/F1/ROC-AUC and a model.onnx artifact.

### T7. Explicit user confirmation before skipping Feature Extract for DL/TI
**Status:** ✅ Done | **Commits:** `a5fc476`
The earlier "silent auto-skip FE for DL" fix was invisible to the user.
Added:
- Main pipeline Windowing view: new "Skip Features & Go to Training"
  button + confirmation dialog. Auto-switches trainingApproach to `dl` if
  the current choice was `ml`/`custom`.
- App Builder editor: when the user switches an endpoint to a DL model
  and a `transform.feature_extract` node exists, prompt to remove it or
  keep it. Backend runner-skip retained as safety net either way.

### T8. TimesNet unusable in App Builder pipeline
**Status:** ✅ Done | **Commits:** `680ceda`
`is_dl` flag now derived from saved model's algorithm (starts with
"timesnet") or `pipeline_config.training_approach == 'dl'`, surfaced in
`/api/melab/endpoints`. Backend runner skips `transform.feature_extract`
for DL endpoints, mirroring the raw-mode skip. Frontend validation exempts
DL from the "needs Feature Extract upstream" error.

### T9. App Builder feature-extraction mismatch (0 predictions symptom)
**Status:** ✅ Done | **Commits:** `c0e51a8`
`_apply_feature_extraction` silently fell through when its preconditions
failed and substituted zeros for missing features → model got junk data.
Now logs each fallback reason explicitly and raises `ValueError` with
actionable info when all requested features are missing.

### T10. Multi-model label decoding for accuracy metrics
**Status:** ✅ Done | **Commits:** `a96a994`
App Builder pipeline discarded the pre-fetch's `dataset_labels` and broke
out of the local endpoint scan on the first hit. Multi-model apps with
Edge Impulse integer targets showed 0% accuracy because raw_target stayed
integer while predictions were strings. Now iterates all endpoints and
only breaks once both `target_col` and `dataset_labels` are collected.

### T11. Table View row limit → configurable
**Status:** ✅ Done | **Commits:** `a089f8a`
The `output.table` node's `max_rows` config was declared in the schema
but ignored by `PublishedAppView.vue`, which hardcoded `.slice(0, 100)` in
four places. Now derived from the node config with 100 fallback. Schema
default bumped 50 → 100 so existing apps don't regress.

### T12. TI REGR 400 BAD REQUEST for files outside shared/
**Status:** ✅ Done | **Commits:** `4ab3701`
Path-remap in `backend/app/routes/ti_tinyml.py` became stale after commit
`1bce223` widened the volume mount from `/app/datasets/shared` to
`/app/datasets`. Fixed the remap and added error surfacing so TI container
errors reach the customer instead of "400 BAD REQUEST for url: ...".

### T13. Housekeeping — .gitignore
**Status:** ✅ Done | **Commits:** `7ccc163`
Replaced ~90 stale per-file `shared/*.csv` entries with blanket
`datasets/` and `shared/` ignores. Cut 163 pending changes → 4. Also
removed the previously-tracked deleted files under `shared/` from git
history.

---

## VERIFIED — already fixed by prior commits (no work needed this round)

Discovered during this round while reading code, then confirmed done.

- **Multi-model endpoint selector filter by mode** — `2eb16d8`
- **Line Chart node Target Column config** — `840fb91` (auto-fill),
  `c79adc2` (pipeline runner honors it)
- **Multi-file selection at Data Source (Select All / Scan Dataset)** —
  `7345f5e` "Multi-model classification timeline chart + Select All for
  CSV files". Frontend has "Select All (N)" button
  [DataSourceView.vue:100](../frontend/src/views/pipeline/DataSourceView.vue#L100)
  and "Scan Dataset" for CBOR folders with training/testing subdirs
  [DataSourceView.vue:167](../frontend/src/views/pipeline/DataSourceView.vue#L167).
  Documented in USER_MANUAL §4.1. Customer's original request in the
  2026-06 email said "อาจารย์แจ้งว่าได้ดำเนินการแล้ว" ("teacher says
  already done") — confirmed.
- **App Builder Normalize node zscore support** — part of the initial
  App Builder implementation. Runner implements minmax + zscore; UI
  schema also exposes `robust` but the runner treats it as zscore.
  This partial coverage is why F3 is 🔧 partial not ✅ done.

## Discovered follow-ups — not yet fixed

Small items surfaced during other work. Fixes are known, scope is
trivial, but they were out of scope for the session that discovered
them. Tracked in their own file:

**→ [docs/FOLLOW-UPS.md](./FOLLOW-UPS.md)**

Currently 3 open (D1 normalize UI literals, D2 CLAW exporter NoneType,
D3 stale `./shared` admin UI hint). Total effort: ~40 minutes.

---

## Known future work (not customer-requested)

Larger initiatives not blocked, not queued — parked so we don't lose
the context. Reassess at planning cycles.

**→ [docs/FUTURE-WORK.md](./FUTURE-WORK.md)**

Currently 3 items: FW1 TCN for MCU, FW2 Web Serial API for browser
flashing, FW3 ONNX Runtime Web (WASM) for browser inference.

---

## Design decisions log

Design questions the team decided internally (rather than pushing back
to the customer) — recorded here so the "why" survives context
compression and next session doesn't relitigate.

### 2026-07-04 — F1 / F2 / F4 unblocked as a batch (11 decisions)

Rather than block on customer email, we made all 11 design choices
internally as the "smallest v1 that ships useful, doesn't paint us into
a corner". The full recommendations were reviewed and approved verbatim.
Individual decisions are folded into each feature's OPEN ITEMS entry
above under the **Decided design** table. Summary of the pattern:

- **F1 Folder Watcher (6 decisions):** single model per watcher, auto-
  detected headered/headerless with user override, container-side path
  only (no code for SMB/NFS), file-only output (no DB audit),
  per-user watchers, Raw Mode only for v1.
- **F2 Multi-dataset Wizard (2 decisions):** reject mismatched schemas
  with clear error (no silent auto-align), cell shows confidence +
  predicted label with per-class breakdown on hover.
- **F4 Project Status view (3 decisions):** aggregated Deploy badge
  with hover breakdown, strict per-user visibility (admin sees all),
  one project = one dataset (swap → clone).

The customer can override any of these later if they surface a real
need — we haven't foreclosed any bigger version.

---

## PROCESS — how to keep this file current

At the end of any session that touched customer-visible behavior:

1. If a new customer request arrived → add it under **OPEN ITEMS** with
   type (feature/bug), status, source date, and summary. If it's a big
   feature, also write a separate `docs/PLAN_*.md` for the design.
2. If work was completed → move the item into **DONE — this round** with
   commit SHAs and a 3-5 sentence root-cause + fix summary. Include what
   was verified (curl / browser / etc).
3. If a customer question got answered → move the question from **OPEN
   QUESTIONS** into the relevant feature's design notes.
4. If a fix broke something and got reverted → move the item back to
   **OPEN ITEMS** with a note explaining what broke.
5. Commit the change to the tracker with a message like
   `Tracker: mark T## done` or `Tracker: add F## from 2026-XX-XX email`.

At the start of any session touching customer work:

1. Read this file first.
2. Then read the specific `docs/PLAN_*.md` if working on a large feature.
3. Cross-check `git log` in case commits landed since the file was last
   updated.

Related files:
- `docs/PLAN_customer_feedback_2026-06.md` — original June planning
  document (still authoritative for F4 architecture).
- `docs/USER_MANUAL.md` — user-facing docs, source of truth for shipped
  behavior.
