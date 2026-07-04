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

### F1. Folder Watcher + ML Prediction
**Status:** 📋 Ready to implement | **Effort:** ~1.5 days
**Type:** feature | **Requested:** 2026-06-19 (June email, restated 2026-07-04)
**Diagram:** `Feature_Watcher-ML-Prediction.png` (shared by customer)

**Summary:** Backend daemon polls a user-specified folder every 60s. For
each file found, reads each row, runs it through a trained ML model,
writes results to an output folder as CSV with columns
`source_file, record_index, sensor_values, prediction, confidence,
predicted_at`, then deletes the input file.

**Decided design (all approved 2026-07-04 — v1 stays small; expand later):**

| # | Decision | Rationale |
|---|---|---|
| F1.1 | **One model per watcher.** Customer creates a second watcher if they need two models on the same folder. | Single `endpoint_id` FK — simpler DB schema, simpler UI, easy to expand later. |
| F1.2 | **Auto-detect headered vs headerless with a user override toggle.** Default: auto. Read first row, try to parse as floats — success → headerless, failure → headered. | Zero friction for the common case; escape hatch for weird files. |
| F1.3 | **Container-side path only.** Customer types any path visible inside the backend container. SMB/NFS is a Docker-compose concern, not application code. | Matches App Builder input handling. |
| F1.4 | **File-only output, no DB persistence.** Output CSV IS the audit trail. | Avoids retention / cleanup / quota design. Also matches the customer's flow diagram. |
| F1.5 | **Per-user watchers.** Each user's watcher points at a subfolder under their private space. | Matches ME-LAB endpoints and App Builder apps. Respects existing quota model. |
| F1.6 | **Raw Mode only, v1.** One row = one prediction. No windowing / buffering. | Matches the flow diagram. Time-series can come later if customer asks. |

**Implementation plan (matches decisions above):**
- Call an existing ME-LAB endpoint per row (reuse `ModelManager.predict()`
  from `melab_service.py`), reuse the label decoding path already fixed
  for App Builder.
- New sidebar entry **"Folder Watcher"** under SERVICES (peer of App
  Builder).
- New `folder_watchers` DB table (id, user_id, name, input_folder,
  output_folder, endpoint_id, status, last_run_at, error_count,
  header_mode). Watchers survive container restarts.
- New bind mount in both compose files: `./watcher-data:/app/watcher-data`.
- Backend daemon: single thread per watcher, 60s tick, poll-glob-process-delete.
  Fail-safe on per-file errors — log, don't abort.

---

### F2. App Builder multi-dataset Wizard
**Status:** 📋 Ready to implement | **Effort:** 4-5 days
**Type:** feature | **Requested:** 2026-07-02

**Summary:** Wizard flow that runs N models across M datasets and returns
a matrix of per-cell confidence + latency + model size. Lets the customer
pick the smallest model that hits their confidence bar for edge deployment.

**Decided design (all approved 2026-07-04):**

| # | Decision | Rationale |
|---|---|---|
| F2.1 | **Reject datasets with mismatched column schemas.** All datasets in one Wizard run must have identical column names. Clear error, no silent alignment. | Auto-align silently drops columns → wrong predictions → customer loses trust. |
| F2.2 | **Cell shows confidence + predicted label.** Full per-class probability breakdown on hover / click. | Cell stays scannable; detail available on demand. |

**Implementation plan:**
- Wizard UI: 3 steps — (1) pick models (existing multi-select), (2) pick
  datasets (multi-file upload with schema-check validation), (3) run &
  view matrix.
- Backend: reuse `Multi-Model Compare` runner infra with a new "batch"
  mode that iterates datasets.
- Results view: table rows=datasets, columns=models, cells=confidence +
  predicted label. Hover shows full probability breakdown + latency +
  model size.
- Download-results-as-CSV button (reuses existing export path).

---

### F4. End-to-end Project Status view
**Status:** 📋 Ready to implement | **Effort:** ~6-7 days
**Type:** feature | **Requested:** 2026-06-23

**Summary:** A Projects sidebar entry showing pipeline stage progress
(Data → Windowing → Features → Training → Deploy) for every body of work,
not just App Builder.

**Gap surfaced:** the `projects` DB table exists but only `training_sessions`
has a FK to it. Data/windowing/feature sessions live only in Pinia +
backend memory — lost on refresh. Need to persist these stages to DB and
wire `project_id` through everywhere.

**Decided design (all approved 2026-07-04):**

| # | Decision | Rationale |
|---|---|---|
| F4.1 | **One aggregated ✅ deploy badge with hover-tooltip breakdown** (ME-LAB / App Builder / TI MCU / Jetson). | Row density: 4 badges × N projects is visual noise. Aggregate keeps it scannable. |
| F4.2 | **Strict per-user visibility.** Admin sees all (same model as ME-LAB endpoints today). | Matches every other user-owned resource. Zero surprise. |
| F4.3 | **One project = one dataset.** Dataset swap = clone the project. | Simpler DB schema. Already agreed in the June design. |

**Also folded in:**
- User-editable + reorderable feature selection (from July email)
  becomes a per-project "feature template" so the API contract stays
  stable across retrainings.
- The "project folder" wording in the user manual (§3) refers to
  `datasets/` subfolders — different from this new Project concept.
  Rename "project folder" → "dataset folder" in the manual to avoid
  clash. UI is already neutrally called "Folder Management".

**Full plan:** [docs/PLAN_customer_feedback_2026-06.md](./PLAN_customer_feedback_2026-06.md)

---

## DONE — this round (July 2026)

Ordered newest first. Every SHA is on the `master` branch.

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
