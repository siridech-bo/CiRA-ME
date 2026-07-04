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
**Status:** ⏸ blocked on customer answers (see Open Questions below)
**Type:** feature | **Requested:** 2026-06-19 (June email, restated 2026-07-04)
**Diagram:** `Feature_Watcher-ML-Prediction.png` (shared by customer)

**Summary:** Backend daemon polls `/input` folder every 60s. For each file
found, reads each row, runs it through a trained ML model, writes results
to `/output` as CSV with columns `source_file, record_index, sensor_values,
prediction, confidence, predicted_at`, then deletes the input file.

**Agreed design:**
- Implement as **Raw Mode classification** calling an existing ME-LAB
  endpoint — each input row is one feature vector already. No new pipeline.
- New sidebar entry **"Folder Watcher"** under SERVICES, peer of App Builder.
- Watcher daemon lives inside the backend container. State persisted to a
  new `folder_watchers` DB table so watchers survive container restarts.
- Reuses `ModelManager.predict()` from `melab_service.py`. Reuses the label
  decoding path already fixed for App Builder.
- New bind mount in both compose files: `./watcher-data:/app/watcher-data`.
- **Effort:** ~1.5 days once open questions are answered.

**Open questions (6):** in the Open Questions section at the bottom.

---

### F2. App Builder multi-dataset Wizard
**Status:** 📋 Open
**Type:** feature | **Requested:** 2026-07-02

**Summary:** Wizard flow that runs an AI model across many datasets and
returns per-dataset confidence, so the user can pick the smallest model
that hits their confidence bar for edge deployment.

**Agreed design:** Full matrix — N models × M datasets. Results table has
rows=datasets, columns=models, cells=confidence + latency estimate + model
size. Reuses the existing Multi-Model Compare backend; only the results view
is new.

**Effort:** 4-5 days.

**Open questions:** in the Open Questions section.

---

### F3. Normalization method choice
**Status:** 📋 Open
**Type:** feature | **Requested:** 2026-07-02

**Summary:** Currently min-max normalization is hardcoded. Let the user
pick min-max / z-score / robust / none at pipeline setup time.

**Non-obvious constraint:** the chosen method AND its fitted parameters
(min/max, mean/std, median/IQR) must be persisted with the SavedModel and
applied identically at every inference site — ME-LAB endpoint, App Builder
runner, live MQTT, TI export. A silent mismatch produces garbage
predictions. Audit whether the current min-max params are already
end-to-end persisted before touching this.

**Effort:** 1-2 days if persistence layer is already there, ~2 if not.

---

### F4. End-to-end Project Status view
**Status:** 📋 Open (large — ~6-7 days)
**Type:** feature | **Requested:** 2026-06-23

**Summary:** A Projects sidebar entry showing pipeline stage progress
(Data → Windowing → Features → Training → Deploy) for every body of work,
not just App Builder.

**Gap surfaced:** the `projects` DB table exists but only `training_sessions`
has a FK to it. Data/windowing/feature sessions live only in Pinia +
backend memory — lost on refresh. Need to persist these stages to DB and
wire project_id through everywhere.

**Design decisions already made:**
- One project = one dataset (swap = clone the project).
- User-editable + reorderable feature selection (July item, folded in here)
  becomes a per-project "feature template" so the API contract is stable
  across retrainings.
- The "project folder" wording in the user manual (§3) refers to
  `datasets/` subfolders — different from this new Project concept.
  Rename "project folder" → "dataset folder" in the manual to avoid clash.
  UI is already neutrally called "Folder Management".

**Full plan:** [docs/PLAN_customer_feedback_2026-06.md](./PLAN_customer_feedback_2026-06.md)

**Open questions:** in the Open Questions section.

---

## DONE — this round (July 2026)

Ordered newest first. Every SHA is on the `master` branch.

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
  documented in USER_MANUAL §4.1; confirmed present in code

---

## OPEN QUESTIONS — waiting on customer

Grouped by feature. Answer these to unblock implementation.

### For F1 (Folder Watcher)
1. **One watcher = one model, or multi-model per watcher?** Affects DB
   schema (one endpoint_id vs many).
2. **Input files headered or headerless?** Default value of the toggle.
3. **Network share or local folder?** If SMB, who handles the Windows
   mount into Docker? Biggest deployment risk.
4. **Persist predictions to DB (audit trail) or output file only?**
5. **Per-user folders or one shared system folder?**
6. **Time-series files (multi-row → 1 prediction with windowing) or Raw
   Mode only (1 row → 1 prediction)?**

### For F2 (Multi-dataset Wizard)
7. **Datasets with different column schemas** — reject as error, or
   attempt auto-alignment?
8. **Cell contents** — just confidence, or also predicted label and
   per-class probability breakdown?

### For F4 (Project Status)
9. **Deploy status** — 4 separate badges (ME-LAB, App Builder, TI MCU,
   Jetson SSH) or one aggregated ✅?
10. **Cross-user project visibility** — admin sees all, everyone sees
    everyone's read-only, or strict per-user?

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
