# Discovered Follow-ups

Small items surfaced during other work — fixes are known, scope is
trivial, but they were out of scope for the session that discovered
them. Chip away when there's a quiet moment between customer requests.

Each entry links back to the session that found it. When you fix one,
move it to the "Done" section at the bottom with the commit SHA.

## Open

### D1. App Builder Normalize node UI uses old literal names
**Discovered:** 2026-07-04 (during F3 QA) | **Effort:** ~30 min
**Files:**
- [frontend/src/views/AppBuilderEditorView.vue:838](../frontend/src/views/AppBuilderEditorView.vue#L838)
- [backend/app/routes/app_builder.py](../backend/app/routes/app_builder.py) → `_apply_normalization`

The App Builder `transform.normalize` node's `configSchema.options` uses
`minmax`/`zscore`/`robust` (no underscores). F3 introduced training-time
literals `min_max`/`z_score`/`robust`/`none` and the runner accepts BOTH
naming conventions for backwards-compat, so there's no functional bug —
but the UX is inconsistent and the `none` option is missing from the
node UI.

**Fix:** align the schema values to `min_max`/`z_score`/`robust`/`none`
and add `none`. Verify no published app has the old string values hardcoded
in the node config before flipping. If any exist in the DB, add a small
migration or accept both literal names in the runner permanently.

---

### D2. `cira_claw_exporter.py:196` NoneType.get on null feature_selection
**Discovered:** 2026-07-04 (during F3 QA) | **Effort:** ~5 min
**File:** [backend/app/services/cira_claw_exporter.py:196](../backend/app/services/cira_claw_exporter.py#L196)

Same pattern as the fix in `deployer.py` back on 2026-07-02 (commit
`d3c39b8`): the row has `pipeline_config.feature_selection: null` in
SQLite, so `saved.get('feature_selection', {})` returns `None`, and the
subsequent `.get('selected_features', [])` raises
`'NoneType' object has no attribute 'get'`.

**Fix:** replace `sel = pipeline_config.get('feature_selection', {})`
with `sel = pipeline_config.get('feature_selection') or {}`.

Occurs on any TI NN or DL model export via CLAW.

---

### D3. `admin.py:451,458` stale `./shared` UI hint
**Discovered:** 2026-06 (during TI REGR QA) | **Effort:** ~5 min
**File:** [backend/app/routes/admin.py:451,458](../backend/app/routes/admin.py#L451)

Cosmetic — comment and the `host_hint` string in the admin Storage panel
still say `./shared` after the May 2026 migration that moved everything
to `./datasets/`. The actual path resolution uses `DATASETS_ROOT_PATH`
correctly. Only the UI string misleads admins about where data lives on
disk.

**Fix:** replace `./shared` with `./datasets/shared` (or
`./datasets/<user_folder>` for per-user context) in the two strings.

## Done

_(none yet — move entries here with commit SHA when you fix them.)_
