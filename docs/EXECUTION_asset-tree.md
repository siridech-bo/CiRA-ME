# Execution plan — Asset Tree, Machine Groups, Continuous Ingest

**Design plan** (source of truth for WHAT we're building): [PLAN_2026-07-18_asset-tree.md](PLAN_2026-07-18_asset-tree.md)
**Live progress dashboard**: [asset-tree-traces.html](asset-tree-traces.html) — updated at every subtask transition.

This document is the HOW. Each phase decomposes into concrete subtasks with acceptance criteria, QA hooks, and hand-off gates.

Confirmed answers to the four open questions from the design plan:
1. **Migration UX**: quiet indicator (small chip in the top bar, no nagging modal).
2. **Sensor unit taxonomy**: dropdown presets + "custom" free-text option.
3. **Node lifecycle**: retire-only, never destructive delete.
4. **Group visibility**: all users see all groups in MVP; per-user ACL later.

## Workflow (applies to every phase)

1. **Implement** — coder agent dispatched, or hand-implemented for tiny phases (0).
2. **Adversarial QA** — QA agent verifies against acceptance criteria + hunts for defects.
3. **Personal browser test** — Claude drives the actual UI and confirms behavior.
4. **Traces updated** — asset-tree-traces.html reflects each subtask transition.
5. **Commit + push** — clean commit per phase, one per phase generally.
6. **Hand-off note to user** — summary + test-in-browser instructions.
7. **User signs off** → next phase kicks off. Otherwise fix + re-QA.

---

## Phase 0 — Dark/Light mode toggle (~2 hours)

Standalone customer request. Warm-up before the big work.

### Subtasks

| ID | Task | Deliverable |
|----|------|-------------|
| 0.1 | Add composable `useThemePref()` that wraps Vuetify's `useTheme()` + syncs to localStorage under key `cira.theme` | New file `frontend/src/composables/useThemePref.ts` |
| 0.2 | Toggle button in the top app bar (sun/moon icon, tooltip) | `App.vue` (or wherever the top bar lives) header edit |
| 0.3 | Boot-time apply — read localStorage in main.ts before mount so first paint matches saved theme | `main.ts` |
| 0.4 | Audit hardcoded dark styles — grep for `background: #0` / `color: #fff` etc. and replace with Vuetify theme colors OR add light-mode fallback | Diffs across `.vue` files that hardcode |

### Acceptance criteria

- Toggle click → theme flips within one frame. No jank.
- Reload preserves the last-chosen theme.
- No console errors in either theme.
- Both themes readable on: Dashboard, Data Source, Pipeline steps, App Builder, Folder Watcher, Published App view.
- Default remains dark (no surprise for existing users on first load).

### QA subtasks (dispatched agent)

- **0.QA.1** Verify localStorage key + value pattern.
- **0.QA.2** Visit every top-level route in both themes; capture 6 pairs of screenshots.
- **0.QA.3** Contrast check: any hardcoded near-black text on newly-white surfaces?
- **0.QA.4** Toggle rapidly 10× — no memory leak or state desync.

### Personal browser tests

- Toggle. Reload. Toggle. Reload. Navigate all sections.
- Fresh incognito → default = dark.

### Hand-off signal

Traces `phase-0` marked complete. Commit pushed. Screenshot pair posted.

---

## Phase A — Data model + first-run wizard + basic tree admin (~2 weeks)

The foundation. All later phases depend on this. Ships without breaking existing pipeline (legacy compat).

### A.1 Backend — schema + migrations
- `backend/app/models.py` — add `AssetTreeConfig`, `AssetNode`, `AssetSensorMeta`, `MachineGroup`, `MachineGroupMember`, `ModelMachineBinding`, `AssetTreeAudit` (matching design plan §3).
- Idempotent `ALTER TABLE` / `CREATE TABLE IF NOT EXISTS` migrations.
- Existing tables untouched.

### A.2 Backend — endpoints (tree CRUD)
- `GET /api/asset-tree/config` — returns level names, root name, topic mode, meta prefixes.
- `PUT /api/asset-tree/config` — upsert single config row.
- `GET /api/asset-tree/nodes` — full tree as nested JSON.
- `POST /api/asset-tree/nodes` — add a node (validates parent, name uniqueness).
- `PATCH /api/asset-tree/nodes/<id>` — rename / update metadata.
- `POST /api/asset-tree/nodes/<id>/retire` — soft-delete with audit entry.
- `POST /api/asset-tree/nodes/<id>/move` — reparent within same depth.
- `GET /api/asset-tree/topics/test` — accepts `?topic=…`, returns which node it routes to.
- `POST /api/asset-tree/import` — accept YAML/JSON spec, build tree.
- `GET /api/asset-tree/audit` — paginated audit log.

### A.3 Backend — endpoints (groups CRUD)
- `GET / POST / PATCH / DELETE /api/asset-tree/groups`
- Group membership add/remove.
- Compatibility validator: `POST /api/asset-tree/validate-compatibility` with a list of machine IDs → returns pass/fail + per-machine diff.

### A.4 Backend — sensor metadata presets
- Serve a canonical list from a Python constant: `Vibration monitor`, `Thermal`, `Rotating machinery`, plus common units + sample rates.

### A.5 Frontend — first-run wizard shell
- New route `/setup/asset-tree` (only accessible when config row missing).
- Route guard: any authenticated user hitting `/` with missing config → redirect to `/setup/asset-tree`.
- 5-step stepper skeleton (Vuetify `v-stepper`).

### A.6 Frontend — wizard steps
- **Step 1 (preset)** — card grid, 5 options.
- **Step 2 (level names)** — editable list, add/remove level, live topic preview.
- **Step 3 (tree builder)** — two-pane; recursive tree component; node detail form; skip-for-later option.
- **Step 4 (MQTT rules)** — strict/learn radio, meta prefixes editor, topic test widget.
- **Step 5 (confirm)** — summary + finish button.

### A.7 Frontend — ongoing admin
- New route `/settings/asset-tree` reusing the wizard's tree builder + detail form.
- Extras: audit log tab, retire button, move-subtree affordance (deferred to Phase B if tight).

### A.8 Frontend — legacy compat surfacing
- On first wizard save, backend auto-creates a `_legacy` synthetic machine per existing project.
- Home page (or Dashboard) shows a small quiet chip: *"3 legacy projects available in _legacy"*.

### QA subtasks

- **A.QA.1** Schema round-trip: create a full tree via API, restart backend, verify tree is intact and audit log preserved.
- **A.QA.2** Topic-test endpoint against 20 handcrafted topics (valid + wildcarded + rejected + meta).
- **A.QA.3** Compatibility validator across mismatched machine sets.
- **A.QA.4** Wizard end-to-end in browser: fresh install → preset → level rename → build 2 plants × 3 machines × 3 sensors each → strict rules → confirm.
- **A.QA.5** Post-wizard, hit `/setup/asset-tree` again — should redirect to `/settings/asset-tree` (wizard is one-shot).
- **A.QA.6** Legacy chip appears when at least one project exists.

### Hand-off signal

Fresh install can complete the wizard and end up with a persistent tree + browsable admin view. Traces `phase-A` complete.

---

## Phase B — Sidebar restructure + machine workspace (~1 week)

Turns the pipeline pages into tab content within a machine workspace. Existing routes redirect.

### Subtasks

- **B.1** New sidebar layout (tree + global tools + settings). Search box + auto-collapse.
- **B.2** Machine workspace route `/machine/:id` with 6 tabs.
- **B.3** Overview tab: live sensor tiles (subscribe via MQTT if available), recent activity feed.
- **B.4** Data tab: File Manager scoped to machine folder.
- **B.5** Models tab: table of models attached to this machine.
- **B.6** Deploy tab: App Builder apps scoped to machine.
- **B.7** Labels tab: placeholder card ("Coming in Phase E").
- **B.8** History tab: long-timeline chart with sensor dropdown + prediction overlay.
- **B.9** Route redirects: `/pipeline/data`, `/pipeline/windowing`, etc. → prompt user to pick a machine first, or redirect if a "current machine" is set in session.

### QA subtasks

- **B.QA.1** Navigation flow: sidebar → pick machine → workspace loads, breadcrumb correct.
- **B.QA.2** All 6 tabs render without errors even on a machine with no data.
- **B.QA.3** Legacy path (`/pipeline/*`) surfaces the right message and doesn't 404.
- **B.QA.4** Sidebar search filters the tree correctly.

### Hand-off signal

End-to-end pipeline (Data → Windowing → Features → Training → Deploy) works when scoped to a real machine node. Traces `phase-B` complete.

---

## Phase C — Machine Groups + cross-machine training (~1 week)

### Subtasks

- **C.1** `Settings → Machine Groups` page (table + create/edit modal with mini-tree picker).
- **C.2** Training view — scope selector at top (Just this / Group / Ad-hoc).
- **C.3** Compatibility validation UI (uses the endpoint from A.3).
- **C.4** Model artifact metadata — training records `trained_on_machines`, `trained_via_group`, `deploy_targets`.
- **C.5** Model detail view — Rebind machines dialog.

### QA subtasks

- **C.QA.1** Create a group with 3 machines, train, verify metadata persisted.
- **C.QA.2** Try training with mismatched sensors — verify clear error blocks the run.
- **C.QA.3** Retired machine stays in the group with a "retired" flag but is excluded from new training runs.
- **C.QA.4** Rebind: pick 1 of 3 trained-on machines as sole deploy target, verify only that machine can bind.

### Hand-off signal

A model trained on 12 machines can be deployed to a subset. Traces `phase-C` complete.

---

## Phase D — MQTT ingest router + workshop-scale defenses (~1 week)

### Subtasks

- **D.1** New service `backend/app/services/mqtt_ingest_router.py`. Subscribes to `#`, routes to tree.
- **D.2** Rolling daily CSV writer. Rotation cutoff midnight UTC.
- **D.3** Rejected-topic log + viewer UI under `Settings → MQTT Rules`.
- **D.4** Config UI: enable, rotation policy, storage format, retention.
- **D.5** Retention janitor: nightly job, respects `Settings → MQTT Rules → keep for N days`.
- **D.6** Companion: publisher rate-limit clamp at 5 msg/s in `MqttTestPublisher.vue` (workshop safety). Topic namespace docs.

### QA subtasks

- **D.QA.1** Publish a known payload → file appears in the expected tree folder within 1 s.
- **D.QA.2** Publish an unknown topic in strict mode → rejected + logged.
- **D.QA.3** Publish an unknown topic in learn mode → node auto-created.
- **D.QA.4** Publish across a midnight boundary → new file created cleanly.
- **D.QA.5** Retention: force clock forward N+1 days, run janitor, verify old files gone.

### Hand-off signal

Traces `phase-D` complete. Files accumulate under the tree; browsable via Data tab.

---

## Phase E — OculusT labels integration (~1-2 weeks)

Depends on Docker network + auth alignment.

### Subtasks

- **E.1** Docker Compose extension so OculusT can read `data/factory/...` from CiRA ME's mount.
- **E.2** Shared session cookie or SSO handoff (a token URL that OculusT accepts).
- **E.3** Labels tab in machine workspace: "Open labeling" button links to OculusT prefilled with the machine's dataset paths.
- **E.4** OculusT-produced labeled CSVs written back to `data/factory/{plant}/{machine}/labels/{campaign}/`.
- **E.5** Windowing view: auto-detect labeled versions and offer to use them.

### QA subtasks

- **E.QA.1** Click "Open labeling" → OculusT loads with the correct dataset.
- **E.QA.2** Label a range, save, close → labeled CSV appears in the expected path.
- **E.QA.3** Windowing detects and offers to switch.
- **E.QA.4** Auth handoff — no double login.

### Hand-off signal

Full loop end to end: ingest → label → train → deploy from a machine workspace. Traces `phase-E` complete → entire plan done.

---

## Rollback / abort rules

- Any phase can be **paused mid-way**. The traces file records where we stopped.
- **Never break existing customer flows**. Legacy pipeline stays reachable throughout A–D. E is the point of no return for pre-tree workflows.
- **Roll back a phase**: `git revert <phase-commit>` + traces update to `blocked`.

## Sequencing options

Default: **0 → A → B → C → D → E** in order.

Alternative if MOMENT lands mid-project: **A → B → MOMENT Phase A** (see [PLAN_2026-07-XX_moment-foundation-model.md](PLAN_2026-07-XX_moment-foundation-model.md)) → **C → D → E**. Foundation model plugs into cross-machine training naturally.

Decision deferred until Phase A ships.
