# Phase G — In-app labeler + simulator profile-swap

**Date:** 2026-07-22
**Effort:** ~2.5 days
**Depends on:** Phase F (Machine Simulator) for Q1; existing Data Source pipeline for Q2

---

## Motivation

Two customer-facing gaps came up in a single session:

**Q1** — the Machine Simulator lets you pick any profile with any topic_base independently. User selected `industrial_boiler` profile but pointed it at `air_compressor_01`. Sim now publishes boiler sensors under a machine named "compressor". No in-app fix — user has to delete the sim, retire nodes, recreate. Ugly.

**Q2** — labeling. State transitions ARE audited but not written to sensor CSVs. Users can flip states on the sim (or wait for real machines to fault), but the resulting CSVs have no class labels for training. Options considered:

- OculusT integration (Phase E — deferred as heavier than needed)
- Session-based recording (one class per session) — user rejected: doesn't match "continuous recording, label post-hoc" reality
- In-app timeline labeler on a NEW page — user rejected: reuse the existing DataSourceView chart instead

Ship the in-app labeler on DataSourceView. That's simpler, less new surface, matches how users already look at their data.

## Decisions (locked in)

### Q1

- Backend: on `POST /simulators/`, if the target machine already has active children whose names don't match the profile's sensors → 400 with clear error listing the mismatch
- Backend: new `POST /simulators/<id>/change-profile` endpoint — stops sim → retires old sensor children → autoprovisions new profile's sensors → restarts sim with the new profile at the initial_state passed in (or the current state if it's valid for the new profile)
- Frontend: **Change profile** button on the sim card → dialog with profile picker → confirm

### Q2

| Decision | Value |
|---|---|
| Loading | Batch-100 only, no "Load all" — safer for memory |
| Selection UX | Two vertical lines (start / end) with editable time inputs — not a shaded box |
| Line placement | Click-on-chart to place; drag to move; type in the input for precision |
| Persistence | Sidecar `<csv>.labels.json` next to the CSV |
| Save trigger | **Auto-save on "Load next" button click** — no manual Save |
| Cross-batch labels | Allowed — user can type an end time outside the currently-loaded window; label saves with the absolute time value |
| Class dropdown source | **Free text** — user types whatever class name they want |
| Label rendering | Thin colored bars along the x-axis (not full shaded regions — signal stays visible) |
| Existing labels on load | Sidecar read at page load, labels reappear on the chart wherever their time range overlaps the visible window |
| Editing | Click a label in the labels list → its two lines reappear on the chart → edit → apply |
| Training pipeline hook | Data loader reads sidecar; rows/windows outside any label range → user-configurable: drop OR label "unlabeled" |

## Labels sidecar file format

Written to `data/<topic_path>/<csv_stem>.labels.json` (or wherever the source CSV lives — colocated).

```json
{
  "csv": "2026-07-22.csv",
  "x_column": "timestamp_iso",
  "labels": [
    {"from": 0.0, "to": 45.12, "class": "idle"},
    {"from": 45.12, "to": 60.20, "class": "fault"},
    {"from": 60.20, "to": 99.26, "class": "idle"}
  ],
  "updated_at": "2026-07-22T15:00:00Z",
  "updated_by": "admin"
}
```

`from` / `to` are numeric — same units as the chart's x-axis. If the CSV timestamps are ISO strings, values are the numeric offset in seconds from the first row. Data loader converts back to ISO for row-matching.

## Backend endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| GET | `/api/pipeline/data/labels?csv_path=<path>` | Read sidecar labels (empty array if none) | any |
| PUT | `/api/pipeline/data/labels` | Write sidecar labels (replaces entire array — client sends the full list, server writes atomically) | any authed |
| POST | `/api/simulators/<id>/change-profile` | Stop → retire → autoprovision → restart with new profile | admin |

## Data loader integration

- `data_loader.py`: after loading rows, look for `<csv>.labels.json` sidecar
- If present, for each row compute its class by finding which label range covers its x-value
- Add a `label` column to the loaded DataFrame
- Rows outside any range: `label = None` (or user's configured default)
- Windowing/features/training already know how to consume a labeled DataFrame — no changes there

## UI shape (DataSourceView)

New "Labeling" mode toggle above the chart:

```
[ 🔍 Zoom/Pan ]  [ 🏷 Label mode ]  [ Zoom reset ]
```

In Label mode:
- Click on chart → places start line (green)
- Click again → places end line (red)
- Both lines get an editable time input rendered below the chart
- Drag either line to reposition
- **Apply** button → inline popover with free-text class field → save → range added to labels list
- **Cancel** clears the pending lines

Labels panel below chart:

```
Labels                              [ + Add manually ]
────────────────────────────────────────────────────────
🟢 idle    0.00 → 45.12   (45.12 s)   [edit] [delete]
🔴 fault  45.12 → 60.20   (15.08 s)   [edit] [delete]
🟢 idle   60.20 → 99.26   (39.06 s)   [edit] [delete]
────────────────────────────────────────────────────────
Coverage 100 % · 3 labels · 2 classes  ● unsaved
```

`● unsaved` indicator turns green after auto-save-on-next fires.

Load next flow:
- User clicks "Load more rows"
- If unsaved labels exist: `PUT /labels` fires first (auto-save), then the next batch loads
- Chart clears, then re-renders with:
  - Sensor data from the newly loaded rows
  - Any labels whose range overlaps the loaded window, drawn as bars

## Deliverables (subtasks)

| ID | Owner | Description |
|---|---|---|
| G.1 | backend | Simulator profile-swap endpoint + name-mismatch validation on create |
| G.2 | backend | Labels sidecar read/write endpoints under pipeline/data |
| G.3 | backend | Data loader sidecar integration |
| G.4 | frontend | Change-profile button + dialog on SimulatorCard |
| G.5 | frontend | DataSourceView label mode: vertical lines + inputs + free-text class field |
| G.6 | frontend | Labels panel + auto-save-on-next hook + bar overlays |
| G.QA | agent | Adversarial QA pass |
| G.T | user | Personal browser test |

## Edge cases

- Two overlapping labels on the same range → keep both, or reject? **Reject** — 400 with "range overlaps existing label 'fault' at 45.12–60.20"; user must delete the existing one first
- Cross-batch label where end < start (user typed weird times) → auto-swap
- Label with class = empty string → 400
- Label with `to <= from` → 400
- Sidecar file missing at load → treat as empty, no error
- Sidecar file malformed JSON → log warning, treat as empty (don't crash the data view)
- User navigates away with unsaved labels → `beforeunload` prompt "You have unsaved labels — save now?"
- Machine's topic_path changes (rename in tree) → labels sidecar stays at the old physical file path (correct — labels are per-CSV, not per-tree-node)

## Not doing (v1)

- OculusT integration (Phase E — deferred separately)
- Multi-signal simultaneous labeling — v1 labels apply to the whole row regardless of which sensor is being viewed
- Batch label operations (e.g. "label every window >X threshold as fault") — future
- Undo / redo — future
- Import labels from external CSV — future

## Definition of done

- User can flip through 100-row batches, drop label ranges via vertical lines with typed-precise times, and auto-save on advance
- Labels persist across sessions and reappear on the chart when their range is in view
- Data loader applies labels; Windowing tab shows labeled window count
- User can swap a simulator's profile from the card without deleting/recreating manually
- Zero blockers from QA
- Personal browser test passes end-to-end
