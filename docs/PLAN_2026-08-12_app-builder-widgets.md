# App Builder Widgets — Tier 1 Expansion

**Status:** 🔧 Active — start immediately
**Date:** 2026-08-12
**Owner:** siridech.bo@kmitl.ac.th
**Related tracker item:** to be added under OPEN ITEMS as `F7`

---

## Why this exists

Customer wants App Builder published apps to look more like real
operator dashboards, not just "here's a table of predictions."
Today's output nodes are limited to `output.table`, `output.line_chart`,
`output.alert_badge`, `output.signal_recorder`, `output.multi_model_compare`.
Missing the widget vocabulary factory operators expect:
big numbers, gauges, status lights, buttons that do things, text
sections for context.

## Locked decisions (from the 2026-08-12 discussion)

- **Tier 1 only** — 5 new node types below. Same architectural pattern
  as existing output nodes: JSON schema in `AppBuilderEditorView.vue`,
  render component in `PublishedAppView.vue`, backend runner mostly
  passes through (widgets are frontend-only for rendering).
- **Skip Tier 2 (variable/expression system)** — no `${var_name}`
  references between widgets. Each widget derives its value from
  pipeline output only.
- **Skip Tier 3 (full low-code / drag-and-drop layout)** — out of scope.

## The 5 widgets

### 1. `output.button` — Action Button

Configurable button that triggers a defined action when the operator
clicks it.

**Config schema:**
- `label` (string) — button text
- `icon` (string, optional) — MDI icon name
- `color` (enum) — primary / success / warning / error
- `action` (enum) — `mqtt.publish` | `http.get` | `http.post` | `download.csv` | `download.pdf`
- Action-specific fields based on `action`:
  - `mqtt.publish` → `topic` (string), `payload` (string), `broker_url` (optional, defaults to app broker)
  - `http.get` / `http.post` → `url`, `headers`, `body`
  - `download.csv` / `download.pdf` → `source` (which pipeline output to export), `filename`

**Runtime:** frontend fires the action when clicked. `mqtt.publish` uses the
same MQTT client the published app already opens for live streaming.
`http.*` and `download.*` don't need backend involvement.

**Effort:** 3d (schema + editor UI + frontend action dispatch + tests)

> **Implementation note (2026-08-12):** `download.pdf` was dropped from the
> `action` enum for v1. It needs `jsPDF` (or similar), which is not in
> `frontend/package.json` and would add ~200 KB to the bundle — out of
> scope for a same-session Tier 1 ship without an explicit go-ahead. The
> other four actions (`mqtt.publish`, `http.get`, `http.post`,
> `download.csv`) are fully implemented. Add PDF export as a follow-up if
> the customer asks for it.

### 2. `output.big_number` — Big Number Display

One prominent metric with unit, optional threshold coloring.
The classic "44°C / 92% / 1,247 units" tile.

**Config schema:**
- `label` (string) — small label above the number
- `source_field` (string) — which field from pipeline output to display
- `unit` (string, optional) — appended to the number ("°C", "%", "units/hr")
- `decimal_places` (int, default 1)
- `thresholds` (array, optional) — `[{"below": 20, "color": "info"}, {"below": 80, "color": "success"}, {"above": 80, "color": "warning"}]`
- `size` (enum) — sm / md / lg / xl (visual size)

**Runtime:** subscribes to latest pipeline output value, renders with
threshold-based color. Updates on each new prediction/measurement.

**Effort:** 2d

### 3. `output.gauge` — Circular Gauge

Radial progress gauge, min → max, with optional threshold bands
(green/yellow/red) and center value.

**Config schema:**
- `label` (string)
- `source_field` (string)
- `min` (number, default 0)
- `max` (number, default 100)
- `unit` (string, optional)
- `bands` (array, optional) — `[{"from": 0, "to": 70, "color": "success"}, {"from": 70, "to": 90, "color": "warning"}, {"from": 90, "to": 100, "color": "error"}]`
- `show_needle` (bool, default true)

**Runtime:** SVG-rendered gauge. Vuetify has no built-in — use
`vue-echarts` (already imported for line chart) with the gauge series.

**Effort:** 3d (mostly ECharts gauge config)

### 4. `output.status_indicator` — Status Light

Colored circle + label. Driven by a boolean or category from pipeline output.
"Machine A — Running (green)", "Machine B — Fault (red)".

**Config schema:**
- `label` (string)
- `source_field` (string)
- `state_map` (object) — maps values to `{color, label, icon}`. E.g.:
  ```json
  {
    "running": {"color": "success", "label": "Running", "icon": "mdi-check-circle"},
    "warning": {"color": "warning", "label": "Attention", "icon": "mdi-alert"},
    "stopped": {"color": "error", "label": "Stopped", "icon": "mdi-close-circle"}
  }
  ```
- `default` (object, optional) — used when value not in map

**Runtime:** looks up value in `state_map`, renders colored dot + label.

**Effort:** 2d

### 5. `output.text_block` — Text / Markdown

Static text with markdown support. For section headers, instructions,
contact info, links to docs.

**Config schema:**
- `content` (string, markdown) — the body text
- `heading_level` (enum, optional) — h1 / h2 / h3 / p
- `alignment` (enum) — left / center / right

**Runtime:** render markdown to HTML. Use `marked.js` (small, already available).

**Effort:** 1d (mostly Vuetify styling)

## Architecture — where each piece lives

- **Schema definition:** `frontend/src/views/AppBuilderEditorView.vue`
  — add each widget to the node type registry with its config schema
- **Editor UI (config panel on the right):** already auto-renders from the
  schema. Just needs a few new field types (`markdown` for text_block,
  `state_map` for status_indicator).
- **Palette (left sidebar):** register each with icon + color + category
  ("Display" for big_number/gauge/status_indicator/text_block, "Action"
  for button).
- **Runtime rendering:** `frontend/src/views/PublishedAppView.vue`
  — new render components per widget type.
- **Backend runner:** `backend/app/routes/app_builder.py`
  — pass-through: widgets don't need backend processing beyond what
  `output.table` already does (just serialize the current pipeline output).
  For `output.button.action=mqtt.publish`, no backend involvement — frontend
  publishes directly.

## Widget order (recommend sequential shipping)

1. **`output.text_block`** — 1d — simplest, unblocks anyone doing dashboard
   design that needs headers.
2. **`output.big_number`** — 2d — highest visual impact, most common ask.
3. **`output.status_indicator`** — 2d — closely related to big_number;
   share styling patterns.
4. **`output.gauge`** — 3d — needs ECharts wiring, slightly more complex.
5. **`output.button`** — 3d — the trickiest because "action" is a whole
   sub-system. Ship last.

Ship each as it lands (they're independent). Milestone: **all 5 shipped
in ~2 weeks** including polish + docs.

## Open questions

1. **Button MQTT auth** — does the published app's MQTT client have write
   permission? Today it only subscribes. Broker config may need per-app
   ACLs. Confirm with customer's MQTT setup.
2. **State map UX in editor** — the `state_map` field is a JSON object.
   Auto-render a "add state" mini-form or accept raw JSON? Recommend
   mini-form for v1 (better UX).
3. **Threshold band UX** — same question for `output.big_number.thresholds`
   and `output.gauge.bands`. Recommend visual band editor.
4. **Do widgets persist state?** — e.g., button click history. Recommend
   no — widgets are stateless in v1. If needed later, tie to Tier 2
   variables (deferred).
5. **Multi-source widgets** — some customer dashboards want a big number
   that's `output_A / output_B * 100`. That needs expression support
   (Tier 2). Not in scope for v1. If asked, defer.

## Concrete customer use case (to build against)

Ask customer for **one real dashboard they want** — probably a
per-machine operator view with:
- Big number for current OEE
- Gauge for cycle time
- Status indicator for machine state
- Button that publishes "reset" to `factory/machine-B079/control`
- Text block with contact info for the shift supervisor

Building against a real dashboard flushes out schema gaps and UX issues
faster than 5 abstract widgets in isolation.

## Testing plan

- **Editor tests:** each widget appears in palette, can be dropped on
  canvas, config panel renders correctly, form validates required fields.
- **Runtime tests:** each widget renders in a Published App with mocked
  pipeline output. State updates on new predictions.
- **Cross-widget integration:** all 5 widgets on one canvas, all render
  simultaneously, no layout conflicts.
- **`output.button` action integration:** MQTT publish reaches the broker,
  HTTP GET/POST returns 200 on mock endpoint, download.csv produces
  a valid CSV.

## Handoff / sequencing

- Widgets are individually shippable — start with `output.text_block`
  today, iterate one-per-day pace, finish in 2 weeks.
- No dependencies on the Partner SDK plan or SQL feed plan. Fully parallel.
- Docs: update `docs/USER_MANUAL.md` App Builder section with each widget
  as it ships. Screenshots optional but nice.

## Companion plans

- **`PLAN_2026-08-12_partner-sdk.md`** — parallel workstream
- **`PLAN_2026-08-12_sql-data-feed.md`** — parked, later
