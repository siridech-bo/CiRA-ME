# Phase J — Multi-topic Live Stream + Signal Recorder

**Date:** 2026-07-31
**Effort:** ~1.5 days
**Depends on:** Phase D (ingest router in-memory topic tail), Phase F (asset tree store), Phase H (cross-sensor JOIN pattern)

---

## Motivation

Customer feedback (2026-07-31): current LIVE STREAM (MQTT) input node takes ONE topic. For the Signal Recorder app template — the primary way users build labeled training datasets — that means running 4 separate recorder apps side-by-side to capture a 4-sensor machine. Their ask: "multitopic and can search and add on the app."

## Decisions (locked)

| Decision | Value |
|---|---|
| Field type | `v-combobox` (multiple) — chip-style multi-select |
| Autocomplete sources | (a) asset tree active sensor topic_paths, (b) recently-published topics via new `/recent-topics` endpoint, (c) freeform (typed) |
| "Load from machine" button | Opens the existing MachineTreePickerDialog with single-machine mode → adds all its sensor topics as chips |
| Recording session storage | `data/recordings/<session_id>/<topic-slugged>.csv` + `session.json` metadata |
| Session filename slug | replace `/` with `_` (e.g. `factory_plant_A_air_compressor_01_pressure.csv`) |
| Backward compat | Old single-topic config → auto-migrate to `topics: [topic]` on node load. Server accepts either shape. |
| Non-recorder apps with multi-topic | Toast warning surfaced downstream ("Multiple topics selected — pick one for inference or use Machine Live Stream node instead"). Don't hard-block. |
| Wildcard subscriptions (`+`, `#`) | Not in v1 — explicit topic list only |
| Recent-topics endpoint | `GET /api/asset-tree/recent-topics?window_s=900&limit=100` — any authed user, from the ingest router's in-memory tail |

## Backend

### J.1 — `/recent-topics` endpoint

`backend/app/routes/asset_tree.py`:
- `GET /api/asset-tree/recent-topics?window_s=900&limit=100`
- Reads router's `_stats['last_message_topic']` tail (already tracked). If router doesn't currently keep a per-topic first-seen map, add one: `_topic_last_seen: Dict[str, float]` (monotonic seconds, per-message updated in `_route`, capped at 500 entries by an LRU-style purge)
- Response:
  ```json
  {
    "topics": [
      {"topic": "factory/plant_A/air_compressor_01/pressure",
       "last_seen": "2026-07-31T13:22:04.123Z",
       "seconds_ago": 4.2},
      ...
    ]
  }
  ```
- Any authed user (matches the Rejected-topics endpoint's auth)
- Query params: `window_s` default 900 (15 min), `limit` default 100. Both hard-capped at 3600s / 500 entries.

### J.2 — Signal Recorder multi-topic capture

Locate the current Signal Recorder capture code (search for `recorder` + the `POST /run/<slug>` handler in `app_builder.py` OR wherever the recorder mode writes CSVs today).

- Current behaviour: single topic → single CSV in the session folder
- New behaviour:
  - Accept `topics: string[]` in the app config
  - For each incoming MQTT message on any subscribed topic, buffer + write to `<session_dir>/<topic_slugged>.csv`
  - Header per file: `timestamp_iso,<value_columns>` where value_columns are inferred per-topic (e.g. `value` for single, `x,y,z` for multi-axis payloads — matches ingest router's parse logic)
  - Session end (Disconnect): write `session.json` with `{session_id, started_at, ended_at, topics: [...], row_counts: {topic: N}, per_topic_files: [...]}`
- Reuse `MqttIngestRouter._parse_payload` / `_parse_multi_axis_payload` logic if it's importable, else duplicate. Keep parsing consistent so downstream loader sees the same shape as ingest router output.
- Backward compat: old apps with `topic: "..."` single string → treat as `topics: [topic]` on load.

## Frontend

### J.3 — Multi-topic combobox on Live Stream node

Locate the LIVE STREAM (MQTT) node's config panel (grep for the current "Topic" input). Replace with a `v-combobox`:

```vue
<v-combobox
  v-model="config.topics"
  :items="topicSuggestions"
  chips
  closable-chips
  multiple
  hide-no-data
  hide-selected
  label="Topic(s)"
  placeholder="Type a topic path or pick one below"
/>
```

- `topicSuggestions` computed: merge (a) `assetTreeStore.tree` walked for sensor-level nodes' `topic_path`, (b) recent-topics API result, deduped, ranked by (recency first, then alphabetical).
- Fetch recent-topics on node mount + refresh every 30 s while the node's config panel is open.
- Empty-state hint below the combobox: "Add topics to record from — pick from your asset tree or type manually."

### J.4 — "Load from machine" button

Below the combobox:

```
[ + Load from machine… ] [ Clear all ]
```

Clicking **Load from machine…** opens the existing `MachineTreePickerDialog` in single-select mode → on selection, walks the machine's active sensor children + appends each `topic_path` as a chip. Dedupes.

### J.5 — Backward-compat migration on node load

When loading an existing app config:
```typescript
// Node's saved config
config.topics = Array.isArray(config.topics)
  ? config.topics
  : (config.topic ? [config.topic] : [])
delete config.topic  // don't keep both — the multi-topic shape wins
```

## Non-recorder apps with multi-topic

- If the app has a `model.endpoint.*` or `transform.feature_extract` node downstream AND the input has >1 topic, show a **warning banner** below the combobox:
  > *"Inference apps expect one topic per Live Stream node. Downstream nodes will use the FIRST topic only. For multi-sensor inference, use the Machine Live Stream node instead."*
- Don't hard-block — customer might genuinely want raw multi-topic capture with a downstream inference on the first.

## Deliverables

| ID | Owner | Description |
|---|---|---|
| J.1 | backend | `/recent-topics` endpoint + router `_topic_last_seen` map |
| J.2 | backend | Signal Recorder multi-topic capture + session.json |
| J.3 | frontend | Multi-topic combobox on Live Stream node config panel |
| J.4 | frontend | "Load from machine" button + tree picker integration |
| J.5 | frontend | Backward-compat migration on node load |
| J.QA | agent | Adversarial QA pass |
| J.T | user | Personal browser test |

## Edge cases

- Zero topics selected + Connect clicked → validation error "add at least one topic"
- Duplicate topic added → silent dedupe
- Topic string with spaces / special chars → validate against MQTT topic grammar `[A-Za-z0-9_\-\+\#\/]+`, reject others with inline chip error
- 20+ topics selected → still works but warn about throughput ("Recording 20 topics at 10 msg/s each may drop messages")
- Session folder disk-full → recorder logs + surfaces error via existing app-runtime error path
- Broker disconnect mid-session → session.json marks `status: "interrupted"` with partial `row_counts`
- App loaded with legacy `topic` field only → migrated silently to `topics: [topic]` (no notification — the load should just work)
- Machine picker: machine with retired sensors → include ONLY active children as chips

## Not doing (v1)

- Wildcard subscriptions (`factory/+/+/pressure`)
- Cross-machine recording session (each session = one machine OR arbitrary topic set)
- Real-time recording preview graph (viz stays as the existing Signal Recorder's chart, showing only the FIRST topic — call it out in a follow-up if customer asks)
- Server-side JOIN of the per-topic files at session end (loader does JOIN at training time already)

## Definition of done

- User adds 4 topics via the combobox + connects → 4 CSVs appear under `data/recordings/<session_id>/`
- Autocomplete surfaces sensors from the asset tree + recently-published topics
- "Load from machine" button adds all a machine's active sensor topics in one click
- Old single-topic apps load without regression + auto-migrate to topics[]
- Zero blockers from QA
- Personal browser test passes end-to-end
