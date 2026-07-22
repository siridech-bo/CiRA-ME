# Phase H — Multi-axis sensors + Machine Live Stream app-builder node

**Date:** 2026-07-22
**Effort:** ~2 days
**Depends on:** Phase D (ingest router), Phase F (machine profiles + simulator), Phase G (data-loader label sidecar)

---

## Motivation

Two customer-visible gaps in the App Builder:

**Q1** — the Live Stream (MQTT) input node subscribes to ONE topic. For a real machine that publishes 4-5 sensor topics, users must wire 4-5 Live Stream nodes side-by-side and merge them manually. Doesn't scale.

**Q2** — real accelerometers and gyroscopes publish `{"x": 1.2, "y": -0.3, "z": 9.8}` on ONE topic. Current router expects `{"value": N}` — single scalar per topic. Multi-axis payloads get rejected, so training pipelines can't ingest common IMU data.

Phase H closes both: one new app-builder node that binds to a machine (auto-subscribes to all its sensor topics), plus a new "multi_axis" sensor kind that lets the router demultiplex `{"x", "y", "z"}` payloads into a multi-column CSV.

## Decisions (locked)

| Decision | Value |
|---|---|
| Multi-axis storage | ONE CSV per topic, columns = axes (`timestamp,x,y,z`) |
| Tree schema for multi-axis | `AssetSensorMeta.channels` JSON list; empty = single-value, non-empty = multi-axis |
| Backward compat for existing profiles | Untouched. CNC keeps `vibration_x/y/z` as 3 separate sensors. New "3-axis Accelerometer" and "3-axis Gyroscope" profiles added for the multi-axis case. |
| Payload keys → columns | Order comes from `channels` list; missing keys → column value = None (row still written; row silently dropped ONLY when ALL channel values missing) |
| App Builder node type | New `input.machine_live_stream` node — binds to a machine node in the asset tree |
| Machine Live Stream buffering | Client-side (mqtt.js): per-topic ring buffer + nearest-alignment on emit, 200 ms tolerance default |
| Node output | One channel per single-value sensor + one channel per axis of each multi-axis sensor. Channel naming: `<sensor>` or `<sensor>.<axis>` |
| Multi-axis on the existing Live Stream node | Also works — Channel Names field already exists client-side. Server ingest now matches. |

## Backend

### H.1 — Tree schema + admin UI

**Migration**: add `channels TEXT` column to `asset_sensor_meta` (nullable, JSON-encoded string array). Default NULL.

**Models** (`backend/app/models.py`):
- `AssetSensorMeta.upsert(...)` gains an optional `channels: Optional[List[str]] = None` parameter
- Serialize/deserialize the channels list to/from JSON string
- Existing calls with no `channels` arg → column stays NULL → single-value behavior

**Routes** (`backend/app/routes/asset_tree.py`):
- `PATCH /nodes/<id>` accepts a `channels` field on sensor-meta updates
- Validate: list of 1-16 unique strings matching `^[A-Za-z0-9_]+$`. Empty list → treated as NULL. Invalid → 400.

**Frontend** (`SensorMetaEditor.vue` or wherever sensor meta is edited):
- New field "Channels (comma-separated, leave blank for single-value)" under Unit + Sample Rate
- Hint text explaining multi-axis vs single-value
- Placeholder: `x, y, z`

### H.2 — Router: multi-axis payload demultiplex

`backend/app/services/mqtt_ingest_router.py`:

The route pipeline currently ends at `_parse_payload → str value → _append_row(topic, timestamp, value)`. Extend it:

1. Add `_lookup_sensor_channels(topic)` — reads sensor meta from the cache, returns the channels list or None
2. After the tree-membership check but BEFORE `_parse_payload`, branch:
   - Channels is None → existing single-value flow (no change)
   - Channels is non-empty → new `_parse_multi_axis_payload(payload_bytes, channels)`:
     - Must parse as a JSON dict
     - Extract each channel value → validate numeric, cast to str
     - If ALL keys missing → return None (rejected as "unparseable payload" — matches Phase D)
     - If SOME keys present → present ones become row values, missing ones become empty strings
     - Returns `List[Optional[str]]` in the channels order
3. `_append_row` gets a variant `_append_multi_row(topic, timestamp, values, channels)`:
   - First write: writes header `timestamp,<ch1>,<ch2>,...` (once per CSV file, matches existing header-once logic)
   - Every row after: writes `<ts>,<v1>,<v2>,...`
   - Buffered write cadence identical to single-value path
   - Same UTC-midnight rotation
4. Stats: `messages_routed` counts one per successful multi-axis publish (not one per channel). Add `channels_written` counter.

Edge cases:
- Payload is `{"value": N}` on a multi-axis sensor → reject as `payload has no channel keys (expected: x, y, z)`
- Payload has extra unknown keys (e.g. `{"x":1, "y":2, "z":3, "temperature":42}` on a `[x,y,z]` sensor) → silently ignore extras, write the 3 known channels
- Router cache reloads on tree mutation (existing) → catches channel changes without restart

### H.3 — Machine profiles + simulator support

`backend/app/constants/machine_profiles.py`:

Extend `SensorDef` dataclass:
```python
@dataclass
class SensorDef:
    name: str
    unit: str
    sample_rate_hz: float
    channels: Optional[List[str]] = None   # NEW — None = single, non-empty = multi-axis
```

CNC Spindle profile: NO CHANGE (keeps 3 separate sensors for backward compat).

Add TWO new profiles:

**3-axis Accelerometer** (`accelerometer_3axis`)
- Sensors: `accel` with `channels=['x', 'y', 'z']`, unit=`g`, sample_rate_hz=100
- States: `off`, `still` (means ~0/0/9.8, all near zero std), `walking` (sinusoidal x+y at 2Hz + steady z), `impact` (spike on one axis)

**3-axis Gyroscope** (`gyroscope_3axis`)
- Sensors: `gyro` with `channels=['x', 'y', 'z']`, unit=`dps` (degrees per second), sample_rate_hz=100
- States: `off`, `still` (~0 all), `rotating_y` (sinusoidal on y, others near zero), `tumbling` (all three moving)

Simulator (`backend/app/services/machine_simulator.py`):
- `_SimulatedMachine._publish_signal` for multi-axis sensors: build `{"x": vx, "y": vy, "z": vz}` payload per tick (one MQTT publish per sensor per tick, not per channel)
- Autoprovision (`_autoprovision_tree`): when creating a sensor node with `channels`, upsert `AssetSensorMeta` with the channels list
- Chaos state: also injects poisons on multi-axis topics (missing keys, non-JSON, etc.)

### H.4 — Data loader multi-column CSV

`backend/app/services/data_loader.py`:

Existing multi-column CSV load already works (any CSV with N columns → N-column DataFrame). Verify:
- `load_csv` on a multi-axis CSV `timestamp,x,y,z` returns a DataFrame with 4 columns, sensor_columns metadata lists `['x','y','z']`
- Cross-sensor JOIN of a mixed machine (single-value + multi-axis) merges correctly on timestamp
- Labels sidecar still works on multi-column CSVs

Add: `metadata.channels_by_sensor` — a dict mapping sensor name to channels list, so downstream UI can render each channel as a distinct signal.

## Frontend

### H.5 — Machine Live Stream node

`frontend/src/components/app-builder/` (or wherever nodes live):

New node type `input.machine_live_stream` (labelled "Machine Live Stream"). Config panel:
- **Machine picker** — dropdown or tree-picker sourced from `assetTreeStore` (only machine-level nodes)
- Read-only display of the machine's sensor topology: e.g.
  ```
  Auto-detected 4 sensor topics
    ✓ pressure          (single)
    ✓ temperature       (single)
    ✓ accel             (multi-axis: x, y, z)
    ✓ current           (single)
  ```
- Broker URL (defaults from asset tree config)
- Sample rate / alignment / tolerance / buffer window (advanced expander)

Runtime behavior (browser mqtt.js):
- On start: subscribe to `<machine>/<sensor>` for each active sensor child
- For each incoming message:
  - Parse per the sensor's channels config (single-value or multi-axis)
  - Push into per-topic ring buffer keyed by timestamp
- On tick (at `sample_rate_hz`):
  - Find the timestamp closest to "now - buffer/2" (aligned emission)
  - For each topic, take the buffered sample nearest that timestamp (within tolerance)
  - Emit a row: `{timestamp, sensor1, sensor2.x, sensor2.y, ...}` — one channel per key
  - Downstream nodes see this as a joined N-channel signal, same as CSV JOIN

### H.6 — Downstream node updates

Verify chart / normalize / windowing / feature-extract nodes handle N-channel input correctly. Most should already — they were designed for CSV JOIN. Explicit fixes only if a specific node hardcodes single-channel.

Channel naming in the pipeline: use `<sensor>` for single-value, `<sensor>.<axis>` for multi-axis. Chart legend, Normalize's "which columns to normalize" picker, Feature Extract's "input signals" — all display these names.

## Deliverables (subtasks)

| ID | Owner | Description |
|---|---|---|
| H.1 | backend | `asset_sensor_meta.channels` column + upsert() param + PATCH validation |
| H.2 | backend | Ingest router multi-axis parse + multi-column CSV write |
| H.3 | backend | Machine profiles: 3-axis Accelerometer + 3-axis Gyroscope; simulator publishes multi-axis payloads |
| H.4 | backend | Data loader metadata: channels_by_sensor for downstream UI |
| H.5 | frontend | Machine Live Stream node + tree-picker config panel + mqtt.js runtime |
| H.6 | frontend | Verify chart / normalize / windowing / features render multi-channel + sensor.axis naming |
| H.QA | agent | Adversarial QA pass |
| H.T | user | Personal browser test |

## Edge cases + failure modes

- Multi-axis payload missing all keys → rejected as `payload has no channel keys (expected: x, y, z)`
- Multi-axis payload missing SOME keys → row written with empty cells; loader treats empties as NaN downstream
- Multi-axis payload with wrong types (e.g. `{"x": "foo"}`) → row written with empty cell for that channel + warning logged
- Sensor's channels list changes while router is running → cache reload picks it up; first message after reload uses new columns (new CSV file starts fresh with new header)
- Backend restart mid-day → header is written once per file per boot; restart doesn't corrupt an existing multi-axis CSV (header check via file existence, same as single-value)
- Autoprovisioned multi-axis sensor whose child asset node already exists with channels=NULL → PATCH upserts channels; no CSV mid-file schema change (new schema takes effect at next UTC midnight rotation OR next boot)
- Machine Live Stream node: machine has zero active sensors → node shows warning, doesn't crash
- Machine Live Stream node: machine gets its tree changed while app is running → node picks up new topology on next config reload (user clicks "Refresh topology" button)

## Not doing (v1)

- Cross-machine Live Stream (subscribing across multiple machines in one node) — future
- Multi-axis sensor visualization in the tree UI beyond a chip label — v1 just shows "accel (x, y, z)"
- Auto-detection of multi-axis payloads on unknown sensors — user must declare `channels` explicitly
- Payload key aliasing (e.g. `ax` → `x`) — user must publish keys that match `channels` exactly

## Definition of done

- A simulator using the 3-axis Accelerometer profile publishes `{"x", "y", "z"}` to one topic; ingest router writes a `timestamp,x,y,z` CSV; data loader reads it as 3 channels
- Multi-axis sensor added via tree admin (channels field) accepts multi-axis payloads immediately
- Machine Live Stream node in App Builder subscribes to a machine's topics, emits joined multi-channel signal downstream
- Existing single-value sensors continue to work with zero regression
- Zero blockers from QA
- Personal browser test passes end-to-end
