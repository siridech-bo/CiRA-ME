# MQTT Topic Namespacing — Publishing to CiRA ME

Phase D of the Asset Tree project (2026-07-19) adds a background service —
the **MQTT Ingest Router** — that subscribes to `#` on the Mosquitto broker
and routes every message by topic path to a folder under
`datasets/<topic_path>/`. Rolling daily CSVs land there automatically, and
retention is enforced by a nightly janitor.

This doc is for anyone publishing data into CiRA ME — device firmware
authors, integrators, and QA scripts.

## The 30-second version

- Topic format: `<root>/<plant>/<machine>/<sensor>` (levels are configurable).
- Payload: JSON `{"value": <number>}`, or a bare number as a string.
- Publish continuously; a CSV per sensor per day appears under
  `datasets/<root>/<plant>/<machine>/<sensor>/YYYY-MM-DD.csv`.
- Unknown topics get **rejected** in Strict mode (default) or **auto-created**
  in Learn mode. Flip the mode in **Settings → MQTT Rules**.

## Topic format

Every published topic must start with the configured **root_name** (default
`factory`). Everything after the root is a path through the asset tree —
one segment per level. For the default 4-level tree:

```
factory / plant_A / machine_1 / temperature
   root      plant     machine     sensor
```

Segment names must match `^[A-Za-z0-9_-]+$` (letters, digits, dash,
underscore). No spaces, no unicode, no slashes inside a segment. Depth
must match the configured level count exactly — a topic that ends at the
machine level is not a valid data topic.

### Example

Given the default `small_factory` template:

```
factory
  plant_A
    machine_1
      temperature
      vibration
      pressure
```

Valid data topics:

```
factory/plant_A/machine_1/temperature
factory/plant_A/machine_1/vibration
factory/plant_A/machine_1/pressure
```

## Payload format

The router accepts three shapes, tried in order:

1. **JSON with `value`**  (recommended)
   ```
   {"value": 22.5}
   ```
2. **JSON with `v` or `val`** — same semantics, shorter for constrained devices.
3. **Bare numeric literal** — the entire payload parses as `float(payload)`.
   ```
   22.5
   ```

Anything else (arbitrary JSON, binary blobs, empty payloads) is dropped
with a `unparseable payload` entry in the rejected-topics log.

Booleans get coerced: `true → 1`, `false → 0`.

## Strict vs Learn mode

Set in **Settings → MQTT Rules → Config → Topic mode**.

| Mode | Unknown topic behavior |
|------|------------------------|
| **Strict** (default) | Rejected + logged. Nothing written to disk. |
| **Learn** | Any missing tree nodes are created on-the-fly and the message is written normally. |

Learn mode is the fast path for commissioning new machines — you can
publish first and register later. Switch back to Strict once the fleet
stabilises so a typo doesn't silently create garbage tree nodes.

## Meta prefixes — the escape hatch for non-data channels

Devices publishing heartbeats, config, or command traffic on their own
sub-topic can share the tree namespace without polluting the CSV. Any
segment that matches a configured meta prefix (default: `_meta`, `_health`,
`_config`, `_cmd`) accepts the topic **but** skips the write.

Example — this heartbeat topic accepts (counter +1 in **Stats → Meta**) but
never touches disk:

```
factory/plant_A/machine_1/_meta/heartbeat
```

Prefixes are matched by **exact segment equality**, not prefix-of-string.
`_metastasized` does not match `_meta`. Add new prefixes in
**Settings → MQTT Rules → Config → Meta prefixes**.

## Testing your publisher

### With `mosquitto_pub` from CLI

Inside the `cirame-backend` container (or any host on the compose network):

```bash
docker exec -it cirame-mosquitto mosquitto_pub \
    -h localhost -p 1883 \
    -t 'factory/plant_A/machine_1/temperature' \
    -m '{"value": 22.5}'
```

Then in another shell:

```bash
tail -f datasets/factory/plant_A/machine_1/temperature/$(date -u +%F).csv
```

You should see a new `timestamp_iso,value` row appear within ~200 ms
(the router batches writes; the flush ticks every 200 ms).

### With the built-in browser publisher

The **MQTT Broker** page has a browser-side **MQTT Test Publisher** panel
that streams a CSV to the broker. Two things to know:

1. **The rate input is capped at 5 msg/s.** You can type a higher number
   into the field, but the actual publish loop caps at 5 to protect the
   broker at workshop scale (~50 attendees each publishing = broker
   collapse if uncapped). Real IoT devices publishing directly are not
   affected — this only applies to the in-browser test publisher.
2. **The publisher's topic is a single string** — pick a real sensor path
   like `factory/plant_A/machine_1/temperature` to see the ingest router
   consume it.

### With a Python one-liner

```python
import paho.mqtt.publish as pub
pub.single(
    'factory/plant_A/machine_1/temperature',
    payload='{"value": 22.5}',
    hostname='cirame-mosquitto',
    port=1883,
)
```

## Verifying the write

- **File on disk**:
  `datasets/<root>/<plant>/<machine>/<sensor>/YYYY-MM-DD.csv`. Header is
  `timestamp_iso,value`. Rows batch every 200 ms.
- **Stats tab**: **Settings → MQTT Rules → Stats** shows counters that
  bump in near-real-time. `messages_routed` = disk writes; `messages_rejected`
  = topics that failed validation.
- **Rejected tab**: **Settings → MQTT Rules → Rejected**. TSV log under
  `datasets/_rejected_topics/YYYY-MM-DD.log`.

## File layout summary

```
datasets/
  factory/
    plant_A/
      machine_1/
        temperature/2026-07-19.csv    ← rolling daily CSV per sensor
        vibration/2026-07-19.csv
        pressure/2026-07-19.csv
      machine_2/…
    plant_B/…
  _rejected_topics/
    2026-07-19.log                    ← TSV of rejected topics + reasons
```

Directories are created on demand; you never have to pre-create a sensor
folder before publishing to it (Learn mode) or the router won't rehearse
its own layout (Strict mode — the folder appears the first time the
topic passes the tree check).

## Retention

Files whose date-in-filename is older than **Settings → MQTT Rules →
Retention (days)** get physically deleted every 6 hours. Default: 30.
Admin can force a sweep from **Settings → MQTT Rules → Stats → Run
retention sweep** for QA.

## Related

- [PLAN_2026-07-18_asset-tree.md](PLAN_2026-07-18_asset-tree.md) §12 — full design.
- [EXECUTION_asset-tree.md](EXECUTION_asset-tree.md) — Phase D subtasks (D.1–D.6).
- `datasets/shared/mqtt-test/README.md` — legacy MQTT test CSVs (predate Phase D).
