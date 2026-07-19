# Phase F — Machine Simulator + Sidebar Reorganization

**Date:** 2026-07-19
**Effort:** ~1.5-2 dev days
**Depends on:** Phase D (MQTT ingest router) — router receives simulator output

---

## 1. Motivation

Workshops and customer demos keep hitting the "no signals, can't demo the ingestion" gap.
- Real IoT devices need setup + hardware
- Browser Test Publisher (Phase D) is capped at 5 msg/s and stops when the tab closes
- No way to demonstrate fault-injection or state-driven signal changes

Phase F ships a **built-in Machine Simulator**: server-side threads inside the existing backend that publish realistic sensor signals to `cirame-mosquitto` under user-picked topics, driven by user-picked machine profiles + states.

Signals flow through the Phase D ingest router → daily CSVs → all existing training/inference paths work as if a real device were connected.

## 2. Design decisions (locked in)

| Decision | Value |
|---|---|
| Deployment | Backend threads, no new container |
| Persistence | In-memory only (like `_publishers`, `_recording_jobs`) — restart wipes |
| Auth on all `/api/simulators/*` CRUD | Admin-only (matches MQTT Rules pattern) |
| Read-only list endpoint | Any authenticated user (for demo audience visibility) |
| Profile set | All 6 shipped: Compressor / Boiler / Pump / Conveyor / CNC / Chiller |
| Auto-provision tree entries | Checkbox in create dialog, defaults **ON** |
| Delete simulator behaviour | Tree entries NOT auto-retired (manual cleanup via tree admin) |
| Chaos state | Ships with all profiles — injects null / garbage / wrong-root every ~30 s |
| Rate cap | None (server-side, no browser clamp needed) — soft warning when total msg/s across all simulators > 200 |

## 3. Machine profiles

Each profile is a `MachineProfile` dataclass in `backend/app/constants/machine_profiles.py`:

```python
@dataclass
class SensorDef:
    name: str          # topic segment
    unit: str          # 'bar', 'C', 'mm/s', 'A', 'Hz', 'percent'
    sample_rate_hz: float  # publish rate

@dataclass
class StateParams:
    # Per-sensor: (mean, std_dev, sinusoid_amplitude, sinusoid_period_s)
    # None → sensor stays silent in this state (e.g. off / maintenance)
    per_sensor: Dict[str, Optional[Tuple[float, float, float, float]]]
    # Optional fault spike probability per tick per sensor
    fault_spikes: Dict[str, float] = field(default_factory=dict)

@dataclass
class MachineProfile:
    id: str
    display_name: str
    icon: str            # mdi- name
    description: str
    sensors: List[SensorDef]
    states: Dict[str, StateParams]  # keys: 'off', 'idle', 'running', ...
    default_state: str
```

### 3.1 Air Compressor
Sensors: `pressure` (bar, 1 Hz) · `temperature` (C, 1 Hz) · `vibration` (mm/s, 5 Hz) · `current` (A, 1 Hz)

| State | pressure | temperature | vibration | current | Notes |
|---|---|---|---|---|---|
| idle | 1.0 ± 0.05 | 25 ± 1 | 0.3 ± 0.1 | 0.5 ± 0.2 | ambient |
| running | 4.0 + 2.0·sin(30s) ± 0.2 | 55 ± 3 | 2.5 ± 0.4 | 15 ± 1 | normal load |
| loaded | 6.8 ± 0.3 | 72 ± 3 | 3.2 ± 0.5 | 22 ± 1.5 | high load |
| fault | 3.5 ± 0.5 (drops) | 88 ± 4 + spike | **7.5 ± 1.2** (bearing) | 27 ± 2 | occasional 12+ mm/s spike |
| maintenance | silent | silent | silent | silent | no publish |

### 3.2 Industrial Boiler
Sensors: `flame` (0/1, 0.5 Hz) · `feedwater_temp` (C, 1 Hz) · `steam_pressure` (bar, 1 Hz) · `flue_gas_temp` (C, 1 Hz) · `o2_percent` (%, 1 Hz)

| State | flame | feedwater_temp | steam_pressure | flue_gas_temp | o2_percent |
|---|---|---|---|---|---|
| off | 0 | 25 | 0 | 25 | 20.9 |
| warming | 1 (duty 0.7) | ramps 25→80 | 0 → 2 | ramps 25→180 | 5 ± 1 |
| steaming | 1 (duty 0.5) | 85 ± 3 | 8 + 0.5·sin(60s) | 220 ± 15 | 4 ± 0.5 |
| blowdown | 0 | 85 → 60 | 8 → 3 | 220 → 150 | 20.9 |
| fault | 1 (stuck) | 95 (overheat) | 12 (over-pressure) | 320 (unsafe) | 1 ± 0.3 (rich) |

### 3.3 Centrifugal Pump
Sensors: `inlet_p` (bar, 1 Hz) · `outlet_p` (bar, 1 Hz) · `flow_rate` (L/min, 1 Hz) · `motor_current` (A, 1 Hz) · `vibration` (mm/s, 5 Hz)

| State | inlet | outlet | flow | current | vibration |
|---|---|---|---|---|---|
| off | 1.0 ± 0.05 | 1.0 ± 0.05 | 0 | 0.3 | 0.2 |
| running | 1.5 ± 0.1 | 4.5 ± 0.2 | 120 ± 10 | 8 ± 0.5 | 1.8 ± 0.3 |
| cavitation | 0.4 ± 0.3 (jumpy) | 3.5 ± 0.8 (jumpy) | 90 ± 25 | 9.5 ± 1 | 5.5 ± 1.5 (harsh) |
| dry_run | 0.9 ± 0.05 | 1.1 ± 0.1 (no lift) | 0 ± 2 | 6 ± 0.5 | 4.5 ± 0.8 (bearing hot) |

### 3.4 Conveyor
Sensors: `speed_rpm` (1 Hz) · `motor_current` (A, 1 Hz) · `belt_tension` (kN, 1 Hz) · `vibration` (mm/s, 5 Hz) · `product_count` (integer, 0.2 Hz)

| State | speed_rpm | current | belt_tension | vibration | product_count |
|---|---|---|---|---|---|
| off | 0 | 0.3 | 2.0 ± 0.2 | 0.1 | 0 |
| running | 900 ± 20 | 4 ± 0.5 | 3.5 ± 0.2 | 1.5 ± 0.3 | +N per interval |
| jam | 0 | 12 (motor stall) | 6.0 (over-tension) | 3.5 (vibrating) | 0 |
| belt_slip | 900 (motor) but 400 (belt) | 6 ± 1 | 1.8 ± 0.5 (slack) | 4.2 (whipping) | +N sporadic |

### 3.5 CNC Spindle
Sensors: `spindle_rpm` (1 Hz) · `spindle_load` (%, 1 Hz) · `temperature` (C, 1 Hz) · `vibration_x/y/z` (mm/s, 10 Hz each)

| State | rpm | load | temp | vib_x/y/z |
|---|---|---|---|---|
| off | 0 | 0 | 22 | 0.05 each |
| idle | 800 ± 20 | 5 ± 1 | 30 ± 2 | 0.4 each |
| cutting | 6000 + 400·sin(15s) | 55 ± 8 | 45 ± 3 | 1.8 / 2.1 / 1.5 |
| chatter | 6000 ± 200 | 75 ± 15 | 52 ± 4 | 6.5 / 8.2 / 5.0 (correlated peaks) |

### 3.6 Chiller / HVAC
Sensors: `refrigerant_p` (bar, 1 Hz) · `evap_temp` (C, 1 Hz) · `cond_temp` (C, 1 Hz) · `compressor_current` (A, 1 Hz)

| State | ref_p | evap_temp | cond_temp | comp_current |
|---|---|---|---|---|
| off | 5 ± 0.1 | 25 ± 1 | 25 ± 1 | 0 |
| cooling | 8.5 ± 0.3 | -2 ± 1 | 42 ± 2 | 14 ± 0.5 |
| defrost | 6 ± 0.2 | 10 ± 2 | 30 ± 2 | 3 ± 0.3 |
| fault | 12 (over-pressure) | 15 (no cooling) | 65 (overheat) | 20 (high load) |

### 3.7 Chaos (all profiles)
Every ~30 s (random 20-40 s), inject ONE of:
- `{"value": null}` → tests null-value rejection
- Garbage bytes `b'\x00\x01\xff'` → tests parse error
- Publish to `wrong_root/{name}/temperature` → tests root mismatch
- Publish to `factory/plant_X/{name}/temperature` (unknown plant) → tests strict-mode rejection

## 4. Backend

### 4.1 `backend/app/services/machine_simulator.py`

```python
class MachineSimulator:
    """Singleton — spawns/kills per-machine threads."""
    def __init__(self): ...
    def start(self, flask_app): ...        # boot; no-op if already running
    def stop(self): ...                    # atexit; kills all threads
    def list_instances(self) -> list[dict]: ...
    def create_instance(self, profile_id, name, topic_base, initial_state, autoprovision_tree, actor_user_id) -> dict: ...
    def patch_state(self, instance_id, new_state, actor_user_id): ...
    def delete_instance(self, instance_id, actor_user_id): ...
    def publish_raw(self, topic, payload_bytes, actor_user_id): ...
    def snapshot(self) -> dict:            # for Stats — total msg/s, per-machine msg/s, alive counts
        ...

class _SimulatedMachine:
    """One thread per machine — publishes at each sensor's sample rate."""
    def __init__(self, profile, name, topic_base, initial_state): ...
    def set_state(self, state): ...
    def run(self):                         # ticks at min(sample_rate_hz); publishes each sensor per its rate
        ...
```

- One shared paho client (like `MqttIngestRouter`) — connects on boot, publishes on each tick
- Per-machine `_stats`: `messages_published`, `state_since_ts`, `current_state`
- Chaos ticks fire from the main run-loop on an independent 20-40 s random schedule
- All exceptions swallowed + logged (never crash the backend)

### 4.2 `backend/app/routes/simulators.py` — new blueprint on `/api/simulators`

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/profiles` | any | List built-in profiles (id, display_name, icon, sensors, states) |
| GET | `` (list) | any | Current running instances + per-instance stats |
| POST | `` | **admin** | Create + start instance |
| PATCH | `/<id>` | **admin** | Change state |
| DELETE | `/<id>` | **admin** | Stop + remove instance |
| POST | `/publish-raw` | **admin** | One-shot arbitrary publish (topic + payload) |
| GET | `/snapshot` | any | Global stats — total msg/s, instance count, chaos events |

Create payload:
```json
{
  "profile_id": "air_compressor",
  "name": "compressor_01",
  "topic_base": "factory/plant_A/compressor_01",
  "initial_state": "running",
  "autoprovision_tree": true
}
```

Auto-provision behaviour: on create with `autoprovision_tree: true`, backend walks `topic_base` segments and creates any missing asset nodes (via existing `AssetNode.create` + same validation as `POST /api/asset-tree/nodes`). Sensors created as level-N children (leaf) with `unit` + `sample_rate_hz` from the profile. Uses the current `AssetTreeConfig` root check — if `topic_base` doesn't start with `<root_name>/...`, returns 400.

### 4.3 Boot integration

Same pattern as `MqttIngestRouter`:
- `backend/app/__init__.py`: after `MqttIngestRouter` boot, call `machine_simulator.start(app)` behind the same `WERKZEUG_RUN_MAIN` gate
- `atexit` hook via `machine_simulator.stop()`

## 5. Frontend

### 5.1 New view: `frontend/src/views/SimulatorsView.vue`

- Header bar: total msg/s ticker · instance count · **Add machine** button · **Stop all** button (admin only)
- Grid of `SimulatorCard.vue` — one per running instance
- Bottom section: `RawPublishWidget.vue` (admin only) — topic + payload textarea + Fire button

### 5.2 `frontend/src/components/SimulatorCard.vue`

Per-card contents:
- Profile icon + name + topic_base (monospace)
- State dropdown (populated from profile.states, current highlighted)
- Start/stop toggle (admin only)
- Live msg/s + uptime
- Per-sensor mini-sparklines (Chart.js `chart-sparkline`, last 60 samples, updated every 500 ms via WebSocket or 1 s poll — pick poll for simplicity)
- Delete button (admin only, with confirm)

### 5.3 `frontend/src/components/SimulatorNewDialog.vue`

Steps:
1. Profile picker (grid of 6 cards with icon + description)
2. Instance name + topic base (prefilled from asset tree if a machine is selected in sidebar)
3. Initial state dropdown
4. Autoprovision checkbox (default ON)
5. Preview: shows which sensors will be created + at what rate

Validation: name must match `^[a-zA-Z0-9_-]+$`, topic_base must start with `<root_name>/`, name must be unique among running instances.

### 5.4 Router entry

`frontend/src/router/index.ts`:
```typescript
{
  path: '/global/simulators',
  name: 'simulators',
  component: () => import('@/views/SimulatorsView.vue'),
  meta: { requiresAuth: true },
}
```

## 6. Sidebar reorganization (Phase F.7)

### 6.1 Renames

| Was | Now | Where |
|---|---|---|
| "Asset Tree" (section header above tree) | *(dropped entirely — redundant with tree itself)* | Sidebar top |
| "Asset Tree" (Settings menu item) | **"Factory Setup"** (or dynamic per root name — see 6.2) | Settings section |
| "Asset Tree Configuration" (page H1) | **"Factory Setup"** (same source as menu label) | `/asset-tree-admin` view |

### 6.2 Dynamic root-name label

Menu label + page title read from `assetTreeStore.config.root_name`:

```typescript
const rootSetupLabel = computed(() => {
  const root = assetTreeStore.config?.root_name
  if (!root) return 'Structure Setup'
  return `${root.charAt(0).toUpperCase() + root.slice(1)} Setup`
})
```

Examples: `factory` → **"Factory Setup"**, `stores` → **"Stores Setup"**, `site` → **"Site Setup"**, `hospital` → **"Hospital Setup"**.

### 6.3 Legacy tools group

New `SidebarLegacyGroup.vue` at the bottom of the sidebar:
- Header row: `▸ 🕐 Legacy tools` (click to toggle)
- Default: collapsed
- Persisted state key: `cira.sidebar.legacyExpanded` (boolean, default `false`)
- Contains: `📁 Projects [legacy]` + `📊 Dashboard [legacy]`
- Chip color: muted grey (`text-medium-emphasis`), not error

### 6.4 New "Machine Simulators" entry

Under `GLOBAL TOOLS` section, right after `Data Source`:
- Icon: `mdi-gauge` or `mdi-tune-vertical`
- Label: `Machine Simulators`
- Route: `/global/simulators`

### 6.5 Section dividers

Each group (Asset Tree · Global Tools · Settings · Legacy) gets a thin `<v-divider>` above its header for visual rhythm.

## 7. Deliverables (subtasks)

| ID | Owner | Description |
|---|---|---|
| F.1 | backend | `machine_profiles.py` — 6 profiles, all states, signal generators |
| F.2 | backend | `machine_simulator.py` service — thread-per-machine, shared paho, chaos ticks |
| F.3 | backend | `routes/simulators.py` — 7 endpoints, admin gating, auto-provision |
| F.4 | frontend | Sidebar reorg — legacy group, renames, dynamic label, dividers |
| F.5 | frontend | `SimulatorsView.vue` + `SimulatorCard.vue` + `SimulatorNewDialog.vue` |
| F.6 | frontend | Sparklines + raw-publish widget + chaos-event visualisation |
| F.QA | agent | Adversarial full-stack QA pass |
| F.T | user | Personal browser test + Phase D reject-path verification |

## 8. Edge cases + failure modes to test

- Simulator with `topic_base` outside `<root_name>/` — must 400
- Auto-provision when tree segment already retired — must 400 (learn-mode router already handles this)
- Delete simulator whose tree entries had multiple simulators publishing to them — other simulators unaffected
- Broker down at simulator create — instance starts in `_connected=False`, reconnects when broker returns
- Backend restart with 5 simulators running — all lost, no crash on boot
- Chaos state on a machine whose tree isn't auto-provisioned — rejections written to `/rejected-topics` log (that's the point)
- 20 simulators × 5 Hz vibration each = 100 msg/s — smoke test this doesn't OOM
- Two simulators same name — second create 409s
- PATCH state to unknown state name — 400
- Non-admin POST/PATCH/DELETE — 403
- Non-admin GET /snapshot — 200

## 9. What's explicitly out of scope

- Persistence across backend restart (would require DB table + resume logic; explicitly deferred per decision table)
- Custom user-defined profiles (v1 ships the 6 built-in; user-defined is a v2 conversation)
- Simulating multiple machines under one card (v1 = one card = one machine)
- Historical replay from a CSV (that's what folder watcher already does)
- CPU/memory limits per simulator (rely on the ~10-15 practical cap; add a hard cap if we see issues)

## 10. Definition of done

- All 6 profiles publish sensible signals in all their states
- Auto-provision creates the tree nodes + ingest router routes the resulting messages
- CSV files appear under `datasets/<topic_path>/` for each simulated sensor
- Chaos state produces visible entries in `/rejected-topics`
- Sidebar renamed, Legacy group collapses/persists, Machine Simulators reachable
- Zero blockers from QA
- Personal browser test passes end-to-end
