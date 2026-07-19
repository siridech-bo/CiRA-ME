"""CiRA ME — Machine Simulator Service (Phase F, 2026-07-19).

Server-side signal generator that publishes realistic sensor data to
`cirame-mosquitto` under user-chosen topic paths so demos/workshops
can exercise the full MQTT ingest → CSV → training pipeline without
real hardware.

Mirrors the structure of `services/mqtt_ingest_router.py`:
- One shared paho client (client_id `cira-simulator-<pid>`) with a
  connect-retry loop that never blocks Flask startup.
- One daemon thread per simulated machine (`_SimulatedMachine`) that
  ticks at `min(sensor.sample_rate_hz)` and publishes each sensor
  according to its individual rate.
- Chaos state additionally schedules poison-message publishes every
  20-40 s to exercise the ingest router's rejection paths.

Failure policy: every thread's run-loop is wrapped in
`try/except Exception: log; continue`. This service must NEVER crash
the backend — a broken simulator is a demo problem, not an outage.

Public surface (used by routes/simulators.py):
  machine_simulator                     — module singleton
  machine_simulator.start(flask_app)    — boot, no-op if running
  machine_simulator.stop()              — atexit hook
  machine_simulator.list_profiles()     — profile catalog
  machine_simulator.list_instances()    — running instances + stats
  machine_simulator.create_instance(**) — spawn a new simulator
  machine_simulator.patch_state(...)    — swap a simulator's state
  machine_simulator.delete_instance(id) — stop + remove
  machine_simulator.publish_raw(...)    — one-shot arbitrary publish
  machine_simulator.snapshot()          — global stats for the UI header
"""

import atexit
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..constants.machine_profiles import (
    MachineProfile, get_all_profiles, get_profile, sample as sample_signal,
)

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────
# Chaos poison-message cadence (uniform random each cycle).
_CHAOS_MIN_S = 20.0
_CHAOS_MAX_S = 40.0
# How many recent samples per sensor we retain for sparkline rendering.
# 60 points × 500 ms poll = 30 s of history in the UI.
_RECENT_MAXLEN = 60
# Connect / reconnect backoff.
_RECONNECT_MIN_S = 2.0
_RECONNECT_MAX_S = 60.0
# Regex mirror of routes/asset_tree._sanitize_name so an
# auto-provisioned segment can never fail the tree-side check.
_NAME_REGEX = re.compile(r'^[A-Za-z0-9_-]+$')


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _monotonic_now() -> float:
    return time.monotonic()


class _SimulatedMachine(threading.Thread):
    """One thread per simulated machine.

    Ticks at min(sensor.sample_rate_hz) for the profile. Each tick,
    checks per-sensor last-published timestamps and publishes any
    sensors whose next-due time has arrived. Publish payload is
    `{"value": <float>}` on topic `<topic_base>/<sensor_name>`.

    Chaos: if `current_state == 'chaos'`, in addition to normal signal
    publishes the thread schedules ONE poison message every
    random 20-40 s. Poison messages cycle through: null value, garbage
    bytes, wrong-root topic, and unknown-plant topic (see `_do_chaos`).
    """

    def __init__(
        self,
        parent: 'MachineSimulator',
        instance_id: str,
        profile: MachineProfile,
        name: str,
        topic_base: str,
        initial_state: str,
    ):
        super().__init__(name=f'sim-{name}', daemon=True)
        self._parent = parent
        self.instance_id = instance_id
        self.profile = profile
        self.name = name
        self.topic_base = topic_base
        self.state = initial_state
        self.created_at = _iso_now()
        self.state_since_ts = self.created_at

        # Per-sensor next-publish deadline (monotonic seconds). Prime so
        # the first tick publishes every sensor.
        self._next_due: Dict[str, float] = {
            s.name: 0.0 for s in profile.sensors
        }
        # Ring buffers for sparklines. Populated on every publish.
        self._recent_values: Dict[str, deque] = {
            s.name: deque(maxlen=_RECENT_MAXLEN) for s in profile.sensors
        }
        # Stats.
        self.messages_published = 0
        self.chaos_events = 0
        # Next chaos deadline (monotonic seconds).
        self._next_chaos = _monotonic_now() + random.uniform(
            _CHAOS_MIN_S, _CHAOS_MAX_S,
        )

        self._stop_flag = threading.Event()
        self._lock = threading.RLock()

    # ── State control ────────────────────────────────────────────────

    def set_state(self, new_state: str) -> None:
        if new_state not in self.profile.states:
            raise ValueError(
                f'Unknown state {new_state!r} for profile {self.profile.id!r}'
            )
        with self._lock:
            self.state = new_state
            self.state_since_ts = _iso_now()
            # Reset chaos cadence on state change so entering chaos
            # doesn't fire instantly (or feel stuck if we just left it).
            self._next_chaos = _monotonic_now() + random.uniform(
                _CHAOS_MIN_S, _CHAOS_MAX_S,
            )

    def stop(self) -> None:
        self._stop_flag.set()

    # ── Snapshot for UI ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            recent = {
                name: [round(v, 4) for v in list(buf)]
                for name, buf in self._recent_values.items()
            }
            return {
                'id': self.instance_id,
                'profile_id': self.profile.id,
                'profile_display_name': self.profile.display_name,
                'profile_icon': self.profile.icon,
                'name': self.name,
                'topic_base': self.topic_base,
                'state': self.state,
                'states': sorted(list(self.profile.states.keys())),
                'sensors': [
                    {
                        'name': s.name,
                        'unit': s.unit,
                        'sample_rate_hz': s.sample_rate_hz,
                    }
                    for s in self.profile.sensors
                ],
                'messages_published': self.messages_published,
                'chaos_events': self.chaos_events,
                'created_at': self.created_at,
                'state_since_ts': self.state_since_ts,
                'recent_values': recent,
                'alive': self.is_alive(),
            }

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901 — one flat loop is easier to read
        # Tick rate: at least fast enough for the highest-rate sensor.
        max_rate = max((s.sample_rate_hz for s in self.profile.sensors),
                       default=1.0)
        tick_s = max(0.02, 1.0 / max(max_rate, 0.1))
        # PHASE-F-FOLLOWUP: could adapt tick_s per state (silent states
        # need no cadence), but per-sensor `_next_due` already skips the
        # publish so wasted CPU is minimal.
        start_time = _monotonic_now()
        while not self._stop_flag.is_set():
            loop_start = _monotonic_now()
            try:
                self._tick(loop_start - start_time)
            except Exception:
                # Hard requirement: never crash. Log + carry on.
                logger.exception(
                    '[sim:%s] tick crashed — swallowing to keep thread alive',
                    self.name,
                )
            # Sleep in short chunks so stop is responsive.
            elapsed = _monotonic_now() - loop_start
            remaining = tick_s - elapsed
            while remaining > 0 and not self._stop_flag.is_set():
                nap = min(0.05, remaining)
                time.sleep(nap)
                remaining -= nap

    def _tick(self, t_elapsed: float) -> None:
        """One tick: publish any sensor whose next-due time is <= now, plus
        run the chaos scheduler."""
        with self._lock:
            state_name = self.state
            state_params = self.profile.states.get(state_name)
        now = _monotonic_now()

        # Signal publishes — every state (including chaos) emits clean signals.
        for sensor in self.profile.sensors:
            due = self._next_due.get(sensor.name, 0.0)
            if now < due:
                continue
            # Compute value; None means silent this state.
            value = sample_signal(state_params, sensor.name, t_elapsed)
            # Schedule the next publish based on nominal rate. Doing it
            # BEFORE the publish keeps cadence stable if publish blocks.
            period = 1.0 / max(sensor.sample_rate_hz, 0.05)
            self._next_due[sensor.name] = now + period
            if value is None:
                continue  # silent state
            self._publish_signal(sensor.name, value)

        # Chaos scheduler — only triggers when in the chaos state, but the
        # timer counts regardless so entering chaos doesn't fire instantly.
        if now >= self._next_chaos:
            self._next_chaos = now + random.uniform(_CHAOS_MIN_S, _CHAOS_MAX_S)
            if state_name == 'chaos':
                try:
                    self._do_chaos()
                except Exception:
                    logger.exception('[sim:%s] chaos step failed', self.name)

    def _publish_signal(self, sensor_name: str, value: float) -> None:
        """Publish `{"value": <float>}` on `<topic_base>/<sensor_name>`."""
        if not self._parent._connected:
            # Broker down — spec says drop rather than buffer.
            return
        topic = f'{self.topic_base}/{sensor_name}'
        payload = json.dumps({'value': round(float(value), 6)})
        try:
            self._parent._publish_bytes(topic, payload.encode('utf-8'))
        except Exception:
            logger.warning('[sim:%s] publish failed on %s', self.name, topic,
                           exc_info=True)
            return
        self.messages_published += 1
        with self._lock:
            self._recent_values[sensor_name].append(float(value))
        with self._parent._stats_lock:
            self._parent._stats['messages_published'] += 1

    def _do_chaos(self) -> None:
        """Fire ONE poison message. Rotates through 4 flavours."""
        # Pick a real sensor name from this profile — hard-coding
        # 'temperature' meant only 1 of 6 profiles ever hit the intended
        # payload-parse paths (QA F.QA polish #3).
        sensor_name = random.choice(self.profile.sensors).name
        variant = random.choice(('null', 'garbage', 'wrong_root', 'unknown_plant'))
        if variant == 'null':
            topic = f'{self.topic_base}/{sensor_name}'
            payload = b'{"value": null}'
        elif variant == 'garbage':
            # Garbage payload on a legit sensor — exercises payload-parse
            # failure rather than topic rejection.
            topic = f'{self.topic_base}/{sensor_name}'
            payload = b'\x00\x01\xff\xfeGARBAGE'
        elif variant == 'wrong_root':
            topic = f'wrong_root/{self.name}/{sensor_name}'
            payload = b'{"value": 1.23}'
        else:  # unknown_plant
            # Splice an unknown-plant segment. If topic_base is
            # `factory/plant_A/compressor_test` this becomes
            # `factory/plant_XX_nonexistent/compressor_test/<sensor>`.
            segments = self.topic_base.split('/')
            if len(segments) >= 2:
                mangled = segments[0] + '/plant_XX_nonexistent/' + '/'.join(segments[1:])
            else:
                mangled = segments[0] + '/plant_XX_nonexistent'
            topic = f'{mangled}/{sensor_name}'
            payload = b'{"value": 4.56}'

        if not self._parent._connected:
            return
        try:
            self._parent._publish_bytes(topic, payload)
            self.chaos_events += 1
            with self._parent._stats_lock:
                self._parent._stats['chaos_events'] += 1
        except Exception:
            logger.warning('[sim:%s] chaos publish failed on %s',
                           self.name, topic, exc_info=True)


# ── Singleton ────────────────────────────────────────────────────────────


class MachineSimulator:
    """Spawns and manages `_SimulatedMachine` threads.

    Owns exactly one paho client that all machines share for publishing.
    """

    def __init__(self):
        self._flask_app = None
        self._client = None
        self._connected = False
        self._broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
        self._broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

        # instance_id → _SimulatedMachine
        self._instances: Dict[str, _SimulatedMachine] = {}
        self._instances_lock = threading.RLock()

        # Global stats.
        self._stats = {
            'messages_published': 0,
            'chaos_events': 0,
            'connect_attempts': 0,
            'reconnects': 0,
            'started_at': None,
            'last_connected_at': None,
        }
        self._stats_lock = threading.Lock()

        self._stop_flag = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None

    # ── Boot / shutdown ──────────────────────────────────────────────

    def start(self, flask_app=None) -> None:
        """Boot the shared paho client. Idempotent."""
        if self._connect_thread and self._connect_thread.is_alive():
            return
        self._flask_app = flask_app
        self._stop_flag.clear()
        with self._stats_lock:
            self._stats['started_at'] = _iso_now()
        self._connect_thread = threading.Thread(
            target=self._connect_loop,
            name='sim-mqtt-connect',
            daemon=True,
        )
        self._connect_thread.start()
        atexit.register(self.stop)
        logger.info('[sim] service started (broker=%s:%s)',
                    self._broker_host, self._broker_port)

    def stop(self, timeout_s: float = 2.0) -> None:
        if self._stop_flag.is_set():
            return
        self._stop_flag.set()
        # Stop every machine thread.
        with self._instances_lock:
            for m in list(self._instances.values()):
                try:
                    m.stop()
                except Exception:
                    pass
            self._instances = {}
        # Disconnect the shared client.
        try:
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass
        # Join connect thread.
        if self._connect_thread and self._connect_thread.is_alive():
            try:
                self._connect_thread.join(timeout_s)
            except Exception:
                pass

    # ── Public API ───────────────────────────────────────────────────

    def list_profiles(self) -> List[dict]:
        return [p.to_dict() for p in get_all_profiles()]

    def list_instances(self) -> List[dict]:
        with self._instances_lock:
            return [m.to_dict() for m in self._instances.values()]

    def create_instance(
        self,
        profile_id: str,
        name: str,
        topic_base: str,
        initial_state: Optional[str] = None,
        autoprovision_tree: bool = False,
        actor_user_id: int = 0,
    ) -> dict:
        """Spawn a new simulator. Returns the created instance's dict.

        Raises ValueError with a HTTP-safe message on validation failure.
        Callers (routes/simulators.py) map ValueError → 400.
        """
        profile = get_profile(profile_id)
        if profile is None:
            raise ValueError(f'Unknown profile_id: {profile_id!r}')
        if not name or not _NAME_REGEX.match(name):
            raise ValueError("name must match ^[A-Za-z0-9_-]+$")
        if not topic_base or not isinstance(topic_base, str):
            raise ValueError('topic_base (string) required')
        topic_base = topic_base.strip().strip('/')
        if not topic_base:
            raise ValueError('topic_base cannot be empty')
        # Every segment topic-safe.
        for seg in topic_base.split('/'):
            if not _NAME_REGEX.match(seg):
                raise ValueError(
                    f"topic_base segment {seg!r} must match ^[A-Za-z0-9_-]+$"
                )
        state = initial_state or profile.default_state
        if state not in profile.states:
            raise ValueError(
                f'initial_state {state!r} not in profile {profile_id!r}'
            )

        # Name uniqueness across running instances.
        with self._instances_lock:
            for m in self._instances.values():
                if m.name == name:
                    raise ValueError(f"A simulator named {name!r} is already running")

        # Root check — topic_base must start with the configured root
        # else ingest router will reject every message we produce.
        from ..models import AssetTreeConfig
        cfg = AssetTreeConfig.get() or {}
        root_name = cfg.get('root_name')
        # Refuse to run against a half-configured tree — silent no-route is
        # worse than a clear error. (QA F.QA polish #6.)
        if not root_name:
            raise ValueError(
                "Asset tree root_name is not set — run the setup wizard "
                "before starting simulators"
            )
        if topic_base.split('/', 1)[0] != root_name:
            raise ValueError(
                f"topic_base must start with '{root_name}/' "
                f"(current asset-tree root); got '{topic_base}'"
            )

        # Auto-provision tree segments BEFORE we start the thread so early
        # ticks don't hit the rejection log while nodes are being created.
        created_node_ids: List[int] = []
        if autoprovision_tree:
            created_node_ids = self._autoprovision_tree(
                topic_base=topic_base,
                profile=profile,
                actor_user_id=actor_user_id,
            )
            # Nudge the ingest router so it picks up the new nodes.
            self._reload_ingest_router()

        instance_id = uuid.uuid4().hex[:12]
        machine = _SimulatedMachine(
            parent=self,
            instance_id=instance_id,
            profile=profile,
            name=name,
            topic_base=topic_base,
            initial_state=state,
        )
        with self._instances_lock:
            self._instances[instance_id] = machine
        machine.start()
        logger.info(
            '[sim] created instance %s (profile=%s, name=%s, topic_base=%s, '
            'state=%s, autoprovision_nodes=%d)',
            instance_id, profile_id, name, topic_base, state,
            len(created_node_ids),
        )
        result = machine.to_dict()
        result['autoprovisioned_node_ids'] = created_node_ids
        return result

    def patch_state(self, instance_id: str, new_state: str,
                    actor_user_id: int = 0) -> dict:
        with self._instances_lock:
            m = self._instances.get(instance_id)
        if m is None:
            raise KeyError(f'Instance {instance_id!r} not found')
        m.set_state(new_state)  # raises ValueError on unknown state
        logger.info('[sim] instance %s → state=%s (actor=%s)',
                    instance_id, new_state, actor_user_id)
        return m.to_dict()

    def delete_instance(self, instance_id: str,
                        actor_user_id: int = 0) -> None:
        with self._instances_lock:
            m = self._instances.pop(instance_id, None)
        if m is None:
            raise KeyError(f'Instance {instance_id!r} not found')
        try:
            m.stop()
        except Exception:
            pass
        # Best-effort join so the thread has a chance to exit cleanly.
        try:
            m.join(0.5)
        except Exception:
            pass
        logger.info('[sim] deleted instance %s (name=%s, actor=%s)',
                    instance_id, m.name, actor_user_id)

    def publish_raw(self, topic: str, payload_bytes: bytes,
                    actor_user_id: int = 0) -> None:
        """One-shot arbitrary publish. Used by the raw-publish widget."""
        if not topic or not isinstance(topic, str):
            raise ValueError('topic (string) required')
        if not self._connected:
            raise RuntimeError('MQTT broker not connected')
        if not isinstance(payload_bytes, (bytes, bytearray)):
            payload_bytes = str(payload_bytes).encode('utf-8', errors='replace')
        self._publish_bytes(topic, bytes(payload_bytes))
        logger.info('[sim] raw publish → %s (%d bytes, actor=%s)',
                    topic, len(payload_bytes), actor_user_id)

    def snapshot(self) -> dict:
        with self._stats_lock:
            data = dict(self._stats)
        with self._instances_lock:
            instance_count = len(self._instances)
            per_instance = [m.to_dict() for m in self._instances.values()]
        data.update({
            'connected': self._connected,
            'broker_host': self._broker_host,
            'broker_port': self._broker_port,
            'instance_count': instance_count,
            'instances': per_instance,
        })
        return data

    # ── Internals ────────────────────────────────────────────────────

    def _publish_bytes(self, topic: str, payload: bytes) -> None:
        """Direct paho publish. Callers must check `_connected` first."""
        if self._client is None:
            return
        # paho publish returns MQTTMessageInfo; we don't wait for QoS 1 ack.
        # QoS 0 is fine for sim data — a lost message is a demo artefact.
        self._client.publish(topic, payload, qos=0)

    def _connect_loop(self) -> None:
        """Connect + auto-reconnect. Modeled on MqttIngestRouter."""
        import paho.mqtt.client as paho_mqtt

        backoff = _RECONNECT_MIN_S
        while not self._stop_flag.is_set():
            try:
                with self._stats_lock:
                    self._stats['connect_attempts'] += 1
                self._client = paho_mqtt.Client(
                    paho_mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f'cira-simulator-{os.getpid()}',
                    clean_session=True,
                )
                self._client.on_connect = self._on_connect
                self._client.on_disconnect = self._on_disconnect
                self._client.connect(self._broker_host, self._broker_port,
                                     keepalive=60)
                self._client.loop_start()
                # Wait up to 5 s for on_connect to fire.
                deadline = time.monotonic() + 5.0
                while (not self._connected
                       and time.monotonic() < deadline
                       and not self._stop_flag.is_set()):
                    time.sleep(0.1)
                if not self._connected:
                    raise TimeoutError('on_connect did not fire within 5 s')
                backoff = _RECONNECT_MIN_S
                # Poll until disconnect or stop.
                while not self._stop_flag.is_set() and self._connected:
                    time.sleep(0.5)
                if self._stop_flag.is_set():
                    return
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except Exception:
                    pass
                with self._stats_lock:
                    self._stats['reconnects'] += 1
            except Exception as e:
                logger.warning('[sim] broker connect failed: %s', e)
                try:
                    self._client and self._client.loop_stop()
                except Exception:
                    pass
                self._connected = False
            # Backoff.
            for _ in range(int(backoff * 2)):
                if self._stop_flag.is_set():
                    return
                time.sleep(0.5)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        try:
            if rc == 0:
                self._connected = True
                with self._stats_lock:
                    self._stats['last_connected_at'] = _iso_now()
                logger.info('[sim] connected to %s:%s',
                            self._broker_host, self._broker_port)
            else:
                logger.warning('[sim] on_connect rc=%s', rc)
                self._connected = False
        except Exception:
            logger.exception('[sim] on_connect failed')

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        self._connected = False
        logger.info('[sim] disconnected from broker')

    # ── Auto-provision (asset-tree nodes) ────────────────────────────

    def _autoprovision_tree(
        self,
        topic_base: str,
        profile: MachineProfile,
        actor_user_id: int,
    ) -> List[int]:
        """Ensure every segment of `topic_base` exists as an asset node, plus
        one child sensor node per profile sensor. Skips segments that are
        already active. Raises ValueError if any segment is retired or
        would exceed configured max depth.

        Returns list of newly-created node ids (empty if all pre-existed).
        """
        from ..models import (
            AssetTreeConfig, AssetNode, AssetSensorMeta, AssetTreeAudit,
        )

        cfg = AssetTreeConfig.get()
        if not cfg:
            raise ValueError(
                'Asset tree not yet configured — set up the tree first'
            )
        level_names = cfg.get('level_names') or []
        # sensor level index = last level; parent-of-sensor = max_depth-1
        # (this matches routes/asset_tree._max_depth semantics).
        max_depth = max(0, len(level_names) - 1)

        segments = topic_base.split('/')
        # topic_base must sit exactly one level above the sensor leaf so the
        # profile's sensors autoprovision at the correct depth. Reject too-
        # short (would mount sensors above the machine level, polluting the
        # taxonomy) and too-deep (would leave the sim publishing forever to
        # unroutable paths). (QA F.QA polish #1 + #2.)
        machine_level = max(0, len(segments) - 1)
        if machine_level != max_depth - 1:
            expected_segments = max_depth  # segments count = level index + 1
            raise ValueError(
                f'topic_base must have exactly {expected_segments} segments '
                f'(one per level, ending at the machine level); got '
                f'{len(segments)} segments in {topic_base!r}. Level names: '
                f'{level_names!r}.'
            )
        created_ids: List[int] = []

        # Walk each segment: reuse existing active nodes; create missing.
        parent_id: Optional[int] = None
        for level, seg in enumerate(segments):
            if level > max_depth:
                raise ValueError(
                    f'topic_base exceeds configured max depth {max_depth} '
                    f'(segments={len(segments)})'
                )
            if not _NAME_REGEX.match(seg):
                raise ValueError(f"segment {seg!r} must match ^[A-Za-z0-9_-]+$")
            path = '/'.join(segments[:level + 1])
            existing = AssetNode.get_by_topic_path(path)
            if existing is not None:
                if existing.get('status') == 'retired':
                    raise ValueError(
                        f"segment {seg!r} ({path}) is retired — cannot "
                        f"auto-provision through retired nodes"
                    )
                parent_id = existing['id']
                continue
            # Create.
            node_id = AssetNode.create(
                parent_id=parent_id, level=level, name=seg,
                topic_path=path, status='active',
            )
            created_ids.append(node_id)
            parent_id = node_id

        # `parent_id` now points at the machine (the level check above
        # guarantees machine_level == max_depth - 1, so sensor children go
        # at max_depth exactly).
        for sensor in profile.sensors:
            sensor_path = f'{topic_base}/{sensor.name}'
            existing = AssetNode.get_by_topic_path(sensor_path)
            if existing is not None:
                # Already there (active or retired) — leave it alone.
                # Retired sensors will drop messages; the operator can
                # un-retire from the tree admin if that's a mistake.
                continue
            sensor_id = AssetNode.create(
                parent_id=parent_id,
                level=max_depth,
                name=sensor.name,
                topic_path=sensor_path,
                status='active',
            )
            created_ids.append(sensor_id)
            # Attach sensor meta so the tree UI shows unit + rate.
            try:
                AssetSensorMeta.upsert(
                    asset_id=sensor_id,
                    unit=sensor.unit,
                    sample_rate_hz=sensor.sample_rate_hz,
                    data_type='float',
                )
            except Exception:
                logger.warning(
                    '[sim] failed to attach sensor meta to %s',
                    sensor_path, exc_info=True,
                )

        if created_ids:
            try:
                AssetTreeAudit.log(
                    actor_user_id=actor_user_id or 0,
                    event_type='simulator_autoprovision',
                    target_type='node',
                    target_id=created_ids[-1],
                    payload={
                        'topic_base': topic_base,
                        'profile_id': profile.id,
                        'created_ids': created_ids,
                        'actor': 'machine_simulator',
                    },
                )
            except Exception:
                logger.warning('[sim] audit log failed', exc_info=True)

        return created_ids

    def _reload_ingest_router(self) -> None:
        """Synchronous cache refresh on the ingest router so auto-provisioned
        paths route immediately. Called BEFORE the sim thread starts so the
        first tick doesn't race into an unknown-topic rejection."""
        try:
            from .mqtt_ingest_router import router as _ingest_router
            # sync=True — cheap DB read, and we need the cache primed
            # before the sim thread emits its first publish.
            _ingest_router.reload_tree(sync=True)
        except Exception:
            logger.warning('[sim] ingest router reload failed', exc_info=True)


# ── Module singleton ─────────────────────────────────────────────────────
machine_simulator = MachineSimulator()
