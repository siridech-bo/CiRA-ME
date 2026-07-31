"""CiRA ME — MQTT Ingest Router (Phase D, 2026-07-19).

Background service that owns a paho-mqtt client subscribed to `#` on the
Mosquitto broker, routes incoming messages by topic path to the on-disk
asset tree, and rotates one CSV per sensor per day. Strictly additive —
does not touch any existing endpoint or scheduler.

Design plan   : docs/PLAN_2026-07-18_asset-tree.md §12
Execution plan: docs/EXECUTION_asset-tree.md — subtasks D.1–D.5

Public surface (used by routes/asset_tree.py):
  router                       — module singleton, boot from create_app
  router.start(flask_app)      — spawn connect + writer + janitor threads
  router.stop()                — atexit hook; safe to call more than once
  router.reload_tree()         — fire-and-forget cache refresh
  router.snapshot()            — dict for the Stats tab
  router.run_janitor_once()    — synchronous sweep, used by /run-now
  router.list_rejected(date, limit) — for the Rejected-topics viewer

Failure policy: every path swallows exceptions and logs. The router is a
side-quest; it must never crash the backend.
"""

import atexit
import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..models import (
    AssetTreeConfig, AssetNode, AssetSensorMeta, AssetTreeAudit, get_db,
)

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────
# How often the writer's flush loop wakes up. 200 ms matches the spec cap.
_FLUSH_INTERVAL_S = 0.2
# Force-flush after this many messages regardless of flush-interval timing.
_FLUSH_BATCH = 100
# Retention janitor cadence. 6 h matches the spec.
_JANITOR_INTERVAL_S = 6 * 60 * 60
# Rejected-topic log ring buffer (in-memory tail so the UI's Rejected tab
# renders instantly). Persistent-on-disk log is always the source of truth.
_REJECTED_TAIL_SIZE = 500
# Connect / reconnect backoff. Doubles each miss up to the ceiling.
_RECONNECT_MIN_S = 2.0
_RECONNECT_MAX_S = 60.0


def _iso_now() -> str:
    """UTC ISO timestamp with 'Z' suffix (matches the sensor recorder)."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _parse_iso_utc(s: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp into an aware UTC datetime.

    Accepts the router's own `_iso_now()` output (`Z`-suffixed millisecond
    precision) as well as bare `YYYY-MM-DDTHH:MM:SS[.fff][+HH:MM|Z]`. Returns
    None on any parse failure so callers can fail-open. Kept module-private
    because it's a one-line helper we don't want to leak.
    """
    if not s or not isinstance(s, str):
        return None
    raw = s.strip()
    if not raw:
        return None
    # Python's fromisoformat prior to 3.11 doesn't understand 'Z' suffix.
    if raw.endswith('Z'):
        raw = raw[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MqttIngestRouter:
    """Subscribes to `#`, routes by topic_path, buffers per-sensor CSVs.

    Not thread-safe to instantiate multiple times per process. Use the
    module-level `router` singleton.
    """

    def __init__(self):
        # ── Configuration snapshot (refreshed on reload_tree / connect) ──
        # All values are cached on a background thread and read from the
        # message handler thread without a lock (they're plain immutable
        # scalars / frozensets after assignment). Reads use local names
        # to avoid TOCTOU during a mid-flight reload.
        self._enabled = False
        self._topic_mode = 'strict'
        self._root_name: Optional[str] = None
        self._meta_prefixes: frozenset = frozenset()
        # topic_path (str) → node dict. Populated from asset_nodes on
        # boot and on every reload_tree() call.
        self._path_cache: Dict[str, dict] = {}
        self._cache_lock = threading.RLock()

        # ── MQTT client + broker settings ────────────────────────────────
        # Client instantiated inside start() so tests can preempt.
        self._client = None
        self._broker_host = os.environ.get('MQTT_BROKER_HOST', 'cirame-mosquitto')
        self._broker_port = int(os.environ.get('MQTT_BROKER_PORT', '1883'))

        # ── Writer buffer + rotation state ──────────────────────────────
        # { topic_path: [(iso_ts, value_str_or_None), ...] }
        self._buffer: Dict[str, list] = {}
        self._buffer_lock = threading.RLock()
        self._buffer_count = 0
        # header-written cache so we don't stat() the same file every flush.
        self._headers_written: set = set()
        # Currently open date (UTC). Reset triggers a per-file rollover.
        self._current_date = _today_utc()

        # ── Rejected-topic tail (in-memory) ──────────────────────────────
        self._rejected_tail: deque = deque(maxlen=_REJECTED_TAIL_SIZE)
        self._rejected_lock = threading.Lock()

        # ── Stats counters ───────────────────────────────────────────────
        # Bare ints are atomic in CPython, but count updates happen in the
        # paho callback thread so we still take the lock for read/write
        # symmetry against snapshot().
        self._stats = {
            'messages_received': 0,
            'messages_routed': 0,
            'messages_rejected': 0,
            'messages_meta': 0,
            'messages_parse_errors': 0,
            'files_written': 0,
            # Phase H — multi-axis stats. `channels_written` counts total
            # (rows × channels) written on multi-axis topics so operators
            # can see the fan-out volume separately from `messages_routed`
            # (which stays a per-message counter).
            'channels_written': 0,
            # Phase I Q3 — messages that passed routing but were suppressed
            # by per-sensor recording controls (ingest_enabled=False, or
            # min_write_interval throttle, or record_until expiry). Silent
            # drop; NOT counted in messages_rejected because the topic is
            # legitimate — we just chose not to persist the row.
            'messages_ingest_skipped': 0,
            'last_message_at': None,
            'last_message_topic': None,
            'last_connected_at': None,
            'connect_attempts': 0,
            'reconnects': 0,
            'started_at': None,
        }
        self._stats_lock = threading.Lock()

        # ── Phase J — per-topic last-seen timestamps ────────────────────
        # Two parallel maps: monotonic seconds for the LRU purge (immune to
        # wall-clock jumps) and real-time epoch seconds for the API's ISO
        # 8601 formatting. Kept in sync on every _route call.
        #   monotonic  → used ONLY for internal ordering / LRU
        #   epoch      → used ONLY for API response formatting
        # Capped at ~500 entries via a lazy LRU purge triggered every
        # 100 writes (see `_maybe_purge_topic_last_seen`). Hot path is a
        # single dict assign; the sort only runs on the 100th write.
        self._topic_last_seen: Dict[str, float] = {}
        self._topic_last_seen_epoch: Dict[str, float] = {}
        self._topic_last_seen_lock = threading.Lock()
        self._topic_last_seen_write_counter = 0

        # ── Phase I Q3 — per-topic throttle state ────────────────────────
        # topic_path → time.monotonic() timestamp of the last accepted write.
        # Compared against sensor's `min_write_interval_ms` to enforce
        # sample-rate throttling. Cleaned lazily — retired sensors leave a
        # stale entry until the next `_refresh_cache_and_config` cycle,
        # which is fine (bounded size = cache size).
        self._last_write_ts: Dict[str, float] = {}
        self._last_write_lock = threading.Lock()
        # Guards `_schedule_auto_stop` re-entry so a burst of messages after
        # a record_until expiry doesn't spawn N duplicate DB updates before
        # the first one lands. Cleared inside the thread's finally clause.
        self._auto_stop_pending: set = set()
        self._auto_stop_lock = threading.Lock()

        # ── Threads / shutdown flag ─────────────────────────────────────
        self._stop_flag = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._janitor_thread: Optional[threading.Thread] = None
        self._reload_executor_lock = threading.Lock()

        self._flask_app = None
        self._connected = False

    # ── Public API ──────────────────────────────────────────────────────

    def start(self, flask_app=None) -> None:
        """Boot connect + writer + janitor threads. Idempotent."""
        if self._connect_thread and self._connect_thread.is_alive():
            return
        self._flask_app = flask_app
        self._stop_flag.clear()
        with self._stats_lock:
            self._stats['started_at'] = _iso_now()

        # Prime the cache + config synchronously so the first message
        # already has something to route against.
        try:
            self._refresh_cache_and_config()
        except Exception:
            logger.exception('[ingest] initial cache refresh failed')

        self._connect_thread = threading.Thread(
            target=self._connect_loop, name='ingest-mqtt-connect', daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name='ingest-writer', daemon=True,
        )
        self._janitor_thread = threading.Thread(
            target=self._janitor_loop, name='ingest-janitor', daemon=True,
        )
        self._connect_thread.start()
        self._writer_thread.start()
        self._janitor_thread.start()
        atexit.register(self.stop)
        logger.info('[ingest] router started (broker=%s:%s)',
                    self._broker_host, self._broker_port)

    def stop(self, timeout_s: float = 2.0) -> None:
        """Graceful shutdown. Safe to call multiple times."""
        if self._stop_flag.is_set():
            return
        self._stop_flag.set()
        # Disconnect the paho client so its loop thread exits.
        try:
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass
        # Best-effort flush of anything still buffered.
        try:
            self._flush_buffer(force=True)
        except Exception:
            pass
        # Join background threads. Daemon=True so process exit isn't blocked
        # if a join times out.
        for t in (self._connect_thread, self._writer_thread, self._janitor_thread):
            if t and t.is_alive():
                try:
                    t.join(timeout_s)
                except Exception:
                    pass

    def reload_tree(self, *, sync: bool = False) -> None:
        """Refresh the in-memory config + tree cache.

        Called from tree-mutation routes (POST /nodes, PATCH /nodes/<id>,
        import, tree-templates/apply). Fire-and-forget by default so the
        HTTP request never blocks on DB reads. Serialised through
        `_reload_executor_lock` to avoid stampedes.
        """
        def _run():
            with self._reload_executor_lock:
                try:
                    self._refresh_cache_and_config()
                except Exception:
                    logger.exception('[ingest] reload_tree failed')
        if sync:
            _run()
            return
        threading.Thread(
            target=_run, name='ingest-reload', daemon=True,
        ).start()

    def snapshot(self) -> dict:
        """Read-only view for the Stats tab + /api/asset-tree/ingest-stats."""
        with self._stats_lock:
            data = dict(self._stats)
        data.update({
            'enabled': self._enabled,
            'connected': self._connected,
            'topic_mode': self._topic_mode,
            'root_name': self._root_name,
            'meta_prefixes': sorted(list(self._meta_prefixes)),
            'broker_host': self._broker_host,
            'broker_port': self._broker_port,
            'cache_size': len(self._path_cache),
            # Buffer depth is useful for spotting a stuck writer thread.
            'buffered_rows': self._buffer_count,
            'writer_alive': bool(self._writer_thread and self._writer_thread.is_alive()),
            'connect_alive': bool(self._connect_thread and self._connect_thread.is_alive()),
            'janitor_alive': bool(self._janitor_thread and self._janitor_thread.is_alive()),
        })
        return data

    def list_rejected(self, date: str = None, limit: int = 200) -> list:
        """Return the tail of the rejected-topics log for `date` (defaults
        to today's UTC). Reads from the persistent file so entries survive
        a restart; the in-memory `_rejected_tail` is just a cache."""
        date = (date or _today_utc()).strip()
        # Very light sanitation: the reader treats `date` as a filename.
        # Reject anything that could path-traverse.
        if '/' in date or '\\' in date or '..' in date:
            return []
        path = os.path.join(self._rejected_dir(), f'{date}.log')
        if not os.path.exists(path):
            # Nothing on disk yet — fall back to the in-memory tail if
            # today's log exists there, else empty.
            if date == _today_utc():
                with self._rejected_lock:
                    return list(self._rejected_tail)[-limit:]
            return []
        entries = []
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    parts = line.split('\t', 2)
                    if len(parts) == 3:
                        entries.append({
                            'timestamp': parts[0],
                            'topic': parts[1],
                            'reason': parts[2],
                        })
                    else:
                        entries.append({'timestamp': '', 'topic': '', 'reason': line})
        except Exception as e:
            logger.warning('[ingest] read rejected log %s failed: %s', path, e)
            return []
        # Tail (last N).
        return entries[-int(max(1, min(int(limit), 5000))):]

    def run_janitor_once(self) -> dict:
        """Synchronous retention sweep. Used by the admin QA endpoint."""
        return self._janitor_sweep()

    # ── Path helpers ────────────────────────────────────────────────────

    def _datasets_root(self) -> str:
        """Base folder where topic_path folders live. Match what routes /
        File Manager already browse so ingested files show up naturally."""
        if self._flask_app is not None:
            try:
                return self._flask_app.config['DATASETS_ROOT_PATH']
            except Exception:
                pass
        # Env-var fallback for isolated test / early boot cases.
        return os.environ.get(
            'DATASETS_ROOT_PATH',
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'datasets',
            ),
        )

    def _rejected_dir(self) -> str:
        return os.path.join(self._datasets_root(), '_rejected_topics')

    # ── Cache refresh ───────────────────────────────────────────────────

    def _refresh_cache_and_config(self) -> None:
        """Reload config + full active-node map from SQLite. Called at boot,
        by reload_tree(), and periodically as a cheap safety net (once every
        10 flush ticks — see writer_loop)."""
        cfg = AssetTreeConfig.get()
        if not cfg:
            # No config yet → router idles.
            self._enabled = False
            self._root_name = None
            self._topic_mode = 'strict'
            self._meta_prefixes = frozenset()
            with self._cache_lock:
                self._path_cache = {}
            return
        # ingest_enabled defaults to 1 when the column is missing on OLD
        # rows created before Phase D migration ran (SQLite returned None
        # for the missing column pre-ALTER; after ALTER the default kicks
        # in for new rows only). Treat missing / None as enabled so we
        # don't silently turn ingest off on an upgrade.
        raw_enabled = cfg.get('ingest_enabled')
        self._enabled = True if raw_enabled is None else bool(raw_enabled)
        self._topic_mode = (cfg.get('topic_mode') or 'strict').lower()
        self._root_name = cfg.get('root_name') or None
        self._meta_prefixes = frozenset(cfg.get('meta_prefixes') or [])

        # Full active-node scan. Retired paths are omitted so newly-published
        # messages against a retired sensor rejection-log rather than routing.
        # Phase H — also fetch sensor_meta.channels so the router can
        # demultiplex multi-axis payloads without a per-message DB read.
        # Phase I Q3 — also fetch ingest_enabled / min_write_interval_ms /
        # record_until so the recording-controls checks in _route() are a
        # pure dict lookup (no DB per message).
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, topic_path, level, parent_id, status "
                "FROM asset_nodes WHERE status = 'active'"
            )
            rows = [dict(r) for r in cursor.fetchall()]
            # Query sensor_meta defensively — the 3 Phase I Q3 columns may
            # not exist yet on a DB where the migration hasn't run (fresh
            # boot ordering). Fall back to the pre-Phase-I column set so
            # the router still starts.
            try:
                cursor.execute(
                    'SELECT asset_id, channels, ingest_enabled, '
                    'min_write_interval_ms, record_until '
                    'FROM asset_sensor_meta'
                )
                meta_rows = [dict(r) for r in cursor.fetchall()]
                has_recording_cols = True
            except Exception:
                cursor.execute(
                    'SELECT asset_id, channels FROM asset_sensor_meta'
                )
                meta_rows = [dict(r) for r in cursor.fetchall()]
                has_recording_cols = False
        # asset_id → per-sensor cache dict. Decoded once here so the
        # message-thread hot path is a plain dict lookup.
        meta_by_asset: Dict[int, dict] = {}
        for mr in meta_rows:
            raw_channels = mr.get('channels') if isinstance(mr, dict) else None
            channels_list: Optional[list] = None
            if raw_channels:
                try:
                    parsed = json.loads(raw_channels)
                    if isinstance(parsed, list) and parsed:
                        channels_list = [str(c) for c in parsed]
                except Exception:
                    pass
            if has_recording_cols:
                raw_ie = mr.get('ingest_enabled')
                ingest_enabled = True if raw_ie is None else bool(raw_ie)
                raw_mwi = mr.get('min_write_interval_ms')
                try:
                    min_write_interval_ms = int(raw_mwi) if raw_mwi is not None else None
                except (TypeError, ValueError):
                    min_write_interval_ms = None
                record_until = mr.get('record_until')
                if record_until is not None and not isinstance(record_until, str):
                    record_until = str(record_until)
            else:
                ingest_enabled = True
                min_write_interval_ms = None
                record_until = None
            meta_by_asset[mr['asset_id']] = {
                'channels': channels_list,
                'ingest_enabled': ingest_enabled,
                'min_write_interval_ms': min_write_interval_ms,
                'record_until': record_until,
            }
        # Attach the sensor-meta bag directly onto each cached node dict so
        # the route hot path is a single lookup.
        for r in rows:
            meta = meta_by_asset.get(r['id']) or {}
            r['_channels'] = meta.get('channels')
            r['_ingest_enabled'] = meta.get('ingest_enabled', True)
            r['_min_write_interval_ms'] = meta.get('min_write_interval_ms')
            r['_record_until'] = meta.get('record_until')
        new_cache: Dict[str, dict] = {r['topic_path']: r for r in rows}
        with self._cache_lock:
            self._path_cache = new_cache

    # ── MQTT connect / message handler ──────────────────────────────────

    def _connect_loop(self) -> None:
        """Connects to the broker and reconnects with exponential backoff
        on drop. Never blocks Flask startup; on repeated failure just
        keeps trying — the router idles until the broker is reachable."""
        import paho.mqtt.client as paho_mqtt

        backoff = _RECONNECT_MIN_S
        while not self._stop_flag.is_set():
            try:
                with self._stats_lock:
                    self._stats['connect_attempts'] += 1
                self._client = paho_mqtt.Client(
                    paho_mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f'cira-ingest-{os.getpid()}',
                    clean_session=True,
                )
                self._client.on_connect = self._on_connect
                self._client.on_message = self._on_message
                self._client.on_disconnect = self._on_disconnect
                self._client.connect(self._broker_host, self._broker_port, keepalive=60)
                self._client.loop_start()
                # Wait for on_connect to fire (up to 5 s). Without this the
                # main loop below sees _connected=False (default) and falls
                # straight through into a reconnect storm, because the paho
                # network thread hasn't had a chance to call on_connect yet.
                connect_deadline = time.monotonic() + 5.0
                while (not self._connected
                       and time.monotonic() < connect_deadline
                       and not self._stop_flag.is_set()):
                    time.sleep(0.1)
                if not self._connected:
                    # Timeout — treat as a failed attempt, back off, retry.
                    raise TimeoutError('on_connect did not fire within 5 s')
                # Successful connect — clear the reconnect backoff so the
                # NEXT drop starts from the minimum. Without this, once the
                # backoff climbs (initial storm or a bad-broker window), it
                # stays high forever and even a healthy broker with periodic
                # blips punishes each subsequent reconnect.
                backoff = _RECONNECT_MIN_S
                # Wait either until stop or a disconnect flips _connected.
                # Poll every 500 ms — no busy-wait.
                while not self._stop_flag.is_set() and self._connected:
                    time.sleep(0.5)
                if self._stop_flag.is_set():
                    return
                # Fell out because on_disconnect fired.
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except Exception:
                    pass
                with self._stats_lock:
                    self._stats['reconnects'] += 1
                # Refresh config on every reconnect — a config PATCH may
                # have flipped `ingest_enabled` while we were down.
                try:
                    self._refresh_cache_and_config()
                except Exception:
                    pass
            except Exception as e:
                logger.warning('[ingest] broker connect failed: %s', e)
                try:
                    self._client and self._client.loop_stop()
                except Exception:
                    pass
                self._connected = False
            # Backoff before the next attempt.
            for _ in range(int(backoff * 2)):
                if self._stop_flag.is_set():
                    return
                time.sleep(0.5)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        try:
            if rc == 0:
                self._connected = True
                client.subscribe('#')
                with self._stats_lock:
                    self._stats['last_connected_at'] = _iso_now()
                logger.info(
                    '[ingest] connected to %s:%s; subscribed to `#`',
                    self._broker_host, self._broker_port,
                )
            else:
                logger.warning('[ingest] on_connect rc=%s', rc)
                self._connected = False
        except Exception:
            logger.exception('[ingest] on_connect failed')

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        self._connected = False
        logger.info('[ingest] disconnected from broker')

    def _on_message(self, client, userdata, msg):
        """Route a single MQTT message. NEVER raises — swallows everything."""
        try:
            with self._stats_lock:
                self._stats['messages_received'] += 1
                self._stats['last_message_at'] = _iso_now()
                self._stats['last_message_topic'] = msg.topic
            # $SYS heartbeats etc. — always skipped, never rejected.
            if msg.topic.startswith('$SYS'):
                return
            # If the router is disabled OR the config is missing, drop
            # silently (no rejection-log noise, no counter bump — matches
            # the spec's "idle" mode). This is the on/off switch.
            if not self._enabled or not self._root_name:
                return
            self._route(msg.topic, msg.payload)
        except Exception:
            logger.exception('[ingest] on_message crashed for %s', msg.topic)

    # ── Routing pipeline ────────────────────────────────────────────────

    def _route(self, topic: str, payload: bytes) -> None:
        # Phase J — record last-seen for /recent-topics BEFORE any routing
        # decision so the endpoint surfaces topics even when they'd be
        # rejected (unknown topic in strict mode is a common early-config
        # symptom the autocomplete list is meant to help debug).
        try:
            self._touch_topic_last_seen(topic)
        except Exception:
            logger.exception('[ingest] _touch_topic_last_seen failed')
        segments = topic.split('/')
        # 1. Root check
        if not segments or segments[0] != self._root_name:
            self._reject(topic, f"root mismatch (expected '{self._root_name}')")
            return
        # 2. Meta-prefix check — exact segment match, not prefix.
        #    Any segment beyond the root that matches a meta prefix accepts
        #    the topic as valid but skips the CSV write.
        meta_set = self._meta_prefixes
        for seg in segments[1:]:
            if seg in meta_set:
                with self._stats_lock:
                    self._stats['messages_meta'] += 1
                return
        # 3. Tree lookup
        with self._cache_lock:
            node = self._path_cache.get(topic)
        if node is None:
            if self._topic_mode == 'learn':
                node = self._autocreate_path(topic, segments)
                if node is None:
                    return
            else:
                self._reject(topic, 'unknown topic (strict mode)')
                return
        # 3.5 Phase H — multi-axis branch. If the target sensor declares
        # `channels`, demultiplex ONE JSON dict payload into a multi-column
        # row. Falls through to the single-value path when channels is None.
        channels = node.get('_channels') if isinstance(node, dict) else None
        canonical_path = node.get('topic_path') or topic
        if channels:
            row_values = self._parse_multi_axis_payload(payload, channels)
            if row_values is None:
                with self._stats_lock:
                    self._stats['messages_parse_errors'] += 1
                expected = ', '.join(channels)
                self._reject(
                    topic,
                    f'payload has no channel keys (expected: {expected})',
                )
                return
            # Phase I Q3 — recording-controls gate (post-parse so parse
            # errors still surface as rejections; a suppressed sensor
            # shouldn't hide "sensor sending malformed payloads" alerts).
            if not self._recording_allowed(node, canonical_path):
                return
            self._append_multi_row(
                canonical_path, _iso_now(), row_values, channels,
            )
            return  # multi-axis path handled — do NOT fall through.
        # 4. Parse payload (single-value path — unchanged from Phase D)
        value = self._parse_payload(payload)
        if value is None:
            with self._stats_lock:
                self._stats['messages_parse_errors'] += 1
            self._reject(topic, 'unparseable payload')
            return
        # 4.5 Phase I Q3 — recording-controls gate (single-value path).
        if not self._recording_allowed(node, canonical_path):
            return
        # 5. Buffer for the writer thread. topic_path may differ from `topic`
        #    only in weird edge cases (learn mode); use the node's canonical
        #    path so the file layout matches the tree.
        row = (_iso_now(), value)
        with self._buffer_lock:
            self._buffer.setdefault(canonical_path, []).append(row)
            self._buffer_count += 1
            need_flush = self._buffer_count >= _FLUSH_BATCH
        with self._stats_lock:
            self._stats['messages_routed'] += 1
        if need_flush:
            # Force an early flush; writer_loop's tick will still run at
            # its normal cadence.
            try:
                self._flush_buffer(force=True)
            except Exception:
                logger.exception('[ingest] force flush failed')

    # ── Phase J — recent-topics tail ─────────────────────────────────────

    def _touch_topic_last_seen(self, topic: str) -> None:
        """Update `_topic_last_seen` + `_topic_last_seen_epoch` for a topic.

        Called from every message on the hot path — must be O(1) amortised.
        The LRU purge runs at most once per 100 writes and sorts the map
        (bounded to 500 entries) so the amortised cost stays well under 1 ms.
        """
        if not topic or not isinstance(topic, str):
            return
        # Skip $SYS heartbeats — they'd dominate the recent list without
        # telling anyone anything useful for the autocomplete.
        if topic.startswith('$SYS'):
            return
        now_mono = time.monotonic()
        now_epoch = time.time()
        need_purge = False
        with self._topic_last_seen_lock:
            self._topic_last_seen[topic] = now_mono
            self._topic_last_seen_epoch[topic] = now_epoch
            self._topic_last_seen_write_counter += 1
            if self._topic_last_seen_write_counter >= 100:
                self._topic_last_seen_write_counter = 0
                if len(self._topic_last_seen) > 500:
                    need_purge = True
        if need_purge:
            self._purge_topic_last_seen()

    def _purge_topic_last_seen(self) -> None:
        """Drop the oldest 100 entries from `_topic_last_seen`.

        Called at most once per 100 writes to keep the hot path cheap. Sort
        + slice on a 500-entry dict is well under 1 ms on any modern CPU.
        """
        try:
            with self._topic_last_seen_lock:
                items = sorted(self._topic_last_seen.items(), key=lambda kv: kv[1])
                to_drop = items[:100]
                for k, _ in to_drop:
                    self._topic_last_seen.pop(k, None)
                    self._topic_last_seen_epoch.pop(k, None)
        except Exception:
            logger.exception('[ingest] _purge_topic_last_seen failed')

    def list_recent_topics(self, window_s: float = 900.0,
                           limit: int = 100) -> list:
        """Snapshot of topics seen in the last `window_s` seconds.

        Returned entries are dicts:
            {'topic': str, 'last_seen': iso8601Z, 'seconds_ago': float}
        Ordered newest-first. Cap at `limit`.
        """
        try:
            window_s = float(window_s)
            limit = int(limit)
        except (TypeError, ValueError):
            return []
        if window_s <= 0 or limit <= 0:
            return []
        now_mono = time.monotonic()
        now_epoch = time.time()
        with self._topic_last_seen_lock:
            # Snapshot both maps under the lock (they're always updated
            # together, so keys align).
            snap_mono = dict(self._topic_last_seen)
            snap_epoch = dict(self._topic_last_seen_epoch)
        recent = []
        for topic, mono_ts in snap_mono.items():
            elapsed = now_mono - mono_ts
            if elapsed > window_s:
                continue
            epoch_ts = snap_epoch.get(topic)
            # Estimate wall-clock ts from now - elapsed if epoch missing
            # (shouldn't happen — belt-and-braces).
            if epoch_ts is None:
                epoch_ts = now_epoch - elapsed
            iso = datetime.fromtimestamp(epoch_ts, tz=timezone.utc)\
                .strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            recent.append({
                'topic': topic,
                'last_seen': iso,
                'seconds_ago': round(elapsed, 3),
            })
        recent.sort(key=lambda e: e['seconds_ago'])
        return recent[:limit]

    # ── Phase I Q3 — recording controls gate ─────────────────────────────

    def _recording_allowed(self, node: dict, canonical_path: str) -> bool:
        """Return True if the router should persist the current message.

        Called AFTER payload parsing has succeeded so parse errors still
        surface as rejections. Three gates, applied in this order:
          1. `ingest_enabled=False`   → silently drop (user chose not to store)
          2. `record_until` expired   → auto-flip ingest_enabled=False + drop
          3. `min_write_interval_ms`  → drop if last write was too recent

        Never raises; on any error, fail-open (allow write) to avoid data
        loss from a router-internal bug.
        """
        try:
            ingest_enabled = node.get('_ingest_enabled', True)
            record_until = node.get('_record_until')
            min_interval = node.get('_min_write_interval_ms')
            node_id = node.get('id')

            # 1) Master toggle.
            if not ingest_enabled:
                with self._stats_lock:
                    self._stats['messages_ingest_skipped'] += 1
                return False

            # 2) Auto-stop: record_until passed → flip enabled=False in DB
            #    asynchronously so we don't block this message thread on an
            #    UPDATE + reload. The in-memory cache flip is done inline
            #    so subsequent messages (before the reload completes) also
            #    see the disabled state and don't hammer _auto_stop_sensor.
            if record_until:
                try:
                    ru_dt = _parse_iso_utc(record_until)
                except Exception:
                    ru_dt = None
                if ru_dt is not None and datetime.now(timezone.utc) > ru_dt:
                    # Inline cache flip (best-effort — cache is a dict copy
                    # under a lock; mutating a value in place is safe for
                    # the reader's dict lookup path).
                    node['_ingest_enabled'] = False
                    if node_id is not None:
                        self._schedule_auto_stop(node_id, canonical_path)
                    with self._stats_lock:
                        self._stats['messages_ingest_skipped'] += 1
                    return False

            # 3) Sample-rate throttle.
            if min_interval and min_interval > 0:
                now_mono = time.monotonic()
                with self._last_write_lock:
                    last = self._last_write_ts.get(canonical_path)
                if last is not None:
                    elapsed_ms = (now_mono - last) * 1000.0
                    if elapsed_ms < min_interval:
                        with self._stats_lock:
                            self._stats['messages_ingest_skipped'] += 1
                        return False
                with self._last_write_lock:
                    self._last_write_ts[canonical_path] = now_mono
            return True
        except Exception:
            logger.exception('[ingest] _recording_allowed crashed; fail-open')
            return True

    def _schedule_auto_stop(self, node_id: int, topic_path: str) -> None:
        """Fire off a background thread that flips ingest_enabled=False in
        the DB and refreshes the router cache. Must NOT block the message
        thread — a long UPDATE + reload would drop broker messages under
        load. Serialised through a module-scope set so the same node isn't
        scheduled twice while the first flip is in flight."""
        with self._auto_stop_lock:
            if node_id in self._auto_stop_pending:
                return
            self._auto_stop_pending.add(node_id)

        def _run():
            try:
                # Import lazily so a broken import at module-load time can't
                # take out the whole ingest thread.
                from ..models import AssetSensorMeta, AssetTreeAudit
                AssetSensorMeta.upsert(
                    asset_id=node_id, ingest_enabled=False,
                )
                try:
                    AssetTreeAudit.log(
                        actor_user_id=0,
                        event_type='sensor_recording_auto_stop',
                        target_type='node',
                        target_id=node_id,
                        payload={
                            'topic_path': topic_path,
                            'actor': 'mqtt_ingest_router',
                            'reason': 'record_until expired',
                        },
                    )
                except Exception:
                    logger.warning('[ingest] auto-stop audit failed', exc_info=True)
                logger.info(
                    '[ingest] auto-stopped recording on %s (node_id=%s) — record_until expired',
                    topic_path, node_id,
                )
                # Refresh cache so subsequent messages see the DB flip AND
                # any other sensors on the same node also pick up any
                # concurrent changes.
                try:
                    self._refresh_cache_and_config()
                except Exception:
                    logger.exception('[ingest] auto-stop cache reload failed')
            except Exception:
                logger.exception('[ingest] auto-stop DB update failed for %s', node_id)
            finally:
                with self._auto_stop_lock:
                    self._auto_stop_pending.discard(node_id)

        threading.Thread(
            target=_run, name=f'ingest-auto-stop-{node_id}', daemon=True,
        ).start()

    # ── Multi-axis payload path (Phase H) ────────────────────────────────

    def _parse_multi_axis_payload(self, payload: bytes, channels: list):
        """Demultiplex a JSON dict payload into per-channel scalar strings.

        Returns a list of length ``len(channels)``, in the same order.
        Missing keys → None in that slot. Non-numeric / bool values → the
        same coercion as ``_value_to_str``. If EVERY channel is missing
        (or the payload can't be parsed as a dict), returns None so the
        caller can reject with a helpful message.
        """
        if payload is None:
            return None
        try:
            raw = payload.decode('utf-8', errors='replace').strip()
        except Exception:
            return None
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if not isinstance(obj, dict):
            # A bare number / string on a multi-axis sensor is a config
            # mismatch — reject rather than silently dropping columns.
            return None
        out: list = []
        any_present = False
        for ch in channels:
            if ch not in obj:
                out.append(None)
                continue
            v = obj[ch]
            if v is None:
                out.append(None)
                continue
            any_present = True
            if isinstance(v, bool):
                out.append('1' if v else '0')
            elif isinstance(v, (int, float)):
                out.append(self._value_to_str(v))
            else:
                # Non-numeric value (e.g. `{"x": "foo"}`) — drop that cell.
                # Warning-log level so noisy publishers surface in the tail.
                logger.debug(
                    '[ingest] multi-axis %s: non-numeric %s=%r → empty cell',
                    channels, ch, v,
                )
                out.append(None)
        if not any_present:
            return None
        return out

    def _append_multi_row(
        self, topic_path: str, ts: str, values: list, channels: list,
    ) -> None:
        """Buffer one multi-axis row for the writer thread.

        Structure mirrors `_route`'s single-value branch — buffer entries
        are ``(ts, values_list, channels_tuple)`` so the writer can render
        the CSV header and rows without hitting the DB again.
        """
        row = (ts, tuple(values), tuple(channels))
        with self._buffer_lock:
            self._buffer.setdefault(topic_path, []).append(row)
            self._buffer_count += 1
            need_flush = self._buffer_count >= _FLUSH_BATCH
        # Message-level counters: ONE published message = one increment even
        # if it fanned out to N channels. `channels_written` tracks fan-out.
        with self._stats_lock:
            self._stats['messages_routed'] += 1
            self._stats['channels_written'] += len(channels)
        if need_flush:
            try:
                self._flush_buffer(force=True)
            except Exception:
                logger.exception('[ingest] force flush (multi-axis) failed')

    def _parse_payload(self, payload: bytes):
        """Return a string suitable for the CSV `value` column, or None
        if the payload can't be interpreted. Accepts JSON `{"value": x}`
        or a bare numeric literal or any short scalar string."""
        if payload is None:
            return None
        try:
            raw = payload.decode('utf-8', errors='replace').strip()
        except Exception:
            return None
        if not raw:
            return None
        # Try JSON first — most publishers send `{"value": 1.23}`.
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for k in ('value', 'v', 'val'):
                if k in obj:
                    v = obj[k]
                    # `{"value": null}` must be treated as a parse error, not
                    # coerced to the string 'None' (which contaminates the
                    # CSV with a categorical marker and poisons training).
                    if v is None:
                        return None
                    return self._value_to_str(v)
            # Dict without `value` — fall through and treat the raw string
            # as opaque so at least SOMETHING lands. Spec calls for
            # `{"value": ...}` or bare number, so anything else counts as
            # unparseable.
            return None
        if isinstance(obj, (int, float)):
            return self._value_to_str(obj)
        # Bare number literal?
        try:
            f = float(raw)
            return self._value_to_str(f)
        except (TypeError, ValueError):
            pass
        return None

    def _value_to_str(self, v) -> str:
        if isinstance(v, bool):
            return '1' if v else '0'
        if isinstance(v, (int, float)):
            # Avoid noisy scientific notation for common ranges but keep
            # precision for large/small floats.
            if isinstance(v, float) and (abs(v) >= 1e-4 and abs(v) < 1e12 or v == 0.0):
                # 6 dp is plenty for sensor readouts; trailing zeros stripped.
                s = f'{v:.6f}'.rstrip('0').rstrip('.')
                return s if s else '0'
            return str(v)
        return str(v)

    def _reject(self, topic: str, reason: str) -> None:
        with self._stats_lock:
            self._stats['messages_rejected'] += 1
        entry = {'timestamp': _iso_now(), 'topic': topic, 'reason': reason}
        with self._rejected_lock:
            self._rejected_tail.append(entry)
        # Persist to disk — the log is grep-friendly TSV.
        try:
            os.makedirs(self._rejected_dir(), exist_ok=True)
            path = os.path.join(self._rejected_dir(), f'{_today_utc()}.log')
            with open(path, 'a', encoding='utf-8', newline='') as fh:
                fh.write(f"{entry['timestamp']}\t{entry['topic']}\t{entry['reason']}\n")
        except Exception:
            logger.exception('[ingest] failed to persist rejection %s', topic)

    # ── Learn-mode auto-create ──────────────────────────────────────────

    def _autocreate_path(self, topic: str, segments: list) -> Optional[dict]:
        """Learn mode: create any missing ancestors + a leaf for `topic`.

        Called from the message thread, so we mutate the DB directly rather
        than going through the HTTP endpoint. Depth is bounded by the
        configured `level_names` length; over-deep topics are rejected.
        """
        cfg = AssetTreeConfig.get()
        if not cfg:
            self._reject(topic, 'no asset tree config (learn mode)')
            return None
        max_depth = max(0, len(cfg.get('level_names') or []) - 1)
        if len(segments) - 1 > max_depth:
            self._reject(topic, f'exceeds max_depth={max_depth} (learn mode)')
            return None

        created_ids = []
        parent_id = None
        node = None
        try:
            for level, seg in enumerate(segments):
                candidate = '/'.join(segments[:level + 1])
                # Prefer cache hit; fall back to DB.
                with self._cache_lock:
                    existing = self._path_cache.get(candidate)
                if existing is None:
                    existing = AssetNode.get_by_topic_path(candidate)
                if existing is not None:
                    if existing.get('status') == 'retired':
                        self._reject(topic, f"segment '{seg}' is retired (learn mode)")
                        return None
                    parent_id = existing['id']
                    node = existing
                    continue
                # Skip creation on invalid segment names (regex mirrors
                # routes/asset_tree._sanitize_name). Rejecting is safer
                # than persisting an unusable node.
                if not re.match(r'^[A-Za-z0-9_-]+$', seg):
                    self._reject(topic, f"invalid segment '{seg}' (learn mode)")
                    return None
                node_id = AssetNode.create(
                    parent_id=parent_id, level=level, name=seg,
                    topic_path=candidate, status='active',
                )
                created_ids.append(node_id)
                # Refresh cache incrementally so subsequent messages on the
                # same path hit the fast path.
                fresh = AssetNode.get_by_id(node_id) or {}
                with self._cache_lock:
                    self._path_cache[candidate] = fresh
                parent_id = node_id
                node = fresh

            if created_ids:
                try:
                    # actor_user_id is NOT NULL in the schema; use 0 as a
                    # sentinel for "system-generated" (no real user did this).
                    # FK isn't enforced (no PRAGMA foreign_keys=ON), so the
                    # dangling reference is harmless. Payload records the
                    # semantic actor so the audit log stays readable.
                    AssetTreeAudit.log(
                        actor_user_id=0,
                        event_type='node_autocreate_learn_mode',
                        target_type='node',
                        target_id=created_ids[-1],
                        payload={
                            'topic': topic,
                            'created_ids': created_ids,
                            'actor': 'mqtt_ingest_router',
                            'segments': segments,
                        },
                    )
                except Exception:
                    # Audit is best-effort — nodes are already committed.
                    logger.warning('[ingest] learn-mode audit failed', exc_info=True)
            return node
        except Exception:
            logger.exception('[ingest] autocreate failed for %s', topic)
            self._reject(topic, 'autocreate failed')
            return None

    # ── Writer / rotation loop ──────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Flush the buffer every _FLUSH_INTERVAL_S. Rotate day at 00:00 UTC."""
        cache_refresh_counter = 0
        while not self._stop_flag.is_set():
            # Cheap safety net: refresh cache every ~2 s (10 * 200 ms). If
            # a tree mutation route fired reload_tree() and it already
            # completed, this is a no-op. If reload_tree() somehow was
            # missed, this catches it within 2 s.
            cache_refresh_counter += 1
            if cache_refresh_counter >= 10:
                cache_refresh_counter = 0
                try:
                    self._refresh_cache_and_config()
                except Exception:
                    logger.exception('[ingest] periodic cache refresh failed')

            # Day rotation check.
            today = _today_utc()
            if today != self._current_date:
                # New day — invalidate the header cache so tomorrow's file
                # gets its header, and remember the new date.
                self._current_date = today
                self._headers_written = set()

            try:
                self._flush_buffer(force=False)
            except Exception:
                logger.exception('[ingest] flush tick failed')
            # Sleep responsively so shutdown is fast.
            for _ in range(int(_FLUSH_INTERVAL_S * 20)):
                if self._stop_flag.is_set():
                    break
                time.sleep(0.05)

    def _flush_buffer(self, force: bool = False) -> None:
        """Drain the in-memory buffer to per-sensor CSVs.

        Two row shapes may appear in the buffer:
          * single-value (Phase D): ``(ts, value_str)`` — 2-tuple
          * multi-axis (Phase H) : ``(ts, values_tuple, channels_tuple)``
            — 3-tuple. Detected by length.
        Rows for a single topic_path always share one shape because the
        sensor's channels config either declared multi-axis or didn't
        when the messages were routed. If the config flips mid-day the
        NEXT UTC-midnight rotation gives the new schema a fresh file.
        """
        # Swap the buffer atomically so appenders don't stall on the write.
        with self._buffer_lock:
            if not self._buffer:
                return
            local = self._buffer
            self._buffer = {}
            self._buffer_count = 0
        root = self._datasets_root()
        for topic_path, rows in local.items():
            if not rows:
                continue
            try:
                folder = os.path.join(root, *topic_path.split('/'))
                os.makedirs(folder, exist_ok=True)
                file_path = os.path.join(folder, f'{_today_utc()}.csv')
                # Detect row shape from the first row. Multi-axis is 3-tuple.
                first = rows[0]
                is_multi = isinstance(first, tuple) and len(first) == 3
                # Header on first write of the file. `_headers_written` is a
                # process-local memo; if the file existed before this process
                # started, don't overwrite its header — check disk.
                write_header = False
                if file_path not in self._headers_written:
                    if not os.path.exists(file_path):
                        write_header = True
                    self._headers_written.add(file_path)
                with open(file_path, 'a', encoding='utf-8', newline='\n') as fh:
                    if is_multi:
                        # Multi-axis: header + N-column rows. First row's
                        # channels tuple defines the header (routing state
                        # is consistent within a flush).
                        _ts0, _vals0, channels0 = first
                        if write_header:
                            fh.write(
                                'timestamp_iso,' + ','.join(channels0) + '\n'
                            )
                        for entry in rows:
                            ts, vals, _ = entry
                            # Empty string for missing / dropped cells.
                            cells = []
                            for v in vals:
                                if v is None:
                                    cells.append('')
                                    continue
                                s = v if isinstance(v, str) else str(v)
                                if ',' in s or '"' in s or '\n' in s:
                                    s = '"' + s.replace('"', '""') + '"'
                                cells.append(s)
                            fh.write(f'{ts},' + ','.join(cells) + '\n')
                    else:
                        if write_header:
                            fh.write('timestamp_iso,value\n')
                        for ts, value in rows:
                            # Escape commas / quotes minimally.
                            v = value if isinstance(value, str) else str(value)
                            if ',' in v or '"' in v or '\n' in v:
                                v = '"' + v.replace('"', '""') + '"'
                            fh.write(f'{ts},{v}\n')
                with self._stats_lock:
                    self._stats['files_written'] += 1
            except Exception:
                logger.exception('[ingest] write failed for %s', topic_path)

    # ── Retention janitor ───────────────────────────────────────────────

    def _janitor_loop(self) -> None:
        """Sweep old CSV / log files every _JANITOR_INTERVAL_S. Kicks off
        an initial sweep 30 s after start so operators don't have to wait
        6 h to see it work."""
        # Delay initial sweep so it doesn't fight with startup migrations.
        for _ in range(60):  # 60 * 0.5 s = 30 s
            if self._stop_flag.is_set():
                return
            time.sleep(0.5)
        while not self._stop_flag.is_set():
            try:
                self._janitor_sweep()
            except Exception:
                logger.exception('[ingest] janitor sweep crashed')
            # Sleep responsively.
            elapsed = 0
            while elapsed < _JANITOR_INTERVAL_S and not self._stop_flag.is_set():
                time.sleep(1.0)
                elapsed += 1

    def _janitor_sweep(self) -> dict:
        """Physical delete of CSV/log files whose filename date is older
        than retention_days ago. Returns a summary dict; also logged."""
        cfg = AssetTreeConfig.get() or {}
        raw = cfg.get('ingest_retention_days')
        try:
            retention_days = int(raw) if raw is not None else 30
        except (TypeError, ValueError):
            retention_days = 30
        if retention_days <= 0:
            return {'skipped': True, 'reason': 'retention_days <= 0'}
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
        root_name = cfg.get('root_name')
        deleted_csvs: list = []
        deleted_logs: list = []
        errors = 0

        def _try_parse_date_from_filename(fname: str):
            # Expect exactly YYYY-MM-DD.csv or YYYY-MM-DD.log
            base = os.path.splitext(fname)[0]
            try:
                return datetime.strptime(base, '%Y-%m-%d').date()
            except ValueError:
                return None

        def _sweep(root_dir: str, ext: str, bucket: list):
            nonlocal errors
            if not os.path.isdir(root_dir):
                return
            for dirpath, _dirnames, filenames in os.walk(root_dir):
                # Don't descend into the rejected-topics dir when sweeping
                # CSVs, and don't leak into unrelated system folders.
                for fn in filenames:
                    if not fn.endswith(ext):
                        continue
                    d = _try_parse_date_from_filename(fn)
                    if d is None or d >= cutoff:
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        os.remove(full)
                        bucket.append(full)
                    except Exception:
                        errors += 1

        root = self._datasets_root()
        if root_name:
            _sweep(os.path.join(root, root_name), '.csv', deleted_csvs)
        _sweep(self._rejected_dir(), '.log', deleted_logs)
        summary = {
            'ran_at': _iso_now(),
            'retention_days': retention_days,
            'deleted_csv_count': len(deleted_csvs),
            'deleted_log_count': len(deleted_logs),
            'errors': errors,
        }
        logger.info(
            '[ingest] janitor sweep: retention=%sd, csvs=%s, logs=%s, errors=%s',
            retention_days, len(deleted_csvs), len(deleted_logs), errors,
        )
        return summary


# ── Module singleton ─────────────────────────────────────────────────────
# Import from routes as: `from ..services.mqtt_ingest_router import router`.
router = MqttIngestRouter()
