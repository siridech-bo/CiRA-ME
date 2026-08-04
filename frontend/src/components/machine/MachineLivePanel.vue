<template>
  <div>
    <!-- Empty state — machine has zero sensor children. -->
    <div v-if="!sensors.length" class="empty-block">
      <v-icon size="40" color="grey">mdi-chip</v-icon>
      <p class="text-body-2 text-medium-emphasis mt-2">
        No sensors registered on this machine yet.
        <router-link :to="{ name: 'asset-tree-admin' }">
          Add sensors in Factory Setup
        </router-link>
        to see live signals here.
      </p>
    </div>

    <template v-else>
      <!-- Top bar — connection chip + rolling-window selector. -->
      <div class="d-flex align-center flex-wrap ga-3 mb-3">
        <v-chip
          size="small"
          :color="mqttConnected ? 'success' : 'default'"
          :prepend-icon="mqttConnected ? 'mdi-lan-connect' : 'mdi-lan-disconnect'"
          variant="tonal"
        >
          {{ mqttConnected ? 'MQTT connected' : (mqttError || 'Connecting…') }}
        </v-chip>
        <v-chip
          size="small"
          variant="tonal"
          prepend-icon="mdi-message-outline"
        >
          {{ mqttMessageCount.toLocaleString() }} msgs
        </v-chip>
        <v-spacer />
        <v-select
          v-model="liveWindowS"
          :items="WINDOW_OPTIONS"
          item-title="label"
          item-value="seconds"
          label="Rolling window"
          density="compact"
          variant="outlined"
          hide-details
          style="max-width: 200px;"
        />
      </div>

      <!-- Sparkline grid — one SensorSparkline per sensor. -->
      <div class="sparkline-grid mb-4">
        <SensorSparkline
          v-for="s in sensors"
          :key="s.id"
          :name="s.name"
          :channels="channelsFor(s)"
          :values="bufferForSensor(s)"
          :unit="s.sensor_meta?.unit || ''"
          height="70px"
        />
      </div>

      <!-- Recording controls panel. -->
      <v-card variant="tonal" class="mt-2">
        <v-card-title class="d-flex align-center py-2 flex-wrap ga-2">
          <v-icon color="primary" class="mr-2" size="small">
            mdi-record-circle-outline
          </v-icon>
          <span class="text-subtitle-1">Recording controls</span>
          <v-chip
            v-if="!isAdmin"
            size="x-small"
            variant="tonal"
            color="warning"
            class="ml-2"
          >
            Read-only — admins can edit
          </v-chip>
          <v-spacer />
          <v-btn
            v-if="isAdmin"
            size="small"
            variant="text"
            color="error"
            prepend-icon="mdi-stop-circle-outline"
            :disabled="globalActionBusy"
            :loading="globalActionBusy === 'stop'"
            @click="stopAll"
          >
            Stop all
          </v-btn>
          <v-btn
            v-if="isAdmin"
            size="small"
            variant="text"
            color="success"
            prepend-icon="mdi-play-circle-outline"
            :disabled="globalActionBusy"
            :loading="globalActionBusy === 'resume'"
            @click="resumeAll"
          >
            Resume all
          </v-btn>
          <v-btn
            size="x-small"
            variant="text"
            :loading="loadingStats"
            @click="refreshStats"
          >
            <v-icon start size="14">mdi-refresh</v-icon>
            Refresh stats
          </v-btn>
        </v-card-title>
        <v-divider />
        <v-list density="compact">
          <v-list-item
            v-for="s in sensors"
            :key="s.id"
            class="recording-row"
          >
            <div class="d-flex align-center flex-wrap ga-3 w-100 py-1">
              <!-- Sensor name -->
              <div class="sensor-name" :title="s.topic_path">
                <div class="text-body-2 font-weight-medium text-truncate">
                  {{ s.display_name || s.name }}
                </div>
                <div class="text-caption text-medium-emphasis text-truncate">
                  {{ s.topic_path }}
                </div>
              </div>

              <!-- Toggle -->
              <v-switch
                :model-value="ingestEnabledFor(s.id)"
                :disabled="!isAdmin || rowBusy[s.id]"
                :loading="rowBusy[s.id] === 'toggle'"
                color="primary"
                density="compact"
                hide-details
                :label="ingestEnabledFor(s.id) ? 'Recording' : 'Off'"
                @update:model-value="onToggle(s, $event)"
              >
                <template v-if="!isAdmin" #append>
                  <v-tooltip location="top" text="Admins only">
                    <template #activator="{ props: tp }">
                      <v-icon v-bind="tp" size="14" color="grey">
                        mdi-lock-outline
                      </v-icon>
                    </template>
                  </v-tooltip>
                </template>
              </v-switch>

              <!-- Sample-every input -->
              <v-text-field
                v-model.number="intervalDrafts[s.id]"
                type="number"
                min="0"
                label="Sample every"
                suffix="ms"
                :placeholder="ingestEnabledFor(s.id) ? 'every msg' : ''"
                :disabled="!isAdmin || rowBusy[s.id]"
                :loading="rowBusy[s.id] === 'interval'"
                density="compact"
                variant="outlined"
                hide-details
                style="max-width: 160px;"
                @keydown.enter="onIntervalCommit(s)"
                @blur="onIntervalCommit(s)"
              />

              <!-- Auto-stop dropdown -->
              <v-select
                :model-value="autoStopChoiceFor(s.id)"
                :items="AUTO_STOP_OPTIONS"
                item-title="label"
                item-value="key"
                label="Auto-stop"
                density="compact"
                variant="outlined"
                hide-details
                :disabled="!isAdmin || rowBusy[s.id]"
                :loading="rowBusy[s.id] === 'auto-stop'"
                style="max-width: 180px;"
                @update:model-value="onAutoStopChange(s, $event)"
              />

              <v-chip
                v-if="recordUntilFor(s.id)"
                size="x-small"
                variant="tonal"
                color="warning"
                prepend-icon="mdi-timer-sand"
              >
                until {{ formatRecordUntil(recordUntilFor(s.id)) }}
              </v-chip>

              <v-spacer />

              <!-- Storage stat -->
              <div class="text-caption text-medium-emphasis stat-cell">
                <div>
                  <v-icon size="12">mdi-database</v-icon>
                  {{ formatBytes(statsFor(s.id).bytes) }}
                </div>
                <div>
                  {{ statsFor(s.id).row_count.toLocaleString() }} rows today
                </div>
              </div>
            </div>
          </v-list-item>
        </v-list>
      </v-card>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * MachineLivePanel — Phase I Q3 (2026-07-25).
 *
 * Renders the Live tab on the machine workspace. Subscribes via mqtt.js
 * to `<machineTopic>/+` and shows one sparkline per sensor plus the
 * admin-gated recording controls that call PATCH /nodes/<id>/recording.
 *
 * mqtt.js WebSocket URL resolution mirrors the pattern already used by
 * MachineOverviewTab.vue (/api/mqtt/ws-info + nginx wss proxy on HTTPS).
 * The Overview tab holds its own mqtt.js client; both tabs run their own
 * subscription because switching tabs unmounts the invisible one and its
 * onBeforeUnmount hook tears the client down cleanly. If a user hops back
 * and forth quickly this is a couple of extra CONNECT/DISCONNECT round-
 * trips per second, which is well inside mosquitto's tolerance.
 *
 * Buffers are per-topic-suffix, per-axis, capped at `windowSize()` samples.
 * When the user changes the rolling window we truncate the front of every
 * buffer so the sparkline reflects the new window on the next tick.
 */
import { ref, computed, onMounted, onBeforeUnmount, watch, reactive } from 'vue'
import mqtt, { type MqttClient } from 'mqtt'
import api from '@/services/api'
import type { AssetNode } from '@/stores/assetTree'
import { useNotificationStore } from '@/stores/notification'
import SensorSparkline from '@/components/SensorSparkline.vue'
import { useAssetTreeStore } from '@/stores/assetTree'

const notify = useNotificationStore()
const assetTreeStore = useAssetTreeStore()

const props = defineProps<{
  machineTopic: string
  machineNodeId: number
  sensors: AssetNode[]
  isAdmin: boolean
}>()

// ── Rolling window ────────────────────────────────────────────────────────
const WINDOW_OPTIONS = [
  { label: 'Last 5 sec',   seconds: 5 },
  { label: 'Last 30 sec',  seconds: 30 },
  { label: 'Last 60 sec',  seconds: 60 },
  { label: 'Last 5 min',   seconds: 5 * 60 },
  { label: 'Last 30 min',  seconds: 30 * 60 },
]
const DEFAULT_WINDOW = 60
const WINDOW_STORAGE_KEY = computed(
  () => `cira.machine.${props.machineNodeId}.liveWindowS`,
)

function loadWindowPref(): number {
  try {
    const raw = localStorage.getItem(WINDOW_STORAGE_KEY.value)
    if (!raw) return DEFAULT_WINDOW
    const n = Number(raw)
    if (WINDOW_OPTIONS.some(w => w.seconds === n)) return n
    return DEFAULT_WINDOW
  } catch {
    return DEFAULT_WINDOW
  }
}
const liveWindowS = ref<number>(loadWindowPref())

watch(liveWindowS, (v) => {
  try {
    localStorage.setItem(WINDOW_STORAGE_KEY.value, String(v))
  } catch { /* ignore quota / disabled storage */ }
  // Truncate every buffer so the new window is reflected immediately —
  // without this we'd keep serving up-to-30-min-old samples for up to
  // 30 min after the user picked "5 sec".
  const cap = windowCapFor
  for (const [k, arr] of Object.entries(buffers)) {
    const [sensorId] = k.split('|')
    const s = props.sensors.find(x => String(x.id) === sensorId)
    const c = s ? cap(s) : 60
    if (arr.length > c) buffers[k] = arr.slice(arr.length - c)
  }
})

// Compute how many samples to retain per axis given the current window
// and the sensor's configured sample_rate_hz. Falls back to 10 Hz when
// the rate isn't set — enough to render a smooth line without OOM.
function windowCapFor(sensor: AssetNode): number {
  const rate = Number(sensor.sensor_meta?.sample_rate_hz) || 10
  const cap = Math.ceil(liveWindowS.value * rate)
  // Absolute upper bound so a misconfigured 10 kHz sensor doesn't allocate
  // 18M-slot arrays for the 30-minute window (200k per buffer is enough
  // for any human-viewable sparkline; the visualization down-samples).
  return Math.min(Math.max(cap, 30), 200_000)
}

// ── MQTT state + per-topic buffers ────────────────────────────────────────
let client: MqttClient | null = null
const mqttConnected = ref(false)
const mqttError = ref<string | null>(null)
const mqttMessageCount = ref(0)
// Key format: `${sensorId}|${axisOrSensorName}`. Values are number[]. Not
// wrapped in `ref<Record>` because reactive replacement on every message
// would trigger a full re-render of every sparkline; the reactive proxy
// below lets Vue see per-cell writes and the sparkline computed re-runs
// only when its keys' entries change.
const buffers = reactive<Record<string, number[]>>({})

// Build the sparkline value bag shape SensorSparkline.vue expects:
//   single-value:  { [sensor.name]: number[] }
//   multi-axis:    { [`${sensor.name}.${axis}`]: number[] }
function bufferForSensor(sensor: AssetNode): Record<string, number[]> {
  // Prefer declared channels; fall back to any channel keys we discovered
  // live (Phase K #4 fallback for sensors with no channels config).
  const ch = channelsFor(sensor)
  const out: Record<string, number[]> = {}
  if (ch && ch.length > 0) {
    for (const axis of ch) {
      out[`${sensor.name}.${axis}`] = buffers[`${sensor.id}|${axis}`] || []
    }
    return out
  }
  out[sensor.name] = buffers[`${sensor.id}|__value__`] || []
  return out
}

// ── Broker URL + connection ───────────────────────────────────────────────
async function resolveBrokerUrl(): Promise<string | null> {
  try {
    const r = await api.get('/api/mqtt/ws-info')
    const host: string = String(r.data?.host || '')
    const wsPort: number = Number(r.data?.ws_port || 9001)
    const isHttps = window.location.protocol === 'https:'
    let effectiveHost = host
    if (!effectiveHost
        || effectiveHost === 'localhost'
        || effectiveHost === '127.0.0.1'
        || effectiveHost === 'cirame-mosquitto') {
      effectiveHost = window.location.hostname
    }
    if (isHttps) return `wss://${window.location.host}/mqtt`
    return `ws://${effectiveHost}:${wsPort}/mqtt`
  } catch (e: any) {
    mqttError.value = e.response?.data?.error || 'MQTT info unavailable'
    return null
  }
}

async function startMqtt() {
  stopMqtt()
  const brokerUrl = await resolveBrokerUrl()
  if (!brokerUrl) return
  const prefix = props.machineTopic
  if (!prefix) return
  try {
    client = mqtt.connect(brokerUrl, {
      clientId: `cira-live-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      clean: true,
      keepalive: 30,
      reconnectPeriod: 3000,
    })
    client.on('connect', () => {
      mqttConnected.value = true
      mqttError.value = null
      // Subscribe to any direct child of the machine topic. `+` matches
      // exactly one level so we don't also pull in deeper sub-topics
      // (there shouldn't be any, but strict is safer).
      client?.subscribe(`${prefix}/+`, { qos: 0 })
    })
    client.on('close', () => { mqttConnected.value = false })
    client.on('offline', () => { mqttConnected.value = false })
    client.on('reconnect', () => { mqttError.value = null })
    client.on('error', (err: Error) => {
      mqttError.value = err.message
      mqttConnected.value = false
    })
    client.on('message', (topic: string, payload: Buffer) => {
      mqttMessageCount.value++
      onMessage(topic, payload)
    })
  } catch (e: any) {
    mqttError.value = e.message || 'Failed to connect'
  }
}

function stopMqtt() {
  if (client) {
    try { client.end(true) } catch { /* ignore */ }
    client = null
  }
  mqttConnected.value = false
}

// Parse an incoming MQTT payload against the matching sensor's channels
// config. Single-value: extract from `{"value": N}` / `{"v": N}` / bare
// number. Multi-axis: extract `{x, y, z, …}` keys the sensor declared.
function onMessage(topic: string, payload: Buffer) {
  const prefix = props.machineTopic
  if (!topic.startsWith(prefix + '/')) return
  const suffix = topic.slice(prefix.length + 1)
  const sensor = props.sensors.find(s => s.name === suffix)
  if (!sensor) return
  let parsed: any
  try {
    parsed = JSON.parse(payload.toString())
  } catch {
    // Non-JSON: treat body as a raw scalar published to the sensor topic.
    const asNum = Number(payload.toString())
    parsed = Number.isFinite(asNum) ? asNum : null
  }
  const ch = sensor.sensor_meta?.channels
  const cap = windowCapFor(sensor)
  if (ch && ch.length > 0) {
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      // Build a case-insensitive lookup so `{X, Y, Z}` matches
      // declared channels `[x, y, z]` (and vice versa). Real-world
      // publishers (Arduino/ESP32 accelerometer libs) mix cases.
      const lowerMap: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(parsed)) lowerMap[k.toLowerCase()] = v
      for (const axis of ch) {
        const raw = (axis in parsed) ? parsed[axis] : lowerMap[axis.toLowerCase()]
        if (raw === undefined) continue
        const v = Number(raw)
        if (!Number.isFinite(v)) continue
        pushToBuffer(`${sensor.id}|${axis}`, v, cap)
      }
    }
    return
  }
  // Single-value: accept several common shapes.
  let value: number | null = null
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    for (const k of ['value', 'v', 'val']) {
      if (k in parsed && Number.isFinite(Number(parsed[k]))) {
        value = Number(parsed[k])
        break
      }
    }
    // Fallback for multi-axis payloads on sensors that were created without
    // declaring channels (learn-mode auto-creation, Phase F migrations).
    // Discover per-axis keys and demux under a `discovered:` channel scheme
    // so sparklines render immediately instead of silently dropping data.
    if (value === null) {
      const discovered: [string, number][] = []
      for (const [k, v] of Object.entries(parsed)) {
        if (typeof v === 'number' && Number.isFinite(v)) discovered.push([k, v])
      }
      if (discovered.length > 0) {
        for (const [k, v] of discovered) pushToBuffer(`${sensor.id}|${k}`, v, cap)
        registerDiscoveredChannels(sensor.id, discovered.map(([k]) => k))
        return
      }
    }
  } else if (typeof parsed === 'number' && Number.isFinite(parsed)) {
    value = parsed
  }
  if (value !== null) {
    pushToBuffer(`${sensor.id}|__value__`, value, cap)
  }
}

// Tracks channel names discovered from live payloads for sensors whose
// sensor_meta.channels is null. Feeds the sparkline via a lookup so the
// grid can render multiple axes without waiting on the admin to update
// the sensor's tree config.
const discoveredChannels = reactive<Record<number, string[]>>({})
function registerDiscoveredChannels(sensorId: number, keys: string[]) {
  const existing = discoveredChannels[sensorId] || []
  const merged = Array.from(new Set([...existing, ...keys]))
  if (merged.length !== existing.length) discoveredChannels[sensorId] = merged
}
function channelsFor(sensor: AssetNode): string[] | null {
  const declared = sensor.sensor_meta?.channels
  if (declared && declared.length > 0) return declared
  return discoveredChannels[sensor.id] || null
}

function pushToBuffer(key: string, value: number, cap: number) {
  const cur = buffers[key]
  if (!cur) {
    buffers[key] = [value]
    return
  }
  cur.push(value)
  if (cur.length > cap) {
    // In-place slice keeps the reactive array reference stable so
    // Chart.js re-renders incrementally instead of dropping/adding the
    // dataset (which flashes the canvas).
    cur.splice(0, cur.length - cap)
  }
}

// ── Recording controls state + calls ──────────────────────────────────────
interface SensorMetaExt {
  ingest_enabled: boolean
  min_write_interval_ms: number | null
  record_until: string | null
}
const meta = reactive<Record<number, SensorMetaExt>>({})
const intervalDrafts = reactive<Record<number, number | null>>({})
const rowBusy = reactive<Record<number, false | 'toggle' | 'interval' | 'auto-stop'>>({})
const globalActionBusy = ref<false | 'stop' | 'resume'>(false)

// Storage stats per sensor.
interface StatBag { bytes: number; row_count: number }
const stats = reactive<Record<number, StatBag>>({})
const loadingStats = ref(false)
let statsInterval: ReturnType<typeof setInterval> | null = null

function ingestEnabledFor(id: number): boolean {
  return meta[id]?.ingest_enabled ?? true
}
function recordUntilFor(id: number): string | null {
  return meta[id]?.record_until ?? null
}
function statsFor(id: number): StatBag {
  return stats[id] || { bytes: 0, row_count: 0 }
}

const AUTO_STOP_OPTIONS = [
  { key: 'never',  label: 'Never' },
  { key: '5m',     label: 'In 5 min' },
  { key: '1h',     label: 'In 1 hour' },
  { key: '8h',     label: 'In 8 hours' },
  // QA B3: any active record_until that doesn't match a preset renders as
  // "Custom" (with the deadline chip on the row header showing the exact
  // time). Without this option in the items list the dropdown blanks out
  // the moment the user picks a preset (Vuetify hides the label when the
  // model-value isn't in items).
  { key: 'custom', label: 'Custom' },
]

function autoStopChoiceFor(id: number): string {
  const ru = meta[id]?.record_until
  if (!ru) return 'never'
  // Any active record_until — reflect it as a generic "custom" so the
  // dropdown doesn't lie about being one of the presets.
  return 'custom'
}

async function seedMeta() {
  // Read initial sensor meta values from the passed sensor prop. When the
  // asset-tree store hasn't populated recording fields yet (an older
  // tree fetch predating this deploy), fall back to defaults so the UI
  // renders — the first refreshStats() will backfill anyway.
  for (const s of props.sensors) {
    const sm: any = s.sensor_meta || {}
    meta[s.id] = {
      ingest_enabled: sm.ingest_enabled !== false,  // undefined → true
      min_write_interval_ms: sm.min_write_interval_ms ?? null,
      record_until: sm.record_until ?? null,
    }
    intervalDrafts[s.id] = meta[s.id].min_write_interval_ms
    rowBusy[s.id] = false
  }
}

async function refreshStats() {
  loadingStats.value = true
  const today = new Date().toISOString().slice(0, 10)
  try {
    await Promise.all(props.sensors.map(async (s) => {
      try {
        const r = await api.get(
          `/api/asset-tree/nodes/${s.id}/data-stats`,
          { params: { date: today } },
        )
        stats[s.id] = {
          bytes: Number(r.data?.bytes) || 0,
          row_count: Number(r.data?.row_count) || 0,
        }
      } catch {
        // Row-level stat failure is silent — one bad sensor shouldn't
        // hide the others' numbers.
      }
    }))
  } finally {
    loadingStats.value = false
  }
}

async function patchRecording(sensor: AssetNode, body: Record<string, unknown>,
                              busyKind: 'toggle' | 'interval' | 'auto-stop') {
  if (!props.isAdmin) return
  rowBusy[sensor.id] = busyKind
  try {
    const r = await api.patch(
      `/api/asset-tree/nodes/${sensor.id}/recording`,
      body,
    )
    meta[sensor.id] = {
      ingest_enabled: !!r.data?.ingest_enabled,
      min_write_interval_ms: r.data?.min_write_interval_ms ?? null,
      record_until: r.data?.record_until ?? null,
    }
    intervalDrafts[sensor.id] = meta[sensor.id].min_write_interval_ms
    // Refresh the tree store so future mounts / other tabs see our just-
    // PATCHed values. IMPORTANT: use fetchTree(true) not invalidateTree()
    // — invalidate synchronously clears `tree.value = []`, which collapses
    // the parent MachineWorkspaceView's currentMachineNode lookup to null
    // and blanks the entire page for a beat. fetchTree(true) preserves
    // the old tree until the fresh response lands.
    assetTreeStore.fetchTree(true).catch(() => { /* soft-fail */ })
    notify.showSuccess(`Recording updated for ${sensor.name}`)
  } catch (e: any) {
    notify.showError(
      e.response?.data?.error || `Failed to update ${sensor.name}`,
    )
  } finally {
    rowBusy[sensor.id] = false
  }
}

function onToggle(sensor: AssetNode, enabled: boolean) {
  // QA P3: when the user flips a sensor back ON that had auto-stopped,
  // also clear record_until — otherwise the router sees the (past)
  // expiry timestamp on the very next message and immediately re-flips
  // ingest_enabled back to false, filling the audit log with alternating
  // patch + auto_stop events. Matches the resumeAll behaviour.
  const body: Record<string, any> = { ingest_enabled: !!enabled }
  const isPastRecordUntil = enabled && meta[sensor.id]?.record_until
    && new Date(meta[sensor.id].record_until).getTime() < Date.now()
  if (isPastRecordUntil) body.record_until = null
  patchRecording(sensor, body, 'toggle')
}

function onIntervalCommit(sensor: AssetNode) {
  const draft = intervalDrafts[sensor.id]
  const current = meta[sensor.id]?.min_write_interval_ms ?? null
  const normalized: number | null =
    draft === null || draft === undefined || (draft as any) === ''
      ? null
      : Number(draft)
  if (normalized === current) return
  if (normalized !== null
      && (!Number.isFinite(normalized) || normalized < 1 || normalized > 86_400_000)) {
    notify.showError('Interval must be 1..86400000 ms, or blank for none')
    intervalDrafts[sensor.id] = current
    return
  }
  patchRecording(sensor, {
    min_write_interval_ms: normalized === null ? null : Math.round(normalized),
  }, 'interval')
}

function onAutoStopChange(sensor: AssetNode, choice: string) {
  let iso: string | null = null
  const now = Date.now()
  if (choice === 'never') iso = null
  else if (choice === '5m') iso = new Date(now + 5 * 60_000).toISOString()
  else if (choice === '1h') iso = new Date(now + 60 * 60_000).toISOString()
  else if (choice === '8h') iso = new Date(now + 8 * 60 * 60_000).toISOString()
  else return
  patchRecording(sensor, { record_until: iso }, 'auto-stop')
}

async function stopAll() {
  if (!props.isAdmin) return
  globalActionBusy.value = 'stop'
  try {
    await Promise.all(props.sensors.map(s =>
      api.patch(`/api/asset-tree/nodes/${s.id}/recording`,
                { ingest_enabled: false }),
    ))
    for (const s of props.sensors) {
      if (!meta[s.id]) meta[s.id] = { ingest_enabled: false, min_write_interval_ms: null, record_until: null }
      else meta[s.id].ingest_enabled = false
    }
    notify.showSuccess('All sensor recording stopped')
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Stop-all failed for one or more sensors')
  } finally {
    globalActionBusy.value = false
  }
}

async function resumeAll() {
  if (!props.isAdmin) return
  globalActionBusy.value = 'resume'
  try {
    await Promise.all(props.sensors.map(s =>
      api.patch(`/api/asset-tree/nodes/${s.id}/recording`,
                { ingest_enabled: true, record_until: null }),
    ))
    for (const s of props.sensors) {
      if (!meta[s.id]) meta[s.id] = { ingest_enabled: true, min_write_interval_ms: null, record_until: null }
      else {
        meta[s.id].ingest_enabled = true
        meta[s.id].record_until = null
      }
    }
    notify.showSuccess('All sensor recording resumed')
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Resume-all failed for one or more sensors')
  } finally {
    globalActionBusy.value = false
  }
}

// ── Formatting helpers ────────────────────────────────────────────────────
function formatBytes(b: number): string {
  if (!b) return '0 B'
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}
function formatRecordUntil(iso: string | null): string {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────
const sensors = computed(() => props.sensors || [])

// Restart MQTT + reseed meta when the machine prop changes (e.g. sidebar
// nav between two machines reuses this component instance).
watch(() => props.machineNodeId, async () => {
  await seedMeta()
  Object.keys(buffers).forEach(k => delete buffers[k])
  mqttMessageCount.value = 0
  if (statsInterval) clearInterval(statsInterval)
  startMqtt()
  refreshStats()
  statsInterval = setInterval(refreshStats, 30_000)
})

onMounted(async () => {
  // Force-refresh the tree store so a tab-switch-and-return picks up any
  // recording-state changes that landed elsewhere (other admin session,
  // router auto-stop, our own PATCH from an earlier mount). Without this
  // the tab reads a cached snapshot and shows stale toggle positions.
  // seedMeta re-runs via the props watcher below once the store settles.
  try { await assetTreeStore.fetchTree(true) } catch { /* soft-fail — seed with what we have */ }
  await seedMeta()
  startMqtt()
  refreshStats()
  statsInterval = setInterval(refreshStats, 30_000)
})

// When the parent recomputes `sensors` (post-fetchTree), re-seed local
// recording state from the fresh sensor_meta values so the toggle UI
// mirrors the DB.
watch(() => props.sensors, async () => {
  await seedMeta()
}, { deep: true })

onBeforeUnmount(() => {
  stopMqtt()
  if (statsInterval) {
    clearInterval(statsInterval)
    statsInterval = null
  }
})
</script>

<style scoped>
.empty-block {
  text-align: center;
  padding: 32px 16px;
}
.sparkline-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}
.recording-row {
  border-bottom: 1px solid rgba(128, 128, 128, 0.08);
}
.recording-row:last-child {
  border-bottom: none;
}
.sensor-name {
  min-width: 180px;
  max-width: 220px;
  flex-shrink: 0;
}
.stat-cell {
  text-align: right;
  min-width: 110px;
}
</style>
