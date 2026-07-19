<template>
  <div>
    <!-- Quick actions -->
    <div class="d-flex flex-wrap ga-2 mb-4">
      <v-btn
        color="primary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-database-plus"
        :to="{ name: 'pipeline-data' }"
      >
        Upload data
      </v-btn>
      <v-btn
        color="secondary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-brain"
        :to="{ name: 'pipeline-training' }"
      >
        Train a model
      </v-btn>
      <v-btn
        color="secondary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-view-dashboard-variant"
        :to="{ name: 'app-builder' }"
      >
        Deploy an app
      </v-btn>
      <v-spacer />
      <v-chip
        size="small"
        :color="mqttConnected ? 'success' : 'default'"
        :prepend-icon="mqttConnected ? 'mdi-lan-connect' : 'mdi-lan-disconnect'"
        variant="tonal"
      >
        {{ mqttConnected ? 'MQTT connected' : (mqttError || 'MQTT offline') }}
      </v-chip>
    </div>

    <!-- Sensor tiles -->
    <div v-if="sensors.length === 0" class="empty-block">
      <v-icon size="40" color="grey">mdi-chip</v-icon>
      <p class="text-body-2 text-medium-emphasis mt-2">
        This machine has no sensors yet. Add sensor children in the
        <router-link :to="{ name: 'asset-tree-admin' }">Asset Tree</router-link>
        first.
      </p>
    </div>
    <v-row v-else dense>
      <v-col
        v-for="s in sensors"
        :key="s.id"
        cols="12"
        sm="6"
        md="4"
        lg="3"
      >
        <v-card
          class="sensor-card"
          variant="tonal"
          :color="statusFor(s.id) === 'live' ? 'primary' : undefined"
        >
          <div class="d-flex align-center pa-3">
            <span
              class="status-dot mr-2"
              :class="{ 'is-live': statusFor(s.id) === 'live' }"
            />
            <div class="flex-grow-1 min-width-0">
              <div class="text-caption text-medium-emphasis text-truncate">
                {{ s.display_name || s.name }}
              </div>
              <div class="d-flex align-baseline flex-wrap">
                <span class="text-h6 font-weight-bold">
                  {{ latestFor(s.id) ?? '—' }}
                </span>
                <span
                  v-if="s.sensor_meta?.unit"
                  class="text-caption text-medium-emphasis ml-1"
                >
                  {{ s.sensor_meta.unit }}
                </span>
              </div>
              <div class="text-caption text-medium-emphasis">
                {{ s.topic_path }}
              </div>
            </div>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Recent activity -->
    <v-card variant="tonal" class="mt-4">
      <v-card-title class="d-flex align-center py-2">
        <v-icon color="primary" class="mr-2" size="small">mdi-history</v-icon>
        <span class="text-subtitle-1">Recent activity</span>
        <v-spacer />
        <v-btn
          size="x-small"
          variant="text"
          :loading="loadingAudit"
          @click="refreshAudit"
        >
          <v-icon start size="14">mdi-refresh</v-icon>
          Refresh
        </v-btn>
      </v-card-title>
      <v-divider />
      <div v-if="loadingAudit && audit.length === 0" class="pa-4 text-center text-caption">
        Loading…
      </div>
      <div v-else-if="audit.length === 0" class="pa-4 text-center text-caption text-medium-emphasis">
        No audit events touch this machine yet.
      </div>
      <v-list v-else density="compact">
        <v-list-item
          v-for="ev in audit"
          :key="ev.id"
          class="activity-row"
        >
          <template #prepend>
            <v-icon size="18" :color="eventColor(ev.event_type)">
              {{ eventIcon(ev.event_type) }}
            </v-icon>
          </template>
          <v-list-item-title class="text-body-2">
            {{ describeEvent(ev) }}
          </v-list-item-title>
          <v-list-item-subtitle class="text-caption">
            {{ formatTime(ev.created_at) }} · user #{{ ev.actor_user_id ?? '—' }}
          </v-list-item-subtitle>
        </v-list-item>
      </v-list>
    </v-card>
  </div>
</template>

<script setup lang="ts">
/**
 * Phase B.3 — Overview tab.
 * Subscribes to MQTT via mqtt.js (WebSocket via /api/mqtt/ws-info) at the
 * machine's topic prefix. Tiles show latest value + connection dot. Recent
 * activity pulls the shared /api/asset-tree/audit feed and filters to
 * events that touched this machine or its descendants.
 */
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import mqtt, { type MqttClient } from 'mqtt'
import api from '@/services/api'
import type { AssetNode } from '@/stores/assetTree'

const props = defineProps<{ machine: AssetNode }>()

// ── MQTT state ────────────────────────────────────────────────────────────
let client: MqttClient | null = null
const mqttConnected = ref(false)
const mqttError = ref<string | null>(null)
// map sensor_id → { value, ts }
const latest = ref<Record<number, { value: unknown; ts: number }>>({})
const now = ref(Date.now())
let clockInterval: ReturnType<typeof setInterval> | null = null

const sensors = computed(() => props.machine?.children || [])

// Reset per-machine when the prop changes (tab switch is cheap; navigation
// changes id though). Kills any existing client.
watch(() => props.machine?.id, () => {
  restart()
})

async function resolveBrokerUrl(): Promise<string | null> {
  // /api/mqtt/ws-info returns { host, ws_port } — mirror the URL-building
  // heuristic used by MqttTestPublisher.vue so behavior is consistent
  // across the app (nginx wss proxy on HTTPS, else ws://<host>:9001/mqtt,
  // normalize container-name/localhost to page hostname).
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
  const prefix = props.machine.topic_path
  if (!prefix) return
  try {
    client = mqtt.connect(brokerUrl, {
      clientId: `cira-ws-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      clean: true,
      keepalive: 30,
      reconnectPeriod: 3000,
    })
    client.on('connect', () => {
      mqttConnected.value = true
      mqttError.value = null
      // Subscribe to any topic under this machine (sensors publish at
      // topic_path or topic_path/<sensor>). Wildcard subscription is cheap
      // for one machine at a time.
      client?.subscribe(`${prefix}/#`, { qos: 0 })
      client?.subscribe(prefix, { qos: 0 })
    })
    client.on('close', () => { mqttConnected.value = false })
    client.on('offline', () => { mqttConnected.value = false })
    client.on('error', (err: Error) => {
      mqttError.value = err.message
      mqttConnected.value = false
    })
    client.on('message', (topic: string, payload: Buffer) => {
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
  latest.value = {}
}

function restart() {
  stopMqtt()
  startMqtt()
}

// Decode payload → per-sensor updates. Accept:
//   1) sensor-topic path — topic ends with the sensor.name and payload is
//      a scalar or { value: ... } object.
//   2) machine-topic path — payload is an object keyed by sensor.name.
function onMessage(topic: string, payload: Buffer) {
  let parsed: any
  try {
    parsed = JSON.parse(payload.toString())
  } catch {
    // Non-JSON: treat as a raw scalar published to a sensor topic.
    parsed = payload.toString()
  }
  const machineTopic = props.machine.topic_path
  const relative = topic.startsWith(machineTopic + '/')
    ? topic.slice(machineTopic.length + 1)
    : ''
  // Case 1: topic includes a sensor segment.
  if (relative) {
    const sensor = sensors.value.find(s => s.name === relative)
    if (sensor) {
      updateSensor(sensor.id, extractScalar(parsed))
      return
    }
  }
  // Case 2: payload keyed by sensor name.
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    for (const sensor of sensors.value) {
      if (sensor.name in parsed) {
        updateSensor(sensor.id, extractScalar(parsed[sensor.name]))
      }
    }
  }
}

function extractScalar(v: any): unknown {
  if (v == null) return null
  if (typeof v === 'object') {
    // Common shapes: { value: X } or { v: X } or { data: X }
    if ('value' in v) return v.value
    if ('v' in v) return v.v
    if ('data' in v) return v.data
    return JSON.stringify(v)
  }
  if (typeof v === 'number') {
    return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2)
  }
  return v
}

function updateSensor(sensorId: number, value: unknown) {
  latest.value = {
    ...latest.value,
    [sensorId]: { value, ts: Date.now() },
  }
}

function latestFor(sensorId: number): unknown {
  return latest.value[sensorId]?.value ?? null
}

function statusFor(sensorId: number): 'live' | 'stale' {
  const entry = latest.value[sensorId]
  if (!entry) return 'stale'
  return now.value - entry.ts < 10_000 ? 'live' : 'stale'
}

// ── Audit feed ────────────────────────────────────────────────────────────
const audit = ref<any[]>([])
const loadingAudit = ref(false)

function collectDescendantIds(node: AssetNode | null | undefined, out: Set<number>) {
  if (!node) return
  out.add(node.id)
  for (const c of node.children || []) collectDescendantIds(c, out)
}

async function refreshAudit() {
  loadingAudit.value = true
  try {
    // The audit endpoint doesn't filter by target — cheap to fetch a page
    // and filter client-side. 100 rows is plenty for the last-20 view.
    const r = await api.get('/api/asset-tree/audit', {
      params: { limit: 100, offset: 0 },
    })
    const rows: any[] = r.data?.audit || []
    const targetIds = new Set<number>()
    collectDescendantIds(props.machine, targetIds)
    audit.value = rows
      .filter(ev => {
        if (ev.target_type === 'node' && targetIds.has(ev.target_id)) return true
        // Group + config events that don't target a specific node get folded
        // in only when they mention this machine in the payload.
        try {
          const p = typeof ev.payload === 'string' ? JSON.parse(ev.payload) : ev.payload
          const members = p?.members || p?.after?.members || []
          if (Array.isArray(members) && members.some((m: any) => targetIds.has(Number(m?.id ?? m)))) {
            return true
          }
        } catch { /* ignore */ }
        return false
      })
      .slice(0, 20)
  } catch {
    // Silent — this is a widget, not a page.
  } finally {
    loadingAudit.value = false
  }
}

function eventIcon(t: string): string {
  if (t.includes('create')) return 'mdi-plus-circle-outline'
  if (t.includes('retire') || t.includes('delete')) return 'mdi-archive-outline'
  if (t.includes('move')) return 'mdi-swap-horizontal'
  if (t.includes('patch') || t.includes('update')) return 'mdi-pencil-outline'
  if (t.includes('group')) return 'mdi-account-group-outline'
  return 'mdi-information-outline'
}
function eventColor(t: string): string {
  if (t.includes('create')) return 'success'
  if (t.includes('retire') || t.includes('delete')) return 'error'
  if (t.includes('move')) return 'warning'
  return 'primary'
}
function describeEvent(ev: any): string {
  const t = String(ev.event_type || '')
  const target = `${ev.target_type}#${ev.target_id ?? '?'}`
  return `${t} (${target})`
}
function formatTime(s: string): string {
  try { return new Date(s + (s.endsWith('Z') ? '' : 'Z')).toLocaleString() }
  catch { return s }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────
onMounted(() => {
  startMqtt()
  refreshAudit()
  clockInterval = setInterval(() => { now.value = Date.now() }, 1000)
})

onBeforeUnmount(() => {
  stopMqtt()
  if (clockInterval) clearInterval(clockInterval)
})
</script>

<style scoped>
.status-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(var(--v-theme-on-surface), 0.25);
  transition: background 0.2s;
}
.status-dot.is-live {
  background: rgb(var(--v-theme-success));
  box-shadow: 0 0 6px rgba(var(--v-theme-success), 0.6);
}
.empty-block {
  text-align: center;
  padding: 32px 16px;
}
.sensor-card {
  height: 100%;
}
.min-width-0 { min-width: 0; }
.activity-row {
  padding-left: 12px;
  padding-right: 12px;
}
</style>
