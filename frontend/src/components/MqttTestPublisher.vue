<template>
  <div>
    <!-- MQTT Test Publisher panel -->
    <v-card class="pa-4">
      <h3 class="text-subtitle-1 font-weight-bold mb-3">
        <v-icon start size="small">mdi-access-point</v-icon>
        MQTT Test Publisher
      </h3>
      <p class="text-caption text-medium-emphasis mb-3">
        Simulate sensor data by publishing CSV rows to the MQTT broker. Use this to test live stream apps without real sensors.
      </p>

      <div class="d-flex align-center gap-2 mb-3" style="flex-wrap: wrap;">
        <v-btn
          variant="outlined"
          density="comfortable"
          :prepend-icon="mqttTestFile ? 'mdi-file-delimited-outline' : 'mdi-folder-search-outline'"
          style="min-width: 320px; max-width: 500px; justify-content: flex-start; text-transform: none; height: 40px;"
          @click="openCsvPicker"
        >
          <div class="text-truncate" style="text-align: left;">
            <div v-if="mqttTestFile" class="text-body-2">
              {{ mqttTestFile.split('/').pop() }}
            </div>
            <div v-else class="text-body-2 text-medium-emphasis">
              Choose CSV dataset…
            </div>
            <div v-if="mqttTestFile" class="text-caption text-medium-emphasis text-truncate">
              {{ mqttTestFile }}
            </div>
          </div>
        </v-btn>
        <v-text-field
          v-model="mqttTestTopic"
          label="Topic"
          variant="outlined"
          density="compact"
          hide-details
          style="max-width: 200px;"
        />
        <v-text-field
          v-model.number="mqttTestRate"
          label="Rate (msg/s)"
          type="number"
          variant="outlined"
          density="compact"
          hide-details
          style="max-width: 120px;"
        />
        <v-checkbox
          v-model="mqttTestLoop"
          label="Loop"
          density="compact"
          hide-details
          style="max-width: 80px;"
        />
      </div>

      <div class="d-flex align-center gap-2">
        <v-btn
          v-if="!mqttPublishing"
          color="success"
          variant="flat"
          size="small"
          :disabled="!mqttTestFile"
          :loading="mqttStarting"
          @click="startMqttPublish"
        >
          <v-icon start size="small">mdi-play</v-icon>
          Start Publishing
        </v-btn>
        <v-btn
          v-else
          color="error"
          variant="tonal"
          size="small"
          @click="stopMqttPublish"
        >
          <v-icon start size="small">mdi-stop</v-icon>
          Stop
        </v-btn>

        <span v-if="mqttPublishing" class="text-caption text-success">
          <v-icon size="10" color="success" class="mr-1">mdi-circle</v-icon>
          Publishing to {{ mqttTestTopic }} — {{ mqttPublished }}/{{ mqttTotal }} rows
        </span>
        <span v-if="mqttBrokerStatus !== null" class="text-caption" :class="mqttBrokerStatus ? 'text-success' : 'text-error'">
          Broker: {{ mqttBrokerStatus ? 'Connected' : 'Not available' }}
        </span>
      </div>

      <div class="text-caption text-medium-emphasis mt-2">
        <v-icon size="12" class="mr-1">mdi-web</v-icon>
        Publishing runs in your browser — no server load.
      </div>

      <v-alert v-if="mqttError" type="error" variant="tonal" density="compact" class="mt-2" closable @click:close="mqttError = ''">
        {{ mqttError }}
      </v-alert>
    </v-card>

    <!-- Unexpected-disconnect prompt (bug 3 fix). Fires when the WebSocket
         drops mid-publish for any non-operator-initiated reason. Reconnect
         resumes from the current row; Stop resets. -->
    <v-dialog v-model="mqttDisconnectDialog" max-width="480" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon start color="warning">mdi-connection</v-icon>
          Publishing stopped
        </v-card-title>
        <v-card-text>
          <p class="text-body-2 mb-3">{{ mqttDisconnectReason }}</p>
          <p class="text-caption text-medium-emphasis mb-0">
            Sent {{ mqttPublished }} / {{ mqttTotal }} rows before the drop.
            Reconnect will resume from row {{ mqttRowIndex }}.
          </p>
        </v-card-text>
        <v-card-actions class="pb-4 px-4">
          <v-spacer />
          <v-btn variant="text" @click="dismissDisconnectDialog">
            Stop
          </v-btn>
          <v-btn color="primary" variant="flat" :loading="mqttReconnecting" @click="reconnectMqttPublish">
            <v-icon start size="small">mdi-refresh</v-icon>
            Reconnect
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- CSV File Picker Dialog -->
    <v-dialog v-model="showCsvPicker" max-width="720" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center pt-4 pb-2 px-4">
          <v-icon class="mr-2">mdi-folder-search-outline</v-icon>
          Choose CSV dataset
          <v-spacer />
          <v-text-field
            v-model="csvPickerSearch"
            density="compact"
            variant="outlined"
            placeholder="Filter…"
            hide-details
            prepend-inner-icon="mdi-magnify"
            clearable
            style="max-width: 240px;"
          />
        </v-card-title>

        <!-- Breadcrumbs -->
        <div class="px-4 py-2" style="border-top: 1px solid rgba(255,255,255,0.08); border-bottom: 1px solid rgba(255,255,255,0.08);">
          <span class="text-caption text-medium-emphasis mr-1">Path:</span>
          <a href="#" class="text-caption" style="color: #a78bfa;" @click.prevent="csvPickerFolder = ''">Root</a>
          <template v-for="(seg, idx) in csvPickerCrumbs" :key="idx">
            <v-icon size="12" class="mx-1 text-medium-emphasis">mdi-chevron-right</v-icon>
            <a
              href="#"
              class="text-caption"
              style="color: #a78bfa;"
              @click.prevent="csvPickerFolder = csvPickerCrumbs.slice(0, idx + 1).join('/')"
            >{{ seg }}</a>
          </template>
        </div>

        <v-card-text style="max-height: 60vh; padding: 8px 0;">
          <v-progress-linear v-if="loadingDatasets" indeterminate color="primary" />
          <v-list v-else density="compact" nav>
            <v-list-item
              v-if="csvPickerFolder"
              class="csv-picker-row"
              @click="navigateUp"
            >
              <template #prepend>
                <v-icon color="grey">mdi-arrow-up-bold-box-outline</v-icon>
              </template>
              <v-list-item-title class="text-body-2 text-medium-emphasis">
                .. (up one level)
              </v-list-item-title>
            </v-list-item>

            <v-list-item
              v-for="folder in csvPickerFolders"
              :key="'d-' + folder.name"
              class="csv-picker-row"
              @click="enterFolder(folder.name)"
            >
              <template #prepend>
                <v-icon color="warning">mdi-folder-outline</v-icon>
              </template>
              <v-list-item-title class="text-body-2 font-weight-medium">
                {{ folder.name }}
              </v-list-item-title>
              <v-list-item-subtitle class="text-caption">
                {{ folder.file_count }} CSV{{ folder.file_count === 1 ? '' : 's' }}
              </v-list-item-subtitle>
            </v-list-item>

            <v-list-item
              v-for="file in csvPickerFiles"
              :key="'f-' + file.path"
              class="csv-picker-row"
              @click="selectCsv(file.path)"
            >
              <template #prepend>
                <v-icon color="success">mdi-file-delimited-outline</v-icon>
              </template>
              <v-list-item-title class="text-body-2">
                {{ file.name }}
              </v-list-item-title>
              <v-list-item-subtitle class="text-caption text-medium-emphasis">
                {{ file.size_kb }} KB
              </v-list-item-subtitle>
            </v-list-item>

            <v-list-item v-if="!loadingDatasets && csvPickerFolders.length === 0 && csvPickerFiles.length === 0">
              <v-list-item-title class="text-caption text-medium-emphasis text-center py-4">
                {{ csvPickerSearch ? `No CSVs matching "${csvPickerSearch}"` : 'This folder has no CSV files' }}
              </v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>

        <v-card-actions class="px-4 pb-3">
          <span class="text-caption text-medium-emphasis">
            {{ mqttDatasets.length }} CSV{{ mqttDatasets.length === 1 ? '' : 's' }} in library
          </span>
          <v-spacer />
          <v-btn variant="text" @click="showCsvPicker = false">Cancel</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import api from '@/services/api'

// MQTT Test Publisher
const mqttTestFile = ref('')
const mqttTestTopic = ref('sensors/test')
const mqttTestRate = ref(10)
const mqttTestLoop = ref(false)
const mqttDatasets = ref<any[]>([])
const loadingDatasets = ref(false)

// CSV File Picker dialog state
const showCsvPicker = ref(false)
const csvPickerFolder = ref('') // current folder within the datasets tree (empty = root)
const csvPickerSearch = ref('')

function openCsvPicker() {
  showCsvPicker.value = true
  csvPickerFolder.value = ''
  csvPickerSearch.value = ''
  if (mqttDatasets.value.length === 0) fetchMqttDatasets()
}

const csvPickerCrumbs = computed(() => (csvPickerFolder.value || '').split('/').filter(Boolean))

// Deduplicated file entries with derived filename + folder path.
const csvPickerAllEntries = computed(() => {
  const seen = new Set()
  const items: any[] = []
  for (const d of (mqttDatasets.value || [])) {
    const path = d.path || ''
    if (!path || seen.has(path)) continue
    seen.add(path)
    const slash = path.lastIndexOf('/')
    const name = slash >= 0 ? path.slice(slash + 1) : path
    const folder = slash >= 0 ? path.slice(0, slash) : ''
    items.push({ path, name, folder, size_kb: d.size_kb })
  }
  return items
})

// Files in the current folder (with optional search filter across all files).
const csvPickerFiles = computed(() => {
  const search = (csvPickerSearch.value || '').trim().toLowerCase()
  if (search) {
    return csvPickerAllEntries.value
      .filter(e => e.path.toLowerCase().includes(search))
      .sort((a, b) => a.path.localeCompare(b.path))
      .slice(0, 200)
  }
  const cur = csvPickerFolder.value
  return csvPickerAllEntries.value
    .filter(e => e.folder === cur)
    .sort((a, b) => a.name.localeCompare(b.name))
})

// Direct child folders of the current folder (only shown when NOT searching).
const csvPickerFolders = computed(() => {
  if ((csvPickerSearch.value || '').trim()) return []
  const cur = csvPickerFolder.value
  const prefix = cur ? cur + '/' : ''
  const seen = new Map<string, number>() // name → cumulative file count under it
  for (const e of csvPickerAllEntries.value) {
    if (!e.folder) continue
    if (cur && !e.folder.startsWith(prefix)) continue
    if (!cur && e.folder.includes('/')) {
      const top = e.folder.split('/')[0]
      seen.set(top, (seen.get(top) || 0) + 1)
      continue
    }
    if (!cur) {
      seen.set(e.folder, (seen.get(e.folder) || 0) + 1)
      continue
    }
    const rest = e.folder.slice(prefix.length)
    const top = rest.split('/')[0]
    if (top) seen.set(top, (seen.get(top) || 0) + 1)
  }
  return Array.from(seen.entries())
    .map(([name, file_count]) => ({ name, file_count }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

function enterFolder(name: string) {
  csvPickerFolder.value = csvPickerFolder.value ? `${csvPickerFolder.value}/${name}` : name
}

function navigateUp() {
  const parts = (csvPickerFolder.value || '').split('/').filter(Boolean)
  parts.pop()
  csvPickerFolder.value = parts.join('/')
}

function selectCsv(path: string) {
  mqttTestFile.value = path
  showCsvPicker.value = false
}

const mqttStarting = ref(false)
const mqttPublishing = ref(false)
const mqttPublished = ref(0)
const mqttTotal = ref(0)
const mqttBrokerStatus = ref<boolean | null>(null)
const mqttError = ref('')

// Browser-side MQTT publisher state.
// Publishing runs entirely in the browser via mqtt.js WebSockets so 65+
// workshop attendees don't fork threads on the Flask dev server. Server-side
// /publish endpoint is kept as a fallback for scripts but no longer used here.
let mqttClient: any = null
let mqttPublishInterval: any = null
let mqttModule: any = null       // cached mqtt.js module (dynamic import)
let mqttRows: any[] = []          // CSV rows fetched once at Start
let mqttSensorCols: string[] = []
let mqttLabelCol: string | null = null
let mqttLabelDecodeMap: Record<string, any> | null = null
let mqttRowIndex = 0
// Cache ws-info between Start and any subsequent Reconnect so a transient
// drop doesn't force another /api/mqtt/ws-info round-trip.
let mqttWsInfo: any = null
// Distinguishes an operator-initiated stop from an unexpected drop. Set
// true right before we tear down the client cleanly; the close-handler
// checks it before deciding whether to prompt Reconnect/Stop.
let mqttIntentionalStop = false

// Reconnect-or-stop prompt state — bug 3 fix. When the WebSocket closes or
// errors mid-publish (SPA navigation, background-tab throttling, broker
// blip), we DON'T silently stop; we surface a dialog so the operator can
// resume from the current row or dismiss.
const mqttDisconnectDialog = ref(false)
const mqttDisconnectReason = ref('')
const mqttReconnecting = ref(false)

async function fetchMqttDatasets() {
  loadingDatasets.value = true
  try {
    const resp = await api.get('/api/mqtt/datasets')
    mqttDatasets.value = resp.data || []
  } catch { mqttDatasets.value = [] }
  loadingDatasets.value = false
}

// Lightweight broker check — hits the new /ws-info endpoint which does NOT
// connect to Mosquitto. Cheap enough to run on mount without stalling requests.
async function checkMqttBroker() {
  try {
    await api.get('/api/mqtt/ws-info')
    mqttBrokerStatus.value = true
  } catch { mqttBrokerStatus.value = false }
}

// Build the WebSocket URL for mqtt.js. Mirrors the pattern in
// PublishedAppView.vue (line ~1863): default ws://<host>:9001/mqtt, and swap
// to the nginx wss://…/mqtt proxy when the page is HTTPS.
function buildBrokerWsUrl(host: string, wsPort: number): string {
  const isHttps = window.location.protocol === 'https:'
  // Normalize localhost/container-name host to the page hostname so browsers
  // on other machines still reach the same broker.
  let effectiveHost = host
  if (!effectiveHost || effectiveHost === 'localhost' || effectiveHost === '127.0.0.1'
      || effectiveHost === 'cirame-mosquitto') {
    effectiveHost = window.location.hostname
  }
  if (isHttps) {
    // Prefer the nginx /mqtt proxy on the current origin (matches
    // PublishedAppView behavior for HTTPS deployments).
    return `wss://${window.location.host}/mqtt`
  }
  return `ws://${effectiveHost}:${wsPort}/mqtt`
}

async function startMqttPublish() {
  if (!mqttTestFile.value) return
  mqttStarting.value = true
  mqttError.value = ''
  try {
    // 1. Fetch CSV rows + sensor/label metadata (single request, cached
    // in module-scope so a subsequent reconnect can resume without re-fetching).
    const [csvResp, wsResp] = await Promise.all([
      api.get('/api/mqtt/csv-rows', { params: { path: mqttTestFile.value } }),
      api.get('/api/mqtt/ws-info'),
    ])
    mqttRows = csvResp.data?.rows || []
    mqttSensorCols = csvResp.data?.sensor_columns || []
    mqttLabelCol = csvResp.data?.label_column || null
    mqttLabelDecodeMap = csvResp.data?.label_decode_map || null
    mqttTotal.value = csvResp.data?.total_rows || mqttRows.length
    mqttPublished.value = 0
    mqttRowIndex = 0
    // Cache the ws-info response on the module so reconnect doesn't need
    // to re-fetch (avoids an extra round-trip on transient drops).
    mqttWsInfo = wsResp.data || {}

    if (mqttRows.length === 0) {
      throw new Error('CSV returned no rows')
    }

    // 2. Dynamic-import mqtt.js (matches PublishedAppView pattern).
    if (!mqttModule) {
      const mod: any = await import('mqtt')
      mqttModule = mod.default || mod
    }

    // 3. Delegate to the shared connect+publish loop (also called by
    //    Reconnect from the disconnect dialog).
    await _connectAndPublish()
  } catch (e: any) {
    mqttError.value = e?.response?.data?.error || e?.message || 'Failed to start publisher'
  }
  mqttStarting.value = false
}

// Connect the WebSocket and start the setInterval publish loop. Shared
// between initial start (row 0) and reconnect (resumes from mqttRowIndex).
// Assumes mqttRows / mqttSensorCols / mqttLabelCol / mqttWsInfo are already
// populated by startMqttPublish.
async function _connectAndPublish() {
  const brokerUrl = buildBrokerWsUrl(mqttWsInfo?.host || '', mqttWsInfo?.ws_port || 9001)
  mqttIntentionalStop = false
  mqttClient = mqttModule.connect(brokerUrl, {
    clientId: `cira-testpub-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    clean: true,
    keepalive: 30,
    // Do NOT auto-reconnect at the mqtt.js layer. We surface a Reconnect
    // dialog to the operator instead so they can decide (workshop policy).
    reconnectPeriod: 0,
  })

  const topic = mqttTestTopic.value
  const rate = mqttTestRate.value > 0 ? mqttTestRate.value : 10
  const interval = Math.max(1, Math.floor(1000 / rate))

  mqttClient.on('connect', () => {
    mqttPublishing.value = true
    // Publishing loop.
    mqttPublishInterval = setInterval(() => {
      if (!mqttClient || !mqttPublishing.value) return
      const row = mqttRows[mqttRowIndex]
      // Payload shape MUST match server-side _publish_worker (lines 415-427):
      //   { <sensor>: float, ..., _timestamp, _index, [label]: decoded }
      const payload: Record<string, any> = {}
      for (const col of mqttSensorCols) {
        payload[col] = row[col]
      }
      payload._timestamp = Date.now() / 1000   // Python time.time() = seconds
      payload._index = mqttRowIndex
      if (mqttLabelCol && row[mqttLabelCol] !== undefined) {
        const raw = row[mqttLabelCol]
        if (mqttLabelDecodeMap && mqttLabelDecodeMap[String(raw)] !== undefined) {
          payload[mqttLabelCol] = mqttLabelDecodeMap[String(raw)]
        } else {
          payload[mqttLabelCol] = raw
        }
      }
      try {
        mqttClient.publish(topic, JSON.stringify(payload), { qos: 0 })
        mqttPublished.value++
      } catch (err) {
        console.warn('[MQTT test publisher] publish failed:', err)
      }
      mqttRowIndex++
      if (mqttRowIndex >= mqttRows.length) {
        if (mqttTestLoop.value) {
          mqttRowIndex = 0
        } else {
          stopMqttPublish()
        }
      }
    }, interval)
  })

  mqttClient.on('error', (err: any) => {
    console.warn('[MQTT test publisher] connection error:', err?.message || err)
    _handleUnexpectedDrop(err?.message || 'WebSocket connection failed')
  })

  mqttClient.on('close', () => {
    // Drop mid-publish → surface the Reconnect/Stop dialog rather than
    // silently ending the run. Intentional stops set mqttIntentionalStop
    // first so we skip the prompt in that case.
    if (mqttIntentionalStop) return
    if (mqttPublishing.value || mqttReconnecting.value) {
      _handleUnexpectedDrop('WebSocket closed unexpectedly (broker drop, tab-throttling, or network blip).')
    }
  })
}

// Tear down the current client+interval WITHOUT resetting row cursor —
// so a follow-up reconnect resumes from where we left off. Called from
// both the error and close paths, and from the Reconnect dialog itself
// before it re-invokes _connectAndPublish.
function _tearDownForReconnect() {
  mqttIntentionalStop = true   // suppress the dialog for our own close
  if (mqttPublishInterval) {
    clearInterval(mqttPublishInterval)
    mqttPublishInterval = null
  }
  if (mqttClient) {
    try { mqttClient.end(true) } catch { /* ignore */ }
    mqttClient = null
  }
  mqttPublishing.value = false
}

// Called from the mqtt.js close/error handlers when the drop was not
// operator-initiated. Preserves mqttRowIndex so Reconnect can resume.
function _handleUnexpectedDrop(reason: string) {
  if (mqttIntentionalStop) return
  _tearDownForReconnect()
  mqttReconnecting.value = false
  mqttDisconnectReason.value = reason
  mqttDisconnectDialog.value = true
}

// Reconnect handler for the dialog. Reuses the cached CSV rows + ws-info
// so no fetch happens; just re-opens the WebSocket and resumes the
// publishing loop from the current mqttRowIndex.
async function reconnectMqttPublish() {
  mqttReconnecting.value = true
  mqttError.value = ''
  mqttDisconnectDialog.value = false
  try {
    if (!mqttModule) {
      const mod: any = await import('mqtt')
      mqttModule = mod.default || mod
    }
    await _connectAndPublish()
  } catch (e: any) {
    mqttError.value = e?.message || 'Failed to reconnect'
  }
  mqttReconnecting.value = false
}

// Dismiss handler for the dialog — user chose Stop instead of Reconnect.
function dismissDisconnectDialog() {
  mqttDisconnectDialog.value = false
  mqttDisconnectReason.value = ''
  // Reset the cursor so a subsequent Start begins at row 0.
  mqttRowIndex = 0
  mqttPublished.value = 0
}

function stopMqttPublish() {
  // Mark as intentional BEFORE ending the client so the close handler
  // doesn't misinterpret this as a broker drop and pop the Reconnect
  // dialog on top of the operator who just clicked Stop.
  mqttIntentionalStop = true
  mqttPublishing.value = false
  if (mqttPublishInterval) {
    clearInterval(mqttPublishInterval)
    mqttPublishInterval = null
  }
  if (mqttClient) {
    try { mqttClient.end(true) } catch { /* ignore */ }
    mqttClient = null
  }
  // Reset cursor so a subsequent Start begins fresh.
  mqttRowIndex = 0
  mqttPublished.value = 0
  // Any leftover disconnect dialog is now moot.
  mqttDisconnectDialog.value = false
}

// Stop cleanly on tab close (beforeunload) and on component unmount so we
// never leak a WS connection or interval when the operator navigates away.
function handleBeforeUnload() {
  if (mqttClient) {
    try { mqttClient.end(true) } catch { /* ignore */ }
  }
}

onMounted(() => {
  fetchMqttDatasets()
  checkMqttBroker()
  window.addEventListener('beforeunload', handleBeforeUnload)
})

onBeforeUnmount(() => {
  stopMqttPublish()
  window.removeEventListener('beforeunload', handleBeforeUnload)
})
</script>

<style scoped>
.csv-picker-row {
  cursor: pointer;
  border-radius: 6px;
  padding: 4px 12px !important;
  min-height: 40px !important;
}
.csv-picker-row:hover {
  background: rgba(99, 102, 241, 0.10);
}
</style>
