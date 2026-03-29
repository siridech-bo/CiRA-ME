<template>
  <v-container fluid class="pa-6">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">App Builder</h1>
        <p class="text-body-2 text-medium-emphasis">
          Build visual inference apps from ME-LAB endpoints
        </p>
      </div>
      <v-spacer />
      <v-btn color="primary" @click="openCreateDialog">
        <v-icon start>mdi-plus</v-icon>
        New App
      </v-btn>
    </div>

    <!-- Apps Table -->
    <v-card class="pa-4">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">
        <v-icon start size="small">mdi-view-dashboard-outline</v-icon>
        Your Apps
      </h3>

      <v-table v-if="apps.length > 0" density="comfortable">
        <thead>
          <tr>
            <th>Name</th>
            <th>Mode</th>
            <th>Status</th>
            <th>Access</th>
            <th>Created</th>
            <th class="text-center">Calls</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="app in apps" :key="app.id">
            <td>
              <div class="font-weight-medium">{{ app.name }}</div>
              <div class="text-caption text-medium-emphasis">{{ app.id }}</div>
            </td>
            <td>
              <v-chip
                v-if="app.mode"
                size="x-small"
                variant="tonal"
                :color="modeColor(app.mode)"
              >
                {{ app.mode }}
              </v-chip>
              <span v-else class="text-caption text-medium-emphasis">—</span>
            </td>
            <td>
              <v-chip
                size="x-small"
                variant="flat"
                :color="app.status === 'published' ? 'success' : 'grey'"
              >
                <v-icon v-if="app.status === 'published'" start size="10">mdi-circle</v-icon>
                {{ app.status || 'draft' }}
              </v-chip>
              <div v-if="app.status === 'published' && app.slug" class="mt-1">
                <a
                  :href="`/standalone/${app.slug}`"
                  target="_blank"
                  class="published-link"
                  @click.stop
                >
                  <v-icon size="10" class="mr-1">mdi-open-in-new</v-icon>
                  {{ app.slug }}
                </a>
              </div>
            </td>
            <td>
              <v-select
                v-if="app.status === 'published'"
                :model-value="app.access || 'private'"
                @update:model-value="v => updateAccess(app, v)"
                :items="accessOptions"
                item-title="label"
                item-value="value"
                variant="outlined"
                density="compact"
                hide-details
                style="max-width: 160px; font-size: 11px;"
              >
                <template #selection="{ item }">
                  <v-icon size="12" :color="item.raw.color" class="mr-1">{{ item.raw.icon }}</v-icon>
                  <span style="font-size: 11px;">{{ item.raw.label }}</span>
                </template>
              </v-select>
              <span v-else class="text-caption text-medium-emphasis">—</span>
            </td>
            <td class="text-caption">{{ formatDate(app.created_at) }}</td>
            <td class="text-center">{{ app.calls_count ?? app.calls ?? 0 }}</td>
            <td>
              <v-btn
                v-if="app.status === 'published' && app.slug"
                icon
                size="x-small"
                variant="text"
                color="purple"
                title="Open Standalone App"
                :href="`/standalone/${app.slug}`"
                target="_blank"
              >
                <v-icon size="small">mdi-open-in-new</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="info"
                title="Edit App"
                @click="navigateToApp(app.id)"
              >
                <v-icon size="small">mdi-pencil-outline</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="error"
                title="Delete App"
                @click="openDeleteDialog(app)"
              >
                <v-icon size="small">mdi-delete-outline</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>

      <!-- Empty state -->
      <div v-else-if="!loading" class="text-center pa-8">
        <v-icon size="48" color="grey" class="mb-3">mdi-view-dashboard-edit-outline</v-icon>
        <div class="text-body-1 text-medium-emphasis">No apps yet.</div>
        <div class="text-caption text-medium-emphasis mt-1">
          Create your first app to get started.
        </div>
        <v-btn color="primary" variant="tonal" class="mt-4" @click="openCreateDialog">
          <v-icon start>mdi-plus</v-icon>
          New App
        </v-btn>
      </div>

      <!-- Loading state -->
      <div v-else class="text-center pa-8">
        <v-progress-circular indeterminate color="primary" size="36" />
      </div>
    </v-card>

    <!-- MQTT Test Publisher -->
    <v-card class="pa-4 mt-6">
      <h3 class="text-subtitle-1 font-weight-bold mb-3">
        <v-icon start size="small">mdi-access-point</v-icon>
        MQTT Test Publisher
      </h3>
      <p class="text-caption text-medium-emphasis mb-3">
        Simulate sensor data by publishing CSV rows to the MQTT broker. Use this to test live stream apps without real sensors.
      </p>

      <div class="d-flex align-center gap-2 mb-3" style="flex-wrap: wrap;">
        <v-select
          v-model="mqttTestFile"
          :items="mqttDatasets"
          item-title="name"
          item-value="path"
          label="CSV Dataset"
          variant="outlined"
          density="compact"
          hide-details
          style="max-width: 300px;"
          :loading="loadingDatasets"
        />
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

      <v-alert v-if="mqttError" type="error" variant="tonal" density="compact" class="mt-2" closable @click:close="mqttError = ''">
        {{ mqttError }}
      </v-alert>
    </v-card>

    <!-- Create App Dialog -->
    <v-dialog v-model="showCreateDialog" max-width="560">
      <v-card>
        <v-card-title class="pt-5 pb-2 px-5">
          <v-icon start size="small">mdi-plus-circle-outline</v-icon>
          New App
        </v-card-title>
        <v-card-text class="px-5 pb-2">
          <v-text-field
            ref="createNameField"
            v-model="newAppName"
            label="App Name"
            variant="outlined"
            density="compact"
            placeholder="e.g. Vibration Monitor"
            :error-messages="createError"
            autofocus
            class="mb-4"
            @input="createError = ''"
            @keydown.enter="createApp"
          />

          <div class="text-caption font-weight-bold text-medium-emphasis mb-2">TEMPLATE</div>
          <v-row dense>
            <v-col v-for="tpl in TEMPLATES" :key="tpl.id" cols="6">
              <v-card
                variant="outlined"
                :color="selectedTemplate === tpl.id ? 'purple' : undefined"
                class="pa-3 template-card"
                :class="{ 'template-selected': selectedTemplate === tpl.id }"
                @click="selectedTemplate = tpl.id"
                style="cursor: pointer; min-height: 90px"
              >
                <div class="d-flex align-center gap-2 mb-1">
                  <v-icon size="16" :color="tpl.color">{{ tpl.icon }}</v-icon>
                  <span class="text-body-2 font-weight-bold">{{ tpl.name }}</span>
                </div>
                <div class="text-caption text-medium-emphasis">{{ tpl.description }}</div>
                <div class="mt-1">
                  <v-chip v-for="node in tpl.nodeLabels" :key="node" size="x-small" variant="tonal" class="mr-1 mt-1" style="font-size:9px">
                    {{ node }}
                  </v-chip>
                </div>
              </v-card>
            </v-col>
          </v-row>
        </v-card-text>
        <v-card-actions class="px-5 pb-4">
          <v-spacer />
          <v-btn variant="text" :disabled="creating" @click="closeCreateDialog">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :disabled="!newAppName.trim()"
            :loading="creating"
            @click="createApp"
          >
            Create
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete Confirm Dialog -->
    <v-dialog v-model="showDeleteDialog" max-width="400">
      <v-card>
        <v-card-title class="pt-5 pb-2 px-5">Delete App</v-card-title>
        <v-card-text class="px-5 pb-2">
          <span>Are you sure you want to delete </span>
          <strong>{{ appToDelete?.name }}</strong>
          <span>? This action cannot be undone.</span>
        </v-card-text>
        <v-card-actions class="px-5 pb-4">
          <v-spacer />
          <v-btn variant="text" :disabled="deleting" @click="closeDeleteDialog">Cancel</v-btn>
          <v-btn
            color="error"
            variant="flat"
            :loading="deleting"
            @click="deleteApp"
          >
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'

interface App {
  id: string | number
  name: string
  mode?: string
  status?: string
  created_at?: string
  calls_count?: number
}

const router = useRouter()
const notificationStore = useNotificationStore()

// State
const apps = ref<App[]>([])
const loading = ref(false)

// Templates
const TEMPLATES = [
  {
    id: 'regression_monitor',
    name: 'Regression Monitor',
    description: 'Normalize → Window → Features → Model → Chart',
    icon: 'mdi-chart-timeline-variant',
    color: '#a78bfa',
    nodeLabels: ['Normalize', 'Window', 'Features', 'Model', 'Chart'],
    nodes: [
      { id: 'n1', type: 'input.csv_upload', config: { timestamp_col: 'timestamp', value_cols: 'value' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 32, step: 16 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      // model node added by user
      { id: 'n6', type: 'output.line_chart', config: { title: 'Predictions', target_column: '', show_anomalies: false } },
    ],
  },
  {
    id: 'anomaly_detector',
    name: 'Anomaly Detector',
    description: 'Normalize → Window → Features → Model → Alert',
    icon: 'mdi-shield-alert',
    color: '#f87171',
    nodeLabels: ['Normalize', 'Window', 'Features', 'Model', 'Alert'],
    nodes: [
      { id: 'n1', type: 'input.csv_upload', config: { timestamp_col: 'timestamp', value_cols: 'value' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 32, step: 16 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      { id: 'n6', type: 'output.alert_badge', config: { label_normal: 'Normal', label_anomaly: 'Anomaly Detected', webhook_url: '' } },
    ],
  },
  {
    id: 'classifier',
    name: 'Classifier',
    description: 'Normalize → Window → Features → Model → Table',
    icon: 'mdi-shape',
    color: '#34d399',
    nodeLabels: ['Normalize', 'Window', 'Features', 'Model', 'Table'],
    nodes: [
      { id: 'n1', type: 'input.csv_upload', config: { timestamp_col: 'timestamp', value_cols: 'value' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 128, step: 64 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      { id: 'n6', type: 'output.table', config: { max_rows: 50, show_confidence: true } },
    ],
  },
  // ── MQTT Live Stream Templates ──
  {
    id: 'live_regression',
    name: 'Live Regression',
    description: 'MQTT → Normalize → Window → Features → Model → Chart',
    icon: 'mdi-chart-timeline-variant',
    color: '#a78bfa',
    nodeLabels: ['MQTT', 'Normalize', 'Window', 'Features', 'Model', 'Chart'],
    nodes: [
      { id: 'n1', type: 'input.live_stream', config: { broker_url: 'ws://localhost:9001/mqtt', topic: 'sensors/machine1/#', channels: '' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 32, step: 16 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      { id: 'n6', type: 'output.line_chart', config: { title: 'Live Predictions', target_column: '', show_anomalies: false } },
    ],
  },
  {
    id: 'live_anomaly',
    name: 'Live Anomaly Detector',
    description: 'MQTT → Normalize → Window → Features → Model → Alert',
    icon: 'mdi-shield-alert',
    color: '#f87171',
    nodeLabels: ['MQTT', 'Normalize', 'Window', 'Features', 'Model', 'Alert'],
    nodes: [
      { id: 'n1', type: 'input.live_stream', config: { broker_url: 'ws://localhost:9001/mqtt', topic: 'sensors/machine1/#', channels: '' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 32, step: 16 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      { id: 'n6', type: 'output.alert_badge', config: { label_normal: 'Normal', label_anomaly: 'Anomaly Detected', webhook_url: '' } },
    ],
  },
  {
    id: 'live_classifier',
    name: 'Live Classifier',
    description: 'MQTT → Normalize → Window → Features → Model → Table',
    icon: 'mdi-shape',
    color: '#34d399',
    nodeLabels: ['MQTT', 'Normalize', 'Window', 'Features', 'Model', 'Table'],
    nodes: [
      { id: 'n1', type: 'input.live_stream', config: { broker_url: 'ws://localhost:9001/mqtt', topic: 'sensors/machine1/#', channels: '' } },
      { id: 'n2', type: 'transform.normalize', config: { method: 'minmax' } },
      { id: 'n3', type: 'transform.window', config: { window_size: 128, step: 64 } },
      { id: 'n4', type: 'transform.feature_extract', config: { features: [] } },
      { id: 'n6', type: 'output.table', config: { max_rows: 50, show_confidence: true } },
    ],
  },
  // ── Recording Template ──
  {
    id: 'mqtt_recorder',
    name: 'MQTT Signal Recorder',
    description: 'MQTT → Record & Label sensor data → Download CSV for training',
    icon: 'mdi-record-circle',
    color: '#ef4444',
    nodeLabels: ['MQTT', 'Recorder'],
    nodes: [
      { id: 'n1', type: 'input.live_stream', config: { broker_url: 'ws://localhost:9001/mqtt', topic: 'sensors/#', channels: '' } },
      { id: 'n6', type: 'output.signal_recorder', config: { labels: 'idle, wave, snake, updown', target_sample_rate: 62.5, max_duration: 300, file_prefix: 'sensor_data' } },
    ],
  },
  // ── Blank ──
  {
    id: 'blank',
    name: 'Blank',
    description: 'Start from scratch',
    icon: 'mdi-file-outline',
    color: '#94a3b8',
    nodeLabels: [],
    nodes: [],
  },
]

// MQTT Test Publisher
const mqttTestFile = ref('')
const mqttTestTopic = ref('sensors/test')
const mqttTestRate = ref(10)
const mqttTestLoop = ref(false)
const mqttDatasets = ref([])
const loadingDatasets = ref(false)
const mqttStarting = ref(false)
const mqttPublishing = ref(false)
const mqttPublished = ref(0)
const mqttTotal = ref(0)
const mqttSessionId = ref('')
const mqttBrokerStatus = ref(null)
const mqttError = ref('')
let mqttPollInterval = null

async function fetchMqttDatasets() {
  loadingDatasets.value = true
  try {
    const resp = await api.get('/api/mqtt/datasets')
    mqttDatasets.value = resp.data || []
  } catch { mqttDatasets.value = [] }
  loadingDatasets.value = false
}

async function checkMqttBroker() {
  try {
    const resp = await api.get('/api/mqtt/status')
    mqttBrokerStatus.value = resp.data?.broker_connected || false
    // Check if any publisher is still running
    const active = resp.data?.active_publishers || []
    const running = active.find(p => p.running)
    if (running) {
      mqttPublishing.value = true
      mqttPublished.value = running.published
      mqttTotal.value = running.total
      mqttSessionId.value = running.session_id
    }
  } catch { mqttBrokerStatus.value = null }
}

async function startMqttPublish() {
  if (!mqttTestFile.value) return
  mqttStarting.value = true
  mqttError.value = ''
  try {
    const resp = await api.post('/api/mqtt/publish', {
      file_path: mqttTestFile.value,
      topic: mqttTestTopic.value,
      rate: mqttTestRate.value,
      loop: mqttTestLoop.value,
    })
    mqttSessionId.value = resp.data.session_id
    mqttTotal.value = resp.data.total_rows
    mqttPublished.value = 0
    mqttPublishing.value = true
    // Poll for progress
    mqttPollInterval = setInterval(async () => {
      try {
        const s = await api.get('/api/mqtt/status')
        const pub = (s.data?.active_publishers || []).find(p => p.session_id === mqttSessionId.value)
        if (pub) {
          mqttPublished.value = pub.published
          if (!pub.running) {
            mqttPublishing.value = false
            clearInterval(mqttPollInterval)
          }
        }
      } catch {}
    }, 1000)
  } catch (e) {
    mqttError.value = e.response?.data?.error || 'Failed to start publisher'
  }
  mqttStarting.value = false
}

async function stopMqttPublish() {
  if (!mqttSessionId.value) return
  try {
    await api.post(`/api/mqtt/publish/${mqttSessionId.value}/stop`)
  } catch {}
  mqttPublishing.value = false
  if (mqttPollInterval) clearInterval(mqttPollInterval)
}

// Access control options
const accessOptions = [
  { value: 'public', label: 'Public', icon: 'mdi-earth', color: 'success', hint: 'Anyone with the link' },
  { value: 'team', label: 'Team', icon: 'mdi-account-group', color: 'info', hint: 'Logged-in users only' },
  { value: 'private', label: 'Private', icon: 'mdi-lock', color: 'warning', hint: 'Owner only + API key' },
]

async function updateAccess(app: App, newAccess: string) {
  try {
    await api.put(`/api/app-builder/apps/${app.id}`, { access: newAccess })
    app.access = newAccess
    notificationStore.showSuccess(`Access updated to "${newAccess}"`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to update access')
  }
}

// Create dialog
const showCreateDialog = ref(false)
const newAppName = ref('')
const createError = ref('')
const creating = ref(false)
const createNameField = ref<HTMLElement | null>(null)
const selectedTemplate = ref('regression_monitor')

// Delete dialog
const showDeleteDialog = ref(false)
const appToDelete = ref<App | null>(null)
const deleting = ref(false)

// Helpers
function modeColor(mode: string): string {
  switch (mode?.toLowerCase()) {
    case 'anomaly':        return 'error'
    case 'classification': return 'success'
    case 'regression':     return 'purple'
    default:               return 'grey'
  }
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return dateStr
  }
}

function navigateToApp(id: string | number) {
  router.push(`/app-builder/${id}`)
}

// Data fetching
async function fetchApps() {
  loading.value = true
  try {
    const resp = await api.get('/api/app-builder/apps')
    apps.value = resp.data ?? []
  } catch {
    apps.value = []
  } finally {
    loading.value = false
  }
}

// Create flow
function openCreateDialog() {
  newAppName.value = ''
  createError.value = ''
  selectedTemplate.value = 'regression_monitor'
  showCreateDialog.value = true
  // autofocus is handled by the text-field attribute, but nextTick ensures the dialog is rendered
  nextTick(() => {
    if (createNameField.value) {
      (createNameField.value as any)?.focus?.()
    }
  })
}

function closeCreateDialog() {
  showCreateDialog.value = false
  newAppName.value = ''
  createError.value = ''
}

async function createApp() {
  const name = newAppName.value.trim()
  if (!name) {
    createError.value = 'App name is required.'
    return
  }
  creating.value = true
  try {
    const tpl = TEMPLATES.find(t => t.id === selectedTemplate.value)
    const nodes = tpl ? JSON.parse(JSON.stringify(tpl.nodes)) : []

    // Auto-insert first active ME-LAB endpoint as model node (between feature_extract and output)
    if (tpl && tpl.id !== 'blank') {
      try {
        const epResp = await api.get('/api/melab/endpoints')
        const endpoints = (epResp.data || []).filter((e: any) => e.status === 'active')
        if (endpoints.length > 0) {
          const ep = endpoints[0]
          const modelNode = {
            id: `n_model_${Date.now()}`,
            type: `model.endpoint.${ep.id}`,
            config: ep.mode === 'regression' ? { horizon: 10 }
                  : ep.mode === 'anomaly' ? { threshold: 0.8, sensitivity: 'medium' }
                  : { top_k: 1 },
          }
          // Insert before the last node (output node)
          const outputIdx = nodes.findIndex((n: any) => n.type.startsWith('output.'))
          if (outputIdx >= 0) {
            nodes.splice(outputIdx, 0, modelNode)
          } else {
            nodes.push(modelNode)
          }
        }
      } catch { /* ME-LAB not available, user will add model manually */ }
    }

    const edges = nodes.slice(1).map((n: any, i: number) => ({
      id: `e${i}`, source: nodes[i].id, target: n.id,
    }))
    const resp = await api.post('/api/app-builder/apps', { name, nodes, edges })
    const newId = resp.data?.id ?? resp.data?.app?.id
    notificationStore.showSuccess(`App "${name}" created.`)
    closeCreateDialog()
    if (newId) {
      router.push(`/app-builder/${newId}`)
    } else {
      fetchApps()
    }
  } catch (e: any) {
    const msg = e.response?.data?.error ?? 'Failed to create app.'
    notificationStore.showError(msg)
    createError.value = msg
  } finally {
    creating.value = false
  }
}

// Delete flow
function openDeleteDialog(app: App) {
  appToDelete.value = app
  showDeleteDialog.value = true
}

function closeDeleteDialog() {
  showDeleteDialog.value = false
  appToDelete.value = null
}

async function deleteApp() {
  if (!appToDelete.value) return
  deleting.value = true
  try {
    await api.delete(`/api/app-builder/apps/${appToDelete.value.id}`)
    notificationStore.showSuccess(`App "${appToDelete.value.name}" deleted.`)
    closeDeleteDialog()
    fetchApps()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error ?? 'Failed to delete app.')
  } finally {
    deleting.value = false
  }
}

onMounted(() => {
  fetchApps()
  fetchMqttDatasets()
  checkMqttBroker()
})
</script>

<style scoped>
.template-card {
  transition: all 0.15s;
}
.template-card:hover {
  border-color: #555 !important;
}
.template-selected {
  background: rgba(167, 139, 250, 0.08) !important;
}
.published-link {
  font-size: 10px;
  font-family: monospace;
  color: #a78bfa;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
}
.published-link:hover {
  color: #c4b5fd;
  text-decoration: underline;
}
</style>
