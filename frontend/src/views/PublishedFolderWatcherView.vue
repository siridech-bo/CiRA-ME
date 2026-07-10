<template>
  <div class="folder-watcher-app" :class="{ 'dashboard-mode': dashboardMode }">
    <!-- Loading -->
    <div v-if="loading && !watcher" class="loading-container">
      <v-progress-circular indeterminate color="primary" size="36" />
      <div class="text-caption text-medium-emphasis mt-3">Loading watcher…</div>
    </div>

    <!-- Error -->
    <v-alert v-else-if="error" type="error" variant="tonal" class="ma-6">
      {{ error }}
    </v-alert>

    <!-- Content -->
    <template v-else-if="watcher">
      <!-- Header -->
      <div class="fw-header">
        <div class="fw-header-left">
          <v-icon size="28" color="orange">mdi-file-search-outline</v-icon>
          <div>
            <div class="fw-title">{{ watcher.name }}</div>
            <div class="fw-subtitle">Log Watcher monitor · auto-refresh 5 s</div>
          </div>
        </div>
        <div class="fw-header-right">
          <v-chip size="small" variant="flat" :color="statusColor(watcher.status)">
            <v-icon start size="12">mdi-circle</v-icon>
            {{ watcher.status }}
          </v-chip>
          <v-btn
            size="small"
            variant="tonal"
            color="grey"
            :title="isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'"
            @click="toggleFullscreen"
          >
            <v-icon size="small">{{ isFullscreen ? 'mdi-fullscreen-exit' : 'mdi-fullscreen' }}</v-icon>
          </v-btn>
        </div>
      </div>

      <!-- ═══ Dashboard (two-pane) layout ═══ -->
      <div v-if="dashboardMode" class="fw-body">
        <!-- LEFT RAIL -->
        <div class="fw-rail">
          <div class="rail-section-title">
            <v-icon size="14" color="orange">mdi-cog-outline</v-icon>Watcher Config
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">Input folder</div>
            <div class="fw-cfg-val fw-cfg-mono">{{ files.input.folder || '—' }}</div>
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">Output folder</div>
            <div class="fw-cfg-val fw-cfg-mono">{{ files.output.folder || '—' }}</div>
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">Error folder</div>
            <div class="fw-cfg-val fw-cfg-mono">{{ files.error.folder || '—' }}</div>
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">Parse mode</div>
            <div class="fw-cfg-val">{{ watcher.parse_mode || 'csv' }}</div>
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">MQTT publishing</div>
            <div class="fw-cfg-val">
              <v-chip size="x-small" :color="watcher.mqtt_enabled ? 'success' : 'grey'" variant="tonal">
                {{ watcher.mqtt_enabled ? 'ENABLED' : 'off' }}
              </v-chip>
            </div>
          </div>
          <div class="fw-cfg-row">
            <div class="fw-cfg-label">Daily CSV</div>
            <div class="fw-cfg-val">
              <v-chip size="x-small" :color="watcher.daily_csv_enabled ? 'success' : 'grey'" variant="tonal">
                {{ watcher.daily_csv_enabled ? 'ENABLED' : 'off' }}
              </v-chip>
            </div>
          </div>

          <v-alert
            v-if="watcher.status === 'error' && watcher.last_error"
            type="error"
            variant="tonal"
            class="mt-3"
            density="compact"
          >
            {{ watcher.last_error }}
          </v-alert>
        </div>

        <!-- RIGHT PANE -->
        <div class="fw-main">
          <!-- Big status cards -->
          <div class="fw-stat-grid">
            <div class="fw-stat-card">
              <div class="fw-stat-label">Files processed</div>
              <div class="fw-stat-value">{{ watcher.files_processed || 0 }}</div>
            </div>
            <div class="fw-stat-card">
              <div class="fw-stat-label">Rows processed</div>
              <div class="fw-stat-value">{{ (watcher.rows_processed || 0).toLocaleString() }}</div>
            </div>
            <div class="fw-stat-card">
              <div class="fw-stat-label">Last run</div>
              <div class="fw-stat-value fw-stat-value-sm">{{ formatDate(watcher.last_run_at) }}</div>
            </div>
            <div class="fw-stat-card">
              <div class="fw-stat-label">Status</div>
              <div class="fw-stat-value fw-stat-value-sm" :style="{ color: statusTextColor(watcher.status) }">
                {{ watcher.status }}
              </div>
            </div>
          </div>

          <!-- Recent files -->
          <div class="fw-files-card">
            <div class="fw-files-head">
              <div class="rail-section-title" style="margin: 0;">
                <v-icon size="14" color="orange">mdi-folder-outline</v-icon>Recent Files
              </div>
              <v-spacer />
              <v-btn size="x-small" variant="text" :loading="filesLoading" @click="loadFiles">
                <v-icon start size="small">mdi-refresh</v-icon>Refresh
              </v-btn>
            </div>

            <v-tabs v-model="kindTab" density="compact" class="mb-1">
              <v-tab value="output">
                <v-icon start size="small">mdi-check-circle-outline</v-icon>
                Output <v-chip size="x-small" class="ml-2">{{ files.output.total }}</v-chip>
              </v-tab>
              <v-tab value="input">
                <v-icon start size="small">mdi-inbox</v-icon>
                Input <v-chip size="x-small" class="ml-2">{{ files.input.total }}</v-chip>
              </v-tab>
              <v-tab value="error">
                <v-icon start size="small">mdi-alert-circle-outline</v-icon>
                Errors <v-chip size="x-small" class="ml-2" :color="files.error.total > 0 ? 'error' : undefined">{{ files.error.total }}</v-chip>
              </v-tab>
            </v-tabs>

            <div class="fw-files-body">
              <v-alert v-if="!filesLoading && files[kindTab].files.length === 0" type="info" variant="tonal" density="compact">
                Nothing here yet.
              </v-alert>
              <v-table v-else density="compact" class="fw-files-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th class="text-right">Size</th>
                    <th>Modified</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="f in files[kindTab].files"
                    :key="f.name"
                    class="fw-file-row"
                    @click="openPreview(kindTab, f.name)"
                  >
                    <td class="font-weight-medium">
                      <v-icon size="14" class="mr-1" color="grey">mdi-file-outline</v-icon>
                      {{ f.name }}
                    </td>
                    <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                    <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
                  </tr>
                </tbody>
              </v-table>
            </div>
          </div>
        </div>
      </div>

      <!-- ═══ Fallback (stacked) layout for narrow screens ═══ -->
      <v-container v-else fluid class="pa-6">
        <!-- Error banner -->
        <v-alert
          v-if="watcher.status === 'error' && watcher.last_error"
          type="error"
          variant="tonal"
          class="mb-4"
          density="compact"
        >
          {{ watcher.last_error }}
        </v-alert>

        <!-- Summary stats -->
        <v-row dense class="mb-4">
          <v-col cols="12" sm="6" md="3">
            <v-card variant="tonal" class="pa-3">
              <div class="text-caption text-medium-emphasis">Files processed</div>
              <div class="text-h5 font-weight-bold">{{ watcher.files_processed || 0 }}</div>
            </v-card>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <v-card variant="tonal" class="pa-3">
              <div class="text-caption text-medium-emphasis">Rows processed</div>
              <div class="text-h5 font-weight-bold">{{ watcher.rows_processed || 0 }}</div>
            </v-card>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <v-card variant="tonal" class="pa-3">
              <div class="text-caption text-medium-emphasis">Last run</div>
              <div class="text-body-2 font-weight-medium">{{ formatDate(watcher.last_run_at) }}</div>
            </v-card>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <v-card variant="tonal" class="pa-3">
              <div class="text-caption text-medium-emphasis">Parse mode</div>
              <div class="text-body-2 font-weight-medium">
                {{ watcher.parse_mode || 'csv' }}
              </div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Files table -->
        <v-card class="pa-4">
          <div class="d-flex align-center mb-3">
            <h3 class="text-subtitle-1 font-weight-bold">Recent Files</h3>
            <v-spacer />
            <v-btn size="x-small" variant="text" :loading="filesLoading" @click="loadFiles">
              <v-icon start size="small">mdi-refresh</v-icon> Refresh
            </v-btn>
          </div>

          <v-tabs v-model="kindTab" density="compact" class="mb-2">
            <v-tab value="output">
              <v-icon start size="small">mdi-check-circle-outline</v-icon>
              Output <v-chip size="x-small" class="ml-2">{{ files.output.total }}</v-chip>
            </v-tab>
            <v-tab value="input">
              <v-icon start size="small">mdi-inbox</v-icon>
              Input Queue <v-chip size="x-small" class="ml-2">{{ files.input.total }}</v-chip>
            </v-tab>
            <v-tab value="error">
              <v-icon start size="small">mdi-alert-circle-outline</v-icon>
              Errors <v-chip size="x-small" class="ml-2" :color="files.error.total > 0 ? 'error' : undefined">{{ files.error.total }}</v-chip>
            </v-tab>
          </v-tabs>

          <div class="text-caption text-medium-emphasis mb-2">
            Folder: <code>{{ files[kindTab].folder }}</code>
          </div>

          <v-alert v-if="!filesLoading && files[kindTab].files.length === 0" type="info" variant="tonal" density="compact">
            Nothing here yet.
          </v-alert>

          <v-table v-else density="compact">
            <thead>
              <tr>
                <th>Name</th>
                <th class="text-right">Size</th>
                <th>Modified</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="f in files[kindTab].files"
                :key="f.name"
                class="fw-file-row"
                @click="openPreview(kindTab, f.name)"
              >
                <td class="font-weight-medium">
                  <v-icon size="14" class="mr-1" color="grey">mdi-file-outline</v-icon>
                  {{ f.name }}
                </td>
                <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-container>
    </template>

    <!-- File preview dialog -->
    <v-dialog v-model="previewOpen" max-width="900" scrollable>
      <v-card v-if="preview.name">
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="primary">mdi-file-eye-outline</v-icon>
          <span class="text-truncate">{{ preview.name }}</span>
          <v-chip size="x-small" class="ml-2" variant="tonal" :color="preview.kind === 'error' ? 'error' : 'default'">
            {{ preview.kind }}
          </v-chip>
          <v-spacer />
          <span class="text-caption text-medium-emphasis mr-2">
            {{ formatSize(preview.size) }}{{ preview.truncated ? ' · truncated' : '' }}
          </span>
          <v-btn icon variant="text" size="small" @click="previewOpen = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>
        <v-divider />
        <v-card-text style="max-height: 70vh; padding: 0;">
          <v-progress-linear v-if="preview.loading" indeterminate color="primary" />
          <v-alert v-else-if="preview.error" type="error" variant="tonal" density="compact" class="ma-3">
            {{ preview.error }}
          </v-alert>
          <pre v-else class="fw-preview-pre">{{ preview.content }}</pre>
        </v-card-text>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '@/services/api'

interface WatcherSummary {
  id: number
  name: string
  status: string
  last_run_at?: string | null
  last_error?: string | null
  files_processed?: number
  rows_processed?: number
  parse_mode?: string
  mqtt_enabled?: boolean
  daily_csv_enabled?: boolean
}

interface FileInfo { name: string; size: number; mtime: number }
interface FolderData { files: FileInfo[]; total: number; folder: string }

const route = useRoute()
const watcherId = Number(route.params.id)

const loading = ref(false)
const watcher = ref<WatcherSummary | null>(null)
const error = ref('')

const kindTab = ref<'output' | 'input' | 'error'>('output')
const files = ref<Record<'output' | 'input' | 'error', FolderData>>({
  output: { files: [], total: 0, folder: '' },
  input:  { files: [], total: 0, folder: '' },
  error:  { files: [], total: 0, folder: '' },
})
const filesLoading = ref(false)

// ── File preview dialog ───────────────────────────
const previewOpen = ref(false)
const preview = ref<{
  kind: 'output' | 'input' | 'error' | ''
  name: string
  content: string
  size: number
  truncated: boolean
  loading: boolean
  error: string
}>({
  kind: '',
  name: '',
  content: '',
  size: 0,
  truncated: false,
  loading: false,
  error: '',
})

async function openPreview(kind: 'output' | 'input' | 'error', name: string) {
  const id = watcher.value?.id
  if (!id) return
  preview.value = {
    kind,
    name,
    content: '',
    size: 0,
    truncated: false,
    loading: true,
    error: '',
  }
  previewOpen.value = true
  try {
    const resp = await api.get(`/api/folder-watchers/${id}/files/${kind}/${encodeURIComponent(name)}`)
    preview.value.content = resp.data.content ?? ''
    preview.value.size = resp.data.size ?? 0
    preview.value.truncated = !!resp.data.truncated
  } catch (e: any) {
    preview.value.error = e.response?.data?.error || 'Failed to load file'
  } finally {
    preview.value.loading = false
  }
}

let refreshTimer: ReturnType<typeof setInterval> | null = null

// ── Dashboard layout state ─────────────────────────
const viewportWidth = ref(typeof window !== 'undefined' ? window.innerWidth : 1920)
function onResize() { viewportWidth.value = window.innerWidth }
const dashboardMode = computed(() => viewportWidth.value >= 1200)

const isFullscreen = ref(false)
function updateFullscreenState() { isFullscreen.value = !!document.fullscreenElement }
async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen()
    } else {
      await document.exitFullscreen()
    }
  } catch (e) {
    console.warn('Fullscreen toggle failed:', e)
  }
}

const loadWatcher = async () => {
  try {
    loading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId}`)
    watcher.value = res.data
    error.value = ''
  } catch (e: any) {
    // 401 / 404 both surface as an error card; the poll below keeps trying.
    error.value = e.response?.data?.error || 'Watcher not found'
  } finally {
    loading.value = false
  }
}

const loadFiles = async () => {
  try {
    filesLoading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId}/files`)
    files.value = res.data
  } catch (e: any) {
    // Silent — file listing errors shouldn't take over the whole view when
    // the watcher summary still loads fine.
  } finally {
    filesLoading.value = false
  }
}

const refreshAll = async () => {
  await Promise.all([loadWatcher(), loadFiles()])
}

const statusColor = (status: string) => {
  switch (status) {
    case 'running': return 'success'
    case 'error':   return 'error'
    default:        return 'grey'
  }
}

const statusTextColor = (status: string) => {
  switch (status) {
    case 'running': return '#34d399'
    case 'error':   return '#f87171'
    default:        return '#8b949e'
  }
}

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const formatDate = (dt?: string | number | null) => {
  if (dt === null || dt === undefined || dt === '') return '—'
  try {
    const d = new Date(dt as any)
    if (isNaN(d.getTime())) return String(dt)
    return d.toLocaleString()
  } catch { return String(dt) }
}

onMounted(async () => {
  await refreshAll()
  // Wall-display refresh cadence — 5 s matches spec.
  refreshTimer = setInterval(refreshAll, 5000)
  window.addEventListener('resize', onResize)
  document.addEventListener('fullscreenchange', updateFullscreenState)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  window.removeEventListener('resize', onResize)
  document.removeEventListener('fullscreenchange', updateFullscreenState)
})
</script>

<style scoped>
.folder-watcher-app {
  background: #0d1117;
  color: #e6edf3;
  min-height: 100vh;
}
.folder-watcher-app.dashboard-mode {
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
}

/* Header */
.fw-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid #21262d;
  min-height: 56px;
  flex-shrink: 0;
}
.fw-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.fw-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.fw-title {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.1;
}
.fw-subtitle {
  font-size: 11px;
  color: #8b949e;
  margin-top: 2px;
}

/* Two-pane body */
.fw-body {
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
  overflow: hidden;
}

/* Left rail */
.fw-rail {
  width: 280px;
  flex-shrink: 0;
  border-right: 1px solid #21262d;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  background: #0d1117;
}
.rail-section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 700;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.fw-cfg-row {
  padding-bottom: 6px;
  border-bottom: 1px solid #161b22;
}
.fw-cfg-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 2px;
}
.fw-cfg-val {
  font-size: 12px;
  color: #e6edf3;
}
.fw-cfg-mono {
  font-family: monospace;
  font-size: 11px;
  word-break: break-all;
  color: #c9d1d9;
}

/* Right pane */
.fw-main {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 14px;
  overflow: hidden;
}

.fw-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  flex-shrink: 0;
}
.fw-stat-card {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px 18px;
}
.fw-stat-label {
  font-size: 11px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.fw-stat-value {
  font-size: 32px;
  font-weight: 700;
  font-family: monospace;
  line-height: 1.05;
  color: #e6edf3;
}
.fw-stat-value-sm {
  font-size: 16px;
}

.fw-files-card {
  flex: 1 1 auto;
  min-height: 0;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.fw-files-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.fw-files-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  margin-top: 4px;
}
.fw-files-table {
  background: transparent !important;
}
.fw-file-row {
  cursor: pointer;
  transition: background 0.12s ease;
}
.fw-file-row:hover td {
  background: rgba(99, 102, 241, 0.08);
}
.fw-preview-pre {
  margin: 0;
  padding: 14px 16px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
  line-height: 1.5;
  color: #e6edf3;
  background: #0d1117;
  white-space: pre;
  overflow-x: auto;
  max-height: 70vh;
}
</style>
