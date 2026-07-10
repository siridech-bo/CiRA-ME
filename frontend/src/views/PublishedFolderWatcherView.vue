<template>
  <v-container fluid class="pa-6">
    <!-- Loading -->
    <div v-if="loading && !watcher" class="text-center pa-8">
      <v-progress-circular indeterminate color="primary" size="36" />
      <div class="text-caption text-medium-emphasis mt-3">Loading watcher…</div>
    </div>

    <!-- Error -->
    <v-alert v-else-if="error" type="error" variant="tonal" class="mb-4">
      {{ error }}
    </v-alert>

    <!-- Content -->
    <template v-else-if="watcher">
      <!-- Header -->
      <div class="d-flex align-center mb-4" style="gap: 12px;">
        <v-icon size="32" color="orange">mdi-file-search-outline</v-icon>
        <div class="flex-grow-1">
          <h1 class="text-h5 font-weight-bold mb-0">{{ watcher.name }}</h1>
          <div class="text-caption text-medium-emphasis">
            Log Watcher monitor · auto-refresh every 5 s
          </div>
        </div>
        <v-chip
          size="small"
          variant="flat"
          :color="statusColor(watcher.status)"
        >
          <v-icon start size="12">mdi-circle</v-icon>
          {{ watcher.status }}
        </v-chip>
      </div>

      <!-- Error banner from watcher itself -->
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
            <tr v-for="f in files[kindTab].files" :key="f.name">
              <td class="font-weight-medium">{{ f.name }}</td>
              <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
              <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
            </tr>
          </tbody>
        </v-table>
      </v-card>
    </template>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
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

let refreshTimer: ReturnType<typeof setInterval> | null = null

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
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})
</script>
