<template>
  <v-container fluid class="pa-6">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">Folder Watcher</h1>
        <p class="text-body-2 text-medium-emphasis">
          Poll a folder every N seconds and run each file's rows through a ME-LAB model
        </p>
      </div>
      <v-spacer />
      <v-btn color="primary" @click="createNew">
        <v-icon start>mdi-plus</v-icon>
        New Watcher
      </v-btn>
    </div>

    <!-- Watchers Table -->
    <v-card class="pa-4">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">
        <v-icon start size="small">mdi-folder-eye-outline</v-icon>
        Your Watchers
      </h3>

      <v-table v-if="watchers.length > 0" density="comfortable">
        <thead>
          <tr>
            <th>Name</th>
            <th>Endpoint</th>
            <th>Input Folder</th>
            <th>Status</th>
            <th class="text-center">Files</th>
            <th>Last Run</th>
            <th class="text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="w in watchers" :key="w.id">
            <td>
              <div class="font-weight-medium">{{ w.name }}</div>
              <div class="text-caption text-medium-emphasis">
                every {{ w.poll_interval_s }}s · {{ w.file_glob }}
              </div>
            </td>
            <td>
              <div class="text-body-2">{{ w.endpoint_name || w.endpoint_id }}</div>
              <div v-if="w.endpoint_algorithm" class="text-caption text-medium-emphasis">
                {{ w.endpoint_algorithm }}
              </div>
            </td>
            <td>
              <code class="text-caption">{{ w.input_folder }}</code>
            </td>
            <td>
              <v-chip
                size="x-small"
                variant="flat"
                :color="statusColor(w.status)"
              >
                <v-icon start size="10">mdi-circle</v-icon>
                {{ w.status }}
              </v-chip>
              <div v-if="w.status === 'error' && w.last_error" class="text-caption text-error mt-1" style="max-width: 240px; white-space: normal;">
                {{ w.last_error }}
              </div>
            </td>
            <td class="text-center">
              <div>{{ w.files_processed || 0 }}</div>
              <div class="text-caption text-medium-emphasis">
                {{ w.rows_processed || 0 }} rows
              </div>
            </td>
            <td class="text-caption">{{ formatDate(w.last_run_at) }}</td>
            <td class="text-right">
              <v-btn
                v-if="w.status !== 'running'"
                icon
                size="x-small"
                variant="text"
                color="success"
                title="Start Watcher"
                :loading="busy[w.id] === 'start'"
                @click="startWatcher(w)"
              >
                <v-icon size="small">mdi-play</v-icon>
              </v-btn>
              <v-btn
                v-else
                icon
                size="x-small"
                variant="text"
                color="warning"
                title="Stop Watcher"
                :loading="busy[w.id] === 'stop'"
                @click="stopWatcher(w)"
              >
                <v-icon size="small">mdi-stop</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="info"
                title="View files (input / output / errors)"
                @click="openFilesDialog(w)"
              >
                <v-icon size="small">mdi-folder-eye-outline</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="purple"
                title="Open public monitor view"
                :href="monitorUrl(w)"
                target="_blank"
              >
                <v-icon size="small">mdi-monitor-dashboard</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="grey"
                title="Copy monitor URL"
                @click="copyMonitorUrl(w)"
              >
                <v-icon size="small">mdi-content-copy</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="info"
                title="Edit"
                @click="editWatcher(w)"
              >
                <v-icon size="small">mdi-pencil-outline</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="error"
                title="Delete"
                @click="openDeleteDialog(w)"
              >
                <v-icon size="small">mdi-delete-outline</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>

      <!-- Empty state -->
      <div v-else-if="!loading" class="text-center pa-8">
        <v-icon size="48" color="grey" class="mb-3">mdi-folder-eye-outline</v-icon>
        <div class="text-body-1 text-medium-emphasis">No watchers yet.</div>
        <div class="text-caption text-medium-emphasis mt-1">
          Automate ML inference on files a machine or PLC writes to a folder.
        </div>
        <v-btn color="primary" variant="tonal" class="mt-4" @click="createNew">
          <v-icon start>mdi-plus</v-icon>
          New Watcher
        </v-btn>
      </div>

      <!-- Loading state -->
      <div v-else class="text-center pa-8">
        <v-progress-circular indeterminate color="primary" size="36" />
      </div>
    </v-card>

    <!-- Files Dialog: view input queue / output CSVs / errors -->
    <v-dialog v-model="showFilesDialog" max-width="900" scrollable>
      <v-card v-if="filesWatcher">
        <v-card-title class="pt-4 pb-2 px-5">
          <div class="d-flex align-center">
            <v-icon class="mr-2">mdi-folder-eye-outline</v-icon>
            <span class="text-truncate">{{ filesWatcher.name }}</span>
            <v-spacer />
            <v-btn icon size="small" variant="text" :loading="filesLoading" @click="loadFiles(filesWatcher!.id)">
              <v-icon>mdi-refresh</v-icon>
            </v-btn>
          </div>
        </v-card-title>

        <v-tabs v-model="filesTab" density="compact">
          <v-tab value="output">
            <v-icon start size="small">mdi-check-circle-outline</v-icon>
            Output <v-chip size="x-small" class="ml-2">{{ filesData.output.total }}</v-chip>
          </v-tab>
          <v-tab value="input">
            <v-icon start size="small">mdi-inbox</v-icon>
            Input Queue <v-chip size="x-small" class="ml-2">{{ filesData.input.total }}</v-chip>
          </v-tab>
          <v-tab value="error">
            <v-icon start size="small">mdi-alert-circle-outline</v-icon>
            Errors <v-chip size="x-small" class="ml-2" :color="filesData.error.total > 0 ? 'error' : undefined">{{ filesData.error.total }}</v-chip>
          </v-tab>
        </v-tabs>

        <v-card-text class="px-0" style="max-height: 60vh;">
          <v-window v-model="filesTab">
            <v-window-item v-for="kind in (['output','input','error'] as const)" :key="kind" :value="kind">
              <div class="px-5 pt-3">
                <div class="text-caption text-medium-emphasis mb-2">
                  Folder: <code>{{ filesData[kind].folder }}</code>
                  <template v-if="filesData[kind].total > filesData[kind].files.length">
                    &nbsp;·&nbsp; Showing newest {{ filesData[kind].files.length }} of {{ filesData[kind].total }}
                  </template>
                </div>

                <v-alert v-if="!filesLoading && filesData[kind].files.length === 0" type="info" variant="tonal" density="compact" class="mb-2">
                  <template v-if="kind === 'output'">No output files yet — nothing has been processed successfully.</template>
                  <template v-else-if="kind === 'input'">No files waiting in the input folder.</template>
                  <template v-else>No errored files — everything's been processed cleanly.</template>
                </v-alert>

                <v-table v-else density="compact" hover>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th class="text-right">Size</th>
                      <th>Modified</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="f in filesData[kind].files" :key="f.name"
                        :class="{ 'selected-row': selectedFile?.kind === kind && selectedFile?.name === f.name }"
                        @click="previewFile(kind, f.name)"
                        style="cursor: pointer">
                      <td class="font-weight-medium">{{ f.name }}</td>
                      <td class="text-right text-caption">{{ formatSize(f.size) }}</td>
                      <td class="text-caption">{{ formatDate(f.mtime * 1000) }}</td>
                      <td class="text-right">
                        <v-icon v-if="selectedFile?.kind === kind && selectedFile?.name === f.name" size="small" color="primary">mdi-eye</v-icon>
                      </td>
                    </tr>
                  </tbody>
                </v-table>

                <!-- File preview panel -->
                <div v-if="selectedFile && selectedFile.kind === kind" class="mt-3">
                  <div class="d-flex align-center mb-1">
                    <span class="text-caption font-weight-medium">Preview: {{ selectedFile.name }}</span>
                    <span v-if="preview?.truncated" class="text-caption text-warning ml-2">(truncated to first 200 KB — refresh to see fresh data)</span>
                    <v-spacer />
                    <v-btn size="x-small" variant="text" :loading="previewLoading" @click="loadPreview">
                      <v-icon start size="x-small">mdi-refresh</v-icon> Refresh
                    </v-btn>
                  </div>
                  <v-sheet
                    v-if="preview"
                    color="grey-darken-4"
                    class="pa-3"
                    rounded
                    style="max-height: 300px; overflow: auto;"
                  >
                    <pre style="margin: 0; font-size: 11px; white-space: pre; font-family: monospace;">{{ preview.content }}</pre>
                  </v-sheet>
                  <v-progress-linear v-else-if="previewLoading" indeterminate />
                </div>
              </div>
            </v-window-item>
          </v-window>
        </v-card-text>

        <v-card-actions class="px-5 pb-4">
          <v-spacer />
          <v-btn variant="text" @click="closeFilesDialog">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete Confirm Dialog -->
    <v-dialog v-model="showDeleteDialog" max-width="400">
      <v-card>
        <v-card-title class="pt-5 pb-2 px-5">Delete Watcher</v-card-title>
        <v-card-text class="px-5 pb-2">
          <span>Are you sure you want to delete </span>
          <strong>{{ watcherToDelete?.name }}</strong>
          <span>? The worker will stop and its history will be lost. Input/output folders on disk are NOT touched.</span>
        </v-card-text>
        <v-card-actions class="px-5 pb-4">
          <v-spacer />
          <v-btn variant="text" :disabled="deleting" @click="closeDeleteDialog">Cancel</v-btn>
          <v-btn
            color="error"
            variant="flat"
            :loading="deleting"
            @click="deleteWatcher"
          >
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'

interface Watcher {
  id: number
  name: string
  endpoint_id: string
  endpoint_name?: string | null
  endpoint_algorithm?: string | null
  endpoint_mode?: string | null
  input_folder: string
  output_folder: string
  poll_interval_s: number
  file_glob: string
  header_mode: string
  status: string
  last_run_at?: string | null
  last_error?: string | null
  files_processed?: number
  rows_processed?: number
}

const router = useRouter()
const notify = useNotificationStore()

const watchers = ref<Watcher[]>([])
const loading = ref(false)
const busy = ref<Record<number, string>>({})

const showDeleteDialog = ref(false)
const watcherToDelete = ref<Watcher | null>(null)
const deleting = ref(false)

// --- Files dialog state ---------------------------------------------------
interface FileInfo {
  name: string
  size: number
  mtime: number
}
interface FolderData {
  files: FileInfo[]
  total: number
  folder: string
}
const showFilesDialog = ref(false)
const filesWatcher = ref<Watcher | null>(null)
const filesTab = ref<'output' | 'input' | 'error'>('output')
const filesLoading = ref(false)
const filesData = ref<Record<'output' | 'input' | 'error', FolderData>>({
  output: { files: [], total: 0, folder: '' },
  input:  { files: [], total: 0, folder: '' },
  error:  { files: [], total: 0, folder: '' },
})
const selectedFile = ref<{ kind: 'output' | 'input' | 'error'; name: string } | null>(null)
const preview = ref<{ name: string; content: string; truncated: boolean; size: number } | null>(null)
const previewLoading = ref(false)

let pollTimer: ReturnType<typeof setInterval> | null = null

const load = async () => {
  try {
    loading.value = true
    const res = await api.get('/api/folder-watchers/')
    watchers.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load watchers')
  } finally {
    loading.value = false
  }
}

const createNew = () => {
  router.push({ name: 'folder-watcher-edit' })
}

const editWatcher = (w: Watcher) => {
  router.push({ name: 'folder-watcher-edit', params: { id: String(w.id) } })
}

const startWatcher = async (w: Watcher) => {
  try {
    busy.value[w.id] = 'start'
    await api.post(`/api/folder-watchers/${w.id}/start`)
    notify.showSuccess(`Watcher "${w.name}" started`)
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to start watcher')
  } finally {
    delete busy.value[w.id]
  }
}

const stopWatcher = async (w: Watcher) => {
  try {
    busy.value[w.id] = 'stop'
    await api.post(`/api/folder-watchers/${w.id}/stop`)
    notify.showSuccess(`Watcher "${w.name}" stopped`)
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to stop watcher')
  } finally {
    delete busy.value[w.id]
  }
}

const openDeleteDialog = (w: Watcher) => {
  watcherToDelete.value = w
  showDeleteDialog.value = true
}

const closeDeleteDialog = () => {
  showDeleteDialog.value = false
  watcherToDelete.value = null
}

const deleteWatcher = async () => {
  if (!watcherToDelete.value) return
  try {
    deleting.value = true
    await api.delete(`/api/folder-watchers/${watcherToDelete.value.id}`)
    notify.showSuccess('Watcher deleted')
    closeDeleteDialog()
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to delete watcher')
  } finally {
    deleting.value = false
  }
}

// --- Files dialog logic ---------------------------------------------------
const openFilesDialog = async (w: Watcher) => {
  filesWatcher.value = w
  filesTab.value = 'output'
  selectedFile.value = null
  preview.value = null
  showFilesDialog.value = true
  await loadFiles(w.id)
}

const closeFilesDialog = () => {
  showFilesDialog.value = false
  filesWatcher.value = null
  selectedFile.value = null
  preview.value = null
}

const loadFiles = async (id: number) => {
  try {
    filesLoading.value = true
    const res = await api.get(`/api/folder-watchers/${id}/files`)
    filesData.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load files')
  } finally {
    filesLoading.value = false
  }
}

const previewFile = async (kind: 'output' | 'input' | 'error', name: string) => {
  selectedFile.value = { kind, name }
  preview.value = null
  await loadPreview()
}

const loadPreview = async () => {
  if (!filesWatcher.value || !selectedFile.value) return
  try {
    previewLoading.value = true
    const { kind, name } = selectedFile.value
    const res = await api.get(
      `/api/folder-watchers/${filesWatcher.value.id}/files/${kind}/${encodeURIComponent(name)}`,
    )
    preview.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load file content')
    preview.value = null
  } finally {
    previewLoading.value = false
  }
}

// Build an absolute URL to the public monitor view. Uses window.location.origin
// so it works whether the SPA is served on localhost, a LAN IP, or a domain.
const monitorUrl = (w: Watcher): string => {
  const base = typeof window !== 'undefined' ? window.location.origin : ''
  return `${base}/watcher-view/${w.id}`
}

const copyMonitorUrl = async (w: Watcher) => {
  const url = monitorUrl(w)
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(url)
    } else {
      // Fallback for insecure contexts (http on LAN) — Clipboard API is
      // https-only in most browsers.
      const ta = document.createElement('textarea')
      ta.value = url
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    notify.showSuccess('Monitor URL copied to clipboard')
  } catch {
    notify.showError('Could not copy URL. Copy manually: ' + url)
  }
}

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const statusColor = (status: string) => {
  switch (status) {
    case 'running': return 'success'
    case 'error':   return 'error'
    default:        return 'grey'
  }
}

const formatDate = (dt?: string | number | null) => {
  if (dt === null || dt === undefined || dt === '') return '—'
  try {
    const d = new Date(dt as any)
    if (isNaN(d.getTime())) return String(dt)
    return d.toLocaleString()
  } catch { return String(dt) }
}

onMounted(() => {
  load()
  // Refresh every 15s so counters/status stay fresh while user is on the page
  pollTimer = setInterval(load, 15000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
