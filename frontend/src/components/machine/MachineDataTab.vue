<template>
  <v-card variant="flat">
    <div class="d-flex align-center pa-3 flex-wrap ga-2">
      <div>
        <div class="text-subtitle-1 font-weight-medium">Files</div>
        <div class="text-caption text-medium-emphasis">
          Scoped to
          <code>data/{{ relPath }}</code>
        </div>
      </div>
      <v-spacer />
      <v-btn
        variant="text"
        size="small"
        :loading="loading"
        prepend-icon="mdi-refresh"
        @click="load"
      >
        Refresh
      </v-btn>
      <v-btn
        color="primary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-upload"
        @click="triggerUpload"
      >
        Upload files
      </v-btn>
      <input
        ref="uploadInput"
        type="file"
        multiple
        hidden
        @change="onUpload"
      />
    </div>
    <v-divider />

    <!-- Path status alert -->
    <div v-if="loadError" class="pa-3">
      <v-alert type="warning" density="compact" variant="tonal">
        <div class="d-flex align-center flex-wrap ga-2">
          <span>
            {{ loadError }}
          </span>
          <v-spacer />
          <v-btn
            v-if="canCreate"
            size="small"
            color="warning"
            variant="text"
            prepend-icon="mdi-folder-plus"
            :loading="creating"
            @click="createMachineFolder"
          >
            Create folder
          </v-btn>
        </div>
      </v-alert>
    </div>

    <!-- Empty folder -->
    <div
      v-else-if="items.length === 0 && !loading"
      class="pa-6 text-center"
    >
      <v-icon size="48" color="grey">mdi-folder-open-outline</v-icon>
      <p class="text-body-2 text-medium-emphasis mt-2">
        No files here yet. Upload sensor CSVs to make them available to
        this machine's pipeline runs.
      </p>
      <v-btn
        color="primary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-upload"
        @click="triggerUpload"
      >
        Upload files
      </v-btn>
    </div>

    <!-- File table -->
    <v-table v-else density="compact">
      <thead>
        <tr>
          <th style="width: 40px"></th>
          <th>Name</th>
          <th class="text-right" style="width: 120px">Size</th>
          <th class="text-right" style="width: 180px">Modified</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="item in items" :key="item.path">
          <td>
            <v-icon size="16" :color="item.is_dir ? 'amber-darken-2' : 'grey'">
              {{ item.is_dir ? 'mdi-folder' : fileIcon(item.extension) }}
            </v-icon>
          </td>
          <td>{{ item.name }}</td>
          <td class="text-right">{{ item.is_dir ? '—' : formatSize(item.size) }}</td>
          <td class="text-right text-caption">{{ formatModified(item) }}</td>
        </tr>
        <tr v-if="loading">
          <td colspan="4" class="text-center py-3">
            <v-progress-circular indeterminate size="18" width="2" color="primary" />
          </td>
        </tr>
      </tbody>
    </v-table>

    <!-- Upload progress -->
    <v-progress-linear
      v-if="uploading"
      :model-value="uploadProgress"
      color="primary"
      height="3"
    />
  </v-card>
</template>

<script setup lang="ts">
/**
 * Phase B.4 — Data tab.
 * A lightweight inline browser targeting the machine's data folder. Full
 * folder manipulation opens the shared FileManagerDialog modal (existing
 * component) — the spec forbids embedding it inline, but as an on-demand
 * modal it's the right escape hatch for rename/move/delete.
 */
import { ref, computed, watch, onMounted } from 'vue'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'
import type { AssetNode } from '@/stores/assetTree'

const props = defineProps<{ machine: AssetNode }>()
const notify = useNotificationStore()

interface FileItem {
  name: string
  path: string
  is_dir: boolean
  extension: string | null
  size: number | null
  modified?: number
}

const items = ref<FileItem[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)
const uploadInput = ref<HTMLInputElement | null>(null)
const uploading = ref(false)
const uploadProgress = ref(0)
const creating = ref(false)
const datasetsRoot = ref<string>('')
const separator = ref<'/' | '\\'>('/')

const relPath = computed(() => props.machine.topic_path.replace(/\./g, '/'))
const folderPath = computed(() => {
  if (!datasetsRoot.value) return ''
  const sep = separator.value
  return `${datasetsRoot.value}${sep}${relPath.value.replaceAll('/', sep)}`
})

// The parent must exist for "Create folder" to be safe. We only surface
// the create button when the error is "Path not found".
const canCreate = computed(() => loadError.value?.toLowerCase().includes('not found'))

watch(() => props.machine?.id, () => load())

async function ensureRoot(): Promise<string | null> {
  if (datasetsRoot.value) return datasetsRoot.value
  try {
    const r = await api.get('/api/data/datasets-root')
    datasetsRoot.value = String(r.data?.path || '')
    // Detect separator from the root string. Windows datasets_root will
    // contain backslashes; POSIX uses forward slashes.
    if (datasetsRoot.value.includes('\\')) separator.value = '\\'
    else separator.value = '/'
    return datasetsRoot.value
  } catch {
    notify.showError('Failed to resolve datasets root')
    return null
  }
}

async function load() {
  loading.value = true
  loadError.value = null
  items.value = []
  const root = await ensureRoot()
  if (!root) { loading.value = false; return }
  try {
    const r = await api.post('/api/data/browse', { path: folderPath.value })
    items.value = r.data?.items || []
  } catch (e: any) {
    loadError.value = e.response?.data?.error || 'Failed to load folder'
  } finally {
    loading.value = false
  }
}

async function createMachineFolder() {
  // Walk the topic-path segments, creating each missing directory. Backend
  // rejects paths outside DATASETS_ROOT; we rely on that guardrail.
  creating.value = true
  try {
    const root = await ensureRoot()
    if (!root) return
    const sep = separator.value
    const parts = relPath.value.split('/').filter(Boolean)
    let parent = root
    for (const p of parts) {
      try {
        await api.post('/api/data/mkdir', { folder: parent, name: p })
      } catch (e: any) {
        // Ignore "already exists"; anything else surfaces.
        const err = e.response?.data?.error || ''
        if (!/exist/i.test(err)) throw e
      }
      parent = `${parent}${sep}${p}`
    }
    notify.showSuccess('Folder created')
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to create folder')
  } finally {
    creating.value = false
  }
}

function triggerUpload() {
  if (loadError.value && canCreate.value) {
    // Auto-create the folder before triggering the file picker so the first
    // upload lands somewhere valid.
    createMachineFolder().then(() => uploadInput.value?.click())
    return
  }
  uploadInput.value?.click()
}

async function onUpload(evt: Event) {
  const target = evt.target as HTMLInputElement
  const files = target.files ? Array.from(target.files) : []
  if (files.length === 0) return
  uploading.value = true
  uploadProgress.value = 0
  let done = 0
  try {
    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('folder', folderPath.value)
      try {
        await api.post('/api/data/upload', fd, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      } catch (e: any) {
        notify.showError(`Upload failed for ${file.name}: ${e.response?.data?.error || e.message}`)
      }
      done++
      uploadProgress.value = Math.round((done / files.length) * 100)
    }
    notify.showSuccess(`Uploaded ${done} file(s)`)
    await load()
  } finally {
    uploading.value = false
    if (uploadInput.value) uploadInput.value.value = ''
  }
}

function fileIcon(ext: string | null | undefined) {
  switch ((ext || '').toLowerCase()) {
    case '.csv': return 'mdi-file-delimited-outline'
    case '.json': return 'mdi-code-json'
    case '.cbor': return 'mdi-file-code-outline'
    case '.txt':
    case '.tsv':
    case '.log': return 'mdi-file-document-outline'
    default: return 'mdi-file-outline'
  }
}
function formatSize(bytes: number | null | undefined) {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
function formatModified(item: FileItem) {
  if (!item.modified) return '—'
  try { return new Date(item.modified * 1000).toLocaleString() }
  catch { return '—' }
}

onMounted(load)
</script>
