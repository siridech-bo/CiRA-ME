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

const statusColor = (status: string) => {
  switch (status) {
    case 'running': return 'success'
    case 'error':   return 'error'
    default:        return 'grey'
  }
}

const formatDate = (dt?: string | null) => {
  if (!dt) return '—'
  try {
    const d = new Date(dt)
    if (isNaN(d.getTime())) return dt
    return d.toLocaleString()
  } catch { return dt }
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
