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
                {{ app.status || 'draft' }}
              </v-chip>
            </td>
            <td class="text-caption">{{ formatDate(app.created_at) }}</td>
            <td class="text-center">{{ app.calls_count ?? 0 }}</td>
            <td>
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

    <!-- Create App Dialog -->
    <v-dialog v-model="showCreateDialog" max-width="440" @keydown.enter="createApp">
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
            @input="createError = ''"
          />
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

// Create dialog
const showCreateDialog = ref(false)
const newAppName = ref('')
const createError = ref('')
const creating = ref(false)
const createNameField = ref<HTMLElement | null>(null)

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
    const resp = await api.post('/api/app-builder/apps', { name })
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
})
</script>
