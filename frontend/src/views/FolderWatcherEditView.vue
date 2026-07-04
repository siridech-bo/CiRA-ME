<template>
  <v-container fluid class="pa-6" style="max-width: 900px;">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <v-btn icon size="small" variant="text" class="mr-2" @click="cancel">
        <v-icon>mdi-arrow-left</v-icon>
      </v-btn>
      <div>
        <h1 class="text-h5 font-weight-bold">
          {{ isEdit ? 'Edit Watcher' : 'New Folder Watcher' }}
        </h1>
        <p class="text-body-2 text-medium-emphasis mb-0">
          Poll a folder, run each row through a ME-LAB model, write results to a CSV.
        </p>
      </div>
    </div>

    <v-card class="pa-6" :loading="loading">
      <v-form ref="formRef" @submit.prevent="save">
        <v-text-field
          v-model="form.name"
          label="Name"
          placeholder="e.g. Machine 001 monitor"
          variant="outlined"
          density="comfortable"
          :rules="[v => !!v || 'Name is required']"
          class="mb-4"
        />

        <v-select
          v-model="form.endpoint_id"
          :items="endpointOptions"
          item-title="label"
          item-value="value"
          label="Endpoint"
          placeholder="Pick a trained ME-LAB model"
          variant="outlined"
          density="comfortable"
          :rules="[v => !!v || 'Endpoint is required']"
          :loading="endpointsLoading"
          :disabled="isEdit"
          class="mb-4"
          :hint="isEdit ? 'Endpoint is fixed once created. Delete this watcher and create a new one to change models.' : 'Only active endpoints appear here'"
          persistent-hint
        />

        <v-text-field
          v-model="form.input_folder"
          label="Input Folder"
          :placeholder="defaultInputFolder"
          variant="outlined"
          density="comfortable"
          :rules="[v => !!v || 'Input folder is required']"
          class="mb-4"
          hint="Any path visible inside the backend container"
          persistent-hint
        />

        <v-text-field
          v-model="form.output_folder"
          label="Output Folder"
          :placeholder="defaultOutputFolder"
          variant="outlined"
          density="comfortable"
          :rules="[v => !!v || 'Output folder is required']"
          class="mb-4"
        />

        <div class="d-flex align-center gap-4 mb-4" style="flex-wrap: wrap;">
          <v-text-field
            v-model.number="form.poll_interval_s"
            label="Poll interval (seconds)"
            type="number"
            :min="10"
            :max="3600"
            variant="outlined"
            density="comfortable"
            style="max-width: 220px;"
            :rules="[
              v => (v !== null && v !== undefined && v !== '') || 'Required',
              v => v >= 10 || 'Minimum 10',
              v => v <= 3600 || 'Maximum 3600',
            ]"
          />
          <v-text-field
            v-model="form.file_glob"
            label="File glob"
            placeholder="*.txt"
            variant="outlined"
            density="comfortable"
            hint="e.g. *.csv or machine_*.txt"
            persistent-hint
            style="max-width: 260px;"
          />
        </div>

        <div class="mb-2 text-body-2 font-weight-medium">Header Mode</div>
        <v-radio-group
          v-model="form.header_mode"
          inline
          hide-details
          class="mb-6"
        >
          <v-radio label="Auto (detect from first row)" value="auto" />
          <v-radio label="Headered" value="headered" />
          <v-radio label="Headerless" value="headerless" />
        </v-radio-group>

        <div class="d-flex align-center justify-end gap-2">
          <v-btn variant="text" :disabled="saving" @click="cancel">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="saving"
            type="submit"
          >
            {{ isEdit ? 'Save Changes' : 'Create Watcher' }}
          </v-btn>
        </div>
      </v-form>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import api from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'

interface Endpoint {
  id: string
  name: string
  algorithm: string
  mode: string
  status: string
}

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const notify = useNotificationStore()

const formRef = ref<any>(null)

const isEdit = computed(() => !!route.params.id)
const watcherId = computed(() =>
  route.params.id ? Number(route.params.id) : null
)

const loading = ref(false)
const saving = ref(false)
const endpointsLoading = ref(false)

const endpoints = ref<Endpoint[]>([])

const form = ref({
  name: '',
  endpoint_id: '',
  input_folder: '',
  output_folder: '',
  poll_interval_s: 60,
  file_glob: '*.txt',
  header_mode: 'auto' as 'auto' | 'headered' | 'headerless',
})

// Slug-safe user id for default folder suggestions
const userSlug = computed(() => {
  const u = auth.user?.username || 'user'
  return String(u).replace(/[^a-zA-Z0-9_-]/g, '_')
})
const nameSlug = computed(() => {
  return (form.value.name || 'watcher')
    .toLowerCase()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'watcher'
})
const defaultInputFolder = computed(() =>
  `/app/watcher-data/${userSlug.value}/${nameSlug.value}/input`
)
const defaultOutputFolder = computed(() =>
  `/app/watcher-data/${userSlug.value}/${nameSlug.value}/output`
)

const endpointOptions = computed(() =>
  endpoints.value
    .filter(ep => ep.status === 'active')
    .map(ep => ({
      value: ep.id,
      label: `${ep.name} — ${ep.algorithm} (${ep.mode})`,
    }))
)

const loadEndpoints = async () => {
  try {
    endpointsLoading.value = true
    const res = await api.get('/api/melab/endpoints')
    endpoints.value = res.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load endpoints')
  } finally {
    endpointsLoading.value = false
  }
}

const loadWatcher = async () => {
  if (!watcherId.value) return
  try {
    loading.value = true
    const res = await api.get(`/api/folder-watchers/${watcherId.value}`)
    const w = res.data
    form.value = {
      name: w.name,
      endpoint_id: w.endpoint_id,
      input_folder: w.input_folder,
      output_folder: w.output_folder,
      poll_interval_s: w.poll_interval_s,
      file_glob: w.file_glob,
      header_mode: w.header_mode,
    }
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Watcher not found')
    router.push({ name: 'folder-watcher-list' })
  } finally {
    loading.value = false
  }
}

const save = async () => {
  const { valid } = await formRef.value.validate()
  if (!valid) return
  try {
    saving.value = true
    // Fill in defaults if user left folders blank (unlikely — rules require them,
    // but paste-blur-order edge cases exist).
    const payload: Record<string, any> = { ...form.value }
    if (!payload.input_folder) payload.input_folder = defaultInputFolder.value
    if (!payload.output_folder) payload.output_folder = defaultOutputFolder.value

    if (isEdit.value) {
      // endpoint_id is immutable on the backend PATCH route. Don't send it,
      // so the request doesn't get 400'd just because the form re-loaded it.
      delete payload.endpoint_id
      await api.patch(`/api/folder-watchers/${watcherId.value}`, payload)
      notify.showSuccess('Watcher updated')
    } else {
      await api.post('/api/folder-watchers/', payload)
      notify.showSuccess('Watcher created')
    }
    router.push({ name: 'folder-watcher-list' })
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to save watcher')
  } finally {
    saving.value = false
  }
}

const cancel = () => {
  router.push({ name: 'folder-watcher-list' })
}

onMounted(async () => {
  await loadEndpoints()
  if (isEdit.value) await loadWatcher()
})
</script>
