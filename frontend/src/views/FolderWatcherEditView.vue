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
          @update:model-value="onInputFolderInput"
        />

        <v-text-field
          v-model="form.output_folder"
          label="Output Folder"
          :placeholder="defaultOutputFolder"
          variant="outlined"
          density="comfortable"
          :rules="[v => !!v || 'Output folder is required']"
          class="mb-4"
          @update:model-value="onOutputFolderInput"
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

        <!-- ── Log Watcher: parse mode + optional regex ───────────────── -->
        <v-select
          v-model="form.parse_mode"
          :items="parseModeOptions"
          item-title="label"
          item-value="value"
          label="Parse mode"
          variant="outlined"
          density="comfortable"
          class="mb-4"
          hint="How each line inside a file is turned into a row"
          persistent-hint
        />

        <v-textarea
          v-if="form.parse_mode === 'regex'"
          v-model="form.parse_regex"
          label="Parse regex"
          placeholder="(?P<time>\S+)\s+temp=(?P<temperature>\d+\.\d+)\s+vib=(?P<vibration>\d+\.\d+)"
          variant="outlined"
          density="comfortable"
          rows="2"
          auto-grow
          class="mb-4"
          style="font-family: monospace;"
          hint="Python regex with named capture groups. Each match becomes a row."
          persistent-hint
          :rules="[
            v => form.parse_mode !== 'regex' || (!!v && !!v.trim()) || 'Regex is required'
          ]"
        />

        <!-- ── Log Watcher: MQTT publish sink ─────────────────────────── -->
        <v-switch
          v-model="form.mqtt_enabled"
          label="Publish predictions to MQTT"
          color="primary"
          density="comfortable"
          hide-details
          class="mb-2"
        />
        <v-text-field
          v-if="form.mqtt_enabled"
          v-model="form.mqtt_topic"
          label="MQTT topic"
          placeholder="alerts/{name}"
          variant="outlined"
          density="comfortable"
          class="mb-4"
          hint="{name} is replaced with the watcher's slug at publish time"
          persistent-hint
          :rules="[
            v => !form.mqtt_enabled || (!!v && !!String(v).trim()) || 'Topic is required'
          ]"
        />

        <!-- ── Log Watcher: daily aggregated CSV sink ─────────────────── -->
        <v-switch
          v-model="form.daily_csv_enabled"
          label="Write daily aggregated CSV"
          color="primary"
          density="comfortable"
          hide-details
          class="mb-2"
          :hint="dailyCsvHint"
          persistent-hint
        />
        <div class="mb-6" />

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
import { ref, computed, onMounted, watch, nextTick } from 'vue'
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
  parse_mode: 'csv' as 'csv' | 'regex' | 'json',
  parse_regex: '',
  mqtt_enabled: false,
  mqtt_topic: 'alerts/{name}',
  daily_csv_enabled: false,
})

const parseModeOptions = [
  { value: 'csv',   label: 'CSV (comma-separated rows)' },
  { value: 'regex', label: 'Regex (named groups per line)' },
  { value: 'json',  label: 'JSON (one JSON object per line)' },
]

const dailyCsvHint = computed(() =>
  form.value.daily_csv_enabled
    ? `Appends to shared/log_watcher/${nameSlug.value}/<YYYY-MM-DD>.csv on the server`
    : 'When enabled, every prediction is appended to a per-day aggregated CSV'
)

// Track whether the user has manually edited the folder fields — used to
// decide when it's safe to keep autofilling from the name field.
const hasUserEditedInputFolder = ref(false)
const hasUserEditedOutputFolder = ref(false)

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
      parse_mode: (w.parse_mode as any) || 'csv',
      parse_regex: w.parse_regex || '',
      mqtt_enabled: !!w.mqtt_enabled,
      mqtt_topic: w.mqtt_topic || 'alerts/{name}',
      daily_csv_enabled: !!w.daily_csv_enabled,
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

    // Don't ship an unused regex when the mode isn't regex — keeps the DB
    // row tidy and lets the backend's PATCH validator skip the compile check.
    if (payload.parse_mode !== 'regex') {
      payload.parse_regex = null
    }
    // Same for MQTT topic: only send when the sink is on. Empty strings would
    // otherwise trip the "required when mqtt_enabled" validator on a two-step
    // edit that toggled off in the same submit.
    if (!payload.mqtt_enabled) {
      payload.mqtt_topic = null
    }

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

// Programmatic-write guard — prevents autofill from tripping the "user edited"
// flags via the @update:model-value handlers.
let autofillingFolders = false

function setFolderAutofill(input: string, output: string) {
  autofillingFolders = true
  form.value.input_folder = input
  form.value.output_folder = output
  nextTick(() => { autofillingFolders = false })
}

function onInputFolderInput() {
  if (!autofillingFolders) hasUserEditedInputFolder.value = true
}
function onOutputFolderInput() {
  if (!autofillingFolders) hasUserEditedOutputFolder.value = true
}

// Autofill folders as the user types a name — but only for NEW watchers, and
// only for fields the user hasn't manually edited.
watch(() => form.value.name, () => {
  if (isEdit.value) return
  const input = defaultInputFolder.value
  const output = defaultOutputFolder.value
  autofillingFolders = true
  if (!hasUserEditedInputFolder.value) form.value.input_folder = input
  if (!hasUserEditedOutputFolder.value) form.value.output_folder = output
  nextTick(() => { autofillingFolders = false })
})

onMounted(async () => {
  await loadEndpoints()
  if (isEdit.value) {
    await loadWatcher()
    // Editing an existing watcher: never autofill, treat both fields as
    // user-owned so the name watcher above is a no-op even if it fires.
    hasUserEditedInputFolder.value = true
    hasUserEditedOutputFolder.value = true
  } else {
    // New watcher: autofill both folder fields once at mount using the
    // current default paths (name is likely empty → uses 'watcher' slug).
    setFolderAutofill(defaultInputFolder.value, defaultOutputFolder.value)
  }
})
</script>
