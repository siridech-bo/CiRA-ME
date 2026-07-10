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

        <!-- ── Log Watcher: parse mode + per-mode config ──────────────── -->
        <v-select
          v-model="form.parse_mode"
          :items="parseModeOptions"
          item-title="label"
          item-value="value"
          label="Parse mode"
          variant="outlined"
          density="comfortable"
          class="mb-2"
          :hint="parseModeHint"
          persistent-hint
        />
        <div class="mb-2 d-flex justify-end">
          <v-btn
            variant="text"
            size="x-small"
            color="grey"
            :prepend-icon="showAdvancedParseModes ? 'mdi-eye-off-outline' : 'mdi-cog-outline'"
            @click="toggleAdvancedParseModes"
          >
            {{ showAdvancedParseModes ? 'Hide advanced parse modes' : 'Show advanced parse modes' }}
          </v-btn>
        </div>
        <div class="mb-4" />

        <!-- key_value mode: just list column names -->
        <v-text-field
          v-if="form.parse_mode === 'key_value'"
          v-model="form.parse_columns"
          label="Column names (comma-separated)"
          placeholder="temperature, vibration, pressure"
          variant="outlined"
          density="comfortable"
          class="mb-4"
          hint="The watcher will look for temperature=X, vibration=X, pressure=X (or : X) on each line, ignoring surrounding text."
          persistent-hint
          :rules="[
            v => form.parse_mode !== 'key_value' || (!!v && !!String(v).trim()) || 'List at least one column'
          ]"
        />

        <!-- regex mode: template picker + textarea -->
        <template v-if="form.parse_mode === 'regex'">
          <v-alert
            type="info"
            variant="tonal"
            density="compact"
            class="mb-3"
            icon="mdi-lightbulb-outline"
          >
            If your log has <code>key=value</code> or <code>key:value</code> pairs (like
            <code>temperature=45.32 vibration=0.87</code>), use
            <strong>Key = Value pairs</strong> mode instead — no regex needed.
            Regex is only for unusual formats.
          </v-alert>
          <v-select
            v-model="regexTemplate"
            :items="regexTemplateOptions"
            item-title="label"
            item-value="value"
            label="Template"
            variant="outlined"
            density="comfortable"
            class="mb-3"
            hint="Pick a starting point, then edit as needed. Choosing a template overwrites the regex below."
            persistent-hint
            @update:model-value="onRegexTemplateChange"
          />

          <v-textarea
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
            @update:model-value="onRegexManualEdit"
          />
        </template>

        <!-- ── Live sample preview panel ──────────────────────────────── -->
        <v-expansion-panels v-model="previewPanel" class="mb-4">
          <v-expansion-panel>
            <v-expansion-panel-title>
              <v-icon size="small" class="mr-2">mdi-flask-outline</v-icon>
              Try a sample from your log file
            </v-expansion-panel-title>
            <v-expansion-panel-text>
              <v-alert type="info" variant="tonal" density="compact" class="mb-3" icon="mdi-information-outline">
                Paste <strong>actual log lines</strong> from a file — <em>not</em> your regex/config.
                Example: <code>2026-07-10T08:00:00 INFO | temperature=45.32 vibration=0.87 pressure=47.52</code>
              </v-alert>
              <v-textarea
                v-model="form.samplePreviewText"
                :rows="3"
                variant="outlined"
                density="comfortable"
                label="Paste 1-5 raw log lines here"
                placeholder="2026-07-10T08:00:00 INFO | temperature=45.32 vibration=0.87 pressure=47.52"
                hide-details
                class="mb-3"
                style="font-family: monospace;"
              />
              <div class="d-flex align-center gap-2 mb-2 flex-wrap">
                <v-btn
                  size="small"
                  variant="tonal"
                  color="primary"
                  :loading="previewLoading"
                  :disabled="!form.samplePreviewText || !form.samplePreviewText.trim()"
                  @click="runPreview"
                >
                  Test parse
                </v-btn>
                <v-btn
                  v-if="form.parse_mode === 'key_value' && !form.parse_columns.trim() && form.samplePreviewText.trim()"
                  size="small"
                  variant="tonal"
                  color="warning"
                  prepend-icon="mdi-lightning-bolt-outline"
                  :loading="detectLoading"
                  @click="detectColumnsFromSample"
                  title="Scan the sample for key=value patterns and fill the Column names field"
                >
                  Auto-detect columns
                </v-btn>
                <span v-if="previewResult" class="text-caption text-medium-emphasis">
                  {{ previewResult.row_count }} row{{ previewResult.row_count === 1 ? '' : 's' }} parsed
                </span>
              </div>

              <v-alert
                v-if="previewError"
                type="error"
                variant="tonal"
                density="compact"
                class="mb-2"
              >
                <div class="d-flex align-center gap-2 mb-1">
                  <v-chip
                    v-if="previewError.error_code"
                    size="x-small"
                    color="error"
                    variant="flat"
                  >
                    {{ previewError.error_code }}
                  </v-chip>
                  <strong>{{ previewError.error }}</strong>
                </div>
                <div v-if="previewError.hint" class="text-caption">
                  {{ previewError.hint }}
                </div>
              </v-alert>

              <template v-if="previewResult && !previewError">
                <div v-if="previewResult.warnings && previewResult.warnings.length" class="mb-2">
                  <v-chip
                    v-for="(w, i) in previewResult.warnings"
                    :key="i"
                    size="x-small"
                    color="warning"
                    variant="tonal"
                    class="mr-1"
                  >
                    {{ w }}
                  </v-chip>
                </div>
                <v-table v-if="previewResult.columns.length" density="compact" class="preview-table">
                  <thead>
                    <tr>
                      <th
                        v-for="(c, i) in previewResult.columns"
                        :key="i"
                        class="text-left"
                      >
                        {{ c }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, ri) in previewResult.rows.slice(0, 5)" :key="ri">
                      <td v-for="(cell, ci) in row" :key="ci">
                        {{ cell === null || cell === undefined ? '—' : cell }}
                      </td>
                    </tr>
                  </tbody>
                </v-table>
                <div
                  v-else
                  class="text-caption text-medium-emphasis pa-2"
                >
                  No columns detected. Check the parse configuration above.
                </div>
              </template>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>

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
  // New watchers default to key_value; existing watchers load their saved mode
  // in loadWatcher() so this default only matters for the "New" flow.
  parse_mode: 'key_value' as 'csv' | 'regex' | 'json' | 'key_value',
  parse_regex: '',
  parse_columns: '',
  mqtt_enabled: false,
  mqtt_topic: 'alerts/{name}',
  daily_csv_enabled: false,
  // Not persisted — just used by the "Try a sample line" panel
  samplePreviewText: '',
})

// Advanced parse modes are hidden by default. Regex is the only advanced
// mode today — kept out of the way of factory operators who won't write it.
const showAdvancedParseModes = ref(false)
const BASIC_PARSE_MODES = ['key_value', 'json', 'csv']
const ADVANCED_PARSE_MODES = ['regex']
const parseModeOptions = computed(() => {
  const base = [
    { value: 'key_value', label: 'Key = Value pairs (recommended)' },
    { value: 'json',      label: 'JSON — one object per line' },
    { value: 'csv',       label: 'CSV — headered rows' },
  ]
  if (showAdvancedParseModes.value) {
    base.push({ value: 'regex', label: 'Regex (named groups per line) — advanced' })
  }
  return base
})

function toggleAdvancedParseModes() {
  showAdvancedParseModes.value = !showAdvancedParseModes.value
  // If the operator hides advanced while an advanced mode is selected,
  // snap the form back to the safe key_value default so the dropdown
  // doesn't show a value that isn't in its option list.
  if (!showAdvancedParseModes.value && ADVANCED_PARSE_MODES.includes(form.value.parse_mode as any)) {
    form.value.parse_mode = 'key_value'
    form.value.parse_regex = ''
  }
}

const parseModeHints: Record<string, string> = {
  key_value: "Each line's key=value pairs are extracted by column name. Simplest for factory logs.",
  regex:     'Full regex power with named capture groups. Best when the format is unusual.',
  json:      'Each line must be a valid JSON object. Best when logs are already structured.',
  csv:       "Each file's first row is the header, subsequent rows are records.",
}
const parseModeHint = computed(() =>
  parseModeHints[form.value.parse_mode] || parseModeHints.key_value
)

// ── Regex templates ──────────────────────────────────────────────────────
// Picking one auto-fills the parse_regex textarea. If the user then edits
// the textarea, regexTemplate flips back to '' (Custom).
const regexTemplate = ref<string>('')
const regexTemplateOptions = [
  { value: '',          label: 'Custom (write your own)' },
  { value: 'apache',    label: 'Apache-style access log' },
  { value: 'syslog',    label: 'Syslog line' },
  { value: 'csv_line',  label: 'CSV-like (comma-separated per line)' },
  { value: 'space_sep', label: 'Space-separated numbers' },
  { value: 'factory',   label: 'Factory sensor line (temperature / vibration / pressure)' },
]
const regexTemplates: Record<string, string> = {
  apache:    '^(?P<ip>\\S+)\\s+\\S+\\s+(?P<user>\\S+)\\s+\\[(?P<time>[^\\]]+)\\]\\s+"(?P<method>\\S+)\\s+(?P<path>\\S+)\\s+HTTP/[\\d.]+"\\s+(?P<status>\\d+)\\s+(?P<bytes>\\d+)',
  syslog:    '^(?P<time>\\S+\\s+\\S+\\s+\\S+)\\s+(?P<host>\\S+)\\s+(?P<process>\\S+?)(\\[(?P<pid>\\d+)\\])?:\\s+(?P<message>.*)$',
  csv_line:  '^(?P<col1>[^,]*),(?P<col2>[^,]*),(?P<col3>[^,]*)',
  space_sep: '^(?P<col1>\\S+)\\s+(?P<col2>\\S+)\\s+(?P<col3>\\S+)',
  factory:   'temperature=(?P<temperature>-?\\d+\\.?\\d*)\\s+vibration=(?P<vibration>-?\\d+\\.?\\d*)\\s+pressure=(?P<pressure>-?\\d+\\.?\\d*)',
}
// Programmatic-write guard for regex textarea (mirrors the folder-autofill
// pattern above) — prevents the template picker from tripping onRegexManualEdit
// via its @update:model-value handler.
let regexAutofilling = false
function onRegexTemplateChange(v: string) {
  if (!v) return  // Custom → don't touch textarea
  regexAutofilling = true
  form.value.parse_regex = regexTemplates[v] || ''
  nextTick(() => { regexAutofilling = false })
}
function onRegexManualEdit() {
  if (regexAutofilling) return
  if (regexTemplate.value !== '') regexTemplate.value = ''
}

// ── Live sample preview ──────────────────────────────────────────────────
const previewPanel = ref<number | undefined>(undefined)
const previewLoading = ref(false)
const previewResult = ref<{
  columns: string[]
  rows: any[][]
  row_count: number
  skipped_lines: number
  warnings: string[]
} | null>(null)
const previewError = ref<{ error: string; error_code?: string; hint?: string } | null>(null)
const detectLoading = ref(false)

async function detectColumnsFromSample() {
  try {
    detectLoading.value = true
    const res = await api.post('/api/folder-watchers/detect-columns', {
      sample_content: form.value.samplePreviewText || '',
    })
    const cols: string[] = res.data?.columns || []
    if (cols.length === 0) {
      notify.showError('No key=value or key:value patterns found in the sample.')
      return
    }
    form.value.parse_columns = cols.join(', ')
    notify.showSuccess(
      `Detected ${cols.length} column${cols.length === 1 ? '' : 's'}: ${cols.join(', ')}. ` +
      `Trim the list if any look like non-sensor keys (pid, port, etc.).`
    )
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to detect columns from sample')
  } finally {
    detectLoading.value = false
  }
}

async function runPreview() {
  previewError.value = null
  previewResult.value = null
  try {
    previewLoading.value = true
    const payload: Record<string, any> = {
      parse_mode: form.value.parse_mode,
      sample_content: form.value.samplePreviewText || '',
    }
    if (form.value.parse_mode === 'regex') payload.parse_regex = form.value.parse_regex
    if (form.value.parse_mode === 'key_value') payload.parse_columns = form.value.parse_columns
    if (form.value.parse_mode === 'csv') payload.header_mode = form.value.header_mode
    const res = await api.post('/api/folder-watchers/preview-parse', payload)
    previewResult.value = res.data
  } catch (e: any) {
    const data = e.response?.data || {}
    previewError.value = {
      error: data.error || 'Preview failed',
      error_code: data.error_code,
      hint: data.hint,
    }
  } finally {
    previewLoading.value = false
  }
}

// Any parse-config change invalidates the previous preview result.
watch(
  () => [
    form.value.parse_mode,
    form.value.parse_regex,
    form.value.parse_columns,
    form.value.header_mode,
  ],
  () => {
    previewResult.value = null
    previewError.value = null
  }
)

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
      // Existing watchers keep their saved mode (csv/regex/json/key_value).
      // Only NEW watchers get the key_value default in the form ref above.
      parse_mode: (w.parse_mode as any) || 'csv',
      parse_regex: w.parse_regex || '',
      parse_columns: w.parse_columns || '',
      mqtt_enabled: !!w.mqtt_enabled,
      mqtt_topic: w.mqtt_topic || 'alerts/{name}',
      daily_csv_enabled: !!w.daily_csv_enabled,
      samplePreviewText: '',
    }
    // If the watcher is on an advanced mode (regex), auto-reveal the toggle so
    // the mode is visible in the dropdown after load.
    if (ADVANCED_PARSE_MODES.includes(w.parse_mode)) {
      showAdvancedParseModes.value = true
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
    // Same for parse_columns: only relevant in key_value mode.
    if (payload.parse_mode !== 'key_value') {
      payload.parse_columns = null
    }
    // samplePreviewText is UI-only — never sent to the backend.
    delete payload.samplePreviewText
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

<style scoped>
.preview-table {
  border: 1px solid rgb(var(--v-theme-outline-variant, 224, 224, 224));
  border-radius: 4px;
  max-height: 240px;
  overflow-y: auto;
}
</style>
