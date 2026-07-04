<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">Multi-Dataset Wizard</h1>
        <p class="text-body-2 text-medium-emphasis">
          Run up to {{ MAX_MODELS }} ME-LAB endpoints against up to {{ MAX_DATASETS }}
          CSV datasets and compare results in one matrix.
        </p>
      </div>
      <v-spacer />
      <v-btn variant="text" @click="resetAll" v-if="step === 3">
        <v-icon start>mdi-refresh</v-icon>
        New Comparison
      </v-btn>
    </div>

    <v-stepper v-model="step" flat editable="false" class="mb-4">
      <v-stepper-header>
        <v-stepper-item :value="1" title="Pick Models" :complete="step > 1" />
        <v-divider />
        <v-stepper-item :value="2" title="Pick Datasets" :complete="step > 2" />
        <v-divider />
        <v-stepper-item :value="3" title="Compare" :complete="step > 3" />
      </v-stepper-header>
    </v-stepper>

    <!-- STEP 1 — Pick Models -->
    <v-card v-if="step === 1" class="pa-4">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          Select ME-LAB Endpoints
          <span class="text-caption text-medium-emphasis ml-2">
            {{ selectedEndpointIds.length }} / {{ MAX_MODELS }} selected
          </span>
        </h3>
        <v-spacer />
        <v-chip v-if="pickedMode" size="small" color="primary" variant="tonal">
          mode: {{ pickedMode }}
        </v-chip>
      </div>

      <v-alert
        v-if="pickedMode"
        density="compact"
        variant="tonal"
        type="info"
        class="mb-3"
      >
        You've picked a <strong>{{ pickedMode }}</strong> endpoint. Endpoints
        from other modes are disabled — one wizard run compares one mode.
      </v-alert>

      <v-progress-linear v-if="loadingEndpoints" indeterminate class="mb-4" />

      <div v-if="!loadingEndpoints && endpointsByMode">
        <div
          v-for="mode in modeOrder"
          :key="mode"
          class="mb-4"
          v-show="(endpointsByMode[mode] || []).length > 0"
        >
          <div class="text-subtitle-2 font-weight-bold mb-2">
            <v-icon start size="small">{{ modeIcon(mode) }}</v-icon>
            {{ modeLabel(mode) }}
          </div>
          <v-list density="comfortable" class="pa-0">
            <v-list-item
              v-for="ep in endpointsByMode[mode]"
              :key="ep.id"
              :disabled="isEndpointDisabled(ep)"
              @click="toggleEndpoint(ep)"
              rounded="lg"
            >
              <template #prepend>
                <v-checkbox-btn
                  :model-value="selectedEndpointIds.includes(ep.id)"
                  :disabled="isEndpointDisabled(ep)"
                />
              </template>
              <v-list-item-title class="font-weight-medium">
                {{ ep.name }}
              </v-list-item-title>
              <v-list-item-subtitle>
                <v-chip
                  size="x-small"
                  variant="tonal"
                  color="grey"
                  class="mr-1"
                >
                  {{ ep.algorithm || 'unknown' }}
                </v-chip>
                <v-chip
                  v-if="ep.is_dl || !ep.no_windowing"
                  size="x-small"
                  variant="tonal"
                  color="orange"
                  class="mr-1"
                  title="Not raw-mode — will be rejected at Step 2"
                >
                  {{ ep.is_dl ? 'DL' : 'windowed' }}
                </v-chip>
              </v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </div>
        <v-alert
          v-if="!hasAnyEndpoints"
          type="warning"
          density="compact"
          variant="tonal"
        >
          No active ME-LAB endpoints found. Create one in ME-LAB first.
        </v-alert>
      </div>

      <div class="d-flex mt-4">
        <v-spacer />
        <v-btn
          color="primary"
          :disabled="selectedEndpointIds.length === 0"
          @click="step = 2"
        >
          Next: Pick Datasets
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </v-card>

    <!-- STEP 2 — Pick Datasets -->
    <v-card v-if="step === 2" class="pa-4">
      <h3 class="text-subtitle-1 font-weight-bold mb-3">
        Pick up to {{ MAX_DATASETS }} CSV Datasets
      </h3>

      <v-tabs v-model="datasetTab" density="compact" class="mb-3">
        <v-tab value="upload">Upload New</v-tab>
        <v-tab value="browse">From Datasets</v-tab>
      </v-tabs>

      <div v-if="datasetTab === 'upload'">
        <v-file-input
          v-model="uploadedFiles"
          label="Select up to 5 CSVs"
          accept=".csv,text/csv"
          multiple
          chips
          show-size
          prepend-icon="mdi-file-upload"
          density="comfortable"
        />
        <div class="text-caption text-medium-emphasis">
          Each file must have &lt; {{ ROW_CAP.toLocaleString() }} data rows. All
          selected CSVs must have <strong>identical sensor column names</strong>
          (set equality, not order). Labels and timestamps are ignored.
        </div>
      </div>

      <div v-else>
        <div class="d-flex align-center mb-2">
          <v-btn
            size="small"
            variant="text"
            :disabled="!canGoUp"
            @click="browseUp"
          >
            <v-icon start>mdi-arrow-up</v-icon>
            Up
          </v-btn>
          <v-chip size="small" class="ml-2">{{ browsePath || '/' }}</v-chip>
          <v-spacer />
          <span class="text-caption text-medium-emphasis">
            {{ selectedDatasetPaths.length }} / {{ MAX_DATASETS }} selected
          </span>
        </div>
        <v-list density="compact" class="pa-0" style="max-height: 320px; overflow-y: auto;">
          <v-list-item
            v-for="item in browseItems"
            :key="item.path"
            @click="onBrowseClick(item)"
            :active="item.type === 'file' && selectedDatasetPaths.includes(item.path)"
            rounded="lg"
          >
            <template #prepend>
              <v-icon>
                {{ item.type === 'directory' ? 'mdi-folder-outline' : 'mdi-file-delimited-outline' }}
              </v-icon>
            </template>
            <v-list-item-title>{{ item.name }}</v-list-item-title>
            <template #append v-if="item.type === 'file'">
              <v-icon v-if="selectedDatasetPaths.includes(item.path)" color="primary">
                mdi-check-circle
              </v-icon>
            </template>
          </v-list-item>
          <div v-if="browseItems.length === 0 && !browseLoading" class="text-body-2 text-medium-emphasis pa-4 text-center">
            No CSVs in this folder.
          </div>
        </v-list>
      </div>

      <v-alert
        v-if="schemaError"
        type="error"
        density="compact"
        class="mt-4"
        variant="tonal"
      >
        <div class="font-weight-bold">Schema mismatch — cannot proceed.</div>
        <div class="mt-2">Expected columns:</div>
        <div class="mt-1">
          <v-chip
            v-for="c in schemaError.expected"
            :key="c"
            size="x-small"
            class="mr-1 mb-1"
            variant="tonal"
          >
            {{ c }}
          </v-chip>
        </div>
        <div class="mt-3" v-for="m in schemaError.mismatches" :key="m.dataset">
          <div class="font-weight-medium">{{ m.dataset }}</div>
          <div class="text-caption">
            <span v-if="m.extra.length > 0" class="text-error">
              extra: {{ m.extra.join(', ') }}
            </span>
            <span v-if="m.missing.length > 0" class="text-warning ml-2">
              missing: {{ m.missing.join(', ') }}
            </span>
          </div>
        </div>
        <div class="mt-2 text-caption">
          Fix your CSV columns and try again — auto-align is intentionally disabled.
        </div>
      </v-alert>
      <v-alert
        v-else-if="rowCapError"
        type="error"
        density="compact"
        class="mt-4"
        variant="tonal"
      >
        <div class="font-weight-bold">Dataset too large.</div>
        <div class="mt-1 text-caption">{{ rowCapError }}</div>
      </v-alert>
      <v-alert
        v-else-if="genericError"
        type="error"
        density="compact"
        class="mt-4"
        variant="tonal"
      >
        {{ genericError }}
      </v-alert>

      <div class="d-flex mt-4">
        <v-btn variant="text" @click="step = 1">
          <v-icon start>mdi-arrow-left</v-icon>
          Back
        </v-btn>
        <v-spacer />
        <v-btn
          color="primary"
          :loading="validating"
          :disabled="!canValidate"
          @click="validateDatasets"
        >
          Validate &amp; Continue
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </v-card>

    <!-- STEP 3 — Compare -->
    <v-card v-if="step === 3" class="pa-4">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">Comparison Matrix</h3>
        <v-spacer />
        <v-btn
          v-if="matrixResult"
          size="small"
          variant="tonal"
          class="mr-2"
          @click="downloadExport('aggregated')"
        >
          <v-icon start>mdi-download</v-icon>
          Aggregated CSV
        </v-btn>
        <v-btn
          v-if="matrixResult"
          size="small"
          variant="tonal"
          @click="downloadExport('per_row')"
        >
          <v-icon start>mdi-download</v-icon>
          Per-Row CSV
        </v-btn>
      </div>

      <v-btn
        v-if="!matrixResult && !running"
        color="primary"
        block
        @click="runComparison"
      >
        <v-icon start>mdi-play</v-icon>
        Run Comparison
      </v-btn>

      <div v-if="running" class="my-4">
        <v-progress-linear indeterminate />
        <div class="text-caption text-medium-emphasis text-center mt-2">
          Running comparison...
        </div>
      </div>

      <v-alert
        v-if="runError"
        type="error"
        density="compact"
        class="mt-4"
        variant="tonal"
      >
        {{ runError }}
      </v-alert>

      <div v-if="matrixResult" class="mt-4" style="overflow-x: auto;">
        <table class="wizard-matrix">
          <thead>
            <tr>
              <th class="sticky-col">Dataset \ Model</th>
              <th v-for="m in matrixResult.models" :key="m.id">
                <div class="font-weight-medium">{{ m.name }}</div>
                <div class="text-caption text-medium-emphasis">{{ m.algorithm }}</div>
                <div class="text-caption text-medium-emphasis">
                  {{ formatBytes(m.size_bytes) }}
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(ds, di) in matrixResult.datasets" :key="ds.id">
              <td class="sticky-col">
                <div class="font-weight-medium">{{ ds.name }}</div>
                <div class="text-caption text-medium-emphasis">
                  {{ ds.row_count.toLocaleString() }} rows
                </div>
              </td>
              <td
                v-for="(cell, mi) in matrixResult.matrix[di]"
                :key="mi"
                :style="cellStyle(cell)"
                class="wizard-cell"
              >
                <v-menu open-on-hover open-on-click :close-on-content-click="false">
                  <template #activator="{ props: menuProps }">
                    <div v-bind="menuProps" class="cell-inner">
                      <div v-if="cell.error" class="text-error">
                        <v-icon size="small" color="error">mdi-alert-circle</v-icon>
                        error
                      </div>
                      <template v-else>
                        <div class="font-weight-medium">
                          {{ formatLabel(cell.predicted_label) }}
                        </div>
                        <div v-if="cell.confidence != null" class="text-caption">
                          {{ (cell.confidence * 100).toFixed(1) }}%
                        </div>
                        <div v-else-if="cell.score != null" class="text-caption">
                          score {{ Number(cell.score).toFixed(3) }}
                        </div>
                      </template>
                    </div>
                  </template>
                  <v-card min-width="260" class="pa-3">
                    <div v-if="cell.error">
                      <div class="text-subtitle-2 mb-1">Error</div>
                      <div class="text-caption text-error">{{ cell.error }}</div>
                    </div>
                    <div v-else>
                      <div class="text-subtitle-2 mb-1">
                        {{ formatLabel(cell.predicted_label) }}
                      </div>
                      <div v-if="cell.probabilities" class="mb-2">
                        <div
                          v-for="(v, k) in cell.probabilities"
                          :key="k"
                          class="mb-1"
                        >
                          <div class="d-flex text-caption">
                            <span>{{ k }}</span>
                            <v-spacer />
                            <span>{{ (Number(v) * 100).toFixed(1) }}%</span>
                          </div>
                          <v-progress-linear
                            :model-value="Number(v) * 100"
                            height="4"
                          />
                        </div>
                      </div>
                      <div class="text-caption text-medium-emphasis mt-2">
                        Latency: {{ cell.latency_ms != null ? cell.latency_ms.toFixed(2) : '–' }} ms/row
                      </div>
                      <div class="text-caption text-medium-emphasis">
                        Model size: {{ formatBytes(matrixResult.models[mi].size_bytes) }}
                      </div>
                    </div>
                  </v-card>
                </v-menu>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="d-flex mt-4">
        <v-btn variant="text" @click="step = 2" :disabled="running">
          <v-icon start>mdi-arrow-left</v-icon>
          Back
        </v-btn>
        <v-spacer />
        <v-btn variant="text" color="primary" @click="resetAll" v-if="matrixResult">
          <v-icon start>mdi-refresh</v-icon>
          New Comparison
        </v-btn>
      </div>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'

// ── Locked customer decisions ─────────────────────────────────────────
const MAX_MODELS = 5
const MAX_DATASETS = 5
const ROW_CAP = 100_000

interface Endpoint {
  id: string
  name: string
  mode: string
  algorithm: string
  status: string
  no_windowing?: boolean
  is_dl?: boolean
}

interface Cell {
  predicted_label?: string | number
  confidence?: number | null
  probabilities?: Record<string, number> | null
  score?: number | null
  latency_ms?: number
  error?: string
}

interface MatrixResult {
  run_id: string
  mode: string
  models: { id: string; name: string; algorithm: string; size_bytes: number }[]
  datasets: { id: string; name: string; row_count: number }[]
  matrix: Cell[][]
}

interface BrowseItem {
  name: string
  path: string
  type: 'file' | 'directory'
}

const notify = useNotificationStore()
const step = ref(1)

// Step 1 state
const loadingEndpoints = ref(false)
const endpoints = ref<Endpoint[]>([])
const selectedEndpointIds = ref<string[]>([])

const modeOrder = ['classification', 'regression', 'anomaly']

const endpointsByMode = computed<Record<string, Endpoint[]>>(() => {
  const g: Record<string, Endpoint[]> = { classification: [], regression: [], anomaly: [] }
  for (const ep of endpoints.value) {
    if (ep.status !== 'active') continue
    const m = ep.mode || 'classification'
    if (!g[m]) g[m] = []
    g[m].push(ep)
  }
  return g
})
const hasAnyEndpoints = computed(() =>
  Object.values(endpointsByMode.value).some(l => l.length > 0)
)
const pickedMode = computed<string | null>(() => {
  if (selectedEndpointIds.value.length === 0) return null
  const firstId = selectedEndpointIds.value[0]
  const ep = endpoints.value.find(e => e.id === firstId)
  return ep?.mode || null
})

function isEndpointDisabled(ep: Endpoint): boolean {
  if (pickedMode.value && ep.mode !== pickedMode.value) return true
  if (
    !selectedEndpointIds.value.includes(ep.id) &&
    selectedEndpointIds.value.length >= MAX_MODELS
  ) return true
  return false
}
function toggleEndpoint(ep: Endpoint) {
  if (isEndpointDisabled(ep)) return
  const i = selectedEndpointIds.value.indexOf(ep.id)
  if (i >= 0) selectedEndpointIds.value.splice(i, 1)
  else selectedEndpointIds.value.push(ep.id)
}
function modeIcon(m: string) {
  return m === 'classification'
    ? 'mdi-tag-multiple'
    : m === 'regression'
      ? 'mdi-chart-line'
      : 'mdi-alert'
}
function modeLabel(m: string) {
  return m.charAt(0).toUpperCase() + m.slice(1)
}

// Step 2 state
const datasetTab = ref<'upload' | 'browse'>('upload')
const uploadedFiles = ref<File[]>([])
const browsePath = ref<string>('')
const browseBasePath = ref<string>('')
const browseItems = ref<BrowseItem[]>([])
const browseLoading = ref(false)
const selectedDatasetPaths = ref<string[]>([])

const canGoUp = computed(() => browsePath.value && browsePath.value !== browseBasePath.value)

async function loadBrowse() {
  browseLoading.value = true
  try {
    const resp = await api.post('/api/data/browse', {
      path: browsePath.value || null,
    })
    const items: BrowseItem[] = (resp.data.items || []).filter((it: any) => {
      if (it.type === 'directory') return true
      return String(it.name || '').toLowerCase().endsWith('.csv')
    })
    browseItems.value = items
    browsePath.value = resp.data.current_path
    if (!browseBasePath.value) browseBasePath.value = resp.data.current_path || ''
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Browse failed')
  } finally {
    browseLoading.value = false
  }
}
function browseUp() {
  if (!canGoUp.value) return
  const p = browsePath.value.replace(/[\\/][^\\/]*$/, '')
  browsePath.value = p
  loadBrowse()
}
function onBrowseClick(item: BrowseItem) {
  if (item.type === 'directory') {
    browsePath.value = item.path
    loadBrowse()
    return
  }
  const i = selectedDatasetPaths.value.indexOf(item.path)
  if (i >= 0) selectedDatasetPaths.value.splice(i, 1)
  else {
    if (selectedDatasetPaths.value.length >= MAX_DATASETS) {
      notify.showError(`Max ${MAX_DATASETS} datasets`)
      return
    }
    selectedDatasetPaths.value.push(item.path)
  }
}

const canValidate = computed(() => {
  if (datasetTab.value === 'upload') return uploadedFiles.value.length > 0
  return selectedDatasetPaths.value.length > 0
})

const validating = ref(false)
const runId = ref<string | null>(null)
const schemaError = ref<any | null>(null)
const rowCapError = ref<string | null>(null)
const genericError = ref<string | null>(null)

async function validateDatasets() {
  schemaError.value = null
  rowCapError.value = null
  genericError.value = null
  validating.value = true

  try {
    let resp: any
    if (datasetTab.value === 'upload') {
      const fd = new FormData()
      for (const f of uploadedFiles.value) fd.append('files[]', f)
      resp = await api.post('/api/wizard/validate-datasets', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
    } else {
      resp = await api.post('/api/wizard/validate-datasets', {
        dataset_paths: selectedDatasetPaths.value,
      })
    }
    runId.value = resp.data.run_id
    step.value = 3
  } catch (e: any) {
    const data = e.response?.data
    const status = e.response?.status
    if (status === 409 && data?.error === 'SCHEMA_MISMATCH') {
      schemaError.value = data
    } else if (status === 413) {
      rowCapError.value = data?.message || 'Row cap exceeded'
    } else {
      genericError.value = data?.error || 'Validation failed'
    }
  } finally {
    validating.value = false
  }
}

// Step 3 — Run
const running = ref(false)
const runError = ref<string | null>(null)
const matrixResult = ref<MatrixResult | null>(null)

async function runComparison() {
  if (!runId.value) return
  running.value = true
  runError.value = null
  matrixResult.value = null
  try {
    const resp = await api.post('/api/wizard/run', {
      run_id: runId.value,
      endpoint_ids: selectedEndpointIds.value,
    })
    matrixResult.value = resp.data
  } catch (e: any) {
    runError.value = e.response?.data?.error || 'Run failed'
  } finally {
    running.value = false
  }
}

async function downloadExport(level: 'aggregated' | 'per_row') {
  if (!runId.value) return
  try {
    const resp = await api.post(
      '/api/wizard/export',
      { run_id: runId.value, level },
      { responseType: 'blob' }
    )
    const blob = new Blob([resp.data], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `wizard_${runId.value}_${level}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Export failed')
  }
}

async function resetAll() {
  // Delete the current run's temp dir if any.
  if (runId.value) {
    try {
      await api.delete(`/api/wizard/runs/${runId.value}`)
    } catch { /* best-effort */ }
  }
  runId.value = null
  matrixResult.value = null
  runError.value = null
  schemaError.value = null
  rowCapError.value = null
  genericError.value = null
  selectedDatasetPaths.value = []
  uploadedFiles.value = []
  selectedEndpointIds.value = []
  step.value = 1
}

// Format helpers
function formatBytes(n: number): string {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}
function formatLabel(v: any): string {
  if (v == null) return '—'
  if (typeof v === 'number') return v.toFixed(3)
  return String(v)
}
function cellStyle(cell: Cell): Record<string, string> {
  if (cell.error) return { background: 'rgba(244, 67, 54, 0.12)' }
  const conf = cell.confidence
  if (conf == null) return {}
  // Higher confidence → deeper green; lower → paler.
  const alpha = Math.min(0.55, Math.max(0.08, conf))
  return { background: `rgba(76, 175, 80, ${alpha})` }
}

async function loadEndpoints() {
  loadingEndpoints.value = true
  try {
    const resp = await api.get('/api/melab/endpoints')
    endpoints.value = (resp.data || []).filter((e: Endpoint) => e.status === 'active')
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load endpoints')
  } finally {
    loadingEndpoints.value = false
  }
}

// Cleanup any leftover run on unmount.
onBeforeUnmount(async () => {
  if (runId.value) {
    try { await api.delete(`/api/wizard/runs/${runId.value}`) } catch { /* no-op */ }
  }
})

watch(datasetTab, (t) => {
  if (t === 'browse' && browseItems.value.length === 0) loadBrowse()
})

onMounted(() => {
  loadEndpoints()
})
</script>

<style scoped>
.wizard-matrix {
  border-collapse: collapse;
  width: 100%;
}
.wizard-matrix th,
.wizard-matrix td {
  border: 1px solid rgba(0, 0, 0, 0.08);
  padding: 8px;
  vertical-align: top;
  text-align: left;
  min-width: 140px;
}
.wizard-matrix th {
  background: rgba(0, 0, 0, 0.04);
  font-weight: 600;
}
.sticky-col {
  position: sticky;
  left: 0;
  background: var(--v-theme-surface, #fff);
  z-index: 1;
  min-width: 200px;
}
.wizard-cell {
  cursor: pointer;
}
.cell-inner {
  min-height: 40px;
}
</style>
