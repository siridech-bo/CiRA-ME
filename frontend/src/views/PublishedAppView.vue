<template>
  <div class="published-app">
    <!-- Loading -->
    <div v-if="loading" class="app-loading">
      <v-progress-circular indeterminate color="purple" />
      <div class="text-caption text-medium-emphasis mt-3">Loading app...</div>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="app-loading">
      <v-icon size="48" color="error">mdi-alert-circle-outline</v-icon>
      <div class="text-h6 mt-3">App not found</div>
      <div class="text-caption text-medium-emphasis mt-1">{{ error }}</div>
      <v-btn class="mt-4" variant="outlined" to="/">Go to Dashboard</v-btn>
    </div>

    <!-- App content -->
    <div v-else class="app-content">
      <!-- Header -->
      <div class="app-header">
        <div class="app-header-left">
          <img src="/logo.svg" alt="CiRA" class="app-logo" />
          <div>
            <div class="app-title">{{ appData.name }}</div>
            <div class="app-subtitle">
              <span class="app-mode-badge" :style="{ color: modeColor, background: modeColor + '18' }">
                {{ appMode?.toUpperCase() }}
              </span>
              <span class="app-algo">{{ appAlgorithm }}</span>
            </div>
          </div>
        </div>
        <div class="app-header-right">
          <v-btn
            v-if="!isStandalone"
            size="small"
            variant="tonal"
            color="purple"
            :href="`/standalone/${slug}`"
            target="_blank"
            class="mr-2"
          >
            <v-icon start size="small">mdi-open-in-new</v-icon>
            Open Standalone
          </v-btn>
          <span class="app-powered">Powered by CiRA ME</span>
        </div>
      </div>

      <!-- Pipeline info -->
      <div v-if="parsedNodes.length > 0" class="app-section" style="padding: 10px 16px;">
        <div class="d-flex flex-wrap align-center" style="gap: 6px;">
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Window: {{ pipelineInfo.window_size }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Stride: {{ pipelineInfo.stride }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Features: {{ pipelineInfo.n_features }}</v-chip>
          <v-chip size="x-small" :color="modeColor" variant="tonal">{{ appMode?.toUpperCase() || 'MODEL' }}</v-chip>
          <v-chip size="x-small" color="purple" variant="tonal">{{ appAlgorithm || 'model' }}</v-chip>
          <v-chip size="x-small" variant="outlined" style="font-size:9px">{{ parsedNodes.length }} nodes</v-chip>
        </div>
      </div>

      <!-- Input section -->
      <div class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="blue">mdi-upload</v-icon>
          Upload Data
        </div>

        <!-- Expected CSV format -->
        <div v-if="expectedColumns.length > 0" class="expected-format">
          <div class="text-caption text-medium-emphasis mb-1">
            <v-icon size="12" class="mr-1">mdi-information-outline</v-icon>
            Expected CSV columns:
          </div>
          <div class="d-flex flex-wrap" style="gap: 3px;">
            <span v-for="col in expectedColumns" :key="col" class="expected-col">{{ col }}</span>
          </div>
        </div>

        <div class="app-upload-area">
          <input type="file" ref="fileInput" accept=".csv" @change="onFileSelect" style="display:none" />
          <div v-if="!selectedFile" class="app-dropzone" @click="$refs.fileInput.click()">
            <v-icon size="32" color="grey">mdi-file-upload-outline</v-icon>
            <div class="text-caption text-medium-emphasis mt-2">Click to upload CSV file</div>
          </div>
          <div v-else class="app-file-selected">
            <v-icon size="16" color="success">mdi-file-check</v-icon>
            <span>{{ selectedFile.name }} ({{ (selectedFile.size / 1024).toFixed(1) }} KB)</span>
            <v-btn icon size="x-small" variant="text" @click="clearFile">
              <v-icon size="14">mdi-close</v-icon>
            </v-btn>
          </div>
          <v-btn
            :disabled="!selectedFile || running"
            :loading="running"
            color="purple"
            variant="flat"
            class="mt-3"
            @click="runPipeline"
          >
            <v-icon start>mdi-play</v-icon>
            Run Inference
          </v-btn>
        </div>
      </div>

      <!-- Results section -->
      <div v-if="result" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="success">mdi-chart-line</v-icon>
          Results
        </div>

        <!-- Regression results -->
        <div v-if="appMode === 'regression'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Predictions</div>
              <div class="result-stat-value">{{ result.predictions?.length || 0 }}</div>
            </div>
            <div v-if="result.mean !== undefined" class="result-stat">
              <div class="result-stat-label">Mean</div>
              <div class="result-stat-value">{{ result.mean?.toFixed(4) }}</div>
            </div>
            <div v-if="result.std !== undefined" class="result-stat">
              <div class="result-stat-label">Std</div>
              <div class="result-stat-value">{{ result.std?.toFixed(4) }}</div>
            </div>
            <div v-if="result.r2 !== undefined" class="result-stat">
              <div class="result-stat-label">R²</div>
              <div class="result-stat-value" :style="{ color: result.r2 > 0.8 ? '#34d399' : result.r2 > 0.5 ? '#fbbf24' : '#f87171' }">{{ result.r2?.toFixed(4) }}</div>
            </div>
            <div v-else-if="result.min !== undefined" class="result-stat">
              <div class="result-stat-label">Range</div>
              <div class="result-stat-value">{{ result.min?.toFixed(1) }} – {{ result.max?.toFixed(1) }}</div>
            </div>
            <div v-if="result.rmse !== undefined" class="result-stat">
              <div class="result-stat-label">RMSE</div>
              <div class="result-stat-value">{{ result.rmse?.toFixed(4) }}</div>
            </div>
          </div>

          <!-- Line Chart -->
          <div v-if="chartData.length > 0" class="chart-container">
            <div class="chart-header">
              <span class="chart-title-text">{{ actualData.length > 0 ? 'Actual vs Predicted' : 'Predictions over Time' }}</span>
            </div>
            <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" class="prediction-chart">
              <!-- Grid lines -->
              <line v-for="i in 4" :key="'g'+i"
                :x1="chartPadding" :y1="chartPadding + (i-1) * (chartInnerH / 3)"
                :x2="chartWidth - chartPadding" :y2="chartPadding + (i-1) * (chartInnerH / 3)"
                stroke="#21262d" stroke-width="0.5" />
              <!-- Y axis labels -->
              <text v-for="i in 4" :key="'y'+i"
                :x="chartPadding - 4" :y="chartPadding + (i-1) * (chartInnerH / 3) + 3"
                text-anchor="end" fill="#8b949e" font-size="8" font-family="monospace">
                {{ chartYLabel(i-1) }}
              </text>
              <!-- Area fill -->
              <path :d="chartAreaPath" fill="url(#pred-gradient)" />
              <!-- Actual line (cyan solid) -->
              <path v-if="actualLinePath" :d="actualLinePath" fill="none" stroke="#22d3ee" stroke-width="1.5" />
              <!-- Prediction line (purple, dashed if actual shown) -->
              <path :d="chartLinePath" fill="none" stroke="#a78bfa" stroke-width="1.5" :stroke-dasharray="actualLinePath ? '4,2' : 'none'" />
              <defs>
                <linearGradient id="pred-gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.15" />
                  <stop offset="100%" stop-color="#a78bfa" stop-opacity="0" />
                </linearGradient>
              </defs>
            </svg>
            <div class="chart-legend-items" style="margin-top:8px">
              <span v-if="actualData.length > 0" class="chart-legend-dot" style="background: #22d3ee"></span>
              <span v-if="actualData.length > 0" class="chart-legend-label">Actual</span>
              <span class="chart-legend-dot" style="background: #a78bfa"></span>
              <span class="chart-legend-label">Predicted</span>
            </div>
          </div>

          <!-- Data Table (collapsible) -->
          <div class="table-toggle" @click="showTable = !showTable">
            <v-icon size="14">{{ showTable ? 'mdi-chevron-down' : 'mdi-chevron-right' }}</v-icon>
            <span>{{ showTable ? 'Hide' : 'Show' }} data table ({{ result.predictions?.length || 0 }} rows)</span>
          </div>
          <div v-if="showTable" class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Prediction</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions || []).slice(0, 100)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: modeColor }">{{ typeof val === 'number' ? val.toFixed(4) : val }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Classification results -->
        <div v-else-if="appMode === 'classification'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Windows</div>
              <div class="result-stat-value">{{ result.count || 0 }}</div>
            </div>
          </div>
          <div class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Label</th><th>Confidence</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions_full || result.predictions || []).slice(0, 100)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: modeColor }">{{ val.label || val }}</td>
                  <td>{{ val.confidence ? (val.confidence * 100).toFixed(1) + '%' : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Anomaly results -->
        <div v-else-if="appMode === 'anomaly'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Total Windows</div>
              <div class="result-stat-value">{{ result.count || 0 }}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Anomalies</div>
              <div class="result-stat-value" style="color: #f87171">{{ result.anomaly_count || 0 }}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Normal</div>
              <div class="result-stat-value" style="color: #34d399">{{ result.normal_count || 0 }}</div>
            </div>
          </div>
          <div class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Label</th><th>Score</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions_full || result.predictions || []).slice(0, 100)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: (val.label || val) === 'anomaly' ? '#f87171' : '#34d399' }">
                    {{ val.label || val }}
                  </td>
                  <td>{{ val.score ? val.score.toFixed(4) : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
              <div class="result-stat-label">Anomalies</div>
              <div class="result-stat-value" style="color: #f87171">
                {{ (result.predictions || []).filter(p => (p.label || p) === 'Anomaly' || p === -1).length }}
              </div>
            </div>
          </div>
          <div class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Label</th><th>Score</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions || []).slice(0, 50)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: (val.label || val) === 'Anomaly' || val === -1 ? '#f87171' : '#34d399' }">
                    {{ val.label || (val === -1 ? 'Anomaly' : 'Normal') }}
                  </td>
                  <td>{{ val.score ? val.score.toFixed(3) : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Raw JSON fallback -->
        <div v-else class="app-results">
          <pre class="result-json">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>

      <!-- Run error -->
      <v-alert v-if="runError" type="error" variant="tonal" class="mt-4" closable @click:close="runError = null">
        {{ runError }}
      </v-alert>

      <!-- Upload new file button after results -->
      <div v-if="result" class="text-center mt-4">
        <v-btn variant="outlined" color="purple" @click="clearFile">
          <v-icon start size="small">mdi-upload</v-icon>
          Upload New File
        </v-btn>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '@/services/api'

const route = useRoute()
const slug = computed(() => route.params.slug)
const isStandalone = computed(() => route.meta?.standalone === true)

const loading = ref(true)
const error = ref(null)
const appData = ref({})
const selectedFile = ref(null)
const running = ref(false)
const result = ref(null)
const showTable = ref(false)

// Chart dimensions
const chartWidth = 700
const chartHeight = 200
const chartPadding = 40
const chartInnerW = chartWidth - chartPadding * 2
const chartInnerH = chartHeight - chartPadding * 2

// Chart data (downsampled predictions for rendering)
const chartData = computed(() => {
  const preds = result.value?.predictions || []
  if (preds.length === 0) return []
  const nums = preds.filter(v => typeof v === 'number')
  if (nums.length === 0) return []
  // Downsample to max 200 points
  if (nums.length <= 200) return nums
  const step = nums.length / 200
  return Array.from({ length: 200 }, (_, i) => nums[Math.floor(i * step)])
})

const chartMinY = computed(() => {
  if (allChartValues.value.length === 0) return 0
  return Math.min(...allChartValues.value) - (Math.max(...allChartValues.value) - Math.min(...allChartValues.value)) * 0.05
})
const chartMaxY = computed(() => {
  if (allChartValues.value.length === 0) return 1
  return Math.max(...allChartValues.value) + (Math.max(...allChartValues.value) - Math.min(...allChartValues.value)) * 0.05
})
const chartRangeY = computed(() => chartMaxY.value - chartMinY.value || 1)

function chartX(i) {
  return chartPadding + (i / (chartData.value.length - 1 || 1)) * chartInnerW
}
function chartY(v) {
  return chartPadding + chartInnerH - ((v - chartMinY.value) / chartRangeY.value) * chartInnerH
}
function chartYLabel(idx) {
  const v = chartMaxY.value - (idx / 3) * chartRangeY.value
  return v.toFixed(1)
}

// Actual values (downsampled same way)
const actualData = computed(() => {
  const actuals = result.value?.actual || []
  if (actuals.length === 0) return []
  if (actuals.length <= 200) return actuals
  const step = actuals.length / 200
  return Array.from({ length: 200 }, (_, i) => actuals[Math.floor(i * step)])
})

// Adjust Y range to include both predicted and actual
const allChartValues = computed(() => {
  return [...chartData.value, ...actualData.value].filter(v => typeof v === 'number')
})

const chartLinePath = computed(() => {
  if (chartData.value.length === 0) return ''
  return chartData.value.map((v, i) => `${i === 0 ? 'M' : 'L'}${chartX(i).toFixed(1)},${chartY(v).toFixed(1)}`).join(' ')
})
const actualLinePath = computed(() => {
  if (actualData.value.length === 0) return ''
  const n = actualData.value.length
  const maxPts = chartData.value.length || n
  return actualData.value.map((v, i) => {
    const x = chartPadding + (i / (n - 1 || 1)) * chartInnerW
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${chartY(v).toFixed(1)}`
  }).join(' ')
})

const chartAreaPath = computed(() => {
  if (chartData.value.length === 0) return ''
  const n = chartData.value.length
  return `${chartLinePath.value} L${chartX(n-1).toFixed(1)},${chartHeight - chartPadding} L${chartX(0).toFixed(1)},${chartHeight - chartPadding} Z`
})
const runError = ref(null)

const MODE_COLORS = {
  anomaly: '#f87171',
  classification: '#34d399',
  regression: '#a78bfa',
}

// Parse nodes once (API may return string or array)
const parsedNodes = computed(() => {
  let nodes = appData.value.nodes || []
  if (typeof nodes === 'string') {
    try { nodes = JSON.parse(nodes) } catch { nodes = [] }
  }
  return Array.isArray(nodes) ? nodes : []
})

const appMode = computed(() => {
  return appData.value.mode || 'regression'
})

const modeColor = computed(() => MODE_COLORS[appMode.value] || '#94a3b8')

const appAlgorithm = computed(() => {
  return appData.value.algorithm || ''
})

// Pipeline info from nodes
const pipelineInfo = computed(() => {
  const windowNode = parsedNodes.value.find(n => n.type === 'transform.window')
  const featNode = parsedNodes.value.find(n => n.type === 'transform.feature_extract')
  if (!windowNode) return null
  return {
    window_size: windowNode.config?.window_size || '?',
    stride: windowNode.config?.step || windowNode.config?.stride || '?',
    n_features: featNode?.config?.features?.length || '?',
    algorithm: appData.value.algorithm || 'model',
  }
})

// Expected CSV columns from model's sensor_columns
const expectedColumns = computed(() => {
  const nodes = appData.value.nodes || []
  const normNode = nodes.find(n => n.type === 'transform.normalize')
  // The model endpoint's pipeline_config has sensor_columns
  // These are returned by the by-slug API
  return appData.value.sensor_columns || []
})

onMounted(async () => {
  try {
    const resp = await api.get(`/api/app-builder/apps/by-slug/${slug.value}`)
    appData.value = resp.data
  } catch (e) {
    // Try getting by ID as fallback
    try {
      const resp = await api.get(`/api/app-builder/apps/${slug.value}`)
      appData.value = resp.data
    } catch {
      error.value = 'This app does not exist or has not been published.'
    }
  }
  loading.value = false
})

const fileInput = ref(null)

function onFileSelect(e) {
  selectedFile.value = e.target.files[0] || null
  result.value = null
  runError.value = null
}

function clearFile() {
  selectedFile.value = null
  result.value = null
  runError.value = null
  // Reset file input so the same file can be re-selected
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

async function runPipeline() {
  if (!selectedFile.value) return
  running.value = true
  runError.value = null
  result.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const resp = await api.post(`/api/app-builder/run/${slug.value}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
    result.value = resp.data
  } catch (e) {
    runError.value = e.response?.data?.error || e.message || 'Pipeline execution failed'
  }
  running.value = false
}
</script>

<style scoped>
.published-app {
  min-height: 100vh;
  background: #0d1117;
  color: #e6edf3;
}

.app-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
}

.app-content {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px 16px;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 0;
  border-bottom: 1px solid #21262d;
  margin-bottom: 24px;
}

.app-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-logo {
  width: 32px;
  height: 32px;
}

.app-title {
  font-size: 18px;
  font-weight: 600;
}

.app-subtitle {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
}

.app-mode-badge {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.5px;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}

.app-algo {
  font-size: 11px;
  color: #8b949e;
  font-family: monospace;
}

.app-powered {
  font-size: 10px;
  color: #484f58;
}

.app-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.app-section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #c9d1d9;
}

.app-dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px;
  border: 2px dashed #30363d;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s;
}
.app-dropzone:hover {
  border-color: #a78bfa;
}

.app-file-selected {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  font-size: 12px;
  font-family: monospace;
}

.result-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.result-stat {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 12px 16px;
  flex: 1;
}

.result-stat-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.result-stat-value {
  font-size: 20px;
  font-weight: 600;
  font-family: monospace;
  margin-top: 4px;
}

.result-table-wrap {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #21262d;
  border-radius: 6px;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: monospace;
}

.result-table th {
  background: #0d1117;
  padding: 8px 12px;
  text-align: left;
  color: #8b949e;
  font-weight: 600;
  border-bottom: 1px solid #21262d;
  position: sticky;
  top: 0;
}

.result-table td {
  padding: 6px 12px;
  border-bottom: 1px solid #161b22;
  color: #e6edf3;
}

.result-table tr:hover td {
  background: #161b22;
}

.result-json {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 12px;
  font-size: 11px;
  font-family: monospace;
  color: #8b949e;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.chart-container {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.chart-title-text {
  font-size: 13px;
  font-weight: 600;
  color: #c9d1d9;
}

.chart-legend-items {
  display: flex;
  align-items: center;
  gap: 4px;
}

.chart-legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.chart-legend-label {
  font-size: 10px;
  color: #8b949e;
}

.prediction-chart {
  width: 100%;
  height: auto;
}

.table-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
  cursor: pointer;
  font-size: 12px;
  color: #8b949e;
  user-select: none;
}
.table-toggle:hover {
  color: #c9d1d9;
}

.expected-format {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 12px;
}

.expected-col {
  font-size: 10px;
  font-family: monospace;
  padding: 2px 6px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 3px;
  color: #a78bfa;
}
</style>
