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
              <span v-if="isRecorderMode" class="app-mode-badge" style="color: #ef4444; background: rgba(239,68,68,0.1)">
                SIGNAL RECORDER
              </span>
              <span v-else-if="appMode" class="app-mode-badge" :style="{ color: modeColor, background: modeColor + '18' }">
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

      <!-- MQTT Live Stream Mode -->
      <div v-if="isLiveStream" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" :color="mqttConnected ? 'success' : 'grey'">mdi-access-point</v-icon>
          Live Stream (MQTT)
          <v-chip v-if="mqttConnected" size="x-small" color="success" variant="flat" class="ml-2">
            <v-icon start size="8">mdi-circle</v-icon> LIVE
          </v-chip>
          <v-chip v-else-if="mqttError" size="x-small" color="error" variant="tonal" class="ml-2">
            Error
          </v-chip>
        </div>

        <!-- Connection config -->
        <div class="d-flex align-center gap-2 mb-3" style="flex-wrap: wrap;">
          <v-text-field
            v-model="mqttBrokerUrl"
            label="Broker URL"
            variant="outlined"
            density="compact"
            hide-details
            style="max-width: 300px; font-size: 12px;"
            :disabled="mqttConnected"
          />
          <v-text-field
            v-model="mqttTopic"
            label="Topic"
            variant="outlined"
            density="compact"
            hide-details
            style="max-width: 200px; font-size: 12px;"
            :disabled="mqttConnected"
          />
          <v-btn
            v-if="!mqttConnected"
            color="success"
            variant="flat"
            size="small"
            @click="startLiveStream"
          >
            <v-icon start size="small">mdi-play</v-icon>
            Connect
          </v-btn>
          <v-btn
            v-else
            color="error"
            variant="tonal"
            size="small"
            @click="stopLiveStream"
          >
            <v-icon start size="small">mdi-stop</v-icon>
            Disconnect
          </v-btn>
        </div>

        <!-- Error message -->
        <v-alert v-if="mqttError" type="error" variant="tonal" density="compact" class="mb-3" closable>
          {{ mqttError }}
        </v-alert>

        <!-- Live stats -->
        <div v-if="mqttConnected" class="d-flex flex-wrap gap-3 mb-3">
          <div class="live-stat">
            <div class="live-stat-label">Messages</div>
            <div class="live-stat-value">{{ mqttMessageCount.toLocaleString() }}</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Rate</div>
            <div class="live-stat-value">{{ mqttMessagesPerSec }}/s</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Buffer</div>
            <div class="live-stat-value">{{ sensorBufferLen }}/{{ liveWindowSize }}</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Inferences</div>
            <div class="live-stat-value">{{ liveInferenceCount }}</div>
          </div>
        </div>

        <!-- Buffer progress bar -->
        <v-progress-linear
          v-if="mqttConnected"
          :model-value="sensorBufferProgress * 100"
          color="purple"
          height="6"
          rounded
          class="mb-3"
        />

        <!-- Live prediction (inference mode) -->
        <div v-if="livePrediction !== null && !isRecorderMode && !isMultiModelApp" class="live-prediction">
          <div class="live-prediction-label">Latest Prediction</div>
          <div class="live-prediction-value" :style="{ color: modeColor }">
            {{ livePrediction }}
          </div>
          <div v-if="liveLastUpdated" class="live-prediction-time">
            {{ liveLastUpdatedText }}
          </div>
        </div>

        <!-- Signal Recorder Mode -->
        <div v-if="isRecorderMode && mqttConnected" class="recorder-section">
          <!-- Label buttons -->
          <div class="recorder-label-header">
            <span class="text-caption font-weight-bold text-medium-emphasis">CURRENT LABEL</span>
            <v-chip v-if="recorderState.recording" size="x-small" color="error" variant="flat" class="ml-2">
              <v-icon start size="8">mdi-circle</v-icon> REC
            </v-chip>
          </div>
          <div class="recorder-labels">
            <button
              v-for="lbl in recorderLabels"
              :key="lbl"
              class="recorder-label-btn"
              :class="{ active: recorderState.currentLabel === lbl }"
              @click="recorderState.currentLabel = lbl"
            >{{ lbl }}</button>
          </div>

          <!-- Custom label input -->
          <div class="d-flex align-center gap-2 mt-2 mb-3">
            <v-text-field
              v-model="recorderCustomLabel"
              label="Add custom label"
              variant="outlined"
              density="compact"
              hide-details
              style="max-width: 200px; font-size: 11px;"
              @keydown.enter="addCustomLabel"
            />
            <v-btn size="x-small" variant="tonal" @click="addCustomLabel" :disabled="!recorderCustomLabel.trim()">
              <v-icon size="small">mdi-plus</v-icon>
            </v-btn>
          </div>

          <!-- Recording controls -->
          <div class="d-flex align-center gap-2 mb-3">
            <v-btn
              v-if="!recorderState.recording"
              color="error"
              variant="flat"
              size="small"
              :disabled="!recorderState.currentLabel"
              @click="startRecording"
            >
              <v-icon start size="small">mdi-record</v-icon>
              Start Recording
            </v-btn>
            <v-btn
              v-else
              color="warning"
              variant="flat"
              size="small"
              @click="stopRecording"
            >
              <v-icon start size="small">mdi-stop</v-icon>
              Stop Recording
            </v-btn>
            <v-btn
              v-if="recorderState.samples.length > 0"
              color="success"
              variant="tonal"
              size="small"
              @click="downloadRecordedCSV"
            >
              <v-icon start size="small">mdi-download</v-icon>
              Download CSV ({{ recorderState.samples.length }} samples)
            </v-btn>
            <v-btn
              v-if="recorderState.samples.length > 0 && !recorderState.recording"
              variant="text"
              size="small"
              color="error"
              @click="clearRecording"
            >
              Clear
            </v-btn>
          </div>

          <!-- Recording stats -->
          <div class="d-flex flex-wrap gap-3 mb-2">
            <div class="live-stat">
              <div class="live-stat-label">Samples</div>
              <div class="live-stat-value">{{ recorderState.samples.length }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Duration</div>
              <div class="live-stat-value">{{ recorderDuration }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Labels</div>
              <div class="live-stat-value">{{ recorderLabelCounts }}</div>
            </div>
          </div>

          <!-- Label timeline -->
          <div v-if="recorderState.samples.length > 0" class="recorder-timeline">
            <div
              v-for="(seg, i) in recorderSegments"
              :key="i"
              class="recorder-segment"
              :style="{ flex: seg.count, background: seg.color }"
              :title="`${seg.label}: ${seg.count} samples`"
            />
          </div>
        </div>
      </div>

      <!-- CSV Upload Mode -->
      <div v-else class="app-section">
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

      <!-- Multi-Model Comparison Results -->
      <div v-if="result && result.multi_model" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="amber">mdi-compare-horizontal</v-icon>
          Multi-Model Comparison
          <v-chip size="x-small" color="amber" variant="tonal" class="ml-2">
            {{ Object.keys(result.models || {}).length }} models
          </v-chip>
        </div>

        <!-- Metrics comparison table -->
        <v-table density="compact" class="mb-4">
          <thead>
            <tr>
              <th>Model</th>
              <th>Algorithm</th>
              <template v-if="result.mode === 'regression'">
                <th class="text-center">R²</th>
                <th class="text-center">RMSE</th>
                <th class="text-center">MAE</th>
              </template>
              <template v-else-if="result.mode === 'classification'">
                <th class="text-center">Accuracy</th>
                <th class="text-center">Precision</th>
                <th class="text-center">F1</th>
              </template>
              <template v-else>
                <th class="text-center">Predictions</th>
              </template>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, eid) in result.models" :key="eid"
                :class="{
                  'bg-amber-darken-4': result.mode === 'regression' ? m.r2 === bestR2 : m.accuracy === bestAccuracy
                }">
              <td class="font-weight-medium">
                {{ m.name }}
                <v-icon v-if="(result.mode === 'regression' && m.r2 === bestR2) || (result.mode === 'classification' && m.accuracy === bestAccuracy)"
                        size="14" color="amber" class="ml-1">mdi-trophy</v-icon>
              </td>
              <td class="text-caption">{{ m.algorithm }}</td>
              <template v-if="result.mode === 'regression'">
                <td class="text-center" :style="{ color: m.r2 > 0.8 ? '#34d399' : m.r2 > 0.5 ? '#fbbf24' : '#f87171' }">
                  {{ m.r2 != null ? m.r2.toFixed(4) : '-' }}
                </td>
                <td class="text-center">{{ m.rmse != null ? m.rmse.toFixed(4) : '-' }}</td>
                <td class="text-center">{{ m.mae != null ? m.mae.toFixed(4) : '-' }}</td>
              </template>
              <template v-else-if="result.mode === 'classification'">
                <td class="text-center" :style="{ color: m.accuracy > 0.9 ? '#34d399' : m.accuracy > 0.7 ? '#fbbf24' : '#f87171' }">
                  {{ m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '-' }}
                </td>
                <td class="text-center">{{ m.precision != null ? (m.precision * 100).toFixed(1) + '%' : '-' }}</td>
                <td class="text-center">{{ m.f1 != null ? (m.f1 * 100).toFixed(1) + '%' : '-' }}</td>
              </template>
              <template v-else>
                <td class="text-center">{{ m.count || 0 }}</td>
              </template>
            </tr>
          </tbody>
        </v-table>

        <!-- Multi-model chart (regression) -->
        <div v-if="result.mode === 'regression' && multiChartData.length > 0" class="chart-container">
          <div class="chart-header">
            <span class="chart-title-text">Model Predictions Comparison</span>
          </div>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight + 20}`" class="prediction-chart">
            <line v-for="i in 4" :key="'g'+i"
              :x1="chartPadding" :y1="chartPadding + (i-1) * (chartInnerH / 3)"
              :x2="chartWidth - chartPadding" :y2="chartPadding + (i-1) * (chartInnerH / 3)"
              stroke="#21262d" stroke-width="0.5" />
            <!-- Actual line -->
            <path v-if="multiActualPath" :d="multiActualPath" fill="none" stroke="#22d3ee" stroke-width="2" />
            <!-- Model lines -->
            <path v-for="(mp, idx) in multiModelPaths" :key="'mp'+idx"
              :d="mp.path" fill="none" :stroke="mp.color" stroke-width="1.5"
              :stroke-dasharray="idx > 0 ? '4,2' : 'none'" />
          </svg>
          <div class="chart-legend-items" style="margin-top:4px; flex-wrap: wrap;">
            <span v-if="result.actual" class="chart-legend-dot" style="background: #22d3ee"></span>
            <span v-if="result.actual" class="chart-legend-label">Actual</span>
            <template v-for="(mp, idx) in multiModelPaths" :key="'leg'+idx">
              <span class="chart-legend-dot" :style="{ background: mp.color }"></span>
              <span class="chart-legend-label">{{ mp.name }}</span>
            </template>
          </div>
        </div>

        <!-- Download CSV -->
        <v-btn color="success" variant="tonal" size="small" class="mt-3" @click="downloadMultiModelCsv">
          <v-icon start size="small">mdi-download</v-icon>
          Download Comparison CSV
        </v-btn>
      </div>

      <!-- Single Model Results section -->
      <div v-else-if="result" class="app-section">
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
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import api from '@/services/api'
// mqtt loaded dynamically only when needed (live stream mode)
let mqtt = null

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
  // Initialize MQTT config from pipeline nodes
  if (isLiveStream.value) {
    const cfg = liveStreamConfig.value
    // Auto-resolve broker URL: replace 'localhost' with actual server hostname
    // so the app works when opened from other machines on the LAN
    let brokerUrl = cfg.broker_url || 'ws://localhost:9001/mqtt'
    if (brokerUrl.includes('localhost') || brokerUrl.includes('127.0.0.1')) {
      brokerUrl = brokerUrl.replace('localhost', window.location.hostname)
                           .replace('127.0.0.1', window.location.hostname)
    }
    mqttBrokerUrl.value = brokerUrl
    mqttTopic.value = cfg.topic || 'sensors/#'
  }
  loading.value = false
})

// Multi-model comparison
const MULTI_COLORS = ['#a78bfa', '#f59e0b', '#34d399', '#f87171', '#60a5fa']

const bestR2 = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return null
  let best = -Infinity
  for (const m of Object.values(result.value.models)) {
    if (m.r2 != null && m.r2 > best) best = m.r2
  }
  return best > -Infinity ? best : null
})

const bestAccuracy = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return null
  let best = -1
  for (const m of Object.values(result.value.models)) {
    if (m.accuracy != null && m.accuracy > best) best = m.accuracy
  }
  return best >= 0 ? best : null
})

const multiChartData = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return []
  return Object.entries(result.value.models)
    .filter(([, m]) => m.predictions && m.predictions.length > 0)
    .map(([eid, m], idx) => ({
      eid, name: m.name,
      predictions: m.predictions.filter(v => typeof v === 'number'),
      color: MULTI_COLORS[idx % MULTI_COLORS.length],
    }))
})

const multiAllValues = computed(() => {
  const all = []
  for (const mc of multiChartData.value) all.push(...mc.predictions)
  if (result.value?.actual) all.push(...result.value.actual)
  return all.filter(v => typeof v === 'number')
})

const multiMinY = computed(() => multiAllValues.value.length ? Math.min(...multiAllValues.value) * 0.98 : 0)
const multiMaxY = computed(() => multiAllValues.value.length ? Math.max(...multiAllValues.value) * 1.02 : 1)
const multiRangeY = computed(() => multiMaxY.value - multiMinY.value || 1)

function multiChartX(i, total) {
  return chartPadding + (i / (total - 1 || 1)) * chartInnerW
}
function multiChartY(v) {
  return chartPadding + chartInnerH - ((v - multiMinY.value) / multiRangeY.value) * chartInnerH
}

const multiActualPath = computed(() => {
  const actuals = result.value?.actual
  if (!actuals || actuals.length === 0) return ''
  const ds = actuals.length <= 200 ? actuals : Array.from({length:200}, (_,i) => actuals[Math.floor(i*actuals.length/200)])
  return ds.map((v,i) => `${i===0?'M':'L'}${multiChartX(i,ds.length).toFixed(1)},${multiChartY(v).toFixed(1)}`).join(' ')
})

const multiModelPaths = computed(() => {
  return multiChartData.value.map(mc => {
    const ds = mc.predictions.length <= 200 ? mc.predictions : Array.from({length:200}, (_,i) => mc.predictions[Math.floor(i*mc.predictions.length/200)])
    const path = ds.map((v,i) => `${i===0?'M':'L'}${multiChartX(i,ds.length).toFixed(1)},${multiChartY(v).toFixed(1)}`).join(' ')
    return { name: mc.name, color: mc.color, path }
  })
})

function downloadMultiModelCsv() {
  if (!result.value?.multi_model || !result.value?.models) return
  const models = Object.values(result.value.models).filter(m => m.predictions)
  const actuals = result.value.actual || []
  const maxLen = Math.max(...models.map(m => m.predictions?.length || 0), actuals.length)

  const header = ['datapoint']
  if (actuals.length > 0) header.push('actual')
  for (const m of models) header.push(m.name || 'model')

  const rows = [header.join(',')]
  for (let i = 0; i < maxLen; i++) {
    const row = [i]
    if (actuals.length > 0) row.push(actuals[i] != null ? actuals[i] : '')
    for (const m of models) row.push(m.predictions?.[i] != null ? m.predictions[i] : '')
    rows.push(row.join(','))
  }

  // Summary
  rows.push('')
  rows.push('--- Metrics ---')
  if (result.value?.mode === 'regression') {
    rows.push('model,r2,rmse,mae')
    for (const m of models) {
      rows.push(`${m.name},${m.r2 ?? ''},${m.rmse ?? ''},${m.mae ?? ''}`)
    }
  } else {
    rows.push('model,accuracy,precision,recall,f1')
    for (const m of models) {
      rows.push(`${m.name},${m.accuracy ?? ''},${m.precision ?? ''},${m.recall ?? ''},${m.f1 ?? ''}`)
    }
  }

  const csv = rows.join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `multi_model_comparison_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const fileInput = ref(null)

// ── Live Stream (MQTT) ─────────────────────────────────
const isLiveStream = computed(() => {
  return parsedNodes.value.some(n => n.type === 'input.live_stream')
})

const liveStreamConfig = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'input.live_stream')
  return node?.config || {}
})

const liveWindowSize = computed(() => {
  const wNode = parsedNodes.value.find(n => n.type === 'transform.window')
  return wNode?.config?.window_size || 128
})

const liveStride = computed(() => {
  const wNode = parsedNodes.value.find(n => n.type === 'transform.window')
  return wNode?.config?.step || wNode?.config?.stride || 64
})

const autoDetectedChannels = ref([])

const liveChannels = computed(() => {
  const channels = liveStreamConfig.value.channels || ''
  if (channels) return channels.split(',').map((c) => c.trim()).filter(Boolean)
  if (appData.value.sensor_columns && appData.value.sensor_columns.length > 0) {
    return appData.value.sensor_columns
  }
  // Use auto-detected channels from first MQTT message
  return autoDetectedChannels.value
})

const mqttBrokerUrl = ref('')
const mqttTopic = ref('')
const mqttConnected = ref(false)
const mqttError = ref(null)
const mqttMessageCount = ref(0)
const mqttMessagesPerSec = ref(0)
const sensorBufferLen = ref(0)
const sensorBufferProgress = ref(0)
const liveInferenceCount = ref(0)
const livePrediction = ref(null)
const livePredictionHistory = ref([])
const MAX_LIVE_HISTORY = 200
const liveLastUpdated = ref(null)
let mqttClient = null
let sensorBuffer = []
let rateCounter = 0
let rateInterval = null

const liveLastUpdatedText = computed(() => {
  if (!liveLastUpdated.value) return ''
  const ago = Math.round((Date.now() - liveLastUpdated.value) / 1000)
  return ago < 2 ? 'just now' : `${ago}s ago`
})

// Update "ago" text reactively
const liveUpdateTicker = ref(0)
let tickerInterval = null

async function startLiveStream() {
  mqttError.value = null
  try {
    // Dynamic import — only load mqtt when needed
    if (!mqtt) {
      const mod = await import('mqtt')
      mqtt = mod.default || mod
    }
    mqttClient = mqtt.connect(mqttBrokerUrl.value, {
      clientId: `cira-live-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      clean: true,
      keepalive: 30,
      reconnectPeriod: 3000,
    })

    mqttClient.on('connect', () => {
      mqttConnected.value = true
      mqttError.value = null
      mqttClient.subscribe(mqttTopic.value, { qos: 0 })
    })

    mqttClient.on('error', (err) => {
      mqttError.value = err.message || 'Connection failed'
      mqttConnected.value = false
    })

    mqttClient.on('close', () => {
      mqttConnected.value = false
    })

    mqttClient.on('message', (_topic, payload) => {
      mqttMessageCount.value++
      rateCounter++
      try {
        const raw = JSON.parse(payload.toString())
        const sample = parseSensorPayload(raw)
        if (sample) {
          // Record if in recorder mode and recording
          if (isRecorderMode.value && recorderState.value.recording && recorderState.value.currentLabel) {
            const maxDur = (recorderConfig.value.max_duration || 300) * 1000
            const elapsed = recorderState.value.startTime ? Date.now() - recorderState.value.startTime : 0
            if (elapsed < maxDur) {
              recorderState.value.samples.push({
                ...sample,
                label: recorderState.value.currentLabel,
                _ts: Date.now(),
              })
            } else {
              recorderState.value.recording = false
            }
          }
          // Buffer for inference (non-recorder mode)
          if (!isRecorderMode.value) {
            pushSensorSample(sample)
          }
        }
      } catch { /* ignore non-JSON */ }
    })

    // Rate counter
    rateInterval = setInterval(() => {
      mqttMessagesPerSec.value = rateCounter
      rateCounter = 0
      liveUpdateTicker.value++
    }, 1000)

  } catch (e) {
    mqttError.value = e.message || 'Failed to connect'
  }
}

function stopLiveStream() {
  if (mqttClient) {
    mqttClient.end(true)
    mqttClient = null
  }
  if (rateInterval) {
    clearInterval(rateInterval)
    rateInterval = null
  }
  mqttConnected.value = false
  sensorBuffer = []
  sensorBufferLen.value = 0
  sensorBufferProgress.value = 0
}

// ── Signal Recorder ─────────────────────────────────
const isMultiModelApp = computed(() => {
  return parsedNodes.value.some(n => n.type === 'output.multi_model_compare')
})

const isRecorderMode = computed(() => {
  return parsedNodes.value.some(n => n.type === 'output.signal_recorder')
})

const recorderConfig = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'output.signal_recorder')
  return node?.config || {}
})

const recorderLabels = ref([])
const recorderCustomLabel = ref('')
const recorderState = ref({
  recording: false,
  currentLabel: '',
  samples: [],
  startTime: null,
})

// Initialize labels from config
watch(() => recorderConfig.value, (cfg) => {
  if (cfg.labels) {
    recorderLabels.value = cfg.labels.split(',').map(l => l.trim()).filter(Boolean)
    if (recorderLabels.value.length > 0 && !recorderState.value.currentLabel) {
      recorderState.value.currentLabel = recorderLabels.value[0]
    }
  }
}, { immediate: true })

function addCustomLabel() {
  const lbl = recorderCustomLabel.value.trim()
  if (lbl && !recorderLabels.value.includes(lbl)) {
    recorderLabels.value.push(lbl)
  }
  recorderCustomLabel.value = ''
}

function startRecording() {
  recorderState.value.recording = true
  recorderState.value.startTime = Date.now()
}

function stopRecording() {
  recorderState.value.recording = false
}

function clearRecording() {
  recorderState.value.samples = []
  recorderState.value.startTime = null
}

const recorderDuration = computed(() => {
  const samples = recorderState.value.samples
  if (samples.length === 0) return '0s'
  const first = samples[0]._ts || 0
  const last = samples[samples.length - 1]._ts || 0
  const sec = Math.round((last - first) / 1000)
  return sec < 60 ? `${sec}s` : `${Math.floor(sec/60)}m ${sec%60}s`
})

const recorderLabelCounts = computed(() => {
  const counts = {}
  for (const s of recorderState.value.samples) {
    counts[s.label] = (counts[s.label] || 0) + 1
  }
  return Object.entries(counts).map(([k,v]) => `${k}:${v}`).join(' ')
})

const LABEL_COLORS = ['#60a5fa','#34d399','#f87171','#fbbf24','#a78bfa','#f472b6','#22d3ee','#fb923c']

const recorderSegments = computed(() => {
  const samples = recorderState.value.samples
  if (samples.length === 0) return []
  const segments = []
  let cur = { label: samples[0].label, count: 0, color: '' }
  const labelIdx = {}
  for (const s of samples) {
    if (s.label !== cur.label) {
      segments.push({...cur})
      cur = { label: s.label, count: 0, color: '' }
    }
    cur.count++
    if (!(s.label in labelIdx)) labelIdx[s.label] = Object.keys(labelIdx).length
    cur.color = LABEL_COLORS[labelIdx[s.label] % LABEL_COLORS.length]
  }
  if (cur.count > 0) segments.push({...cur})
  return segments
})

function downloadRecordedCSV() {
  const samples = recorderState.value.samples
  if (samples.length === 0) return
  const channels = liveChannels.value
  const prefix = recorderConfig.value.file_prefix || 'sensor_data'

  // Build CSV
  const header = ['timestamp', ...channels, 'label'].join(',')
  const rows = samples.map((s, i) => {
    const vals = channels.map(ch => s[ch] ?? 0)
    return [i * (1.0 / (recorderConfig.value.target_sample_rate || 62.5)), ...vals, s.label].join(',')
  })
  const csv = [header, ...rows].join('\n')

  // Download
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${prefix}_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function parseSensorPayload(raw) {
  // Normalize different MQTT payload formats into { channel: value } object
  const channels = liveChannels.value

  // Format 1: { "values": [1.2, 3.4, 5.6] } (Android SensorSpot, array)
  if (raw.values && Array.isArray(raw.values)) {
    // Auto-detect channel names if not configured
    if (channels.length === 0 && autoDetectedChannels.value.length === 0) {
      autoDetectedChannels.value = raw.values.map((_, i) => `ch${i}`)
    }
    const ch = channels.length > 0 ? channels : autoDetectedChannels.value
    const sample = {}
    raw.values.forEach((v, i) => {
      const name = ch[i] || `ch${i}`
      sample[name] = typeof v === 'number' ? v : parseFloat(v) || 0
    })
    return sample
  }

  // Format 2: { "values": { "v0": 1.2, "v1": 3.4 } } (SensorSpot named)
  if (raw.values && typeof raw.values === 'object' && !Array.isArray(raw.values)) {
    const keys = Object.keys(raw.values)
    if (channels.length === 0 && autoDetectedChannels.value.length === 0) {
      autoDetectedChannels.value = keys
    }
    const ch = channels.length > 0 ? channels : autoDetectedChannels.value
    const sample = {}
    keys.forEach((k, i) => {
      const name = ch[i] || k
      sample[name] = typeof raw.values[k] === 'number' ? raw.values[k] : parseFloat(raw.values[k]) || 0
    })
    return sample
  }

  // Format 3: { "accX": 1.2, "accY": 3.4, "accZ": 5.6 } (flat object with sensor keys)
  if (typeof raw === 'object') {
    const skip = new Set(['type', 'timestamp', 'time', '_timestamp', '_index', 'name', 'id'])
    const sample = {}
    let hasNumeric = false
    const detectedKeys = []
    for (const [k, v] of Object.entries(raw)) {
      if (!skip.has(k) && typeof v === 'number') {
        sample[k] = v
        hasNumeric = true
        detectedKeys.push(k)
      }
    }
    if (hasNumeric) {
      if (channels.length === 0 && autoDetectedChannels.value.length === 0) {
        autoDetectedChannels.value = detectedKeys
      }
      return sample
    }
  }

  return null
}

function pushSensorSample(sample) {
  sensorBuffer.push(sample)
  sensorBufferLen.value = sensorBuffer.length

  const ws = liveWindowSize.value
  sensorBufferProgress.value = Math.min(sensorBuffer.length / ws, 1)

  if (sensorBuffer.length >= ws) {
    // Extract window and run inference
    const window = sensorBuffer.slice(0, ws)
    const stride = liveStride.value
    sensorBuffer = sensorBuffer.slice(stride)
    sensorBufferLen.value = sensorBuffer.length
    sensorBufferProgress.value = Math.min(sensorBuffer.length / ws, 1)
    runLiveInference(window)
  }
}

async function runLiveInference(windowData) {
  // Convert to CSV-like format for the pipeline runner
  const channels = liveChannels.value
  const csvRows = windowData.map(sample => {
    if (Array.isArray(sample)) return sample
    return channels.map((ch) => sample[ch] ?? 0)
  })

  try {
    const resp = await api.post(`/api/app-builder/run/${slug.value}`, {
      data: csvRows,
      channels: channels,
    }, { timeout: 30000 })

    liveInferenceCount.value++
    liveLastUpdated.value = Date.now()

    if (resp.data?.multi_model) {
      // Multi-model response — show comparison results directly
      const models = resp.data.models || {}
      const names = Object.values(models).map(m => m.name).join(', ')
      livePrediction.value = `${Object.keys(models).length} models compared`
      result.value = resp.data
    } else {
      // Single model response
      const preds = resp.data?.predictions || []
      if (preds.length > 0) {
        const lastPred = preds[preds.length - 1]
        livePrediction.value = lastPred
        if (typeof lastPred === 'number') {
          livePredictionHistory.value.push(lastPred)
          if (livePredictionHistory.value.length > MAX_LIVE_HISTORY) {
            livePredictionHistory.value = livePredictionHistory.value.slice(-MAX_LIVE_HISTORY)
          }
        }
      }
      result.value = {
        ...resp.data,
        predictions: livePredictionHistory.value.length > 0 ? [...livePredictionHistory.value] : resp.data?.predictions,
        count: livePredictionHistory.value.length || resp.data?.count,
      }
    }
  } catch (e) {
    console.error('Live inference error:', e)
  }
}

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

.live-stat {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 80px;
}
.live-stat-label {
  font-size: 9px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.live-stat-value {
  font-size: 16px;
  font-weight: 600;
  font-family: monospace;
  color: #e6edf3;
}
.live-prediction {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.live-prediction-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}
.live-prediction-value {
  font-size: 28px;
  font-weight: 700;
  font-family: monospace;
}
.live-prediction-time {
  font-size: 10px;
  color: #484f58;
  margin-top: 4px;
}

.recorder-section {
  margin-top: 12px;
}
.recorder-label-header {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}
.recorder-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.recorder-label-btn {
  padding: 6px 16px;
  border-radius: 6px;
  border: 2px solid #30363d;
  background: #0d1117;
  color: #8b949e;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.recorder-label-btn:hover {
  border-color: #555;
  color: #c9d1d9;
}
.recorder-label-btn.active {
  border-color: #34d399;
  background: rgba(52, 211, 153, 0.1);
  color: #34d399;
}
.recorder-timeline {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  gap: 1px;
}
.recorder-segment {
  min-width: 2px;
  border-radius: 2px;
  opacity: 0.7;
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
