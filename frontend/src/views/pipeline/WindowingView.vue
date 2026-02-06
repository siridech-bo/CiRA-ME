<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="windowing" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Windowing & Preprocessing</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Configure time-series segmentation settings
    </p>

    <v-row>
      <!-- Window Configuration -->
      <v-col cols="12" md="6">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Window Settings</h3>

          <!-- Window Size -->
          <div class="mb-6">
            <div class="d-flex justify-space-between mb-2">
              <span class="text-body-2">Window Size (samples)</span>
              <span class="font-weight-medium">{{ windowingConfig.window_size }}</span>
            </div>
            <v-slider
              v-model="windowingConfig.window_size"
              :min="16"
              :max="sliderMaxWindowSize"
              :step="16"
              color="primary"
              hide-details
            />
            <div v-if="minSampleLength > 0" class="text-caption text-medium-emphasis mt-1">
              Max: {{ minSampleLength.toLocaleString() }} (smallest sample in dataset)
            </div>
          </div>

          <!-- Warning if no windows possible -->
          <v-alert
            v-if="estimatedWindows === 0 && totalSamples > 0"
            type="warning"
            variant="tonal"
            density="compact"
            class="mb-4"
          >
            Window size ({{ windowingConfig.window_size }}) exceeds data length ({{ minSampleLength }}).
            Try {{ recommendedWindowSize }} or smaller.
          </v-alert>

          <!-- Stride -->
          <div class="mb-6">
            <div class="d-flex justify-space-between mb-2">
              <span class="text-body-2">Stride (samples)</span>
              <span class="font-weight-medium">{{ windowingConfig.stride }}</span>
            </div>
            <v-slider
              v-model="windowingConfig.stride"
              :min="16"
              :max="windowingConfig.window_size"
              :step="16"
              color="secondary"
              hide-details
            />
          </div>

          <!-- Overlap Display -->
          <v-alert type="info" variant="tonal" density="compact" class="mb-4">
            <strong>Overlap:</strong> {{ overlapPercent }}%
          </v-alert>

          <!-- Label Preservation -->
          <h4 class="text-subtitle-2 font-weight-bold mb-3">Label Preservation</h4>
          <v-radio-group v-model="windowingConfig.label_method" hide-details>
            <v-radio value="majority">
              <template #label>
                <div>
                  <div class="font-weight-medium">Majority Voting</div>
                  <div class="text-caption text-medium-emphasis">
                    Assign most common label in window
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="first">
              <template #label>
                <div>
                  <div class="font-weight-medium">First Label</div>
                  <div class="text-caption text-medium-emphasis">
                    Use label from first sample
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="last">
              <template #label>
                <div>
                  <div class="font-weight-medium">Last Label</div>
                  <div class="text-caption text-medium-emphasis">
                    Use label from last sample
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="threshold">
              <template #label>
                <div>
                  <div class="font-weight-medium">Threshold (>50%)</div>
                  <div class="text-caption text-medium-emphasis">
                    Label if majority exceeds 50%
                  </div>
                </div>
              </template>
            </v-radio>
          </v-radio-group>
        </v-card>
      </v-col>

      <!-- Preview -->
      <v-col cols="12" md="6">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Window Preview</h3>

          <!-- Visualization -->
          <div class="window-preview mb-4">
            <svg viewBox="0 0 400 100" class="w-100">
              <!-- Signal line -->
              <path
                d="M 10 50 Q 50 20, 90 50 T 170 50 T 250 50 T 330 50 L 390 50"
                stroke="#6366F1"
                stroke-width="2"
                fill="none"
              />

              <!-- Windows -->
              <rect
                v-for="(w, i) in previewWindows"
                :key="i"
                :x="w.x"
                y="10"
                :width="w.width"
                height="80"
                :fill="w.color"
                :opacity="0.2"
                stroke="#22D3EE"
                stroke-width="1"
                rx="4"
              />

              <!-- Window labels -->
              <text
                v-for="(w, i) in previewWindows"
                :key="'t' + i"
                :x="w.x + w.width / 2"
                y="55"
                text-anchor="middle"
                fill="#94A3B8"
                font-size="10"
              >
                W{{ i + 1 }}
              </text>
            </svg>
          </div>

          <!-- Stats -->
          <v-row dense>
            <v-col cols="6">
              <v-card variant="tonal" class="pa-3 text-center">
                <div class="text-caption text-medium-emphasis">Total Samples</div>
                <div class="text-h6">{{ totalSamples.toLocaleString() }}</div>
              </v-card>
            </v-col>
            <v-col cols="6">
              <v-card variant="tonal" class="pa-3 text-center">
                <div class="text-caption text-medium-emphasis">Windows Created</div>
                <div class="text-h6">{{ estimatedWindows }}</div>
              </v-card>
            </v-col>
            <v-col cols="6">
              <v-card variant="tonal" class="pa-3 text-center">
                <div class="text-caption text-medium-emphasis">Samples per Window</div>
                <div class="text-h6">{{ windowingConfig.window_size }}</div>
              </v-card>
            </v-col>
            <v-col cols="6">
              <v-card variant="tonal" class="pa-3 text-center">
                <div class="text-caption text-medium-emphasis">Sensor Channels</div>
                <div class="text-h6">{{ sensorColumns }}</div>
              </v-card>
            </v-col>
          </v-row>
        </v-card>

        <!-- Windowing Result -->
        <v-card v-if="windowedResult" class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Windowing Complete</h3>

          <v-alert type="success" variant="tonal" class="mb-4">
            <strong>{{ windowedResult.num_windows }}</strong> windows created successfully
          </v-alert>

          <div v-if="windowedResult.summary?.label_distribution">
            <h4 class="text-subtitle-2 mb-2">Label Distribution</h4>
            <v-chip
              v-for="(count, label) in windowedResult.summary.label_distribution"
              :key="label"
              size="small"
              class="mr-2 mb-2"
              :color="label === 'anomaly' ? 'error' : 'success'"
              variant="flat"
            >
              {{ label }}: {{ count }}
            </v-chip>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Signal Visualization Section -->
    <v-row v-if="windowedResult" class="mt-4">
      <v-col cols="12">
        <v-card class="pa-4">
          <div class="d-flex justify-space-between align-center mb-4">
            <h3 class="text-subtitle-1 font-weight-bold">Signal Visualization</h3>
            <div class="d-flex align-center">
              <v-btn
                icon="mdi-chevron-left"
                variant="text"
                size="small"
                :disabled="navigationInfo.current <= 1 || loadingChart"
                @click="prevWindow"
              />
              <span class="mx-3 text-body-2">
                Window <strong>{{ navigationInfo.current }}</strong> of <strong>{{ navigationInfo.total }}</strong>
                <span v-if="selectedLabel" class="text-caption text-medium-emphasis ml-1">({{ selectedLabel }})</span>
              </span>
              <v-btn
                icon="mdi-chevron-right"
                variant="text"
                size="small"
                :disabled="navigationInfo.current >= navigationInfo.total || loadingChart"
                @click="nextWindow"
              />
            </div>
          </div>

          <!-- Label Filter Selector -->
          <div class="mb-3">
            <span class="text-caption text-medium-emphasis mr-2">Filter by label:</span>
            <v-chip
              size="small"
              class="mr-1 mb-1"
              :color="!selectedLabel ? 'primary' : 'default'"
              :variant="!selectedLabel ? 'flat' : 'outlined'"
              @click="selectLabel(null)"
            >
              All
            </v-chip>
            <v-chip
              v-for="label in availableLabels"
              :key="label"
              size="small"
              class="mr-1 mb-1"
              :color="selectedLabel === label ? (label === 'anomaly' ? 'error' : 'primary') : 'default'"
              :variant="selectedLabel === label ? 'flat' : 'outlined'"
              @click="selectLabel(label)"
            >
              {{ label }}
              <span class="ml-1 text-caption">({{ labelIndices[label]?.length || 0 }})</span>
            </v-chip>
          </div>

          <!-- Current Window Label Badge -->
          <div v-if="windowSample?.label" class="mb-3">
            <v-chip
              :color="windowSample.label === 'anomaly' ? 'error' : 'success'"
              variant="tonal"
              size="small"
            >
              Current: {{ windowSample.label }}
            </v-chip>
          </div>

          <!-- Chart -->
          <div class="chart-container" style="height: 300px; position: relative;">
            <v-progress-circular
              v-if="loadingChart"
              indeterminate
              color="primary"
              class="chart-loading"
            />
            <Line
              v-else-if="windowSample"
              :data="chartData"
              :options="chartOptions"
            />
            <div v-else class="d-flex align-center justify-center h-100 text-medium-emphasis">
              Loading signal data...
            </div>
          </div>

          <!-- Window Index Slider -->
          <div class="mt-4">
            <v-slider
              v-if="!selectedLabel"
              v-model="currentWindowIndex"
              :min="0"
              :max="(windowSample?.total_windows || windowedResult.num_windows) - 1"
              :step="1"
              color="primary"
              hide-details
              :disabled="loadingChart"
              @update:model-value="onSliderChange"
            >
              <template #prepend>
                <span class="text-caption">0</span>
              </template>
              <template #append>
                <span class="text-caption">{{ (windowSample?.total_windows || windowedResult.num_windows) - 1 }}</span>
              </template>
            </v-slider>
            <v-slider
              v-else
              v-model="filteredWindowIndex"
              :min="0"
              :max="(currentFilteredIndices?.length || 1) - 1"
              :step="1"
              color="primary"
              hide-details
              :disabled="loadingChart"
              @update:model-value="(idx: number) => { currentWindowIndex = currentFilteredIndices![idx]; fetchWindowSample(currentWindowIndex) }"
            >
              <template #prepend>
                <span class="text-caption">1</span>
              </template>
              <template #append>
                <span class="text-caption">{{ currentFilteredIndices?.length || 0 }}</span>
              </template>
            </v-slider>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="router.push({ name: 'pipeline-data' })"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          color="secondary"
          size="large"
          class="mr-2"
          :loading="loading"
          @click="applyWindowing"
        >
          Apply Windowing
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :disabled="!windowedResult"
          @click="router.push({ name: 'pipeline-features' })"
        >
          Continue to Features
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, reactive, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import api from '@/services/api'

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

const loading = ref(false)
const windowedResult = ref<any>(null)

// Signal visualization state
const currentWindowIndex = ref(0)
const windowSample = ref<any>(null)
const loadingChart = ref(false)

// Label filtering state
const selectedLabel = ref<string | null>(null)
const labelIndices = ref<Record<string, number[]>>({})
const filteredWindowIndex = ref(0) // Index within filtered windows

// Chart colors for different channels
const channelColors = [
  '#6366F1', // indigo
  '#22D3EE', // cyan
  '#F59E0B', // amber
  '#10B981', // emerald
  '#EF4444', // red
  '#8B5CF6'  // violet
]

// Chart configuration
const chartData = computed(() => {
  if (!windowSample.value) return { labels: [], datasets: [] }

  const labels = Array.from({ length: windowSample.value.window_size }, (_, i) => i)

  const datasets = windowSample.value.channels.map((channel: string, idx: number) => ({
    label: channel,
    data: windowSample.value.data[idx],
    borderColor: channelColors[idx % channelColors.length],
    backgroundColor: channelColors[idx % channelColors.length] + '20',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1
  }))

  return { labels, datasets }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom' as const,
      labels: {
        usePointStyle: true,
        padding: 15
      }
    },
    tooltip: {
      mode: 'index' as const,
      intersect: false
    }
  },
  scales: {
    x: {
      title: {
        display: true,
        text: 'Sample Index'
      },
      grid: {
        display: false
      }
    },
    y: {
      title: {
        display: true,
        text: 'Amplitude'
      },
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      }
    }
  },
  interaction: {
    mode: 'nearest' as const,
    axis: 'x' as const,
    intersect: false
  }
}

// Get available labels from windowing result
const availableLabels = computed(() => {
  if (!windowedResult.value?.summary?.label_distribution) return []
  return Object.keys(windowedResult.value.summary.label_distribution)
})

// Get current filtered window indices
const currentFilteredIndices = computed(() => {
  if (!selectedLabel.value || !labelIndices.value[selectedLabel.value]) {
    return null // No filter - use all windows
  }
  return labelIndices.value[selectedLabel.value]
})

// Get display info for navigation
const navigationInfo = computed(() => {
  if (currentFilteredIndices.value) {
    return {
      current: filteredWindowIndex.value + 1,
      total: currentFilteredIndices.value.length,
      actualIndex: currentFilteredIndices.value[filteredWindowIndex.value]
    }
  }
  return {
    current: currentWindowIndex.value + 1,
    total: windowSample.value?.total_windows || windowedResult.value?.num_windows || 0,
    actualIndex: currentWindowIndex.value
  }
})

// Fetch label indices for filtering
async function fetchLabelIndices() {
  if (!windowedResult.value?.session_id) return

  try {
    const response = await api.post('/api/data/windows-by-label', {
      session_id: windowedResult.value.session_id
    })
    labelIndices.value = response.data.labels || {}
  } catch (e: any) {
    console.error('Failed to fetch label indices:', e)
  }
}

// Fetch window sample data
async function fetchWindowSample(index: number) {
  if (!windowedResult.value?.session_id) return

  loadingChart.value = true
  try {
    const response = await api.post('/api/data/window-sample', {
      session_id: windowedResult.value.session_id,
      window_index: index
    })
    windowSample.value = response.data
  } catch (e: any) {
    console.error('Failed to fetch window sample:', e)
    notificationStore.showError('Failed to load window data')
  } finally {
    loadingChart.value = false
  }
}

// Select a label to filter by
function selectLabel(label: string | null) {
  selectedLabel.value = label
  filteredWindowIndex.value = 0

  if (label && labelIndices.value[label]?.length > 0) {
    // Navigate to first window of selected label
    currentWindowIndex.value = labelIndices.value[label][0]
    fetchWindowSample(currentWindowIndex.value)
  } else if (!label) {
    // Clear filter - go to first window
    currentWindowIndex.value = 0
    fetchWindowSample(0)
  }
}

// Navigate to previous window (respecting filter)
function prevWindow() {
  if (currentFilteredIndices.value) {
    // Filtered navigation
    if (filteredWindowIndex.value > 0) {
      filteredWindowIndex.value--
      currentWindowIndex.value = currentFilteredIndices.value[filteredWindowIndex.value]
      fetchWindowSample(currentWindowIndex.value)
    }
  } else {
    // Unfiltered navigation
    if (currentWindowIndex.value > 0) {
      currentWindowIndex.value--
      fetchWindowSample(currentWindowIndex.value)
    }
  }
}

// Navigate to next window (respecting filter)
function nextWindow() {
  if (currentFilteredIndices.value) {
    // Filtered navigation
    if (filteredWindowIndex.value < currentFilteredIndices.value.length - 1) {
      filteredWindowIndex.value++
      currentWindowIndex.value = currentFilteredIndices.value[filteredWindowIndex.value]
      fetchWindowSample(currentWindowIndex.value)
    }
  } else {
    // Unfiltered navigation
    if (windowSample.value && currentWindowIndex.value < windowSample.value.total_windows - 1) {
      currentWindowIndex.value++
      fetchWindowSample(currentWindowIndex.value)
    }
  }
}

// Handle slider change (only when not filtered)
function onSliderChange(index: number) {
  if (!currentFilteredIndices.value) {
    fetchWindowSample(index)
  }
}

// Watch for windowed result changes to load first sample and label indices
watch(windowedResult, (newVal) => {
  if (newVal?.session_id) {
    currentWindowIndex.value = 0
    filteredWindowIndex.value = 0
    selectedLabel.value = null
    fetchWindowSample(0)
    fetchLabelIndices()
  }
})

const windowingConfig = reactive({
  window_size: pipelineStore.windowingConfig.window_size,
  stride: pipelineStore.windowingConfig.stride,
  label_method: pipelineStore.windowingConfig.label_method
})

const totalSamples = computed(() =>
  pipelineStore.dataSession?.metadata?.total_rows || 0
)

const minSampleLength = computed(() =>
  pipelineStore.dataSession?.metadata?.min_sample_length || totalSamples.value
)

const sensorColumns = computed(() =>
  pipelineStore.dataSession?.metadata?.sensor_columns?.length || 0
)

const sliderMaxWindowSize = computed(() => {
  if (minSampleLength.value <= 0) return 512
  // Round down to nearest step of 16, minimum 16
  return Math.max(16, Math.floor(minSampleLength.value / 16) * 16)
})

const recommendedWindowSize = computed(() => {
  const maxValid = minSampleLength.value
  // Largest power of 2 that fits
  let size = 16
  while (size * 2 <= maxValid) size *= 2
  return size
})

const overlapPercent = computed(() => {
  const overlap = windowingConfig.window_size - windowingConfig.stride
  return Math.round((overlap / windowingConfig.window_size) * 100)
})

const estimatedWindows = computed(() => {
  if (totalSamples.value === 0) return 0
  const sampleLen = minSampleLength.value
  if (sampleLen < windowingConfig.window_size) return 0
  const numSamples = pipelineStore.dataSession?.metadata?.total_samples || 1
  const windowsPerSample = Math.floor((sampleLen - windowingConfig.window_size) / windowingConfig.stride) + 1
  return windowsPerSample * numSamples
})

const previewWindows = computed(() => {
  const windows = []
  const totalWidth = 380
  const windowWidth = 80
  const strideWidth = windowWidth * (windowingConfig.stride / windowingConfig.window_size)

  for (let i = 0; i < 4; i++) {
    const x = 10 + i * strideWidth
    if (x + windowWidth <= totalWidth) {
      windows.push({
        x,
        width: windowWidth,
        color: i % 2 === 0 ? '#6366F1' : '#22D3EE'
      })
    }
  }

  return windows
})

// Clamp window size when data changes and current size exceeds max
watch(sliderMaxWindowSize, (newMax) => {
  if (windowingConfig.window_size > newMax) {
    windowingConfig.window_size = newMax
  }
  if (windowingConfig.stride > windowingConfig.window_size) {
    windowingConfig.stride = windowingConfig.window_size
  }
})

async function applyWindowing() {
  // Update store config
  pipelineStore.windowingConfig.window_size = windowingConfig.window_size
  pipelineStore.windowingConfig.stride = windowingConfig.stride
  pipelineStore.windowingConfig.label_method = windowingConfig.label_method

  loading.value = true

  const result = await pipelineStore.applyWindowing()

  if (result.success) {
    windowedResult.value = result.data
    notificationStore.showSuccess('Windowing applied successfully')
  } else {
    notificationStore.showError(result.error || 'Failed to apply windowing')
  }

  loading.value = false
}
</script>

<style scoped lang="scss">
.window-preview {
  background: rgba(var(--v-theme-surface-variant), 0.3);
  border-radius: 8px;
  padding: 16px;
}

.chart-container {
  background: rgba(var(--v-theme-surface-variant), 0.1);
  border-radius: 8px;
  padding: 16px;
}

.chart-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
</style>
