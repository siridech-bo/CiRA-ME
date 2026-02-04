<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="features" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Feature Engineering</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Select features for {{ pipelineStore.mode === 'anomaly' ? 'anomaly detection' : 'classification' }}
    </p>

    <v-row>
      <!-- Feature Selection -->
      <v-col cols="12" md="8">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4">
            <h3 class="text-subtitle-1 font-weight-bold">Available Features</h3>
            <v-spacer />
            <v-btn-group density="compact">
              <v-btn size="small" @click="selectAll">Select All</v-btn>
              <v-btn size="small" @click="clearSelection">Clear</v-btn>
              <v-btn size="small" @click="selectTSFresh">TSFresh</v-btn>
              <v-btn size="small" @click="selectDSP">DSP</v-btn>
            </v-btn-group>
          </div>

          <!-- Search -->
          <v-text-field
            v-model="searchQuery"
            prepend-inner-icon="mdi-magnify"
            label="Search features..."
            hide-details
            density="compact"
            class="mb-4"
          />

          <!-- Feature List -->
          <v-list
            density="compact"
            class="feature-list"
            max-height="400"
            style="overflow-y: auto"
          >
            <v-list-subheader>TSFresh Features</v-list-subheader>
            <v-list-item
              v-for="feature in filteredTSFreshFeatures"
              :key="feature"
              :class="{ 'selected': selectedFeatures.includes(feature) }"
              @click="toggleFeature(feature)"
            >
              <template #prepend>
                <v-checkbox
                  :model-value="selectedFeatures.includes(feature)"
                  hide-details
                  density="compact"
                  @click.stop="toggleFeature(feature)"
                />
              </template>
              <v-list-item-title>{{ feature }}</v-list-item-title>
              <template #append>
                <v-chip size="x-small" color="info" variant="flat">TSFresh</v-chip>
              </template>
            </v-list-item>

            <v-divider class="my-2" />

            <v-list-subheader>Custom DSP Features</v-list-subheader>
            <v-list-item
              v-for="feature in filteredDSPFeatures"
              :key="feature"
              :class="{ 'selected': selectedFeatures.includes(feature) }"
              @click="toggleFeature(feature)"
            >
              <template #prepend>
                <v-checkbox
                  :model-value="selectedFeatures.includes(feature)"
                  hide-details
                  density="compact"
                  @click.stop="toggleFeature(feature)"
                />
              </template>
              <v-list-item-title>{{ feature }}</v-list-item-title>
              <template #append>
                <v-chip size="x-small" color="secondary" variant="flat">DSP</v-chip>
              </template>
            </v-list-item>
          </v-list>

          <!-- Selection Summary -->
          <v-alert type="info" variant="tonal" class="mt-4">
            <strong>{{ selectedFeatures.length }}</strong> features selected
            ({{ selectedFeatures.length * sensorColumns }} total with {{ sensorColumns }} sensor channels)
          </v-alert>
        </v-card>
      </v-col>

      <!-- LLM Assistant -->
      <v-col cols="12" md="4">
        <v-card class="llm-assistant pa-4">
          <div class="d-flex align-center mb-4">
            <v-icon color="secondary" class="mr-2">mdi-robot</v-icon>
            <h3 class="text-subtitle-1 font-weight-bold">LLM Assistant</h3>
            <v-spacer />
            <v-chip
              size="x-small"
              :color="llmStatus?.available ? 'success' : 'error'"
              variant="flat"
            >
              <v-icon
                size="x-small"
                :icon="llmStatus?.available ? 'mdi-check-circle' : 'mdi-alert-circle'"
                class="mr-1"
              />
              {{ llmStatus?.model || 'Ollama' }}
            </v-chip>
          </div>

          <!-- LLM Status Details -->
          <div v-if="llmStatus" class="mb-4">
            <div v-if="llmStatus.available" class="d-flex align-center mb-2">
              <v-icon
                size="small"
                :color="llmStatus.gpu_loaded ? 'success' : 'warning'"
                class="mr-2"
              >
                {{ llmStatus.gpu_loaded ? 'mdi-chip' : 'mdi-memory' }}
              </v-icon>
              <span class="text-caption">
                {{ llmStatus.gpu_loaded ? 'GPU Accelerated' : 'CPU Mode' }}
                <span v-if="llmStatus.gpu_info?.vram_used_mb" class="text-medium-emphasis">
                  ({{ llmStatus.gpu_info.vram_used_mb.toFixed(0) }} MB VRAM)
                </span>
              </span>
            </div>
            <div v-else class="text-caption text-error mb-2">
              {{ llmStatus.error || 'LLM service not available' }}
            </div>
          </div>

          <!-- Recommendations -->
          <div v-if="recommendations" class="assistant-message mb-4">
            <div class="d-flex align-center mb-2">
              <p class="mb-0">Based on your {{ pipelineStore.mode }} task, I recommend:</p>
              <v-chip
                v-if="recommendations.llm_used"
                size="x-small"
                color="secondary"
                variant="tonal"
                class="ml-2"
              >
                LLM
              </v-chip>
              <v-chip
                v-else
                size="x-small"
                color="grey"
                variant="tonal"
                class="ml-2"
              >
                Rule-based
              </v-chip>
            </div>
            <ul class="pl-4">
              <li v-for="feature in recommendations.recommended_features.slice(0, 8)" :key="feature">
                {{ feature }}
              </li>
            </ul>
          </div>

          <div v-if="recommendations?.reasoning" class="mb-4">
            <p
              v-for="(reason, index) in recommendations.reasoning"
              :key="index"
              class="text-caption text-medium-emphasis mb-1"
            >
              • {{ reason }}
            </p>
          </div>

          <v-btn
            color="secondary"
            block
            :loading="loadingRecommendations"
            :disabled="!llmStatus?.available && !pipelineStore.windowedSession"
            @click="getRecommendations"
          >
            <v-icon start>mdi-auto-fix</v-icon>
            Get Recommendations
          </v-btn>

          <v-btn
            v-if="recommendations"
            color="primary"
            variant="outlined"
            block
            class="mt-2"
            @click="applyRecommendations"
          >
            Apply Recommendations
          </v-btn>
        </v-card>

        <!-- Extraction Progress -->
        <v-card v-if="extractionResult" class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Extraction Complete</h3>

          <v-alert type="success" variant="tonal" class="mb-4">
            <strong>{{ extractionResult.num_features }}</strong> features extracted
            from <strong>{{ extractionResult.num_windows }}</strong> windows
          </v-alert>

          <div class="text-caption text-medium-emphasis">
            Session ID: {{ extractionResult.session_id }}
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Feature Visualization (shown after extraction) -->
    <v-row v-if="extractionResult && featurePreview" class="mt-4">
      <v-col cols="12">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4">
            <v-icon color="primary" class="mr-2">mdi-chart-histogram</v-icon>
            <h3 class="text-subtitle-1 font-weight-bold">Feature Visualization</h3>
          </div>

          <v-row>
            <!-- Feature Selector and Chart -->
            <v-col cols="12" md="8">
              <v-autocomplete
                v-model="selectedFeatureForViz"
                :items="extractedFeatureNames"
                label="Select Feature to Visualize"
                density="compact"
                hide-details
                class="mb-4"
                clearable
                auto-select-first
              >
                <template #item="{ item, props }">
                  <v-list-item v-bind="props">
                    <template #append>
                      <v-chip size="x-small" :color="getFeatureTypeColor(item.value)" variant="flat">
                        {{ getFeatureType(item.value) }}
                      </v-chip>
                    </template>
                  </v-list-item>
                </template>
              </v-autocomplete>

              <div v-if="loadingDistribution" class="d-flex justify-center align-center" style="height: 300px">
                <v-progress-circular indeterminate color="primary" />
              </div>

              <div v-else-if="featureDistribution" style="height: 300px">
                <Bar :data="distributionChartData" :options="distributionChartOptions" />
              </div>

              <div v-else class="d-flex justify-center align-center text-medium-emphasis" style="height: 300px">
                Select a feature to view its distribution
              </div>
            </v-col>

            <!-- Feature Statistics -->
            <v-col cols="12" md="4">
              <v-card variant="outlined" class="pa-4">
                <h4 class="text-subtitle-2 font-weight-bold mb-3">Feature Statistics</h4>

                <div v-if="featureDistribution?.statistics">
                  <div class="stat-row">
                    <span class="text-medium-emphasis">Mean:</span>
                    <span class="font-weight-medium">{{ featureDistribution.statistics.mean.toFixed(4) }}</span>
                  </div>
                  <div class="stat-row">
                    <span class="text-medium-emphasis">Std Dev:</span>
                    <span class="font-weight-medium">{{ featureDistribution.statistics.std.toFixed(4) }}</span>
                  </div>
                  <div class="stat-row">
                    <span class="text-medium-emphasis">Min:</span>
                    <span class="font-weight-medium">{{ featureDistribution.statistics.min.toFixed(4) }}</span>
                  </div>
                  <div class="stat-row">
                    <span class="text-medium-emphasis">Max:</span>
                    <span class="font-weight-medium">{{ featureDistribution.statistics.max.toFixed(4) }}</span>
                  </div>
                  <div class="stat-row">
                    <span class="text-medium-emphasis">Median:</span>
                    <span class="font-weight-medium">{{ featureDistribution.statistics.median.toFixed(4) }}</span>
                  </div>

                  <v-divider class="my-3" />

                  <div class="text-caption text-medium-emphasis">
                    Total samples: {{ featureDistribution.statistics.count }}
                  </div>
                </div>

                <div v-else class="text-caption text-medium-emphasis">
                  Select a feature to view statistics
                </div>
              </v-card>

              <!-- Label Distribution Summary -->
              <v-card v-if="featurePreview?.label_counts" variant="outlined" class="pa-4 mt-4">
                <h4 class="text-subtitle-2 font-weight-bold mb-3">Label Distribution</h4>
                <div v-for="(count, label) in featurePreview.label_counts" :key="label" class="stat-row">
                  <v-chip size="x-small" :color="getLabelColor(label as string)" class="mr-2">
                    {{ label }}
                  </v-chip>
                  <span class="font-weight-medium">{{ count }} samples</span>
                </div>
              </v-card>
            </v-col>
          </v-row>
        </v-card>
      </v-col>
    </v-row>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="router.push({ name: 'pipeline-windowing' })"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          color="secondary"
          size="large"
          class="mr-2"
          :loading="extracting"
          :disabled="selectedFeatures.length === 0"
          @click="extractFeatures"
        >
          Extract Features
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :disabled="!extractionResult"
          @click="router.push({ name: 'pipeline-training' })"
        >
          Continue to Training
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

const searchQuery = ref('')
const selectedFeatures = ref<string[]>([])
const recommendations = ref<any>(null)
const extractionResult = ref<any>(null)
const loadingRecommendations = ref(false)
const extracting = ref(false)

// Feature visualization state
const featurePreview = ref<any>(null)
const featureDistribution = ref<any>(null)
const selectedFeatureForViz = ref<string>('')
const loadingPreview = ref(false)
const loadingDistribution = ref(false)

// LLM status
const llmStatus = ref<any>(null)

// TSFresh statistical features (25 features)
const tsfreshFeatures = [
  'mean', 'std', 'min', 'max', 'median', 'sum', 'variance',
  'skewness', 'kurtosis', 'abs_energy', 'root_mean_square',
  'mean_abs_change', 'mean_change', 'count_above_mean', 'count_below_mean',
  'first_location_of_maximum', 'first_location_of_minimum',
  'last_location_of_maximum', 'last_location_of_minimum',
  'percentage_of_reoccurring_values', 'sum_of_reoccurring_values',
  'abs_sum_of_changes', 'range', 'interquartile_range', 'mean_second_derivative'
]

// Custom DSP features (22 features)
const dspFeatures = [
  'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
  'impulse_factor', 'margin_factor', 'zero_crossing_rate',
  'autocorr_lag1', 'autocorr_lag5', 'binned_entropy',
  'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
  'spectral_flatness', 'spectral_entropy', 'peak_frequency',
  'spectral_skewness', 'spectral_kurtosis',
  'band_power_low', 'band_power_mid', 'band_power_high'
]

const sensorColumns = computed(() =>
  pipelineStore.dataSession?.metadata?.sensor_columns?.length || 3
)

const filteredTSFreshFeatures = computed(() =>
  tsfreshFeatures.filter(f => f.toLowerCase().includes(searchQuery.value.toLowerCase()))
)

const filteredDSPFeatures = computed(() =>
  dspFeatures.filter(f => f.toLowerCase().includes(searchQuery.value.toLowerCase()))
)

// Available features for visualization (from extracted data)
const extractedFeatureNames = computed(() => {
  if (!featurePreview.value?.columns) return []
  return featurePreview.value.columns.filter((c: string) => c !== 'label')
})

// Distribution chart data
const distributionChartData = computed(() => {
  if (!featureDistribution.value) {
    return { labels: [], datasets: [] }
  }

  const dist = featureDistribution.value
  return {
    labels: dist.bin_edges.slice(0, -1).map((edge: number, i: number) => {
      const nextEdge = dist.bin_edges[i + 1]
      return `${edge.toFixed(2)}`
    }),
    datasets: [{
      label: selectedFeatureForViz.value,
      data: dist.counts,
      backgroundColor: 'rgba(99, 102, 241, 0.7)',
      borderColor: 'rgba(99, 102, 241, 1)',
      borderWidth: 1
    }]
  }
})

const distributionChartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    title: {
      display: true,
      text: `Distribution: ${selectedFeatureForViz.value}`
    }
  },
  scales: {
    x: {
      title: {
        display: true,
        text: 'Value'
      },
      ticks: {
        maxRotation: 45,
        minRotation: 45
      }
    },
    y: {
      title: {
        display: true,
        text: 'Count'
      },
      beginAtZero: true
    }
  }
}))

function toggleFeature(feature: string) {
  const index = selectedFeatures.value.indexOf(feature)
  if (index >= 0) {
    selectedFeatures.value.splice(index, 1)
  } else {
    selectedFeatures.value.push(feature)
  }
}

function selectAll() {
  selectedFeatures.value = [...tsfreshFeatures, ...dspFeatures]
}

function clearSelection() {
  selectedFeatures.value = []
}

function selectTSFresh() {
  selectedFeatures.value = [...tsfreshFeatures]
}

function selectDSP() {
  selectedFeatures.value = [...dspFeatures]
}

async function getRecommendations() {
  if (!pipelineStore.windowedSession) {
    notificationStore.showError('No windowed data available')
    return
  }

  try {
    loadingRecommendations.value = true

    const response = await api.post('/api/features/recommend', {
      session_id: pipelineStore.windowedSession.session_id,
      mode: pipelineStore.mode,
      use_llm: true
    })

    recommendations.value = response.data

    // Update LLM status if provided in response
    if (response.data.llm_status) {
      llmStatus.value = {
        ...llmStatus.value,
        ...response.data.llm_status
      }
    }

    const source = response.data.llm_used ? 'LLM' : 'rule-based'
    notificationStore.showSuccess(`Recommendations received (${source})`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to get recommendations')
  } finally {
    loadingRecommendations.value = false
  }
}

function applyRecommendations() {
  if (recommendations.value?.recommended_features) {
    selectedFeatures.value = recommendations.value.recommended_features.filter(
      (f: string) => tsfreshFeatures.includes(f) || dspFeatures.includes(f)
    )
    notificationStore.showSuccess('Recommendations applied')
  }
}

async function extractFeatures() {
  pipelineStore.selectedFeatures = selectedFeatures.value

  extracting.value = true

  const result = await pipelineStore.extractFeatures(selectedFeatures.value)

  if (result.success) {
    extractionResult.value = result.data
    notificationStore.showSuccess('Features extracted successfully')
    // Load feature preview after extraction
    await fetchFeaturePreview()
  } else {
    notificationStore.showError(result.error || 'Failed to extract features')
  }

  extracting.value = false
}

async function fetchFeaturePreview() {
  if (!extractionResult.value?.session_id) return

  try {
    loadingPreview.value = true
    const response = await api.post('/api/features/preview', {
      session_id: extractionResult.value.session_id,
      num_rows: 100
    })
    featurePreview.value = response.data

    // Auto-select first feature for visualization
    if (extractedFeatureNames.value.length > 0 && !selectedFeatureForViz.value) {
      selectedFeatureForViz.value = extractedFeatureNames.value[0]
    }
  } catch (e: any) {
    notificationStore.showError('Failed to load feature preview')
  } finally {
    loadingPreview.value = false
  }
}

async function fetchFeatureDistribution(featureName: string) {
  if (!extractionResult.value?.session_id || !featureName) return

  try {
    loadingDistribution.value = true
    const response = await api.post('/api/features/distribution', {
      session_id: extractionResult.value.session_id,
      feature_name: featureName,
      bins: 20
    })
    featureDistribution.value = response.data
  } catch (e: any) {
    notificationStore.showError('Failed to load feature distribution')
  } finally {
    loadingDistribution.value = false
  }
}

// Watch for feature selection changes to update distribution
watch(selectedFeatureForViz, (newFeature) => {
  if (newFeature) {
    fetchFeatureDistribution(newFeature)
  }
})

// Helper function for label colors
const labelColors = ['primary', 'secondary', 'success', 'warning', 'error', 'info']
function getLabelColor(label: string): string {
  const index = Object.keys(featurePreview.value?.label_counts || {}).indexOf(label)
  return labelColors[index % labelColors.length]
}

// Helper functions for feature type identification
function getFeatureType(featureName: string): string {
  const baseName = featureName.split('_').slice(0, -1).join('_') // Remove sensor suffix
  if (tsfreshFeatures.includes(baseName)) return 'TSFresh'
  if (dspFeatures.includes(baseName)) return 'DSP'
  return 'Other'
}

function getFeatureTypeColor(featureName: string): string {
  const type = getFeatureType(featureName)
  if (type === 'TSFresh') return 'info'
  if (type === 'DSP') return 'secondary'
  return 'default'
}

async function fetchLLMStatus() {
  try {
    const response = await api.get('/api/features/llm-status')
    llmStatus.value = response.data
  } catch (e: any) {
    llmStatus.value = {
      available: false,
      error: 'Failed to check LLM status'
    }
  }
}

onMounted(() => {
  // Default selection
  if (pipelineStore.selectedFeatures.length > 0) {
    selectedFeatures.value = [...pipelineStore.selectedFeatures]
  } else {
    selectedFeatures.value = ['mean', 'std', 'rms', 'kurtosis', 'spectral_entropy']
  }

  // Check LLM status
  fetchLLMStatus()
})
</script>

<style scoped lang="scss">
.feature-list {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;

  .v-list-item {
    &.selected {
      background: rgba(99, 102, 241, 0.1);
    }

    &:hover {
      background: rgba(var(--v-theme-surface-variant), 0.5);
    }
  }
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  border-bottom: 1px solid rgba(var(--v-border-color), 0.1);

  &:last-child {
    border-bottom: none;
  }
}
</style>
