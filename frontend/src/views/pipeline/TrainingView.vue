<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="training" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Model Training</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Train a {{ pipelineStore.mode === 'anomaly' ? 'anomaly detection' : 'classification' }} model
    </p>

    <!-- ML vs DL Toggle -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center">
        <h3 class="text-subtitle-1 font-weight-bold mr-4">Training Approach</h3>
        <v-btn-toggle
          v-model="trainingApproach"
          mandatory
          color="primary"
          rounded="lg"
        >
          <v-btn value="ml" size="small">
            <v-icon start>mdi-chart-scatter-plot</v-icon>
            Traditional ML
          </v-btn>
          <v-btn value="dl" size="small">
            <v-icon start>mdi-brain</v-icon>
            Deep Learning (TimesNet)
          </v-btn>
        </v-btn-toggle>
      </div>

      <v-alert
        :type="trainingApproach === 'ml' ? 'info' : 'warning'"
        variant="tonal"
        density="compact"
        class="mt-4"
      >
        <template v-if="trainingApproach === 'ml'">
          <strong>Traditional ML:</strong> Uses extracted features (TSFresh + DSP) with PyOD/Scikit-learn algorithms.
          Best for interpretability and specific sensor metrics.
        </template>
        <template v-else>
          <strong>Deep Learning (TimesNet):</strong> End-to-end learning directly from windowed data.
          Best for complex, high-dimensional patterns. No manual feature extraction needed.
        </template>
      </v-alert>
    </v-card>

    <v-row>
      <!-- Traditional ML Section -->
      <template v-if="trainingApproach === 'ml'">
        <!-- Algorithm Selection -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <div class="d-flex align-center mb-4">
              <h3 class="text-subtitle-1 font-weight-bold">Algorithm</h3>
              <v-spacer />
              <v-chip
                :color="pipelineStore.mode === 'anomaly' ? 'error' : 'success'"
                size="small"
                variant="flat"
              >
                {{ pipelineStore.mode === 'anomaly' ? 'Anomaly Detection' : 'Classification' }}
              </v-chip>
            </div>

            <!-- Anomaly Algorithms -->
            <v-radio-group v-if="pipelineStore.mode === 'anomaly'" v-model="selectedAlgorithm">
              <v-radio
                v-for="algo in anomalyAlgorithms"
                :key="algo.id"
                :value="algo.id"
              >
                <template #label>
                  <div class="d-flex align-center">
                    <div>
                      <div class="font-weight-medium">{{ algo.name }}</div>
                      <div class="text-caption text-medium-emphasis">{{ algo.description }}</div>
                    </div>
                    <v-chip
                      v-if="algo.recommended"
                      size="x-small"
                      color="warning"
                      variant="flat"
                      class="ml-2"
                    >
                      Recommended
                    </v-chip>
                  </div>
                </template>
              </v-radio>
            </v-radio-group>

            <!-- Classification Algorithms -->
            <v-radio-group v-else v-model="selectedAlgorithm">
              <v-radio
                v-for="algo in classificationAlgorithms"
                :key="algo.id"
                :value="algo.id"
              >
                <template #label>
                  <div class="d-flex align-center">
                    <div>
                      <div class="font-weight-medium">{{ algo.name }}</div>
                      <div class="text-caption text-medium-emphasis">{{ algo.description }}</div>
                    </div>
                    <v-chip
                      v-if="algo.recommended"
                      size="x-small"
                      color="warning"
                      variant="flat"
                      class="ml-2"
                    >
                      Recommended
                    </v-chip>
                  </div>
                </template>
              </v-radio>
            </v-radio-group>
          </v-card>
        </v-col>

        <!-- ML Hyperparameters -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <h3 class="text-subtitle-1 font-weight-bold mb-4">Hyperparameters</h3>

            <template v-if="pipelineStore.mode === 'anomaly'">
              <v-text-field
                v-model.number="mlHyperparameters.n_estimators"
                label="Number of Estimators"
                type="number"
                :min="10"
                :max="500"
                hint="More estimators = better accuracy but slower"
              />

              <div class="mb-4">
                <div class="d-flex justify-space-between mb-2">
                  <span class="text-body-2">Contamination</span>
                  <span class="font-weight-medium">{{ mlHyperparameters.contamination }}</span>
                </div>
                <v-slider
                  v-model="mlHyperparameters.contamination"
                  :min="0.01"
                  :max="0.5"
                  :step="0.01"
                  color="error"
                  hide-details
                />
              </div>
            </template>

            <template v-else>
              <v-text-field
                v-model.number="mlHyperparameters.n_estimators"
                label="Number of Estimators"
                type="number"
                :min="10"
                :max="500"
              />

              <v-text-field
                v-model.number="mlHyperparameters.max_depth"
                label="Max Depth"
                type="number"
                :min="1"
                :max="50"
              />

              <div class="mb-4">
                <div class="d-flex justify-space-between mb-2">
                  <span class="text-body-2">Test Split</span>
                  <span class="font-weight-medium">{{ (mlHyperparameters.test_size * 100).toFixed(0) }}%</span>
                </div>
                <v-slider
                  v-model="mlHyperparameters.test_size"
                  :min="0.1"
                  :max="0.4"
                  :step="0.05"
                  color="info"
                  hide-details
                />
              </div>
            </template>
          </v-card>
        </v-col>
      </template>

      <!-- Deep Learning (TimesNet) Section -->
      <template v-else>
        <!-- GPU Status Card -->
        <v-col cols="12">
          <v-card class="pa-4 mb-4">
            <div class="d-flex align-center justify-space-between">
              <div class="d-flex align-center">
                <v-icon
                  :color="gpuStatus.available ? 'success' : 'warning'"
                  class="mr-2"
                >
                  {{ gpuStatus.available ? 'mdi-chip' : 'mdi-cpu-64-bit' }}
                </v-icon>
                <div>
                  <h3 class="text-subtitle-1 font-weight-bold">
                    {{ gpuStatus.available ? 'GPU Available' : 'GPU Not Available' }}
                  </h3>
                  <div class="text-caption text-medium-emphasis">
                    <template v-if="gpuStatus.available">
                      {{ gpuStatus.device_name }} -
                      {{ gpuStatus.memory_free_gb }}GB free / {{ gpuStatus.memory_total_gb }}GB total
                    </template>
                    <template v-else-if="gpuStatus.error">
                      {{ gpuStatus.error }}
                    </template>
                    <template v-else>
                      {{ gpuStatus.info || 'CUDA not available' }}
                    </template>
                  </div>
                </div>
              </div>

              <div class="d-flex align-center">
                <span class="text-body-2 mr-3">Training Device:</span>
                <v-btn-toggle
                  v-model="selectedDevice"
                  mandatory
                  color="primary"
                  density="compact"
                >
                  <v-btn value="cpu" size="small">
                    <v-icon start size="small">mdi-cpu-64-bit</v-icon>
                    CPU
                  </v-btn>
                  <v-btn
                    value="cuda"
                    size="small"
                    :disabled="!gpuStatus.available"
                  >
                    <v-icon start size="small">mdi-chip</v-icon>
                    GPU
                  </v-btn>
                </v-btn-toggle>
              </div>
            </div>

            <v-alert
              v-if="gpuStatus.warning"
              type="warning"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              {{ gpuStatus.warning }}
            </v-alert>

            <v-alert
              v-if="gpuStatus.dll_error"
              type="warning"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              <strong>PyTorch DLL Conflict:</strong> Another GPU application is running.
              TimesNet deep learning is unavailable. You can still train using traditional ML algorithms
              which don't require PyTorch.
            </v-alert>

            <v-alert
              v-if="!gpuStatus.torch_available && !gpuStatus.dll_error"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              {{ gpuStatus.info || 'PyTorch not available. Using fallback ML methods.' }}
            </v-alert>

            <v-alert
              v-if="selectedDevice === 'cpu' && gpuStatus.available"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              Training on CPU will be slower but avoids GPU memory conflicts with other applications.
            </v-alert>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <div class="d-flex align-center mb-4">
              <v-icon color="secondary" class="mr-2">mdi-brain</v-icon>
              <h3 class="text-subtitle-1 font-weight-bold">TimesNet Configuration</h3>
            </div>

            <v-alert type="info" variant="tonal" density="compact" class="mb-4">
              TimesNet learns directly from raw windowed data using multi-periodic analysis.
              No feature extraction required!
            </v-alert>

            <!-- Model Architecture -->
            <v-text-field
              v-model.number="timesnetConfig.d_model"
              label="Model Dimension (d_model)"
              type="number"
              :min="32"
              :max="256"
              hint="Embedding dimension"
            />

            <v-text-field
              v-model.number="timesnetConfig.d_ff"
              label="Feed-Forward Dimension (d_ff)"
              type="number"
              :min="64"
              :max="512"
              hint="Hidden layer dimension"
            />

            <v-text-field
              v-model.number="timesnetConfig.e_layers"
              label="Encoder Layers"
              type="number"
              :min="1"
              :max="6"
              hint="Number of transformer layers"
            />

            <div class="mb-4">
              <div class="d-flex justify-space-between mb-2">
                <span class="text-body-2">Dropout</span>
                <span class="font-weight-medium">{{ timesnetConfig.dropout }}</span>
              </div>
              <v-slider
                v-model="timesnetConfig.dropout"
                :min="0"
                :max="0.5"
                :step="0.05"
                color="secondary"
                hide-details
              />
            </div>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <h3 class="text-subtitle-1 font-weight-bold mb-4">Period Configuration</h3>

            <p class="text-body-2 text-medium-emphasis mb-4">
              TimesNet analyzes data at multiple periods to capture intraperiod and interperiod variations.
            </p>

            <!-- Period Selection -->
            <v-chip-group
              v-model="selectedPeriods"
              multiple
              column
              selected-class="text-primary"
            >
              <v-chip
                v-for="period in availablePeriods"
                :key="period"
                :value="period"
                filter
                variant="outlined"
              >
                {{ period }}
              </v-chip>
            </v-chip-group>

            <v-text-field
              v-model.number="timesnetConfig.top_k"
              label="Top-K Periods"
              type="number"
              :min="1"
              :max="selectedPeriods.length || 5"
              hint="Number of dominant periods to use"
              class="mt-4"
            />

            <v-divider class="my-4" />

            <!-- Training Parameters -->
            <h4 class="text-subtitle-2 font-weight-bold mb-3">Training Parameters</h4>

            <v-text-field
              v-model.number="timesnetConfig.epochs"
              label="Epochs"
              type="number"
              :min="10"
              :max="500"
            />

            <v-text-field
              v-model.number="timesnetConfig.batch_size"
              label="Batch Size"
              type="number"
              :min="8"
              :max="128"
            />

            <v-text-field
              v-model.number="timesnetConfig.learning_rate"
              label="Learning Rate"
              type="number"
              :min="0.0001"
              :max="0.1"
              :step="0.0001"
            />
          </v-card>
        </v-col>
      </template>
    </v-row>

    <!-- Training Results -->
    <v-card v-if="trainingResult" class="pa-4 mt-6">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">Training Complete</h3>

      <v-alert type="success" variant="tonal" class="mb-4">
        Model trained successfully using
        <strong>{{ trainingResult.algorithm }}</strong>
        <template v-if="trainingApproach === 'dl'"> (Deep Learning)</template>
      </v-alert>

      <!-- Split method info -->
      <v-alert
        v-if="trainingResult.metrics.split_method || trainingResult.metrics.train_samples"
        type="info"
        variant="tonal"
        density="compact"
        class="mb-4"
      >
        <v-icon size="small" class="mr-1">mdi-chart-pie</v-icon>
        <template v-if="trainingResult.metrics.split_method === 'category'">
          Using dataset's built-in train/test split
        </template>
        <template v-else>
          Train/Test split
        </template>
        <span v-if="trainingResult.metrics.train_samples">
          ({{ trainingResult.metrics.train_samples }} train / {{ trainingResult.metrics.test_samples || 0 }} test)
        </span>
      </v-alert>

      <!-- Metrics info message (when labels are missing/incomplete) -->
      <v-alert
        v-if="trainingResult.metrics.metrics_info"
        type="warning"
        variant="tonal"
        density="compact"
        class="mb-4"
      >
        <v-icon size="small" class="mr-1">mdi-alert-circle</v-icon>
        {{ trainingResult.metrics.metrics_info }}
      </v-alert>

      <!-- Primary Metrics Row -->
      <v-row dense>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Accuracy</div>
            <div class="text-h5 text-success">
              {{ ((trainingResult.metrics.accuracy || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">F1 Score</div>
            <div class="text-h5 text-info">
              {{ ((trainingResult.metrics.f1 || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Precision</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.precision || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Recall</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.recall || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2" v-if="trainingResult.metrics.roc_auc">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">ROC-AUC</div>
            <div class="text-h6 text-purple">
              {{ (trainingResult.metrics.roc_auc || 0).toFixed(3) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2" v-if="trainingResult.metrics.specificity !== undefined">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Specificity</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.specificity || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Evaluation Charts Row -->
      <v-row class="mt-4">
        <!-- Confusion Matrix -->
        <v-col cols="12" md="6" v-if="trainingResult.metrics.confusion_matrix">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-grid</v-icon>
              Confusion Matrix
            </h4>
            <div class="confusion-matrix-container">
              <table class="confusion-matrix-table">
                <thead>
                  <tr>
                    <th></th>
                    <th v-for="(label, j) in confusionMatrixLabels" :key="'h'+j" class="header-cell">
                      {{ label }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, i) in trainingResult.metrics.confusion_matrix" :key="'r'+i">
                    <th class="row-header">{{ confusionMatrixLabels[i] || 'Class ' + i }}</th>
                    <td
                      v-for="(cell, j) in row"
                      :key="'c'+j"
                      :class="getCellClass(i, j, cell)"
                      :style="getCellStyle(cell)"
                    >
                      <div class="cell-value">{{ cell }}</div>
                      <div class="cell-percent">{{ getCellPercent(i, j, cell) }}</div>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div class="matrix-legend mt-2">
                <span class="legend-item">
                  <span class="legend-color diagonal"></span> Correct (TP/TN)
                </span>
                <span class="legend-item">
                  <span class="legend-color off-diagonal"></span> Errors (FP/FN)
                </span>
              </div>
            </div>
          </v-card>
        </v-col>

        <!-- ROC Curve -->
        <v-col cols="12" md="6" v-if="trainingResult.metrics.roc_curve">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-chart-line</v-icon>
              ROC Curve (AUC = {{ (trainingResult.metrics.roc_auc || 0).toFixed(3) }})
            </h4>
            <div class="roc-chart-container">
              <svg viewBox="0 0 300 300" class="roc-chart">
                <!-- Grid lines -->
                <g class="grid-lines">
                  <line v-for="i in 5" :key="'hg'+i" :x1="50" :x2="290" :y1="50 + (i-1)*50" :y2="50 + (i-1)*50" />
                  <line v-for="i in 5" :key="'vg'+i" :x1="50 + (i-1)*60" :x2="50 + (i-1)*60" :y1="10" :y2="250" />
                </g>

                <!-- Diagonal line (random classifier) -->
                <line x1="50" y1="250" x2="290" y2="10" class="diagonal-line" />

                <!-- ROC curve -->
                <polyline
                  :points="rocCurvePoints"
                  class="roc-line"
                  fill="none"
                />

                <!-- Fill area under curve -->
                <polygon
                  :points="rocAreaPoints"
                  class="roc-area"
                />

                <!-- Axes -->
                <line x1="50" y1="250" x2="290" y2="250" class="axis" />
                <line x1="50" y1="10" x2="50" y2="250" class="axis" />

                <!-- Labels -->
                <text x="170" y="280" class="axis-label">False Positive Rate</text>
                <text x="15" y="130" class="axis-label" transform="rotate(-90, 15, 130)">True Positive Rate</text>

                <!-- Tick labels -->
                <text x="50" y="265" class="tick-label">0</text>
                <text x="170" y="265" class="tick-label">0.5</text>
                <text x="285" y="265" class="tick-label">1</text>
                <text x="40" y="255" class="tick-label">0</text>
                <text x="40" y="135" class="tick-label">0.5</text>
                <text x="40" y="15" class="tick-label">1</text>
              </svg>
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Detailed Stats for Anomaly Detection -->
      <v-row class="mt-2" v-if="trainingResult.metrics.true_positives !== undefined">
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="success" class="pa-2 text-center">
            <div class="text-caption">True Positives</div>
            <div class="text-h6">{{ trainingResult.metrics.true_positives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="success" class="pa-2 text-center">
            <div class="text-caption">True Negatives</div>
            <div class="text-h6">{{ trainingResult.metrics.true_negatives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="error" class="pa-2 text-center">
            <div class="text-caption">False Positives</div>
            <div class="text-h6">{{ trainingResult.metrics.false_positives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="error" class="pa-2 text-center">
            <div class="text-caption">False Negatives</div>
            <div class="text-h6">{{ trainingResult.metrics.false_negatives }}</div>
          </v-card>
        </v-col>
      </v-row>

      <!-- TimesNet specific info -->
      <div v-if="trainingApproach === 'dl' && trainingResult.config" class="mt-4">
        <h4 class="text-subtitle-2 mb-2">Model Configuration</h4>
        <v-chip size="small" class="mr-2 mb-2">d_model: {{ trainingResult.config.d_model }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">layers: {{ trainingResult.config.e_layers }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">top_k: {{ trainingResult.config.top_k }}</v-chip>
        <v-chip
          v-if="trainingResult.device"
          size="small"
          :color="trainingResult.device === 'cuda' ? 'success' : 'info'"
          class="mr-2 mb-2"
        >
          <v-icon start size="small">{{ trainingResult.device === 'cuda' ? 'mdi-chip' : 'mdi-cpu-64-bit' }}</v-icon>
          {{ trainingResult.device === 'cuda' ? 'GPU' : 'CPU' }}
        </v-chip>
      </div>
    </v-card>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="goBack"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          color="secondary"
          size="large"
          class="mr-2"
          :loading="training"
          :disabled="!canTrain"
          @click="trainModel"
        >
          <v-icon start>mdi-play</v-icon>
          Train Model
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :disabled="!trainingResult"
          @click="router.push({ name: 'pipeline-deploy' })"
        >
          Continue to Deploy
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

// Training approach synced with pipeline store
const trainingApproach = computed({
  get: () => pipelineStore.trainingApproach,
  set: (val) => pipelineStore.setTrainingApproach(val as 'ml' | 'dl')
})

// ML state
const selectedAlgorithm = ref('')
const training = ref(false)
const trainingResult = ref<any>(null)

const mlHyperparameters = reactive({
  n_estimators: 100,
  contamination: 0.1,
  max_depth: null as number | null,
  test_size: 0.2
})

// TimesNet state
const timesnetConfig = reactive({
  d_model: 64,
  d_ff: 128,
  e_layers: 2,
  dropout: 0.1,
  top_k: 3,
  epochs: 50,
  batch_size: 32,
  learning_rate: 0.001
})

// GPU status
const gpuStatus = reactive({
  available: false,
  cuda_available: false,
  torch_available: false,
  dll_error: false,
  device_name: null as string | null,
  memory_total_gb: null as number | null,
  memory_used_gb: null as number | null,
  memory_free_gb: null as number | null,
  error: null as string | null,
  warning: null as string | null,
  info: null as string | null,
  recommendation: 'cpu' as 'cpu' | 'cuda'
})

const selectedDevice = ref<'cpu' | 'cuda'>('cpu')

const availablePeriods = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
const selectedPeriods = ref([8, 16, 32, 64])

const anomalyAlgorithms = [
  { id: 'iforest', name: 'Isolation Forest', description: 'Tree-based anomaly detection', recommended: true },
  { id: 'lof', name: 'Local Outlier Factor', description: 'Density-based local anomalies' },
  { id: 'ocsvm', name: 'One-Class SVM', description: 'Support vector method' },
  { id: 'hbos', name: 'HBOS', description: 'Histogram-based outlier score' },
  { id: 'knn', name: 'KNN', description: 'K-nearest neighbors based' },
  { id: 'copod', name: 'COPOD', description: 'Copula-based outlier detection' },
  { id: 'ecod', name: 'ECOD', description: 'Empirical cumulative distribution' },
  { id: 'autoencoder', name: 'AutoEncoder', description: 'Neural network reconstruction' },
]

const classificationAlgorithms = [
  { id: 'rf', name: 'Random Forest', description: 'Ensemble of decision trees', recommended: true },
  { id: 'gb', name: 'Gradient Boosting', description: 'Sequential ensemble method' },
  { id: 'svm', name: 'Support Vector Machine', description: 'Margin-based classification' },
  { id: 'mlp', name: 'Multi-Layer Perceptron', description: 'Neural network classifier' },
  { id: 'knn', name: 'K-Nearest Neighbors', description: 'Instance-based learning' },
  { id: 'dt', name: 'Decision Tree', description: 'Single tree classifier' },
  { id: 'nb', name: 'Naive Bayes', description: 'Probabilistic classifier' },
  { id: 'lr', name: 'Logistic Regression', description: 'Linear classifier' },
]

const canTrain = computed(() => {
  if (trainingApproach.value === 'ml') {
    return !!selectedAlgorithm.value && !!pipelineStore.featureSession
  } else {
    return !!pipelineStore.windowedSession
  }
})

// Confusion matrix helpers
const confusionMatrixLabels = computed(() => {
  if (!trainingResult.value?.metrics) return ['Class 0', 'Class 1']
  return trainingResult.value.metrics.confusion_matrix_labels ||
         trainingResult.value.metrics.class_names ||
         ['Normal', 'Anomaly']
})

const confusionMatrixTotal = computed(() => {
  if (!trainingResult.value?.metrics?.confusion_matrix) return 0
  return trainingResult.value.metrics.confusion_matrix.flat().reduce((a: number, b: number) => a + b, 0)
})

function getCellClass(i: number, j: number, value: number): string {
  const classes = ['matrix-cell']
  if (i === j) {
    classes.push('diagonal')
  } else {
    classes.push('off-diagonal')
  }
  return classes.join(' ')
}

function getCellStyle(value: number): Record<string, string> {
  const total = confusionMatrixTotal.value
  if (total === 0) return {}
  const intensity = Math.min(value / total * 3, 1) // Scale for visibility
  return {
    '--cell-intensity': intensity.toString()
  }
}

function getCellPercent(i: number, j: number, value: number): string {
  const total = confusionMatrixTotal.value
  if (total === 0) return ''
  return `${((value / total) * 100).toFixed(1)}%`
}

// ROC curve helpers
const rocCurvePoints = computed(() => {
  if (!trainingResult.value?.metrics?.roc_curve) return ''
  const roc = trainingResult.value.metrics.roc_curve
  return roc.fpr.map((fpr: number, i: number) => {
    const x = 50 + fpr * 240  // Scale to SVG coordinates
    const y = 250 - roc.tpr[i] * 240
    return `${x},${y}`
  }).join(' ')
})

const rocAreaPoints = computed(() => {
  if (!trainingResult.value?.metrics?.roc_curve) return ''
  const roc = trainingResult.value.metrics.roc_curve
  const points = roc.fpr.map((fpr: number, i: number) => {
    const x = 50 + fpr * 240
    const y = 250 - roc.tpr[i] * 240
    return `${x},${y}`
  })
  // Close the polygon
  points.push('290,250')  // Bottom right
  points.push('50,250')   // Bottom left
  return points.join(' ')
})

function goBack() {
  if (trainingApproach.value === 'ml') {
    router.push({ name: 'pipeline-features' })
  } else {
    // DL skips features, go back to windowing
    router.push({ name: 'pipeline-windowing' })
  }
}

async function trainModel() {
  training.value = true
  trainingResult.value = null

  try {
    if (trainingApproach.value === 'ml') {
      await trainMLModel()
    } else {
      await trainTimesNetModel()
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Training failed')
  } finally {
    training.value = false
  }
}

async function trainMLModel() {
  if (!pipelineStore.featureSession) {
    notificationStore.showError('No features extracted. Please go back and extract features first.')
    return
  }

  pipelineStore.selectedAlgorithm = selectedAlgorithm.value
  pipelineStore.hyperparameters = { ...mlHyperparameters }

  const result = await pipelineStore.trainModel()

  if (result.success) {
    trainingResult.value = result.data
    notificationStore.showSuccess('Model trained successfully!')
  } else {
    notificationStore.showError(result.error || 'Training failed')
  }
}

async function trainTimesNetModel() {
  if (!pipelineStore.windowedSession) {
    notificationStore.showError('No windowed data. Please go back and apply windowing first.')
    return
  }

  const endpoint = pipelineStore.mode === 'anomaly'
    ? '/api/training/timesnet/train/anomaly'
    : '/api/training/timesnet/train/classification'

  const response = await api.post(endpoint, {
    windowed_session_id: pipelineStore.windowedSession.session_id,
    config: {
      d_model: timesnetConfig.d_model,
      d_ff: timesnetConfig.d_ff,
      e_layers: timesnetConfig.e_layers,
      dropout: timesnetConfig.dropout,
      top_k: timesnetConfig.top_k,
      period_list: selectedPeriods.value
    },
    epochs: timesnetConfig.epochs,
    batch_size: timesnetConfig.batch_size,
    learning_rate: timesnetConfig.learning_rate,
    test_size: mlHyperparameters.test_size,
    device: selectedDevice.value  // CPU or CUDA
  })

  trainingResult.value = response.data
  pipelineStore.trainingSession = response.data
  notificationStore.showSuccess('TimesNet model trained successfully!')
}

// Fetch GPU status
async function fetchGpuStatus() {
  try {
    const response = await api.get('/api/training/gpu-status')
    Object.assign(gpuStatus, response.data)
    // Set recommended device
    selectedDevice.value = gpuStatus.recommendation
  } catch (e: any) {
    gpuStatus.error = 'Failed to check GPU status'
    gpuStatus.available = false
    selectedDevice.value = 'cpu'
  }
}

// Watch for approach changes
watch(trainingApproach, (newVal) => {
  trainingResult.value = null
  // Fetch GPU status when switching to deep learning
  if (newVal === 'dl') {
    fetchGpuStatus()
  }
})

onMounted(() => {
  if (pipelineStore.mode === 'anomaly') {
    selectedAlgorithm.value = 'iforest'
  } else {
    selectedAlgorithm.value = 'rf'
  }

  // Fetch GPU status if starting with deep learning approach
  if (trainingApproach.value === 'dl') {
    fetchGpuStatus()
  }
})
</script>

<style scoped lang="scss">
// Confusion Matrix Styles
.confusion-matrix-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.confusion-matrix-table {
  border-collapse: collapse;
  width: 100%;
  max-width: 400px;

  th, td {
    border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
    padding: 8px 12px;
    text-align: center;
  }

  th {
    background: rgba(var(--v-theme-surface-variant), 0.5);
    font-weight: 600;
    font-size: 0.85rem;
  }

  .header-cell {
    min-width: 80px;
  }

  .row-header {
    text-align: right;
    font-weight: 600;
    background: rgba(var(--v-theme-surface-variant), 0.3);
  }

  .matrix-cell {
    position: relative;
    min-width: 80px;
    min-height: 60px;

    .cell-value {
      font-size: 1.25rem;
      font-weight: 700;
    }

    .cell-percent {
      font-size: 0.75rem;
      opacity: 0.7;
    }

    &.diagonal {
      background: rgba(16, 185, 129, calc(0.15 + var(--cell-intensity, 0) * 0.4));
      color: #10B981;
    }

    &.off-diagonal {
      background: rgba(239, 68, 68, calc(0.1 + var(--cell-intensity, 0) * 0.3));
      color: #EF4444;
    }
  }
}

.matrix-legend {
  display: flex;
  gap: 16px;
  font-size: 0.8rem;

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;

    &.diagonal {
      background: rgba(16, 185, 129, 0.4);
    }

    &.off-diagonal {
      background: rgba(239, 68, 68, 0.3);
    }
  }
}

// ROC Curve Styles
.roc-chart-container {
  display: flex;
  justify-content: center;
}

.roc-chart {
  width: 100%;
  max-width: 350px;
  height: auto;

  .grid-lines line {
    stroke: rgba(var(--v-border-color), 0.2);
    stroke-width: 0.5;
  }

  .diagonal-line {
    stroke: rgba(var(--v-theme-on-surface), 0.3);
    stroke-width: 1;
    stroke-dasharray: 5, 5;
  }

  .roc-line {
    stroke: #8B5CF6;
    stroke-width: 2.5;
  }

  .roc-area {
    fill: rgba(139, 92, 246, 0.15);
  }

  .axis {
    stroke: rgba(var(--v-theme-on-surface), 0.5);
    stroke-width: 1.5;
  }

  .axis-label {
    font-size: 11px;
    fill: rgba(var(--v-theme-on-surface), 0.7);
    text-anchor: middle;
  }

  .tick-label {
    font-size: 9px;
    fill: rgba(var(--v-theme-on-surface), 0.6);
    text-anchor: middle;
  }
}

// Legacy confusion matrix (keep for backwards compatibility)
.confusion-matrix {
  table {
    border-collapse: collapse;
    width: 100%;
    max-width: 200px;

    td {
      border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
      padding: 12px;
      text-align: center;
      font-weight: 500;

      &.diagonal {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
      }
    }
  }
}
</style>
