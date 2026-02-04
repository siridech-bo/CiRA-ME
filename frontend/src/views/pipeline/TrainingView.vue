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
        v-if="trainingResult.metrics.split_method"
        :type="trainingResult.metrics.split_method === 'category' ? 'info' : 'warning'"
        variant="tonal"
        density="compact"
        class="mb-4"
      >
        <template v-if="trainingResult.metrics.split_method === 'category'">
          <v-icon size="small" class="mr-1">mdi-shield-check</v-icon>
          Using dataset's built-in train/test split
          ({{ trainingResult.metrics.train_samples }} train / {{ trainingResult.metrics.test_samples }} test)
        </template>
        <template v-else>
          <v-icon size="small" class="mr-1">mdi-shuffle-variant</v-icon>
          Random 80/20 split (no category column detected)
          ({{ trainingResult.metrics.train_samples }} train / {{ trainingResult.metrics.test_samples }} test)
        </template>
      </v-alert>

      <!-- Metrics -->
      <v-row dense>
        <v-col cols="6" md="3">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Accuracy</div>
            <div class="text-h5 text-success">
              {{ ((trainingResult.metrics.accuracy || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">F1 Score</div>
            <div class="text-h5 text-info">
              {{ ((trainingResult.metrics.f1 || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Precision</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.precision || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Recall</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.recall || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Confusion Matrix -->
      <div v-if="trainingResult.metrics.confusion_matrix" class="mt-4">
        <h4 class="text-subtitle-2 mb-2">Confusion Matrix</h4>
        <div class="confusion-matrix">
          <table>
            <tr v-for="(row, i) in trainingResult.metrics.confusion_matrix" :key="i">
              <td
                v-for="(cell, j) in row"
                :key="j"
                :class="{ 'diagonal': i === j }"
              >
                {{ cell }}
              </td>
            </tr>
          </table>
        </div>
      </div>

      <!-- TimesNet specific info -->
      <div v-if="trainingApproach === 'dl' && trainingResult.config" class="mt-4">
        <h4 class="text-subtitle-2 mb-2">Model Configuration</h4>
        <v-chip size="small" class="mr-2 mb-2">d_model: {{ trainingResult.config.d_model }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">layers: {{ trainingResult.config.e_layers }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">top_k: {{ trainingResult.config.top_k }}</v-chip>
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
    test_size: mlHyperparameters.test_size
  })

  trainingResult.value = response.data
  pipelineStore.trainingSession = response.data
  notificationStore.showSuccess('TimesNet model trained successfully!')
}

// Watch for approach changes
watch(trainingApproach, (newVal) => {
  trainingResult.value = null
})

onMounted(() => {
  if (pipelineStore.mode === 'anomaly') {
    selectedAlgorithm.value = 'iforest'
  } else {
    selectedAlgorithm.value = 'rf'
  }
})
</script>

<style scoped lang="scss">
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
