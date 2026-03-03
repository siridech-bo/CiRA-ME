<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="deploy" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Deploy to Edge</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Select a model and deploy it to an edge device
    </p>

    <!-- Model Selection -->
    <v-card class="pa-4 mb-6">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">
        <v-icon start color="primary">mdi-brain</v-icon>
        Select Model
      </h3>

      <v-radio-group v-model="modelSource" hide-details>
        <!-- Current Session -->
        <v-radio
          value="session"
          :disabled="!pipelineStore.trainingSession"
        >
          <template #label>
            <div class="d-flex align-center flex-wrap ga-2" style="width: 100%">
              <span class="font-weight-medium">Current Training Session</span>
              <template v-if="pipelineStore.trainingSession">
                <v-chip size="x-small" color="primary" variant="tonal">
                  {{ pipelineStore.trainingSession.algorithm }}
                </v-chip>
                <v-chip size="x-small" color="info" variant="tonal">
                  {{ pipelineStore.trainingSession.mode }}
                </v-chip>
                <v-chip
                  v-if="pipelineStore.trainingSession.metrics?.accuracy != null"
                  size="x-small" color="success" variant="tonal"
                >
                  Acc: {{ (pipelineStore.trainingSession.metrics.accuracy * 100).toFixed(1) }}%
                </v-chip>
                <v-chip
                  v-if="pipelineStore.trainingSession.metrics?.f1 != null"
                  size="x-small" color="warning" variant="tonal"
                >
                  F1: {{ (pipelineStore.trainingSession.metrics.f1 * 100).toFixed(1) }}%
                </v-chip>
              </template>
              <span v-else class="text-caption text-medium-emphasis">(no model trained in this session)</span>
            </div>
          </template>
        </v-radio>

        <!-- Saved Models -->
        <v-radio value="saved">
          <template #label>
            <span class="font-weight-medium">Saved Benchmark Model</span>
          </template>
        </v-radio>
      </v-radio-group>

      <!-- Saved Models Table (shown when 'saved' is selected) -->
      <div v-if="modelSource === 'saved'" class="mt-4">
        <div v-if="loadingSavedModels" class="text-center pa-4">
          <v-progress-circular indeterminate size="24" />
          <span class="ml-2 text-medium-emphasis">Loading saved models...</span>
        </div>

        <v-alert v-else-if="savedModels.length === 0" type="info" variant="tonal">
          No saved models found. Save a benchmark from the Training page first.
        </v-alert>

        <v-table v-else dense hover>
          <thead>
            <tr>
              <th></th>
              <th>Name</th>
              <th>Algorithm</th>
              <th>Mode</th>
              <th>Accuracy</th>
              <th>F1</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="model in savedModels"
              :key="model.id"
              :class="{ 'bg-primary-darken-3': selectedSavedModelId === model.id }"
              style="cursor: pointer"
              @click="selectSavedModel(model)"
            >
              <td>
                <v-radio-group v-model="selectedSavedModelId" hide-details inline>
                  <v-radio :value="model.id" density="compact" hide-details />
                </v-radio-group>
              </td>
              <td class="font-weight-medium">{{ model.name }}</td>
              <td>{{ model.algorithm }}</td>
              <td>
                <v-chip size="x-small" :color="model.mode === 'anomaly' ? 'warning' : 'info'" variant="tonal">
                  {{ model.mode }}
                </v-chip>
              </td>
              <td>{{ model.metrics?.accuracy != null ? (model.metrics.accuracy * 100).toFixed(1) + '%' : '-' }}</td>
              <td>{{ model.metrics?.f1 != null ? (model.metrics.f1 * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-caption">{{ formatDate(model.created_at) }}</td>
            </tr>
          </tbody>
        </v-table>
      </div>

      <!-- Selected Model Summary -->
      <v-alert
        v-if="selectedModelSummary"
        type="success"
        variant="tonal"
        class="mt-4"
        density="compact"
      >
        <strong>Selected:</strong> {{ selectedModelSummary }}
      </v-alert>
    </v-card>

    <v-row>
      <!-- Target Device -->
      <v-col cols="12" md="6">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Target Device</h3>

          <v-radio-group v-model="targetDevice">
            <v-radio value="jetson_nano">
              <template #label>
                <div>
                  <div class="font-weight-medium">NVIDIA Jetson Nano</div>
                  <div class="text-caption text-medium-emphasis">
                    4GB RAM, Maxwell GPU
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="jetson_xavier">
              <template #label>
                <div>
                  <div class="font-weight-medium">NVIDIA Jetson Xavier NX</div>
                  <div class="text-caption text-medium-emphasis">
                    8GB RAM, Volta GPU
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="raspberry_pi">
              <template #label>
                <div>
                  <div class="font-weight-medium">Raspberry Pi 4</div>
                  <div class="text-caption text-medium-emphasis">
                    ARM Cortex-A72
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="custom_ssh">
              <template #label>
                <div>
                  <div class="font-weight-medium">Custom SSH Target</div>
                  <div class="text-caption text-medium-emphasis">
                    Any Linux device with SSH
                  </div>
                </div>
              </template>
            </v-radio>
          </v-radio-group>
        </v-card>

        <!-- Export Options -->
        <v-card class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Export Options</h3>

          <v-radio-group v-model="exportFormat">
            <v-radio value="onnx">
              <template #label>
                <div>
                  <div class="font-weight-medium">ONNX Runtime</div>
                  <div class="text-caption text-medium-emphasis">
                    Cross-platform, optimized inference
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="pickle">
              <template #label>
                <div>
                  <div class="font-weight-medium">Scikit-learn (Pickle)</div>
                  <div class="text-caption text-medium-emphasis">
                    Native Python format
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="joblib">
              <template #label>
                <div>
                  <div class="font-weight-medium">Joblib</div>
                  <div class="text-caption text-medium-emphasis">
                    Efficient for large arrays
                  </div>
                </div>
              </template>
            </v-radio>
          </v-radio-group>

          <v-divider class="my-4" />

          <v-checkbox
            v-model="includeScaler"
            label="Include feature scaler"
            hide-details
          />
          <v-checkbox
            v-model="includeInferenceScript"
            label="Include inference script"
            hide-details
          />
          <v-checkbox
            v-model="includeRequirements"
            label="Include requirements.txt"
            hide-details
          />
        </v-card>
      </v-col>

      <!-- SSH Configuration -->
      <v-col cols="12" md="6">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">SSH Configuration</h3>

          <v-text-field
            v-model="sshConfig.host"
            label="Host / IP Address"
            prepend-inner-icon="mdi-server"
            placeholder="192.168.1.100"
          />

          <v-text-field
            v-model="sshConfig.username"
            label="Username"
            prepend-inner-icon="mdi-account"
            placeholder="jetson"
          />

          <v-text-field
            v-model="sshConfig.password"
            label="Password"
            prepend-inner-icon="mdi-lock"
            :type="showPassword ? 'text' : 'password'"
            :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
            @click:append-inner="showPassword = !showPassword"
          />

          <v-text-field
            v-model.number="sshConfig.port"
            label="Port"
            prepend-inner-icon="mdi-ethernet"
            type="number"
          />

          <v-text-field
            v-model="sshConfig.remote_path"
            label="Remote Path"
            prepend-inner-icon="mdi-folder"
            placeholder="/home/jetson/models"
          />

          <v-btn
            color="info"
            variant="outlined"
            block
            :loading="testingConnection"
            @click="testConnection"
          >
            <v-icon start>mdi-connection</v-icon>
            Test Connection
          </v-btn>

          <v-alert
            v-if="connectionStatus"
            :type="connectionStatus.status === 'connected' ? 'success' : 'error'"
            variant="tonal"
            class="mt-4"
          >
            {{ connectionStatus.message }}
            <div v-if="connectionStatus.system_info" class="text-caption mt-2">
              {{ connectionStatus.system_info }}
            </div>
          </v-alert>
        </v-card>

        <!-- Deployment Progress -->
        <v-card v-if="deploymentResult" class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Deployment Progress</h3>

          <div class="deployment-progress">
            <div
              v-for="step in deploymentResult.steps"
              :key="step.step"
              class="step"
            >
              <div class="step-icon" :class="step.status">
                <v-icon size="14" v-if="step.status === 'completed'">mdi-check</v-icon>
                <v-icon size="14" v-else-if="step.status === 'failed'">mdi-close</v-icon>
                <v-progress-circular
                  v-else-if="step.status === 'in_progress'"
                  size="14"
                  width="2"
                  indeterminate
                />
              </div>
              <span class="text-capitalize">{{ step.step.replace('_', ' ') }}</span>
            </div>
          </div>

          <v-alert
            v-if="deploymentResult.status === 'completed'"
            type="success"
            variant="tonal"
            class="mt-4"
          >
            <strong>Deployment successful!</strong>
            <div class="text-caption">
              Model deployed to: {{ deploymentResult.remote_path }}
            </div>
          </v-alert>
        </v-card>
      </v-col>
    </v-row>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="router.push({ name: 'pipeline-training' })"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          color="secondary"
          size="large"
          class="mr-2"
          variant="outlined"
          @click="exportOnly"
          :loading="exporting"
          :disabled="!hasModelSelected"
        >
          <v-icon start>mdi-download</v-icon>
          Export Only
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :loading="deploying"
          :disabled="!canDeploy"
          @click="deploy"
        >
          <v-icon start>mdi-rocket-launch</v-icon>
          Deploy Now
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

// Model selection
const modelSource = ref<'session' | 'saved'>(pipelineStore.trainingSession ? 'session' : 'saved')
const savedModels = ref<any[]>([])
const selectedSavedModelId = ref<number | null>(null)
const loadingSavedModels = ref(false)

// Deploy config
const targetDevice = ref('jetson_nano')
const exportFormat = ref('onnx')
const includeScaler = ref(true)
const includeInferenceScript = ref(true)
const includeRequirements = ref(true)
const showPassword = ref(false)

const sshConfig = reactive({
  host: '',
  username: 'jetson',
  password: '',
  port: 22,
  remote_path: '/home/jetson/models'
})

const testingConnection = ref(false)
const connectionStatus = ref<any>(null)
const deploying = ref(false)
const exporting = ref(false)
const deploymentResult = ref<any>(null)

// Computed
const hasModelSelected = computed(() => {
  if (modelSource.value === 'session') return !!pipelineStore.trainingSession
  if (modelSource.value === 'saved') return !!selectedSavedModelId.value
  return false
})

const canDeploy = computed(() =>
  hasModelSelected.value && sshConfig.host && sshConfig.username
)

const selectedModelSummary = computed(() => {
  if (modelSource.value === 'session' && pipelineStore.trainingSession) {
    const s = pipelineStore.trainingSession
    const acc = s.metrics?.accuracy != null ? ` | Acc: ${(s.metrics.accuracy * 100).toFixed(1)}%` : ''
    return `${s.algorithm} (${s.mode})${acc}`
  }
  if (modelSource.value === 'saved' && selectedSavedModelId.value) {
    const model = savedModels.value.find(m => m.id === selectedSavedModelId.value)
    if (model) {
      const acc = model.metrics?.accuracy != null ? ` | Acc: ${(model.metrics.accuracy * 100).toFixed(1)}%` : ''
      return `${model.name} — ${model.algorithm} (${model.mode})${acc}`
    }
  }
  return null
})

// Functions
function selectSavedModel(model: any) {
  selectedSavedModelId.value = model.id
}

function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

async function loadSavedModels() {
  try {
    loadingSavedModels.value = true
    const response = await api.get('/api/training/saved-models')
    savedModels.value = response.data || []
  } catch (e: any) {
    console.error('Failed to load saved models:', e)
  } finally {
    loadingSavedModels.value = false
  }
}

async function testConnection() {
  try {
    testingConnection.value = true

    const response = await api.post('/api/deployment/test-connection', sshConfig)
    connectionStatus.value = response.data

    if (response.data.status === 'connected') {
      notificationStore.showSuccess('Connection successful!')
    }
  } catch (e: any) {
    connectionStatus.value = {
      status: 'failed',
      message: e.response?.data?.error || 'Connection failed'
    }
    notificationStore.showError('Connection failed')
  } finally {
    testingConnection.value = false
  }
}

async function exportOnly() {
  if (!hasModelSelected.value) {
    notificationStore.showError('No model selected')
    return
  }

  try {
    exporting.value = true

    let url: string
    if (modelSource.value === 'session' && pipelineStore.trainingSession) {
      url = `/api/training/export/${pipelineStore.trainingSession.training_session_id}`
    } else {
      url = `/api/training/export-saved/${selectedSavedModelId.value}`
    }

    const response = await api.post(url, { format: exportFormat.value })
    notificationStore.showSuccess(`Model exported as ${exportFormat.value}`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Export failed')
  } finally {
    exporting.value = false
  }
}

async function deploy() {
  if (!hasModelSelected.value) {
    notificationStore.showError('No model selected')
    return
  }

  try {
    deploying.value = true

    const payload: any = {
      target_type: targetDevice.value,
      export_format: exportFormat.value,
      ...sshConfig,
      include_scaler: includeScaler.value,
      include_inference_script: includeInferenceScript.value,
      include_requirements: includeRequirements.value
    }

    if (modelSource.value === 'session' && pipelineStore.trainingSession) {
      payload.training_session_id = pipelineStore.trainingSession.training_session_id
    } else {
      payload.saved_model_id = selectedSavedModelId.value
    }

    const response = await api.post('/api/deployment/deploy', payload)
    deploymentResult.value = response.data
    notificationStore.showSuccess('Deployment successful!')
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Deployment failed')
  } finally {
    deploying.value = false
  }
}

onMounted(() => {
  loadSavedModels()
})
</script>
