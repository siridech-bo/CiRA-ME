<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="deploy" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Deploy to Edge</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Deploy your trained model to an edge device
    </p>

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
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

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

const canDeploy = computed(() =>
  sshConfig.host && sshConfig.username && pipelineStore.trainingSession
)

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
  if (!pipelineStore.trainingSession) {
    notificationStore.showError('No trained model available')
    return
  }

  try {
    exporting.value = true

    const response = await api.post(
      `/api/training/export/${pipelineStore.trainingSession.training_session_id}`,
      { format: exportFormat.value }
    )

    notificationStore.showSuccess(`Model exported as ${exportFormat.value}`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Export failed')
  } finally {
    exporting.value = false
  }
}

async function deploy() {
  if (!pipelineStore.trainingSession) {
    notificationStore.showError('No trained model available')
    return
  }

  try {
    deploying.value = true

    const response = await api.post('/api/deployment/deploy', {
      training_session_id: pipelineStore.trainingSession.training_session_id,
      target_type: targetDevice.value,
      export_format: exportFormat.value,
      ...sshConfig,
      include_scaler: includeScaler.value,
      include_inference_script: includeInferenceScript.value,
      include_requirements: includeRequirements.value
    })

    deploymentResult.value = response.data
    notificationStore.showSuccess('Deployment successful!')
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Deployment failed')
  } finally {
    deploying.value = false
  }
}
</script>
