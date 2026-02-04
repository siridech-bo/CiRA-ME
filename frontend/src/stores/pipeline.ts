/**
 * CiRA ME - Pipeline Store
 * Manages ML pipeline state across views
 * Supports both Traditional ML (feature-based) and Deep Learning (TimesNet) workflows
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/services/api'

export type PipelineMode = 'anomaly' | 'classification'
export type PipelineStep = 'data' | 'windowing' | 'features' | 'training' | 'deploy'
export type TrainingApproach = 'ml' | 'dl'

interface DataSession {
  session_id: string
  metadata: {
    format: string
    file_path: string
    total_rows: number
    columns: string[]
    sensor_columns: string[]
    label_column: string | null
    labels: string[] | null
  }
}

interface WindowingConfig {
  window_size: number
  stride: number
  label_method: 'majority' | 'first' | 'last' | 'threshold'
}

interface WindowedSession {
  session_id: string
  num_windows: number
  window_shape: [number, number]
}

interface FeatureSession {
  session_id: string
  num_features: number
  feature_names: string[]
}

interface TrainingSession {
  training_session_id: string
  algorithm: string
  mode: string
  metrics: Record<string, any>
  model_path: string
}

export const usePipelineStore = defineStore('pipeline', () => {
  // Mode
  const mode = ref<PipelineMode>('anomaly')

  // Training approach: 'ml' (feature-based) or 'dl' (TimesNet end-to-end)
  const trainingApproach = ref<TrainingApproach>('ml')

  // Current step
  const currentStep = ref<PipelineStep>('data')

  // Data session
  const dataSession = ref<DataSession | null>(null)

  // Windowing
  const windowingConfig = ref<WindowingConfig>({
    window_size: 128,
    stride: 64,
    label_method: 'majority'
  })
  const windowedSession = ref<WindowedSession | null>(null)

  // Features
  const selectedFeatures = ref<string[]>([])
  const featureSession = ref<FeatureSession | null>(null)

  // Training
  const selectedAlgorithm = ref<string>('')
  const hyperparameters = ref<Record<string, any>>({})
  const trainingSession = ref<TrainingSession | null>(null)

  // Loading states
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Computed
  const canProceedToWindowing = computed(() => !!dataSession.value)
  const canProceedToFeatures = computed(() => !!windowedSession.value)

  // For DL mode, we can proceed to training directly from windowing (skip features)
  const canProceedToTraining = computed(() => {
    if (trainingApproach.value === 'dl') {
      return !!windowedSession.value
    }
    return !!featureSession.value
  })

  const canProceedToDeploy = computed(() => !!trainingSession.value)

  // Step status adapts based on training approach
  const stepStatus = computed(() => {
    if (trainingApproach.value === 'dl') {
      // DL workflow: Data → Windowing → Training → Deploy (skip Features)
      return {
        data: dataSession.value ? 'complete' : 'current',
        windowing: windowedSession.value ? 'complete' : (canProceedToWindowing.value ? 'current' : 'disabled'),
        features: 'skipped', // Features are skipped for DL
        training: trainingSession.value ? 'complete' : (canProceedToTraining.value ? 'current' : 'disabled'),
        deploy: canProceedToDeploy.value ? 'current' : 'disabled'
      }
    }
    // ML workflow: Data → Windowing → Features → Training → Deploy
    return {
      data: dataSession.value ? 'complete' : 'current',
      windowing: windowedSession.value ? 'complete' : (canProceedToWindowing.value ? 'current' : 'disabled'),
      features: featureSession.value ? 'complete' : (canProceedToFeatures.value ? 'current' : 'disabled'),
      training: trainingSession.value ? 'complete' : (canProceedToTraining.value ? 'current' : 'disabled'),
      deploy: canProceedToDeploy.value ? 'current' : 'disabled'
    }
  })

  // Actions
  async function loadData(filePath: string, format: string) {
    try {
      loading.value = true
      error.value = null

      const endpoint = `/api/data/ingest/${format}`
      const response = await api.post(endpoint, { file_path: filePath })

      dataSession.value = response.data
      currentStep.value = 'windowing'

      return { success: true, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to load data'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  async function applyWindowing() {
    if (!dataSession.value) {
      return { success: false, error: 'No data loaded' }
    }

    try {
      loading.value = true
      error.value = null

      const response = await api.post('/api/data/windowing', {
        session_id: dataSession.value.session_id,
        ...windowingConfig.value
      })

      windowedSession.value = response.data
      currentStep.value = 'features'

      return { success: true, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to apply windowing'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  async function extractFeatures(features?: string[]) {
    if (!windowedSession.value) {
      return { success: false, error: 'No windowed data' }
    }

    try {
      loading.value = true
      error.value = null

      const response = await api.post('/api/features/extract', {
        session_id: windowedSession.value.session_id,
        features: features || selectedFeatures.value.length > 0 ? selectedFeatures.value : null
      })

      featureSession.value = response.data
      currentStep.value = 'training'

      return { success: true, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to extract features'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  async function trainModel() {
    if (!featureSession.value) {
      return { success: false, error: 'No features extracted' }
    }

    try {
      loading.value = true
      error.value = null

      const endpoint = mode.value === 'anomaly'
        ? '/api/training/train/anomaly'
        : '/api/training/train/classification'

      const response = await api.post(endpoint, {
        feature_session_id: featureSession.value.session_id,
        algorithm: selectedAlgorithm.value,
        hyperparameters: hyperparameters.value
      })

      trainingSession.value = response.data
      currentStep.value = 'deploy'

      return { success: true, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to train model'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  async function trainTimesNet(config: {
    d_model: number
    d_ff: number
    e_layers: number
    dropout: number
    top_k: number
    period_list: number[]
    epochs: number
    batch_size: number
    learning_rate: number
    test_size?: number
  }) {
    if (!windowedSession.value) {
      return { success: false, error: 'No windowed data available' }
    }

    try {
      loading.value = true
      error.value = null

      const endpoint = mode.value === 'anomaly'
        ? '/api/training/timesnet/train/anomaly'
        : '/api/training/timesnet/train/classification'

      const response = await api.post(endpoint, {
        windowed_session_id: windowedSession.value.session_id,
        config: {
          d_model: config.d_model,
          d_ff: config.d_ff,
          e_layers: config.e_layers,
          dropout: config.dropout,
          top_k: config.top_k,
          period_list: config.period_list
        },
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        test_size: config.test_size || 0.2
      })

      trainingSession.value = response.data
      currentStep.value = 'deploy'

      return { success: true, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to train TimesNet model'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  function reset() {
    dataSession.value = null
    windowedSession.value = null
    featureSession.value = null
    trainingSession.value = null
    selectedFeatures.value = []
    selectedAlgorithm.value = ''
    hyperparameters.value = {}
    currentStep.value = 'data'
    error.value = null
    // Keep trainingApproach on reset - user preference
  }

  function setMode(newMode: PipelineMode) {
    mode.value = newMode
    // Reset algorithm when mode changes
    selectedAlgorithm.value = ''
    hyperparameters.value = {}
  }

  function setTrainingApproach(approach: TrainingApproach) {
    trainingApproach.value = approach
    // Reset training session when approach changes
    trainingSession.value = null
  }

  return {
    mode,
    trainingApproach,
    currentStep,
    dataSession,
    windowingConfig,
    windowedSession,
    selectedFeatures,
    featureSession,
    selectedAlgorithm,
    hyperparameters,
    trainingSession,
    loading,
    error,
    canProceedToWindowing,
    canProceedToFeatures,
    canProceedToTraining,
    canProceedToDeploy,
    stepStatus,
    loadData,
    applyWindowing,
    extractFeatures,
    trainModel,
    trainTimesNet,
    reset,
    setMode,
    setTrainingApproach
  }
})
