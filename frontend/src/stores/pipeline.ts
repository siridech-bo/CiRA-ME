/**
 * CiRA ME - Pipeline Store
 * Manages ML pipeline state across views
 * Supports both Traditional ML (feature-based) and Deep Learning (TimesNet) workflows
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/services/api'
import { SUPPORTED_FEATURES } from '@/lib/featureExtraction'

export type PipelineMode = 'anomaly' | 'classification' | 'regression'
export type PipelineStep = 'data' | 'windowing' | 'features' | 'training' | 'deploy'
export type TrainingApproach = 'ml' | 'dl' | 'custom' | 'ti'

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
  test_ratio: number
  split_strategy: 'temporal_end' | 'temporal_blocks' | 'random'
  no_windowing: boolean
  // F3: user-selectable normalization method. Default 'min_max' preserves prior behavior.
  normalization_method: 'min_max' | 'z_score' | 'robust' | 'none'
}

interface WindowedSession {
  session_id: string
  num_windows: number
  window_shape: [number, number]
  metadata?: {
    normalization?: {
      method: string
      channel_min: number[]
      channel_max: number[]
      sensor_columns: string[]
      dropped_columns: string[]
    }
    window_size?: number
    stride?: number
    label_method?: string
    test_ratio?: number
    split_method?: string
    [key: string]: any
  }
  summary?: Record<string, any>
}

interface FeatureSession {
  session_id: string
  num_features: number
  feature_names: string[]
}

interface FeatureSelectionState {
  // Extraction result
  extractionResult: {
    session_id: string
    num_features: number
    num_windows: number
    feature_set?: string
  } | null
  // Selection result (before applying)
  selectionResult: {
    session_id: string
    selected_features: string[]
    original_count: number
    final_count: number
    method?: string
    fdr_level?: number
    after_fresh_count?: number
  } | null
  // Whether selection has been applied
  selectionApplied: boolean
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
  const mode = ref<PipelineMode>('classification')

  // Training approach: 'ml' (feature-based) or 'dl' (TimesNet end-to-end)
  const trainingApproach = ref<TrainingApproach>('ml')

  // Current step
  const currentStep = ref<PipelineStep>('data')

  // F4: Active project — threaded through Apply endpoints. Null means legacy /
  // ad-hoc pipeline (no project persistence).
  const projectId = ref<number | null>(null)

  // Data session
  const dataSession = ref<DataSession | null>(null)

  // Windowing
  const windowingConfig = ref<WindowingConfig>({
    window_size: 128,
    stride: 64,
    label_method: 'majority',
    test_ratio: 0.2,
    split_strategy: 'temporal_end',
    no_windowing: false,
    normalization_method: 'min_max'
  })
  const windowedSession = ref<WindowedSession | null>(null)

  // Features
  const selectedFeatures = ref<string[]>([])
  const featureSession = ref<FeatureSession | null>(null)

  // Feature selection state (for persistence between steps)
  const featureSelectionState = ref<FeatureSelectionState>({
    extractionResult: null,
    selectionResult: null,
    selectionApplied: false
  })

  // Column selection (which sensor columns to use)
  const selectedColumns = ref<string[]>([])

  // Regression target column
  const targetColumn = ref<string | null>(null)

  // Custom feature selections (persisted across navigation)
  const customFeatureToggles = ref<string[]>([])
  const rawSignals = ref<string[]>([])
  const rawSignalMethod = ref<'last' | 'first'>('last')

  // Training
  const selectedAlgorithm = ref<string>('')
  const hyperparameters = ref<Record<string, any>>({})
  const trainingSession = ref<TrainingSession | null>(null)

  // Loading states
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Confirm-restart-pipeline dialog state (driven from Data Source nav clicks)
  const showResetDialog = ref(false)

  const hasDownstreamState = computed(() =>
    !!windowedSession.value ||
    !!featureSession.value ||
    !!trainingSession.value ||
    !!featureSelectionState.value.extractionResult ||
    !!featureSelectionState.value.selectionResult
  )

  // Computed
  const canProceedToWindowing = computed(() => !!dataSession.value)
  const canProceedToFeatures = computed(() => !!windowedSession.value)

  // For DL mode, we can proceed to training directly from windowing (skip features)
  // For custom and ML modes, features are required
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
      // Reset column selection when new dataset is loaded
      selectedColumns.value = []
      targetColumn.value = null
      windowedSession.value = null
      featureSession.value = null
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

      // Auto-create a project on first Windowing apply so F4 Projects list
      // shows work-in-progress. Name uses dataset filename stem for locatability.
      if (projectId.value === null) {
        const stem = (dataSession.value.metadata?.filename || 'Pipeline')
          .replace(/\.[^.]+$/, '')
        const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ')
        await createProjectAndAdopt(`${stem} ${stamp}`)
      }

      const response = await api.post('/api/data/windowing', {
        session_id: dataSession.value.session_id,
        ...windowingConfig.value,
        target_column: targetColumn.value || undefined,
        selected_columns: selectedColumns.value.length > 0 ? selectedColumns.value : undefined,
        project_id: projectId.value || undefined
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

  /**
   * Async feature extraction. POST /extract returns 202 + job_id; the caller
   * polls via `pollExtractionJob`. This helper just kicks the job off — the
   * view is expected to handle the polling loop so it can render queue
   * position, elapsed time, and a cancel button.
   */
  async function submitExtractionJob(features?: string[]) {
    if (!windowedSession.value) {
      return { success: false as const, error: 'No windowed data' }
    }

    try {
      loading.value = true
      error.value = null

      const featureList = features && features.length > 0
        ? features
        : (selectedFeatures.value.length > 0 ? selectedFeatures.value : null)

      const response = await api.post('/api/features/extract', {
        session_id: windowedSession.value.session_id,
        features: featureList,
        project_id: projectId.value || undefined
      })

      // Backend returns { job_id, status, queue_position, estimated_wait_seconds }.
      return { success: true as const, data: response.data }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to submit feature extraction'
      return { success: false as const, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * P2 Phase 2 — Fast Mode client-side feature extraction.
   *
   * Downloads the raw windows for the current windowed session, spins up a
   * Web Worker (see `frontend/src/workers/features.worker.ts`), and lets the
   * worker compute the lightweight feature set entirely in the browser.
   * Bypasses the 5-slot backend queue completely so 60 concurrent workshop
   * users don't have to wait on each other.
   *
   * Returns a handle so the calling view can:
   *   - listen to `worker.onmessage` for `progress` frames
   *   - call `terminate()` to cancel a run mid-flight
   *
   * On the `done` frame the store's `featureSession` / `extractionResult`
   * are already updated and `currentStep` has moved to 'training'; the view
   * just needs to close its progress card.
   */
  async function extractFeaturesFast(features?: string[]) {
    if (!windowedSession.value) {
      return { success: false as const, error: 'No windowed data' }
    }

    const featureList = features && features.length > 0
      ? features
      : (selectedFeatures.value.length > 0 ? selectedFeatures.value : SUPPORTED_FEATURES)

    try {
      loading.value = true
      error.value = null

      // Fetch raw windows. Backend gzips the JSON on the wire.
      const sid = windowedSession.value.session_id
      const resp = await api.get(`/api/features/windows/${sid}`)
      const { windows, sensor_columns, sampling_rate, num_windows } = resp.data

      // ?worker suffix is the Vite convention — the module is compiled to a
      // separate chunk with a proper Worker constructor.
      // Note: we use the new URL() form (not `?worker`) because the ?worker
      // import must be a static string and TS gets grumpy about the type.
      const worker = new Worker(
        new URL('../workers/features.worker.ts', import.meta.url),
        { type: 'module' },
      )

      // Wire success + error handlers before posting the extract message so
      // a synchronous failure inside the worker doesn't race the listener.
      const donePromise = new Promise<{ success: true; data: any } | { success: false; error: string }>(
        (resolve) => {
          worker.onmessage = async (evt: MessageEvent<any>) => {
            const msg = evt.data
            if (!msg) return
            if (msg.type === 'done') {
              worker.terminate()
              // Register the Fast Mode result with the backend so downstream
              // steps (selection, visualization, training) can look it up in
              // _feature_sessions like any server-side extraction.
              try {
                const reg = await api.post('/api/features/register-fast', {
                  windowed_session_id: sid,
                  feature_names: msg.feature_names,
                  features_df: msg.features_df,
                  project_id: projectId.value || undefined,
                })
                const featureData = {
                  session_id: reg.data.session_id,
                  num_windows: reg.data.num_windows,
                  num_features: reg.data.num_features,
                  feature_names: reg.data.feature_names,
                  feature_set: 'fast_mode',
                  preview: reg.data.preview,
                  features_df: msg.features_df,
                  extraction_ms: msg.extraction_ms,
                }
                featureSession.value = featureData as any
                setExtractionResult({
                  session_id: featureData.session_id,
                  num_features: featureData.num_features,
                  num_windows: featureData.num_windows,
                  feature_set: 'fast_mode',
                })
                currentStep.value = 'training'
                resolve({ success: true, data: featureData })
              } catch (regErr: any) {
                error.value = regErr.response?.data?.error || 'Failed to register Fast Mode session on server'
                resolve({ success: false, error: error.value })
              }
            } else if (msg.type === 'error') {
              error.value = msg.message || 'Fast Mode extraction failed'
              worker.terminate()
              resolve({ success: false, error: error.value })
            }
            // 'progress' frames are handled by the view via its own listener.
          }
          worker.onerror = (e) => {
            error.value = e.message || 'Fast Mode worker crashed'
            worker.terminate()
            resolve({ success: false, error: error.value })
          }
        },
      )

      worker.postMessage({
        type: 'extract',
        windows,
        channelNames: sensor_columns,
        selectedFeatures: featureList,
        samplingRate: sampling_rate,
        sessionId: `fast_${sid}`,
      })

      return {
        success: true as const,
        worker,
        donePromise,
        totalWindows: num_windows,
        terminate: () => worker.terminate(),
      }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Failed to start Fast Mode extraction'
      return { success: false as const, error: error.value }
    } finally {
      loading.value = false
    }
  }

  /**
   * Utility for the FeaturesView to decide which lightweight features to
   * grey out when Fast Mode is on. Right now the client covers 100% of the
   * lightweight set — this returns `unsupported: []` today — but leaving
   * this as a helper means adding a backend-only feature later just requires
   * omitting it from SUPPORTED_FEATURES.
   */
  function fastModeAvailable(features: string[]): { available: boolean; unsupported: string[] } {
    const supportedSet = new Set(SUPPORTED_FEATURES)
    const unsupported = features.filter((f) => !supportedSet.has(f))
    return { available: unsupported.length === 0, unsupported }
  }

  /**
   * Kept for backwards compatibility with callers that expect the old
   * synchronous shape. Wraps submit + polling internally. Views that need to
   * render queue status should call `submitExtractionJob` directly.
   */
  async function extractFeatures(features?: string[]) {
    const submitResult = await submitExtractionJob(features)
    if (!submitResult.success) return submitResult

    const jobId = submitResult.data.job_id
    // Poll until terminal state.
    // Safety cap: 20 min at 2s intervals = 600 polls.
    for (let i = 0; i < 600; i++) {
      await new Promise((r) => setTimeout(r, 2000))
      try {
        const s = await api.get(`/api/features/extract/${jobId}`)
        const status = s.data.status
        if (status === 'done') {
          featureSession.value = s.data.features
          currentStep.value = 'training'
          return { success: true, data: s.data.features }
        }
        if (status === 'error') {
          error.value = s.data.error || 'Feature extraction failed'
          return { success: false, error: error.value }
        }
        if (status === 'cancelled') {
          error.value = 'Feature extraction cancelled'
          return { success: false, error: error.value }
        }
      } catch (e: any) {
        error.value = e.response?.data?.error || 'Failed to poll feature extraction'
        return { success: false, error: error.value }
      }
    }
    error.value = 'Feature extraction timed out'
    return { success: false, error: error.value }
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
        : mode.value === 'regression'
        ? '/api/training/train/regression'
        : '/api/training/train/classification'

      const response = await api.post(endpoint, {
        feature_session_id: featureSession.value.session_id,
        algorithm: selectedAlgorithm.value,
        hyperparameters: hyperparameters.value,
        project_id: projectId.value || undefined
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
        test_size: config.test_size || 0.2,
        project_id: projectId.value || undefined
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
    selectedColumns.value = []
    targetColumn.value = null
    customFeatureToggles.value = []
    rawSignals.value = []
    rawSignalMethod.value = 'last'
    currentStep.value = 'data'
    error.value = null
    // Reset feature selection state
    featureSelectionState.value = {
      extractionResult: null,
      selectionResult: null,
      selectionApplied: false
    }
    // Keep projectId and trainingApproach on reset - user preferences
  }

  async function setActiveProject(id: number | null) {
    projectId.value = id
    if (id === null) return

    try {
      const res = await api.get(`/api/projects/${id}`)
      if (res.data?.mode) {
        mode.value = res.data.mode === 'mixed' ? mode.value : res.data.mode
      }
    } catch { /* ignore metadata hydration errors */ }

    // Rehydrate the persisted data / windowing / features into the working
    // pipeline state so clicking a stage chip on the Projects list lands the
    // user on a pre-loaded view, not an empty one.
    try {
      const res = await api.post(`/api/projects/${id}/hydrate`)
      const h = res.data
      if (h?.data_session) {
        dataSession.value = h.data_session
      }
      if (h?.windowing_config) {
        windowingConfig.value = { ...windowingConfig.value, ...h.windowing_config }
      }
      if (h?.feature_session?.feature_names?.length) {
        selectedFeatures.value = h.feature_session.feature_names
      }
    } catch { /* nothing persisted yet — new project */ }
  }

  async function createProjectAndAdopt(name: string): Promise<number | null> {
    try {
      const res = await api.post('/api/projects', {
        name,
        mode: mode.value,
      })
      const newId = res.data?.id
      if (newId) {
        projectId.value = newId
        return newId
      }
    } catch { /* ignore */ }
    return null
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

  // Feature selection state management
  function setExtractionResult(result: FeatureSelectionState['extractionResult']) {
    featureSelectionState.value.extractionResult = result
    // Reset selection when new extraction is done
    featureSelectionState.value.selectionResult = null
    featureSelectionState.value.selectionApplied = false
  }

  function setSelectionResult(result: FeatureSelectionState['selectionResult']) {
    featureSelectionState.value.selectionResult = result
    featureSelectionState.value.selectionApplied = false
  }

  function markSelectionApplied() {
    featureSelectionState.value.selectionApplied = true
  }

  // Computed: check if there's an unapplied selection
  const hasUnappliedSelection = computed(() => {
    return featureSelectionState.value.selectionResult !== null &&
           !featureSelectionState.value.selectionApplied
  })

  // Get the active feature count (selected if applied, otherwise extracted)
  const activeFeatureCount = computed(() => {
    if (featureSelectionState.value.selectionApplied && featureSelectionState.value.selectionResult) {
      return featureSelectionState.value.selectionResult.final_count
    }
    if (featureSelectionState.value.extractionResult) {
      return featureSelectionState.value.extractionResult.num_features
    }
    if (featureSession.value) {
      return featureSession.value.num_features
    }
    return 0
  })

  // Get the list of selected feature names (for display in training)
  const selectedFeatureNames = computed(() => {
    if (featureSelectionState.value.selectionApplied && featureSelectionState.value.selectionResult) {
      return featureSelectionState.value.selectionResult.selected_features
    }
    if (featureSession.value) {
      return featureSession.value.feature_names
    }
    return []
  })

  return {
    mode,
    trainingApproach,
    currentStep,
    projectId,
    setActiveProject,
    createProjectAndAdopt,
    dataSession,
    windowingConfig,
    windowedSession,
    selectedFeatures,
    featureSession,
    selectedAlgorithm,
    selectedColumns,
    targetColumn,
    customFeatureToggles,
    rawSignals,
    rawSignalMethod,
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
    submitExtractionJob,
    extractFeaturesFast,
    fastModeAvailable,
    trainModel,
    trainTimesNet,
    reset,
    setMode,
    setTrainingApproach,
    // Feature selection state
    featureSelectionState,
    setExtractionResult,
    setSelectionResult,
    markSelectionApplied,
    hasUnappliedSelection,
    activeFeatureCount,
    selectedFeatureNames,
    // Restart-pipeline confirmation
    showResetDialog,
    hasDownstreamState
  }
})
