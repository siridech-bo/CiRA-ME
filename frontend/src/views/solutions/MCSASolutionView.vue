<template>
  <v-container fluid class="pa-6" style="max-width: 1100px">
    <div class="d-flex align-center mb-1">
      <v-icon size="32" color="primary" class="mr-3">mdi-flash</v-icon>
      <h1 class="text-h4 font-weight-bold">Motor Current Diagnosis</h1>
      <v-spacer />
      <v-btn variant="text" size="small" :to="{ name: 'solutions' }">
        <v-icon start>mdi-arrow-left</v-icon> Catalog
      </v-btn>
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6" style="max-width: 760px">
      Diagnose induction-motor faults (broken rotor bars, eccentricity) from 3-phase
      stator current. Enter the motor nameplate, load your current recording, and
      analyze — the app handles windowing, MCSA feature extraction, and model
      selection for you.
    </p>

    <!-- 1. Motor & acquisition -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon start size="small">mdi-engine</v-icon> 1 · Motor &amp; acquisition
        <v-btn icon="mdi-information-outline" size="x-small" variant="text" class="ml-1"
          title="Example settings" @click="infoOpen = true" />
        <v-spacer />
        <v-btn size="x-small" variant="text" :prepend-icon="showDiagram ? 'mdi-chevron-up' : 'mdi-image-outline'"
          @click="showDiagram = !showDiagram">{{ showDiagram ? 'Hide' : 'Motor & sensor' }}</v-btn>
      </v-card-title>
      <v-card-text>
        <v-expand-transition>
          <div v-show="showDiagram" class="diagram-wrap mb-4 pa-3">
            <MotorCurrentDiagram :line-hz="form.line_freq_hz ?? 50" />
          </div>
        </v-expand-transition>
        <v-row dense>
          <v-col cols="6" sm="4" md="2">
            <v-text-field v-model.number="form.pole_pairs" type="number" min="1"
              label="Pole pairs" density="compact" hint="4-pole motor = 2" persistent-hint />
          </v-col>
          <v-col cols="6" sm="4" md="2">
            <v-select v-model.number="form.line_freq_hz" :items="[50, 60]"
              label="Line freq (Hz)" density="compact" />
          </v-col>
          <v-col cols="6" sm="4" md="2">
            <v-text-field v-model.number="form.rotor_bars" type="number" min="1"
              label="Rotor bars" density="compact" placeholder="optional" />
          </v-col>
          <v-col cols="6" sm="4" md="2">
            <v-text-field v-model.number="form.rated_rpm" type="number" min="1"
              label="Rated rpm" density="compact" placeholder="optional" />
          </v-col>
          <v-col cols="6" sm="4" md="2">
            <v-text-field v-model.number="form.sampling_rate" type="number" min="1"
              label="Sample rate (Hz)" density="compact" hint="Your DAQ rate" persistent-hint />
          </v-col>
          <v-col cols="6" sm="4" md="2">
            <v-text-field v-model.number="form.window_s" type="number" min="1"
              label="Window (s)" density="compact" hint="10 recommended" persistent-hint />
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <!-- 2. Data -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon start size="small">mdi-database</v-icon> 2 · Current data
        <v-spacer />
        <v-btn size="small" variant="tonal" @click="openBrowse">
          <v-icon start>mdi-folder-open</v-icon> Browse files
        </v-btn>
      </v-card-title>
      <v-card-text>
        <v-alert v-if="!loaded" type="info" variant="tonal" density="compact">
          Load a CSV of 3-phase current. Provide <strong>healthy-only</strong> data
          for an anomaly baseline, or <strong>labeled healthy + fault</strong> data
          (a label column) for a classifier — the app picks the right model.
        </v-alert>

        <div v-else>
          <div class="d-flex align-center mb-3">
            <v-icon start color="success">mdi-check-circle</v-icon>
            <span class="font-weight-medium">{{ loaded.name }}</span>
            <v-chip size="x-small" class="ml-2" variant="tonal">
              {{ loaded.total_rows?.toLocaleString() }} rows · {{ loaded.sensor_columns?.length }} columns
            </v-chip>
          </div>
          <v-select
            v-model="selectedPhases"
            :items="loaded.sensor_columns"
            label="Current phase columns (pick 3: Ia, Ib, Ic)"
            multiple chips closable-chips density="compact"
            :hint="selectedPhases.length === 3 ? '' : 'Select exactly 3 current columns'"
            persistent-hint
          />
        </div>
      </v-card-text>
    </v-card>

    <!-- 3. Approach & settings -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon start size="small">mdi-tune</v-icon> 3 · Approach &amp; model
      </v-card-title>
      <v-card-text>
        <div class="text-caption text-medium-emphasis mb-2">Approach</div>
        <v-btn-toggle v-model="form.approach" mandatory density="compact" color="primary" class="mb-1">
          <v-btn value="auto" size="small">Auto</v-btn>
          <v-btn value="anomaly" size="small">Anomaly (healthy baseline)</v-btn>
          <v-btn value="classification" size="small">Classification (labeled)</v-btn>
        </v-btn-toggle>
        <p class="text-caption text-medium-emphasis mb-4">
          <template v-if="form.approach === 'auto'">Auto picks Classification if your data has ≥2 labels, else an Anomaly baseline.</template>
          <template v-else-if="form.approach === 'anomaly'">Learns a healthy baseline (needs no fault labels) and flags drift.</template>
          <template v-else>Trains a fault classifier — requires a label column with ≥2 classes.</template>
        </p>

        <v-expansion-panels variant="accordion">
          <v-expansion-panel>
            <v-expansion-panel-title>
              <v-icon start size="small">mdi-cog</v-icon> Advanced settings (model, windowing, feature extraction)
            </v-expansion-panel-title>
            <v-expansion-panel-text>
              <v-row dense>
                <v-col cols="12" sm="6" md="4">
                  <v-select v-model="form.algorithm" :items="algoOptions" label="Model algorithm"
                    density="compact" clearable hint="Leave empty for the default" persistent-hint />
                </v-col>
                <v-col cols="12" sm="6" md="4">
                  <div class="text-caption mb-1">Window overlap: {{ Math.round(form.overlap * 100) }}%</div>
                  <v-slider v-model="form.overlap" :min="0" :max="0.9" :step="0.05" density="compact" hide-details />
                </v-col>
                <v-col cols="6" sm="6" md="4">
                  <v-text-field :model-value="`${form.window_s}s → ${Math.round(form.window_s * form.sampling_rate)} samples`"
                    label="Window size" density="compact" readonly />
                </v-col>
              </v-row>
              <v-divider class="my-3" />
              <div class="text-caption text-medium-emphasis mb-2">MCSA feature extraction</div>
              <v-row dense>
                <v-col cols="6" sm="4" md="3">
                  <v-switch v-model="form.calibrate" label="Grid-lock (calibrate)" color="primary" density="compact" hide-details />
                </v-col>
                <v-col cols="6" sm="4" md="3">
                  <v-select v-model.number="form.phase" :items="[0, 1, 2]" label="Analysis phase" density="compact" />
                </v-col>
                <v-col cols="6" sm="4" md="3">
                  <v-text-field v-model.number="form.absent_dbc" type="number" label="Absent floor (dBc)"
                    density="compact" hint="-90 for anomaly" persistent-hint />
                </v-col>
              </v-row>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
      </v-card-text>
    </v-card>

    <!-- Analyze -->
    <div class="d-flex justify-end align-center ga-4 mb-6">
      <v-chip v-if="solutionRun.isBusy.value" color="info" variant="tonal">
        <v-icon start size="small">
          {{ solutionRun.status.value === 'queued' ? 'mdi-clock-outline' : 'mdi-loading mdi-spin' }}
        </v-icon>
        {{ solutionRun.progressText.value }}
      </v-chip>
      <v-btn color="primary" size="large" :loading="running"
        :disabled="!canAnalyze" @click="analyze">
        <v-icon start>mdi-play</v-icon> Analyze
      </v-btn>
    </div>

    <!-- 4. Results -->
    <v-card v-if="result" class="mb-4" variant="elevated">
      <v-card-title class="text-subtitle-1">
        <v-icon start size="small">mdi-clipboard-pulse</v-icon> Results
      </v-card-title>
      <v-card-text>
        <v-row dense class="mb-2">
          <v-col cols="6" md="3">
            <div class="stat-key">Approach</div>
            <div class="stat-val">{{ result.mode === 'classification' ? 'Classification' : 'Anomaly baseline' }}</div>
          </v-col>
          <v-col cols="6" md="3">
            <div class="stat-key">Model</div>
            <div class="stat-val">{{ result.algorithm_name || result.algorithm }}</div>
          </v-col>
          <v-col cols="6" md="3">
            <div class="stat-key">Windows</div>
            <div class="stat-val">{{ result.num_windows }}<span v-if="result.windows_failed" class="text-caption text-warning"> ({{ result.windows_failed }} skipped)</span></div>
          </v-col>
          <v-col cols="6" md="3">
            <div class="stat-key">Key metric</div>
            <div class="stat-val">{{ headlineMetric }}</div>
          </v-col>
        </v-row>

        <!-- Resolved settings (transparency) -->
        <div v-if="settings" class="text-caption text-medium-emphasis mb-3">
          Windows of {{ settings.window_size.toLocaleString() }} samples
          ({{ settings.window_s }}s @ {{ settings.sampling_rate }} Hz),
          stride {{ settings.stride.toLocaleString() }} ({{ Math.round(settings.overlap * 100) }}% overlap) ·
          {{ result.num_features }} MCSA features ·
          extractor: pole_pairs={{ settings.extractor_params?.pole_pairs }},
          line={{ settings.extractor_params?.line_freq_hz }}Hz,
          calibrate={{ settings.extractor_params?.calibrate }},
          phase={{ settings.extractor_params?.phase }}
        </div>

        <v-divider class="my-3" />

        <!-- Per-window signal graph -->
        <div v-if="previews.length" class="mb-4">
          <div class="d-flex align-center mb-2">
            <v-spacer />
            <v-select
              v-model="selectedPreview" :items="previewItems"
              density="compact" hide-details variant="outlined"
              style="max-width: 240px"
            />
          </div>
          <SignalChart :channels="previews[selectedPreview]?.channels || {}"
            :fs="settings?.sampling_rate || form.sampling_rate" title="Current per window" y-label="current" :height="260" />
          <p class="text-caption text-medium-emphasis mt-1">
            Real current, native resolution (first {{ (previews[selectedPreview]?.preview_samples || 0).toLocaleString() }}
            of {{ (previews[selectedPreview]?.n_samples || 0).toLocaleString() }} samples in the window).
            Toggle phases with the chips; scroll to zoom, drag to pan.
          </p>
        </div>

        <!-- Frequency-domain MCSA spectrum -->
        <div v-if="spectrum" class="mb-4">
          <div class="d-flex align-center mb-2" style="gap: 8px;">
            <div class="text-subtitle-2">Current spectrum (MCSA)</div>
            <v-spacer />
            <v-btn size="x-small" variant="text" prepend-icon="mdi-magnify-minus-outline" @click="resetSpecZoom">
              Reset zoom
            </v-btn>
          </div>
          <div style="height: 260px">
            <Line ref="specRef" :data="spectrumData" :options="spectrumOptions" />
          </div>
          <p class="text-caption text-medium-emphasis mt-1">
            Amplitude in dBc vs the fundamental (<span style="color:#bbb">f₀</span>).
            <span style="color:#ffa726">Dashed lines</span> mark the broken-bar sidebands
            (1&nbsp;±&nbsp;2ks)·f₀. <strong>Zoom into f₀ ± a few Hz</strong> to see the sidebands rise
            with fault severity — that separation is what the model learns.
          </p>
        </div>

        <v-divider class="my-3" />

        <!-- Evidence: physically-meaningful features by group -->
        <div class="text-subtitle-2 mb-2">Evidence (mean per group)</div>
        <v-table density="compact" class="evidence-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th v-for="g in evidenceGroups" :key="g" class="text-right">{{ g }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in evidenceFeatures" :key="f">
              <td>{{ featureLabel(f) }}</td>
              <td v-for="g in evidenceGroups" :key="g" class="text-right mono">
                {{ formatVal(result.evidence.by_group[g]?.[f]) }}
              </td>
            </tr>
          </tbody>
        </v-table>
        <p class="text-caption text-medium-emphasis mt-2">
          Sideband levels are dBc relative to the fundamental. A less-negative
          broken-bar value (e.g. −33 vs −50) indicates a stronger fault. −120
          means no measurable sideband.
        </p>
      </v-card-text>
    </v-card>

    <!-- 5. Save & deploy -->
    <v-card v-if="result" class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon start size="small">mdi-rocket-launch</v-icon> Save &amp; deploy
      </v-card-title>
      <v-card-text>
        <p class="text-body-2 text-medium-emphasis mb-3">
          Save this trained model to deploy it as an inference endpoint. The
          deployed model runs the same windowing → MCSA features → prediction
          on live current.
        </p>
        <div class="d-flex align-center flex-wrap" style="gap: 12px;">
          <v-text-field
            v-model="modelName" label="Model name" density="compact"
            placeholder="Motor Current (MCSA)" hide-details style="max-width: 340px"
          />
          <v-btn color="primary" :loading="saving" @click="saveModel">
            <v-icon start>mdi-content-save</v-icon> Save model
          </v-btn>
          <template v-if="savedModelId">
            <v-icon color="success">mdi-check-circle</v-icon>
            <span class="text-body-2">Saved (#{{ savedModelId }}).</span>
            <v-btn variant="tonal" size="small" :to="{ name: 'pipeline-deploy' }">
              <v-icon start>mdi-rocket-launch</v-icon> Go to Deploy
            </v-btn>
          </template>
        </div>
        <p v-if="savedModelId" class="text-caption text-medium-emphasis mt-2">
          The model is in your saved-models list. On the Deploy page, choose it and
          deploy to your target device (SSH / Docker) as an inference endpoint.
        </p>
      </v-card-text>
    </v-card>

    <!-- Example-settings info dialog -->
    <v-dialog v-model="infoOpen" max-width="760">
      <v-card>
        <v-card-title class="text-subtitle-1 d-flex align-center">
          <v-icon start>mdi-information</v-icon> Example settings — motor current (MCSA)
          <v-spacer />
          <v-btn icon="mdi-close" variant="text" size="small" @click="infoOpen = false" />
        </v-card-title>
        <v-card-text>
          <p class="text-body-2 text-medium-emphasis mb-3">
            Example values from the built-in demo and a real UNESP motor. For your
            own motor, read these off the nameplate / datasheet.
          </p>
          <v-table density="compact">
            <thead><tr><th>Field</th><th>Demo</th><th>UNESP</th><th>Where to get it</th></tr></thead>
            <tbody>
              <tr><td>Sample rate (Hz)</td><td class="mono">5000</td><td class="mono">50000</td><td>Your DAQ rate</td></tr>
              <tr><td>Window (s)</td><td class="mono">10</td><td class="mono">10</td><td>Keep 10 s — needed for sidebands</td></tr>
              <tr><td>Pole pairs</td><td class="mono">2</td><td class="mono">2</td><td>Nameplate: poles ÷ 2 (4-pole = 2)</td></tr>
              <tr><td>Line frequency (Hz)</td><td class="mono">50</td><td class="mono">60</td><td>Mains supply (50 / 60)</td></tr>
              <tr><td>Rotor bars (opt)</td><td class="mono">28</td><td class="mono">34</td><td>Motor datasheet</td></tr>
              <tr><td>Rated rpm (opt)</td><td class="mono">—</td><td class="mono">1715</td><td>Nameplate full-load speed</td></tr>
              <tr><td>Phase columns</td><td class="mono" colspan="2">Ia, Ib, Ic</td><td>The 3 current channels</td></tr>
            </tbody>
          </v-table>
          <p class="text-caption text-medium-emphasis mt-3">
            Less-negative broken-bar sideband (e.g. −33 vs −50 dBc) = stronger fault.
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="tonal" color="primary" @click="applyExample">
            <v-icon start>mdi-auto-fix</v-icon> Apply the demo example values
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Browse dialog -->
    <v-dialog v-model="browse.open" max-width="620">
      <v-card>
        <v-card-title class="text-subtitle-1 d-flex align-center">
          <v-icon start size="small">mdi-folder</v-icon>
          <span class="text-truncate">{{ browse.path || 'Files' }}</span>
          <v-spacer />
          <v-btn icon="mdi-arrow-up" size="small" variant="text" @click="browseUp" />
        </v-card-title>
        <v-card-text style="max-height: 60vh; overflow-y: auto;">
          <v-list density="compact">
            <v-list-item
              v-for="item in browse.items"
              :key="item.path"
              :prepend-icon="item.is_dir ? 'mdi-folder' : 'mdi-file-delimited'"
              :title="item.name"
              :disabled="!item.is_dir && item.extension !== '.csv'"
              @click="item.is_dir ? browseTo(item.path) : pickFile(item)"
            />
            <v-list-item v-if="!browse.items.length" title="(empty)" disabled />
          </v-list>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import annotationPlugin from 'chartjs-plugin-annotation'
import api from '@/services/api'
import SignalChart from '@/components/SignalChart.vue'
import MotorCurrentDiagram from '@/components/MotorCurrentDiagram.vue'
import { useNotificationStore } from '@/stores/notification'
import { usePipelineStore } from '@/stores/pipeline'
import { useSolutionRun } from '@/composables/useSolutionRun'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, zoomPlugin, annotationPlugin)

const notify = useNotificationStore()
const pipeline = usePipelineStore()

const form = reactive({
  pole_pairs: 2,
  line_freq_hz: 50,
  rotor_bars: null as number | null,
  rated_rpm: null as number | null,
  sampling_rate: 5000,
  window_s: 10,
  // advanced (with sensible defaults)
  approach: 'auto' as 'auto' | 'anomaly' | 'classification',
  algorithm: null as string | null,
  overlap: 0.5,
  calibrate: true,
  phase: 0,
  absent_dbc: -120,
})

const ANOMALY_ALGOS = [
  { value: 'iforest', title: 'Isolation Forest' },
  { value: 'ecod', title: 'ECOD' },
  { value: 'copod', title: 'COPOD' },
  { value: 'ocsvm', title: 'One-Class SVM' },
  { value: 'lof', title: 'Local Outlier Factor' },
  { value: 'hbos', title: 'HBOS' },
]
const CLASSIFICATION_ALGOS = [
  { value: 'rf', title: 'Random Forest' },
  { value: 'gb', title: 'Gradient Boosting' },
  { value: 'svm', title: 'SVM' },
  { value: 'knn', title: 'k-NN' },
  { value: 'dt', title: 'Decision Tree' },
  { value: 'lr', title: 'Logistic Regression' },
]
const algoOptions = computed(() =>
  form.approach === 'classification' ? CLASSIFICATION_ALGOS
  : form.approach === 'anomaly' ? ANOMALY_ALGOS
  : [{ value: null as any, title: 'Auto (default for chosen approach)' }, ...ANOMALY_ALGOS, ...CLASSIFICATION_ALGOS])

// Seed defaults from the active solution's data profile, if launched from the catalog.
onMounted(() => {
  const s = pipeline.activeSolution
  if (s?.data_profile?.sample_rate_hz) form.sampling_rate = s.data_profile.sample_rate_hz
  if (s?.windowing?.window_s) form.window_s = s.windowing.window_s
})

const infoOpen = ref(false)
const showDiagram = ref(true)
function applyExample() {
  Object.assign(form, {
    pole_pairs: 2, line_freq_hz: 50, rotor_bars: 28, rated_rpm: null,
    sampling_rate: 5000, window_s: 10,
  })
  infoOpen.value = false
  notify.showSuccess('Applied the demo example settings.')
}

// ── Data loading ──
const loaded = ref<any>(null)
const selectedPhases = ref<string[]>([])
const browse = reactive({ open: false, path: '', items: [] as any[] })

async function openBrowse() {
  browse.open = true
  await browseTo('')
}
async function browseTo(path: string) {
  try {
    const resp = await api.post('/api/data/browse', { path })
    browse.path = resp.data.current_path || path
    browse.items = (resp.data.items || []).sort((a: any, b: any) =>
      (b.is_dir ? 1 : 0) - (a.is_dir ? 1 : 0) || a.name.localeCompare(b.name))
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Browse failed')
  }
}
function browseUp() {
  const p = browse.path
  if (!p) return
  const parent = p.replace(/[/\\][^/\\]+$/, '')
  browseTo(parent)
}
async function pickFile(item: any) {
  try {
    const resp = await api.post('/api/data/preview', { file_path: item.path, format: 'csv', rows: 5 })
    const meta = resp.data.metadata || {}
    loaded.value = {
      name: item.name,
      session_id: resp.data.session_id,
      sensor_columns: meta.sensor_columns || [],
      total_rows: meta.total_rows,
    }
    // default to the first 3 sensor columns as the phases
    selectedPhases.value = (meta.sensor_columns || []).slice(0, 3)
    browse.open = false
    notify.showSuccess(`Loaded ${item.name}`)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load file')
  }
}

// ── Analyze (queued via /api/solutions/mcsa/run — see useSolutionRun) ──
const solutionRun = useSolutionRun('/api/solutions/mcsa/run')
const running = computed(() => solutionRun.isBusy.value)
const result = ref<any>(null)

const canAnalyze = computed(() =>
  !!loaded.value?.session_id &&
  selectedPhases.value.length === 3 &&
  form.pole_pairs >= 1 &&
  form.sampling_rate > 0,
)

async function analyze() {
  result.value = null
  try {
    const params: Record<string, any> = {
      pole_pairs: form.pole_pairs,
      line_freq_hz: form.line_freq_hz,
      calibrate: form.calibrate,
      phase: form.phase,
      absent_dbc: form.absent_dbc,
    }
    if (form.rotor_bars) params.rotor_bars = form.rotor_bars
    if (form.rated_rpm) params.rated_rpm = form.rated_rpm

    const runResult: any = await solutionRun.run({
      data_session_id: loaded.value.session_id,
      sampling_rate: form.sampling_rate,
      window_s: form.window_s,
      overlap: form.overlap,
      approach: form.approach,
      algorithm: form.algorithm,
      selected_columns: selectedPhases.value,
      params,
      project_id: pipeline.projectId || undefined,
    })
    result.value = runResult
    selectedPreview.value = 0
    const mode = runResult.mode === 'classification' ? 'classifier' : 'anomaly baseline'
    notify.showSuccess(`Analysis complete — trained ${mode} on ${runResult.num_windows} windows.`)
  } catch (e: any) {
    notify.showError(e?.message || e.response?.data?.error || 'Analysis failed')
  }
}

// ── Results rendering ──
const evidenceGroups = computed<string[]>(() =>
  result.value?.evidence?.by_group ? Object.keys(result.value.evidence.by_group) : [])
const evidenceFeatures = computed<string[]>(() =>
  result.value?.evidence?.features || [])

const FEATURE_LABELS: Record<string, string> = {
  mcsa_brb_strongest_db: 'Broken-bar sideband (dBc)',
  mcsa_brb_lsb_db: 'BRB lower sideband (dBc)',
  mcsa_brb_usb_db: 'BRB upper sideband (dBc)',
  mcsa_ecc_strongest_db: 'Eccentricity (dBc)',
  mcsa_stator_strongest_db: 'Stator fault (dBc)',
  mcsa_slip: 'Slip',
  mcsa_slip_confident: 'Slip confident',
  mcsa_supported: 'Data supported',
}
function featureLabel(f: string) { return FEATURE_LABELS[f] || f }
function formatVal(v: any) {
  if (v === undefined || v === null) return '–'
  return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v
}

const headlineMetric = computed(() => {
  const m = result.value?.training?.metrics
  if (!m) return '—'
  if (result.value.mode === 'classification') {
    const acc = m.accuracy ?? m.test_accuracy
    return acc != null ? `${(acc * 100).toFixed(0)}% accuracy` : '—'
  }
  // anomaly: surface a contamination / anomaly-rate style metric if present
  const ar = m.anomaly_rate ?? m.contamination
  return ar != null ? `${(ar * 100).toFixed(0)}% flagged` : 'baseline trained'
})

// ── Per-window signal graph ──
const selectedPreview = ref(0)
const previews = computed<any[]>(() => result.value?.window_previews || [])
const previewItems = computed(() =>
  previews.value.map((p, i) => ({
    value: i,
    title: `Window ${p.index}${p.label ? ' · ' + p.label : ''}`,
  })))

const PHASE_COLORS = ['#42a5f5', '#66bb6a', '#ffa726']
const chartRef = ref<any>(null)
const chartData = computed(() => {
  const p = previews.value[selectedPreview.value]
  if (!p) return { labels: [], datasets: [] }
  const chNames = Object.keys(p.channels)
  const n = p.channels[chNames[0]]?.length || 0
  const fs = settings.value?.sampling_rate || form.sampling_rate || 1
  // native-resolution slice; x axis = time in milliseconds
  const labels = Array.from({ length: n }, (_, i) => +(1000 * i / fs).toFixed(2))
  return {
    labels,
    datasets: chNames.map((name, idx) => ({
      label: name,
      data: p.channels[name],
      borderColor: PHASE_COLORS[idx % PHASE_COLORS.length],
      borderWidth: 1,
      pointRadius: 0,
      tension: 0,
    })),
  }
})
const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  scales: {
    x: { type: 'linear' as const, title: { display: true, text: 'time (ms)' }, ticks: { maxTicksLimit: 10 } },
    y: { title: { display: true, text: 'current' } },
  },
  plugins: {
    legend: { position: 'top' as const },
    zoom: {
      pan: { enabled: true, mode: 'x' as const },
      zoom: { wheel: { enabled: true }, pinch: { enabled: true }, drag: { enabled: false }, mode: 'x' as const },
    },
  },
}))
function resetZoom() {
  chartRef.value?.chart?.resetZoom?.()
}

// ── Frequency-domain (MCSA spectrum) ──
const specRef = ref<any>(null)
const spectrum = computed(() => previews.value[selectedPreview.value]?.spectrum || null)
const spectrumData = computed(() => {
  const s = spectrum.value
  if (!s) return { labels: [], datasets: [] }
  return {
    labels: s.freq,
    datasets: [{
      label: 'Current spectrum (dBc)',
      data: s.db,
      borderColor: '#42a5f5',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0,
      fill: false,
    }],
  }
})
const spectrumOptions = computed(() => {
  const s = spectrum.value
  const anns: Record<string, any> = {}
  if (s) {
    anns.fund = {
      type: 'line', xMin: s.fundamental_hz, xMax: s.fundamental_hz,
      borderColor: 'rgba(255,255,255,0.55)', borderWidth: 1,
      label: { display: true, content: `f₀ ${s.fundamental_hz}Hz`, position: 'start', font: { size: 9 } },
    }
    ;(s.sidebands || []).forEach((f: number, i: number) => {
      anns[`sb${i}`] = {
        type: 'line', xMin: f, xMax: f, borderColor: 'rgba(255,167,38,0.7)',
        borderWidth: 1, borderDash: [4, 3],
      }
    })
  }
  return {
    responsive: true, maintainAspectRatio: false, animation: false as const,
    scales: {
      x: { type: 'linear' as const, title: { display: true, text: 'frequency (Hz)' }, ticks: { maxTicksLimit: 12 } },
      y: { title: { display: true, text: 'level (dBc)' }, suggestedMin: -100, suggestedMax: 5 },
    },
    plugins: {
      legend: { display: false },
      annotation: { annotations: anns },
      zoom: {
        pan: { enabled: true, mode: 'x' as const },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' as const },
      },
    },
  }
})
function resetSpecZoom() {
  specRef.value?.chart?.resetZoom?.()
}

// Resolved settings echo (shown for transparency)
const settings = computed(() => result.value?.settings || null)

// ── Save & deploy ──
const modelName = ref('')
const saving = ref(false)
const savedModelId = ref<number | null>(null)
async function saveModel() {
  if (!result.value) return
  const name = modelName.value.trim()
    || `Motor Current (MCSA) ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`
  saving.value = true
  try {
    const resp = await api.post('/api/solutions/mcsa/save', {
      name,
      training_session_id: result.value.training?.training_session_id,
      windowed_session_id: result.value.windowed_session_id,
      feature_session_id: result.value.feature_session_id,
      settings: result.value.settings,
    })
    savedModelId.value = resp.data.saved_model_id
    notify.showSuccess(`Saved "${resp.data.name}" — it's now deployable from the Deploy page.`)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Save failed')
  } finally {
    saving.value = false
  }
}
</script>

<style scoped lang="scss">
.stat-key { font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 600; }
.stat-val { font-size: 1.05rem; font-weight: 600; }
.mono { font-family: ui-monospace, monospace; }
.evidence-table th { font-weight: 600; }
.diagram-wrap { border: 1px solid rgba(127, 127, 127, 0.25); border-radius: 8px; background: rgba(127, 127, 127, 0.04); }
</style>
