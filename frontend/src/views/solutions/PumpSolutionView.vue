<template>
  <v-container fluid class="pa-6" style="max-width: 1100px">
    <div class="d-flex align-center mb-1">
      <v-icon size="32" color="primary" class="mr-3">mdi-pump</v-icon>
      <h1 class="text-h4 font-weight-bold">Pump Cavitation Diagnosis</h1>
      <v-spacer />
      <v-btn variant="text" size="small" :to="{ name: 'solutions' }"><v-icon start>mdi-arrow-left</v-icon> Catalog</v-btn>
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6" style="max-width: 760px">
      Detect centrifugal-pump cavitation and impeller faults from vibration or pressure.
      Enter the shaft speed and impeller blade count, load your recording, and analyze —
      the app extracts blade-pass and cavitation (broadband) features and picks a model.
    </p>

    <!-- 1. Pump & acquisition -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon start size="small">mdi-cog</v-icon> 1 · Pump &amp; acquisition
        <v-btn icon="mdi-information-outline" size="x-small" variant="text" class="ml-1" title="Example settings" @click="infoOpen = true" />
        <v-spacer />
        <v-btn size="x-small" variant="text" :prepend-icon="showDiagram ? 'mdi-chevron-up' : 'mdi-image-outline'"
          @click="showDiagram = !showDiagram">{{ showDiagram ? 'Hide' : 'Pump & sensor' }}</v-btn>
      </v-card-title>
      <v-card-text>
        <v-expand-transition>
          <div v-show="showDiagram" class="diagram-wrap mb-4 pa-3">
            <PumpDiagram :shaft-hz="params.shaft_speed_hz ?? 25" :n-blades="params.n_blades ?? 6" />
          </div>
        </v-expand-transition>
        <v-row dense>
          <v-col cols="6" sm="4" md="3"><v-text-field v-model.number="sampling_rate" type="number" min="1" label="Sample rate (Hz)" density="compact" hint="Your DAQ rate" persistent-hint /></v-col>
          <v-col cols="6" sm="4" md="3"><v-text-field v-model.number="window_s" type="number" min="0.1" step="0.1" label="Window (s)" density="compact" /></v-col>
          <v-col v-for="p in formParams" :key="p.name" cols="6" sm="4" md="3">
            <v-switch v-if="p.type === 'bool'" v-model="params[p.name]" :label="p.label" color="primary" density="compact" hide-details />
            <v-select v-else-if="p.choices" v-model="params[p.name]" :items="p.choices" :label="p.label + (p.unit ? ` (${p.unit})` : '')" density="compact" />
            <v-text-field v-else v-model.number="params[p.name]" type="number" :label="p.label + (p.unit ? ` (${p.unit})` : '')" density="compact" :placeholder="p.required ? '' : 'optional'" :messages="p.help ? [p.help] : []" />
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <!-- 2. Data -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon start size="small">mdi-database</v-icon> 2 · Pump data
        <v-spacer /><v-btn size="small" variant="tonal" @click="openBrowse"><v-icon start>mdi-folder-open</v-icon> Browse files</v-btn>
      </v-card-title>
      <v-card-text>
        <v-alert v-if="!loaded" type="info" variant="tonal" density="compact">
          Load a CSV of pump vibration or pressure. <strong>Healthy-only</strong> → anomaly baseline;
          <strong>labeled</strong> (a label column, e.g. healthy/cavitation) → classifier.
        </v-alert>
        <div v-else>
          <div class="d-flex align-center mb-3">
            <v-icon start color="success">mdi-check-circle</v-icon>
            <span class="font-weight-medium">{{ loaded.name }}</span>
            <v-chip size="x-small" class="ml-2" variant="tonal">{{ loaded.total_rows?.toLocaleString() }} rows · {{ loaded.sensor_columns?.length }} columns</v-chip>
          </div>
          <v-select v-model="selectedChannels" :items="loaded.sensor_columns" label="Signal channel(s)" multiple chips closable-chips density="compact" hint="Pick the vibration/pressure column(s). 'Channel' above selects among them." persistent-hint />
        </div>
      </v-card-text>
    </v-card>

    <!-- 3. Approach -->
    <v-card class="mb-4">
      <v-card-title class="text-subtitle-1"><v-icon start size="small">mdi-tune</v-icon> 3 · Approach &amp; model</v-card-title>
      <v-card-text>
        <div class="text-caption text-medium-emphasis mb-2">Approach</div>
        <v-btn-toggle v-model="approach" mandatory density="compact" color="primary" class="mb-1">
          <v-btn value="auto" size="small">Auto</v-btn>
          <v-btn value="anomaly" size="small">Anomaly (healthy baseline)</v-btn>
          <v-btn value="classification" size="small">Classification (labeled)</v-btn>
        </v-btn-toggle>
        <v-expansion-panels variant="accordion" class="mt-2">
          <v-expansion-panel>
            <v-expansion-panel-title><v-icon start size="small">mdi-cog</v-icon> Advanced (model, overlap)</v-expansion-panel-title>
            <v-expansion-panel-text>
              <v-row dense>
                <v-col cols="12" sm="6" md="4"><v-select v-model="algorithm" :items="algoOptions" label="Model algorithm" density="compact" clearable hint="Empty = default" persistent-hint /></v-col>
                <v-col cols="12" sm="6" md="4"><div class="text-caption mb-1">Window overlap: {{ Math.round(overlap * 100) }}%</div><v-slider v-model="overlap" :min="0" :max="0.9" :step="0.05" density="compact" hide-details /></v-col>
              </v-row>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
      </v-card-text>
    </v-card>

    <div class="d-flex justify-end mb-6">
      <v-btn color="primary" size="large" :loading="running" :disabled="!canAnalyze" @click="analyze"><v-icon start>mdi-play</v-icon> Analyze</v-btn>
    </div>

    <!-- Results -->
    <v-card v-if="result" class="mb-4" variant="elevated">
      <v-card-title class="text-subtitle-1"><v-icon start size="small">mdi-clipboard-pulse</v-icon> Results</v-card-title>
      <v-card-text>
        <v-row dense class="mb-2">
          <v-col cols="6" md="3"><div class="stat-key">Approach</div><div class="stat-val">{{ result.mode === 'classification' ? 'Classification' : 'Anomaly baseline' }}</div></v-col>
          <v-col cols="6" md="3"><div class="stat-key">Model</div><div class="stat-val">{{ result.algorithm_name || result.algorithm }}</div></v-col>
          <v-col cols="6" md="3"><div class="stat-key">Windows</div><div class="stat-val">{{ result.num_windows }}</div></v-col>
          <v-col cols="6" md="3"><div class="stat-key">Key metric</div><div class="stat-val">{{ headlineMetric }}</div></v-col>
        </v-row>
        <div v-if="settings" class="text-caption text-medium-emphasis mb-3">
          Windows of {{ settings.window_size?.toLocaleString() }} samples ({{ settings.window_s }}s @ {{ settings.sampling_rate }} Hz),
          {{ Math.round(settings.overlap * 100) }}% overlap · {{ result.num_features }} pump features ·
          blade-pass = {{ settings.extractor_params?.n_blades }} × {{ settings.extractor_params?.shaft_speed_hz }} Hz
        </div>
        <v-divider class="my-3" />

        <div v-if="previews.length" class="mb-4">
          <div class="d-flex align-center mb-2"><v-spacer />
            <v-select v-model="selectedPreview" :items="previewItems" density="compact" hide-details variant="outlined" style="max-width: 240px" />
          </div>
          <SignalChart :channels="previews[selectedPreview]?.channels || {}"
            :fs="settings?.sampling_rate || sampling_rate" title="Signal per window" y-label="acceleration" />
        </div>
        <v-divider class="my-3" />

        <div v-if="spectrum" class="mb-4">
          <div class="d-flex align-center mb-2" style="gap: 8px;"><div class="text-subtitle-2">Spectrum</div><v-spacer /><v-btn size="x-small" variant="text" prepend-icon="mdi-magnify-minus-outline" @click="resetSpecZoom">Reset zoom</v-btn></div>
          <div style="height: 260px"><Line ref="specRef" :data="spectrumData" :options="spectrumOptions" /></div>
          <p class="text-caption text-medium-emphasis mt-1">
            <span style="color:#ffa726">Dashed lines</span> mark shaft &amp; blade-pass frequencies. A broadband hump at
            high frequency (5–10 kHz) indicates <strong>cavitation</strong>; a peak at blade-pass indicates an impeller/hydraulic fault.
          </p>
        </div>
        <v-divider class="my-3" />

        <div class="text-subtitle-2 mb-2">Evidence (mean per group)</div>
        <v-table density="compact">
          <thead><tr><th>Feature</th><th v-for="g in evidenceGroups" :key="g" class="text-right">{{ g }}</th></tr></thead>
          <tbody><tr v-for="f in evidenceFeatures" :key="f"><td>{{ featureLabel(f) }}</td><td v-for="g in evidenceGroups" :key="g" class="text-right mono">{{ formatVal(result.evidence.by_group[g]?.[f]) }}</td></tr></tbody>
        </v-table>
      </v-card-text>
    </v-card>

    <!-- Save & deploy -->
    <v-card v-if="result" class="mb-4">
      <v-card-title class="text-subtitle-1"><v-icon start size="small">mdi-rocket-launch</v-icon> Save &amp; deploy</v-card-title>
      <v-card-text>
        <div class="d-flex align-center flex-wrap" style="gap: 12px;">
          <v-text-field v-model="modelName" label="Model name" density="compact" placeholder="Pump Cavitation" hide-details style="max-width: 340px" />
          <v-btn color="primary" :loading="saving" @click="saveModel"><v-icon start>mdi-content-save</v-icon> Save model</v-btn>
          <template v-if="savedModelId"><v-icon color="success">mdi-check-circle</v-icon><span class="text-body-2">Saved (#{{ savedModelId }}).</span><v-btn variant="tonal" size="small" :to="{ name: 'pipeline-deploy' }"><v-icon start>mdi-rocket-launch</v-icon> Go to Deploy</v-btn></template>
        </div>
      </v-card-text>
    </v-card>

    <!-- Info dialog -->
    <v-dialog v-model="infoOpen" max-width="720">
      <v-card>
        <v-card-title class="text-subtitle-1 d-flex align-center"><v-icon start>mdi-information</v-icon> Example settings — pump<v-spacer /><v-btn icon="mdi-close" variant="text" size="small" @click="infoOpen = false" /></v-card-title>
        <v-card-text>
          <p class="text-body-2 text-medium-emphasis mb-3">Values from the real 4TU pump demo (Motor-4 @ 70%). Use your pump's data for shaft speed and blade count.</p>
          <v-table density="compact">
            <thead><tr><th>Field</th><th>Example (4TU demo)</th><th>Where to get it</th></tr></thead>
            <tbody>
              <tr><td>Sample rate (Hz)</td><td class="mono">20000</td><td>Your DAQ rate (cavitation needs ≥ ~8 kHz)</td></tr>
              <tr><td>Window (s)</td><td class="mono">1</td><td>1 s is plenty at kHz rates</td></tr>
              <tr><td>Shaft speed (Hz)</td><td class="mono">34.4</td><td>Pump rpm ÷ 60 (Motor-4 @ 70% ≈ 34.4)</td></tr>
              <tr><td>Impeller blades</td><td class="mono">6</td><td>Pump datasheet (vane count)</td></tr>
              <tr><td>Cavitation band (Hz)</td><td class="mono">1000 – 9500</td><td>Pump-specific — validated ~1–5 kHz on 4TU; tune to where your cavitation energy rises</td></tr>
              <tr><td>Signal channel</td><td class="mono">pump-side accel</td><td>Use a PUMP-side accelerometer (not motor-side) — cavitation shows there</td></tr>
            </tbody>
          </v-table>
          <p class="text-caption text-medium-emphasis mt-3">Cavitation → high HF-energy-ratio &amp; spectral entropy · impeller fault → elevated blade-pass. <strong>Real 4TU demo: cavitation HF-ratio 0.45 vs healthy 0.26.</strong></p>
        </v-card-text>
        <v-card-actions><v-spacer /><v-btn variant="tonal" color="primary" @click="applyExample"><v-icon start>mdi-auto-fix</v-icon> Apply these example values</v-btn></v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Browse dialog -->
    <v-dialog v-model="browse.open" max-width="620">
      <v-card>
        <v-card-title class="text-subtitle-1 d-flex align-center"><v-icon start size="small">mdi-folder</v-icon><span class="text-truncate">{{ browse.path || 'Files' }}</span><v-spacer /><v-btn icon="mdi-arrow-up" size="small" variant="text" @click="browseUp" /></v-card-title>
        <v-card-text style="max-height: 60vh; overflow-y: auto;">
          <v-list density="compact">
            <v-list-item v-for="item in browse.items" :key="item.path" :prepend-icon="item.is_dir ? 'mdi-folder' : 'mdi-file-delimited'" :title="item.name" :disabled="!item.is_dir && item.extension !== '.csv'" @click="item.is_dir ? browseTo(item.path) : pickFile(item)" />
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
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import annotationPlugin from 'chartjs-plugin-annotation'
import api from '@/services/api'
import SignalChart from '@/components/SignalChart.vue'
import PumpDiagram from '@/components/PumpDiagram.vue'
import { useNotificationStore } from '@/stores/notification'
import { usePipelineStore } from '@/stores/pipeline'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, zoomPlugin, annotationPlugin)
const notify = useNotificationStore()
const pipeline = usePipelineStore()

const sampling_rate = ref(20000)
const window_s = ref(1.0)
const params = reactive<Record<string, any>>({})
const formParams = ref<any[]>([])
const approach = ref<'auto' | 'anomaly' | 'classification'>('auto')
const algorithm = ref<string | null>(null)
const overlap = ref(0.5)

const infoOpen = ref(false)
const showDiagram = ref(true)
function applyExample() {
  sampling_rate.value = 20000; window_s.value = 1
  Object.assign(params, { shaft_speed_hz: 34.4, n_blades: 6, cav_band_low_hz: 1000, cav_band_high_hz: 9500, axis: 0 })
  infoOpen.value = false
  notify.showSuccess('Applied the 4TU pump-demo example settings.')
}

const ANOMALY_ALGOS = [{ value: 'iforest', title: 'Isolation Forest' }, { value: 'ecod', title: 'ECOD' }, { value: 'copod', title: 'COPOD' }, { value: 'ocsvm', title: 'One-Class SVM' }, { value: 'lof', title: 'Local Outlier Factor' }, { value: 'hbos', title: 'HBOS' }]
const CLASS_ALGOS = [{ value: 'rf', title: 'Random Forest' }, { value: 'gb', title: 'Gradient Boosting' }, { value: 'svm', title: 'SVM' }, { value: 'knn', title: 'k-NN' }, { value: 'dt', title: 'Decision Tree' }, { value: 'lr', title: 'Logistic Regression' }]
const algoOptions = computed(() => approach.value === 'classification' ? CLASS_ALGOS : approach.value === 'anomaly' ? ANOMALY_ALGOS : [{ value: null as any, title: 'Auto' }, ...ANOMALY_ALGOS, ...CLASS_ALGOS])

onMounted(async () => {
  try {
    const resp = await api.get('/api/solutions/pump_analysis')
    const sol = resp.data.solution
    if (sol?.data_profile?.sample_rate_hz) sampling_rate.value = sol.data_profile.sample_rate_hz
    if (sol?.windowing?.window_s) window_s.value = sol.windowing.window_s
    formParams.value = sol?.param_schema || []
    for (const p of formParams.value) params[p.name] = p.default
  } catch { /* ignore */ }
})

const loaded = ref<any>(null)
const selectedChannels = ref<string[]>([])
const browse = reactive({ open: false, path: '', items: [] as any[] })
async function openBrowse() { browse.open = true; await browseTo('') }
async function browseTo(path: string) {
  try {
    const resp = await api.post('/api/data/browse', { path })
    browse.path = resp.data.current_path || path
    browse.items = (resp.data.items || []).sort((a: any, b: any) => (b.is_dir ? 1 : 0) - (a.is_dir ? 1 : 0) || a.name.localeCompare(b.name))
  } catch (e: any) { notify.showError(e.response?.data?.error || 'Browse failed') }
}
function browseUp() { if (browse.path) browseTo(browse.path.replace(/[/\\][^/\\]+$/, '')) }
async function pickFile(item: any) {
  try {
    const resp = await api.post('/api/data/preview', { file_path: item.path, format: 'csv', rows: 5 })
    const meta = resp.data.metadata || {}
    loaded.value = { name: item.name, session_id: resp.data.session_id, sensor_columns: meta.sensor_columns || [], total_rows: meta.total_rows }
    selectedChannels.value = (meta.sensor_columns || []).slice(0, 1)
    browse.open = false; notify.showSuccess(`Loaded ${item.name}`)
  } catch (e: any) { notify.showError(e.response?.data?.error || 'Failed to load file') }
}

const running = ref(false)
const result = ref<any>(null)
const canAnalyze = computed(() => !!loaded.value?.session_id && selectedChannels.value.length >= 1 && sampling_rate.value > 0)
async function analyze() {
  running.value = true; result.value = null
  try {
    const resp = await api.post('/api/solutions/pump/run', {
      data_session_id: loaded.value.session_id, sampling_rate: sampling_rate.value, window_s: window_s.value,
      overlap: overlap.value, approach: approach.value, algorithm: algorithm.value,
      selected_columns: selectedChannels.value, params: { ...params }, project_id: pipeline.projectId || undefined,
    })
    result.value = resp.data; selectedPreview.value = 0
    notify.showSuccess(`Analysis complete — ${resp.data.num_windows} windows.`)
  } catch (e: any) { notify.showError(e.response?.data?.error || 'Analysis failed') }
  finally { running.value = false }
}

const selectedPreview = ref(0)
const previews = computed<any[]>(() => result.value?.window_previews || [])
const previewItems = computed(() => previews.value.map((p, i) => ({ value: i, title: `Window ${p.index}${p.label ? ' · ' + p.label : ''}` })))
const PHASE_COLORS = ['#42a5f5', '#66bb6a', '#ffa726']
const chartRef = ref<any>(null)
const chartData = computed(() => {
  const p = previews.value[selectedPreview.value]; if (!p) return { labels: [], datasets: [] }
  const chNames = Object.keys(p.channels); const n = p.channels[chNames[0]]?.length || 0
  const fs = settings.value?.sampling_rate || sampling_rate.value || 1
  return { labels: Array.from({ length: n }, (_, i) => +(1000 * i / fs).toFixed(2)), datasets: chNames.map((name, idx) => ({ label: name, data: p.channels[name], borderColor: PHASE_COLORS[idx % 3], borderWidth: 1, pointRadius: 0, tension: 0 })) }
})
const chartOptions = computed(() => ({ responsive: true, maintainAspectRatio: false, animation: false as const, scales: { x: { type: 'linear' as const, title: { display: true, text: 'time (ms)' }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: 'signal' } } }, plugins: { legend: { position: 'top' as const }, zoom: { pan: { enabled: true, mode: 'x' as const }, zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' as const } } } }))
function resetZoom() { chartRef.value?.chart?.resetZoom?.() }

const specRef = ref<any>(null)
const spectrum = computed(() => previews.value[selectedPreview.value]?.spectrum || null)
const spectrumData = computed(() => { const s = spectrum.value; if (!s) return { labels: [], datasets: [] }; return { labels: s.freq, datasets: [{ label: 'Spectrum', data: s.db, borderColor: '#42a5f5', borderWidth: 1, pointRadius: 0, tension: 0 }] } })
const spectrumOptions = computed(() => {
  const s = spectrum.value; const anns: Record<string, any> = {}
  if (s) (s.sidebands || []).forEach((f: number, i: number) => { anns[`sb${i}`] = { type: 'line', xMin: f, xMax: f, borderColor: 'rgba(255,167,38,0.7)', borderWidth: 1, borderDash: [4, 3] } })
  return { responsive: true, maintainAspectRatio: false, animation: false as const, scales: { x: { type: 'linear' as const, title: { display: true, text: 'frequency (Hz)' }, ticks: { maxTicksLimit: 12 } }, y: { title: { display: true, text: 'level (dB)' }, suggestedMin: -100, suggestedMax: 5 } }, plugins: { legend: { display: false }, annotation: { annotations: anns }, zoom: { pan: { enabled: true, mode: 'x' as const }, zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' as const } } } }
})
function resetSpecZoom() { specRef.value?.chart?.resetZoom?.() }
const settings = computed(() => result.value?.settings || null)

const evidenceGroups = computed<string[]>(() => result.value?.evidence?.by_group ? Object.keys(result.value.evidence.by_group) : [])
const evidenceFeatures = computed<string[]>(() => result.value?.evidence?.features || [])
const FEATURE_LABELS: Record<string, string> = {
  pump_hf_energy_ratio: 'HF energy ratio (cavitation)', pump_spectral_entropy: 'Spectral entropy',
  pump_spectral_centroid_hz: 'Spectral centroid (Hz)', pump_bpf_db: 'Blade-pass (dB)',
  pump_kurtosis: 'Kurtosis', pump_crest_factor: 'Crest factor', pump_rms: 'RMS',
}
function featureLabel(f: string) { return FEATURE_LABELS[f] || f }
function formatVal(v: any) { return v === undefined || v === null ? '–' : (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v) }
const headlineMetric = computed(() => {
  const m = result.value?.training?.metrics; if (!m) return '—'
  if (result.value.mode === 'classification') { const a = m.accuracy ?? m.test_accuracy; return a != null ? `${(a * 100).toFixed(0)}% accuracy` : '—' }
  const ar = m.anomaly_rate ?? m.contamination; return ar != null ? `${(ar * 100).toFixed(0)}% flagged` : 'baseline trained'
})

const modelName = ref(''); const saving = ref(false); const savedModelId = ref<number | null>(null)
async function saveModel() {
  if (!result.value) return
  const name = modelName.value.trim() || `Pump Cavitation ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`
  saving.value = true
  try {
    const resp = await api.post('/api/solutions/pump/save', { name, training_session_id: result.value.training?.training_session_id, windowed_session_id: result.value.windowed_session_id, feature_session_id: result.value.feature_session_id, settings: result.value.settings })
    savedModelId.value = resp.data.saved_model_id
    notify.showSuccess(`Saved "${resp.data.name}" — deployable from the Deploy page.`)
  } catch (e: any) { notify.showError(e.response?.data?.error || 'Save failed') }
  finally { saving.value = false }
}
</script>

<style scoped lang="scss">
.stat-key { font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 600; }
.stat-val { font-size: 1.05rem; font-weight: 600; }
.mono { font-family: ui-monospace, monospace; }
.diagram-wrap { border: 1px solid rgba(127, 127, 127, 0.25); border-radius: 8px; background: rgba(127, 127, 127, 0.04); }
</style>
