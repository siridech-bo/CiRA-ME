<template>
  <div>
    <v-card variant="tonal" class="mb-3">
      <div class="pa-3 d-flex flex-wrap ga-2 align-end">
        <v-select
          v-model="selectedFile"
          :items="csvFiles"
          item-title="name"
          item-value="path"
          label="Data file"
          density="compact"
          hide-details
          style="min-width: 240px"
          :disabled="csvFiles.length === 0"
        />
        <v-select
          v-model="selectedSensor"
          :items="sensorOptions"
          item-title="label"
          item-value="value"
          label="Sensor / column"
          density="compact"
          hide-details
          style="min-width: 220px"
          :disabled="!selectedFile"
        />
        <v-text-field
          v-model.number="maxRows"
          label="Max rows"
          type="number"
          density="compact"
          hide-details
          style="max-width: 140px"
          min="100"
          max="20000"
        />
        <v-spacer />
        <v-btn
          size="small"
          color="primary"
          variant="tonal"
          :loading="loading"
          :disabled="!selectedFile || !selectedSensor"
          prepend-icon="mdi-chart-line"
          @click="loadSeries"
        >
          Plot
        </v-btn>
      </div>
    </v-card>

    <!-- No files -->
    <div v-if="csvFiles.length === 0 && !loadingFiles" class="empty-block">
      <v-icon size="40" color="grey">mdi-file-search-outline</v-icon>
      <p class="text-body-2 text-medium-emphasis mt-2">
        No CSV files found for this machine. Upload data on the Data tab to
        enable history plotting.
      </p>
    </div>

    <!-- Chart -->
    <v-card v-else class="chart-card">
      <div class="pa-3">
        <div v-if="!series.length && !loading" class="empty-block">
          <v-icon size="40" color="grey">mdi-chart-line</v-icon>
          <p class="text-body-2 text-medium-emphasis mt-2">
            {{ selectedFile
              ? 'Select a sensor column and press Plot.'
              : 'Pick a data file to visualize.' }}
          </p>
        </div>
        <Line
          v-else-if="chartData"
          :data="chartData"
          :options="chartOptions"
          :style="{ height: '360px' }"
        />
      </div>
    </v-card>

    <v-alert
      v-if="loadError"
      type="warning"
      density="compact"
      variant="tonal"
      class="mt-3"
    >
      {{ loadError }}
    </v-alert>
  </div>
</template>

<script setup lang="ts">
/**
 * Phase B.8 — History tab.
 * Uses vue-chartjs (already in package.json) to plot one sensor column
 * from a CSV file scoped to this machine's folder. Since the platform
 * has no dedicated time-series store, the file itself is the source of
 * truth — this is honest and covers the workshop use case.
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import api from '@/services/api'
import type { AssetNode } from '@/stores/assetTree'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
)

const props = defineProps<{ machine: AssetNode }>()

interface CsvFile { name: string; path: string; size: number | null }

const csvFiles = ref<CsvFile[]>([])
const loadingFiles = ref(false)
const selectedFile = ref<string | null>(null)
const selectedSensor = ref<string | null>(null)
const maxRows = ref(2000)
const series = ref<Array<{ ts: string | number; value: number }>>([])
const availableColumns = ref<string[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)
const datasetsRoot = ref('')
const separator = ref<'/' | '\\'>('/')

const relPath = computed(() => props.machine.topic_path.replace(/\./g, '/'))
const folderPath = computed(() => {
  if (!datasetsRoot.value) return ''
  const sep = separator.value
  return `${datasetsRoot.value}${sep}${relPath.value.replaceAll('/', sep)}`
})

// Sensor options: prefer the machine's declared sensors, then fall back to
// numeric columns detected in the loaded CSV.
const sensorOptions = computed(() => {
  const machineSensors = (props.machine.children || []).map(s => ({
    value: s.name,
    label: `${s.display_name || s.name}${s.sensor_meta?.unit ? ` (${s.sensor_meta.unit})` : ''}`,
  }))
  const extras = availableColumns.value
    .filter(c => !machineSensors.some(m => m.value === c))
    .map(c => ({ value: c, label: c }))
  return [...machineSensors, ...extras]
})

async function ensureRoot(): Promise<string | null> {
  if (datasetsRoot.value) return datasetsRoot.value
  try {
    const r = await api.get('/api/data/datasets-root')
    datasetsRoot.value = String(r.data?.path || '')
    if (datasetsRoot.value.includes('\\')) separator.value = '\\'
    else separator.value = '/'
    return datasetsRoot.value
  } catch { return null }
}

async function loadFiles() {
  loadingFiles.value = true
  loadError.value = null
  await ensureRoot()
  if (!datasetsRoot.value) { loadingFiles.value = false; return }
  try {
    const r = await api.post('/api/data/browse', { path: folderPath.value })
    const items = r.data?.items || []
    csvFiles.value = items
      .filter((i: any) => !i.is_dir && String(i.extension || '').toLowerCase() === '.csv')
      .map((i: any) => ({ name: i.name, path: i.path, size: i.size }))
    if (csvFiles.value.length > 0 && !selectedFile.value) {
      selectedFile.value = csvFiles.value[0].path
    }
  } catch (e: any) {
    // Empty state is handled visually — don't surface "Path not found" as
    // an error; the user just hasn't uploaded anything yet.
    const err = e.response?.data?.error || ''
    if (!/not found/i.test(err)) {
      loadError.value = err || 'Failed to list data files'
    }
  } finally {
    loadingFiles.value = false
  }
}

async function loadSeries() {
  if (!selectedFile.value || !selectedSensor.value) return
  loading.value = true
  loadError.value = null
  series.value = []
  try {
    const r = await api.post('/api/data/preview', {
      file_path: selectedFile.value,
      rows: Math.max(100, Math.min(20000, Number(maxRows.value) || 2000)),
    })
    const preview: any[] = r.data?.preview || r.data?.data || []
    if (!Array.isArray(preview) || preview.length === 0) {
      loadError.value = 'File has no readable rows.'
      return
    }
    // Discover numeric columns for the sensor dropdown.
    const cols = Object.keys(preview[0])
    availableColumns.value = cols.filter(c => {
      const v = preview[0][c]
      return typeof v === 'number' || (typeof v === 'string' && v.trim() !== '' && !isNaN(Number(v)))
    })

    if (!cols.includes(selectedSensor.value)) {
      loadError.value = `Column "${selectedSensor.value}" not present in file.`
      return
    }

    // Timestamp column heuristic — prefer common names.
    const tsCol = cols.find(c => /^(ts|time|timestamp|date|datetime)$/i.test(c))
    series.value = preview.map((row, i) => ({
      ts: tsCol ? row[tsCol] : i,
      value: Number(row[selectedSensor.value!]),
    })).filter(p => Number.isFinite(p.value))

    if (series.value.length === 0) {
      loadError.value = 'Selected column has no numeric values.'
    }
  } catch (e: any) {
    loadError.value = e.response?.data?.error || 'Failed to load data'
  } finally {
    loading.value = false
  }
}

const chartData = computed(() => {
  if (series.value.length === 0) return null
  const labels = series.value.map(p => String(p.ts))
  return {
    labels,
    datasets: [
      {
        label: selectedSensor.value || '',
        data: series.value.map(p => p.value),
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.15)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
        fill: true,
      },
    ],
  }
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  interaction: { mode: 'index' as const, intersect: false },
  scales: {
    x: {
      ticks: { autoSkip: true, maxTicksLimit: 8 },
      grid: { color: 'rgba(128,128,128,0.12)' },
    },
    y: {
      grid: { color: 'rgba(128,128,128,0.12)' },
    },
  },
  plugins: {
    legend: { display: true, position: 'top' as const },
    tooltip: { enabled: true },
  },
}))

// Default the sensor to the first sensor child so plotting is 1 click away
// once a file loads.
watch(csvFiles, () => {
  if (!selectedSensor.value) {
    const first = (props.machine.children || [])[0]
    if (first) selectedSensor.value = first.name
  }
})

watch(() => props.machine?.id, () => {
  csvFiles.value = []
  selectedFile.value = null
  selectedSensor.value = null
  series.value = []
  // Clear columns discovered from the previous machine — otherwise
  // stale entries linger in the dropdown until a new plot succeeds.
  availableColumns.value = []
  loadFiles()
})

// Read ?sensor= from the URL when this tab mounts (or the query changes)
// so clicking a sensor in the sidebar preselects it here. Falls back to
// the tab's existing default-first-sensor behavior when the query is
// absent or the requested sensor doesn't exist on this machine.
const route = useRoute()
function applySensorQuery() {
  const q = route.query.sensor
  if (typeof q !== 'string' || !q) return
  const match = (props.machine.children || []).find(c => c.name === q)
  if (match) selectedSensor.value = match.name
}
watch(
  () => [route.query.sensor, props.machine?.id],
  applySensorQuery,
  { immediate: true },
)

onMounted(loadFiles)
</script>

<style scoped>
.empty-block {
  text-align: center;
  padding: 32px 16px;
}
.chart-card {
  min-height: 380px;
}
</style>
