<template>
  <v-card class="h-100" :class="{ 'chaos-card': isChaos }">
    <v-card-title class="d-flex align-center">
      <v-icon :color="isChaos ? 'warning' : 'primary'" class="mr-2">
        {{ instance.profile_icon || 'mdi-tune-vertical' }}
      </v-icon>
      <div class="flex-grow-1" style="min-width: 0;">
        <div class="text-truncate">{{ instance.name }}</div>
        <div class="text-caption text-medium-emphasis text-truncate">
          {{ instance.profile_display_name }}
        </div>
      </div>
      <v-chip
        v-if="isChaos"
        color="warning"
        size="x-small"
        variant="tonal"
        class="chaos-badge"
      >
        <v-icon size="12" start>mdi-alert-outline</v-icon>
        chaos {{ instance.chaos_events }}
      </v-chip>
    </v-card-title>

    <v-card-text class="pt-0">
      <!-- Topic + stats -->
      <div class="topic-line mb-2 text-caption text-medium-emphasis">
        <v-icon size="14" class="mr-1">mdi-router-wireless</v-icon>
        <code class="topic-code">{{ instance.topic_base }}</code>
      </div>

      <div class="d-flex align-center flex-wrap ga-2 mb-3">
        <v-chip size="x-small" variant="tonal" color="info">
          {{ instance.messages_published.toLocaleString() }} msgs
        </v-chip>
        <v-chip size="x-small" variant="tonal">
          uptime {{ uptimeDisplay }}
        </v-chip>
        <v-chip size="x-small" variant="tonal" :color="instance.alive ? 'success' : 'error'">
          {{ instance.alive ? 'alive' : 'dead' }}
        </v-chip>
      </div>

      <!-- State controls -->
      <div class="d-flex align-center ga-2 mb-3">
        <v-select
          :model-value="instance.state"
          :items="instance.states"
          label="State"
          density="compact"
          variant="outlined"
          hide-details
          :disabled="!isAdmin"
          @update:model-value="onStateChange"
        />
        <v-menu v-if="isAdmin" location="bottom end">
          <template #activator="{ props: menuProps }">
            <v-btn
              icon
              size="small"
              variant="tonal"
              v-bind="menuProps"
              :title="'More actions'"
            >
              <v-icon>mdi-dots-vertical</v-icon>
            </v-btn>
          </template>
          <v-list density="compact">
            <v-list-item
              prepend-icon="mdi-swap-horizontal"
              title="Change profile…"
              @click="onChangeProfile"
            />
            <v-divider />
            <v-list-item
              prepend-icon="mdi-delete-outline"
              base-color="error"
              title="Delete simulator"
              @click="confirmDelete"
            />
          </v-list>
        </v-menu>
      </div>

      <!-- Sparklines -->
      <div class="sparkline-grid">
        <div
          v-for="sensor in instance.sensors"
          :key="sensor.name"
          class="sparkline-cell"
        >
          <div class="sparkline-label d-flex align-center justify-space-between">
            <span class="text-truncate d-flex align-center">
              {{ sensor.name }}
              <v-chip
                v-if="sensor.channels && sensor.channels.length > 1"
                size="x-small"
                variant="tonal"
                color="primary"
                class="ml-1"
                style="height: 14px; font-size: 9px;"
              >
                {{ sensor.channels.join('/') }}
              </v-chip>
            </span>
            <span class="text-caption text-medium-emphasis ml-2 font-mono">
              {{ lastValueFor(sensor) }}
            </span>
          </div>
          <div class="sparkline-canvas-wrap">
            <Line
              v-if="hasValuesFor(sensor)"
              :data="chartDataFor(sensor)"
              :options="chartOptions"
              class="sparkline-canvas"
            />
            <div v-else class="sparkline-empty text-caption text-medium-emphasis">
              waiting…
            </div>
          </div>
        </div>
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement,
                 Filler, Tooltip)

interface SensorInfo {
  name: string
  unit: string
  sample_rate_hz: number
  channels?: string[] | null
}

// Distinct hues for the axis lines in a multi-axis sparkline. Match the
// convention most IMU tools use — X blue, Y green, Z amber.
const AXIS_COLORS = [
  'rgb(33, 150, 243)',   // blue
  'rgb(76, 175, 80)',    // green
  'rgb(255, 152, 0)',    // amber
  'rgb(233, 30, 99)',    // pink (4th axis, if any)
  'rgb(156, 39, 176)',   // purple
  'rgb(0, 188, 212)',    // cyan
]
interface Instance {
  id: string
  profile_id: string
  profile_display_name: string
  profile_icon: string
  name: string
  topic_base: string
  state: string
  states: string[]
  sensors: SensorInfo[]
  messages_published: number
  chaos_events: number
  created_at: string
  state_since_ts: string
  recent_values: Record<string, number[]>
  alive: boolean
}

const props = defineProps<{
  instance: Instance
  isAdmin: boolean
}>()

const emit = defineEmits<{
  (e: 'patch-state', id: string, newState: string): void
  (e: 'delete', id: string): void
  (e: 'change-profile', id: string): void
}>()

const isChaos = computed(() => props.instance.state === 'chaos')

const uptimeDisplay = computed(() => {
  try {
    const start = Date.parse(props.instance.created_at)
    if (Number.isNaN(start)) return '—'
    const s = Math.max(0, Math.floor((Date.now() - start) / 1000))
    if (s < 60) return `${s}s`
    const m = Math.floor(s / 60)
    if (m < 60) return `${m}m ${s % 60}s`
    const h = Math.floor(m / 60)
    return `${h}h ${m % 60}m`
  } catch {
    return '—'
  }
})

// For a single-value sensor the recent_values key is the sensor name;
// for a multi-axis sensor it's `<name>.<axis>` and there's one entry
// per channel. This helper enumerates the keys the UI should read.
function sparklineKeysFor(sensor: SensorInfo): string[] {
  if (sensor.channels && sensor.channels.length > 0) {
    return sensor.channels.map((axis) => `${sensor.name}.${axis}`)
  }
  return [sensor.name]
}

function hasValuesFor(sensor: SensorInfo): boolean {
  const keys = sparklineKeysFor(sensor)
  return keys.some((k) => {
    const v = props.instance.recent_values?.[k]
    return Array.isArray(v) && v.length > 0
  })
}

function lastValueFor(sensor: SensorInfo): string {
  // Multi-axis: show last value of each axis separated by pipes, e.g.
  // "0.02 | -0.01 | 9.80". Single-value: just the number.
  const keys = sparklineKeysFor(sensor)
  const parts = keys.map((k) => {
    const arr = props.instance.recent_values?.[k] || []
    if (arr.length === 0) return '—'
    const v = arr[arr.length - 1]
    return typeof v === 'number' ? v.toFixed(2) : '—'
  })
  return parts.join(' | ')
}

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: {
    legend: { display: false },
    tooltip: { enabled: false },
  },
  scales: {
    x: { display: false, grid: { display: false } },
    y: { display: false, grid: { display: false } },
  },
  elements: {
    point: { radius: 0 },
    line: { borderWidth: 1.5, tension: 0.35 },
  },
} as any

function chartDataFor(sensor: SensorInfo) {
  const keys = sparklineKeysFor(sensor)
  // Longest per-axis buffer sets the x-axis so a lagging channel doesn't
  // truncate the others.
  const maxLen = Math.max(
    0,
    ...keys.map((k) => (props.instance.recent_values?.[k] || []).length),
  )
  const labels = Array.from({ length: maxLen }, (_, i) => i)

  const isMulti = keys.length > 1
  const datasets = keys.map((k, idx) => {
    const arr = props.instance.recent_values?.[k] || []
    // Left-pad with nulls so all series line up on the right (newest sample).
    const padded = arr.length < maxLen
      ? [...Array(maxLen - arr.length).fill(null), ...arr]
      : arr
    const axisName = k.includes('.') ? k.split('.').pop() : k
    return {
      label: axisName,
      data: padded,
      borderColor: isChaos.value
        ? 'rgb(255, 152, 0)'
        : AXIS_COLORS[idx % AXIS_COLORS.length],
      backgroundColor: isMulti
        ? 'transparent'
        : isChaos.value ? 'rgba(255, 152, 0, 0.15)' : 'rgba(33, 150, 243, 0.15)',
      fill: !isMulti,
      spanGaps: true,
    }
  })
  return { labels, datasets }
}

function onStateChange(v: string) {
  emit('patch-state', props.instance.id, v)
}

function confirmDelete() {
  if (!confirm(`Stop simulator "${props.instance.name}"?`)) return
  emit('delete', props.instance.id)
}

function onChangeProfile() {
  emit('change-profile', props.instance.id)
}
</script>

<style scoped>
.topic-code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 12px;
  word-break: break-all;
}
.sparkline-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
.sparkline-cell {
  border: 1px solid rgba(128, 128, 128, 0.15);
  border-radius: 6px;
  padding: 4px 6px;
  background: rgba(128, 128, 128, 0.04);
}
.sparkline-label {
  font-size: 11px;
  font-weight: 500;
  margin-bottom: 2px;
}
.sparkline-canvas-wrap {
  height: 34px;
  position: relative;
}
.sparkline-canvas {
  width: 100%;
  height: 34px;
}
.sparkline-empty {
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
}
.chaos-card {
  border: 1px solid rgba(255, 152, 0, 0.4);
  box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.08);
}
.chaos-badge {
  animation: pulse-amber 1.4s ease-in-out infinite;
}
@keyframes pulse-amber {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.55; }
}
</style>
