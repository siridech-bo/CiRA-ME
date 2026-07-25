<template>
  <div class="sparkline-cell" :class="{ 'chaos-cell': isChaos }">
    <div class="sparkline-label d-flex align-center justify-space-between">
      <span class="text-truncate d-flex align-center">
        {{ name }}
        <v-chip
          v-if="channels && channels.length > 1"
          size="x-small"
          variant="tonal"
          color="primary"
          class="ml-1"
          style="height: 14px; font-size: 9px;"
        >
          {{ channels.join('/') }}
        </v-chip>
        <span
          v-if="unit"
          class="text-caption text-medium-emphasis ml-1"
        >
          {{ unit }}
        </span>
      </span>
      <span class="text-caption text-medium-emphasis ml-2 font-mono">
        {{ lastValueDisplay }}
      </span>
    </div>
    <div class="sparkline-canvas-wrap" :style="{ height: canvasHeight }">
      <Line
        v-if="hasAnyValues"
        :data="chartData"
        :options="chartOptions"
        class="sparkline-canvas"
        :style="{ height: canvasHeight }"
      />
      <div v-else class="sparkline-empty text-caption text-medium-emphasis">
        waiting…
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * SensorSparkline — Phase I Q3 (2026-07-25) shared sparkline.
 *
 * Extracted from SimulatorCard.vue so both the SERVICES-group Machine
 * Simulator page and the new Machine workspace Live tab render the same
 * compact multi-axis-aware sparkline. Rendering rules preserved verbatim:
 *   - single-value sensor → one blue line, fill on
 *   - multi-axis sensor   → blue/green/amber per axis (IMU convention),
 *                           no fill so overlapping lines stay legible
 *   - left-pad shorter buffers with nulls + spanGaps so all lines share
 *     a common right-edge (newest sample)
 *
 * Prop shapes deliberately match how SimulatorCard.vue already keys its
 * `recent_values` map:
 *   - single-value:  values[name] = number[]
 *   - multi-axis:    values[`${name}.${axis}`] = number[]
 */
import { computed } from 'vue'
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

const props = withDefaults(defineProps<{
  name: string
  channels?: string[] | null
  values: Record<string, number[]>
  isChaos?: boolean
  height?: string
  unit?: string
}>(), {
  channels: () => null,
  isChaos: false,
  height: '34px',
  unit: '',
})

const canvasHeight = computed(() => props.height || '34px')

// For a single-value sensor the values key is the sensor name; for a
// multi-axis sensor it's `<name>.<axis>` and there's one entry per channel.
const sparklineKeys = computed<string[]>(() => {
  if (props.channels && props.channels.length > 0) {
    return props.channels.map((axis) => `${props.name}.${axis}`)
  }
  return [props.name]
})

const hasAnyValues = computed<boolean>(() => {
  return sparklineKeys.value.some((k) => {
    const v = props.values?.[k]
    return Array.isArray(v) && v.length > 0
  })
})

const lastValueDisplay = computed<string>(() => {
  // Multi-axis: show last value of each axis separated by pipes.
  const parts = sparklineKeys.value.map((k) => {
    const arr = props.values?.[k] || []
    if (arr.length === 0) return '—'
    const v = arr[arr.length - 1]
    return typeof v === 'number' ? v.toFixed(2) : '—'
  })
  return parts.join(' | ')
})

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

const chartData = computed(() => {
  const keys = sparklineKeys.value
  // Longest per-axis buffer sets the x-axis so a lagging channel doesn't
  // truncate the others.
  const maxLen = Math.max(
    0,
    ...keys.map((k) => (props.values?.[k] || []).length),
  )
  const labels = Array.from({ length: maxLen }, (_, i) => i)

  const isMulti = keys.length > 1
  const datasets = keys.map((k, idx) => {
    const arr = props.values?.[k] || []
    // Left-pad with nulls so all series line up on the right (newest sample).
    // CRITICAL: shallow-copy in BOTH branches. Handing Chart.js the raw
    // reactive array reference (props.values[k]) lets its internal in-place
    // dataset.push() trip Vue's reactive Proxy → invalidates this computed
    // → re-renders → Chart.js re-mounts → mutates again → RangeError
    // "Maximum call stack size exceeded". Simulator card avoided this by
    // polling every 1s (coalesced frames); MachineLivePanel pushes per
    // MQTT message so the loop explodes within ~100 msgs.
    const padded = arr.length < maxLen
      ? [...Array(maxLen - arr.length).fill(null), ...arr]
      : [...arr]
    const axisName = k.includes('.') ? k.split('.').pop() : k
    return {
      label: axisName,
      data: padded,
      borderColor: props.isChaos
        ? 'rgb(255, 152, 0)'
        : AXIS_COLORS[idx % AXIS_COLORS.length],
      backgroundColor: isMulti
        ? 'transparent'
        : props.isChaos ? 'rgba(255, 152, 0, 0.15)' : 'rgba(33, 150, 243, 0.15)',
      fill: !isMulti,
      spanGaps: true,
    }
  })
  return { labels, datasets }
})
</script>

<style scoped>
.sparkline-cell {
  border: 1px solid rgba(128, 128, 128, 0.15);
  border-radius: 6px;
  padding: 4px 6px;
  background: rgba(128, 128, 128, 0.04);
}
.chaos-cell {
  border-color: rgba(255, 152, 0, 0.4);
}
.sparkline-label {
  font-size: 11px;
  font-weight: 500;
  margin-bottom: 2px;
}
.sparkline-canvas-wrap {
  position: relative;
}
.sparkline-canvas {
  width: 100%;
}
.sparkline-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  height: 100%;
}
.font-mono {
  font-family: 'Fira Code', 'Consolas', monospace;
}
</style>
