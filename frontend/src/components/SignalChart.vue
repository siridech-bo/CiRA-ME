<template>
  <div>
    <div class="d-flex align-center mb-2" style="gap: 6px; flex-wrap: wrap;">
      <div class="text-subtitle-2">{{ title }}</div>
      <v-spacer />
      <v-chip
        v-for="(ch, i) in channelNames"
        :key="ch"
        size="x-small"
        :variant="hidden[ch] ? 'outlined' : 'flat'"
        :color="hidden[ch] ? undefined : colorFor(i)"
        style="cursor: pointer"
        @click="toggle(ch)"
      >
        <v-icon start size="x-small">{{ hidden[ch] ? 'mdi-eye-off-outline' : 'mdi-eye-outline' }}</v-icon>
        {{ ch }}
      </v-chip>
      <v-btn size="x-small" variant="text" prepend-icon="mdi-magnify-minus-outline" @click="resetZoom">Reset zoom</v-btn>
    </div>
    <div :style="{ height: height + 'px' }">
      <Line ref="chartRef" :data="chartData" :options="chartOptions" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend,
} from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, zoomPlugin)

const props = withDefaults(defineProps<{
  channels: Record<string, number[]>
  fs: number
  title?: string
  yLabel?: string
  initialHidden?: string[]
  height?: number
}>(), { title: 'Signal per window', yLabel: 'signal', height: 240 })

const PALETTE = ['#42a5f5', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc', '#26c6da']
function colorFor(i: number) { return PALETTE[i % PALETTE.length] }

const channelNames = computed(() => Object.keys(props.channels || {}))
const hidden = reactive<Record<string, boolean>>({})
// Apply initialHidden defaults once channels are known / change identity.
watch(channelNames, (names) => {
  for (const n of names) if (!(n in hidden)) hidden[n] = (props.initialHidden || []).includes(n)
}, { immediate: true })
function toggle(name: string) { hidden[name] = !hidden[name] }

const chartRef = ref<any>(null)
function resetZoom() { chartRef.value?.chart?.resetZoom?.() }

const chartData = computed(() => {
  const names = channelNames.value
  const n = names.length ? (props.channels[names[0]]?.length || 0) : 0
  const fs = props.fs || 1
  return {
    labels: Array.from({ length: n }, (_, i) => +(1000 * i / fs).toFixed(2)),
    datasets: names.map((name, i) => ({
      label: name,
      data: props.channels[name],
      borderColor: colorFor(i),
      borderWidth: 1,
      pointRadius: 0,
      tension: 0,
      hidden: !!hidden[name],
    })),
  }
})
const chartOptions = computed(() => ({
  responsive: true, maintainAspectRatio: false, animation: false as const,
  scales: {
    x: { type: 'linear' as const, title: { display: true, text: 'time (ms)' }, ticks: { maxTicksLimit: 10 } },
    y: { title: { display: true, text: props.yLabel } },
  },
  plugins: {
    legend: { position: 'top' as const },
    zoom: {
      pan: { enabled: true, mode: 'x' as const },
      zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' as const },
    },
  },
}))
</script>
