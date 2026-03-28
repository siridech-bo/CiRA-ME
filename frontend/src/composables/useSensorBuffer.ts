/**
 * CiRA ME - Sensor Buffer Composable
 * Accumulates sensor samples into windows for ML inference.
 * When a window is full, triggers the onWindowReady callback.
 */

import { ref, computed } from 'vue'

export interface SensorSample {
  [channel: string]: number
}

export interface BufferConfig {
  windowSize: number
  stride: number
  channels: string[]
}

export function useSensorBuffer(
  config: BufferConfig,
  onWindowReady: (windowData: number[][]) => void
) {
  const buffer = ref<SensorSample[]>([])
  const windowCount = ref(0)
  const totalSamples = ref(0)
  const lastInferenceTime = ref<number | null>(null)

  const bufferProgress = computed(() => {
    return Math.min(buffer.value.length / config.windowSize, 1)
  })

  const bufferFull = computed(() => {
    return buffer.value.length >= config.windowSize
  })

  function push(sample: SensorSample) {
    buffer.value.push(sample)
    totalSamples.value++

    if (buffer.value.length >= config.windowSize) {
      // Extract window as 2D array [window_size × channels]
      const window = buffer.value.slice(0, config.windowSize)
      const windowData = window.map(s =>
        config.channels.map(ch => s[ch] ?? 0)
      )

      // Trigger inference callback
      windowCount.value++
      lastInferenceTime.value = Date.now()
      onWindowReady(windowData)

      // Slide by stride
      buffer.value = buffer.value.slice(config.stride)
    }
  }

  function pushBatch(samples: SensorSample[]) {
    for (const s of samples) {
      push(s)
    }
  }

  function clear() {
    buffer.value = []
    windowCount.value = 0
    totalSamples.value = 0
    lastInferenceTime.value = null
  }

  // Get recent raw values for charting (last N samples)
  function getRecentValues(channel: string, maxPoints: number = 500): number[] {
    const start = Math.max(0, buffer.value.length - maxPoints)
    return buffer.value.slice(start).map(s => s[channel] ?? 0)
  }

  return {
    buffer,
    bufferProgress,
    bufferFull,
    windowCount,
    totalSamples,
    lastInferenceTime,
    push,
    pushBatch,
    clear,
    getRecentValues,
  }
}
