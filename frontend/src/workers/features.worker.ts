/// <reference lib="webworker" />
/**
 * CiRA ME - Fast Mode feature-extraction Web Worker
 *
 * Owner: Fast Mode (Phase 2 hybrid). The main thread posts a batch of raw
 * windows; the worker computes the lightweight feature set and streams
 * progress back every ~100 windows so the UI stays snappy under a 65-user
 * workshop load without blocking the render loop.
 *
 * Message contract is documented in the PLAN file (search for
 * "features.worker.ts"). Any change to the message shapes must be mirrored
 * in `pipeline.ts::extractFeaturesFast`.
 */

import {
  computeWindowFeatures,
  partitionFeatures,
} from '../lib/featureExtraction'

interface ExtractMessage {
  type: 'extract'
  windows: number[][][]
  channelNames: string[]
  selectedFeatures: string[]
  samplingRate: number
  sessionId?: string           // opaque pass-through for the caller
}

interface ProgressMessage {
  type: 'progress'
  done: number
  total: number
}

interface DoneMessage {
  type: 'done'
  session_id: string
  num_windows: number
  num_features: number
  feature_names: string[]
  features_df: number[][]
  feature_set: 'fast_mode'
  extraction_ms: number
}

interface ErrorMessage {
  type: 'error'
  message: string
}

type WorkerOut = ProgressMessage | DoneMessage | ErrorMessage

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope

function post(msg: WorkerOut) {
  ctx.postMessage(msg)
}

ctx.addEventListener('message', (evt: MessageEvent<ExtractMessage>) => {
  const msg = evt.data
  if (!msg || msg.type !== 'extract') return

  const started = performance.now()
  try {
    const { windows, channelNames, selectedFeatures, samplingRate } = msg
    const sessionId = msg.sessionId ?? `fast_${Date.now()}`

    if (!Array.isArray(windows) || windows.length === 0) {
      post({ type: 'error', message: 'No windows provided to worker' })
      return
    }

    // Filter to only client-side portable features. Anything unsupported is
    // silently dropped — the view has already warned the user via the greyed
    // out checkboxes.
    const { supported } = partitionFeatures(selectedFeatures)
    if (supported.length === 0) {
      post({
        type: 'error',
        message: 'No supported features selected for Fast Mode',
      })
      return
    }

    const total = windows.length
    const featuresDf: number[][] = new Array(total)
    let featureNames: string[] = []

    for (let i = 0; i < total; i++) {
      const res = computeWindowFeatures(
        windows[i],
        channelNames,
        supported,
        samplingRate,
      )
      if (i === 0) featureNames = res.feature_names
      featuresDf[i] = res.values

      // Progress every ~100 windows keeps the main thread notified without
      // flooding postMessage (which serializes each call).
      if (i > 0 && i % 100 === 0) {
        post({ type: 'progress', done: i, total })
      }
    }
    // Final progress tick so the bar reaches 100% before the done frame.
    post({ type: 'progress', done: total, total })

    post({
      type: 'done',
      session_id: sessionId,
      num_windows: total,
      num_features: featureNames.length,
      feature_names: featureNames,
      features_df: featuresDf,
      feature_set: 'fast_mode',
      extraction_ms: Math.round(performance.now() - started),
    })
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e)
    post({ type: 'error', message })
  }
})
