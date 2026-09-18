/**
 * Solutions Analysis run — client-side of the async job queue (F8 Phase 1,
 * 2026-09-18).
 *
 * The 4 Solution views (MCSA / Vibration / Pump / PumpFusion) used to
 * `await api.post('/api/solutions/<kind>/run', body)` and pin the result
 * straight to a ref. That worked with 1 user; with 40 concurrent workshop
 * clicks the backend gunicorn thread pool went silent.
 *
 * The endpoint now returns 202 + `{job_id}` immediately and the actual
 * compute runs behind a Semaphore(4) on the server. This composable
 * wraps the submit-then-poll pattern so each view stays clean:
 *
 *   const runner = useSolutionRun('/api/solutions/mcsa/run')
 *   ...
 *   const result = await runner.run(body)
 *   // While `run` is in flight, `runner.status`, `runner.queuePosition`,
 *   // and `runner.elapsedS` update reactively so the view can show a
 *   // "You're #17 in queue" progress card.
 */
import { ref, computed } from 'vue'
import api from '../services/api'

export type SolutionJobStatus = 'idle' | 'queued' | 'running' | 'done' | 'failed'

export function useSolutionRun(runUrl: string) {
  const status = ref<SolutionJobStatus>('idle')
  const queuePosition = ref<number>(0)
  const elapsedS = ref<number>(0)
  const jobId = ref<string | null>(null)
  const error = ref<string | null>(null)

  const isBusy = computed(() =>
    status.value === 'queued' || status.value === 'running'
  )

  // Human-readable progress string for the "Waiting for slot / Running…"
  // card the views render while polling.
  const progressText = computed(() => {
    if (status.value === 'queued') {
      const p = queuePosition.value
      if (p > 1) return `Waiting in queue — position ${p}`
      return 'Next in queue…'
    }
    if (status.value === 'running') {
      return `Analyzing… ${elapsedS.value.toFixed(0)} s`
    }
    if (status.value === 'failed') return error.value || 'Analysis failed'
    return ''
  })

  function reset() {
    status.value = 'idle'
    queuePosition.value = 0
    elapsedS.value = 0
    jobId.value = null
    error.value = null
  }

  /**
   * Submit a Solution analysis and poll until done/failed. Returns the
   * runner's result object on success, throws on failure so the caller
   * can `try/catch` exactly like it did with the old sync API.
   */
  async function run(body: Record<string, unknown>): Promise<unknown> {
    reset()
    status.value = 'queued'
    // 1) Submit → get job_id
    const submit = await api.post(runUrl, body)
    jobId.value = submit.data.job_id
    queuePosition.value = submit.data.queue_position ?? 0
    status.value = submit.data.status || 'queued'
    // 2) Poll until terminal state. 500 ms interval keeps the queue
    //    position responsive without hammering the backend during a
    //    workshop storm (a 40-user poll @ 500 ms = 80 req/s to a cheap
    //    endpoint — trivial).
    while (true) {
      await new Promise((r) => setTimeout(r, 500))
      const poll = await api.get(`/api/solutions/jobs/${jobId.value}`)
      const s = poll.data.status as SolutionJobStatus
      status.value = s
      queuePosition.value = poll.data.queue_position ?? 0
      elapsedS.value = poll.data.elapsed_s ?? 0
      if (s === 'done') {
        return poll.data.result
      }
      if (s === 'failed') {
        error.value = poll.data.error || 'Analysis failed'
        throw new Error(error.value ?? 'Analysis failed')
      }
      // status still 'queued' or 'running' → keep polling
    }
  }

  return {
    // reactive state
    status,
    queuePosition,
    elapsedS,
    jobId,
    error,
    isBusy,
    progressText,
    // actions
    run,
    reset,
  }
}
