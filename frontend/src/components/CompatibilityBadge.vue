<template>
  <div class="compat-badge">
    <div v-if="loading" class="d-flex align-center">
      <v-progress-circular indeterminate size="14" width="2" class="mr-2" />
      <span class="text-caption text-medium-emphasis">Checking sensor sets…</span>
    </div>

    <div v-else-if="machineIds.length < 1" class="d-flex align-center">
      <v-chip
        size="small"
        variant="tonal"
        color="grey"
        prepend-icon="mdi-help-circle-outline"
      >
        No machines to check
      </v-chip>
    </div>

    <div v-else-if="machineIds.length === 1" class="d-flex align-center">
      <v-chip
        size="small"
        variant="tonal"
        color="info"
        prepend-icon="mdi-information-outline"
      >
        Single machine — compatibility not applicable
      </v-chip>
    </div>

    <div v-else-if="error" class="d-flex align-center">
      <v-chip
        size="small"
        variant="tonal"
        color="warning"
        prepend-icon="mdi-alert-circle-outline"
      >
        Compatibility check failed
      </v-chip>
      <span class="text-caption text-medium-emphasis ml-2">{{ error }}</span>
    </div>

    <div v-else-if="result?.compatible" class="d-flex align-center">
      <v-chip
        size="small"
        variant="tonal"
        color="success"
        prepend-icon="mdi-check-circle"
      >
        {{ machineIds.length }} machines compatible
      </v-chip>
    </div>

    <div v-else-if="result">
      <div class="d-flex align-center">
        <v-chip
          size="small"
          variant="tonal"
          color="error"
          prepend-icon="mdi-alert-octagon"
          @click="expanded = !expanded"
        >
          Sensor mismatch — click for details
          <v-icon end size="14">
            {{ expanded ? 'mdi-chevron-up' : 'mdi-chevron-down' }}
          </v-icon>
        </v-chip>
      </div>

      <v-expand-transition>
        <v-card
          v-if="expanded"
          variant="tonal"
          color="error"
          class="mt-2 pa-3"
        >
          <div class="text-caption mb-2">
            Reference machine:
            <code>{{ result.reference_machine?.topic_path || '—' }}</code>
          </div>

          <!-- Per-machine diff -->
          <v-table
            v-if="result.per_machine_diff && result.per_machine_diff.length > 0"
            density="compact"
            class="compat-table mb-2"
          >
            <thead>
              <tr>
                <th>Machine</th>
                <th>Missing</th>
                <th>Extra</th>
                <th>Renamed</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in result.per_machine_diff" :key="d.id">
                <td class="text-caption">
                  <code>{{ d.topic_path }}</code>
                </td>
                <td>
                  <v-chip
                    v-for="s in d.missing_sensors"
                    :key="s"
                    size="x-small"
                    variant="tonal"
                    color="error"
                    class="mr-1 mb-1"
                  >
                    {{ s }}
                  </v-chip>
                  <span v-if="!d.missing_sensors || d.missing_sensors.length === 0" class="text-caption text-medium-emphasis">—</span>
                </td>
                <td>
                  <v-chip
                    v-for="s in d.extra_sensors"
                    :key="s"
                    size="x-small"
                    variant="tonal"
                    color="warning"
                    class="mr-1 mb-1"
                  >
                    {{ s }}
                  </v-chip>
                  <span v-if="!d.extra_sensors || d.extra_sensors.length === 0" class="text-caption text-medium-emphasis">—</span>
                </td>
                <td>
                  <span
                    v-for="r in d.renamed_sensors"
                    :key="`${r.from}-${r.to}`"
                    class="text-caption d-block"
                  >
                    <code>{{ r.from }}</code> → <code>{{ r.to }}</code>
                  </span>
                  <span v-if="!d.renamed_sensors || d.renamed_sensors.length === 0" class="text-caption text-medium-emphasis">—</span>
                </td>
              </tr>
            </tbody>
          </v-table>

          <!-- Unit mismatches -->
          <div
            v-if="result.unit_mismatches && result.unit_mismatches.length > 0"
            class="mt-1"
          >
            <div class="text-caption font-weight-bold mb-1">Unit mismatches</div>
            <ul class="text-caption ma-0 pl-4">
              <li
                v-for="(u, i) in result.unit_mismatches"
                :key="`u-${i}`"
              >
                <code>{{ u.sensor }}</code>: reference
                <strong>{{ u.reference_unit || '—' }}</strong>,
                other <strong>{{ u.other_unit || '—' }}</strong>
              </li>
            </ul>
          </div>

          <!-- Sample rate mismatches -->
          <div
            v-if="result.sample_rate_mismatches && result.sample_rate_mismatches.length > 0"
            class="mt-1"
          >
            <div class="text-caption font-weight-bold mb-1">Sample rate mismatches</div>
            <ul class="text-caption ma-0 pl-4">
              <li
                v-for="(s, i) in result.sample_rate_mismatches"
                :key="`s-${i}`"
              >
                <code>{{ s.sensor }}</code>: reference
                <strong>{{ s.reference_sample_rate_hz ?? '—' }} Hz</strong>,
                other <strong>{{ s.other_sample_rate_hz ?? '—' }} Hz</strong>
              </li>
            </ul>
          </div>
        </v-card>
      </v-expand-transition>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Phase C.3 — reusable compatibility badge.
 * Watches `machineIds`, debounces 400ms, calls
 * POST /api/asset-tree/validate-compatibility, renders result inline.
 * Consumers:
 *  - MachineGroupEditDialog (Machine Groups tab)
 *  - TrainingView scope selector
 *
 * Emits `update:compatible` (boolean) so parents can gate a Save button.
 * Single-machine sets short-circuit to "compatibility not applicable" —
 * the backend treats a single-id list as "trivially compatible" but we
 * surface a clearer state.
 */
import { ref, watch, onUnmounted } from 'vue'
import api from '@/services/api'

interface CompatResponse {
  compatible: boolean
  reference_machine?: { id: number; topic_path: string }
  per_machine_diff?: Array<{
    id: number
    topic_path: string
    missing_sensors: string[]
    extra_sensors: string[]
    renamed_sensors: Array<{ from: string; to: string }>
  }>
  unit_mismatches?: Array<{
    sensor: string
    reference_unit?: string | null
    other_unit?: string | null
  }>
  sample_rate_mismatches?: Array<{
    sensor: string
    reference_sample_rate_hz?: number | null
    other_sample_rate_hz?: number | null
  }>
}

const props = defineProps<{
  machineIds: number[]
  autoRun?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:compatible', v: boolean | null): void
  (e: 'update:result', v: CompatResponse | null): void
}>()

const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<CompatResponse | null>(null)
const expanded = ref(false)

let debounceTimer: ReturnType<typeof setTimeout> | null = null

async function runValidation() {
  if (!props.machineIds || props.machineIds.length < 2) {
    result.value = null
    error.value = null
    // A single machine is trivially compatible from the caller's POV.
    emit('update:compatible', props.machineIds.length === 1 ? true : null)
    emit('update:result', null)
    return
  }
  loading.value = true
  error.value = null
  try {
    const r = await api.post('/api/asset-tree/validate-compatibility', {
      machine_asset_ids: props.machineIds,
    })
    result.value = r.data
    emit('update:compatible', !!r.data?.compatible)
    emit('update:result', r.data)
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Request failed'
    result.value = null
    emit('update:compatible', false)
    emit('update:result', null)
  } finally {
    loading.value = false
  }
}

function scheduleRun() {
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(() => {
    void runValidation()
  }, 400)
}

// Auto-run by default; parents can pass :auto-run="false" and invoke via ref.
if (props.autoRun !== false) {
  watch(
    () => props.machineIds,
    () => {
      // Reset expansion on prop change so a fresh problem opens closed.
      expanded.value = false
      scheduleRun()
    },
    { immediate: true, deep: true },
  )
}

onUnmounted(() => {
  if (debounceTimer) clearTimeout(debounceTimer)
})

defineExpose({ runValidation })
</script>

<style scoped>
.compat-table {
  background: rgba(var(--v-theme-surface-bright), 0.5);
  border-radius: 6px;
}
.compat-table :deep(th) {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: rgba(var(--v-theme-on-surface), 0.7);
}
</style>
