<template>
  <v-dialog v-model="open" max-width="560" persistent>
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon color="primary" class="mr-2">mdi-source-merge</v-icon>
        <span>Combine sensor files</span>
      </v-card-title>

      <v-card-subtitle class="text-body-2 mt-2" style="white-space: normal">
        You picked <strong>{{ sensors.length }}</strong> CSV files across different
        folders. They'll be joined into one dataset, with each file's <code>value</code>
        column renamed to its parent folder (the sensor name).
      </v-card-subtitle>

      <v-card-text>
        <div class="text-caption text-medium-emphasis mb-1">Sensors detected:</div>
        <div class="mb-4">
          <v-chip
            v-for="s in sensors"
            :key="s"
            size="x-small"
            variant="outlined"
            class="mr-1 mb-1"
          >
            {{ s }}
          </v-chip>
        </div>

        <div class="text-subtitle-2 mb-2">How should timestamps line up?</div>
        <v-radio-group v-model="alignment" density="compact" hide-details>
          <v-radio value="exact">
            <template #label>
              <div>
                <div><strong>Exact match</strong></div>
                <div class="text-caption text-medium-emphasis">
                  Keep only timestamps present in every sensor file. Safest;
                  may drop rows if sensors' clocks aren't perfectly synced.
                </div>
              </div>
            </template>
          </v-radio>
          <v-radio value="nearest" class="mt-2">
            <template #label>
              <div>
                <div><strong>Nearest within tolerance</strong></div>
                <div class="text-caption text-medium-emphasis">
                  Match each row to the nearest timestamp on other sensors,
                  as long as they're within the tolerance below.
                </div>
              </div>
            </template>
          </v-radio>
          <v-radio value="resample" class="mt-2">
            <template #label>
              <div>
                <div><strong>Resample to fixed rate</strong></div>
                <div class="text-caption text-medium-emphasis">
                  Interpolate each sensor onto a common time grid at the rate
                  below. Best when sample rates differ.
                </div>
              </div>
            </template>
          </v-radio>
        </v-radio-group>

        <div v-if="alignment === 'nearest'" class="mt-4">
          <v-text-field
            v-model.number="toleranceMs"
            label="Tolerance (milliseconds)"
            type="number"
            variant="outlined"
            density="compact"
            min="1"
            :rules="[v => (v && v > 0) || 'Enter a positive number']"
            hide-details="auto"
          />
          <div class="text-caption text-medium-emphasis mt-1">
            Two timestamps count as "the same" if they differ by less than this.
            Typical: 100–1000 ms for asynchronous sensors.
          </div>
        </div>

        <div v-if="alignment === 'resample'" class="mt-4">
          <v-text-field
            v-model.number="resampleHz"
            label="Rate (Hz)"
            type="number"
            variant="outlined"
            density="compact"
            min="0.1"
            step="0.1"
            :rules="[v => (v && v > 0) || 'Enter a positive number']"
            hide-details="auto"
          />
          <div class="text-caption text-medium-emphasis mt-1">
            One row per <code>1/{{ resampleHz || '?' }}</code> seconds. Gaps
            are linearly interpolated.
          </div>
        </div>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="cancel">Cancel</v-btn>
        <v-btn color="primary" variant="flat" :disabled="!isValid" @click="confirm">
          Combine
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

interface Props {
  modelValue: boolean
  sensors: string[]
}

const props = defineProps<Props>()
const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'confirm', payload: {
    alignment: 'exact' | 'nearest' | 'resample'
    tolerance_ms: number | null
    resample_hz: number | null
  }): void
  (e: 'cancel'): void
}>()

const open = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
})

const alignment = ref<'exact' | 'nearest' | 'resample'>('exact')
const toleranceMs = ref<number>(500)
const resampleHz = ref<number>(10)

// Reset radio to 'exact' every time the dialog opens so a stale choice from
// a previous session doesn't quietly apply to a different sensor set.
watch(open, (isOpen) => {
  if (isOpen) alignment.value = 'exact'
})

const isValid = computed(() => {
  if (alignment.value === 'nearest') return toleranceMs.value > 0
  if (alignment.value === 'resample') return resampleHz.value > 0
  return true
})

function confirm() {
  emit('confirm', {
    alignment: alignment.value,
    tolerance_ms: alignment.value === 'nearest' ? toleranceMs.value : null,
    resample_hz: alignment.value === 'resample' ? resampleHz.value : null,
  })
  open.value = false
}

function cancel() {
  emit('cancel')
  open.value = false
}
</script>
