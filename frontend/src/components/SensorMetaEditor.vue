<template>
  <div class="sensor-meta-editor">
    <!-- Unit -->
    <div class="d-flex ga-2 mb-2">
      <v-select
        :model-value="unitSelectValue"
        :items="unitItemsForSelect"
        item-title="label"
        item-value="value"
        label="Unit"
        density="compact"
        class="flex-grow-1"
        hide-details
        @update:model-value="onUnitChange"
      />
    </div>
    <v-text-field
      v-if="customUnitMode"
      v-model="customUnit"
      label="Custom unit"
      density="compact"
      class="mb-2"
      placeholder="e.g. mmol/L"
      hide-details
      autofocus
      @update:model-value="onCustomUnitInput"
    />

    <!-- Sample rate -->
    <div class="d-flex ga-2 mb-2">
      <v-select
        :model-value="rateSelectValue"
        :items="rateItemsForSelect"
        item-title="label"
        item-value="value"
        label="Sample rate (Hz)"
        density="compact"
        class="flex-grow-1"
        hide-details
        @update:model-value="onRateChange"
      />
    </div>
    <v-text-field
      v-if="customRateMode"
      v-model.number="customRate"
      label="Custom Hz"
      type="number"
      min="0"
      density="compact"
      class="mb-2"
      hide-details
      autofocus
      @update:model-value="onCustomRateInput"
    />

    <!-- Data type -->
    <v-select
      :model-value="meta.data_type"
      :items="dataTypes"
      label="Data type"
      density="compact"
      class="mb-2"
      hide-details
      @update:model-value="update({ data_type: $event })"
    />

    <!-- Expected min / max -->
    <div class="d-flex ga-2 mb-3">
      <v-text-field
        :model-value="meta.expected_min"
        label="Expected min"
        type="number"
        density="compact"
        hide-details
        @update:model-value="update({ expected_min: coerceNumber($event) })"
      />
      <v-text-field
        :model-value="meta.expected_max"
        label="Expected max"
        type="number"
        density="compact"
        hide-details
        @update:model-value="update({ expected_max: coerceNumber($event) })"
      />
    </div>

    <!-- Channels (Phase H — multi-axis payloads on ONE topic) -->
    <v-text-field
      :model-value="channelsText"
      label="Channels (comma-separated, blank = single value)"
      placeholder="x, y, z"
      density="compact"
      hide-details="auto"
      hint="For accelerometers/gyroscopes publishing multiple axes in one MQTT payload. Leave blank for standard single-value sensors."
      persistent-hint
      :error-messages="channelsError ? [channelsError] : []"
      @update:model-value="onChannelsInput"
    />
  </div>
</template>

<script setup lang="ts">
/**
 * Sensor meta editor — used at both the wizard's Step 3 and the admin
 * page. Emits an object each time a field changes; parent may reassign
 * the reactive object each render.
 */
import { computed, ref, watch } from 'vue'
import type { SensorMeta } from '@/stores/assetTree'

const props = defineProps<{
  modelValue?: SensorMeta | null
  unitPresets: Array<{ value: string; label: string }>
  ratePresets: number[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: SensorMeta): void
}>()

const CUSTOM_UNIT_MARKER = 'custom'
const CUSTOM_RATE_MARKER = -1

const meta = computed<SensorMeta>(() => props.modelValue || {})

function update(patch: Partial<SensorMeta>) {
  emit('update:modelValue', { ...meta.value, ...patch })
}

// ── Unit ─────────────────────────────────────────────────────────────────
//
// Bug fix (Phase A QA blocker #1): custom-unit mode used to be inferred
// from the emitted value not-matching-a-preset. Selecting "Custom…" first
// emitted unit='' (customUnit is initially empty), which then made
// isCustomUnit=false, hiding the custom text input — a dead end.
// Solution: explicit `customUnitMode` flag toggled by the dropdown handler
// and re-hydrated from the model on external changes. The text input is
// gated on the mode flag, not on the emitted value.

const unitItemsForSelect = computed(() => props.unitPresets)

const customUnit = ref('')
const customUnitMode = ref(false)

// Value shown in the dropdown: either the preset value, or the "custom"
// marker while the user is typing a custom unit. When neither applies,
// null → the dropdown clears.
const unitSelectValue = computed<string | null>(() => {
  if (customUnitMode.value) return CUSTOM_UNIT_MARKER
  const u = meta.value.unit
  if (!u) return null
  const isPresetValue = props.unitPresets.some(p => p.value === u && p.value !== CUSTOM_UNIT_MARKER)
  return isPresetValue ? u : CUSTOM_UNIT_MARKER
})

// Watch the incoming model — if it holds a non-preset value, switch on
// custom mode so a re-mount / external assignment doesn't strand us.
watch(
  () => meta.value.unit,
  (v) => {
    if (!v) {
      if (!customUnitMode.value) customUnit.value = ''
      return
    }
    const isPreset = props.unitPresets.some(p => p.value === v && p.value !== CUSTOM_UNIT_MARKER)
    if (!isPreset) {
      customUnit.value = v
      customUnitMode.value = true
    } else {
      customUnitMode.value = false
      customUnit.value = ''
    }
  },
  { immediate: true },
)

function onUnitChange(v: string | null) {
  if (v === CUSTOM_UNIT_MARKER) {
    customUnitMode.value = true
    // Do NOT emit yet — wait until the user types something in the
    // custom field. Emitting '' would poison the DB with an empty unit.
    return
  }
  customUnitMode.value = false
  customUnit.value = ''
  update({ unit: v || null })
}
function onCustomUnitInput(v: string) {
  const trimmed = (v || '').trim()
  update({ unit: trimmed || null })
}

// ── Sample rate ──────────────────────────────────────────────────────────
// Same fix as Unit — explicit `customRateMode` flag decoupled from
// emitted value.

const rateItemsForSelect = computed(() => {
  const out = props.ratePresets.map(r => ({ value: r as number | string, label: `${r} Hz` }))
  out.push({ value: CUSTOM_RATE_MARKER, label: 'Custom…' })
  return out
})

const customRate = ref<number | null>(null)
const customRateMode = ref(false)

const rateSelectValue = computed<number | null>(() => {
  if (customRateMode.value) return CUSTOM_RATE_MARKER
  const r = meta.value.sample_rate_hz
  if (r == null) return null
  return props.ratePresets.includes(r) ? r : CUSTOM_RATE_MARKER
})

watch(
  () => meta.value.sample_rate_hz,
  (v) => {
    if (v == null) {
      if (!customRateMode.value) customRate.value = null
      return
    }
    if (!props.ratePresets.includes(v)) {
      customRate.value = v
      customRateMode.value = true
    } else {
      customRateMode.value = false
      customRate.value = null
    }
  },
  { immediate: true },
)

function onRateChange(v: number | null) {
  if (v === CUSTOM_RATE_MARKER) {
    customRateMode.value = true
    // Do NOT emit yet — wait until user types a number.
    return
  }
  customRateMode.value = false
  customRate.value = null
  update({ sample_rate_hz: v == null ? null : v })
}
function onCustomRateInput(v: number | string) {
  const num = typeof v === 'number' ? v : parseFloat(v as string)
  update({ sample_rate_hz: Number.isFinite(num) ? num : null })
}

// ── Data type ────────────────────────────────────────────────────────────

const dataTypes = ['float', 'int', 'string']

function coerceNumber(v: number | string | null): number | null {
  if (v == null || v === '') return null
  const n = typeof v === 'number' ? v : parseFloat(v)
  return Number.isFinite(n) ? n : null
}

// ── Channels (Phase H) ───────────────────────────────────────────────────
// Mirrors backend _CHANNEL_NAME_REGEX exactly. Enforced client-side so the
// user gets an immediate inline hint instead of a 400 on save.
const CHANNEL_NAME_REGEX = /^[A-Za-z0-9_]+$/
const MAX_CHANNELS = 16

const channelsText = computed<string>(() => {
  const arr = meta.value.channels
  return Array.isArray(arr) ? arr.join(', ') : ''
})
const channelsError = ref('')

function onChannelsInput(v: string) {
  const raw = (v || '').trim()
  channelsError.value = ''
  if (!raw) {
    update({ channels: null })
    return
  }
  const parts = raw.split(',').map(s => s.trim()).filter(Boolean)
  if (parts.length > MAX_CHANNELS) {
    channelsError.value = `Max ${MAX_CHANNELS} channels`
    return
  }
  const seen = new Set<string>()
  for (const p of parts) {
    if (!CHANNEL_NAME_REGEX.test(p)) {
      channelsError.value = `"${p}" must match letters, digits, underscore`
      return
    }
    if (seen.has(p)) {
      channelsError.value = `Duplicate channel "${p}"`
      return
    }
    seen.add(p)
  }
  update({ channels: parts })
}
</script>

<style scoped>
.sensor-meta-editor {
  display: flex;
  flex-direction: column;
}
</style>
