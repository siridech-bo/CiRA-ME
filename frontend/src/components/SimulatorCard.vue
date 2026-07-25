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

      <!-- Sparklines — shared component (Phase I Q3, 2026-07-25). -->
      <div class="sparkline-grid">
        <SensorSparkline
          v-for="sensor in instance.sensors"
          :key="sensor.name"
          :name="sensor.name"
          :unit="sensor.unit"
          :channels="sensor.channels || null"
          :values="instance.recent_values || {}"
          :is-chaos="isChaos"
        />
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import SensorSparkline from '@/components/SensorSparkline.vue'

interface SensorInfo {
  name: string
  unit: string
  sample_rate_hz: number
  channels?: string[] | null
}

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
/* Phase I Q3: sparkline chrome moved into SensorSparkline.vue. This card
   only owns the grid layout that arranges N sparklines per instance. */
.sparkline-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
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
