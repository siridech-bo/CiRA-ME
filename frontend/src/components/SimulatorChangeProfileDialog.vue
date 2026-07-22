<template>
  <v-dialog
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    max-width="720"
    persistent
    scrollable
  >
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon color="secondary" class="mr-2">mdi-swap-horizontal</v-icon>
        Change simulator profile
      </v-card-title>
      <v-card-subtitle v-if="instance" class="text-body-2" style="white-space: normal">
        <code>{{ instance.name }}</code> · currently
        <strong>{{ instance.profile_display_name }}</strong> at
        <code>{{ instance.topic_base }}</code>
      </v-card-subtitle>

      <v-card-text style="max-height: 60vh;">
        <v-alert
          type="warning"
          variant="tonal"
          density="compact"
          class="mb-4"
        >
          This will retire the current profile's sensor children under
          <code>{{ instance?.topic_base }}</code> and auto-provision the
          new profile's sensors. Historical CSV data remains on disk but no
          new data will land under the retired sensor paths.
        </v-alert>

        <div v-if="loadingProfiles" class="text-center py-6">
          <v-progress-circular indeterminate size="24" width="2" />
          <span class="ml-2 text-caption">Loading profiles…</span>
        </div>

        <template v-else>
          <!-- Profile picker — matches SimulatorNewDialog's grid layout -->
          <div class="text-caption text-medium-emphasis mb-2 font-weight-medium">
            1. Pick the new profile
          </div>
          <v-row dense class="mb-4">
            <v-col
              v-for="p in profiles"
              :key="p.id"
              cols="6"
              md="4"
            >
              <v-card
                :variant="selectedProfileId === p.id ? 'elevated' : 'outlined'"
                :color="selectedProfileId === p.id ? 'secondary' : undefined"
                class="pa-3 h-100"
                :class="{ 'profile-disabled': p.id === instance?.profile_id }"
                :style="{
                  cursor: p.id === instance?.profile_id ? 'not-allowed' : 'pointer',
                  opacity: p.id === instance?.profile_id ? 0.55 : 1,
                }"
                @click="onSelectProfile(p)"
              >
                <div class="d-flex align-center mb-1">
                  <v-icon
                    :color="selectedProfileId === p.id ? 'white' : 'secondary'"
                    class="mr-2"
                  >
                    {{ p.icon }}
                  </v-icon>
                  <div class="font-weight-medium text-truncate">
                    {{ p.display_name }}
                  </div>
                  <v-spacer />
                  <v-chip
                    v-if="p.id === instance?.profile_id"
                    size="x-small"
                    variant="tonal"
                  >
                    current
                  </v-chip>
                </div>
                <div class="text-caption" style="line-height: 1.2;">
                  {{ p.description }}
                </div>
                <div class="text-caption text-medium-emphasis mt-1">
                  {{ p.sensors.length }} sensors · {{ p.states.length }} states
                </div>
              </v-card>
            </v-col>
          </v-row>

          <template v-if="selectedProfile">
            <!-- State picker -->
            <div class="text-caption text-medium-emphasis mb-2 font-weight-medium">
              2. Initial state after swap
            </div>
            <v-row dense class="mb-2">
              <v-col cols="12" md="6">
                <v-select
                  v-model="targetState"
                  :items="selectedProfile.states"
                  label="Initial state"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
            </v-row>

            <!-- Preview of the swap -->
            <v-alert
              type="info"
              variant="tonal"
              density="compact"
              class="mb-2 mt-3"
            >
              <div class="text-caption font-weight-medium mb-1">
                Sensor children that will be provisioned:
              </div>
              <ul class="preview-list">
                <li
                  v-for="s in selectedProfile.sensors"
                  :key="s.name"
                >
                  <code>{{ instance?.topic_base }}/{{ s.name }}</code>
                  <span class="text-caption text-medium-emphasis ml-2">
                    ({{ s.unit }} · {{ s.sample_rate_hz }} Hz)
                  </span>
                </li>
              </ul>
            </v-alert>
          </template>
        </template>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" :disabled="submitting" @click="close">Cancel</v-btn>
        <v-btn
          color="secondary"
          variant="flat"
          :loading="submitting"
          :disabled="!canSubmit"
          @click="submit"
        >
          Change profile
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'

interface ProfileSensor {
  name: string
  unit: string
  sample_rate_hz: number
}
interface Profile {
  id: string
  display_name: string
  icon: string
  description: string
  sensors: ProfileSensor[]
  states: string[]
  default_state: string
}
interface Instance {
  id: string
  profile_id: string
  profile_display_name: string
  name: string
  topic_base: string
  state: string
}

const props = defineProps<{
  modelValue: boolean
  instance: Instance | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'changed'): void
}>()

const notify = useNotificationStore()

const profiles = ref<Profile[]>([])
const loadingProfiles = ref(false)
const selectedProfileId = ref<string | null>(null)
const targetState = ref<string>('')
const submitting = ref(false)

const selectedProfile = computed(() => {
  if (!selectedProfileId.value) return null
  return profiles.value.find(p => p.id === selectedProfileId.value) || null
})

const canSubmit = computed(() => {
  if (!props.instance) return false
  if (!selectedProfileId.value) return false
  if (selectedProfileId.value === props.instance.profile_id) return false
  if (!targetState.value) return false
  return true
})

function onSelectProfile(p: Profile) {
  // Refuse re-selecting the current profile — backend rejects it too.
  if (p.id === props.instance?.profile_id) return
  selectedProfileId.value = p.id
  // If the previously-typed state exists on the new profile, keep it —
  // otherwise fall back to the profile's default. Matches the backend
  // behavior (which defaults to new profile's default_state when the
  // caller doesn't send `state` or sends an incompatible one).
  if (p.states.includes(targetState.value)) return
  targetState.value = p.default_state
}

async function loadProfiles() {
  loadingProfiles.value = true
  try {
    const r = await api.get<{ profiles: Profile[] }>('/api/simulators/profiles')
    profiles.value = r.data.profiles || []
  } catch (e: any) {
    notify.showError('Failed to load profile catalog')
  } finally {
    loadingProfiles.value = false
  }
}

function reset() {
  selectedProfileId.value = null
  targetState.value = ''
}

function close() {
  reset()
  emit('update:modelValue', false)
}

async function submit() {
  if (!canSubmit.value || !props.instance) return
  submitting.value = true
  try {
    await api.post(
      `/api/simulators/${props.instance.id}/change-profile`,
      {
        profile_id: selectedProfileId.value,
        state: targetState.value,
      },
    )
    notify.showSuccess(
      `Simulator "${props.instance.name}" now on profile "${selectedProfileId.value}".`,
    )
    reset()
    emit('changed')
    emit('update:modelValue', false)
  } catch (e: any) {
    notify.showError(
      e.response?.data?.error || 'Failed to change profile',
    )
  } finally {
    submitting.value = false
  }
}

// Load the profile catalog on first open; cache across open/close.
watch(() => props.modelValue, (v) => {
  if (v && profiles.value.length === 0) loadProfiles()
})
</script>

<style scoped>
.preview-list {
  list-style: disc;
  padding-left: 20px;
  margin: 0;
  max-height: 140px;
  overflow-y: auto;
}
.preview-list li {
  font-size: 12px;
  line-height: 1.6;
}
.profile-disabled {
  pointer-events: none;
}
.profile-disabled >>> * {
  cursor: not-allowed !important;
}
</style>
