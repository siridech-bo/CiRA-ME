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
        <v-icon color="primary" class="mr-2">mdi-tune-vertical</v-icon>
        New machine simulator
      </v-card-title>

      <v-card-text style="max-height: 68vh;">
        <div v-if="loadingProfiles" class="text-center py-6">
          <v-progress-circular indeterminate size="24" width="2" />
          <span class="ml-2 text-caption">Loading profiles…</span>
        </div>

        <template v-else>
          <!-- Step 1: profile picker -->
          <div class="text-caption text-medium-emphasis mb-2 font-weight-medium">
            1. Pick a machine profile
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
                :color="selectedProfileId === p.id ? 'primary' : undefined"
                class="pa-3 h-100"
                style="cursor: pointer;"
                @click="onSelectProfile(p.id)"
              >
                <div class="d-flex align-center mb-1">
                  <v-icon
                    :color="selectedProfileId === p.id ? 'white' : 'primary'"
                    class="mr-2"
                  >
                    {{ p.icon }}
                  </v-icon>
                  <div class="font-weight-medium text-truncate">
                    {{ p.display_name }}
                  </div>
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

          <!-- Step 2: name + topic_base -->
          <template v-if="selectedProfile">
            <div class="text-caption text-medium-emphasis mb-2 font-weight-medium">
              2. Name + topic base
            </div>
            <v-row dense class="mb-1">
              <v-col cols="12" md="5">
                <v-text-field
                  v-model="name"
                  label="Instance name"
                  hint="Segment-safe [A-Za-z0-9_-]+"
                  density="compact"
                  variant="outlined"
                  persistent-hint
                  :rules="[nameRule]"
                />
              </v-col>
              <v-col cols="12" md="7">
                <v-text-field
                  v-model="topicBase"
                  label="Topic base"
                  :hint="`Must start with '${rootName}/'`"
                  density="compact"
                  variant="outlined"
                  persistent-hint
                  :rules="[topicRule]"
                />
              </v-col>
            </v-row>

            <!-- Step 3: initial state + autoprovision -->
            <v-row dense class="mb-2">
              <v-col cols="12" md="6">
                <v-select
                  v-model="initialState"
                  :items="selectedProfile.states"
                  label="Initial state"
                  density="compact"
                  variant="outlined"
                />
              </v-col>
              <v-col cols="12" md="6" class="d-flex align-center">
                <v-checkbox
                  v-model="autoprovision"
                  label="Auto-provision asset-tree nodes"
                  density="compact"
                  hide-details
                />
              </v-col>
            </v-row>

            <!-- Preview -->
            <v-alert
              v-if="autoprovision"
              type="info"
              variant="tonal"
              density="compact"
              class="mb-2"
            >
              <div class="text-caption font-weight-medium mb-1">
                Nodes that will be created (if missing):
              </div>
              <ul class="preview-list">
                <li v-for="seg in previewPath" :key="seg">
                  <code>{{ seg }}</code>
                </li>
                <li
                  v-for="s in selectedProfile.sensors"
                  :key="s.name"
                >
                  <code>{{ topicBase }}/{{ s.name }}</code>
                  <span class="text-caption text-medium-emphasis ml-2">
                    ({{ s.unit }} · {{ s.sample_rate_hz }} Hz)
                  </span>
                </li>
              </ul>
            </v-alert>
            <v-alert
              v-else
              type="warning"
              variant="tonal"
              density="compact"
              class="mb-2"
            >
              Autoprovision is OFF — messages will be rejected (strict mode)
              unless the tree already contains these nodes.
            </v-alert>
          </template>
        </template>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="close">Cancel</v-btn>
        <v-btn
          color="primary"
          :loading="creating"
          :disabled="!canCreate"
          @click="submit"
        >
          Create + start
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
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

const props = defineProps<{
  modelValue: boolean
  rootName: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'created'): void
}>()

const notify = useNotificationStore()

const profiles = ref<Profile[]>([])
const loadingProfiles = ref(false)
const selectedProfileId = ref<string | null>(null)
const name = ref('')
const topicBase = ref('')
const initialState = ref<string>('')
const autoprovision = ref(true)
const creating = ref(false)

const selectedProfile = computed(() => {
  if (!selectedProfileId.value) return null
  return profiles.value.find(p => p.id === selectedProfileId.value) || null
})

const previewPath = computed(() => {
  const segs = topicBase.value.split('/').filter(Boolean)
  const out: string[] = []
  for (let i = 0; i < segs.length; i++) {
    out.push(segs.slice(0, i + 1).join('/'))
  }
  return out
})

const NAME_RE = /^[A-Za-z0-9_-]+$/
const nameRule = (v: string) => {
  if (!v) return 'Required'
  if (!NAME_RE.test(v)) return 'Must match [A-Za-z0-9_-]+'
  return true
}
const topicRule = (v: string) => {
  if (!v) return 'Required'
  const trimmed = v.trim()
  if (!trimmed.startsWith(props.rootName + '/')) {
    return `Must start with '${props.rootName}/'`
  }
  const segs = trimmed.split('/').filter(Boolean)
  for (const s of segs) {
    if (!NAME_RE.test(s)) return `Segment '${s}' must match [A-Za-z0-9_-]+`
  }
  return true
}

const canCreate = computed(() => {
  if (!selectedProfile.value) return false
  if (nameRule(name.value) !== true) return false
  if (topicRule(topicBase.value) !== true) return false
  if (!initialState.value) return false
  return true
})

function onSelectProfile(id: string) {
  selectedProfileId.value = id
  const p = profiles.value.find(x => x.id === id)
  if (p) {
    initialState.value = p.default_state
    // Prefill name + topic if empty.
    if (!name.value) name.value = `${p.id}_01`
    if (!topicBase.value) {
      topicBase.value = `${props.rootName}/plant_A/${name.value}`
    }
  }
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
  name.value = ''
  topicBase.value = ''
  initialState.value = ''
  autoprovision.value = true
}

function close() {
  reset()
  emit('update:modelValue', false)
}

async function submit() {
  if (!canCreate.value || !selectedProfile.value) return
  creating.value = true
  try {
    await api.post('/api/simulators/', {
      profile_id: selectedProfile.value.id,
      name: name.value.trim(),
      topic_base: topicBase.value.trim(),
      initial_state: initialState.value,
      autoprovision_tree: autoprovision.value,
    })
    notify.showSuccess(`Simulator "${name.value.trim()}" started.`)
    reset()
    emit('created')
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to create simulator')
  } finally {
    creating.value = false
  }
}

// Load profiles when the dialog opens; keep them cached across open/close.
watch(() => props.modelValue, (v) => {
  if (v && profiles.value.length === 0) loadProfiles()
  if (v && !name.value && selectedProfileId.value === null) {
    // Fresh open — nothing to do.
  }
})

onMounted(() => {
  if (props.modelValue) loadProfiles()
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
</style>
