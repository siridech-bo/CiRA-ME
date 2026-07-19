<template>
  <v-container fluid class="pa-6">
    <!-- Header bar -->
    <div class="d-flex align-center flex-wrap mb-4 ga-3">
      <div>
        <h1 class="text-h4 font-weight-bold">Machine Simulators</h1>
        <p class="text-body-2 text-medium-emphasis mb-0">
          Server-side signal generators that publish to Mosquitto. Use these to
          demo the ingest pipeline without real hardware.
        </p>
      </div>
      <v-spacer />

      <!-- Broker connection dot -->
      <v-chip
        :color="snapshot.connected ? 'success' : 'error'"
        variant="tonal"
        size="small"
        :prepend-icon="snapshot.connected ? 'mdi-check-circle' : 'mdi-alert-circle'"
      >
        {{ snapshot.connected ? 'Broker connected' : 'Broker disconnected' }}
      </v-chip>

      <!-- Rate ticker -->
      <v-chip
        color="info"
        variant="tonal"
        size="small"
        prepend-icon="mdi-swap-vertical-bold"
      >
        {{ rateDisplay }} msg/s
      </v-chip>

      <v-chip
        size="small"
        variant="tonal"
        prepend-icon="mdi-view-grid"
      >
        {{ snapshot.instance_count || 0 }} instances
      </v-chip>

      <v-chip
        v-if="totalChaos > 0"
        color="warning"
        size="small"
        variant="tonal"
        prepend-icon="mdi-alert-outline"
      >
        {{ totalChaos }} chaos events
      </v-chip>

      <v-btn
        v-if="isAdmin"
        color="primary"
        prepend-icon="mdi-plus"
        @click="showNewDialog = true"
      >
        Add machine
      </v-btn>

      <v-btn
        v-if="isAdmin && instances.length > 0"
        color="error"
        variant="tonal"
        prepend-icon="mdi-stop-circle-outline"
        @click="stopAll"
        :loading="stoppingAll"
      >
        Stop all
      </v-btn>

      <v-chip
        v-if="!isAdmin"
        color="warning"
        size="small"
        variant="tonal"
        prepend-icon="mdi-eye"
      >
        Read-only — admins can edit
      </v-chip>
    </div>

    <!-- Empty state -->
    <v-card
      v-if="instances.length === 0 && !loadingList"
      variant="tonal"
      color="surface"
      class="pa-8 text-center"
    >
      <v-icon size="64" color="grey">mdi-tune-vertical</v-icon>
      <p class="text-body-1 mt-4 mb-1">No simulators running.</p>
      <p class="text-body-2 text-medium-emphasis mb-4">
        Spin up a virtual machine — it publishes realistic sensor data to
        Mosquitto so downstream ingest, training and inference all work.
      </p>
      <v-btn
        v-if="isAdmin"
        color="primary"
        prepend-icon="mdi-plus"
        @click="showNewDialog = true"
      >
        Add your first machine
      </v-btn>
    </v-card>

    <!-- Card grid -->
    <v-row v-else dense>
      <v-col
        v-for="inst in instances"
        :key="inst.id"
        cols="12"
        md="6"
        xl="4"
      >
        <SimulatorCard
          :instance="inst"
          :is-admin="isAdmin"
          @patch-state="onPatchState"
          @delete="onDelete"
        />
      </v-col>
    </v-row>

    <!-- Raw-publish widget (admin only) -->
    <v-divider v-if="isAdmin" class="my-6" />

    <v-card v-if="isAdmin" class="mt-2">
      <v-card-title class="d-flex align-center">
        <v-icon color="warning" class="mr-2">mdi-flash</v-icon>
        Raw publish
        <v-spacer />
        <v-chip size="x-small" variant="tonal" color="warning">
          admin
        </v-chip>
      </v-card-title>
      <v-card-text>
        <p class="text-body-2 text-medium-emphasis mb-3">
          One-shot arbitrary MQTT publish. Useful for probing the ingest
          router's parse and rejection paths. Escape bytes with
          <code>\x00</code>-style sequences.
        </p>
        <v-row>
          <v-col cols="12" md="6">
            <v-text-field
              v-model="rawTopic"
              label="Topic"
              placeholder="factory/plant_A/machine_01/temperature"
              density="compact"
              variant="outlined"
              hide-details
            />
          </v-col>
          <v-col cols="12" md="6">
            <v-textarea
              v-model="rawPayload"
              label='Payload (JSON, plain text, or \x00 escapes)'
              placeholder='{"value": 42.0}'
              density="compact"
              variant="outlined"
              rows="2"
              auto-grow
              hide-details
            />
          </v-col>
        </v-row>
        <div class="d-flex justify-end mt-3">
          <v-btn
            color="warning"
            variant="tonal"
            prepend-icon="mdi-rocket-launch-outline"
            @click="fireRawPublish"
            :loading="firingRaw"
            :disabled="!rawTopic.trim()"
          >
            Fire
          </v-btn>
        </div>
      </v-card-text>
    </v-card>

    <!-- New-simulator dialog -->
    <SimulatorNewDialog
      v-model="showNewDialog"
      :root-name="rootName"
      @created="onCreated"
    />
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import api from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { useAssetTreeStore } from '@/stores/assetTree'
import { useNotificationStore } from '@/stores/notification'
import SimulatorCard from '@/components/SimulatorCard.vue'
import SimulatorNewDialog from '@/components/SimulatorNewDialog.vue'

interface SensorInfo {
  name: string
  unit: string
  sample_rate_hz: number
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

interface Snapshot {
  connected: boolean
  broker_host: string
  broker_port: number
  instance_count: number
  messages_published: number
  chaos_events: number
  instances: Instance[]
  started_at?: string | null
  last_connected_at?: string | null
}

const authStore = useAuthStore()
const assetTreeStore = useAssetTreeStore()
const notify = useNotificationStore()

const isAdmin = computed(() => authStore.isAdmin)
const rootName = computed(() => assetTreeStore.config?.root_name || 'factory')

const instances = ref<Instance[]>([])
const snapshot = ref<Snapshot>({
  connected: false,
  broker_host: '',
  broker_port: 0,
  instance_count: 0,
  messages_published: 0,
  chaos_events: 0,
  instances: [],
})
const loadingList = ref(false)
const showNewDialog = ref(false)

// Raw-publish widget state
const rawTopic = ref('')
const rawPayload = ref('')
const firingRaw = ref(false)
const stoppingAll = ref(false)

// Rate ticker — computed from published-message delta / interval.
const lastPublishedTotal = ref(0)
const lastPollAt = ref(Date.now())
const rateMsgS = ref(0)

const totalChaos = computed(() => snapshot.value.chaos_events || 0)
const rateDisplay = computed(() => rateMsgS.value.toFixed(1))

let pollTimer: number | null = null

async function refresh() {
  try {
    const r = await api.get<Snapshot>('/api/simulators/snapshot')
    const now = Date.now()
    const elapsedS = Math.max(0.001, (now - lastPollAt.value) / 1000)
    const publishedNow = r.data.messages_published || 0
    const delta = Math.max(0, publishedNow - lastPublishedTotal.value)
    // First tick primes the counter without an inflated spike.
    if (lastPublishedTotal.value > 0) {
      rateMsgS.value = delta / elapsedS
    }
    lastPublishedTotal.value = publishedNow
    lastPollAt.value = now

    snapshot.value = r.data
    instances.value = r.data.instances || []
  } catch (e: any) {
    // Silent — poller retries next tick. Only surface hard errors on user
    // actions.
  }
}

async function onPatchState(id: string, newState: string) {
  try {
    await api.patch(`/api/simulators/${id}`, { state: newState })
    notify.showSuccess(`State changed to "${newState}".`)
    await refresh()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to change state')
  }
}

async function onDelete(id: string) {
  try {
    await api.delete(`/api/simulators/${id}`)
    notify.showSuccess('Simulator stopped.')
    await refresh()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to delete simulator')
  }
}

async function onCreated() {
  showNewDialog.value = false
  await refresh()
  // Nudge the tree store so newly auto-provisioned nodes show in the sidebar.
  try {
    assetTreeStore.invalidateTree?.()
    await assetTreeStore.fetchTree?.()
  } catch {
    /* store may not expose invalidate on older builds */
  }
}

async function stopAll() {
  if (!confirm(`Stop all ${instances.value.length} simulators?`)) return
  stoppingAll.value = true
  try {
    // Fire deletes in parallel — a slow one won't block the batch.
    await Promise.all(instances.value.map(i => api.delete(`/api/simulators/${i.id}`)))
    notify.showSuccess('All simulators stopped.')
    await refresh()
  } catch (e: any) {
    notify.showError('Some simulators failed to stop; refreshing…')
    await refresh()
  } finally {
    stoppingAll.value = false
  }
}

async function fireRawPublish() {
  firingRaw.value = true
  try {
    await api.post('/api/simulators/publish-raw', {
      topic: rawTopic.value.trim(),
      payload: rawPayload.value,
    })
    notify.showSuccess(`Published to ${rawTopic.value.trim()}`)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Publish failed')
  } finally {
    firingRaw.value = false
  }
}

onMounted(async () => {
  loadingList.value = true
  await refresh()
  loadingList.value = false
  // 1 s poll matches spec §5.2.
  pollTimer = window.setInterval(refresh, 1000)
})

onBeforeUnmount(() => {
  if (pollTimer) window.clearInterval(pollTimer)
})
</script>
