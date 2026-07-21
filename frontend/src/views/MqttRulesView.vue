<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center flex-wrap mb-4 ga-2">
      <div>
        <h1 class="text-h4 font-weight-bold">MQTT Rules</h1>
        <p class="text-body-2 text-medium-emphasis mb-0">
          Ingest router config, rejected-topic log, and live router stats.
          Messages published to Mosquitto are routed by topic path to the
          asset tree and written as daily CSVs.
        </p>
      </div>
      <v-spacer />
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

    <v-tabs v-model="activeTab" density="compact" class="mb-4">
      <v-tab value="setup" prepend-icon="mdi-rocket-launch-outline">Setup</v-tab>
      <v-tab value="config" prepend-icon="mdi-cog">Config</v-tab>
      <v-tab value="rejected" prepend-icon="mdi-cancel">Rejected</v-tab>
      <v-tab value="stats" prepend-icon="mdi-chart-line">Stats</v-tab>
    </v-tabs>

    <v-window v-model="activeTab">
      <!-- ── Setup tab (landing) ────────────────────────────────────── -->
      <v-window-item value="setup">
        <MqttSetupTab
          :config="config"
          @go-to-rejected="activeTab = 'rejected'"
        />
      </v-window-item>

      <!-- ── Config tab ─────────────────────────────────────────────── -->
      <v-window-item value="config">
        <v-alert
          type="info"
          variant="tonal"
          density="compact"
          class="mb-4"
          icon="mdi-shield-alert-outline"
        >
          <strong>Advanced settings</strong> — most users don't need to
          change these. If you're new to CiRA ME, start with the
          <a
            href="#"
            class="text-decoration-underline"
            @click.prevent="activeTab = 'setup'"
          >Setup tab</a>
          instead.
        </v-alert>
        <v-card>
          <v-card-text>
            <div v-if="configLoading" class="pa-6 text-center text-caption">
              <v-progress-circular indeterminate size="20" width="2" class="mr-2" />
              Loading config…
            </div>
            <div v-else-if="!config" class="pa-6 text-center">
              <v-icon size="48" color="grey">mdi-file-tree-outline</v-icon>
              <p class="text-body-1 mt-3 mb-1">Asset tree not yet configured.</p>
              <p class="text-body-2 text-medium-emphasis">
                Run the first-run wizard to create a tree — MQTT ingest
                depends on it.
              </p>
              <v-btn
                color="primary"
                variant="tonal"
                class="mt-3"
                :to="{ name: 'asset-tree-setup' }"
              >
                Go to setup wizard
              </v-btn>
            </div>
            <template v-else>
              <v-row>
                <v-col cols="12" md="6">
                  <div class="text-subtitle-2 mb-2">
                    <v-icon size="16" class="mr-1">mdi-toggle-switch-outline</v-icon>
                    Ingest routing
                  </div>
                  <v-switch
                    v-model="editable.ingest_enabled"
                    :disabled="!isAdmin"
                    :label="editable.ingest_enabled ? 'Enabled — messages routed to CSV' : 'Disabled — router idles (still connected)'"
                    color="primary"
                    density="comfortable"
                    hide-details
                    class="mb-2"
                  />
                  <p class="text-caption text-medium-emphasis">
                    When disabled, the router stays subscribed but skips all
                    writes. Flip this to pause ingest during
                    maintenance without losing broker connection.
                  </p>
                </v-col>
                <v-col cols="12" md="6">
                  <div class="text-subtitle-2 mb-2">
                    <v-icon size="16" class="mr-1">mdi-shield-check-outline</v-icon>
                    Topic mode
                  </div>
                  <v-radio-group
                    v-model="editable.topic_mode"
                    :disabled="!isAdmin"
                    density="comfortable"
                    hide-details
                    class="mt-0"
                  >
                    <v-radio value="strict" label="Strict — only pre-registered topics" />
                    <v-radio value="learn" label="Learn — auto-create tree nodes on first message" />
                  </v-radio-group>
                  <p class="text-caption text-medium-emphasis mt-2">
                    Learn mode is useful for onboarding new machines fast;
                    switch back to Strict once the fleet is stable.
                  </p>
                </v-col>
              </v-row>

              <v-divider class="my-4" />

              <v-row>
                <v-col cols="12" md="8">
                  <div class="text-subtitle-2 mb-2">
                    <v-icon size="16" class="mr-1">mdi-tag-multiple-outline</v-icon>
                    Meta prefixes (comma-separated)
                  </div>
                  <v-text-field
                    v-model="metaPrefixesText"
                    :disabled="!isAdmin"
                    variant="outlined"
                    density="comfortable"
                    hide-details
                    placeholder="_meta, _health, _config, _cmd"
                  />
                  <p class="text-caption text-medium-emphasis mt-1">
                    Any segment matching one of these prefixes accepts the
                    topic but skips the CSV write. Used for heartbeat /
                    config / command channels that share the tree namespace.
                  </p>
                </v-col>
                <v-col cols="12" md="4">
                  <div class="text-subtitle-2 mb-2">
                    <v-icon size="16" class="mr-1">mdi-calendar-clock</v-icon>
                    Retention (days)
                  </div>
                  <v-text-field
                    v-model.number="editable.ingest_retention_days"
                    :disabled="!isAdmin"
                    type="number"
                    :min="1"
                    :max="3650"
                    variant="outlined"
                    density="comfortable"
                    hide-details
                  />
                  <p class="text-caption text-medium-emphasis mt-1">
                    Daily CSVs and rejected-topic logs older than this get
                    physically deleted every 6 h.
                  </p>
                </v-col>
              </v-row>

              <div class="d-flex align-center mt-6">
                <v-chip
                  size="small"
                  variant="tonal"
                  :color="config.topic_mode === 'strict' ? 'info' : 'warning'"
                  class="mr-2"
                >
                  Current: {{ config.topic_mode }} mode,
                  {{ config.ingest_enabled === false ? 'ingest OFF' : 'ingest ON' }}
                </v-chip>
                <v-chip
                  size="small"
                  variant="text"
                  class="text-caption text-medium-emphasis"
                >
                  root: <code class="ml-1">{{ config.root_name }}</code>
                </v-chip>
                <v-spacer />
                <v-btn
                  v-if="isAdmin"
                  variant="text"
                  :disabled="!hasChanges"
                  @click="resetChanges"
                >
                  Reset
                </v-btn>
                <v-btn
                  v-if="isAdmin"
                  color="primary"
                  :loading="saving"
                  :disabled="!hasChanges"
                  @click="saveConfig"
                >
                  Save changes
                </v-btn>
              </div>
            </template>
          </v-card-text>
        </v-card>
      </v-window-item>

      <!-- ── Rejected tab ───────────────────────────────────────────── -->
      <v-window-item value="rejected">
        <v-card>
          <v-card-text>
            <div class="d-flex align-center flex-wrap ga-3 mb-3">
              <v-text-field
                v-model="rejectedDate"
                label="Date (UTC)"
                type="date"
                variant="outlined"
                density="compact"
                hide-details
                style="max-width: 200px"
              />
              <v-select
                v-model="rejectedLimit"
                :items="[50, 100, 200, 500, 1000]"
                label="Limit"
                variant="outlined"
                density="compact"
                hide-details
                style="max-width: 120px"
              />
              <v-btn
                variant="tonal"
                prepend-icon="mdi-refresh"
                :loading="rejectedLoading"
                @click="fetchRejected"
              >
                Refresh
              </v-btn>
              <v-spacer />
              <span class="text-caption text-medium-emphasis">
                {{ rejectedEntries.length }} entries
              </span>
            </div>

            <div v-if="rejectedLoading" class="pa-6 text-center text-caption">
              <v-progress-circular indeterminate size="20" width="2" class="mr-2" />
              Loading…
            </div>
            <div
              v-else-if="rejectedEntries.length === 0"
              class="pa-8 text-center"
            >
              <v-icon size="48" color="success">mdi-check-circle-outline</v-icon>
              <p class="text-body-1 mt-3 mb-1">No rejections on this date.</p>
              <p class="text-body-2 text-medium-emphasis">
                Either everything is going through cleanly, or the router
                hasn't seen any traffic yet.
              </p>
            </div>
            <div v-else class="rejected-scroll">
              <v-table density="compact">
                <thead>
                  <tr>
                    <th style="width: 220px">Timestamp (UTC)</th>
                    <th>Topic</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(e, idx) in rejectedEntries" :key="idx">
                    <td class="text-caption">{{ e.timestamp }}</td>
                    <td class="text-caption"><code>{{ e.topic }}</code></td>
                    <td class="text-caption text-medium-emphasis">
                      {{ e.reason }}
                    </td>
                  </tr>
                </tbody>
              </v-table>
            </div>
          </v-card-text>
        </v-card>
      </v-window-item>

      <!-- ── Stats tab ──────────────────────────────────────────────── -->
      <v-window-item value="stats">
        <v-card>
          <v-card-text>
            <div class="d-flex align-center mb-3">
              <div class="text-subtitle-2">Router status</div>
              <v-spacer />
              <v-btn
                v-if="isAdmin"
                variant="text"
                size="small"
                prepend-icon="mdi-broom"
                :loading="janitorRunning"
                @click="runJanitor"
              >
                Run retention sweep
              </v-btn>
              <v-btn
                variant="tonal"
                size="small"
                prepend-icon="mdi-refresh"
                :loading="statsLoading"
                @click="fetchStats"
              >
                Refresh
              </v-btn>
            </div>

            <div v-if="statsLoading && !stats" class="pa-6 text-center text-caption">
              <v-progress-circular indeterminate size="20" width="2" class="mr-2" />
              Loading stats…
            </div>
            <template v-else-if="stats">
              <v-row dense>
                <v-col cols="6" md="3">
                  <v-card variant="tonal" color="primary">
                    <v-card-text class="pa-3">
                      <div class="text-caption text-medium-emphasis">Enabled</div>
                      <div class="text-h6">
                        <v-icon :color="stats.enabled ? 'success' : 'warning'" size="20">
                          {{ stats.enabled ? 'mdi-check-circle' : 'mdi-pause-circle' }}
                        </v-icon>
                        {{ stats.enabled ? 'Yes' : 'Idle' }}
                      </div>
                    </v-card-text>
                  </v-card>
                </v-col>
                <v-col cols="6" md="3">
                  <v-card variant="tonal" :color="stats.connected ? 'success' : 'error'">
                    <v-card-text class="pa-3">
                      <div class="text-caption text-medium-emphasis">Broker</div>
                      <div class="text-h6">
                        <v-icon size="20">
                          {{ stats.connected ? 'mdi-lan-connect' : 'mdi-lan-disconnect' }}
                        </v-icon>
                        {{ stats.connected ? 'Connected' : 'Disconnected' }}
                      </div>
                    </v-card-text>
                  </v-card>
                </v-col>
                <v-col cols="6" md="3">
                  <v-card variant="tonal">
                    <v-card-text class="pa-3">
                      <div class="text-caption text-medium-emphasis">Topic mode</div>
                      <div class="text-h6">{{ stats.topic_mode }}</div>
                    </v-card-text>
                  </v-card>
                </v-col>
                <v-col cols="6" md="3">
                  <v-card variant="tonal">
                    <v-card-text class="pa-3">
                      <div class="text-caption text-medium-emphasis">Cache size</div>
                      <div class="text-h6">{{ stats.cache_size ?? '—' }}</div>
                    </v-card-text>
                  </v-card>
                </v-col>
              </v-row>

              <div class="text-subtitle-2 mt-6 mb-2">Counters</div>
              <v-row dense>
                <v-col cols="6" md="3" v-for="c in counterTiles" :key="c.key">
                  <div class="counter-tile pa-3">
                    <div class="text-caption text-medium-emphasis">{{ c.label }}</div>
                    <div class="text-h5 font-weight-medium">
                      {{ formatCount(stats[c.key]) }}
                    </div>
                  </div>
                </v-col>
              </v-row>

              <div class="text-subtitle-2 mt-6 mb-2">Threads</div>
              <v-chip-group class="mb-3">
                <v-chip
                  size="small"
                  :color="stats.connect_alive ? 'success' : 'error'"
                  variant="tonal"
                >
                  Connect thread: {{ stats.connect_alive ? 'alive' : 'dead' }}
                </v-chip>
                <v-chip
                  size="small"
                  :color="stats.writer_alive ? 'success' : 'error'"
                  variant="tonal"
                >
                  Writer thread: {{ stats.writer_alive ? 'alive' : 'dead' }}
                </v-chip>
                <v-chip
                  size="small"
                  :color="stats.janitor_alive ? 'success' : 'error'"
                  variant="tonal"
                >
                  Janitor thread: {{ stats.janitor_alive ? 'alive' : 'dead' }}
                </v-chip>
              </v-chip-group>

              <v-table density="compact" class="details-table">
                <tbody>
                  <tr>
                    <td class="text-caption font-weight-medium">Broker</td>
                    <td class="text-caption">
                      <code>{{ stats.broker_host }}:{{ stats.broker_port }}</code>
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Root name</td>
                    <td class="text-caption">
                      <code>{{ stats.root_name || '—' }}</code>
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Meta prefixes</td>
                    <td class="text-caption">
                      <span v-if="!stats.meta_prefixes?.length">—</span>
                      <v-chip
                        v-for="p in stats.meta_prefixes"
                        :key="p"
                        size="x-small"
                        class="mr-1"
                      >{{ p }}</v-chip>
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Last message at</td>
                    <td class="text-caption">
                      {{ stats.last_message_at || '—' }}
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Last topic</td>
                    <td class="text-caption">
                      <code>{{ stats.last_message_topic || '—' }}</code>
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Last connected at</td>
                    <td class="text-caption">
                      {{ stats.last_connected_at || '—' }}
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Started at</td>
                    <td class="text-caption">
                      {{ stats.started_at || '—' }}
                    </td>
                  </tr>
                  <tr>
                    <td class="text-caption font-weight-medium">Buffered rows</td>
                    <td class="text-caption">{{ stats.buffered_rows ?? 0 }}</td>
                  </tr>
                </tbody>
              </v-table>

              <v-alert
                v-if="janitorSummary"
                type="success"
                variant="tonal"
                density="compact"
                class="mt-3"
                closable
                @click:close="janitorSummary = null"
              >
                Retention sweep: deleted
                {{ janitorSummary.deleted_csv_count }} CSVs,
                {{ janitorSummary.deleted_log_count }} logs
                (retention = {{ janitorSummary.retention_days }} days,
                errors = {{ janitorSummary.errors }})
              </v-alert>
            </template>
          </v-card-text>
        </v-card>
      </v-window-item>
    </v-window>
  </v-container>
</template>

<script setup lang="ts">
/**
 * Phase D — MQTT Rules view (config + rejected log + stats).
 * Reads are open to any authenticated user; writes admin-only. The API
 * also enforces admin, so the frontend guards are UX (not security).
 */
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import { useAssetTreeStore } from '@/stores/assetTree'
import MqttSetupTab from '@/components/MqttSetupTab.vue'
import api from '@/services/api'

interface Config {
  level_names?: string[]
  root_name?: string
  topic_mode?: string
  meta_prefixes?: string[]
  ingest_enabled?: boolean | number
  ingest_retention_days?: number
}

interface RejectedEntry {
  timestamp: string
  topic: string
  reason: string
}

interface StatsSnapshot {
  enabled: boolean
  connected: boolean
  topic_mode: string
  root_name: string | null
  meta_prefixes: string[]
  broker_host: string
  broker_port: number
  cache_size: number
  buffered_rows: number
  messages_received: number
  messages_routed: number
  messages_rejected: number
  messages_meta: number
  messages_parse_errors?: number
  files_written: number
  last_message_at: string | null
  last_message_topic: string | null
  last_connected_at: string | null
  connect_attempts: number
  reconnects: number
  started_at: string | null
  writer_alive: boolean
  connect_alive: boolean
  janitor_alive: boolean
}

const authStore = useAuthStore()
const notify = useNotificationStore()

const isAdmin = computed(() => authStore.user?.role === 'admin')

const activeTab = ref<'setup' | 'config' | 'rejected' | 'stats'>('setup')

// ── Config tab ────────────────────────────────────────────────────────
const config = ref<Config | null>(null)
const configLoading = ref(false)
const saving = ref(false)
const editable = ref<{
  topic_mode: string
  ingest_enabled: boolean
  ingest_retention_days: number
}>({
  topic_mode: 'strict',
  ingest_enabled: true,
  ingest_retention_days: 30,
})
const metaPrefixesText = ref('')

const hasChanges = computed(() => {
  if (!config.value) return false
  const currentPrefixes = normalizePrefixes(metaPrefixesText.value).join(',')
  const storedPrefixes = (config.value.meta_prefixes || []).slice().sort().join(',')
  return (
    editable.value.topic_mode !== config.value.topic_mode ||
    Boolean(editable.value.ingest_enabled) !== (config.value.ingest_enabled !== false && config.value.ingest_enabled !== 0) ||
    Number(editable.value.ingest_retention_days) !== Number(config.value.ingest_retention_days ?? 30) ||
    currentPrefixes !== storedPrefixes
  )
})

function normalizePrefixes(text: string): string[] {
  const seen = new Set<string>()
  for (const raw of (text || '').split(',')) {
    const trimmed = raw.trim()
    if (trimmed) seen.add(trimmed)
  }
  return Array.from(seen).sort()
}

function resetChanges() {
  if (!config.value) return
  applyConfigToEditable(config.value)
}

function applyConfigToEditable(cfg: Config) {
  editable.value = {
    topic_mode: cfg.topic_mode || 'strict',
    ingest_enabled: cfg.ingest_enabled !== false && cfg.ingest_enabled !== 0,
    ingest_retention_days: Number(cfg.ingest_retention_days ?? 30),
  }
  metaPrefixesText.value = (cfg.meta_prefixes || []).join(', ')
}

async function fetchConfig() {
  configLoading.value = true
  try {
    const r = await api.get('/api/asset-tree/config')
    const cfg = r.data || {}
    config.value = Object.keys(cfg).length ? cfg : null
    if (config.value) applyConfigToEditable(config.value)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load config')
  } finally {
    configLoading.value = false
  }
}

async function saveConfig() {
  if (!config.value) return
  saving.value = true
  try {
    const body = {
      topic_mode: editable.value.topic_mode,
      ingest_enabled: editable.value.ingest_enabled,
      ingest_retention_days: Number(editable.value.ingest_retention_days),
      meta_prefixes: normalizePrefixes(metaPrefixesText.value),
    }
    const r = await api.patch('/api/asset-tree/config', body)
    config.value = r.data
    if (config.value) applyConfigToEditable(config.value)
    notify.showSuccess('MQTT rules updated')
    // Refresh stats — mode / enabled flags are shown there too.
    fetchStats()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Save failed')
  } finally {
    saving.value = false
  }
}

// ── Rejected tab ──────────────────────────────────────────────────────
const rejectedDate = ref<string>(todayISO())
const rejectedLimit = ref<number>(200)
const rejectedEntries = ref<RejectedEntry[]>([])
const rejectedLoading = ref(false)

function todayISO(): string {
  const now = new Date()
  const y = now.getUTCFullYear()
  const m = String(now.getUTCMonth() + 1).padStart(2, '0')
  const d = String(now.getUTCDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

async function fetchRejected() {
  rejectedLoading.value = true
  try {
    const r = await api.get('/api/asset-tree/rejected-topics', {
      params: { date: rejectedDate.value, limit: rejectedLimit.value },
    })
    rejectedEntries.value = r.data?.entries || []
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load rejected topics')
    rejectedEntries.value = []
  } finally {
    rejectedLoading.value = false
  }
}

watch([rejectedDate, rejectedLimit], () => {
  // Auto-refresh on filter change so operators don't have to click.
  fetchRejected()
})

// ── Stats tab ─────────────────────────────────────────────────────────
const stats = ref<StatsSnapshot | null>(null)
const statsLoading = ref(false)
const janitorRunning = ref(false)
const janitorSummary = ref<any>(null)
let statsPollTimer: number | null = null

const counterTiles = [
  { key: 'messages_received', label: 'Received' },
  { key: 'messages_routed',   label: 'Routed to CSV' },
  { key: 'messages_meta',     label: 'Meta (skipped)' },
  { key: 'messages_rejected', label: 'Rejected' },
]

function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  return Number(n).toLocaleString()
}

async function fetchStats() {
  statsLoading.value = true
  try {
    const r = await api.get('/api/asset-tree/ingest-stats')
    stats.value = r.data
  } catch (e: any) {
    // Swallow — the router might genuinely be idle / unavailable.
    stats.value = null
  } finally {
    statsLoading.value = false
  }
}

async function runJanitor() {
  janitorRunning.value = true
  try {
    const r = await api.post('/api/asset-tree/ingest-janitor/run-now')
    janitorSummary.value = r.data
    notify.showSuccess('Retention sweep complete')
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Janitor run failed')
  } finally {
    janitorRunning.value = false
  }
}

// Auto-poll stats every 3 s while the Stats tab is open. Cleared on unmount.
watch(activeTab, (t) => {
  if (statsPollTimer !== null) {
    clearInterval(statsPollTimer)
    statsPollTimer = null
  }
  if (t === 'stats') {
    fetchStats()
    statsPollTimer = window.setInterval(fetchStats, 3000)
  } else if (t === 'rejected') {
    fetchRejected()
  }
})

onMounted(async () => {
  await fetchConfig()
  // Setup tab needs the tree to build the "example machine" snippet.
  const treeStore = useAssetTreeStore()
  if (!treeStore.treeLoaded && !treeStore.loadingTree) {
    treeStore.fetchTree()
  }
})

onUnmounted(() => {
  if (statsPollTimer !== null) {
    clearInterval(statsPollTimer)
    statsPollTimer = null
  }
})
</script>

<style scoped>
.counter-tile {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;
  background: rgba(var(--v-theme-surface), 0.6);
}
.details-table :deep(td) {
  border-bottom: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}
.rejected-scroll {
  max-height: 60vh;
  overflow: auto;
}
</style>
