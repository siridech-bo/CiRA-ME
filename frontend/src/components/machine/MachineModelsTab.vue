<template>
  <div>
    <!-- Direct bindings -->
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="d-flex align-center py-2">
        <v-icon color="primary" class="mr-2" size="small">mdi-brain</v-icon>
        <span class="text-subtitle-1">Trained on this machine</span>
        <v-spacer />
        <v-btn
          size="x-small"
          variant="text"
          :loading="loading"
          @click="load"
        >
          <v-icon start size="14">mdi-refresh</v-icon>Refresh
        </v-btn>
      </v-card-title>
      <v-divider />
      <div v-if="loading && !dataReady" class="pa-4 text-center text-caption">
        Loading…
      </div>
      <div
        v-else-if="trainedOn.length === 0"
        class="pa-6 text-center"
      >
        <v-icon size="40" color="grey">mdi-brain</v-icon>
        <p class="text-body-2 text-medium-emphasis mt-2">
          No models trained for this machine yet.
        </p>
        <v-btn
          color="primary"
          variant="tonal"
          size="small"
          prepend-icon="mdi-plus"
          :to="{ name: 'pipeline-training' }"
        >
          Train new model
        </v-btn>
      </div>
      <v-table v-else density="compact">
        <thead>
          <tr>
            <th>Name</th>
            <th>Algorithm</th>
            <th>Mode</th>
            <th>Endpoints</th>
            <th>Created</th>
            <th style="width: 130px" class="text-right"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in trainedOn" :key="`t-${m.id}`">
            <td class="font-weight-medium">{{ m.name }}</td>
            <td>
              <v-chip size="x-small" variant="tonal">{{ m.algorithm }}</v-chip>
            </td>
            <td>
              <v-chip size="x-small" variant="tonal" :color="modeColor(m.mode)">
                {{ m.mode }}
              </v-chip>
            </td>
            <td>
              <span v-if="!m.endpoints || m.endpoints.length === 0" class="text-caption text-medium-emphasis">
                —
              </span>
              <v-chip
                v-for="ep in (m.endpoints || [])"
                :key="ep.id"
                size="x-small"
                variant="tonal"
                color="success"
                class="mr-1"
              >
                {{ ep.name }}
              </v-chip>
            </td>
            <td class="text-caption">{{ formatTime(m.created_at) }}</td>
            <td class="text-right">
              <v-btn
                v-if="isAdmin"
                size="x-small"
                variant="text"
                prepend-icon="mdi-link-variant"
                @click="onRebind(m)"
              >
                Rebind
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>

    <!-- Group models -->
    <v-card variant="tonal">
      <v-card-title class="d-flex align-center py-2">
        <v-icon color="secondary" class="mr-2" size="small">mdi-account-group</v-icon>
        <span class="text-subtitle-1">Group models this machine participates in</span>
      </v-card-title>
      <v-divider />
      <div
        v-if="groupModels.length === 0"
        class="pa-4 text-center text-caption text-medium-emphasis"
      >
        This machine is not a member of any groups that share models.
      </div>
      <v-table v-else density="compact">
        <thead>
          <tr>
            <th>Name</th>
            <th>Group</th>
            <th>Algorithm</th>
            <th>Endpoints</th>
            <th>Created</th>
            <th style="width: 130px" class="text-right"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in groupModels" :key="`g-${m.id}`">
            <td class="font-weight-medium">{{ m.name }}</td>
            <td>
              <v-chip size="x-small" variant="tonal" color="secondary">
                {{ m.group_name }}
              </v-chip>
            </td>
            <td>
              <v-chip size="x-small" variant="tonal">{{ m.algorithm }}</v-chip>
            </td>
            <td>
              <span v-if="!m.endpoints || m.endpoints.length === 0" class="text-caption text-medium-emphasis">
                —
              </span>
              <v-chip
                v-for="ep in (m.endpoints || [])"
                :key="ep.id"
                size="x-small"
                variant="tonal"
                color="success"
                class="mr-1"
              >
                {{ ep.name }}
              </v-chip>
            </td>
            <td class="text-caption">{{ formatTime(m.created_at) }}</td>
            <td class="text-right">
              <v-btn
                v-if="isAdmin"
                size="x-small"
                variant="text"
                prepend-icon="mdi-link-variant"
                @click="onRebind(m)"
              >
                Rebind
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>

    <div v-if="loadError" class="pt-3">
      <v-alert type="error" density="compact" variant="tonal">
        {{ loadError }}
      </v-alert>
    </div>

    <RebindMachinesDialog
      v-model="rebindDialogOpen"
      :saved-model-id="rebindTargetId"
      :model-name="rebindTargetName"
      @saved="onRebindSaved"
    />
  </div>
</template>

<script setup lang="ts">
/**
 * Phase B.5 — Models tab.
 * Wraps GET /api/asset-tree/nodes/<id>/models (added in Phase B backend).
 * Splits results into two tables per the spec.
 */
import { ref, computed, watch, onMounted } from 'vue'
import api from '@/services/api'
import type { AssetNode } from '@/stores/assetTree'
import { useAuthStore } from '@/stores/auth'
import RebindMachinesDialog from '@/components/RebindMachinesDialog.vue'

const props = defineProps<{ machine: AssetNode }>()

const authStore = useAuthStore()
const isAdmin = computed(() => authStore.user?.role === 'admin')

const trainedOn = ref<any[]>([])
const deployedTo = ref<any[]>([])
const groupModels = ref<any[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)
const dataReady = ref(false)

// deployedTo is displayed on the Deploy tab, but a user may glance here for
// completeness; leave it out of the primary tables to avoid duplicating the
// Deploy tab's app view. We keep the ref for future extension.
void deployedTo

async function load() {
  loading.value = true
  loadError.value = null
  try {
    const r = await api.get(`/api/asset-tree/nodes/${props.machine.id}/models`)
    trainedOn.value = r.data?.trained_on || []
    deployedTo.value = r.data?.deployed_to || []
    groupModels.value = r.data?.group_models || []
    dataReady.value = true
  } catch (e: any) {
    loadError.value = e.response?.data?.error || 'Failed to load models'
  } finally {
    loading.value = false
  }
}

watch(() => props.machine?.id, () => load())
onMounted(load)

// ── Rebind dialog wiring (Phase C.5) ─────────────────────────────────────

const rebindDialogOpen = ref(false)
const rebindTargetId = ref<number | null>(null)
const rebindTargetName = ref<string | null>(null)

function onRebind(m: any) {
  rebindTargetId.value = m.id
  rebindTargetName.value = m.name || null
  rebindDialogOpen.value = true
}
async function onRebindSaved() {
  await load()
}

function modeColor(m: string) {
  if (m === 'anomaly') return 'orange'
  if (m === 'regression') return 'blue'
  if (m === 'classification') return 'purple'
  return undefined
}
function formatTime(s: string): string {
  if (!s) return '—'
  try { return new Date(s + (s.endsWith('Z') ? '' : 'Z')).toLocaleString() }
  catch { return s }
}
</script>
