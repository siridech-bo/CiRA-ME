<template>
  <v-card variant="tonal">
    <v-card-title class="d-flex align-center py-2">
      <v-icon color="primary" class="mr-2" size="small">mdi-rocket-launch</v-icon>
      <span class="text-subtitle-1">Deployed apps</span>
      <v-spacer />
      <v-btn
        size="x-small"
        variant="text"
        :loading="loading"
        @click="load"
      >
        <v-icon start size="14">mdi-refresh</v-icon>Refresh
      </v-btn>
      <v-btn
        color="primary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-plus"
        :to="{ name: 'app-builder' }"
        class="ml-2"
      >
        Publish new app
      </v-btn>
    </v-card-title>
    <v-divider />

    <div v-if="loading && !dataReady" class="pa-4 text-center text-caption">
      Loading…
    </div>
    <div
      v-else-if="matchingApps.length === 0"
      class="pa-6 text-center"
    >
      <v-icon size="40" color="grey">mdi-rocket-launch-outline</v-icon>
      <p class="text-body-2 text-medium-emphasis mt-2">
        No apps deploy models bound to this machine yet.
      </p>
    </div>
    <v-table v-else density="compact">
      <thead>
        <tr>
          <th>Name</th>
          <th>Slug</th>
          <th>Status</th>
          <th>Access</th>
          <th class="text-right">Calls</th>
          <th>Updated</th>
          <th style="width: 90px"></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="a in matchingApps" :key="a.id">
          <td class="font-weight-medium">{{ a.name }}</td>
          <td class="text-caption">{{ a.slug || '—' }}</td>
          <td>
            <v-chip size="x-small" variant="tonal" :color="a.status === 'published' ? 'success' : undefined">
              {{ a.status }}
            </v-chip>
          </td>
          <td>
            <v-chip size="x-small" variant="tonal">{{ a.access }}</v-chip>
          </td>
          <td class="text-right">{{ a.calls || 0 }}</td>
          <td class="text-caption">{{ formatTime(a.updated_at || a.created_at) }}</td>
          <td>
            <v-btn
              size="x-small"
              variant="text"
              prepend-icon="mdi-pencil"
              :to="{ name: 'app-builder-editor', params: { id: String(a.id) } }"
            >
              Edit
            </v-btn>
          </td>
        </tr>
      </tbody>
    </v-table>

    <div v-if="loadError" class="pa-3">
      <v-alert type="error" density="compact" variant="tonal">
        {{ loadError }}
      </v-alert>
    </div>
  </v-card>
</template>

<script setup lang="ts">
/**
 * Phase B.6 — Deploy tab.
 * Cross-references three sources to answer "which apps are deployed for
 * this machine":
 *   1. GET /api/asset-tree/nodes/<id>/models → deployed_to model ids
 *   2. GET /api/melab/endpoints → maps endpoint_id → saved_model_id
 *   3. GET /api/app-builder/apps → app list, then per-app GET to inspect
 *      nodes for `model.endpoint.<endpoint_id>` references.
 *
 * The per-app fetch is N-way but N is small in practice (< 20 apps per
 * user). If this ever gets slow, promote the join to a backend endpoint.
 */
import { ref, computed, watch, onMounted } from 'vue'
import api from '@/services/api'
import type { AssetNode } from '@/stores/assetTree'

const props = defineProps<{ machine: AssetNode }>()

const apps = ref<any[]>([])
const matchingApps = ref<any[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)
const dataReady = ref(false)

async function load() {
  loading.value = true
  loadError.value = null
  try {
    // 1. Machine deploy-bindings
    const modelsResp = await api.get(`/api/asset-tree/nodes/${props.machine.id}/models`)
    const deployedTo = new Set<number>(
      (modelsResp.data?.deployed_to || []).map((m: any) => m.id),
    )
    // 2. Endpoint → model
    const endpointsResp = await api.get('/api/melab/endpoints')
    const endpointsArr = Array.isArray(endpointsResp.data) ? endpointsResp.data : (endpointsResp.data?.endpoints || [])
    const relevantEndpointIds = new Set<string>()
    for (const ep of endpointsArr) {
      if (deployedTo.has(ep.saved_model_id)) {
        relevantEndpointIds.add(String(ep.id))
      }
    }
    // 3. Apps + per-app node scan
    const appsResp = await api.get('/api/app-builder/apps')
    apps.value = Array.isArray(appsResp.data) ? appsResp.data : []
    const matches: any[] = []
    if (relevantEndpointIds.size === 0) {
      matchingApps.value = []
      dataReady.value = true
      return
    }
    for (const app of apps.value) {
      try {
        const detail = await api.get(`/api/app-builder/apps/${app.id}`)
        const nodes = detail.data?.nodes || []
        const refsMachine = nodes.some((n: any) => {
          const t = String(n?.type || '')
          if (!t.startsWith('model.endpoint.')) return false
          const eid = t.replace('model.endpoint.', '')
          return relevantEndpointIds.has(eid)
        })
        if (refsMachine) matches.push(app)
      } catch { /* skip on error */ }
    }
    matchingApps.value = matches
    dataReady.value = true
  } catch (e: any) {
    loadError.value = e.response?.data?.error || 'Failed to load apps'
  } finally {
    loading.value = false
  }
}

watch(() => props.machine?.id, () => load())
onMounted(load)

function formatTime(s: string): string {
  if (!s) return '—'
  try { return new Date(s + (s.endsWith('Z') ? '' : 'Z')).toLocaleString() }
  catch { return s }
}
</script>
