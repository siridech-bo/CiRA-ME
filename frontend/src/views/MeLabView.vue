<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">ME-LAB</h1>
        <p class="text-body-2 text-medium-emphasis">
          AI-as-a-Service inference endpoints
        </p>
      </div>
      <v-spacer />
      <v-btn color="primary" @click="showCreateDialog = true" :disabled="!savedModels.length">
        <v-icon start>mdi-plus</v-icon>
        New Endpoint
      </v-btn>
    </div>

    <!-- API Keys Section -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-key</v-icon>
          API Keys
        </h3>
        <v-spacer />
        <v-btn size="small" color="info" variant="tonal" @click="createApiKey">
          <v-icon start size="small">mdi-plus</v-icon>
          Generate Key
        </v-btn>
      </div>

      <v-alert v-if="newKeyValue" type="success" variant="tonal" class="mb-3" closable @click:close="newKeyValue = ''">
        <div class="font-weight-bold mb-1">New API Key (save this — it won't be shown again!):</div>
        <code class="d-block pa-2" style="background: rgba(0,0,0,0.2); border-radius: 4px; word-break: break-all;">{{ newKeyValue }}</code>
      </v-alert>

      <div v-if="apiKeys.length > 0" class="d-flex flex-wrap ga-2">
        <v-chip
          v-for="key in apiKeys"
          :key="key.id"
          :color="key.is_active ? 'primary' : 'grey'"
          variant="tonal"
          size="small"
          closable
          @click:close="revokeKey(key.id)"
        >
          <v-icon start size="x-small">mdi-key</v-icon>
          {{ key.prefix }}... ({{ key.name }})
          <span v-if="key.last_used_at" class="ml-1 text-caption opacity-70">
            | used {{ formatDate(key.last_used_at) }}
          </span>
        </v-chip>
      </div>
      <div v-else class="text-caption text-medium-emphasis">
        No API keys. Generate one to start using inference endpoints.
      </div>
    </v-card>

    <!-- Endpoints Table -->
    <v-card class="pa-4">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">
        <v-icon start size="small">mdi-api</v-icon>
        Inference Endpoints
      </h3>

      <v-table v-if="endpoints.length > 0" density="comfortable">
        <thead>
          <tr>
            <th>Name</th>
            <th>Mode</th>
            <th>Algorithm</th>
            <th>Features</th>
            <th>Status</th>
            <th class="text-center">Calls</th>
            <th>Last Used</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="ep in endpoints" :key="ep.id">
            <td>
              <div class="font-weight-medium">{{ ep.name }}</div>
              <div class="text-caption text-medium-emphasis">{{ ep.id }}</div>
            </td>
            <td>
              <v-chip size="x-small" :color="ep.mode === 'anomaly' ? 'error' : ep.mode === 'regression' ? 'purple' : 'success'" variant="tonal">
                {{ ep.mode }}
              </v-chip>
            </td>
            <td class="text-caption">{{ ep.algorithm }}</td>
            <td class="text-caption">{{ ep.n_features }}</td>
            <td>
              <v-chip size="x-small" :color="ep.status === 'active' ? 'success' : 'grey'" variant="flat">
                {{ ep.status }}
              </v-chip>
            </td>
            <td class="text-center">{{ ep.inference_count || 0 }}</td>
            <td class="text-caption">{{ ep.last_inference_at ? formatDate(ep.last_inference_at) : 'never' }}</td>
            <td>
              <v-btn icon size="x-small" variant="text" color="info" @click="showEndpointInfo(ep)" title="API Info">
                <v-icon size="small">mdi-code-braces</v-icon>
              </v-btn>
              <v-btn icon size="x-small" variant="text" @click="toggleStatus(ep)" :title="ep.status === 'active' ? 'Pause' : 'Activate'">
                <v-icon size="small">{{ ep.status === 'active' ? 'mdi-pause' : 'mdi-play' }}</v-icon>
              </v-btn>
              <v-btn icon size="x-small" variant="text" color="error" @click="deleteEndpoint(ep)" title="Delete">
                <v-icon size="small">mdi-delete</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>

      <div v-else class="text-center pa-8">
        <v-icon size="48" color="grey" class="mb-3">mdi-api-off</v-icon>
        <div class="text-body-1 text-medium-emphasis">No inference endpoints yet</div>
        <div class="text-caption text-medium-emphasis mt-1">
          Save a trained model, then create an endpoint to serve it as an API.
        </div>
      </div>
    </v-card>

    <!-- Create Endpoint Dialog -->
    <v-dialog v-model="showCreateDialog" max-width="500">
      <v-card>
        <v-card-title>Create Inference Endpoint</v-card-title>
        <v-card-text>
          <v-select
            v-model="newEndpoint.saved_model_id"
            :items="savedModels"
            item-title="displayName"
            item-value="id"
            label="Select Saved Model"
            variant="outlined"
            density="compact"
            class="mb-3"
          />
          <v-text-field
            v-model="newEndpoint.name"
            label="Endpoint Name"
            variant="outlined"
            density="compact"
            class="mb-3"
          />
          <v-text-field
            v-model="newEndpoint.description"
            label="Description (optional)"
            variant="outlined"
            density="compact"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCreateDialog = false">Cancel</v-btn>
          <v-btn color="primary" variant="flat" :disabled="!newEndpoint.saved_model_id" @click="createEndpoint">
            Create
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Endpoint Info Dialog -->
    <v-dialog v-model="showInfoDialog" max-width="650" scrollable>
      <v-card v-if="selectedEndpoint">
        <v-card-title>
          <v-icon start size="small">mdi-code-braces</v-icon>
          API Documentation — {{ selectedEndpoint.name }}
        </v-card-title>
        <v-card-text>
          <h4 class="text-subtitle-2 font-weight-bold mb-2">Endpoint URL</h4>
          <v-card variant="outlined" class="pa-2 mb-4" style="background: #1e1e1e;">
            <code style="color: #d4d4d4;">POST {{ baseUrl }}/api/melab/v1/{{ selectedEndpoint.id }}/predict</code>
          </v-card>

          <h4 class="text-subtitle-2 font-weight-bold mb-2">cURL Example</h4>
          <v-card variant="outlined" class="pa-2 mb-4" style="background: #1e1e1e;">
            <pre style="color: #d4d4d4; white-space: pre-wrap; margin: 0; font-size: 12px;">curl -X POST {{ baseUrl }}/api/melab/v1/{{ selectedEndpoint.id }}/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"data": [[{{ exampleFeatures }}]]}'</pre>
          </v-card>

          <h4 class="text-subtitle-2 font-weight-bold mb-2">Python Example</h4>
          <v-card variant="outlined" class="pa-2 mb-4" style="background: #1e1e1e;">
            <pre style="color: #d4d4d4; white-space: pre-wrap; margin: 0; font-size: 12px;">import requests

url = "{{ baseUrl }}/api/melab/v1/{{ selectedEndpoint.id }}/predict"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "YOUR_API_KEY"
}
data = {"data": [[{{ exampleFeatures }}]]}

response = requests.post(url, json=data, headers=headers)
print(response.json())</pre>
          </v-card>

          <h4 class="text-subtitle-2 font-weight-bold mb-2">Expected Input</h4>
          <div class="text-caption mb-2">
            <strong>{{ selectedEndpoint.n_features }}</strong> features per sample
          </div>
          <div v-if="selectedEndpoint.feature_names?.length" class="d-flex flex-wrap ga-1 mb-4">
            <v-chip v-for="f in selectedEndpoint.feature_names" :key="f" size="x-small" variant="tonal">{{ f }}</v-chip>
          </div>

          <h4 class="text-subtitle-2 font-weight-bold mb-2">Response Format ({{ selectedEndpoint.mode }})</h4>
          <v-card variant="outlined" class="pa-2" style="background: #1e1e1e;">
            <pre v-if="selectedEndpoint.mode === 'regression'" style="color: #d4d4d4; white-space: pre-wrap; margin: 0; font-size: 12px;">{"predictions": [{"value": 42.7}], "latency_ms": 5.2}</pre>
            <pre v-else-if="selectedEndpoint.mode === 'anomaly'" style="color: #d4d4d4; white-space: pre-wrap; margin: 0; font-size: 12px;">{"predictions": [{"label": "normal", "score": 0.23}], "latency_ms": 5.2}</pre>
            <pre v-else style="color: #d4d4d4; white-space: pre-wrap; margin: 0; font-size: 12px;">{"predictions": [{"label": "class_A", "confidence": 0.92}], "latency_ms": 5.2}</pre>
          </v-card>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showInfoDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'

const notificationStore = useNotificationStore()

const endpoints = ref<any[]>([])
const apiKeys = ref<any[]>([])
const savedModels = ref<any[]>([])
const newKeyValue = ref('')
const showCreateDialog = ref(false)
const showInfoDialog = ref(false)
const selectedEndpoint = ref<any>(null)
const newEndpoint = ref({ saved_model_id: null, name: '', description: '' })

const baseUrl = computed(() => window.location.origin)

const exampleFeatures = computed(() => {
  if (!selectedEndpoint.value) return '0.0, 0.0, 0.0'
  const n = selectedEndpoint.value.n_features || 3
  return Array(Math.min(n, 5)).fill('0.0').join(', ') + (n > 5 ? ', ...' : '')
})

function formatDate(dateStr: string) {
  if (!dateStr) return ''
  try { return new Date(dateStr).toLocaleDateString() } catch { return dateStr }
}

async function fetchEndpoints() {
  try {
    const resp = await api.get('/api/melab/endpoints')
    endpoints.value = resp.data
  } catch { endpoints.value = [] }
}

async function fetchApiKeys() {
  try {
    const resp = await api.get('/api/melab/keys')
    apiKeys.value = resp.data
  } catch { apiKeys.value = [] }
}

async function fetchSavedModels() {
  try {
    const resp = await api.get('/api/training/saved-models')
    savedModels.value = (resp.data || []).map((m: any) => ({
      ...m,
      displayName: `${m.name} (${m.algorithm || m.mode})`,
    }))
  } catch { savedModels.value = [] }
}

async function createEndpoint() {
  try {
    const resp = await api.post('/api/melab/endpoints', newEndpoint.value)
    notificationStore.showSuccess(`Endpoint created: ${resp.data.endpoint_id}`)
    showCreateDialog.value = false
    newEndpoint.value = { saved_model_id: null, name: '', description: '' }
    fetchEndpoints()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to create endpoint')
  }
}

async function deleteEndpoint(ep: any) {
  if (!confirm(`Delete endpoint "${ep.name}"?`)) return
  try {
    await api.delete(`/api/melab/endpoints/${ep.id}`)
    notificationStore.showSuccess('Endpoint deleted')
    fetchEndpoints()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to delete')
  }
}

async function toggleStatus(ep: any) {
  const newStatus = ep.status === 'active' ? 'paused' : 'active'
  try {
    await api.put(`/api/melab/endpoints/${ep.id}`, { status: newStatus })
    ep.status = newStatus
    notificationStore.showSuccess(`Endpoint ${newStatus}`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to update')
  }
}

async function createApiKey() {
  try {
    const resp = await api.post('/api/melab/keys', { name: 'default' })
    newKeyValue.value = resp.data.key
    notificationStore.showSuccess('API key generated — save it now!')
    fetchApiKeys()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to generate key')
  }
}

async function revokeKey(keyId: number) {
  if (!confirm('Revoke this API key?')) return
  try {
    await api.delete(`/api/melab/keys/${keyId}`)
    notificationStore.showSuccess('API key revoked')
    fetchApiKeys()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to revoke')
  }
}

function showEndpointInfo(ep: any) {
  selectedEndpoint.value = ep
  showInfoDialog.value = true
}

onMounted(() => {
  fetchEndpoints()
  fetchApiKeys()
  fetchSavedModels()
})
</script>
