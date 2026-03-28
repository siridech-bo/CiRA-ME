<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">MQTT Broker</h1>
        <p class="text-body-2 text-medium-emphasis">
          Monitor broker status, discover topics, and test subscriptions
        </p>
      </div>
    </div>

    <!-- Broker Status Card -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-server</v-icon>
          Broker Status
        </h3>
        <v-spacer />
        <v-btn size="small" variant="tonal" :loading="loadingBroker" @click="fetchBrokerInfo">
          <v-icon start size="small">mdi-refresh</v-icon>
          Refresh
        </v-btn>
      </div>

      <v-alert v-if="brokerError" type="error" variant="tonal" class="mb-3" density="compact">
        {{ brokerError }}
      </v-alert>

      <div class="d-flex flex-wrap ga-4">
        <div>
          <div class="text-caption text-medium-emphasis">Connection</div>
          <v-chip
            :color="broker.connected ? 'success' : 'error'"
            variant="flat"
            size="small"
          >
            <v-icon start size="x-small">{{ broker.connected ? 'mdi-check-circle' : 'mdi-close-circle' }}</v-icon>
            {{ broker.connected ? 'Connected' : 'Disconnected' }}
          </v-chip>
        </div>
        <div>
          <div class="text-caption text-medium-emphasis">Host</div>
          <div class="text-body-2 font-weight-medium">{{ broker.host || '-' }}</div>
        </div>
        <div>
          <div class="text-caption text-medium-emphasis">MQTT Port</div>
          <div class="text-body-2 font-weight-medium">{{ broker.port || '-' }}</div>
        </div>
        <div>
          <div class="text-caption text-medium-emphasis">WebSocket Port</div>
          <div class="text-body-2 font-weight-medium">{{ broker.ws_port || '-' }}</div>
        </div>
        <div v-if="broker.version">
          <div class="text-caption text-medium-emphasis">Version</div>
          <div class="text-body-2 font-weight-medium">{{ broker.version }}</div>
        </div>
        <div v-if="broker.clients_connected != null">
          <div class="text-caption text-medium-emphasis">Clients Connected</div>
          <div class="text-body-2 font-weight-medium">{{ broker.clients_connected }}</div>
        </div>
        <div v-if="broker.messages_received != null">
          <div class="text-caption text-medium-emphasis">Messages Received</div>
          <div class="text-body-2 font-weight-medium">{{ broker.messages_received }}</div>
        </div>
        <div v-if="broker.messages_sent != null">
          <div class="text-caption text-medium-emphasis">Messages Sent</div>
          <div class="text-body-2 font-weight-medium">{{ broker.messages_sent }}</div>
        </div>
        <div v-if="broker.uptime">
          <div class="text-caption text-medium-emphasis">Uptime</div>
          <div class="text-body-2 font-weight-medium">{{ formatUptime(broker.uptime) }}</div>
        </div>
      </div>
    </v-card>

    <!-- Active Topics Card -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-format-list-bulleted</v-icon>
          Active Topics
        </h3>
        <v-spacer />
        <v-btn size="small" variant="tonal" :loading="loadingTopics" @click="fetchTopics">
          <v-icon start size="small">mdi-refresh</v-icon>
          Discover Topics
        </v-btn>
      </div>

      <v-alert v-if="!loadingTopics && topics.length === 0" type="info" variant="tonal" density="compact">
        No active topics discovered. Make sure data is being published to the broker, then click "Discover Topics".
      </v-alert>

      <v-table v-if="topics.length > 0" density="comfortable">
        <thead>
          <tr>
            <th>Topic</th>
            <th>Last Seen</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in topics" :key="t.topic">
            <td>
              <code class="text-body-2">{{ t.topic }}</code>
            </td>
            <td class="text-caption">{{ formatTimestamp(t.last_seen) }}</td>
            <td>
              <v-btn
                size="x-small"
                variant="text"
                color="primary"
                @click="testTopic = t.topic; fetchTestMessages()"
                title="Subscribe & Test"
              >
                <v-icon size="small">mdi-play</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>

    <!-- Topic Test Card -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-test-tube</v-icon>
          Topic Test
        </h3>
      </div>

      <div class="d-flex align-center ga-3 mb-4">
        <v-text-field
          v-model="testTopic"
          label="Topic"
          placeholder="sensors/test"
          variant="outlined"
          density="compact"
          hide-details
          class="flex-grow-1"
        />
        <v-text-field
          v-model.number="testCount"
          label="Messages"
          type="number"
          variant="outlined"
          density="compact"
          hide-details
          style="max-width: 120px;"
        />
        <v-btn
          color="primary"
          variant="flat"
          :loading="loadingTest"
          :disabled="!testTopic"
          @click="fetchTestMessages"
        >
          <v-icon start>mdi-antenna</v-icon>
          Subscribe
        </v-btn>
      </div>

      <v-alert v-if="testError" type="error" variant="tonal" class="mb-3" density="compact">
        {{ testError }}
      </v-alert>

      <v-alert v-if="!loadingTest && testMessages.length === 0 && testAttempted" type="info" variant="tonal" density="compact">
        No messages received within the timeout period. Check that data is being published to "{{ testTopic }}".
      </v-alert>

      <div v-if="testMessages.length > 0">
        <div class="text-caption text-medium-emphasis mb-2">
          Received {{ testMessages.length }} message(s):
        </div>
        <v-card
          v-for="(msg, i) in testMessages"
          :key="i"
          variant="outlined"
          class="pa-3 mb-2"
          style="background: rgba(0,0,0,0.15);"
        >
          <div class="d-flex align-center mb-1">
            <v-chip size="x-small" variant="tonal" color="primary" class="mr-2">
              {{ msg.topic }}
            </v-chip>
            <span class="text-caption text-medium-emphasis">
              QoS {{ msg.qos }} | {{ formatTimestamp(msg.timestamp) }}
            </span>
          </div>
          <pre class="text-body-2" style="white-space: pre-wrap; margin: 0; font-family: monospace;">{{ formatPayload(msg.payload) }}</pre>
        </v-card>
      </div>
    </v-card>

    <!-- Active Publishers Card -->
    <v-card class="pa-4">
      <div class="d-flex align-center mb-3">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-publish</v-icon>
          Active Publishers
        </h3>
        <v-spacer />
        <v-btn size="small" variant="tonal" :loading="loadingPublishers" @click="fetchPublishers">
          <v-icon start size="small">mdi-refresh</v-icon>
          Refresh
        </v-btn>
      </div>

      <v-alert v-if="publishers.length === 0 && !loadingPublishers" type="info" variant="tonal" density="compact">
        No active MQTT test publishers running.
      </v-alert>

      <v-table v-if="publishers.length > 0" density="comfortable">
        <thead>
          <tr>
            <th>Session</th>
            <th>Topic</th>
            <th>File</th>
            <th>Rate</th>
            <th>Progress</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="pub in publishers" :key="pub.session_id">
            <td class="text-caption">{{ pub.session_id }}</td>
            <td><code class="text-body-2">{{ pub.topic }}</code></td>
            <td class="text-caption">{{ pub.file }}</td>
            <td class="text-caption">{{ pub.rate }}/s</td>
            <td class="text-caption">{{ pub.published }} / {{ pub.total }}</td>
            <td>
              <v-chip
                size="x-small"
                :color="pub.running ? 'success' : 'grey'"
                variant="flat"
              >
                {{ pub.running ? 'Running' : 'Stopped' }}
              </v-chip>
            </td>
            <td>
              <v-btn
                v-if="pub.running"
                icon
                size="x-small"
                variant="text"
                color="error"
                @click="stopPublisher(pub.session_id)"
                title="Stop"
              >
                <v-icon size="small">mdi-stop</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>
  </v-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import api from '@/services/api'

const broker = ref({
  host: null,
  port: null,
  ws_port: null,
  connected: false,
  version: null,
  clients_connected: null,
  messages_received: null,
  messages_sent: null,
  uptime: null,
})
const brokerError = ref('')
const loadingBroker = ref(false)

const topics = ref([])
const loadingTopics = ref(false)

const testTopic = ref('')
const testCount = ref(5)
const testMessages = ref([])
const testError = ref('')
const testAttempted = ref(false)
const loadingTest = ref(false)

const publishers = ref([])
const loadingPublishers = ref(false)

async function fetchBrokerInfo() {
  loadingBroker.value = true
  brokerError.value = ''
  try {
    const resp = await api.get('/api/mqtt/broker-info')
    broker.value = resp.data
  } catch (e) {
    brokerError.value = e.response?.data?.error || 'Failed to fetch broker info'
  } finally {
    loadingBroker.value = false
  }
}

async function fetchTopics() {
  loadingTopics.value = true
  try {
    const resp = await api.get('/api/mqtt/topics')
    topics.value = resp.data
  } catch {
    topics.value = []
  } finally {
    loadingTopics.value = false
  }
}

async function fetchTestMessages() {
  if (!testTopic.value) return
  loadingTest.value = true
  testError.value = ''
  testMessages.value = []
  testAttempted.value = true
  try {
    const resp = await api.post('/api/mqtt/topics/subscribe-test', {
      topic: testTopic.value,
      count: testCount.value,
    })
    testMessages.value = resp.data.messages || []
  } catch (e) {
    testError.value = e.response?.data?.error || 'Subscribe test failed'
  } finally {
    loadingTest.value = false
  }
}

async function fetchPublishers() {
  loadingPublishers.value = true
  try {
    const resp = await api.get('/api/mqtt/status')
    publishers.value = resp.data.active_publishers || []
  } catch {
    publishers.value = []
  } finally {
    loadingPublishers.value = false
  }
}

async function stopPublisher(sessionId) {
  try {
    await api.post(`/api/mqtt/publish/${sessionId}/stop`)
    fetchPublishers()
  } catch {
    // ignore
  }
}

function formatTimestamp(ts) {
  if (!ts) return '-'
  try {
    return new Date(ts * 1000).toLocaleTimeString()
  } catch {
    return String(ts)
  }
}

function formatUptime(uptime) {
  if (!uptime) return '-'
  const seconds = parseInt(uptime)
  if (isNaN(seconds)) return uptime
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  if (days > 0) return `${days}d ${hours}h ${mins}m`
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m`
}

function formatPayload(payload) {
  try {
    const parsed = JSON.parse(payload)
    return JSON.stringify(parsed, null, 2)
  } catch {
    return payload
  }
}

onMounted(() => {
  fetchBrokerInfo()
  fetchPublishers()
})
</script>
