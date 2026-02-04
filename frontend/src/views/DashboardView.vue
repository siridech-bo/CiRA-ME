<template>
  <v-container fluid class="pa-6">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">Dashboard</h1>
        <p class="text-body-2 text-medium-emphasis">
          Real-time monitoring and analytics
        </p>
      </div>
      <v-spacer />
      <v-btn
        variant="outlined"
        prepend-icon="mdi-refresh"
        @click="refreshData"
        :loading="loading"
      >
        Refresh
      </v-btn>
    </div>

    <!-- Metric Cards -->
    <v-row class="mb-6">
      <v-col cols="12" sm="6" md="3">
        <v-card class="metric-card pa-4">
          <div class="text-body-2 text-medium-emphasis">Total Samples</div>
          <div class="metric-value">{{ metrics.totalSamples.toLocaleString() }}</div>
          <div class="metric-change positive">
            <v-icon size="small">mdi-arrow-up</v-icon>
            +{{ metrics.samplesChange }}%
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="metric-card pa-4">
          <div class="text-body-2 text-medium-emphasis">
            {{ pipelineStore.mode === 'anomaly' ? 'Anomalies Detected' : 'Classifications' }}
          </div>
          <div class="metric-value">{{ metrics.detections }}</div>
          <div class="metric-change" :class="metrics.detectionsChange < 0 ? 'positive' : 'negative'">
            <v-icon size="small">{{ metrics.detectionsChange < 0 ? 'mdi-arrow-down' : 'mdi-arrow-up' }}</v-icon>
            {{ Math.abs(metrics.detectionsChange) }}%
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="metric-card pa-4">
          <div class="text-body-2 text-medium-emphasis">Model Status</div>
          <div class="d-flex align-center mt-2">
            <span class="status-dot connected mr-2" />
            <span class="text-h6">ACTIVE</span>
          </div>
          <div class="text-caption text-medium-emphasis mt-1">
            {{ currentModel }}
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="metric-card pa-4">
          <div class="text-body-2 text-medium-emphasis">Accuracy</div>
          <div class="metric-value">{{ metrics.accuracy }}%</div>
          <div class="metric-change positive">
            <v-icon size="small">mdi-arrow-up</v-icon>
            +{{ metrics.accuracyChange }}%
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Charts Row -->
    <v-row class="mb-6">
      <!-- Sensor Stream -->
      <v-col cols="12" md="8">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4">
            <h3 class="text-h6">Live Sensor Stream</h3>
            <v-spacer />
            <v-chip size="small" color="success" variant="flat">
              <span class="status-dot connected mr-2" style="width:6px;height:6px" />
              Streaming
            </v-chip>
          </div>
          <div class="chart-container">
            <Line :data="sensorChartData" :options="chartOptions" />
          </div>
        </v-card>
      </v-col>

      <!-- Distribution -->
      <v-col cols="12" md="4">
        <v-card class="pa-4">
          <h3 class="text-h6 mb-4">
            {{ pipelineStore.mode === 'anomaly' ? 'Anomaly Distribution' : 'Class Distribution' }}
          </h3>
          <div class="chart-container">
            <Doughnut :data="distributionChartData" :options="doughnutOptions" />
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Recent Events -->
    <v-row>
      <v-col cols="12" md="8">
        <v-card>
          <v-card-title class="d-flex align-center">
            <span>Recent Events</span>
            <v-spacer />
            <v-btn variant="text" size="small">View All</v-btn>
          </v-card-title>
          <v-data-table
            :headers="eventHeaders"
            :items="recentEvents"
            :items-per-page="5"
            density="comfortable"
          >
            <template #item.status="{ item }">
              <v-chip
                :color="item.status === 'anomaly' ? 'error' : 'success'"
                size="small"
                variant="flat"
              >
                {{ item.status.toUpperCase() }}
              </v-chip>
            </template>
            <template #item.actions="{ item }">
              <v-btn variant="text" size="small" color="primary">
                View Details
              </v-btn>
            </template>
          </v-data-table>
        </v-card>
      </v-col>

      <!-- Feature Importance -->
      <v-col cols="12" md="4">
        <v-card class="pa-4">
          <h3 class="text-h6 mb-4">Top Features</h3>
          <v-list density="compact">
            <v-list-item v-for="(feature, index) in topFeatures" :key="index">
              <template #prepend>
                <span class="text-caption text-medium-emphasis mr-2">{{ index + 1 }}</span>
              </template>
              <v-list-item-title class="text-body-2">{{ feature.name }}</v-list-item-title>
              <template #append>
                <v-progress-linear
                  :model-value="feature.importance * 100"
                  color="primary"
                  height="6"
                  rounded
                  style="width: 100px"
                />
              </template>
            </v-list-item>
          </v-list>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Line, Doughnut } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { usePipelineStore } from '@/stores/pipeline'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const pipelineStore = usePipelineStore()
const loading = ref(false)

const metrics = ref({
  totalSamples: 5432,
  samplesChange: 12,
  detections: 23,
  detectionsChange: -5,
  accuracy: 94.2,
  accuracyChange: 2.1
})

const currentModel = computed(() =>
  pipelineStore.mode === 'anomaly' ? 'Isolation Forest' : 'Random Forest'
)

const eventHeaders = [
  { title: 'Timestamp', key: 'timestamp' },
  { title: 'Score', key: 'score' },
  { title: 'Threshold', key: 'threshold' },
  { title: 'Status', key: 'status' },
  { title: 'Actions', key: 'actions', sortable: false }
]

const recentEvents = ref([
  { timestamp: '14:32:05', score: 0.89, threshold: 0.65, status: 'anomaly' },
  { timestamp: '14:28:12', score: 0.72, threshold: 0.65, status: 'anomaly' },
  { timestamp: '14:15:33', score: 0.31, threshold: 0.65, status: 'normal' },
  { timestamp: '14:10:21', score: 0.28, threshold: 0.65, status: 'normal' },
  { timestamp: '14:05:45', score: 0.45, threshold: 0.65, status: 'normal' }
])

const topFeatures = ref([
  { name: 'rms_accel_x', importance: 0.85 },
  { name: 'std_accel_y', importance: 0.72 },
  { name: 'kurtosis_accel_z', importance: 0.65 },
  { name: 'peak_frequency', importance: 0.58 },
  { name: 'spectral_entropy', importance: 0.45 }
])

// Chart data
const sensorChartData = computed(() => ({
  labels: Array.from({ length: 50 }, (_, i) => i.toString()),
  datasets: [
    {
      label: 'Accel X',
      data: Array.from({ length: 50 }, () => Math.sin(Math.random() * Math.PI) * 0.5 + Math.random() * 0.2),
      borderColor: '#6366F1',
      backgroundColor: 'rgba(99, 102, 241, 0.1)',
      fill: true,
      tension: 0.4
    },
    {
      label: 'Accel Y',
      data: Array.from({ length: 50 }, () => Math.cos(Math.random() * Math.PI) * 0.4 + Math.random() * 0.2),
      borderColor: '#22D3EE',
      backgroundColor: 'rgba(34, 211, 238, 0.1)',
      fill: true,
      tension: 0.4
    },
    {
      label: 'Accel Z',
      data: Array.from({ length: 50 }, () => Math.random() * 0.3 + 0.2),
      borderColor: '#A855F7',
      backgroundColor: 'rgba(168, 85, 247, 0.1)',
      fill: true,
      tension: 0.4
    }
  ]
}))

const distributionChartData = computed(() => ({
  labels: pipelineStore.mode === 'anomaly' ? ['Normal', 'Anomaly'] : ['Class A', 'Class B', 'Class C'],
  datasets: [{
    data: pipelineStore.mode === 'anomaly' ? [412, 23] : [180, 150, 105],
    backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
    borderWidth: 0
  }]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top' as const
    }
  },
  scales: {
    y: {
      beginAtZero: true
    }
  }
}

const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom' as const
    }
  }
}

function refreshData() {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 1000)
}

onMounted(() => {
  // Initial data load
})
</script>
