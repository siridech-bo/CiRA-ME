<template>
  <v-container fluid class="pa-6">
    <!-- Header -->
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">Dashboard</h1>
        <p class="text-body-2 text-medium-emphasis">
          System overview and model analytics
        </p>
      </div>
      <v-spacer />
      <v-btn
        variant="outlined"
        prepend-icon="mdi-refresh"
        @click="fetchStats"
        :loading="loading"
      >
        Refresh
      </v-btn>
    </div>

    <!-- Summary Cards -->
    <v-row class="mb-6">
      <v-col cols="12" sm="6" md="3">
        <v-card class="pa-4">
          <div class="d-flex align-center">
            <v-avatar color="primary" variant="tonal" size="48" class="mr-3">
              <v-icon>mdi-account-group</v-icon>
            </v-avatar>
            <div>
              <div class="text-caption text-medium-emphasis">Total Users</div>
              <div class="text-h4 font-weight-bold">{{ stats.users?.total || 0 }}</div>
              <div class="text-caption text-medium-emphasis">
                {{ stats.users?.active || 0 }} active
              </div>
            </div>
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="pa-4">
          <div class="d-flex align-center">
            <v-avatar color="success" variant="tonal" size="48" class="mr-3">
              <v-icon>mdi-brain</v-icon>
            </v-avatar>
            <div>
              <div class="text-caption text-medium-emphasis">Saved Models</div>
              <div class="text-h4 font-weight-bold">{{ stats.models?.total || 0 }}</div>
              <div class="text-caption text-medium-emphasis">
                across all users
              </div>
            </div>
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="pa-4">
          <div class="d-flex align-center">
            <v-avatar color="info" variant="tonal" size="48" class="mr-3">
              <v-icon>mdi-server</v-icon>
            </v-avatar>
            <div>
              <div class="text-caption text-medium-emphasis">System Uptime</div>
              <div class="text-h5 font-weight-bold">{{ stats.system?.uptime || '-' }}</div>
              <div class="text-caption text-medium-emphasis">
                v{{ stats.system?.version || '1.0.0' }}
              </div>
            </div>
          </div>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="3">
        <v-card class="pa-4">
          <div class="d-flex align-center">
            <v-avatar color="purple" variant="tonal" size="48" class="mr-3">
              <v-icon>mdi-memory</v-icon>
            </v-avatar>
            <div>
              <div class="text-caption text-medium-emphasis">Memory</div>
              <div class="text-h5 font-weight-bold">
                {{ stats.system?.memory?.usage_percent || 0 }}%
              </div>
              <div class="text-caption text-medium-emphasis">
                {{ stats.system?.memory?.used_gb || 0 }} / {{ stats.system?.memory?.total_gb || 0 }} GB
              </div>
            </div>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Charts Row -->
    <v-row class="mb-6">
      <!-- Models by Mode -->
      <v-col cols="12" md="5">
        <v-card class="pa-4" style="height: 100%;">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">
            <v-icon size="small" class="mr-1">mdi-chart-donut</v-icon>
            Models by Mode
          </h3>
          <div v-if="hasModels" style="height: 250px; display: flex; align-items: center; justify-content: center;">
            <Doughnut :data="modeChartData" :options="doughnutOptions" />
          </div>
          <div v-else class="d-flex flex-column align-center justify-center" style="height: 250px;">
            <v-icon size="64" color="grey" class="mb-3">mdi-chart-donut-variant</v-icon>
            <div class="text-body-2 text-medium-emphasis">No models saved yet</div>
            <v-btn
              color="primary"
              variant="tonal"
              size="small"
              class="mt-2"
              @click="$router.push({ name: 'pipeline-training' })"
            >
              Train Your First Model
            </v-btn>
          </div>
        </v-card>
      </v-col>

      <!-- Models by Algorithm -->
      <v-col cols="12" md="7">
        <v-card class="pa-4" style="height: 100%;">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">
            <v-icon size="small" class="mr-1">mdi-chart-bar</v-icon>
            Models by Algorithm
          </h3>
          <div v-if="hasModels" style="height: 250px;">
            <Bar :data="algorithmChartData" :options="barOptions" />
          </div>
          <div v-else class="d-flex flex-column align-center justify-center" style="height: 250px;">
            <v-icon size="64" color="grey" class="mb-3">mdi-chart-bar</v-icon>
            <div class="text-body-2 text-medium-emphasis">Train and save models to see statistics</div>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- System Info Row -->
    <v-row class="mb-6">
      <!-- CPU -->
      <v-col cols="12" md="4">
        <v-card class="pa-4" style="height: 100%;">
          <div class="d-flex align-center mb-3">
            <v-icon color="info" class="mr-2">mdi-cpu-64-bit</v-icon>
            <h3 class="text-subtitle-1 font-weight-bold">CPU</h3>
            <v-spacer />
            <v-chip size="small" :color="(stats.system?.cpu?.usage_percent || 0) > 80 ? 'error' : 'success'" variant="tonal">
              {{ stats.system?.cpu?.usage_percent || 0 }}%
            </v-chip>
          </div>
          <v-table density="compact">
            <tbody>
              <tr>
                <td class="text-medium-emphasis">Processor</td>
                <td class="text-right font-weight-medium">{{ stats.system?.cpu?.processor || '-' }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Cores (Physical)</td>
                <td class="text-right font-weight-medium">{{ stats.system?.cpu?.cores_physical || '-' }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Cores (Logical)</td>
                <td class="text-right font-weight-medium">{{ stats.system?.cpu?.cores_logical || '-' }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Platform</td>
                <td class="text-right font-weight-medium">{{ stats.system?.platform || '-' }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Python</td>
                <td class="text-right font-weight-medium">{{ stats.system?.python_version || '-' }}</td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-col>

      <!-- GPU -->
      <v-col cols="12" md="4">
        <v-card class="pa-4" style="height: 100%;">
          <div class="d-flex align-center mb-3">
            <v-icon :color="stats.system?.cuda_available ? 'success' : 'grey'" class="mr-2">mdi-chip</v-icon>
            <h3 class="text-subtitle-1 font-weight-bold">GPU</h3>
            <v-spacer />
            <v-chip size="small" :color="stats.system?.cuda_available ? 'success' : 'grey'" variant="tonal">
              {{ stats.system?.cuda_available ? 'CUDA' : 'N/A' }}
            </v-chip>
          </div>
          <v-table v-if="stats.system?.gpu" density="compact">
            <tbody>
              <tr>
                <td class="text-medium-emphasis">GPU Name</td>
                <td class="text-right font-weight-medium">{{ stats.system.gpu.name }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">GPU Count</td>
                <td class="text-right font-weight-medium">{{ stats.system.gpu.count }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">VRAM Total</td>
                <td class="text-right font-weight-medium">{{ stats.system.gpu.memory_total_gb }} GB</td>
              </tr>
              <tr v-if="stats.system.gpu.memory_used_gb != null">
                <td class="text-medium-emphasis">VRAM Used</td>
                <td class="text-right font-weight-medium">{{ stats.system.gpu.memory_used_gb }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">CUDA Version</td>
                <td class="text-right font-weight-medium">{{ stats.system.gpu.cuda_version }}</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">PyTorch</td>
                <td class="text-right font-weight-medium">{{ stats.system.torch_version || '-' }}</td>
              </tr>
            </tbody>
          </v-table>
          <div v-else class="d-flex flex-column align-center justify-center pa-4">
            <v-icon size="48" color="grey" class="mb-2">mdi-chip</v-icon>
            <div class="text-body-2 text-medium-emphasis text-center">
              {{ stats.system?.torch_available ? 'PyTorch available (CPU only)' : 'No GPU detected' }}
            </div>
            <div v-if="stats.system?.torch_version" class="text-caption text-medium-emphasis mt-1">
              PyTorch {{ stats.system.torch_version }}
            </div>
          </div>
        </v-card>
      </v-col>

      <!-- Storage -->
      <v-col cols="12" md="4">
        <v-card class="pa-4" style="height: 100%;">
          <div class="d-flex align-center mb-3">
            <v-icon color="warning" class="mr-2">mdi-harddisk</v-icon>
            <h3 class="text-subtitle-1 font-weight-bold">Storage</h3>
            <v-spacer />
            <v-chip size="small" :color="(stats.system?.disk?.usage_percent || 0) > 90 ? 'error' : 'success'" variant="tonal">
              {{ stats.system?.disk?.usage_percent || 0 }}%
            </v-chip>
          </div>
          <v-table density="compact">
            <tbody>
              <tr>
                <td class="text-medium-emphasis">Disk Total</td>
                <td class="text-right font-weight-medium">{{ stats.system?.disk?.total_gb || '-' }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Disk Used</td>
                <td class="text-right font-weight-medium">{{ stats.system?.disk?.used_gb || '-' }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Disk Free</td>
                <td class="text-right font-weight-medium">{{ stats.system?.disk?.free_gb || '-' }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">RAM Total</td>
                <td class="text-right font-weight-medium">{{ stats.system?.memory?.total_gb || '-' }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">RAM Available</td>
                <td class="text-right font-weight-medium">{{ stats.system?.memory?.available_gb || '-' }} GB</td>
              </tr>
              <tr>
                <td class="text-medium-emphasis">Models on Disk</td>
                <td class="text-right font-weight-medium">{{ stats.system?.models_disk_mb || 0 }} MB</td>
              </tr>
            </tbody>
          </v-table>
        </v-card>
      </v-col>
    </v-row>

    <!-- Recent Models Table -->
    <v-row>
      <v-col cols="12">
        <v-card>
          <v-card-title class="d-flex align-center">
            <v-icon class="mr-2">mdi-history</v-icon>
            Recent Models
          </v-card-title>
          <v-data-table
            v-if="stats.recent_models?.length > 0"
            :headers="modelHeaders"
            :items="stats.recent_models"
            :items-per-page="10"
            density="comfortable"
          >
            <template #item.mode="{ item }">
              <v-chip
                size="small"
                :color="item.mode === 'anomaly' ? 'error' : item.mode === 'regression' ? 'purple' : 'success'"
                variant="tonal"
              >
                {{ item.mode }}
              </v-chip>
            </template>
            <template #item.algorithm="{ item }">
              <span class="font-weight-medium">{{ formatAlgorithm(item.algorithm) }}</span>
            </template>
            <template #item.metrics="{ item }">
              <span v-if="item.mode === 'regression' && item.metrics?.r2 != null">
                R²: {{ item.metrics.r2.toFixed(3) }}
              </span>
              <span v-else-if="item.metrics?.f1 != null">
                F1: {{ (item.metrics.f1 * 100).toFixed(1) }}%
              </span>
              <span v-else-if="item.metrics?.accuracy != null">
                Acc: {{ (item.metrics.accuracy * 100).toFixed(1) }}%
              </span>
              <span v-else class="text-medium-emphasis">-</span>
            </template>
            <template #item.created_at="{ item }">
              {{ formatDate(item.created_at) }}
            </template>
            <template #item.user_name="{ item }">
              <v-avatar size="24" color="primary" class="mr-1">
                <span class="text-caption">{{ (item.user_name || 'U')[0] }}</span>
              </v-avatar>
              {{ item.user_name || 'Unknown' }}
            </template>
          </v-data-table>
          <div v-else class="pa-8 text-center">
            <v-icon size="48" color="grey" class="mb-2">mdi-database-off</v-icon>
            <div class="text-body-1 text-medium-emphasis">No models saved yet</div>
            <div class="text-caption text-medium-emphasis mt-1">
              Train a model and save it as a benchmark to see it here
            </div>
          </div>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Doughnut, Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import api from '@/services/api'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
)

const loading = ref(false)
const stats = ref<any>({})

const hasModels = computed(() => (stats.value.models?.total || 0) > 0)

const modelHeaders = [
  { title: 'Name', key: 'name' },
  { title: 'User', key: 'user_name' },
  { title: 'Algorithm', key: 'algorithm' },
  { title: 'Mode', key: 'mode' },
  { title: 'Performance', key: 'metrics', sortable: false },
  { title: 'Date', key: 'created_at' },
]

const modeColors: Record<string, string> = {
  anomaly: '#EF4444',
  classification: '#10B981',
  regression: '#A855F7',
}

const modeChartData = computed(() => {
  const byMode = stats.value.models?.by_mode || {}
  const labels = Object.keys(byMode)
  return {
    labels: labels.map(m => m.charAt(0).toUpperCase() + m.slice(1)),
    datasets: [{
      data: labels.map(m => byMode[m]),
      backgroundColor: labels.map(m => modeColors[m] || '#6366F1'),
      borderWidth: 0,
    }]
  }
})

const algorithmChartData = computed(() => {
  const byAlgo = stats.value.models?.by_algorithm || {}
  const labels = Object.keys(byAlgo)
  return {
    labels: labels.map(a => formatAlgorithm(a)),
    datasets: [{
      label: 'Models',
      data: labels.map(a => byAlgo[a]),
      backgroundColor: '#6366F1',
      borderRadius: 6,
    }]
  }
})

const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'bottom' as const },
  }
}

const barOptions = {
  responsive: true,
  maintainAspectRatio: false,
  indexAxis: 'y' as const,
  plugins: {
    legend: { display: false },
  },
  scales: {
    x: {
      beginAtZero: true,
      ticks: { stepSize: 1 },
    }
  }
}

const algorithmNames: Record<string, string> = {
  iforest: 'Isolation Forest',
  lof: 'LOF',
  ocsvm: 'One-Class SVM',
  hbos: 'HBOS',
  rf: 'Random Forest',
  gb: 'Gradient Boosting',
  svm: 'SVM',
  mlp: 'MLP',
  dt: 'Decision Tree',
  knn: 'KNN',
  rf_reg: 'RF Regressor',
  xgb_reg: 'XGBoost Reg.',
  lgbm_reg: 'LightGBM Reg.',
  dt_reg: 'DT Regressor',
  svr: 'SVR',
  knn_reg: 'KNN Regressor',
  timesnet: 'TimesNet',
  custom: 'Custom Model',
}

function formatAlgorithm(algo: string): string {
  return algorithmNames[algo] || algo
}

function formatDate(dateStr: string): string {
  if (!dateStr) return '-'
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return dateStr
  }
}

async function fetchStats() {
  loading.value = true
  try {
    const response = await api.get('/api/admin/dashboard-stats')
    stats.value = response.data
  } catch (e: any) {
    console.error('Failed to fetch dashboard stats:', e)
  }
  loading.value = false
}

onMounted(() => {
  fetchStats()
})
</script>
