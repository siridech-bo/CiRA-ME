<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-1">
      <h1 class="text-h4 font-weight-bold">Solution Templates</h1>
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6" style="max-width: 780px">
      Pre-built, end-to-end pipelines for specific industrial use cases. Each
      template pre-selects the data profile, windowing, physics-aware features,
      model, and deploy target — so you don't need to know the DSP details.
      <strong>Ready</strong> templates can be used now; the physics-aware
      extractor for each ships per phase.
    </p>

    <v-alert
      v-if="error"
      type="error" variant="tonal" density="compact" class="mb-4"
    >{{ error }}</v-alert>

    <v-row v-if="loading">
      <v-col v-for="n in 3" :key="n" cols="12" md="4">
        <v-skeleton-loader type="card" />
      </v-col>
    </v-row>

    <v-row v-else>
      <v-col
        v-for="s in solutions"
        :key="s.id"
        cols="12" md="4"
      >
        <v-card
          class="h-100 d-flex flex-column"
          :variant="s.status === 'ready' ? 'elevated' : 'tonal'"
        >
          <v-card-item>
            <template #prepend>
              <v-avatar :color="s.status === 'ready' ? 'primary' : 'grey'" variant="tonal">
                <v-icon>{{ s.icon || 'mdi-shape-outline' }}</v-icon>
              </v-avatar>
            </template>
            <v-card-title class="d-flex align-center">
              {{ s.display_name }}
              <v-chip
                class="ml-2"
                size="x-small"
                :color="s.status === 'ready' ? 'success' : 'grey'"
                variant="flat"
              >{{ s.status === 'ready' ? 'Ready' : 'Coming soon' }}</v-chip>
            </v-card-title>
          </v-card-item>

          <v-card-text class="flex-grow-1">
            <p class="text-body-2 mb-4">{{ s.description }}</p>

            <div class="recipe">
              <div class="recipe-row">
                <span class="recipe-key">Signal</span>
                <span>{{ s.data_profile.channels }} ch @ {{ s.data_profile.sample_rate_hz }} Hz</span>
              </div>
              <div class="recipe-row">
                <span class="recipe-key">Window</span>
                <span>{{ s.windowing.window_s }} s</span>
              </div>
              <div class="recipe-row">
                <span class="recipe-key">Features</span>
                <span>{{ s.feature_extractor }}</span>
              </div>
              <div class="recipe-row">
                <span class="recipe-key">Model</span>
                <span>{{ s.models.default }} / {{ s.models.advanced }}</span>
              </div>
              <div class="recipe-row">
                <span class="recipe-key">Deploy</span>
                <span>{{ (s.deploy.targets || []).join(', ') }} ({{ s.deploy.format }})</span>
              </div>
            </div>
          </v-card-text>

          <v-card-actions>
            <v-btn
              :disabled="s.status !== 'ready'"
              color="primary"
              variant="flat"
              @click="startSolution(s)"
            >
              <v-icon start>mdi-play</v-icon>
              {{ s.status === 'ready' ? 'Start' : 'Not yet available' }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'
import { usePipelineStore } from '@/stores/pipeline'

const router = useRouter()
const notify = useNotificationStore()
const pipeline = usePipelineStore()

const solutions = ref<any[]>([])
const loading = ref(true)
const error = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    const resp = await api.get('/api/solutions')
    solutions.value = resp.data.solutions || []
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Failed to load solution templates'
  } finally {
    loading.value = false
  }
}

// Each Ready solution opens its own self-contained app (a dedicated, guided
// flow). setActiveSolution seeds that app's defaults (sample rate, window).
const SOLUTION_ROUTES: Record<string, string> = {
  motor_current_mcsa: 'mcsa-solution',
  machine_vibration: 'vibration-solution',
  pump_analysis: 'pump-solution',
  pump_fusion: 'pump-fusion-solution',
}
function startSolution(s: any) {
  pipeline.setActiveSolution(s)
  const routeName = SOLUTION_ROUTES[s.id]
  if (routeName) {
    router.push({ name: routeName })
  } else {
    notify.showInfo(`"${s.display_name}" does not have a dedicated app yet.`)
  }
}

onMounted(load)
</script>

<style scoped lang="scss">
.recipe {
  border-top: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  padding-top: 8px;
}
.recipe-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 0.8rem;
  padding: 2px 0;
}
.recipe-key {
  color: rgb(var(--v-theme-on-surface));
  opacity: 0.6;
  font-weight: 600;
}
</style>
