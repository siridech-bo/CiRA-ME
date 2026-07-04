<template>
  <v-container fluid class="pa-6">
    <div v-if="loading" class="text-center pa-8">
      <v-progress-circular indeterminate />
    </div>
    <template v-else-if="project">
      <div class="d-flex align-center mb-6">
        <v-btn icon size="small" variant="text" :to="{ name: 'projects-list' }">
          <v-icon>mdi-arrow-left</v-icon>
        </v-btn>
        <div class="ml-3">
          <h1 class="text-h4 font-weight-bold">{{ project.name }}</h1>
          <div class="text-caption text-medium-emphasis">
            <v-chip size="x-small" variant="tonal" class="mr-2">{{ project.mode || 'mixed' }}</v-chip>
            id #{{ project.id }} · updated {{ formatDate(project.updated_at) }}
          </div>
        </div>
        <v-spacer />
        <v-btn variant="tonal" color="primary" @click="continueAtCurrentStage">
          <v-icon start>mdi-arrow-right-circle-outline</v-icon>
          Continue
        </v-btn>
      </div>

      <v-row>
        <v-col cols="12" md="6" lg="4">
          <v-card class="pa-4 h-100">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-database</v-icon> Data
            </div>
            <div v-if="latestData" class="text-body-2">
              <div><strong>Format:</strong> {{ latestData.format }}</div>
              <div><strong>Rows:</strong> {{ latestData.total_rows }}</div>
              <div class="text-caption text-medium-emphasis">
                {{ latestData.file_path }}
              </div>
            </div>
            <div v-else class="text-caption text-medium-emphasis">Not yet ingested.</div>
            <v-btn variant="text" color="primary" size="small" class="mt-2" @click="jumpTo('data')">
              Open Data Source
            </v-btn>
          </v-card>
        </v-col>

        <v-col cols="12" md="6" lg="4">
          <v-card class="pa-4 h-100">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-tune-vertical</v-icon> Windowing
            </div>
            <div v-if="latestWindowed" class="text-body-2">
              <div><strong>Windows:</strong> {{ latestWindowed.num_windows }}</div>
              <div v-if="parseJson(latestWindowed.config)?.window_size">
                <strong>Window size:</strong> {{ parseJson(latestWindowed.config).window_size }}
              </div>
              <div v-if="parseJson(latestWindowed.config)?.stride">
                <strong>Stride:</strong> {{ parseJson(latestWindowed.config).stride }}
              </div>
            </div>
            <div v-else class="text-caption text-medium-emphasis">Not yet windowed.</div>
            <v-btn variant="text" color="primary" size="small" class="mt-2" @click="jumpTo('windowing')">
              Open Windowing
            </v-btn>
          </v-card>
        </v-col>

        <v-col cols="12" md="6" lg="4">
          <v-card class="pa-4 h-100">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-auto-fix</v-icon> Features
            </div>
            <div v-if="latestFeature" class="text-body-2">
              <div><strong>Method:</strong> {{ latestFeature.method }}</div>
              <div><strong>Features:</strong> {{ latestFeature.num_features }}</div>
              <div v-if="project.feature_template" class="text-caption">
                Template active (v{{ project.feature_template.version }}) —
                {{ project.feature_template.ordered_feature_names?.length }} columns pinned
              </div>
            </div>
            <div v-else class="text-caption text-medium-emphasis">Not yet extracted.</div>
            <v-btn variant="text" color="primary" size="small" class="mt-2" @click="jumpTo('features')">
              Open Features
            </v-btn>
          </v-card>
        </v-col>

        <v-col cols="12" md="6" lg="4">
          <v-card class="pa-4 h-100">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-brain</v-icon> Training
            </div>
            <div v-if="project.saved_models?.length" class="text-body-2">
              <div><strong>{{ project.saved_models.length }}</strong> saved model(s)</div>
              <ul class="text-caption pl-4">
                <li v-for="m in project.saved_models.slice(0, 4)" :key="m.id">
                  {{ m.name }} ({{ m.algorithm }})
                </li>
                <li v-if="project.saved_models.length > 4" class="text-medium-emphasis">
                  +{{ project.saved_models.length - 4 }} more
                </li>
              </ul>
            </div>
            <div v-else class="text-caption text-medium-emphasis">No models saved.</div>
            <v-btn variant="text" color="primary" size="small" class="mt-2" @click="jumpTo('training')">
              Open Training
            </v-btn>
          </v-card>
        </v-col>

        <v-col cols="12" md="6" lg="8">
          <v-card class="pa-4 h-100">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-rocket-launch</v-icon> Deploy
            </div>
            <div class="d-flex flex-wrap gap-2 mb-2">
              <template v-for="tgt in ['melab','app_builder','ti_mcu','jetson']" :key="tgt">
                <div class="mr-4">
                  <div class="text-caption text-medium-emphasis">{{ targetLabel(tgt) }}</div>
                  <div class="text-h6">{{ countFor(tgt) }}</div>
                </div>
              </template>
            </div>
            <div class="text-body-2 mt-2">
              <div v-if="project.melab_endpoints?.length" class="mb-1">
                <strong>ME-LAB endpoints:</strong>
                <v-chip
                  v-for="e in project.melab_endpoints"
                  :key="e.id"
                  size="x-small"
                  class="ml-1"
                  variant="tonal"
                >
                  {{ e.name }}
                </v-chip>
              </div>
              <div v-if="project.app_builder_apps?.length" class="mb-1">
                <strong>Apps:</strong>
                <v-chip
                  v-for="a in project.app_builder_apps"
                  :key="a.id"
                  size="x-small"
                  class="ml-1"
                  variant="tonal"
                >
                  {{ a.name }}
                </v-chip>
              </div>
              <div v-if="tiRecords.length" class="mb-1">
                <strong>TI MCU packages:</strong>
                <v-chip
                  v-for="d in tiRecords"
                  :key="d.id"
                  size="x-small"
                  class="ml-1"
                  variant="tonal"
                >
                  {{ d.ref_id || `#${d.id}` }}
                </v-chip>
              </div>
              <div v-if="jetsonRecords.length" class="mb-1">
                <strong>Jetson deploys:</strong>
                <v-chip
                  v-for="d in jetsonRecords"
                  :key="d.id"
                  size="x-small"
                  class="ml-1"
                  variant="tonal"
                >
                  {{ d.ref_id || `#${d.id}` }}
                </v-chip>
              </div>
            </div>
          </v-card>
        </v-col>

        <v-col cols="12">
          <v-card class="pa-4">
            <div class="text-subtitle-2 font-weight-bold mb-2">
              <v-icon start size="small">mdi-file-tree</v-icon> Feature Template
              <span v-if="project.feature_template" class="text-caption text-medium-emphasis ml-2">
                v{{ project.feature_template.version }} — updated {{ formatDate(project.feature_template.updated_at) }}
              </span>
            </div>
            <p class="text-caption text-medium-emphasis mb-3">
              The template pins the ordered feature contract so ME-LAB /
              App Builder payload shape stays stable across retrainings.
            </p>
            <v-alert
              v-if="!templateItems.length"
              type="info"
              variant="tonal"
              density="compact"
              class="mb-2"
            >
              No template yet. Extract features on this project once to populate it, then reorder here.
            </v-alert>
            <v-list v-else density="compact" class="mb-2" style="max-height: 320px; overflow-y: auto;">
              <v-list-item v-for="(feat, idx) in templateItems" :key="feat + idx">
                <template #prepend>
                  <span class="text-caption text-medium-emphasis mr-2">{{ idx + 1 }}.</span>
                </template>
                <v-list-item-title class="text-body-2">{{ feat }}</v-list-item-title>
                <template #append>
                  <v-btn icon size="x-small" variant="text" :disabled="idx === 0" @click="moveUp(idx)">
                    <v-icon size="small">mdi-arrow-up</v-icon>
                  </v-btn>
                  <v-btn icon size="x-small" variant="text" :disabled="idx === templateItems.length - 1" @click="moveDown(idx)">
                    <v-icon size="small">mdi-arrow-down</v-icon>
                  </v-btn>
                  <v-btn icon size="x-small" variant="text" color="error" @click="removeAt(idx)">
                    <v-icon size="small">mdi-close</v-icon>
                  </v-btn>
                </template>
              </v-list-item>
            </v-list>

            <v-btn variant="tonal" color="primary" size="small" :loading="templateSaving" @click="saveTemplate">
              Save Template
            </v-btn>
            <v-btn variant="text" size="small" class="ml-2" @click="importFromLatestExtract">
              Import from latest extraction
            </v-btn>
          </v-card>
        </v-col>
      </v-row>
    </template>
    <div v-else class="text-center pa-8 text-medium-emphasis">
      Project not found.
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'
import { usePipelineStore } from '@/stores/pipeline'

const route = useRoute()
const router = useRouter()
const notify = useNotificationStore()
const pipeline = usePipelineStore()

const project = ref<any>(null)
const loading = ref(false)
const templateItems = ref<string[]>([])
const templateSaving = ref(false)

const latestData = computed(() => project.value?.data_sessions?.[0])
const latestWindowed = computed(() => project.value?.windowed_sessions?.[0])
const latestFeature = computed(() => project.value?.feature_sessions?.[0])

const tiRecords = computed(() => project.value?.deploy_records?.filter((r: any) => r.target === 'ti_mcu') || [])
const jetsonRecords = computed(() => project.value?.deploy_records?.filter((r: any) => r.target === 'jetson') || [])

async function load() {
  try {
    loading.value = true
    const res = await api.get(`/api/projects/${route.params.id}`)
    project.value = res.data
    templateItems.value = [...(project.value?.feature_template?.ordered_feature_names || [])]
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load project')
  } finally {
    loading.value = false
  }
}

function parseJson(v: any) {
  if (!v) return null
  if (typeof v === 'object') return v
  try { return JSON.parse(v) } catch { return null }
}

function countFor(t: string) {
  if (!project.value) return 0
  if (t === 'melab') return (project.value.melab_endpoints || []).filter((e: any) => e.status === 'active').length
  if (t === 'app_builder') return (project.value.app_builder_apps || []).filter((a: any) => a.status === 'published').length
  if (t === 'ti_mcu') return tiRecords.value.filter((r: any) => r.status === 'active').length
  if (t === 'jetson') return jetsonRecords.value.filter((r: any) => r.status === 'active').length
  return 0
}
function targetLabel(t: string) {
  return ({ melab: 'ME-LAB', app_builder: 'App Builder', ti_mcu: 'TI MCU', jetson: 'Jetson' } as any)[t] || t
}

function formatDate(dt?: string | null) {
  if (!dt) return '—'
  try { return new Date(dt).toLocaleString() } catch { return dt }
}

async function jumpTo(stage: string) {
  await pipeline.setActiveProject(project.value.id)
  const routeMap: Record<string, string> = {
    data: 'pipeline-data',
    windowing: 'pipeline-windowing',
    features: 'pipeline-features',
    training: 'pipeline-training',
    deploy: 'pipeline-deploy',
  }
  router.push({ name: routeMap[stage] || 'pipeline-data', query: { project_id: String(project.value.id) } })
}

async function continueAtCurrentStage() {
  const stage = project.value.current_stage || 'data'
  await jumpTo(stage)
}

function moveUp(i: number) {
  if (i === 0) return
  const t = templateItems.value.slice()
  ;[t[i - 1], t[i]] = [t[i], t[i - 1]]
  templateItems.value = t
}
function moveDown(i: number) {
  if (i === templateItems.value.length - 1) return
  const t = templateItems.value.slice()
  ;[t[i + 1], t[i]] = [t[i], t[i + 1]]
  templateItems.value = t
}
function removeAt(i: number) {
  templateItems.value = templateItems.value.filter((_, idx) => idx !== i)
}

function importFromLatestExtract() {
  const feat = latestFeature.value
  if (!feat) {
    notify.showError('No feature extraction rows on this project yet.')
    return
  }
  let names: string[] = []
  try {
    names = typeof feat.feature_names === 'string'
      ? JSON.parse(feat.feature_names)
      : (feat.feature_names || [])
  } catch { names = [] }
  templateItems.value = names
  notify.showSuccess(`Imported ${names.length} features`)
}

async function saveTemplate() {
  try {
    templateSaving.value = true
    await api.put(`/api/projects/${project.value.id}/feature-template`, {
      ordered_feature_names: templateItems.value,
    })
    notify.showSuccess('Feature template saved')
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Save failed')
  } finally {
    templateSaving.value = false
  }
}

onMounted(load)
</script>
