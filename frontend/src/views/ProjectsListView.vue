<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-6">
      <div>
        <h1 class="text-h4 font-weight-bold">Projects</h1>
        <p class="text-body-2 text-medium-emphasis">
          Track pipeline progress and deployments for each dataset.
        </p>
      </div>
      <v-spacer />
      <v-btn-toggle
        v-if="isAdmin"
        v-model="viewScope"
        density="comfortable"
        rounded="lg"
        color="primary"
        class="mr-3"
      >
        <v-btn value="mine" size="small">Mine</v-btn>
        <v-btn value="all" size="small">All users</v-btn>
      </v-btn-toggle>
      <v-btn color="primary" @click="openCreate">
        <v-icon start>mdi-plus</v-icon>
        New Project
      </v-btn>
    </div>

    <v-card class="pa-4">
      <v-table v-if="projects.length > 0" density="comfortable" hover>
        <thead>
          <tr>
            <th>Name</th>
            <th>Mode</th>
            <th class="text-center">Data</th>
            <th class="text-center">Windowing</th>
            <th class="text-center">Features</th>
            <th class="text-center">Training</th>
            <th class="text-center">Deploy</th>
            <th>Best</th>
            <th>Updated</th>
            <th class="text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in projects" :key="p.id">
            <td>
              <div class="font-weight-medium">{{ p.name }}</div>
              <div v-if="isAdmin && viewScope === 'all'" class="text-caption text-medium-emphasis">
                by {{ p.owner_username || `user #${p.user_id}` }}
              </div>
            </td>
            <td>
              <v-chip
                size="x-small"
                variant="tonal"
                :color="modeColor(p.mode)"
              >
                {{ p.mode || 'mixed' }}
              </v-chip>
            </td>
            <td class="text-center">
              <span :title="p.stages?.data?.summary || p.stages?.data?.status" style="cursor:pointer" @click="jumpToStage(p, 'data')">
                <v-icon :color="stageColor(p.stages?.data?.status)" size="20">{{ stageIcon(p.stages?.data?.status) }}</v-icon>
              </span>
            </td>
            <td class="text-center">
              <span :title="p.stages?.windowing?.summary || p.stages?.windowing?.status" style="cursor:pointer" @click="jumpToStage(p, 'windowing')">
                <v-icon :color="stageColor(p.stages?.windowing?.status)" size="20">{{ stageIcon(p.stages?.windowing?.status) }}</v-icon>
              </span>
            </td>
            <td class="text-center">
              <span :title="p.stages?.features?.summary || p.stages?.features?.status" style="cursor:pointer" @click="jumpToStage(p, 'features')">
                <v-icon :color="stageColor(p.stages?.features?.status)" size="20">{{ stageIcon(p.stages?.features?.status) }}</v-icon>
              </span>
            </td>
            <td class="text-center">
              <span :title="p.stages?.training?.summary || p.stages?.training?.status" style="cursor:pointer" @click="jumpToStage(p, 'training')">
                <v-icon :color="stageColor(p.stages?.training?.status)" size="20">{{ stageIcon(p.stages?.training?.status) }}</v-icon>
              </span>
            </td>
            <td class="text-center">
              <v-tooltip location="top">
                <template #activator="{ props }">
                  <v-chip
                    v-bind="props"
                    size="small"
                    variant="flat"
                    :color="deployColor(p.stages?.deploy?.status)"
                    style="cursor: pointer"
                    @click="jumpToStage(p, 'deploy')"
                  >
                    <v-icon start size="12">{{ deployIcon(p.stages?.deploy?.status) }}</v-icon>
                    {{ deployLabel(p) }}
                  </v-chip>
                </template>
                <div class="pa-1">
                  <div v-for="tgt in ['melab','app_builder','ti_mcu','jetson']" :key="tgt" class="text-caption">
                    <strong>{{ targetLabel(tgt) }}:</strong>
                    {{ p.deploy_breakdown?.[tgt]?.count || 0 }}
                    <span v-if="p.deploy_breakdown?.[tgt]?.items?.length">
                      —
                      <span
                        v-for="(it, i) in p.deploy_breakdown[tgt].items.slice(0, 3)"
                        :key="i"
                      >
                        {{ it.name }}<span v-if="i < Math.min(2, p.deploy_breakdown[tgt].items.length - 1)">, </span>
                      </span>
                      <span v-if="p.deploy_breakdown[tgt].items.length > 3">
                        &nbsp;(+{{ p.deploy_breakdown[tgt].items.length - 3 }})
                      </span>
                    </span>
                  </div>
                </div>
              </v-tooltip>
            </td>
            <td>
              <div v-if="p.best_metric" class="text-caption">
                <strong>{{ p.best_metric.name }}:</strong>
                {{ Number(p.best_metric.value).toFixed(3) }}
              </div>
              <div v-else class="text-caption text-medium-emphasis">—</div>
            </td>
            <td class="text-caption">{{ formatDate(p.updated_at || p.created_at) }}</td>
            <td class="text-right">
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="primary"
                title="Continue at current stage"
                @click="continueAtCurrentStage(p)"
              >
                <v-icon size="small">mdi-arrow-right-circle-outline</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="info"
                title="View detail"
                :to="{ path: `/projects/${p.id}` }"
              >
                <v-icon size="small">mdi-eye-outline</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="secondary"
                title="Clone project (swap dataset)"
                @click="openClone(p)"
              >
                <v-icon size="small">mdi-content-copy</v-icon>
              </v-btn>
              <v-btn
                icon
                size="x-small"
                variant="text"
                color="error"
                title="Delete"
                @click="openDelete(p)"
              >
                <v-icon size="small">mdi-delete-outline</v-icon>
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>

      <div v-else-if="!loading" class="text-center pa-8">
        <v-icon size="48" color="grey" class="mb-3">mdi-folder-multiple-outline</v-icon>
        <div class="text-body-1 text-medium-emphasis">No projects yet.</div>
        <div class="text-caption text-medium-emphasis mt-1">
          A project is created automatically the first time you Apply Windowing.
        </div>
        <v-btn color="primary" variant="tonal" class="mt-4" @click="openCreate">
          <v-icon start>mdi-plus</v-icon>
          New Project
        </v-btn>
      </div>

      <div v-else class="text-center pa-8">
        <v-progress-circular indeterminate color="primary" size="36" />
      </div>
    </v-card>

    <!-- Create Dialog -->
    <v-dialog v-model="showCreate" max-width="480">
      <v-card>
        <v-card-title>New Project</v-card-title>
        <v-card-text>
          <v-text-field v-model="createForm.name" label="Name" autofocus />
          <v-select
            v-model="createForm.mode"
            :items="['anomaly','classification','regression']"
            label="Mode"
          />
          <v-textarea v-model="createForm.description" label="Description (optional)" rows="2" />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCreate = false">Cancel</v-btn>
          <v-btn color="primary" :loading="saving" @click="createProject">Create</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Clone Dialog -->
    <v-dialog v-model="showClone" max-width="600">
      <v-card v-if="cloneSrc">
        <v-card-title>Clone {{ cloneSrc.name }}</v-card-title>
        <v-card-text>
          <v-text-field v-model="cloneForm.name" label="New project name" />
          <p class="text-caption text-medium-emphasis mb-2">
            The clone copies configuration and the feature template only. Data,
            windowing, features and models must be regenerated. Open the clone
            to pick its dataset from the Data Source page.
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showClone = false">Cancel</v-btn>
          <v-btn color="primary" :loading="saving" @click="doClone">Clone</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete Dialog -->
    <v-dialog v-model="showDelete" max-width="440">
      <v-card v-if="deleteTarget">
        <v-card-title class="d-flex align-center">
          <v-icon color="warning" class="mr-2">mdi-alert</v-icon>
          Delete {{ deleteTarget.name }}?
        </v-card-title>
        <v-card-text>
          <p>This removes project rows and pipeline snapshots. Saved models,
          ME-LAB endpoints and App Builder apps stay but detach from the
          project.</p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showDelete = false">Cancel</v-btn>
          <v-btn color="error" :loading="saving" @click="doDelete">Delete</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/services/api'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import { usePipelineStore } from '@/stores/pipeline'

const router = useRouter()
const auth = useAuthStore()
const notify = useNotificationStore()
const pipeline = usePipelineStore()

const isAdmin = computed(() => auth.isAdmin)

interface StageCell { status?: string; summary?: string; id?: number }
interface Project {
  id: number
  name: string
  mode: string
  user_id: number
  owner_username?: string
  current_stage?: string
  updated_at?: string
  created_at?: string
  stages?: Record<string, StageCell>
  deploy_breakdown?: Record<string, { count: number; items: any[] }>
  best_metric?: { name: string; value: number } | null
}

const projects = ref<Project[]>([])
const loading = ref(false)
const viewScope = ref<'mine' | 'all'>('mine')

const showCreate = ref(false)
const createForm = ref({ name: '', mode: 'classification', description: '' })
const saving = ref(false)

const showClone = ref(false)
const cloneSrc = ref<Project | null>(null)
const cloneForm = ref({ name: '' })

const showDelete = ref(false)
const deleteTarget = ref<Project | null>(null)

let pollTimer: ReturnType<typeof setInterval> | null = null

async function load() {
  try {
    loading.value = true
    const qs = (isAdmin.value && viewScope.value === 'all') ? '?all=1' : ''
    const res = await api.get(`/api/projects${qs}`)
    projects.value = res.data?.projects || []
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load projects')
  } finally {
    loading.value = false
  }
}

function openCreate() {
  createForm.value = { name: '', mode: 'classification', description: '' }
  showCreate.value = true
}

async function createProject() {
  if (!createForm.value.name.trim()) {
    notify.showError('Name is required')
    return
  }
  try {
    saving.value = true
    await api.post('/api/projects', createForm.value)
    notify.showSuccess('Project created')
    showCreate.value = false
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to create')
  } finally {
    saving.value = false
  }
}

function openClone(p: Project) {
  cloneSrc.value = p
  cloneForm.value = { name: `${p.name} (clone)` }
  showClone.value = true
}

async function doClone() {
  if (!cloneSrc.value) return
  try {
    saving.value = true
    const res = await api.post(`/api/projects/${cloneSrc.value.id}/clone`, cloneForm.value)
    notify.showSuccess('Project cloned')
    showClone.value = false
    // Redirect to Data page with the cloned project active
    if (res.data?.id) {
      await pipeline.setActiveProject(res.data.id)
      router.push({ name: 'pipeline-data', query: { project_id: String(res.data.id) } })
    } else {
      await load()
    }
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Clone failed')
  } finally {
    saving.value = false
  }
}

function openDelete(p: Project) {
  deleteTarget.value = p
  showDelete.value = true
}

async function doDelete() {
  if (!deleteTarget.value) return
  try {
    saving.value = true
    await api.delete(`/api/projects/${deleteTarget.value.id}`)
    notify.showSuccess('Project deleted')
    showDelete.value = false
    await load()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Delete failed')
  } finally {
    saving.value = false
  }
}

async function jumpToStage(p: Project, stage: string) {
  await pipeline.setActiveProject(p.id)
  const routeMap: Record<string, string> = {
    data: 'pipeline-data',
    windowing: 'pipeline-windowing',
    features: 'pipeline-features',
    training: 'pipeline-training',
    deploy: 'pipeline-deploy',
  }
  router.push({ name: routeMap[stage] || 'pipeline-data', query: { project_id: String(p.id) } })
}

async function continueAtCurrentStage(p: Project) {
  const stage = p.current_stage || 'data'
  await jumpToStage(p, stage)
}

function modeColor(mode?: string) {
  switch ((mode || '').toLowerCase()) {
    case 'anomaly': return 'purple'
    case 'classification': return 'blue'
    case 'regression': return 'teal'
    case 'mixed': return 'grey'
    default: return 'grey'
  }
}
function deployColor(status?: string) {
  switch (status) {
    case 'complete': return 'success'
    case 'in_progress': return 'warning'
    case 'not_started': return 'grey'
    default: return 'grey'
  }
}
function deployIcon(status?: string) {
  switch (status) {
    case 'complete': return 'mdi-check-circle'
    case 'in_progress': return 'mdi-progress-clock'
    default: return 'mdi-circle-outline'
  }
}
function deployLabel(p: Project) {
  const bd = p.deploy_breakdown || {}
  const total = ['melab', 'app_builder', 'ti_mcu', 'jetson']
    .reduce((s, k) => s + (bd[k]?.count || 0), 0)
  const targets = ['melab', 'app_builder', 'ti_mcu', 'jetson']
    .filter(k => (bd[k]?.count || 0) > 0).length
  if (total === 0) return 'none'
  return `${total} / ${targets} tgt`
}
function targetLabel(t: string) {
  return ({ melab: 'ME-LAB', app_builder: 'App Builder', ti_mcu: 'TI MCU', jetson: 'Jetson' } as any)[t] || t
}

function formatDate(dt?: string | null) {
  if (!dt) return '—'
  try {
    return new Date(dt).toLocaleString()
  } catch { return dt }
}

function stageIcon(status?: string) {
  switch (status) {
    case 'complete': return 'mdi-check-circle'
    case 'in_progress': return 'mdi-progress-clock'
    case 'skipped': return 'mdi-minus-circle'
    case 'error': return 'mdi-alert-circle'
    default: return 'mdi-circle-outline'
  }
}
function stageColor(status?: string) {
  switch (status) {
    case 'complete': return 'success'
    case 'in_progress': return 'warning'
    case 'skipped': return 'grey-lighten-1'
    case 'error': return 'error'
    default: return 'grey'
  }
}

watch(viewScope, () => load())

onMounted(() => {
  load()
  // Auto-refresh every 15s while page visible
  const tick = () => {
    if (document.visibilityState !== 'hidden') load()
  }
  pollTimer = setInterval(tick, 15000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
