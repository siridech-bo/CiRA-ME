<template>
  <v-dialog
    :model-value="modelValue"
    max-width="640"
    scrollable
    persistent
    @update:model-value="onDialogUpdate"
  >
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon start>mdi-link-variant</v-icon>
        Rebind deploy targets
        <v-spacer />
        <v-btn icon size="small" variant="text" @click="onCancel">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-card-title>

      <v-divider />

      <v-card-text style="max-height: 65vh;">
        <div class="mb-2">
          <div class="text-caption text-medium-emphasis">Model</div>
          <div class="font-weight-medium">{{ modelName || `#${savedModelId}` }}</div>
        </div>

        <v-alert type="info" variant="tonal" density="compact" class="mb-3">
          Rebinding changes which machines can serve this model in production.
          It does not affect the model file itself or its training history —
          <strong>trained_on</strong> bindings stay locked to the original set.
        </v-alert>

        <div v-if="loading" class="pa-4 text-center text-caption">
          <v-progress-circular indeterminate size="18" width="2" class="mr-2" />
          Loading current bindings…
        </div>

        <template v-else>
          <div v-if="trainedOn.length > 0" class="mb-3">
            <div class="text-caption font-weight-bold mb-1">
              Trained on ({{ trainedOn.length }})
            </div>
            <div>
              <v-chip
                v-for="m in trainedOn"
                :key="`t-${m.asset_id}`"
                size="small"
                variant="tonal"
                color="info"
                class="mr-1 mb-1"
              >
                {{ m.display_name || m.name }}
              </v-chip>
            </div>
          </div>

          <v-divider class="my-3" />

          <div class="d-flex align-center mb-2">
            <span class="text-subtitle-2">Deploy targets</span>
            <v-spacer />
            <v-checkbox
              v-model="hideRetired"
              label="Hide retired"
              density="compact"
              hide-details
            />
          </div>

          <v-text-field
            v-model="query"
            density="compact"
            variant="outlined"
            hide-details
            placeholder="Search machines…"
            prepend-inner-icon="mdi-magnify"
            clearable
            class="mb-2"
          />

          <div class="picker-container">
            <div v-if="assetTreeStore.loadingTree" class="pa-4 text-center text-caption">
              Loading tree…
            </div>
            <template v-else>
              <MachineTreePickerNode
                v-for="root in visibleTree"
                :key="root.id"
                :node="root"
                :depth="0"
                :machine-level="assetTreeStore.machineLevel"
                :selected-ids="selectedSet"
                :hide-retired="hideRetired"
                @toggle-machine="onToggleMachine"
              />
            </template>
          </div>

          <div class="mt-2 text-body-2">
            <strong>{{ selectedIds.length }}</strong>
            {{ selectedIds.length === 1 ? 'machine' : 'machines' }} selected
          </div>
        </template>
      </v-card-text>

      <v-divider />

      <v-card-actions class="px-4 py-3">
        <v-btn variant="text" @click="onCancel">Cancel</v-btn>
        <v-spacer />
        <v-btn
          color="primary"
          :loading="saving"
          :disabled="!hasChanges"
          @click="onSave"
        >
          Save deploy targets
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
/**
 * Phase C.5 — Rebind machines dialog.
 * Backed by GET/PATCH /api/asset-tree/models/<id>/deploy-targets endpoints.
 * Opens with the model's current deploy_targets preselected; user checks /
 * unchecks machines and saves. Trained-on bindings are shown for context
 * but are read-only.
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'
import MachineTreePickerNode from './MachineTreePickerNode.vue'

const props = defineProps<{
  modelValue: boolean
  savedModelId: number | null
  modelName?: string | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'saved', payload: { savedModelId: number; deployed_to: any[] }): void
}>()

const assetTreeStore = useAssetTreeStore()
const notify = useNotificationStore()

const loading = ref(false)
const saving = ref(false)
const trainedOn = ref<any[]>([])
const originalDeployIds = ref<number[]>([])
const selectedIds = ref<number[]>([])
const selectedSet = computed(() => new Set(selectedIds.value))

const query = ref('')
const hideRetired = ref(true)

async function fetchBindings() {
  if (props.savedModelId == null) return
  loading.value = true
  try {
    const r = await api.get(`/api/asset-tree/models/${props.savedModelId}/bindings`)
    trainedOn.value = r.data?.trained_on || []
    const deployIds = (r.data?.deployed_to || [])
      .map((m: any) => m.asset_id)
      .filter((n: any) => Number.isFinite(n))
    originalDeployIds.value = [...deployIds]
    selectedIds.value = [...deployIds]
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load bindings')
  } finally {
    loading.value = false
  }
}

watch(
  () => props.modelValue,
  async (open) => {
    if (!open) return
    query.value = ''
    // Ensure the tree is loaded before showing checkboxes.
    if (!assetTreeStore.treeLoaded && !assetTreeStore.loadingTree) {
      await assetTreeStore.fetchTree()
    }
    await fetchBindings()
  },
)

onMounted(async () => {
  if (!assetTreeStore.treeLoaded && !assetTreeStore.loadingTree) {
    await assetTreeStore.fetchTree()
  }
})

function pruneTree(nodes: AssetNode[]): AssetNode[] {
  const q = (query.value || '').trim().toLowerCase()
  const walk = (list: AssetNode[]): AssetNode[] => {
    const out: AssetNode[] = []
    for (const n of list) {
      const kids = n.children ? walk(n.children) : []
      const isRetired = n.status === 'retired'
      if (isRetired && hideRetired.value && kids.length === 0) continue
      const hay = `${n.name} ${n.display_name || ''}`.toLowerCase()
      const selfMatch = !q || hay.includes(q)
      if (selfMatch || kids.length > 0) {
        out.push({ ...n, children: kids })
      }
    }
    return out
  }
  return walk(nodes)
}

const visibleTree = computed(() => pruneTree(assetTreeStore.tree))

const hasChanges = computed(() => {
  const a = [...originalDeployIds.value].sort().join(',')
  const b = [...selectedIds.value].sort().join(',')
  return a !== b
})

function onToggleMachine(node: AssetNode) {
  const id = node.id
  const idx = selectedIds.value.indexOf(id)
  if (idx >= 0) selectedIds.value.splice(idx, 1)
  else selectedIds.value.push(id)
}

async function onSave() {
  if (props.savedModelId == null) return
  saving.value = true
  try {
    const r = await api.patch(
      `/api/asset-tree/models/${props.savedModelId}/deploy-targets`,
      { machine_asset_ids: selectedIds.value },
    )
    notify.showSuccess('Deploy targets updated')
    emit('saved', {
      savedModelId: props.savedModelId,
      deployed_to: r.data?.deployed_to || [],
    })
    emit('update:modelValue', false)
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Rebind failed')
  } finally {
    saving.value = false
  }
}

function onCancel() {
  emit('update:modelValue', false)
}
function onDialogUpdate(v: boolean) {
  if (!v) onCancel()
}
</script>

<style scoped>
.picker-container {
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 6px;
  max-height: 280px;
  overflow-y: auto;
  padding: 4px;
  background: rgba(var(--v-theme-surface-bright), 0.4);
}
</style>
