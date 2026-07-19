<template>
  <v-dialog
    :model-value="modelValue"
    max-width="620"
    scrollable
    persistent
    @update:model-value="onDialogUpdate"
  >
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon start>mdi-file-tree</v-icon>
        {{ title || 'Select machines' }}
        <v-spacer />
        <v-btn icon size="small" variant="text" @click="onCancel">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-card-title>

      <v-divider />

      <div class="picker-toolbar px-4 py-2">
        <v-text-field
          v-model="query"
          density="compact"
          variant="outlined"
          hide-details
          placeholder="Search machines…"
          prepend-inner-icon="mdi-magnify"
          clearable
        />
        <div class="d-flex align-center mt-2 flex-wrap ga-2">
          <v-checkbox
            v-model="hideRetired"
            label="Hide retired"
            density="compact"
            hide-details
          />
          <v-spacer />
          <v-btn
            variant="text"
            size="small"
            @click="selectAllVisible"
          >
            Select all visible
          </v-btn>
          <v-btn
            variant="text"
            size="small"
            @click="clearAll"
          >
            Clear
          </v-btn>
        </div>
      </div>

      <v-divider />

      <v-card-text style="max-height: 55vh;">
        <div v-if="assetTreeStore.loadingTree" class="pa-4 text-center text-caption">
          Loading tree…
        </div>
        <div
          v-else-if="!assetTreeStore.tree || assetTreeStore.tree.length === 0"
          class="pa-6 text-center text-caption text-medium-emphasis"
        >
          No tree yet. Set one up in Settings → Asset Tree.
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
      </v-card-text>

      <v-divider />

      <v-card-actions class="px-4 py-3">
        <div class="text-body-2">
          <strong>{{ selectedIds.length }}</strong>
          {{ selectedIds.length === 1 ? 'machine' : 'machines' }} selected
        </div>
        <v-spacer />
        <v-btn variant="text" @click="onCancel">Cancel</v-btn>
        <v-btn
          color="primary"
          :disabled="!allowEmpty && selectedIds.length === 0"
          @click="onSave"
        >
          Confirm selection
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
/**
 * Phase C — reusable machine tree picker dialog.
 * Renders the current asset tree with a checkbox on each machine-level node
 * (non-machine rows are expandable but never selectable). Filters retired
 * machines by default. Emits the final id list on Save via
 * v-model:selected. Retains the initial preselection until the user
 * cancels.
 *
 * Used by:
 *  - MachineGroupEditDialog (Machine Groups edit modal)
 *  - TrainingView (Ad-hoc scope picker)
 *  - RebindMachinesDialog (deploy-targets rebind)
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'
import MachineTreePickerNode from './MachineTreePickerNode.vue'

const props = defineProps<{
  modelValue: boolean
  selected: number[]
  title?: string
  allowEmpty?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'update:selected', ids: number[]): void
  (e: 'confirm', ids: number[]): void
  (e: 'cancel'): void
}>()

const assetTreeStore = useAssetTreeStore()

const query = ref('')
const hideRetired = ref(true)
// Local mutable snapshot — we only commit on Save.
const draft = ref<number[]>([...props.selected])

const selectedSet = computed(() => new Set(draft.value))
const selectedIds = computed(() => draft.value)

// Reset draft each time the dialog opens.
watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      draft.value = [...props.selected]
      query.value = ''
    }
  },
)

onMounted(async () => {
  if (!assetTreeStore.treeLoaded && !assetTreeStore.loadingTree) {
    await assetTreeStore.fetchTree()
  }
})

// Prune the tree so search + hide-retired trim children while keeping
// ancestor rows visible enough to reach matches.
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

function collectMachineIds(nodes: AssetNode[], out: number[] = []): number[] {
  for (const n of nodes) {
    if (assetTreeStore.isMachineNode(n)) {
      if (!(hideRetired.value && n.status === 'retired')) {
        out.push(n.id)
      }
    }
    if (n.children) collectMachineIds(n.children, out)
  }
  return out
}

function selectAllVisible() {
  const ids = collectMachineIds(visibleTree.value)
  const set = new Set([...draft.value, ...ids])
  draft.value = [...set]
}
function clearAll() {
  draft.value = []
}

function onToggleMachine(node: AssetNode) {
  const id = node.id
  const idx = draft.value.indexOf(id)
  if (idx >= 0) draft.value.splice(idx, 1)
  else draft.value.push(id)
}

function onSave() {
  emit('update:selected', [...draft.value])
  emit('confirm', [...draft.value])
  emit('update:modelValue', false)
}
function onCancel() {
  emit('cancel')
  emit('update:modelValue', false)
}
function onDialogUpdate(v: boolean) {
  if (!v) onCancel()
}
</script>

<style scoped>
.picker-toolbar {
  background: rgba(var(--v-theme-on-surface), 0.03);
}
</style>
