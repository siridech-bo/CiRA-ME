<template>
  <v-dialog
    :model-value="modelValue"
    max-width="720"
    scrollable
    persistent
    @update:model-value="onDialogUpdate"
  >
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon start>{{ isEdit ? 'mdi-account-group' : 'mdi-plus-circle-outline' }}</v-icon>
        {{ isEdit ? 'Edit machine group' : 'Create machine group' }}
        <v-spacer />
        <v-btn icon size="small" variant="text" @click="onCancel">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-card-title>

      <v-divider />

      <v-card-text style="max-height: 70vh;">
        <v-text-field
          v-model="form.name"
          label="Name"
          density="compact"
          :error-messages="nameError"
          @blur="validateName"
        />
        <v-textarea
          v-model="form.description"
          label="Description (optional)"
          rows="2"
          density="compact"
        />

        <v-divider class="my-3" />

        <div class="d-flex align-center mb-2">
          <span class="text-subtitle-2">Machines</span>
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
          <div
            v-else-if="!assetTreeStore.tree || assetTreeStore.tree.length === 0"
            class="pa-4 text-center text-caption text-medium-emphasis"
          >
            No tree yet. Set one up in Settings → Asset Tree first.
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

        <v-divider class="my-3" />

        <div class="d-flex align-center flex-wrap ga-2 mb-2">
          <span class="text-body-2">
            <strong>{{ selectedIds.length }}</strong>
            {{ selectedIds.length === 1 ? 'machine' : 'machines' }} selected
          </span>
          <v-btn
            size="small"
            variant="tonal"
            prepend-icon="mdi-check-decagram-outline"
            :disabled="selectedIds.length < 2"
            @click="runCompatibility"
          >
            Validate compatibility
          </v-btn>
        </div>

        <CompatibilityBadge
          ref="compatBadgeRef"
          :machine-ids="selectedIds"
          :auto-run="false"
          @update:compatible="onCompatibleChange"
        />

        <v-checkbox
          v-if="lastValidatedIncompatible"
          v-model="saveAnyway"
          label="Save anyway (I understand training may fail on incompatible sensor sets)"
          density="compact"
          hide-details
          color="warning"
          class="mt-2"
        />
      </v-card-text>

      <v-divider />

      <v-card-actions class="px-4 py-3">
        <v-btn variant="text" @click="onCancel">Cancel</v-btn>
        <v-spacer />
        <v-btn
          color="primary"
          :disabled="!canSave"
          :loading="saving"
          @click="onSave"
        >
          {{ isEdit ? 'Save changes' : 'Create group' }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
/**
 * Phase C.1 — Machine Group create/edit modal.
 * Uses MachineTreePickerNode for the tree picker (only machines checkable),
 * and CompatibilityBadge for the inline validation UX.
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'
import MachineTreePickerNode from './MachineTreePickerNode.vue'
import CompatibilityBadge from './CompatibilityBadge.vue'

interface Group {
  id: number
  name: string
  description?: string | null
  members?: any[]
}

const props = defineProps<{
  modelValue: boolean
  group?: Group | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'saved', group: Group): void
}>()

const assetTreeStore = useAssetTreeStore()
const notify = useNotificationStore()

const isEdit = computed(() => !!props.group?.id)

const form = ref({ name: '', description: '' })
const nameError = ref<string | null>(null)

const query = ref('')
const hideRetired = ref(true)
const selectedIds = ref<number[]>([])
const selectedSet = computed(() => new Set(selectedIds.value))

const saving = ref(false)
const compatBadgeRef = ref<InstanceType<typeof CompatibilityBadge> | null>(null)
const lastCompatible = ref<boolean | null>(null)
const saveAnyway = ref(false)

// True only once the user has run validation and it came back incompatible.
const lastValidatedIncompatible = computed(() => lastCompatible.value === false)

// Reset when the dialog opens, populating from the existing group if edit-mode.
watch(
  () => props.modelValue,
  (open) => {
    if (!open) return
    nameError.value = null
    query.value = ''
    saveAnyway.value = false
    lastCompatible.value = null
    if (props.group) {
      form.value = {
        name: props.group.name || '',
        description: props.group.description || '',
      }
      // Filter retired members out of the hydrate — they'd otherwise be
      // invisibly re-submitted on Save, defeating the QA #1 backend guard.
      // Legacy groups created before that guard can now be cleaned up by
      // simply opening + saving them.
      const raw = (props.group.members || []) as any[]
      selectedIds.value = raw
        .filter(m => m && m.status !== 'retired')
        .map(m => m.id)
        .filter(n => Number.isFinite(n))
    } else {
      form.value = { name: '', description: '' }
      selectedIds.value = []
    }
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

function onToggleMachine(node: AssetNode) {
  const id = node.id
  const idx = selectedIds.value.indexOf(id)
  if (idx >= 0) selectedIds.value.splice(idx, 1)
  else selectedIds.value.push(id)
  // Invalidate the "saved-anyway" opt-in whenever membership changes.
  saveAnyway.value = false
  lastCompatible.value = null
}

function validateName() {
  const v = (form.value.name || '').trim()
  if (!v) {
    nameError.value = 'Name is required'
    return false
  }
  nameError.value = null
  return true
}

async function runCompatibility() {
  if (selectedIds.value.length < 2) return
  await compatBadgeRef.value?.runValidation()
}

function onCompatibleChange(v: boolean | null) {
  lastCompatible.value = v
}

const canSave = computed(() => {
  if (saving.value) return false
  if (!(form.value.name || '').trim()) return false
  if (selectedIds.value.length < 1) return false
  // If the user ran validation and it came back false, they must tick
  // "Save anyway" to proceed. If they never validated, save is allowed.
  if (lastCompatible.value === false && !saveAnyway.value) return false
  return true
})

async function onSave() {
  if (!validateName()) return
  saving.value = true
  try {
    const body = {
      name: form.value.name.trim(),
      description: form.value.description || null,
      machine_asset_ids: selectedIds.value,
    }
    let resp
    if (isEdit.value && props.group) {
      resp = await api.patch(`/api/asset-tree/groups/${props.group.id}`, body)
    } else {
      resp = await api.post('/api/asset-tree/groups', body)
    }
    notify.showSuccess(isEdit.value ? 'Group updated' : 'Group created')
    emit('saved', resp.data)
    emit('update:modelValue', false)
  } catch (e: any) {
    const msg = e.response?.data?.error || 'Save failed'
    if (/name/i.test(msg)) nameError.value = msg
    else notify.showError(msg)
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
  max-height: 320px;
  overflow-y: auto;
  padding: 4px;
  background: rgba(var(--v-theme-surface-bright), 0.4);
}
</style>
