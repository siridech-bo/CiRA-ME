<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center mb-4">
      <div>
        <h1 class="text-h4 font-weight-bold">Asset Tree</h1>
        <p class="text-body-2 text-medium-emphasis">
          Edit your physical-asset hierarchy. All write operations are
          audited.
        </p>
      </div>
      <v-spacer />
      <v-chip
        v-if="!isAdmin"
        color="warning"
        size="small"
        variant="tonal"
        prepend-icon="mdi-eye"
      >
        Read-only — admins can edit
      </v-chip>
    </div>

    <v-tabs v-model="activeTab" density="compact" class="mb-4">
      <v-tab value="tree" prepend-icon="mdi-file-tree">Tree</v-tab>
      <v-tab value="groups" prepend-icon="mdi-account-group">Machine Groups</v-tab>
      <v-tab value="audit" prepend-icon="mdi-clipboard-text-clock">Audit Log</v-tab>
    </v-tabs>

    <v-window v-model="activeTab">
      <!-- ── Tree tab ────────────────────────────────────────────────── -->
      <v-window-item value="tree">
        <div class="d-flex align-center mb-3 flex-wrap ga-2">
          <v-checkbox
            v-model="includeRetired"
            label="Show retired"
            density="compact"
            hide-details
            @update:model-value="fetchTree"
          />
          <v-spacer />
          <v-btn
            variant="text"
            size="small"
            prepend-icon="mdi-refresh"
            @click="fetchTree"
            :loading="loadingTree"
          >
            Refresh
          </v-btn>
        </div>

        <div class="tree-editor-grid">
          <div class="tree-editor-left">
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-2">Structure</span>
              <v-spacer />
              <v-btn
                v-if="!tree || tree.length === 0"
                v-show="isAdmin"
                size="x-small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-plus"
                @click="onAddRoot"
              >
                Add root
              </v-btn>
            </div>

            <div v-if="loadingTree" class="text-caption text-medium-emphasis pa-4 text-center">
              Loading…
            </div>
            <div v-else-if="!tree || tree.length === 0" class="empty-tree-block">
              <p class="text-body-2 text-medium-emphasis mb-3">
                No nodes yet. Add a root manually, or pick a template to
                get started in one click.
              </p>
              <v-btn
                v-if="isAdmin"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-file-tree-outline"
                size="small"
                @click="showTemplateDialog = true"
              >
                Browse templates
              </v-btn>
            </div>
            <div v-else>
              <AssetTreeNodeEditor
                v-for="root in tree"
                :key="root.id"
                :node="root"
                :depth="0"
                :max-depth="maxDepth"
                :selected-id="selectedNodeId"
                :readonly="!isAdmin"
                :level-names="config?.level_names"
                @select="onSelectNode"
                @add-child="onAddChild"
                @delete-node="promptRetire"
                @move-node="onMoveNode"
              />
            </div>
          </div>

          <div class="tree-editor-right">
            <div v-if="!selectedNode" class="empty-detail text-caption text-medium-emphasis">
              Select a node to view / edit its details.
            </div>
            <div v-else>
              <div class="d-flex align-center mb-3">
                <v-icon color="primary" class="mr-2">mdi-pencil</v-icon>
                <span class="text-subtitle-1 font-weight-bold">
                  {{ (config?.level_names || [])[selectedNode.level] || 'Node' }} details
                </span>
                <v-spacer />
                <v-chip
                  v-if="selectedNode.status === 'retired'"
                  size="x-small"
                  color="grey"
                  variant="tonal"
                >
                  retired
                </v-chip>
              </div>

              <v-text-field
                v-model="editForm.name"
                label="Name (topic segment)"
                :rules="[validNameRule]"
                :readonly="!isAdmin"
                hint="Renaming updates topic_path here and for all descendants"
                persistent-hint
                density="compact"
                class="mb-2"
              />
              <v-text-field
                v-model="editForm.display_name"
                label="Display name"
                :readonly="!isAdmin"
                density="compact"
                class="mb-2"
              />
              <v-textarea
                v-model="editForm.description"
                label="Description"
                :readonly="!isAdmin"
                rows="2"
                density="compact"
                class="mb-2"
              />
              <v-text-field
                v-model="editForm.location_tag"
                label="Location tag"
                :readonly="!isAdmin"
                density="compact"
                class="mb-3"
              />

              <v-card variant="tonal" class="pa-2 mb-3">
                <div class="text-caption text-medium-emphasis mb-1">
                  Topic path
                </div>
                <div class="d-flex align-center">
                  <code class="flex-grow-1">{{ selectedNode.topic_path }}</code>
                  <v-btn
                    icon="mdi-content-copy"
                    size="x-small"
                    variant="text"
                    title="Copy topic path"
                    @click="copyText(selectedNode.topic_path)"
                  />
                </div>
              </v-card>

              <template v-if="isSensorLevel(selectedNode) && isAdmin">
                <v-divider class="mb-3" />
                <div class="text-subtitle-2 mb-2">Sensor metadata</div>
                <SensorMetaEditor
                  v-model="editForm.sensor_meta"
                  :unit-presets="unitPresets"
                  :rate-presets="ratePresets"
                />
              </template>

              <div class="d-flex justify-space-between align-center mt-4 flex-wrap ga-2">
                <div>
                  <v-btn
                    v-if="isAdmin && canMove(selectedNode)"
                    variant="text"
                    size="small"
                    prepend-icon="mdi-swap-horizontal"
                    @click="showMoveDialog = true"
                  >
                    Move to…
                  </v-btn>
                  <v-btn
                    v-if="isAdmin && selectedNode.status !== 'retired' && selectedNode.parent_id !== null"
                    variant="text"
                    color="error"
                    size="small"
                    prepend-icon="mdi-archive-outline"
                    @click="promptRetire(selectedNode)"
                  >
                    Retire this node
                  </v-btn>
                </div>
                <v-btn
                  v-if="isAdmin"
                  color="primary"
                  size="small"
                  :loading="saving"
                  :disabled="!hasChanges"
                  @click="saveNode"
                >
                  Save changes
                </v-btn>
              </div>
            </div>
          </div>
        </div>
      </v-window-item>

      <!-- ── Groups tab (Phase C placeholder) ────────────────────────── -->
      <v-window-item value="groups">
        <v-card class="pa-6" variant="tonal">
          <div class="d-flex flex-column align-center text-center">
            <v-icon size="48" color="secondary" class="mb-2">mdi-account-group</v-icon>
            <h3 class="text-h6">Machine Groups</h3>
            <p class="text-body-2 text-medium-emphasis mt-2">
              Cross-machine training groups land in Phase C. The backend
              endpoints already exist — the UI ships next.
            </p>
          </div>
        </v-card>
      </v-window-item>

      <!-- ── Audit tab ────────────────────────────────────────────────── -->
      <v-window-item value="audit">
        <v-card>
          <v-data-table
            :headers="auditHeaders"
            :items="auditRows"
            :loading="loadingAudit"
            :items-per-page="auditLimit"
            item-value="id"
            density="comfortable"
            hide-default-footer
          >
            <template #item.event_type="{ item }">
              <v-chip size="small" variant="tonal" :color="eventTypeColor(item.event_type)">
                {{ item.event_type }}
              </v-chip>
            </template>
            <template #item.target_type="{ item }">
              <span class="text-caption">
                {{ item.target_type }} #{{ item.target_id ?? '—' }}
              </span>
            </template>
            <template #item.actor_user_id="{ item }">
              <span class="text-caption">user #{{ item.actor_user_id ?? '—' }}</span>
            </template>
            <template #item.payload="{ item }">
              <v-btn
                variant="text"
                size="x-small"
                prepend-icon="mdi-code-json"
                @click="showPayload(item)"
              >
                view
              </v-btn>
            </template>
            <template #item.created_at="{ item }">
              <span class="text-caption">{{ item.created_at }}</span>
            </template>
          </v-data-table>
          <div class="d-flex align-center pa-3 ga-2">
            <span class="text-caption text-medium-emphasis">
              {{ auditOffset + 1 }}–{{ auditOffset + auditRows.length }} of {{ auditTotal }}
            </span>
            <v-spacer />
            <v-btn size="small" variant="text" :disabled="auditOffset === 0" @click="pageAudit(-1)">
              <v-icon start>mdi-chevron-left</v-icon>Prev
            </v-btn>
            <v-btn
              size="small"
              variant="text"
              :disabled="auditOffset + auditRows.length >= auditTotal"
              @click="pageAudit(1)"
            >
              Next<v-icon end>mdi-chevron-right</v-icon>
            </v-btn>
          </div>
        </v-card>
      </v-window-item>
    </v-window>

    <!-- Rename warning modal -->
    <v-dialog v-model="showRenameWarning" max-width="480">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="warning" class="mr-2">mdi-alert-outline</v-icon>
          Rename node?
        </v-card-title>
        <v-card-text>
          <p class="mb-2">
            Renaming will not change existing MQTT subscriptions. Devices must
            be reconfigured to publish to the new topic. Existing data on the
            old path stays where it is.
          </p>
          <p class="text-body-2 text-medium-emphasis">
            Rename <code>{{ renameFrom }}</code> → <code>{{ renameTo }}</code>?
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="cancelRename">Cancel</v-btn>
          <v-btn color="warning" @click="confirmRename">Rename</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Retire confirmation modal -->
    <v-dialog v-model="showRetireDialog" max-width="480">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="error" class="mr-2">mdi-archive-outline</v-icon>
          Retire node?
        </v-card-title>
        <v-card-text>
          <p class="mb-2">
            Retiring <code>{{ retireTarget?.topic_path }}</code> will also retire
            all its descendants. Historical data is untouched.
          </p>
          <p class="text-body-2 text-medium-emphasis">
            Retired nodes stop accepting new MQTT publishes in strict mode.
            You can view them by ticking "Show retired".
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showRetireDialog = false">Cancel</v-btn>
          <v-btn color="error" :loading="retiring" @click="confirmRetire">Retire</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Move dialog -->
    <v-dialog v-model="showMoveDialog" max-width="480">
      <v-card>
        <v-card-title>Move to a new parent</v-card-title>
        <v-card-text>
          <p class="text-body-2 text-medium-emphasis mb-3">
            Same-level moves only. Node
            <code>{{ selectedNode?.name }}</code> must reparent to a peer of
            its current parent.
          </p>
          <v-select
            v-model="moveTargetId"
            :items="moveCandidates"
            item-title="topic_path"
            item-value="id"
            label="New parent"
            density="compact"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showMoveDialog = false">Cancel</v-btn>
          <v-btn
            color="primary"
            :disabled="!moveTargetId"
            :loading="moving"
            @click="confirmMove"
          >
            Move
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Add child dialog -->
    <v-dialog v-model="showAddDialog" max-width="480">
      <v-card>
        <v-card-title>
          Add {{ addingIsRoot ? 'root node' : childLevelLabel }}
        </v-card-title>
        <v-card-text>
          <v-text-field
            v-model="addForm.name"
            label="Name"
            :rules="[validNameRule]"
            density="compact"
            class="mb-2"
          />
          <v-text-field
            v-model="addForm.display_name"
            label="Display name (optional)"
            density="compact"
            class="mb-2"
          />
          <v-textarea
            v-model="addForm.description"
            label="Description (optional)"
            rows="2"
            density="compact"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showAddDialog = false">Cancel</v-btn>
          <v-btn color="primary" :loading="adding" @click="confirmAdd">Add</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Tree template picker — empty-state shortcut for admins so they can
         drop a working tree in one click instead of building from scratch. -->
    <v-dialog v-model="showTemplateDialog" max-width="900" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon start>mdi-file-tree-outline</v-icon>
          Start from a template
          <v-spacer />
          <v-btn icon size="small" variant="text" @click="showTemplateDialog = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>
        <v-card-text>
          <TreeTemplatePicker
            allow-cancel
            @applied="onTemplateApplied"
            @cancel="showTemplateDialog = false"
          />
        </v-card-text>
      </v-card>
    </v-dialog>

    <!-- Audit payload viewer -->
    <v-dialog v-model="showPayloadDialog" max-width="720">
      <v-card>
        <v-card-title>Audit payload</v-card-title>
        <v-card-text>
          <pre class="payload-block">{{ payloadPretty }}</pre>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showPayloadDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
/**
 * Ongoing admin view — Phase A.7.
 * Reuses AssetTreeNodeEditor + SensorMetaEditor from the wizard so the UX
 * is symmetrical. Non-admins see the same layout with all edit buttons
 * hidden; the read paths (GET /nodes, /audit) are open to all users.
 */
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import { useAssetTreeStore, type AssetNode, type SensorMeta } from '@/stores/assetTree'
import api from '@/services/api'
import AssetTreeNodeEditor from '@/components/AssetTreeNodeEditor.vue'
import SensorMetaEditor from '@/components/SensorMetaEditor.vue'
import TreeTemplatePicker from '@/components/TreeTemplatePicker.vue'

const NAME_REGEX = /^[A-Za-z0-9_-]+$/

const props = defineProps<{
  defaultTab?: 'tree' | 'groups' | 'audit'
}>()

const route = useRoute()
const authStore = useAuthStore()
const notificationStore = useNotificationStore()
const assetTreeStore = useAssetTreeStore()

const isAdmin = computed(() => authStore.user?.role === 'admin')

const activeTab = ref(props.defaultTab || 'tree')
watch(
  () => route.fullPath,
  () => {
    if (props.defaultTab) activeTab.value = props.defaultTab
  },
)

// ── Tree state ───────────────────────────────────────────────────────────

const tree = ref<AssetNode[]>([])
const loadingTree = ref(false)
const includeRetired = ref(false)
const selectedNodeId = ref<number | null>(null)
const config = computed(() => assetTreeStore.config)
const maxDepth = computed(() => Math.max(0, (config.value?.level_names?.length || 4) - 1))
const unitPresets = ref<Array<{ value: string; label: string }>>([])
const ratePresets = ref<number[]>([])

const editForm = ref<{
  name: string
  display_name: string
  description: string
  location_tag: string
  sensor_meta: SensorMeta
}>({
  name: '',
  display_name: '',
  description: '',
  location_tag: '',
  sensor_meta: {},
})

function findInTree(nodes: AssetNode[], id: number): AssetNode | null {
  for (const n of nodes) {
    if (n.id === id) return n
    if (n.children) {
      const hit = findInTree(n.children, id)
      if (hit) return hit
    }
  }
  return null
}

function walkTree(nodes: AssetNode[], fn: (n: AssetNode) => void) {
  for (const n of nodes) {
    fn(n)
    if (n.children) walkTree(n.children, fn)
  }
}

const selectedNode = computed(() =>
  selectedNodeId.value != null ? findInTree(tree.value, selectedNodeId.value) : null,
)

const hasChanges = computed(() => {
  if (!selectedNode.value) return false
  const sn = selectedNode.value
  if (editForm.value.name !== sn.name) return true
  if ((editForm.value.display_name || '') !== (sn.display_name || '')) return true
  if ((editForm.value.description || '') !== (sn.description || '')) return true
  if ((editForm.value.location_tag || '') !== (sn.location_tag || '')) return true
  if (isSensorLevel(sn)) {
    const a = editForm.value.sensor_meta || {}
    const b = sn.sensor_meta || {}
    if ((a.unit ?? null) !== (b.unit ?? null)) return true
    if ((a.sample_rate_hz ?? null) !== (b.sample_rate_hz ?? null)) return true
    if ((a.data_type ?? null) !== (b.data_type ?? null)) return true
    if ((a.expected_min ?? null) !== (b.expected_min ?? null)) return true
    if ((a.expected_max ?? null) !== (b.expected_max ?? null)) return true
  }
  return false
})

async function fetchTree() {
  loadingTree.value = true
  try {
    const r = await api.get('/api/asset-tree/nodes', {
      params: { include_retired: includeRetired.value ? 'true' : undefined },
    })
    tree.value = r.data?.tree || []
    // Re-select if the selected node still exists
    if (selectedNodeId.value != null) {
      const still = findInTree(tree.value, selectedNodeId.value)
      if (!still) selectedNodeId.value = null
      else loadFormFromNode(still)
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to load tree')
  } finally {
    loadingTree.value = false
  }
}

async function loadPresets() {
  try {
    const p = await assetTreeStore.loadPresets()
    unitPresets.value = p.unit_presets
    ratePresets.value = p.sample_rate_presets
  } catch { /* ignore, defaults are fine */ }
}

onMounted(async () => {
  await assetTreeStore.ensureConfigChecked()
  await Promise.all([fetchTree(), loadPresets(), refreshAudit()])
})

function isSensorLevel(n: AssetNode | null | undefined): boolean {
  if (!n) return false
  return n.level === maxDepth.value
}

function loadFormFromNode(n: AssetNode) {
  editForm.value = {
    name: n.name,
    display_name: n.display_name || '',
    description: n.description || '',
    location_tag: n.location_tag || '',
    sensor_meta: { ...(n.sensor_meta || {}) },
  }
}

function onSelectNode(n: AssetNode) {
  selectedNodeId.value = n.id
  loadFormFromNode(n)
}

function validNameRule(v: string) {
  if (!v) return 'Required'
  if (!NAME_REGEX.test(v)) return 'Letters, digits, _ or - only'
  if (v.length > 64) return 'Max 64 characters'
  return true
}

async function copyText(t: string) {
  try {
    await navigator.clipboard.writeText(t)
    notificationStore.showInfo('Topic copied')
  } catch {
    notificationStore.showError('Copy failed')
  }
}

// ── Save ─────────────────────────────────────────────────────────────────

const saving = ref(false)
const showRenameWarning = ref(false)
const renameFrom = ref('')
const renameTo = ref('')

async function saveNode() {
  if (!selectedNode.value) return
  if (editForm.value.name !== selectedNode.value.name) {
    renameFrom.value = selectedNode.value.name
    renameTo.value = editForm.value.name
    showRenameWarning.value = true
    return
  }
  await doSaveNode()
}

async function doSaveNode() {
  if (!selectedNode.value) return
  const body: any = {
    name: editForm.value.name,
    display_name: editForm.value.display_name || null,
    description: editForm.value.description || null,
    location_tag: editForm.value.location_tag || null,
  }
  if (isSensorLevel(selectedNode.value)) {
    body.sensor_meta = editForm.value.sensor_meta
  }
  saving.value = true
  try {
    await api.patch(`/api/asset-tree/nodes/${selectedNode.value.id}`, body)
    notificationStore.showSuccess('Saved')
    await fetchTree()
    await refreshAudit()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Save failed')
  } finally {
    saving.value = false
  }
}

function cancelRename() {
  showRenameWarning.value = false
  // Revert edit form
  if (selectedNode.value) loadFormFromNode(selectedNode.value)
}
async function confirmRename() {
  showRenameWarning.value = false
  await doSaveNode()
}

// ── Retire ───────────────────────────────────────────────────────────────

const showRetireDialog = ref(false)
const retireTarget = ref<AssetNode | null>(null)
const retiring = ref(false)

function promptRetire(node: AssetNode) {
  retireTarget.value = node
  showRetireDialog.value = true
}
async function confirmRetire() {
  if (!retireTarget.value) return
  retiring.value = true
  try {
    await api.post(`/api/asset-tree/nodes/${retireTarget.value.id}/retire`)
    notificationStore.showSuccess('Node retired')
    showRetireDialog.value = false
    retireTarget.value = null
    await fetchTree()
    await refreshAudit()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Retire failed')
  } finally {
    retiring.value = false
  }
}

// ── Tree template shortcut (empty state) ─────────────────────────────────

const showTemplateDialog = ref(false)

async function onTemplateApplied(payload: { template_id: string; count: number }) {
  notificationStore.showSuccess(
    `Template applied — ${payload.count} node${payload.count === 1 ? '' : 's'} created`,
  )
  showTemplateDialog.value = false
  // Store is stale — force a re-fetch of config + tree so the whole page
  // rebuilds against the freshly-populated tree.
  assetTreeStore.reset()
  await assetTreeStore.ensureConfigChecked()
  await Promise.all([fetchTree(), refreshAudit()])
}

// ── Move ─────────────────────────────────────────────────────────────────

const showMoveDialog = ref(false)
const moveTargetId = ref<number | null>(null)
const moving = ref(false)

function canMove(n: AssetNode | null | undefined): boolean {
  if (!n) return false
  return n.parent_id !== null
}

const moveCandidates = computed(() => {
  if (!selectedNode.value) return []
  const currentParentId = selectedNode.value.parent_id
  // We need siblings-of-parent: same-level nodes as the current parent, except the current parent itself.
  const currentParent = currentParentId != null ? findInTree(tree.value, currentParentId) : null
  if (!currentParent) return []
  const out: Array<{ id: number; topic_path: string }> = []
  walkTree(tree.value, (n) => {
    if (n.level === currentParent.level && n.id !== currentParent.id && n.status !== 'retired') {
      out.push({ id: n.id, topic_path: n.topic_path })
    }
  })
  return out
})

async function confirmMove() {
  if (!selectedNode.value || moveTargetId.value == null) return
  moving.value = true
  try {
    await api.post(`/api/asset-tree/nodes/${selectedNode.value.id}/move`, {
      new_parent_id: moveTargetId.value,
    })
    notificationStore.showSuccess('Node moved')
    showMoveDialog.value = false
    moveTargetId.value = null
    await fetchTree()
    await refreshAudit()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Move failed')
  } finally {
    moving.value = false
  }
}

async function onMoveNode(payload: { sourceId: number; targetParentId: number }) {
  try {
    await api.post(`/api/asset-tree/nodes/${payload.sourceId}/move`, {
      new_parent_id: payload.targetParentId,
    })
    notificationStore.showSuccess('Node moved')
    await fetchTree()
    await refreshAudit()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Move failed')
  }
}

// ── Add child ────────────────────────────────────────────────────────────

const showAddDialog = ref(false)
const addForm = ref({ name: '', display_name: '', description: '' })
const addingIsRoot = ref(false)
const addingParent = ref<AssetNode | null>(null)
const adding = ref(false)

const childLevelLabel = computed(() => {
  const names = config.value?.level_names || []
  if (addingParent.value) return names[addingParent.value.level + 1] || 'child'
  return 'root'
})

function onAddRoot() {
  addingIsRoot.value = true
  addingParent.value = null
  addForm.value = {
    name: config.value?.root_name || (config.value?.level_names || [])[0] || '',
    display_name: '',
    description: '',
  }
  showAddDialog.value = true
}
function onAddChild(parent: AssetNode) {
  addingIsRoot.value = false
  addingParent.value = parent
  addForm.value = { name: '', display_name: '', description: '' }
  showAddDialog.value = true
}
async function confirmAdd() {
  const rule = validNameRule(addForm.value.name)
  if (rule !== true) {
    notificationStore.showError(rule)
    return
  }
  adding.value = true
  try {
    const body: any = {
      parent_id: addingIsRoot.value ? null : addingParent.value?.id,
      name: addForm.value.name,
      display_name: addForm.value.display_name || null,
      description: addForm.value.description || null,
    }
    await api.post('/api/asset-tree/nodes', body)
    notificationStore.showSuccess('Node added')
    showAddDialog.value = false
    await fetchTree()
    await refreshAudit()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Add failed')
  } finally {
    adding.value = false
  }
}

// ── Audit tab ────────────────────────────────────────────────────────────

const auditRows = ref<any[]>([])
const auditTotal = ref(0)
const auditLimit = ref(20)
const auditOffset = ref(0)
const loadingAudit = ref(false)
const auditHeaders = [
  { title: 'When', value: 'created_at', width: 180 },
  { title: 'Event', value: 'event_type', width: 140 },
  { title: 'Target', value: 'target_type', width: 140 },
  { title: 'Actor', value: 'actor_user_id', width: 100 },
  { title: 'Payload', value: 'payload', width: 100 },
]

const showPayloadDialog = ref(false)
const payloadPretty = ref('')

function showPayload(item: any) {
  try {
    const parsed = typeof item.payload === 'string' ? JSON.parse(item.payload) : item.payload
    payloadPretty.value = JSON.stringify(parsed, null, 2)
  } catch {
    payloadPretty.value = String(item.payload)
  }
  showPayloadDialog.value = true
}

async function refreshAudit() {
  loadingAudit.value = true
  try {
    const r = await api.get('/api/asset-tree/audit', {
      params: { limit: auditLimit.value, offset: auditOffset.value },
    })
    auditRows.value = r.data?.audit || []
    auditTotal.value = r.data?.total || 0
  } catch (e: any) {
    // Don't spam errors on tab switch; leave rows as-is.
    if (activeTab.value === 'audit') {
      notificationStore.showError(e.response?.data?.error || 'Failed to load audit')
    }
  } finally {
    loadingAudit.value = false
  }
}

async function pageAudit(dir: number) {
  const next = auditOffset.value + dir * auditLimit.value
  if (next < 0) return
  auditOffset.value = next
  await refreshAudit()
}

function eventTypeColor(t: string) {
  if (t.includes('create')) return 'success'
  if (t.includes('retire') || t.includes('delete')) return 'error'
  if (t.includes('move') || t.includes('patch')) return 'warning'
  return 'primary'
}
</script>

<style scoped>
.tree-editor-grid {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.4fr);
  gap: 16px;
  align-items: start;
}
.tree-editor-left,
.tree-editor-right {
  background: rgba(var(--v-theme-surface-bright), 0.5);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 8px;
  padding: 12px;
  min-height: 380px;
}
.empty-tree,
.empty-detail {
  padding: 24px 8px;
  text-align: center;
}
.payload-block {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  background: rgba(var(--v-theme-surface-bright), 0.6);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  padding: 12px;
  border-radius: 6px;
  max-height: 400px;
  overflow: auto;
  font-size: 0.8rem;
  white-space: pre-wrap;
  word-break: break-word;
}
@media (max-width: 700px) {
  .tree-editor-grid {
    grid-template-columns: 1fr;
  }
}
</style>
