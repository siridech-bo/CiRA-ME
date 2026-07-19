<template>
  <div class="asset-tree-node" :class="{ 'is-root': depth === 0 }">
    <div
      class="asset-tree-row"
      :class="{
        'is-selected': node.id === selectedId,
        'is-retired': isRetired,
      }"
      :draggable="canDrag"
      @click.stop="$emit('select', node)"
      @dragstart="onDragStart"
      @dragover.prevent="onDragOver($event)"
      @dragleave="onDragLeave"
      @drop.prevent="onDrop($event)"
    >
      <v-icon
        v-if="hasChildren"
        size="16"
        class="expand-toggle"
        @click.stop="toggleExpanded"
      >
        {{ expanded ? 'mdi-chevron-down' : 'mdi-chevron-right' }}
      </v-icon>
      <span v-else class="expand-spacer" />

      <v-icon
        size="18"
        :color="isRetired ? 'grey' : levelColor"
        class="mr-2"
      >
        {{ levelIcon }}
      </v-icon>

      <span class="node-name" :title="node.topic_path">
        {{ node.display_name || node.name }}
        <span v-if="isRetired" class="text-caption text-medium-emphasis">
          (retired)
        </span>
      </span>

      <v-spacer />

      <v-btn
        v-if="canAddChild && !isRetired"
        icon="mdi-plus"
        size="x-small"
        variant="text"
        :title="`Add child under ${node.name}`"
        @click.stop="$emit('add-child', node)"
      />
      <v-btn
        v-if="canDelete && !isRetired"
        icon="mdi-delete-outline"
        size="x-small"
        variant="text"
        :title="`Delete ${node.name}`"
        @click.stop="$emit('delete-node', node)"
      />
    </div>

    <div v-if="hasChildren && expanded" class="children">
      <AssetTreeNodeEditor
        v-for="child in (node.children || [])"
        :key="child.id"
        :node="child"
        :depth="depth + 1"
        :max-depth="maxDepth"
        :selected-id="selectedId"
        :readonly="readonly"
        :level-names="levelNames"
        :level-icons="levelIcons"
        @select="$emit('select', $event)"
        @add-child="$emit('add-child', $event)"
        @delete-node="$emit('delete-node', $event)"
        @move-node="$emit('move-node', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Recursive tree row + children. Used inside both the wizard's Step 3 and
 * the ongoing admin view (A.6 + A.7). Emits UP; parent decides what to do
 * with add / delete / select / move events so this component stays pure.
 *
 * The drag-drop-move implementation is deliberately minimal — same-level
 * moves only. Cross-level moves aren't allowed anywhere in the product;
 * see backend/app/routes/asset_tree.py move_node handler.
 */
import { ref, computed } from 'vue'
import type { AssetNode } from '@/stores/assetTree'

const props = defineProps<{
  node: AssetNode
  depth: number
  maxDepth: number
  selectedId?: number | null
  readonly?: boolean
  levelNames?: string[]
  levelIcons?: string[]
}>()

const emit = defineEmits<{
  (e: 'select', node: AssetNode): void
  (e: 'add-child', node: AssetNode): void
  (e: 'delete-node', node: AssetNode): void
  (e: 'move-node', payload: { sourceId: number; targetParentId: number }): void
}>()

const expanded = ref(props.depth <= 1)

const hasChildren = computed(() => (props.node.children?.length ?? 0) > 0)
const isRetired = computed(() => props.node.status === 'retired')

const canAddChild = computed(() => {
  if (props.readonly) return false
  // Root of tree still has room for children? Depth check.
  return props.depth < props.maxDepth
})
const canDelete = computed(() => {
  if (props.readonly) return false
  // Never allow deleting the root — you rename it via config.
  return props.depth > 0
})
const canDrag = computed(() => !props.readonly && props.depth > 0)

const defaultIcons = [
  'mdi-earth', 'mdi-domain', 'mdi-factory', 'mdi-cog', 'mdi-chip', 'mdi-radar',
]
const levelIcon = computed(() => {
  const arr = props.levelIcons || defaultIcons
  return arr[props.depth] || 'mdi-circle-medium'
})

const defaultColors = ['primary', 'info', 'secondary', 'success', 'warning', 'error']
const levelColor = computed(() => defaultColors[props.depth] || 'primary')

function toggleExpanded() {
  expanded.value = !expanded.value
}

// ── Drag & drop (same-level moves only) ────────────────────────────────
function onDragStart(evt: DragEvent) {
  if (!canDrag.value) return
  evt.dataTransfer?.setData(
    'application/x-asset-node',
    JSON.stringify({ id: props.node.id, level: props.node.level, parent_id: props.node.parent_id }),
  )
  evt.dataTransfer!.effectAllowed = 'move'
}

function onDragOver(evt: DragEvent) {
  if (props.readonly) return
  // Accept a drop only if the dragged node's level matches ours — i.e. we
  // become the new parent of a same-level *child* — wait, no. Same-level
  // moves = same-parent-level, meaning the *new parent* has same level as
  // the old parent. So the target of the drop is the *new parent*, and it
  // must have the same level as the source's current parent.
  const raw = evt.dataTransfer?.types?.includes('application/x-asset-node')
  if (!raw) return
  ;(evt.currentTarget as HTMLElement).classList.add('is-drop-target')
  evt.dataTransfer!.dropEffect = 'move'
}

function onDragLeave(evt: DragEvent) {
  ;(evt.currentTarget as HTMLElement).classList.remove('is-drop-target')
}

function onDrop(evt: DragEvent) {
  ;(evt.currentTarget as HTMLElement).classList.remove('is-drop-target')
  const raw = evt.dataTransfer?.getData('application/x-asset-node')
  if (!raw) return
  try {
    const source = JSON.parse(raw) as { id: number; level: number; parent_id: number | null }
    if (source.id === props.node.id) return
    // Target parent = *this* node. The source's future parent has level =
    // this.level. Its old parent's level equals source.level - 1 = target.level?
    // For that to be a same-level move we need source.level - 1 === props.node.level.
    if (source.level - 1 !== props.node.level) {
      // Not a valid same-level move → silent no-op.
      return
    }
    emit('move-node', { sourceId: source.id, targetParentId: props.node.id })
  } catch {
    /* ignore malformed drag payload */
  }
}
</script>

<style scoped>
.asset-tree-node {
  font-size: 0.875rem;
}

.asset-tree-row {
  display: flex;
  align-items: center;
  padding: 4px 8px;
  border-radius: 6px;
  cursor: pointer;
  min-height: 32px;
  border: 1px solid transparent;
  transition: background-color 120ms ease;
}

.asset-tree-row:hover {
  background: rgba(var(--v-theme-on-surface), 0.06);
}

.asset-tree-row.is-selected {
  background: rgba(var(--v-theme-primary), 0.12);
  border-color: rgba(var(--v-theme-primary), 0.4);
}

.asset-tree-row.is-retired {
  opacity: 0.55;
  font-style: italic;
}

.asset-tree-row.is-drop-target {
  background: rgba(var(--v-theme-success), 0.14);
  border-color: rgba(var(--v-theme-success), 0.5);
}

.expand-toggle {
  cursor: pointer;
  margin-right: 2px;
}

.expand-spacer {
  display: inline-block;
  width: 16px;
  margin-right: 2px;
}

.node-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.children {
  margin-left: 18px;
  border-left: 1px dashed rgba(var(--v-theme-on-surface), 0.14);
  padding-left: 6px;
}

.is-root > .asset-tree-row {
  font-weight: 600;
}
</style>
