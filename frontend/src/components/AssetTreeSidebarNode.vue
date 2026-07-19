<template>
  <div class="atn-node">
    <div
      class="atn-row"
      :class="{
        'atn-machine': isMachine,
        'atn-active': isCurrentMachine,
        'atn-leaf': !hasChildren,
      }"
      :style="{ paddingLeft: `${8 + depth * 12}px` }"
      role="button"
      tabindex="0"
      @click="onRowClick"
      @keydown.enter.prevent="onRowClick"
      @keydown.space.prevent="onRowClick"
    >
      <!-- Chevron / spacer -->
      <span class="atn-chevron">
        <v-icon
          v-if="hasChildren"
          size="14"
          :style="{ transform: expanded ? 'rotate(90deg)' : 'none' }"
          @click.stop="toggle"
        >
          mdi-chevron-right
        </v-icon>
        <span v-else class="atn-chevron-spacer" />
      </span>

      <!-- Level icon -->
      <v-icon
        size="16"
        class="atn-icon"
        :color="iconColor"
      >{{ iconName }}</v-icon>

      <!-- Label -->
      <span class="atn-label" :title="fullLabelTitle">
        {{ node.display_name || node.name }}
      </span>

      <!-- Machine subtitle chip: sensor count on the machine. Kept small so
           it doesn't crowd narrow sidebars. -->
      <v-chip
        v-if="isMachine && sensorCount > 0 && !rail"
        size="x-small"
        variant="tonal"
        density="compact"
        class="atn-chip"
        :color="isCurrentMachine ? 'primary' : undefined"
      >
        {{ sensorCount }}
      </v-chip>
    </div>

    <!-- Children -->
    <div v-if="expanded && hasChildren">
      <AssetTreeSidebarNode
        v-for="child in node.children"
        :key="child.id"
        :node="child"
        :depth="depth + 1"
        :query="query"
        :machine-level="machineLevel"
        :current-machine-id="currentMachineId"
        :rail="rail"
        @select-machine="$emit('select-machine', $event)"
        @select-sensor="$emit('select-sensor', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'

const props = defineProps<{
  node: AssetNode
  depth: number
  query: string
  machineLevel: number
  currentMachineId: number | null
  rail?: boolean
}>()

const emit = defineEmits<{
  (e: 'select-machine', node: AssetNode): void
  (e: 'select-sensor', payload: { machine: AssetNode; sensor: AssetNode }): void
}>()

const assetTreeStore = useAssetTreeStore()

const isMachine = computed(() => props.node.level === props.machineLevel)
const isSensor = computed(() => props.node.level > props.machineLevel)
const isCurrentMachine = computed(() =>
  isMachine.value && props.currentMachineId === props.node.id,
)
const hasChildren = computed(() => (props.node.children?.length || 0) > 0)
const expanded = computed(() => assetTreeStore.isNodeExpanded(props.node.id))

const sensorCount = computed(() => props.node.children?.length || 0)

// Root=warehouse, plant=factory, machine=cog, sensor=leaf/tune.
const iconName = computed(() => {
  const l = props.node.level
  if (l === 0) return 'mdi-warehouse'
  if (l < props.machineLevel) return 'mdi-factory'
  if (l === props.machineLevel) return 'mdi-cog'
  return 'mdi-chip'
})
const iconColor = computed(() => {
  if (isCurrentMachine.value) return 'primary'
  if (isMachine.value) return 'secondary'
  return undefined
})

const fullLabelTitle = computed(() => {
  const dn = props.node.display_name
  return dn && dn !== props.node.name
    ? `${dn} (${props.node.topic_path})`
    : props.node.topic_path
})

function toggle() {
  assetTreeStore.toggleNodeExpanded(props.node.id)
}

// Clicking a machine navigates to that machine's workspace.
// Clicking a sensor navigates to its parent machine + preselects the
// History tab filtered to that sensor. Clicking a non-machine/non-sensor
// (root / plant / rack / etc.) expands/collapses so the tree stays
// browsable. Prevents the previous "click sensor → nothing happens"
// dead-end and preserves the spec's "click-to-filter parent machine"
// intent.
function onRowClick() {
  if (isMachine.value) {
    emit('select-machine', props.node)
    return
  }
  if (isSensor.value) {
    const machine = assetTreeStore.findMachineAncestor(props.node.id)
    if (machine) {
      emit('select-sensor', { machine, sensor: props.node })
    }
    return
  }
  if (hasChildren.value) toggle()
}
</script>

<style scoped>
.atn-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 6px;
  cursor: pointer;
  user-select: none;
  font-size: 0.85rem;
  line-height: 1.2;
  color: rgba(var(--v-theme-on-surface), 0.85);
}
.atn-row:hover {
  background: rgba(var(--v-theme-on-surface), 0.06);
}
.atn-row:focus-visible {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: -2px;
}
.atn-row.atn-machine {
  font-weight: 500;
}
.atn-row.atn-active {
  background: rgba(var(--v-theme-primary), 0.15);
  color: rgb(var(--v-theme-primary));
}
.atn-row.atn-active:hover {
  background: rgba(var(--v-theme-primary), 0.22);
}
.atn-chevron {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  color: rgba(var(--v-theme-on-surface), 0.55);
}
.atn-chevron-spacer {
  display: inline-block;
  width: 14px;
  height: 14px;
}
.atn-icon {
  flex-shrink: 0;
}
.atn-label {
  flex-grow: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.atn-chip {
  flex-shrink: 0;
  height: 18px;
  font-size: 0.7rem;
  padding: 0 6px;
}
</style>
