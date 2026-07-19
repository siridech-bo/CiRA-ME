<template>
  <div class="mtpn-node">
    <div
      class="mtpn-row"
      :class="{ 'mtpn-machine': isMachine, 'mtpn-retired': isRetired }"
      :style="{ paddingLeft: `${4 + depth * 14}px` }"
    >
      <!-- Chevron / spacer -->
      <span class="mtpn-chevron">
        <v-icon
          v-if="hasChildren"
          size="14"
          :style="{ transform: expanded ? 'rotate(90deg)' : 'none' }"
          @click.stop="expanded = !expanded"
        >
          mdi-chevron-right
        </v-icon>
        <span v-else class="mtpn-chevron-spacer" />
      </span>

      <!-- Checkbox on machine rows only -->
      <v-checkbox-btn
        v-if="isMachine"
        :model-value="checked"
        density="compact"
        hide-details
        class="mtpn-check"
        :disabled="isRetired"
        @update:model-value="onToggle"
      />
      <span v-else class="mtpn-check-spacer" />

      <!-- Level icon -->
      <v-icon size="15" :color="iconColor" class="mtpn-icon">
        {{ iconName }}
      </v-icon>

      <!-- Label -->
      <span
        class="mtpn-label"
        :title="node.topic_path"
        :class="{ 'is-clickable': hasChildren }"
        @click="hasChildren && (expanded = !expanded)"
      >
        {{ node.display_name || node.name }}
      </span>

      <v-chip
        v-if="isRetired"
        size="x-small"
        variant="tonal"
        color="grey"
        class="ml-1"
      >
        retired
      </v-chip>
    </div>

    <div v-if="expanded && hasChildren">
      <MachineTreePickerNode
        v-for="child in visibleChildren"
        :key="child.id"
        :node="child"
        :depth="depth + 1"
        :machine-level="machineLevel"
        :selected-ids="selectedIds"
        :hide-retired="hideRetired"
        @toggle-machine="$emit('toggle-machine', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { AssetNode } from '@/stores/assetTree'

const props = defineProps<{
  node: AssetNode
  depth: number
  machineLevel: number
  selectedIds: Set<number>
  hideRetired: boolean
}>()

const emit = defineEmits<{
  (e: 'toggle-machine', node: AssetNode): void
}>()

// Auto-expand non-machine rows by default so pickers open with the tree
// visible without extra clicks.
const expanded = ref(props.depth < props.machineLevel)

const isMachine = computed(() => props.node.level === props.machineLevel)
const isRetired = computed(() => props.node.status === 'retired')
const hasChildren = computed(() => (props.node.children?.length || 0) > 0)
const checked = computed(() => props.selectedIds.has(props.node.id))

const visibleChildren = computed(() => {
  const kids = props.node.children || []
  if (!props.hideRetired) return kids
  return kids.filter(k => !(k.status === 'retired' && (!k.children || k.children.length === 0)))
})

const iconName = computed(() => {
  const l = props.node.level
  if (l === 0) return 'mdi-warehouse'
  if (l < props.machineLevel) return 'mdi-factory'
  if (l === props.machineLevel) return 'mdi-cog'
  return 'mdi-chip'
})
const iconColor = computed(() => {
  if (isRetired.value) return 'grey'
  if (isMachine.value) return 'primary'
  return undefined
})

function onToggle() {
  if (isRetired.value) return
  emit('toggle-machine', props.node)
}
</script>

<style scoped>
.mtpn-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 4px;
  border-radius: 4px;
  font-size: 0.86rem;
  line-height: 1.2;
  color: rgba(var(--v-theme-on-surface), 0.88);
}
.mtpn-row.mtpn-machine {
  font-weight: 500;
}
.mtpn-row.mtpn-retired {
  opacity: 0.55;
}
.mtpn-chevron,
.mtpn-check-spacer {
  display: inline-flex;
  width: 18px;
  height: 18px;
  align-items: center;
  justify-content: center;
  color: rgba(var(--v-theme-on-surface), 0.5);
  flex-shrink: 0;
}
.mtpn-chevron-spacer {
  display: inline-block;
  width: 14px;
  height: 14px;
}
.mtpn-check {
  padding: 0;
  margin: 0;
  min-height: 0;
  flex-shrink: 0;
}
.mtpn-icon {
  flex-shrink: 0;
}
.mtpn-label {
  flex-grow: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mtpn-label.is-clickable {
  cursor: pointer;
}
</style>
