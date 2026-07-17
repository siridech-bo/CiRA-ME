<template>
  <div class="ft-node">
    <div
      class="ft-row"
      :class="{ 'ft-row-selected': node.path === selectedPath }"
      :style="{ paddingLeft: `${depth * 12 + 8}px` }"
      @click="onClick"
    >
      <v-icon
        size="small"
        class="ft-chevron"
        :class="{ 'ft-chevron-hidden': !expandable }"
        @click.stop="toggle"
      >
        {{ expanded ? 'mdi-chevron-down' : 'mdi-chevron-right' }}
      </v-icon>
      <v-icon size="small" class="mr-1" color="amber-darken-2">
        {{ expanded ? 'mdi-folder-open' : 'mdi-folder' }}
      </v-icon>
      <span class="ft-label text-body-2">{{ node.name }}</span>
      <v-progress-circular
        v-if="loading"
        indeterminate
        size="12"
        width="2"
        class="ml-2"
      />
    </div>
    <div v-if="expanded && children.length > 0" class="ft-children">
      <FileTreeNode
        v-for="child in children"
        :key="child.path"
        :node="child"
        :depth="depth + 1"
        :selected-path="selectedPath"
        :folders-only="foldersOnly"
        @select="(p) => $emit('select', p)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import api from '@/services/api'

interface TreeNode {
  name: string
  path: string
  is_dir?: boolean
  type?: string
}

const props = withDefaults(defineProps<{
  node: TreeNode
  depth?: number
  selectedPath?: string
  foldersOnly?: boolean
}>(), {
  depth: 0,
  selectedPath: '',
  foldersOnly: true,
})

const emit = defineEmits<{
  (e: 'select', path: string): void
}>()

const expanded = ref(false)
const loaded = ref(false)
const loading = ref(false)
const children = ref<TreeNode[]>([])

// Root folders don't have is_dir; treat them as expandable.
// For loaded children, only folders are expandable.
const expandable = computed(() => {
  if (props.node.is_dir === false) return false
  return true
})

async function toggle() {
  if (!expandable.value) return
  if (expanded.value) {
    expanded.value = false
    return
  }
  expanded.value = true
  if (loaded.value) return
  await loadChildren()
}

async function loadChildren() {
  loading.value = true
  try {
    const resp = await api.post('/api/data/browse', { path: props.node.path })
    const items = (resp.data.items || []) as any[]
    children.value = items
      .filter(it => !props.foldersOnly || it.is_dir)
      .map(it => ({
        name: it.name,
        path: it.path,
        is_dir: it.is_dir,
      }))
    loaded.value = true
  } catch {
    children.value = []
  } finally {
    loading.value = false
  }
}

function onClick() {
  emit('select', props.node.path)
  // Also auto-expand on click for convenience if not yet expanded
  if (expandable.value && !expanded.value) {
    toggle()
  }
}
</script>

<style scoped lang="scss">
.ft-node {
  user-select: none;
}

.ft-row {
  display: flex;
  align-items: center;
  padding: 4px 8px 4px 0;
  cursor: pointer;
  border-radius: 4px;
  font-size: 0.875rem;

  &:hover {
    background: rgba(99, 102, 241, 0.08);
  }

  &.ft-row-selected {
    background: rgba(99, 102, 241, 0.18);
    font-weight: 500;
  }
}

.ft-chevron {
  margin-right: 4px;
  opacity: 0.7;

  &.ft-chevron-hidden {
    visibility: hidden;
  }
}

.ft-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
