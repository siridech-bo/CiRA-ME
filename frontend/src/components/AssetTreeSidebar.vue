<template>
  <!--
    Phase B — Asset tree sidebar navigation.
    Renders the current tree from useAssetTreeStore, with:
      - a client-side search box (name + display_name),
      - persistent expand/collapse (via store),
      - machine-level nodes navigate to /machine/:id,
      - non-machine nodes expand/collapse on click.
    Retired nodes are excluded (fetchTree already filters).
  -->
  <div class="asset-tree-sidebar">
    <div class="asset-tree-header">
      <v-text-field
        v-if="!rail"
        v-model="query"
        density="compact"
        variant="outlined"
        hide-details
        placeholder="Search tree…"
        prepend-inner-icon="mdi-magnify"
        clearable
        class="asset-tree-search"
      />
      <div v-else class="asset-tree-rail-icon">
        <v-icon size="small">mdi-file-tree</v-icon>
      </div>
    </div>

    <div class="asset-tree-scroll">
      <div v-if="assetTreeStore.loadingTree" class="asset-tree-empty">
        <v-progress-circular indeterminate size="18" width="2" color="primary" />
        <span class="ml-2 text-caption text-medium-emphasis">Loading tree…</span>
      </div>
      <div
        v-else-if="!assetTreeStore.tree || assetTreeStore.tree.length === 0"
        class="asset-tree-empty text-caption text-medium-emphasis"
      >
        No tree yet.
        <router-link :to="{ name: 'asset-tree-admin' }" class="ml-1">Set up →</router-link>
      </div>
      <template v-else>
        <AssetTreeSidebarNode
          v-for="root in visibleTree"
          :key="root.id"
          :node="root"
          :depth="0"
          :query="query || ''"
          :machine-level="assetTreeStore.machineLevel"
          :current-machine-id="assetTreeStore.currentMachineId"
          :rail="rail"
          @select-machine="onSelectMachine"
          @select-sensor="onSelectSensor"
        />
        <div v-if="query && visibleTree.length === 0" class="asset-tree-empty text-caption text-medium-emphasis">
          No matches for “{{ query }}”
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'
import AssetTreeSidebarNode from './AssetTreeSidebarNode.vue'

// Rail mode — provided by parent so the collapsed drawer hides the search
// input and just shows a tree icon. Toggle handled at the App.vue level.
defineProps<{ rail?: boolean }>()

const router = useRouter()
const assetTreeStore = useAssetTreeStore()

const query = ref('')

// Fetch once on mount; the store caches. Parent should call
// assetTreeStore.invalidateTree() after admin mutations.
onMounted(async () => {
  if (!assetTreeStore.treeLoaded && !assetTreeStore.loadingTree) {
    await assetTreeStore.fetchTree()
  }
})

// Client-side filter: keep any node whose name/display_name matches (case-
// insensitive) OR any of whose descendants match. Ancestor visibility keeps
// the tree navigable while filtered.
function matchesQuery(node: AssetNode, q: string): boolean {
  if (!q) return true
  const hay = `${node.name} ${node.display_name || ''}`.toLowerCase()
  if (hay.includes(q)) return true
  if (node.children) {
    return node.children.some(c => matchesQuery(c, q))
  }
  return false
}

function pruneTree(nodes: AssetNode[], q: string): AssetNode[] {
  if (!q) return nodes
  const lower = q.trim().toLowerCase()
  if (!lower) return nodes
  const out: AssetNode[] = []
  for (const n of nodes) {
    const kids = n.children ? pruneTree(n.children, lower) : []
    const selfHit = `${n.name} ${n.display_name || ''}`.toLowerCase().includes(lower)
    if (selfHit || kids.length > 0) {
      out.push({ ...n, children: kids.length > 0 ? kids : n.children })
    }
  }
  return out
}

const visibleTree = computed(() => pruneTree(assetTreeStore.tree, query.value || ''))

// While a filter is active, force everything to appear expanded so matches
// are visible without extra clicks. We do this by temporarily adding all
// visible ids to the expanded set — but only in memory (no persistence).
watch(query, (q) => {
  if (!q) return
  for (const root of visibleTree.value) walkExpand(root)
})

function walkExpand(n: AssetNode) {
  if (!assetTreeStore.isNodeExpanded(n.id)) {
    assetTreeStore.setNodeExpanded(n.id, true)
  }
  if (n.children) n.children.forEach(walkExpand)
}

function onSelectMachine(node: AssetNode) {
  router.push({ name: 'machine-workspace', params: { id: String(node.id) } })
}

// Sensor clicks route to the machine workspace's History tab, preselected
// to the clicked sensor via the ?sensor= query. MachineHistoryTab reads
// this on mount and auto-selects the dropdown.
function onSelectSensor(payload: { machine: AssetNode; sensor: AssetNode }) {
  router.push({
    name: 'machine-workspace',
    params: { id: String(payload.machine.id) },
    query: { tab: 'history', sensor: payload.sensor.name },
  })
}
</script>

<style scoped>
.asset-tree-sidebar {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.asset-tree-header {
  padding: 4px 8px 6px;
}
.asset-tree-search :deep(.v-field) {
  font-size: 0.85rem;
}
.asset-tree-rail-icon {
  text-align: center;
  padding: 4px 0;
  color: rgba(var(--v-theme-on-surface), 0.6);
}
.asset-tree-scroll {
  max-height: 40vh;
  overflow-y: auto;
  padding: 2px 4px 6px;
}
.asset-tree-empty {
  padding: 8px 10px;
  display: flex;
  align-items: center;
}
</style>
