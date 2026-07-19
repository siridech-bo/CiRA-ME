<template>
  <v-container fluid class="pa-6">
    <!-- Header: breadcrumb + retire -->
    <div class="d-flex align-center flex-wrap mb-4 ga-2">
      <v-icon color="primary" size="28" class="mr-2">mdi-cog</v-icon>
      <div class="flex-grow-1 min-width-0">
        <div class="text-h5 font-weight-bold text-truncate">
          {{ machine?.display_name || machine?.name || 'Machine' }}
        </div>
        <v-breadcrumbs
          v-if="breadcrumbs.length > 0"
          :items="breadcrumbs"
          density="compact"
          class="pa-0 breadcrumbs-row"
        >
          <template #divider>
            <v-icon size="small">mdi-chevron-right</v-icon>
          </template>
          <template #item="{ item }">
            <v-breadcrumbs-item :disabled="item.disabled">
              {{ item.title }}
            </v-breadcrumbs-item>
          </template>
        </v-breadcrumbs>
      </div>
      <v-spacer />
      <v-btn
        v-if="isAdmin && machine"
        variant="tonal"
        color="error"
        size="small"
        prepend-icon="mdi-archive-outline"
        @click="showRetireDialog = true"
      >
        Retire
      </v-btn>
    </div>

    <!-- Tabs -->
    <v-tabs v-model="activeTab" density="compact" class="mb-3">
      <v-tab value="overview" prepend-icon="mdi-view-dashboard-outline">Overview</v-tab>
      <v-tab value="data" prepend-icon="mdi-database">Data</v-tab>
      <v-tab value="models" prepend-icon="mdi-brain">Models</v-tab>
      <v-tab value="deploy" prepend-icon="mdi-rocket-launch">Deploy</v-tab>
      <v-tab value="labels" prepend-icon="mdi-tag-outline">Labels</v-tab>
      <v-tab value="history" prepend-icon="mdi-chart-line">History</v-tab>
    </v-tabs>

    <v-window v-model="activeTab">
      <v-window-item value="overview">
        <MachineOverviewTab
          v-if="machine"
          :machine="machine"
        />
      </v-window-item>
      <v-window-item value="data">
        <MachineDataTab v-if="machine" :machine="machine" />
      </v-window-item>
      <v-window-item value="models">
        <MachineModelsTab v-if="machine" :machine="machine" />
      </v-window-item>
      <v-window-item value="deploy">
        <MachineDeployTab v-if="machine" :machine="machine" />
      </v-window-item>
      <v-window-item value="labels">
        <MachineLabelsTab v-if="machine" :machine="machine" />
      </v-window-item>
      <v-window-item value="history">
        <MachineHistoryTab v-if="machine" :machine="machine" />
      </v-window-item>
    </v-window>

    <!-- Retire confirmation -->
    <v-dialog v-model="showRetireDialog" max-width="480">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="error" class="mr-2">mdi-archive-outline</v-icon>
          Retire this machine?
        </v-card-title>
        <v-card-text>
          <p class="mb-2">
            Retiring <code>{{ machine?.topic_path }}</code> also retires all
            its sensors. Historical data on disk is untouched.
          </p>
          <p class="text-body-2 text-medium-emphasis">
            After retiring, this workspace will redirect back to the tree.
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showRetireDialog = false">Cancel</v-btn>
          <v-btn color="error" :loading="retiring" @click="confirmRetire">
            Retire machine
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
/**
 * Phase B.2 — Machine workspace shell.
 * Tab state is mirrored to the URL (?tab=data) so refresh persists it.
 * The route guard already asserts machine-level; we defensively re-check
 * on mount to survive stale tree caches.
 */
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAssetTreeStore, type AssetNode } from '@/stores/assetTree'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'
import MachineOverviewTab from '@/components/machine/MachineOverviewTab.vue'
import MachineDataTab from '@/components/machine/MachineDataTab.vue'
import MachineModelsTab from '@/components/machine/MachineModelsTab.vue'
import MachineDeployTab from '@/components/machine/MachineDeployTab.vue'
import MachineLabelsTab from '@/components/machine/MachineLabelsTab.vue'
import MachineHistoryTab from '@/components/machine/MachineHistoryTab.vue'

const route = useRoute()
const router = useRouter()
const assetTreeStore = useAssetTreeStore()
const authStore = useAuthStore()
const notify = useNotificationStore()

const isAdmin = computed(() => authStore.user?.role === 'admin')

const VALID_TABS = ['overview', 'data', 'models', 'deploy', 'labels', 'history']
const activeTab = ref<string>(readTabFromRoute())

function readTabFromRoute(): string {
  const q = String(route.query.tab || 'overview')
  return VALID_TABS.includes(q) ? q : 'overview'
}

// Reflect tab into the URL so refresh restores it. Uses replace so we
// don't spam browser history with every tab click.
watch(activeTab, (v) => {
  if (String(route.query.tab || '') !== v) {
    router.replace({ query: { ...route.query, tab: v } })
  }
})
watch(
  () => route.query.tab,
  () => {
    const q = readTabFromRoute()
    if (q !== activeTab.value) activeTab.value = q
  },
)

const machineId = computed(() => {
  const n = Number(route.params.id)
  return Number.isFinite(n) ? n : null
})

const machine = computed<AssetNode | null>(() => {
  if (machineId.value == null) return null
  return assetTreeStore.findNode(machineId.value)
})

const breadcrumbs = computed(() => {
  const m = machine.value
  if (!m) return []
  // Walk up: gather ancestors by tracing topic_path segments against the
  // tree. Cheap enough for typical trees (< 100 nodes).
  const path: AssetNode[] = []
  const collect = (node: AssetNode | null) => {
    if (!node) return
    path.unshift(node)
    if (node.parent_id != null) {
      collect(assetTreeStore.findNode(node.parent_id))
    }
  }
  collect(m)
  return path.map((n, i) => ({
    title: n.display_name || n.name,
    disabled: i === path.length - 1,
  }))
})

const showRetireDialog = ref(false)
const retiring = ref(false)

async function confirmRetire() {
  if (!machine.value) return
  retiring.value = true
  try {
    await api.post(`/api/asset-tree/nodes/${machine.value.id}/retire`)
    notify.showSuccess('Machine retired')
    assetTreeStore.invalidateTree()
    await assetTreeStore.fetchTree(true)
    router.push({ name: 'asset-tree-admin' })
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Retire failed')
  } finally {
    retiring.value = false
    showRetireDialog.value = false
  }
}

onMounted(async () => {
  // Ensure the tree is loaded so `machine` resolves.
  if (!assetTreeStore.treeLoaded) {
    await assetTreeStore.fetchTree()
  }
})

// Track machine id from the URL. Vue Router reuses this same component
// instance when navigating between different `/machine/:id` params, so
// `onMounted` fires only once — a watcher is required to update the
// store's currentMachineId when the user clicks a different machine in
// the sidebar. Without this, the sidebar's active highlight and the
// legacy pipeline pages' banner both keep pointing at the first machine
// the user visited. Immediate so the initial mount also fires.
watch(
  machineId,
  async (id) => {
    if (id == null) return
    // Wait for the tree to be loaded if it hasn't yet — guard covers the
    // very-first-visit case.
    if (!assetTreeStore.treeLoaded) {
      await assetTreeStore.fetchTree()
    }
    const node = assetTreeStore.findNode(id)
    if (!node || !assetTreeStore.isMachineNode(node)) {
      notify.showError('Machine not found or not at machine level.')
      router.replace({ name: 'asset-tree-admin' })
      return
    }
    assetTreeStore.setCurrentMachineId(id)
  },
  { immediate: true },
)

// Leave the machine context intact when navigating to legacy pipeline
// pages — the banner reads it. We only clear it on explicit unmount when
// navigating to another /machine/:id (handled by the next mount's setCurrent).
onBeforeUnmount(() => {
  // Intentional no-op: currentMachineId persists across routes so the
  // legacy pipeline banner keeps context.
})
</script>

<style scoped>
.breadcrumbs-row {
  font-size: 0.8rem;
}
.min-width-0 {
  min-width: 0;
}
</style>
