<template>
  <!--
    Phase B — B.9. Compact banner mounted at the top of every legacy
    pipeline page. Shows either:
      - "No machine selected — pick one from the tree" (subtle warning), or
      - "Working on machine <topic_path>" (info, clickable to open workspace).
    Hidden entirely on unauthenticated / standalone views (parent decides).
  -->
  <v-alert
    v-if="!currentMachine"
    density="compact"
    variant="tonal"
    color="warning"
    class="mb-3"
    icon="mdi-file-tree-outline"
  >
    <div class="d-flex align-center flex-wrap ga-2">
      <span class="text-body-2">
        No machine selected — pick one from the tree to bind this pipeline
        run to an asset.
      </span>
      <v-spacer />
      <v-btn
        size="small"
        variant="text"
        color="warning"
        density="comfortable"
        :to="{ name: 'asset-tree-admin' }"
      >
        Open Asset Tree
      </v-btn>
    </div>
  </v-alert>

  <v-alert
    v-else
    density="compact"
    variant="tonal"
    color="primary"
    class="mb-3"
    icon="mdi-cog"
  >
    <div class="d-flex align-center flex-wrap ga-2">
      <span class="text-body-2">
        Working on
        <router-link
          :to="{ name: 'machine-workspace', params: { id: String(currentMachine.id) } }"
          class="font-weight-medium"
        >
          {{ currentMachine.display_name || currentMachine.name }}
        </router-link>
        <span class="text-caption text-medium-emphasis ml-2">
          {{ currentMachine.topic_path }}
        </span>
      </span>
      <v-spacer />
      <v-btn
        size="small"
        variant="text"
        color="primary"
        density="comfortable"
        prepend-icon="mdi-open-in-new"
        :to="{ name: 'machine-workspace', params: { id: String(currentMachine.id) } }"
      >
        Open workspace
      </v-btn>
    </div>
  </v-alert>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useAssetTreeStore } from '@/stores/assetTree'

const assetTreeStore = useAssetTreeStore()

// Ensure the tree is loaded so currentMachineNode resolves to a real node.
onMounted(async () => {
  if (!assetTreeStore.treeLoaded && !assetTreeStore.loadingTree) {
    await assetTreeStore.fetchTree()
  }
})

const currentMachine = computed(() => assetTreeStore.currentMachineNode)
</script>
