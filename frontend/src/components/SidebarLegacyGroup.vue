<template>
  <SidebarCollapsibleGroup
    ref="innerGroup"
    title="Legacy tools"
    storage-key="cira.sidebar.legacyExpanded"
    icon="mdi-history"
    :active-routes="['projects-list', 'dashboard']"
    :rail="rail"
  >
    <v-list-item
      prepend-icon="mdi-folder-multiple-outline"
      :to="{ name: 'projects-list' }"
      value="legacy-projects"
      rounded="lg"
      density="compact"
    >
      <v-list-item-title>
        <span>Projects</span>
        <v-chip
          v-if="!rail"
          size="x-small"
          variant="tonal"
          class="ml-2 legacy-chip"
        >
          legacy
        </v-chip>
      </v-list-item-title>
    </v-list-item>

    <v-list-item
      prepend-icon="mdi-view-dashboard"
      :to="{ name: 'dashboard' }"
      value="legacy-dashboard"
      rounded="lg"
      density="compact"
    >
      <v-list-item-title>
        <span>Dashboard</span>
        <v-chip
          v-if="!rail"
          size="x-small"
          variant="tonal"
          class="ml-2 legacy-chip"
        >
          legacy
        </v-chip>
      </v-list-item-title>
    </v-list-item>
  </SidebarCollapsibleGroup>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import SidebarCollapsibleGroup from './SidebarCollapsibleGroup.vue'

defineProps<{
  rail?: boolean
}>()

const innerGroup = ref<InstanceType<typeof SidebarCollapsibleGroup> | null>(null)

// Expose collapse() so the sidebar's "collapse all" button can reach in
// through this wrapper.
function collapse() {
  innerGroup.value?.collapse?.()
}
defineExpose({ collapse })
</script>

<style scoped>
.legacy-chip {
  color: rgb(var(--v-theme-on-surface-variant, 120, 120, 120)) !important;
  opacity: 0.75;
  font-size: 9px;
  height: 16px !important;
}
</style>
