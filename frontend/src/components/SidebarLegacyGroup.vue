<template>
  <div class="sidebar-legacy-group">
    <v-divider v-if="!rail" class="my-1" />
    <v-list-item
      v-if="!rail"
      class="legacy-header"
      density="compact"
      @click.stop="toggle"
      rounded="lg"
    >
      <template #prepend>
        <v-icon size="small" class="mr-1">
          {{ expanded ? 'mdi-menu-down' : 'mdi-menu-right' }}
        </v-icon>
        <v-icon size="small">mdi-history</v-icon>
      </template>
      <v-list-item-title class="text-caption text-uppercase text-medium-emphasis">
        Legacy tools
      </v-list-item-title>
    </v-list-item>

    <v-expand-transition>
      <div v-show="expanded || rail">
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
      </div>
    </v-expand-transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

defineProps<{
  rail?: boolean
}>()

const LS_KEY = 'cira.sidebar.legacyExpanded'
const expanded = ref(false)

function load() {
  try {
    const raw = window.localStorage.getItem(LS_KEY)
    expanded.value = raw === 'true'
  } catch {
    expanded.value = false
  }
}

function save() {
  try {
    window.localStorage.setItem(LS_KEY, expanded.value ? 'true' : 'false')
  } catch {
    /* localStorage full or blocked — ignore */
  }
}

function toggle() {
  expanded.value = !expanded.value
  save()
}

onMounted(load)
watch(expanded, save)
</script>

<style scoped>
.legacy-header {
  cursor: pointer;
  user-select: none;
}
.legacy-chip {
  color: rgb(var(--v-theme-on-surface-variant, 120, 120, 120)) !important;
  opacity: 0.75;
  font-size: 9px;
  height: 16px !important;
}
</style>
