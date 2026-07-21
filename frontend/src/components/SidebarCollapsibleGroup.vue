<template>
  <div class="sidebar-collapsible-group">
    <v-divider v-if="!rail" class="my-1" />
    <v-list-item
      v-if="!rail"
      class="group-header"
      density="compact"
      @click.stop="toggle"
      rounded="lg"
    >
      <template #prepend>
        <v-icon size="small" class="mr-1">
          {{ expanded ? 'mdi-menu-down' : 'mdi-menu-right' }}
        </v-icon>
        <v-icon v-if="icon" size="small">{{ icon }}</v-icon>
      </template>
      <v-list-item-title class="text-caption text-uppercase text-medium-emphasis">
        {{ title }}
      </v-list-item-title>
    </v-list-item>

    <v-expand-transition>
      <div v-show="expanded || rail">
        <slot />
      </div>
    </v-expand-transition>
  </div>
</template>

<script setup lang="ts">
/**
 * Generic collapsible sidebar group.
 *
 * State model:
 *   - Fresh install (no localStorage entry): expanded iff the current
 *     route is in `activeRoutes` (auto-mode). Route change re-evaluates.
 *   - After the user clicks the header once: their choice sticks
 *     forever, ignoring the route. Persisted in `storageKey`.
 *
 * This keeps the sidebar compact by default while auto-opening the
 * section you're currently working in — until you tell it otherwise.
 */
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'

const props = defineProps<{
  title: string
  storageKey: string
  activeRoutes?: string[]
  icon?: string
  rail?: boolean
}>()

const route = useRoute()

// null = user has never toggled; true/false = user's sticky choice.
const userOverride = ref<boolean | null>(null)

const isActiveRoute = computed(() =>
  (props.activeRoutes ?? []).includes(String(route.name ?? ''))
)

const expanded = computed(() =>
  userOverride.value !== null ? userOverride.value : isActiveRoute.value,
)

function toggle() {
  userOverride.value = !expanded.value
  try {
    window.localStorage.setItem(
      props.storageKey,
      userOverride.value ? 'true' : 'false',
    )
  } catch {
    /* localStorage blocked — degrade silently, state lives in memory only */
  }
}

// Public API: force this group closed. Used by the sidebar-wide
// "collapse all" button so one click collapses everything regardless
// of the auto-open heuristic.
function collapse() {
  userOverride.value = false
  try {
    window.localStorage.setItem(props.storageKey, 'false')
  } catch { /* ignore */ }
}
defineExpose({ collapse })

onMounted(() => {
  try {
    const raw = window.localStorage.getItem(props.storageKey)
    if (raw === 'true') userOverride.value = true
    else if (raw === 'false') userOverride.value = false
    // else stays null → auto-mode based on active route
  } catch {
    /* localStorage blocked — stay in auto-mode */
  }
})
</script>

<style scoped>
.group-header {
  cursor: pointer;
  user-select: none;
}
</style>
