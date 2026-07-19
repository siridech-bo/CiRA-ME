<template>
  <v-app>
    <!-- App Bar (hidden on standalone pages and the fullscreen setup wizard) -->
    <v-app-bar
      v-if="authStore.isAuthenticated && !isStandalone && !isFullscreen"
      elevation="0"
      color="surface"
      border="b"
    >
      <template #prepend>
        <v-app-bar-nav-icon @click="drawer = !drawer" />
      </template>

      <LogoFull compact />

      <v-spacer />

      <!-- Mode Toggle -->
      <v-btn-toggle
        v-model="pipelineStore.mode"
        class="mode-toggle mx-4"
        mandatory
        rounded="lg"
        density="comfortable"
      >
        <v-btn value="anomaly" size="small">
          <v-icon start>mdi-chart-bell-curve</v-icon>
          Anomaly
        </v-btn>
        <v-btn value="classification" size="small">
          <v-icon start>mdi-shape</v-icon>
          Classification
        </v-btn>
        <v-btn value="regression" size="small">
          <v-icon start>mdi-chart-timeline-variant</v-icon>
          Regression
        </v-btn>
      </v-btn-toggle>

      <v-spacer />

      <!-- Legacy projects chip — surfaces pre-tree projects so they're
           findable. Non-blocking, migration wizard lands in Phase B. -->
      <v-menu v-if="legacyCount > 0" location="bottom end" :close-on-content-click="false">
        <template #activator="{ props }">
          <v-btn
            v-bind="props"
            size="small"
            variant="tonal"
            color="warning"
            class="mr-2 legacy-chip"
            prepend-icon="mdi-archive-outline"
          >
            {{ legacyCount }} legacy project{{ legacyCount === 1 ? '' : 's' }}
          </v-btn>
        </template>
        <v-card min-width="320" max-width="380">
          <v-card-title class="d-flex align-center">
            <v-icon color="warning" class="mr-2">mdi-archive-outline</v-icon>
            Legacy projects
          </v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-2">
              These predate the asset tree. Open them via the classic Projects
              view (deprecated). A migration wizard will land in Phase B.
            </p>
            <v-list density="compact" nav>
              <v-list-item
                v-for="p in assetTreeStore.legacyProjects.slice(0, 10)"
                :key="p.id"
                :to="{ name: 'projects-detail', params: { id: p.id } }"
                prepend-icon="mdi-folder-outline"
              >
                {{ p.name }}
              </v-list-item>
              <v-list-item
                v-if="assetTreeStore.legacyProjects.length > 10"
                prepend-icon="mdi-dots-horizontal"
                :subtitle="`${assetTreeStore.legacyProjects.length - 10} more`"
                :to="{ name: 'projects-list' }"
              >
                See all
              </v-list-item>
            </v-list>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn variant="text" :to="{ name: 'projects-list' }" size="small">
              Open Projects
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-menu>

      <!-- Theme toggle — prominent icon so users don't have to hunt through
           the user menu. Same handler as the menu item below so both stay
           in sync via the useThemePref composable (Phase 0). -->
      <v-btn
        icon
        variant="text"
        density="comfortable"
        :aria-label="themePref.isDark.value ? 'Switch to light mode' : 'Switch to dark mode'"
        :title="themePref.isDark.value ? 'Switch to light mode' : 'Switch to dark mode'"
        @click="themePref.toggle()"
      >
        <v-icon>{{ themePref.isDark.value ? 'mdi-weather-sunny' : 'mdi-weather-night' }}</v-icon>
      </v-btn>

      <!-- User Menu -->
      <v-menu>
        <template #activator="{ props }">
          <v-btn v-bind="props" variant="text" class="text-none">
            <v-avatar size="32" color="primary">
              <span class="text-caption">{{ userInitials }}</span>
            </v-avatar>
            <span class="ml-2 d-none d-sm-inline">{{ authStore.user?.display_name }}</span>
            <v-icon end>mdi-chevron-down</v-icon>
          </v-btn>
        </template>
        <v-list density="compact" nav>
          <v-list-item prepend-icon="mdi-account" :subtitle="authStore.user?.role">
            {{ authStore.user?.username }}
          </v-list-item>
          <v-divider />
          <v-list-item prepend-icon="mdi-key" @click="showChangePassword = true">
            Change Password
          </v-list-item>
          <v-list-item
            :prepend-icon="themePref.isDark.value ? 'mdi-weather-sunny' : 'mdi-weather-night'"
            @click="themePref.toggle()"
          >
            {{ themePref.isDark.value ? 'Switch to Light Mode' : 'Switch to Dark Mode' }}
          </v-list-item>
          <v-list-item prepend-icon="mdi-logout" @click="logout" class="text-error">
            Logout
          </v-list-item>
        </v-list>
      </v-menu>
    </v-app-bar>

    <!-- Navigation Drawer -->
    <v-navigation-drawer
      v-if="authStore.isAuthenticated && !isStandalone && !isFullscreen"
      v-model="drawer"
      :rail="rail"
      permanent
      @click="rail = false"
    >
      <v-list density="compact" nav>
        <!--
          Phase B — Asset tree is the primary navigation. Header dropped
          in Phase F because the tree itself is self-labelling and the
          section header was visual noise.
        -->
        <AssetTreeSidebar :rail="rail" />

        <v-divider v-if="!rail" class="my-1" />
        <v-list-subheader v-if="!rail">GLOBAL TOOLS</v-list-subheader>

        <v-list-item
          prepend-icon="mdi-database"
          title="Data Source"
          value="data"
          :active="route.name === 'pipeline-data'"
          rounded="lg"
          @click="requestNavigateToDataSource"
        />

        <v-list-item
          prepend-icon="mdi-gauge"
          title="Machine Simulators"
          value="simulators"
          :to="{ name: 'simulators' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-tune-vertical"
          title="Windowing"
          value="windowing"
          :to="{ name: 'pipeline-windowing' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-auto-fix"
          title="Features"
          value="features"
          :to="{ name: 'pipeline-features' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-brain"
          title="Training"
          value="training"
          :to="{ name: 'pipeline-training' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-rocket-launch"
          title="Deploy"
          value="deploy"
          :to="{ name: 'pipeline-deploy' }"
          rounded="lg"
        />

        <v-divider v-if="!rail" class="my-1" />
        <v-list-subheader v-if="!rail">SERVICES</v-list-subheader>

        <v-list-item
          prepend-icon="mdi-api"
          title="ME-LAB"
          value="melab"
          :to="{ name: 'melab' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-lan"
          title="MQTT Broker"
          value="mqtt-management"
          :to="{ name: 'mqtt-management' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-view-dashboard-variant"
          title="App Builder"
          value="app-builder"
          :to="{ name: 'app-builder' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-folder-eye"
          title="Folder Watcher"
          value="folder-watcher"
          :to="{ name: 'folder-watcher-list' }"
          rounded="lg"
        />

        <v-list-item
          prepend-icon="mdi-table-search"
          title="Multi-Dataset Wizard"
          value="wizard"
          :to="{ name: 'wizard' }"
          rounded="lg"
        />
      </v-list>

      <template #append>
        <v-list density="compact" nav>
          <v-divider class="mb-2" />

          <v-list-subheader v-if="!rail">SETTINGS</v-list-subheader>

          <v-list-item
            prepend-icon="mdi-file-tree"
            :title="rootSetupLabel"
            value="asset-tree"
            :to="{ name: 'asset-tree-admin' }"
            rounded="lg"
          />

          <v-list-item
            prepend-icon="mdi-account-group"
            title="Machine Groups"
            value="machine-groups"
            :to="{ name: 'machine-groups' }"
            rounded="lg"
          />

          <v-list-item
            prepend-icon="mdi-router-network"
            title="MQTT Rules"
            value="mqtt-rules"
            :to="{ name: 'mqtt-rules' }"
            rounded="lg"
          />

          <v-list-item
            v-if="authStore.isAdmin"
            prepend-icon="mdi-shield-account"
            title="Admin"
            value="admin"
            :to="{ name: 'admin' }"
            rounded="lg"
          />

          <!-- Phase F — Legacy tools group. Collapsed by default. Persists
               state in localStorage (key: cira.sidebar.legacyExpanded). -->
          <SidebarLegacyGroup :rail="rail" />

          <v-divider v-if="!rail" class="my-1" />

          <v-list-item
            prepend-icon="mdi-chevron-left"
            :title="rail ? '' : 'Collapse'"
            @click.stop="rail = !rail"
            rounded="lg"
          >
            <template #prepend>
              <v-icon>{{ rail ? 'mdi-chevron-right' : 'mdi-chevron-left' }}</v-icon>
            </template>
          </v-list-item>
        </v-list>
      </template>
    </v-navigation-drawer>

    <!-- Main Content -->
    <v-main>
      <router-view />
    </v-main>

    <!-- Footer -->
    <v-footer
      v-if="authStore.isAuthenticated && !isFullscreen"
      app
      height="32"
      color="surface"
      border="t"
    >
      <div class="d-flex align-center w-100 px-4">
        <span class="status-dot" :class="backendStatus" />
        <span class="ml-2 text-caption text-medium-emphasis">
          {{ backendStatus === 'connected' ? 'Backend Connected' : 'Connecting...' }}
        </span>

        <v-divider vertical class="mx-4" />

        <span class="text-caption text-medium-emphasis">
          Mode: <strong>{{ pipelineStore.mode === 'anomaly' ? 'Anomaly Detection' : pipelineStore.mode === 'regression' ? 'Regression' : 'Classification' }}</strong>
        </span>

        <v-spacer />

        <span class="text-caption text-medium-emphasis">CiRA ME v1.0.0</span>
      </div>
    </v-footer>

    <!-- Change Password Dialog -->
    <v-dialog v-model="showChangePassword" max-width="400" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="primary" class="mr-2">mdi-key</v-icon>
          Change Password
        </v-card-title>
        <v-card-text>
          <v-form ref="passwordForm" @submit.prevent="changePassword">
            <v-text-field
              v-model="passwordData.current_password"
              label="Current Password"
              type="password"
              :rules="[v => !!v || 'Current password is required']"
              prepend-inner-icon="mdi-lock"
              class="mb-2"
            />
            <v-text-field
              v-model="passwordData.new_password"
              label="New Password"
              type="password"
              :rules="[
                v => !!v || 'New password is required',
                v => v.length >= 6 || 'Password must be at least 6 characters'
              ]"
              prepend-inner-icon="mdi-lock-plus"
              class="mb-2"
            />
            <v-text-field
              v-model="passwordData.confirm_password"
              label="Confirm New Password"
              type="password"
              :rules="[
                v => !!v || 'Please confirm your password',
                v => v === passwordData.new_password || 'Passwords do not match'
              ]"
              prepend-inner-icon="mdi-lock-check"
            />
          </v-form>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="closePasswordDialog">Cancel</v-btn>
          <v-btn
            color="primary"
            :loading="changingPassword"
            @click="changePassword"
          >
            Change Password
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Restart Pipeline Confirmation Dialog -->
    <v-dialog v-model="pipelineStore.showResetDialog" max-width="440" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="warning" class="mr-2">mdi-restart-alert</v-icon>
          Restart Pipeline?
        </v-card-title>
        <v-card-text>
          <p class="mb-2">Do you want to restart all the process?</p>
          <p class="text-caption text-medium-emphasis">
            Windowing, Features, and the current Training run will be cleared.
            Models already saved to the Benchmark are not affected.
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="onCancelResetPipeline">No</v-btn>
          <v-btn color="warning" variant="flat" @click="onConfirmResetPipeline">
            Yes, Restart
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Global Snackbar -->
    <v-snackbar
      v-model="snackbar.show"
      :color="snackbar.color"
      :timeout="snackbar.timeout"
      location="bottom right"
    >
      {{ snackbar.message }}
      <template #actions>
        <v-btn variant="text" @click="snackbar.show = false">Close</v-btn>
      </template>
    </v-snackbar>
  </v-app>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useThemePref } from '@/composables/useThemePref'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import { useAssetTreeStore } from '@/stores/assetTree'
import LogoFull from '@/assets/LogoFull.vue'
import AssetTreeSidebar from '@/components/AssetTreeSidebar.vue'
import SidebarLegacyGroup from '@/components/SidebarLegacyGroup.vue'
import api from '@/services/api'

// Persisted theme (dark/light) — Phase 0. Both the top-bar icon and
// the user-menu item call themePref.toggle(); localStorage sync happens
// inside the composable.
const themePref = useThemePref()
const router = useRouter()
const authStore = useAuthStore()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()
const assetTreeStore = useAssetTreeStore()

const route = useRoute()
const isStandalone = computed(() => route.meta?.standalone === true)
// Fullscreen routes (e.g. asset-tree setup wizard) hide the sidebar + app bar.
const isFullscreen = computed(() => route.meta?.fullscreen === true)
const legacyCount = computed(() => assetTreeStore.legacyProjects.length)

// Phase F — dynamic Settings menu label that follows the tree's root_name.
// Examples: 'factory' → 'Factory Setup', 'stores' → 'Stores Setup'.
// Falls back to 'Structure Setup' if no config is loaded yet.
const rootSetupLabel = computed(() => {
  const root = assetTreeStore.config?.root_name
  if (!root) return 'Structure Setup'
  return `${root.charAt(0).toUpperCase() + root.slice(1)} Setup`
})

const drawer = ref(true)
const rail = ref(false)
const backendStatus = ref('loading')

// Change password state
const showChangePassword = ref(false)
const changingPassword = ref(false)
const passwordForm = ref<any>(null)
const passwordData = ref({
  current_password: '',
  new_password: '',
  confirm_password: ''
})

const snackbar = computed(() => notificationStore.snackbar)

const userInitials = computed(() => {
  const name = authStore.user?.display_name || authStore.user?.username || 'U'
  return name.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2)
})

const logout = async () => {
  await authStore.logout()
  router.push({ name: 'login' })
}

const requestNavigateToDataSource = () => {
  if (route.name === 'pipeline-data') return
  if (pipelineStore.hasDownstreamState) {
    pipelineStore.showResetDialog = true
  } else {
    router.push({ name: 'pipeline-data' })
  }
}

const onConfirmResetPipeline = () => {
  pipelineStore.reset()
  pipelineStore.showResetDialog = false
  router.push({ name: 'pipeline-data' })
}

const onCancelResetPipeline = () => {
  pipelineStore.showResetDialog = false
}

const closePasswordDialog = () => {
  showChangePassword.value = false
  passwordData.value = {
    current_password: '',
    new_password: '',
    confirm_password: ''
  }
}

const changePassword = async () => {
  if (!passwordForm.value) return

  const { valid } = await passwordForm.value.validate()
  if (!valid) return

  try {
    changingPassword.value = true

    await api.post('/api/auth/change-password', {
      current_password: passwordData.value.current_password,
      new_password: passwordData.value.new_password
    })

    notificationStore.showSuccess('Password changed successfully')
    closePasswordDialog()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to change password')
  } finally {
    changingPassword.value = false
  }
}

const checkBackendHealth = async () => {
  try {
    await api.get('/api/health')
    backendStatus.value = 'connected'
  } catch {
    backendStatus.value = 'disconnected'
  }
}

onMounted(() => {
  checkBackendHealth()
  setInterval(checkBackendHealth, 30000)
})

// When the user is authenticated (either on boot or after login), fetch
// the legacy-projects list once so the top-bar chip can render. The store
// caches the result; re-triggers on logout+login by re-watching auth state.
watch(
  () => authStore.isAuthenticated,
  (isAuth) => {
    if (isAuth) {
      assetTreeStore.ensureLegacyChecked()
    } else {
      assetTreeStore.reset()
    }
  },
  { immediate: true },
)
</script>
