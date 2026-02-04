<template>
  <v-app>
    <!-- App Bar -->
    <v-app-bar
      v-if="authStore.isAuthenticated"
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
      </v-btn-toggle>

      <v-spacer />

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
          <v-list-item prepend-icon="mdi-theme-light-dark" @click="toggleTheme">
            Toggle Theme
          </v-list-item>
          <v-list-item prepend-icon="mdi-logout" @click="logout" class="text-error">
            Logout
          </v-list-item>
        </v-list>
      </v-menu>
    </v-app-bar>

    <!-- Navigation Drawer -->
    <v-navigation-drawer
      v-if="authStore.isAuthenticated"
      v-model="drawer"
      :rail="rail"
      permanent
      @click="rail = false"
    >
      <v-list density="compact" nav>
        <v-list-item
          prepend-icon="mdi-view-dashboard"
          title="Dashboard"
          value="dashboard"
          :to="{ name: 'dashboard' }"
          rounded="lg"
        />

        <v-list-subheader v-if="!rail">PIPELINE</v-list-subheader>

        <v-list-item
          prepend-icon="mdi-database"
          title="Data Source"
          value="data"
          :to="{ name: 'pipeline-data' }"
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
      </v-list>

      <template #append>
        <v-list density="compact" nav>
          <v-divider class="mb-2" />

          <v-list-item
            v-if="authStore.isAdmin"
            prepend-icon="mdi-shield-account"
            title="Admin"
            value="admin"
            :to="{ name: 'admin' }"
            rounded="lg"
          />

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
      v-if="authStore.isAuthenticated"
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
          Mode: <strong>{{ pipelineStore.mode === 'anomaly' ? 'Anomaly Detection' : 'Classification' }}</strong>
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
import { ref, computed, onMounted } from 'vue'
import { useTheme } from 'vuetify'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import LogoFull from '@/assets/LogoFull.vue'
import api from '@/services/api'

const theme = useTheme()
const router = useRouter()
const authStore = useAuthStore()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

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

const toggleTheme = () => {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}

const logout = async () => {
  await authStore.logout()
  router.push({ name: 'login' })
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
</script>
