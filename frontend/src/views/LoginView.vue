<template>
  <v-container fluid class="fill-height login-container">
    <v-row align="center" justify="center">
      <v-col cols="12" sm="8" md="4">
        <v-card class="pa-6" elevation="8">
          <!-- Logo -->
          <div class="text-center mb-6">
            <LogoFull show-tagline />
          </div>

          <!-- Login Form -->
          <v-form ref="form" @submit.prevent="handleLogin">
            <v-text-field
              v-model="username"
              label="Username"
              prepend-inner-icon="mdi-account"
              :rules="[rules.required]"
              :disabled="loading"
              autofocus
            />

            <v-text-field
              v-model="password"
              label="Password"
              prepend-inner-icon="mdi-lock"
              :type="showPassword ? 'text' : 'password'"
              :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
              :rules="[rules.required]"
              :disabled="loading"
              @click:append-inner="showPassword = !showPassword"
            />

            <v-alert
              v-if="error"
              type="error"
              variant="tonal"
              density="compact"
              class="mb-4"
            >
              {{ error }}
            </v-alert>

            <v-btn
              type="submit"
              color="primary"
              size="large"
              block
              :loading="loading"
            >
              <v-icon start>mdi-login</v-icon>
              Sign In
            </v-btn>
          </v-form>

          <!-- Footer -->
          <div class="text-center mt-6 text-caption text-medium-emphasis">
            Default: admin / admin123
          </div>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import LogoFull from '@/assets/LogoFull.vue'

const router = useRouter()
const authStore = useAuthStore()
const notificationStore = useNotificationStore()

const username = ref('')
const password = ref('')
const showPassword = ref(false)
const loading = ref(false)
const error = ref('')

const rules = {
  required: (v: string) => !!v || 'This field is required'
}

async function handleLogin() {
  if (!username.value || !password.value) return

  loading.value = true
  error.value = ''

  const result = await authStore.login(username.value, password.value)

  if (result.success) {
    notificationStore.showSuccess('Welcome to CiRA ME!')
    router.push({ name: 'dashboard' })
  } else {
    error.value = result.error || 'Login failed'
  }

  loading.value = false
}
</script>

<style scoped lang="scss">
.login-container {
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
  min-height: 100vh;
}
</style>
