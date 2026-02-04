/**
 * CiRA ME - Auth Store
 * Manages authentication state and user info
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/services/api'

interface User {
  id: number
  username: string
  display_name: string
  role: string
  private_folder?: string
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const initialized = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const isAuthenticated = computed(() => !!user.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function initialize() {
    if (initialized.value) return

    try {
      loading.value = true
      const response = await api.get('/api/auth/me')
      user.value = response.data
    } catch {
      user.value = null
    } finally {
      loading.value = false
      initialized.value = true
    }
  }

  async function login(username: string, password: string) {
    try {
      loading.value = true
      error.value = null

      const response = await api.post('/api/auth/login', { username, password })
      user.value = response.data.user

      return { success: true }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Login failed'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  async function logout() {
    try {
      await api.post('/api/auth/logout')
    } finally {
      user.value = null
    }
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    try {
      loading.value = true
      error.value = null

      await api.post('/api/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword
      })

      return { success: true }
    } catch (e: any) {
      error.value = e.response?.data?.error || 'Password change failed'
      return { success: false, error: error.value }
    } finally {
      loading.value = false
    }
  }

  return {
    user,
    initialized,
    loading,
    error,
    isAuthenticated,
    isAdmin,
    initialize,
    login,
    logout,
    changePassword
  }
})
