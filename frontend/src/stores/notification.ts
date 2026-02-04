/**
 * CiRA ME - Notification Store
 * Manages global notifications and snackbars
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'

interface Snackbar {
  show: boolean
  message: string
  color: string
  timeout: number
}

export const useNotificationStore = defineStore('notification', () => {
  const snackbar = ref<Snackbar>({
    show: false,
    message: '',
    color: 'info',
    timeout: 3000
  })

  function showSuccess(message: string, timeout = 3000) {
    snackbar.value = {
      show: true,
      message,
      color: 'success',
      timeout
    }
  }

  function showError(message: string, timeout = 5000) {
    snackbar.value = {
      show: true,
      message,
      color: 'error',
      timeout
    }
  }

  function showWarning(message: string, timeout = 4000) {
    snackbar.value = {
      show: true,
      message,
      color: 'warning',
      timeout
    }
  }

  function showInfo(message: string, timeout = 3000) {
    snackbar.value = {
      show: true,
      message,
      color: 'info',
      timeout
    }
  }

  function hide() {
    snackbar.value.show = false
  }

  return {
    snackbar,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    hide
  }
})
