/**
 * CiRA ME - Main Entry Point
 * Machine Intelligence for Edge Computing
 */

import { createApp } from 'vue'
import { createPinia } from 'pinia'

// Vuetify
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'

// App
import App from './App.vue'
import router from './router'

// Custom styles
import './styles/main.scss'

// Create Vuetify instance with custom theme
const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        dark: true,
        colors: {
          background: '#0F172A',
          surface: '#1E293B',
          'surface-bright': '#334155',
          'surface-light': '#475569',
          'surface-variant': '#64748B',
          'on-surface-variant': '#E2E8F0',
          primary: '#6366F1',
          'primary-darken-1': '#4F46E5',
          secondary: '#22D3EE',
          'secondary-darken-1': '#06B6D4',
          error: '#EF4444',
          info: '#3B82F6',
          success: '#10B981',
          warning: '#F59E0B',
        },
      },
      light: {
        dark: false,
        colors: {
          background: '#F8FAFC',
          surface: '#FFFFFF',
          'surface-bright': '#F1F5F9',
          'surface-light': '#E2E8F0',
          'surface-variant': '#CBD5E1',
          'on-surface-variant': '#334155',
          primary: '#6366F1',
          'primary-darken-1': '#4F46E5',
          secondary: '#06B6D4',
          'secondary-darken-1': '#0891B2',
          error: '#DC2626',
          info: '#2563EB',
          success: '#059669',
          warning: '#D97706',
        },
      },
    },
  },
  defaults: {
    VBtn: {
      variant: 'flat',
      rounded: 'lg',
    },
    VCard: {
      rounded: 'lg',
      elevation: 0,
    },
    VTextField: {
      variant: 'outlined',
      density: 'comfortable',
    },
    VSelect: {
      variant: 'outlined',
      density: 'comfortable',
    },
  },
})

// Create app
const app = createApp(App)

// Use plugins
app.use(createPinia())
app.use(router)
app.use(vuetify)

// Mount app
app.mount('#app')
