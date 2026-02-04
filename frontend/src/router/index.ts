/**
 * CiRA ME - Vue Router Configuration
 */

import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { requiresGuest: true }
    },
    {
      path: '/',
      name: 'dashboard',
      component: () => import('@/views/DashboardView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline',
      name: 'pipeline',
      redirect: { name: 'pipeline-data' },
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline/data',
      name: 'pipeline-data',
      component: () => import('@/views/pipeline/DataSourceView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline/windowing',
      name: 'pipeline-windowing',
      component: () => import('@/views/pipeline/WindowingView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline/features',
      name: 'pipeline-features',
      component: () => import('@/views/pipeline/FeaturesView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline/training',
      name: 'pipeline-training',
      component: () => import('@/views/pipeline/TrainingView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/pipeline/deploy',
      name: 'pipeline-deploy',
      component: () => import('@/views/pipeline/DeployView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/admin',
      name: 'admin',
      component: () => import('@/views/AdminView.vue'),
      meta: { requiresAuth: true, requiresAdmin: true }
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/'
    }
  ]
})

// Navigation guards
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()

  // Initialize auth state if needed
  if (!authStore.initialized) {
    await authStore.initialize()
  }

  const isAuthenticated = authStore.isAuthenticated
  const isAdmin = authStore.isAdmin

  // Guest-only routes (login)
  if (to.meta.requiresGuest && isAuthenticated) {
    return next({ name: 'dashboard' })
  }

  // Protected routes
  if (to.meta.requiresAuth && !isAuthenticated) {
    return next({ name: 'login' })
  }

  // Admin-only routes
  if (to.meta.requiresAdmin && !isAdmin) {
    return next({ name: 'dashboard' })
  }

  next()
})

export default router
