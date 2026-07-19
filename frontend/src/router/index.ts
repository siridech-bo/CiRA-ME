/**
 * CiRA ME - Vue Router Configuration
 */

import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useAssetTreeStore } from '@/stores/assetTree'
import { useNotificationStore } from '@/stores/notification'

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
      path: '/melab',
      name: 'melab',
      component: () => import('@/views/MeLabView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/mqtt',
      name: 'mqtt-management',
      component: () => import('@/views/MqttManagementView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/app-builder',
      name: 'app-builder',
      component: () => import('@/views/AppBuilderListView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/app-builder/:id',
      name: 'app-builder-editor',
      component: () => import('@/views/AppBuilderEditorView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/folder-watcher',
      name: 'folder-watcher-list',
      component: () => import('@/views/FolderWatcherListView.vue'),
      meta: { requiresAuth: true }
    },
    {
      // Alias for the App Builder "Log Watcher" template — jumps into the
      // Folder Watcher edit view with no :id, i.e. "new watcher" mode.
      path: '/folder-watcher/new',
      name: 'folder-watcher-new',
      component: () => import('@/views/FolderWatcherEditView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/projects',
      name: 'projects-list',
      component: () => import('@/views/ProjectsListView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/projects/:id',
      name: 'projects-detail',
      component: () => import('@/views/ProjectDetailView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/wizard',
      name: 'wizard',
      component: () => import('@/views/MultiDatasetWizardView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/folder-watcher/:id?',
      name: 'folder-watcher-edit',
      component: () => import('@/views/FolderWatcherEditView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/admin',
      name: 'admin',
      component: () => import('@/views/AdminView.vue'),
      meta: { requiresAuth: true, requiresAdmin: true }
    },
    // ── Machine Workspace (Phase B) ──────────────────────────────────────
    {
      // Per-machine dashboard: Overview / Data / Models / Deploy / Labels /
      // History tabs. Route guard (below) rejects non-machine ids so users
      // hitting a stale bookmark get redirected instead of a blank shell.
      path: '/machine/:id',
      name: 'machine-workspace',
      component: () => import('@/views/MachineWorkspaceView.vue'),
      meta: { requiresAuth: true, requiresMachineNode: true },
    },
    // ── Asset Tree (Phase A) ─────────────────────────────────────────────
    {
      // First-run wizard — fullscreen, no sidebar. Bypasses the
      // "config missing → redirect to wizard" guard for obvious reasons.
      path: '/setup/asset-tree',
      name: 'asset-tree-setup',
      component: () => import('@/views/AssetTreeSetupView.vue'),
      meta: { requiresAuth: true, fullscreen: true, skipAssetTreeGuard: true }
    },
    {
      path: '/settings/asset-tree',
      name: 'asset-tree-admin',
      component: () => import('@/views/AssetTreeAdminView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/settings/asset-tree/audit',
      name: 'asset-tree-audit',
      component: () => import('@/views/AssetTreeAdminView.vue'),
      props: { defaultTab: 'audit' },
      meta: { requiresAuth: true }
    },
    {
      // Phase C — dedicated Machine Groups view (was a stub tab under
      // AssetTreeAdminView until Phase C shipped).
      path: '/settings/machine-groups',
      name: 'machine-groups',
      component: () => import('@/views/MachineGroupsView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/apps/:slug',
      name: 'published-app',
      component: () => import('@/views/PublishedAppView.vue'),
      meta: { requiresAuth: false }
    },
    {
      path: '/standalone/:slug',
      name: 'standalone-app',
      component: () => import('@/views/PublishedAppView.vue'),
      meta: { requiresAuth: false, standalone: true }
    },
    {
      // Public-ish monitor view for a folder watcher. Backend endpoint is
      // still auth-gated; a future PR can add a public token model.
      path: '/watcher-view/:id',
      name: 'published-folder-watcher',
      component: () => import('@/views/PublishedFolderWatcherView.vue'),
      meta: { requiresAuth: true }
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

  // Asset-tree config guard — Phase A.5.
  // If authenticated + config missing → force wizard.
  // If wizard is done + user is on the wizard route → send home.
  // We skip this entire block for public/standalone routes and the login
  // page so guests aren't hit with a config fetch on their way in.
  if (
    isAuthenticated &&
    to.meta.requiresAuth &&
    !to.meta.skipAssetTreeGuard
  ) {
    const assetTreeStore = useAssetTreeStore()
    const configured = await assetTreeStore.ensureConfigChecked()
    if (!configured) {
      return next({ name: 'asset-tree-setup' })
    }
  }
  if (to.name === 'asset-tree-setup' && isAuthenticated) {
    const assetTreeStore = useAssetTreeStore()
    // Also verify at the destination: if config exists, don't let users
    // stay on the wizard route.
    const configured = await assetTreeStore.ensureConfigChecked()
    if (configured) {
      return next({ name: 'dashboard' })
    }
  }

  // Phase B — machine-workspace guard. Rejects stale ids that aren't at
  // machine level (or are retired). Redirect to the tree admin so the
  // user can pick a real one. Skip the guard on unauthenticated flows
  // — the earlier auth guard already handled those.
  if (isAuthenticated && to.meta.requiresMachineNode) {
    const notify = useNotificationStore()
    const idRaw = Number(to.params.id)
    if (!Number.isFinite(idRaw)) {
      notify.showError('Invalid machine id.')
      return next({ name: 'asset-tree-admin' })
    }
    try {
      const assetTreeStore = useAssetTreeStore()
      if (!assetTreeStore.treeLoaded) {
        await assetTreeStore.fetchTree()
      }
      const node = assetTreeStore.findNode(idRaw)
      if (!node) {
        notify.showError(`Machine ${idRaw} not found.`)
        return next({ name: 'asset-tree-admin' })
      }
      if (node.status === 'retired') {
        notify.showError(`Machine "${node.name}" is retired.`)
        return next({ name: 'asset-tree-admin' })
      }
      if (!assetTreeStore.isMachineNode(node)) {
        notify.showError(`"${node.name}" is not at the machine level.`)
        return next({ name: 'asset-tree-admin' })
      }
    } catch {
      notify.showError('Failed to verify machine.')
      return next({ name: 'asset-tree-admin' })
    }
  }

  next()
})

export default router
