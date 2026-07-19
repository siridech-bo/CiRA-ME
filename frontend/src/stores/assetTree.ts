/**
 * CiRA ME - Asset Tree Store (Phase A, 2026-07-18)
 *
 * Caches:
 *  - whether the asset-tree config exists (used by the router guard so we
 *    don't hit /api/asset-tree/config on every navigation),
 *  - the raw config object,
 *  - the preset payload (hierarchy / sensor / unit / rate).
 *
 * The router guard calls `ensureConfigChecked()` once per session; wizard
 * finish calls `invalidateConfig()` so the next navigation re-fetches and
 * unblocks the app.
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/services/api'

// ── Types ────────────────────────────────────────────────────────────────

export interface AssetTreeConfig {
  level_names?: string[]
  root_name?: string
  topic_mode?: 'strict' | 'learn'
  meta_prefixes?: string[]
  // Backend may add other fields (id, created_at, …) — we keep the rest.
  [key: string]: unknown
}

export interface SensorMeta {
  unit?: string | null
  sample_rate_hz?: number | null
  expected_min?: number | null
  expected_max?: number | null
  data_type?: 'float' | 'int' | 'string' | null
}

export interface AssetNode {
  id: number
  parent_id: number | null
  level: number
  name: string
  topic_path: string
  display_name?: string | null
  description?: string | null
  location_tag?: string | null
  status?: 'active' | 'retired'
  sensor_meta?: SensorMeta
  children?: AssetNode[]
}

export interface HierarchyPreset {
  value: string
  label: string
  levels: string[]
}

export interface SensorTemplate {
  value: string
  label: string
  sensors: Array<{
    name: string
    unit?: string
    sample_rate_hz?: number
    data_type?: 'float' | 'int' | 'string'
  }>
}

export interface UnitPreset {
  value: string
  label: string
}

export interface PresetsPayload {
  hierarchy_presets: HierarchyPreset[]
  sensor_templates: SensorTemplate[]
  unit_presets: UnitPreset[]
  sample_rate_presets: number[]
}

// ── Store ────────────────────────────────────────────────────────────────

export const useAssetTreeStore = defineStore('assetTree', () => {
  const config = ref<AssetTreeConfig | null>(null)
  const configChecked = ref(false)
  const configExists = ref(false)
  const presets = ref<PresetsPayload | null>(null)
  const loadingConfig = ref(false)
  const loadingPresets = ref(false)
  // Legacy projects — surfaced as a quiet chip in the top bar.
  const legacyProjects = ref<Array<{ id: number; name: string }>>([])
  const legacyChecked = ref(false)

  // Phase B — sidebar tree cache + navigation state.
  // Kept lightweight: fetched on demand, invalidated by wizard/admin writes
  // (call invalidateTree()).
  const tree = ref<AssetNode[]>([])
  const treeLoaded = ref(false)
  const loadingTree = ref(false)
  // Node ids the sidebar has expanded. Persisted in localStorage so the
  // shape survives navigations and reloads. Populated with sensible defaults
  // on first tree fetch (root + plants expanded).
  const expandedNodes = ref<Set<number>>(new Set())
  // The machine the user is currently "focused on" (via /machine/:id or
  // sidebar click). Read by legacy pipeline views to show the context banner.
  const currentMachineId = ref<number | null>(null)

  const isConfigured = computed(() => configExists.value)

  // Convenience: machine level index (=level_names.length - 2). For the
  // default 4-level preset (factory/plant/machine/sensor) → level 2.
  // For 3-level configs → level 1. Falls back to 2 when no config.
  const machineLevel = computed(() => {
    const names = config.value?.level_names
    if (!names || names.length < 2) return 2
    return names.length - 2
  })
  const sensorLevel = computed(() => {
    const names = config.value?.level_names
    if (!names || names.length === 0) return 3
    return names.length - 1
  })

  /**
   * Fetch /config exactly once per session, cache the result. Called by the
   * router guard before every non-login, non-setup navigation.
   */
  async function ensureConfigChecked(force = false): Promise<boolean> {
    if (configChecked.value && !force) return configExists.value
    loadingConfig.value = true
    try {
      const r = await api.get('/api/asset-tree/config')
      const body = r.data ?? {}
      // Backend returns {} when no config row exists.
      const hasConfig =
        body &&
        typeof body === 'object' &&
        Array.isArray((body as AssetTreeConfig).level_names) &&
        (body as AssetTreeConfig).level_names!.length > 0
      configExists.value = !!hasConfig
      config.value = hasConfig ? (body as AssetTreeConfig) : null
    } catch {
      // Treat network / 401 failures as "not configured" so we don't block
      // login flows; the guard falls back to normal behavior.
      configExists.value = false
      config.value = null
    } finally {
      loadingConfig.value = false
      configChecked.value = true
    }
    return configExists.value
  }

  function invalidateConfig() {
    configChecked.value = false
    configExists.value = false
    config.value = null
  }

  async function loadPresets(): Promise<PresetsPayload> {
    if (presets.value) return presets.value
    loadingPresets.value = true
    try {
      const r = await api.get('/api/asset-tree/presets')
      presets.value = r.data as PresetsPayload
      return presets.value
    } finally {
      loadingPresets.value = false
    }
  }

  async function saveConfig(payload: {
    level_names: string[]
    root_name: string
    topic_mode: 'strict' | 'learn'
    meta_prefixes: string[]
  }): Promise<AssetTreeConfig> {
    const r = await api.put('/api/asset-tree/config', payload)
    config.value = r.data
    configExists.value = true
    configChecked.value = true
    return r.data
  }

  async function ensureLegacyChecked(force = false): Promise<number> {
    if (legacyChecked.value && !force) return legacyProjects.value.length
    try {
      const r = await api.get('/api/projects')
      const rows = Array.isArray(r.data?.projects) ? r.data.projects : []
      legacyProjects.value = rows.map((p: any) => ({
        id: p.id,
        name: p.name,
      }))
    } catch {
      legacyProjects.value = []
    } finally {
      legacyChecked.value = true
    }
    return legacyProjects.value.length
  }

  // ── Tree cache (Phase B — sidebar + workspace shared source) ──────────

  const EXPANSION_STORAGE_KEY = 'cirame.assetTree.expanded'

  function loadExpansionFromStorage(): Set<number> {
    try {
      const raw = localStorage.getItem(EXPANSION_STORAGE_KEY)
      if (!raw) return new Set()
      const arr = JSON.parse(raw)
      if (!Array.isArray(arr)) return new Set()
      return new Set(arr.filter((v: unknown) => typeof v === 'number'))
    } catch { return new Set() }
  }

  function persistExpansion() {
    try {
      localStorage.setItem(
        EXPANSION_STORAGE_KEY,
        JSON.stringify([...expandedNodes.value]),
      )
    } catch { /* ignore quota errors */ }
  }

  // Hydrate expansion on store init (before any tree fetch).
  expandedNodes.value = loadExpansionFromStorage()

  function setNodeExpanded(nodeId: number, expanded: boolean) {
    const next = new Set(expandedNodes.value)
    if (expanded) next.add(nodeId)
    else next.delete(nodeId)
    expandedNodes.value = next
    persistExpansion()
  }

  function toggleNodeExpanded(nodeId: number) {
    setNodeExpanded(nodeId, !expandedNodes.value.has(nodeId))
  }

  function isNodeExpanded(nodeId: number): boolean {
    return expandedNodes.value.has(nodeId)
  }

  function walkTree(nodes: AssetNode[], fn: (n: AssetNode) => void) {
    for (const n of nodes) {
      fn(n)
      if (n.children) walkTree(n.children, fn)
    }
  }

  function findNode(id: number, nodes: AssetNode[] = tree.value): AssetNode | null {
    for (const n of nodes) {
      if (n.id === id) return n
      if (n.children) {
        const hit = findNode(id, n.children)
        if (hit) return hit
      }
    }
    return null
  }

  function isMachineNode(node: AssetNode | null | undefined): boolean {
    if (!node) return false
    return node.level === machineLevel.value
  }

  /**
   * Walk up parent_id chain to find the machine-level ancestor of any
   * node — used by the sidebar so clicking a sensor navigates to its
   * owning machine's workspace. Returns null if the node isn't in a
   * machine-rooted subtree (e.g. it IS the root, or the tree isn't
   * loaded yet).
   */
  function findMachineAncestor(id: number): AssetNode | null {
    let cursor = findNode(id)
    while (cursor) {
      if (cursor.level === machineLevel.value) return cursor
      if (cursor.parent_id == null) return null
      cursor = findNode(cursor.parent_id)
    }
    return null
  }

  /**
   * Fetch nested tree (active nodes only) with sensor_meta on leaves.
   * Auto-populates default expansion on the first successful fetch:
   *   - roots and plant-level nodes are open,
   *   - anything deeper stays closed unless the user opened it before
   *     (persisted).
   */
  async function fetchTree(force = false): Promise<AssetNode[]> {
    if (treeLoaded.value && !force) return tree.value
    loadingTree.value = true
    try {
      const r = await api.get('/api/asset-tree/nodes')
      tree.value = (r.data?.tree || []) as AssetNode[]
      treeLoaded.value = true

      // Seed expansion: everything at level ≤ plant (machineLevel - 1)
      // opens on first render. Existing user expansions in localStorage
      // are preserved.
      const seed = new Set(expandedNodes.value)
      const plantCutoff = Math.max(0, machineLevel.value - 1)
      walkTree(tree.value, (n) => {
        if (n.level <= plantCutoff) seed.add(n.id)
      })
      expandedNodes.value = seed
      persistExpansion()
    } catch {
      tree.value = []
      treeLoaded.value = false
    } finally {
      loadingTree.value = false
    }
    return tree.value
  }

  function invalidateTree() {
    treeLoaded.value = false
    tree.value = []
  }

  function setCurrentMachineId(id: number | null) {
    currentMachineId.value = id
  }

  const currentMachineNode = computed(() => {
    if (currentMachineId.value == null) return null
    return findNode(currentMachineId.value)
  })

  function reset() {
    config.value = null
    configChecked.value = false
    configExists.value = false
    presets.value = null
    legacyProjects.value = []
    legacyChecked.value = false
    tree.value = []
    treeLoaded.value = false
    currentMachineId.value = null
    // Note: we intentionally do NOT clear expandedNodes so the user's
    // sidebar layout persists across logout / login.
  }

  return {
    config,
    configChecked,
    configExists,
    presets,
    loadingConfig,
    loadingPresets,
    legacyProjects,
    legacyChecked,
    isConfigured,
    // Phase B — tree cache + nav
    tree,
    treeLoaded,
    loadingTree,
    expandedNodes,
    currentMachineId,
    machineLevel,
    sensorLevel,
    currentMachineNode,
    ensureConfigChecked,
    invalidateConfig,
    loadPresets,
    saveConfig,
    ensureLegacyChecked,
    fetchTree,
    invalidateTree,
    isNodeExpanded,
    setNodeExpanded,
    toggleNodeExpanded,
    findNode,
    findMachineAncestor,
    isMachineNode,
    setCurrentMachineId,
    reset,
  }
})
