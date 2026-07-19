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

  const isConfigured = computed(() => configExists.value)

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

  function reset() {
    config.value = null
    configChecked.value = false
    configExists.value = false
    presets.value = null
    legacyProjects.value = []
    legacyChecked.value = false
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
    ensureConfigChecked,
    invalidateConfig,
    loadPresets,
    saveConfig,
    ensureLegacyChecked,
    reset,
  }
})
