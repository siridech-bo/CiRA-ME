/**
 * Theme preference composable — Phase 0 of the Asset Tree project.
 *
 * Wraps Vuetify's useTheme() with a localStorage layer so the user's choice
 * survives reload. Consumers (the top-bar toggle button, the user-menu
 * item) call `toggle()` or `setTheme('light' | 'dark')` and never touch
 * localStorage directly.
 *
 * The boot path in main.ts uses `readSavedTheme()` synchronously BEFORE
 * createVuetify so the very first paint matches the saved value — no
 * light→dark flash on reload for users who prefer dark, and no dark→light
 * flash for users who prefer light.
 */

import { computed } from 'vue'
import { useTheme } from 'vuetify'

const STORAGE_KEY = 'cira.theme'
type ThemeName = 'dark' | 'light'

const DEFAULT: ThemeName = 'dark'

/**
 * Read whichever theme was saved last, or fall back to the default.
 * Safe to call before Vue mounts — used in main.ts to seed
 * createVuetify({ theme: { defaultTheme } }).
 */
export function readSavedTheme(): ThemeName {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    // localStorage can be blocked in incognito or by policy — degrade silently.
  }
  return DEFAULT
}

function writeSavedTheme(name: ThemeName): void {
  try {
    localStorage.setItem(STORAGE_KEY, name)
  } catch {
    // ignore — theme still applies for this session
  }
}

/**
 * Composable for components that need to toggle or read the current theme.
 * MUST be called inside a component's setup function (uses useTheme).
 */
export function useThemePref() {
  const theme = useTheme()

  const current = computed<ThemeName>(() =>
    (theme.global.name.value as ThemeName) || DEFAULT,
  )

  const isDark = computed(() => current.value === 'dark')

  function setTheme(name: ThemeName): void {
    theme.global.name.value = name
    writeSavedTheme(name)
  }

  function toggle(): void {
    setTheme(current.value === 'dark' ? 'light' : 'dark')
  }

  return { current, isDark, setTheme, toggle }
}
