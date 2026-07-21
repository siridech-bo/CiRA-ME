<template>
  <div class="copy-block-wrap">
    <div class="copy-block" :class="{ multiline }">
      <div v-if="label" class="copy-block-label">{{ label }}</div>
      <pre class="copy-block-code" :class="{ multiline }"><code>{{ text }}</code></pre>
      <v-tooltip location="top" :text="copied ? 'Copied!' : 'Copy'">
        <template #activator="{ props: tipProps }">
          <v-btn
            v-bind="tipProps"
            :icon="copied ? 'mdi-check' : 'mdi-content-copy'"
            size="x-small"
            variant="text"
            :color="copied ? 'success' : undefined"
            class="copy-btn"
            @click="copy"
          />
        </template>
      </v-tooltip>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Small reusable "code with copy button" block.
 * Falls back to a textarea + document.execCommand path when the
 * Clipboard API isn't available (older browsers, insecure contexts).
 */
import { ref } from 'vue'

const props = defineProps<{
  text: string
  label?: string
  multiline?: boolean
}>()

const copied = ref(false)

async function copy() {
  const value = props.text
  let ok = false
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      ok = true
    } else {
      const ta = document.createElement('textarea')
      ta.value = value
      ta.style.position = 'fixed'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.select()
      ok = document.execCommand('copy')
      document.body.removeChild(ta)
    }
  } catch {
    ok = false
  }
  if (ok) {
    copied.value = true
    setTimeout(() => (copied.value = false), 1400)
  }
}
</script>

<style scoped>
.copy-block-wrap {
  margin-bottom: 8px;
}
.copy-block {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(var(--v-theme-on-surface), 0.06);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 6px;
  padding: 6px 8px 6px 12px;
  position: relative;
}
.copy-block.multiline {
  align-items: flex-start;
}
.copy-block-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: rgb(var(--v-theme-primary));
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin-right: 6px;
  padding: 2px 6px;
  background: rgba(var(--v-theme-primary), 0.12);
  border-radius: 3px;
  flex-shrink: 0;
}
.copy-block-code {
  font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  font-size: 0.85rem;
  margin: 0;
  padding: 2px 0;
  flex: 1;
  overflow-x: auto;
  white-space: nowrap;
}
.copy-block-code.multiline {
  white-space: pre;
  overflow-x: auto;
  padding: 4px 0;
}
.copy-btn {
  flex-shrink: 0;
}
</style>
