<template>
  <v-card variant="tonal" class="pa-6">
    <div class="d-flex flex-column align-center text-center">
      <v-icon size="48" color="secondary" class="mb-3">mdi-tag-outline</v-icon>
      <h3 class="text-h6">Coming in Phase E</h3>
      <p class="text-body-2 text-medium-emphasis mt-2 mb-4">
        Interactive labeling via OculusT — window playback, keyboard
        shortcuts, and label review — lands in Phase E. For now, upload
        pre-labeled CSVs to the Data tab.
      </p>
      <v-btn
        color="secondary"
        variant="tonal"
        size="small"
        prepend-icon="mdi-open-in-new"
        :href="oculustUrl"
        target="_blank"
        rel="noopener"
      >
        Open OculusT
      </v-btn>
    </div>
  </v-card>
</template>

<script setup lang="ts">
/**
 * Phase B.7 — Labels placeholder.
 * OculusT integration is deferred to Phase E; this card just tells the
 * user how to get labeled data in the meantime.
 *
 * OculusT runs on port 3010 alongside CiRA ME. Reuse the current page's
 * hostname (not a hardcoded localhost) so links stay valid whether the
 * user is on localhost:3030, a LAN IP, or the .103 production host.
 */
import { computed } from 'vue'
import type { AssetNode } from '@/stores/assetTree'

defineProps<{ machine: AssetNode }>()

const oculustUrl = computed(() => {
  if (typeof window === 'undefined') return 'http://localhost:3010'
  const host = window.location.hostname || 'localhost'
  const isHttps = window.location.protocol === 'https:'
  return `${isHttps ? 'https' : 'http'}://${host}:3010`
})
</script>
