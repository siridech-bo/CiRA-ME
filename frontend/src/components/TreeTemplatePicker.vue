<template>
  <div class="tree-template-picker">
    <!-- Category tabs so the list stays scannable when we have 13+ templates. -->
    <v-chip-group
      v-model="selectedCategory"
      mandatory
      selected-class="text-primary"
      class="mb-3"
    >
      <v-chip
        v-for="cat in categories"
        :key="cat.value"
        :value="cat.value"
        size="small"
        variant="tonal"
      >
        {{ cat.label }} · {{ cat.count }}
      </v-chip>
    </v-chip-group>

    <!-- Card grid -->
    <v-progress-linear v-if="loading" indeterminate class="mb-3" />
    <div v-else class="template-grid">
      <v-card
        v-for="t in visibleTemplates"
        :key="t.id"
        :variant="selectedId === t.id ? 'elevated' : 'outlined'"
        :color="selectedId === t.id ? 'primary' : undefined"
        class="template-card pa-4"
        @click="selectedId = t.id"
      >
        <div class="d-flex align-start ga-3">
          <v-icon size="28" :color="selectedId === t.id ? 'primary' : 'medium-emphasis'">
            {{ t.icon || 'mdi-file-tree-outline' }}
          </v-icon>
          <div class="flex-grow-1" style="min-width: 0;">
            <div class="text-subtitle-2 font-weight-bold">
              {{ t.name }}
            </div>
            <div class="text-caption text-medium-emphasis mt-1">
              {{ t.description }}
            </div>
            <!-- Structure hint: show the level-names as a mini pipe. -->
            <div class="mt-2 template-levels">
              <span
                v-for="(lvl, i) in t.config.level_names"
                :key="i"
                class="template-level"
              >
                {{ lvl }}<span v-if="i < t.config.level_names.length - 1" class="mx-1">›</span>
              </span>
            </div>
          </div>
        </div>
      </v-card>
    </div>

    <!-- Apply button — enabled when a template is selected. -->
    <div class="mt-4 d-flex align-center ga-3">
      <v-btn
        color="primary"
        :disabled="!selectedId || applying"
        :loading="applying"
        @click="onApply"
      >
        <v-icon start>mdi-check-circle-outline</v-icon>
        Use this template
      </v-btn>
      <v-btn v-if="allowCancel" variant="text" @click="$emit('cancel')">
        Cancel
      </v-btn>
      <v-spacer />
      <span v-if="selectedTemplate" class="text-caption text-medium-emphasis">
        Root: <code>{{ selectedTemplate.config.root_name }}</code> ·
        {{ selectedTemplate.config.level_names.length }} levels
      </span>
    </div>

    <v-alert
      v-if="error"
      type="error"
      variant="tonal"
      density="compact"
      class="mt-3"
      closable
      @click:close="error = ''"
    >
      {{ error }}
    </v-alert>
  </div>
</template>

<script setup lang="ts">
/**
 * Reusable tree-template picker used in two places:
 *   1. Wizard Step 1 — shown below the "Or design your own" preset row.
 *   2. Admin view empty state — shown when the tree has zero nodes.
 *
 * Fetches /tree-templates on mount, groups by category, applies the
 * chosen one via POST /tree-templates/<id>/apply. On success, emits
 * 'applied' with the response so the parent can navigate / refresh.
 */
import { ref, computed, onMounted } from 'vue'
import api from '@/services/api'

interface TreeTemplate {
  id: string
  name: string
  description: string
  category: string
  category_label: string
  icon: string
  config: {
    level_names: string[]
    root_name: string
    topic_mode: string
    meta_prefixes: string[]
  }
  tree: any
}

const props = defineProps<{
  allowCancel?: boolean
}>()

const emit = defineEmits<{
  (e: 'applied', payload: { template_id: string; count: number }): void
  (e: 'cancel'): void
}>()

const templates = ref<TreeTemplate[]>([])
const loading = ref(true)
const applying = ref(false)
const error = ref('')
const selectedId = ref<string | null>(null)
const selectedCategory = ref<string>('all')

onMounted(async () => {
  await fetchTemplates()
})

async function fetchTemplates() {
  loading.value = true
  try {
    const res = await api.get('/api/asset-tree/tree-templates')
    templates.value = res.data?.templates || []
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Failed to load templates'
  } finally {
    loading.value = false
  }
}

const categories = computed(() => {
  const counts = new Map<string, { label: string; count: number }>()
  counts.set('all', { label: 'All', count: templates.value.length })
  for (const t of templates.value) {
    const existing = counts.get(t.category)
    if (existing) {
      existing.count++
    } else {
      counts.set(t.category, { label: t.category_label, count: 1 })
    }
  }
  return Array.from(counts, ([value, v]) => ({ value, label: v.label, count: v.count }))
})

const visibleTemplates = computed(() => {
  if (selectedCategory.value === 'all') return templates.value
  return templates.value.filter(t => t.category === selectedCategory.value)
})

const selectedTemplate = computed(() =>
  templates.value.find(t => t.id === selectedId.value) || null,
)

async function onApply() {
  if (!selectedId.value) return
  applying.value = true
  error.value = ''
  try {
    const res = await api.post(
      `/api/asset-tree/tree-templates/${selectedId.value}/apply`,
    )
    emit('applied', {
      template_id: res.data?.template_id || selectedId.value,
      count: res.data?.count || 0,
    })
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Failed to apply template'
  } finally {
    applying.value = false
  }
}
</script>

<style scoped>
.template-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}

.template-card {
  cursor: pointer;
  transition: transform 120ms ease, border-color 120ms ease;
}
.template-card:hover {
  transform: translateY(-1px);
}

.template-levels {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.75em;
  color: rgba(var(--v-theme-on-surface), 0.7);
  word-break: break-word;
}
.template-level {
  display: inline-block;
}
</style>
