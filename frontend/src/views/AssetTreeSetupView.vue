<template>
  <v-app-bar
    elevation="0"
    color="surface"
    border="b"
    density="comfortable"
  >
    <template #prepend>
      <v-icon color="primary" class="ml-4">mdi-file-tree</v-icon>
    </template>
    <span class="text-h6 font-weight-bold">Asset Tree Setup</span>
    <v-spacer />
    <v-btn
      icon
      variant="text"
      density="comfortable"
      :aria-label="themePref.isDark.value ? 'Switch to light mode' : 'Switch to dark mode'"
      :title="themePref.isDark.value ? 'Switch to light mode' : 'Switch to dark mode'"
      @click="themePref.toggle()"
    >
      <v-icon>{{ themePref.isDark.value ? 'mdi-weather-sunny' : 'mdi-weather-night' }}</v-icon>
    </v-btn>
    <v-btn
      variant="text"
      prepend-icon="mdi-logout"
      @click="onLogout"
    >
      {{ authStore.user?.username }}
    </v-btn>
  </v-app-bar>

  <v-container v-if="!isAdmin" fluid class="pa-6">
    <v-card class="pa-8 mx-auto" max-width="600">
      <div class="d-flex flex-column align-center">
        <v-icon size="64" color="warning" class="mb-4">mdi-shield-alert</v-icon>
        <h2 class="text-h5 font-weight-bold mb-3">Setup pending</h2>
        <p class="text-body-1 text-center mb-4">
          Please ask an administrator to complete asset-tree setup before using CiRA ME.
        </p>
        <p class="text-body-2 text-medium-emphasis text-center">
          Your account: <strong>{{ authStore.user?.username }}</strong> ({{ authStore.user?.role }}).
          Sign out and back in as an administrator, or wait for one to configure the system.
        </p>
      </div>
    </v-card>
  </v-container>

  <v-container v-else fluid class="pa-6 setup-container">
    <div class="setup-inner mx-auto">
      <div class="d-flex align-center mb-4">
        <div>
          <h1 class="text-h4 font-weight-bold">Welcome to CiRA ME</h1>
          <p class="text-body-2 text-medium-emphasis">
            Let's set up your physical-asset hierarchy. This runs once —
            you can edit everything later under Settings → Asset Tree.
          </p>
        </div>
      </div>

      <v-stepper
        v-model="step"
        :items="stepItems"
        hide-actions
        alt-labels
        class="setup-stepper"
      >
        <template #item.1>
          <v-card flat class="pa-4">
            <h2 class="text-h6 mb-2">1. Choose a starting preset</h2>
            <p class="text-body-2 text-medium-emphasis mb-4">
              These are common industry hierarchies. Pick the closest match — you
              can rename levels in the next step. Choose <strong>Custom</strong>
              to start from a blank slate.
            </p>

            <v-row v-if="loadingPresets">
              <v-col cols="12">
                <v-skeleton-loader type="card" />
              </v-col>
            </v-row>

            <v-row v-else>
              <v-col
                v-for="p in hierarchyPresets"
                :key="p.value"
                cols="12"
                sm="6"
                md="4"
                lg="2"
                class="d-flex"
              >
                <v-card
                  class="preset-card flex-grow-1"
                  :class="{
                    'preset-selected': selectedPreset === p.value,
                    'preset-custom': p.value === 'custom',
                  }"
                  @click="pickPreset(p)"
                >
                  <div class="pa-4 text-center d-flex flex-column align-center">
                    <v-icon size="42" :color="presetColor(p.value)">
                      {{ presetIcon(p.value) }}
                    </v-icon>
                    <div class="text-subtitle-1 font-weight-bold mt-2">
                      {{ p.label }}
                    </div>
                    <div class="text-caption text-medium-emphasis mt-2 preset-preview">
                      <template v-if="p.levels.length">
                        <div
                          v-for="(lvl, i) in p.levels"
                          :key="lvl"
                          :style="{ paddingLeft: `${i * 6}px` }"
                        >
                          <v-icon size="12" class="mr-1">mdi-subdirectory-arrow-right</v-icon>{{ lvl }}
                        </div>
                      </template>
                      <template v-else>
                        <em>Start blank — build your own levels</em>
                      </template>
                    </div>
                  </div>
                </v-card>
              </v-col>
            </v-row>

            <div class="d-flex justify-end mt-4">
              <v-btn
                color="primary"
                :disabled="!selectedPreset"
                append-icon="mdi-arrow-right"
                @click="goStep(2)"
              >
                Next
              </v-btn>
            </div>
          </v-card>
        </template>

        <template #item.2>
          <v-card flat class="pa-4">
            <h2 class="text-h6 mb-2">2. Level names</h2>
            <p class="text-body-2 text-medium-emphasis mb-4">
              These become MQTT topic segments and folder names on disk.
              Use lowercase, no spaces. Between 2 and 6 levels.
            </p>

            <div class="d-flex flex-column ga-2 mb-4">
              <div
                v-for="(lvl, i) in levelNames"
                :key="i"
                class="d-flex align-center ga-2"
              >
                <v-chip size="small" color="primary" variant="tonal">
                  Level {{ i + 1 }}
                </v-chip>
                <v-text-field
                  v-model="levelNames[i]"
                  :rules="[validNameRule]"
                  hide-details="auto"
                  density="compact"
                  :placeholder="i === 0 ? 'e.g. factory' : 'name'"
                  class="flex-grow-1"
                  @update:model-value="onLevelNameChange"
                />
                <v-btn
                  v-if="canRemoveLevel(i)"
                  icon="mdi-close"
                  size="x-small"
                  variant="text"
                  :title="`Remove level ${i + 1}`"
                  @click="removeLevel(i)"
                />
                <span v-else class="remove-spacer" />
              </div>
              <v-btn
                v-if="levelNames.length < 6"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-plus"
                class="align-self-start"
                @click="addLevel"
              >
                Add level
              </v-btn>
              <div v-else class="text-caption text-medium-emphasis">
                Maximum 6 levels.
              </div>
            </div>

            <v-card variant="tonal" class="pa-3 topic-preview">
              <div class="text-caption text-medium-emphasis mb-1">
                Topic pattern
              </div>
              <code class="topic-code">{{ topicPattern }}</code>
              <div class="text-caption text-medium-emphasis mt-3 mb-1">
                Example
              </div>
              <code class="topic-code">{{ topicExample }}</code>
            </v-card>

            <div class="d-flex justify-space-between mt-4">
              <v-btn variant="text" prepend-icon="mdi-arrow-left" @click="goStep(1)">
                Back
              </v-btn>
              <v-btn
                color="primary"
                :disabled="!levelsValid"
                append-icon="mdi-arrow-right"
                @click="goStep(3)"
              >
                Next
              </v-btn>
            </div>
          </v-card>
        </template>

        <template #item.3>
          <v-card flat class="pa-4">
            <div class="d-flex align-center mb-2">
              <h2 class="text-h6">3. Build your tree</h2>
              <v-spacer />
              <v-btn
                variant="text"
                size="small"
                prepend-icon="mdi-code-json"
                @click="showImportDialog = true"
              >
                Import
              </v-btn>
            </div>
            <p class="text-body-2 text-medium-emphasis mb-4">
              Optional. Add plants, machines, and sensors now — or skip and
              finish setup, then build them later under Settings.
            </p>

            <div class="tree-editor-grid">
              <div class="tree-editor-left">
                <div class="d-flex align-center mb-2">
                  <span class="text-subtitle-2">Structure</span>
                  <v-spacer />
                  <v-btn
                    v-if="!rootNode"
                    size="x-small"
                    variant="tonal"
                    color="primary"
                    prepend-icon="mdi-plus"
                    @click="addRootNode"
                  >
                    Add {{ levelNames[0] || 'root' }}
                  </v-btn>
                </div>
                <div v-if="!rootNode" class="empty-tree text-caption text-medium-emphasis">
                  No nodes yet. Click "Add {{ levelNames[0] || 'root' }}" or skip
                  this step to finish setup with an empty tree.
                </div>
                <AssetTreeNodeEditor
                  v-else
                  :node="rootNode"
                  :depth="0"
                  :max-depth="maxDepth"
                  :selected-id="selectedNodeId"
                  :level-names="levelNames"
                  @select="selectNode"
                  @add-child="onAddChild"
                  @delete-node="onDeleteLocalNode"
                  @move-node="onMoveNode"
                />
              </div>

              <div class="tree-editor-right">
                <div v-if="!selectedNode" class="empty-detail text-caption text-medium-emphasis">
                  Select a node to edit its details.
                </div>
                <div v-else>
                  <div class="d-flex align-center mb-3">
                    <v-icon color="primary" class="mr-2">mdi-pencil</v-icon>
                    <span class="text-subtitle-1 font-weight-bold">
                      {{ levelNames[selectedNode.level] || 'Node' }} details
                    </span>
                  </div>

                  <v-text-field
                    v-model="selectedNode.name"
                    label="Name (topic segment)"
                    :rules="[validNameRule]"
                    hint="Letters, digits, underscore, hyphen only"
                    persistent-hint
                    density="compact"
                    class="mb-2"
                  />
                  <v-text-field
                    v-model="selectedNode.display_name"
                    label="Display name (optional)"
                    density="compact"
                    class="mb-2"
                  />
                  <v-textarea
                    v-model="selectedNode.description"
                    label="Description (optional)"
                    rows="2"
                    density="compact"
                    class="mb-2"
                  />
                  <v-text-field
                    v-model="selectedNode.location_tag"
                    label="Location tag (optional)"
                    density="compact"
                    class="mb-3"
                  />

                  <v-card variant="tonal" class="pa-2 mb-3">
                    <div class="text-caption text-medium-emphasis mb-1">
                      Topic path
                    </div>
                    <div class="d-flex align-center">
                      <code class="flex-grow-1">{{ selectedTopicPath }}</code>
                      <v-btn
                        icon="mdi-content-copy"
                        size="x-small"
                        variant="text"
                        title="Copy topic path"
                        @click="copyText(selectedTopicPath)"
                      />
                    </div>
                  </v-card>

                  <!-- Machine-level actions -->
                  <template v-if="isMachineLevel(selectedNode)">
                    <v-btn
                      variant="tonal"
                      color="secondary"
                      prepend-icon="mdi-content-duplicate"
                      block
                      class="mb-3"
                      @click="showCopyMachineDialog = true"
                    >
                      Copy sensors from another machine
                    </v-btn>
                  </template>

                  <!-- Sensor-leaf detail -->
                  <template v-if="isSensorLevel(selectedNode)">
                    <v-divider class="mb-3" />
                    <div class="text-subtitle-2 mb-2">Sensor metadata</div>
                    <SensorMetaEditor
                      v-model="selectedNode.sensor_meta"
                      :unit-presets="unitPresets"
                      :rate-presets="ratePresets"
                    />
                  </template>

                  <!-- Template panel (only when a machine node is selected) -->
                  <template v-if="isMachineLevel(selectedNode) && sensorTemplates.length">
                    <v-divider class="mb-3 mt-2" />
                    <div class="text-subtitle-2 mb-2">Apply a sensor template</div>
                    <div class="text-caption text-medium-emphasis mb-2">
                      Replaces this machine's current sensor children.
                    </div>
                    <v-btn
                      v-for="tpl in sensorTemplates"
                      :key="tpl.value"
                      size="small"
                      variant="tonal"
                      class="mr-2 mb-2"
                      @click="applyTemplate(tpl)"
                    >
                      {{ tpl.label }}
                    </v-btn>
                  </template>
                </div>
              </div>
            </div>

            <div class="d-flex justify-space-between mt-4">
              <v-btn variant="text" prepend-icon="mdi-arrow-left" @click="goStep(2)">
                Back
              </v-btn>
              <div>
                <v-btn variant="text" class="mr-2" @click="skipToConfirm">
                  Skip for now
                </v-btn>
                <v-btn
                  color="primary"
                  append-icon="mdi-arrow-right"
                  @click="goStep(4)"
                >
                  Next
                </v-btn>
              </div>
            </div>
          </v-card>
        </template>

        <template #item.4>
          <v-card flat class="pa-4">
            <h2 class="text-h6 mb-2">4. MQTT publisher rules</h2>
            <p class="text-body-2 text-medium-emphasis mb-4">
              How CiRA ME handles incoming MQTT topics. You can change this later.
            </p>

            <v-card variant="tonal" class="pa-3 mb-4">
              <div class="text-caption text-medium-emphasis mb-1">
                Devices must publish to
              </div>
              <code class="topic-code">{{ topicPattern }}</code>
            </v-card>

            <div class="text-subtitle-2 mb-2">Mode</div>
            <v-radio-group v-model="topicMode" density="compact" class="mb-3">
              <v-radio value="strict">
                <template #label>
                  <div>
                    <strong>Strict</strong>
                    <span class="text-caption text-medium-emphasis d-block">
                      Only accept topics that match a known tree path. Unknown
                      topics are rejected + logged. Recommended.
                    </span>
                  </div>
                </template>
              </v-radio>
              <v-radio value="learn">
                <template #label>
                  <div>
                    <strong>Learn</strong>
                    <span class="text-caption text-medium-emphasis d-block">
                      Auto-create nodes for unknown topics. Useful during
                      initial device rollout.
                    </span>
                  </div>
                </template>
              </v-radio>
            </v-radio-group>

            <div class="text-subtitle-2 mb-2">Meta-topic exceptions</div>
            <p class="text-caption text-medium-emphasis mb-2">
              Comma-separated segment prefixes that bypass tree matching
              (typically device health / config / commands).
            </p>
            <v-text-field
              v-model="metaPrefixesText"
              placeholder="_meta,_health,_config,_cmd"
              density="compact"
              class="mb-4"
            />

            <v-card variant="outlined" class="pa-3 mb-3">
              <div class="text-subtitle-2 mb-2">Test a topic</div>
              <p class="text-caption text-medium-emphasis mb-2">
                Type a topic to see how it would be routed once the tree is saved.
                This requires nodes to exist — skip if you're doing this step first.
              </p>
              <v-text-field
                v-model="testTopic"
                placeholder="factory/plant_A/machine_1/temperature"
                density="compact"
                clearable
                append-inner-icon="mdi-magnify"
                hide-details
              />
              <div v-if="testResult" class="mt-3 test-result">
                <template v-if="testResult.valid && testResult.route_to">
                  <v-icon color="success" class="mr-1">mdi-check-circle</v-icon>
                  <span>
                    Routes to <code>{{ testResult.route_to.topic_path }}</code>
                  </span>
                </template>
                <template v-else-if="testResult.valid && !testResult.route_to">
                  <v-icon color="warning" class="mr-1">mdi-alert-outline</v-icon>
                  <span>
                    Meta topic —
                    {{ (testResult.warnings || []).join(', ') || 'not routed to data' }}
                  </span>
                </template>
                <template v-else>
                  <v-icon color="error" class="mr-1">mdi-close-circle</v-icon>
                  <span>Invalid — {{ testResult.reason }}</span>
                </template>
              </div>
              <div v-else-if="testing" class="text-caption text-medium-emphasis mt-2">
                Testing…
              </div>
            </v-card>

            <div class="d-flex justify-space-between">
              <v-btn variant="text" prepend-icon="mdi-arrow-left" @click="goStep(3)">
                Back
              </v-btn>
              <v-btn color="primary" append-icon="mdi-arrow-right" @click="goStep(5)">
                Next
              </v-btn>
            </div>
          </v-card>
        </template>

        <template #item.5>
          <v-card flat class="pa-4">
            <h2 class="text-h6 mb-2">5. Confirm &amp; finish</h2>
            <p class="text-body-2 text-medium-emphasis mb-4">
              Review your setup. Click <strong>Finish</strong> to save.
            </p>

            <v-card variant="tonal" class="pa-4 mb-4">
              <div class="summary-row">
                <div class="summary-label">Preset</div>
                <div>{{ selectedPresetLabel }}</div>
              </div>
              <div class="summary-row">
                <div class="summary-label">Levels ({{ levelNames.length }})</div>
                <div>
                  <v-chip
                    v-for="(lvl, i) in levelNames"
                    :key="i"
                    size="x-small"
                    class="mr-1 mb-1"
                    variant="tonal"
                  >
                    {{ lvl }}
                  </v-chip>
                </div>
              </div>
              <div class="summary-row">
                <div class="summary-label">Root name</div>
                <div><code>{{ rootName || levelNames[0] || '(none)' }}</code></div>
              </div>
              <div class="summary-row">
                <div class="summary-label">Node count</div>
                <div>{{ nodeCountByLevelText }}</div>
              </div>
              <div class="summary-row">
                <div class="summary-label">MQTT mode</div>
                <div>
                  <v-chip size="small" :color="topicMode === 'strict' ? 'success' : 'warning'">
                    {{ topicMode }}
                  </v-chip>
                </div>
              </div>
              <div class="summary-row">
                <div class="summary-label">Meta prefixes</div>
                <div>
                  <code>{{ metaPrefixesArray.join(', ') || '(none)' }}</code>
                </div>
              </div>
            </v-card>

            <v-alert
              v-if="saveError"
              type="error"
              variant="tonal"
              class="mb-3"
            >
              {{ saveError }}
            </v-alert>

            <div class="d-flex justify-space-between">
              <v-btn variant="text" prepend-icon="mdi-arrow-left" @click="goStep(4)">
                Back
              </v-btn>
              <v-btn
                color="primary"
                append-icon="mdi-check"
                :loading="saving"
                @click="finish"
              >
                Finish setup
              </v-btn>
            </div>
          </v-card>
        </template>
      </v-stepper>
    </div>

    <!-- Import dialog -->
    <v-dialog v-model="showImportDialog" max-width="640">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="primary" class="mr-2">mdi-code-json</v-icon>
          Import tree from JSON
        </v-card-title>
        <v-card-text>
          <p class="text-body-2 text-medium-emphasis mb-3">
            Paste a nested JSON spec. Root must match your top-level name.
            <em>Note:</em> import replaces any locally-drafted tree and calls
            <code>POST /import</code> immediately.
          </p>
          <v-textarea
            v-model="importText"
            rows="10"
            placeholder='{"name":"factory","children":[{"name":"plant_A","children":[...]}]}'
            monospace
          />
          <v-alert
            v-if="importError"
            type="error"
            variant="tonal"
            class="mt-2"
            density="compact"
          >
            {{ importError }}
          </v-alert>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showImportDialog = false">Cancel</v-btn>
          <v-btn color="primary" :loading="importing" @click="doImport">
            Import
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Copy sensors from another machine -->
    <v-dialog v-model="showCopyMachineDialog" max-width="480">
      <v-card>
        <v-card-title>Copy sensors from…</v-card-title>
        <v-card-text>
          <p class="text-body-2 text-medium-emphasis mb-3">
            Choose a machine to copy its sensor children onto <strong>{{ selectedNode?.name }}</strong>.
            Existing sensors on this machine will be replaced.
          </p>
          <v-select
            v-model="copySourceMachineId"
            :items="machineNodesForPicker"
            item-title="topic_path"
            item-value="id"
            label="Source machine"
            density="compact"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCopyMachineDialog = false">Cancel</v-btn>
          <v-btn
            color="primary"
            :disabled="!copySourceMachineId"
            @click="copyMachineSensors"
          >
            Copy
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
/**
 * First-run wizard — Phase A.5 + A.6.
 * Renders fullscreen (no sidebar); the router guard blocks all other routes
 * until PUT /config succeeds. See router/index.ts.
 *
 * Wizard state lives entirely in this component. On finish we:
 *   1. PUT /api/asset-tree/config    (level names, root, topic mode, meta prefixes)
 *   2. For each drafted node in order → POST /api/asset-tree/nodes
 *   3. Invalidate the assetTree store's config cache → router lets us leave.
 *
 * "Skip for now" jumps to the confirm step with the tree still empty. Users
 * can build the tree later at /settings/asset-tree.
 */

import { ref, computed, watch, onMounted, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useThemePref } from '@/composables/useThemePref'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import {
  useAssetTreeStore,
  type AssetNode,
  type SensorMeta,
  type HierarchyPreset,
  type SensorTemplate,
} from '@/stores/assetTree'
import api from '@/services/api'
import AssetTreeNodeEditor from '@/components/AssetTreeNodeEditor.vue'
import SensorMetaEditor from '@/components/SensorMetaEditor.vue'

const NAME_REGEX = /^[A-Za-z0-9_-]+$/

const router = useRouter()
const themePref = useThemePref()
const authStore = useAuthStore()
const notificationStore = useNotificationStore()
const assetTreeStore = useAssetTreeStore()

const isAdmin = computed(() => authStore.user?.role === 'admin')

const step = ref(1)
const stepItems = [
  '1. Preset',
  '2. Level names',
  '3. Build tree',
  '4. MQTT rules',
  '5. Confirm',
]

const loadingPresets = ref(true)
const hierarchyPresets = ref<HierarchyPreset[]>([])
const sensorTemplates = ref<SensorTemplate[]>([])
const unitPresets = ref<Array<{ value: string; label: string }>>([])
const ratePresets = ref<number[]>([])

const selectedPreset = ref<string>('')
const selectedPresetLabel = computed(() => {
  const p = hierarchyPresets.value.find(h => h.value === selectedPreset.value)
  return p?.label || selectedPreset.value || '(none)'
})

const levelNames = ref<string[]>([])
const topicMode = ref<'strict' | 'learn'>('strict')
const metaPrefixesText = ref('_meta,_health,_config,_cmd')
const rootName = ref('')

// Local tree state. Nodes get negative IDs until saved to backend, so we
// can track and diff without collisions.
let idCounter = -1
type LocalNode = AssetNode & {
  children: LocalNode[]
  sensor_meta?: SensorMeta
}
const rootNode = ref<LocalNode | null>(null)
const selectedNodeId = ref<number | null>(null)

const selectedNode = computed<LocalNode | null>(() => {
  if (selectedNodeId.value == null || !rootNode.value) return null
  return findNode(rootNode.value, selectedNodeId.value)
})

const selectedTopicPath = computed(() => {
  if (!selectedNode.value) return ''
  return computeTopicPath(selectedNode.value)
})

// ── Presets ──────────────────────────────────────────────────────────────
onMounted(async () => {
  try {
    const p = await assetTreeStore.loadPresets()
    hierarchyPresets.value = p.hierarchy_presets
    sensorTemplates.value = p.sensor_templates
    unitPresets.value = p.unit_presets
    ratePresets.value = p.sample_rate_presets
  } catch {
    notificationStore.showError('Failed to load presets')
  } finally {
    loadingPresets.value = false
  }
})

function presetIcon(v: string) {
  switch (v) {
    case 'factory': return 'mdi-factory'
    case 'hospital': return 'mdi-hospital-building'
    case 'fleet': return 'mdi-truck'
    case 'farm': return 'mdi-tractor'
    case 'custom': return 'mdi-cog'
    default: return 'mdi-shape'
  }
}
function presetColor(v: string) {
  switch (v) {
    case 'factory': return 'primary'
    case 'hospital': return 'error'
    case 'fleet': return 'info'
    case 'farm': return 'success'
    case 'custom': return 'secondary'
    default: return 'primary'
  }
}

function pickPreset(p: HierarchyPreset) {
  selectedPreset.value = p.value
  if (p.value === 'custom') {
    // Start with 2 empty levels (min).
    levelNames.value = ['', '']
  } else {
    levelNames.value = [...p.levels]
    // Also seed root name to first level for the summary.
    rootName.value = p.levels[0] || ''
  }
}

// ── Step 2 — level names ─────────────────────────────────────────────────

function validNameRule(v: string) {
  if (!v) return 'Required'
  if (!NAME_REGEX.test(v)) return 'Letters, digits, _ or - only'
  if (v.length > 64) return 'Max 64 characters'
  return true
}

const levelsValid = computed(() => {
  if (levelNames.value.length < 2) return false
  if (levelNames.value.length > 6) return false
  for (const n of levelNames.value) {
    if (typeof n !== 'string' || !NAME_REGEX.test(n)) return false
    if (n.length > 64) return false
  }
  return true
})

function canRemoveLevel(i: number) {
  if (levelNames.value.length <= 2) return false
  // Never remove first or last — only middles.
  if (i === 0 || i === levelNames.value.length - 1) return false
  return true
}

function addLevel() {
  if (levelNames.value.length >= 6) return
  levelNames.value.push('')
}
function removeLevel(i: number) {
  if (!canRemoveLevel(i)) return
  levelNames.value.splice(i, 1)
}

function onLevelNameChange() {
  // Keep rootName synced to the first level unless the user has customised.
  if (levelNames.value[0]) rootName.value = levelNames.value[0]
}

const topicPattern = computed(() => {
  if (!levelNames.value.length) return '{root}'
  return levelNames.value.map(n => `{${n || '?'}}`).join('/')
})

const topicExample = computed(() => {
  if (!levelNames.value.length) return ''
  const first = levelNames.value[0] || 'factory'
  const rest = levelNames.value.slice(1)
  const examples: string[] = [first]
  rest.forEach((_, i) => {
    if (i === rest.length - 1) examples.push('temperature')
    else if (i === rest.length - 2) examples.push('machine_1')
    else examples.push(`${_ || 'segment'}_A`)
  })
  return examples.join('/')
})

// ── Step 3 — tree building ───────────────────────────────────────────────

const maxDepth = computed(() => Math.max(0, levelNames.value.length - 1))

function newLocalNode(partial: Partial<LocalNode>): LocalNode {
  const id = idCounter--
  return {
    id,
    parent_id: null,
    level: 0,
    name: '',
    topic_path: '',
    display_name: '',
    description: '',
    location_tag: '',
    status: 'active',
    children: [],
    sensor_meta: undefined,
    ...partial,
  } as LocalNode
}

function findNode(root: LocalNode, id: number): LocalNode | null {
  if (root.id === id) return root
  for (const c of root.children) {
    const hit = findNode(c, id)
    if (hit) return hit
  }
  return null
}

function findParent(root: LocalNode, id: number, parent: LocalNode | null = null): LocalNode | null {
  if (root.id === id) return parent
  for (const c of root.children) {
    const hit = findParent(c, id, root)
    if (hit !== null) return hit
    if (root.children.some(x => x.id === id)) return root
  }
  return null
}

function computeTopicPath(node: LocalNode): string {
  if (!rootNode.value) return node.name
  const parts: string[] = []
  let cur: LocalNode | null = node
  while (cur) {
    parts.unshift(cur.name || '?')
    if (cur.id === rootNode.value.id) break
    cur = findParent(rootNode.value, cur.id)
  }
  return parts.join('/')
}

function selectNode(node: AssetNode) {
  selectedNodeId.value = node.id
}

function addRootNode() {
  const name = rootName.value || levelNames.value[0] || 'root'
  rootNode.value = newLocalNode({
    name,
    display_name: '',
    level: 0,
    topic_path: name,
  })
  selectedNodeId.value = rootNode.value.id
}

function onAddChild(parent: AssetNode) {
  if (!rootNode.value) return
  const p = findNode(rootNode.value, parent.id)
  if (!p) return
  if (p.level >= maxDepth.value) return
  const child = newLocalNode({
    parent_id: p.id,
    level: p.level + 1,
    name: `new_${levelNames.value[p.level + 1] || 'node'}`,
  })
  p.children.push(child)
  selectedNodeId.value = child.id
}

function onDeleteLocalNode(node: AssetNode) {
  if (!rootNode.value) return
  if (node.id === rootNode.value.id) {
    rootNode.value = null
    selectedNodeId.value = null
    return
  }
  const parent = findParent(rootNode.value, node.id)
  if (!parent) return
  const idx = parent.children.findIndex(c => c.id === node.id)
  if (idx >= 0) parent.children.splice(idx, 1)
  if (selectedNodeId.value === node.id) selectedNodeId.value = null
}

function onMoveNode(payload: { sourceId: number; targetParentId: number }) {
  if (!rootNode.value) return
  const source = findNode(rootNode.value, payload.sourceId)
  const targetParent = findNode(rootNode.value, payload.targetParentId)
  if (!source || !targetParent) return
  const currentParent = findParent(rootNode.value, source.id)
  if (!currentParent) return
  if (currentParent.id === targetParent.id) return
  const idx = currentParent.children.findIndex(c => c.id === source.id)
  if (idx < 0) return
  currentParent.children.splice(idx, 1)
  source.parent_id = targetParent.id
  targetParent.children.push(source)
}

function isMachineLevel(n: LocalNode | null | undefined): boolean {
  if (!n) return false
  return n.level === Math.max(0, maxDepth.value - 1)
}

function isSensorLevel(n: LocalNode | null | undefined): boolean {
  if (!n) return false
  return n.level === maxDepth.value
}

function applyTemplate(tpl: SensorTemplate) {
  if (!selectedNode.value) return
  const machine = selectedNode.value
  machine.children = tpl.sensors.map(s => {
    return newLocalNode({
      parent_id: machine.id,
      level: machine.level + 1,
      name: s.name,
      sensor_meta: reactive<SensorMeta>({
        unit: s.unit ?? null,
        sample_rate_hz: s.sample_rate_hz ?? null,
        data_type: s.data_type ?? 'float',
        expected_min: null,
        expected_max: null,
      }),
    })
  })
  notificationStore.showSuccess(`Applied "${tpl.label}"`)
}

// Copy-from-another-machine
const showCopyMachineDialog = ref(false)
const copySourceMachineId = ref<number | null>(null)
const machineNodesForPicker = computed(() => {
  if (!rootNode.value) return []
  const out: Array<{ id: number; topic_path: string }> = []
  const walk = (n: LocalNode) => {
    if (isMachineLevel(n) && n.id !== selectedNode.value?.id) {
      out.push({ id: n.id, topic_path: computeTopicPath(n) })
    }
    for (const c of n.children) walk(c)
  }
  walk(rootNode.value)
  return out
})

function copyMachineSensors() {
  if (!rootNode.value || !selectedNode.value || copySourceMachineId.value == null) return
  const src = findNode(rootNode.value, copySourceMachineId.value)
  if (!src) return
  const targetMachine = selectedNode.value
  targetMachine.children = src.children.map(child => newLocalNode({
    parent_id: targetMachine.id,
    level: targetMachine.level + 1,
    name: child.name,
    display_name: child.display_name,
    description: child.description,
    location_tag: child.location_tag,
    sensor_meta: child.sensor_meta ? reactive({ ...child.sensor_meta }) : undefined,
  }))
  showCopyMachineDialog.value = false
  copySourceMachineId.value = null
  notificationStore.showSuccess(`Copied ${targetMachine.children.length} sensors`)
}

// ── Import dialog ────────────────────────────────────────────────────────

const showImportDialog = ref(false)
const importText = ref('')
const importing = ref(false)
const importError = ref<string | null>(null)

async function doImport() {
  importError.value = null
  let spec: unknown
  try {
    spec = JSON.parse(importText.value)
  } catch {
    importError.value = 'Invalid JSON'
    return
  }

  // For import to work backend needs the config row present. Save config
  // first if not already, then POST /import, then refetch the tree.
  importing.value = true
  try {
    await saveConfigOnly()
    const r = await api.post('/api/asset-tree/import', { spec })
    // Load the freshly-imported tree and replace local state
    const tree = await api.get('/api/asset-tree/nodes')
    const roots = tree.data?.tree || []
    if (roots.length) rootNode.value = adoptRemoteTree(roots[0])
    notificationStore.showSuccess(`Imported ${r.data?.count ?? 0} nodes`)
    showImportDialog.value = false
    importText.value = ''
  } catch (e: any) {
    importError.value = e.response?.data?.error || e.message || 'Import failed'
  } finally {
    importing.value = false
  }
}

function adoptRemoteTree(node: any): LocalNode {
  const out = newLocalNode({
    id: node.id,
    parent_id: node.parent_id ?? null,
    level: node.level,
    name: node.name,
    topic_path: node.topic_path,
    display_name: node.display_name || '',
    description: node.description || '',
    location_tag: node.location_tag || '',
    status: node.status || 'active',
    sensor_meta: node.sensor_meta || undefined,
  })
  out.children = (node.children || []).map((c: any) => {
    const child = adoptRemoteTree(c)
    child.parent_id = out.id
    return child
  })
  return out
}

// ── MQTT test widget ─────────────────────────────────────────────────────

const testTopic = ref('')
const testResult = ref<any | null>(null)
const testing = ref(false)
let testDebounceTimer: number | undefined

watch(testTopic, (v) => {
  if (testDebounceTimer) window.clearTimeout(testDebounceTimer)
  if (!v || !v.trim()) {
    testResult.value = null
    return
  }
  testing.value = true
  testDebounceTimer = window.setTimeout(async () => {
    try {
      const r = await api.get(`/api/asset-tree/topics/test`, { params: { topic: v.trim() } })
      testResult.value = r.data
    } catch (e: any) {
      testResult.value = {
        valid: false,
        reason: e.response?.data?.error || 'test failed',
        route_to: null,
      }
    } finally {
      testing.value = false
    }
  }, 300)
})

// ── Step navigation ──────────────────────────────────────────────────────

function goStep(n: number) {
  step.value = n
}

function skipToConfirm() {
  step.value = 5
}

// ── Confirm / save ───────────────────────────────────────────────────────

const saving = ref(false)
const saveError = ref<string | null>(null)

const metaPrefixesArray = computed(() =>
  metaPrefixesText.value
    .split(',')
    .map(s => s.trim())
    .filter(Boolean),
)

const nodeCountByLevel = computed(() => {
  const counts: number[] = Array(levelNames.value.length).fill(0)
  const walk = (n: LocalNode) => {
    counts[n.level] = (counts[n.level] || 0) + 1
    for (const c of n.children) walk(c)
  }
  if (rootNode.value) walk(rootNode.value)
  return counts
})

const nodeCountByLevelText = computed(() => {
  if (!rootNode.value) return 'Empty tree (build later under Settings)'
  return nodeCountByLevel.value
    .map((c, i) => `${levelNames.value[i]}: ${c}`)
    .join(' • ')
})

async function saveConfigOnly() {
  const payload = {
    level_names: levelNames.value,
    root_name: rootName.value || levelNames.value[0] || 'root',
    topic_mode: topicMode.value,
    meta_prefixes: metaPrefixesArray.value,
  }
  await assetTreeStore.saveConfig(payload)
}

async function persistNodesRecursively(node: LocalNode, parentServerId: number | null): Promise<number> {
  // If node already has a positive id (adopted from a prior import), skip creation for it,
  // but still walk children.
  let serverId: number
  if (node.id > 0) {
    serverId = node.id
  } else {
    const body: any = {
      parent_id: parentServerId,
      name: node.name,
    }
    if (node.display_name) body.display_name = node.display_name
    if (node.description) body.description = node.description
    if (node.location_tag) body.location_tag = node.location_tag
    if (node.sensor_meta) {
      const sm = node.sensor_meta
      body.sensor_meta = {
        unit: sm.unit ?? null,
        sample_rate_hz: sm.sample_rate_hz ?? null,
        expected_min: sm.expected_min ?? null,
        expected_max: sm.expected_max ?? null,
        data_type: sm.data_type ?? null,
      }
    }
    const r = await api.post('/api/asset-tree/nodes', body)
    serverId = r.data.id
    node.id = serverId
  }
  for (const c of node.children) {
    await persistNodesRecursively(c, serverId)
  }
  return serverId
}

async function finish() {
  saveError.value = null
  saving.value = true
  try {
    await saveConfigOnly()
    if (rootNode.value) {
      await persistNodesRecursively(rootNode.value, null)
    }
    assetTreeStore.invalidateConfig()
    await assetTreeStore.ensureConfigChecked(true)
    notificationStore.showSuccess('Asset tree saved. Welcome!')
    router.push({ name: 'dashboard' })
  } catch (e: any) {
    saveError.value = e.response?.data?.error || e.message || 'Save failed'
  } finally {
    saving.value = false
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

async function copyText(t: string) {
  try {
    await navigator.clipboard.writeText(t)
    notificationStore.showInfo('Topic copied')
  } catch {
    notificationStore.showError('Copy failed')
  }
}

async function onLogout() {
  await authStore.logout()
  router.push({ name: 'login' })
}
</script>

<style scoped>
.setup-container {
  min-height: calc(100vh - 64px);
  background: rgba(var(--v-theme-background), 1);
}
.setup-inner {
  max-width: 1200px;
}

.setup-stepper {
  background: transparent;
}

.preset-card {
  cursor: pointer;
  border: 2px solid transparent;
  transition: transform 120ms ease, border-color 120ms ease;
}
.preset-card:hover {
  transform: translateY(-2px);
  border-color: rgba(var(--v-theme-primary), 0.35);
}
.preset-selected {
  border-color: rgb(var(--v-theme-primary));
  background: rgba(var(--v-theme-primary), 0.08);
}
.preset-custom {
  border-style: dashed;
  border-color: rgba(var(--v-theme-on-surface), 0.25);
}
.preset-preview {
  text-align: left;
  min-height: 4.5rem;
}

.topic-code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  color: rgb(var(--v-theme-primary));
  word-break: break-all;
}

.topic-preview code {
  display: block;
}

.remove-spacer {
  width: 32px;
  display: inline-block;
}

.tree-editor-grid {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.4fr);
  gap: 16px;
  align-items: start;
}
.tree-editor-left,
.tree-editor-right {
  background: rgba(var(--v-theme-surface-bright), 0.5);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 8px;
  padding: 12px;
  min-height: 380px;
}
.empty-tree,
.empty-detail {
  padding: 24px 8px;
  text-align: center;
}

.summary-row {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px dashed rgba(var(--v-theme-on-surface), 0.08);
}
.summary-row:last-child {
  border-bottom: none;
}
.summary-label {
  color: rgba(var(--v-theme-on-surface), 0.7);
  font-size: 0.85rem;
}

.test-result {
  display: flex;
  align-items: center;
}

@media (max-width: 700px) {
  .tree-editor-grid {
    grid-template-columns: 1fr;
  }
}
</style>
