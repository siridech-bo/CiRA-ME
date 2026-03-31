<template>
  <div class="editor-root">

    <!-- Top Bar -->
    <header class="editor-topbar">
      <div class="d-flex align-center gap-3">
        <v-btn icon size="small" variant="text" @click="router.push('/app-builder')" title="Back to App Builder">
          <v-icon size="18">mdi-arrow-left</v-icon>
        </v-btn>
        <div class="topbar-brand d-flex align-center gap-2">
          <v-icon size="16" color="purple">mdi-hexagon-outline</v-icon>
          <span class="topbar-label-cira">CiRA</span>
          <span class="topbar-label-ab">APP BUILDER</span>
        </div>
        <span class="topbar-sep">/</span>
        <v-text-field
          v-model="appName"
          variant="plain"
          density="compact"
          hide-details
          class="app-name-field"
          style="max-width: 200px;"
        />
      </div>

      <div class="topbar-tabs d-flex">
        <button
          v-for="tab in ['BUILD','PREVIEW','PUBLISH']"
          :key="tab"
          class="topbar-tab"
          :class="{ active: activeTab === tab }"
          @click="activeTab = tab"
        >{{ tab }}</button>
      </div>

      <div class="d-flex align-center gap-3">
        <span v-if="validationErrors.length > 0" class="status-error">
          <v-icon size="14" color="error">mdi-alert</v-icon>
          {{ validationErrors.length }} error{{ validationErrors.length > 1 ? 's' : '' }}
        </span>
        <span v-else class="status-ok">
          <v-icon size="14" color="success">mdi-check</v-icon>
          valid
        </span>
        <span class="status-meta">{{ nodes.length }} nodes</span>
        <v-btn
          size="small"
          color="primary"
          variant="tonal"
          :loading="saving"
          @click="saveApp"
        >
          <v-icon start size="14">mdi-content-save</v-icon>
          Save
        </v-btn>
      </div>
    </header>

    <!-- BUILD Tab -->
    <div v-if="activeTab === 'BUILD'" class="editor-body">

      <!-- Left: Node Palette -->
      <aside class="palette-panel">
        <div v-for="group in paletteGroups" :key="group.category" class="palette-group">
          <div class="palette-group-header">{{ group.category }}</div>

          <!-- Model nodes grouped by mode -->
          <template v-if="group.category === 'Model'">
            <div v-for="(items, mode) in group.byMode" :key="mode" class="palette-mode-group">
              <div class="palette-mode-label" :style="{ color: MODE_META[mode].color + 'bb' }">
                {{ mode }}
              </div>
              <div
                v-for="item in items"
                :key="item.type"
                class="palette-item"
                @click="addNode(item.type)"
                :title="`Add ${item.label}`"
              >
                <v-icon size="16" :style="{ color: MODE_META[mode].color }">{{ item.icon }}</v-icon>
                <div class="palette-item-info">
                  <div class="palette-item-label">{{ item.label }}</div>
                  <div class="palette-item-sub">{{ item.feature_count }}f · {{ item.algorithm }}</div>
                </div>
                <v-chip
                  size="x-small"
                  variant="tonal"
                  :style="{ background: MODE_META[mode].color + '18', color: MODE_META[mode].color, borderColor: MODE_META[mode].color + '44' }"
                  class="mode-badge"
                >
                  {{ mode.slice(0,3).toUpperCase() }}
                </v-chip>
              </div>
            </div>
            <div v-if="!Object.keys(group.byMode).length" class="palette-empty">
              No active endpoints
            </div>
          </template>

          <!-- Static nodes -->
          <template v-else>
            <div
              v-for="item in group.items"
              :key="item.type"
              class="palette-item"
              @click="addNode(item.type)"
              :title="`Add ${item.label}`"
            >
              <v-icon size="16" :style="{ color: item.color }">{{ item.icon }}</v-icon>
              <span class="palette-item-label">{{ item.label }}</span>
            </div>
          </template>
        </div>

        <div v-if="inactiveEndpointCount > 0" class="palette-footer">
          {{ inactiveEndpointCount }} inactive endpoint{{ inactiveEndpointCount > 1 ? 's' : '' }} hidden
        </div>
      </aside>

      <!-- Center: Pipeline Canvas -->
      <main class="canvas-panel" ref="canvasRef"
            @click="selectedId = null"
            @mousedown.self="startPan"
            @mousemove="doPan"
            @mouseup="endPan"
            @mouseleave="endPan">
        <div v-if="nodes.length === 0" class="canvas-empty">
          <v-icon size="40" color="grey">mdi-view-grid-plus-outline</v-icon>
          <div class="text-caption text-medium-emphasis mt-2">Click a node from the palette to add it</div>
        </div>

        <div v-else class="canvas-chain" @click.stop>
          <template v-for="(node, index) in nodes" :key="node.id">
            <!-- Node Card -->
            <div
              class="node-card"
              :class="{
                selected: selectedId === node.id,
                'has-error': nodeHasError(node.id),
              }"
              @click.stop="selectedId = node.id"
            >
              <!-- Delete button -->
              <button class="node-delete" @click.stop="removeNode(node.id)" title="Remove node">
                <v-icon size="12">mdi-close</v-icon>
              </button>

              <!-- Mode badge for model nodes -->
              <div
                v-if="isModelNode(node.type)"
                class="node-mode-badge"
                :style="{
                  background: MODE_META[capabilities[node.type]?.mode]?.color + '15',
                  color: MODE_META[capabilities[node.type]?.mode]?.color,
                  borderColor: MODE_META[capabilities[node.type]?.mode]?.color + '40',
                }"
              >
                {{ MODE_META[capabilities[node.type]?.mode]?.label }}
              </div>

              <!-- Icon + label -->
              <div class="node-header">
                <v-icon size="16" :style="{ color: capabilities[node.type]?.color }">
                  {{ capabilities[node.type]?.icon }}
                </v-icon>
                <span class="node-label">{{ capabilities[node.type]?.label }}</span>
              </div>

              <!-- Algorithm sub-label -->
              <div v-if="isModelNode(node.type)" class="node-algorithm">
                {{ capabilities[node.type]?.algorithm }}
              </div>

              <!-- Feature count badge -->
              <div v-if="isModelNode(node.type)" class="node-footer">
                <span class="node-feat-badge">{{ capabilities[node.type]?.feature_count }}f</span>
                <v-icon v-if="nodeHasError(node.id)" size="12" color="error">mdi-alert</v-icon>
              </div>

              <!-- Output type -->
              <div class="node-output-type">{{ capabilities[node.type]?.outputs?.[0] }}</div>

              <!-- Selected indicator bar -->
              <div v-if="selectedId === node.id" class="node-selected-bar" />
            </div>

            <!-- Arrow connector -->
            <div v-if="index < nodes.length - 1" class="node-arrow">
              <div class="arrow-line" />
              <svg width="5" height="8" viewBox="0 0 5 8" class="arrow-head">
                <polygon points="0,0 5,4 0,8" fill="#444c56" />
              </svg>
            </div>
          </template>

          <!-- Drop zone -->
          <div class="canvas-dropzone">
            <v-icon size="18" color="grey">mdi-plus</v-icon>
          </div>
        </div>
      </main>

      <!-- Right: Config Panel -->
      <aside class="config-panel">
        <div class="config-panel-header">
          {{ selectedNode ? 'Node Config' : 'Config Panel' }}
        </div>

        <!-- Empty state -->
        <div v-if="!selectedNode" class="config-empty">
          <v-icon size="32" color="grey">mdi-cursor-default-click-outline</v-icon>
          <div class="text-caption text-medium-emphasis mt-2">Click a node to configure</div>
        </div>

        <!-- Node config -->
        <div v-else class="config-body">
          <!-- Node header -->
          <div class="config-node-header">
            <div class="d-flex align-center gap-2">
              <v-icon size="16" :style="{ color: selectedCap?.color }">{{ selectedCap?.icon }}</v-icon>
              <span class="config-node-label">{{ selectedCap?.label }}</span>
            </div>
            <div class="config-node-type">{{ selectedNode.type }}</div>

            <!-- Model info block -->
            <div v-if="isModelNode(selectedNode.type)" class="config-model-block">
              <!-- Endpoint switcher -->
              <div class="mb-3">
                <div class="config-model-key mb-1">Switch Endpoint</div>
                <v-select
                  :model-value="selectedCap?.endpoint_id"
                  @update:model-value="switchEndpoint"
                  :items="availableEndpoints"
                  item-title="label"
                  item-value="id"
                  variant="outlined"
                  density="compact"
                  hide-details
                  style="font-size: 11px"
                />
              </div>
              <div class="config-model-row">
                <span class="config-model-key">Mode</span>
                <span class="config-model-val font-weight-bold" :style="{ color: MODE_META[selectedCap.mode]?.color }">
                  {{ selectedCap.mode?.toUpperCase() }}
                </span>
              </div>
              <div class="config-model-row">
                <span class="config-model-key">Algorithm</span>
                <span class="config-model-val">{{ selectedCap.algorithm }}</span>
              </div>
              <div class="config-model-row">
                <span class="config-model-key">Required Features</span>
                <span class="config-model-val">{{ selectedCap.feature_count }}</span>
              </div>
              <div class="pt-2 border-t" style="border-color: #21262d;">
                <div class="config-model-key mb-1">Expects</div>
                <div class="d-flex flex-wrap" style="gap: 3px;">
                  <span
                    v-for="f in selectedCap.feature_names"
                    :key="f"
                    class="feature-tag"
                  >{{ f }}</span>
                </div>
              </div>

              <v-btn
                block
                size="x-small"
                variant="outlined"
                color="purple"
                class="mt-3"
                @click="autoConfigureFeatures"
              >
                <v-icon start size="12">mdi-auto-fix</v-icon>
                Auto-configure Feature Extract
              </v-btn>
            </div>
          </div>

          <!-- Validation error -->
          <div v-if="selectedNodeError" class="config-error-block">
            <v-icon size="12" color="error" class="mr-1">mdi-alert</v-icon>
            {{ selectedNodeError.msg }}
          </div>

          <!-- I/O type badges -->
          <div class="config-io-row">
            <span v-for="t in (selectedCap?.inputs || [])" :key="'in-'+t" class="io-badge io-badge-in">
              &larr; {{ t }}
            </span>
            <span
              v-for="t in (selectedCap?.outputs || [])"
              :key="'out-'+t"
              class="io-badge io-badge-out"
              :style="{ background: selectedCap.color + '10', borderColor: selectedCap.color + '40', color: selectedCap.color }"
            >
              &rarr; {{ t }}
            </span>
          </div>

          <!-- Config fields -->
          <div class="config-fields">
            <div v-if="!selectedCap?.configSchema?.length" class="text-caption text-medium-emphasis">
              No configuration needed.
            </div>

            <div v-for="field in selectedCap?.configSchema" :key="field.key" class="config-field">
              <div class="d-flex align-center justify-space-between mb-1">
                <label class="config-field-label">{{ field.label }}</label>
                <!-- toggle inline -->
                <v-switch
                  v-if="field.type === 'toggle'"
                  :model-value="getConfigVal(selectedNode, field)"
                  @update:model-value="v => updateConfig(selectedNode.id, field.key, v)"
                  density="compact"
                  hide-details
                  color="purple"
                  class="toggle-inline"
                />
              </div>

              <!-- MQTT topic field with discover button -->
              <div v-if="field.type === 'text' && field.key === 'topic' && selectedNode?.type === 'input.live_stream'" class="mb-2">
                <div class="d-flex align-center gap-1">
                  <v-text-field
                    :model-value="getConfigVal(selectedNode, field)"
                    @update:model-value="v => updateConfig(selectedNode.id, field.key, v)"
                    variant="outlined"
                    density="compact"
                    hide-details
                    class="config-input flex-grow-1"
                  />
                  <v-btn size="small" variant="tonal" color="info" :loading="discoveringTopics" @click="discoverTopics">
                    <v-icon size="small">mdi-magnify</v-icon>
                  </v-btn>
                </div>
                <div v-if="discoveredTopics.length > 0" class="mt-1 d-flex flex-wrap" style="gap: 3px;">
                  <button
                    v-for="t in discoveredTopics"
                    :key="t"
                    class="discovered-topic-chip"
                    :class="{ active: getConfigVal(selectedNode, field) === t }"
                    @click="updateConfig(selectedNode.id, field.key, t)"
                  >{{ t }}</button>
                </div>
              </div>

              <!-- text (generic) -->
              <v-text-field
                v-else-if="field.type === 'text'"
                :model-value="getConfigVal(selectedNode, field)"
                @update:model-value="v => updateConfig(selectedNode.id, field.key, v)"
                variant="outlined"
                density="compact"
                :hide-details="!(field.key === 'target_column' && targetColumnHint)"
                :hint="field.key === 'target_column' ? targetColumnHint : ''"
                :persistent-hint="field.key === 'target_column'"
                class="config-input"
              />

              <!-- number -->
              <v-text-field
                v-else-if="field.type === 'number'"
                :model-value="getConfigVal(selectedNode, field)"
                @update:model-value="v => updateConfig(selectedNode.id, field.key, parseFloat(v) || 0)"
                type="number"
                variant="outlined"
                density="compact"
                hide-details
                class="config-input"
              />

              <!-- select -->
              <v-select
                v-else-if="field.type === 'select'"
                :model-value="getConfigVal(selectedNode, field)"
                @update:model-value="v => updateConfig(selectedNode.id, field.key, v)"
                :items="field.options"
                variant="outlined"
                density="compact"
                hide-details
                class="config-input"
              />

              <!-- slider -->
              <div v-else-if="field.type === 'slider'" class="d-flex align-center gap-2">
                <v-slider
                  :model-value="getConfigVal(selectedNode, field)"
                  @update:model-value="v => updateConfig(selectedNode.id, field.key, v)"
                  :min="field.min"
                  :max="field.max"
                  :step="field.step"
                  color="purple"
                  hide-details
                  density="compact"
                  class="flex-grow-1"
                />
                <span class="slider-val">{{ getConfigVal(selectedNode, field) }}</span>
              </div>

              <!-- multiselect chips -->
              <div v-else-if="field.type === 'multiselect'" class="multiselect-chips">
                <button
                  v-for="opt in getMultiselectOptions(selectedNode, field)"
                  :key="opt"
                  class="multiselect-chip"
                  :class="{ active: (getConfigVal(selectedNode, field) || []).includes(opt) }"
                  @click="toggleMultiselect(selectedNode.id, field.key, opt, getConfigVal(selectedNode, field))"
                >{{ opt.includes(':') ? opt.split(':').slice(1).join(':') : opt }}</button>
                <div class="multiselect-count">
                  {{ (getConfigVal(selectedNode, field) || []).length }} selected
                </div>

                <!-- Auto-configure button for multi-model endpoint selection -->
                <v-btn
                  v-if="selectedNode?.type === 'output.multi_model_compare' && field.key === 'endpoint_ids' && (getConfigVal(selectedNode, field) || []).length > 0"
                  block
                  size="x-small"
                  variant="outlined"
                  color="purple"
                  class="mt-2"
                  @click="autoConfigureFromMultiModel"
                >
                  <v-icon start size="12">mdi-auto-fix</v-icon>
                  Auto-configure Feature Extract
                </v-btn>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>

    <!-- PREVIEW Tab -->
    <div v-else-if="activeTab === 'PREVIEW'" class="editor-body">

      <!-- Pipeline steps -->
      <aside class="preview-steps-panel">
        <div class="panel-section-header">Pipeline Steps</div>
        <div v-for="(node, index) in nodes" :key="node.id" class="preview-step">
          <div class="preview-step-indicator">
            <div
              class="step-dot"
              :class="{ 'step-dot-error': nodeHasError(node.id) }"
              :style="!nodeHasError(node.id) ? { color: capabilities[node.type]?.color } : {}"
            >
              <v-icon v-if="nodeHasError(node.id)" size="10" color="error">mdi-alert</v-icon>
              <v-icon v-else size="12" :style="{ color: capabilities[node.type]?.color }">
                {{ capabilities[node.type]?.icon }}
              </v-icon>
            </div>
            <div v-if="index < nodes.length - 1" class="step-line" />
          </div>
          <div class="preview-step-info">
            <div class="preview-step-label">{{ capabilities[node.type]?.label }}</div>
            <div v-if="isModelNode(node.type)" class="preview-step-mode" :style="{ color: MODE_META[capabilities[node.type]?.mode]?.color }">
              {{ capabilities[node.type]?.mode?.toUpperCase() }} · {{ capabilities[node.type]?.feature_count }}f
            </div>
            <div v-if="nodeHasError(node.id)" class="preview-step-error">
              {{ getNodeError(node.id)?.msg }}
            </div>
            <div v-else-if="!isModelNode(node.type)" class="preview-step-config">
              {{ firstConfigSummary(node) }}
            </div>
          </div>
        </div>
      </aside>

      <!-- Output preview -->
      <main class="preview-main">
        <div class="output-preview-wrap">
          <div v-if="!previewOutputNode || !previewModelNode" class="output-preview-empty">
            <v-icon size="32" color="grey">mdi-eye-off-outline</v-icon>
            <div class="text-caption text-medium-emphasis mt-2">Add a Model endpoint + Output node to preview</div>
          </div>

          <div v-else>
            <!-- Browser chrome -->
            <div class="browser-bar">
              <div class="browser-dots">
                <div class="browser-dot" /><div class="browser-dot" /><div class="browser-dot" />
              </div>
              <span class="browser-url">localhost:3030/apps/{{ previewSlug }}</span>
              <span v-if="previewMeta" class="browser-mode-badge" :style="{ color: previewMeta.color, background: previewMeta.color + '18' }">
                {{ previewMeta.label }}
              </span>
            </div>

            <div class="browser-body">
              <div class="d-flex align-center justify-space-between mb-4">
                <span class="browser-app-title">{{ appName }}</span>
                <span class="browser-model-info">{{ previewModelCap?.label }} · {{ previewModelCap?.algorithm }}</span>
              </div>

              <!-- Input widget -->
              <div v-if="previewInputNode" class="browser-input-widget">
                <div class="browser-widget-label">
                  <v-icon size="12" :style="{ color: capabilities[previewInputNode.type]?.color }">{{ capabilities[previewInputNode.type]?.icon }}</v-icon>
                  {{ capabilities[previewInputNode.type]?.label }}
                </div>
                <div v-if="previewInputNode.type === 'input.csv_upload'" class="csv-upload-row">
                  <div class="csv-drop-zone">drop .csv · {{ previewInputNode.config.value_cols || 'value' }}</div>
                  <div class="run-btn" :style="{ background: previewMeta?.color }">Run</div>
                </div>
                <div v-else class="live-stream-row">
                  <div class="live-dot" :style="{ background: previewMeta?.color }" />
                  <span class="live-label">live · {{ previewInputNode.config.interval_ms }}ms</span>
                </div>
              </div>

              <!-- Line chart output -->
              <div v-if="previewOutputNode?.type === 'output.line_chart'">
                <div class="chart-title">{{ previewOutputNode.config.title || 'Results' }}</div>
                <svg :width="chartW" :height="chartH" style="width:100%;" viewBox="0 0 360 110">
                  <defs>
                    <linearGradient id="preview-grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" :stop-color="previewMeta?.color" stop-opacity="0.25"/>
                      <stop offset="100%" :stop-color="previewMeta?.color" stop-opacity="0"/>
                    </linearGradient>
                  </defs>
                  <path :d="chartAreaD" fill="url(#preview-grad)"/>
                  <path :d="chartPathD" fill="none" :stroke="previewMeta?.color" stroke-width="1.5"/>
                  <template v-if="previewModelCap?.mode === 'anomaly' && previewOutputNode.config.show_anomalies">
                    <g v-for="i in chartAIdx" :key="i">
                      <line :x1="chartSx(i)" y1="0" :x2="chartSx(i)" :y2="chartH" stroke="#f87171" stroke-width="1" stroke-dasharray="3,2" opacity="0.35"/>
                      <circle :cx="chartSx(i)" :cy="chartSy(chartPts[i])" r="4" fill="#f87171"/>
                    </g>
                  </template>
                  <path v-if="previewModelCap?.mode === 'regression'"
                    :d="'M' + chartSx(32) + ',' + chartSy(52) + ' L' + chartSx(39) + ',' + chartSy(58)"
                    :stroke="previewMeta?.color" stroke-width="2" stroke-dasharray="4,3" opacity="0.5"/>
                </svg>
                <div class="chart-legend">
                  <div class="legend-item">
                    <div class="legend-line" :style="{ background: previewMeta?.color }"/>
                    <span>signal</span>
                  </div>
                  <div v-if="previewModelCap?.mode === 'regression'" class="legend-item">
                    <div class="legend-line" :style="{ background: previewMeta?.color, opacity: 0.4 }"/>
                    <span>forecast</span>
                  </div>
                </div>
              </div>

              <!-- Alert badge output -->
              <div v-else-if="previewOutputNode?.type === 'output.alert_badge'" class="d-flex gap-3 mt-4">
                <div class="alert-badge-item" style="border-color: #34d399; color: #34d399;">
                  <v-icon size="18" color="#34d399">mdi-check-circle</v-icon>
                  {{ previewOutputNode.config.label_normal || 'Normal' }}
                </div>
                <div class="alert-badge-item" style="border-color: #f87171; color: #f87171; opacity: 0.4;">
                  <v-icon size="18" color="#f87171">mdi-alert-circle</v-icon>
                  {{ previewOutputNode.config.label_anomaly || 'Anomaly' }}
                </div>
              </div>

              <!-- Table output -->
              <div v-else-if="previewOutputNode?.type === 'output.table'" class="mt-4">
                <table class="preview-table">
                  <thead><tr><th>#</th><th>Input</th><th>Prediction</th><th v-if="previewOutputNode.config.show_confidence">Conf.</th></tr></thead>
                  <tbody>
                    <tr v-for="r in 3" :key="r">
                      <td>{{ r }}</td>
                      <td style="color:#94a3b8">[...]</td>
                      <td :style="{ color: previewMeta?.color }">{{ previewModelCap?.mode === 'regression' ? (50 + r * 2.3).toFixed(1) : 'Class_' + r }}</td>
                      <td v-if="previewOutputNode.config.show_confidence" style="color:#94a3b8">{{ (0.95 - r * 0.05).toFixed(2) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>

    <!-- PUBLISH Tab -->
    <div v-else-if="activeTab === 'PUBLISH'" class="editor-body publish-body">
      <div class="publish-container">
        <div class="panel-section-header mb-4">Publish App</div>

        <!-- Validation checklist -->
        <v-card variant="outlined" class="publish-card mb-4">
          <div class="publish-card-header">Validation</div>
          <div v-for="node in nodes" :key="node.id" class="publish-check-row">
            <div class="d-flex align-center gap-2">
              <v-icon size="14" :style="{ color: capabilities[node.type]?.color }">
                {{ capabilities[node.type]?.icon }}
              </v-icon>
              <span class="publish-check-label">{{ capabilities[node.type]?.label }}</span>
            </div>
            <div v-if="getNodeError(node.id)" class="publish-check-error">
              <v-icon size="12" color="error">mdi-alert</v-icon>
              {{ getNodeError(node.id)?.msg }}
            </div>
            <v-icon v-else size="14" color="success">mdi-check-circle</v-icon>
          </div>
          <div v-if="nodes.length === 0" class="text-caption text-medium-emphasis pa-2">
            No nodes in pipeline yet.
          </div>
        </v-card>

        <!-- App settings -->
        <v-card variant="outlined" class="publish-card mb-4">
          <div class="publish-card-header">App Settings</div>
          <div class="publish-settings">
            <div class="mb-3">
              <label class="config-field-label">App Name</label>
              <v-text-field
                v-model="appName"
                variant="outlined"
                density="compact"
                hide-details
                class="config-input mt-1"
              />
            </div>
            <div class="mb-3">
              <label class="config-field-label">Access Control</label>
              <v-select
                v-model="publishSettings.access"
                :items="['Private · API Key required', 'Team · Authenticated users', 'Public']"
                variant="outlined"
                density="compact"
                hide-details
                class="config-input mt-1"
              />
            </div>
            <div>
              <label class="config-field-label">Rate Limit</label>
              <v-select
                v-model="publishSettings.rateLimit"
                :items="['100 req / day', '1,000 req / day', 'Unlimited']"
                variant="outlined"
                density="compact"
                hide-details
                class="config-input mt-1"
              />
            </div>
          </div>
        </v-card>

        <!-- Publish button -->
        <v-btn
          v-if="!publishResult"
          block
          :color="readyToPublish ? 'purple' : 'grey'"
          :disabled="!readyToPublish"
          :loading="publishing"
          size="large"
          variant="flat"
          class="publish-btn"
          @click="publishApp"
        >
          <v-icon start>mdi-rocket-launch</v-icon>
          {{ readyToPublish ? 'Publish App' : `Fix ${validationErrors.length} error${validationErrors.length > 1 ? 's' : ''} first` }}
        </v-btn>

        <!-- Published result -->
        <v-card v-else variant="outlined" class="publish-result-card">
          <div class="publish-result-header">
            <v-icon color="success" size="16">mdi-check-circle</v-icon>
            Published
          </div>

          <div v-for="item in publishLinks" :key="item.label" class="publish-result-row">
            <div class="publish-result-label">{{ item.label }}</div>
            <div class="publish-result-value-row">
              <code class="publish-result-code" :style="{ color: item.color }">{{ item.value }}</code>
              <v-btn
                icon
                size="x-small"
                variant="text"
                :title="`Copy ${item.label}`"
                @click="copyToClipboard(item.value)"
              >
                <v-icon size="14">mdi-content-copy</v-icon>
              </v-btn>
            </div>
          </div>

          <v-btn
            block
            variant="tonal"
            color="grey"
            size="small"
            class="mt-3"
            @click="publishResult = null"
          >
            Unpublish / Edit
          </v-btn>
        </v-card>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import api from '@/services/api'

// ── Router ──────────────────────────────────────────────────────────
const router = useRouter()
const route  = useRoute()
const appId  = computed(() => route.params.id)

// ── Constants ───────────────────────────────────────────────────────
const MODE_META = {
  regression:     { icon: 'mdi-wave', color: '#a78bfa', label: 'REGRESSION',    output: 'regression_result'     },
  classification: { icon: 'mdi-shape', color: '#34d399', label: 'CLASSIFICATION', output: 'classification_result' },
  anomaly:        { icon: 'mdi-hexagon-outline', color: '#f87171', label: 'ANOMALY', output: 'anomaly_result'    },
}

const MODE_CONFIG_SCHEMA = {
  regression:     [{ key: 'horizon',     label: 'Forecast Horizon', type: 'number', default: 10 }],
  classification: [{ key: 'top_k',       label: 'Top K Results',    type: 'number', default: 1  }],
  anomaly: [
    { key: 'threshold',   label: 'Threshold',   type: 'slider', min: 0.1, max: 1.0, step: 0.05, default: 0.8 },
    { key: 'sensitivity', label: 'Sensitivity', type: 'select', options: ['low','medium','high'], default: 'medium' },
  ],
}

const ALL_FEATURES = ['mean','std','rms','max','min','fft_peak','entropy','skew','kurt','zcr']

const STATIC_CAPS = {
  'input.csv_upload':          { label: 'CSV Upload',      icon: 'mdi-file-delimited',       color: '#60a5fa', category: 'Input',     outputs: ['timeseries'], configSchema: [{ key: 'timestamp_col', label: 'Timestamp Column', type: 'text', default: 'timestamp' }, { key: 'value_cols', label: 'Value Columns', type: 'text', default: 'value' }] },
  'input.live_stream':         { label: 'Live Stream (MQTT)', icon: 'mdi-access-point',       color: '#60a5fa', category: 'Input',     outputs: ['timeseries'], configSchema: [{ key: 'broker_url', label: 'MQTT Broker URL', type: 'text', default: 'ws://localhost:9001/mqtt' }, { key: 'topic', label: 'MQTT Topic', type: 'text', default: 'sensors/machine1/#' }, { key: 'channels', label: 'Channel Names (comma-separated)', type: 'text', default: '' }] },
  'transform.window':          { label: 'Windowing',       icon: 'mdi-view-grid-outline',     color: '#818cf8', category: 'Transform', inputs: ['timeseries'], outputs: ['timeseries'], configSchema: [{ key: 'window_size', label: 'Window Size', type: 'number', default: 32 }, { key: 'step', label: 'Stride', type: 'number', default: 16 }] },
  'transform.fill_missing':    { label: 'Fill Missing',    icon: 'mdi-format-align-justify',  color: '#818cf8', category: 'Transform', inputs: ['timeseries'], outputs: ['timeseries'], configSchema: [{ key: 'method', label: 'Method', type: 'select', options: ['ffill','bfill','interpolate','zero'], default: 'interpolate' }] },
  'transform.normalize':       { label: 'Normalize',       icon: 'mdi-arrow-expand-vertical', color: '#818cf8', category: 'Transform', inputs: ['timeseries'], outputs: ['timeseries'], configSchema: [{ key: 'method', label: 'Method', type: 'select', options: ['minmax','zscore','robust'], default: 'zscore' }] },
  'transform.feature_extract': { label: 'Feature Extract', icon: 'mdi-star-four-points',      color: '#818cf8', category: 'Transform', inputs: ['timeseries'], outputs: ['features'],   configSchema: [{ key: 'features', label: 'Features', type: 'multiselect', options: ALL_FEATURES, default: [] }] },
  'output.line_chart':         { label: 'Line Chart',      icon: 'mdi-chart-line',            color: '#94a3b8', category: 'Output',    inputs: ['timeseries','anomaly_result','regression_result'], configSchema: [{ key: 'title', label: 'Chart Title', type: 'text', default: 'Results' }, { key: 'target_column', label: 'Target Column (ground truth)', type: 'text', default: '' }, { key: 'show_anomalies', label: 'Highlight Anomalies', type: 'toggle', default: true }] },
  'output.alert_badge':        { label: 'Alert Badge',     icon: 'mdi-shield-alert',          color: '#94a3b8', category: 'Output',    inputs: ['anomaly_result'], configSchema: [{ key: 'label_normal', label: 'Normal Label', type: 'text', default: 'Normal' }, { key: 'label_anomaly', label: 'Anomaly Label', type: 'text', default: 'Anomaly Detected' }, { key: 'webhook_url', label: 'Webhook URL', type: 'text', default: '' }] },
  'output.table':              { label: 'Table View',      icon: 'mdi-table',                 color: '#94a3b8', category: 'Output',    inputs: ['classification_result','regression_result'], configSchema: [{ key: 'max_rows', label: 'Max Rows', type: 'number', default: 50 }, { key: 'show_confidence', label: 'Show Confidence', type: 'toggle', default: true }] },
  'output.signal_recorder':    { label: 'Signal Recorder', icon: 'mdi-record-circle',         color: '#ef4444', category: 'Output',    inputs: ['timeseries'], configSchema: [{ key: 'labels', label: 'Label Names (comma-separated)', type: 'text', default: 'idle, wave, snake, updown' }, { key: 'target_sample_rate', label: 'Target Sample Rate (Hz)', type: 'number', default: 62.5 }, { key: 'max_duration', label: 'Max Duration (seconds)', type: 'number', default: 300 }, { key: 'file_prefix', label: 'File Name Prefix', type: 'text', default: 'sensor_data' }] },
  'output.multi_model_compare': { label: 'Multi-Model Compare', icon: 'mdi-compare-horizontal', color: '#f59e0b', category: 'Output', inputs: ['features'], configSchema: [{ key: 'endpoint_ids', label: 'Model Endpoints (select up to 5)', type: 'multiselect', options: [], default: [] }, { key: 'target_column', label: 'Target Column (ground truth)', type: 'text', default: '' }, { key: 'show_chart', label: 'Show Comparison Chart', type: 'toggle', default: true }, { key: 'show_metrics', label: 'Show Metrics Table', type: 'toggle', default: true }] },
}

const PALETTE_ORDER = ['Input', 'Transform', 'Model', 'Output']

// ── State ────────────────────────────────────────────────────────────
const appName       = ref('My App')
const activeTab     = ref('BUILD')
const nodes         = ref([])
const selectedId    = ref(null)
const saving        = ref(false)
const publishing    = ref(false)
const publishResult = ref(null)
const melabEndpoints = ref([])
const canvasRef     = ref(null)
const isPanning     = ref(false)
const panStart      = ref({ x: 0, y: 0, scrollLeft: 0, scrollTop: 0 })

const publishSettings = ref({
  access: 'Private · API Key required',
  rateLimit: '100 req / day',
})

// ── Capabilities (static + ME-LAB model nodes) ────────────────────
const capabilities = computed(() => {
  const caps = { ...STATIC_CAPS }
  melabEndpoints.value
    .filter(e => e.status === 'active')
    .forEach(e => {
      const meta = MODE_META[e.mode]
      if (!meta) return
      caps[`model.endpoint.${e.id}`] = {
        label: e.name,
        icon: meta.icon,
        color: meta.color,
        category: 'Model',
        mode: e.mode,
        algorithm: e.algorithm,
        feature_count: e.n_features,
        feature_names: e.feature_names || [],
        target_column: e.target_column || null,
        endpoint_id: e.id,
        inputs: ['features'],
        outputs: [meta.output],
        configSchema: MODE_CONFIG_SCHEMA[e.mode] || [],
        calls: e.inference_count || 0,
        last_used: e.last_inference_at || null,
      }
    })
  return caps
})

// ── Palette groups ────────────────────────────────────────────────
const paletteGroups = computed(() => {
  return PALETTE_ORDER.map(cat => {
    const entries = Object.entries(capabilities.value).filter(([, v]) => v.category === cat)
    if (cat === 'Model') {
      const byMode = {}
      entries.forEach(([type, v]) => {
        if (!byMode[v.mode]) byMode[v.mode] = []
        byMode[v.mode].push({ type, ...v })
      })
      return { category: cat, byMode }
    }
    return { category: cat, items: entries.map(([type, v]) => ({ type, ...v })) }
  })
})

const inactiveEndpointCount = computed(() =>
  melabEndpoints.value.filter(e => e.status !== 'active').length
)

// ── Derived selections ────────────────────────────────────────────
const selectedNode = computed(() =>
  nodes.value.find(n => n.id === selectedId.value) ?? null
)

const selectedCap = computed(() =>
  selectedNode.value ? capabilities.value[selectedNode.value.type] : null
)

// ── Validation ────────────────────────────────────────────────────
const validationErrors = computed(() => {
  const errors = []
  nodes.value.forEach((node, i) => {
    if (!node.type.startsWith('model.endpoint.')) return
    const cap = capabilities.value[node.type]
    if (!cap) return
    const featNode = nodes.value.slice(0, i).reverse()
      .find(n => n.type === 'transform.feature_extract')
    if (!featNode) {
      errors.push({ nodeId: node.id, msg: `"${cap.label}" needs Feature Extract upstream` })
      return
    }
    const sel = featNode.config.features || []
    if (sel.length !== cap.feature_count) {
      errors.push({ nodeId: node.id, msg: `"${cap.label}" needs ${cap.feature_count} features, got ${sel.length}` })
    } else {
      const missing = cap.feature_names.filter(f => !sel.includes(f))
      if (missing.length) errors.push({ nodeId: node.id, msg: `Missing: ${missing.join(', ')}` })
    }
  })
  return errors
})

const readyToPublish = computed(() =>
  nodes.value.length >= 2 && validationErrors.value.length === 0
)

const selectedNodeError = computed(() =>
  selectedNode.value
    ? validationErrors.value.find(e => e.nodeId === selectedNode.value.id) ?? null
    : null
)

// ── Publish links ─────────────────────────────────────────────────
const appSlug = computed(() => appName.value.toLowerCase().replace(/\s+/g, '-'))

const publishLinks = computed(() => {
  if (!publishResult.value) return []
  const slug = publishResult.value.slug || appSlug.value
  const base = window.location.origin
  return [
    { label: 'Standalone App', value: `${base}/standalone/${slug}`,                   color: '#a78bfa' },
    { label: 'API Endpoint',   value: `${base}/api/app-builder/run/${slug}`,          color: '#34d399' },
    { label: 'Embed',          value: `<iframe src="${base}/standalone/${slug}" />`,   color: '#94a3b8' },
  ]
})

// ── Helpers ──────────────────────────────────────────────────────
function isModelNode(type) {
  return type?.startsWith('model.endpoint.')
}

function nodeHasError(nodeId) {
  return validationErrors.value.some(e => e.nodeId === nodeId)
}

function getNodeError(nodeId) {
  return validationErrors.value.find(e => e.nodeId === nodeId) ?? null
}

function initConfig(cap) {
  return Object.fromEntries((cap?.configSchema || []).map(f => [f.key, f.default]))
}

function makeNode(type, cap) {
  return { id: `n${Date.now()}_${Math.random().toString(36).slice(2)}`, type, config: initConfig(cap) }
}

function getConfigVal(node, field) {
  return node.config[field.key] !== undefined ? node.config[field.key] : field.default
}

// ── Preview computeds ─────────────────────────────────────────────
const previewInputNode  = computed(() => nodes.value.find(n => n.type.startsWith('input.')))
const previewOutputNode = computed(() => nodes.value.find(n => n.type.startsWith('output.')))
const previewModelNode  = computed(() => nodes.value.find(n => n.type.startsWith('model.endpoint.')))
const previewModelCap   = computed(() => previewModelNode.value ? capabilities.value[previewModelNode.value.type] : null)
const previewMeta       = computed(() => previewModelCap.value ? MODE_META[previewModelCap.value.mode] : null)
const previewSlug       = computed(() => (appName.value || '').toLowerCase().replace(/\s+/g, '-'))

// Simulated chart data
const chartPts  = Array.from({ length: 40 }, (_, i) => 50 + Math.sin(i * 0.4) * 18 + Math.sin(i * 1.7) * 5)
const chartAIdx = [10, 11, 28]
const chartW = 360, chartH = 110
const chartMinY = Math.min(...chartPts) - 4
const chartMaxY = Math.max(...chartPts) + 4
const chartSx = (i) => (i / 39) * (chartW - 30) + 15
const chartSy = (v) => chartH - 8 - ((v - chartMinY) / (chartMaxY - chartMinY)) * (chartH - 16)
const chartPathD = chartPts.map((v, i) => `${i === 0 ? 'M' : 'L'}${chartSx(i)},${chartSy(v)}`).join(' ')
const chartAreaD = `${chartPathD} L${chartSx(39)},${chartH} L${chartSx(0)},${chartH} Z`

// Target column: auto-detect from model's pipeline_config
const modelTargetColumn = computed(() => {
  // Check single model node
  const modelNode = nodes.value.find(n => n.type.startsWith('model.endpoint.'))
  if (modelNode) {
    const cap = capabilities.value[modelNode.type]
    return cap?.target_column || null
  }
  // Check multi-model compare endpoints
  const multiNode = nodes.value.find(n => n.type === 'output.multi_model_compare')
  if (multiNode) {
    const endpointIds = (multiNode.config?.endpoint_ids || [])
    for (const eidStr of endpointIds) {
      const eid = eidStr.split(':')[0]
      const ep = melabEndpoints.value.find(e => e.id === eid)
      if (ep?.target_column) return ep.target_column
    }
  }
  return null
})

const targetColumnHint = computed(() => {
  if (modelTargetColumn.value) {
    return `Target column from model: ${modelTargetColumn.value}`
  }
  // For classification, suggest label column name
  const multiNode = nodes.value.find(n => n.type === 'output.multi_model_compare')
  if (multiNode) {
    const endpointIds = (multiNode.config?.endpoint_ids || [])
    for (const eidStr of endpointIds) {
      const eid = eidStr.split(':')[0]
      const ep = melabEndpoints.value.find(e => e.id === eid)
      if (ep?.mode === 'classification') return 'For classification: enter the label column name (e.g., "label")'
    }
  }
  const modelNode = nodes.value.find(n => n.type.startsWith('model.endpoint.'))
  if (!modelNode) return ''
  const cap = capabilities.value[modelNode.type]
  if (!cap || cap.mode !== 'regression') return ''
  const featureNames = cap.feature_names || []
  const sensorCols = new Set()
  const prefixes = ['abs_energy','abs_sum_of_changes','spectral_bandwidth','margin_factor','peak_to_peak','rms','mean','std','max','min','crest_factor','shape_factor']
  for (const fname of featureNames) {
    for (const p of prefixes.sort((a,b) => b.length - a.length)) {
      if (fname.startsWith(p + '_')) {
        sensorCols.add(fname.slice(p.length + 1))
        break
      }
    }
  }
  if (sensorCols.size > 0) {
    return `Model uses: ${[...sensorCols].join(', ')}. Target is likely the column NOT in this list.`
  }
  return ''
})

// Auto-fill target column on Line Chart when model has target_column
watch(modelTargetColumn, (tc) => {
  if (!tc) return
  const lineChart = nodes.value.find(n => n.type === 'output.line_chart')
  if (lineChart && !lineChart.config.target_column) {
    lineChart.config.target_column = tc
  }
})

function firstConfigSummary(node) {
  const cap = capabilities.value[node.type]
  if (!cap?.configSchema?.length) return ''
  const [k, v] = Object.entries(node.config)[0] || []
  if (!k) return ''
  const display = Array.isArray(v) ? v.join(',') : v
  return `${k}: ${display}`
}

// ── Node operations ───────────────────────────────────────────────
function addNode(type) {
  const cap = capabilities.value[type]
  if (!cap) return
  nodes.value.push(makeNode(type, cap))
}

function removeNode(id) {
  nodes.value = nodes.value.filter(n => n.id !== id)
  if (selectedId.value === id) selectedId.value = null
}

function updateConfig(nodeId, key, val) {
  const node = nodes.value.find(n => n.id === nodeId)
  if (node) node.config[key] = val
}

// Available endpoints for switching (same mode as current model node)
const availableEndpoints = computed(() => {
  return melabEndpoints.value
    .filter(e => e.status === 'active')
    .map(e => ({
      id: e.id,
      label: `${e.name} (${e.algorithm})`,
      mode: e.mode,
    }))
})

function switchEndpoint(newEndpointId) {
  if (!selectedNode.value || !isModelNode(selectedNode.value.type)) return
  const oldIdx = nodes.value.findIndex(n => n.id === selectedNode.value.id)
  if (oldIdx < 0) return

  const newType = `model.endpoint.${newEndpointId}`
  const newCap = capabilities.value[newType]
  if (!newCap) return

  // Update the node's type and reset config
  nodes.value[oldIdx] = {
    ...nodes.value[oldIdx],
    type: newType,
    config: Object.fromEntries((newCap.configSchema || []).map(f => [f.key, f.default])),
  }
}

// MQTT topic discovery
const discoveringTopics = ref(false)
const discoveredTopics = ref([])

async function discoverTopics() {
  discoveringTopics.value = true
  try {
    const resp = await api.get('/api/mqtt/topics?duration=5')
    discoveredTopics.value = (resp.data || []).map(t => t.topic)
  } catch {
    discoveredTopics.value = []
  }
  discoveringTopics.value = false
}

function getMultiselectOptions(node, field) {
  // For multi_model_compare endpoint selection
  if (node?.type === 'output.multi_model_compare' && field.key === 'endpoint_ids') {
    return melabEndpoints.value
      .filter(e => e.status === 'active')
      .map(e => e.id + ':' + e.name)
  }
  // For feature_extract nodes, merge generic features with model-required features
  if (node?.type === 'transform.feature_extract' && field.key === 'features') {
    const baseOptions = new Set(field.options || [])
    // Collect feature names from single model nodes
    for (const n of nodes.value) {
      if (n.type.startsWith('model.endpoint.')) {
        const cap = capabilities.value[n.type]
        if (cap?.feature_names) {
          cap.feature_names.forEach(f => baseOptions.add(f))
        }
      }
      // Collect from multi-model compare endpoints
      if (n.type === 'output.multi_model_compare') {
        const endpointIds = n.config?.endpoint_ids || []
        for (const eidStr of endpointIds) {
          const eid = eidStr.split(':')[0]
          const cap = capabilities.value[`model.endpoint.${eid}`]
          if (cap?.feature_names) {
            cap.feature_names.forEach(f => baseOptions.add(f))
          }
          // Also check melabEndpoints directly
          const ep = melabEndpoints.value.find(e => e.id === eid)
          if (ep?.feature_names) {
            ep.feature_names.forEach(f => baseOptions.add(f))
          }
        }
      }
    }
    return [...baseOptions]
  }
  return field.options || []
}

function startPan(e) {
  if (!canvasRef.value) return
  isPanning.value = true
  panStart.value = {
    x: e.clientX,
    y: e.clientY,
    scrollLeft: canvasRef.value.scrollLeft,
    scrollTop: canvasRef.value.scrollTop,
  }
}
function doPan(e) {
  if (!isPanning.value || !canvasRef.value) return
  e.preventDefault()
  canvasRef.value.scrollLeft = panStart.value.scrollLeft - (e.clientX - panStart.value.x)
  canvasRef.value.scrollTop  = panStart.value.scrollTop  - (e.clientY - panStart.value.y)
}
function endPan() {
  isPanning.value = false
}

function toggleMultiselect(nodeId, key, opt, current) {
  const sel = Array.isArray(current) ? [...current] : []
  const idx = sel.indexOf(opt)
  if (idx >= 0) sel.splice(idx, 1)
  else sel.push(opt)
  updateConfig(nodeId, key, sel)
}

function autoConfigureFeatures() {
  if (!selectedCap.value?.feature_names) return
  const names = selectedCap.value.feature_names
  nodes.value.forEach(n => {
    if (n.type === 'transform.feature_extract') {
      n.config.features = [...names]
    }
    // Auto-fill target column on Line Chart
    if (n.type === 'output.line_chart' && selectedCap.value.target_column && !n.config.target_column) {
      n.config.target_column = selectedCap.value.target_column
    }
  })
}

function autoConfigureFromMultiModel() {
  const multiNode = nodes.value.find(n => n.type === 'output.multi_model_compare')
  if (!multiNode) return
  const endpointIds = multiNode.config?.endpoint_ids || []
  if (endpointIds.length === 0) return

  for (const eidStr of endpointIds) {
    const eid = eidStr.split(':')[0]
    const cap = capabilities.value[`model.endpoint.${eid}`]
    const ep = melabEndpoints.value.find(e => e.id === eid)
    const featureNames = cap?.feature_names || ep?.feature_names || []
    const targetCol = cap?.target_column || ep?.target_column || null

    if (featureNames.length > 0) {
      // Auto-fill Feature Extract
      nodes.value.forEach(n => {
        if (n.type === 'transform.feature_extract') {
          n.config.features = [...featureNames]
        }
      })
      // Auto-fill target column
      if (targetCol && !multiNode.config.target_column) {
        multiNode.config.target_column = targetCol
      }
      // For classification without target_column, set 'label' as default
      const mode = cap?.mode || ep?.mode
      if (mode === 'classification' && !multiNode.config.target_column) {
        multiNode.config.target_column = 'label'
      }
      return
    }
  }
}

// ── Clipboard ────────────────────────────────────────────────────
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => {})
}

// ── API calls ────────────────────────────────────────────────────
async function loadApp() {
  if (!appId.value) return
  try {
    const resp = await api.get(`/api/app-builder/apps/${appId.value}`)
    const data = resp.data
    appName.value = data.name || 'My App'
    nodes.value   = data.nodes || []
  } catch {
    // New app — start empty
  }
}

async function loadMelabEndpoints() {
  try {
    const resp = await api.get('/api/melab/endpoints')
    melabEndpoints.value = resp.data || []
  } catch {
    melabEndpoints.value = []
  }
}

async function saveApp() {
  if (!appId.value) return
  saving.value = true
  try {
    await api.put(`/api/app-builder/apps/${appId.value}`, {
      name: appName.value,
      nodes: nodes.value,
      edges: buildEdges(),
    })
  } catch (e) {
    console.error('Save failed', e)
  } finally {
    saving.value = false
  }
}

async function publishApp() {
  if (!readyToPublish.value || !appId.value) return
  publishing.value = true
  publishResult.value = null
  try {
    // Save first to ensure nodes are persisted
    await api.put(`/api/app-builder/apps/${appId.value}`, {
      name: appName.value,
      nodes: nodes.value,
      edges: buildEdges(),
    })
    // Then publish
    const resp = await api.post(`/api/app-builder/apps/${appId.value}/publish`, {
      name: appName.value,
      access: publishSettings.value.access,
      rate_limit: publishSettings.value.rateLimit,
    })
    publishResult.value = resp.data
  } catch (e) {
    const msg = e.response?.data?.error || 'Publish failed'
    alert(msg)
    publishResult.value = null
  } finally {
    publishing.value = false
  }
}

function buildEdges() {
  return nodes.value.slice(1).map((n, i) => ({
    id: `e${i}`,
    source: nodes.value[i].id,
    target: n.id,
  }))
}

// ── Lifecycle ────────────────────────────────────────────────────
onMounted(async () => {
  await loadMelabEndpoints()
  await loadApp()
})
</script>


<style scoped>
/* ── Root layout ───────────────────────────────────────────────── */
.editor-root {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #0d1117;
  color: #c9d1d9;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  overflow: hidden;
}

/* ── Top Bar ────────────────────────────────────────────────────── */
.editor-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 16px;
  border-bottom: 1px solid #21262d;
  background: #161b22;
  flex-shrink: 0;
  gap: 16px;
}

.topbar-brand {
  display: flex;
  align-items: center;
  gap: 6px;
}

.topbar-label-cira {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: #8b949e;
}

.topbar-label-ab {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: #a78bfa;
}

.topbar-sep {
  color: #30363d;
  font-size: 14px;
}

.app-name-field :deep(.v-field__input) {
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  color: #c9d1d9;
  min-height: unset;
  padding: 4px 4px;
}

.app-name-field :deep(.v-field) {
  padding: 0;
}

/* Tabs */
.topbar-tabs {
  display: flex;
  gap: 2px;
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 4px;
}

.topbar-tab {
  padding: 4px 14px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: #6e7681;
  background: transparent;
  border: none;
  cursor: pointer;
  transition: all 0.15s;
}

.topbar-tab:hover { color: #c9d1d9; }
.topbar-tab.active { background: #7c3aed; color: #fff; }

/* Status */
.status-error,
.status-ok {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  font-weight: 600;
}

.status-error { color: #f87171; }
.status-ok    { color: #34d399; }

.status-meta {
  font-size: 10px;
  color: #6e7681;
}

/* ── Editor body ────────────────────────────────────────────────── */
.editor-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Palette ────────────────────────────────────────────────────── */
.palette-panel {
  width: 200px;
  flex-shrink: 0;
  background: #0d1117;
  border-right: 1px solid #21262d;
  overflow-y: auto;
  padding: 10px 8px;
}

.palette-group {
  margin-bottom: 16px;
}

.palette-group-header {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #6e7681;
  padding: 0 8px;
  margin-bottom: 6px;
}

.palette-mode-group {
  margin-bottom: 8px;
}

.palette-mode-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0 8px;
  margin-bottom: 4px;
}

.palette-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.12s;
  margin-bottom: 2px;
}

.palette-item:hover { background: #161b22; }

.palette-item-info {
  flex: 1;
  min-width: 0;
}

.palette-item-label {
  font-size: 10px;
  color: #8b949e;
  transition: color 0.12s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.palette-item:hover .palette-item-label { color: #c9d1d9; }

.palette-item-sub {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
}

.mode-badge {
  font-size: 8px !important;
  height: 16px !important;
  padding: 0 4px !important;
  flex-shrink: 0;
}

.palette-empty {
  font-size: 9px;
  color: #484f58;
  padding: 4px 8px;
  font-style: italic;
}

.palette-footer {
  font-size: 9px;
  color: #484f58;
  padding: 8px 8px 0;
  border-top: 1px solid #21262d;
  margin-top: 8px;
}

/* ── Canvas ────────────────────────────────────────────────────── */
.canvas-panel {
  flex: 1;
  min-width: 0;
  overflow-x: auto;
  overflow-y: auto;
  background: #0d1117;
  display: flex;
  align-items: center;
  padding: 24px;
  position: relative;
  cursor: grab;
}
.canvas-panel:active {
  cursor: grabbing;
}

.canvas-empty {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #484f58;
}

.canvas-chain {
  display: flex;
  align-items: center;
  gap: 0;
  flex-shrink: 0;
  padding-right: 40px;
}

/* ── Node Cards ─────────────────────────────────────────────────── */
.node-card {
  position: relative;
  min-width: 110px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 10px;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  user-select: none;
  flex-shrink: 0;
}

.node-card:hover { border-color: #555; }
.node-card:hover .node-delete { opacity: 1; }

.node-card.selected {
  border-color: #7c3aed;
  background: #1a1230;
  box-shadow: 0 0 16px rgba(124, 58, 237, 0.15);
}

.node-card.has-error {
  border-color: #7f1d1d;
  background: #130a0a;
}

.node-card.has-error:hover { border-color: #b91c1c; }

.node-delete {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #30363d;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.15s, background 0.15s;
  z-index: 10;
  color: #8b949e;
}

.node-delete:hover {
  background: #dc2626;
  color: #fff;
}

.node-mode-badge {
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.12em;
  padding: 2px 5px;
  border-radius: 3px;
  border: 1px solid;
  display: inline-block;
  margin-bottom: 6px;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 2px;
}

.node-label {
  font-size: 11px;
  font-weight: 600;
  color: #c9d1d9;
  line-height: 1.2;
}

.node-algorithm {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
  margin-top: 2px;
}

.node-footer {
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.node-feat-badge {
  font-size: 9px;
  font-family: monospace;
  padding: 2px 6px;
  border-radius: 4px;
  background: #0d1117;
  border: 1px solid #30363d;
  color: #6e7681;
}

.node-output-type {
  margin-top: 6px;
  font-size: 8px;
  font-family: monospace;
  color: #484f58;
}

.node-selected-bar {
  position: absolute;
  bottom: 0;
  left: 10px;
  right: 10px;
  height: 2px;
  background: #7c3aed;
  border-radius: 2px;
}

/* ── Arrow connector ───────────────────────────────────────────── */
.node-arrow {
  display: flex;
  align-items: center;
  padding: 0 4px;
  flex-shrink: 0;
}

.arrow-line {
  width: 12px;
  height: 1px;
  background: #30363d;
}

.arrow-head {
  display: block;
}

/* ── Drop zone ──────────────────────────────────────────────────── */
.canvas-dropzone {
  width: 52px;
  height: 52px;
  border: 1px dashed #30363d;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #484f58;
  margin-left: 12px;
  flex-shrink: 0;
  cursor: default;
  transition: border-color 0.15s;
}

.canvas-dropzone:hover { border-color: #555; }

/* ── Config Panel ───────────────────────────────────────────────── */
.config-panel {
  width: 280px;
  flex-shrink: 0;
  background: #161b22;
  border-left: 1px solid #21262d;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.config-panel-header {
  padding: 8px 16px;
  border-bottom: 1px solid #21262d;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #6e7681;
  flex-shrink: 0;
}

.config-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #484f58;
}

.config-body {
  flex: 1;
  overflow-y: auto;
}

.config-node-header {
  padding: 12px 16px;
  border-bottom: 1px solid #21262d;
}

.config-node-label {
  font-size: 13px;
  font-weight: 600;
  color: #c9d1d9;
}

.config-node-type {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
  margin-top: 2px;
  margin-bottom: 8px;
}

.config-model-block {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 10px;
  margin-top: 6px;
  space-y: 6px;
}

.config-model-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 5px;
}

.config-model-key {
  font-size: 10px;
  color: #6e7681;
}

.config-model-val {
  font-size: 10px;
  color: #8b949e;
  font-family: monospace;
}

.feature-tag {
  font-size: 9px;
  font-family: monospace;
  padding: 2px 5px;
  border-radius: 3px;
  background: #1e1043;
  color: #a78bfa;
  border: 1px solid rgba(167, 139, 250, 0.25);
}

.config-error-block {
  margin: 10px 16px;
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(127, 29, 29, 0.25);
  border: 1px solid rgba(185, 28, 28, 0.4);
  font-size: 10px;
  color: #f87171;
  line-height: 1.4;
}

.config-io-row {
  padding: 8px 16px;
  border-bottom: 1px solid #21262d;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.io-badge {
  font-size: 9px;
  font-family: monospace;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid;
}

.io-badge-in {
  background: #0d1117;
  border-color: #21262d;
  color: #6e7681;
}

.config-fields {
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.config-field-label {
  font-size: 11px;
  color: #8b949e;
}

.config-input :deep(.v-field__outline) {
  border-color: #30363d !important;
}

.config-input :deep(.v-field__input) {
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  color: #c9d1d9;
  min-height: unset;
}

.toggle-inline {
  margin: 0 !important;
  padding: 0 !important;
}

.slider-val {
  font-size: 11px;
  font-family: monospace;
  color: #a78bfa;
  min-width: 28px;
  text-align: right;
}

.multiselect-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 2px;
}

.multiselect-chip {
  padding: 4px 10px;
  border-radius: 5px;
  font-size: 12px;
  font-family: monospace;
  border: 1px solid #30363d;
  background: #0d1117;
  color: #8b949e;
  cursor: pointer;
  transition: all 0.12s;
}

.multiselect-chip:hover { border-color: #555; }

.multiselect-chip.active {
  background: rgba(124, 58, 237, 0.15);
  border-color: #7c3aed;
  color: #c4b5fd;
}

.discovered-topic-chip {
  font-size: 9px;
  font-family: monospace;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid #30363d;
  background: #0d1117;
  color: #60a5fa;
  cursor: pointer;
  transition: all 0.15s;
}
.discovered-topic-chip:hover { border-color: #60a5fa; }
.discovered-topic-chip.active {
  background: rgba(96, 165, 250, 0.15);
  border-color: #60a5fa;
  color: #93c5fd;
}

.multiselect-count {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
  width: 100%;
  margin-top: 2px;
}

/* ── Preview ────────────────────────────────────────────────────── */
.panel-section-header {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #6e7681;
  margin-bottom: 12px;
}

.preview-steps-panel {
  width: 200px;
  flex-shrink: 0;
  background: #0d1117;
  border-right: 1px solid #21262d;
  overflow-y: auto;
  padding: 12px;
}

.preview-step {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 12px;
}

.preview-step-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex-shrink: 0;
}

.step-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #161b22;
  display: flex;
  align-items: center;
  justify-content: center;
}

.step-dot-error { background: #1c0a0a; }

.step-line {
  width: 1px;
  height: 18px;
  background: #21262d;
  margin-top: 2px;
}

.preview-step-info {
  flex: 1;
  min-width: 0;
  padding-top: 3px;
}

.preview-step-label {
  font-size: 10px;
  font-weight: 600;
  color: #c9d1d9;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preview-step-mode {
  font-size: 9px;
  font-weight: 700;
  margin-top: 2px;
}

.preview-step-error {
  font-size: 9px;
  color: #f87171;
  margin-top: 2px;
  line-height: 1.3;
}

.preview-step-config {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
  margin-top: 2px;
}

.preview-main {
  flex: 1;
  overflow: auto;
  background: #0d1117;
  padding: 24px;
}

/* ── Publish ────────────────────────────────────────────────────── */
.publish-body {
  justify-content: center;
}

.publish-container {
  width: 100%;
  max-width: 460px;
  overflow-y: auto;
  padding: 24px 16px;
}

.publish-card {
  border-color: #21262d !important;
  background: #161b22 !important;
  border-radius: 10px !important;
}

.publish-card-header {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #6e7681;
  padding: 12px 16px 8px;
  border-bottom: 1px solid #21262d;
}

.publish-check-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  gap: 8px;
  border-bottom: 1px solid #0d1117;
}

.publish-check-label {
  font-size: 10px;
  color: #8b949e;
}

.publish-check-error {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 9px;
  color: #f87171;
  text-align: right;
  max-width: 160px;
  line-height: 1.3;
}

.publish-settings {
  padding: 12px 16px;
}

.publish-btn {
  letter-spacing: 0.08em;
  font-weight: 700;
}

.publish-result-card {
  border-color: rgba(52, 211, 153, 0.25) !important;
  background: #161b22 !important;
  border-radius: 10px !important;
  padding: 16px;
}

.publish-result-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #34d399;
  margin-bottom: 12px;
}

.publish-result-row {
  margin-bottom: 10px;
}

.publish-result-label {
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6e7681;
  margin-bottom: 4px;
}

.publish-result-value-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.publish-result-code {
  flex: 1;
  font-size: 10px;
  font-family: monospace;
  background: #0d1117;
  padding: 6px 8px;
  border-radius: 4px;
  border: 1px solid #21262d;
  word-break: break-all;
  display: block;
}

/* ── Output Preview (browser mockup) ────────────────────────────── */
.output-preview-wrap {
  max-width: 480px;
}

.output-preview-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 0;
  color: #484f58;
  text-align: center;
}

.browser-bar {
  background: #0d1117;
  border: 1px solid #21262d;
  border-bottom: none;
  border-radius: 8px 8px 0 0;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.browser-dots {
  display: flex;
  gap: 4px;
}

.browser-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #21262d;
}

.browser-url {
  font-size: 10px;
  color: #6e7681;
  font-family: monospace;
  margin-left: 6px;
  flex: 1;
}

.browser-mode-badge {
  font-size: 9px;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 3px;
}

.browser-body {
  background: #161b22;
  border: 1px solid #21262d;
  border-top: none;
  border-radius: 0 0 8px 8px;
  padding: 16px;
}

.browser-app-title {
  font-size: 13px;
  font-weight: 600;
  color: #c9d1d9;
}

.browser-model-info {
  font-size: 9px;
  color: #484f58;
  font-family: monospace;
}

.browser-input-widget {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 12px;
}

.browser-widget-label {
  font-size: 9px;
  color: #6e7681;
  font-family: monospace;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.csv-upload-row {
  display: flex;
  gap: 8px;
}

.csv-drop-zone {
  flex: 1;
  font-size: 10px;
  color: #484f58;
  border: 1px dashed #30363d;
  border-radius: 4px;
  padding: 6px 10px;
  font-family: monospace;
  text-align: center;
}

.run-btn {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  display: flex;
  align-items: center;
}

.live-stream-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}

.live-label {
  font-size: 10px;
  color: #6e7681;
  font-family: monospace;
}

.chart-title {
  font-size: 10px;
  color: #6e7681;
  font-family: monospace;
  margin-bottom: 6px;
}

.chart-legend {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 4px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 9px;
  color: #6e7681;
  font-family: monospace;
}

.legend-line {
  width: 16px;
  height: 2px;
  border-radius: 1px;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.alert-badge-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.alert-badge-normal,
.alert-badge-anomaly {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid;
}

.alert-badge-normal {
  color: #34d399;
  background: rgba(52, 211, 153, 0.08);
  border-color: rgba(52, 211, 153, 0.3);
}

.alert-badge-anomaly {
  color: #f87171;
  background: rgba(248, 113, 113, 0.08);
  border-color: rgba(248, 113, 113, 0.3);
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
  font-family: monospace;
}

.preview-table th {
  text-align: left;
  padding: 4px 10px 4px 0;
  color: #6e7681;
  border-bottom: 1px solid #21262d;
  font-weight: 600;
}

.preview-table td {
  padding: 5px 10px 5px 0;
  color: #c9d1d9;
  border-bottom: 1px solid #0d1117;
}

.table-label-badge {
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 9px;
}

/* Scrollbar styling */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
