<template>
  <div class="published-app" :class="{ 'dashboard-mode': dashboardMode }">
    <!-- Loading -->
    <div v-if="loading" class="app-loading">
      <v-progress-circular indeterminate color="purple" />
      <div class="text-caption text-medium-emphasis mt-3">Loading app...</div>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="app-loading">
      <v-icon size="48" color="error">mdi-alert-circle-outline</v-icon>
      <div class="text-h6 mt-3">App not found</div>
      <div class="text-caption text-medium-emphasis mt-1">{{ error }}</div>
      <v-btn class="mt-4" variant="outlined" to="/">Go to Dashboard</v-btn>
    </div>

    <!-- App content -->
    <div v-else class="app-content">
      <!-- Header (shared between dashboard + fallback layouts) -->
      <div class="app-header">
        <div class="app-header-left">
          <img src="/logo.svg" alt="CiRA" class="app-logo" />
          <div>
            <div class="app-title">{{ appData.name }}</div>
            <div class="app-subtitle">
              <span v-if="isRecorderMode" class="app-mode-badge" style="color: #ef4444; background: rgba(239,68,68,0.1)">
                SIGNAL RECORDER
              </span>
              <span v-else-if="appMode" class="app-mode-badge" :style="{ color: modeColor, background: modeColor + '18' }">
                {{ appMode?.toUpperCase() }}
              </span>
              <span class="app-algo">{{ appAlgorithm }}</span>
              <!-- Inline pipeline chips (dashboard header) -->
              <template v-if="dashboardMode && pipelineInfo">
                <v-chip size="x-small" color="info" variant="tonal" class="ml-2">W {{ pipelineInfo.window_size }}</v-chip>
                <v-chip size="x-small" color="info" variant="tonal">S {{ pipelineInfo.stride }}</v-chip>
                <v-chip size="x-small" color="info" variant="tonal">F {{ pipelineInfo.n_features }}</v-chip>
              </template>
            </div>
          </div>
        </div>
        <div class="app-header-right">
          <v-btn
            v-if="!isStandalone"
            size="small"
            variant="tonal"
            color="purple"
            :href="`/standalone/${slug}`"
            target="_blank"
            class="mr-2"
          >
            <v-icon start size="small">mdi-open-in-new</v-icon>
            Open Standalone
          </v-btn>
          <v-menu v-if="dashboardMode" :close-on-content-click="false">
            <template #activator="{ props }">
              <v-btn
                v-bind="props"
                size="small"
                variant="tonal"
                color="grey"
                class="mr-2"
                title="Display settings"
              >
                <v-icon size="small">mdi-cog-outline</v-icon>
              </v-btn>
            </template>
            <v-card min-width="280" class="pa-3">
              <div class="text-subtitle-2 mb-2">Display</div>
              <div class="text-caption text-medium-emphasis mb-1">Font size</div>
              <v-btn-toggle
                v-model="displayFontSize"
                density="compact"
                mandatory
                divided
                variant="outlined"
                class="mb-3 d-block"
              >
                <v-btn value="S" size="x-small">S</v-btn>
                <v-btn value="M" size="x-small">M</v-btn>
                <v-btn value="L" size="x-small">L</v-btn>
                <v-btn value="XL" size="x-small">XL</v-btn>
              </v-btn-toggle>
              <div class="text-caption text-medium-emphasis mb-1">Decimal places</div>
              <v-btn-toggle
                v-model="displayPrecision"
                density="compact"
                mandatory
                divided
                variant="outlined"
                class="mb-3 d-block"
              >
                <v-btn :value="0" size="x-small">0</v-btn>
                <v-btn :value="1" size="x-small">1</v-btn>
                <v-btn :value="2" size="x-small">2</v-btn>
                <v-btn :value="3" size="x-small">3</v-btn>
                <v-btn :value="4" size="x-small">4</v-btn>
                <v-btn :value="6" size="x-small">6</v-btn>
              </v-btn-toggle>
              <div class="text-caption text-medium-emphasis mb-1">View</div>
              <v-btn-toggle
                v-model="displayView"
                density="compact"
                mandatory
                divided
                variant="outlined"
                class="d-block"
              >
                <v-btn value="chart" size="x-small">
                  <v-icon size="14" start>mdi-chart-line</v-icon>Chart
                </v-btn>
                <v-btn value="table" size="x-small">
                  <v-icon size="14" start>mdi-table</v-icon>Table
                </v-btn>
              </v-btn-toggle>
            </v-card>
          </v-menu>
          <v-btn
            size="small"
            variant="tonal"
            color="grey"
            class="mr-2"
            :title="isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'"
            @click="toggleFullscreen"
          >
            <v-icon size="small">{{ isFullscreen ? 'mdi-fullscreen-exit' : 'mdi-fullscreen' }}</v-icon>
          </v-btn>
          <span class="app-powered">Powered by CiRA ME</span>
        </div>
      </div>

      <!-- ═══════════════════════════════════════════════════════ -->
      <!-- DASHBOARD (two-pane) LAYOUT — MQTT / Signal Recorder     -->
      <!-- ═══════════════════════════════════════════════════════ -->
      <div v-if="dashboardMode" class="dashboard-body">
        <!-- LEFT RAIL -->
        <div class="dash-rail" :class="{ collapsed: railCollapsed }">
          <button
            class="rail-toggle"
            :title="railCollapsed ? 'Expand settings' : 'Collapse settings'"
            @click="railCollapsed = !railCollapsed"
          >
            <v-icon size="18">{{ railCollapsed ? 'mdi-chevron-right' : 'mdi-chevron-left' }}</v-icon>
          </button>

          <!-- Collapsed icon strip -->
          <div v-if="railCollapsed" class="rail-icons">
            <div class="rail-icon-btn" :title="mqttConnected ? 'LIVE' : (mqttError ? 'Error' : 'Disconnected')">
              <v-icon size="24" :color="mqttConnected ? 'success' : (mqttError ? 'error' : 'grey')">
                mdi-circle
              </v-icon>
              <span class="rail-icon-lbl">{{ mqttConnected ? 'LIVE' : 'OFF' }}</span>
            </div>
            <div class="rail-icon-btn" title="Messages">
              <v-icon size="22" color="grey">mdi-message-flash-outline</v-icon>
              <span class="rail-icon-lbl">{{ mqttMessageCount.toLocaleString() }}</span>
            </div>
            <div class="rail-icon-btn" title="Inferences">
              <v-icon size="22" color="purple">mdi-brain</v-icon>
              <span class="rail-icon-lbl">{{ liveInferenceCount }}</span>
            </div>
            <div
              v-if="isRecordingPredictions"
              class="rail-icon-btn"
              :title="`Recording · ${predictionRecordBuffer.length} rows`"
            >
              <v-icon size="22" color="error">mdi-record-circle</v-icon>
              <span class="rail-icon-lbl" style="color:#ef5350">REC</span>
            </div>
            <button
              v-if="mqttConnected"
              class="rail-icon-btn rail-icon-btn-danger"
              title="Disconnect"
              @click="stopLiveStream"
            >
              <v-icon size="22" color="error">mdi-stop-circle-outline</v-icon>
              <span class="rail-icon-lbl">STOP</span>
            </button>
          </div>

          <!-- Expanded config -->
          <div v-else class="rail-expanded">
            <div class="rail-section-title">
              <v-icon size="14" :color="mqttConnected ? 'success' : 'grey'">mdi-access-point</v-icon>
              Live Stream (MQTT)
              <v-chip v-if="mqttConnected" size="x-small" color="success" variant="flat" class="ml-1">
                <v-icon start size="8">mdi-circle</v-icon>LIVE
              </v-chip>
              <v-chip v-else-if="mqttError" size="x-small" color="error" variant="tonal" class="ml-1">
                Error
              </v-chip>
            </div>

            <v-text-field
              v-model="mqttBrokerUrl"
              label="Broker URL"
              variant="outlined"
              density="compact"
              hide-details
              class="mb-2"
              style="font-size: 12px;"
              :disabled="mqttConnected"
            />
            <v-text-field
              v-model="mqttTopic"
              label="Topic"
              variant="outlined"
              density="compact"
              hide-details
              class="mb-2"
              style="font-size: 12px;"
              :disabled="mqttConnected"
            />

            <!-- Auto-record predictions -->
            <div v-if="!isRecorderMode" class="rail-record-group">
              <v-checkbox
                v-model="autoRecordPredictions"
                :disabled="mqttConnected"
                density="compact"
                hide-details
                color="error"
              >
                <template #label>
                  <span class="text-caption">
                    <v-icon size="x-small" color="error" class="mr-1">mdi-record-circle-outline</v-icon>
                    Auto-record to CSV
                  </span>
                </template>
              </v-checkbox>
              <v-select
                v-model="predictionRecordMode"
                :items="[
                  { title: 'Per inference', value: 'per_inference' },
                  { title: 'Per sample', value: 'per_sample' },
                  { title: 'Full window', value: 'full_window' },
                ]"
                :disabled="!autoRecordPredictions || mqttConnected"
                density="compact"
                variant="outlined"
                hide-details
                label="Granularity"
                class="mt-1"
                style="font-size: 12px;"
              />
              <div class="text-caption text-medium-emphasis mt-1" style="font-size: 10px;">
                CSV saves on Disconnect.
              </div>
            </div>

            <!-- Regression: actual column selector (single-model) -->
            <v-select
              v-if="mqttConnected && !isRecorderMode && !isMultiModelApp && appMode === 'regression' && (autoDetectedChannels.length > 0 || liveChannels.length > 0)"
              v-model="liveActualColumn"
              :items="['(none)', ...(autoDetectedChannels.length > 0 ? autoDetectedChannels : liveChannels)]"
              label="Compare with column"
              density="compact"
              variant="outlined"
              hide-details
              prepend-inner-icon="mdi-chart-line"
              class="mt-3"
              style="font-size: 12px;"
            />

            <v-alert v-if="mqttError" type="error" variant="tonal" density="compact" class="mt-3" style="font-size: 11px;">
              {{ mqttError }}
            </v-alert>

            <!-- Fast Mode toggle (dashboard rail) — P2 Phase 3 -->
            <div v-if="fastModeAvailable" class="rail-fast-mode mt-3">
              <div class="d-flex align-center gap-1">
                <v-icon size="14" :color="fastModeEnabled ? 'success' : 'grey'">mdi-lightning-bolt</v-icon>
                <span class="text-caption" style="font-weight: 600;">Fast Mode</span>
                <v-switch
                  v-model="fastModeEnabled"
                  color="success"
                  hide-details
                  density="compact"
                  inset
                  class="mt-0 ml-auto"
                  style="transform: scale(0.75); transform-origin: right;"
                />
              </div>
              <div class="text-caption text-medium-emphasis" style="font-size: 10px; line-height: 1.3;">
                Extract features in browser (skips server tsfresh — 47 features only).
              </div>
              <span
                class="fast-mode-badge mt-1"
                :class="{ on: fastModeEnabled }"
                style="font-size: 9px;"
              >
                <v-icon size="10" class="mr-1">{{ fastModeEnabled ? 'mdi-lightning-bolt' : 'mdi-server' }}</v-icon>
                {{ fastModeEnabled ? 'Fast (browser)' : 'Server extraction' }}
              </span>
            </div>

            <!-- Connect / Disconnect -->
            <v-btn
              v-if="!mqttConnected"
              color="success"
              variant="flat"
              block
              class="mt-3"
              @click="startLiveStream"
            >
              <v-icon start>mdi-play</v-icon>Connect
            </v-btn>
            <v-btn
              v-else
              color="error"
              variant="tonal"
              block
              class="mt-3"
              @click="stopLiveStream"
            >
              <v-icon start>mdi-stop</v-icon>Disconnect
            </v-btn>

            <!-- Test Publisher (rail) — logged-in users only. Rail is too
                 narrow for the full panel; open in a dialog on demand. -->
            <v-btn
              v-if="isAuthenticated && isLiveStream"
              variant="tonal"
              color="info"
              size="small"
              block
              class="mt-2"
              @click="showTestPublisherDialog = true"
            >
              <v-icon start size="14">mdi-test-tube</v-icon>
              Test Publisher
            </v-btn>

            <!-- Recorder-specific rail controls -->
            <template v-if="isRecorderMode && mqttConnected">
              <div class="rail-section-title mt-3">
                <v-icon size="14" color="error">mdi-record</v-icon>Current label
                <v-chip v-if="recorderState.recording" size="x-small" color="error" variant="flat" class="ml-1">
                  <v-icon start size="8">mdi-circle</v-icon>REC
                </v-chip>
              </div>
              <div class="recorder-labels" style="margin-bottom: 6px;">
                <button
                  v-for="lbl in recorderLabels"
                  :key="lbl"
                  class="recorder-label-btn"
                  :class="{ active: recorderState.currentLabel === lbl }"
                  @click="recorderState.currentLabel = lbl"
                >{{ lbl }}</button>
              </div>
              <div class="d-flex align-center gap-2 mb-2">
                <v-text-field
                  v-model="recorderCustomLabel"
                  label="Custom"
                  variant="outlined"
                  density="compact"
                  hide-details
                  style="font-size: 11px;"
                  @keydown.enter="addCustomLabel"
                />
                <v-btn size="x-small" variant="tonal" @click="addCustomLabel" :disabled="!recorderCustomLabel.trim()">
                  <v-icon size="small">mdi-plus</v-icon>
                </v-btn>
              </div>
              <v-btn
                v-if="!recorderState.recording"
                color="error"
                variant="flat"
                block
                :disabled="!recorderState.currentLabel"
                @click="startRecording"
              >
                <v-icon start size="small">mdi-record</v-icon>Start Recording
              </v-btn>
              <v-btn
                v-else
                color="warning"
                variant="flat"
                block
                @click="stopRecording"
              >
                <v-icon start size="small">mdi-stop</v-icon>Stop Recording
              </v-btn>
              <v-btn
                v-if="recorderState.samples.length > 0"
                color="success"
                variant="tonal"
                block
                class="mt-2"
                @click="downloadRecordedCSV"
              >
                <v-icon start size="small">mdi-download</v-icon>
                Download ({{ recorderState.samples.length }})
              </v-btn>
              <v-btn
                v-if="recorderState.samples.length > 0 && !recorderState.recording"
                variant="text"
                color="error"
                block
                class="mt-1"
                @click="clearRecording"
              >Clear</v-btn>
            </template>
          </div>
        </div>

        <!-- RIGHT PANE -->
        <div class="dash-main">
          <!-- Compact live stats strip -->
          <div v-if="mqttConnected" class="dash-stats-strip">
            <div class="live-stat">
              <div class="live-stat-label">Messages</div>
              <div class="live-stat-value">{{ mqttMessageCount.toLocaleString() }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Rate</div>
              <div class="live-stat-value">{{ mqttMessagesPerSec }}/s</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Buffer</div>
              <div class="live-stat-value">{{ sensorBufferLen }}/{{ liveWindowSize }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Inferences</div>
              <div class="live-stat-value">{{ liveInferenceCount }}</div>
            </div>
            <div
              v-if="isRecordingPredictions"
              class="live-stat"
              style="background: rgba(244, 67, 54, 0.12); border-color: rgba(244, 67, 54, 0.4);"
            >
              <div class="live-stat-label" style="color: #ef5350;">
                <v-icon size="x-small" color="error" class="mr-1">mdi-circle</v-icon>Recording
              </div>
              <div class="live-stat-value" style="color: #ef5350;">
                {{ predictionRecordBuffer.length }} rows · {{ predictionRecordDuration }}
              </div>
            </div>
            <div class="dash-progress">
              <v-progress-linear
                :model-value="sensorBufferProgress * 100"
                color="purple"
                height="6"
                rounded
              />
            </div>
          </div>

          <!-- ────────── Signal Recorder preview ────────── -->
          <div v-if="isRecorderMode && mqttConnected" class="dash-content dash-recorder">
            <!-- Auto-fallback warning banner (Layer 1) — shown when the parser had
                 to positionally match discovered numeric leaves to configured
                 channel names because no names overlapped. -->
            <v-alert
              v-if="sensorAutoFallbackActive"
              type="warning"
              variant="tonal"
              density="compact"
              closable
              @click:close="sensorAutoFallbackActive = false"
            >
              Configured channels
              <strong>{{ (sensorAutoFallbackInfo?.configured || []).join(', ') }}</strong>
              didn't match any payload keys. Auto-matched by position from detected keys
              <strong>{{ (sensorAutoFallbackInfo?.detected || []).join(', ') }}</strong>.
              Update your App Builder MQTT node to future-proof.
            </v-alert>

            <!-- Show raw MQTT toggle (Layer 3) — diagnostic view of the last 3
                 payloads so the operator can eyeball payload shape without
                 leaving the recorder. -->
            <div class="raw-mqtt-controls">
              <v-btn
                size="x-small"
                variant="tonal"
                :prepend-icon="showRawMqtt ? 'mdi-eye-off-outline' : 'mdi-eye-outline'"
                @click="showRawMqtt = !showRawMqtt"
              >
                {{ showRawMqtt ? 'Hide raw MQTT' : 'Show raw MQTT' }}
              </v-btn>
            </div>
            <div v-if="showRawMqtt && rawMqttBuffer.length > 0" class="raw-mqtt-wrap">
              <pre
                v-for="(msg, i) in rawMqttBuffer"
                :key="'raw-' + i"
                class="raw-mqtt-block"
              >{{ msg.length > 500 ? msg.slice(0, 500) + '…' : msg }}</pre>
            </div>

            <div v-if="liveChannels.length > 0" class="preview-panel dash-preview">
              <div class="preview-header">
                <span class="text-caption font-weight-bold text-medium-emphasis">LIVE PREVIEW</span>
                <v-select
                  v-model="previewWindowSec"
                  :items="[5, 10, 30, 60, 120]"
                  label="Preview window (s)"
                  variant="outlined"
                  density="compact"
                  hide-details
                  style="max-width: 180px; font-size: 11px;"
                />
              </div>

              <div class="preview-cards">
                <div v-for="ch in liveChannels" :key="'pv-' + ch" class="preview-card">
                  <div class="preview-card-label">{{ ch }}</div>
                  <div class="preview-card-value">{{ fmtPreview(previewStats[ch]?.latest) }}</div>
                  <div class="preview-card-stats">
                    <span>Min: {{ fmtPreview(previewStats[ch]?.min) }}</span>
                    <span class="preview-card-stats-sep">·</span>
                    <span>Max: {{ fmtPreview(previewStats[ch]?.max) }}</span>
                    <span class="preview-card-stats-sep">·</span>
                    <span>Mean: {{ fmtPreview(previewStats[ch]?.mean) }}</span>
                  </div>
                </div>
              </div>

              <div v-if="previewBuffer.length > 1" class="preview-chart dash-preview-chart">
                <Line :data="previewChartData" :options="previewChartOptions" />
              </div>
              <div v-else class="preview-chart-empty dash-preview-chart">
                Waiting for samples…
              </div>
            </div>

            <!-- Recording stats -->
            <div v-if="recorderState.samples.length > 0" class="dash-recorder-stats">
              <div class="live-stat">
                <div class="live-stat-label">Samples</div>
                <div class="live-stat-value">{{ recorderState.samples.length }}</div>
              </div>
              <div class="live-stat">
                <div class="live-stat-label">Duration</div>
                <div class="live-stat-value">{{ recorderDuration }}</div>
              </div>
              <div class="live-stat" style="flex: 1;">
                <div class="live-stat-label">Labels</div>
                <div class="live-stat-value" style="font-size: 12px; white-space: normal;">{{ recorderLabelCounts }}</div>
              </div>
            </div>

            <div v-if="recorderState.samples.length > 0" class="recorder-timeline">
              <div
                v-for="(seg, i) in recorderSegments"
                :key="i"
                class="recorder-segment"
                :style="{ flex: seg.count, background: seg.color }"
                :title="`${seg.label}: ${seg.count} samples`"
              />
            </div>
          </div>

          <!-- ────────── Multi-model TILE GRID ────────── -->
          <div v-else-if="isMultiModelApp" class="dash-content dash-multi-grid">
            <div v-if="!result || !result.models" class="dash-placeholder">
              <v-icon size="48" color="grey">mdi-compare-horizontal</v-icon>
              <div class="text-caption text-medium-emphasis mt-2">
                {{ mqttConnected ? 'Waiting for inference…' : 'Connect to start comparing models' }}
              </div>
            </div>
            <div v-else class="multi-tile-grid">
              <div
                v-for="(m, eid) in result.models"
                :key="'tile-' + eid"
                class="multi-tile"
                :class="{ 'multi-tile-error': !!m.error }"
              >
                <div class="multi-tile-head">
                  <div class="multi-tile-name">
                    {{ m.name }}
                    <v-icon
                      v-if="!m.error && result.actual && ((result.mode === 'regression' && m.r2 === bestR2) || (result.mode === 'classification' && m.accuracy === bestAccuracy))"
                      size="16" color="amber"
                    >mdi-trophy</v-icon>
                  </div>
                  <v-chip size="x-small" variant="tonal" color="purple">{{ m.algorithm || '—' }}</v-chip>
                </div>

                <div v-if="m.error" class="multi-tile-error-msg">
                  <v-icon size="20" color="error">mdi-alert-circle</v-icon>
                  {{ m.error }}
                </div>

                <template v-else>
                  <div class="multi-tile-pred" :style="{ color: multiTilePredColor(m, result.mode) }">
                    {{ multiTileLatest(m) }}
                  </div>

                  <!-- Regression metrics -->
                  <div v-if="result.mode === 'regression' && result.actual" class="multi-tile-metrics">
                    <span :style="{ color: m.r2 > 0.8 ? '#34d399' : m.r2 > 0.5 ? '#fbbf24' : '#f87171' }">
                      R² {{ m.r2 != null ? m.r2.toFixed(3) : '—' }}
                    </span>
                    <span>RMSE {{ m.rmse != null ? m.rmse.toFixed(3) : '—' }}</span>
                  </div>

                  <!-- Classification confidence bar (latest window from predictions_full if present) -->
                  <div v-else-if="result.mode === 'classification'" class="multi-tile-metrics">
                    <span v-if="m.accuracy != null" :style="{ color: m.accuracy > 0.9 ? '#34d399' : m.accuracy > 0.7 ? '#fbbf24' : '#f87171' }">
                      Accuracy {{ (m.accuracy * 100).toFixed(1) }}%
                    </span>
                    <span v-else>Windows {{ m.count || 0 }}</span>
                  </div>

                  <!-- Mini live chart -->
                  <div v-if="m.predictions && m.predictions.length > 0" class="multi-tile-chart">
                    <Line
                      :data="tileChartData(eid, m)"
                      :options="tileChartOptions(result.mode)"
                    />
                  </div>
                </template>
              </div>
            </div>
          </div>

          <!-- ────────── Single-model prediction + live chart ────────── -->
          <div v-else class="dash-content dash-single">
            <!-- Big prediction card -->
            <div v-if="livePrediction !== null" class="dash-prediction-card" :style="{ borderColor: modeColor + '55' }">
              <div class="live-prediction-label">Latest Prediction</div>
              <div
                class="dash-prediction-value"
                :class="['fs-' + displayFontSize]"
                :style="{ color: modeColor }"
              >
                {{ displayedPrediction }}
              </div>
              <div v-if="liveLastUpdated" class="live-prediction-time">
                {{ liveLastUpdatedText }}
              </div>
              <!-- Confidence bar for classification -->
              <div v-if="appMode === 'classification' && livePredictionHistoryFull.length > 0" class="dash-confidence">
                <template v-if="latestConfidence != null">
                  <div class="dash-confidence-label">Confidence · {{ (latestConfidence * 100).toFixed(1) }}%</div>
                  <v-progress-linear
                    :model-value="latestConfidence * 100"
                    :color="latestConfidence > 0.85 ? 'success' : latestConfidence > 0.6 ? 'warning' : 'error'"
                    height="8"
                    rounded
                  />
                </template>
              </div>
            </div>
            <div v-else class="dash-placeholder dash-prediction-card">
              <v-icon size="48" color="grey">mdi-brain</v-icon>
              <div class="text-caption text-medium-emphasis mt-2">
                {{ mqttConnected ? 'Waiting for enough samples to run inference…' : 'Connect to start streaming' }}
              </div>
            </div>

            <!-- Live chart OR table (view is persisted per-app in localStorage) -->
            <template v-if="displayView === 'chart'">
              <div v-if="chartData.length > 0" class="chart-container dash-chart-container">
                <div class="chart-header">
                  <span class="chart-title-text">{{ actualData.length > 0 ? 'Actual vs Predicted' : 'Predictions over Time' }}</span>
                </div>
                <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" preserveAspectRatio="none" class="prediction-chart dash-svg-chart">
                  <line v-for="i in 4" :key="'g'+i"
                    :x1="chartPadding" :y1="chartPadding + (i-1) * (chartInnerH / 3)"
                    :x2="chartWidth - chartPadding" :y2="chartPadding + (i-1) * (chartInnerH / 3)"
                    stroke="#21262d" stroke-width="0.5" />
                  <text v-for="i in 4" :key="'y'+i"
                    :x="chartPadding - 4" :y="chartPadding + (i-1) * (chartInnerH / 3) + 3"
                    text-anchor="end" fill="#8b949e" font-size="8" font-family="monospace">
                    {{ chartYLabel(i-1) }}
                  </text>
                  <path :d="chartAreaPath" fill="url(#pred-gradient-d)" />
                  <path v-if="actualLinePath" :d="actualLinePath" fill="none" stroke="#22d3ee" stroke-width="1.5" />
                  <path :d="chartLinePath" fill="none" stroke="#a78bfa" stroke-width="1.5" :stroke-dasharray="actualLinePath ? '4,2' : 'none'" />
                  <defs>
                    <linearGradient id="pred-gradient-d" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.15" />
                      <stop offset="100%" stop-color="#a78bfa" stop-opacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
                <div class="chart-legend-items" style="margin-top:8px">
                  <span v-if="actualData.length > 0" class="chart-legend-dot" style="background: #22d3ee"></span>
                  <span v-if="actualData.length > 0" class="chart-legend-label">Actual</span>
                  <span class="chart-legend-dot" style="background: #a78bfa"></span>
                  <span class="chart-legend-label">Predicted</span>
                </div>
              </div>
              <div v-else class="dash-placeholder dash-chart-container">
                <v-icon size="40" color="grey">mdi-chart-line-variant</v-icon>
                <div class="text-caption text-medium-emphasis mt-2">Chart appears once inferences begin</div>
              </div>
            </template>
            <template v-else>
              <div v-if="dashboardTableRows.length > 0" class="dash-table-container">
                <table class="dash-table">
                  <thead>
                    <tr>
                      <th style="width: 100px;">Time</th>
                      <th>Prediction</th>
                      <th v-if="appMode === 'classification' && dashboardTableRows.some(r => r.hasConfidence)" style="width: 110px;">Confidence</th>
                      <th v-if="dashboardTableRows.some(r => r.hasActual)" style="width: 110px;">Actual</th>
                      <th v-if="appMode === 'regression' && dashboardTableRows.some(r => r.hasActual)" style="width: 110px;">Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in dashboardTableRows" :key="row.i">
                      <td>{{ row.timeText }}</td>
                      <td>{{ row.predText }}</td>
                      <td v-if="appMode === 'classification' && dashboardTableRows.some(r => r.hasConfidence)">
                        {{ row.confidenceText }}
                      </td>
                      <td v-if="dashboardTableRows.some(r => r.hasActual)">{{ row.actualText }}</td>
                      <td v-if="appMode === 'regression' && dashboardTableRows.some(r => r.hasActual)">
                        {{ row.errorText }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-else class="dash-placeholder dash-chart-container">
                <v-icon size="40" color="grey">mdi-table</v-icon>
                <div class="text-caption text-medium-emphasis mt-2">Table appears once inferences begin</div>
              </div>
            </template>
          </div>
        </div>
      </div>

      <!-- ═══════════════════════════════════════════════════════ -->
      <!-- FALLBACK LAYOUT — CSV upload apps + narrow viewports    -->
      <!-- ═══════════════════════════════════════════════════════ -->
      <template v-else>
      <!-- Pipeline info -->
      <div v-if="parsedNodes.length > 0" class="app-section" style="padding: 10px 16px;">
        <div class="d-flex flex-wrap align-center" style="gap: 6px;">
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Window: {{ pipelineInfo.window_size }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Stride: {{ pipelineInfo.stride }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Features: {{ pipelineInfo.n_features }}</v-chip>
          <v-chip size="x-small" :color="modeColor" variant="tonal">{{ appMode?.toUpperCase() || 'MODEL' }}</v-chip>
          <v-chip size="x-small" color="purple" variant="tonal">{{ appAlgorithm || 'model' }}</v-chip>
          <v-chip size="x-small" variant="outlined" style="font-size:9px">{{ parsedNodes.length }} nodes</v-chip>
        </div>
      </div>

      <!-- Pipeline info -->
      <div v-if="parsedNodes.length > 0" class="app-section" style="padding: 10px 16px;">
        <div class="d-flex flex-wrap align-center" style="gap: 6px;">
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Window: {{ pipelineInfo.window_size }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Stride: {{ pipelineInfo.stride }}</v-chip>
          <v-chip v-if="pipelineInfo" size="x-small" color="info" variant="tonal">Features: {{ pipelineInfo.n_features }}</v-chip>
          <v-chip size="x-small" :color="modeColor" variant="tonal">{{ appMode?.toUpperCase() || 'MODEL' }}</v-chip>
          <v-chip size="x-small" color="purple" variant="tonal">{{ appAlgorithm || 'model' }}</v-chip>
          <v-chip size="x-small" variant="outlined" style="font-size:9px">{{ parsedNodes.length }} nodes</v-chip>
        </div>
      </div>

      <!-- MQTT Live Stream Mode -->
      <div v-if="isLiveStream" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" :color="mqttConnected ? 'success' : 'grey'">mdi-access-point</v-icon>
          Live Stream (MQTT)
          <v-chip v-if="mqttConnected" size="x-small" color="success" variant="flat" class="ml-2">
            <v-icon start size="8">mdi-circle</v-icon> LIVE
          </v-chip>
          <v-chip v-else-if="mqttError" size="x-small" color="error" variant="tonal" class="ml-2">
            Error
          </v-chip>
        </div>

        <!-- Connection config -->
        <div class="d-flex align-center gap-2 mb-3" style="flex-wrap: wrap;">
          <v-text-field
            v-model="mqttBrokerUrl"
            label="Broker URL"
            variant="outlined"
            density="compact"
            hide-details
            style="max-width: 300px; font-size: 12px;"
            :disabled="mqttConnected"
          />
          <v-text-field
            v-model="mqttTopic"
            label="Topic"
            variant="outlined"
            density="compact"
            hide-details
            style="max-width: 200px; font-size: 12px;"
            :disabled="mqttConnected"
          />
          <v-btn
            v-if="!mqttConnected"
            color="success"
            variant="flat"
            size="small"
            @click="startLiveStream"
          >
            <v-icon start size="small">mdi-play</v-icon>
            Connect
          </v-btn>
          <v-btn
            v-else
            color="error"
            variant="tonal"
            size="small"
            @click="stopLiveStream"
          >
            <v-icon start size="small">mdi-stop</v-icon>
            Disconnect
          </v-btn>
        </div>

        <!-- Auto-record predictions toggle (inference apps incl. multi-model; not recorder) -->
        <div
          v-if="!isRecorderMode"
          class="d-flex align-center flex-wrap gap-2 mb-3"
        >
          <v-checkbox
            v-model="autoRecordPredictions"
            :disabled="mqttConnected"
            density="compact"
            hide-details
            color="error"
            class="mr-2"
          >
            <template #label>
              <span class="text-caption">
                <v-icon size="x-small" color="error" class="mr-1">mdi-record-circle-outline</v-icon>
                Auto-record predictions to CSV
              </span>
            </template>
          </v-checkbox>
          <v-select
            v-model="predictionRecordMode"
            :items="[
              { title: 'Per inference (1 row per prediction)', value: 'per_inference' },
              { title: 'Per sample (1 row per MQTT message)', value: 'per_sample' },
              { title: 'Full window (all samples per prediction)', value: 'full_window' },
            ]"
            :disabled="!autoRecordPredictions || mqttConnected"
            density="compact"
            variant="outlined"
            hide-details
            style="max-width: 280px; font-size: 12px;"
            label="Granularity"
          />
          <span class="text-caption text-medium-emphasis">
            Saves a CSV when you Disconnect.
          </span>
        </div>

        <!-- Error message -->
        <v-alert v-if="mqttError" type="error" variant="tonal" density="compact" class="mb-3" closable>
          {{ mqttError }}
        </v-alert>

        <!-- Auto-fallback warning banner (Layer 1) — surfaced here too so operators
             running in inference mode (not recorder) also see the miscoding warning. -->
        <v-alert
          v-if="sensorAutoFallbackActive"
          type="warning"
          variant="tonal"
          density="compact"
          class="mb-3"
          closable
          @click:close="sensorAutoFallbackActive = false"
        >
          Configured channels
          <strong>{{ (sensorAutoFallbackInfo?.configured || []).join(', ') }}</strong>
          didn't match any payload keys. Auto-matched by position from detected keys
          <strong>{{ (sensorAutoFallbackInfo?.detected || []).join(', ') }}</strong>.
          Update your App Builder MQTT node to future-proof.
        </v-alert>

        <!-- Fast Mode toggle (P2 Phase 3) — only for MQTT apps with a
             feature_extract node; CSV-upload / raw-mode apps never see it. -->
        <div v-if="fastModeAvailable" class="fast-mode-card mb-3">
          <div class="d-flex align-center flex-wrap gap-2">
            <v-icon size="18" :color="fastModeEnabled ? 'success' : 'grey'">mdi-lightning-bolt</v-icon>
            <span class="text-subtitle-2" style="font-weight: 600;">Fast Mode</span>
            <v-switch
              v-model="fastModeEnabled"
              color="success"
              hide-details
              density="compact"
              inset
              class="mt-0 ml-2"
            />
            <span
              class="fast-mode-badge ml-2"
              :class="{ on: fastModeEnabled }"
            >
              <v-icon size="12" class="mr-1">{{ fastModeEnabled ? 'mdi-lightning-bolt' : 'mdi-server' }}</v-icon>
              {{ fastModeEnabled ? 'Fast (browser)' : 'Server extraction' }}
            </span>
          </div>
          <div class="text-caption text-medium-emphasis mt-1" style="line-height: 1.35;">
            Compute features in your browser. Skips server-side extraction — inference is faster and doesn't load the server. (47 lightweight features only.)
          </div>
        </div>

        <!-- Live stats -->
        <div v-if="mqttConnected" class="d-flex flex-wrap gap-3 mb-3">
          <div class="live-stat">
            <div class="live-stat-label">Messages</div>
            <div class="live-stat-value">{{ mqttMessageCount.toLocaleString() }}</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Rate</div>
            <div class="live-stat-value">{{ mqttMessagesPerSec }}/s</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Buffer</div>
            <div class="live-stat-value">{{ sensorBufferLen }}/{{ liveWindowSize }}</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">Inferences</div>
            <div class="live-stat-value">{{ liveInferenceCount }}</div>
          </div>
          <div
            v-if="isRecordingPredictions"
            class="live-stat"
            style="background: rgba(244, 67, 54, 0.12); border-color: rgba(244, 67, 54, 0.4);"
          >
            <div class="live-stat-label" style="color: #ef5350;">
              <v-icon size="x-small" color="error" class="mr-1">mdi-circle</v-icon>Recording
            </div>
            <div class="live-stat-value" style="color: #ef5350;">
              {{ predictionRecordBuffer.length }} rows
            </div>
          </div>
        </div>

        <!-- Buffer progress bar -->
        <v-progress-linear
          v-if="mqttConnected"
          :model-value="sensorBufferProgress * 100"
          color="purple"
          height="6"
          rounded
          class="mb-3"
        />

        <!-- Live prediction (inference mode) -->
        <div v-if="livePrediction !== null && !isRecorderMode && !isMultiModelApp" class="live-prediction">
          <div class="live-prediction-label">Latest Prediction</div>
          <div class="live-prediction-value" :style="{ color: modeColor }">
            {{ livePrediction }}
          </div>
          <div v-if="liveLastUpdated" class="live-prediction-time">
            {{ liveLastUpdatedText }}
          </div>
        </div>

        <!-- Actual Column Selector (regression MQTT apps) -->
        <div v-if="mqttConnected && !isRecorderMode && !isMultiModelApp && appMode === 'regression' && (autoDetectedChannels.length > 0 || liveChannels.length > 0)"
             class="actual-col-selector mb-3">
          <v-select
            v-model="liveActualColumn"
            :items="['(none)', ...(autoDetectedChannels.length > 0 ? autoDetectedChannels : liveChannels)]"
            label="Compare with column (actual)"
            density="compact"
            variant="outlined"
            hide-details
            style="max-width: 300px"
            prepend-inner-icon="mdi-chart-line"
          />
        </div>

        <!-- Signal Recorder Mode -->
        <div v-if="isRecorderMode && mqttConnected" class="recorder-section">
          <!-- Auto-fallback warning banner (Layer 1) -->
          <v-alert
            v-if="sensorAutoFallbackActive"
            type="warning"
            variant="tonal"
            density="compact"
            closable
            class="mb-3"
            @click:close="sensorAutoFallbackActive = false"
          >
            Configured channels
            <strong>{{ (sensorAutoFallbackInfo?.configured || []).join(', ') }}</strong>
            didn't match any payload keys. Auto-matched by position from detected keys
            <strong>{{ (sensorAutoFallbackInfo?.detected || []).join(', ') }}</strong>.
            Update your App Builder MQTT node to future-proof.
          </v-alert>

          <!-- Show raw MQTT toggle (Layer 3) -->
          <div class="raw-mqtt-controls">
            <v-btn
              size="x-small"
              variant="tonal"
              :prepend-icon="showRawMqtt ? 'mdi-eye-off-outline' : 'mdi-eye-outline'"
              @click="showRawMqtt = !showRawMqtt"
            >
              {{ showRawMqtt ? 'Hide raw MQTT' : 'Show raw MQTT' }}
            </v-btn>
          </div>
          <div v-if="showRawMqtt && rawMqttBuffer.length > 0" class="raw-mqtt-wrap">
            <pre
              v-for="(msg, i) in rawMqttBuffer"
              :key="'raw-fb-' + i"
              class="raw-mqtt-block"
            >{{ msg.length > 500 ? msg.slice(0, 500) + '…' : msg }}</pre>
          </div>

          <!-- Live preview panel: always shown while connected, even before Record -->
          <div v-if="liveChannels.length > 0" class="preview-panel">
            <div class="preview-header">
              <span class="text-caption font-weight-bold text-medium-emphasis">LIVE PREVIEW</span>
              <v-select
                v-model="previewWindowSec"
                :items="[5, 10, 30, 60, 120]"
                label="Preview window (s)"
                variant="outlined"
                density="compact"
                hide-details
                style="max-width: 180px; font-size: 11px;"
              />
            </div>

            <!-- Per-channel numeric readout -->
            <div class="preview-cards">
              <div v-for="ch in liveChannels" :key="'pv-' + ch" class="preview-card">
                <div class="preview-card-label">{{ ch }}</div>
                <div class="preview-card-value">{{ fmtPreview(previewStats[ch]?.latest) }}</div>
                <div class="preview-card-stats">
                  <span>Min: {{ fmtPreview(previewStats[ch]?.min) }}</span>
                  <span class="preview-card-stats-sep">·</span>
                  <span>Max: {{ fmtPreview(previewStats[ch]?.max) }}</span>
                  <span class="preview-card-stats-sep">·</span>
                  <span>Mean: {{ fmtPreview(previewStats[ch]?.mean) }}</span>
                </div>
              </div>
            </div>

            <!-- Live oscilloscope chart -->
            <div v-if="previewBuffer.length > 1" class="preview-chart">
              <Line :data="previewChartData" :options="previewChartOptions" />
            </div>
            <div v-else class="preview-chart-empty">
              Waiting for samples…
            </div>
          </div>

          <!-- Label buttons -->
          <div class="recorder-label-header">
            <span class="text-caption font-weight-bold text-medium-emphasis">CURRENT LABEL</span>
            <v-chip v-if="recorderState.recording" size="x-small" color="error" variant="flat" class="ml-2">
              <v-icon start size="8">mdi-circle</v-icon> REC
            </v-chip>
          </div>
          <div class="recorder-labels">
            <button
              v-for="lbl in recorderLabels"
              :key="lbl"
              class="recorder-label-btn"
              :class="{ active: recorderState.currentLabel === lbl }"
              @click="recorderState.currentLabel = lbl"
            >{{ lbl }}</button>
          </div>

          <!-- Custom label input -->
          <div class="d-flex align-center gap-2 mt-2 mb-3">
            <v-text-field
              v-model="recorderCustomLabel"
              label="Add custom label"
              variant="outlined"
              density="compact"
              hide-details
              style="max-width: 200px; font-size: 11px;"
              @keydown.enter="addCustomLabel"
            />
            <v-btn size="x-small" variant="tonal" @click="addCustomLabel" :disabled="!recorderCustomLabel.trim()">
              <v-icon size="small">mdi-plus</v-icon>
            </v-btn>
          </div>

          <!-- Recording controls -->
          <div class="d-flex align-center gap-2 mb-3">
            <v-btn
              v-if="!recorderState.recording"
              color="error"
              variant="flat"
              size="small"
              :disabled="!recorderState.currentLabel"
              @click="startRecording"
            >
              <v-icon start size="small">mdi-record</v-icon>
              Start Recording
            </v-btn>
            <v-btn
              v-else
              color="warning"
              variant="flat"
              size="small"
              @click="stopRecording"
            >
              <v-icon start size="small">mdi-stop</v-icon>
              Stop Recording
            </v-btn>
            <v-btn
              v-if="recorderState.samples.length > 0"
              color="success"
              variant="tonal"
              size="small"
              @click="downloadRecordedCSV"
            >
              <v-icon start size="small">mdi-download</v-icon>
              Download CSV ({{ recorderState.samples.length }} samples)
            </v-btn>
            <v-btn
              v-if="recorderState.samples.length > 0 && !recorderState.recording"
              variant="text"
              size="small"
              color="error"
              @click="clearRecording"
            >
              Clear
            </v-btn>
          </div>

          <!-- Recording stats -->
          <div class="d-flex flex-wrap gap-3 mb-2">
            <div class="live-stat">
              <div class="live-stat-label">Samples</div>
              <div class="live-stat-value">{{ recorderState.samples.length }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Duration</div>
              <div class="live-stat-value">{{ recorderDuration }}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Labels</div>
              <div class="live-stat-value">{{ recorderLabelCounts }}</div>
            </div>
          </div>

          <!-- Label timeline -->
          <div v-if="recorderState.samples.length > 0" class="recorder-timeline">
            <div
              v-for="(seg, i) in recorderSegments"
              :key="i"
              class="recorder-segment"
              :style="{ flex: seg.count, background: seg.color }"
              :title="`${seg.label}: ${seg.count} samples`"
            />
          </div>
        </div>
      </div>

      <!-- MQTT Test Publisher (logged-in users only, MQTT apps only).
           Wrapped in a collapsed accordion so it stays out of the way of the
           primary live-inference view — one click to open when needed. -->
      <div v-if="isAuthenticated && isLiveStream" class="app-section" style="padding: 10px 16px;">
        <v-expansion-panels variant="accordion">
          <v-expansion-panel>
            <v-expansion-panel-title>
              <v-icon start size="small" class="mr-2">mdi-test-tube</v-icon>
              Test Publisher (simulate sensor data)
            </v-expansion-panel-title>
            <v-expansion-panel-text>
              <MqttTestPublisher />
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
      </div>

      <!-- CSV Upload Mode -->
      <div v-else class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="blue">mdi-upload</v-icon>
          Upload Data
        </div>

        <!-- Expected CSV format -->
        <div v-if="expectedColumns.length > 0" class="expected-format">
          <div class="text-caption text-medium-emphasis mb-1">
            <v-icon size="12" class="mr-1">mdi-information-outline</v-icon>
            Expected CSV columns:
          </div>
          <div class="d-flex flex-wrap" style="gap: 3px;">
            <span v-for="col in expectedColumns" :key="col" class="expected-col">{{ col }}</span>
          </div>
        </div>

        <div class="app-upload-area">
          <input type="file" ref="fileInput" accept=".csv" @change="onFileSelect" style="display:none" />
          <div v-if="!selectedFile" class="app-dropzone" @click="$refs.fileInput.click()">
            <v-icon size="32" color="grey">mdi-file-upload-outline</v-icon>
            <div class="text-caption text-medium-emphasis mt-2">Click to upload CSV file</div>
          </div>
          <div v-else class="app-file-selected">
            <v-icon size="16" color="success">mdi-file-check</v-icon>
            <span>{{ selectedFile.name }} ({{ (selectedFile.size / 1024).toFixed(1) }} KB)</span>
            <v-btn icon size="x-small" variant="text" @click="clearFile">
              <v-icon size="14">mdi-close</v-icon>
            </v-btn>
          </div>
          <v-btn
            :disabled="!selectedFile || running"
            :loading="running"
            color="purple"
            variant="flat"
            class="mt-3"
            @click="runPipeline"
          >
            <v-icon start>mdi-play</v-icon>
            Run Inference
          </v-btn>
        </div>
      </div>

      <!-- Multi-Model Comparison Results -->
      <div v-if="result && result.multi_model" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="amber">mdi-compare-horizontal</v-icon>
          Multi-Model Comparison
          <v-chip size="x-small" color="amber" variant="tonal" class="ml-2">
            {{ Object.keys(result.models || {}).length }} models
          </v-chip>
        </div>

        <!-- Metrics comparison table -->
        <v-table density="compact" class="mb-4">
          <thead>
            <tr>
              <th>Model</th>
              <th>Algorithm</th>
              <template v-if="result.actual && result.mode === 'regression'">
                <th class="text-center">R²</th>
                <th class="text-center">RMSE</th>
                <th class="text-center">MAE</th>
              </template>
              <template v-else-if="result.actual && result.mode === 'classification'">
                <th class="text-center">Accuracy</th>
                <th class="text-center">Precision</th>
                <th class="text-center">F1</th>
              </template>
              <template v-else>
                <th class="text-center">Latest Prediction</th>
                <th class="text-center">Windows</th>
              </template>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, eid) in result.models" :key="eid"
                :class="{
                  'bg-amber-darken-4': !m.error && result.actual && (result.mode === 'regression' ? m.r2 === bestR2 : m.accuracy === bestAccuracy),
                  'bg-red-darken-4': !!m.error
                }">
              <td class="font-weight-medium">
                {{ m.name }}
                <v-icon v-if="m.error" size="14" color="error" class="ml-1">mdi-alert-circle</v-icon>
                <v-icon v-else-if="result.actual && ((result.mode === 'regression' && m.r2 === bestR2) || (result.mode === 'classification' && m.accuracy === bestAccuracy))"
                        size="14" color="amber" class="ml-1">mdi-trophy</v-icon>
              </td>
              <td class="text-caption">
                <span v-if="m.error" class="text-error text-caption">{{ m.error }}</span>
                <span v-else>{{ m.algorithm }}</span>
              </td>
              <template v-if="m.error">
                <!-- Error: skip metric columns -->
              </template>
              <template v-else-if="result.actual && result.mode === 'regression'">
                <td class="text-center" :style="{ color: m.r2 > 0.8 ? '#34d399' : m.r2 > 0.5 ? '#fbbf24' : '#f87171' }">
                  {{ m.r2 != null ? m.r2.toFixed(4) : '-' }}
                </td>
                <td class="text-center">{{ m.rmse != null ? m.rmse.toFixed(4) : '-' }}</td>
                <td class="text-center">{{ m.mae != null ? m.mae.toFixed(4) : '-' }}</td>
              </template>
              <template v-else-if="result.actual && result.mode === 'classification'">
                <td class="text-center" :style="{ color: m.accuracy > 0.9 ? '#34d399' : m.accuracy > 0.7 ? '#fbbf24' : '#f87171' }">
                  {{ m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '-' }}
                </td>
                <td class="text-center">{{ m.precision != null ? (m.precision * 100).toFixed(1) + '%' : '-' }}</td>
                <td class="text-center">{{ m.f1 != null ? (m.f1 * 100).toFixed(1) + '%' : '-' }}</td>
              </template>
              <template v-else>
                <td class="text-center" :style="{ color: '#34d399' }">
                  {{ m.predictions?.length > 0 ? m.predictions[m.predictions.length - 1] : '-' }}
                </td>
                <td class="text-center">{{ m.count || 0 }}</td>
              </template>
            </tr>
          </tbody>
        </v-table>

        <!-- Multi-model chart (regression) -->
        <div v-if="result.mode === 'regression' && multiChartData.length > 0" class="chart-container">
          <div class="chart-header">
            <span class="chart-title-text">Model Predictions Comparison</span>
          </div>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight + 20}`" class="prediction-chart">
            <line v-for="i in 4" :key="'g'+i"
              :x1="chartPadding" :y1="chartPadding + (i-1) * (chartInnerH / 3)"
              :x2="chartWidth - chartPadding" :y2="chartPadding + (i-1) * (chartInnerH / 3)"
              stroke="#21262d" stroke-width="0.5" />
            <!-- Actual line -->
            <path v-if="multiActualPath" :d="multiActualPath" fill="none" stroke="#22d3ee" stroke-width="2" />
            <!-- Model lines -->
            <path v-for="(mp, idx) in multiModelPaths" :key="'mp'+idx"
              :d="mp.path" fill="none" :stroke="mp.color" stroke-width="1.5"
              :stroke-dasharray="idx > 0 ? '4,2' : 'none'" />
          </svg>
          <div class="chart-legend-items" style="margin-top:4px; flex-wrap: wrap;">
            <span v-if="result.actual" class="chart-legend-dot" style="background: #22d3ee"></span>
            <span v-if="result.actual" class="chart-legend-label">Actual</span>
            <template v-for="(mp, idx) in multiModelPaths" :key="'leg'+idx">
              <span class="chart-legend-dot" :style="{ background: mp.color }"></span>
              <span class="chart-legend-label">{{ mp.name }}</span>
            </template>
          </div>
        </div>

        <!-- Multi-model classification timeline chart -->
        <div v-if="result.mode === 'classification' && classTimelineRows.length > 0" class="chart-container">
          <div class="chart-header">
            <span class="chart-title-text">Predictions Timeline</span>
          </div>
          <svg :viewBox="`0 0 ${classChartW} ${classChartH}`" style="width: 100%; min-width: 500px;" class="prediction-chart">
            <!-- Background grid -->
            <line v-for="i in 5" :key="'cg'+i"
              :x1="classPadL" :y1="20 + (i-1) * (classBandsH / 4)"
              :x2="classChartW - 10" :y2="20 + (i-1) * (classBandsH / 4)"
              stroke="#21262d" stroke-width="0.5" />

            <!-- Class bands (one row per model + actual if present) -->
            <g v-for="(row, ri) in classTimelineRows" :key="'row'+ri">
              <!-- Row label -->
              <text :x="classPadL - 6" :y="row.y + row.h / 2 + 4"
                    text-anchor="end" fill="#94a3b8" font-size="10" font-family="monospace">
                {{ row.label }}
              </text>
              <!-- Class rectangles -->
              <rect v-for="(seg, si) in row.segments" :key="'seg'+ri+'-'+si"
                :x="seg.x" :y="row.y" :width="seg.w" :height="row.h"
                :fill="seg.color" fill-opacity="0.7"
                :stroke="seg.mismatch ? '#f87171' : 'none'" :stroke-width="seg.mismatch ? 1 : 0">
                <title>{{ row.label }} | Window {{ seg.idx + 1 }}: {{ seg.label }}</title>
              </rect>
            </g>

            <!-- Signal plot below the bands -->
            <g v-if="classSignalPath">
              <line :x1="classPadL" :y1="classBandsBottom + 4" :x2="classChartW - 10" :y2="classBandsBottom + 4"
                    stroke="#30363d" stroke-width="0.5" />
              <text :x="classPadL - 6" :y="classBandsBottom + classSignalH / 2 + 10"
                    text-anchor="end" fill="#94a3b8" font-size="10" font-family="monospace">Signal</text>
              <path :d="classSignalPath" fill="none" stroke="#22d3ee" stroke-width="1" stroke-opacity="0.8" />
            </g>

            <!-- X-axis label -->
            <text :x="classChartW / 2" :y="classChartH - 4" text-anchor="middle"
                  fill="#8b949e" font-size="9">Window Index (time →)</text>
          </svg>

          <!-- Class legend -->
          <div class="chart-legend-items" style="margin-top: 6px; flex-wrap: wrap; gap: 4px 12px;">
            <template v-for="(color, label) in classColorMap" :key="'lg'+label">
              <span class="chart-legend-dot" :style="{ background: color, opacity: 0.7 }"></span>
              <span class="chart-legend-label">{{ label }}</span>
            </template>
            <template v-if="hasMismatch">
              <span style="display:inline-block; width: 10px; height: 10px; border: 1px solid #f87171; border-radius: 2px; margin-left: 8px;"></span>
              <span class="chart-legend-label" style="color: #f87171;">= mismatch with actual</span>
            </template>
          </div>
        </div>

        <!-- Per-window predictions table -->
        <div class="table-toggle mt-3" @click="showMultiTable = !showMultiTable">
          <v-icon size="14">{{ showMultiTable ? 'mdi-chevron-down' : 'mdi-chevron-right' }}</v-icon>
          <span>{{ showMultiTable ? 'Hide' : 'Show' }} per-window predictions ({{ result.num_windows || '?' }} windows)</span>
        </div>
        <div v-if="showMultiTable" class="result-table-wrap">
          <table class="result-table">
            <thead>
              <tr>
                <th>#</th>
                <th v-if="result.actual">Actual</th>
                <th v-for="(m, eid) in result.models" :key="'th'+eid">{{ m.name }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="i in Math.min(result.num_windows || 0, tableMaxRows)" :key="i">
                <td>{{ i }}</td>
                <td v-if="result.actual" style="color: #22d3ee;">{{ result.actual[i-1] ?? '-' }}</td>
                <td v-for="(m, eid) in result.models" :key="'td'+eid+i"
                    :style="{ color: result.actual && String(result.actual[i-1]).toLowerCase() === String(m.predictions?.[i-1] ?? '').toLowerCase() ? '#34d399' : result.actual ? '#f87171' : '#e6edf3' }">
                  {{ m.predictions?.[i-1] ?? '-' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Download CSV -->
        <v-btn color="success" variant="tonal" size="small" class="mt-3" @click="downloadMultiModelCsv">
          <v-icon start size="small">mdi-download</v-icon>
          Download Comparison CSV
        </v-btn>
      </div>

      <!-- Single Model Results section -->
      <div v-else-if="result" class="app-section">
        <div class="app-section-title">
          <v-icon size="16" color="success">mdi-chart-line</v-icon>
          Results
        </div>

        <!-- Regression results -->
        <div v-if="appMode === 'regression'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Predictions</div>
              <div class="result-stat-value">{{ result.predictions?.length || 0 }}</div>
            </div>
            <div v-if="result.mean !== undefined" class="result-stat">
              <div class="result-stat-label">Mean</div>
              <div class="result-stat-value">{{ result.mean?.toFixed(4) }}</div>
            </div>
            <div v-if="result.std !== undefined" class="result-stat">
              <div class="result-stat-label">Std</div>
              <div class="result-stat-value">{{ result.std?.toFixed(4) }}</div>
            </div>
            <div v-if="result.r2 !== undefined" class="result-stat">
              <div class="result-stat-label">R²</div>
              <div class="result-stat-value" :style="{ color: result.r2 > 0.8 ? '#34d399' : result.r2 > 0.5 ? '#fbbf24' : '#f87171' }">{{ result.r2?.toFixed(4) }}</div>
            </div>
            <div v-else-if="result.min !== undefined" class="result-stat">
              <div class="result-stat-label">Range</div>
              <div class="result-stat-value">{{ result.min?.toFixed(1) }} – {{ result.max?.toFixed(1) }}</div>
            </div>
            <div v-if="result.rmse !== undefined" class="result-stat">
              <div class="result-stat-label">RMSE</div>
              <div class="result-stat-value">{{ result.rmse?.toFixed(4) }}</div>
            </div>
          </div>

          <!-- Line Chart -->
          <div v-if="chartData.length > 0" class="chart-container">
            <div class="chart-header">
              <span class="chart-title-text">{{ actualData.length > 0 ? 'Actual vs Predicted' : 'Predictions over Time' }}</span>
            </div>
            <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" class="prediction-chart">
              <!-- Grid lines -->
              <line v-for="i in 4" :key="'g'+i"
                :x1="chartPadding" :y1="chartPadding + (i-1) * (chartInnerH / 3)"
                :x2="chartWidth - chartPadding" :y2="chartPadding + (i-1) * (chartInnerH / 3)"
                stroke="#21262d" stroke-width="0.5" />
              <!-- Y axis labels -->
              <text v-for="i in 4" :key="'y'+i"
                :x="chartPadding - 4" :y="chartPadding + (i-1) * (chartInnerH / 3) + 3"
                text-anchor="end" fill="#8b949e" font-size="8" font-family="monospace">
                {{ chartYLabel(i-1) }}
              </text>
              <!-- Area fill -->
              <path :d="chartAreaPath" fill="url(#pred-gradient)" />
              <!-- Actual line (cyan solid) -->
              <path v-if="actualLinePath" :d="actualLinePath" fill="none" stroke="#22d3ee" stroke-width="1.5" />
              <!-- Prediction line (purple, dashed if actual shown) -->
              <path :d="chartLinePath" fill="none" stroke="#a78bfa" stroke-width="1.5" :stroke-dasharray="actualLinePath ? '4,2' : 'none'" />
              <defs>
                <linearGradient id="pred-gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.15" />
                  <stop offset="100%" stop-color="#a78bfa" stop-opacity="0" />
                </linearGradient>
              </defs>
            </svg>
            <div class="chart-legend-items" style="margin-top:8px">
              <span v-if="actualData.length > 0" class="chart-legend-dot" style="background: #22d3ee"></span>
              <span v-if="actualData.length > 0" class="chart-legend-label">Actual</span>
              <span class="chart-legend-dot" style="background: #a78bfa"></span>
              <span class="chart-legend-label">Predicted</span>
            </div>
          </div>

          <!-- Data Table (collapsible) -->
          <div class="table-toggle" @click="showTable = !showTable">
            <v-icon size="14">{{ showTable ? 'mdi-chevron-down' : 'mdi-chevron-right' }}</v-icon>
            <span>{{ showTable ? 'Hide' : 'Show' }} data table ({{ result.predictions?.length || 0 }} rows)</span>
          </div>
          <div v-if="showTable" class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Prediction</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions || []).slice(0, tableMaxRows)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: modeColor }">{{ typeof val === 'number' ? val.toFixed(4) : val }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Classification results -->
        <div v-else-if="appMode === 'classification'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Windows</div>
              <div class="result-stat-value">{{ result.count || 0 }}</div>
            </div>
          </div>
          <div class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Label</th><th>Confidence</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions_full || result.predictions || []).slice(0, tableMaxRows)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: modeColor }">{{ val.label || val }}</td>
                  <td>{{ val.confidence ? (val.confidence * 100).toFixed(1) + '%' : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Anomaly results -->
        <div v-else-if="appMode === 'anomaly'" class="app-results">
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Total Windows</div>
              <div class="result-stat-value">{{ result.count || 0 }}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Anomalies</div>
              <div class="result-stat-value" style="color: #f87171">{{ result.anomaly_count || 0 }}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Normal</div>
              <div class="result-stat-value" style="color: #34d399">{{ result.normal_count || 0 }}</div>
            </div>
          </div>
          <div class="result-table-wrap">
            <table class="result-table">
              <thead><tr><th>#</th><th>Label</th><th>Score</th></tr></thead>
              <tbody>
                <tr v-for="(val, i) in (result.predictions_full || result.predictions || []).slice(0, tableMaxRows)" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td :style="{ color: (val.label || val) === 'anomaly' ? '#f87171' : '#34d399' }">
                    {{ val.label || val }}
                  </td>
                  <td>{{ val.score ? val.score.toFixed(4) : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Raw JSON fallback -->
        <div v-else class="app-results">
          <pre class="result-json">{{ JSON.stringify(result, null, 2) }}</pre>
        </div>
      </div>

      <!-- Run error -->
      <v-alert v-if="runError" type="error" variant="tonal" class="mt-4" closable @click:close="runError = null">
        {{ runError }}
      </v-alert>

      <!-- Upload new file button after results — CSV mode only.
           In live-stream mode the input is the MQTT connection, so the
           equivalent action is Disconnect / Reconnect, not File Upload. -->
      <div v-if="result && !isLiveStream" class="text-center mt-4">
        <v-btn variant="outlined" color="purple" @click="clearFile">
          <v-icon start size="small">mdi-upload</v-icon>
          Upload New File
        </v-btn>
      </div>
      </template>
      <!-- /fallback layout -->
    </div>

    <!-- Test Publisher dialog (opened from rail button). Auth-gated at the
         outer v-if. `eager` keeps the panel mounted even when the dialog is
         closed, so an active publish session survives the user closing the
         dialog to watch inference charts — reopening resumes the same run
         instead of terminating the WebSocket. -->
    <v-dialog
      v-if="isAuthenticated && isLiveStream"
      v-model="showTestPublisherDialog"
      max-width="720"
      scrollable
      eager
    >
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon start>mdi-test-tube</v-icon>
          Test Publisher
          <v-spacer />
          <v-btn icon size="small" variant="text" @click="showTestPublisherDialog = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>
        <v-card-text style="max-height: 70vh;">
          <MqttTestPublisher />
        </v-card-text>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import api from '@/services/api'
import { Line } from 'vue-chartjs'
import MqttTestPublisher from '@/components/MqttTestPublisher.vue'
import { useAuthStore } from '@/stores/auth'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

// mqtt loaded dynamically only when needed (live stream mode)
let mqtt = null

const route = useRoute()
const slug = computed(() => route.params.slug)
const isStandalone = computed(() => route.meta?.standalone === true)

// Auth store — used to gate the MQTT Test Publisher panel to logged-in users only.
// Public/anonymous viewers of /standalone/<slug> won't see the panel.
const authStore = useAuthStore()
const isAuthenticated = computed(() => authStore.isAuthenticated)

// Test Publisher dialog — opened from the rail button (rail is too narrow to
// host the full panel inline).
const showTestPublisherDialog = ref(false)

const loading = ref(true)
const error = ref(null)
const appData = ref({})
const selectedFile = ref(null)
const running = ref(false)
const result = ref(null)
const showTable = ref(false)

// ── Dashboard layout state (production monitor) ─────────
// Track viewport width so we can fall back to the old single-column layout
// on smaller screens (developers/mobile still need a usable view).
const viewportWidth = ref(typeof window !== 'undefined' ? window.innerWidth : 1920)
function onResize() {
  viewportWidth.value = window.innerWidth
}
const isWideScreen = computed(() => viewportWidth.value >= 1200)

// Left rail collapses to an icon strip after Connect. A small chevron
// expands it back to the full 280 px so operators can change settings.
const railCollapsed = ref(false)

// Fullscreen API — toggled by the ⛶ button in the header.
const isFullscreen = ref(false)
function updateFullscreenState() {
  isFullscreen.value = !!document.fullscreenElement
}
async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen()
    } else {
      await document.exitFullscreen()
    }
  } catch (e) {
    // Some browsers throw if the gesture is stale; ignore silently.
    console.warn('Fullscreen toggle failed:', e)
  }
}

// Chart dimensions
const chartWidth = 700
const chartHeight = 200
const chartPadding = 40
const chartInnerW = chartWidth - chartPadding * 2
const chartInnerH = chartHeight - chartPadding * 2

// Chart data (downsampled predictions for rendering)
const chartData = computed(() => {
  const preds = result.value?.predictions || []
  if (preds.length === 0) return []
  const nums = preds.filter(v => typeof v === 'number')
  if (nums.length === 0) return []
  // Downsample to max 200 points
  if (nums.length <= 200) return nums
  const step = nums.length / 200
  return Array.from({ length: 200 }, (_, i) => nums[Math.floor(i * step)])
})

const chartMinY = computed(() => {
  if (allChartValues.value.length === 0) return 0
  return Math.min(...allChartValues.value) - (Math.max(...allChartValues.value) - Math.min(...allChartValues.value)) * 0.05
})
const chartMaxY = computed(() => {
  if (allChartValues.value.length === 0) return 1
  return Math.max(...allChartValues.value) + (Math.max(...allChartValues.value) - Math.min(...allChartValues.value)) * 0.05
})
const chartRangeY = computed(() => chartMaxY.value - chartMinY.value || 1)

function chartX(i) {
  return chartPadding + (i / (chartData.value.length - 1 || 1)) * chartInnerW
}
function chartY(v) {
  return chartPadding + chartInnerH - ((v - chartMinY.value) / chartRangeY.value) * chartInnerH
}
function chartYLabel(idx) {
  const v = chartMaxY.value - (idx / 3) * chartRangeY.value
  return v.toFixed(1)
}

// Actual values (downsampled same way)
const actualData = computed(() => {
  const actuals = result.value?.actual || []
  if (actuals.length === 0) return []
  if (actuals.length <= 200) return actuals
  const step = actuals.length / 200
  return Array.from({ length: 200 }, (_, i) => actuals[Math.floor(i * step)])
})

// Adjust Y range to include both predicted and actual
const allChartValues = computed(() => {
  return [...chartData.value, ...actualData.value].filter(v => typeof v === 'number')
})

const chartLinePath = computed(() => {
  if (chartData.value.length === 0) return ''
  return chartData.value.map((v, i) => `${i === 0 ? 'M' : 'L'}${chartX(i).toFixed(1)},${chartY(v).toFixed(1)}`).join(' ')
})
const actualLinePath = computed(() => {
  if (actualData.value.length === 0) return ''
  const n = actualData.value.length
  const maxPts = chartData.value.length || n
  return actualData.value.map((v, i) => {
    const x = chartPadding + (i / (n - 1 || 1)) * chartInnerW
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${chartY(v).toFixed(1)}`
  }).join(' ')
})

const chartAreaPath = computed(() => {
  if (chartData.value.length === 0) return ''
  const n = chartData.value.length
  return `${chartLinePath.value} L${chartX(n-1).toFixed(1)},${chartHeight - chartPadding} L${chartX(0).toFixed(1)},${chartHeight - chartPadding} Z`
})
const runError = ref(null)

const MODE_COLORS = {
  anomaly: '#f87171',
  classification: '#34d399',
  regression: '#a78bfa',
}

// Parse nodes once (API may return string or array)
const parsedNodes = computed(() => {
  let nodes = appData.value.nodes || []
  if (typeof nodes === 'string') {
    try { nodes = JSON.parse(nodes) } catch { nodes = [] }
  }
  return Array.isArray(nodes) ? nodes : []
})

const appMode = computed(() => {
  return appData.value.mode || 'regression'
})

// Max rows to render in result tables; falls back to 100 for backwards compatibility
// (older apps had no output.table node config or set max_rows to a smaller default).
const tableMaxRows = computed(() => {
  const tableNode = parsedNodes.value.find(n => n.type === 'output.table')
  const configured = Number(tableNode?.config?.max_rows)
  return Number.isFinite(configured) && configured > 0 ? configured : 100
})

const modeColor = computed(() => MODE_COLORS[appMode.value] || '#94a3b8')

// Show the new two-pane monitor layout for MQTT-driven apps at wide widths.
// CSV upload apps and narrow viewports fall back to the classic stacked layout.
const dashboardMode = computed(() => {
  return isWideScreen.value && (isLiveStream.value || isRecorderMode.value)
})

// ── Dashboard display preferences (persisted per-app in localStorage) ───────
const displayPrecision = ref(3)
const displayFontSize = ref('L') // 'S' | 'M' | 'L' | 'XL'
const displayView = ref('chart')  // 'chart' | 'table'
const DASH_TABLE_ROWS = 20

// P1 Layer 1 — auto-fallback banner state (referenced by templates above; kept
// here so the ref exists before display-prefs load/save reference it).
const sensorAutoFallbackActive = ref(false)
const sensorAutoFallbackInfo = ref(null) // { configured: string[], detected: string[] } | null

// P1 Layer 3 — Show raw MQTT toggle + ring buffer (max 3 messages).
const showRawMqtt = ref(false)
const rawMqttBuffer = ref([]) // string[] — newest first, max 3

function _prefsKey() { return `dash-prefs-${slug.value || 'app'}` }
function loadDisplayPrefs() {
  const key = _prefsKey()
  const stored = localStorage.getItem(key)
  const mode = appMode.value
  const defaultView = (mode === 'classification') ? 'table' : 'chart'
  if (stored) {
    try {
      const p = JSON.parse(stored)
      displayPrecision.value = typeof p.precision === 'number' ? p.precision : 3
      displayFontSize.value = p.fontSize || 'L'
      displayView.value = p.view || defaultView
      showRawMqtt.value = !!p.showRawMqtt
      return
    } catch { /* fall through */ }
  }
  displayPrecision.value = 3
  displayFontSize.value = 'L'
  displayView.value = defaultView
  showRawMqtt.value = false
}
function saveDisplayPrefs() {
  const key = _prefsKey()
  localStorage.setItem(key, JSON.stringify({
    precision: displayPrecision.value,
    fontSize: displayFontSize.value,
    view: displayView.value,
    showRawMqtt: showRawMqtt.value,
  }))
}
watch(slug, () => { if (slug.value) loadDisplayPrefs() })
watch(() => appMode.value, () => { if (slug.value) loadDisplayPrefs() }, { immediate: false })
watch([displayPrecision, displayFontSize, displayView, showRawMqtt], () => { saveDisplayPrefs() })

// Precision-formatted latest prediction (used in the dashboard card).
const displayedPrediction = computed(() => {
  const v = livePrediction.value
  if (v === null || v === undefined) return ''
  if (typeof v === 'number' && Number.isFinite(v)) {
    return v.toFixed(displayPrecision.value)
  }
  return String(v)
})

function _fmtNum(v, decimals) {
  if (v === null || v === undefined || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return String(v)
  return n.toFixed(decimals ?? displayPrecision.value)
}

// Rows for the dashboard's Table view — newest first, up to DASH_TABLE_ROWS.
const dashboardTableRows = computed(() => {
  const preds = livePredictionHistory.value || []
  const actuals = liveActualHistory.value || []
  const full = livePredictionHistoryFull.value || []
  const startTs = liveLastUpdated.value ? Number(liveLastUpdated.value) : Date.now()
  const total = preds.length
  const rows = []
  const end = total
  const begin = Math.max(0, total - DASH_TABLE_ROWS)
  for (let i = end - 1; i >= begin; i--) {
    const pred = preds[i]
    const actual = actuals[i]
    let confidence = null
    const fullEntry = full[i]
    if (fullEntry && typeof fullEntry === 'object' && 'confidence' in fullEntry
        && typeof fullEntry.confidence === 'number') {
      confidence = fullEntry.confidence
    }
    const isNumericPred = typeof pred === 'number' && Number.isFinite(pred)
    rows.push({
      i,
      // Approximate timestamp: infer from position back from last update.
      timeText: new Date(startTs - (total - 1 - i) * 1000).toLocaleTimeString(),
      predText: isNumericPred ? _fmtNum(pred) : String(pred ?? '—'),
      actualText: actual !== undefined ? _fmtNum(actual) : '—',
      hasActual: actual !== undefined && actual !== null,
      confidenceText: confidence != null ? (confidence * 100).toFixed(1) + '%' : '—',
      hasConfidence: confidence != null,
      errorText: (isNumericPred && typeof actual === 'number' && Number.isFinite(actual))
        ? _fmtNum(actual - pred)
        : '—',
    })
  }
  return rows
})

const appAlgorithm = computed(() => {
  return appData.value.algorithm || ''
})

// Pipeline info from nodes
const pipelineInfo = computed(() => {
  const windowNode = parsedNodes.value.find(n => n.type === 'transform.window')
  const featNode = parsedNodes.value.find(n => n.type === 'transform.feature_extract')
  if (!windowNode) return null
  return {
    window_size: windowNode.config?.window_size || '?',
    stride: windowNode.config?.step || windowNode.config?.stride || '?',
    n_features: featNode?.config?.features?.length || '?',
    algorithm: appData.value.algorithm || 'model',
  }
})

// Expected CSV columns from model's sensor_columns
const expectedColumns = computed(() => {
  const nodes = appData.value.nodes || []
  const normNode = nodes.find(n => n.type === 'transform.normalize')
  // The model endpoint's pipeline_config has sensor_columns
  // These are returned by the by-slug API
  return appData.value.sensor_columns || []
})

onMounted(async () => {
  try {
    const resp = await api.get(`/api/app-builder/apps/by-slug/${slug.value}`)
    appData.value = resp.data
  } catch (e) {
    // Try getting by ID as fallback
    try {
      const resp = await api.get(`/api/app-builder/apps/${slug.value}`)
      appData.value = resp.data
    } catch {
      error.value = 'This app does not exist or has not been published.'
    }
  }
  // Initialize MQTT config from pipeline nodes
  if (isLiveStream.value) {
    const cfg = liveStreamConfig.value
    // Auto-resolve broker URL: replace 'localhost' with actual server hostname
    // so the app works when opened from other machines on the LAN
    let brokerUrl = cfg.broker_url || 'ws://localhost:9001/mqtt'
    if (brokerUrl.includes('localhost') || brokerUrl.includes('127.0.0.1')) {
      brokerUrl = brokerUrl.replace('localhost', window.location.hostname)
                           .replace('127.0.0.1', window.location.hostname)
    }
    // Browsers block ws:// from https:// pages. When the page is HTTPS and the
    // broker is on the same host over plain ws://, use the nginx /mqtt WSS proxy.
    if (window.location.protocol === 'https:' && brokerUrl.startsWith('ws://')) {
      try {
        const u = new URL(brokerUrl)
        if (u.hostname === window.location.hostname) {
          brokerUrl = `wss://${window.location.host}/mqtt`
        }
      } catch { /* leave URL unchanged */ }
    }
    mqttBrokerUrl.value = brokerUrl
    mqttTopic.value = cfg.topic || 'sensors/#'
  }
  // Load Fast Mode preference (per-app localStorage key). If it was ON from a
  // previous session, fetch the inference-config now so the first live window
  // can be normalized correctly — otherwise we'd race with the first MQTT tick.
  loadFastModePref()
  if (fastModeEnabled.value && fastModeAvailable.value) {
    fetchFastInferenceConfig()
  }
  loading.value = false

  // Dashboard layout listeners
  window.addEventListener('resize', onResize)
  document.addEventListener('fullscreenchange', updateFullscreenState)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  document.removeEventListener('fullscreenchange', updateFullscreenState)
  // Clean up the Fast Mode worker if it's still running (page navigation)
  terminateFastModeWorker()
})

// Multi-model comparison
const showMultiTable = ref(true)
const MULTI_COLORS = ['#a78bfa', '#f59e0b', '#34d399', '#f87171', '#60a5fa']

const bestR2 = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return null
  let best = -Infinity
  for (const m of Object.values(result.value.models)) {
    if (m.r2 != null && m.r2 > best) best = m.r2
  }
  return best > -Infinity ? best : null
})

const bestAccuracy = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return null
  let best = -1
  for (const m of Object.values(result.value.models)) {
    if (m.accuracy != null && m.accuracy > best) best = m.accuracy
  }
  return best >= 0 ? best : null
})

const multiChartData = computed(() => {
  if (!result.value?.multi_model || !result.value?.models) return []
  return Object.entries(result.value.models)
    .filter(([, m]) => m.predictions && m.predictions.length > 0)
    .map(([eid, m], idx) => ({
      eid, name: m.name,
      predictions: m.predictions.filter(v => typeof v === 'number'),
      color: MULTI_COLORS[idx % MULTI_COLORS.length],
    }))
})

const multiAllValues = computed(() => {
  const all = []
  for (const mc of multiChartData.value) all.push(...mc.predictions)
  if (result.value?.actual) all.push(...result.value.actual)
  return all.filter(v => typeof v === 'number')
})

const multiMinY = computed(() => multiAllValues.value.length ? Math.min(...multiAllValues.value) * 0.98 : 0)
const multiMaxY = computed(() => multiAllValues.value.length ? Math.max(...multiAllValues.value) * 1.02 : 1)
const multiRangeY = computed(() => multiMaxY.value - multiMinY.value || 1)

function multiChartX(i, total) {
  return chartPadding + (i / (total - 1 || 1)) * chartInnerW
}
function multiChartY(v) {
  return chartPadding + chartInnerH - ((v - multiMinY.value) / multiRangeY.value) * chartInnerH
}

const multiActualPath = computed(() => {
  const actuals = result.value?.actual
  if (!actuals || actuals.length === 0) return ''
  const ds = actuals.length <= 200 ? actuals : Array.from({length:200}, (_,i) => actuals[Math.floor(i*actuals.length/200)])
  return ds.map((v,i) => `${i===0?'M':'L'}${multiChartX(i,ds.length).toFixed(1)},${multiChartY(v).toFixed(1)}`).join(' ')
})

const multiModelPaths = computed(() => {
  return multiChartData.value.map(mc => {
    const ds = mc.predictions.length <= 200 ? mc.predictions : Array.from({length:200}, (_,i) => mc.predictions[Math.floor(i*mc.predictions.length/200)])
    const path = ds.map((v,i) => `${i===0?'M':'L'}${multiChartX(i,ds.length).toFixed(1)},${multiChartY(v).toFixed(1)}`).join(' ')
    return { name: mc.name, color: mc.color, path }
  })
})

// ── Multi-Model Classification Timeline ──────────────────
const CLASS_COLOR_PALETTE = [
  '#a78bfa', '#34d399', '#f59e0b', '#22d3ee', '#f472b6',
  '#fbbf24', '#60a5fa', '#fb7185', '#10b981', '#c084fc',
]
const classPadL = 70
const classChartW = 800
const classBandsTop = 20
const classRowH = 22
const classRowGap = 4
const classSignalH = 60

const classColorMap = computed(() => {
  if (!result.value?.multi_model || result.value.mode !== 'classification') return {}
  const labels = new Set()
  if (result.value.actual) result.value.actual.forEach(v => labels.add(String(v)))
  Object.values(result.value.models || {}).forEach(m => {
    (m.predictions || []).forEach(v => labels.add(String(v)))
  })
  const map = {}
  Array.from(labels).sort().forEach((lbl, i) => {
    map[lbl] = CLASS_COLOR_PALETTE[i % CLASS_COLOR_PALETTE.length]
  })
  return map
})

const classTimelineRows = computed(() => {
  if (!result.value?.multi_model || result.value.mode !== 'classification') return []
  const colors = classColorMap.value
  const actuals = result.value.actual || null
  const models = Object.entries(result.value.models || {}).filter(([, m]) => m.predictions && m.predictions.length > 0)
  if (models.length === 0) return []

  // Determine total window count
  const total = Math.max(
    actuals ? actuals.length : 0,
    ...models.map(([, m]) => m.predictions.length)
  )
  if (total === 0) return []

  // Downsample if too many windows (>300)
  const MAX = 300
  let displayCount = total
  let strideStep = 1
  if (total > MAX) {
    displayCount = MAX
    strideStep = total / MAX
  }

  const innerW = classChartW - classPadL - 10
  const segW = innerW / displayCount

  const buildSegments = (arr) => {
    if (!arr) return []
    const out = []
    for (let i = 0; i < displayCount; i++) {
      const srcIdx = Math.min(arr.length - 1, Math.floor(i * strideStep))
      const lbl = String(arr[srcIdx])
      const actLbl = actuals ? String(actuals[Math.min(actuals.length - 1, Math.floor(i * strideStep))]) : null
      out.push({
        x: classPadL + i * segW,
        w: segW + 0.5,
        idx: srcIdx,
        label: lbl,
        color: colors[lbl] || '#666',
        mismatch: actLbl !== null && actLbl !== lbl && arr !== actuals,
      })
    }
    return out
  }

  const rows = []
  let y = classBandsTop

  if (actuals) {
    rows.push({ label: 'Actual', y, h: classRowH, segments: buildSegments(actuals) })
    y += classRowH + classRowGap
  }
  for (const [eid, m] of models) {
    rows.push({ label: m.name || eid, y, h: classRowH, segments: buildSegments(m.predictions) })
    y += classRowH + classRowGap
  }
  return rows
})

const classBandsH = computed(() => classTimelineRows.value.length * (classRowH + classRowGap))
const classBandsBottom = computed(() => classBandsTop + classBandsH.value)
const classChartH = computed(() => {
  const sig = result.value?.signal_preview && result.value.signal_preview.length > 0 ? classSignalH : 0
  return classBandsBottom.value + sig + 20
})

const classSignalPath = computed(() => {
  const sig = result.value?.signal_preview
  if (!sig || sig.length === 0 || classTimelineRows.value.length === 0) return ''
  const innerW = classChartW - classPadL - 10
  const minV = Math.min(...sig)
  const maxV = Math.max(...sig)
  const range = (maxV - minV) || 1
  const sigTop = classBandsBottom.value + 10
  return sig.map((v, i) => {
    const x = classPadL + (i / (sig.length - 1 || 1)) * innerW
    const y = sigTop + (1 - (v - minV) / range) * (classSignalH - 12)
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})

const hasMismatch = computed(() => {
  return classTimelineRows.value.some(r => r.segments.some(s => s.mismatch))
})

function downloadMultiModelCsv() {
  if (!result.value?.multi_model || !result.value?.models) return
  const models = Object.values(result.value.models).filter(m => m.predictions)
  const actuals = result.value.actual || []
  const maxLen = Math.max(...models.map(m => m.predictions?.length || 0), actuals.length)

  const header = ['datapoint']
  if (actuals.length > 0) header.push('actual')
  for (const m of models) header.push(m.name || 'model')

  const rows = [header.join(',')]
  for (let i = 0; i < maxLen; i++) {
    const row = [i]
    if (actuals.length > 0) row.push(actuals[i] != null ? actuals[i] : '')
    for (const m of models) row.push(m.predictions?.[i] != null ? m.predictions[i] : '')
    rows.push(row.join(','))
  }

  // Summary
  rows.push('')
  rows.push('--- Metrics ---')
  if (result.value?.mode === 'regression') {
    rows.push('model,r2,rmse,mae')
    for (const m of models) {
      rows.push(`${m.name},${m.r2 ?? ''},${m.rmse ?? ''},${m.mae ?? ''}`)
    }
  } else {
    rows.push('model,accuracy,precision,recall,f1')
    for (const m of models) {
      rows.push(`${m.name},${m.accuracy ?? ''},${m.precision ?? ''},${m.recall ?? ''},${m.f1 ?? ''}`)
    }
  }

  const csv = rows.join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `multi_model_comparison_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const fileInput = ref(null)

// ── Live Stream (MQTT) ─────────────────────────────────
const isLiveStream = computed(() => {
  return parsedNodes.value.some(n => n.type === 'input.live_stream')
})

const liveStreamConfig = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'input.live_stream')
  return node?.config || {}
})

const liveWindowSize = computed(() => {
  const wNode = parsedNodes.value.find(n => n.type === 'transform.window')
  return wNode?.config?.window_size || 128
})

const liveStride = computed(() => {
  const wNode = parsedNodes.value.find(n => n.type === 'transform.window')
  return wNode?.config?.step || wNode?.config?.stride || 64
})

const autoDetectedChannels = ref([])

// (sensorAutoFallback* and showRawMqtt / rawMqttBuffer refs are declared
//  earlier in the display-prefs block so load/save can see them before use.)

const liveChannels = computed(() => {
  const channels = liveStreamConfig.value.channels || ''
  if (channels) return channels.split(',').map((c) => c.trim()).filter(Boolean)
  if (appData.value.sensor_columns && appData.value.sensor_columns.length > 0) {
    return appData.value.sensor_columns
  }
  // Use auto-detected channels from first MQTT message
  return autoDetectedChannels.value
})

const mqttBrokerUrl = ref('')
const mqttTopic = ref('')
const mqttConnected = ref(false)
const mqttError = ref(null)
const mqttMessageCount = ref(0)
const mqttMessagesPerSec = ref(0)
const sensorBufferLen = ref(0)
const sensorBufferProgress = ref(0)
const liveInferenceCount = ref(0)
const livePrediction = ref(null)
const livePredictionHistory = ref([])
// Object-shape history for classification/anomaly ({label, confidence, ...}).
// Kept separately from livePredictionHistory (which is numbers for regression).
const livePredictionHistoryFull = ref([])
const liveActualColumn = ref('(none)')  // Selected MQTT column for actual comparison
const liveActualHistory = ref([])  // accumulated actual values from selected column
watch(liveActualColumn, () => { liveActualHistory.value = [] })
const liveMultiHistory = ref({})  // { eid: { name, algorithm, mode, predictions: [] } }
const liveMultiActuals = ref([])  // accumulated actual values from MQTT target column
const liveSignalHistory = ref([])  // accumulated signal data for classification timeline chart
const MAX_LIVE_HISTORY = 200
const liveLastUpdated = ref(null)

// ── Prediction recording (auto on Connect → CSV on Disconnect) ──
const autoRecordPredictions = ref(false)
const predictionRecordMode = ref('per_inference') // 'per_inference' | 'per_sample' | 'full_window'
const isRecordingPredictions = ref(false)
const predictionRecordBuffer = ref([])
const predictionRecordStart = ref(0)
const predictionRecordTick = ref(0)
let predictionRecordTimer = null
let recordModeInitialized = false
// Latest inference output, used to stamp per-sample rows in Mode 2
const latestPredictionState = ref({
  prediction: null,
  confidence: null,
  score: null,
  modelPredictions: null,
})

const predictionRecordDuration = computed(() => {
  if (!predictionRecordStart.value) return '00:00'
  void predictionRecordTick.value
  const sec = Math.floor((Date.now() - predictionRecordStart.value) / 1000)
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  const pad = (n) => String(n).padStart(2, '0')
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
})
let mqttClient = null
let sensorBuffer = []
let rateCounter = 0
let rateInterval = null

// ── Fast Mode (P2 Phase 3) ─────────────────────────────────────
// When ON, live MQTT inference computes features in the browser (Web Worker
// reused from Phase 2) and posts a `feature_vector` instead of raw samples.
// The backend skips its tsfresh path — protects the server under workshop load.
// Persisted per-app in localStorage so different apps can independently opt in.
const fastModeKey = computed(() => `cira.publishedApp.fastMode.${slug.value || 'app'}`)
const fastModeEnabled = ref(false)
// Feature names the trained model expects (fetched from the pipeline's
// transform.feature_extract node config). If empty, Fast Mode won't be shown.
const modelFeatureNames = computed(() => {
  const featNode = parsedNodes.value.find(n => n.type === 'transform.feature_extract')
  const feats = featNode?.config?.features
  return Array.isArray(feats) ? feats : []
})
const hasFeatureExtractNode = computed(() => modelFeatureNames.value.length > 0)
// Only meaningful for MQTT-live apps; toggle stays hidden for CSV-upload apps.
const fastModeAvailable = computed(() => isLiveStream.value && hasFeatureExtractNode.value)
// One long-lived worker for the whole MQTT session. Spawned when Fast Mode is
// enabled + MQTT connects; torn down on disable / disconnect / unmount.
let fastModeWorker = null
// Pending window promises keyed by session_id → resolver
const fastPending = new Map()
let fastSessionCounter = 0

// Model's training normalization + feature contract, fetched from the backend
// when Fast Mode is turned on. Without this, client-computed features won't
// match what the model was trained on (min-max/z-score scaling matters).
// See backend `/api/app-builder/run/<slug>/inference-config`.
const fastInferenceConfig = ref(null)  // { normalization, sensor_columns, feature_names } | null

async function fetchFastInferenceConfig() {
  if (!slug.value) return
  try {
    const resp = await api.get(`/api/app-builder/run/${slug.value}/inference-config`)
    fastInferenceConfig.value = resp.data
  } catch (e) {
    console.warn('[Fast Mode] failed to fetch inference-config:', e?.response?.data?.error || e?.message || e)
    fastInferenceConfig.value = null
  }
}

/** Apply the model's training normalization to a raw window client-side.
 *  Matches backend `_apply_normalization` when `_model_norm` is supplied.
 *  Returns a fresh 2D array; original csvRows are not mutated. */
function normalizeWindowForFastMode(csvRows, channels) {
  const cfg = fastInferenceConfig.value?.normalization
  // No normalization params from server, or model was trained without it —
  // pass through untouched. Same as backend's train_method === 'none' branch.
  if (!cfg || cfg.method === 'none') return csvRows

  const trainCols = cfg.sensor_columns || []
  const nCh = channels.length
  if (trainCols.length === 0 || nCh === 0) return csvRows

  // Build per-channel scale/offset arrays aligned to the incoming `channels`
  // order. If a channel doesn't appear in the model's sensor_columns, leave
  // it alone (offset 0, scale 1) — matches the backend loop that only writes
  // into positions where `col in sensor_cols`.
  const offset = new Array(nCh).fill(0)
  const scale = new Array(nCh).fill(1)

  const method = cfg.method || 'min_max'
  if (method === 'z_score') {
    const mean = cfg.channel_mean || []
    const std = cfg.channel_std || []
    for (let i = 0; i < nCh; i++) {
      const idx = trainCols.indexOf(channels[i])
      if (idx >= 0 && idx < mean.length && idx < std.length) {
        offset[i] = mean[idx]
        const s = std[idx]
        scale[i] = s === 0 ? 1 : s
      }
    }
  } else if (method === 'robust') {
    const median = cfg.channel_median || []
    const iqr = cfg.channel_iqr || []
    for (let i = 0; i < nCh; i++) {
      const idx = trainCols.indexOf(channels[i])
      if (idx >= 0 && idx < median.length && idx < iqr.length) {
        offset[i] = median[idx]
        const s = iqr[idx]
        scale[i] = s === 0 ? 1 : s
      }
    }
  } else {
    // min_max (default) — offset = min, scale = max - min (0 → 1)
    const cMin = cfg.channel_min || []
    const cMax = cfg.channel_max || []
    for (let i = 0; i < nCh; i++) {
      const idx = trainCols.indexOf(channels[i])
      if (idx >= 0 && idx < cMin.length && idx < cMax.length) {
        offset[i] = cMin[idx]
        const denom = cMax[idx] - cMin[idx]
        scale[i] = denom === 0 ? 1 : denom
      }
    }
  }

  const out = new Array(csvRows.length)
  for (let r = 0; r < csvRows.length; r++) {
    const row = csvRows[r]
    const newRow = new Array(nCh)
    for (let i = 0; i < nCh; i++) newRow[i] = (row[i] - offset[i]) / scale[i]
    out[r] = newRow
  }
  return out
}

function loadFastModePref() {
  try {
    const v = localStorage.getItem(fastModeKey.value)
    if (v === '1' || v === 'true') fastModeEnabled.value = true
    else if (v === '0' || v === 'false') fastModeEnabled.value = false
  } catch { /* localStorage may be blocked */ }
}
function persistFastModePref() {
  try { localStorage.setItem(fastModeKey.value, fastModeEnabled.value ? '1' : '0') } catch { /* ignore */ }
}

function spawnFastModeWorker() {
  if (fastModeWorker) return fastModeWorker
  try {
    fastModeWorker = new Worker(
      new URL('../workers/features.worker.ts', import.meta.url),
      { type: 'module' },
    )
    fastModeWorker.onmessage = (evt) => {
      const msg = evt.data
      if (!msg) return
      if (msg.type === 'done') {
        const pending = fastPending.get(msg.session_id)
        if (pending) {
          fastPending.delete(msg.session_id)
          pending.resolve({
            feature_vector: msg.features_df,
            feature_names: msg.feature_names,
          })
        }
      } else if (msg.type === 'error') {
        // Reject all outstanding — main thread falls back to raw path per-window.
        console.warn('[Fast Mode] worker error:', msg.message)
        for (const [, pending] of fastPending) pending.reject(new Error(msg.message))
        fastPending.clear()
      }
    }
    fastModeWorker.onerror = (e) => {
      console.warn('[Fast Mode] worker crashed:', e.message || e)
      for (const [, pending] of fastPending) pending.reject(new Error(e.message || 'worker crashed'))
      fastPending.clear()
      // Force teardown so the next window either respawns or falls back.
      try { fastModeWorker.terminate() } catch { /* ignore */ }
      fastModeWorker = null
    }
  } catch (e) {
    console.warn('[Fast Mode] failed to spawn worker:', e)
    fastModeWorker = null
  }
  return fastModeWorker
}

function terminateFastModeWorker() {
  if (fastModeWorker) {
    try { fastModeWorker.terminate() } catch { /* ignore */ }
    fastModeWorker = null
  }
  for (const [, pending] of fastPending) pending.reject(new Error('worker terminated'))
  fastPending.clear()
}

/** Compute features for a single window in the worker. Resolves with
 *  {feature_vector: number[][], feature_names: string[]} or rejects on error. */
function extractWindowFeaturesFast(csvRows, channels) {
  return new Promise((resolve, reject) => {
    const worker = spawnFastModeWorker()
    if (!worker) {
      reject(new Error('worker unavailable'))
      return
    }
    const sessionId = `fastlive_${++fastSessionCounter}`
    fastPending.set(sessionId, { resolve, reject })
    // Apply the model's training normalization to the raw window before
    // handing it to the worker so features match server-side pipeline output.
    // No-op when the model was trained without normalization.
    const normRows = normalizeWindowForFastMode(csvRows, channels)
    worker.postMessage({
      type: 'extract',
      windows: [normRows],   // single-window batch (already normalized)
      channelNames: channels,
      selectedFeatures: modelFeatureNames.value,
      samplingRate: 100,    // used only by spectral features; matches Phase 2 default
      sessionId,
    })
    // Safety: if the worker never responds within 15s, reject and let the
    // caller fall back to raw so a stuck window doesn't block inference.
    setTimeout(() => {
      const pending = fastPending.get(sessionId)
      if (pending) {
        fastPending.delete(sessionId)
        pending.reject(new Error('worker timeout'))
      }
    }, 15000)
  })
}

// Toggle handler: persist + tear down worker when disabling so we don't leak
// a worker when the user flips off mid-session. When enabling, fetch the
// inference-config so subsequent window inferences can be normalized correctly.
watch(fastModeEnabled, (val) => {
  persistFastModePref()
  if (val) {
    fetchFastInferenceConfig()
  } else {
    terminateFastModeWorker()
    fastInferenceConfig.value = null
  }
})

// ── Signal Recorder live preview ──────────────────────────────
const previewWindowSec = ref(10)
const previewBuffer = ref([]) // Array<{ts: number, [channel]: number}>
const previewTicker = ref(0)  // bumped on message to force chart re-eval
let previewPruneInterval = null

function prunePreviewBuffer() {
  const cutoff = Date.now() - previewWindowSec.value * 1000
  const buf = previewBuffer.value
  let dropFrom = 0
  while (dropFrom < buf.length && buf[dropFrom].ts < cutoff) dropFrom++
  if (dropFrom > 0) previewBuffer.value = buf.slice(dropFrom)
}

const PREVIEW_COLORS = ['#a78bfa', '#f59e0b', '#34d399', '#f87171', '#60a5fa', '#f472b6', '#22d3ee', '#facc15']

const previewStats = computed(() => {
  void previewTicker.value
  const buf = previewBuffer.value
  const channels = liveChannels.value || []
  const out = {}
  for (const ch of channels) {
    let latest = null
    let min = Infinity
    let max = -Infinity
    let sum = 0
    let count = 0
    for (const s of buf) {
      const v = s[ch]
      if (typeof v === 'number' && !isNaN(v)) {
        latest = v
        if (v < min) min = v
        if (v > max) max = v
        sum += v
        count++
      }
    }
    out[ch] = {
      latest,
      min: count > 0 ? min : null,
      max: count > 0 ? max : null,
      mean: count > 0 ? sum / count : null,
    }
  }
  return out
})

const previewChartData = computed(() => {
  void previewTicker.value
  const buf = previewBuffer.value
  const channels = liveChannels.value || []
  const now = Date.now()
  const labels = buf.map(s => ((s.ts - now) / 1000).toFixed(1))
  const datasets = channels.map((ch, idx) => ({
    label: ch,
    data: buf.map(s => (typeof s[ch] === 'number' ? s[ch] : null)),
    borderColor: PREVIEW_COLORS[idx % PREVIEW_COLORS.length],
    backgroundColor: PREVIEW_COLORS[idx % PREVIEW_COLORS.length] + '22',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.15,
    spanGaps: true,
  }))
  return { labels, datasets }
})

const previewChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { intersect: false, mode: 'nearest' },
  plugins: {
    legend: {
      display: true,
      position: 'bottom',
      labels: { color: '#8b949e', boxWidth: 10, font: { size: 10 } },
    },
    tooltip: { enabled: true },
  },
  scales: {
    x: {
      title: { display: true, text: 'seconds ago', color: '#8b949e', font: { size: 10 } },
      ticks: { color: '#8b949e', maxTicksLimit: 6, font: { size: 9 } },
      grid: { color: '#21262d' },
    },
    y: {
      ticks: { color: '#8b949e', font: { size: 9 } },
      grid: { color: '#21262d' },
    },
  },
}

function fmtPreview(v) {
  if (v === null || v === undefined || isNaN(v)) return '--'
  const abs = Math.abs(v)
  if (abs >= 1000) return v.toFixed(1)
  return v.toFixed(3)
}

// ── Dashboard helpers ─────────────────────────────
const latestConfidence = computed(() => {
  const hist = livePredictionHistoryFull.value
  if (!hist || hist.length === 0) return null
  const last = hist[hist.length - 1]
  if (last && typeof last === 'object' && 'confidence' in last && typeof last.confidence === 'number') {
    return last.confidence
  }
  return null
})

function multiTileLatest(m) {
  const preds = m?.predictions || []
  if (preds.length === 0) return '—'
  const v = preds[preds.length - 1]
  if (typeof v === 'number') return v.toFixed(3)
  return String(v)
}

function multiTilePredColor(m, mode) {
  if (mode === 'classification') return '#34d399'
  if (mode === 'anomaly') return '#f87171'
  return '#a78bfa'
}

// Downsample helper for mini tile charts (keeps DOM light on large histories).
function tileDownsample(arr, max = 60) {
  if (!arr || arr.length === 0) return []
  if (arr.length <= max) return arr.slice()
  const step = arr.length / max
  return Array.from({ length: max }, (_, i) => arr[Math.floor(i * step)])
}

function tileChartData(eid, m) {
  const mode = result.value?.mode
  const preds = m.predictions || []
  if (mode === 'classification') {
    // Classification: convert labels to indices so we get a step-like line.
    const labelToIdx = {}
    const labels = Object.keys(classColorMap.value)
    labels.forEach((l, i) => (labelToIdx[l] = i))
    const ds = tileDownsample(preds, 80)
    return {
      labels: ds.map((_, i) => i),
      datasets: [{
        label: m.name,
        data: ds.map(v => (v in labelToIdx ? labelToIdx[v] : 0)),
        borderColor: '#34d399',
        backgroundColor: '#34d39922',
        borderWidth: 1.5,
        pointRadius: 0,
        stepped: 'before',
        tension: 0,
      }],
    }
  }
  // Regression / anomaly: numeric line
  const nums = preds.filter(v => typeof v === 'number')
  const ds = tileDownsample(nums, 80)
  const actuals = result.value?.actual
  const datasets = [{
    label: m.name,
    data: ds,
    borderColor: '#a78bfa',
    backgroundColor: '#a78bfa22',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.15,
    spanGaps: true,
  }]
  if (actuals && actuals.length > 0 && mode === 'regression') {
    const aNums = actuals.filter(v => typeof v === 'number')
    if (aNums.length > 0) {
      const dsA = tileDownsample(aNums, 80)
      datasets.unshift({
        label: 'Actual',
        data: dsA,
        borderColor: '#22d3ee',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
        spanGaps: true,
      })
    }
  }
  return {
    labels: ds.map((_, i) => i),
    datasets,
  }
}

function tileChartOptions(_mode) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: { display: false },
      y: {
        ticks: { color: '#8b949e', font: { size: 8 }, maxTicksLimit: 3 },
        grid: { color: '#21262d' },
      },
    },
  }
}

// Re-prune when the user shrinks the window
watch(previewWindowSec, () => {
  prunePreviewBuffer()
  previewTicker.value++
})

// Auto-collapse the left rail once we're connected so the dashboard
// gives the full width to charts / prediction tiles. Expanding is manual.
watch(mqttConnected, (connected) => {
  if (connected) railCollapsed.value = true
  else railCollapsed.value = false
})
// ──────────────────────────────────────────────────────────────

const liveLastUpdatedText = computed(() => {
  if (!liveLastUpdated.value) return ''
  const ago = Math.round((Date.now() - liveLastUpdated.value) / 1000)
  return ago < 2 ? 'just now' : `${ago}s ago`
})

// Update "ago" text reactively
const liveUpdateTicker = ref(0)
let tickerInterval = null

async function startLiveStream() {
  mqttError.value = null
  try {
    // Dynamic import — only load mqtt when needed
    if (!mqtt) {
      const mod = await import('mqtt')
      mqtt = mod.default || mod
    }
    mqttClient = mqtt.connect(mqttBrokerUrl.value, {
      clientId: `cira-live-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      clean: true,
      keepalive: 30,
      reconnectPeriod: 3000,
    })

    mqttClient.on('connect', () => {
      mqttConnected.value = true
      mqttError.value = null
      mqttClient.subscribe(mqttTopic.value, { qos: 0 })
      if (autoRecordPredictions.value && !isRecorderMode.value) {
        startPredictionRecording()
      }
      // Warm up the Fast Mode worker so the first window doesn't pay the cold-
      // start cost (dynamic import + parse) — only when the app supports it.
      if (fastModeEnabled.value && fastModeAvailable.value) {
        spawnFastModeWorker()
      }
    })

    mqttClient.on('error', (err) => {
      mqttError.value = err.message || 'Connection failed'
      mqttConnected.value = false
    })

    mqttClient.on('close', () => {
      mqttConnected.value = false
    })

    mqttClient.on('message', (_topic, payload) => {
      mqttMessageCount.value++
      rateCounter++
      try {
        const raw = JSON.parse(payload.toString())
        // Layer 3: ring-buffer the raw payload for the "Show raw MQTT" panel.
        // Pretty-print to 2-space JSON so operators can eyeball payload shape
        // without a separate viewer. Newest-first, capped at 3 entries.
        try {
          const pretty = JSON.stringify(raw, null, 2)
          rawMqttBuffer.value.unshift(pretty)
          if (rawMqttBuffer.value.length > 3) {
            rawMqttBuffer.value = rawMqttBuffer.value.slice(0, 3)
          }
        } catch { /* stringify should not fail on parseable JSON */ }
        const sample = parseSensorPayload(raw)
        if (sample) {
          // Preview buffer (recorder-mode only): always updates while connected,
          // independent of whether the user has pressed Record.
          if (isRecorderMode.value) {
            previewBuffer.value.push({ ts: Date.now(), ...sample })
            prunePreviewBuffer()
            previewTicker.value++
          }
          // Record if in recorder mode and recording
          if (isRecorderMode.value && recorderState.value.recording && recorderState.value.currentLabel) {
            const maxDur = (recorderConfig.value.max_duration || 300) * 1000
            const elapsed = recorderState.value.startTime ? Date.now() - recorderState.value.startTime : 0
            if (elapsed < maxDur) {
              recorderState.value.samples.push({
                ...sample,
                label: recorderState.value.currentLabel,
                _ts: Date.now(),
              })
            } else {
              recorderState.value.recording = false
            }
          }
          // Buffer for inference (non-recorder mode)
          if (!isRecorderMode.value) {
            sample._ts = Date.now()
            pushSensorSample(sample)
            if (isRecordingPredictions.value && predictionRecordMode.value === 'per_sample') {
              pushPerSampleRecord(sample)
            }
          }
        }
      } catch { /* ignore non-JSON */ }
    })

    // Rate counter
    rateInterval = setInterval(() => {
      mqttMessagesPerSec.value = rateCounter
      rateCounter = 0
      liveUpdateTicker.value++
    }, 1000)

    // Prune preview buffer even when messages slow down, so old points drop off
    // the rolling window on the chart.
    if (previewPruneInterval) clearInterval(previewPruneInterval)
    previewPruneInterval = setInterval(() => {
      if (isRecorderMode.value && previewBuffer.value.length > 0) {
        prunePreviewBuffer()
        previewTicker.value++
      }
    }, 500)

  } catch (e) {
    mqttError.value = e.message || 'Failed to connect'
  }
}

function stopLiveStream() {
  if (isRecordingPredictions.value) {
    stopAndDownloadPredictionRecording()
  }
  if (mqttClient) {
    mqttClient.end(true)
    mqttClient = null
  }
  if (rateInterval) {
    clearInterval(rateInterval)
    rateInterval = null
  }
  if (previewPruneInterval) {
    clearInterval(previewPruneInterval)
    previewPruneInterval = null
  }
  previewBuffer.value = []
  mqttConnected.value = false
  sensorBuffer = []
  sensorBufferLen.value = 0
  sensorBufferProgress.value = 0
  // Free the Fast Mode worker when the stream stops (spawn again on reconnect)
  terminateFastModeWorker()
}

function startPredictionRecording() {
  predictionRecordBuffer.value = []
  predictionRecordStart.value = Date.now()
  isRecordingPredictions.value = true
  latestPredictionState.value = { prediction: null, confidence: null, score: null, modelPredictions: null }
  if (predictionRecordTimer) clearInterval(predictionRecordTimer)
  predictionRecordTimer = setInterval(() => { predictionRecordTick.value++ }, 1000)
}

function getActualFromSample(samp) {
  let col = null
  if (isMultiModelApp.value) {
    col = multiModelTargetCol.value
  } else if (liveActualColumn.value && liveActualColumn.value !== '(none)') {
    col = liveActualColumn.value
  }
  if (!col || !samp || typeof samp !== 'object' || Array.isArray(samp)) return null
  return col in samp ? samp[col] : null
}

function extractChannelVals(samp) {
  const channels = liveChannels.value || []
  const out = {}
  for (const ch of channels) {
    if (Array.isArray(samp)) {
      const idx = channels.indexOf(ch)
      out[ch] = idx >= 0 ? samp[idx] : null
    } else if (samp && typeof samp === 'object') {
      out[ch] = samp[ch] ?? null
    } else {
      out[ch] = null
    }
  }
  return out
}

function pushPerSampleRecord(sample) {
  const channelVals = extractChannelVals(sample)
  const state = latestPredictionState.value || {}
  const actual = getActualFromSample(sample)
  const timestamp = new Date(sample._ts || Date.now()).toISOString()
  if (state.modelPredictions) {
    predictionRecordBuffer.value.push({
      timestamp,
      ...channelVals,
      modelPredictions: { ...state.modelPredictions },
      actual,
    })
  } else {
    predictionRecordBuffer.value.push({
      timestamp,
      ...channelVals,
      prediction: state.prediction,
      confidence: state.confidence,
      score: state.score,
      actual,
    })
  }
}

function recordInferenceRows(opts) {
  const { windowData, prediction, confidence, score, modelPredictions } = opts
  latestPredictionState.value = {
    prediction: prediction ?? null,
    confidence: confidence ?? null,
    score: score ?? null,
    modelPredictions: modelPredictions ? { ...modelPredictions } : null,
  }
  if (!isRecordingPredictions.value) return
  if (predictionRecordMode.value === 'per_sample') return // handled in MQTT handler

  const samples = predictionRecordMode.value === 'full_window'
    ? windowData
    : [windowData[windowData.length - 1]]

  const nowIso = new Date().toISOString()
  for (const samp of samples) {
    const channelVals = extractChannelVals(samp)
    const actual = getActualFromSample(samp)
    const ts = (samp && typeof samp === 'object' && !Array.isArray(samp) && samp._ts)
      ? new Date(samp._ts).toISOString()
      : nowIso
    if (modelPredictions) {
      predictionRecordBuffer.value.push({
        timestamp: ts,
        ...channelVals,
        modelPredictions: { ...modelPredictions },
        actual,
      })
    } else {
      predictionRecordBuffer.value.push({
        timestamp: ts,
        ...channelVals,
        prediction: prediction ?? null,
        confidence: confidence ?? null,
        score: score ?? null,
        actual,
      })
    }
  }
}

function stopAndDownloadPredictionRecording() {
  isRecordingPredictions.value = false
  if (predictionRecordTimer) {
    clearInterval(predictionRecordTimer)
    predictionRecordTimer = null
  }
  if (predictionRecordBuffer.value.length === 0) {
    predictionRecordStart.value = 0
    return
  }
  const csv = buildPredictionCSV()
  downloadCSV(csv, predictionCsvFilename())
  predictionRecordBuffer.value = []
  predictionRecordStart.value = 0
}

function csvEscape(val) {
  if (val === null || val === undefined) return ''
  const s = String(val)
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return '"' + s.replace(/"/g, '""') + '"'
  }
  return s
}

function formatNum(v) {
  if (v === null || v === undefined || v === '') return ''
  if (typeof v === 'number' && Number.isFinite(v)) return String(v)
  return csvEscape(v)
}

function buildPredictionCSV() {
  const rows = predictionRecordBuffer.value
  const channels = liveChannels.value || []
  const hasActual = rows.some(r => r.actual !== null && r.actual !== undefined)
  const isMulti = rows.some(r => r.modelPredictions && typeof r.modelPredictions === 'object')

  if (isMulti) {
    const modelNames = []
    const seen = new Set()
    for (const r of rows) {
      if (!r.modelPredictions) continue
      for (const name of Object.keys(r.modelPredictions)) {
        if (!seen.has(name)) { seen.add(name); modelNames.push(name) }
      }
    }
    const header = ['timestamp', ...channels, ...modelNames]
    if (hasActual) header.push('actual')
    const csvRows = rows.map(r => {
      const vals = [csvEscape(r.timestamp)]
      for (const ch of channels) vals.push(formatNum(r[ch]))
      for (const name of modelNames) {
        const v = r.modelPredictions ? r.modelPredictions[name] : null
        vals.push(formatNum(v))
      }
      if (hasActual) vals.push(formatNum(r.actual))
      return vals.join(',')
    })
    return [header.join(','), ...csvRows].join('\n')
  }

  const hasConfidence = rows.some(r => r.confidence !== null && r.confidence !== undefined)
  const hasScore = rows.some(r => r.score !== null && r.score !== undefined)

  const header = ['timestamp', ...channels, 'prediction']
  if (hasConfidence) header.push('confidence')
  if (hasScore) header.push('anomaly_score')
  if (hasActual) header.push('actual')

  const csvRows = rows.map(r => {
    const vals = [csvEscape(r.timestamp)]
    for (const ch of channels) vals.push(formatNum(r[ch]))
    vals.push(formatNum(r.prediction))
    if (hasConfidence) vals.push(formatNum(r.confidence))
    if (hasScore) vals.push(formatNum(r.score))
    if (hasActual) vals.push(formatNum(r.actual))
    return vals.join(',')
  })

  return [header.join(','), ...csvRows].join('\n')
}

function predictionCsvFilename() {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const base = (slug.value || 'app').replace(/[^a-zA-Z0-9_-]/g, '_')
  return `predictions_${base}_${ts}.csv`
}

function downloadCSV(csvText, filename) {
  const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ── Signal Recorder ─────────────────────────────────
const isMultiModelApp = computed(() => {
  return parsedNodes.value.some(n => n.type === 'output.multi_model_compare')
})

const multiModelTargetCol = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'output.multi_model_compare')
  return node?.config?.target_column || ''
})

const multiModelMode = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'output.multi_model_compare')
  return node?.config?.mode || 'regression'
})

// Default record granularity once we know the app mode: classification → Per sample, else → Per inference
watch([appMode, multiModelMode, isMultiModelApp], ([am, mm, isMulti]) => {
  if (recordModeInitialized) return
  const m = isMulti ? mm : am
  if (!m) return
  recordModeInitialized = true
  predictionRecordMode.value = m === 'classification' ? 'per_sample' : 'per_inference'
}, { immediate: true })

const isRecorderMode = computed(() => {
  return parsedNodes.value.some(n => n.type === 'output.signal_recorder')
})

const recorderConfig = computed(() => {
  const node = parsedNodes.value.find(n => n.type === 'output.signal_recorder')
  return node?.config || {}
})

const recorderLabels = ref([])
const recorderCustomLabel = ref('')
const recorderState = ref({
  recording: false,
  currentLabel: '',
  samples: [],
  startTime: null,
})

// Initialize labels from config
watch(() => recorderConfig.value, (cfg) => {
  if (cfg.labels) {
    recorderLabels.value = cfg.labels.split(',').map(l => l.trim()).filter(Boolean)
    if (recorderLabels.value.length > 0 && !recorderState.value.currentLabel) {
      recorderState.value.currentLabel = recorderLabels.value[0]
    }
  }
}, { immediate: true })

function addCustomLabel() {
  const lbl = recorderCustomLabel.value.trim()
  if (lbl && !recorderLabels.value.includes(lbl)) {
    recorderLabels.value.push(lbl)
  }
  recorderCustomLabel.value = ''
}

function startRecording() {
  recorderState.value.recording = true
  recorderState.value.startTime = Date.now()
}

function stopRecording() {
  recorderState.value.recording = false
}

function clearRecording() {
  recorderState.value.samples = []
  recorderState.value.startTime = null
}

const recorderDuration = computed(() => {
  const samples = recorderState.value.samples
  if (samples.length === 0) return '0s'
  const first = samples[0]._ts || 0
  const last = samples[samples.length - 1]._ts || 0
  const sec = Math.round((last - first) / 1000)
  return sec < 60 ? `${sec}s` : `${Math.floor(sec/60)}m ${sec%60}s`
})

const recorderLabelCounts = computed(() => {
  const counts = {}
  for (const s of recorderState.value.samples) {
    counts[s.label] = (counts[s.label] || 0) + 1
  }
  return Object.entries(counts).map(([k,v]) => `${k}:${v}`).join(' ')
})

const LABEL_COLORS = ['#60a5fa','#34d399','#f87171','#fbbf24','#a78bfa','#f472b6','#22d3ee','#fb923c']

const recorderSegments = computed(() => {
  const samples = recorderState.value.samples
  if (samples.length === 0) return []
  const segments = []
  let cur = { label: samples[0].label, count: 0, color: '' }
  const labelIdx = {}
  for (const s of samples) {
    if (s.label !== cur.label) {
      segments.push({...cur})
      cur = { label: s.label, count: 0, color: '' }
    }
    cur.count++
    if (!(s.label in labelIdx)) labelIdx[s.label] = Object.keys(labelIdx).length
    cur.color = LABEL_COLORS[labelIdx[s.label] % LABEL_COLORS.length]
  }
  if (cur.count > 0) segments.push({...cur})
  return segments
})

function downloadRecordedCSV() {
  const samples = recorderState.value.samples
  if (samples.length === 0) return
  const channels = liveChannels.value
  const prefix = recorderConfig.value.file_prefix || 'sensor_data'

  // Build CSV
  const header = ['timestamp', ...channels, 'label'].join(',')
  const rows = samples.map((s, i) => {
    const vals = channels.map(ch => s[ch] ?? 0)
    return [i * (1.0 / (recorderConfig.value.target_sample_rate || 62.5)), ...vals, s.label].join(',')
  })
  const csv = [header, ...rows].join('\n')

  // Download
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${prefix}_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// Recursive numeric-leaf flattener. Walks any JSON value and returns
// { [dotted_key]: number } for every numeric leaf.
//
//   {X:1, Y:2}                    -> {X:1, Y:2}
//   {data:{x:1, y:2}}             -> {"data.x":1, "data.y":2}
//   {values:[1,2,3]}              -> {"values[0]":1, "values[1]":2, "values[2]":3}
//   [1,2,3]                       -> {"[0]":1, "[1]":2, "[2]":3}
//   5.5                           -> {value:5.5}
//   {ts:1, temp:2, device:"a"}    -> {ts:1, temp:2}  (non-numeric dropped)
//
// Insertion order is preserved by Object.entries + push semantics — the caller
// relies on this for the auto-fallback positional assignment.
function flattenNumericLeaves(obj, prefix = '') {
  const out = {}
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

  // Bare number — return under 'value' when at the root, otherwise honor prefix.
  if (isNum(obj)) {
    out[prefix || 'value'] = obj
    return out
  }
  if (obj === null || obj === undefined) return out

  if (Array.isArray(obj)) {
    obj.forEach((v, i) => {
      const key = `${prefix}[${i}]`
      if (isNum(v)) {
        out[key] = v
      } else if (v !== null && typeof v === 'object') {
        Object.assign(out, flattenNumericLeaves(v, key))
      }
      // strings / bools / null at leaves are dropped (not numeric)
    })
    return out
  }

  if (typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (isNum(v)) {
        out[key] = v
      } else if (v !== null && typeof v === 'object') {
        Object.assign(out, flattenNumericLeaves(v, key))
      }
      // strings / bools / null at leaves are dropped
    }
    return out
  }

  return out
}

function parseSensorPayload(raw) {
  // Normalize different MQTT payload formats into { channel: value } object
  const channels = liveChannels.value

  // ── Format 1 (preserved): { "values": [1.2, 3.4, 5.6] } ──────
  // Android SensorSpot and other array-shaped emitters. Only use this branch's
  // result if it fully resolves — otherwise fall through to the generic
  // flatten/match logic below.
  if (raw && raw.values && Array.isArray(raw.values)) {
    if (channels.length === 0 && autoDetectedChannels.value.length === 0) {
      autoDetectedChannels.value = raw.values.map((_, i) => `ch${i}`)
    }
    const ch = channels.length > 0 ? channels : autoDetectedChannels.value
    const sample = {}
    let ok = true
    raw.values.forEach((v, i) => {
      const name = ch[i] || `ch${i}`
      const num = typeof v === 'number' ? v : parseFloat(v)
      if (!Number.isFinite(num)) { ok = false; return }
      sample[name] = num
    })
    // Attach string target column if present (for ground-truth comparison)
    const targetCol = multiModelTargetCol.value
    if (targetCol && raw && typeof raw === 'object' && typeof raw[targetCol] === 'string') {
      sample[targetCol] = raw[targetCol]
    }
    if (ok && Object.keys(sample).length > 0) return sample
    // fall through
  }

  // ── Format 2 (preserved): { "values": { "v0": 1.2, "v1": 3.4 } } ──
  if (raw && raw.values && typeof raw.values === 'object' && !Array.isArray(raw.values)) {
    const keys = Object.keys(raw.values)
    if (channels.length === 0 && autoDetectedChannels.value.length === 0) {
      autoDetectedChannels.value = keys
    }
    const ch = channels.length > 0 ? channels : autoDetectedChannels.value
    const sample = {}
    let ok = true
    keys.forEach((k, i) => {
      const name = ch[i] || k
      const num = typeof raw.values[k] === 'number' ? raw.values[k] : parseFloat(raw.values[k])
      if (!Number.isFinite(num)) { ok = false; return }
      sample[name] = num
    })
    const targetCol = multiModelTargetCol.value
    if (targetCol && typeof raw[targetCol] === 'string') {
      sample[targetCol] = raw[targetCol]
    }
    if (ok && Object.keys(sample).length > 0) return sample
    // fall through
  }

  // ── Generic path: flatten to numeric leaves, then case-insensitive
  //    match, then positional auto-fallback. Handles bare numbers, bare
  //    arrays, uppercase-keyed flat objects, nested objects, etc.
  const flat = flattenNumericLeaves(raw)
  const flatKeys = Object.keys(flat)
  if (flatKeys.length === 0) return null

  const targetCol = multiModelTargetCol.value

  // If channels are configured, do a two-pass fill:
  //  1) case-insensitive full-name match for each configured channel
  //  2) for any channel that didn't match, positional fallback from the
  //     UNUSED flattened leaves (in insertion order)
  // If ANY channel needed positional fallback, surface the warning banner.
  if (channels.length > 0) {
    const lowerIndex = {}
    for (const fk of flatKeys) lowerIndex[fk.toLowerCase()] = fk

    const sample = {}
    const usedKeys = new Set()      // flat keys already claimed by a match
    const unmatchedChannels = []    // configured channel names we still need

    // Pass 1 — case-insensitive full-name matches.
    for (const ch of channels) {
      const hit = lowerIndex[ch.toLowerCase()]
      if (hit !== undefined) {
        sample[ch] = flat[hit]
        usedKeys.add(hit)
      } else {
        unmatchedChannels.push(ch)
      }
    }

    // Pass 2 — positional fallback for unmatched channels from unused leaves.
    // BEFORE positional fill, drop obvious meta/time fields from the pool.
    // Without this guard, a numeric `_timestamp` / `epoch` / `ts` column
    // from the publisher takes the first fallback slot and offsets every
    // real sensor channel by one — the bug where "MQTT only worked when
    // the index column was named 'time'" (the publisher happened to strip
    // that specific name; any other alias slipped through).
    const META_FIELD_NAMES = new Set([
      '_timestamp', '_index',
      'timestamp', 'time', 'ts', 'datetime', 'date', 'epoch', 'unix_time',
      'elapsed', 'sec', 'seconds', 'ms', 'milliseconds', 't',
      'index', 'idx', 'i', 'row', 'row_id', 'sample_id',
    ])
    const isMetaField = (k) => {
      // Handle dotted keys from flattenNumericLeaves (e.g. "meta._timestamp")
      // by checking the leaf segment.
      const leaf = k.split('.').pop() || k
      return META_FIELD_NAMES.has(leaf.toLowerCase())
    }
    const unusedKeys = flatKeys.filter(k => !usedKeys.has(k) && !isMetaField(k))
    const fallbackConfigured = []
    const fallbackDetected = []
    for (let i = 0; i < unmatchedChannels.length && i < unusedKeys.length; i++) {
      const ch = unmatchedChannels[i]
      const src = unusedKeys[i]
      sample[ch] = flat[src]
      fallbackConfigured.push(ch)
      fallbackDetected.push(src)
    }

    // If any channel needed positional fallback, flag the session banner.
    if (fallbackConfigured.length > 0) {
      sensorAutoFallbackActive.value = true
      sensorAutoFallbackInfo.value = {
        configured: fallbackConfigured,
        detected: fallbackDetected,
      }
    }

    if (targetCol && raw && typeof raw === 'object' && typeof raw[targetCol] === 'string') {
      sample[targetCol] = raw[targetCol]
    }
    if (Object.keys(sample).length === 0) return null
    return sample
  }

  // No configured channels — publish the flattened leaves as-is under their
  // dotted keys and let the caller auto-detect channel names.
  const sample = { ...flat }
  if (autoDetectedChannels.value.length === 0) {
    autoDetectedChannels.value = flatKeys
  }
  if (targetCol && raw && typeof raw === 'object' && typeof raw[targetCol] === 'string') {
    sample[targetCol] = raw[targetCol]
  }
  return sample
}

function pushSensorSample(sample) {
  sensorBuffer.push(sample)
  sensorBufferLen.value = sensorBuffer.length

  const ws = liveWindowSize.value
  sensorBufferProgress.value = Math.min(sensorBuffer.length / ws, 1)

  if (sensorBuffer.length >= ws) {
    // Extract window and run inference
    const window = sensorBuffer.slice(0, ws)
    const stride = liveStride.value
    sensorBuffer = sensorBuffer.slice(stride)
    sensorBufferLen.value = sensorBuffer.length
    sensorBufferProgress.value = Math.min(sensorBuffer.length / ws, 1)
    runLiveInference(window)
  }
}

async function runLiveInference(windowData) {
  // Convert to CSV-like format for the pipeline runner
  const channels = liveChannels.value
  const csvRows = windowData.map(sample => {
    if (Array.isArray(sample)) return sample
    return channels.map((ch) => sample[ch] ?? 0)
  })

  // Extract target column values from MQTT data for live ground truth comparison
  let targetColValues = null
  const targetCol = multiModelTargetCol.value
  if (targetCol && isMultiModelApp.value && windowData.length > 0) {
    const firstSample = windowData[0]
    if (typeof firstSample === 'object' && !Array.isArray(firstSample) && targetCol in firstSample) {
      targetColValues = windowData.map(s => s[targetCol])
    }
  }

  // ── P2 Phase 3: Fast Mode ────────────────────────────────────────
  // If enabled + we have a feature list to compute + the worker is healthy,
  // extract features in the browser and POST a feature_vector. Server skips
  // its tsfresh path entirely. On any worker failure we fall back to the
  // raw payload for THIS window only — Fast Mode stays enabled for the next.
  let usedFastMode = false
  let fastPayload = null
  if (fastModeEnabled.value && fastModeAvailable.value && windowData.length > 0) {
    try {
      const fastRes = await extractWindowFeaturesFast(csvRows, channels)
      fastPayload = {
        feature_vector: fastRes.feature_vector,
        feature_names: fastRes.feature_names,
        target_values: targetColValues,
      }
      usedFastMode = true
    } catch (fmErr) {
      console.warn('[Fast Mode] falling back to raw for this window:', fmErr?.message || fmErr)
    }
  }

  try {
    const resp = await api.post(
      `/api/app-builder/run/${slug.value}`,
      usedFastMode ? fastPayload : {
        data: csvRows,
        channels: channels,
        target_values: targetColValues,
      },
      { timeout: 30000 },
    )

    liveInferenceCount.value++
    liveLastUpdated.value = Date.now()

    if (resp.data?.multi_model) {
      // Multi-model response — accumulate prediction history per model
      const models = resp.data.models || {}
      livePrediction.value = `${Object.keys(models).length} models compared`

      // Accumulate each model's predictions into history
      for (const [eid, m] of Object.entries(models)) {
        if (m.error) {
          // Log error only once per model
          if (!liveMultiHistory.value[eid]) {
            console.warn(`[Multi-Model] ${m.name || eid} error: ${m.error}`)
            liveMultiHistory.value[eid] = {
              name: m.name || eid, algorithm: m.algorithm || '', mode: m.mode || '',
              predictions: [], error: m.error,
            }
          }
          continue
        }
        if (!liveMultiHistory.value[eid]) {
          liveMultiHistory.value[eid] = {
            name: m.name, algorithm: m.algorithm, mode: m.mode,
            predictions: [],
          }
        }
        const hist = liveMultiHistory.value[eid]
        // Append latest prediction(s)
        if (m.predictions && m.predictions.length > 0) {
          hist.predictions.push(...m.predictions)
          if (hist.predictions.length > MAX_LIVE_HISTORY) {
            hist.predictions = hist.predictions.slice(-MAX_LIVE_HISTORY)
          }
        }
      }

      // Accumulate actual values from backend response (decoded labels)
      if (resp.data.actual && resp.data.actual.length > 0) {
        liveMultiActuals.value.push(...resp.data.actual)
        if (liveMultiActuals.value.length > MAX_LIVE_HISTORY) {
          liveMultiActuals.value = liveMultiActuals.value.slice(-MAX_LIVE_HISTORY)
        }
      }

      // Build accumulated result for display
      const accModels = {}
      for (const [eid, hist] of Object.entries(liveMultiHistory.value)) {
        accModels[eid] = {
          name: hist.name,
          algorithm: hist.algorithm,
          mode: hist.mode,
          predictions: [...hist.predictions],
          count: hist.predictions.length,
        }
        if (hist.error) accModels[eid].error = hist.error
      }
      const numWindows = Math.max(...Object.values(accModels).map(m => m.predictions?.length || 0), 0)
      // Include accumulated actuals if target column detected in MQTT stream
      const hasLiveActuals = liveMultiActuals.value.length > 0
      const actuals = hasLiveActuals ? [...liveMultiActuals.value] : resp.data.actual

      // Compute metrics client-side when we have live actuals
      if (actuals && actuals.length > 0) {
        const mode = multiModelMode.value || resp.data.mode
        for (const [eid, mm] of Object.entries(accModels)) {
          const preds = mm.predictions || []
          const len = Math.min(actuals.length, preds.length)
          if (len === 0) continue
          if (mode === 'regression') {
            const a = actuals.slice(0, len).map(Number)
            const p = preds.slice(0, len).map(Number)
            const meanA = a.reduce((s, v) => s + v, 0) / len
            const ssRes = a.reduce((s, v, i) => s + (v - p[i]) ** 2, 0)
            const ssTot = a.reduce((s, v) => s + (v - meanA) ** 2, 0)
            mm.r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0
            mm.rmse = Math.sqrt(ssRes / len)
            mm.mae = a.reduce((s, v, i) => s + Math.abs(v - p[i]), 0) / len
          } else if (mode === 'classification') {
            const aStr = actuals.slice(0, len).map(v => String(v).trim().toLowerCase())
            const pStr = preds.slice(0, len).map(v => String(v).trim().toLowerCase())
            let correct = 0
            for (let i = 0; i < len; i++) { if (aStr[i] === pStr[i]) correct++ }
            mm.accuracy = correct / len
            mm.precision = mm.accuracy  // simplified for live
            mm.f1 = mm.accuracy
          }
        }
      }

      // Accumulate signal preview for classification timeline (one sample per window)
      if (resp.data.signal_preview && resp.data.signal_preview.length > 0) {
        // Take the mean of current window's signal as one timeline point
        const winSig = resp.data.signal_preview
        const winMean = winSig.reduce((a, b) => a + b, 0) / winSig.length
        liveSignalHistory.value.push(winMean)
        if (liveSignalHistory.value.length > MAX_LIVE_HISTORY) {
          liveSignalHistory.value = liveSignalHistory.value.slice(-MAX_LIVE_HISTORY)
        }
      }

      result.value = {
        ...resp.data,
        models: accModels,
        num_windows: numWindows,
        actual: actuals,
        signal_preview: liveSignalHistory.value.length > 0 ? [...liveSignalHistory.value] : resp.data.signal_preview,
      }

      // Multi-model: update latest state + record rows per chosen granularity
      const mPreds = {}
      for (const [eid, m] of Object.entries(resp.data.models || {})) {
        if (m.error || !m.predictions || m.predictions.length === 0) continue
        const name = m.name || eid
        mPreds[name] = m.predictions[m.predictions.length - 1]
      }
      if (Object.keys(mPreds).length > 0) {
        recordInferenceRows({ windowData, modelPredictions: mPreds })
      }
    } else {
      // Single model response
      const preds = resp.data?.predictions || []
      const predsFull = resp.data?.predictions_full || []
      if (preds.length > 0) {
        const lastPred = preds[preds.length - 1]
        livePrediction.value = lastPred
        if (typeof lastPred === 'number') {
          // Regression: numeric history
          livePredictionHistory.value.push(lastPred)
          if (livePredictionHistory.value.length > MAX_LIVE_HISTORY) {
            livePredictionHistory.value = livePredictionHistory.value.slice(-MAX_LIVE_HISTORY)
          }
        } else {
          // Classification / anomaly: also keep the raw label list so
          // `predictions` on the accumulated result stays non-empty and
          // downstream tables/charts have something to iterate over.
          livePredictionHistory.value.push(lastPred)
          if (livePredictionHistory.value.length > MAX_LIVE_HISTORY) {
            livePredictionHistory.value = livePredictionHistory.value.slice(-MAX_LIVE_HISTORY)
          }
        }
        // Accumulate the rich per-window record (label + confidence + probs)
        // for classification/anomaly tables. Falls back to the plain label
        // if the backend didn't send predictions_full.
        if (predsFull.length > 0) {
          livePredictionHistoryFull.value.push(predsFull[predsFull.length - 1])
        } else {
          livePredictionHistoryFull.value.push(lastPred)
        }
        if (livePredictionHistoryFull.value.length > MAX_LIVE_HISTORY) {
          livePredictionHistoryFull.value = livePredictionHistoryFull.value.slice(-MAX_LIVE_HISTORY)
        }
      }

      // Extract actual value from selected column for live comparison (regression)
      if (liveActualColumn.value && liveActualColumn.value !== '(none)' && windowData.length > 0) {
        const col = liveActualColumn.value
        const firstSample = windowData[0]
        if (typeof firstSample === 'object' && !Array.isArray(firstSample) && col in firstSample) {
          const colValues = windowData.map(s => s[col]).filter(v => typeof v === 'number')
          if (colValues.length > 0) {
            const windowMean = colValues.reduce((a, b) => a + b, 0) / colValues.length
            liveActualHistory.value.push(windowMean)
            if (liveActualHistory.value.length > MAX_LIVE_HISTORY) {
              liveActualHistory.value = liveActualHistory.value.slice(-MAX_LIVE_HISTORY)
            }
          }
        }
      }

      result.value = {
        ...resp.data,
        predictions: livePredictionHistory.value.length > 0 ? [...livePredictionHistory.value] : resp.data?.predictions,
        predictions_full: livePredictionHistoryFull.value.length > 0 ? [...livePredictionHistoryFull.value] : resp.data?.predictions_full,
        count: livePredictionHistory.value.length || resp.data?.count,
        num_windows: livePredictionHistory.value.length || resp.data?.num_windows,
        actual: liveActualHistory.value.length > 0 ? [...liveActualHistory.value] : resp.data?.actual,
      }

      // Single-model: update latest state + record rows per chosen granularity
      if (preds.length > 0) {
        const lastPred = preds[preds.length - 1]
        const full = resp.data?.predictions_full
        const lastFull = Array.isArray(full) && full.length > 0 ? full[full.length - 1] : null
        const confidence = lastFull && typeof lastFull === 'object' && 'confidence' in lastFull
          ? lastFull.confidence : null
        const score = lastFull && typeof lastFull === 'object' && 'score' in lastFull
          ? lastFull.score : null
        recordInferenceRows({ windowData, prediction: lastPred, confidence, score })
      }
    }
  } catch (e) {
    console.error('Live inference error:', e)
  }
}

function onFileSelect(e) {
  selectedFile.value = e.target.files[0] || null
  result.value = null
  runError.value = null
}

function clearFile() {
  selectedFile.value = null
  result.value = null
  runError.value = null
  // Reset file input so the same file can be re-selected
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

async function runPipeline() {
  if (!selectedFile.value) return
  running.value = true
  runError.value = null
  result.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const resp = await api.post(`/api/app-builder/run/${slug.value}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
    result.value = resp.data
  } catch (e) {
    runError.value = e.response?.data?.error || e.message || 'Pipeline execution failed'
  }
  running.value = false
}
</script>

<style scoped>
.published-app {
  min-height: 100vh;
  background: #0d1117;
  color: #e6edf3;
}

/* Dashboard mode: full-viewport, no scroll. Wall-monitor friendly. */
.published-app.dashboard-mode {
  height: 100vh;
  min-height: 0;
  overflow: hidden;
}
.published-app.dashboard-mode .app-content {
  max-width: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.published-app.dashboard-mode .app-header {
  padding: 8px 16px;
  margin-bottom: 0;
  min-height: 56px;
  flex-shrink: 0;
}

.app-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
}

.app-content {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px 16px;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 0;
  border-bottom: 1px solid #21262d;
  margin-bottom: 24px;
}

.app-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-logo {
  width: 32px;
  height: 32px;
}

.app-title {
  font-size: 18px;
  font-weight: 600;
}

.app-subtitle {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
}

.app-mode-badge {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.5px;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}

.app-algo {
  font-size: 11px;
  color: #8b949e;
  font-family: monospace;
}

.app-powered {
  font-size: 10px;
  color: #484f58;
}

.app-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.app-section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #c9d1d9;
}

.app-dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px;
  border: 2px dashed #30363d;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s;
}
.app-dropzone:hover {
  border-color: #a78bfa;
}

.app-file-selected {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  font-size: 12px;
  font-family: monospace;
}

.result-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.result-stat {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 12px 16px;
  flex: 1;
}

.result-stat-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.result-stat-value {
  font-size: 20px;
  font-weight: 600;
  font-family: monospace;
  margin-top: 4px;
}

.result-table-wrap {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #21262d;
  border-radius: 6px;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: monospace;
}

.result-table th {
  background: #0d1117;
  padding: 8px 12px;
  text-align: left;
  color: #8b949e;
  font-weight: 600;
  border-bottom: 1px solid #21262d;
  position: sticky;
  top: 0;
}

.result-table td {
  padding: 6px 12px;
  border-bottom: 1px solid #161b22;
  color: #e6edf3;
}

.result-table tr:hover td {
  background: #161b22;
}

.result-json {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 12px;
  font-size: 11px;
  font-family: monospace;
  color: #8b949e;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.chart-container {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.chart-title-text {
  font-size: 13px;
  font-weight: 600;
  color: #c9d1d9;
}

.chart-legend-items {
  display: flex;
  align-items: center;
  gap: 4px;
}

.chart-legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.chart-legend-label {
  font-size: 10px;
  color: #8b949e;
}

.prediction-chart {
  width: 100%;
  height: auto;
}

.table-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
  cursor: pointer;
  font-size: 12px;
  color: #8b949e;
  user-select: none;
}
.table-toggle:hover {
  color: #c9d1d9;
}

.fast-mode-card {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 10px 14px;
}
.fast-mode-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
  background: rgba(139, 148, 158, 0.15);
  color: #8b949e;
}
.fast-mode-badge.on {
  background: rgba(46, 160, 67, 0.18);
  color: #56d364;
}
.rail-fast-mode {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 6px 8px;
}

.live-stat {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 80px;
}
.live-stat-label {
  font-size: 9px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.live-stat-value {
  font-size: 16px;
  font-weight: 600;
  font-family: monospace;
  color: #e6edf3;
}
.live-prediction {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.live-prediction-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}
.live-prediction-value {
  font-size: 28px;
  font-weight: 700;
  font-family: monospace;
}
.live-prediction-time {
  font-size: 10px;
  color: #484f58;
  margin-top: 4px;
}

.preview-panel {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 16px;
}
.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}
.preview-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.preview-card {
  flex: 1 1 140px;
  min-width: 140px;
  background: #010409;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 12px;
}
.preview-card-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 2px;
}
.preview-card-value {
  font-size: 24px;
  font-weight: 700;
  font-family: monospace;
  color: #e6edf3;
  line-height: 1.1;
}
.preview-card-stats {
  font-size: 10px;
  color: #8b949e;
  font-family: monospace;
  margin-top: 4px;
}
.preview-card-stats-sep {
  margin: 0 4px;
  color: #484f58;
}
.preview-chart {
  position: relative;
  height: 200px;
  background: #010409;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px;
}
.preview-chart-empty {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #484f58;
  font-size: 12px;
  background: #010409;
  border: 1px dashed #21262d;
  border-radius: 6px;
}

.recorder-section {
  margin-top: 12px;
}
.recorder-label-header {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}
.recorder-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.recorder-label-btn {
  padding: 6px 16px;
  border-radius: 6px;
  border: 2px solid #30363d;
  background: #0d1117;
  color: #8b949e;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.recorder-label-btn:hover {
  border-color: #555;
  color: #c9d1d9;
}
.recorder-label-btn.active {
  border-color: #34d399;
  background: rgba(52, 211, 153, 0.1);
  color: #34d399;
}
.recorder-timeline {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  gap: 1px;
}
.recorder-segment {
  min-width: 2px;
  border-radius: 2px;
  opacity: 0.7;
}

.expected-format {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 12px;
}

.expected-col {
  font-size: 10px;
  font-family: monospace;
  padding: 2px 6px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 3px;
  color: #a78bfa;
}

/* ═══════════════════════════════════════════════════════════ */
/* Dashboard (two-pane) layout                                 */
/* ═══════════════════════════════════════════════════════════ */
.dashboard-body {
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
  overflow: hidden;
  border-top: 1px solid #21262d;
}

/* ── Left rail ───────────────────────────────────────────── */
.dash-rail {
  width: 280px;
  flex-shrink: 0;
  background: #0d1117;
  border-right: 1px solid #21262d;
  display: flex;
  flex-direction: column;
  position: relative;
  padding: 12px;
  overflow-y: auto;
  transition: width 0.15s ease;
}
.dash-rail.collapsed {
  width: 72px;
  padding: 12px 6px;
  align-items: center;
}
.rail-toggle {
  position: absolute;
  top: 8px;
  right: 6px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  color: #8b949e;
  cursor: pointer;
  padding: 2px 4px;
  z-index: 2;
}
.rail-toggle:hover {
  color: #e6edf3;
  border-color: #a78bfa;
}
.dash-rail.collapsed .rail-toggle {
  right: 50%;
  transform: translateX(50%);
  top: 8px;
}

.rail-icons {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  margin-top: 40px;
  width: 100%;
}
.rail-icon-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 6px 4px;
  width: 60px;
  cursor: default;
}
.rail-icon-btn.rail-icon-btn-danger {
  cursor: pointer;
}
.rail-icon-btn.rail-icon-btn-danger:hover {
  border-color: #ef5350;
  background: rgba(239, 83, 80, 0.08);
}
.rail-icon-lbl {
  font-size: 9px;
  color: #8b949e;
  font-family: monospace;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.rail-expanded {
  margin-top: 28px;
  display: flex;
  flex-direction: column;
}
.rail-section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 700;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
  margin-top: 4px;
}
.rail-record-group {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 8px 10px;
  margin-top: 6px;
}

/* ── Right pane ─────────────────────────────────────────── */
.dash-main {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: 12px 16px;
  overflow: hidden;
  gap: 10px;
}

.dash-stats-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  flex-wrap: nowrap;
}
.dash-stats-strip .live-stat {
  min-width: 90px;
  padding: 6px 12px;
}
.dash-stats-strip .live-stat-value {
  font-size: 14px;
}
.dash-progress {
  flex: 1;
  min-width: 100px;
}

.dash-content {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
}

.dash-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #0d1117;
  border: 1px dashed #30363d;
  border-radius: 8px;
  padding: 24px;
}

/* Single-model layout */
.dash-single {
  display: grid;
  grid-template-columns: minmax(340px, 380px) 1fr;
  grid-template-rows: 1fr;
  gap: 12px;
  height: 100%;
}
.dash-prediction-card {
  background: #0d1117;
  border: 2px solid #21262d;
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.dash-prediction-value {
  font-size: 68px;
  font-weight: 800;
  font-family: monospace;
  line-height: 1.05;
  margin: 8px 0 4px;
  word-break: break-word;
}
.dash-prediction-value.fs-S { font-size: 32px; }
.dash-prediction-value.fs-M { font-size: 48px; }
.dash-prediction-value.fs-L { font-size: 68px; }
.dash-prediction-value.fs-XL { font-size: 96px; line-height: 1; }
.dash-table-container {
  flex: 1;
  overflow: auto;
  border: 1px solid #21262d;
  border-radius: 8px;
  background: #0d1117;
  padding: 0;
}
.dash-table {
  width: 100%;
  border-collapse: collapse;
  color: #e6edf3;
  font-family: monospace;
  font-size: 13px;
}
.dash-table th {
  background: #161b22;
  padding: 8px 10px;
  text-align: left;
  border-bottom: 1px solid #21262d;
  color: #8b949e;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.5px;
  position: sticky;
  top: 0;
  z-index: 1;
}
.dash-table td {
  padding: 6px 10px;
  border-bottom: 1px solid rgba(33, 38, 45, 0.5);
}
.dash-table tbody tr:first-child td {
  color: #a78bfa;
  font-weight: 700;
}
.dash-confidence {
  width: 100%;
  margin-top: 24px;
}
.dash-confidence-label {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.dash-chart-container {
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.dash-svg-chart {
  flex: 1 1 auto;
  width: 100%;
  height: 100%;
  min-height: 0;
}

/* Multi-model tile grid */
.dash-multi-grid {
  overflow: hidden;
}
.multi-tile-grid {
  flex: 1 1 auto;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  grid-auto-rows: minmax(220px, 1fr);
  gap: 12px;
  overflow-y: auto;
  align-content: start;
  min-height: 0;
}
.multi-tile {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}
.multi-tile-error {
  border-color: rgba(248, 113, 113, 0.4);
  background: rgba(248, 113, 113, 0.05);
}
.multi-tile-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.multi-tile-name {
  font-size: 13px;
  font-weight: 700;
  color: #e6edf3;
  display: flex;
  align-items: center;
  gap: 4px;
}
.multi-tile-pred {
  font-size: 30px;
  font-weight: 700;
  font-family: monospace;
  line-height: 1;
  margin: 2px 0;
  word-break: break-word;
}
.multi-tile-metrics {
  display: flex;
  gap: 10px;
  font-size: 11px;
  font-family: monospace;
  color: #8b949e;
}
.multi-tile-error-msg {
  color: #f87171;
  font-size: 11px;
  font-family: monospace;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
}
.multi-tile-chart {
  flex: 1 1 auto;
  min-height: 80px;
  max-height: 140px;
  position: relative;
}

/* Recorder in dashboard mode */
.dash-recorder {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dash-preview {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  min-height: 0;
  margin-bottom: 0;
}
.dash-preview-chart {
  flex: 1 1 auto;
  min-height: 0;
  height: auto;
}
.dash-recorder-stats {
  display: flex;
  gap: 10px;
  flex-shrink: 0;
}

/* Ensure the recorder-labels wrap nicely in the rail */
.dash-rail .recorder-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.dash-rail .recorder-label-btn {
  padding: 4px 10px;
  font-size: 11px;
}

/* ── Raw MQTT diagnostic panel (Layer 3) ────────────────── */
.raw-mqtt-controls {
  display: flex;
  justify-content: flex-end;
  margin: 4px 0;
}
.raw-mqtt-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 6px 0 10px;
}
.raw-mqtt-block {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  padding: 10px 12px;
  margin: 0;
  font-size: 11px;
  font-family: monospace;
  color: #8b949e;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
