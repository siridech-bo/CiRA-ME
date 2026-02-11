<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="features" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Feature Engineering</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Extract and select optimal features for {{ pipelineStore.mode === 'anomaly' ? 'anomaly detection' : 'classification' }}
    </p>

    <!-- Workflow Tabs -->
    <v-tabs v-model="activeTab" class="mb-4">
      <v-tab value="extract">1. Extract</v-tab>
      <v-tab value="select" :disabled="!extractionResult">2. Select</v-tab>
      <v-tab value="visualize" :disabled="!extractionResult">3. Visualize</v-tab>
    </v-tabs>

    <v-window v-model="activeTab">
      <!-- Tab 1: Feature Extraction -->
      <v-window-item value="extract">
        <v-row>
          <!-- Feature Selection -->
          <v-col cols="12" md="8">
            <v-card class="pa-4">
              <!-- Extraction Mode Toggle -->
              <div class="d-flex align-center mb-4">
                <v-btn-toggle
                  v-model="extractionMode"
                  mandatory
                  density="compact"
                  color="primary"
                >
                  <v-btn value="lightweight" size="small">
                    <v-icon start size="small">mdi-lightning-bolt</v-icon>
                    Lightweight (44 features)
                  </v-btn>
                  <v-btn value="tsfresh" size="small">
                    <v-icon start size="small">mdi-atom</v-icon>
                    TSFresh Library (800+)
                  </v-btn>
                </v-btn-toggle>
              </div>

              <!-- TSFresh Library Mode -->
              <div v-if="extractionMode === 'tsfresh'" class="mb-4">
                <v-alert type="info" variant="tonal" class="mb-4">
                  <strong>TSFresh Library</strong> - Extract comprehensive time-series features using the real tsfresh library with hypothesis-tested feature extraction.
                </v-alert>

                <v-select
                  v-model="tsfreshFeatureSet"
                  :items="tsfreshFeatureSets"
                  item-title="name"
                  item-value="value"
                  label="Feature Set"
                  density="compact"
                  class="mb-4"
                >
                  <template #item="{ item, props }">
                    <v-list-item v-bind="props">
                      <template #subtitle>
                        {{ item.raw.description }}
                      </template>
                    </v-list-item>
                  </template>
                </v-select>

                <v-alert
                  :type="tsfreshFeatureSet === 'comprehensive' ? 'warning' : 'info'"
                  variant="tonal"
                  density="compact"
                >
                  <template v-if="tsfreshFeatureSet === 'minimal'">
                    ~10 features per sensor - Fast extraction with essential statistics
                  </template>
                  <template v-else-if="tsfreshFeatureSet === 'efficient'">
                    ~100 features per sensor - Balanced extraction without slow calculators
                  </template>
                  <template v-else>
                    ~800 features per sensor - Full comprehensive extraction (may take longer)
                  </template>
                </v-alert>
              </div>

              <!-- Lightweight Mode (Original) -->
              <div v-else>
                <div class="d-flex align-center mb-4">
                  <h3 class="text-subtitle-1 font-weight-bold">Available Features (44)</h3>
                  <v-spacer />
                  <v-btn-group density="compact">
                    <v-btn size="small" @click="selectAll">Select All</v-btn>
                    <v-btn size="small" @click="clearSelection">Clear</v-btn>
                    <v-btn size="small" @click="selectTSFresh">TSFresh</v-btn>
                    <v-btn size="small" @click="selectDSP">DSP</v-btn>
                  </v-btn-group>
                </div>

              <!-- Search -->
              <v-text-field
                v-model="searchQuery"
                prepend-inner-icon="mdi-magnify"
                label="Search features..."
                hide-details
                density="compact"
                class="mb-4"
              />

              <!-- Feature List -->
              <v-list
                density="compact"
                class="feature-list"
                max-height="400"
                style="overflow-y: auto"
              >
                <v-list-subheader>TSFresh Features (Statistical)</v-list-subheader>
                <v-list-item
                  v-for="feature in filteredTSFreshFeatures"
                  :key="feature"
                  :class="{ 'selected': selectedFeatures.includes(feature) }"
                  @click="toggleFeature(feature)"
                >
                  <template #prepend>
                    <v-checkbox
                      :model-value="selectedFeatures.includes(feature)"
                      hide-details
                      density="compact"
                      @click.stop="toggleFeature(feature)"
                    />
                  </template>
                  <v-list-item-title>{{ feature }}</v-list-item-title>
                  <template #append>
                    <v-chip size="x-small" color="info" variant="flat">TSFresh</v-chip>
                  </template>
                </v-list-item>

                <v-divider class="my-2" />

                <v-list-subheader>Custom DSP Features (Frequency Domain)</v-list-subheader>
                <v-list-item
                  v-for="feature in filteredDSPFeatures"
                  :key="feature"
                  :class="{ 'selected': selectedFeatures.includes(feature) }"
                  @click="toggleFeature(feature)"
                >
                  <template #prepend>
                    <v-checkbox
                      :model-value="selectedFeatures.includes(feature)"
                      hide-details
                      density="compact"
                      @click.stop="toggleFeature(feature)"
                    />
                  </template>
                  <v-list-item-title>{{ feature }}</v-list-item-title>
                  <template #append>
                    <v-chip size="x-small" color="secondary" variant="flat">DSP</v-chip>
                  </template>
                </v-list-item>
              </v-list>

                <!-- Selection Summary -->
                <v-alert type="info" variant="tonal" class="mt-4">
                  <strong>{{ selectedFeatures.length }}</strong> features selected
                  ({{ selectedFeatures.length * sensorColumns }} total with {{ sensorColumns }} sensor channels)
                </v-alert>
              </div>
            </v-card>
          </v-col>

          <!-- LLM Assistant -->
          <v-col cols="12" md="4">
            <v-card class="llm-assistant pa-4">
              <div class="d-flex align-center mb-4">
                <v-icon color="secondary" class="mr-2">mdi-robot</v-icon>
                <h3 class="text-subtitle-1 font-weight-bold">LLM Assistant</h3>
                <v-spacer />
                <v-chip
                  size="x-small"
                  :color="llmStatus?.available ? 'success' : 'error'"
                  variant="flat"
                >
                  <v-icon
                    size="x-small"
                    :icon="llmStatus?.available ? 'mdi-check-circle' : 'mdi-alert-circle'"
                    class="mr-1"
                  />
                  {{ llmStatus?.model || 'Ollama' }}
                </v-chip>
              </div>

              <!-- LLM Status Details -->
              <div v-if="llmStatus" class="mb-4">
                <div v-if="llmStatus.available" class="d-flex align-center mb-2">
                  <v-icon
                    size="small"
                    :color="llmStatus.gpu_loaded ? 'success' : 'warning'"
                    class="mr-2"
                  >
                    {{ llmStatus.gpu_loaded ? 'mdi-chip' : 'mdi-memory' }}
                  </v-icon>
                  <span class="text-caption">
                    {{ llmStatus.gpu_loaded ? 'GPU Accelerated' : 'CPU Mode' }}
                    <span v-if="llmStatus.gpu_info?.vram_used_mb" class="text-medium-emphasis">
                      ({{ llmStatus.gpu_info.vram_used_mb.toFixed(0) }} MB VRAM)
                    </span>
                  </span>
                </div>
                <div v-else class="text-caption text-error mb-2">
                  {{ llmStatus.error || 'LLM service not available' }}
                </div>
              </div>

              <!-- Recommendations -->
              <div v-if="recommendations" class="assistant-message mb-4">
                <div class="d-flex align-center mb-2">
                  <p class="mb-0">Recommended features:</p>
                  <v-chip
                    v-if="recommendations.llm_used"
                    size="x-small"
                    color="secondary"
                    variant="tonal"
                    class="ml-2"
                  >
                    LLM
                  </v-chip>
                  <v-chip
                    v-else
                    size="x-small"
                    color="grey"
                    variant="tonal"
                    class="ml-2"
                  >
                    Rule-based
                  </v-chip>
                </div>
                <ul class="pl-4">
                  <li v-for="feature in recommendations.recommended_features.slice(0, 8)" :key="feature">
                    {{ feature }}
                  </li>
                </ul>
              </div>

              <div v-if="recommendations?.reasoning" class="mb-4">
                <p
                  v-for="(reason, index) in recommendations.reasoning"
                  :key="index"
                  class="text-caption text-medium-emphasis mb-1"
                >
                  • {{ reason }}
                </p>
              </div>

              <v-btn
                color="secondary"
                block
                :loading="loadingRecommendations"
                :disabled="!pipelineStore.windowedSession"
                @click="getRecommendations"
              >
                <v-icon start>mdi-auto-fix</v-icon>
                Get Recommendations
              </v-btn>

              <v-btn
                v-if="recommendations"
                color="primary"
                variant="outlined"
                block
                class="mt-2"
                @click="applyRecommendations"
              >
                Apply Recommendations
              </v-btn>
            </v-card>

            <!-- Feature Count Summary Card -->
            <v-card v-if="extractionResult" class="pa-4 mt-4 feature-count-card">
              <h3 class="text-subtitle-1 font-weight-bold mb-4">
                <v-icon color="success" class="mr-2">mdi-check-circle</v-icon>
                Feature Status
              </h3>

              <!-- Feature Count Display -->
              <div class="feature-counts mb-4">
                <div class="count-item">
                  <div class="count-label">Extracted</div>
                  <div class="count-value text-info">{{ extractionResult.num_features }}</div>
                </div>
                <v-icon class="mx-2">mdi-arrow-right</v-icon>
                <div class="count-item">
                  <div class="count-label">Selected</div>
                  <div class="count-value" :class="selectionResult ? 'text-success' : 'text-medium-emphasis'">
                    {{ selectionResult ? selectionResult.final_count : '-' }}
                  </div>
                </div>
                <v-icon class="mx-2">mdi-arrow-right</v-icon>
                <div class="count-item">
                  <div class="count-label">For Training</div>
                  <div class="count-value" :class="appliedSelection ? 'text-primary font-weight-bold' : 'text-medium-emphasis'">
                    {{ appliedSelection ? appliedSelection.num_features : (selectionResult ? 'Not Applied' : extractionResult.num_features) }}
                  </div>
                </div>
              </div>

              <!-- Status Alert -->
              <v-alert
                v-if="selectionResult && !appliedSelection"
                type="warning"
                variant="tonal"
                density="compact"
                class="mb-4"
              >
                <v-icon size="small" class="mr-1">mdi-alert</v-icon>
                Selection not applied yet. Click "Apply Selection" to use selected features for training.
              </v-alert>

              <v-alert
                v-else-if="appliedSelection"
                type="success"
                variant="tonal"
                density="compact"
                class="mb-4"
              >
                <v-icon size="small" class="mr-1">mdi-check</v-icon>
                <strong>{{ appliedSelection.num_features }}</strong> features ready for training.
              </v-alert>

              <v-alert
                v-else
                type="info"
                variant="tonal"
                density="compact"
                class="mb-4"
              >
                <strong>{{ extractionResult.num_features }}</strong> features extracted
                from <strong>{{ extractionResult.num_windows }}</strong> windows.
                Proceed to selection or use all features for training.
              </v-alert>

              <!-- TSFresh-specific info -->
              <div v-if="extractionResult.feature_set" class="mb-4">
                <v-chip size="small" color="secondary" variant="tonal" class="mr-2">
                  <v-icon start size="small">mdi-atom</v-icon>
                  TSFresh {{ extractionResult.feature_set }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">
                  {{ extractionResult.num_features }} features across {{ sensorColumns }} sensors
                </span>
              </div>

              <!-- Recommendation for FRESH selection when using tsfresh -->
              <v-alert
                v-if="extractionResult.feature_set && extractionResult.num_features > 100"
                type="info"
                variant="tonal"
                density="compact"
                class="mb-4"
              >
                <strong>Tip:</strong> Use FRESH selection to automatically identify statistically significant features.
              </v-alert>

              <v-btn
                color="primary"
                block
                @click="activeTab = 'select'"
              >
                <v-icon start>mdi-filter-variant</v-icon>
                Continue to Selection
              </v-btn>
            </v-card>
          </v-col>
        </v-row>
      </v-window-item>

      <!-- Tab 2: Feature Selection -->
      <v-window-item value="select">
        <v-row>
          <v-col cols="12" md="8">
            <v-card class="pa-4">
              <div class="d-flex align-center mb-4">
                <v-icon color="primary" class="mr-2">mdi-filter-variant</v-icon>
                <h3 class="text-subtitle-1 font-weight-bold">Intelligent Feature Selection</h3>
              </div>

              <p class="text-body-2 text-medium-emphasis mb-4">
                Reduce the {{ extractionResult?.num_features || 0 }} extracted features to an optimal subset
                using statistical methods and LLM-powered analysis.
              </p>

              <!-- Selection Method -->
              <v-select
                v-model="selectionMethod"
                :items="selectionMethods"
                item-title="name"
                item-value="value"
                label="Selection Method"
                density="compact"
                class="mb-4"
              >
                <template #item="{ item, props }">
                  <v-list-item v-bind="props">
                    <template #subtitle>
                      {{ item.raw.description }}
                    </template>
                  </v-list-item>
                </template>
              </v-select>

              <!-- FRESH-specific options -->
              <div v-if="selectionMethod === 'fresh' || selectionMethod === 'fresh_combined'" class="mb-4">
                <v-alert type="info" variant="tonal" density="compact" class="mb-4">
                  <strong>FRESH Algorithm</strong> - Feature Extraction based on Scalable Hypothesis tests.
                  Uses statistical hypothesis testing with Benjamini-Hochberg FDR correction.
                  <template v-if="selectionMethod === 'fresh_combined'">
                    <br><strong>+ Target Count</strong> - Then reduces to your specified target using mutual information ranking.
                  </template>
                </v-alert>

                <v-slider
                  v-model="fdrLevel"
                  :min="0.01"
                  :max="0.20"
                  :step="0.01"
                  label="FDR Level"
                  thumb-label
                  :thumb-size="24"
                >
                  <template #thumb-label="{ modelValue }">
                    {{ (modelValue * 100).toFixed(0) }}%
                  </template>
                </v-slider>
                <p class="text-caption text-medium-emphasis mt-n2 mb-4">
                  False Discovery Rate level (default 5%). Lower values are more conservative.
                </p>

                <!-- Target count for chained selection -->
                <v-slider
                  v-if="selectionMethod === 'fresh_combined'"
                  v-model="targetFeatures"
                  :min="5"
                  :max="Math.min(100, extractionResult?.num_features || 100)"
                  :step="1"
                  label="Target Features"
                  thumb-label
                  class="mb-4"
                />
              </div>

              <!-- Target Features (for non-FRESH methods) -->
              <v-slider
                v-if="selectionMethod !== 'fresh' && selectionMethod !== 'fresh_combined'"
                v-model="targetFeatures"
                :min="5"
                :max="Math.min(50, extractionResult?.num_features || 50)"
                :step="1"
                label="Target Features"
                thumb-label
                class="mb-4"
              />

              <!-- Run Selection Button -->
              <v-btn
                v-if="selectionMethod === 'fresh'"
                color="secondary"
                :loading="selectingFeatures"
                @click="runFRESHSelection"
              >
                <v-icon start>mdi-flask</v-icon>
                Run FRESH Selection
              </v-btn>

              <v-btn
                v-else-if="selectionMethod === 'fresh_combined'"
                color="secondary"
                :loading="selectingFeatures"
                @click="runFRESHCombinedSelection"
              >
                <v-icon start>mdi-flask-plus</v-icon>
                Run FRESH + Target
              </v-btn>

              <v-btn
                v-else
                color="secondary"
                :loading="selectingFeatures"
                @click="runFeatureSelection"
              >
                <v-icon start>mdi-auto-fix</v-icon>
                Run Selection
              </v-btn>

              <v-btn
                v-if="llmStatus?.available && selectionMethod !== 'fresh' && selectionMethod !== 'fresh_combined'"
                color="primary"
                class="ml-2"
                :loading="selectingFeatures"
                @click="runLLMSelection"
              >
                <v-icon start>mdi-robot</v-icon>
                LLM Selection
              </v-btn>
            </v-card>

            <!-- Selection Results -->
            <v-card v-if="selectionResult" class="pa-4 mt-4">
              <div class="d-flex align-center mb-4">
                <v-icon color="success" class="mr-2">mdi-check-circle</v-icon>
                <h3 class="text-subtitle-1 font-weight-bold">Selection Results</h3>
                <v-spacer />
                <v-chip
                  size="small"
                  :color="selectionResult.llm_used ? 'secondary' : 'info'"
                  variant="flat"
                >
                  {{ selectionResult.llm_used ? 'LLM-Powered' : 'Statistical' }}
                </v-chip>
              </div>

              <v-alert type="success" variant="tonal" class="mb-4">
                Reduced from <strong>{{ selectionResult.original_count }}</strong> to
                <strong>{{ selectionResult.final_count }}</strong> features
              </v-alert>

              <!-- Selection Log -->
              <div v-if="selectionResult.selection_log?.length" class="mb-4">
                <h4 class="text-subtitle-2 mb-2">Selection Steps:</h4>
                <p
                  v-for="(log, idx) in selectionResult.selection_log"
                  :key="idx"
                  class="text-caption text-medium-emphasis mb-1"
                >
                  {{ idx + 1 }}. {{ log }}
                </p>
              </div>

              <!-- LLM Reasoning -->
              <div v-if="selectionResult.reasoning?.length" class="mb-4">
                <h4 class="text-subtitle-2 mb-2">LLM Reasoning:</h4>
                <p
                  v-for="(reason, idx) in selectionResult.reasoning"
                  :key="idx"
                  class="text-caption text-medium-emphasis mb-1"
                >
                  • {{ reason }}
                </p>
              </div>

              <!-- FRESH-specific results -->
              <div v-if="selectionResult.fdr_level !== undefined" class="mb-4">
                <h4 class="text-subtitle-2 mb-2">
                  {{ selectionResult.method === 'fresh_combined' ? 'FRESH + Target Selection Details:' : 'FRESH Selection Details:' }}
                </h4>
                <div class="d-flex flex-wrap gap-2 mb-2">
                  <v-chip size="small" color="info" variant="tonal">
                    FDR: {{ (selectionResult.fdr_level * 100).toFixed(0) }}%
                  </v-chip>
                  <v-chip v-if="selectionResult.after_fresh_count" size="small" color="secondary" variant="tonal">
                    After FRESH: {{ selectionResult.after_fresh_count }}
                  </v-chip>
                  <v-chip v-if="selectionResult.target_features" size="small" color="primary" variant="tonal">
                    Target: {{ selectionResult.target_features }}
                  </v-chip>
                </div>
                <p class="text-caption text-medium-emphasis">
                  <template v-if="selectionResult.method === 'fresh_combined'">
                    Step 1: FRESH hypothesis testing with Benjamini-Hochberg FDR correction.
                    Step 2: Mutual information ranking to reach target count.
                  </template>
                  <template v-else>
                    Features selected based on statistical significance using Benjamini-Hochberg procedure.
                  </template>
                </p>
              </div>

              <!-- Feature Importance Chart -->
              <div v-if="importanceChartData.labels.length" style="height: 300px" class="mb-4">
                <Bar :data="importanceChartData" :options="importanceChartOptions" />
              </div>

              <!-- Selected Features List -->
              <h4 class="text-subtitle-2 mb-2">Selected Features:</h4>
              <div class="d-flex flex-wrap gap-2">
                <v-chip
                  v-for="feat in selectionResult.selected_features"
                  :key="feat"
                  size="small"
                  :color="getFeatureTypeColor(feat)"
                  variant="tonal"
                >
                  {{ feat }}
                </v-chip>
              </div>
            </v-card>
          </v-col>

          <!-- Removed Features Sidebar -->
          <v-col cols="12" md="4">
            <v-card v-if="selectionResult?.removed_features" class="pa-4">
              <h3 class="text-subtitle-1 font-weight-bold mb-4">Filtered Features</h3>

              <v-expansion-panels variant="accordion">
                <v-expansion-panel
                  v-if="selectionResult.removed_features.low_variance?.length"
                >
                  <v-expansion-panel-title>
                    <v-icon size="small" class="mr-2">mdi-chart-line-variant</v-icon>
                    Low Variance ({{ selectionResult.removed_features.low_variance.length }})
                  </v-expansion-panel-title>
                  <v-expansion-panel-text>
                    <v-chip
                      v-for="feat in selectionResult.removed_features.low_variance"
                      :key="feat"
                      size="x-small"
                      class="ma-1"
                      variant="outlined"
                    >
                      {{ feat }}
                    </v-chip>
                  </v-expansion-panel-text>
                </v-expansion-panel>

                <v-expansion-panel
                  v-if="selectionResult.removed_features.high_correlation?.length"
                >
                  <v-expansion-panel-title>
                    <v-icon size="small" class="mr-2">mdi-chart-scatter-plot</v-icon>
                    High Correlation ({{ selectionResult.removed_features.high_correlation.length }})
                  </v-expansion-panel-title>
                  <v-expansion-panel-text>
                    <v-chip
                      v-for="feat in selectionResult.removed_features.high_correlation"
                      :key="feat"
                      size="x-small"
                      class="ma-1"
                      variant="outlined"
                    >
                      {{ feat }}
                    </v-chip>
                  </v-expansion-panel-text>
                </v-expansion-panel>
              </v-expansion-panels>
            </v-card>

            <!-- Apply Selection Card -->
            <v-card v-if="selectionResult" class="pa-4 mt-4">
              <h3 class="text-subtitle-1 font-weight-bold mb-4">Apply Selection</h3>
              <p class="text-caption text-medium-emphasis mb-4">
                Create a reduced feature set with only the selected features for training.
              </p>
              <v-btn
                color="primary"
                block
                :loading="applyingSelection"
                @click="applyFeatureSelection"
              >
                <v-icon start>mdi-check</v-icon>
                Apply Selection
              </v-btn>
            </v-card>
          </v-col>
        </v-row>
      </v-window-item>

      <!-- Tab 3: Visualization -->
      <v-window-item value="visualize">
        <v-row>
          <v-col cols="12" md="8">
            <v-card class="pa-4">
              <div class="d-flex align-center mb-4">
                <v-icon color="primary" class="mr-2">mdi-chart-histogram</v-icon>
                <h3 class="text-subtitle-1 font-weight-bold">Feature Distribution</h3>
              </div>

              <v-autocomplete
                v-model="selectedFeatureForViz"
                :items="extractedFeatureNames"
                label="Select Feature to Visualize"
                density="compact"
                hide-details
                class="mb-4"
                clearable
                auto-select-first
              >
                <template #item="{ item, props }">
                  <v-list-item v-bind="props">
                    <template #append>
                      <v-chip size="x-small" :color="getFeatureTypeColor(item.value)" variant="flat">
                        {{ getFeatureType(item.value) }}
                      </v-chip>
                    </template>
                  </v-list-item>
                </template>
              </v-autocomplete>

              <div v-if="loadingDistribution" class="d-flex justify-center align-center" style="height: 300px">
                <v-progress-circular indeterminate color="primary" />
              </div>

              <div v-else-if="featureDistribution" style="height: 300px">
                <Bar :data="distributionChartData" :options="distributionChartOptions" />
              </div>

              <div v-else class="d-flex justify-center align-center text-medium-emphasis" style="height: 300px">
                Select a feature to view its distribution
              </div>
            </v-card>
          </v-col>

          <v-col cols="12" md="4">
            <!-- Feature Statistics -->
            <v-card class="pa-4">
              <h4 class="text-subtitle-2 font-weight-bold mb-3">Feature Statistics</h4>

              <div v-if="featureDistribution?.statistics">
                <div class="stat-row">
                  <span class="text-medium-emphasis">Mean:</span>
                  <span class="font-weight-medium">{{ featureDistribution.statistics.mean.toFixed(4) }}</span>
                </div>
                <div class="stat-row">
                  <span class="text-medium-emphasis">Std Dev:</span>
                  <span class="font-weight-medium">{{ featureDistribution.statistics.std.toFixed(4) }}</span>
                </div>
                <div class="stat-row">
                  <span class="text-medium-emphasis">Min:</span>
                  <span class="font-weight-medium">{{ featureDistribution.statistics.min.toFixed(4) }}</span>
                </div>
                <div class="stat-row">
                  <span class="text-medium-emphasis">Max:</span>
                  <span class="font-weight-medium">{{ featureDistribution.statistics.max.toFixed(4) }}</span>
                </div>
                <div class="stat-row">
                  <span class="text-medium-emphasis">Median:</span>
                  <span class="font-weight-medium">{{ featureDistribution.statistics.median.toFixed(4) }}</span>
                </div>

                <v-divider class="my-3" />

                <div class="text-caption text-medium-emphasis">
                  Total samples: {{ featureDistribution.statistics.count }}
                </div>
              </div>

              <div v-else class="text-caption text-medium-emphasis">
                Select a feature to view statistics
              </div>
            </v-card>

            <!-- Label Distribution Summary -->
            <v-card v-if="featurePreview?.label_counts" class="pa-4 mt-4">
              <h4 class="text-subtitle-2 font-weight-bold mb-3">Label Distribution</h4>
              <div v-for="(count, label) in featurePreview.label_counts" :key="label" class="stat-row">
                <v-chip size="x-small" :color="getLabelColor(label as string)" class="mr-2">
                  {{ label }}
                </v-chip>
                <span class="font-weight-medium">{{ count }} samples</span>
              </div>
            </v-card>
          </v-col>
        </v-row>
      </v-window-item>
    </v-window>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="router.push({ name: 'pipeline-windowing' })"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          v-if="activeTab === 'extract'"
          color="secondary"
          size="large"
          class="mr-2"
          :loading="extracting"
          :disabled="extractionMode === 'lightweight' && selectedFeatures.length === 0"
          @click="extractionMode === 'tsfresh' ? extractTSFreshFeatures() : extractFeatures()"
        >
          <v-icon start>{{ extractionMode === 'tsfresh' ? 'mdi-atom' : 'mdi-lightning-bolt' }}</v-icon>
          {{ extractionMode === 'tsfresh' ? 'Extract TSFresh Features' : 'Extract Features' }}
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :disabled="!extractionResult && !appliedSelection"
          @click="router.push({ name: 'pipeline-training' })"
        >
          Continue to Training
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

// Tab state
const activeTab = ref('extract')

// Extraction mode
const extractionMode = ref<'lightweight' | 'tsfresh'>('lightweight')
const tsfreshFeatureSet = ref('efficient')
const tsfreshFeatureSets = [
  { name: 'Minimal (~10 features/sensor)', value: 'minimal', description: 'Fast extraction with essential statistics only' },
  { name: 'Efficient (~100 features/sensor)', value: 'efficient', description: 'Balanced extraction without slow calculators' },
  { name: 'Comprehensive (~800 features/sensor)', value: 'comprehensive', description: 'Full tsfresh extraction with all features' }
]

const searchQuery = ref('')
const selectedFeatures = ref<string[]>([])
const recommendations = ref<any>(null)
const extractionResult = ref<any>(null)
const loadingRecommendations = ref(false)
const extracting = ref(false)

// Feature selection state
const selectionMethod = ref('combined')
const targetFeatures = ref(15)
const fdrLevel = ref(0.05)
const selectionResult = ref<any>(null)
const selectingFeatures = ref(false)
const applyingSelection = ref(false)
const appliedSelection = ref<any>(null)

const selectionMethods = [
  { name: 'Combined (Recommended)', value: 'combined', description: 'Variance + Correlation + Mutual Information filters' },
  { name: 'FRESH (tsfresh)', value: 'fresh', description: 'Hypothesis testing with Benjamini-Hochberg FDR correction' },
  { name: 'FRESH + Target Count', value: 'fresh_combined', description: 'FRESH filtering then reduce to target count' },
  { name: 'Variance Filter', value: 'variance', description: 'Remove low-variance features' },
  { name: 'Correlation Filter', value: 'correlation', description: 'Remove highly correlated redundant features' },
  { name: 'Mutual Information', value: 'mutual_info', description: 'Rank features by mutual information with target' },
  { name: 'ANOVA F-Score', value: 'anova', description: 'Select features by ANOVA F-value' }
]

// Feature visualization state
const featurePreview = ref<any>(null)
const featureDistribution = ref<any>(null)
const selectedFeatureForViz = ref<string>('')
const loadingDistribution = ref(false)

// LLM status
const llmStatus = ref<any>(null)

// TSFresh statistical features (25 features)
const tsfreshFeatures = [
  'mean', 'std', 'min', 'max', 'median', 'sum', 'variance',
  'skewness', 'kurtosis', 'abs_energy', 'root_mean_square',
  'mean_abs_change', 'mean_change', 'count_above_mean', 'count_below_mean',
  'first_location_of_maximum', 'first_location_of_minimum',
  'last_location_of_maximum', 'last_location_of_minimum',
  'percentage_of_reoccurring_values', 'sum_of_reoccurring_values',
  'abs_sum_of_changes', 'range', 'interquartile_range', 'mean_second_derivative'
]

// Custom DSP features (21 features)
const dspFeatures = [
  'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
  'impulse_factor', 'margin_factor', 'zero_crossing_rate',
  'autocorr_lag1', 'autocorr_lag5', 'binned_entropy',
  'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
  'spectral_flatness', 'spectral_entropy', 'peak_frequency',
  'spectral_skewness', 'spectral_kurtosis',
  'band_power_low', 'band_power_mid', 'band_power_high'
]

const sensorColumns = computed(() =>
  pipelineStore.dataSession?.metadata?.sensor_columns?.length || 3
)

const filteredTSFreshFeatures = computed(() =>
  tsfreshFeatures.filter(f => f.toLowerCase().includes(searchQuery.value.toLowerCase()))
)

const filteredDSPFeatures = computed(() =>
  dspFeatures.filter(f => f.toLowerCase().includes(searchQuery.value.toLowerCase()))
)

// Available features for visualization (from extracted data)
const extractedFeatureNames = computed(() => {
  if (!featurePreview.value?.columns) return []
  return featurePreview.value.columns.filter((c: string) => c !== 'label')
})

// Feature importance chart data
const importanceChartData = computed(() => {
  if (!selectionResult.value?.importance_scores) {
    return { labels: [], datasets: [] }
  }

  const scores = selectionResult.value.importance_scores
  const sortedFeatures = Object.entries(scores)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .slice(0, 15)

  return {
    labels: sortedFeatures.map(([name]) => name.split('_').slice(0, -1).join('_') || name),
    datasets: [{
      label: 'Importance Score',
      data: sortedFeatures.map(([, score]) => score),
      backgroundColor: sortedFeatures.map(([name]) => {
        const type = getFeatureType(name)
        return type === 'TSFresh' ? 'rgba(99, 102, 241, 0.7)' : 'rgba(34, 211, 238, 0.7)'
      }),
      borderColor: sortedFeatures.map(([name]) => {
        const type = getFeatureType(name)
        return type === 'TSFresh' ? 'rgba(99, 102, 241, 1)' : 'rgba(34, 211, 238, 1)'
      }),
      borderWidth: 1
    }]
  }
})

const importanceChartOptions = {
  indexAxis: 'y' as const,
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    title: { display: true, text: 'Feature Importance Scores' }
  },
  scales: {
    x: { beginAtZero: true, max: 1 }
  }
}

// Distribution chart data
const distributionChartData = computed(() => {
  if (!featureDistribution.value) {
    return { labels: [], datasets: [] }
  }

  const dist = featureDistribution.value
  return {
    labels: dist.bin_edges.slice(0, -1).map((edge: number) => `${edge.toFixed(2)}`),
    datasets: [{
      label: selectedFeatureForViz.value,
      data: dist.counts,
      backgroundColor: 'rgba(99, 102, 241, 0.7)',
      borderColor: 'rgba(99, 102, 241, 1)',
      borderWidth: 1
    }]
  }
})

const distributionChartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    title: { display: true, text: `Distribution: ${selectedFeatureForViz.value}` }
  },
  scales: {
    x: { title: { display: true, text: 'Value' }, ticks: { maxRotation: 45 } },
    y: { title: { display: true, text: 'Count' }, beginAtZero: true }
  }
}))

function toggleFeature(feature: string) {
  const index = selectedFeatures.value.indexOf(feature)
  if (index >= 0) {
    selectedFeatures.value.splice(index, 1)
  } else {
    selectedFeatures.value.push(feature)
  }
}

function selectAll() {
  selectedFeatures.value = [...tsfreshFeatures, ...dspFeatures]
}

function clearSelection() {
  selectedFeatures.value = []
}

function selectTSFresh() {
  selectedFeatures.value = [...tsfreshFeatures]
}

function selectDSP() {
  selectedFeatures.value = [...dspFeatures]
}

async function getRecommendations() {
  if (!pipelineStore.windowedSession) {
    notificationStore.showError('No windowed data available')
    return
  }

  try {
    loadingRecommendations.value = true
    const response = await api.post('/api/features/recommend', {
      session_id: pipelineStore.windowedSession.session_id,
      mode: pipelineStore.mode,
      use_llm: true
    })

    recommendations.value = response.data
    const source = response.data.llm_used ? 'LLM' : 'rule-based'
    notificationStore.showSuccess(`Recommendations received (${source})`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to get recommendations')
  } finally {
    loadingRecommendations.value = false
  }
}

function applyRecommendations() {
  if (recommendations.value?.recommended_features) {
    selectedFeatures.value = recommendations.value.recommended_features.filter(
      (f: string) => tsfreshFeatures.includes(f) || dspFeatures.includes(f)
    )
    notificationStore.showSuccess('Recommendations applied')
  }
}

async function extractFeatures() {
  pipelineStore.selectedFeatures = selectedFeatures.value
  extracting.value = true

  const result = await pipelineStore.extractFeatures(selectedFeatures.value)

  if (result.success) {
    extractionResult.value = result.data
    // Store extraction result in pipeline store for persistence
    pipelineStore.setExtractionResult({
      session_id: result.data.session_id,
      num_features: result.data.num_features,
      num_windows: result.data.num_windows,
      feature_set: result.data.feature_set
    })
    notificationStore.showSuccess('Features extracted successfully')
    await fetchFeaturePreview()
    activeTab.value = 'select'
  } else {
    notificationStore.showError(result.error || 'Failed to extract features')
  }

  extracting.value = false
}

async function extractTSFreshFeatures() {
  if (!pipelineStore.windowedSession) {
    notificationStore.showError('No windowed data available')
    return
  }

  extracting.value = true

  try {
    const response = await api.post('/api/features/extract-tsfresh', {
      session_id: pipelineStore.windowedSession.session_id,
      feature_set: tsfreshFeatureSet.value
    })

    extractionResult.value = response.data
    pipelineStore.featureSession = response.data
    // Store extraction result in pipeline store for persistence
    pipelineStore.setExtractionResult({
      session_id: response.data.session_id,
      num_features: response.data.num_features,
      num_windows: response.data.num_windows,
      feature_set: response.data.feature_set
    })
    notificationStore.showSuccess(`TSFresh extraction complete: ${response.data.num_features} features`)
    await fetchFeaturePreview()
    activeTab.value = 'select'
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'TSFresh extraction failed')
  } finally {
    extracting.value = false
  }
}

async function runFeatureSelection() {
  if (!extractionResult.value?.session_id) return

  try {
    selectingFeatures.value = true
    const response = await api.post('/api/features/select', {
      session_id: extractionResult.value.session_id,
      method: selectionMethod.value,
      n_features: targetFeatures.value
    })

    selectionResult.value = response.data
    // Store selection result in pipeline store
    pipelineStore.setSelectionResult({
      session_id: response.data.session_id,
      selected_features: response.data.selected_features,
      original_count: response.data.original_count,
      final_count: response.data.final_count,
      method: selectionMethod.value
    })
    notificationStore.showSuccess(`Selected ${response.data.final_count} features`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Selection failed')
  } finally {
    selectingFeatures.value = false
  }
}

async function runFRESHSelection() {
  if (!extractionResult.value?.session_id) return

  try {
    selectingFeatures.value = true
    const response = await api.post('/api/features/select-fresh', {
      session_id: extractionResult.value.session_id,
      fdr_level: fdrLevel.value
    })

    selectionResult.value = {
      ...response.data,
      selection_log: [
        `Applied FRESH algorithm with FDR level: ${(fdrLevel.value * 100).toFixed(0)}%`,
        `Performed ${response.data.num_hypothesis_tests || 'N/A'} hypothesis tests`,
        `Selected ${response.data.final_count} statistically significant features`
      ]
    }
    // Store selection result in pipeline store
    pipelineStore.setSelectionResult({
      session_id: response.data.session_id,
      selected_features: response.data.selected_features,
      original_count: response.data.original_count,
      final_count: response.data.final_count,
      method: 'fresh',
      fdr_level: fdrLevel.value
    })
    notificationStore.showSuccess(`FRESH selected ${response.data.final_count} significant features`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'FRESH selection failed')
  } finally {
    selectingFeatures.value = false
  }
}

async function runFRESHCombinedSelection() {
  if (!extractionResult.value?.session_id) return

  try {
    selectingFeatures.value = true
    const response = await api.post('/api/features/select-fresh-combined', {
      session_id: extractionResult.value.session_id,
      fdr_level: fdrLevel.value,
      n_features: targetFeatures.value
    })

    selectionResult.value = {
      ...response.data,
      selection_log: response.data.selection_log || [
        `Step 1: FRESH with FDR level ${(fdrLevel.value * 100).toFixed(0)}%`,
        `Step 2: Reduced to ${targetFeatures.value} features using mutual information`,
        `Final: ${response.data.final_count} features selected`
      ]
    }
    // Store selection result in pipeline store
    pipelineStore.setSelectionResult({
      session_id: response.data.session_id,
      selected_features: response.data.selected_features,
      original_count: response.data.original_count,
      final_count: response.data.final_count,
      method: 'fresh_combined',
      fdr_level: fdrLevel.value,
      after_fresh_count: response.data.after_fresh_count
    })
    notificationStore.showSuccess(`FRESH + Target: ${response.data.final_count} features selected`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Chained selection failed')
  } finally {
    selectingFeatures.value = false
  }
}

async function runLLMSelection() {
  if (!extractionResult.value?.session_id) return

  try {
    selectingFeatures.value = true
    const response = await api.post('/api/features/llm-select', {
      session_id: extractionResult.value.session_id,
      mode: pipelineStore.mode,
      n_features: targetFeatures.value
    })

    selectionResult.value = response.data
    // Store selection result in pipeline store
    pipelineStore.setSelectionResult({
      session_id: response.data.session_id,
      selected_features: response.data.selected_features,
      original_count: response.data.original_count,
      final_count: response.data.final_count,
      method: 'llm'
    })
    const source = response.data.llm_used ? 'LLM-powered' : 'statistical'
    notificationStore.showSuccess(`${source} selection: ${response.data.final_count} features`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'LLM selection failed')
  } finally {
    selectingFeatures.value = false
  }
}

async function applyFeatureSelection() {
  if (!selectionResult.value?.selected_features) return

  try {
    applyingSelection.value = true
    const response = await api.post('/api/features/apply-selection', {
      session_id: extractionResult.value.session_id,
      selected_features: selectionResult.value.selected_features
    })

    appliedSelection.value = response.data
    pipelineStore.featureSession = response.data
    // Mark selection as applied in pipeline store
    pipelineStore.markSelectionApplied()
    notificationStore.showSuccess('Feature selection applied')
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to apply selection')
  } finally {
    applyingSelection.value = false
  }
}

async function fetchFeaturePreview() {
  if (!extractionResult.value?.session_id) return

  try {
    const response = await api.post('/api/features/preview', {
      session_id: extractionResult.value.session_id,
      num_rows: 100
    })
    featurePreview.value = response.data

    if (extractedFeatureNames.value.length > 0 && !selectedFeatureForViz.value) {
      selectedFeatureForViz.value = extractedFeatureNames.value[0]
    }
  } catch (e: any) {
    notificationStore.showError('Failed to load feature preview')
  }
}

async function fetchFeatureDistribution(featureName: string) {
  if (!extractionResult.value?.session_id || !featureName) return

  try {
    loadingDistribution.value = true
    const response = await api.post('/api/features/distribution', {
      session_id: extractionResult.value.session_id,
      feature_name: featureName,
      bins: 20
    })
    featureDistribution.value = response.data
  } catch (e: any) {
    notificationStore.showError('Failed to load feature distribution')
  } finally {
    loadingDistribution.value = false
  }
}

watch(selectedFeatureForViz, (newFeature) => {
  if (newFeature) fetchFeatureDistribution(newFeature)
})

const labelColors = ['primary', 'secondary', 'success', 'warning', 'error', 'info']
function getLabelColor(label: string): string {
  const index = Object.keys(featurePreview.value?.label_counts || {}).indexOf(label)
  return labelColors[index % labelColors.length]
}

function getFeatureType(featureName: string): string {
  const baseName = featureName.split('_').slice(0, -1).join('_')
  if (tsfreshFeatures.includes(baseName)) return 'TSFresh'
  if (dspFeatures.includes(baseName)) return 'DSP'
  return 'Other'
}

function getFeatureTypeColor(featureName: string): string {
  const type = getFeatureType(featureName)
  if (type === 'TSFresh') return 'info'
  if (type === 'DSP') return 'secondary'
  return 'default'
}

async function fetchLLMStatus() {
  try {
    const response = await api.get('/api/features/llm-status')
    llmStatus.value = response.data
  } catch {
    llmStatus.value = { available: false, error: 'Failed to check LLM status' }
  }
}

onMounted(async () => {
  if (pipelineStore.selectedFeatures.length > 0) {
    selectedFeatures.value = [...pipelineStore.selectedFeatures]
  } else {
    selectedFeatures.value = ['mean', 'std', 'rms', 'kurtosis', 'spectral_entropy']
  }

  // Restore state from pipeline store when coming back
  const storedState = pipelineStore.featureSelectionState
  if (storedState.extractionResult) {
    extractionResult.value = storedState.extractionResult
    await fetchFeaturePreview()
  }
  if (storedState.selectionResult) {
    selectionResult.value = storedState.selectionResult
  }
  if (storedState.selectionApplied && pipelineStore.featureSession) {
    appliedSelection.value = pipelineStore.featureSession
  }

  fetchLLMStatus()
})
</script>

<style scoped lang="scss">
.feature-list {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;

  .v-list-item {
    &.selected {
      background: rgba(99, 102, 241, 0.1);
    }

    &:hover {
      background: rgba(var(--v-theme-surface-variant), 0.5);
    }
  }
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  border-bottom: 1px solid rgba(var(--v-border-color), 0.1);

  &:last-child {
    border-bottom: none;
  }
}

.gap-2 {
  gap: 8px;
}

// Feature count card styles
.feature-count-card {
  border-left: 4px solid rgb(var(--v-theme-success));
}

.feature-counts {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  background: rgba(var(--v-theme-surface-variant), 0.3);
  border-radius: 8px;

  .count-item {
    text-align: center;
    min-width: 80px;

    .count-label {
      font-size: 0.75rem;
      color: rgba(var(--v-theme-on-surface), 0.6);
      margin-bottom: 4px;
    }

    .count-value {
      font-size: 1.5rem;
      font-weight: 700;
    }
  }
}
</style>
