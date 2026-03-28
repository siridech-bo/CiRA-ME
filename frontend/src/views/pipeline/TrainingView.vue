<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="training" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Model Training</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Train a {{ pipelineStore.mode === 'anomaly' ? 'anomaly detection' : pipelineStore.mode === 'regression' ? 'regression' : 'classification' }} model
    </p>

    <!-- Feature Status Warning -->
    <v-alert
      v-if="trainingApproach === 'ml' && pipelineStore.hasUnappliedSelection"
      type="warning"
      variant="tonal"
      class="mb-4"
      prominent
    >
      <div class="d-flex align-center">
        <v-icon class="mr-3">mdi-alert</v-icon>
        <div>
          <div class="font-weight-bold">Feature Selection Not Applied</div>
          <div class="text-body-2">
            You ran feature selection ({{ pipelineStore.featureSelectionState.selectionResult?.final_count }} features selected)
            but didn't apply it. Training will use all
            <strong>{{ pipelineStore.featureSelectionState.extractionResult?.num_features }}</strong> extracted features.
          </div>
        </div>
        <v-spacer />
        <v-btn
          color="warning"
          variant="flat"
          @click="goBackToApplySelection"
        >
          Go Back to Apply
        </v-btn>
      </div>
    </v-alert>

    <!-- Feature Summary Card (for ML approach) -->
    <v-card v-if="(trainingApproach === 'ml' || trainingApproach === 'ti' || trainingApproach === 'custom') && pipelineStore.featureSession" class="pa-4 mb-6">
      <div class="d-flex align-center mb-3">
        <v-icon color="primary" class="mr-2">mdi-feature-search</v-icon>
        <h3 class="text-subtitle-1 font-weight-bold">Features for Training</h3>
        <v-spacer />
        <v-chip
          :color="pipelineStore.featureSelectionState.selectionApplied ? 'success' : 'info'"
          size="small"
          variant="flat"
        >
          {{ pipelineStore.activeFeatureCount }} features
        </v-chip>
      </div>

      <v-alert
        v-if="pipelineStore.featureSelectionState.selectionApplied"
        type="success"
        variant="tonal"
        density="compact"
        class="mb-3"
      >
        Using <strong>{{ pipelineStore.activeFeatureCount }}</strong> selected features
        (reduced from {{ pipelineStore.featureSelectionState.extractionResult?.num_features }})
      </v-alert>

      <v-alert
        v-else
        type="info"
        variant="tonal"
        density="compact"
        class="mb-3"
      >
        Using all <strong>{{ pipelineStore.activeFeatureCount }}</strong> extracted features
      </v-alert>

      <!-- Expandable Feature List -->
      <v-expansion-panels variant="accordion">
        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon size="small" class="mr-2">mdi-format-list-bulleted</v-icon>
            View Feature List ({{ pipelineStore.selectedFeatureNames.length }})
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <div class="feature-list-container">
              <v-chip
                v-for="(feat, idx) in pipelineStore.selectedFeatureNames"
                :key="idx"
                size="x-small"
                :color="getFeatureTypeColor(feat)"
                variant="tonal"
                class="ma-1"
              >
                {{ feat }}
              </v-chip>
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
    </v-card>

    <!-- ML vs DL Toggle -->
    <v-card class="pa-4 mb-6">
      <div class="d-flex align-center">
        <h3 class="text-subtitle-1 font-weight-bold mr-4">Training Approach</h3>
        <v-btn-toggle
          v-model="trainingApproach"
          mandatory
          color="primary"
          rounded="lg"
        >
          <v-btn value="ml" size="small">
            <v-icon start>mdi-chart-scatter-plot</v-icon>
            Traditional ML
          </v-btn>
          <v-btn value="dl" size="small">
            <v-icon start>mdi-brain</v-icon>
            Deep Learning (TimesNet)
          </v-btn>
          <v-btn value="custom" size="small">
            <v-icon start>mdi-code-braces</v-icon>
            Custom Model
          </v-btn>
          <v-btn value="ti" size="small">
            <v-icon start>mdi-chip</v-icon>
            TI TinyML
          </v-btn>
        </v-btn-toggle>
      </div>

      <v-alert
        :type="trainingApproach === 'ml' ? 'info' : trainingApproach === 'custom' ? 'success' : 'warning'"
        variant="tonal"
        density="compact"
        class="mt-4"
      >
        <template v-if="trainingApproach === 'ml'">
          <strong>Traditional ML:</strong> Uses extracted features (TSFresh + DSP) with PyOD/Scikit-learn algorithms.
          Best for interpretability and specific sensor metrics.
        </template>
        <template v-else-if="trainingApproach === 'custom'">
          <strong>Custom Model:</strong> Write your own model in Python using any library (scikit-learn, PyTorch, XGBoost, etc.).
          Your code runs in an isolated subprocess with full access to the pipeline's extracted features.
        </template>
        <template v-else-if="trainingApproach === 'ti'">
          <strong>TI TinyML:</strong> Train quantized models from TI's model zoo for deployment on TMS320 MCUs.
          Train quantized neural networks and traditional ML models for TMS320 MCUs using TI's tinyml-modelmaker.
        </template>
        <template v-else>
          <strong>Deep Learning (TimesNet):</strong> End-to-end learning directly from windowed data.
          Best for complex, high-dimensional patterns. No manual feature extraction needed.
        </template>
      </v-alert>
    </v-card>

    <v-row>
      <!-- Traditional ML Section -->
      <template v-if="trainingApproach === 'ml'">
        <!-- Algorithm Selection -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <div class="d-flex align-center mb-4">
              <h3 class="text-subtitle-1 font-weight-bold">Algorithms</h3>
              <v-spacer />
              <v-chip
                :color="pipelineStore.mode === 'anomaly' ? 'error' : pipelineStore.mode === 'regression' ? 'purple' : 'success'"
                size="small"
                variant="flat"
              >
                {{ pipelineStore.mode === 'anomaly' ? 'Anomaly Detection' : pipelineStore.mode === 'regression' ? 'Regression' : 'Classification' }}
              </v-chip>
            </div>

            <!-- Selection Actions -->
            <div class="d-flex align-center mb-3 gap-2">
              <v-btn size="small" variant="tonal" @click="selectAllAlgorithms">
                Select All
              </v-btn>
              <v-btn size="small" variant="tonal" @click="clearAlgorithmSelection">
                Clear
              </v-btn>
              <v-spacer />
              <v-chip size="small" color="primary" variant="flat">
                {{ selectedAlgorithms.length }} selected
              </v-chip>
            </div>

            <!-- Algorithm Checkboxes (mode-dependent) -->
            <div class="algorithm-list">
              <v-checkbox
                v-for="algo in currentAlgorithmList"
                :key="algo.id"
                v-model="selectedAlgorithms"
                :value="algo.id"
                density="compact"
                hide-details
                class="algorithm-checkbox"
              >
                <template #label>
                  <div class="d-flex align-center flex-grow-1">
                    <div class="flex-grow-1">
                      <div class="font-weight-medium">{{ algo.name }}</div>
                      <div class="text-caption text-medium-emphasis">{{ algo.description }}</div>
                    </div>
                    <v-chip
                      v-if="algo.recommended"
                      size="x-small"
                      color="warning"
                      variant="flat"
                      class="ml-2"
                    >
                      Recommended
                    </v-chip>
                    <v-chip
                      v-if="algo.noOnnx"
                      size="x-small"
                      color="grey"
                      variant="tonal"
                      class="ml-1"
                    >
                      No ONNX
                    </v-chip>
                  </div>
                </template>
              </v-checkbox>
            </div>
          </v-card>
        </v-col>

        <!-- ML Hyperparameters -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <h3 class="text-subtitle-1 font-weight-bold mb-4">Hyperparameters</h3>

            <!-- Anomaly hyperparameters -->
            <template v-if="pipelineStore.mode === 'anomaly'">
              <v-text-field
                v-model.number="mlHyperparameters.n_estimators"
                label="Number of Estimators"
                type="number"
                :min="10"
                :max="500"
                hint="More estimators = better accuracy but slower"
              />

              <div class="mb-4">
                <div class="d-flex justify-space-between mb-2">
                  <span class="text-body-2">Contamination</span>
                  <span class="font-weight-medium">{{ mlHyperparameters.contamination }}</span>
                </div>
                <v-slider
                  v-model="mlHyperparameters.contamination"
                  :min="0.01"
                  :max="0.5"
                  :step="0.01"
                  color="error"
                  hide-details
                />
              </div>
            </template>

            <!-- Regression hyperparameters (context-aware) -->
            <template v-else-if="pipelineStore.mode === 'regression'">
              <!-- n_estimators: only for ensemble models -->
              <v-text-field
                v-if="regNeedsEstimators"
                v-model.number="mlHyperparameters.n_estimators"
                label="Number of Estimators"
                type="number"
                :min="10"
                :max="500"
                hint="Number of trees (RF, XGBoost, LightGBM)"
              />

              <!-- max_depth: for tree-based models -->
              <v-text-field
                v-if="regNeedsMaxDepth"
                v-model.number="mlHyperparameters.max_depth"
                label="Max Depth"
                type="number"
                :min="1"
                :max="50"
                hint="Maximum tree depth"
                clearable
              />

              <!-- n_neighbors: KNN only -->
              <v-text-field
                v-if="regNeedsNeighbors"
                v-model.number="mlHyperparameters.n_neighbors"
                label="Number of Neighbors (K)"
                type="number"
                :min="1"
                :max="50"
                hint="Number of nearest neighbors to consider"
              />

              <!-- SVR kernel -->
              <v-select
                v-if="regNeedsSvrParams"
                v-model="mlHyperparameters.svr_kernel"
                label="Kernel"
                :items="['rbf', 'linear', 'poly', 'sigmoid']"
                hint="SVR kernel function"
              />

              <!-- SVR C -->
              <v-text-field
                v-if="regNeedsSvrParams"
                v-model.number="mlHyperparameters.svr_C"
                label="Regularization (C)"
                type="number"
                :min="0.01"
                :max="1000"
                :step="0.1"
                hint="Higher C = less regularization"
              />

              <!-- SVR epsilon -->
              <v-text-field
                v-if="regNeedsSvrParams"
                v-model.number="mlHyperparameters.svr_epsilon"
                label="Epsilon"
                type="number"
                :min="0.001"
                :max="1"
                :step="0.01"
                hint="Width of the no-penalty tube"
              />

              <div class="mb-4">
                <div class="d-flex justify-space-between mb-2">
                  <span class="text-body-2">Test Split</span>
                  <span class="font-weight-medium">{{ (mlHyperparameters.test_size * 100).toFixed(0) }}%</span>
                </div>
                <v-slider
                  v-model="mlHyperparameters.test_size"
                  :min="0.1"
                  :max="0.4"
                  :step="0.05"
                  color="info"
                  hide-details
                />
              </div>
            </template>

            <!-- Classification hyperparameters -->
            <template v-else>
              <v-text-field
                v-model.number="mlHyperparameters.n_estimators"
                label="Number of Estimators"
                type="number"
                :min="10"
                :max="500"
              />

              <v-text-field
                v-model.number="mlHyperparameters.max_depth"
                label="Max Depth"
                type="number"
                :min="1"
                :max="50"
              />

              <div class="mb-4">
                <div class="d-flex justify-space-between mb-2">
                  <span class="text-body-2">Test Split</span>
                  <span class="font-weight-medium">{{ (mlHyperparameters.test_size * 100).toFixed(0) }}%</span>
                </div>
                <v-slider
                  v-model="mlHyperparameters.test_size"
                  :min="0.1"
                  :max="0.4"
                  :step="0.05"
                  color="info"
                  hide-details
                />
              </div>
            </template>
          </v-card>
        </v-col>
      </template>

      <!-- TI TinyML Section -->
      <template v-else-if="trainingApproach === 'ti'">
        <!-- Device + Model Selection -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <!-- TI Service Status -->
            <v-alert
              v-if="!tiServiceAvailable"
              type="warning"
              variant="tonal"
              density="compact"
              class="mb-3"
            >
              TI ModelMaker service is not running. Start it with:
              <code>docker compose up -d ti-modelmaker</code>
            </v-alert>
            <div v-if="tiServiceAvailable && tiComputeInfo" class="mb-3">
              <v-chip
                size="small"
                :color="tiComputeInfo.startsWith('GPU') ? 'success' : 'info'"
                variant="tonal"
              >
                <v-icon start size="small">{{ tiComputeInfo.startsWith('GPU') ? 'mdi-chip' : 'mdi-cpu-64-bit' }}</v-icon>
                TI Training: {{ tiComputeInfo }}
              </v-chip>
            </div>

            <!-- Real-time Training Progress -->
            <v-card
              v-if="tiProgress"
              variant="outlined"
              class="pa-3 mb-3"
              color="primary"
            >
              <div class="d-flex align-center mb-2">
                <v-progress-circular indeterminate size="20" width="2" color="primary" class="mr-2" />
                <span class="font-weight-medium">{{ tiProgress.phase }}</span>
                <v-spacer />
                <span class="text-caption">
                  {{ tiProgress.epoch + 1 }} / {{ tiProgress.total }}
                </span>
              </div>
              <v-progress-linear
                :model-value="((tiProgress.epoch + 1) / tiProgress.total) * 100"
                color="primary"
                height="6"
                rounded
                class="mb-2"
              />
              <div class="d-flex text-caption text-medium-emphasis" style="gap: 16px;">
                <span v-if="tiProgress.r2 != null">R²: <strong>{{ tiProgress.r2.toFixed(4) }}</strong></span>
                <span v-if="tiProgress.mse != null">RMSE: <strong>{{ tiProgress.mse.toFixed(4) }}</strong></span>
              </div>
            </v-card>

            <h3 class="text-subtitle-1 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-developer-board</v-icon>
              Target Device
            </h3>
            <v-select
              v-model="tiSelectedDevice"
              :items="tiDeviceList"
              item-title="label"
              item-value="id"
              label="TI MCU Device"
              variant="outlined"
              density="comfortable"
              hide-details
              class="mb-4"
              @update:model-value="fetchTiModels"
            >
              <template #item="{ item, props: itemProps }">
                <v-list-item v-bind="itemProps">
                  <v-list-item-subtitle>
                    {{ item.raw.family }} | {{ item.raw.flash_kb }}KB Flash
                    <v-chip v-if="item.raw.npu" size="x-small" color="warning" variant="tonal" class="ml-1">
                      <v-icon start size="10">mdi-alert</v-icon>NPU
                    </v-chip>
                  </v-list-item-subtitle>
                </v-list-item>
              </template>
            </v-select>

            <!-- Model Zoo with checkboxes -->
            <div class="d-flex align-center mb-3">
              <h3 class="text-subtitle-1 font-weight-bold">Models</h3>
              <v-spacer />
              <v-btn-toggle
                v-model="tiModelSource"
                mandatory
                density="compact"
                rounded="lg"
                color="primary"
                class="mr-2"
                @update:model-value="fetchTiModels"
              >
                <v-btn value="all" size="x-small">All</v-btn>
                <v-btn value="ti_zoo" size="x-small">TI NN</v-btn>
                <v-btn value="traditional_ml" size="x-small">Trad. ML</v-btn>
              </v-btn-toggle>
            </div>

            <div class="d-flex align-center mb-2">
              <v-btn size="small" variant="tonal" @click="tiSelectedModels = []">
                Clear
              </v-btn>
              <v-spacer />
              <v-chip size="small" color="primary" variant="flat">
                {{ tiSelectedModels.length }} selected
              </v-chip>
            </div>

            <div v-if="Object.keys(tiModelsFiltered).length > 0" class="algorithm-list" style="max-height: 350px; overflow-y: auto;">
              <v-checkbox
                v-for="(model, key) in tiModelsFiltered"
                :key="key"
                v-model="tiSelectedModels"
                :value="key"
                density="compact"
                hide-details
                class="algorithm-checkbox"
              >
                <template #label>
                  <div class="d-flex align-center flex-grow-1">
                    <div class="flex-grow-1">
                      <div class="font-weight-medium">{{ model.name }}</div>
                      <div class="text-caption text-medium-emphasis">
                        {{ model.architecture }}
                        <template v-if="model.params > 0"> | {{ model.params.toLocaleString() }} params</template>
                        <template v-if="model.estimated_flash_kb"> | ~{{ model.estimated_flash_kb }}KB</template>
                        <template v-if="model.min_epochs"> | min {{ model.min_epochs }} epochs</template>
                      </div>
                    </div>
                    <v-chip v-if="model.source === 'traditional_ml'" size="x-small" color="orange" variant="tonal" class="ml-1"
                      title="Uses CiRA ME's extracted features → emlearn C export">CiRA Features</v-chip>
                    <v-chip v-else-if="model.source === 'ti_zoo'" size="x-small" color="info" variant="tonal" class="ml-1"
                      title="Uses TI's own pipeline with raw windowed data">TI Pipeline</v-chip>
                  </div>
                </template>
              </v-checkbox>
            </div>
            <div v-else-if="tiSelectedDevice" class="text-body-2 text-medium-emphasis pa-4 text-center">
              No compatible models for this device
            </div>
            <div v-else class="text-body-2 text-medium-emphasis pa-4 text-center">
              Select a target device first
            </div>
          </v-card>
        </v-col>

        <!-- TI Hyperparameters -->
        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <h3 class="text-subtitle-1 font-weight-bold mb-4">Hyperparameters</h3>

            <v-text-field
              v-model.number="tiConfig.epochs"
              label="Epochs (for Neural Net models)"
              type="number"
              :min="10"
              :max="500"
              :hint="tiConfig.epochs < tiSuggestedEpochs
                ? `Suggested minimum: ${tiSuggestedEpochs} epochs for selected model(s)`
                : 'Ignored for Traditional ML models'"
              :persistent-hint="tiConfig.epochs < tiSuggestedEpochs"
              :color="tiConfig.epochs < tiSuggestedEpochs ? 'warning' : undefined"
            />

            <v-text-field
              v-model.number="tiConfig.max_depth"
              label="Max Depth"
              type="number"
              :min="1"
              :max="50"
              hint="For tree-based models"
              clearable
            />

            <v-select
              v-model="tiConfig.quantization"
              :items="[
                { title: '8-bit (default)', value: '8bit' },
                { title: '4-bit weights', value: '4bit' },
                { title: '2-bit weights', value: '2bit' },
              ]"
              label="Quantization (for Neural Net models)"
              variant="outlined"
              density="compact"
              class="mb-2"
            />

            <div class="mb-4">
              <div class="d-flex justify-space-between mb-2">
                <span class="text-body-2">Test Split</span>
                <span class="font-weight-medium">{{ (tiConfig.test_size * 100).toFixed(0) }}%</span>
              </div>
              <v-slider
                v-model="tiConfig.test_size"
                :min="0.1"
                :max="0.4"
                :step="0.05"
                color="info"
                hide-details
              />
            </div>

            <!-- Dataset info -->
            <v-alert
              v-if="pipelineStore.dataSession"
              type="success"
              variant="tonal"
              density="compact"
            >
              <v-icon size="small" class="mr-1">mdi-database</v-icon>
              Dataset: {{ pipelineStore.dataSession.metadata.file_path?.split(/[/\\]/).pop() }}
              ({{ pipelineStore.dataSession.metadata.total_rows?.toLocaleString() }} rows)
            </v-alert>

            <!-- Feature pipeline info -->
            <v-alert
              v-if="tiModelSource !== 'ti_zoo' && pipelineStore.featureSession"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-2"
            >
              <v-icon size="small" class="mr-1">mdi-auto-fix</v-icon>
              <strong>CiRA Features</strong> models use your extracted features ({{ pipelineStore.featureSession.num_features }} features,
              window={{ pipelineStore.windowingConfig.window_size }}, stride={{ pipelineStore.windowingConfig.stride }})
            </v-alert>
            <v-alert
              v-else-if="tiModelSource !== 'ti_zoo' && !pipelineStore.featureSession"
              type="warning"
              variant="tonal"
              density="compact"
              class="mt-2"
            >
              <v-icon size="small" class="mr-1">mdi-alert</v-icon>
              Traditional ML models need features. Go to
              <strong @click="$router.push({ name: 'pipeline-features' })" style="cursor:pointer; text-decoration:underline;">Features page</strong> first.
            </v-alert>

            <!-- TI NN pipeline info -->
            <v-alert
              v-if="tiModelSource !== 'traditional_ml'"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-2"
            >
              <v-icon size="small" class="mr-1">mdi-chip</v-icon>
              <strong>TI Pipeline</strong> models use raw signal with SimpleWindow (frame=32, stride=16, 50% overlap)
            </v-alert>
          </v-card>
        </v-col>
      </template>

      <!-- Custom Model Editor Section -->
      <template v-else-if="trainingApproach === 'custom'">
        <v-col cols="12">
          <v-card class="pa-4">
            <div class="d-flex align-center mb-2">
              <h3 class="text-subtitle-1 font-weight-bold">Custom Model Editor</h3>
              <v-spacer />
              <v-select
                v-model="selectedTemplate"
                :items="customTemplates"
                item-title="name"
                item-value="id"
                label="Template"
                density="compact"
                variant="outlined"
                style="max-width: 250px"
                hide-details
                class="mr-2"
                @update:model-value="loadTemplate"
              >
                <template #item="{ item, props: itemProps }">
                  <v-list-item v-bind="itemProps">
                    <v-list-item-subtitle>{{ item.raw.description }}</v-list-item-subtitle>
                  </v-list-item>
                </template>
              </v-select>
            </div>
            <!-- Save / Load user snippets -->
            <div class="d-flex align-center mb-3" style="gap: 6px;">
              <v-select
                v-model="selectedSnippet"
                :items="savedSnippets"
                item-title="name"
                item-value="name"
                label="Saved Snippets"
                density="compact"
                variant="outlined"
                hide-details
                clearable
                style="max-width: 250px; font-size: 12px;"
                @update:model-value="loadSnippet"
              />
              <v-btn
                size="small"
                variant="tonal"
                color="success"
                :disabled="!customModelCode.trim()"
                @click="showSaveSnippetDialog = true"
              >
                <v-icon start size="small">mdi-content-save</v-icon>
                Save
              </v-btn>
              <v-btn
                v-if="selectedSnippet"
                size="small"
                variant="tonal"
                color="error"
                @click="deleteSnippet"
              >
                <v-icon size="small">mdi-delete</v-icon>
              </v-btn>
            </div>

            <!-- Save snippet dialog -->
            <v-dialog v-model="showSaveSnippetDialog" max-width="400">
              <v-card>
                <v-card-title>Save Code Snippet</v-card-title>
                <v-card-text>
                  <v-text-field
                    v-model="snippetSaveName"
                    label="Snippet Name"
                    variant="outlined"
                    density="compact"
                    placeholder="e.g. My XGBoost v2"
                    autofocus
                    @keydown.enter="saveSnippet"
                  />
                </v-card-text>
                <v-card-actions>
                  <v-spacer />
                  <v-btn variant="text" @click="showSaveSnippetDialog = false">Cancel</v-btn>
                  <v-btn color="success" variant="flat" :disabled="!snippetSaveName.trim()" @click="saveSnippet">Save</v-btn>
                </v-card-actions>
              </v-card>
            </v-dialog>

            <CodeEditor
              v-model="customModelCode"
              height="450px"
            />

            <div class="d-flex align-center mt-4">
              <v-chip
                v-if="pipelineStore.featureSession"
                size="small"
                color="primary"
                variant="tonal"
                class="mr-2"
              >
                {{ pipelineStore.featureSession.num_features }} features
              </v-chip>
              <v-chip
                size="small"
                :color="pipelineStore.mode === 'anomaly' ? 'error' : pipelineStore.mode === 'regression' ? 'purple' : 'success'"
                variant="tonal"
                class="mr-2"
              >
                Task: {{ pipelineStore.mode }}
              </v-chip>
              <v-spacer />
              <v-btn
                color="primary"
                size="large"
                :loading="training"
                :disabled="!pipelineStore.featureSession || !customModelCode.trim()"
                @click="trainCustomModel"
              >
                <v-icon start>mdi-play</v-icon>
                Run Custom Model
              </v-btn>
            </div>

            <!-- Execution Logs -->
            <v-expand-transition>
              <div v-if="customModelLogs.length > 0" class="mt-4">
                <h4 class="text-subtitle-2 font-weight-bold mb-2">Execution Logs</h4>
                <v-card variant="outlined" class="pa-3" style="background: #1e1e1e; max-height: 200px; overflow-y: auto;">
                  <pre class="text-caption" style="color: #d4d4d4; white-space: pre-wrap; margin: 0;">{{ customModelLogs.join('\n') }}</pre>
                </v-card>
              </div>
            </v-expand-transition>
          </v-card>
        </v-col>
      </template>

      <!-- Deep Learning (TimesNet) Section -->
      <template v-else-if="trainingApproach === 'dl'">
        <!-- GPU Status Card -->
        <v-col cols="12">
          <v-card class="pa-4 mb-4">
            <div class="d-flex align-center justify-space-between">
              <div class="d-flex align-center">
                <v-icon
                  :color="gpuStatus.available ? 'success' : 'warning'"
                  class="mr-2"
                >
                  {{ gpuStatus.available ? 'mdi-chip' : 'mdi-cpu-64-bit' }}
                </v-icon>
                <div>
                  <h3 class="text-subtitle-1 font-weight-bold">
                    {{ gpuStatus.available ? 'GPU Available' : 'GPU Not Available' }}
                  </h3>
                  <div class="text-caption text-medium-emphasis">
                    <template v-if="gpuStatus.available">
                      {{ gpuStatus.device_name }} -
                      {{ gpuStatus.memory_free_gb }}GB free / {{ gpuStatus.memory_total_gb }}GB total
                    </template>
                    <template v-else-if="gpuStatus.error">
                      {{ gpuStatus.error }}
                    </template>
                    <template v-else>
                      {{ gpuStatus.info || 'CUDA not available' }}
                    </template>
                  </div>
                </div>
              </div>

              <div class="d-flex align-center">
                <span class="text-body-2 mr-3">Training Device:</span>
                <v-btn-toggle
                  v-model="selectedDevice"
                  mandatory
                  color="primary"
                  density="compact"
                >
                  <v-btn value="cpu" size="small">
                    <v-icon start size="small">mdi-cpu-64-bit</v-icon>
                    CPU
                  </v-btn>
                  <v-btn
                    value="cuda"
                    size="small"
                    :disabled="!gpuStatus.available"
                  >
                    <v-icon start size="small">mdi-chip</v-icon>
                    GPU
                  </v-btn>
                </v-btn-toggle>
              </div>
            </div>

            <v-alert
              v-if="gpuStatus.warning"
              type="warning"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              {{ gpuStatus.warning }}
            </v-alert>

            <v-alert
              v-if="gpuStatus.dll_error"
              type="warning"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              <strong>PyTorch DLL Conflict:</strong> Another GPU application is running.
              TimesNet deep learning is unavailable. You can still train using traditional ML algorithms
              which don't require PyTorch.
            </v-alert>

            <v-alert
              v-if="!gpuStatus.torch_available && !gpuStatus.dll_error"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              {{ gpuStatus.info || 'PyTorch not available. Using fallback ML methods.' }}
            </v-alert>

            <v-alert
              v-if="selectedDevice === 'cpu' && gpuStatus.available"
              type="info"
              variant="tonal"
              density="compact"
              class="mt-3"
            >
              Training on CPU will be slower but avoids GPU memory conflicts with other applications.
            </v-alert>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <div class="d-flex align-center mb-4">
              <v-icon color="secondary" class="mr-2">mdi-brain</v-icon>
              <h3 class="text-subtitle-1 font-weight-bold">TimesNet Configuration</h3>
            </div>

            <v-alert type="info" variant="tonal" density="compact" class="mb-4">
              TimesNet learns directly from raw windowed data using multi-periodic analysis.
              No feature extraction required!
            </v-alert>

            <!-- Model Architecture -->
            <v-text-field
              v-model.number="timesnetConfig.d_model"
              label="Model Dimension (d_model)"
              type="number"
              :min="32"
              :max="256"
              hint="Embedding dimension"
            />

            <v-text-field
              v-model.number="timesnetConfig.d_ff"
              label="Feed-Forward Dimension (d_ff)"
              type="number"
              :min="64"
              :max="512"
              hint="Hidden layer dimension"
            />

            <v-text-field
              v-model.number="timesnetConfig.e_layers"
              label="Encoder Layers"
              type="number"
              :min="1"
              :max="6"
              hint="Number of transformer layers"
            />

            <div class="mb-4">
              <div class="d-flex justify-space-between mb-2">
                <span class="text-body-2">Dropout</span>
                <span class="font-weight-medium">{{ timesnetConfig.dropout }}</span>
              </div>
              <v-slider
                v-model="timesnetConfig.dropout"
                :min="0"
                :max="0.5"
                :step="0.05"
                color="secondary"
                hide-details
              />
            </div>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card class="pa-4">
            <h3 class="text-subtitle-1 font-weight-bold mb-4">Period Configuration</h3>

            <p class="text-body-2 text-medium-emphasis mb-4">
              TimesNet analyzes data at multiple periods to capture intraperiod and interperiod variations.
            </p>

            <!-- Period Selection -->
            <v-chip-group
              v-model="selectedPeriods"
              multiple
              column
              selected-class="text-primary"
            >
              <v-chip
                v-for="period in availablePeriods"
                :key="period"
                :value="period"
                filter
                variant="outlined"
              >
                {{ period }}
              </v-chip>
            </v-chip-group>

            <v-text-field
              v-model.number="timesnetConfig.top_k"
              label="Top-K Periods"
              type="number"
              :min="1"
              :max="selectedPeriods.length || 5"
              hint="Number of dominant periods to use"
              class="mt-4"
            />

            <v-divider class="my-4" />

            <!-- Training Parameters -->
            <h4 class="text-subtitle-2 font-weight-bold mb-3">Training Parameters</h4>

            <v-text-field
              v-model.number="timesnetConfig.epochs"
              label="Epochs"
              type="number"
              :min="10"
              :max="500"
            />

            <v-text-field
              v-model.number="timesnetConfig.batch_size"
              label="Batch Size"
              type="number"
              :min="8"
              :max="128"
            />

            <v-text-field
              v-model.number="timesnetConfig.learning_rate"
              label="Learning Rate"
              type="number"
              :min="0.0001"
              :max="0.1"
              :step="0.0001"
            />
          </v-card>
        </v-col>
      </template>
    </v-row>

    <!-- Comparison Results -->
    <v-card v-if="comparisonResult" class="pa-4 mt-6">
      <div class="d-flex align-center mb-4">
        <v-icon color="primary" class="mr-2">mdi-compare</v-icon>
        <h3 class="text-subtitle-1 font-weight-bold">Algorithm Comparison</h3>
        <v-spacer />
        <v-chip color="success" size="small" variant="flat" class="mr-2">
          {{ comparisonResult.successful }} successful
        </v-chip>
        <v-chip v-if="comparisonResult.failed > 0" color="error" size="small" variant="flat">
          {{ comparisonResult.failed }} failed
        </v-chip>
      </div>

      <!-- Best Algorithm Banner -->
      <v-alert
        v-if="comparisonResult.best_algorithm"
        type="success"
        variant="tonal"
        class="mb-4"
      >
        <div class="d-flex align-center">
          <v-icon class="mr-2">mdi-trophy</v-icon>
          <div>
            <strong>Best Performer:</strong> {{ comparisonResult.best_algorithm.algorithm_name }}
            <span class="text-medium-emphasis ml-2">
              <template v-if="pipelineStore.mode === 'regression'">
                ({{ comparisonResult.best_algorithm.metric }}: {{ comparisonResult.best_algorithm.score.toFixed(4) }})
              </template>
              <template v-else>
                ({{ comparisonResult.best_algorithm.metric }}: {{ (comparisonResult.best_algorithm.score * 100).toFixed(1) }}%)
              </template>
            </span>
          </div>
        </div>
      </v-alert>

      <!-- Comparison Table -->
      <v-table density="comfortable" class="comparison-table">
        <thead>
          <tr>
            <th class="text-left">Algorithm</th>
            <th v-for="header in comparisonHeaders" :key="header.key" class="text-center">{{ header.label }}</th>
            <th class="text-center">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="result in comparisonResult.comparison?.rows || []"
            :key="result.algorithm"
            :class="{
              'best-row': comparisonResult.best_algorithm?.algorithm === result.algorithm,
              'selected-row': selectedComparisonAlgo === result.algorithm,
            }"
            style="cursor: pointer;"
            @click="selectComparisonModel(result.algorithm)"
          >
            <td class="font-weight-medium">
              {{ result.algorithm_name }}
              <v-icon
                v-if="comparisonResult.best_algorithm?.algorithm === result.algorithm"
                size="small"
                color="warning"
                class="ml-1"
                title="Best performer"
              >
                mdi-star
              </v-icon>
              <v-icon
                v-if="selectedComparisonAlgo === result.algorithm"
                size="small"
                color="primary"
                class="ml-1"
              >
                mdi-check-circle
              </v-icon>
            </td>
            <td v-for="header in comparisonHeaders" :key="'v-'+header.key" class="text-center">
              <template v-if="result.values[header.key] != null">
                {{ header.format(result.values[header.key]) }}
              </template>
              <template v-else>-</template>
            </td>
            <td class="text-center">
              <v-chip size="x-small" color="success" variant="flat">OK</v-chip>
            </td>
          </tr>
          <tr v-for="error in comparisonResult.errors || []" :key="'err-'+error.algorithm" class="error-row">
            <td class="font-weight-medium text-error">{{ error.algorithm_name }}</td>
            <td :colspan="comparisonHeaders.length" class="text-center text-caption text-error">
              <v-tooltip location="top" max-width="400">
                <template #activator="{ props }">
                  <span v-bind="props" class="error-message-truncate">
                    {{ truncateError(error.error) }}
                  </span>
                </template>
                <span>{{ error.error }}</span>
              </v-tooltip>
            </td>
            <td class="text-center">
              <v-chip size="x-small" color="error" variant="flat">Failed</v-chip>
            </td>
          </tr>
        </tbody>
      </v-table>

      <v-divider class="my-4" />

      <div class="d-flex align-center justify-space-between">
        <p class="text-caption text-medium-emphasis mb-0">
          <template v-if="selectedComparisonAlgo">
            <v-icon size="small" color="primary" class="mr-1">mdi-check-circle</v-icon>
            <strong>{{ getSelectedAlgoName() }}</strong> selected.
            Click any row to switch. Details shown below.
          </template>
          <template v-else>
            Click any algorithm row to select it for deployment and view detailed metrics below.
          </template>
        </p>
        <v-btn
          v-if="selectedComparisonAlgo || comparisonResult.best_algorithm"
          color="warning"
          variant="flat"
          size="small"
          @click="showSaveBenchmarkDialog = true"
        >
          <v-icon start size="small">mdi-content-save</v-icon>
          Save {{ getSelectedAlgoName() }}
        </v-btn>
      </div>
    </v-card>

    <!-- Save Benchmark Dialog -->
    <v-dialog v-model="showSaveBenchmarkDialog" max-width="450">
      <v-card class="pa-4">
        <h3 class="text-subtitle-1 font-weight-bold mb-4">Save Model as Benchmark</h3>
        <v-text-field
          v-model="benchmarkName"
          label="Benchmark Name"
          :placeholder="`${comparisonResult?.best_algorithm?.algorithm_name || 'Model'} - ${new Date().toLocaleDateString()}`"
          variant="outlined"
          density="compact"
          class="mb-4"
        />
        <div class="d-flex justify-end ga-2">
          <v-btn variant="text" @click="showSaveBenchmarkDialog = false">Cancel</v-btn>
          <v-btn color="warning" variant="flat" @click="saveBenchmark" :loading="savingBenchmark">Save</v-btn>
        </div>
      </v-card>
    </v-dialog>

    <!-- Model History -->
    <v-card class="pa-4 mt-6">
      <div class="d-flex align-center justify-space-between mb-4">
        <h3 class="text-subtitle-1 font-weight-bold">
          <v-icon start size="small">mdi-history</v-icon>
          Model History
        </h3>
        <v-btn variant="text" size="small" @click="loadSavedModels" :loading="loadingSavedModels">
          <v-icon start size="small">mdi-refresh</v-icon>
          Refresh
        </v-btn>
      </div>

      <v-alert v-if="savedModels.length === 0 && !loadingSavedModels" type="info" variant="tonal" density="compact">
        No saved benchmarks yet. Train a model and click "Save as Benchmark" to start tracking.
      </v-alert>

      <!-- Regression Models -->
      <div v-if="savedModelsRegression.length > 0" class="mb-4">
        <div class="text-caption font-weight-bold text-medium-emphasis mb-2">
          <v-icon size="x-small" class="mr-1">mdi-chart-timeline-variant</v-icon>
          Regression Models ({{ savedModelsRegression.length }})
        </div>
        <v-table density="compact" class="comparison-table">
          <thead>
            <tr>
              <th><v-checkbox-btn v-model="selectAllModels" @update:model-value="toggleSelectAllModels" density="compact" hide-details /></th>
              <th class="text-left">Name</th>
              <th class="text-center">Algorithm</th>
              <th class="text-center">R²</th>
              <th class="text-center">RMSE</th>
              <th class="text-center">MAE</th>
              <th class="text-center">Date</th>
              <th class="text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in savedModelsRegression" :key="m.id">
              <td><v-checkbox-btn v-model="selectedModelIds" :value="m.id" density="compact" hide-details /></td>
              <td class="font-weight-medium">{{ m.name }}</td>
              <td class="text-center text-caption">{{ m.algorithm }}</td>
              <td class="text-center">{{ m.metrics?.r2 != null ? m.metrics.r2.toFixed(4) : '-' }}</td>
              <td class="text-center">{{ m.metrics?.rmse != null ? m.metrics.rmse.toFixed(4) : '-' }}</td>
              <td class="text-center">{{ m.metrics?.mae != null ? m.metrics.mae.toFixed(4) : '-' }}</td>
              <td class="text-center text-caption">{{ m.created_at?.slice(0, 10) }}</td>
              <td class="text-center">
                <v-btn icon size="x-small" variant="text" color="error" @click="deleteSavedModel(m)" title="Delete">
                  <v-icon size="small">mdi-delete</v-icon>
                </v-btn>
              </td>
            </tr>
          </tbody>
        </v-table>
      </div>

      <!-- Classification & Anomaly Models -->
      <div v-if="savedModelsClassification.length > 0">
        <div class="text-caption font-weight-bold text-medium-emphasis mb-2">
          <v-icon size="x-small" class="mr-1">mdi-shape</v-icon>
          Classification & Anomaly Models ({{ savedModelsClassification.length }})
        </div>
        <v-table density="compact" class="comparison-table">
          <thead>
            <tr>
              <th><v-checkbox-btn v-model="selectAllModels" @update:model-value="toggleSelectAllModels" density="compact" hide-details /></th>
              <th class="text-left">Name</th>
              <th class="text-center">Algorithm</th>
              <th class="text-center">Accuracy</th>
              <th class="text-center">Precision</th>
              <th class="text-center">Recall</th>
              <th class="text-center">F1</th>
              <th class="text-center">Date</th>
              <th class="text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in savedModelsClassification" :key="m.id">
              <td><v-checkbox-btn v-model="selectedModelIds" :value="m.id" density="compact" hide-details /></td>
              <td class="font-weight-medium">{{ m.name }}</td>
              <td class="text-center text-caption">{{ m.algorithm }}</td>
              <td class="text-center">{{ m.metrics?.accuracy != null ? (m.metrics.accuracy * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ m.metrics?.precision != null ? (m.metrics.precision * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ m.metrics?.recall != null ? (m.metrics.recall * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ m.metrics?.f1 != null ? (m.metrics.f1 * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center text-caption">{{ m.created_at?.slice(0, 10) }}</td>
              <td class="text-center">
                <v-btn icon size="x-small" variant="text" color="info" @click="startEvaluation(m)" title="Test with new data">
                  <v-icon size="small">mdi-test-tube</v-icon>
                </v-btn>
                <v-btn icon size="x-small" variant="text" color="error" @click="deleteSavedModel(m)" title="Delete">
                  <v-icon size="small">mdi-delete</v-icon>
                </v-btn>
              </td>
            </tr>
          </tbody>
        </v-table>
      </div>

      <v-btn
        v-if="selectedModelIds.length === 2"
        color="primary"
        variant="flat"
        size="small"
        class="mt-3"
        @click="compareSelectedModels"
        :loading="comparing"
      >
        <v-icon start size="small">mdi-compare</v-icon>
        Compare Selected ({{ selectedModelIds.length }})
      </v-btn>

      <!-- Compare Result -->
      <v-card v-if="compareResult" variant="outlined" class="pa-3 mt-3">
        <h4 class="text-subtitle-2 font-weight-bold mb-2">Comparison: {{ compareResult.model_1.name }} vs {{ compareResult.model_2.name }}</h4>
        <v-table density="compact">
          <thead>
            <tr>
              <th class="text-left">Metric</th>
              <th class="text-center">{{ compareResult.model_1.name }}</th>
              <th class="text-center">{{ compareResult.model_2.name }}</th>
              <th class="text-center">Diff</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in compareResult.comparison" :key="c.metric">
              <td class="font-weight-medium">{{ c.metric }}</td>
              <td class="text-center">{{ c.model_1 != null ? (c.model_1 * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ c.model_2 != null ? (c.model_2 * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center" :class="c.diff > 0 ? 'text-success' : c.diff < 0 ? 'text-error' : ''">
                {{ c.diff != null ? (c.diff > 0 ? '+' : '') + (c.diff * 100).toFixed(1) + '%' : '-' }}
              </td>
            </tr>
          </tbody>
        </v-table>
      </v-card>
    </v-card>

    <!-- Evaluate on New Data Dialog -->
    <v-dialog v-model="showEvalDialog" max-width="600">
      <v-card class="pa-4">
        <h3 class="text-subtitle-1 font-weight-bold mb-4">
          <v-icon start size="small">mdi-test-tube</v-icon>
          Evaluate "{{ evalModel?.name }}" on New Data
        </h3>

        <v-alert type="info" variant="tonal" density="compact" class="mb-4">
          Load new data through the pipeline (Data Source → Windowing → Features), then click Evaluate.
          The saved model will predict on the new features and compare with original metrics.
        </v-alert>

        <div v-if="pipelineStore.featureSession" class="mb-4">
          <v-chip color="success" size="small" variant="flat" class="mr-2">
            Features Ready
          </v-chip>
          <span class="text-caption">
            {{ pipelineStore.featureSession.metadata?.num_features || '?' }} features,
            {{ pipelineStore.featureSession.metadata?.num_windows || '?' }} samples
          </span>
        </div>
        <v-alert v-else type="warning" variant="tonal" density="compact" class="mb-4">
          No feature session available. Go back to the pipeline to load & process new data first.
        </v-alert>

        <div class="d-flex justify-end ga-2">
          <v-btn variant="text" @click="showEvalDialog = false">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :disabled="!pipelineStore.featureSession"
            :loading="evaluating"
            @click="runEvaluation"
          >
            Evaluate
          </v-btn>
        </div>

        <!-- Evaluation Result -->
        <v-card v-if="evalResult" variant="outlined" class="pa-3 mt-4">
          <h4 class="text-subtitle-2 font-weight-bold mb-2">Results: Original vs New Data</h4>
          <v-table density="compact">
            <thead>
              <tr>
                <th class="text-left">Metric</th>
                <th class="text-center">Original</th>
                <th class="text-center">New Data</th>
                <th class="text-center">Diff</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="c in evalResult.comparison" :key="c.metric">
                <td class="font-weight-medium">{{ c.metric }}</td>
                <td class="text-center">{{ c.original != null ? (c.original * 100).toFixed(1) + '%' : '-' }}</td>
                <td class="text-center">{{ c.new_data != null ? (c.new_data * 100).toFixed(1) + '%' : '-' }}</td>
                <td class="text-center" :class="c.diff != null ? (c.diff >= 0 ? 'text-success' : 'text-error') : ''">
                  {{ c.diff != null ? (c.diff > 0 ? '+' : '') + (c.diff * 100).toFixed(1) + '%' : '-' }}
                </td>
              </tr>
            </tbody>
          </v-table>
          <div class="text-caption text-medium-emphasis mt-2">
            Evaluated on {{ evalResult.new_metrics?.test_samples || '?' }} samples
          </div>
        </v-card>
      </v-card>
    </v-dialog>

    <!-- Training Results (Single Algorithm or Best from Comparison) -->
    <v-card v-if="trainingResult" class="pa-4 mt-6">
      <div class="d-flex align-center justify-space-between mb-4">
        <h3 class="text-subtitle-1 font-weight-bold">
          {{ comparisonResult ? 'Best Model Details' : 'Training Complete' }}
        </h3>
        <v-btn
          v-if="!comparisonResult && trainingResult.training_session_id"
          color="warning"
          variant="flat"
          size="small"
          @click="showSaveBenchmarkDialog = true"
        >
          <v-icon start size="small">mdi-content-save</v-icon>
          Save as Benchmark
        </v-btn>
      </div>

      <v-alert type="success" variant="tonal" class="mb-4">
        Model trained successfully using
        <strong>{{ trainingResult.algorithm }}</strong>
        <template v-if="trainingApproach === 'dl'"> (Deep Learning)</template>
      </v-alert>

      <!-- Split method info -->
      <v-alert
        v-if="trainingResult.metrics.split_method || trainingResult.metrics.train_samples"
        type="info"
        variant="tonal"
        density="compact"
        class="mb-4"
      >
        <v-icon size="small" class="mr-1">mdi-chart-pie</v-icon>
        <template v-if="trainingResult.metrics.split_method === 'category'">
          Using dataset's built-in train/test split
        </template>
        <template v-else>
          Train/Test split
        </template>
        <span v-if="trainingResult.metrics.train_samples">
          ({{ trainingResult.metrics.train_samples }} train / {{ trainingResult.metrics.test_samples || 0 }} test)
        </span>
      </v-alert>

      <!-- Metrics info message (when labels are missing/incomplete) -->
      <v-alert
        v-if="trainingResult.metrics.metrics_info"
        type="warning"
        variant="tonal"
        density="compact"
        class="mb-4"
      >
        <v-icon size="small" class="mr-1">mdi-alert-circle</v-icon>
        {{ trainingResult.metrics.metrics_info }}
      </v-alert>

      <!-- Regression Metrics Row -->
      <v-row v-if="pipelineStore.mode === 'regression'" dense>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">R² Score</div>
            <div class="text-h5" :class="trainingResult.metrics.r2 >= 0.8 ? 'text-success' : trainingResult.metrics.r2 >= 0.5 ? 'text-info' : 'text-warning'">
              {{ (trainingResult.metrics.r2 || 0).toFixed(4) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">RMSE</div>
            <div class="text-h5 text-info">
              {{ (trainingResult.metrics.rmse || 0).toFixed(4) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">MAE</div>
            <div class="text-h6">
              {{ (trainingResult.metrics.mae || 0).toFixed(4) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">MSE</div>
            <div class="text-h6">
              {{ (trainingResult.metrics.mse || 0).toFixed(4) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2" v-if="trainingResult.metrics.mape !== undefined">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">MAPE</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.mape || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Train R²</div>
            <div class="text-h6 text-purple">
              {{ (trainingResult.metrics.train_r2 || 0).toFixed(4) }}
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- MCU Memory Budget (TI models only) -->
      <v-row
        v-if="trainingApproach === 'ti' && pipelineStore.mode === 'regression' && trainingResult.metrics.model_size_kb"
        class="mt-4"
      >
        <v-col cols="12">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-memory</v-icon>
              MCU Memory Budget
              <span v-if="tiSelectedDevice" class="text-caption text-medium-emphasis ml-2">
                ({{ tiDevices[tiSelectedDevice]?.name || tiSelectedDevice }})
              </span>
            </h4>
            <v-row dense>
              <v-col cols="6" md="3">
                <v-card variant="tonal" class="pa-3 text-center">
                  <div class="text-caption text-medium-emphasis">Model Size</div>
                  <div class="text-h6 text-info">
                    {{ trainingResult.metrics.model_size_kb?.toFixed(1) || '?' }} KB
                  </div>
                  <div class="text-caption text-medium-emphasis">ONNX (FP32)</div>
                </v-card>
              </v-col>
              <v-col cols="6" md="3">
                <v-card variant="tonal" class="pa-3 text-center">
                  <div class="text-caption text-medium-emphasis">Quantized Size</div>
                  <div class="text-h6 text-success">
                    {{ trainingResult.metrics.model_size_int8_kb?.toFixed(1) || '?' }} KB
                  </div>
                  <div class="text-caption text-medium-emphasis">INT8 (estimated)</div>
                </v-card>
              </v-col>
              <v-col cols="6" md="3">
                <v-card variant="tonal" class="pa-3 text-center">
                  <div class="text-caption text-medium-emphasis">Device Flash</div>
                  <div class="text-h6">
                    {{ tiDevices[tiSelectedDevice]?.flash_kb || '?' }} KB
                  </div>
                  <div class="text-caption text-medium-emphasis">Total available</div>
                </v-card>
              </v-col>
              <v-col cols="6" md="3">
                <v-card variant="tonal" class="pa-3 text-center">
                  <div class="text-caption text-medium-emphasis">Flash Remaining</div>
                  <div
                    class="text-h6"
                    :class="mcuFlashRemaining > 50 ? 'text-success' : mcuFlashRemaining > 20 ? 'text-warning' : 'text-error'"
                  >
                    {{ mcuFlashRemaining }}%
                  </div>
                  <div class="text-caption text-medium-emphasis">For application code</div>
                </v-card>
              </v-col>
            </v-row>
            <v-progress-linear
              :model-value="mcuFlashUsed"
              :color="mcuFlashRemaining > 50 ? 'success' : mcuFlashRemaining > 20 ? 'warning' : 'error'"
              height="20"
              rounded
              class="mt-3"
            >
              <template #default>
                <span class="text-caption font-weight-medium">
                  Model: {{ trainingResult.metrics.model_size_int8_kb?.toFixed(1) }} KB / {{ tiDevices[tiSelectedDevice]?.flash_kb || '?' }} KB Flash
                </span>
              </template>
            </v-progress-linear>
          </v-card>
        </v-col>
      </v-row>

      <!-- Regression Scatter Plot (Predicted vs Actual) -->
      <v-row v-if="pipelineStore.mode === 'regression' && trainingResult.metrics.scatter_data" class="mt-4">
        <v-col cols="12" md="6">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-1">
              <v-icon size="small" class="mr-1">mdi-chart-scatter-plot</v-icon>
              Predicted vs Actual
            </h4>
            <div class="text-caption text-medium-emphasis mb-2">
              {{ trainingResult.metrics.test_samples || trainingResult.metrics.scatter_data?.actual?.length || '?' }} test samples
              <span v-if="trainingApproach === 'ti'"> | {{ trainingResult.metrics.pipeline === 'cira_features' ? 'CiRA ME features' : 'TI windowed data' }}</span>
            </div>
            <svg viewBox="0 0 300 280" class="w-100" style="max-height: 280px;">
              <!-- Grid lines -->
              <line x1="50" y1="10" x2="50" y2="250" stroke="currentColor" stroke-opacity="0.2"/>
              <line x1="50" y1="250" x2="290" y2="250" stroke="currentColor" stroke-opacity="0.2"/>
              <!-- Perfect prediction line -->
              <line x1="50" y1="250" x2="290" y2="10" stroke="#888" stroke-dasharray="4" stroke-opacity="0.5"/>
              <!-- Scatter points -->
              <circle
                v-for="(actual, idx) in trainingResult.metrics.scatter_data.actual"
                :key="'scatter'+idx"
                :cx="50 + ((actual - scatterMin) / (scatterMax - scatterMin || 1)) * 240"
                :cy="250 - ((trainingResult.metrics.scatter_data.predicted[idx] - scatterMin) / (scatterMax - scatterMin || 1)) * 240"
                r="3"
                fill="#6366f1"
                fill-opacity="0.6"
              />
              <!-- Axis labels -->
              <text x="170" y="275" text-anchor="middle" fill="currentColor" font-size="11" opacity="0.7">Actual</text>
              <text x="15" y="130" text-anchor="middle" fill="currentColor" font-size="11" opacity="0.7" transform="rotate(-90,15,130)">Predicted</text>
            </svg>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-chart-bar</v-icon>
              Target Statistics
            </h4>
            <v-table density="compact">
              <tbody>
                <tr><td>Target Mean</td><td class="text-right font-weight-medium">{{ (trainingResult.metrics.target_mean || 0).toFixed(4) }}</td></tr>
                <tr><td>Target Std</td><td class="text-right font-weight-medium">{{ (trainingResult.metrics.target_std || 0).toFixed(4) }}</td></tr>
                <tr><td>Target Min</td><td class="text-right font-weight-medium">{{ (trainingResult.metrics.target_min || 0).toFixed(4) }}</td></tr>
                <tr><td>Target Max</td><td class="text-right font-weight-medium">{{ (trainingResult.metrics.target_max || 0).toFixed(4) }}</td></tr>
                <tr v-if="trainingResult.metrics.residuals">
                  <td>Residual Mean</td>
                  <td class="text-right font-weight-medium">{{ (trainingResult.metrics.residuals.mean || 0).toFixed(4) }}</td>
                </tr>
                <tr v-if="trainingResult.metrics.residuals">
                  <td>Residual Std</td>
                  <td class="text-right font-weight-medium">{{ (trainingResult.metrics.residuals.std || 0).toFixed(4) }}</td>
                </tr>
              </tbody>
            </v-table>
          </v-card>
        </v-col>
      </v-row>

      <!-- Regression Time-Series Overlay (Actual vs Predicted over time) -->
      <v-row v-if="pipelineStore.mode === 'regression' && trainingResult.metrics.timeseries_data" class="mt-4">
        <v-col cols="12">
          <v-card variant="outlined" class="pa-4">
            <div class="d-flex align-center mb-1">
              <h4 class="text-subtitle-2 font-weight-bold">
                <v-icon size="small" class="mr-1">mdi-chart-line</v-icon>
                Actual vs Predicted (Time Series)
              </h4>
              <v-spacer />
              <v-btn-toggle v-model="tsViewMode" mandatory density="compact" rounded="lg" class="mr-2">
                <v-btn value="test" size="x-small">Test Only</v-btn>
                <v-btn value="all" size="x-small">Train + Test</v-btn>
              </v-btn-toggle>
            </div>
            <div class="text-caption text-medium-emphasis mb-2">
              {{ trainingResult.metrics.train_samples || trainingResult.metrics.timeseries_data?.train_actual?.length || '?' }} train /
              {{ trainingResult.metrics.test_samples || trainingResult.metrics.timeseries_data?.test_actual?.length || '?' }} test windows
              <template v-if="trainingApproach === 'ti'">
                | <strong>{{ trainingResult.metrics.pipeline === 'cira_features' ? 'CiRA ME windowed features' : 'TI SimpleWindow (raw signal)' }}</strong>
              </template>
              <template v-else>
                | CiRA ME windowed features
              </template>
            </div>
            <div style="width: 100%; overflow-x: auto;">
              <svg :viewBox="`0 0 ${tsChartWidth} 220`" style="width: 100%; min-width: 500px; height: 220px;">
                <!-- Grid lines -->
                <line v-for="i in 4" :key="'grid'+i" :x1="50" :x2="tsChartWidth - 10"
                  :y1="30 + (i-1) * 45" :y2="30 + (i-1) * 45"
                  stroke="currentColor" stroke-opacity="0.1" />
                <!-- Y axis -->
                <line x1="50" y1="20" x2="50" y2="195" stroke="currentColor" stroke-opacity="0.3" />
                <!-- X axis -->
                <line x1="50" :x2="tsChartWidth - 10" y1="195" y2="195" stroke="currentColor" stroke-opacity="0.3" />

                <!-- Y axis labels -->
                <text v-for="(label, i) in tsYLabels" :key="'yl'+i"
                  x="45" :y="35 + i * 45" text-anchor="end" fill="currentColor" font-size="9" opacity="0.5">
                  {{ label }}
                </text>

                <!-- Train/Test separator line -->
                <line v-if="tsViewMode === 'all' && tsTrainLength > 0"
                  :x1="50 + tsTrainLength * tsXScale" :x2="50 + tsTrainLength * tsXScale"
                  y1="20" y2="195"
                  stroke="#FF9800" stroke-width="1" stroke-dasharray="4" />
                <text v-if="tsViewMode === 'all' && tsTrainLength > 0"
                  :x="50 + tsTrainLength * tsXScale + 4" y="30"
                  fill="#FF9800" font-size="9">Test →</text>

                <!-- Actual line -->
                <polyline :points="tsActualPoints" fill="none" stroke="#22D3EE" stroke-width="1.5" stroke-opacity="0.9" />
                <!-- Predicted line -->
                <polyline :points="tsPredictedPoints" fill="none" stroke="#F472B6" stroke-width="1.5" stroke-opacity="0.9" stroke-dasharray="4" />

                <!-- Axis labels -->
                <text :x="tsChartWidth / 2" y="215" text-anchor="middle" fill="currentColor" font-size="10" opacity="0.6">
                  Window Index (time →)
                </text>
                <text x="12" y="110" text-anchor="middle" fill="currentColor" font-size="10" opacity="0.6"
                  transform="rotate(-90, 12, 110)">Value</text>

                <!-- Legend -->
                <line :x1="tsChartWidth - 180" :x2="tsChartWidth - 160" y1="12" y2="12" stroke="#22D3EE" stroke-width="2" />
                <text :x="tsChartWidth - 155" y="16" fill="#22D3EE" font-size="10">Actual</text>
                <line :x1="tsChartWidth - 100" :x2="tsChartWidth - 80" y1="12" y2="12" stroke="#F472B6" stroke-width="2" stroke-dasharray="4" />
                <text :x="tsChartWidth - 75" y="16" fill="#F472B6" font-size="10">Predicted</text>
              </svg>
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Classification/Anomaly Primary Metrics Row -->
      <v-row v-if="pipelineStore.mode !== 'regression'" dense>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Accuracy</div>
            <div class="text-h5 text-success">
              {{ ((trainingResult.metrics.accuracy || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">F1 Score</div>
            <div class="text-h5 text-info">
              {{ ((trainingResult.metrics.f1 || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Precision</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.precision || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Recall</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.recall || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2" v-if="trainingResult.metrics.roc_auc">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">ROC-AUC</div>
            <div class="text-h6 text-purple">
              {{ (trainingResult.metrics.roc_auc || 0).toFixed(3) }}
            </div>
          </v-card>
        </v-col>
        <v-col cols="6" md="2" v-if="trainingResult.metrics.specificity !== undefined">
          <v-card variant="tonal" class="pa-3 text-center">
            <div class="text-caption text-medium-emphasis">Specificity</div>
            <div class="text-h6">
              {{ ((trainingResult.metrics.specificity || 0) * 100).toFixed(1) }}%
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Evaluation Charts Row -->
      <v-row class="mt-4">
        <!-- Confusion Matrix -->
        <v-col cols="12" md="6" v-if="trainingResult.metrics.confusion_matrix">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-grid</v-icon>
              Confusion Matrix
            </h4>
            <div class="confusion-matrix-container">
              <table class="confusion-matrix-table">
                <thead>
                  <tr>
                    <th></th>
                    <th v-for="(label, j) in confusionMatrixLabels" :key="'h'+j" class="header-cell">
                      {{ label }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, i) in trainingResult.metrics.confusion_matrix" :key="'r'+i">
                    <th class="row-header">{{ confusionMatrixLabels[i] || 'Class ' + i }}</th>
                    <td
                      v-for="(cell, j) in row"
                      :key="'c'+j"
                      :class="getCellClass(i, j, cell)"
                      :style="getCellStyle(cell)"
                    >
                      <div class="cell-value">{{ cell }}</div>
                      <div class="cell-percent">{{ getCellPercent(i, j, cell) }}</div>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div class="matrix-legend mt-2">
                <span class="legend-item">
                  <span class="legend-color diagonal"></span> Correct (TP/TN)
                </span>
                <span class="legend-item">
                  <span class="legend-color off-diagonal"></span> Errors (FP/FN)
                </span>
              </div>
            </div>
          </v-card>
        </v-col>

        <!-- ROC Curve -->
        <v-col cols="12" md="6" v-if="trainingResult.metrics.roc_curve">
          <v-card variant="outlined" class="pa-4">
            <h4 class="text-subtitle-2 font-weight-bold mb-3">
              <v-icon size="small" class="mr-1">mdi-chart-line</v-icon>
              ROC Curve (AUC = {{ (trainingResult.metrics.roc_auc || 0).toFixed(3) }})
            </h4>
            <div class="roc-chart-container">
              <svg viewBox="0 0 300 300" class="roc-chart">
                <!-- Grid lines -->
                <g class="grid-lines">
                  <line v-for="i in 5" :key="'hg'+i" :x1="50" :x2="290" :y1="50 + (i-1)*50" :y2="50 + (i-1)*50" />
                  <line v-for="i in 5" :key="'vg'+i" :x1="50 + (i-1)*60" :x2="50 + (i-1)*60" :y1="10" :y2="250" />
                </g>

                <!-- Diagonal line (random classifier) -->
                <line x1="50" y1="250" x2="290" y2="10" class="diagonal-line" />

                <!-- ROC curve -->
                <polyline
                  :points="rocCurvePoints"
                  class="roc-line"
                  fill="none"
                />

                <!-- Fill area under curve -->
                <polygon
                  :points="rocAreaPoints"
                  class="roc-area"
                />

                <!-- Axes -->
                <line x1="50" y1="250" x2="290" y2="250" class="axis" />
                <line x1="50" y1="10" x2="50" y2="250" class="axis" />

                <!-- Labels -->
                <text x="170" y="280" class="axis-label">False Positive Rate</text>
                <text x="15" y="130" class="axis-label" transform="rotate(-90, 15, 130)">True Positive Rate</text>

                <!-- Tick labels -->
                <text x="50" y="265" class="tick-label">0</text>
                <text x="170" y="265" class="tick-label">0.5</text>
                <text x="285" y="265" class="tick-label">1</text>
                <text x="40" y="255" class="tick-label">0</text>
                <text x="40" y="135" class="tick-label">0.5</text>
                <text x="40" y="15" class="tick-label">1</text>
              </svg>
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Detailed Stats for Anomaly Detection -->
      <v-row class="mt-2" v-if="trainingResult.metrics.true_positives !== undefined">
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="success" class="pa-2 text-center">
            <div class="text-caption">True Positives</div>
            <div class="text-h6">{{ trainingResult.metrics.true_positives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="success" class="pa-2 text-center">
            <div class="text-caption">True Negatives</div>
            <div class="text-h6">{{ trainingResult.metrics.true_negatives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="error" class="pa-2 text-center">
            <div class="text-caption">False Positives</div>
            <div class="text-h6">{{ trainingResult.metrics.false_positives }}</div>
          </v-card>
        </v-col>
        <v-col cols="6" md="3">
          <v-card variant="tonal" color="error" class="pa-2 text-center">
            <div class="text-caption">False Negatives</div>
            <div class="text-h6">{{ trainingResult.metrics.false_negatives }}</div>
          </v-card>
        </v-col>
      </v-row>

      <!-- TimesNet specific info -->
      <div v-if="trainingApproach === 'dl' && trainingResult.config" class="mt-4">
        <h4 class="text-subtitle-2 mb-2">Model Configuration</h4>
        <v-chip size="small" class="mr-2 mb-2">d_model: {{ trainingResult.config.d_model }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">layers: {{ trainingResult.config.e_layers }}</v-chip>
        <v-chip size="small" class="mr-2 mb-2">top_k: {{ trainingResult.config.top_k }}</v-chip>
        <v-chip
          v-if="trainingResult.device"
          size="small"
          :color="trainingResult.device === 'cuda' ? 'success' : 'info'"
          class="mr-2 mb-2"
        >
          <v-icon start size="small">{{ trainingResult.device === 'cuda' ? 'mdi-chip' : 'mdi-cpu-64-bit' }}</v-icon>
          {{ trainingResult.device === 'cuda' ? 'GPU' : 'CPU' }}
        </v-chip>
      </div>
    </v-card>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="goBack"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <div>
        <v-btn
          color="secondary"
          size="large"
          class="mr-2"
          :loading="training"
          :disabled="!canTrain"
          @click="trainModel"
        >
          <v-icon start>mdi-play</v-icon>
          Train Model
        </v-btn>

        <v-btn
          color="primary"
          size="large"
          :disabled="!trainingResult"
          @click="router.push({ name: 'pipeline-deploy' })"
        >
          Continue to Deploy
          <v-icon end>mdi-arrow-right</v-icon>
        </v-btn>
      </div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import CodeEditor from '@/components/CodeEditor.vue'
import api from '@/services/api'

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

// Training approach synced with pipeline store
const trainingApproach = computed({
  get: () => pipelineStore.trainingApproach,
  set: (val) => pipelineStore.setTrainingApproach(val as 'ml' | 'dl' | 'custom')
})

// ML state
const selectedAlgorithms = ref<string[]>([])
const training = ref(false)
const trainingResult = ref<any>(null)
const comparisonResult = ref<any>(null)
const pytorchAvailable = ref(true)  // Assume available, check on mount

// Saved models / benchmark state
const savedModels = ref<any[]>([])
const savedModelsRegression = computed(() =>
  savedModels.value.filter(m => m.mode === 'regression')
)
const savedModelsClassification = computed(() =>
  savedModels.value.filter(m => m.mode !== 'regression')
)
const loadingSavedModels = ref(false)
const showSaveBenchmarkDialog = ref(false)
const benchmarkName = ref('')
const savingBenchmark = ref(false)
const selectedModelIds = ref<number[]>([])
const selectAllModels = ref(false)
const comparing = ref(false)
const compareResult = ref<any>(null)

// Evaluate on new data state
const showEvalDialog = ref(false)
const evalModel = ref<any>(null)
const evaluating = ref(false)
const evalResult = ref<any>(null)

// Selected comparison algorithm
const selectedComparisonAlgo = ref<string | null>(null)

// TI TinyML state
const tiServiceAvailable = ref(false)
const tiDevices = ref<Record<string, any>>({})
const tiDeviceList = computed(() => {
  return Object.entries(tiDevices.value).map(([id, dev]: [string, any]) => ({
    id,
    label: `${dev.name} (${dev.family})`,
    ...dev,
  }))
})
const tiSelectedDevice = ref('')
const tiModels = ref<Record<string, any>>({})
const tiModelSource = ref('all')
const tiSelectedModels = ref<string[]>([])

// Filter out NPU models (require TI NN Compiler which is not integrated)
const tiModelsFiltered = computed(() => {
  const result: Record<string, any> = {}
  for (const [key, model] of Object.entries(tiModels.value)) {
    if (!model.npu_only) {
      result[key] = model
    }
  }
  return result
})
const tiComparisonResult = ref<any>(null)
const tiConfig = reactive({
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.001,
  quantization: '8bit',
  max_depth: 10 as number | null,
  test_size: 0.2,
})
const tiLogs = ref<string[]>([])
const tiRunId = ref('')
const tiProgress = ref<{phase: string, epoch: number, total: number, loss: number, mse: number | null, r2: number | null} | null>(null)
const tiComputeInfo = ref('')

// Custom model state
const customModelCode = ref('')
const selectedSnippet = ref<string | null>(null)
const showSaveSnippetDialog = ref(false)
const snippetSaveName = ref('')
const SNIPPETS_KEY = 'cira_custom_model_snippets'

const savedSnippets = computed(() => {
  try {
    const data = localStorage.getItem(SNIPPETS_KEY)
    return data ? JSON.parse(data) : []
  } catch { return [] }
})

function saveSnippet() {
  const name = snippetSaveName.value.trim()
  if (!name || !customModelCode.value.trim()) return
  const snippets = [...savedSnippets.value]
  const existing = snippets.findIndex((s: any) => s.name === name)
  const entry = { name, code: customModelCode.value, savedAt: new Date().toISOString() }
  if (existing >= 0) {
    snippets[existing] = entry
  } else {
    snippets.push(entry)
  }
  localStorage.setItem(SNIPPETS_KEY, JSON.stringify(snippets))
  selectedSnippet.value = name
  showSaveSnippetDialog.value = false
  snippetSaveName.value = ''
}

function loadSnippet(name: string | null) {
  if (!name) return
  const snippet = savedSnippets.value.find((s: any) => s.name === name)
  if (snippet) {
    customModelCode.value = snippet.code
  }
}

function deleteSnippet() {
  if (!selectedSnippet.value) return
  const snippets = savedSnippets.value.filter((s: any) => s.name !== selectedSnippet.value)
  localStorage.setItem(SNIPPETS_KEY, JSON.stringify(snippets))
  selectedSnippet.value = null
}
const customModelLogs = ref<string[]>([])
const customTemplates = ref<any[]>([])
const selectedTemplate = ref('')

const mlHyperparameters = reactive({
  n_estimators: 100,
  contamination: 0.1,
  max_depth: null as number | null,
  test_size: pipelineStore.windowingConfig.test_ratio || 0.2,
  n_neighbors: 5,
  svr_kernel: 'rbf',
  svr_C: 1.0,
  svr_epsilon: 0.1,
})

// Context-aware hyperparameter visibility for regression
const ENSEMBLE_ALGOS = ['rf_reg', 'xgb_reg', 'lgbm_reg']
const TREE_ALGOS = ['rf_reg', 'xgb_reg', 'lgbm_reg', 'dt_reg']

const regNeedsEstimators = computed(() =>
  selectedAlgorithms.value.some(a => ENSEMBLE_ALGOS.includes(a))
)
const regNeedsMaxDepth = computed(() =>
  selectedAlgorithms.value.some(a => TREE_ALGOS.includes(a))
)
const regNeedsNeighbors = computed(() =>
  selectedAlgorithms.value.includes('knn_reg')
)
const regNeedsSvrParams = computed(() =>
  selectedAlgorithms.value.includes('svr')
)

// TimesNet state
const timesnetConfig = reactive({
  d_model: 64,
  d_ff: 128,
  e_layers: 2,
  dropout: 0.1,
  top_k: 3,
  epochs: 50,
  batch_size: 32,
  learning_rate: 0.001
})

// GPU status
const gpuStatus = reactive({
  available: false,
  cuda_available: false,
  torch_available: false,
  dll_error: false,
  device_name: null as string | null,
  memory_total_gb: null as number | null,
  memory_used_gb: null as number | null,
  memory_free_gb: null as number | null,
  error: null as string | null,
  warning: null as string | null,
  info: null as string | null,
  recommendation: 'cpu' as 'cpu' | 'cuda'
})

const selectedDevice = ref<'cpu' | 'cuda'>('cpu')

const availablePeriods = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
const selectedPeriods = ref([8, 16, 32, 64])

const anomalyAlgorithms = [
  { id: 'iforest', name: 'Isolation Forest', description: 'Tree-based anomaly detection', recommended: true },
  { id: 'lof', name: 'Local Outlier Factor', description: 'Density-based local anomalies' },
  { id: 'ocsvm', name: 'One-Class SVM', description: 'Support vector method' },
  { id: 'hbos', name: 'HBOS', description: 'Histogram-based outlier score' },
  { id: 'knn', name: 'KNN', description: 'K-nearest neighbors based' },
  { id: 'copod', name: 'COPOD', description: 'Copula-based outlier detection' },
  { id: 'ecod', name: 'ECOD', description: 'Empirical cumulative distribution' },
  { id: 'autoencoder', name: 'AutoEncoder', description: 'Neural network reconstruction', requiresPytorch: true },
  { id: 'deep_svdd', name: 'Deep SVDD', description: 'Deep support vector data description', requiresPytorch: true },
]

const classificationAlgorithms = [
  { id: 'rf', name: 'Random Forest', description: 'Ensemble of decision trees', recommended: true },
  { id: 'gb', name: 'Gradient Boosting', description: 'Sequential ensemble method' },
  { id: 'svm', name: 'Support Vector Machine', description: 'Margin-based classification' },
  { id: 'mlp', name: 'Multi-Layer Perceptron', description: 'Neural network classifier' },
  { id: 'knn', name: 'K-Nearest Neighbors', description: 'Instance-based learning' },
  { id: 'dt', name: 'Decision Tree', description: 'Single tree classifier' },
  { id: 'nb', name: 'Naive Bayes', description: 'Probabilistic classifier' },
  { id: 'lr', name: 'Logistic Regression', description: 'Linear classifier' },
]

const regressionAlgorithms = [
  { id: 'rf_reg', name: 'Random Forest Regressor', description: 'Ensemble of decision trees for regression', recommended: true },
  { id: 'xgb_reg', name: 'XGBoost Regressor', description: 'Gradient boosting with regularization', recommended: true },
  { id: 'lgbm_reg', name: 'LightGBM Regressor', description: 'Fast gradient boosting framework' },
  { id: 'dt_reg', name: 'Decision Tree Regressor', description: 'Single tree, smallest MCU footprint' },
  { id: 'knn_reg', name: 'KNN Regressor', description: 'Instance-based prediction', noOnnx: true },
  { id: 'svr', name: 'Support Vector Regressor', description: 'Kernel-based regression' },
]

const canTrain = computed(() => {
  if (trainingApproach.value === 'ml') {
    return selectedAlgorithms.value.length > 0 && !!pipelineStore.featureSession
  } else if (trainingApproach.value === 'custom') {
    return !!pipelineStore.featureSession && !!customModelCode.value.trim()
  } else if (trainingApproach.value === 'ti') {
    return !!tiSelectedDevice.value && tiSelectedModels.value.length > 0 && !!pipelineStore.dataSession
  } else {
    return !!pipelineStore.windowedSession
  }
})

// Filter out PyTorch-dependent algorithms when PyTorch is unavailable
const availableAnomalyAlgorithms = computed(() => {
  return anomalyAlgorithms.filter(algo => !algo.requiresPytorch || pytorchAvailable.value)
})

// Unified algorithm list based on current mode
const currentAlgorithmList = computed(() => {
  if (pipelineStore.mode === 'anomaly') return availableAnomalyAlgorithms.value
  if (pipelineStore.mode === 'regression') return regressionAlgorithms
  return classificationAlgorithms
})

// Algorithm selection helpers
function selectAllAlgorithms() {
  selectedAlgorithms.value = currentAlgorithmList.value.map(a => a.id)
}

function clearAlgorithmSelection() {
  selectedAlgorithms.value = []
}

// Confusion matrix helpers
const confusionMatrixLabels = computed(() => {
  if (!trainingResult.value?.metrics) return ['Class 0', 'Class 1']
  return trainingResult.value.metrics.confusion_matrix_labels ||
         trainingResult.value.metrics.class_names ||
         ['Normal', 'Anomaly']
})

const confusionMatrixTotal = computed(() => {
  if (!trainingResult.value?.metrics?.confusion_matrix) return 0
  return trainingResult.value.metrics.confusion_matrix.flat().reduce((a: number, b: number) => a + b, 0)
})

function getCellClass(i: number, j: number, value: number): string {
  const classes = ['matrix-cell']
  if (i === j) {
    classes.push('diagonal')
  } else {
    classes.push('off-diagonal')
  }
  return classes.join(' ')
}

function getCellStyle(value: number): Record<string, string> {
  const total = confusionMatrixTotal.value
  if (total === 0) return {}
  const intensity = Math.min(value / total * 3, 1) // Scale for visibility
  return {
    '--cell-intensity': intensity.toString()
  }
}

function getCellPercent(i: number, j: number, value: number): string {
  const total = confusionMatrixTotal.value
  if (total === 0) return ''
  return `${((value / total) * 100).toFixed(1)}%`
}

// ROC curve helpers
const rocCurvePoints = computed(() => {
  if (!trainingResult.value?.metrics?.roc_curve) return ''
  const roc = trainingResult.value.metrics.roc_curve
  return roc.fpr.map((fpr: number, i: number) => {
    const x = 50 + fpr * 240  // Scale to SVG coordinates
    const y = 250 - roc.tpr[i] * 240
    return `${x},${y}`
  }).join(' ')
})

const rocAreaPoints = computed(() => {
  if (!trainingResult.value?.metrics?.roc_curve) return ''
  const roc = trainingResult.value.metrics.roc_curve
  const points = roc.fpr.map((fpr: number, i: number) => {
    const x = 50 + fpr * 240
    const y = 250 - roc.tpr[i] * 240
    return `${x},${y}`
  })
  // Close the polygon
  points.push('290,250')  // Bottom right
  points.push('50,250')   // Bottom left
  return points.join(' ')
})

// Time-series chart state and computeds
const tsViewMode = ref<'test' | 'all'>('all')

const tsChartWidth = computed(() => {
  const n = tsViewMode.value === 'all'
    ? (trainingResult.value?.metrics?.timeseries_data?.train_actual?.length || 0)
      + (trainingResult.value?.metrics?.timeseries_data?.test_actual?.length || 0)
    : (trainingResult.value?.metrics?.timeseries_data?.test_actual?.length || 0)
  return Math.max(500, Math.min(60 + n * 6, 1200))
})

const tsTrainLength = computed(() => {
  if (tsViewMode.value !== 'all') return 0
  return trainingResult.value?.metrics?.timeseries_data?.train_actual?.length || 0
})

const tsAllData = computed(() => {
  const ts = trainingResult.value?.metrics?.timeseries_data
  if (!ts) return { actual: [], predicted: [] }
  if (tsViewMode.value === 'all') {
    return {
      actual: [...(ts.train_actual || []), ...(ts.test_actual || [])],
      predicted: [...(ts.train_predicted || []), ...(ts.test_predicted || [])],
    }
  }
  return { actual: ts.test_actual || [], predicted: ts.test_predicted || [] }
})

const tsYMin = computed(() => {
  const all = [...tsAllData.value.actual, ...tsAllData.value.predicted]
  return all.length > 0 ? Math.min(...all) : 0
})

const tsYMax = computed(() => {
  const all = [...tsAllData.value.actual, ...tsAllData.value.predicted]
  return all.length > 0 ? Math.max(...all) : 1
})

const tsYRange = computed(() => tsYMax.value - tsYMin.value || 1)

const tsXScale = computed(() => {
  const n = tsAllData.value.actual.length
  return n > 1 ? (tsChartWidth.value - 60) / (n - 1) : 1
})

const tsYLabels = computed(() => {
  const labels = []
  for (let i = 0; i < 4; i++) {
    labels.push((tsYMax.value - (i / 3) * tsYRange.value).toFixed(1))
  }
  return labels
})

function tsToSvgY(val: number): number {
  return 195 - ((val - tsYMin.value) / tsYRange.value) * 165
}

const tsActualPoints = computed(() => {
  return tsAllData.value.actual.map((v, i) =>
    `${50 + i * tsXScale.value},${tsToSvgY(v)}`
  ).join(' ')
})

const tsPredictedPoints = computed(() => {
  return tsAllData.value.predicted.map((v, i) =>
    `${50 + i * tsXScale.value},${tsToSvgY(v)}`
  ).join(' ')
})

// Comparison table headers based on mode
const comparisonHeaders = computed(() => {
  const sizeCol = { key: 'model_size_kb', label: 'Size (KB)', format: (v: number) => v.toFixed(1) }

  if (pipelineStore.mode === 'regression') {
    const cols = [
      { key: 'r2', label: 'R²', format: (v: number) => v.toFixed(4) },
      { key: 'rmse', label: 'RMSE', format: (v: number) => v.toFixed(4) },
      { key: 'mae', label: 'MAE', format: (v: number) => v.toFixed(4) },
      { key: 'mape', label: 'MAPE', format: (v: number) => (v * 100).toFixed(1) + '%' },
    ]
    if (trainingApproach.value === 'ti') cols.push(sizeCol)
    return cols
  }
  const cols = [
    { key: 'accuracy', label: 'Accuracy', format: (v: number) => (v * 100).toFixed(1) + '%' },
    { key: 'precision', label: 'Precision', format: (v: number) => (v * 100).toFixed(1) + '%' },
    { key: 'recall', label: 'Recall', format: (v: number) => (v * 100).toFixed(1) + '%' },
    { key: 'f1', label: 'F1 Score', format: (v: number) => (v * 100).toFixed(1) + '%' },
    { key: 'roc_auc', label: 'ROC-AUC', format: (v: number) => v.toFixed(3) },
  ]
  if (trainingApproach.value === 'ti') cols.push(sizeCol)
  return cols
})

// MCU memory budget computeds
const mcuFlashUsed = computed(() => {
  const modelKb = trainingResult.value?.metrics?.model_size_int8_kb || 0
  const flashKb = tiDevices.value[tiSelectedDevice.value]?.flash_kb || 1024
  return Math.min(100, (modelKb / flashKb) * 100)
})

const mcuFlashRemaining = computed(() => {
  return Math.max(0, Math.round(100 - mcuFlashUsed.value))
})

// Scatter plot helpers for regression
const scatterMin = computed(() => {
  if (!trainingResult.value?.metrics?.scatter_data) return 0
  const actual = trainingResult.value.metrics.scatter_data.actual
  const predicted = trainingResult.value.metrics.scatter_data.predicted
  return Math.min(...actual, ...predicted)
})

const scatterMax = computed(() => {
  if (!trainingResult.value?.metrics?.scatter_data) return 1
  const actual = trainingResult.value.metrics.scatter_data.actual
  const predicted = trainingResult.value.metrics.scatter_data.predicted
  return Math.max(...actual, ...predicted)
})

function goBack() {
  if (trainingApproach.value === 'ml') {
    router.push({ name: 'pipeline-features' })
  } else {
    // DL skips features, go back to windowing
    router.push({ name: 'pipeline-windowing' })
  }
}

function goBackToApplySelection() {
  router.push({ name: 'pipeline-features' })
}

// TSFresh statistical features for type detection
const tsfreshFeatures = [
  'mean', 'std', 'min', 'max', 'median', 'sum', 'variance',
  'skewness', 'kurtosis', 'abs_energy', 'root_mean_square',
  'mean_abs_change', 'mean_change', 'count_above_mean', 'count_below_mean',
  'first_location_of_maximum', 'first_location_of_minimum',
  'last_location_of_maximum', 'last_location_of_minimum',
  'percentage_of_reoccurring_values', 'sum_of_reoccurring_values',
  'abs_sum_of_changes', 'range', 'interquartile_range', 'mean_second_derivative'
]

// Custom DSP features
const dspFeatures = [
  'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
  'impulse_factor', 'margin_factor', 'zero_crossing_rate',
  'autocorr_lag1', 'autocorr_lag5', 'binned_entropy',
  'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
  'spectral_flatness', 'spectral_entropy', 'peak_frequency',
  'spectral_skewness', 'spectral_kurtosis',
  'band_power_low', 'band_power_mid', 'band_power_high'
]

function getFeatureType(featureName: string): string {
  // Extract base name (remove sensor suffix like _acc_x)
  const parts = featureName.split('__')
  const baseName = parts.length > 1 ? parts[0] : featureName.split('_').slice(0, -1).join('_')
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

// Get F1 score color class based on value
function getF1Class(f1: number | null): string {
  if (f1 == null) return ''
  if (f1 >= 0.9) return 'text-success font-weight-bold'
  if (f1 >= 0.7) return 'text-info font-weight-medium'
  if (f1 >= 0.5) return 'text-warning'
  return 'text-error'
}

// Truncate error messages for display in comparison table
function truncateError(error: string, maxLength: number = 60): string {
  if (!error) return 'Unknown error'
  if (error.length <= maxLength) return error
  return error.substring(0, maxLength) + '...'
}

async function trainModel() {
  training.value = true
  trainingResult.value = null

  try {
    if (trainingApproach.value === 'ml') {
      await trainMLModel()
    } else if (trainingApproach.value === 'custom') {
      await trainCustomModel()
    } else if (trainingApproach.value === 'ti') {
      await trainTiModel()
    } else {
      await trainTimesNetModel()
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Training failed')
  } finally {
    training.value = false
  }
}

async function trainMLModel() {
  if (!pipelineStore.featureSession) {
    notificationStore.showError('No features extracted. Please go back and extract features first.')
    return
  }

  if (selectedAlgorithms.value.length === 0) {
    notificationStore.showError('Please select at least one algorithm.')
    return
  }

  // Reset previous results
  trainingResult.value = null
  comparisonResult.value = null
  selectedComparisonAlgo.value = null

  // Decide whether to do single or comparison training
  if (selectedAlgorithms.value.length === 1) {
    // Single algorithm training
    pipelineStore.selectedAlgorithm = selectedAlgorithms.value[0]
    pipelineStore.hyperparameters = { ...mlHyperparameters }

    const result = await pipelineStore.trainModel()

    if (result.success) {
      trainingResult.value = result.data
      notificationStore.showSuccess('Model trained successfully!')
    } else {
      notificationStore.showError(result.error || 'Training failed')
    }
  } else {
    // Multi-algorithm comparison training
    try {
      const endpoint = pipelineStore.mode === 'anomaly'
        ? '/api/training/train/anomaly/compare'
        : pipelineStore.mode === 'regression'
        ? '/api/training/train/regression/compare'
        : '/api/training/train/classification/compare'

      const response = await api.post(endpoint, {
        feature_session_id: pipelineStore.featureSession.session_id,
        algorithms: selectedAlgorithms.value,
        hyperparameters: { ...mlHyperparameters },
        test_size: mlHyperparameters.test_size
      })

      comparisonResult.value = response.data

      // Auto-select the best algorithm
      if (response.data.best_algorithm) {
        selectedComparisonAlgo.value = response.data.best_algorithm.algorithm
        const bestResult = response.data.results.find(
          (r: any) => r.algorithm === response.data.best_algorithm.algorithm
        )
        if (bestResult) {
          trainingResult.value = {
            training_session_id: bestResult.training_session_id,
            algorithm: bestResult.algorithm_name,
            mode: pipelineStore.mode,
            metrics: bestResult.metrics
          }
          pipelineStore.trainingSession = trainingResult.value
        }
      }

      const successCount = response.data.successful
      const failCount = response.data.failed
      if (failCount > 0) {
        notificationStore.showWarning(`Trained ${successCount} algorithms successfully, ${failCount} failed.`)
      } else {
        notificationStore.showSuccess(`Trained ${successCount} algorithms successfully!`)
      }
    } catch (e: any) {
      notificationStore.showError(e.response?.data?.error || 'Comparison training failed')
    }
  }
}

async function trainTimesNetModel() {
  if (!pipelineStore.windowedSession) {
    notificationStore.showError('No windowed data. Please go back and apply windowing first.')
    return
  }

  if (pipelineStore.mode === 'regression') {
    notificationStore.showError('TimesNet does not support regression yet. Please use Traditional ML approach.')
    return
  }

  const endpoint = pipelineStore.mode === 'anomaly'
    ? '/api/training/timesnet/train/anomaly'
    : '/api/training/timesnet/train/classification'

  const response = await api.post(endpoint, {
    windowed_session_id: pipelineStore.windowedSession.session_id,
    config: {
      d_model: timesnetConfig.d_model,
      d_ff: timesnetConfig.d_ff,
      e_layers: timesnetConfig.e_layers,
      dropout: timesnetConfig.dropout,
      top_k: timesnetConfig.top_k,
      period_list: selectedPeriods.value
    },
    epochs: timesnetConfig.epochs,
    batch_size: timesnetConfig.batch_size,
    learning_rate: timesnetConfig.learning_rate,
    test_size: mlHyperparameters.test_size,
    device: selectedDevice.value  // CPU or CUDA
  })

  if (response.data.error) {
    notificationStore.showError(response.data.error)
    return
  }

  trainingResult.value = response.data
  pipelineStore.trainingSession = response.data
  notificationStore.showSuccess('TimesNet model trained successfully!')
}

// ─── Comparison Model Selection ──────────────────────────────────

function selectComparisonModel(algorithmId: string) {
  if (!comparisonResult.value) return

  selectedComparisonAlgo.value = algorithmId

  // Find the result for this algorithm
  const result = comparisonResult.value.results.find(
    (r: any) => r.algorithm === algorithmId
  )
  if (!result) return

  // Update trainingResult to show this model's details
  trainingResult.value = {
    training_session_id: result.training_session_id,
    algorithm: result.algorithm_name,
    mode: pipelineStore.mode,
    metrics: result.metrics,
  }
  pipelineStore.trainingSession = trainingResult.value
}

function getSelectedAlgoName(): string {
  if (!selectedComparisonAlgo.value || !comparisonResult.value) return ''
  const result = comparisonResult.value.results.find(
    (r: any) => r.algorithm === selectedComparisonAlgo.value
  )
  return result?.algorithm_name || selectedComparisonAlgo.value
}

// ─── TI TinyML Functions ─────────────────────────────────────────

async function fetchTiStatus() {
  try {
    const resp = await api.get('/api/ti/status')
    tiServiceAvailable.value = resp.data.status === 'healthy'
    tiComputeInfo.value = resp.data.compute === 'GPU'
      ? `GPU: ${resp.data.gpu_name || 'Available'}`
      : 'CPU Only'
  } catch {
    tiServiceAvailable.value = false
    tiComputeInfo.value = ''
  }
}

async function fetchTiDevices() {
  try {
    const resp = await api.get('/api/ti/devices')
    tiDevices.value = resp.data

    // Filter devices by current mode
    const tiTask = pipelineStore.mode === 'anomaly' ? 'timeseries_anomalydetection'
      : pipelineStore.mode === 'regression' ? 'timeseries_regression'
      : 'timeseries_classification'

    const filtered: Record<string, any> = {}
    for (const [id, dev] of Object.entries(resp.data) as [string, any][]) {
      if (dev.tasks?.includes(tiTask)) {
        filtered[id] = dev
      }
    }
    tiDevices.value = filtered
  } catch {
    tiDevices.value = {}
  }
}

async function fetchTiModels() {
  if (!tiSelectedDevice.value) return
  const tiTask = pipelineStore.mode === 'anomaly' ? 'timeseries_anomalydetection'
    : pipelineStore.mode === 'regression' ? 'timeseries_regression'
    : 'timeseries_classification'

  try {
    const resp = await api.get('/api/ti/models', {
      params: {
        task: tiTask,
        device: tiSelectedDevice.value,
        source: tiModelSource.value,
      }
    })
    tiModels.value = resp.data
    tiSelectedModels.value = []
  } catch {
    tiModels.value = {}
  }
}

function tiSelectAll() {
  tiSelectedModels.value = Object.keys(tiModelsFiltered.value)
}

// Auto-suggest epochs based on selected models
const tiSuggestedEpochs = computed(() => {
  let maxMin = 0
  for (const key of tiSelectedModels.value) {
    const model = tiModels.value[key]
    if (model?.min_epochs && model.min_epochs > maxMin) {
      maxMin = model.min_epochs
    }
  }
  return maxMin || 50
})

// Watch selection changes and suggest epochs
watch(tiSelectedModels, (newVal) => {
  if (newVal.length > 0 && tiSuggestedEpochs.value > tiConfig.epochs) {
    tiConfig.epochs = tiSuggestedEpochs.value
  }
})

async function trainTiModel() {
  if (!tiSelectedDevice.value || tiSelectedModels.value.length === 0 || !pipelineStore.dataSession) return

  training.value = true
  trainingResult.value = null
  comparisonResult.value = null
  tiComparisonResult.value = null
  selectedComparisonAlgo.value = null
  tiLogs.value = []
  tiRunId.value = ''
  tiProgress.value = null

  // Train models one by one with progress updates
  const allResults: any[] = []
  const allErrors: any[] = []

  for (let i = 0; i < tiSelectedModels.value.length; i++) {
    const modelName = tiSelectedModels.value[i]
    const modelInfo = tiModels.value[modelName] || {}

    tiProgress.value = {
      phase: `Training ${i + 1}/${tiSelectedModels.value.length}: ${modelInfo.name || modelName}`,
      epoch: i,
      total: tiSelectedModels.value.length,
      loss: 0,
      mse: null,
      r2: null,
    }

    try {
      let resultData: any

      if (modelName.startsWith('ML_')) {
        // Traditional ML: use CiRA ME's feature pipeline
        if (!pipelineStore.featureSession) {
          allErrors.push({
            model_name: modelName,
            algorithm_name: modelInfo.name || modelName,
            error: 'Features not extracted. Go to Features page first.',
            status: 'failed',
          })
          tiLogs.value.push(`${modelInfo.name || modelName}: Skipped — no features extracted`)
          continue
        }

        const resp = await api.post('/api/ti/train-ml', {
          feature_session_id: pipelineStore.featureSession.session_id,
          model_name: modelName,
          target_device: tiSelectedDevice.value,
          mode: pipelineStore.mode,
          test_size: tiConfig.test_size,
          hyperparameters: { max_depth: tiConfig.max_depth },
        })

        // Wrap single result in batch format — use real training_session_id for save-benchmark
        const realSessionId = resp.data.training_session_id || modelName
        resultData = {
          results: [{
            model_name: realSessionId,
            algorithm_name: modelInfo.name || modelName,
            status: 'success',
            metrics: { ...(resp.data.metrics || {}), pipeline: 'cira_features' },
            source: 'traditional_ml',
            training_session_id: realSessionId,
          }],
          errors: [],
          run_id: realSessionId,
        }
      } else {
        // TI NN: use TI's own pipeline with raw data
        const resp = await api.post('/api/ti/train', {
          mode: pipelineStore.mode,
          model_names: [modelName],
          target_device: tiSelectedDevice.value,
          dataset_path: pipelineStore.dataSession!.metadata.file_path,
          config: { ...tiConfig },
        })
        resultData = resp.data
      }

      tiRunId.value = resultData.run_id || ''

      for (const r of (resultData.results || [])) {
        allResults.push(r)
        if (r.metrics?.r2 != null) {
          tiProgress.value!.r2 = r.metrics.r2
        }
        if (r.metrics?.rmse != null) {
          tiProgress.value!.mse = r.metrics.rmse
        }
        tiLogs.value.push(`${r.algorithm_name}: R²=${r.metrics?.r2 ?? '?'}, RMSE=${r.metrics?.rmse?.toFixed(2) ?? '?'}`)
      }
      for (const e of (resultData.errors || [])) {
        allErrors.push(e)
        tiLogs.value.push(`${e.algorithm_name}: Failed — ${e.error?.slice(0, 60)}`)
      }
    } catch (e: any) {
      allErrors.push({
        model_name: modelName,
        algorithm_name: modelInfo.name || modelName,
        error: e.response?.data?.error || e.message,
        status: 'failed',
      })
    }
  }

  tiProgress.value = null

  // Build combined comparison result
  const best = _findBestTiResult(allResults)
  const combined = {
    successful: allResults.length,
    failed: allErrors.length,
    results: allResults,
    errors: allErrors,
    best_algorithm: best,
  }
  _processTiResults(combined)
  training.value = false
}

async function trainTiModelStream(modelName: string) {
  tiProgress.value = { phase: 'starting', epoch: 0, total: tiConfig.epochs, loss: 0, mse: null, r2: null }
  tiLogs.value.push(`Starting ${modelName}...`)

  try {
    const resp = await fetch('/api/ti/train-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        mode: pipelineStore.mode,
        model_name: modelName,
        target_device: tiSelectedDevice.value,
        dataset_path: pipelineStore.dataSession!.metadata.file_path,
        config: { ...tiConfig },
      }),
    })

    const reader = resp.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) {
      notificationStore.showError('Streaming not supported')
      return
    }

    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const chunk of lines) {
        if (!chunk.startsWith('data: ')) continue
        try {
          const event = JSON.parse(chunk.slice(6))
          switch (event.type) {
            case 'epoch':
              tiProgress.value = {
                phase: event.phase || 'float',
                epoch: event.epoch,
                total: event.total || tiConfig.epochs,
                loss: event.loss,
                mse: tiProgress.value?.mse ?? null,
                r2: tiProgress.value?.r2 ?? null,
              }
              break
            case 'eval':
              if (tiProgress.value) tiProgress.value.mse = event.mse
              break
            case 'eval_r2':
              if (tiProgress.value) tiProgress.value.r2 = event.r2
              break
            case 'status':
              tiLogs.value.push(event.message)
              break
            case 'params':
              tiLogs.value.push(`Trainable params: ${event.trainable_params.toLocaleString()}`)
              break
            case 'best':
              tiLogs.value.push(`Best ${event.phase}: ${event.metric} = ${event.value}`)
              break
            case 'complete':
              tiRunId.value = event.run_id || ''
              tiProgress.value = null
              // Fetch full results via batch API for comparison table
              if (event.result) {
                // Traditional ML complete result
                _processTiSingleResult(modelName, event.result)
              }
              break
            case 'error':
              notificationStore.showError(event.message)
              break
          }
        } catch {
          // Skip malformed SSE
        }
      }
    }

    // If streaming ended without complete event, fetch results
    if (!trainingResult.value && tiRunId.value) {
      notificationStore.showSuccess(`${modelName} training complete!`)
    }

  } catch (e: any) {
    notificationStore.showError(`Streaming failed: ${e.message}`)
  } finally {
    tiProgress.value = null
  }
}

function _findBestTiResult(results: any[]) {
  let best: any = null
  let bestScore = -Infinity
  const isRegression = pipelineStore.mode === 'regression'

  for (const r of results) {
    const score = isRegression
      ? (r.metrics?.r2 ?? -Infinity)
      : (r.metrics?.f1 ?? r.metrics?.accuracy ?? 0)
    if (score > bestScore) {
      bestScore = score
      best = {
        model_name: r.model_name,
        algorithm_name: r.algorithm_name,
        score,
        metric: isRegression ? 'r2' : 'f1',
      }
    }
  }
  return best
}

function _processTiSingleResult(modelName: string, result: any) {
  const modelInfo = tiModels.value[modelName] || {}
  trainingResult.value = {
    training_session_id: modelName,
    algorithm: modelInfo.name || modelName,
    mode: pipelineStore.mode,
    metrics: result.metrics || {},
  }
  pipelineStore.trainingSession = trainingResult.value
  notificationStore.showSuccess(`${modelInfo.name || modelName} trained!`)
}

function _processTiResults(data: any) {
  // Collect all logs
  for (const r of (data.results || [])) {
    tiLogs.value.push(`--- ${r.algorithm_name} ---`)
    tiLogs.value.push(...(r.logs || []).slice(-5))
  }

  // Build comparison result
  comparisonResult.value = {
    successful: data.successful,
    failed: data.failed,
    best_algorithm: data.best_algorithm,
    comparison: {
      rows: data.results.map((r: any) => ({
        algorithm: r.model_name,
        algorithm_name: r.algorithm_name,
        values: r.metrics,
      })),
    },
    results: data.results.map((r: any) => ({
      algorithm: r.model_name,
      algorithm_name: r.algorithm_name,
      training_session_id: r.training_session_id || r.model_name,
      metrics: r.metrics,
    })),
    errors: data.errors || [],
  }

  // Auto-select best
  if (data.best_algorithm) {
    selectedComparisonAlgo.value = data.best_algorithm.model_name
    const bestResult = data.results.find((r: any) => r.model_name === data.best_algorithm.model_name)
    if (bestResult) {
      trainingResult.value = {
        training_session_id: data.run_id,
        algorithm: bestResult.algorithm_name,
        mode: pipelineStore.mode,
        metrics: bestResult.metrics,
      }
      pipelineStore.trainingSession = trainingResult.value
    }
  }

  if (data.failed > 0) {
    notificationStore.showWarning(`Trained ${data.successful} models, ${data.failed} failed`)
  } else {
    notificationStore.showSuccess(`Trained ${data.successful} models successfully!`)
  }
}

// ─── Custom Model Functions ───────────────────────────────────────

async function trainCustomModel() {
  if (!pipelineStore.featureSession) {
    notificationStore.showError('No features extracted. Please go back and extract features first.')
    return
  }

  training.value = true
  trainingResult.value = null
  comparisonResult.value = null
  customModelLogs.value = []

  try {
    const response = await api.post('/api/training/custom-model/execute', {
      code: customModelCode.value,
      feature_session_id: pipelineStore.featureSession.session_id,
      task: pipelineStore.mode,
      test_size: mlHyperparameters.test_size,
    })

    const data = response.data
    customModelLogs.value = data.logs || []

    if (data.status === 'success') {
      trainingResult.value = {
        training_session_id: data.training_session_id,
        algorithm: 'Custom Model',
        mode: pipelineStore.mode,
        metrics: data.metrics,
      }
      pipelineStore.trainingSession = trainingResult.value
      notificationStore.showSuccess('Custom model trained successfully!')
    } else {
      const errorMsg = data.error || 'Custom model execution failed'
      if (data.traceback) {
        customModelLogs.value.push('--- Traceback ---')
        customModelLogs.value.push(data.traceback)
      }
      notificationStore.showError(errorMsg)
    }
  } catch (e: any) {
    const errorData = e.response?.data
    if (errorData?.logs) {
      customModelLogs.value = errorData.logs
    }
    if (errorData?.traceback) {
      customModelLogs.value.push('--- Traceback ---')
      customModelLogs.value.push(errorData.traceback)
    }
    notificationStore.showError(errorData?.error || 'Custom model execution failed')
  } finally {
    training.value = false
  }
}

async function fetchCustomTemplates() {
  try {
    const response = await api.get('/api/training/custom-model/templates')
    customTemplates.value = response.data
    // Load first template matching current mode
    const match = customTemplates.value.find(t => t.task === pipelineStore.mode)
    if (match && !customModelCode.value) {
      selectedTemplate.value = match.id
      customModelCode.value = match.code
    }
  } catch {
    customTemplates.value = []
  }
}

function loadTemplate(templateId: string) {
  const template = customTemplates.value.find(t => t.id === templateId)
  if (template) {
    customModelCode.value = template.code
    customModelLogs.value = []
  }
}

// Fetch GPU status
async function fetchGpuStatus() {
  try {
    const response = await api.get('/api/training/gpu-status')
    Object.assign(gpuStatus, response.data)
    // Set recommended device
    selectedDevice.value = gpuStatus.recommendation
  } catch (e: any) {
    gpuStatus.error = 'Failed to check GPU status'
    gpuStatus.available = false
    selectedDevice.value = 'cpu'
  }
}

// Watch for approach changes
watch(trainingApproach, (newVal) => {
  trainingResult.value = null
  comparisonResult.value = null
  if (newVal === 'dl') {
    fetchGpuStatus()
  } else if (newVal === 'custom') {
    fetchCustomTemplates()
  } else if (newVal === 'ti') {
    fetchTiStatus()
    fetchTiDevices()
  }
})

// Watch for mode changes to reset algorithm selection
watch(() => pipelineStore.mode, (newMode) => {
  if (newMode === 'anomaly') {
    selectedAlgorithms.value = ['iforest']
  } else if (newMode === 'regression') {
    selectedAlgorithms.value = ['rf_reg']
  } else {
    selectedAlgorithms.value = ['rf']
  }
  trainingResult.value = null
  comparisonResult.value = null
  // Re-fetch templates/models when mode changes
  if (trainingApproach.value === 'custom') {
    fetchCustomTemplates()
  } else if (trainingApproach.value === 'ti') {
    fetchTiDevices()
    tiSelectedModels.value = []
    tiModels.value = {}
    // Re-fetch models for the new mode if device is already selected
    if (tiSelectedDevice.value) {
      fetchTiModels()
    }
  }
})

// ─── Saved Models / Benchmark Functions ───────────────────────────

async function loadSavedModels() {
  loadingSavedModels.value = true
  try {
    const response = await api.get('/api/training/saved-models')
    savedModels.value = response.data
  } catch {
    savedModels.value = []
  }
  loadingSavedModels.value = false
}

async function saveBenchmark() {
  // Determine training_session_id: selected model > best model > single training
  let sessionId: string | undefined
  if (selectedComparisonAlgo.value && comparisonResult.value) {
    const selected = comparisonResult.value.results.find(
      (r: any) => r.algorithm === selectedComparisonAlgo.value
    )
    sessionId = selected?.training_session_id
  } else if (comparisonResult.value?.best_algorithm) {
    sessionId = comparisonResult.value.best_algorithm.training_session_id
  } else if (trainingResult.value?.training_session_id) {
    sessionId = trainingResult.value.training_session_id
  }

  if (!sessionId) {
    notificationStore.showError('No training session to save')
    return
  }

  savingBenchmark.value = true

  try {
    await api.post('/api/training/save-benchmark', {
      training_session_id: sessionId,
      name: benchmarkName.value || undefined,
      // For TI models: include metrics and algorithm since session is in TI container
      metrics: trainingResult.value?.metrics || undefined,
      algorithm: trainingResult.value?.algorithm || undefined,
      mode: pipelineStore.mode,
      pipeline_config: {
        version: 1,
        training_approach: pipelineStore.trainingApproach,
        mode: pipelineStore.mode,
        // Session IDs for backend to pull full data
        windowed_session_id: pipelineStore.windowedSession?.session_id || null,
        feature_session_id: pipelineStore.featureSession?.session_id || null,
        selection_session_id: pipelineStore.featureSelectionState.selectionApplied
          ? pipelineStore.featureSelectionState.selectionResult?.session_id || null
          : null,
        // Target column (regression mode)
        target_column: pipelineStore.targetColumn || null,
        // Windowing params
        windowing: {
          window_size: pipelineStore.windowingConfig.window_size,
          stride: pipelineStore.windowingConfig.stride,
          label_method: pipelineStore.windowingConfig.label_method,
          test_ratio: pipelineStore.windowingConfig.test_ratio,
        },
        // Normalization from windowed session metadata
        normalization: pipelineStore.windowedSession?.metadata?.normalization || null,
        // Feature extraction info (ML only)
        feature_extraction: pipelineStore.featureSession ? {
          feature_names: pipelineStore.featureSession.feature_names,
          num_features: pipelineStore.featureSession.num_features,
          method: pipelineStore.featureSelectionState.extractionResult?.feature_set
            ? 'tsfresh' : 'lightweight',
          feature_set: pipelineStore.featureSelectionState.extractionResult?.feature_set || null,
        } : null,
        // Feature selection info (ML only, if applied)
        feature_selection: pipelineStore.featureSelectionState.selectionApplied
          ? {
            method: pipelineStore.featureSelectionState.selectionResult?.method,
            fdr_level: pipelineStore.featureSelectionState.selectionResult?.fdr_level,
            selected_features: pipelineStore.featureSelectionState.selectionResult?.selected_features,
            num_selected: pipelineStore.featureSelectionState.selectionResult?.final_count,
          }
          : null,
      },
      dataset_info: {
        format: pipelineStore.dataSession?.metadata?.format,
        file_path: pipelineStore.dataSession?.metadata?.file_path,
        filename: pipelineStore.dataSession?.metadata?.file_path?.split(/[/\\]/).pop(),
        total_rows: pipelineStore.dataSession?.metadata?.total_rows,
        columns: pipelineStore.dataSession?.metadata?.columns,
        sensor_columns: pipelineStore.dataSession?.metadata?.sensor_columns,
        label_column: pipelineStore.dataSession?.metadata?.label_column,
        labels: pipelineStore.dataSession?.metadata?.labels,
      }
    })
    notificationStore.showSuccess('Model saved as benchmark')
    showSaveBenchmarkDialog.value = false
    benchmarkName.value = ''
    loadSavedModels()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to save benchmark')
  }
  savingBenchmark.value = false
}

async function deleteSavedModel(model: any) {
  if (!confirm(`Delete benchmark "${model.name}"?`)) return
  try {
    await api.delete(`/api/training/saved-models/${model.id}`)
    notificationStore.showSuccess('Benchmark deleted')
    loadSavedModels()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to delete')
  }
}

function toggleSelectAllModels(val: boolean) {
  selectedModelIds.value = val ? savedModels.value.map((m: any) => m.id) : []
}

async function compareSelectedModels() {
  if (selectedModelIds.value.length !== 2) {
    notificationStore.showError('Select exactly 2 models to compare')
    return
  }
  comparing.value = true
  try {
    const response = await api.post('/api/training/saved-models/compare', {
      model_id_1: selectedModelIds.value[0],
      model_id_2: selectedModelIds.value[1]
    })
    compareResult.value = response.data
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Comparison failed')
  }
  comparing.value = false
}

// ─── Evaluate on New Data Functions ───────────────────────────────

function startEvaluation(model: any) {
  evalModel.value = model
  evalResult.value = null
  showEvalDialog.value = true
}

async function runEvaluation() {
  if (!evalModel.value || !pipelineStore.featureSession) return
  evaluating.value = true
  try {
    const response = await api.post('/api/training/evaluate', {
      saved_model_id: evalModel.value.id,
      feature_session_id: pipelineStore.featureSession.session_id
    })
    evalResult.value = response.data
    notificationStore.showSuccess('Evaluation complete')
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Evaluation failed')
  }
  evaluating.value = false
}

onMounted(async () => {
  // Set default algorithm selection (recommended ones)
  if (pipelineStore.mode === 'anomaly') {
    selectedAlgorithms.value = ['iforest']
  } else if (pipelineStore.mode === 'regression') {
    selectedAlgorithms.value = ['rf_reg']
  } else {
    selectedAlgorithms.value = ['rf']
  }

  // Check GPU/PyTorch availability
  try {
    const gpuResponse = await api.get('/api/training/gpu-status')
    pytorchAvailable.value = gpuResponse.data.torch_available !== false
    if (gpuResponse.data.dll_error) {
      pytorchAvailable.value = false
    }
  } catch {
    pytorchAvailable.value = false
  }

  // Fetch GPU status if starting with deep learning approach
  if (trainingApproach.value === 'dl') {
    fetchGpuStatus()
  } else if (trainingApproach.value === 'custom') {
    fetchCustomTemplates()
  } else if (trainingApproach.value === 'ti') {
    fetchTiStatus()
    fetchTiDevices()
  }

  // Load saved models
  loadSavedModels()
})
</script>

<style scoped lang="scss">
// Confusion Matrix Styles
.confusion-matrix-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.confusion-matrix-table {
  border-collapse: collapse;
  width: 100%;
  max-width: 400px;

  th, td {
    border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
    padding: 8px 12px;
    text-align: center;
  }

  th {
    background: rgba(var(--v-theme-surface-variant), 0.5);
    font-weight: 600;
    font-size: 0.85rem;
  }

  .header-cell {
    min-width: 80px;
  }

  .row-header {
    text-align: right;
    font-weight: 600;
    background: rgba(var(--v-theme-surface-variant), 0.3);
  }

  .matrix-cell {
    position: relative;
    min-width: 80px;
    min-height: 60px;

    .cell-value {
      font-size: 1.25rem;
      font-weight: 700;
    }

    .cell-percent {
      font-size: 0.75rem;
      opacity: 0.7;
    }

    &.diagonal {
      background: rgba(16, 185, 129, calc(0.15 + var(--cell-intensity, 0) * 0.4));
      color: #10B981;
    }

    &.off-diagonal {
      background: rgba(239, 68, 68, calc(0.1 + var(--cell-intensity, 0) * 0.3));
      color: #EF4444;
    }
  }
}

.matrix-legend {
  display: flex;
  gap: 16px;
  font-size: 0.8rem;

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;

    &.diagonal {
      background: rgba(16, 185, 129, 0.4);
    }

    &.off-diagonal {
      background: rgba(239, 68, 68, 0.3);
    }
  }
}

// ROC Curve Styles
.roc-chart-container {
  display: flex;
  justify-content: center;
}

.roc-chart {
  width: 100%;
  max-width: 350px;
  height: auto;

  .grid-lines line {
    stroke: rgba(var(--v-border-color), 0.2);
    stroke-width: 0.5;
  }

  .diagonal-line {
    stroke: rgba(var(--v-theme-on-surface), 0.3);
    stroke-width: 1;
    stroke-dasharray: 5, 5;
  }

  .roc-line {
    stroke: #8B5CF6;
    stroke-width: 2.5;
  }

  .roc-area {
    fill: rgba(139, 92, 246, 0.15);
  }

  .axis {
    stroke: rgba(var(--v-theme-on-surface), 0.5);
    stroke-width: 1.5;
  }

  .axis-label {
    font-size: 11px;
    fill: rgba(var(--v-theme-on-surface), 0.7);
    text-anchor: middle;
  }

  .tick-label {
    font-size: 9px;
    fill: rgba(var(--v-theme-on-surface), 0.6);
    text-anchor: middle;
  }
}

// Legacy confusion matrix (keep for backwards compatibility)
.confusion-matrix {
  table {
    border-collapse: collapse;
    width: 100%;
    max-width: 200px;

    td {
      border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
      padding: 12px;
      text-align: center;
      font-weight: 500;

      &.diagonal {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
      }
    }
  }
}

// Feature list container
.feature-list-container {
  max-height: 200px;
  overflow-y: auto;
  padding: 8px;
  background: rgba(var(--v-theme-surface-variant), 0.2);
  border-radius: 8px;
}

// Algorithm selection list
.algorithm-list {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;
  padding: 8px;

  .algorithm-checkbox {
    padding: 8px 4px;
    border-bottom: 1px solid rgba(var(--v-border-color), 0.1);

    &:last-child {
      border-bottom: none;
    }

    &:hover {
      background: rgba(var(--v-theme-surface-variant), 0.3);
      border-radius: 4px;
    }
  }
}

// Comparison table styles
.comparison-table {
  .best-row {
    background: rgba(16, 185, 129, 0.05);
  }

  .selected-row {
    background: rgba(99, 102, 241, 0.15) !important;
    outline: 2px solid rgba(99, 102, 241, 0.5);
    outline-offset: -2px;
  }

  tbody tr:hover {
    background: rgba(99, 102, 241, 0.08) !important;
  }

  .error-row {
    background: rgba(239, 68, 68, 0.05);
  }

  th {
    background: rgba(var(--v-theme-surface-variant), 0.5);
    font-weight: 600;
  }
}

// Error message truncation in comparison table
.error-message-truncate {
  display: inline-block;
  max-width: 400px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: help;
  vertical-align: middle;
}

.gap-2 {
  gap: 8px;
}
</style>
