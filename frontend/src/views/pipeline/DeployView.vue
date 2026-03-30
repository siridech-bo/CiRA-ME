<template>
  <v-container fluid class="pa-6">
    <!-- Header with Stepper -->
    <PipelineStepper current-step="deploy" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Deploy to Edge</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Select a model and deploy it to an edge device
    </p>

    <!-- Model Selection -->
    <v-card class="pa-4 mb-6">
      <h3 class="text-subtitle-1 font-weight-bold mb-4">
        <v-icon start color="primary">mdi-brain</v-icon>
        Select Model
      </h3>

      <v-radio-group v-model="modelSource" hide-details>
        <!-- Current Session -->
        <v-radio
          value="session"
          :disabled="!pipelineStore.trainingSession"
        >
          <template #label>
            <div class="d-flex align-center flex-wrap ga-2" style="width: 100%">
              <span class="font-weight-medium">Current Training Session</span>
              <template v-if="pipelineStore.trainingSession">
                <v-chip size="x-small" color="primary" variant="tonal">
                  {{ pipelineStore.trainingSession.algorithm }}
                </v-chip>
                <v-chip size="x-small" color="info" variant="tonal">
                  {{ pipelineStore.trainingSession.mode }}
                </v-chip>
                <v-chip
                  v-if="pipelineStore.trainingSession.metrics?.accuracy != null"
                  size="x-small" color="success" variant="tonal"
                >
                  Acc: {{ (pipelineStore.trainingSession.metrics.accuracy * 100).toFixed(1) }}%
                </v-chip>
                <v-chip
                  v-if="pipelineStore.trainingSession.metrics?.f1 != null"
                  size="x-small" color="warning" variant="tonal"
                >
                  F1: {{ (pipelineStore.trainingSession.metrics.f1 * 100).toFixed(1) }}%
                </v-chip>
              </template>
              <span v-else class="text-caption text-medium-emphasis">(no model trained in this session)</span>
            </div>
          </template>
        </v-radio>

        <!-- Saved Models -->
        <v-radio value="saved">
          <template #label>
            <span class="font-weight-medium">Saved Benchmark Model</span>
          </template>
        </v-radio>
      </v-radio-group>

      <!-- Saved Models Table (shown when 'saved' is selected) -->
      <div v-if="modelSource === 'saved'" class="mt-4">
        <div v-if="loadingSavedModels" class="text-center pa-4">
          <v-progress-circular indeterminate size="24" />
          <span class="ml-2 text-medium-emphasis">Loading saved models...</span>
        </div>

        <v-alert v-else-if="savedModels.length === 0" type="info" variant="tonal">
          No saved models found. Save a benchmark from the Training page first.
        </v-alert>

        <!-- Mode tabs for model table -->
        <v-btn-toggle v-else v-model="modelTableTab" mandatory density="compact" class="mb-3">
          <v-btn value="regression" size="small" :color="modelTableTab === 'regression' ? 'purple' : undefined">
            <v-icon start size="small">mdi-chart-timeline-variant</v-icon>
            Regression ({{ regressionModels.length }})
          </v-btn>
          <v-btn value="classification" size="small" :color="modelTableTab === 'classification' ? 'info' : undefined">
            <v-icon start size="small">mdi-shape</v-icon>
            Classification / Anomaly ({{ classAnomalyModels.length }})
          </v-btn>
        </v-btn-toggle>

        <!-- Regression Models Table -->
        <v-table v-if="modelTableTab === 'regression' && regressionModels.length > 0" dense hover>
          <thead>
            <tr>
              <th style="width:40px"></th>
              <th>Name</th>
              <th>Type</th>
              <th>Algorithm</th>
              <th class="text-center">R²</th>
              <th class="text-center">RMSE</th>
              <th class="text-center">MAE</th>
              <th>Date</th>
              <th style="width:40px"></th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="model in regressionModels"
              :key="model.id"
              :class="{ 'bg-purple-darken-4': selectedSavedModelId === model.id }"
              style="cursor: pointer"
              @click="selectSavedModel(model)"
            >
              <td>
                <v-radio-group v-model="selectedSavedModelId" hide-details inline>
                  <v-radio :value="model.id" density="compact" hide-details />
                </v-radio-group>
              </td>
              <td class="font-weight-medium">{{ model.name }}</td>
              <td>
                <v-chip size="x-small" :color="getModelTypeInfo(model).color" variant="flat" class="font-weight-bold" style="font-size:9px; letter-spacing:0.5px">
                  {{ getModelTypeInfo(model).label }}
                </v-chip>
              </td>
              <td class="text-caption">{{ model.algorithm }}</td>
              <td class="text-center" :style="{ color: model.metrics?.r2 > 0.8 ? '#34d399' : model.metrics?.r2 > 0.5 ? '#fbbf24' : '#f87171' }">
                {{ model.metrics?.r2 != null ? model.metrics.r2.toFixed(4) : '-' }}
              </td>
              <td class="text-center">{{ model.metrics?.rmse != null ? model.metrics.rmse.toFixed(4) : '-' }}</td>
              <td class="text-center">{{ model.metrics?.mae != null ? model.metrics.mae.toFixed(4) : '-' }}</td>
              <td class="text-caption">{{ formatDate(model.created_at) }}</td>
              <td>
                <v-btn icon size="x-small" variant="text" color="error" @click.stop="confirmDeleteModel(model)">
                  <v-icon size="small">mdi-delete</v-icon>
                </v-btn>
              </td>
            </tr>
          </tbody>
        </v-table>
        <div v-else-if="modelTableTab === 'regression'" class="text-center text-medium-emphasis py-4">
          No regression models saved yet.
        </div>

        <!-- Classification / Anomaly Models Table -->
        <v-table v-if="modelTableTab === 'classification' && classAnomalyModels.length > 0" dense hover>
          <thead>
            <tr>
              <th style="width:40px"></th>
              <th>Name</th>
              <th>Type</th>
              <th>Algorithm</th>
              <th>Mode</th>
              <th class="text-center">Accuracy</th>
              <th class="text-center">Precision</th>
              <th class="text-center">Recall</th>
              <th class="text-center">F1</th>
              <th>Date</th>
              <th style="width:40px"></th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="model in classAnomalyModels"
              :key="model.id"
              :class="{ 'bg-primary-darken-3': selectedSavedModelId === model.id }"
              style="cursor: pointer"
              @click="selectSavedModel(model)"
            >
              <td>
                <v-radio-group v-model="selectedSavedModelId" hide-details inline>
                  <v-radio :value="model.id" density="compact" hide-details />
                </v-radio-group>
              </td>
              <td class="font-weight-medium">{{ model.name }}</td>
              <td>
                <v-chip size="x-small" :color="getModelTypeInfo(model).color" variant="flat" class="font-weight-bold" style="font-size:9px; letter-spacing:0.5px">
                  {{ getModelTypeInfo(model).label }}
                </v-chip>
              </td>
              <td class="text-caption">{{ model.algorithm }}</td>
              <td>
                <v-chip size="x-small" :color="model.mode === 'anomaly' ? 'warning' : 'info'" variant="tonal">
                  {{ model.mode }}
                </v-chip>
              </td>
              <td class="text-center">{{ model.metrics?.accuracy != null ? (model.metrics.accuracy * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ model.metrics?.precision != null ? (model.metrics.precision * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ model.metrics?.recall != null ? (model.metrics.recall * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-center">{{ model.metrics?.f1 != null ? (model.metrics.f1 * 100).toFixed(1) + '%' : '-' }}</td>
              <td class="text-caption">{{ formatDate(model.created_at) }}</td>
              <td>
                <v-btn icon size="x-small" variant="text" color="error" @click.stop="confirmDeleteModel(model)">
                  <v-icon size="small">mdi-delete</v-icon>
                </v-btn>
              </td>
            </tr>
          </tbody>
        </v-table>
        <div v-else-if="modelTableTab === 'classification'" class="text-center text-medium-emphasis py-4">
          No classification/anomaly models saved yet.
        </div>

        <!-- Delete confirmation dialog -->
        <v-dialog v-model="showDeleteModelDialog" max-width="400">
          <v-card>
            <v-card-title>Delete Model</v-card-title>
            <v-card-text>
              Are you sure you want to delete <strong>{{ deleteTargetModel?.name }}</strong>?
              This action cannot be undone.
            </v-card-text>
            <v-card-actions>
              <v-spacer />
              <v-btn @click="showDeleteModelDialog = false">Cancel</v-btn>
              <v-btn color="error" variant="flat" :loading="deletingModel" @click="deleteModel">Delete</v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>

        <!-- TI MCU Package Info Dialog -->
        <v-dialog v-model="showTiMcuInfo" max-width="700" scrollable>
          <v-card>
            <v-card-title class="d-flex align-center">
              <v-icon start color="pink">mdi-chip</v-icon>
              TI MCU Deployment Package
              <v-spacer />
              <v-btn icon size="small" variant="text" @click="showTiMcuInfo = false">
                <v-icon>mdi-close</v-icon>
              </v-btn>
            </v-card-title>

            <v-card-text>
              <!-- Workflow Diagram -->
              <div class="ti-workflow mb-4">
                <div class="text-subtitle-2 font-weight-bold mb-2">Deployment Workflow</div>
                <div class="ti-flow-diagram">
                  <div class="ti-flow-step">
                    <div class="ti-flow-icon" style="background: #a78bfa20; border-color: #a78bfa;">
                      <v-icon size="20" color="purple">mdi-brain</v-icon>
                    </div>
                    <div class="ti-flow-label">Train Model</div>
                    <div class="ti-flow-sub">CiRA ME</div>
                  </div>
                  <div class="ti-flow-arrow">
                    <v-icon size="16" color="grey">mdi-arrow-right</v-icon>
                  </div>
                  <div class="ti-flow-step">
                    <div class="ti-flow-icon" style="background: #60a5fa20; border-color: #60a5fa;">
                      <v-icon size="20" color="blue">mdi-download</v-icon>
                    </div>
                    <div class="ti-flow-label">Download Package</div>
                    <div class="ti-flow-sub">.zip file</div>
                  </div>
                  <div class="ti-flow-arrow">
                    <v-icon size="16" color="grey">mdi-arrow-right</v-icon>
                  </div>
                  <div class="ti-flow-step">
                    <div class="ti-flow-icon" style="background: #fbbf2420; border-color: #fbbf24;">
                      <v-icon size="20" color="amber">mdi-application-braces</v-icon>
                    </div>
                    <div class="ti-flow-label">Open in CCS</div>
                    <div class="ti-flow-sub">Code Composer Studio</div>
                  </div>
                  <div class="ti-flow-arrow">
                    <v-icon size="16" color="grey">mdi-arrow-right</v-icon>
                  </div>
                  <div class="ti-flow-step">
                    <div class="ti-flow-icon" style="background: #34d39920; border-color: #34d399;">
                      <v-icon size="20" color="success">mdi-flash</v-icon>
                    </div>
                    <div class="ti-flow-label">Build & Flash</div>
                    <div class="ti-flow-sub">TMS320 MCU</div>
                  </div>
                </div>
              </div>

              <v-divider class="mb-4" />

              <!-- Package Contents -->
              <div class="text-subtitle-2 font-weight-bold mb-2">Package Contents</div>
              <v-table density="compact" class="mb-4">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Description</th>
                    <th>Model Type</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #a78bfa;">model_weights.h</td>
                    <td class="text-caption">Neural network weights as C arrays</td>
                    <td><v-chip size="x-small" color="orange" variant="tonal">TI NN</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #a78bfa;">model_config.h</td>
                    <td class="text-caption">Input/output shape definitions</td>
                    <td><v-chip size="x-small" color="orange" variant="tonal">TI NN</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #a78bfa;">model.onnx</td>
                    <td class="text-caption">Trained model for TI NN Compiler (NPU devices)</td>
                    <td><v-chip size="x-small" color="orange" variant="tonal">TI NN</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #60a5fa;">model.h</td>
                    <td class="text-caption">Complete model as C if/else tree (emlearn)</td>
                    <td><v-chip size="x-small" color="blue" variant="tonal">ML</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #94a3b8;">cira_main.c</td>
                    <td class="text-caption">Firmware template with SCI UART + inference loop</td>
                    <td><v-chip size="x-small" variant="tonal">All</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #94a3b8;">cira_serial_test.py</td>
                    <td class="text-caption">Python tool for testing via serial port</td>
                    <td><v-chip size="x-small" variant="tonal">All</v-chip></td>
                  </tr>
                  <tr>
                    <td class="font-weight-medium" style="font-family: monospace; font-size: 12px; color: #94a3b8;">README.txt</td>
                    <td class="text-caption">Step-by-step CCS integration guide</td>
                    <td><v-chip size="x-small" variant="tonal">All</v-chip></td>
                  </tr>
                </tbody>
              </v-table>

              <v-divider class="mb-4" />

              <!-- Step by step -->
              <div class="text-subtitle-2 font-weight-bold mb-2">CCS Integration Steps</div>
              <div class="ti-steps">
                <div class="ti-step">
                  <div class="ti-step-num">1</div>
                  <div>
                    <div class="font-weight-medium">Create CCS Project</div>
                    <div class="text-caption text-medium-emphasis">File > New > CCS Project > Select your TMS320 device</div>
                  </div>
                </div>
                <div class="ti-step">
                  <div class="ti-step-num">2</div>
                  <div>
                    <div class="font-weight-medium">Add Model Files</div>
                    <div class="text-caption text-medium-emphasis">Copy model/ folder contents into your CCS project</div>
                  </div>
                </div>
                <div class="ti-step">
                  <div class="ti-step-num">3</div>
                  <div>
                    <div class="font-weight-medium">Add Firmware Template</div>
                    <div class="text-caption text-medium-emphasis">Add src/cira_main.c or write your own main()</div>
                  </div>
                </div>
                <div class="ti-step">
                  <div class="ti-step-num">4</div>
                  <div>
                    <div class="font-weight-medium">Build & Flash</div>
                    <div class="text-caption text-medium-emphasis">Click Debug button > program flashes to MCU > Run</div>
                  </div>
                </div>
                <div class="ti-step">
                  <div class="ti-step-num">5</div>
                  <div>
                    <div class="font-weight-medium">Test via Serial</div>
                    <div class="text-caption text-medium-emphasis">python cira_serial_test.py --port COM5 --interactive</div>
                  </div>
                </div>
              </div>

              <v-divider class="my-4" />

              <!-- NPU note -->
              <v-alert variant="tonal" color="warning" density="compact" class="mb-2">
                <div class="text-caption">
                  <strong>NPU Models:</strong> For devices with NPU (F28P55x, MSPM0+), use TI NN Compiler in CCS
                  to compile the .onnx file for NPU execution. Command: <code>ti_nn_compiler --model model.onnx --target F28P55</code>
                </div>
              </v-alert>
              <v-alert variant="tonal" color="info" density="compact">
                <div class="text-caption">
                  <strong>CPU Models:</strong> For devices without NPU (F28379D, F280049C), the model_weights.h
                  contains all parameters. Implement the forward pass in C or use the emlearn model.h directly.
                </div>
              </v-alert>
            </v-card-text>
          </v-card>
        </v-dialog>
      </div>

      <!-- Selected Model Metrics Summary Card -->
      <v-card v-if="selectedModel" variant="outlined" class="mt-4 pa-4">
        <div class="d-flex align-center mb-3">
          <v-icon color="success" class="mr-2">mdi-check-circle</v-icon>
          <v-chip size="x-small" :color="getModelTypeInfo(selectedModel).color" variant="flat" class="mr-2 font-weight-bold" style="font-size:9px; letter-spacing:0.5px">
            {{ getModelTypeInfo(selectedModel).label }}
          </v-chip>
          <h4 class="text-subtitle-2 font-weight-bold">
            {{ selectedModel.name }}
            <span class="text-medium-emphasis font-weight-regular ml-1">— {{ selectedModel.algorithm }} ({{ selectedModel.mode }})</span>
          </h4>
          <v-spacer />
          <v-btn
            v-if="modelSource === 'saved' && !isTiNnModel"
            color="info"
            variant="tonal"
            size="small"
            @click="openEvalDialog"
          >
            <v-icon start size="small">mdi-test-tube</v-icon>
            Test with New Data
          </v-btn>
          <v-chip v-if="modelSource === 'saved' && isTiNnModel" size="small" color="warning" variant="tonal">
            <v-icon start size="small">mdi-chip</v-icon>
            TI NN — deploy to MCU for testing
          </v-chip>
          <v-btn
            v-if="modelSource === 'saved' && selectedModel?.pipeline_config?.normalization"
            color="success"
            variant="tonal"
            size="small"
            class="ml-2"
            :loading="downloadingPackage"
            @click="downloadPackage"
          >
            <v-icon start size="small">mdi-package-down</v-icon>
            Download Package
          </v-btn>
        </div>

        <!-- Capabilities row -->
        <div class="d-flex align-center gap-2 mb-3">
          <span class="text-caption text-medium-emphasis">Capabilities:</span>
          <v-chip size="x-small" :color="getModelTypeInfo(selectedModel).canApi ? 'success' : 'grey'" variant="tonal">
            <v-icon start size="10">{{ getModelTypeInfo(selectedModel).canApi ? 'mdi-check' : 'mdi-close' }}</v-icon>
            API / ME-LAB
          </v-chip>
          <v-chip size="x-small" :color="getModelTypeInfo(selectedModel).canMcu ? 'success' : 'grey'" variant="tonal">
            <v-icon start size="10">{{ getModelTypeInfo(selectedModel).canMcu ? 'mdi-check' : 'mdi-close' }}</v-icon>
            TI MCU
          </v-chip>
          <v-chip size="x-small" color="success" variant="tonal">
            <v-icon start size="10">mdi-check</v-icon>
            SSH Deploy
          </v-chip>
          <v-chip size="x-small" :color="!getModelTypeInfo(selectedModel).canApi && getModelTypeInfo(selectedModel).label === 'TI NN' ? 'grey' : 'success'" variant="tonal">
            <v-icon start size="10">{{ !getModelTypeInfo(selectedModel).canApi && getModelTypeInfo(selectedModel).label === 'TI NN' ? 'mdi-close' : 'mdi-check' }}</v-icon>
            Test with Data
          </v-chip>
        </div>

        <!-- Regression metrics -->
        <v-row v-if="selectedModel.mode === 'regression'" dense>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="primary">
              <div class="text-caption text-medium-emphasis">R² Score</div>
              <div class="text-h6">{{ selectedModel.metrics?.r2 != null ? selectedModel.metrics.r2.toFixed(4) : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="info">
              <div class="text-caption text-medium-emphasis">RMSE</div>
              <div class="text-h6">{{ selectedModel.metrics?.rmse != null ? selectedModel.metrics.rmse.toFixed(4) : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="warning">
              <div class="text-caption text-medium-emphasis">MAE</div>
              <div class="text-h6">{{ selectedModel.metrics?.mae != null ? selectedModel.metrics.mae.toFixed(4) : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="success">
              <div class="text-caption text-medium-emphasis">MAPE</div>
              <div class="text-h6">{{ selectedModel.metrics?.mape != null ? (selectedModel.metrics.mape * 100).toFixed(1) + '%' : '-' }}</div>
            </v-card>
          </v-col>
        </v-row>
        <!-- Classification / Anomaly metrics -->
        <v-row v-else dense>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="primary">
              <div class="text-caption text-medium-emphasis">Accuracy</div>
              <div class="text-h6">{{ selectedModel.metrics?.accuracy != null ? (selectedModel.metrics.accuracy * 100).toFixed(1) + '%' : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="info">
              <div class="text-caption text-medium-emphasis">Precision</div>
              <div class="text-h6">{{ selectedModel.metrics?.precision != null ? (selectedModel.metrics.precision * 100).toFixed(1) + '%' : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="warning">
              <div class="text-caption text-medium-emphasis">Recall</div>
              <div class="text-h6">{{ selectedModel.metrics?.recall != null ? (selectedModel.metrics.recall * 100).toFixed(1) + '%' : '-' }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center" color="success">
              <div class="text-caption text-medium-emphasis">F1 Score</div>
              <div class="text-h6">{{ selectedModel.metrics?.f1 != null ? (selectedModel.metrics.f1 * 100).toFixed(1) + '%' : '-' }}</div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Dataset info if available -->
        <div v-if="selectedModel.dataset_info" class="mt-3 text-caption text-medium-emphasis">
          <v-icon size="x-small" class="mr-1">mdi-database</v-icon>
          Trained on: {{ selectedModel.dataset_info.filename || selectedModel.dataset_info.name || 'Unknown dataset' }}
        </div>
      </v-card>
    </v-card>

    <v-row>
      <!-- Step 1: Export Format -->
      <v-col cols="12" md="5">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">
            <v-icon start size="small">mdi-package-variant</v-icon>
            Step 1: Export Format
          </h3>

          <v-radio-group v-model="exportFormat">
            <v-radio value="onnx">
              <template #label>
                <div>
                  <div class="font-weight-medium">ONNX Runtime</div>
                  <div class="text-caption text-medium-emphasis">
                    Cross-platform, optimized inference
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="pickle">
              <template #label>
                <div>
                  <div class="font-weight-medium">Scikit-learn (Pickle)</div>
                  <div class="text-caption text-medium-emphasis">
                    Native Python format
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="joblib">
              <template #label>
                <div>
                  <div class="font-weight-medium">Joblib</div>
                  <div class="text-caption text-medium-emphasis">
                    Efficient for large arrays
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio
              v-if="modelSource === 'saved' && selectedModel?.pipeline_config?.normalization"
              value="cira_claw"
            >
              <template #label>
                <div>
                  <div class="font-weight-medium" style="color: #FF5722;">CiRA CLAW Package</div>
                  <div class="text-caption text-medium-emphasis">
                    ONNX + manifest for CiRA CLAW C runtime
                  </div>
                </div>
              </template>
            </v-radio>
            <v-radio value="ti_mcu">
              <template #label>
                <div class="d-flex align-center">
                  <div>
                    <div class="font-weight-medium" style="color: #E91E63;">TI MCU Package</div>
                    <div class="text-caption text-medium-emphasis">
                      CCS-ready project files for TMS320 deployment
                    </div>
                  </div>
                  <v-btn
                    icon
                    size="x-small"
                    variant="text"
                    color="info"
                    class="ml-2"
                    @click.stop="showTiMcuInfo = true"
                    title="View package contents & instructions"
                  >
                    <v-icon size="small">mdi-information-outline</v-icon>
                  </v-btn>
                </div>
              </template>
            </v-radio>
          </v-radio-group>

        </v-card>
      </v-col>

      <!-- Step 2: Deploy Target -->
      <v-col cols="12" md="7">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">
            <v-icon start size="small">mdi-rocket-launch</v-icon>
            Step 2: Deploy Target
          </h3>

          <v-radio-group v-model="deployTarget" hide-details class="mb-4">
            <v-radio value="download">
              <template #label>
                <div>
                  <div class="font-weight-medium">Download to PC</div>
                  <div class="text-caption text-medium-emphasis">Download model package as zip</div>
                </div>
              </template>
            </v-radio>
            <v-radio value="ssh" :disabled="exportFormat === 'ti_mcu'">
              <template #label>
                <div>
                  <div class="font-weight-medium" :class="{ 'text-medium-emphasis': exportFormat === 'ti_mcu' }">Deploy via SSH</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ exportFormat === 'ti_mcu' ? 'Not available — MCU requires JTAG/CCS flashing' : 'Transfer and run on remote device' }}
                  </div>
                </div>
              </template>
            </v-radio>
          </v-radio-group>

          <template v-if="deployTarget === 'ssh'">
            <v-divider class="mb-4" />

            <div class="d-flex ga-3 mb-4">
              <v-select
                v-model="targetDevice"
                :items="[
                  { title: 'Jetson Nano', value: 'jetson_nano' },
                  { title: 'Jetson Xavier NX', value: 'jetson_xavier' },
                  { title: 'Raspberry Pi 4', value: 'raspberry_pi' },
                  { title: 'Horizon RDK X5', value: 'rdk_x5' },
                  { title: 'Ubuntu x86 PC', value: 'ubuntu_x86' },
                  { title: 'Custom SSH', value: 'custom_ssh' },
                ]"
                label="Target Device"
                variant="outlined"
                density="compact"
                style="max-width: 200px"
                hide-details
              />
              <v-btn-toggle v-model="deployMode" mandatory density="compact">
                <v-btn value="docker" size="small">
                  <v-icon start size="small">mdi-docker</v-icon>
                  Docker
                </v-btn>
                <v-btn value="files" size="small">
                  <v-icon start size="small">mdi-file-code</v-icon>
                  Files
                </v-btn>
              </v-btn-toggle>
            </div>

          <!-- Saved Devices -->
          <div v-if="savedDevices.length > 0" class="mb-4">
            <div class="text-caption text-medium-emphasis mb-2">
              <v-icon size="x-small" class="mr-1">mdi-bookmark-multiple</v-icon>
              Saved Devices
            </div>
            <div class="d-flex flex-wrap ga-2">
              <v-chip
                v-for="dev in savedDevices"
                :key="dev.id"
                :color="sshConfig.host === dev.host && sshConfig.username === dev.username ? 'primary' : 'default'"
                variant="tonal"
                size="small"
                style="cursor: pointer"
                @click="loadDevice(dev)"
              >
                <template v-if="renamingDeviceId === dev.id">
                  <input
                    :ref="el => { if (el) renameInputs[dev.id] = el as HTMLInputElement }"
                    v-model="dev.name"
                    class="rename-input"
                    @blur="finishRename"
                    @keyup.enter="finishRename"
                    @keyup.escape="finishRename"
                    @click.stop
                  />
                </template>
                <template v-else>
                  <v-icon start size="x-small">mdi-server-network</v-icon>
                  {{ dev.name }}
                </template>
                <v-icon
                  end size="x-small"
                  class="ml-1"
                  style="opacity: 0.6"
                  @click.stop="startRename(dev)"
                >mdi-pencil</v-icon>
                <v-icon
                  end size="x-small"
                  class="ml-1"
                  style="opacity: 0.6"
                  @click.stop="deleteDevice(dev.id)"
                >mdi-close</v-icon>
              </v-chip>
            </div>
            <v-divider class="mt-3 mb-3" />
          </div>

          <v-text-field
            v-model="sshConfig.host"
            label="Host / IP Address"
            prepend-inner-icon="mdi-server"
            placeholder="192.168.1.100"
          />

          <v-text-field
            v-model="sshConfig.username"
            label="Username"
            prepend-inner-icon="mdi-account"
            placeholder="cira"
          />

          <v-text-field
            v-model="sshConfig.password"
            label="Password"
            prepend-inner-icon="mdi-lock"
            :type="showPassword ? 'text' : 'password'"
            :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
            @click:append-inner="showPassword = !showPassword"
          />

          <v-text-field
            v-model.number="sshConfig.port"
            label="Port"
            prepend-inner-icon="mdi-ethernet"
            type="number"
          />

          <v-text-field
            v-model="sshConfig.remote_path"
            label="Remote Path"
            prepend-inner-icon="mdi-folder"
            placeholder="~/cira_models"
          />

          <v-btn
            color="info"
            variant="outlined"
            block
            :loading="testingConnection"
            @click="testConnection"
          >
            <v-icon start>mdi-connection</v-icon>
            Test Connection
          </v-btn>

          <v-alert
            v-if="connectionStatus"
            :type="connectionStatus.status === 'connected' ? 'success' : 'error'"
            variant="tonal"
            class="mt-4"
          >
            <div class="d-flex align-center justify-space-between flex-wrap ga-2">
              <div>
                {{ connectionStatus.message }}
                <div v-if="connectionStatus.system_info" class="text-caption mt-1 opacity-80">
                  {{ connectionStatus.system_info }}
                </div>
              </div>
              <v-btn
                v-if="connectionStatus.status === 'connected' && !isCurrentDeviceSaved"
                size="x-small"
                color="success"
                variant="tonal"
                @click="saveCurrentDevice"
              >
                <v-icon start size="x-small">mdi-bookmark-plus</v-icon>
                Save device
              </v-btn>
              <v-chip
                v-else-if="connectionStatus.status === 'connected' && isCurrentDeviceSaved"
                size="x-small" color="success" variant="tonal"
              >
                <v-icon start size="x-small">mdi-bookmark-check</v-icon>
                Saved
              </v-chip>
            </div>
          </v-alert>

          <!-- Detected device info panel -->
          <v-card
            v-if="connectionStatus?.status === 'connected'"
            variant="outlined"
            class="mt-3 pa-3"
          >
            <div class="text-caption font-weight-bold mb-2">Detected Device Info</div>
            <div class="d-flex flex-wrap ga-2">
              <v-chip
                v-if="connectionStatus.is_jetson"
                size="small" color="success" variant="tonal"
              >
                <v-icon start size="x-small">mdi-chip</v-icon>
                Jetson
                {{ connectionStatus.jetpack_version ? `JetPack ${connectionStatus.jetpack_version}` : '' }}
                {{ connectionStatus.l4t_revision ? `(${connectionStatus.l4t_revision})` : '' }}
              </v-chip>
              <v-chip v-else size="small" variant="tonal">
                <v-icon start size="x-small">mdi-desktop-tower</v-icon>
                Non-Jetson
              </v-chip>
              <v-chip size="small" variant="tonal">
                <v-icon start size="x-small">mdi-language-python</v-icon>
                {{ connectionStatus.python_version || 'Python unknown' }}
              </v-chip>
              <v-chip
                v-if="connectionStatus.cuda_version"
                size="small" color="info" variant="tonal"
              >
                <v-icon start size="x-small">mdi-gpu</v-icon>
                CUDA {{ connectionStatus.cuda_version }}
              </v-chip>
              <v-chip
                :color="connectionStatus.nvidia_runtime ? 'success' : 'default'"
                size="small" variant="tonal"
              >
                <v-icon start size="x-small">mdi-docker</v-icon>
                nvidia runtime: {{ connectionStatus.nvidia_runtime ? 'yes' : 'no' }}
              </v-chip>
              <v-chip size="small" variant="tonal">
                <v-icon start size="x-small">mdi-harddisk</v-icon>
                Free: {{ connectionStatus.disk_free }}
              </v-chip>
              <v-chip size="small" variant="tonal">
                <v-icon start size="x-small">mdi-memory</v-icon>
                RAM free: {{ connectionStatus.ram_free }}
              </v-chip>
            </div>

            <!-- GPU toggle (only visible in docker mode) -->
            <div v-if="deployMode === 'docker'" class="mt-3">
              <v-switch
                v-model="enableGpu"
                :label="enableGpu ? 'GPU inference enabled' : 'CPU inference (GPU disabled)'"
                :color="enableGpu ? 'success' : 'default'"
                density="compact"
                hide-details
                :disabled="!connectionStatus.nvidia_runtime"
              />
              <div
                v-if="!connectionStatus.nvidia_runtime"
                class="text-caption text-medium-emphasis mt-1"
              >
                nvidia runtime not detected — GPU disabled
              </div>
            </div>
          </v-card>
          </template>
        </v-card>

        <!-- Deployment Progress -->
        <v-card v-if="deploymentResult" class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Deployment Progress</h3>

          <div class="deployment-progress">
            <div
              v-for="step in deploymentResult.steps"
              :key="step.step"
              class="step"
            >
              <div class="step-icon" :class="step.status">
                <v-icon size="14" v-if="step.status === 'completed'">mdi-check</v-icon>
                <v-icon size="14" v-else-if="step.status === 'failed'">mdi-close</v-icon>
                <v-progress-circular
                  v-else-if="step.status === 'in_progress'"
                  size="14"
                  width="2"
                  indeterminate
                />
              </div>
              <span class="text-capitalize">{{ step.step.replace(/_/g, ' ') }}</span>
            </div>
          </div>

          <!-- Container started successfully -->
          <v-alert
            v-if="deploymentResult.status === 'completed' && deploymentResult.container_started !== false"
            type="success"
            variant="tonal"
            class="mt-4"
          >
            <div class="d-flex align-center justify-space-between flex-wrap ga-2">
              <div>
                <strong>Deployment successful!</strong>
                <div class="text-caption mt-1">
                  Path: {{ deploymentResult.remote_path }}
                </div>
                <div v-if="deploymentResult.container_name" class="text-caption">
                  Container: <code>{{ deploymentResult.container_name }}</code>
                </div>
              </div>
              <!-- Build log button always available for Docker mode -->
              <v-btn
                v-if="deploymentResult.deploy_mode === 'docker' && deploymentResult.build_log_file"
                size="small"
                variant="outlined"
                color="success"
                :loading="fetchingLog"
                @click="buildLogInterval ? stopBuildLogPolling() : startBuildLogPolling()"
              >
                <v-icon start size="small">
                  {{ buildLogInterval ? 'mdi-stop' : 'mdi-console' }}
                </v-icon>
                {{ buildLogInterval ? 'Stop watching' : 'View build log' }}
              </v-btn>
            </div>
          </v-alert>

          <!-- Build in background — container not yet up -->
          <v-alert
            v-if="deploymentResult.status === 'completed' && deploymentResult.container_started === false"
            type="warning"
            variant="tonal"
            class="mt-4"
          >
            <div class="d-flex align-center justify-space-between flex-wrap ga-2">
              <div>
                <strong>Files transferred — Docker build in progress</strong>
                <div class="text-body-2 mt-1">
                  The container image is building on the remote device.
                  First-time builds take <strong>~10 minutes</strong> (base image pull + packages).
                </div>
              </div>
              <v-btn
                size="small"
                variant="flat"
                color="warning"
                :loading="fetchingLog"
                @click="buildLogInterval ? stopBuildLogPolling() : startBuildLogPolling()"
              >
                <v-icon start size="small">
                  {{ buildLogInterval ? 'mdi-stop' : 'mdi-console' }}
                </v-icon>
                {{ buildLogInterval ? 'Stop watching' : 'Watch build log' }}
              </v-btn>
            </div>
          </v-alert>

          <!-- Live build log panel (shown for both states) -->
          <v-card
            v-if="buildLog !== null && deploymentResult.deploy_mode === 'docker'"
            variant="outlined"
            class="mt-3"
          >
            <v-card-title class="text-subtitle-2 d-flex align-center pa-3 pb-0">
              <v-icon size="small" class="mr-1" color="warning">mdi-console-line</v-icon>
              Remote build log
              <v-spacer />
              <v-chip
                v-if="buildLogInterval"
                size="x-small"
                color="warning"
                variant="tonal"
                class="mr-2"
              >
                <v-icon start size="x-small">mdi-refresh</v-icon>
                auto-refresh 15s
              </v-chip>
              <v-btn
                icon size="x-small"
                variant="text"
                :loading="fetchingLog"
                @click="fetchBuildLog"
              >
                <v-icon size="small">mdi-refresh</v-icon>
              </v-btn>
            </v-card-title>
            <v-card-text class="pa-3 pt-2">
              <pre
                style="white-space: pre-wrap; font-family: monospace; font-size: 0.72rem;
                       max-height: 320px; overflow-y: auto; background: #111;
                       color: #d4d4d4; padding: 12px; border-radius: 6px;"
              >{{ buildLog || '(no output yet — build may just be starting)' }}</pre>
            </v-card-text>
          </v-card>
        </v-card>

        <!-- Post-Deployment: Validate & Run Inference -->
        <v-card v-if="deploymentResult?.status === 'completed'" class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-3">
            <v-icon start color="info">mdi-check-network</v-icon>
            Validate & Run Inference
          </h3>

          <!-- Step 5: Check Remote Files -->
          <div class="text-subtitle-2 font-weight-medium mb-2">Step 5 — Verify Remote Files</div>
          <v-btn
            color="info" variant="outlined" size="small"
            :loading="checkingFiles"
            @click="checkRemoteFiles"
          >
            <v-icon start size="small">mdi-folder-search</v-icon>
            Check Remote Files
          </v-btn>

          <v-card v-if="remoteFiles" variant="tonal" class="pa-3 mt-3">
            <div class="text-caption font-weight-bold mb-1">
              <v-icon size="x-small" class="mr-1">mdi-folder</v-icon>
              {{ deploymentResult.remote_path }}
            </div>
            <pre class="text-caption" style="white-space: pre-wrap; font-family: monospace;">{{ remoteFiles.files }}</pre>
            <template v-if="remoteFiles.containers">
              <v-divider class="my-2" />
              <div class="text-caption font-weight-bold mb-1">
                <v-icon size="x-small" class="mr-1">mdi-docker</v-icon>
                Docker Containers
              </div>
              <pre class="text-caption" style="white-space: pre-wrap; font-family: monospace;">{{ remoteFiles.containers }}</pre>
            </template>
          </v-card>

          <v-divider class="my-4" />

          <!-- Step 6: Run Inference -->
          <div class="text-subtitle-2 font-weight-medium mb-2">Step 6 — Run Inference on Remote</div>
          <p class="text-caption text-medium-emphasis mb-3">
            Upload a CSV with the same sensor columns as training data.
            It will be transferred to the remote device and inference will run there.
          </p>

          <v-file-input
            v-model="inferenceFile"
            label="Upload CSV for inference"
            accept=".csv"
            prepend-icon="mdi-file-delimited"
            show-size density="compact"
            class="mb-2"
          />

          <v-btn
            color="primary" variant="flat" size="small"
            :disabled="!inferenceFile"
            :loading="runningInference"
            @click="runRemoteInference"
          >
            <v-icon start size="small">mdi-play-circle</v-icon>
            Run Inference on Remote
          </v-btn>

          <!-- Inference Output -->
          <v-card v-if="inferenceOutput" variant="outlined" class="pa-3 mt-3">
            <div class="d-flex align-center mb-3">
              <v-icon color="success" class="mr-2">mdi-check-circle</v-icon>
              <span class="text-subtitle-2 font-weight-bold">
                Inference Results — {{ inferenceOutput.csv }}
              </span>
            </div>

            <!-- Metric chips -->
            <v-row dense class="mb-3">
              <v-col cols="6" sm="3">
                <v-card variant="tonal" color="primary" class="pa-2 text-center">
                  <div class="text-caption text-medium-emphasis">Windows</div>
                  <div class="text-h6">{{ inferenceOutput.num_windows ?? '—' }}</div>
                </v-card>
              </v-col>
              <v-col cols="6" sm="3">
                <v-card variant="tonal" color="info" class="pa-2 text-center">
                  <div class="text-caption text-medium-emphasis">Features</div>
                  <div class="text-h6">{{ inferenceOutput.num_features ?? '—' }}</div>
                </v-card>
              </v-col>
              <v-col cols="6" sm="3">
                <v-card variant="tonal" color="success" class="pa-2 text-center">
                  <div class="text-caption text-medium-emphasis">Avg Confidence</div>
                  <div class="text-h6">
                    {{ inferenceOutput.avg_confidence != null ? inferenceOutput.avg_confidence + '%' : '—' }}
                  </div>
                </v-card>
              </v-col>
              <v-col cols="6" sm="3">
                <v-card variant="tonal" color="warning" class="pa-2 text-center">
                  <div class="text-caption text-medium-emphasis">Classes</div>
                  <div class="text-h6">
                    {{ inferenceOutput.prediction_distribution ? Object.keys(inferenceOutput.prediction_distribution).length : '—' }}
                  </div>
                </v-card>
              </v-col>
            </v-row>

            <!-- Prediction distribution chips -->
            <div v-if="inferenceOutput.prediction_distribution" class="mb-3">
              <div class="text-caption text-medium-emphasis mb-1">Prediction Distribution</div>
              <div class="d-flex flex-wrap ga-2">
                <v-chip
                  v-for="(count, label) in inferenceOutput.prediction_distribution"
                  :key="label"
                  size="small"
                  variant="tonal"
                  color="primary"
                >
                  {{ label }}: <strong class="ml-1">{{ count }}</strong>
                </v-chip>
              </div>
            </div>

            <!-- Raw console (collapsed by default) -->
            <v-expansion-panels variant="accordion" class="mt-2">
              <v-expansion-panel>
                <v-expansion-panel-title class="text-caption">
                  <v-icon size="x-small" class="mr-1">mdi-console</v-icon>
                  Raw console output
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <pre class="text-caption" style="white-space: pre-wrap; font-family: monospace; max-height: 250px; overflow-y: auto;">{{ inferenceOutput.output }}</pre>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
          </v-card>
          <v-alert v-if="inferenceError" type="error" variant="tonal" class="mt-3" density="compact">
            {{ inferenceError }}
          </v-alert>
        </v-card>
      </v-col>
    </v-row>

    <!-- Actions -->
    <div class="d-flex justify-space-between mt-6">
      <v-btn
        variant="outlined"
        size="large"
        @click="router.push({ name: 'pipeline-training' })"
      >
        <v-icon start>mdi-arrow-left</v-icon>
        Back
      </v-btn>

      <v-btn
        v-if="deployTarget === 'download'"
        color="primary"
        size="large"
        :loading="exporting"
        :disabled="!hasModelSelected"
        @click="exportOnly"
      >
        <v-icon start>mdi-download</v-icon>
        Download Package
      </v-btn>
      <v-btn
        v-else
        color="primary"
        size="large"
        :loading="deploying"
        :disabled="!canDeploy"
        @click="deploy"
      >
        <v-icon start>mdi-rocket-launch</v-icon>
        Deploy to Device
      </v-btn>
    </div>

    <!-- Test with New Data Dialog -->
    <v-dialog v-model="showEvalDialog" max-width="700" scrollable>
      <v-card>
        <v-card-title>
          <v-icon start size="small">mdi-test-tube</v-icon>
          Test "{{ evalModel?.name }}" with New Data
        </v-card-title>
        <v-card-text style="max-height: 70vh;">

        <!-- No pipeline config warning -->
        <v-alert
          v-if="!evalModel?.pipeline_config?.normalization"
          type="warning" variant="tonal" density="compact" class="mb-4"
        >
          This model was saved without full pipeline config.
          Re-save the model from a training session to enable raw CSV evaluation.
        </v-alert>

        <!-- Pipeline info -->
        <template v-else>
          <v-alert type="info" variant="tonal" density="compact" class="mb-4">
            Upload a raw CSV file with the same sensor columns as the training data.
            The saved pipeline will automatically apply windowing, normalization,
            {{ evalModel.pipeline_config.training_approach === 'dl'
               ? 'and TimesNet inference'
               : 'feature extraction, and model prediction' }}.
            If the CSV has a label column, metrics will be computed.
          </v-alert>

          <div class="d-flex flex-wrap ga-2 mb-4">
            <v-chip size="small" color="primary" variant="tonal">
              {{ evalModel.pipeline_config.training_approach === 'dl' ? 'TimesNet' : 'ML' }}
            </v-chip>
            <v-chip size="small" variant="tonal">
              Window: {{ evalModel.pipeline_config.windowing?.window_size || '?' }}
            </v-chip>
            <v-chip size="small" variant="tonal">
              Channels: {{ evalModel.pipeline_config.normalization?.sensor_columns?.length || '?' }}
            </v-chip>
            <v-chip v-if="evalModel.pipeline_config.feature_selection" size="small" color="success" variant="tonal">
              Selected: {{ evalModel.pipeline_config.feature_selection.num_selected }} features
            </v-chip>
            <v-chip v-else-if="evalModel.pipeline_config.feature_extraction" size="small" variant="tonal">
              {{ evalModel.pipeline_config.feature_extraction.num_features }} features
            </v-chip>
          </div>
        </template>

        <!-- File upload -->
        <v-file-input
          v-model="evalFile"
          label="Upload CSV file"
          accept=".csv"
          prepend-icon="mdi-file-delimited"
          :disabled="!evalModel?.pipeline_config?.normalization"
          show-size
          density="compact"
        />

        <!-- Target column for regression -->
        <v-select
          v-if="evalModel?.mode === 'regression' && evalCsvColumns.length > 0"
          v-model="evalTargetColumn"
          :items="evalCsvColumns"
          label="Target column in CSV (for R²/RMSE evaluation)"
          density="compact"
          hint="Select the column to compare predictions against"
          persistent-hint
          clearable
          class="mb-2"
        />

        <div class="d-flex justify-end ga-2">
          <v-btn variant="text" @click="showEvalDialog = false">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :disabled="!evalFile || !evalModel?.pipeline_config?.normalization"
            :loading="evaluating"
            @click="runRawEvaluation"
          >
            Evaluate
          </v-btn>
        </div>

        <!-- Evaluation Result -->
        <v-card v-if="evalResult" variant="outlined" class="pa-3 mt-4">
          <h4 class="text-subtitle-2 font-weight-bold mb-2">
            {{ evalResult.has_labels ? 'Results: Original vs New Data' : 'Prediction Results' }}
          </h4>

          <!-- Metrics comparison (if labels found) -->
          <v-table v-if="evalResult.comparison" density="compact">
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
                <td class="font-weight-medium text-capitalize">{{ c.metric }}</td>
                <template v-if="evalModel?.mode === 'regression'">
                  <td class="text-center">{{ c.original != null ? c.original.toFixed(4) : '-' }}</td>
                  <td class="text-center">{{ c.new_data != null ? c.new_data.toFixed(4) : '-' }}</td>
                  <td class="text-center" :class="c.diff != null ? (['r2'].includes(c.metric) ? (c.diff >= 0 ? 'text-success' : 'text-error') : (c.diff <= 0 ? 'text-success' : 'text-error')) : ''">
                    {{ c.diff != null ? (c.diff > 0 ? '+' : '') + c.diff.toFixed(4) : '-' }}
                  </td>
                </template>
                <template v-else>
                  <td class="text-center">{{ c.original != null ? (c.original * 100).toFixed(1) + '%' : '-' }}</td>
                  <td class="text-center">{{ c.new_data != null ? (c.new_data * 100).toFixed(1) + '%' : '-' }}</td>
                  <td class="text-center" :class="c.diff != null ? (c.diff >= 0 ? 'text-success' : 'text-error') : ''">
                    {{ c.diff != null ? (c.diff > 0 ? '+' : '') + (c.diff * 100).toFixed(1) + '%' : '-' }}
                  </td>
                </template>
              </tr>
            </tbody>
          </v-table>

          <!-- Prediction distribution -->
          <div class="mt-3">
            <div class="text-body-2 mb-1">
              <strong>{{ evalResult.num_windows }}</strong> windows processed
            </div>
            <!-- Regression: show prediction stats -->
            <template v-if="selectedModel?.mode === 'regression' && evalResult.predictions?.length">
              <div class="d-flex flex-wrap ga-2">
                <v-chip size="small" variant="tonal" color="info">
                  Mean: {{ (evalResult.predictions.reduce((a: number, b: number) => a + b, 0) / evalResult.predictions.length).toFixed(4) }}
                </v-chip>
                <v-chip size="small" variant="tonal" color="info">
                  Min: {{ Math.min(...evalResult.predictions).toFixed(4) }}
                </v-chip>
                <v-chip size="small" variant="tonal" color="info">
                  Max: {{ Math.max(...evalResult.predictions).toFixed(4) }}
                </v-chip>
              </div>
            </template>
            <!-- Classification: show class distribution -->
            <div v-else-if="evalResult.prediction_distribution" class="d-flex flex-wrap ga-2">
              <v-chip
                v-for="(count, label) in evalResult.prediction_distribution"
                :key="label"
                size="small"
                variant="tonal"
              >
                {{ label }}: {{ count }}
              </v-chip>
            </div>
          </div>

          <div v-if="evalResult.new_metrics" class="text-caption text-medium-emphasis mt-2">
            Evaluated on {{ evalResult.new_metrics?.test_samples || '?' }} labeled samples
          </div>
          <div v-else class="text-caption text-medium-emphasis mt-2">
            No label column found in CSV — showing predictions only
          </div>

          <v-divider class="my-3" />
          <v-btn
            color="success"
            variant="flat"
            size="small"
            @click="downloadEvalCsv"
          >
            <v-icon start size="small">mdi-download</v-icon>
            Save Results as CSV
          </v-btn>
        </v-card>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

// Model selection
const modelSource = ref<'session' | 'saved'>(pipelineStore.trainingSession ? 'session' : 'saved')
const savedModels = ref<any[]>([])
const selectedSavedModelId = ref<number | null>(null)
const loadingSavedModels = ref(false)
const modelTableTab = ref('regression')
const showDeleteModelDialog = ref(false)
const showTiMcuInfo = ref(false)
const deleteTargetModel = ref<any>(null)
const deletingModel = ref(false)

const regressionModels = computed(() =>
  savedModels.value.filter(m => m.mode === 'regression')
)
const classAnomalyModels = computed(() =>
  savedModels.value.filter(m => m.mode !== 'regression')
)

function confirmDeleteModel(model: any) {
  deleteTargetModel.value = model
  showDeleteModelDialog.value = true
}

async function deleteModel() {
  if (!deleteTargetModel.value) return
  deletingModel.value = true
  try {
    await api.delete(`/api/training/saved-models/${deleteTargetModel.value.id}`)
    savedModels.value = savedModels.value.filter(m => m.id !== deleteTargetModel.value.id)
    if (selectedSavedModelId.value === deleteTargetModel.value.id) {
      selectedSavedModelId.value = null
    }
    showDeleteModelDialog.value = false
  } catch (e: any) {
    alert(e.response?.data?.error || 'Failed to delete model')
  }
  deletingModel.value = false
}

// Deploy config
const deployTarget = ref<'download' | 'ssh'>('download')

// Auto-switch to Download when TI MCU is selected (no SSH for MCU)
watch(() => exportFormat.value, (newFormat) => {
  if (newFormat === 'ti_mcu') {
    deployTarget.value = 'download'
  }
})
const targetDevice = ref('jetson_nano')
const exportFormat = ref('onnx')
const deployMode = ref<'docker' | 'files'>('docker')
const enableGpu = ref(false)

const showPassword = ref(false)

const sshConfig = reactive({
  host: '',
  username: '',
  password: '',
  port: 22,
  remote_path: '~/cira_models'
})

const testingConnection = ref(false)
const connectionStatus = ref<any>(null)
const deploying = ref(false)
const exporting = ref(false)
const deploymentResult = ref<any>(null)

// Evaluation state
const showEvalDialog = ref(false)
const evalModel = ref<any>(null)
const evaluating = ref(false)
const evalResult = ref<any>(null)
const evalFile = ref<File | null>(null)
const evalTargetColumn = ref('')
const evalCsvColumns = ref<string[]>([])

// Parse CSV headers when file is selected
watch(evalFile, async (newFile) => {
  evalCsvColumns.value = []
  evalTargetColumn.value = ''
  if (!newFile) return
  try {
    const text = await newFile.slice(0, 4096).text()
    const firstLine = text.split('\n')[0].trim()
    const headers = firstLine.split(',').map((h: string) => h.trim().replace(/^["']|["']$/g, ''))
    evalCsvColumns.value = headers.filter((h: string) => h && h.length > 0)
    const savedTarget = evalModel.value?.pipeline_config?.target_column
    if (savedTarget && evalCsvColumns.value.includes(savedTarget)) {
      evalTargetColumn.value = savedTarget
    }
  } catch { evalCsvColumns.value = [] }
})
const downloadingPackage = ref(false)

// Saved devices (persisted in localStorage)
interface SavedDevice {
  id: string
  name: string
  host: string
  username: string
  password: string
  port: number
  remote_path: string
}

const STORAGE_KEY = 'cira_saved_devices'

function loadDevicesFromStorage(): SavedDevice[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch { return [] }
}

const savedDevices = ref<SavedDevice[]>(loadDevicesFromStorage())
const renamingDeviceId = ref<string | null>(null)
const renameInputs: Record<string, HTMLInputElement> = {}

function persistDevices() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(savedDevices.value))
}

const isCurrentDeviceSaved = computed(() =>
  savedDevices.value.some(
    d => d.host === sshConfig.host && d.username === sshConfig.username
  )
)

function saveCurrentDevice() {
  const id = `${sshConfig.host}_${sshConfig.username}_${Date.now()}`
  const name = sshConfig.host  // default name = IP, user can rename
  savedDevices.value.push({
    id, name,
    host: sshConfig.host,
    username: sshConfig.username,
    password: sshConfig.password,
    port: sshConfig.port,
    remote_path: sshConfig.remote_path,
  })
  persistDevices()
  notificationStore.showSuccess(`Device "${name}" saved`)
}

function loadDevice(dev: SavedDevice) {
  sshConfig.host = dev.host
  sshConfig.username = dev.username
  sshConfig.password = dev.password
  sshConfig.port = dev.port
  sshConfig.remote_path = dev.remote_path
  connectionStatus.value = null
}

function deleteDevice(id: string) {
  savedDevices.value = savedDevices.value.filter(d => d.id !== id)
  persistDevices()
}

function startRename(dev: SavedDevice) {
  renamingDeviceId.value = dev.id
  setTimeout(() => renameInputs[dev.id]?.focus(), 50)
}

function finishRename() {
  renamingDeviceId.value = null
  persistDevices()
}

// Build log state
const buildLog = ref<string | null>(null)
const fetchingLog = ref(false)
const buildLogInterval = ref<ReturnType<typeof setInterval> | null>(null)

// Post-deployment state
const checkingFiles = ref(false)
const remoteFiles = ref<any>(null)
const inferenceFile = ref<File | null>(null)
const runningInference = ref(false)
const inferenceOutput = ref<any>(null)
const inferenceError = ref<string | null>(null)

// Computed
const hasModelSelected = computed(() => {
  if (modelSource.value === 'session') return !!pipelineStore.trainingSession
  if (modelSource.value === 'saved') return !!selectedSavedModelId.value
  return false
})

const canDeploy = computed(() =>
  hasModelSelected.value && sshConfig.host && sshConfig.username
)

const selectedModel = computed(() => {
  if (modelSource.value === 'session' && pipelineStore.trainingSession) {
    const s = pipelineStore.trainingSession
    return {
      name: s.algorithm,
      algorithm: s.algorithm,
      mode: s.mode,
      metrics: s.metrics,
      dataset_info: null
    }
  }
  if (modelSource.value === 'saved' && selectedSavedModelId.value) {
    return savedModels.value.find(m => m.id === selectedSavedModelId.value) || null
  }
  return null
})

// Functions
function selectSavedModel(model: any) {
  selectedSavedModelId.value = model.id
}

function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

function getModelTypeInfo(model: any) {
  if (!model) return { label: '?', color: 'grey', canApi: false, canMcu: false }
  const algo = (model.algorithm || '').toUpperCase()
  const approach = model.pipeline_config?.training_approach || ''

  // TI NN models (trained in TI container with neural networks)
  if (algo.startsWith('REGR_') || algo.startsWith('REGR ') ||
      algo.startsWith('CLF_TS') || algo.startsWith('AE_TS')) {
    return { label: 'TI NN', color: 'orange', canApi: false, canMcu: true, icon: 'mdi-chip' }
  }
  // TI Traditional ML (trained via TI tab but sklearn models)
  if (approach === 'ti' && !algo.startsWith('REGR')) {
    return { label: 'TI ML', color: 'light-blue', canApi: true, canMcu: true, icon: 'mdi-memory' }
  }
  // Deep Learning (TimesNet)
  if (approach === 'dl' || algo.includes('TIMESNET') || algo.includes('DL_NETWORK') || algo === 'MLP') {
    return { label: 'DL', color: 'amber', canApi: true, canMcu: false, icon: 'mdi-brain' }
  }
  // Custom model
  if (approach === 'custom') {
    return { label: 'Custom', color: 'purple', canApi: true, canMcu: false, icon: 'mdi-code-braces' }
  }
  // Default: Traditional ML
  return { label: 'ML', color: 'blue', canApi: true, canMcu: true, icon: 'mdi-cog' }
}

const isTiNnModel = computed(() => {
  const model = savedModels.value.find(m => m.id === selectedSavedModelId.value)
  if (!model) return false
  return getModelTypeInfo(model).label === 'TI NN'
})

function openEvalDialog() {
  const model = savedModels.value.find(m => m.id === selectedSavedModelId.value)
  if (!model) return
  if (isTiNnModel.value) {
    alert('TI NN models require deployment to MCU for testing. Use "Download Package" to get the C code for Code Composer Studio.')
    return
  }
  evalModel.value = model
  evalResult.value = null
  evalFile.value = null
  showEvalDialog.value = true
}

function downloadEvalCsv() {
  if (!evalResult.value) return

  const rows: string[] = []
  const mode = evalModel.value?.mode || 'classification'
  const modelName = evalModel.value?.name || 'model'

  const preds = evalResult.value.predictions || []
  const actuals = evalResult.value.actuals || []
  const probs = evalResult.value.probabilities || []
  const hasActuals = actuals.length === preds.length

  // Header row
  if (mode === 'regression') {
    rows.push(hasActuals ? 'window_index,actual,prediction,residual' : 'window_index,prediction')
  } else {
    rows.push(hasActuals ? 'window_index,actual,prediction,correct,confidence' : 'window_index,prediction,confidence')
  }

  // Data rows — one row per window
  for (let i = 0; i < preds.length; i++) {
    if (mode === 'regression') {
      if (hasActuals) {
        const actual = parseFloat(actuals[i])
        const pred = parseFloat(preds[i] as any)
        const residual = (actual - pred).toFixed(6)
        rows.push(`${i},${actual},${pred},${residual}`)
      } else {
        rows.push(`${i},${preds[i]}`)
      }
    } else {
      const conf = probs[i] ? Math.max(...probs[i]).toFixed(4) : ''
      if (hasActuals) {
        const correct = String(actuals[i]).toLowerCase() === String(preds[i]).toLowerCase() ? 'YES' : 'NO'
        rows.push(`${i},${actuals[i]},${preds[i]},${correct},${conf}`)
      } else {
        rows.push(`${i},${preds[i]},${conf}`)
      }
    }
  }

  // Add summary section
  rows.push('')
  rows.push('--- Summary ---')
  rows.push(`model,${modelName}`)
  rows.push(`mode,${mode}`)
  rows.push(`num_windows,${preds.length}`)

  if (evalResult.value.new_metrics) {
    rows.push('')
    rows.push('--- Metrics ---')
    for (const [key, val] of Object.entries(evalResult.value.new_metrics)) {
      if (val != null && key !== 'test_samples') {
        rows.push(`${key},${val}`)
      }
    }
  }

  if (evalResult.value.comparison) {
    rows.push('')
    rows.push('--- Comparison (Original vs New) ---')
    rows.push('metric,original,new_data,diff')
    for (const c of evalResult.value.comparison) {
      rows.push(`${c.metric},${c.original ?? ''},${c.new_data ?? ''},${c.diff ?? ''}`)
    }
  }

  // Download
  const csv = rows.join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `eval_${modelName.replace(/\s+/g, '_')}_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
  notificationStore.showSuccess('Evaluation results saved as CSV')
}

async function runRawEvaluation() {
  if (!evalModel.value || !evalFile.value) return
  evaluating.value = true
  evalResult.value = null
  try {
    const formData = new FormData()
    formData.append('file', evalFile.value)
    formData.append('saved_model_id', String(evalModel.value.id))
    if (evalTargetColumn.value) {
      formData.append('target_column', evalTargetColumn.value)
    }

    const response = await api.post('/api/training/evaluate-raw', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    evalResult.value = response.data
    notificationStore.showSuccess('Evaluation complete')
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Evaluation failed')
  }
  evaluating.value = false
}

async function downloadPackage() {
  if (!selectedSavedModelId.value) return
  downloadingPackage.value = true
  try {
    const response = await api.post(
      `/api/deployment/package/${selectedSavedModelId.value}`,
      {},
      { responseType: 'blob' }
    )
    const blob = new Blob([response.data], { type: 'application/zip' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const disposition = response.headers['content-disposition']
    const filename = disposition
      ? disposition.split('filename=')[1]?.replace(/"/g, '')
      : `deployment_package_${selectedSavedModelId.value}.zip`
    a.download = filename
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
    notificationStore.showSuccess('Deployment package downloaded')
  } catch (e: any) {
    if (e.response?.data instanceof Blob) {
      const text = await e.response.data.text()
      try {
        const json = JSON.parse(text)
        notificationStore.showError(json.error || 'Download failed')
      } catch {
        notificationStore.showError('Download failed')
      }
    } else {
      notificationStore.showError(e.response?.data?.error || 'Download failed')
    }
  } finally {
    downloadingPackage.value = false
  }
}


async function loadSavedModels() {
  try {
    loadingSavedModels.value = true
    const response = await api.get('/api/training/saved-models')
    savedModels.value = response.data || []
  } catch (e: any) {
    console.error('Failed to load saved models:', e)
  } finally {
    loadingSavedModels.value = false
  }
}

async function testConnection() {
  try {
    testingConnection.value = true

    const response = await api.post('/api/deployment/test-connection', sshConfig)
    connectionStatus.value = response.data

    if (response.data.status === 'connected') {
      notificationStore.showSuccess('Connection successful!')
      // Auto-enable GPU if Jetson with nvidia runtime detected and docker mode
      if (response.data.is_jetson && response.data.nvidia_runtime && deployMode.value === 'docker') {
        enableGpu.value = true
      }
    }
  } catch (e: any) {
    connectionStatus.value = {
      status: 'failed',
      message: e.response?.data?.error || 'Connection failed'
    }
    notificationStore.showError('Connection failed')
  } finally {
    testingConnection.value = false
  }
}

async function exportOnly() {
  if (!hasModelSelected.value) {
    notificationStore.showError('No model selected')
    return
  }

  try {
    exporting.value = true

    // TI MCU: export saved model as ONNX/C code package
    if (exportFormat.value === 'ti_mcu') {
      if (!selectedSavedModelId.value) {
        notificationStore.showError('TI MCU export requires a saved model')
        return
      }
      try {
        const response = await api.post(
          `/api/ti/export-saved/${selectedSavedModelId.value}`,
          {},
          { responseType: 'blob' }
        )
        const model = savedModels.value.find((m: any) => m.id === selectedSavedModelId.value)
        _downloadBlob(response, `ti_mcu_${model?.algorithm || 'model'}.zip`)
        notificationStore.showSuccess('TI MCU package downloaded')
      } catch (e: any) {
        if (e.response?.data instanceof Blob) {
          const text = await e.response.data.text()
          try { notificationStore.showError(JSON.parse(text).error) } catch { notificationStore.showError('TI MCU export failed') }
        } else {
          notificationStore.showError(e.response?.data?.error || 'TI MCU export failed')
        }
      }
      return
    }

    // CiRA CLAW: download zip with model.onnx + manifest
    if (exportFormat.value === 'cira_claw') {
      if (!selectedSavedModelId.value) {
        notificationStore.showError('CiRA CLAW export requires a saved model')
        return
      }
      const response = await api.post(
        `/api/deployment/cira-claw-package/${selectedSavedModelId.value}`,
        {},
        { responseType: 'blob' }
      )
      _downloadBlob(response, `cira_claw_${selectedSavedModelId.value}.zip`)
      notificationStore.showSuccess('CiRA CLAW package downloaded (model.onnx + cira_model.json)')
      return
    }

    // Other formats: download deployment package zip
    if (modelSource.value === 'saved' && selectedSavedModelId.value) {
      const response = await api.post(
        `/api/deployment/package/${selectedSavedModelId.value}`,
        {},
        { responseType: 'blob' }
      )
      _downloadBlob(response, `deployment_package_${selectedSavedModelId.value}.zip`)
      notificationStore.showSuccess(`Deployment package downloaded (${exportFormat.value})`)
    } else {
      // In-memory session: server-side export only (no download)
      const url = `/api/training/export/${pipelineStore.trainingSession!.training_session_id}`
      await api.post(url, { format: exportFormat.value })
      notificationStore.showSuccess(`Model exported as ${exportFormat.value}`)
    }
  } catch (e: any) {
    if (e.response?.data instanceof Blob) {
      const text = await e.response.data.text()
      try {
        const json = JSON.parse(text)
        notificationStore.showError(json.error || 'Export failed')
      } catch {
        notificationStore.showError('Export failed')
      }
    } else {
      notificationStore.showError(e.response?.data?.error || 'Export failed')
    }
  } finally {
    exporting.value = false
  }
}

function _downloadBlob(response: any, fallbackName: string) {
  const blob = new Blob([response.data], { type: 'application/zip' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const disposition = response.headers['content-disposition']
  a.download = disposition
    ? disposition.split('filename=')[1]?.replace(/"/g, '')
    : fallbackName
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

async function deploy() {
  if (!hasModelSelected.value) {
    notificationStore.showError('No model selected')
    return
  }

  try {
    deploying.value = true

    const payload: any = {
      target_type: targetDevice.value,
      export_format: exportFormat.value,
      deploy_mode: deployMode.value,
      enable_gpu: enableGpu.value,
      jetpack_version: connectionStatus.value?.jetpack_version ?? null,
      ...sshConfig
    }

    if (modelSource.value === 'session' && pipelineStore.trainingSession) {
      payload.training_session_id = pipelineStore.trainingSession.training_session_id
    } else {
      payload.saved_model_id = selectedSavedModelId.value
    }

    const response = await api.post('/api/deployment/deploy', payload)
    deploymentResult.value = response.data
    if (response.data.container_started === false) {
      notificationStore.showSuccess('Files transferred — container building in background')
    } else {
      notificationStore.showSuccess('Deployment successful!')
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Deployment failed')
  } finally {
    deploying.value = false
  }
}

async function fetchBuildLog() {
  if (!deploymentResult.value?.build_log_file || !deploymentResult.value?.container_name) return
  fetchingLog.value = true
  try {
    const res = await api.post('/api/deployment/build-log', {
      host: sshConfig.host,
      username: sshConfig.username,
      password: sshConfig.password,
      port: sshConfig.port,
      log_file: deploymentResult.value.build_log_file,
      container_name: deploymentResult.value.container_name,
    })
    buildLog.value = res.data.log
    if (res.data.container_started) {
      // Container is now up — update result and stop polling
      deploymentResult.value.container_started = true
      stopBuildLogPolling()
      notificationStore.showSuccess('Docker container is now running!')
    }
  } catch (e: any) {
    buildLog.value = `Error fetching log: ${e.response?.data?.error || e.message}`
  } finally {
    fetchingLog.value = false
  }
}

function startBuildLogPolling() {
  fetchBuildLog()
  buildLogInterval.value = setInterval(fetchBuildLog, 15000)
}

function stopBuildLogPolling() {
  if (buildLogInterval.value) {
    clearInterval(buildLogInterval.value)
    buildLogInterval.value = null
  }
}

async function checkRemoteFiles() {
  if (!deploymentResult.value) return
  checkingFiles.value = true
  remoteFiles.value = null
  try {
    const res = await api.post('/api/deployment/remote-files', {
      ...sshConfig,
      remote_path: deploymentResult.value.remote_path
    })
    remoteFiles.value = res.data
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to list remote files')
  } finally {
    checkingFiles.value = false
  }
}

async function runRemoteInference() {
  if (!inferenceFile.value || !deploymentResult.value) return
  runningInference.value = true
  inferenceOutput.value = null
  inferenceError.value = null
  try {
    const form = new FormData()
    form.append('file', inferenceFile.value)
    form.append('host', sshConfig.host)
    form.append('username', sshConfig.username)
    form.append('password', sshConfig.password)
    form.append('port', String(sshConfig.port))
    form.append('remote_path', deploymentResult.value.remote_path)
    form.append('deploy_mode', deploymentResult.value.deploy_mode || 'docker')
    form.append('service_name', deploymentResult.value.service_name || 'inference')
    form.append('container_name', deploymentResult.value.container_name || '')
    const res = await api.post('/api/deployment/run-inference', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180000
    })
    inferenceOutput.value = res.data
    // Update container_name in deploymentResult if backend resolved it via auto-discovery
    if (res.data.container_name && deploymentResult.value) {
      deploymentResult.value.container_name = res.data.container_name
    }
    notificationStore.showSuccess('Inference completed on remote device')
  } catch (e: any) {
    inferenceError.value = e.response?.data?.error || 'Inference failed'
  } finally {
    runningInference.value = false
  }
}

onMounted(() => {
  loadSavedModels()
})

onUnmounted(() => {
  stopBuildLogPolling()
})
</script>

<style scoped>
/* TI MCU Info Dialog */
.ti-flow-diagram {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px 0;
}
.ti-flow-step {
  text-align: center;
  min-width: 90px;
}
.ti-flow-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  border: 2px solid;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 6px;
}
.ti-flow-label {
  font-size: 11px;
  font-weight: 600;
  color: #e6edf3;
}
.ti-flow-sub {
  font-size: 9px;
  color: #8b949e;
}
.ti-flow-arrow {
  padding-top: 0;
}
.ti-steps {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ti-step {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.ti-step-num {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #a78bfa20;
  color: #a78bfa;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}

.rename-input {
  background: transparent;
  border: none;
  outline: none;
  color: inherit;
  font-size: inherit;
  width: 90px;
  min-width: 50px;
}
</style>
