<template>
  <v-container fluid class="pa-6">
    <!-- Phase B — machine context banner. Non-blocking; hidden by the
         component itself when auth/tree isn't ready. -->
    <MachinePipelineBanner />
    <!-- Header with Stepper -->
    <PipelineStepper current-step="data" class="mb-6" />

    <h2 class="text-h5 font-weight-bold mb-2">Data Source</h2>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Select a data file to begin the ML pipeline
    </p>

    <v-row>
      <!-- Source Type Selection -->
      <v-col cols="12" md="4">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">File Format</h3>

          <v-radio-group v-model="selectedFormat" hide-details>
            <v-radio value="csv">
              <template #label>
                <div>
                  <div class="font-weight-medium">CSV File</div>
                  <div class="text-caption text-medium-emphasis">
                    Standard comma-separated values
                  </div>
                </div>
              </template>
            </v-radio>

            <v-radio value="ei-json">
              <template #label>
                <div>
                  <div class="font-weight-medium">Edge Impulse JSON</div>
                  <div class="text-caption text-medium-emphasis">
                    JSON format from Edge Impulse Studio
                  </div>
                </div>
              </template>
            </v-radio>

            <v-radio value="ei-cbor">
              <template #label>
                <div>
                  <div class="font-weight-medium">Edge Impulse CBOR</div>
                  <div class="text-caption text-medium-emphasis">
                    Binary CBOR format (compact)
                  </div>
                </div>
              </template>
            </v-radio>

            <v-radio value="cira-cbor">
              <template #label>
                <div>
                  <div class="font-weight-medium">CiRA CBOR</div>
                  <div class="text-caption text-medium-emphasis">
                    CiRA native recording format
                  </div>
                </div>
              </template>
            </v-radio>

            <v-radio value="text">
              <template #label>
                <div>
                  <div class="font-weight-medium">Text File</div>
                  <div class="text-caption text-medium-emphasis">
                    Any delimited text file (.txt, .tsv, .log). Auto-detects delimiter.
                  </div>
                </div>
              </template>
            </v-radio>

            <v-radio value="url">
              <template #label>
                <div>
                  <div class="font-weight-medium">Load from URL</div>
                  <div class="text-caption text-medium-emphasis">
                    Fetch a CSV / text file over HTTPS (max 100 MB). Includes Factory / PdM catalog.
                  </div>
                </div>
              </template>
            </v-radio>
          </v-radio-group>

          <!-- Format Info -->
          <v-alert
            v-if="formatInfo"
            type="info"
            variant="tonal"
            density="compact"
            class="mt-4"
          >
            {{ formatInfo }}
          </v-alert>
        </v-card>
      </v-col>

      <!-- File Browser -->
      <v-col v-if="!isUrlFormat" cols="12" md="8">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4 flex-wrap">
            <h3 class="text-subtitle-1 font-weight-bold">Browse Files</h3>
            <v-chip
              v-if="isCsvFormat && selectedFiles.length > 0"
              size="x-small"
              :color="isCrossFolderSelection ? 'secondary' : 'primary'"
              variant="tonal"
              class="ml-2"
            >
              <v-icon v-if="isCrossFolderSelection" size="14" class="mr-1">mdi-source-merge</v-icon>
              {{ selectedFiles.length }} selected
              <span v-if="isCrossFolderSelection">
                &nbsp;· {{ basketFolders.length }} folders
              </span>
            </v-chip>
            <v-btn
              v-if="isCsvFormat && selectedFiles.length > 0"
              variant="text"
              size="x-small"
              color="error"
              class="ml-1"
              @click="clearBasket"
            >
              Clear
            </v-btn>
            <v-spacer />
            <v-btn
              v-if="isCsvFormat && folderIsMachineShape"
              variant="tonal"
              size="small"
              color="secondary"
              prepend-icon="mdi-source-branch-plus"
              class="mr-2"
              @click="openLoadAllDialog"
            >
              Load All Sensors
            </v-btn>
            <v-btn
              v-if="isCsvFormat && csvFilesInFolder.length > 0"
              variant="tonal"
              size="small"
              :color="allCsvSelectedInFolder ? 'warning' : 'success'"
              :prepend-icon="allCsvSelectedInFolder ? 'mdi-checkbox-blank-outline' : 'mdi-checkbox-multiple-marked'"
              class="mr-2"
              @click="toggleSelectAllCsv"
            >
              {{ allCsvSelectedInFolder ? 'Deselect All' : `Select All (${csvFilesInFolder.length})` }}
            </v-btn>
            <v-btn
              v-if="authStore.isAdmin"
              variant="tonal"
              size="small"
              color="secondary"
              prepend-icon="mdi-access-point"
              class="mr-2"
              @click="showRecordDialog = true"
            >
              Record Sensors
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              color="info"
              prepend-icon="mdi-information-outline"
              class="mr-2"
              @click="showFormatGuide = true"
            >
              Format Guide
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              color="primary"
              prepend-icon="mdi-upload"
              class="mr-2"
              @click="showUploadDialog = true"
            >
              Upload
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              prepend-icon="mdi-folder-cog-outline"
              class="mr-2"
              @click="showFileManager = true"
            >
              Manage Files
            </v-btn>
            <v-btn
              variant="text"
              size="small"
              prepend-icon="mdi-refresh"
              @click="loadFolders"
              :loading="loadingFolders"
            >
              Refresh
            </v-btn>
          </div>

          <!-- Breadcrumb -->
          <v-breadcrumbs :items="breadcrumbs" density="compact" class="pa-0 mb-2">
            <template #divider>
              <v-icon>mdi-chevron-right</v-icon>
            </template>
            <template #item="{ item }">
              <v-breadcrumbs-item
                :disabled="item.disabled"
                @click="navigateTo(item.path)"
              >
                {{ item.title }}
              </v-breadcrumbs-item>
            </template>
          </v-breadcrumbs>

          <!-- Dataset Folder Detection: Scan prompt (before scan) -->
          <v-alert
            v-if="isDatasetFolder && isCborFormat && !datasetScan"
            type="info"
            variant="tonal"
            density="compact"
            class="mb-3"
          >
            <div class="d-flex align-center">
              <div>
                <div class="font-weight-medium">Dataset folder detected</div>
                <div class="text-caption">Contains training/testing subfolders. Scan to explore partitions.</div>
              </div>
              <v-spacer />
              <v-btn
                color="primary"
                size="small"
                :loading="scanning"
                @click="scanDatasetFolder"
              >
                <v-icon start>mdi-magnify-scan</v-icon>
                Scan Dataset
              </v-btn>
            </div>
          </v-alert>

          <!-- Dataset Partition Selector (after scan) -->
          <v-card v-if="datasetScan" variant="outlined" class="mb-3 pa-4">
            <div class="d-flex align-center mb-3">
              <v-icon class="mr-2" color="primary">mdi-folder-open</v-icon>
              <span class="font-weight-medium">Dataset Structure</span>
              <v-spacer />
              <v-chip size="x-small" color="info" variant="flat">
                {{ datasetScan.total_files }} files total
              </v-chip>
            </div>

            <!-- Category chips -->
            <div class="mb-3">
              <span class="text-caption text-medium-emphasis mr-2">Category:</span>
              <v-chip
                v-for="(catData, catName) in datasetScan.categories"
                :key="catName"
                class="mr-2 mb-1"
                :color="selectedCategory === catName ? 'primary' : 'default'"
                :variant="selectedCategory === catName ? 'flat' : 'outlined'"
                size="small"
                @click="selectCategory(catName as string)"
              >
                {{ catName }}
                <span class="ml-1 text-caption">({{ catData.file_count }} files)</span>
              </v-chip>
            </div>

            <!-- Label chips (shown after category is selected) -->
            <div v-if="selectedCategory && datasetScan.categories[selectedCategory]">
              <span class="text-caption text-medium-emphasis mr-2">Label:</span>
              <v-chip
                class="mr-2 mb-1"
                :color="selectedLabel === null ? 'secondary' : 'default'"
                :variant="selectedLabel === null ? 'flat' : 'outlined'"
                size="small"
                @click="selectLabel(null)"
              >
                All
              </v-chip>
              <v-chip
                v-for="(labelData, labelName) in datasetScan.categories[selectedCategory].labels"
                :key="labelName"
                class="mr-2 mb-1"
                :color="selectedLabel === labelName ? 'secondary' : 'default'"
                :variant="selectedLabel === labelName ? 'flat' : 'outlined'"
                size="small"
                @click="selectLabel(labelName as string)"
              >
                {{ labelName }}
                <span class="ml-1 text-caption">({{ labelData.file_count }})</span>
              </v-chip>
            </div>
          </v-card>

          <!-- File List -->
          <v-list
            density="compact"
            class="file-list"
            max-height="400"
            style="overflow-y: auto"
          >
            <v-list-item
              v-for="item in currentItems"
              :key="item.path"
              :class="{
                'selected': !isCsvFormat || item.is_dir
                  ? selectedFile?.path === item.path
                  : isFileSelected(item.path)
              }"
              @click="handleItemClick(item)"
            >
              <template #prepend>
                <!-- Checkbox for CSV files in CSV mode -->
                <v-checkbox-btn
                  v-if="isCsvFormat && !item.is_dir && item.extension === '.csv'"
                  :model-value="isFileSelected(item.path)"
                  density="compact"
                  class="mr-1"
                  @click.stop="toggleCsvFile(item)"
                />
                <v-icon :color="getFolderColor(item)">
                  {{ item.is_dir ? getFolderIcon(item) : getFileIcon(item.extension) }}
                </v-icon>
              </template>

              <v-list-item-title>{{ item.name }}</v-list-item-title>

              <template #append>
                <v-chip
                  v-if="item.is_dir && isDatasetRootFolder(item)"
                  size="x-small"
                  color="primary"
                  variant="flat"
                  class="mr-2"
                >
                  Dataset
                </v-chip>
                <span v-if="!item.is_dir" class="text-caption text-medium-emphasis mr-2">
                  {{ formatFileSize(item.size) }}
                </span>
                <!-- Download Button (files only) -->
                <v-btn
                  v-if="!item.is_dir"
                  icon
                  variant="text"
                  size="x-small"
                  color="primary"
                  @click.stop="downloadFile(item)"
                  title="Download"
                >
                  <v-icon size="small">mdi-download</v-icon>
                </v-btn>
                <!-- Delete Button (admin or user's own folder) -->
                <v-btn
                  v-if="canDeleteItem(item)"
                  icon
                  variant="text"
                  size="x-small"
                  color="error"
                  @click.stop="confirmDelete(item)"
                  title="Delete"
                >
                  <v-icon size="small">mdi-delete</v-icon>
                </v-btn>
              </template>
            </v-list-item>

            <v-list-item v-if="currentItems.length === 0">
              <v-list-item-title class="text-medium-emphasis">
                No files found in this directory
              </v-list-item-title>
            </v-list-item>
          </v-list>

          <!-- Multi-CSV selection info — grouped by folder so cross-folder
               picks are legible. Each file has an X to drop it from the basket. -->
          <v-alert
            v-if="isCsvFormat && selectedFiles.length > 1"
            :type="isCrossFolderSelection ? 'warning' : 'info'"
            variant="tonal"
            class="mt-4"
            density="compact"
          >
            <div class="d-flex align-center mb-1">
              <div class="font-weight-medium">
                {{ selectedFiles.length }} CSV files selected
                <span v-if="isCrossFolderSelection">
                  · cross-sensor JOIN (columns: {{ sensorsInBasket.join(', ') }})
                </span>
              </div>
            </div>
            <div v-for="grp in basketByFolder" :key="grp.dir" class="mt-1">
              <div class="text-caption font-weight-medium">{{ grp.dir }}/</div>
              <v-chip
                v-for="f in grp.files"
                :key="f.path"
                size="x-small"
                variant="outlined"
                closable
                class="mr-1 mb-1"
                @click:close="removeFromBasket(f)"
              >
                {{ f.name }}
              </v-chip>
            </div>
          </v-alert>

          <!-- Multi-CSV column mismatch error -->
          <v-alert
            v-if="multiCsvError"
            type="error"
            variant="tonal"
            class="mt-4"
          >
            <div class="font-weight-medium">Column mismatch</div>
            <div class="text-caption">{{ multiCsvError }}</div>
          </v-alert>

          <!-- Selected File/Folder Info (single select) -->
          <v-alert
            v-if="selectedFile && !(isCsvFormat && selectedFiles.length > 1)"
            type="success"
            variant="tonal"
            class="mt-4"
          >
            <div class="font-weight-medium">Selected: {{ selectedFile.name }}</div>
            <div class="text-caption">{{ selectedFile.path }}</div>
          </v-alert>
        </v-card>
      </v-col>

      <!-- URL Loader (shown when File Format = Load from URL) -->
      <v-col v-if="isUrlFormat" cols="12" md="8">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4 flex-wrap">
            <h3 class="text-subtitle-1 font-weight-bold">Fetch from URL</h3>
            <v-spacer />
            <v-btn
              variant="tonal"
              size="small"
              color="info"
              prepend-icon="mdi-information-outline"
              @click="showFormatGuide = true"
            >
              Format Guide
            </v-btn>
          </div>

          <p class="text-body-2 text-medium-emphasis mb-3">
            Paste a direct <code>https://</code> link to a CSV or delimited text file.
            The file is streamed to a temporary buffer (100 MB hard cap) and never written to disk.
          </p>

          <v-text-field
            v-model="urlLoader.url"
            label="File URL"
            placeholder="https://…"
            density="compact"
            variant="outlined"
            hide-details="auto"
            hint="HTTPS only. Some hosts (e.g. GitHub) may require a raw-content URL."
            class="mb-3"
          />

          <v-row>
            <v-col cols="12" md="4">
              <v-select
                v-model="urlLoader.format"
                :items="[
                  { title: 'CSV', value: 'csv' },
                  { title: 'Text (delimited)', value: 'text' },
                ]"
                item-title="title"
                item-value="value"
                label="Format"
                density="compact"
                variant="outlined"
                hide-details
              />
            </v-col>

            <template v-if="urlLoader.format === 'text'">
              <v-col cols="6" md="3">
                <v-text-field
                  v-model="urlLoader.delimiter"
                  label="Delimiter"
                  placeholder="auto"
                  maxlength="4"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
              <v-col cols="6" md="2">
                <v-text-field
                  v-model.number="urlLoader.headerRow"
                  type="number"
                  label="Header row"
                  min="0"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
              <v-col cols="6" md="3">
                <v-text-field
                  v-model.number="urlLoader.skipRows"
                  type="number"
                  label="Skip rows"
                  min="0"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
            </template>
          </v-row>

          <div class="d-flex align-center mt-4">
            <v-btn
              color="primary"
              size="small"
              :loading="urlLoader.loading"
              :disabled="!urlLoader.url.trim()"
              prepend-icon="mdi-cloud-download-outline"
              @click="fetchFromUrl"
            >
              Fetch &amp; Preview
            </v-btn>
            <v-spacer />
            <span v-if="dataPreview?.metadata?.source_url" class="text-caption text-medium-emphasis text-truncate" style="max-width: 60%;">
              <v-icon size="small" class="mr-1">mdi-link-variant</v-icon>
              {{ dataPreview.metadata.source_url }}
            </span>
          </div>

          <v-divider class="my-4" />

          <!-- Factory / PdM Quick-Pick chips -->
          <div class="mb-2">
            <div class="text-subtitle-2 font-weight-medium">
              <v-icon size="small" color="primary" class="mr-1">mdi-factory</v-icon>
              Quick-pick from Factory / PdM samples
              <span class="text-caption text-medium-emphasis ml-1">(~1-10 MB each)</span>
            </div>
          </div>
          <div class="mb-2">
            <v-chip
              v-for="key in PDM_QUICK_PICK_KEYS"
              :key="key"
              class="mr-2 mb-1"
              color="primary"
              variant="tonal"
              size="small"
              prepend-icon="mdi-database-outline"
              @click="pickPdmSample(key)"
            >
              {{ key }}
            </v-chip>
          </div>

          <!-- Factory / PdM Full Catalog (expansion panel) -->
          <v-expansion-panels variant="accordion" class="mt-3">
            <v-expansion-panel>
              <v-expansion-panel-title>
                <v-icon class="mr-2" color="secondary">mdi-book-open-variant</v-icon>
                Factory / Predictive-Maintenance Catalog
                <v-chip size="x-small" color="secondary" variant="tonal" class="ml-2">
                  {{ PDM_CATALOG.length }} datasets
                </v-chip>
              </v-expansion-panel-title>
              <v-expansion-panel-text>
                <v-alert
                  type="info"
                  variant="tonal"
                  density="compact"
                  class="mb-3"
                  icon="mdi-license"
                >
                  All datasets are direct-URL CSV or space-delimited TXT streamed to memory
                  (no server storage). Numeric multi-sensor data — fits the CiRA ME pipeline.
                  Licenses vary per row; commercial use permitted for CC BY 4.0, MIT, and US Gov works.
                </v-alert>

                <div
                  v-for="category in pdmCategories"
                  :key="category"
                  class="mb-4"
                >
                  <div class="text-caption text-medium-emphasis font-weight-medium mb-1">
                    {{ category }}
                  </div>
                  <v-table density="compact" class="loghub-table">
                    <thead>
                      <tr>
                        <th>Dataset</th>
                        <th>Description</th>
                        <th class="text-center">Labeled</th>
                        <th class="text-right">#Rows</th>
                        <th class="text-right">Raw Size</th>
                        <th class="text-center">Format</th>
                        <th class="text-center">License</th>
                        <th class="text-center">Load</th>
                        <th class="text-center">Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr
                        v-for="entry in pdmByCategory[category]"
                        :key="entry.key"
                      >
                        <td class="font-weight-medium">{{ entry.key }}</td>
                        <td>{{ entry.description }}</td>
                        <td class="text-center">
                          <v-icon v-if="entry.labeled" size="small" color="success">mdi-check</v-icon>
                          <span v-else class="text-medium-emphasis">—</span>
                        </td>
                        <td class="text-right">{{ entry.rows?.toLocaleString() ?? '—' }}</td>
                        <td class="text-right">{{ entry.sizeRaw }}</td>
                        <td class="text-center">
                          <v-chip size="x-small" :color="entry.format === 'csv' ? 'success' : 'info'" variant="flat">
                            {{ entry.format === 'csv' ? 'CSV' : 'TXT' }}
                          </v-chip>
                        </td>
                        <td class="text-center">
                          <span class="text-caption text-medium-emphasis">{{ entry.license }}</span>
                        </td>
                        <td class="text-center">
                          <v-btn
                            size="x-small"
                            variant="tonal"
                            color="primary"
                            @click="pickPdmSample(entry.key)"
                          >
                            Use
                          </v-btn>
                        </td>
                        <td class="text-center">
                          <a
                            v-if="entry.sourceUrl"
                            :href="entry.sourceUrl"
                            target="_blank"
                            rel="noopener"
                            class="text-caption"
                          >
                            Info
                            <v-icon size="x-small">mdi-open-in-new</v-icon>
                          </a>
                          <span v-else class="text-caption text-medium-emphasis">—</span>
                        </td>
                      </tr>
                    </tbody>
                  </v-table>
                </div>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>

          <v-alert
            v-if="urlLoader.error"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-4"
            closable
            @click:close="urlLoader.error = ''"
          >
            {{ urlLoader.error }}
          </v-alert>
        </v-card>
      </v-col>
    </v-row>

    <!-- Data Preview -->
    <v-card v-if="dataPreview?.metadata" class="mt-6 pa-4">
      <div class="d-flex align-center mb-4">
        <h3 class="text-subtitle-1 font-weight-bold">Data Preview</h3>
        <v-chip
          v-if="dataPreview.metadata?.is_partition_preview"
          size="small"
          color="warning"
          variant="tonal"
          class="ml-3"
        >
          {{ dataPreview.metadata.filter?.category }}{{ dataPreview.metadata.filter?.label ? ' / ' + dataPreview.metadata.filter.label : ' / all labels' }}
        </v-chip>
        <v-spacer />
        <v-chip size="small" color="info" variant="flat">
          {{ dataPreview.metadata.total_rows.toLocaleString() }} rows
        </v-chip>
      </div>

      <!-- Column Selection Info -->
      <div v-if="dataPreview.metadata.sensor_columns?.length" class="d-flex align-center mb-2">
        <v-icon size="small" class="mr-1">mdi-table-column</v-icon>
        <span class="text-caption text-medium-emphasis">
          Click column headers to select/deselect signals for the pipeline.
        </span>
        <v-spacer />
        <v-chip size="x-small" color="primary" variant="flat" class="mr-1">
          {{ pipelineStore.selectedColumns.length }} / {{ dataPreview.metadata.sensor_columns.length }} sensors selected
        </v-chip>
        <v-btn size="x-small" variant="tonal" @click="selectAllColumns" class="mr-1">All</v-btn>
        <v-btn size="x-small" variant="tonal" @click="selectNoColumns">None</v-btn>
      </div>

      <v-data-table
        v-model:items-per-page="previewItemsPerPage"
        :headers="previewHeaders"
        :items="dataPreview.preview"
        :items-per-page-options="[10, 25, 50, 100]"
        density="compact"
        class="preview-table"
      >
        <!-- Custom column headers with checkboxes for sensor columns -->
        <template v-for="col in dataPreview.metadata.columns" :key="'header-'+col" #[`header.${col}`]="{ column }">
          <div
            v-if="col === timestampColumn"
            class="d-flex align-center"
          >
            <v-icon size="x-small" color="success" class="mr-1">mdi-lock</v-icon>
            <span>{{ column.title }}</span>
          </div>
          <div
            v-else-if="isSensorColumn(col)"
            class="d-flex align-center"
            style="cursor: pointer;"
            @click.stop="toggleColumn(col)"
          >
            <v-icon size="x-small" :color="pipelineStore.selectedColumns.includes(col) ? 'primary' : 'grey'" class="mr-1">
              {{ pipelineStore.selectedColumns.includes(col) ? 'mdi-checkbox-marked' : 'mdi-checkbox-blank-outline' }}
            </v-icon>
            <span :style="{ opacity: pipelineStore.selectedColumns.includes(col) ? 1 : 0.4 }">
              {{ column.title }}
            </span>
          </div>
          <span v-else class="text-medium-emphasis">{{ column.title }}</span>
        </template>

        <template #item.label="{ item }">
          <v-chip
            v-if="item.label"
            :color="item.label === 'anomaly' || item.label === '1' ? 'error' : 'success'"
            size="small"
            variant="flat"
          >
            {{ item.label }}
          </v-chip>
        </template>
      </v-data-table>

      <!-- Signal Visualization -->
      <v-expand-transition>
        <div v-if="showVisualization && canVisualize" class="mt-4">
          <v-card variant="outlined" class="pa-4">
            <div class="d-flex align-center mb-2 flex-wrap">
              <v-icon size="small" class="mr-1" color="primary">mdi-chart-line</v-icon>
              <span class="text-subtitle-2 font-weight-medium">
                Signal Visualization
              </span>
              <v-chip size="x-small" color="primary" variant="tonal" class="ml-2">
                {{ plottableColumns.length }} signal{{ plottableColumns.length === 1 ? '' : 's' }}
              </v-chip>
              <!-- Phase G — mode toggle. In label mode click-drag places
                   start/end lines instead of panning. Only offered for
                   single-CSV loads (labels sidecar is per-CSV). -->
              <v-btn-toggle
                v-if="labelModeAvailable"
                v-model="chartMode"
                mandatory
                variant="tonal"
                density="compact"
                divided
                class="ml-3"
              >
                <v-btn value="pan" size="small" :title="'Zoom / pan'">
                  <v-icon size="16" start>mdi-magnify</v-icon>
                  Zoom/Pan
                </v-btn>
                <v-btn value="label" size="small" :title="'Label mode'" color="warning">
                  <v-icon size="16" start>mdi-tag-plus-outline</v-icon>
                  Label
                </v-btn>
              </v-btn-toggle>
              <v-btn
                size="x-small"
                variant="text"
                prepend-icon="mdi-restore"
                class="ml-2"
                :disabled="!isZoomed"
                @click="resetZoom"
              >
                Reset Zoom
              </v-btn>
              <v-spacer />
              <span class="text-caption text-medium-emphasis">
                Plotting {{ dataPreview.preview.length.toLocaleString() }} of {{ dataPreview.metadata.total_rows.toLocaleString() }} rows
              </span>
            </div>
            <div
              class="chart-container"
              :style="{ height: '340px', position: 'relative', cursor: chartCursor, userSelect: 'none' }"
              @wheel="onChartWheel"
              @mousedown="onChartMouseDown"
              @mousemove="onChartMouseMove"
              @mouseup="onChartMouseUp"
              @mouseleave="onChartMouseUp"
              @dblclick="onChartDoubleClick"
              @click="onChartClick"
            >
              <Line ref="chartRef" :data="chartData" :options="chartOptions" :plugins="labelChartPlugins" />
            </div>
            <div class="text-caption text-medium-emphasis mt-1">
              <template v-if="isLabelMode">
                X-axis: {{ timestampColumn }} &nbsp;·&nbsp;
                <strong>Click</strong> to place start line, click again for end,
                type into the boxes below for precision, then <strong>Apply</strong>.
                Colored strip along the axis shows saved label ranges.
              </template>
              <template v-else>
                X-axis: {{ timestampColumn }} &nbsp;·&nbsp; <strong>Scroll</strong> to zoom, <strong>drag</strong> to pan, <strong>double-click</strong> to reset. Toggle signals via the column headers above.
              </template>
            </div>

            <!-- Phase G — placement panel (label mode only) -->
            <div v-if="isLabelMode" class="mt-3 label-placement-panel pa-3">
              <div class="d-flex align-center flex-wrap ga-3">
                <v-text-field
                  v-model.number="labelStart"
                  label="Start"
                  type="number"
                  step="0.01"
                  density="compact"
                  variant="outlined"
                  hide-details
                  style="max-width: 150px"
                  :placeholder="labelPlaceholderMin"
                />
                <v-text-field
                  v-model.number="labelEnd"
                  label="End"
                  type="number"
                  step="0.01"
                  density="compact"
                  variant="outlined"
                  hide-details
                  style="max-width: 150px"
                  :placeholder="labelPlaceholderMax"
                />
                <v-combobox
                  v-model="labelClass"
                  :items="knownClassNames"
                  label="Class"
                  density="compact"
                  variant="outlined"
                  hide-details
                  clearable
                  style="min-width: 180px; max-width: 280px"
                />
                <v-btn
                  color="primary"
                  variant="flat"
                  :disabled="!canApplyLabel"
                  @click="applyPendingLabel"
                >
                  <v-icon start size="16">mdi-check</v-icon>
                  {{ editingLabelIndex === null ? 'Apply' : 'Update' }}
                </v-btn>
                <v-btn variant="text" @click="cancelPendingLabel">
                  Cancel
                </v-btn>
              </div>
              <div v-if="labelValidationError" class="text-caption text-error mt-2">
                {{ labelValidationError }}
              </div>
            </div>
          </v-card>
        </div>
      </v-expand-transition>

      <!-- Phase G — Labels panel (visible whenever we have a single CSV
           loaded and either sidecar labels exist or user is in label
           mode). Renders below the chart, above "Load More". -->
      <div v-if="labelModeAvailable && showVisualization && (labels.length > 0 || isLabelMode)" class="mt-4">
        <v-card variant="outlined" class="pa-4">
          <div class="d-flex align-center mb-2">
            <v-icon size="small" class="mr-1" color="warning">mdi-tag-multiple-outline</v-icon>
            <span class="text-subtitle-2 font-weight-medium">Labels</span>
            <v-chip
              v-if="labels.length > 0"
              size="x-small"
              color="warning"
              variant="tonal"
              class="ml-2"
            >
              {{ labels.length }}
            </v-chip>
            <v-spacer />
            <span
              class="text-caption d-flex align-center mr-3"
              :class="hasUnsavedLabels ? 'text-warning font-weight-bold' : 'text-success'"
            >
              <v-icon size="12" class="mr-1">
                {{ hasUnsavedLabels ? 'mdi-alert-circle' : 'mdi-check-circle' }}
              </v-icon>
              {{ hasUnsavedLabels ? 'Unsaved changes' : 'Saved' }}
            </span>
            <!-- Save button is ALWAYS visible when there's at least one
                 label so users know where to click. Highlights (elevated
                 primary) while unsaved, greys out once saved. This is the
                 explicit action; auto-save-on-Load-More is a bonus for
                 multi-batch files where the label list may be long. -->
            <v-btn
              v-if="labels.length > 0"
              size="small"
              :variant="hasUnsavedLabels ? 'elevated' : 'tonal'"
              :color="hasUnsavedLabels ? 'primary' : 'success'"
              :loading="savingLabels"
              :disabled="!hasUnsavedLabels && !savingLabels"
              prepend-icon="mdi-content-save-outline"
              @click="saveLabels()"
            >
              {{ hasUnsavedLabels ? 'Save' : 'Saved' }}
            </v-btn>
          </div>

          <div v-if="labels.length === 0" class="text-caption text-medium-emphasis py-2">
            No labels yet. Switch to Label mode above, click on the chart to
            place start and end lines, type a class name, then Apply.
          </div>
          <v-list v-else density="compact" class="pa-0">
            <v-list-item
              v-for="(l, idx) in sortedLabels"
              :key="`${l.from}-${l.to}-${idx}`"
              class="px-2 label-row"
              :active="editingLabelIndex === idx"
              @click="previewLabel(idx)"
            >
              <template #prepend>
                <span
                  class="label-dot"
                  :style="{ background: classColor(l.class) }"
                />
              </template>
              <v-list-item-title class="d-flex align-center">
                <strong class="mr-2">{{ l.class }}</strong>
                <code class="text-caption">{{ l.from.toFixed(2) }} → {{ l.to.toFixed(2) }}</code>
                <span class="text-caption text-medium-emphasis ml-2">
                  ({{ (l.to - l.from).toFixed(2) }} s)
                </span>
              </v-list-item-title>
              <template #append>
                <v-btn
                  icon
                  size="x-small"
                  variant="text"
                  :title="'Edit'"
                  @click="startEditLabel(idx)"
                >
                  <v-icon size="16">mdi-pencil</v-icon>
                </v-btn>
                <v-btn
                  icon
                  size="x-small"
                  variant="text"
                  color="error"
                  :title="'Delete'"
                  @click="deleteLabel(idx)"
                >
                  <v-icon size="16">mdi-delete-outline</v-icon>
                </v-btn>
              </template>
            </v-list-item>
          </v-list>

          <div class="d-flex align-center mt-2 text-caption text-medium-emphasis">
            <span>
              Coverage {{ labelCoverageDisplay }} ·
              {{ labels.length }} label{{ labels.length === 1 ? '' : 's' }} ·
              {{ knownClassNames.length }} class{{ knownClassNames.length === 1 ? '' : 'es' }}
            </span>
            <v-spacer />
            <span v-if="labelsLastSavedAt" class="text-caption">
              Last saved {{ labelsLastSavedDisplay }}
            </span>
          </div>
        </v-card>
      </div>

      <!-- Load More Info -->
      <div class="d-flex align-center justify-space-between pa-2 mt-2">
        <span class="text-caption text-medium-emphasis">
          Loaded {{ dataPreview.preview.length }} of {{ dataPreview.metadata.total_rows.toLocaleString() }} total rows
        </span>
        <div class="d-flex align-center">
          <v-btn
            variant="tonal"
            size="small"
            :color="showVisualization ? 'primary' : 'secondary'"
            :prepend-icon="showVisualization ? 'mdi-chart-line-variant' : 'mdi-chart-line'"
            :disabled="!canVisualize"
            class="mr-2"
            @click="showVisualization = !showVisualization"
          >
            {{ showVisualization ? 'Hide Chart' : 'Visualize' }}
          </v-btn>
          <v-btn
            v-if="dataPreview.preview.length < dataPreview.metadata.total_rows && dataPreview.preview.length < maxPreviewRows"
            variant="tonal"
            size="small"
            color="primary"
            :loading="loadingMore || savingLabels"
            @click="loadMorePreviewWithSave"
          >
            <v-icon start>mdi-plus</v-icon>
            {{ hasUnsavedLabels ? 'Save Labels + Load More' : 'Load More Rows' }}
          </v-btn>
          <span v-else-if="dataPreview.preview.length >= maxPreviewRows" class="text-caption text-warning">
            Preview limit reached ({{ maxPreviewRows }} rows max)
          </span>
        </div>
      </div>

      <!-- Metadata -->
      <v-row class="mt-4">
        <v-col cols="12" sm="6" md="3">
          <div class="text-caption text-medium-emphasis">Format</div>
          <div class="font-weight-medium">{{ dataPreview.metadata.format }}</div>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <div class="text-caption text-medium-emphasis">Sensor Columns</div>
          <div class="font-weight-medium">{{ dataPreview.metadata.sensor_columns?.length || 0 }}</div>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <div class="text-caption text-medium-emphasis">Labels</div>
          <div class="font-weight-medium">
            <!-- Prefer classes from the sidecar labeler if present (the
                 authoritative source when a user has hand-labeled ranges),
                 else fall back to filename-convention labels from
                 dataPreview metadata (Edge Impulse style: idle_2024.csv). -->
            <template v-if="knownClassNames.length > 0">
              {{ knownClassNames.join(', ') }}
              <span class="text-caption text-medium-emphasis ml-1">
                ({{ labels.length }} range{{ labels.length === 1 ? '' : 's' }})
              </span>
            </template>
            <template v-else>
              {{ dataPreview.metadata.labels?.join(', ') || 'None' }}
            </template>
          </div>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <div class="text-caption text-medium-emphasis">
            {{ dataPreview.metadata.is_folder || dataPreview.metadata.is_multi_csv ? 'Samples (files)' : 'Session ID' }}
          </div>
          <div class="font-weight-medium text-truncate">
            {{ dataPreview.metadata.is_multi_csv
              ? `${dataPreview.metadata.total_samples} files`
              : dataPreview.metadata.is_folder
                ? `${dataPreview.metadata.total_samples}${dataPreview.metadata.training_samples != null ? ` (Train: ${dataPreview.metadata.training_samples}, Test: ${dataPreview.metadata.testing_samples || 0})` : ''}`
                : dataPreview.session_id
            }}
          </div>
        </v-col>
      </v-row>
    </v-card>

    <!-- Actions -->
    <div class="d-flex justify-end mt-6">
      <v-btn
        color="primary"
        size="large"
        :disabled="!canProceed"
        :loading="loading || loadingFull"
        @click="proceedToWindowing"
      >
        <template v-if="loadingFull">
          Loading Full Dataset...
        </template>
        <template v-else>
          Continue to Windowing
          <v-icon end>mdi-arrow-right</v-icon>
        </template>
      </v-btn>
    </div>

    <!-- Upload Dialog -->
    <v-dialog v-model="showUploadDialog" max-width="600" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2">mdi-upload</v-icon>
          Upload Dataset
          <v-spacer />
          <v-btn icon variant="text" @click="closeUploadDialog">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-tabs
          v-model="uploadTab"
          color="primary"
          density="compact"
          grow
        >
          <v-tab value="files">
            <v-icon start size="small">mdi-file-multiple-outline</v-icon>
            Files
          </v-tab>
          <v-tab value="folder">
            <v-icon start size="small">mdi-folder-upload-outline</v-icon>
            Folder
          </v-tab>
        </v-tabs>

        <v-card-text>
          <v-window v-model="uploadTab">
            <!-- FILES TAB -->
            <v-window-item value="files">
              <!-- Drag and Drop Zone -->
              <div
                class="upload-dropzone"
                :class="{ 'drag-over': isDragging, 'has-files': uploadFiles.length > 0 }"
                @dragover.prevent="isDragging = true"
                @dragleave.prevent="isDragging = false"
                @drop.prevent="handleDrop"
                @click="triggerFileInput"
              >
                <input
                  ref="fileInput"
                  type="file"
                  :accept="allowedFileTypes"
                  multiple
                  hidden
                  @change="handleFileSelect"
                />

                <template v-if="uploadFiles.length === 0">
                  <v-icon size="48" color="primary" class="mb-2">mdi-cloud-upload</v-icon>
                  <div class="text-body-1 font-weight-medium">
                    Drag and drop files here
                  </div>
                  <div class="text-caption text-medium-emphasis">
                    or click to browse
                  </div>
                  <div class="text-caption text-medium-emphasis mt-2">
                    Supported: CSV, JSON, CBOR, text (.txt/.tsv/.dat/.log) — max 100 MB
                  </div>
                </template>

                <template v-else>
                  <v-icon size="32" color="success" class="mb-2">mdi-check-circle</v-icon>
                  <div class="text-body-1 font-weight-medium">
                    {{ uploadFiles.length }} file(s) selected
                  </div>
                </template>
              </div>

              <!-- Selected Files List -->
              <v-list v-if="uploadFiles.length > 0" density="compact" class="mt-4">
                <v-list-subheader>Selected Files</v-list-subheader>
                <v-list-item
                  v-for="(file, index) in uploadFiles"
                  :key="index"
                  class="upload-file-item"
                >
                  <template #prepend>
                    <v-icon :color="getFileTypeColor(file.name)">
                      {{ getFileTypeIcon(file.name) }}
                    </v-icon>
                  </template>

                  <v-list-item-title>{{ file.name }}</v-list-item-title>
                  <v-list-item-subtitle>{{ formatFileSize(file.size) }}</v-list-item-subtitle>

                  <template #append>
                    <v-btn
                      icon
                      variant="text"
                      size="small"
                      color="error"
                      @click.stop="removeFile(index)"
                    >
                      <v-icon>mdi-close</v-icon>
                    </v-btn>
                  </template>
                </v-list-item>
              </v-list>
            </v-window-item>

            <!-- FOLDER TAB -->
            <v-window-item value="folder">
              <div
                class="upload-dropzone"
                :class="{ 'has-files': folderUploadFiles.length > 0 }"
                @click="triggerFolderInput"
              >
                <input
                  ref="folderInput"
                  type="file"
                  webkitdirectory
                  directory
                  multiple
                  hidden
                  @change="handleFolderSelect"
                />

                <template v-if="folderUploadFiles.length === 0">
                  <v-icon size="48" color="primary" class="mb-2">mdi-folder-upload</v-icon>
                  <div class="text-body-1 font-weight-medium">
                    Click to select a folder
                  </div>
                  <div class="text-caption text-medium-emphasis">
                    Nested directory structure will be preserved
                  </div>
                  <div class="text-caption text-medium-emphasis mt-2">
                    Supported: CSV, JSON, CBOR, text (.txt/.tsv/.dat/.log) — max 100 MB per file
                  </div>
                </template>

                <template v-else>
                  <v-icon size="32" color="success" class="mb-2">mdi-folder-check</v-icon>
                  <div class="text-body-1 font-weight-medium">
                    {{ folderUploadFiles.length }} files, {{ folderTopLevelCount }} folders selected
                  </div>
                  <div class="text-caption text-medium-emphasis mt-1">
                    Click to choose a different folder
                  </div>
                </template>
              </div>

              <v-list v-if="folderUploadFiles.length > 0" density="compact" class="mt-4 folder-upload-list">
                <v-list-subheader>Files ({{ folderUploadFiles.length }})</v-list-subheader>
                <v-list-item
                  v-for="(entry, index) in folderUploadFiles"
                  :key="index"
                  class="upload-file-item"
                >
                  <template #prepend>
                    <v-icon :color="getFileTypeColor(entry.file.name)">
                      {{ getFileTypeIcon(entry.file.name) }}
                    </v-icon>
                  </template>

                  <!-- Show basename as the title so a truncated container doesn't hide
                       the only bit of the name that varies between rows. Full path
                       (parent chain from webkitRelativePath) shown smaller below. -->
                  <v-list-item-title>{{ entry.file.name }}</v-list-item-title>
                  <v-list-item-subtitle>
                    <span v-if="entry.relative_path && entry.relative_path !== entry.file.name" class="text-caption text-medium-emphasis">
                      {{ entry.relative_path.slice(0, entry.relative_path.length - entry.file.name.length).replace(/[/\\]$/, '') || '(root)' }}
                      <span class="mx-1">·</span>
                    </span>
                    {{ formatFileSize(entry.file.size) }}
                  </v-list-item-subtitle>
                </v-list-item>
              </v-list>
            </v-window-item>
          </v-window>

          <!-- Upload Progress -->
          <v-progress-linear
            v-if="uploading"
            :model-value="uploadProgress"
            color="primary"
            class="mt-4"
            height="8"
            rounded
          />

          <!-- Upload Error -->
          <v-alert
            v-if="uploadError"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-4"
            closable
            @click:close="uploadError = ''"
          >
            {{ uploadError }}
          </v-alert>

          <!-- Upload Success -->
          <v-alert
            v-if="uploadSuccess"
            type="success"
            variant="tonal"
            density="compact"
            class="mt-4"
          >
            {{ uploadSuccess }}
          </v-alert>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="closeUploadDialog">
            Cancel
          </v-btn>
          <v-btn
            v-if="uploadTab === 'files'"
            color="primary"
            variant="flat"
            :disabled="uploadFiles.length === 0 || uploading"
            :loading="uploading"
            @click="uploadSelectedFiles"
          >
            <v-icon start>mdi-upload</v-icon>
            Upload {{ uploadFiles.length }} File(s)
          </v-btn>
          <v-btn
            v-else
            color="primary"
            variant="flat"
            :disabled="folderUploadFiles.length === 0 || uploading"
            :loading="uploading"
            @click="uploadSelectedFiles"
          >
            <v-icon start>mdi-folder-upload</v-icon>
            Upload {{ folderUploadFiles.length }} File(s)
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Validation Error Dialog -->
    <v-dialog v-model="showValidationError" max-width="520">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="error">mdi-alert-circle-outline</v-icon>
          Can't Load This File
          <v-spacer />
          <v-btn icon variant="text" size="small" @click="showValidationError = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-card-text v-if="validationError">
          <div class="mb-3">
            <v-chip
              size="small"
              color="error"
              variant="tonal"
              label
            >
              {{ validationError.code }}
            </v-chip>
          </div>

          <div class="text-body-1 mb-4">
            {{ validationError.message }}
          </div>

          <v-alert
            v-if="validationError.hint"
            type="info"
            variant="tonal"
            density="compact"
            icon="mdi-lightbulb-outline"
          >
            {{ validationError.hint }}
          </v-alert>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn color="primary" variant="flat" @click="showValidationError = false">
            OK
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Dataset Format Guide Dialog -->
    <v-dialog v-model="showFormatGuide" max-width="720" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="info" class="mr-2">mdi-information-outline</v-icon>
          Dataset Format Guide
          <v-spacer />
          <v-btn icon variant="text" @click="showFormatGuide = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-card-text style="max-height: 70vh;">
          <p class="text-body-2 text-medium-emphasis mb-4">
            Prepare your data with these rules and the pipeline will load it cleanly.
            If loading fails, the error dialog will name the exact issue.
          </p>

          <!-- Supported formats -->
          <div class="format-section">
            <div class="format-section-title">
              <v-icon size="small" color="info" class="mr-1">mdi-file-multiple</v-icon>
              Supported Formats
            </div>
            <v-list density="compact" class="pa-0 bg-transparent">
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="success">mdi-file-delimited</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>CSV</strong> — header row + numeric sensor columns. Multiple CSVs can be selected as one dataset (columns must match).</v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="info">mdi-code-json</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>Edge Impulse JSON</strong> — standard EI export with sensors + values arrays.</v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="secondary">mdi-file-code</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>Edge Impulse CBOR</strong> — folder with <code>training/</code> and <code>testing/</code> subfolders; classes auto-detected from filenames.</v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="secondary">mdi-file-code</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>CiRA CBOR</strong> — folder with <code>train/</code> and <code>test/</code> subfolders.</v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="primary">mdi-file-document-outline</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>Text file</strong> — any delimited text file (<code>.txt</code>, <code>.tsv</code>, <code>.dat</code>, <code>.log</code>). The Text Import wizard auto-detects the delimiter and lets you tweak header row / skip rows with a live preview before loading.</v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0">
                <template #prepend><v-icon size="small" color="primary">mdi-cloud-download-outline</v-icon></template>
                <v-list-item-title class="text-body-2"><strong>Load from URL</strong> — fetch a CSV / text file over <code>https://</code> (max 100 MB, streamed to memory only). Includes a hardcoded catalog of factory / predictive-maintenance datasets (AI4I 2020, NASA CMAPSS turbofan, Azure PdM) with one-click samples.</v-list-item-title>
              </v-list-item>
            </v-list>
          </div>

          <v-divider class="my-3" />

          <!-- CSV column rules -->
          <div class="format-section">
            <div class="format-section-title">
              <v-icon size="small" color="success" class="mr-1">mdi-table-column</v-icon>
              CSV Column Rules
            </div>

            <div class="format-subsection">
              <div class="format-sub-title">Timestamp column (X-axis)</div>
              <p class="text-body-2 mb-1">Detected by name (case-insensitive, exact match):</p>
              <div class="format-chips">
                <v-chip v-for="p in timePatternExact" :key="p" size="x-small" variant="tonal" color="success">{{ p }}</v-chip>
              </div>
              <p class="text-body-2 mt-2 mb-1">Also accepted (prefix match):</p>
              <div class="format-chips">
                <v-chip size="x-small" variant="tonal" color="success">time…</v-chip>
                <v-chip size="x-small" variant="tonal" color="success">timestamp…</v-chip>
                <span class="text-caption text-medium-emphasis ml-2">(e.g. <code>Time (seconds)</code>, <code>timestamp_ms</code>)</span>
              </div>
              <v-alert type="info" variant="tonal" density="compact" class="mt-2 text-caption">
                <strong>Fallback:</strong> if no column matches, the <strong>first column</strong> is used automatically — but only if it contains numeric values (e.g. Unix epoch, day counter).
              </v-alert>
              <v-alert type="success" variant="tonal" density="compact" class="mt-2 text-caption">
                <strong>Datetime strings are auto-converted.</strong> A column like <code>Timestamp</code> with values <code>2025-04-01 08:00:00</code> gets parsed and rewritten to <strong>seconds since the first sample</strong> (0, 60, 120…), so downstream stages work.
              </v-alert>
            </div>

            <div class="format-subsection mt-3">
              <div class="format-sub-title">Label column (optional)</div>
              <p class="text-body-2 mb-1">Detected by name (case-insensitive):</p>
              <div class="format-chips">
                <v-chip v-for="p in labelPatterns" :key="p" size="x-small" variant="tonal" color="warning">{{ p }}</v-chip>
              </div>
              <p class="text-caption text-medium-emphasis mt-2">Values can be strings (<code>anomaly</code>, <code>normal</code>) or integers (<code>0</code>, <code>1</code>). Leave out for unlabelled data.</p>
            </div>

            <div class="format-subsection mt-3">
              <div class="format-sub-title">Sensor columns</div>
              <p class="text-body-2">Every remaining <strong>numeric</strong> column is a sensor. Text columns are ignored unless one of them matches a label name. You need at least one numeric sensor column.</p>
            </div>
          </div>

          <v-divider class="my-3" />

          <!-- Common issues -->
          <div class="format-section">
            <div class="format-section-title">
              <v-icon size="small" color="error" class="mr-1">mdi-alert-circle-outline</v-icon>
              Common Errors &amp; Fixes
            </div>
            <v-list density="compact" class="pa-0 bg-transparent">
              <v-list-item v-for="err in commonErrors" :key="err.code" class="px-0">
                <v-list-item-title class="text-body-2">
                  <v-chip size="x-small" color="error" variant="tonal" class="mr-2">{{ err.code }}</v-chip>
                  <strong>{{ err.title }}</strong>
                </v-list-item-title>
                <v-list-item-subtitle class="text-caption mt-1">{{ err.hint }}</v-list-item-subtitle>
              </v-list-item>
            </v-list>
          </div>

          <v-divider class="my-3" />

          <!-- Quick example -->
          <div class="format-section">
            <div class="format-section-title">
              <v-icon size="small" color="primary" class="mr-1">mdi-lightbulb-outline</v-icon>
              Minimal CSV Example
            </div>
            <pre class="format-example">time,temperature,vibration,label
0.0,45.3,0.87,normal
0.1,45.4,0.65,normal
0.2,45.6,1.05,anomaly</pre>
            <p class="text-caption text-medium-emphasis mt-1">
              Or use <code>index</code> instead of <code>time</code> if you don't have real timestamps.
            </p>
          </div>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn color="primary" variant="flat" @click="showFormatGuide = false">
            Got It
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Record Sensors Dialog -->
    <v-dialog v-model="showRecordDialog" max-width="550" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="secondary">mdi-access-point</v-icon>
          Record System Sensors
          <v-spacer />
          <v-btn icon variant="text" @click="closeRecordDialog" :disabled="recording">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-card-text>
          <!-- Not Recording: Configuration -->
          <template v-if="!recording && !recordingDone">
            <div class="text-body-2 text-medium-emphasis mb-4">
              Record system sensors (CPU, RAM, disk, network{{ recordHasGpu ? ', GPU' : '' }}) as time series CSV for classification training.
            </div>

            <!-- Dataset Preset -->
            <div class="font-weight-medium mb-2">Dataset Preset</div>
            <v-radio-group v-model="recordMode" hide-details class="mb-4">
              <v-radio value="network_traffic">
                <template #label>
                  <div>
                    <div class="font-weight-medium">Network Traffic</div>
                    <div class="text-caption text-medium-emphasis">
                      Manual: Idle → Web Browsing → Video Streaming → File Download
                    </div>
                  </div>
                </template>
              </v-radio>
              <v-radio value="disk_io">
                <template #label>
                  <div>
                    <div class="font-weight-medium">Disk I/O Patterns</div>
                    <div class="text-caption text-medium-emphasis">
                      Auto: Idle → Sequential Read → Random Read → Write Heavy
                    </div>
                  </div>
                </template>
              </v-radio>
              <v-radio value="manual">
                <template #label>
                  <div>
                    <div class="font-weight-medium">Manual Label</div>
                    <div class="text-caption text-medium-emphasis">
                      Record with a single custom label
                    </div>
                  </div>
                </template>
              </v-radio>
            </v-radio-group>

            <!-- Manual label -->
            <v-text-field
              v-if="recordMode === 'manual'"
              v-model="recordLabel"
              label="Label"
              density="compact"
              variant="outlined"
              class="mb-4"
              hide-details
            />

            <!-- Duration & Rate -->
            <v-row>
              <v-col cols="6">
                <v-text-field
                  v-model.number="recordDuration"
                  label="Duration (seconds)"
                  type="number"
                  :min="10"
                  :max="600"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
              <v-col cols="6">
                <v-text-field
                  v-model.number="recordRate"
                  label="Sample Rate (Hz)"
                  type="number"
                  :min="1"
                  :max="10"
                  density="compact"
                  variant="outlined"
                  hide-details
                />
              </v-col>
            </v-row>

            <div class="text-caption text-medium-emphasis mt-2">
              Total samples: {{ recordDuration * recordRate }} ({{ recordDuration }}s at {{ recordRate }} Hz)
            </div>

            <!-- Optional filename -->
            <v-text-field
              v-model="recordFilename"
              label="Filename (optional)"
              density="compact"
              variant="outlined"
              class="mt-4"
              hide-details
              placeholder="Auto-generated if empty"
            />
          </template>

          <!-- Recording in progress -->
          <template v-if="recording">
            <div class="text-center py-4">
              <v-progress-circular
                :model-value="recordProgress"
                :size="100"
                :width="8"
                color="secondary"
                class="mb-4"
              >
                <span class="text-h6">{{ Math.round(recordProgress) }}%</span>
              </v-progress-circular>

              <div class="text-body-1 font-weight-medium mb-1">Recording...</div>
              <div class="text-body-2 text-medium-emphasis mb-2">
                {{ recordElapsed.toFixed(0) }}s / {{ recordDuration }}s
              </div>

              <v-chip
                v-if="recordCurrentPhase"
                :color="recordCurrentPhase === 'idle' ? 'success' : 'info'"
                variant="flat"
                size="small"
                class="mb-2"
              >
                Phase: {{ recordCurrentPhase }}
              </v-chip>

              <!-- Phase instruction for network_traffic mode -->
              <v-alert
                v-if="recordMode === 'network_traffic' && recordCurrentPhase"
                :type="recordCurrentPhase === 'idle' ? 'success' : 'info'"
                variant="tonal"
                density="compact"
                class="mt-3 text-left"
              >
                <div class="font-weight-medium mb-1">{{ phaseInstruction.title }}</div>
                <div class="text-body-2">{{ phaseInstruction.detail }}</div>
              </v-alert>

              <!-- Auto-generation notice for disk_io mode -->
              <v-alert
                v-if="recordMode === 'disk_io' && recordCurrentPhase"
                type="info"
                variant="tonal"
                density="compact"
                class="mt-3 text-left"
              >
                <div class="font-weight-medium">Auto-generating disk activity...</div>
                <div class="text-body-2">{{ recordCurrentPhase === 'idle' ? 'Baseline idle measurement' : `Running: ${recordCurrentPhase.replace('_', ' ')}` }}</div>
              </v-alert>
            </div>

            <v-progress-linear
              :model-value="recordProgress"
              color="secondary"
              height="6"
              rounded
              class="mt-2"
            />
          </template>

          <!-- Recording complete -->
          <template v-if="recordingDone">
            <div class="text-center py-4">
              <v-icon size="64" color="success" class="mb-3">mdi-check-circle</v-icon>
              <div class="text-body-1 font-weight-medium mb-1">Recording Complete</div>
              <div class="text-body-2 text-medium-emphasis mb-2">
                {{ recordTotalSamples }} samples recorded
              </div>
              <v-chip color="info" variant="tonal" size="small">
                {{ recordOutputFilename }}
              </v-chip>
            </div>
          </template>

          <!-- Recording error -->
          <v-alert v-if="recordError" type="error" variant="tonal" density="compact" class="mt-4">
            {{ recordError }}
          </v-alert>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <template v-if="!recording && !recordingDone">
            <v-btn variant="text" @click="closeRecordDialog">Cancel</v-btn>
            <v-btn
              color="secondary"
              variant="flat"
              @click="startRecording"
              :loading="recordStarting"
            >
              <v-icon start>mdi-record-circle</v-icon>
              Start Recording
            </v-btn>
          </template>
          <template v-if="recording">
            <v-btn
              color="warning"
              variant="tonal"
              @click="stopRecording"
            >
              <v-icon start>mdi-stop</v-icon>
              Stop Early
            </v-btn>
          </template>
          <template v-if="recordingDone">
            <v-btn variant="text" @click="closeRecordDialog">Close</v-btn>
            <v-btn
              color="primary"
              variant="flat"
              @click="loadRecordedData"
            >
              <v-icon start>mdi-play</v-icon>
              Load This Data
            </v-btn>
          </template>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Text Import Dialog -->
    <v-dialog v-model="showTextImport" max-width="820" scrollable persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="primary">mdi-file-document-outline</v-icon>
          Text Import
          <v-spacer />
          <v-btn icon variant="text" @click="cancelTextImport" :disabled="textImportLoading">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>

        <v-card-text style="max-height: 70vh;">
          <div v-if="textImportFile" class="text-caption text-medium-emphasis mb-3">
            <v-icon size="small" class="mr-1">mdi-file</v-icon>
            {{ textImportFile.name }}
            <span v-if="textImportDetectedDelimiter" class="ml-2">
              &middot; auto-detected delimiter:
              <code>{{ delimiterDisplay(textImportDetectedDelimiter) }}</code>
            </span>
          </div>

          <v-row>
            <v-col cols="12" md="6">
              <div class="text-subtitle-2 font-weight-medium mb-2">Delimiter</div>
              <v-radio-group
                v-model="textImportSettings.delimiter"
                density="compact"
                hide-details
                inline
              >
                <v-radio :value="','" label="Comma" />
                <v-radio :value="'\t'" label="Tab" />
                <v-radio :value="';'" label="Semicolon" />
                <v-radio :value="' '" label="Space" />
                <v-radio :value="'|'" label="Pipe" />
                <v-radio :value="'other'" label="Other" />
              </v-radio-group>

              <v-text-field
                v-if="textImportSettings.delimiter === 'other'"
                v-model="textImportSettings.delimiterOther"
                label="Custom delimiter (single character)"
                maxlength="1"
                density="compact"
                variant="outlined"
                class="mt-2"
                hide-details
              />
            </v-col>

            <v-col cols="12" md="3">
              <v-text-field
                v-model.number="textImportSettings.headerRow"
                label="Header row"
                type="number"
                min="0"
                density="compact"
                variant="outlined"
                hint="1-based. Use 0 for headerless."
                persistent-hint
              />
            </v-col>

            <v-col cols="12" md="3">
              <v-text-field
                v-model.number="textImportSettings.skipRows"
                label="Skip N rows from top"
                type="number"
                min="0"
                density="compact"
                variant="outlined"
                hint="Applied before the header row."
                persistent-hint
              />
            </v-col>
          </v-row>

          <v-divider class="my-4" />

          <div class="d-flex align-center mb-2">
            <div class="text-subtitle-2 font-weight-medium">Preview</div>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">
              First {{ Math.min(20, textImportPreview.rows.length) }} data rows
            </span>
          </div>

          <div v-if="textImportRawLines.length < 2" class="text-body-2 text-medium-emphasis pa-4 text-center">
            Not enough data to preview.
          </div>
          <div v-else-if="textImportPreview.headers.length === 0" class="text-body-2 text-warning pa-4 text-center">
            Header row is beyond the available lines. Try a smaller header row or fewer skipped rows.
          </div>
          <div v-else class="text-import-preview-wrap">
            <v-table density="compact" class="text-import-preview">
              <thead>
                <tr>
                  <th v-for="(h, i) in textImportPreview.headers" :key="'th-' + i">
                    {{ h }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, ri) in textImportPreview.rows" :key="'tr-' + ri">
                  <td v-for="(cell, ci) in row" :key="'td-' + ri + '-' + ci">
                    {{ cell }}
                  </td>
                </tr>
              </tbody>
            </v-table>
          </div>

          <v-alert
            v-if="textImportError"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-4"
            closable
            @click:close="textImportError = ''"
          >
            {{ textImportError }}
          </v-alert>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="cancelTextImport" :disabled="textImportLoading">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :disabled="!textImportCanImport"
            :loading="textImportLoading"
            @click="confirmTextImport"
          >
            <v-icon start>mdi-import</v-icon>
            Import
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete Confirmation Dialog (Admin only) -->
    <v-dialog v-model="showDeleteDialog" max-width="450" persistent>
      <v-card>
        <v-card-title class="d-flex align-center text-error">
          <v-icon class="mr-2" color="error">mdi-alert-circle</v-icon>
          Confirm Delete
        </v-card-title>

        <v-card-text v-if="itemToDelete">
          <p class="mb-3">
            Are you sure you want to delete this {{ itemToDelete.is_dir ? 'folder' : 'file' }}?
          </p>
          <v-alert type="warning" variant="tonal" density="compact" class="mb-3">
            <div class="font-weight-medium">{{ itemToDelete.name }}</div>
            <div class="text-caption">{{ itemToDelete.path }}</div>
          </v-alert>
          <p v-if="itemToDelete.is_dir" class="text-error text-body-2">
            <v-icon size="small" class="mr-1">mdi-alert</v-icon>
            This will delete the folder and ALL its contents!
          </p>
          <p class="text-body-2 text-medium-emphasis">
            This action cannot be undone.
          </p>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="cancelDelete" :disabled="deleting">
            Cancel
          </v-btn>
          <v-btn
            color="error"
            variant="flat"
            :loading="deleting"
            @click="executeDelete"
          >
            <v-icon start>mdi-delete</v-icon>
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Full File Manager (Manage Files) -->
    <FileManagerDialog
      v-model="showFileManager"
      :initial-path="currentPath"
      @refresh-requested="loadFolders"
    />

    <!-- Cross-sensor JOIN alignment picker (only opens for multi-folder baskets) -->
    <CsvMergeSettingsDialog
      v-model="mergeDialogOpen"
      :sensors="sensorsInBasket"
      @confirm="onMergeDialogConfirm"
      @cancel="onMergeDialogCancel"
    />

    <!-- "Load all sensors" date picker + preview -->
    <v-dialog v-model="loadAllDialogOpen" max-width="560" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="secondary" class="mr-2">mdi-source-branch-plus</v-icon>
          <span>Load all sensors for this machine</span>
        </v-card-title>
        <v-card-subtitle class="text-body-2 mt-2" style="white-space: normal">
          Picks one CSV file per child sensor folder on the chosen date.
          Sensors without a file for that date are shown but excluded.
        </v-card-subtitle>
        <v-card-text>
          <v-text-field
            v-model="loadAllDate"
            type="date"
            label="Date"
            variant="outlined"
            density="compact"
            :max="new Date().toISOString().slice(0,10)"
            @update:model-value="fetchSensorFilesForDate"
            hide-details
            class="mb-4"
          />
          <div v-if="loadingSensorFiles" class="text-center py-4">
            <v-progress-circular indeterminate color="secondary" size="24" />
          </div>
          <div v-else-if="loadAllPreview.length === 0" class="text-center text-medium-emphasis py-4">
            No sensor folders found under this path.
          </div>
          <v-list v-else density="compact" class="pa-0">
            <v-list-item
              v-for="entry in loadAllPreview"
              :key="entry.sensor"
              :class="{ 'text-medium-emphasis': !entry.exists }"
            >
              <template #prepend>
                <v-icon :color="entry.exists ? 'success' : 'grey'" size="18">
                  {{ entry.exists ? 'mdi-check-circle' : 'mdi-close-circle-outline' }}
                </v-icon>
              </template>
              <v-list-item-title>
                {{ entry.sensor }}
                <span v-if="!entry.exists" class="text-caption ml-2">(no file for that date)</span>
              </v-list-item-title>
              <v-list-item-subtitle class="text-caption">
                {{ entry.file }}
              </v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="loadAllDialogOpen = false">Cancel</v-btn>
          <v-btn
            color="secondary"
            variant="flat"
            :disabled="!loadAllPreview.some(x => x.exists)"
            @click="confirmLoadAllSensors"
          >
            Add {{ loadAllPreview.filter(x => x.exists).length }} to basket
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import { useAuthStore } from '@/stores/auth'
import PipelineStepper from '@/components/PipelineStepper.vue'
import MachinePipelineBanner from '@/components/MachinePipelineBanner.vue'
import FileManagerDialog from '@/components/FileManagerDialog.vue'
import CsvMergeSettingsDialog from '@/components/CsvMergeSettingsDialog.vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import api from '@/services/api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const channelColors = [
  '#6366F1',
  '#22D3EE',
  '#F59E0B',
  '#10B981',
  '#EF4444',
  '#8B5CF6',
  '#EC4899',
  '#14B8A6'
]

interface FileItem {
  name: string
  path: string
  is_dir: boolean
  extension: string | null
  size: number | null
  file_type: string | null
}

const router = useRouter()
const route = useRoute()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()
const authStore = useAuthStore()

const selectedFormat = ref('csv')
const currentPath = ref('')
const basePath = ref<string | null>(null) // The root path user has access to
const currentItems = ref<FileItem[]>([])
const selectedFile = ref<FileItem | null>(null)
const selectedFiles = ref<FileItem[]>([])  // Multi-select for CSV
const multiCsvError = ref<string | null>(null)

// Cross-sensor JOIN state — set by CsvMergeSettingsDialog when the basket
// spans multiple folders. Null values mean the backend uses its defaults
// (auto-detect single-folder → concat, cross-folder → exact JOIN).
const mergeDialogOpen = ref(false)
const mergeAlignment = ref<'exact' | 'nearest' | 'resample' | null>(null)
const mergeToleranceMs = ref<number | null>(null)
const mergeResampleHz = ref<number | null>(null)
let pendingLoadAction: null | (() => Promise<void>) = null

// Load-all-sensors state (file browser "Load All" button)
const loadAllDialogOpen = ref(false)
const loadAllDate = ref<string>(new Date().toISOString().slice(0, 10))
const loadAllPreview = ref<Array<{sensor: string; file: string; path: string; exists: boolean}>>([])
const loadingSensorFiles = ref(false)
const dataPreview = ref<any>(null)
const loading = ref(false)
const loadingFolders = ref(false)
const loadingMore = ref(false)
const loadingFull = ref(false)
const previewItemsPerPage = ref(10)
const maxPreviewRows = 50000

const showVisualization = ref(false)
const chartRef = ref<any>(null)
const xMin = ref<number | null>(null)
const xMax = ref<number | null>(null)
const panCursor = ref<'grab' | 'grabbing'>('grab')
let isPanning = false
let panStartClientX = 0
let panStartMin = 0
let panStartMax = 0

// ── Phase G — Label mode state ──────────────────────────────────────────
// Only enabled for single-CSV loads (labels sidecar is per-CSV). Cross-
// sensor JOIN loads are excluded — the sidecar path model doesn't fit a
// synthesized combined dataset.
interface LabelEntry {
  from: number
  to: number
  class: string
}
const chartMode = ref<'pan' | 'label'>('pan')
const labels = ref<LabelEntry[]>([])
const savedLabelsSignature = ref<string>('[]')  // JSON of last-saved labels
const labelsLastSavedAt = ref<string | null>(null)
const savingLabels = ref(false)
const labelsHydratedForPath = ref<string | null>(null)
// Placement — start / end are the current pending line positions. null =
// not placed yet. First chart click sets labelStart, next sets labelEnd.
const labelStart = ref<number | null>(null)
const labelEnd = ref<number | null>(null)
const labelClass = ref<string>('')
const labelValidationError = ref<string | null>(null)
const editingLabelIndex = ref<number | null>(null)
// x_column from the last loaded sidecar (null when nothing loaded / first
// time labeling this CSV — we default to the current timestamp column).
const sidecarXColumn = ref<string | null>(null)

// Dataset scan & partition state
const datasetScan = ref<any>(null)
const selectedCategory = ref<string | null>(null)
const selectedLabel = ref<string | null>(null)
const scanning = ref(false)

// Upload state
const showUploadDialog = ref(false)
const showFormatGuide = ref(false)

// File Manager (Manage Files) dialog state
const showFileManager = ref(false)

// Text Import wizard state
type TextDelimiterChoice = ',' | '\t' | ';' | ' ' | '|' | 'other'
const showTextImport = ref(false)
const textImportFile = ref<FileItem | null>(null)
const textImportLoading = ref(false)
const textImportError = ref('')
const textImportSettings = ref<{
  delimiter: TextDelimiterChoice
  delimiterOther: string
  headerRow: number
  skipRows: number
}>({
  delimiter: ',',
  delimiterOther: '',
  headerRow: 1,
  skipRows: 0,
})
const textImportRawLines = ref<string[]>([])
const textImportDetectedDelimiter = ref<string>(',')

const timePatternExact = [
  'time', 'timestamp', 'index',
  'time (s)', 'time(s)', 'time_s',
  'time (ms)', 'time(ms)', 'time_ms',
  'elapsed', 'elapsed_time',
  't (s)', 't(s)', 't_s',
  'datetime', 'date_time'
]
const labelPatterns = ['label', 'labels', 'class', 'class_name', 'target', 'category']
const commonErrors = [
  { code: 'NO_TIME_OR_INDEX', title: 'No time or index column found', hint: 'Add a column named `time` (seconds/ms) or `index` (row counter). Any column starting with `time` or `timestamp` also works.' },
  { code: 'NO_SENSOR_COLUMNS', title: 'No numeric sensor columns detected', hint: 'Ensure at least one column contains numbers. Text columns are ignored (except for the label column).' },
  { code: 'NON_NUMERIC_SENSOR', title: 'A sensor column has non-numeric values', hint: 'Check for stray text, empty cells, or wrong delimiters. Every non-time, non-label column must be all numbers.' },
  { code: 'EMPTY_FILE', title: 'File is empty or has no data rows', hint: 'Verify the file has a header row plus at least one data row.' },
  { code: 'COLUMN_MISMATCH', title: 'Multi-CSV files have different columns', hint: 'When selecting multiple CSVs, all files must have identical column headers.' }
]
const uploadTab = ref<'files' | 'folder'>('files')
const uploadFiles = ref<File[]>([])
const folderUploadFiles = ref<Array<{ file: File; relative_path: string }>>([])
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadError = ref('')
const uploadSuccess = ref('')
const isDragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const folderInput = ref<HTMLInputElement | null>(null)
const allowedFileTypes = '.csv,.json,.cbor,.txt,.tsv,.dat,.log'

const folderTopLevelCount = computed(() => {
  const tops = new Set<string>()
  for (const entry of folderUploadFiles.value) {
    // relative_path is "topdir/sub/.../file.csv"; if there is no separator
    // (a single-file "folder"), treat the file itself as its own top-level.
    const idx = entry.relative_path.indexOf('/')
    tops.add(idx >= 0 ? entry.relative_path.slice(0, idx) : entry.relative_path)
  }
  return tops.size
})

// Validation-error dialog (surfaces backend DataValidationError responses)
const showValidationError = ref(false)
const validationError = ref<{ code: string; message: string; hint: string } | null>(null)

function tryShowValidationError(e: any): boolean {
  const data = e?.response?.data
  if (data?.error_code) {
    validationError.value = {
      code: data.error_code,
      message: data.error || 'Validation failed',
      hint: data.hint || '',
    }
    showValidationError.value = true
    return true
  }
  return false
}

// Delete state (admin only)
const showDeleteDialog = ref(false)
const itemToDelete = ref<FileItem | null>(null)
const deleting = ref(false)

// Sensor recording state
const showRecordDialog = ref(false)
const recordMode = ref<'manual' | 'network_traffic' | 'disk_io'>('network_traffic')
const recordDuration = ref(120)
const recordRate = ref(10)
const recordLabel = ref('Normal')
const recordFilename = ref('')
const recording = ref(false)
const recordingDone = ref(false)
const recordStarting = ref(false)
const recordJobId = ref<string | null>(null)
const recordElapsed = ref(0)
const recordProgress = ref(0)
const recordCurrentPhase = ref('')
const recordTotalSamples = ref(0)
const recordOutputFilename = ref('')
const recordOutputPath = ref('')
const recordError = ref('')
const recordHasGpu = ref(false)
let recordPollTimer: ReturnType<typeof setInterval> | null = null

const phaseInstruction = computed(() => {
  const instructions: Record<string, { title: string; detail: string }> = {
    idle: { title: 'Idle Phase', detail: 'Keep your computer idle — don\'t touch anything.' },
    web_browsing: { title: 'Web Browsing', detail: 'Browse websites — open pages, scroll, click links.' },
    video_streaming: { title: 'Video Streaming', detail: 'Play a YouTube video or stream media.' },
    file_download: { title: 'File Download', detail: 'Download a large file from the internet.' },
  }
  return instructions[recordCurrentPhase.value] || { title: recordCurrentPhase.value, detail: '' }
})

const formatInfo = computed(() => {
  switch (selectedFormat.value) {
    case 'csv':
      return 'Headers in first row. Requires numeric sensor columns and optional "label" column. Select multiple CSV files with the same columns to load as one dataset.'
    case 'ei-json':
      return 'Standard Edge Impulse JSON export format with sensors and values arrays.'
    case 'ei-cbor':
      return 'Select a dataset folder containing training/testing subfolders. Classes are auto-detected from filenames.'
    case 'cira-cbor':
      return 'Select a dataset folder with train/test subfolders. Classes are auto-detected from filenames.'
    case 'text':
      return 'Pick a delimited text file (.txt, .tsv, .dat, .log). The Text Import wizard lets you tweak delimiter, header row, and skipped rows with a live preview before loading.'
    case 'url':
      return 'Fetch a CSV or delimited text file over HTTPS. Streamed to memory only (never written to disk) with a 100 MB hard cap. Quick-pick chips below load small factory / predictive-maintenance samples.'
    default:
      return ''
  }
})

const isCborFormat = computed(() => {
  return selectedFormat.value === 'ei-cbor' || selectedFormat.value === 'cira-cbor'
})

const isCsvFormat = computed(() => selectedFormat.value === 'csv')
const isTextFormat = computed(() => selectedFormat.value === 'text')
const isUrlFormat = computed(() => selectedFormat.value === 'url')

// --- Load from URL state --------------------------------------------------
interface UrlLoaderState {
  url: string
  format: 'csv' | 'text'
  delimiter: string
  headerRow: number
  skipRows: number
  columnNames: string[] | null
  loading: boolean
  error: string
}

const urlLoader = ref<UrlLoaderState>({
  url: '',
  format: 'csv',
  delimiter: '',
  headerRow: 1,
  skipRows: 0,
  columnNames: null,
  loading: false,
  error: '',
})

// --- Factory / Predictive-Maintenance catalog (hardcoded, direct-URL CSVs/TXTs) ---
interface PdmEntry {
  key: string
  category: string
  description: string
  labeled: boolean
  rows: number | null
  sizeRaw: string
  format: 'csv' | 'text'
  headerRow?: number // 0 = headerless (CMAPSS)
  columnNames?: string[] // Optional canonical column names for headerless files
  sampleUrl: string
  sourceUrl?: string  // Optional link to original / paper
  license: string
}

// Canonical NASA CMAPSS column layout (all 4 FD00X files use the same schema).
const CMAPSS_COLUMNS = [
  'unit_number', 'time_cycles',
  'setting_1', 'setting_2', 'setting_3',
  'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
  'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
  'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
  'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
  'sensor_21',
]

const PDM_CATALOG: PdmEntry[] = [
  // UCI mirror
  { key: 'AI4I_2020_full', category: 'UCI — Predictive Maintenance', description: 'Milling machine: temp, torque, speed, tool wear + fault labels (TWF/HDF/PWF/OSF/RNF)',
    labeled: true, rows: 10000, sizeRaw: '522 KB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/m-kenny/predictive_maintenance/main/ai4i2020.csv',
    sourceUrl: 'https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset',
    license: 'CC BY 4.0' },
  // NASA CMAPSS mirror — space-delimited, headerless. All 4 files share the same 26-column schema.
  { key: 'CMAPSS_FD001_train', category: 'NASA CMAPSS Turbofan', description: 'Turbofan engine sensors — 1 fault mode, 1 op condition (train)',
    labeled: true, rows: 20631, sizeRaw: '3.5 MB', format: 'text', headerRow: 0, columnNames: CMAPSS_COLUMNS,
    sampleUrl: 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt',
    sourceUrl: 'https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/',
    license: 'Public domain (US Gov work)' },
  { key: 'CMAPSS_FD001_test', category: 'NASA CMAPSS Turbofan', description: 'Turbofan test set — matched to FD001_train',
    labeled: true, rows: 13096, sizeRaw: '2.2 MB', format: 'text', headerRow: 0, columnNames: CMAPSS_COLUMNS,
    sampleUrl: 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/test_FD001.txt',
    license: 'Public domain (US Gov work)' },
  { key: 'CMAPSS_FD002_train', category: 'NASA CMAPSS Turbofan', description: '6 operating conditions, 1 fault mode (train)',
    labeled: true, rows: 53759, sizeRaw: '9.1 MB', format: 'text', headerRow: 0, columnNames: CMAPSS_COLUMNS,
    sampleUrl: 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD002.txt',
    license: 'Public domain (US Gov work)' },
  { key: 'CMAPSS_FD003_train', category: 'NASA CMAPSS Turbofan', description: '1 op condition, 2 fault modes (train)',
    labeled: true, rows: 24720, sizeRaw: '4.2 MB', format: 'text', headerRow: 0, columnNames: CMAPSS_COLUMNS,
    sampleUrl: 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD003.txt',
    license: 'Public domain (US Gov work)' },
  { key: 'CMAPSS_FD004_train', category: 'NASA CMAPSS Turbofan', description: '6 op conditions, 2 fault modes — most complex (train)',
    labeled: true, rows: 61249, sizeRaw: '10.4 MB', format: 'text', headerRow: 0, columnNames: CMAPSS_COLUMNS,
    sampleUrl: 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD004.txt',
    license: 'Public domain (US Gov work)' },
  // Azure Predictive Maintenance case study mirror
  { key: 'Azure_PdM_telemetry', category: 'Microsoft Azure PdM Case Study', description: 'Full sensor telemetry (voltage, rotation, pressure, vibration) per machine, hourly for a year',
    labeled: false, rows: 876100, sizeRaw: '79 MB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_telemetry.csv',
    sourceUrl: 'https://learn.microsoft.com/en-us/azure/architecture/industries/manufacturing/predictive-maintenance-overview',
    license: 'MIT (mirror)' },
  { key: 'Azure_PdM_failures', category: 'Microsoft Azure PdM Case Study', description: 'Failure event log (machineID, datetime, failure component)',
    labeled: true, rows: 761, sizeRaw: '23 KB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_failures.csv',
    license: 'MIT (mirror)' },
  { key: 'Azure_PdM_errors', category: 'Microsoft Azure PdM Case Study', description: 'Error event log (machineID, datetime, errorID)',
    labeled: false, rows: 3919, sizeRaw: '125 KB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_errors.csv',
    license: 'MIT (mirror)' },
  { key: 'Azure_PdM_maint', category: 'Microsoft Azure PdM Case Study', description: 'Maintenance log (machineID, datetime, component)',
    labeled: false, rows: 3286, sizeRaw: '101 KB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_maint.csv',
    license: 'MIT (mirror)' },
  { key: 'Azure_PdM_machines', category: 'Microsoft Azure PdM Case Study', description: 'Machine metadata (machineID, model, age)',
    labeled: false, rows: 100, sizeRaw: '1.5 KB', format: 'csv',
    sampleUrl: 'https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_machines.csv',
    license: 'MIT (mirror)' },
]

const PDM_QUICK_PICK_KEYS = [
  'AI4I_2020_full',
  'CMAPSS_FD001_train', 'CMAPSS_FD001_test',
  'CMAPSS_FD002_train', 'CMAPSS_FD003_train', 'CMAPSS_FD004_train',
  'Azure_PdM_failures', 'Azure_PdM_errors', 'Azure_PdM_maint',
]

const pdmCategories = computed(() => {
  const seen: string[] = []
  for (const e of PDM_CATALOG) {
    if (!seen.includes(e.category)) seen.push(e.category)
  }
  return seen
})

const pdmByCategory = computed(() => {
  const groups: Record<string, PdmEntry[]> = {}
  for (const e of PDM_CATALOG) {
    if (!groups[e.category]) groups[e.category] = []
    groups[e.category].push(e)
  }
  return groups
})

function pickPdmSample(key: string) {
  const entry = PDM_CATALOG.find(e => e.key === key)
  if (!entry) return
  urlLoader.value.url = entry.sampleUrl
  urlLoader.value.format = entry.format
  if (entry.format === 'text') {
    // CMAPSS is headerless (space-delimited). Preserve any other text defaults.
    urlLoader.value.headerRow = entry.headerRow ?? 1
    urlLoader.value.delimiter = ''
    urlLoader.value.skipRows = 0
  }
  // Attach canonical column names if the catalog entry provides them (used for
  // headerless files like CMAPSS so the preview shows meaningful names).
  urlLoader.value.columnNames = entry.columnNames ? [...entry.columnNames] : null
  urlLoader.value.error = ''
}

async function fetchFromUrl() {
  const url = urlLoader.value.url.trim()
  if (!url) return

  try {
    urlLoader.value.loading = true
    urlLoader.value.error = ''

    const payload: any = {
      url,
      format: urlLoader.value.format,
    }
    if (urlLoader.value.format === 'text') {
      payload.delimiter = urlLoader.value.delimiter || null
      payload.header_row = Math.floor(Number(urlLoader.value.headerRow) || 0)
      payload.skip_rows = Math.max(0, Math.floor(Number(urlLoader.value.skipRows) || 0))
      if (urlLoader.value.columnNames && urlLoader.value.columnNames.length > 0) {
        payload.column_names = urlLoader.value.columnNames
      }
    }

    const response = await api.post('/api/data/load-from-url', payload)
    const body = response.data
    if (!body || !body.metadata || !body.preview) {
      urlLoader.value.error = body?.error || 'Response was missing preview data. Check the URL points to a plain CSV or text file (not an HTML page or ZIP).'
      return
    }
    dataPreview.value = body
    // Clear file-based selection state so proceedToWindowing takes the
    // "non-folder data" branch (stores dataPreview directly).
    selectedFile.value = null
    selectedFiles.value = []
    notificationStore.showSuccess('File loaded from URL')
  } catch (e: any) {
    if (!tryShowValidationError(e)) {
      urlLoader.value.error = e.response?.data?.error || 'Failed to fetch URL'
    }
  } finally {
    urlLoader.value.loading = false
  }
}

const TEXT_FILE_EXTS = ['.txt', '.tsv', '.dat', '.log']
function isTextExtension(ext: string | null | undefined): boolean {
  return !!ext && TEXT_FILE_EXTS.includes(ext.toLowerCase())
}

const isFileSelected = computed(() => {
  return (path: string) => selectedFiles.value.some(f => f.path === path)
})

// CSV files in the current folder (for Select All)
const csvFilesInFolder = computed(() => {
  return currentItems.value.filter(i => !i.is_dir && i.extension === '.csv')
})

// Parent folder of each file in the basket, deduped. If size > 1, the
// backend switches to cross-sensor JOIN mode (per-sensor value columns).
const basketFolders = computed(() => {
  const dirs = new Set<string>()
  for (const f of selectedFiles.value) {
    const p = f.path.replace(/\\/g, '/')
    const idx = p.lastIndexOf('/')
    dirs.add(idx >= 0 ? p.slice(0, idx) : '')
  }
  return Array.from(dirs)
})

const isCrossFolderSelection = computed(() => basketFolders.value.length > 1)

// Sensor names inferred from the parent folder of each selected file —
// what the JOINed dataset's columns will be named.
const sensorsInBasket = computed(() => {
  const names = new Set<string>()
  for (const f of selectedFiles.value) {
    const p = f.path.replace(/\\/g, '/').split('/')
    if (p.length >= 2) names.add(p[p.length - 2])
  }
  return Array.from(names)
})

// Group basket by folder for the "Selected files" panel.
const basketByFolder = computed(() => {
  const groups: Record<string, FileItem[]> = {}
  for (const f of selectedFiles.value) {
    const p = f.path.replace(/\\/g, '/')
    const idx = p.lastIndexOf('/')
    const dir = idx >= 0 ? p.slice(0, idx) : ''
    if (!groups[dir]) groups[dir] = []
    groups[dir].push(f)
  }
  return Object.entries(groups).map(([dir, files]) => ({ dir, files }))
})

// "Load all sensors" is offered when the current folder looks like a
// machine folder — it has sub-folders (assumed to be sensor folders)
// rather than files. If it has any files, we don't offer it (the user
// is probably already at the sensor level).
const folderIsMachineShape = computed(() => {
  if (currentItems.value.length === 0) return false
  const subFolders = currentItems.value.filter(i => i.is_dir)
  const anyFiles = currentItems.value.some(i => !i.is_dir)
  return subFolders.length >= 2 && !anyFiles
})

function removeFromBasket(item: FileItem) {
  const idx = selectedFiles.value.findIndex(f => f.path === item.path)
  if (idx >= 0) selectedFiles.value.splice(idx, 1)
  if (selectedFiles.value.length <= 1) {
    selectedFile.value = selectedFiles.value[0] || null
    if (selectedFiles.value.length === 1) {
      previewFile(selectedFiles.value[0])
    } else {
      dataPreview.value = null
    }
  } else {
    previewMultipleCsv()
  }
}

function clearBasket() {
  selectedFiles.value = []
  selectedFile.value = null
  dataPreview.value = null
  multiCsvError.value = null
}

// "Load all sensors" — for a machine-shape folder (children are sensor
// folders), fetch one CSV per sensor for a given date and add them to
// the basket. Backend returns entries with `exists:false` for sensors
// missing that date so we can show them to the user.
async function openLoadAllDialog() {
  loadAllDate.value = new Date().toISOString().slice(0, 10)
  loadAllPreview.value = []
  loadAllDialogOpen.value = true
  await fetchSensorFilesForDate()

  // If today has no files for any sensor, walk backwards up to 30 days
  // and land on the most recent date that has ANY data. Silently
  // updates the date input + re-populates the list. Prevents the
  // common "picked today, saw all zeros, gave up" trap when a sim
  // hasn't run yet today OR the writer hasn't flushed the buffer.
  if (loadAllPreview.value.length > 0 &&
      !loadAllPreview.value.some(x => x.exists)) {
    const start = new Date()
    for (let i = 1; i <= 30; i++) {
      const d = new Date(start)
      d.setDate(d.getDate() - i)
      loadAllDate.value = d.toISOString().slice(0, 10)
      await fetchSensorFilesForDate()
      if (loadAllPreview.value.some(x => x.exists)) {
        notificationStore.showSuccess(
          `No data for today — showing latest available (${loadAllDate.value})`,
          3500,
        )
        return
      }
    }
    // Nothing in the last 30 days either — leave the picker on today
    // (with all-zeros) so the message is unambiguous.
    loadAllDate.value = start.toISOString().slice(0, 10)
    await fetchSensorFilesForDate()
  }
}

async function fetchSensorFilesForDate() {
  if (!currentPath.value) return
  try {
    loadingSensorFiles.value = true
    const response = await api.post('/api/data/sensor-files-for-date', {
      folder_path: currentPath.value,
      date: loadAllDate.value,
    })
    loadAllPreview.value = response.data.sensor_files || []
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to list sensor files')
    loadAllPreview.value = []
  } finally {
    loadingSensorFiles.value = false
  }
}

function confirmLoadAllSensors() {
  const toAdd = loadAllPreview.value.filter(x => x.exists)
  if (toAdd.length === 0) {
    notificationStore.showError('No sensor files exist for that date')
    return
  }
  // Replace the basket to make intent obvious. The alternative — appending
  // — silently merged with unrelated leftover picks and confused users.
  selectedFiles.value = toAdd.map(x => ({
    name: x.file,
    path: x.path,
    is_dir: false,
    extension: '.csv',
    size: null,
    file_type: 'csv',
  } as FileItem))
  loadAllDialogOpen.value = false
  notificationStore.showSuccess(`Added ${toAdd.length} sensor file${toAdd.length === 1 ? '' : 's'} to selection`)
  previewMultipleCsv()
}

const allCsvSelectedInFolder = computed(() => {
  if (csvFilesInFolder.value.length === 0) return false
  return csvFilesInFolder.value.every(f => selectedFiles.value.some(s => s.path === f.path))
})

// Detect if current folder is a dataset root (has training/testing subfolders)
const isDatasetFolder = computed(() => {
  if (!isCborFormat.value) return false
  const folderNames = currentItems.value.filter(i => i.is_dir).map(i => i.name.toLowerCase())
  return folderNames.includes('training') || folderNames.includes('testing') ||
         folderNames.includes('train') || folderNames.includes('test') ||
         folderNames.includes('dataset')
})

const breadcrumbs = computed(() => {
  // Only show paths relative to the base path (the user's accessible root)
  const base = basePath.value || ''
  const current = currentPath.value || ''

  // Get the relative path from base
  let relativePath = current
  if (base && current.startsWith(base)) {
    relativePath = current.slice(base.length).replace(/^[/\\]/, '')
  }

  const parts = relativePath.split(/[/\\]/).filter(Boolean)
  // Root navigates back to base path (or null for API to resolve)
  const items = [{ title: 'Root', path: '__ROOT__', disabled: false }]

  let path = base
  for (const part of parts) {
    path += (path ? '/' : '') + part
    items.push({ title: part, path, disabled: false })
  }

  if (items.length > 0) {
    items[items.length - 1].disabled = true
  }

  return items
})

const previewHeaders = computed(() => {
  if (!dataPreview.value) return []

  return dataPreview.value.metadata.columns.map((col: string) => ({
    title: col,
    key: col,
    sortable: true
  }))
})

const canProceed = computed(() => !!dataPreview.value)

// Initialize selected columns when data loads (all selected by default)
watch(() => dataPreview.value, (newVal) => {
  if (newVal?.metadata?.sensor_columns && pipelineStore.selectedColumns.length === 0) {
    const cols = [...newVal.metadata.sensor_columns]
    // Ensure timestamp is included
    const ts = newVal.metadata.timestamp_column
    if (ts && !cols.includes(ts)) {
      cols.unshift(ts)
    }
    pipelineStore.selectedColumns = cols
  }
})

function isSensorColumn(col: string): boolean {
  return dataPreview.value?.metadata?.sensor_columns?.includes(col) || false
}

function toggleColumn(col: string) {
  // Timestamp is always selected, can't toggle
  if (col === timestampColumn.value) return

  const cols = [...pipelineStore.selectedColumns]
  const idx = cols.indexOf(col)
  if (idx >= 0) {
    // Must keep at least 2 columns (1 sensor + timestamp)
    const nonTimestampCount = cols.filter(c => c !== timestampColumn.value).length
    if (nonTimestampCount <= 1) return
    cols.splice(idx, 1)
  } else {
    cols.push(col)
  }
  pipelineStore.selectedColumns = cols
}

function selectAllColumns() {
  pipelineStore.selectedColumns = [...(dataPreview.value?.metadata?.sensor_columns || [])]
}

function selectNoColumns() {
  // Keep timestamp + 1 sensor minimum
  const sensors = dataPreview.value?.metadata?.sensor_columns || []
  const ts = timestampColumn.value
  const nonTs = sensors.filter((s: string) => s !== ts)
  pipelineStore.selectedColumns = ts && sensors.includes(ts)
    ? [ts, ...(nonTs.length > 0 ? [nonTs[0]] : [])]
    : sensors.length > 0 ? [sensors[0]] : []
}

const timestampColumn = computed(() =>
  dataPreview.value?.metadata?.timestamp_column || null
)

const plottableColumns = computed<string[]>(() => {
  const ts = timestampColumn.value
  return pipelineStore.selectedColumns.filter(
    (c: string) => c !== ts && isSensorColumn(c)
  )
})

const canVisualize = computed(() =>
  !!dataPreview.value && !!timestampColumn.value && plottableColumns.value.length > 0
)

const chartData = computed(() => {
  if (!dataPreview.value || !timestampColumn.value) {
    return { labels: [] as (string | number)[], datasets: [] as any[] }
  }
  const rows = dataPreview.value.preview as any[]
  const ts = timestampColumn.value as string
  const labels = rows.map((r) => r[ts])
  const datasets = plottableColumns.value.map((col, idx) => ({
    label: col,
    data: rows.map((r) => Number(r[col])),
    borderColor: channelColors[idx % channelColors.length],
    backgroundColor: channelColors[idx % channelColors.length] + '20',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1,
    spanGaps: true
  }))
  return { labels, datasets }
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: {
    legend: { position: 'bottom' as const, labels: { usePointStyle: true, padding: 15 } },
    tooltip: { mode: 'index' as const, intersect: false }
  },
  scales: {
    x: {
      title: { display: true, text: 'timestamp' },
      ticks: { autoSkip: true, maxTicksLimit: 12 },
      grid: { display: false },
      min: xMin.value ?? undefined,
      max: xMax.value ?? undefined
    },
    y: {
      title: { display: true, text: 'value' },
      grid: { color: 'rgba(127, 127, 127, 0.1)' }
    }
  },
  interaction: { mode: 'nearest' as const, axis: 'x' as const, intersect: false }
}))

const isZoomed = computed(() => xMin.value !== null || xMax.value !== null)

const dataLength = computed(() => dataPreview.value?.preview?.length ?? 0)

function clampRange(min: number, max: number): [number, number] {
  const last = Math.max(0, dataLength.value - 1)
  if (max - min < 2) max = Math.min(last, min + 2)
  if (min < 0) { max -= min; min = 0 }
  if (max > last) { min -= (max - last); max = last }
  if (min < 0) min = 0
  return [Math.round(min), Math.round(max)]
}

function resetZoom() {
  xMin.value = null
  xMax.value = null
}

function onChartWheel(e: WheelEvent) {
  if (!chartRef.value?.chart || dataLength.value < 3) return
  e.preventDefault()
  const chart = chartRef.value.chart
  const rect = chart.canvas.getBoundingClientRect()
  const cursorPx = e.clientX - rect.left
  const xScale = chart.scales.x
  const last = dataLength.value - 1
  const curMin = xMin.value ?? 0
  const curMax = xMax.value ?? last
  let valueAtCursor: number
  try {
    valueAtCursor = xScale.getValueForPixel(cursorPx) ?? (curMin + curMax) / 2
  } catch {
    valueAtCursor = (curMin + curMax) / 2
  }
  const factor = e.deltaY < 0 ? 0.8 : 1.25
  const newMin = valueAtCursor - (valueAtCursor - curMin) * factor
  const newMax = valueAtCursor + (curMax - valueAtCursor) * factor
  const [m, M] = clampRange(newMin, newMax)
  if (m === 0 && M === last) {
    resetZoom()
  } else {
    xMin.value = m
    xMax.value = M
  }
}

function onChartMouseDown(e: MouseEvent) {
  if (e.button !== 0 || !chartRef.value?.chart || dataLength.value < 2) return
  // In label mode, mousedown doesn't pan — the click event handler places
  // the start/end lines instead. Consuming mousedown here would swallow
  // the follow-up click.
  if (chartMode.value === 'label') return
  isPanning = true
  panCursor.value = 'grabbing'
  panStartClientX = e.clientX
  panStartMin = xMin.value ?? 0
  panStartMax = xMax.value ?? (dataLength.value - 1)
}

function onChartMouseMove(e: MouseEvent) {
  if (!isPanning || !chartRef.value?.chart) return
  const chart = chartRef.value.chart
  const rect = chart.canvas.getBoundingClientRect()
  const range = panStartMax - panStartMin
  if (range <= 0 || rect.width <= 0) return
  const pixelsPerUnit = rect.width / range
  const shift = -(e.clientX - panStartClientX) / pixelsPerUnit
  const [m, M] = clampRange(panStartMin + shift, panStartMax + shift)
  const last = Math.max(0, dataLength.value - 1)
  if (m === 0 && M === last) {
    resetZoom()
  } else {
    xMin.value = m
    xMax.value = M
  }
}

function onChartMouseUp() {
  if (isPanning) {
    isPanning = false
    panCursor.value = 'grab'
  }
}

watch([() => showVisualization.value, () => dataLength.value], () => {
  if (xMin.value !== null && xMin.value >= dataLength.value) resetZoom()
  if (xMax.value !== null && xMax.value >= dataLength.value) resetZoom()
})

function getFileIcon(ext: string | null) {
  switch (ext) {
    case '.csv': return 'mdi-file-delimited'
    case '.json': return 'mdi-code-json'
    case '.cbor': return 'mdi-file-code'
    case '.txt':
    case '.tsv':
    case '.dat':
    case '.log':
      return 'mdi-file-document-outline'
    default: return 'mdi-file'
  }
}

function getFileColor(ext: string | null) {
  switch (ext) {
    case '.csv': return 'success'
    case '.json': return 'info'
    case '.cbor': return 'secondary'
    case '.txt':
    case '.tsv':
    case '.dat':
    case '.log':
      return 'primary'
    default: return 'grey'
  }
}

function getFolderIcon(item: FileItem) {
  const name = item.name.toLowerCase()
  if (name === 'training' || name === 'train') return 'mdi-folder-star'
  if (name === 'testing' || name === 'test') return 'mdi-folder-clock'
  if (name === 'dataset') return 'mdi-folder-multiple'
  if (isDatasetRootFolder(item)) return 'mdi-folder-open'
  return 'mdi-folder'
}

function getFolderColor(item: FileItem) {
  if (!item.is_dir) return getFileColor(item.extension)
  const name = item.name.toLowerCase()
  if (name === 'training' || name === 'train') return 'success'
  if (name === 'testing' || name === 'test') return 'info'
  if (isDatasetRootFolder(item)) return 'primary'
  return 'warning'
}

function isDatasetRootFolder(item: FileItem): boolean {
  if (!item.is_dir) return false
  const name = item.name.toLowerCase()
  return name.includes('cbor') || name.includes('dataset') || name.includes('impulse')
}

function formatFileSize(bytes: number | null) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

async function loadFolders() {
  try {
    loadingFolders.value = true
    const response = await api.post('/api/data/browse', { path: currentPath.value || null })
    currentItems.value = response.data.items
    currentPath.value = response.data.current_path

    // Store the base path on first load (when we send null)
    if (basePath.value === null) {
      basePath.value = response.data.current_path || ''
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to load files')
  } finally {
    loadingFolders.value = false
  }
}

function navigateTo(path: string) {
  // Handle special '__ROOT__' path - navigate to base path
  if (path === '__ROOT__') {
    currentPath.value = basePath.value || ''
  } else {
    currentPath.value = path
  }
  datasetScan.value = null
  selectedCategory.value = null
  selectedLabel.value = null
  dataPreview.value = null
  selectedFile.value = null
  // Keep `selectedFiles` (the multi-file basket) so users can pick CSVs
  // from different folders in one go — e.g. one sensor per folder under
  // a machine node. Use "Clear selection" in the header to reset it.
  multiCsvError.value = null
  loadFolders()
}

function toggleSelectAllCsv() {
  multiCsvError.value = null
  const folderFiles = csvFilesInFolder.value
  if (folderFiles.length === 0) return

  if (allCsvSelectedInFolder.value) {
    // Deselect all CSV files in this folder
    const folderPaths = new Set(folderFiles.map(f => f.path))
    selectedFiles.value = selectedFiles.value.filter(f => !folderPaths.has(f.path))
  } else {
    // Add any unselected CSV files from this folder
    const existingPaths = new Set(selectedFiles.value.map(f => f.path))
    for (const f of folderFiles) {
      if (!existingPaths.has(f.path)) selectedFiles.value.push(f)
    }
  }

  // Sync selectedFile + preview state
  if (selectedFiles.value.length === 1) {
    selectedFile.value = selectedFiles.value[0]
    previewFile(selectedFiles.value[0])
  } else if (selectedFiles.value.length > 1) {
    selectedFile.value = null
    previewMultipleCsv()
  } else {
    selectedFile.value = null
    dataPreview.value = null
  }
}

// Row-click on a CSV file → single-select (replace whole selection with this file).
// Matches Windows Explorer / Finder behavior. Multi-select is via the checkbox.
function selectSingleCsvFile(item: FileItem) {
  multiCsvError.value = null
  const wasOnlySelection = selectedFiles.value.length === 1 && selectedFiles.value[0].path === item.path
  if (wasOnlySelection) {
    // Clicking the already-selected file deselects it.
    selectedFiles.value = []
    selectedFile.value = null
    dataPreview.value = null
    return
  }
  selectedFiles.value = [item]
  selectedFile.value = item
  previewFile(item)
}

function toggleCsvFile(item: FileItem) {
  multiCsvError.value = null
  const idx = selectedFiles.value.findIndex(f => f.path === item.path)
  if (idx >= 0) {
    selectedFiles.value.splice(idx, 1)
  } else {
    selectedFiles.value.push(item)
  }

  // Update selectedFile for compatibility
  if (selectedFiles.value.length === 1) {
    selectedFile.value = selectedFiles.value[0]
  } else if (selectedFiles.value.length > 1) {
    selectedFile.value = null
  } else {
    selectedFile.value = null
  }

  // Trigger preview
  if (selectedFiles.value.length === 1) {
    previewFile(selectedFiles.value[0])
  } else if (selectedFiles.value.length > 1) {
    previewMultipleCsv()
  } else {
    dataPreview.value = null
  }
}

async function previewMultipleCsv() {
  // Cross-folder basket + no alignment picked yet → open the merge dialog
  // and defer this call until the user confirms.
  if (isCrossFolderSelection.value && !mergeAlignment.value) {
    pendingLoadAction = previewMultipleCsv
    mergeDialogOpen.value = true
    return
  }
  try {
    loading.value = true
    multiCsvError.value = null

    const payload: Record<string, any> = {
      file_paths: selectedFiles.value.map(f => f.path),
      rows: 100,
      format: 'csv',
    }
    if (isCrossFolderSelection.value) {
      payload.merge_mode = 'join'
      payload.alignment = mergeAlignment.value
      if (mergeToleranceMs.value != null) payload.tolerance_ms = mergeToleranceMs.value
      if (mergeResampleHz.value != null) payload.resample_hz = mergeResampleHz.value
    }
    const response = await api.post('/api/data/preview', payload)

    dataPreview.value = response.data
    notificationStore.showSuccess(`${selectedFiles.value.length} CSV files loaded successfully`)
  } catch (e: any) {
    const errorMsg = e.response?.data?.error || 'Failed to load multiple CSV files'
    if (tryShowValidationError(e)) {
      dataPreview.value = null
    } else if (errorMsg.includes('Column mismatch') || errorMsg.includes('mismatch')) {
      multiCsvError.value = errorMsg
      dataPreview.value = null
    } else {
      notificationStore.showError(errorMsg)
      dataPreview.value = null
    }
  } finally {
    loading.value = false
  }
}

function onMergeDialogConfirm(payload: {
  alignment: 'exact' | 'nearest' | 'resample'
  tolerance_ms: number | null
  resample_hz: number | null
}) {
  mergeAlignment.value = payload.alignment
  mergeToleranceMs.value = payload.tolerance_ms
  mergeResampleHz.value = payload.resample_hz
  const action = pendingLoadAction
  pendingLoadAction = null
  if (action) action()
}

function onMergeDialogCancel() {
  pendingLoadAction = null
  // Leave alignment state as-is so the user can reopen without losing their
  // previous pick, but don't proceed with the load.
}

// Reset the cross-sensor alignment picks whenever the basket changes so the
// dialog re-opens with a clean slate on the next cross-folder load. Only
// clears when the set of parent folders actually changes.
watch(basketFolders, () => {
  mergeAlignment.value = null
  mergeToleranceMs.value = null
  mergeResampleHz.value = null
}, { deep: true })

async function handleItemClick(item: FileItem) {
  if (item.is_dir) {
    currentPath.value = item.path
    datasetScan.value = null
    selectedCategory.value = null
    selectedLabel.value = null
    dataPreview.value = null
    selectedFile.value = null
    // Basket (`selectedFiles`) intentionally preserved across folder hops
    // so cross-folder multi-select works. See navigateTo for the same rule.
    multiCsvError.value = null
    await loadFolders()
  } else if (isCsvFormat.value && item.extension === '.csv') {
    // Row-click = single-select (replace). Checkbox click still toggles multi-select.
    selectSingleCsvFile(item)
  } else if (isTextFormat.value && isTextExtension(item.extension)) {
    // Text file — open the Text Import wizard before parsing
    selectedFile.value = item
    selectedFiles.value = []
    await openTextImportDialog(item)
  } else {
    selectedFile.value = item
    selectedFiles.value = []
    await previewFile(item)
  }
}

async function scanDatasetFolder() {
  try {
    scanning.value = true
    const response = await api.post('/api/data/scan', {
      folder_path: currentPath.value
    })
    datasetScan.value = response.data

    // Auto-select first category
    const categories = Object.keys(response.data.categories)
    if (categories.length > 0) {
      selectCategory(categories[0])
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to scan dataset')
    datasetScan.value = null
  } finally {
    scanning.value = false
  }
}

function selectCategory(category: string) {
  selectedCategory.value = category
  selectedLabel.value = null
  loadPartitionPreview()
}

function selectLabel(label: string | null) {
  selectedLabel.value = label
  loadPartitionPreview()
}

async function loadPartitionPreview() {
  if (!selectedCategory.value) return

  try {
    loading.value = true
    const response = await api.post('/api/data/preview', {
      file_path: currentPath.value,
      rows: 100,
      format: selectedFormat.value,
      category: selectedCategory.value,
      label: selectedLabel.value
    })

    dataPreview.value = response.data
    selectedFile.value = {
      name: currentPath.value.split(/[/\\]/).pop() || 'Dataset',
      path: currentPath.value,
      is_dir: true,
      extension: null,
      size: null,
      file_type: null
    }
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to load partition preview')
    dataPreview.value = null
  } finally {
    loading.value = false
  }
}

async function previewFile(item: FileItem) {
  try {
    loading.value = true

    const response = await api.post('/api/data/preview', {
      file_path: item.path,
      rows: 100,
      format: selectedFormat.value
    })

    dataPreview.value = response.data
    notificationStore.showSuccess('Data loaded successfully')
  } catch (e: any) {
    if (!tryShowValidationError(e)) {
      notificationStore.showError(e.response?.data?.error || 'Failed to preview file')
    }
    dataPreview.value = null
  } finally {
    loading.value = false
  }
}

// --- Text Import wizard -------------------------------------------------
function delimiterCharFromChoice(choice: TextDelimiterChoice, other: string): string {
  if (choice === 'other') return (other || '').slice(0, 1) || ','
  return choice
}

function delimiterDisplay(ch: string): string {
  if (ch === '\t') return '\\t'
  if (ch === ' ') return 'space'
  return ch
}

const textImportEffectiveDelimiter = computed(() =>
  delimiterCharFromChoice(textImportSettings.value.delimiter, textImportSettings.value.delimiterOther)
)

const textImportPreview = computed<{ headers: string[]; rows: string[][] }>(() => {
  const lines = textImportRawLines.value
  const skip = Math.max(0, Math.floor(Number(textImportSettings.value.skipRows) || 0))
  const headerRow = Math.floor(Number(textImportSettings.value.headerRow) || 0)
  const delim = textImportEffectiveDelimiter.value

  const afterSkip = lines.slice(skip)
  if (afterSkip.length === 0) return { headers: [], rows: [] }

  let headers: string[]
  let dataLines: string[]

  if (headerRow <= 0) {
    // Headerless
    const first = afterSkip[0].split(delim)
    headers = first.map((_, i) => `col_${i + 1}`)
    dataLines = afterSkip
  } else {
    const headerIdx = headerRow - 1
    if (headerIdx >= afterSkip.length) return { headers: [], rows: [] }
    headers = afterSkip[headerIdx].split(delim)
    dataLines = afterSkip.slice(headerIdx + 1)
  }

  const rows = dataLines.slice(0, 20).map((line) => line.split(delim))
  return { headers, rows }
})

const textImportCanImport = computed(() => {
  if (textImportLoading.value) return false
  if (textImportSettings.value.delimiter === 'other' && !textImportSettings.value.delimiterOther) {
    return false
  }
  return textImportPreview.value.headers.length > 0
})

function delimiterChoiceFromChar(ch: string): TextDelimiterChoice {
  switch (ch) {
    case ',': return ','
    case '\t': return '\t'
    case ';': return ';'
    case ' ': return ' '
    case '|': return '|'
    default: return 'other'
  }
}

async function openTextImportDialog(item: FileItem) {
  textImportFile.value = item
  textImportError.value = ''
  textImportRawLines.value = []
  // Reset to defaults on each open so the previous file's picks don't leak.
  textImportSettings.value = {
    delimiter: ',',
    delimiterOther: '',
    headerRow: 1,
    skipRows: 0,
  }

  try {
    textImportLoading.value = true
    const response = await api.post('/api/data/text-sniff', {
      file_path: item.path,
    })
    textImportDetectedDelimiter.value = response.data.detected_delimiter || ','
    textImportRawLines.value = response.data.raw_lines || []

    // Pre-select the sniffed delimiter.
    const sniffed = textImportDetectedDelimiter.value
    const choice = delimiterChoiceFromChar(sniffed)
    textImportSettings.value.delimiter = choice
    if (choice === 'other') {
      textImportSettings.value.delimiterOther = sniffed.slice(0, 1)
    }
    // Pre-populate skip-rows from the backend's preamble heuristic so files
    // with leading comment/title lines Just Work on Import.
    const suggested = Number(response.data.suggested_skip_rows)
    if (Number.isFinite(suggested) && suggested > 0) {
      textImportSettings.value.skipRows = suggested
    }
    showTextImport.value = true
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to sniff text file')
    textImportFile.value = null
  } finally {
    textImportLoading.value = false
  }
}

function cancelTextImport() {
  showTextImport.value = false
  textImportFile.value = null
  textImportError.value = ''
  textImportRawLines.value = []
}

async function confirmTextImport() {
  if (!textImportFile.value) return
  const delim = textImportEffectiveDelimiter.value
  if (!delim) {
    textImportError.value = 'Please choose a delimiter.'
    return
  }

  try {
    textImportLoading.value = true
    textImportError.value = ''

    const response = await api.post('/api/data/preview', {
      file_path: textImportFile.value.path,
      rows: 100,
      format: 'text',
      delimiter: delim,
      header_row: Math.floor(Number(textImportSettings.value.headerRow) || 0),
      skip_rows: Math.max(0, Math.floor(Number(textImportSettings.value.skipRows) || 0)),
    })

    dataPreview.value = response.data
    notificationStore.showSuccess('Text file loaded successfully')
    showTextImport.value = false
    textImportFile.value = null
    textImportRawLines.value = []
  } catch (e: any) {
    if (!tryShowValidationError(e)) {
      textImportError.value = e.response?.data?.error || 'Failed to import text file'
    }
  } finally {
    textImportLoading.value = false
  }
}

async function loadMorePreview() {
  if (!dataPreview.value) return
  const sourceUrl = dataPreview.value.metadata?.source_url as string | undefined
  if (!selectedFile.value && selectedFiles.value.length === 0 && !sourceUrl) return

  try {
    loadingMore.value = true

    const currentRows = dataPreview.value.preview.length
    const newRowCount = Math.min(currentRows + 100, maxPreviewRows)

    // URL-loaded data: re-fetch. Cheaper alternative would need a backend
    // "session preview" endpoint; keep it simple and reuse load-from-url.
    if (sourceUrl) {
      const payload: any = {
        url: sourceUrl,
        format: dataPreview.value.metadata?.source_format === 'text' ? 'text' : 'csv',
      }
      if (payload.format === 'text') {
        payload.delimiter = dataPreview.value.metadata?.delimiter || null
        payload.header_row = dataPreview.value.metadata?.header_row ?? 1
        payload.skip_rows = dataPreview.value.metadata?.skip_rows ?? 0
      }
      const response = await api.post('/api/data/load-from-url', payload)
      // Splice more rows out of the reloaded preview (backend caps at 10).
      // Since load_csv/load_text keep the full dataframe in-session but only
      // return .head(10), we can only surface those 10 rows for URL loads
      // without a session-preview endpoint. Merge to at most newRowCount.
      const merged = response.data.preview.slice(0, newRowCount)
      dataPreview.value = { ...response.data, preview: merged }
      notificationStore.showSuccess(`Loaded ${merged.length} rows`)
      return
    }

    // Multi-CSV load more
    if (dataPreview.value.metadata?.is_multi_csv && selectedFiles.value.length > 1) {
      const response = await api.post('/api/data/preview', {
        file_paths: selectedFiles.value.map(f => f.path),
        rows: newRowCount,
        format: 'csv'
      })
      dataPreview.value = response.data
      notificationStore.showSuccess(`Loaded ${response.data.preview.length} rows`)
      return
    }

    const requestData: any = {
      file_path: selectedFile.value!.path,
      rows: newRowCount,
      format: selectedFormat.value
    }

    // Preserve partition filters for dataset folder previews
    if (dataPreview.value.metadata?.is_partition_preview && selectedCategory.value) {
      requestData.category = selectedCategory.value
      requestData.label = selectedLabel.value
    }

    // Preserve Text Import wizard settings so "Load More" re-parses identically
    // to the original import (delimiter + header/skip stay the same).
    if (dataPreview.value.metadata?.source_format === 'text') {
      requestData.format = 'text'
      requestData.delimiter = dataPreview.value.metadata.delimiter
      requestData.header_row = dataPreview.value.metadata.header_row
      requestData.skip_rows = dataPreview.value.metadata.skip_rows
    }

    const response = await api.post('/api/data/preview', requestData)

    dataPreview.value = response.data
    notificationStore.showSuccess(`Loaded ${response.data.preview.length} rows`)
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to load more rows')
  } finally {
    loadingMore.value = false
  }
}

async function proceedToWindowing() {
  if (!dataPreview.value) return

  // Multi-CSV: load full dataset via ingest endpoint
  if (dataPreview.value.metadata?.is_multi_csv && selectedFiles.value.length > 1) {
    // Same "open dialog first" guard as previewMultipleCsv, in case the
    // user reached this button without triggering the preview path first.
    if (isCrossFolderSelection.value && !mergeAlignment.value) {
      pendingLoadAction = proceedToWindowing
      mergeDialogOpen.value = true
      return
    }
    try {
      loadingFull.value = true
      loading.value = true

      const payload: Record<string, any> = {
        file_paths: selectedFiles.value.map(f => f.path),
      }
      if (isCrossFolderSelection.value) {
        payload.merge_mode = 'join'
        payload.alignment = mergeAlignment.value
        if (mergeToleranceMs.value != null) payload.tolerance_ms = mergeToleranceMs.value
        if (mergeResampleHz.value != null) payload.resample_hz = mergeResampleHz.value
      }
      const response = await api.post('/api/data/ingest/csv-multiple', payload)

      pipelineStore.dataSession = response.data
      notificationStore.showSuccess(
        `${response.data.metadata.total_samples} CSV files loaded as one dataset`
      )
    } catch (e: any) {
      notificationStore.showError(e.response?.data?.error || 'Failed to load CSV files')
      return
    } finally {
      loadingFull.value = false
      loading.value = false
    }
  } else if (dataPreview.value.metadata?.is_partition_preview) {
    // Partition preview: load the full dataset first
    try {
      loadingFull.value = true
      loading.value = true

      const response = await api.post('/api/data/load-full', {
        folder_path: currentPath.value,
        format: selectedFormat.value,
        preview_session_id: dataPreview.value.session_id
      })

      // Store the full session for windowing
      pipelineStore.dataSession = response.data
      notificationStore.showSuccess(
        `Full dataset loaded: ${response.data.metadata.total_samples} samples`
      )
    } catch (e: any) {
      notificationStore.showError(e.response?.data?.error || 'Failed to load full dataset')
      return
    } finally {
      loadingFull.value = false
      loading.value = false
    }
  } else {
    // Non-folder data — store directly
    pipelineStore.dataSession = dataPreview.value
  }

  router.push({ name: 'pipeline-windowing' })
}

// Watch format changes to reset state
watch(selectedFormat, () => {
  selectedFile.value = null
  selectedFiles.value = []
  multiCsvError.value = null
  dataPreview.value = null
  datasetScan.value = null
  selectedCategory.value = null
  selectedLabel.value = null
  urlLoader.value.error = ''
  urlLoader.value.loading = false
})

// Upload methods
function triggerFileInput() {
  fileInput.value?.click()
}

function triggerFolderInput() {
  folderInput.value?.click()
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files) {
    addFiles(Array.from(target.files))
  }
}

function handleFolderSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (!target.files) return

  const validExtensions = ['csv', 'json', 'cbor', 'txt', 'tsv', 'dat', 'log']
  const maxSize = 100 * 1024 * 1024 // 100 MB
  const entries: Array<{ file: File; relative_path: string }> = []

  for (const file of Array.from(target.files)) {
    // webkitRelativePath is "topdir/subdir/file.csv"
    const rel = (file as any).webkitRelativePath || file.name
    const ext = file.name.split('.').pop()?.toLowerCase() || ''

    if (!validExtensions.includes(ext)) continue
    if (file.size > maxSize) {
      uploadError.value = `File too large: ${rel}. Max size: 100 MB`
      continue
    }
    entries.push({ file, relative_path: rel })
  }

  folderUploadFiles.value = entries

  // Reset the input so re-selecting the same folder fires @change again
  target.value = ''
}

function handleDrop(event: DragEvent) {
  isDragging.value = false
  if (event.dataTransfer?.files) {
    addFiles(Array.from(event.dataTransfer.files))
  }
}

function addFiles(files: File[]) {
  const validExtensions = ['csv', 'json', 'cbor', 'txt', 'tsv', 'dat', 'log']
  const maxSize = 100 * 1024 * 1024 // 100 MB

  for (const file of files) {
    const ext = file.name.split('.').pop()?.toLowerCase() || ''

    if (!validExtensions.includes(ext)) {
      uploadError.value = `Invalid file type: ${file.name}. Supported: CSV, JSON, CBOR, text (.txt/.tsv/.dat/.log)`
      continue
    }

    if (file.size > maxSize) {
      uploadError.value = `File too large: ${file.name}. Max size: 100 MB`
      continue
    }

    // Avoid duplicates
    if (!uploadFiles.value.find(f => f.name === file.name && f.size === file.size)) {
      uploadFiles.value.push(file)
    }
  }
}

function removeFile(index: number) {
  uploadFiles.value.splice(index, 1)
}

function getFileTypeIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'csv': return 'mdi-file-delimited'
    case 'json': return 'mdi-code-json'
    case 'cbor': return 'mdi-file-code'
    case 'txt':
    case 'tsv':
    case 'dat':
    case 'log':
      return 'mdi-file-document-outline'
    default: return 'mdi-file'
  }
}

function getFileTypeColor(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'csv': return 'success'
    case 'json': return 'info'
    case 'cbor': return 'secondary'
    case 'txt':
    case 'tsv':
    case 'dat':
    case 'log':
      return 'primary'
    default: return 'grey'
  }
}

async function uploadSelectedFiles() {
  const isFolderMode = uploadTab.value === 'folder'
  const totalFiles = isFolderMode ? folderUploadFiles.value.length : uploadFiles.value.length
  if (totalFiles === 0) return

  uploading.value = true
  uploadProgress.value = 0
  uploadError.value = ''
  uploadSuccess.value = ''

  try {
    let uploadedCount = 0

    if (isFolderMode) {
      for (const entry of folderUploadFiles.value) {
        const formData = new FormData()
        formData.append('file', entry.file)
        formData.append('relative_path', entry.relative_path)

        if (currentPath.value) {
          formData.append('folder', currentPath.value)
        }

        await api.post('/api/data/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        uploadedCount++
        uploadProgress.value = Math.round((uploadedCount / totalFiles) * 100)
      }
    } else {
      for (const file of uploadFiles.value) {
        const formData = new FormData()
        formData.append('file', file)

        // Upload to current folder if we're in a user-accessible directory
        if (currentPath.value) {
          formData.append('folder', currentPath.value)
        }

        await api.post('/api/data/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        uploadedCount++
        uploadProgress.value = Math.round((uploadedCount / totalFiles) * 100)
      }
    }

    uploadSuccess.value = `Successfully uploaded ${uploadedCount} file(s)`
    uploadFiles.value = []
    folderUploadFiles.value = []

    // Refresh the file list
    await loadFolders()

    // Auto close after success
    setTimeout(() => {
      closeUploadDialog()
    }, 1500)
  } catch (e: any) {
    if (!tryShowValidationError(e)) {
      uploadError.value = e.response?.data?.error || 'Upload failed'
    }
  } finally {
    uploading.value = false
  }
}

function closeUploadDialog() {
  showUploadDialog.value = false
  uploadFiles.value = []
  folderUploadFiles.value = []
  uploadTab.value = 'files'
  uploadProgress.value = 0
  uploadError.value = ''
  uploadSuccess.value = ''
  isDragging.value = false
}

// Download file
async function downloadFile(item: FileItem) {
  try {
    const response = await api.get('/api/data/download', {
      params: { path: item.path },
      responseType: 'blob'
    })
    const blobUrl = window.URL.createObjectURL(new Blob([response.data]))
    const a = document.createElement('a')
    a.href = blobUrl
    a.download = item.name
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(blobUrl)
    document.body.removeChild(a)
  } catch {
    notificationStore.showError('Download failed')
  }
}

// Delete methods - users can delete from their own folders
function canDeleteItem(item: FileItem): boolean {
  // Admins can delete anything
  if (authStore.isAdmin) return true

  const user = authStore.user
  if (!user) return false

  // Normalize paths for comparison
  const itemPath = item.path.toLowerCase().replace(/\\/g, '/')
  const currPath = currentPath.value.toLowerCase().replace(/\\/g, '/')

  // User can delete from their private folder
  if (user.private_folder) {
    const privateFolderLower = user.private_folder.toLowerCase()
    // Check if item is inside the private folder
    if (itemPath.includes(`/${privateFolderLower}/`) || itemPath.endsWith(`/${privateFolderLower}`)) {
      return true
    }
    // Also check if we're currently inside the private folder
    if (currPath.includes(`/${privateFolderLower}`) || currPath.endsWith(`/${privateFolderLower}`)) {
      return true
    }
  }

  // User can delete from their uploads folder
  const uploadsPattern = `/uploads/user_${user.id}`
  if (itemPath.includes(uploadsPattern) || currPath.includes(uploadsPattern)) {
    return true
  }

  return false
}

function confirmDelete(item: FileItem) {
  itemToDelete.value = item
  showDeleteDialog.value = true
}

function cancelDelete() {
  showDeleteDialog.value = false
  itemToDelete.value = null
}

async function executeDelete() {
  if (!itemToDelete.value) return

  try {
    deleting.value = true

    // Use the appropriate endpoint based on user role
    const endpoint = authStore.isAdmin
      ? '/api/data/admin/delete'
      : '/api/data/delete-upload'

    await api.post(endpoint, {
      file_path: itemToDelete.value.path
    })

    notificationStore.showSuccess(`Deleted: ${itemToDelete.value.name}`)

    // If the deleted item was selected, clear selection
    if (selectedFile.value?.path === itemToDelete.value.path) {
      selectedFile.value = null
      dataPreview.value = null
    }

    // Refresh the file list
    await loadFolders()

    // Close dialog
    showDeleteDialog.value = false
    itemToDelete.value = null
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Delete failed')
  } finally {
    deleting.value = false
  }
}

// Sensor recording methods
async function startRecording() {
  recordStarting.value = true
  recordError.value = ''

  try {
    const response = await api.post('/api/sensors/start', {
      mode: recordMode.value,
      duration: recordDuration.value,
      rate: recordRate.value,
      label: recordLabel.value,
      filename: recordFilename.value || undefined,
    })

    recordJobId.value = response.data.id
    recordHasGpu.value = response.data.has_gpu
    recording.value = true
    recordingDone.value = false
    recordCurrentPhase.value = response.data.current_phase || ''

    // Start polling
    recordPollTimer = setInterval(pollRecordingStatus, 500)
  } catch (e: any) {
    recordError.value = e.response?.data?.error || 'Failed to start recording'
  } finally {
    recordStarting.value = false
  }
}

async function pollRecordingStatus() {
  if (!recordJobId.value) return

  try {
    const response = await api.get(`/api/sensors/status/${recordJobId.value}`)
    const job = response.data

    recordElapsed.value = job.elapsed || 0
    recordCurrentPhase.value = job.current_phase || ''
    recordTotalSamples.value = job.samples_collected || 0
    recordProgress.value = job.total_expected > 0
      ? Math.min(100, (job.samples_collected / job.total_expected) * 100)
      : 0

    if (job.status === 'completed') {
      recording.value = false
      recordingDone.value = true
      recordOutputFilename.value = job.output_filename || ''
      recordOutputPath.value = job.output_path || ''
      if (recordPollTimer) {
        clearInterval(recordPollTimer)
        recordPollTimer = null
      }
    }
  } catch (e: any) {
    // Silently continue polling
  }
}

async function stopRecording() {
  if (!recordJobId.value) return

  try {
    await api.post(`/api/sensors/stop/${recordJobId.value}`)
    // Polling will detect the completed status
  } catch (e: any) {
    recordError.value = e.response?.data?.error || 'Failed to stop recording'
  }
}

async function loadRecordedData() {
  if (!recordOutputPath.value) return

  try {
    loading.value = true
    // Set format to CSV and preview the recorded file
    selectedFormat.value = 'csv'

    const response = await api.post('/api/data/preview', {
      file_path: recordOutputPath.value,
      rows: 100,
      format: 'csv',
    })

    dataPreview.value = response.data
    selectedFile.value = {
      name: recordOutputFilename.value,
      path: recordOutputPath.value,
      is_dir: false,
      extension: '.csv',
      size: null,
      file_type: 'csv',
    }
    selectedFiles.value = []

    closeRecordDialog()
    notificationStore.showSuccess('Sensor recording loaded successfully')

    // Refresh file browser to show the new file
    await loadFolders()
  } catch (e: any) {
    recordError.value = e.response?.data?.error || 'Failed to load recorded data'
  } finally {
    loading.value = false
  }
}

function closeRecordDialog() {
  if (recording.value) return // Don't close while recording
  showRecordDialog.value = false
  recording.value = false
  recordingDone.value = false
  recordJobId.value = null
  recordElapsed.value = 0
  recordProgress.value = 0
  recordCurrentPhase.value = ''
  recordTotalSamples.value = 0
  recordOutputFilename.value = ''
  recordOutputPath.value = ''
  recordError.value = ''
  if (recordPollTimer) {
    clearInterval(recordPollTimer)
    recordPollTimer = null
  }
}

onMounted(async () => {
  // Adopt the project from the URL first — this triggers hydration
  // (POST /api/projects/<id>/hydrate) which repopulates
  // pipelineStore.dataSession from persisted state after a backend restart.
  const qpid = route.query.project_id
  if (qpid && !pipelineStore.projectId) {
    const idNum = Array.isArray(qpid) ? Number(qpid[0]) : Number(qpid)
    if (!Number.isNaN(idNum)) {
      await pipelineStore.setActiveProject(idNum)
    }
  }

  // If hydration (or a still-warm store) has a dataSession, mirror it into
  // the local dataPreview ref so the view renders as if the user had just
  // loaded the file. Without this the picker shows an empty Root even
  // though the Projects list clearly shows "csv · 623 rows".
  if (pipelineStore.dataSession && !dataPreview.value) {
    dataPreview.value = pipelineStore.dataSession
  }

  loadFolders()
})

// ═══════════════════════════════════════════════════════════════════════
// Phase G — Label mode (see docs/PLAN_2026-07-22_labeler-and-profile-swap.md)
// ═══════════════════════════════════════════════════════════════════════

const isLabelMode = computed(() => chartMode.value === 'label')

// Only meaningful for a single-CSV, single-file preview. Multi-CSV / cross-
// sensor JOIN / URL loads / folder previews don't map to a single sidecar.
const labelModeAvailable = computed(() => {
  const meta = dataPreview.value?.metadata
  if (!meta) return false
  if (meta.is_multi_csv) return false
  if (meta.is_cross_sensor_join) return false
  if (meta.is_folder) return false
  if (meta.source_url) return false
  // Require a resolvable local file path.
  const path = selectedFile.value?.path || meta.file_path
  if (!path || typeof path !== 'string') return false
  return true
})

const chartCursor = computed<'grab' | 'grabbing' | 'crosshair'>(() => {
  if (isLabelMode.value) return 'crosshair'
  return panCursor.value
})

// Placeholders for the start/end inputs so users know the visible range.
const labelPlaceholderMin = computed(() => {
  const rows = dataPreview.value?.preview
  if (!rows?.length || !timestampColumn.value) return ''
  const first = Number(rows[0][timestampColumn.value])
  return Number.isFinite(first) ? first.toFixed(2) : ''
})
const labelPlaceholderMax = computed(() => {
  const rows = dataPreview.value?.preview
  if (!rows?.length || !timestampColumn.value) return ''
  const last = Number(rows[rows.length - 1][timestampColumn.value])
  return Number.isFinite(last) ? last.toFixed(2) : ''
})

const knownClassNames = computed(() => {
  const set = new Set<string>()
  for (const l of labels.value) if (l.class) set.add(l.class)
  return Array.from(set).sort()
})

const sortedLabels = computed(() =>
  [...labels.value].sort((a, b) => a.from - b.from),
)

const hasUnsavedLabels = computed(() => {
  return JSON.stringify(labels.value) !== savedLabelsSignature.value
})

const canApplyLabel = computed(() => {
  if (labelStart.value === null || labelEnd.value === null) return false
  if (!Number.isFinite(labelStart.value) || !Number.isFinite(labelEnd.value)) return false
  if (!labelClass.value || !String(labelClass.value).trim()) return false
  return true
})

const labelCoverageDisplay = computed(() => {
  // Coverage % against the currently visible dataset window. Multi-batch
  // labels that extend past the window still count for the visible span.
  const rows = dataPreview.value?.preview
  if (!rows?.length || !timestampColumn.value) return '—'
  const first = Number(rows[0][timestampColumn.value])
  const last = Number(rows[rows.length - 1][timestampColumn.value])
  if (!Number.isFinite(first) || !Number.isFinite(last) || last <= first) return '—'
  const totalSpan = last - first
  let covered = 0
  const clamped = labels.value
    .map((l) => [Math.max(first, l.from), Math.min(last, l.to)] as [number, number])
    .filter(([lo, hi]) => hi > lo)
    .sort((a, b) => a[0] - b[0])
  let cursor = first
  for (const [lo, hi] of clamped) {
    const s = Math.max(cursor, lo)
    if (hi > s) covered += hi - s
    if (hi > cursor) cursor = hi
  }
  return `${((covered / totalSpan) * 100).toFixed(0)}%`
})

const labelsLastSavedDisplay = computed(() => {
  if (!labelsLastSavedAt.value) return ''
  try {
    const d = new Date(labelsLastSavedAt.value)
    return d.toLocaleTimeString()
  } catch {
    return labelsLastSavedAt.value
  }
})

// Deterministic pastel-ish color per class name (hash to hue).
function classColor(name: string): string {
  let h = 0
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) >>> 0
  }
  const hue = h % 360
  return `hsl(${hue}, 62%, 55%)`
}

// ── Chart plugin — draws vertical placement lines + horizontal bars ─────
//
// Registered per-Line component instance via the `:plugins` prop so it
// only runs for the DataSourceView chart, and re-reads reactive state on
// each draw (Chart.js calls plugin.afterDatasetsDraw every render).
const labelChartPlugins = computed(() => [
  {
    id: 'cira-labels-overlay',
    afterDatasetsDraw(chart: any) {
      const ctx = chart.ctx
      const xScale = chart.scales.x
      const yScale = chart.scales.y
      if (!xScale || !yScale) return
      const rows: any[] = dataPreview.value?.preview || []
      const tsCol = timestampColumn.value
      if (!tsCol) return

      // Map an absolute time value to a pixel X inside the chart. The chart's
      // x-axis is category-scale (row index) so we convert time → row index
      // via linear interpolation, then use the scale.
      const nRows = rows.length
      if (nRows < 2) return
      const firstT = Number(rows[0][tsCol])
      const lastT = Number(rows[nRows - 1][tsCol])
      if (!Number.isFinite(firstT) || !Number.isFinite(lastT) || lastT <= firstT) return

      const timeToIndex = (t: number) => {
        const clamped = Math.max(firstT, Math.min(lastT, t))
        return ((clamped - firstT) / (lastT - firstT)) * (nRows - 1)
      }
      const timeToPx = (t: number) => {
        try {
          return xScale.getPixelForValue(timeToIndex(t))
        } catch {
          return null
        }
      }

      // 1) Bar overlays for saved labels — thin strip near the bottom axis.
      const barHeight = 6
      const barY = chart.chartArea.bottom - barHeight - 1
      for (const l of labels.value) {
        const lo = Math.max(firstT, l.from)
        const hi = Math.min(lastT, l.to)
        if (hi <= lo) continue
        const px1 = timeToPx(lo)
        const px2 = timeToPx(hi)
        if (px1 === null || px2 === null) continue
        ctx.save()
        ctx.fillStyle = classColor(l.class) + 'cc'
        ctx.fillRect(px1, barY, Math.max(1, px2 - px1), barHeight)
        ctx.restore()
      }

      // 2) Pending start / end vertical lines (label mode only).
      if (!isLabelMode.value) return
      const drawLine = (t: number | null, color: string, tag: string) => {
        if (t === null || !Number.isFinite(t)) return
        const px = timeToPx(t)
        if (px === null) return
        ctx.save()
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.setLineDash([4, 3])
        ctx.beginPath()
        ctx.moveTo(px, chart.chartArea.top)
        ctx.lineTo(px, chart.chartArea.bottom)
        ctx.stroke()
        // Tag label at top.
        ctx.setLineDash([])
        ctx.fillStyle = color
        ctx.font = '10px sans-serif'
        ctx.fillText(tag, px + 4, chart.chartArea.top + 12)
        ctx.restore()
      }
      drawLine(labelStart.value, '#2e7d32', `start ${labelStart.value?.toFixed(2) ?? ''}`)
      drawLine(labelEnd.value, '#c62828', `end ${labelEnd.value?.toFixed(2) ?? ''}`)
    },
  },
])

// ── Chart click handler (label mode only) ───────────────────────────────

function pxToTimeValue(clientX: number): number | null {
  const chart = chartRef.value?.chart
  if (!chart) return null
  const rect = chart.canvas.getBoundingClientRect()
  const px = clientX - rect.left
  const xScale = chart.scales.x
  let idx: number
  try {
    idx = xScale.getValueForPixel(px)
  } catch {
    return null
  }
  const rows = dataPreview.value?.preview
  if (!rows?.length || !timestampColumn.value) return null
  const nRows = rows.length
  const firstT = Number(rows[0][timestampColumn.value])
  const lastT = Number(rows[nRows - 1][timestampColumn.value])
  if (!Number.isFinite(firstT) || !Number.isFinite(lastT) || lastT <= firstT) return null
  const clampedIdx = Math.max(0, Math.min(nRows - 1, idx))
  return firstT + (clampedIdx / (nRows - 1)) * (lastT - firstT)
}

function onChartClick(e: MouseEvent) {
  if (!isLabelMode.value) return
  const t = pxToTimeValue(e.clientX)
  if (t === null) return
  labelValidationError.value = null
  if (labelStart.value === null) {
    labelStart.value = Number(t.toFixed(3))
    return
  }
  if (labelEnd.value === null) {
    labelEnd.value = Number(t.toFixed(3))
    return
  }
  // Both already placed — nearest one gets moved.
  const dStart = Math.abs(t - (labelStart.value ?? 0))
  const dEnd = Math.abs(t - (labelEnd.value ?? 0))
  if (dStart <= dEnd) labelStart.value = Number(t.toFixed(3))
  else labelEnd.value = Number(t.toFixed(3))
}

function onChartDoubleClick(e: MouseEvent) {
  // Label mode: double-click clears placement so user can start over.
  if (isLabelMode.value) {
    cancelPendingLabel()
    return
  }
  resetZoom()
}

// ── Apply / edit / delete labels ────────────────────────────────────────

function _overlapsOther(from: number, to: number, excludeIdx: number | null): number {
  for (let i = 0; i < labels.value.length; i++) {
    if (i === excludeIdx) continue
    const l = labels.value[i]
    // Half-open [from, to): touching edges is fine (idle 0→45.12, fault 45.12→60.20).
    if (from < l.to && l.from < to) return i
  }
  return -1
}

function applyPendingLabel() {
  labelValidationError.value = null
  if (labelStart.value === null || labelEnd.value === null || !labelClass.value) return
  let from = Number(labelStart.value)
  let to = Number(labelEnd.value)
  if (!Number.isFinite(from) || !Number.isFinite(to)) {
    labelValidationError.value = 'Start and end must be numbers.'
    return
  }
  if (from > to) [from, to] = [to, from]  // spec: auto-swap
  if (from === to) {
    labelValidationError.value = 'Start and end must differ.'
    return
  }
  const cls = String(labelClass.value).trim()
  if (!cls) {
    labelValidationError.value = 'Class is required.'
    return
  }
  const overlap = _overlapsOther(from, to, editingLabelIndex.value)
  if (overlap >= 0) {
    const other = labels.value[overlap]
    labelValidationError.value = `Range overlaps existing label "${other.class}" at ${other.from.toFixed(2)}–${other.to.toFixed(2)}. Delete or shrink that one first.`
    return
  }
  const entry: LabelEntry = { from, to, class: cls }
  if (editingLabelIndex.value !== null) {
    labels.value.splice(editingLabelIndex.value, 1, entry)
  } else {
    labels.value.push(entry)
  }
  cancelPendingLabel()
}

function cancelPendingLabel() {
  labelStart.value = null
  labelEnd.value = null
  labelClass.value = ''
  editingLabelIndex.value = null
  labelValidationError.value = null
}

function startEditLabel(sortedIdx: number) {
  // sortedLabels is the display order — resolve back to the source index.
  const sorted = sortedLabels.value
  const target = sorted[sortedIdx]
  if (!target) return
  const idx = labels.value.findIndex(
    (l) => l.from === target.from && l.to === target.to && l.class === target.class,
  )
  if (idx < 0) return
  editingLabelIndex.value = idx
  labelStart.value = target.from
  labelEnd.value = target.to
  labelClass.value = target.class
  chartMode.value = 'label'  // switch modes so lines are visible
}

function previewLabel(sortedIdx: number) {
  // Row-click behaviour: show the two vertical lines on the chart at this
  // label's range without entering edit mode. Distinct from startEditLabel
  // (which the pencil button calls) — a preview click that lands you in
  // "replace mode" would surprise-overwrite the label on the next Apply.
  const sorted = sortedLabels.value
  const target = sorted[sortedIdx]
  if (!target) return
  editingLabelIndex.value = null
  labelStart.value = target.from
  labelEnd.value = target.to
  labelClass.value = ''  // stays neutral so Apply-without-edit acts as new
  chartMode.value = 'label'
}

function deleteLabel(sortedIdx: number) {
  const sorted = sortedLabels.value
  const target = sorted[sortedIdx]
  if (!target) return
  const idx = labels.value.findIndex(
    (l) => l.from === target.from && l.to === target.to && l.class === target.class,
  )
  if (idx < 0) return
  labels.value.splice(idx, 1)
  if (editingLabelIndex.value === idx) cancelPendingLabel()
}

// ── Sidecar sync (GET on load, PUT on advance) ──────────────────────────

function _currentCsvPath(): string | null {
  const path = selectedFile.value?.path || dataPreview.value?.metadata?.file_path
  if (!path || typeof path !== 'string') return null
  return path
}

async function hydrateLabels(force = false) {
  const path = _currentCsvPath()
  if (!path) return
  if (!force && labelsHydratedForPath.value === path) return
  try {
    const resp = await api.get('/api/data/labels', { params: { csv_path: path } })
    const body = resp.data || {}
    labels.value = Array.isArray(body.labels) ? body.labels.map((l: any) => ({
      from: Number(l.from), to: Number(l.to), class: String(l.class),
    })) : []
    sidecarXColumn.value = body.x_column || null
    labelsLastSavedAt.value = body.updated_at || null
    savedLabelsSignature.value = JSON.stringify(labels.value)
    labelsHydratedForPath.value = path
    if (body.warning) {
      notificationStore.showError(`Labels sidecar warning: ${body.warning}`)
    }
  } catch (e: any) {
    // Non-fatal — the labeler UI still works without hydration.
    console.warn('[labels] hydrate failed', e)
  }
}

async function saveLabels(): Promise<boolean> {
  const path = _currentCsvPath()
  if (!path) return false
  savingLabels.value = true
  try {
    const resp = await api.put('/api/data/labels', {
      csv_path: path,
      x_column: sidecarXColumn.value || timestampColumn.value || null,
      labels: labels.value,
    })
    savedLabelsSignature.value = JSON.stringify(labels.value)
    labelsLastSavedAt.value = resp.data?.updated_at || new Date().toISOString()
    notificationStore.showSuccess(`Saved ${labels.value.length} label${labels.value.length === 1 ? '' : 's'}`)
    return true
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to save labels')
    return false
  } finally {
    savingLabels.value = false
  }
}

// Hydrate labels whenever a new CSV becomes the current preview target.
watch(
  () => _currentCsvPath(),
  (newPath, oldPath) => {
    // Reset state on file switch so labels from a prior file don't leak.
    if (newPath !== oldPath) {
      labels.value = []
      savedLabelsSignature.value = '[]'
      cancelPendingLabel()
      labelsHydratedForPath.value = null
      labelsLastSavedAt.value = null
    }
    if (newPath && labelModeAvailable.value) {
      hydrateLabels()
    }
  },
)

// Also hydrate when labelModeAvailable becomes true (e.g. cross-folder
// basket collapses back to a single file), for good measure.
watch(labelModeAvailable, (v) => {
  if (v) hydrateLabels()
})

// ── Auto-save hook on "Load More Rows" ──────────────────────────────────
//
// Wraps the existing loadMorePreview so unsaved labels get PUT'd first.
// If the save fails, we abort the load — the user sees the error toast
// and the batch counter doesn't advance.
const _originalLoadMorePreview = loadMorePreview
async function loadMorePreviewWithSave() {
  if (labelModeAvailable.value && hasUnsavedLabels.value) {
    const ok = await saveLabels()
    if (!ok) return  // don't advance — spec: "If save fails → toast, don't advance"
  }
  await _originalLoadMorePreview()
}

// ── beforeunload guard ──────────────────────────────────────────────────

function _beforeUnloadHandler(e: BeforeUnloadEvent) {
  if (!hasUnsavedLabels.value) return
  e.preventDefault()
  // Modern browsers ignore custom text but still show a generic warning
  // when returnValue is set.
  e.returnValue = ''
}
onMounted(() => window.addEventListener('beforeunload', _beforeUnloadHandler))
onBeforeUnmount(() => window.removeEventListener('beforeunload', _beforeUnloadHandler))

// Whenever labels change during editing, drop any focused validation error
// so the panel doesn't stick showing an outdated "overlaps" hint.
watch(labels, () => { labelValidationError.value = null }, { deep: true })
</script>

<style scoped lang="scss">
.file-list {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;

  .v-list-item {
    border-bottom: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));

    &:last-child {
      border-bottom: none;
    }

    &.selected {
      background: rgba(99, 102, 241, 0.1);
    }

    &:hover {
      background: rgba(var(--v-theme-surface-variant), 0.5);
    }
  }
}

.preview-table {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;

  :deep(.v-data-table__wrapper) {
    max-height: 400px;
    overflow-y: auto;
  }
}

// Phase G — label mode UI
.label-placement-panel {
  background: rgba(var(--v-theme-surface-variant), 0.35);
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;
}
.label-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
  border: 1px solid rgba(0, 0, 0, 0.15);
}
.label-row {
  border-bottom: 1px dashed rgba(var(--v-border-color), var(--v-border-opacity));
}
.label-row:last-child {
  border-bottom: none;
}

// Upload dropzone styles
.upload-dropzone {
  border: 2px dashed rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 12px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  background: rgba(var(--v-theme-surface-variant), 0.2);

  &:hover {
    border-color: rgb(var(--v-theme-primary));
    background: rgba(var(--v-theme-primary), 0.05);
  }

  &.drag-over {
    border-color: rgb(var(--v-theme-primary));
    background: rgba(var(--v-theme-primary), 0.1);
    border-style: solid;
  }

  &.has-files {
    border-color: rgb(var(--v-theme-success));
    background: rgba(var(--v-theme-success), 0.05);
  }
}

.upload-file-item {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 8px;
  margin-bottom: 4px;
}

.format-section-title {
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.format-subsection {
  padding: 4px 0;
}

.format-sub-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: rgba(var(--v-theme-on-surface), 0.72);
  margin-bottom: 4px;
}

.format-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.format-example {
  background: rgba(var(--v-theme-surface-variant), 0.4);
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 6px;
  padding: 10px 12px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
  margin: 0;
  overflow-x: auto;
  white-space: pre;
}

.text-import-preview-wrap {
  max-height: 340px;
  overflow: auto;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 6px;
}

.text-import-preview {
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;

  :deep(th) {
    font-weight: 600;
    white-space: nowrap;
    background: rgba(var(--v-theme-surface-variant), 0.5);
  }

  :deep(td) {
    white-space: nowrap;
  }
}
</style>
