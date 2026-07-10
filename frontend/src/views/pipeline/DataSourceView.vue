<template>
  <v-container fluid class="pa-6">
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
                    Fetch a CSV / text file over HTTPS (max 100 MB). Includes Loghub catalog.
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
              color="primary"
              variant="tonal"
              class="ml-2"
            >
              {{ selectedFiles.length }} / {{ csvFilesInFolder.length }} selected
            </v-chip>
            <v-spacer />
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

          <!-- Multi-CSV selection info -->
          <v-alert
            v-if="isCsvFormat && selectedFiles.length > 1"
            type="info"
            variant="tonal"
            class="mt-4"
          >
            <div class="font-weight-medium">{{ selectedFiles.length }} CSV files selected</div>
            <div class="text-caption">
              {{ selectedFiles.map(f => f.name).join(', ') }}
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

          <!-- Loghub Quick-Pick chips -->
          <div class="mb-2">
            <div class="text-subtitle-2 font-weight-medium">
              <v-icon size="small" color="primary" class="mr-1">mdi-lightning-bolt</v-icon>
              Quick-pick from Loghub samples
              <span class="text-caption text-medium-emphasis ml-1">(~200-400 KB each)</span>
            </div>
          </div>
          <div class="mb-2">
            <v-chip
              v-for="key in LOGHUB_QUICK_PICK_KEYS"
              :key="key"
              class="mr-2 mb-1"
              color="primary"
              variant="tonal"
              size="small"
              prepend-icon="mdi-database-outline"
              @click="pickLoghubSample(key)"
            >
              {{ key }}
            </v-chip>
          </div>

          <!-- Loghub Full Catalog (expansion panel) -->
          <v-expansion-panels variant="accordion" class="mt-3">
            <v-expansion-panel>
              <v-expansion-panel-title>
                <v-icon class="mr-2" color="secondary">mdi-book-open-variant</v-icon>
                Loghub — Full Dataset Catalog
                <v-chip size="x-small" color="secondary" variant="tonal" class="ml-2">
                  {{ LOGHUB_CATALOG.length }} datasets
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
                  Loghub datasets are freely available for research or academic work.
                  Confirm your license needs before commercial use.
                </v-alert>

                <div
                  v-for="category in loghubCategories"
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
                        <th class="text-right">#Lines</th>
                        <th class="text-right">Raw Size</th>
                        <th class="text-center">Sample (200 KB)</th>
                        <th class="text-center">Full download</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr
                        v-for="entry in loghubByCategory[category]"
                        :key="entry.key"
                      >
                        <td class="font-weight-medium">{{ entry.key }}</td>
                        <td>{{ entry.description }}</td>
                        <td class="text-center">
                          <v-icon v-if="entry.labeled" size="small" color="success">mdi-check</v-icon>
                          <span v-else class="text-medium-emphasis">—</span>
                        </td>
                        <td class="text-right">{{ entry.lines.toLocaleString() }}</td>
                        <td class="text-right">{{ entry.sizeRaw }}</td>
                        <td class="text-center">
                          <v-btn
                            v-if="entry.sampleUrl"
                            size="x-small"
                            variant="tonal"
                            color="primary"
                            @click="pickLoghubSample(entry.key)"
                          >
                            Use
                          </v-btn>
                          <span v-else class="text-caption text-medium-emphasis">N/A</span>
                        </td>
                        <td class="text-center">
                          <a
                            :href="entry.fullUrl"
                            target="_blank"
                            rel="noopener"
                            class="text-caption"
                          >
                            Zenodo
                            <v-icon size="x-small">mdi-open-in-new</v-icon>
                          </a>
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
    <v-card v-if="dataPreview" class="mt-6 pa-4">
      <div class="d-flex align-center mb-4">
        <h3 class="text-subtitle-1 font-weight-bold">Data Preview</h3>
        <v-chip
          v-if="dataPreview.metadata.is_partition_preview"
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
            <div class="d-flex align-center mb-2">
              <v-icon size="small" class="mr-1" color="primary">mdi-chart-line</v-icon>
              <span class="text-subtitle-2 font-weight-medium">
                Signal Visualization
              </span>
              <v-chip size="x-small" color="primary" variant="tonal" class="ml-2">
                {{ plottableColumns.length }} signal{{ plottableColumns.length === 1 ? '' : 's' }}
              </v-chip>
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
              :style="{ height: '340px', position: 'relative', cursor: panCursor, userSelect: 'none' }"
              @wheel="onChartWheel"
              @mousedown="onChartMouseDown"
              @mousemove="onChartMouseMove"
              @mouseup="onChartMouseUp"
              @mouseleave="onChartMouseUp"
              @dblclick="resetZoom"
            >
              <Line ref="chartRef" :data="chartData" :options="chartOptions" />
            </div>
            <div class="text-caption text-medium-emphasis mt-1">
              X-axis: {{ timestampColumn }} &nbsp;·&nbsp; <strong>Scroll</strong> to zoom, <strong>drag</strong> to pan, <strong>double-click</strong> to reset. Toggle signals via the column headers above.
            </div>
          </v-card>
        </div>
      </v-expand-transition>

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
            :loading="loadingMore"
            @click="loadMorePreview"
          >
            <v-icon start>mdi-plus</v-icon>
            Load More Rows
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
            {{ dataPreview.metadata.labels?.join(', ') || 'None' }}
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

                  <v-list-item-title>{{ entry.relative_path }}</v-list-item-title>
                  <v-list-item-subtitle>{{ formatFileSize(entry.file.size) }}</v-list-item-subtitle>
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
                <v-list-item-title class="text-body-2"><strong>Load from URL</strong> — fetch a CSV / text file over <code>https://</code> (max 100 MB, streamed to memory only). Includes a hardcoded catalog of Loghub log datasets with one-click samples for quick experimentation.</v-list-item-title>
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
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import { useAuthStore } from '@/stores/auth'
import PipelineStepper from '@/components/PipelineStepper.vue'
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

// Dataset scan & partition state
const datasetScan = ref<any>(null)
const selectedCategory = ref<string | null>(null)
const selectedLabel = ref<string | null>(null)
const scanning = ref(false)

// Upload state
const showUploadDialog = ref(false)
const showFormatGuide = ref(false)

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
      return 'Fetch a CSV or delimited text file over HTTPS. Streamed to memory only (never written to disk) with a 100 MB hard cap. Quick-pick chips below load small Loghub samples.'
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
  loading: boolean
  error: string
}

const urlLoader = ref<UrlLoaderState>({
  url: '',
  format: 'csv',
  delimiter: '',
  headerRow: 1,
  skipRows: 0,
  loading: false,
  error: '',
})

// --- Loghub catalog (hardcoded, from loghub README) -----------------------
interface LoghubEntry {
  key: string
  category: string
  description: string
  labeled: boolean
  lines: number
  sizeRaw: string
  sampleUrl: string | null
  fullUrl: string
}

const LOGHUB_CATALOG: LoghubEntry[] = [
  // Distributed systems
  { key: 'HDFS_v1', category: 'Distributed systems', description: 'Hadoop distributed file system log', labeled: true, lines: 11175629, sizeRaw: '1.47 GiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1' },
  { key: 'HDFS_v2', category: 'Distributed systems', description: 'Hadoop distributed file system log', labeled: false, lines: 71118073, sizeRaw: '16.06 GiB',
    sampleUrl: null,
    fullUrl: 'https://zenodo.org/records/8196385/files/HDFS_v2.zip?download=1' },
  { key: 'HDFS_v3', category: 'Distributed systems', description: 'Instrumented HDFS trace log (TraceBench)', labeled: true, lines: 14778079, sizeRaw: '2.96 GiB',
    sampleUrl: null,
    fullUrl: 'https://zenodo.org/records/8196385/files/HDFS_v3_TraceBench.zip?download=1' },
  { key: 'Hadoop', category: 'Distributed systems', description: 'Hadoop MapReduce job log', labeled: true, lines: 394308, sizeRaw: '48.61 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Hadoop/Hadoop_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Hadoop.zip?download=1' },
  { key: 'Spark', category: 'Distributed systems', description: 'Spark job log', labeled: false, lines: 33236604, sizeRaw: '2.75 GiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Spark/Spark_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Spark.tar.gz?download=1' },
  { key: 'Zookeeper', category: 'Distributed systems', description: 'ZooKeeper service log', labeled: false, lines: 74380, sizeRaw: '9.95 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Zookeeper/Zookeeper_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1' },
  { key: 'OpenStack', category: 'Distributed systems', description: 'OpenStack infrastructure log', labeled: true, lines: 207820, sizeRaw: '58.61 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/OpenStack/OpenStack_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1' },
  // Super computers
  { key: 'BGL', category: 'Super computers', description: 'Blue Gene/L supercomputer log', labeled: true, lines: 4747963, sizeRaw: '708.76 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/BGL.zip?download=1' },
  { key: 'HPC', category: 'Super computers', description: 'High performance cluster log', labeled: false, lines: 433489, sizeRaw: '32.00 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/HPC/HPC_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/HPC.zip?download=1' },
  { key: 'Thunderbird', category: 'Super computers', description: 'Thunderbird supercomputer log', labeled: true, lines: 211212192, sizeRaw: '29.60 GiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1' },
  // Operating systems
  { key: 'Windows', category: 'Operating systems', description: 'Windows event log', labeled: false, lines: 114608388, sizeRaw: '26.09 GiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Windows/Windows_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Windows.tar.gz?download=1' },
  { key: 'Linux', category: 'Operating systems', description: 'Linux system log', labeled: false, lines: 25567, sizeRaw: '2.25 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1' },
  { key: 'Mac', category: 'Operating systems', description: 'Mac OS log', labeled: false, lines: 117283, sizeRaw: '16.09 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Mac/Mac_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1' },
  // Mobile systems
  { key: 'Android_v1', category: 'Mobile systems', description: 'Android framework log', labeled: false, lines: 1555005, sizeRaw: '183.37 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Android/Android_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Android_v1.zip?download=1' },
  { key: 'Android_v2', category: 'Mobile systems', description: 'Android framework log', labeled: false, lines: 30348042, sizeRaw: '3.38 GiB',
    sampleUrl: null,
    fullUrl: 'https://zenodo.org/records/8196385/files/Android_v2.zip?download=1' },
  { key: 'HealthApp', category: 'Mobile systems', description: 'Health app log', labeled: false, lines: 253395, sizeRaw: '22.44 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/HealthApp.tar.gz?download=1' },
  // Server applications
  { key: 'Apache', category: 'Server applications', description: 'Apache web server error log', labeled: false, lines: 56481, sizeRaw: '4.90 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1' },
  { key: 'OpenSSH', category: 'Server applications', description: 'OpenSSH server log', labeled: false, lines: 655146, sizeRaw: '70.02 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/OpenSSH/OpenSSH_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1' },
  // Standalone software
  { key: 'Proxifier', category: 'Standalone software', description: 'Proxifier software log', labeled: false, lines: 21329, sizeRaw: '2.42 MiB',
    sampleUrl: 'https://raw.githubusercontent.com/logpai/loghub/master/Proxifier/Proxifier_2k.log_structured.csv',
    fullUrl: 'https://zenodo.org/records/8196385/files/Proxifier.tar.gz?download=1' },
]

const LOGHUB_QUICK_PICK_KEYS = [
  'Apache', 'Linux', 'Mac', 'HealthApp', 'OpenSSH',
  'Proxifier', 'HDFS_v1', 'Zookeeper', 'Hadoop',
]

const loghubCategories = computed(() => {
  const seen: string[] = []
  for (const e of LOGHUB_CATALOG) {
    if (!seen.includes(e.category)) seen.push(e.category)
  }
  return seen
})

const loghubByCategory = computed(() => {
  const groups: Record<string, LoghubEntry[]> = {}
  for (const e of LOGHUB_CATALOG) {
    if (!groups[e.category]) groups[e.category] = []
    groups[e.category].push(e)
  }
  return groups
})

function pickLoghubSample(key: string) {
  const entry = LOGHUB_CATALOG.find(e => e.key === key)
  if (!entry || !entry.sampleUrl) return
  urlLoader.value.url = entry.sampleUrl
  urlLoader.value.format = 'csv'
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
    }

    const response = await api.post('/api/data/load-from-url', payload)
    dataPreview.value = response.data
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
  selectedFiles.value = []
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
  try {
    loading.value = true
    multiCsvError.value = null

    const response = await api.post('/api/data/preview', {
      file_paths: selectedFiles.value.map(f => f.path),
      rows: 100,
      format: 'csv'
    })

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

async function handleItemClick(item: FileItem) {
  if (item.is_dir) {
    currentPath.value = item.path
    datasetScan.value = null
    selectedCategory.value = null
    selectedLabel.value = null
    dataPreview.value = null
    selectedFile.value = null
    selectedFiles.value = []
    multiCsvError.value = null
    await loadFolders()
  } else if (isCsvFormat.value && item.extension === '.csv') {
    // CSV multi-select mode
    toggleCsvFile(item)
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
    try {
      loadingFull.value = true
      loading.value = true

      const response = await api.post('/api/data/ingest/csv-multiple', {
        file_paths: selectedFiles.value.map(f => f.path)
      })

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
