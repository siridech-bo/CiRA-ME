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
      <v-col cols="12" md="8">
        <v-card class="pa-4">
          <div class="d-flex align-center mb-4">
            <h3 class="text-subtitle-1 font-weight-bold">Browse Files</h3>
            <v-spacer />
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
              :class="{ 'selected': selectedFile?.path === item.path }"
              @click="handleItemClick(item)"
            >
              <template #prepend>
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
                <span v-if="!item.is_dir" class="text-caption text-medium-emphasis">
                  {{ formatFileSize(item.size) }}
                </span>
              </template>
            </v-list-item>

            <v-list-item v-if="currentItems.length === 0">
              <v-list-item-title class="text-medium-emphasis">
                No files found in this directory
              </v-list-item-title>
            </v-list-item>
          </v-list>

          <!-- Selected File/Folder Info -->
          <v-alert
            v-if="selectedFile"
            type="success"
            variant="tonal"
            class="mt-4"
          >
            <div class="font-weight-medium">Selected: {{ selectedFile.name }}</div>
            <div class="text-caption">{{ selectedFile.path }}</div>
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

      <v-data-table
        v-model:items-per-page="previewItemsPerPage"
        :headers="previewHeaders"
        :items="dataPreview.preview"
        :items-per-page-options="[10, 25, 50, 100]"
        density="compact"
        class="preview-table"
      >
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

      <!-- Load More Info -->
      <div class="d-flex align-center justify-space-between pa-2 mt-2">
        <span class="text-caption text-medium-emphasis">
          Loaded {{ dataPreview.preview.length }} of {{ dataPreview.metadata.total_rows.toLocaleString() }} total rows
        </span>
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
            {{ dataPreview.metadata.is_folder ? 'Samples' : 'Session ID' }}
          </div>
          <div class="font-weight-medium text-truncate">
            {{ dataPreview.metadata.is_folder
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
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { usePipelineStore } from '@/stores/pipeline'
import { useNotificationStore } from '@/stores/notification'
import PipelineStepper from '@/components/PipelineStepper.vue'
import api from '@/services/api'

interface FileItem {
  name: string
  path: string
  is_dir: boolean
  extension: string | null
  size: number | null
  file_type: string | null
}

const router = useRouter()
const pipelineStore = usePipelineStore()
const notificationStore = useNotificationStore()

const selectedFormat = ref('csv')
const currentPath = ref('')
const currentItems = ref<FileItem[]>([])
const selectedFile = ref<FileItem | null>(null)
const dataPreview = ref<any>(null)
const loading = ref(false)
const loadingFolders = ref(false)
const loadingMore = ref(false)
const loadingFull = ref(false)
const previewItemsPerPage = ref(10)
const maxPreviewRows = 500

// Dataset scan & partition state
const datasetScan = ref<any>(null)
const selectedCategory = ref<string | null>(null)
const selectedLabel = ref<string | null>(null)
const scanning = ref(false)

const formatInfo = computed(() => {
  switch (selectedFormat.value) {
    case 'csv':
      return 'Headers in first row. Requires numeric sensor columns and optional "label" column.'
    case 'ei-json':
      return 'Standard Edge Impulse JSON export format with sensors and values arrays.'
    case 'ei-cbor':
      return 'Select a dataset folder containing training/testing subfolders. Classes are auto-detected from filenames.'
    case 'cira-cbor':
      return 'Select a dataset folder with train/test subfolders. Classes are auto-detected from filenames.'
    default:
      return ''
  }
})

const isCborFormat = computed(() => {
  return selectedFormat.value === 'ei-cbor' || selectedFormat.value === 'cira-cbor'
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
  const parts = currentPath.value.split(/[/\\]/).filter(Boolean)
  const items = [{ title: 'Root', path: '', disabled: false }]

  let path = ''
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

function getFileIcon(ext: string | null) {
  switch (ext) {
    case '.csv': return 'mdi-file-delimited'
    case '.json': return 'mdi-code-json'
    case '.cbor': return 'mdi-file-code'
    default: return 'mdi-file'
  }
}

function getFileColor(ext: string | null) {
  switch (ext) {
    case '.csv': return 'success'
    case '.json': return 'info'
    case '.cbor': return 'secondary'
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
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to load files')
  } finally {
    loadingFolders.value = false
  }
}

function navigateTo(path: string) {
  currentPath.value = path
  datasetScan.value = null
  selectedCategory.value = null
  selectedLabel.value = null
  dataPreview.value = null
  selectedFile.value = null
  loadFolders()
}

async function handleItemClick(item: FileItem) {
  if (item.is_dir) {
    currentPath.value = item.path
    datasetScan.value = null
    selectedCategory.value = null
    selectedLabel.value = null
    dataPreview.value = null
    selectedFile.value = null
    await loadFolders()
  } else {
    selectedFile.value = item
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
    notificationStore.showError(e.response?.data?.error || 'Failed to preview file')
    dataPreview.value = null
  } finally {
    loading.value = false
  }
}

async function loadMorePreview() {
  if (!dataPreview.value || !selectedFile.value) return

  try {
    loadingMore.value = true

    const currentRows = dataPreview.value.preview.length
    const newRowCount = Math.min(currentRows + 100, maxPreviewRows)

    const requestData: any = {
      file_path: selectedFile.value.path,
      rows: newRowCount,
      format: selectedFormat.value
    }

    // Preserve partition filters for dataset folder previews
    if (dataPreview.value.metadata?.is_partition_preview && selectedCategory.value) {
      requestData.category = selectedCategory.value
      requestData.label = selectedLabel.value
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

  // If this is a partition preview, load the full dataset first
  if (dataPreview.value.metadata?.is_partition_preview) {
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
  dataPreview.value = null
  datasetScan.value = null
  selectedCategory.value = null
  selectedLabel.value = null
})

onMounted(() => {
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
</style>
