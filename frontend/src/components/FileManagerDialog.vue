<template>
  <v-dialog
    :model-value="modelValue"
    fullscreen
    scrollable
    transition="dialog-bottom-transition"
    @update:model-value="onDialogUpdate"
  >
    <v-card class="d-flex flex-column" style="height: 100vh;">
      <!-- Top bar -->
      <v-toolbar density="compact" color="surface" flat>
        <v-toolbar-title class="d-flex align-center">
          <v-icon class="mr-2" color="primary">mdi-folder-cog-outline</v-icon>
          Manage Files
        </v-toolbar-title>
        <v-spacer />
        <v-btn
          variant="text"
          size="small"
          prepend-icon="mdi-refresh"
          :loading="loadingItems"
          @click="reloadCurrent"
        >
          Refresh
        </v-btn>
        <v-btn icon variant="text" @click="close">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-toolbar>

      <v-divider />

      <!-- Two-pane body -->
      <v-card-text class="pa-0 flex-grow-1 d-flex" style="overflow: hidden;">
        <!-- LEFT PANE: folder tree -->
        <div class="fm-left-pane">
          <div class="pa-2 text-caption text-medium-emphasis font-weight-medium">
            <v-icon size="small" class="mr-1">mdi-file-tree</v-icon>
            Folders
          </div>
          <v-divider />
          <div class="fm-tree-scroll">
            <FileTreeNode
              v-for="root in rootFolders"
              :key="root.path"
              :node="root"
              :selected-path="currentPath"
              :folders-only="true"
              @select="onTreeSelect"
            />
            <div v-if="rootFolders.length === 0 && !loadingRoots" class="pa-3 text-caption text-medium-emphasis">
              No accessible folders.
            </div>
          </div>
        </div>

        <v-divider vertical />

        <!-- RIGHT PANE: current folder contents -->
        <div class="fm-right-pane">
          <!-- Breadcrumb -->
          <div class="fm-breadcrumbs">
            <v-breadcrumbs :items="breadcrumbItems" density="compact" class="pa-0">
              <template #divider>
                <v-icon size="small">mdi-chevron-right</v-icon>
              </template>
              <template #item="{ item }">
                <v-breadcrumbs-item
                  :disabled="item.disabled"
                  style="cursor: pointer;"
                  @click="!item.disabled && navigateTo(item.path)"
                >
                  {{ item.title }}
                </v-breadcrumbs-item>
              </template>
            </v-breadcrumbs>
          </div>

          <!-- Toolbar -->
          <div class="fm-toolbar">
            <v-btn
              variant="tonal"
              size="small"
              color="primary"
              prepend-icon="mdi-folder-plus"
              :disabled="!currentPath"
              @click="openNewFolderDialog"
            >
              New Folder
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              color="primary"
              prepend-icon="mdi-upload"
              :disabled="!currentPath"
              @click="triggerUpload"
            >
              Upload
            </v-btn>
            <input
              ref="uploadInput"
              type="file"
              multiple
              hidden
              @change="handleUploadFiles"
            />
            <v-divider vertical class="mx-1" />
            <v-btn
              variant="tonal"
              size="small"
              prepend-icon="mdi-form-textbox"
              :disabled="selectedItems.length !== 1"
              @click="beginRename"
            >
              Rename
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              prepend-icon="mdi-folder-move"
              :disabled="selectedItems.length === 0"
              @click="openMoveDialog"
            >
              Move
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              prepend-icon="mdi-content-copy"
              :disabled="selectedItems.length === 0"
              @click="openCopyDialog"
            >
              Copy
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              color="error"
              prepend-icon="mdi-delete"
              :disabled="selectedItems.length === 0"
              @click="openDeleteConfirm"
            >
              Delete
            </v-btn>
            <v-btn
              variant="tonal"
              size="small"
              prepend-icon="mdi-download"
              :disabled="selectedItems.length !== 1 || selectedItems[0].is_dir"
              @click="downloadSelected"
            >
              Download
            </v-btn>
            <v-spacer />
            <v-chip
              v-if="selectedItems.length > 0"
              size="small"
              color="primary"
              variant="tonal"
            >
              {{ selectedItems.length }} selected
            </v-chip>
          </div>

          <!-- Upload progress -->
          <v-progress-linear
            v-if="uploading"
            :model-value="uploadProgress"
            color="primary"
            height="4"
            rounded
          />

          <v-divider />

          <!-- Table -->
          <div class="fm-table-scroll">
            <v-table density="compact" class="fm-table">
              <thead>
                <tr>
                  <th style="width: 40px;">
                    <v-checkbox-btn
                      :model-value="allSelected"
                      :indeterminate="someSelected && !allSelected"
                      density="compact"
                      @update:model-value="toggleSelectAll"
                    />
                  </th>
                  <th style="width: 40px;"></th>
                  <th @click="setSort('name')" style="cursor: pointer;">
                    Name
                    <v-icon v-if="sortBy === 'name'" size="x-small">
                      {{ sortDesc ? 'mdi-menu-down' : 'mdi-menu-up' }}
                    </v-icon>
                  </th>
                  <th class="text-right" @click="setSort('size')" style="cursor: pointer; width: 120px;">
                    Size
                    <v-icon v-if="sortBy === 'size'" size="x-small">
                      {{ sortDesc ? 'mdi-menu-down' : 'mdi-menu-up' }}
                    </v-icon>
                  </th>
                  <th class="text-right" @click="setSort('modified')" style="cursor: pointer; width: 180px;">
                    Modified
                    <v-icon v-if="sortBy === 'modified'" size="x-small">
                      {{ sortDesc ? 'mdi-menu-down' : 'mdi-menu-up' }}
                    </v-icon>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="item in sortedItems"
                  :key="item.path"
                  :class="{ 'fm-row-selected': isSelected(item.path) }"
                  tabindex="0"
                  @click="onRowClick($event, item)"
                  @dblclick="onRowDblClick(item)"
                  @keydown.enter="onRowDblClick(item)"
                >
                  <td @click.stop>
                    <v-checkbox-btn
                      :model-value="isSelected(item.path)"
                      density="compact"
                      @update:model-value="toggleSelect(item)"
                    />
                  </td>
                  <td>
                    <v-icon :color="item.is_dir ? 'amber-darken-2' : iconColor(item.extension)">
                      {{ item.is_dir ? 'mdi-folder' : fileIcon(item.extension) }}
                    </v-icon>
                  </td>
                  <td>
                    <template v-if="renamingPath === item.path">
                      <v-text-field
                        v-model="renameValue"
                        density="compact"
                        variant="outlined"
                        hide-details
                        autofocus
                        @keydown.enter="commitRename"
                        @keydown.esc="cancelRename"
                        @click.stop
                        @dblclick.stop
                      >
                        <template #append-inner>
                          <v-btn size="x-small" variant="text" color="success" icon @click.stop="commitRename">
                            <v-icon>mdi-check</v-icon>
                          </v-btn>
                          <v-btn size="x-small" variant="text" icon @click.stop="cancelRename">
                            <v-icon>mdi-close</v-icon>
                          </v-btn>
                        </template>
                      </v-text-field>
                    </template>
                    <span v-else>{{ item.name }}</span>
                  </td>
                  <td class="text-right">{{ item.is_dir ? '—' : formatFileSize(item.size) }}</td>
                  <td class="text-right text-caption">{{ formatModified(item) }}</td>
                </tr>
                <tr v-if="sortedItems.length === 0 && !loadingItems">
                  <td colspan="5" class="text-center py-8 text-medium-emphasis">
                    <v-icon size="48" color="grey">mdi-folder-open-outline</v-icon>
                    <div class="mt-2">
                      This folder is empty. Use <strong>New Folder</strong> or <strong>Upload</strong> to add content.
                    </div>
                  </td>
                </tr>
                <tr v-if="loadingItems">
                  <td colspan="5" class="text-center py-4">
                    <v-progress-circular indeterminate size="24" color="primary" />
                  </td>
                </tr>
              </tbody>
            </v-table>
          </div>
        </div>
      </v-card-text>
    </v-card>

    <!-- New Folder Dialog -->
    <v-dialog v-model="newFolderDialog.open" max-width="420" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="primary">mdi-folder-plus</v-icon>
          New Folder
        </v-card-title>
        <v-card-text>
          <div class="text-caption text-medium-emphasis mb-2">In: {{ currentPath }}</div>
          <v-text-field
            v-model="newFolderDialog.name"
            label="Folder name"
            variant="outlined"
            density="compact"
            hide-details="auto"
            :error-messages="newFolderDialog.error"
            autofocus
            @keydown.enter="submitNewFolder"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="newFolderDialog.open = false">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="newFolderDialog.submitting"
            :disabled="!newFolderDialog.name.trim()"
            @click="submitNewFolder"
          >
            Create
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Move/Copy folder picker -->
    <v-dialog v-model="pickerDialog.open" max-width="560" persistent>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2" color="primary">
            {{ pickerDialog.mode === 'move' ? 'mdi-folder-move' : 'mdi-content-copy' }}
          </v-icon>
          {{ pickerDialog.mode === 'move' ? 'Move to…' : 'Copy to…' }}
        </v-card-title>
        <v-card-text>
          <div class="text-caption text-medium-emphasis mb-2">
            {{ pickerDialog.sources.length }} item(s) selected. Pick a destination folder:
          </div>
          <v-alert
            v-if="pickerDialog.destination"
            type="info"
            variant="tonal"
            density="compact"
            class="mb-2"
          >
            <div class="font-weight-medium">Destination:</div>
            <div class="text-caption">{{ pickerDialog.destination }}</div>
          </v-alert>
          <div class="fm-picker-tree">
            <FileTreeNode
              v-for="root in rootFolders"
              :key="root.path"
              :node="root"
              :selected-path="pickerDialog.destination"
              :folders-only="true"
              @select="(path) => pickerDialog.destination = path"
            />
          </div>
          <v-alert
            v-if="pickerDialog.error"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-2"
          >
            {{ pickerDialog.error }}
          </v-alert>
          <v-progress-linear
            v-if="pickerDialog.submitting && pickerDialog.sources.length > 5"
            indeterminate
            class="mt-2"
            color="primary"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="pickerDialog.open = false">Cancel</v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="pickerDialog.submitting"
            :disabled="!pickerDialog.destination"
            @click="submitPicker"
          >
            {{ pickerDialog.mode === 'move' ? 'Move here' : 'Copy here' }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete confirm -->
    <v-dialog v-model="deleteDialog.open" max-width="480" persistent>
      <v-card>
        <v-card-title class="d-flex align-center text-error">
          <v-icon class="mr-2" color="error">mdi-alert-circle</v-icon>
          Confirm Delete
        </v-card-title>
        <v-card-text>
          <p class="mb-3">
            You are about to delete <strong>{{ deleteDialog.items.length }}</strong> item(s).
            Folders will be deleted recursively. This cannot be undone.
          </p>
          <v-list density="compact" max-height="240" style="overflow-y: auto;">
            <v-list-item
              v-for="item in deleteDialog.items.slice(0, 20)"
              :key="item.path"
            >
              <template #prepend>
                <v-icon size="small" :color="item.is_dir ? 'amber-darken-2' : 'grey'">
                  {{ item.is_dir ? 'mdi-folder' : 'mdi-file-outline' }}
                </v-icon>
              </template>
              <v-list-item-title class="text-body-2">{{ item.name }}</v-list-item-title>
              <v-list-item-subtitle class="text-caption">{{ item.path }}</v-list-item-subtitle>
            </v-list-item>
            <v-list-item v-if="deleteDialog.items.length > 20">
              <v-list-item-title class="text-caption text-medium-emphasis">
                … and {{ deleteDialog.items.length - 20 }} more
              </v-list-item-title>
            </v-list-item>
          </v-list>
          <v-alert
            v-if="deleteDialog.items.length > 10"
            type="warning"
            variant="tonal"
            density="compact"
            class="mt-3"
          >
            You're deleting more than 10 items. Type <strong>DELETE</strong> to confirm.
            <v-text-field
              v-model="deleteDialog.confirmText"
              density="compact"
              variant="outlined"
              hide-details
              class="mt-2"
              placeholder="DELETE"
            />
          </v-alert>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteDialog.open = false" :disabled="deleteDialog.submitting">Cancel</v-btn>
          <v-btn
            color="error"
            variant="flat"
            :loading="deleteDialog.submitting"
            :disabled="deleteDialog.items.length > 10 && deleteDialog.confirmText !== 'DELETE'"
            @click="submitDelete"
          >
            <v-icon start>mdi-delete</v-icon>
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import api from '@/services/api'
import { useNotificationStore } from '@/stores/notification'
import { useAuthStore } from '@/stores/auth'
import FileTreeNode from './FileTreeNode.vue'

interface FileItem {
  name: string
  path: string
  is_dir: boolean
  extension: string | null
  size: number | null
  file_type: string | null
  modified?: number
}

interface RootFolder {
  name: string
  path: string
  type?: string
}

const props = defineProps<{
  modelValue: boolean
  initialPath?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'refreshRequested'): void
}>()

const notify = useNotificationStore()
const authStore = useAuthStore()

// ---- State -------------------------------------------------------------
const rootFolders = ref<RootFolder[]>([])
const loadingRoots = ref(false)

const currentPath = ref<string>('')
const currentItems = ref<FileItem[]>([])
const loadingItems = ref(false)

const selectedPaths = ref<Set<string>>(new Set())
const lastAnchorPath = ref<string | null>(null)

const sortBy = ref<'name' | 'size' | 'modified'>('name')
const sortDesc = ref(false)

const renamingPath = ref<string | null>(null)
const renameValue = ref('')

const uploading = ref(false)
const uploadProgress = ref(0)
const uploadInput = ref<HTMLInputElement | null>(null)

const anyOperationHappened = ref(false)

// New folder sub-dialog
const newFolderDialog = ref({
  open: false,
  name: '',
  error: '',
  submitting: false,
})

// Picker sub-dialog (move / copy)
const pickerDialog = ref({
  open: false,
  mode: 'move' as 'move' | 'copy',
  sources: [] as FileItem[],
  destination: '',
  error: '',
  submitting: false,
})

// Delete confirm
const deleteDialog = ref({
  open: false,
  items: [] as FileItem[],
  confirmText: '',
  submitting: false,
})

// ---- Computed ----------------------------------------------------------
const selectedItems = computed(() =>
  currentItems.value.filter(i => selectedPaths.value.has(i.path))
)

const allSelected = computed(() =>
  currentItems.value.length > 0 && currentItems.value.every(i => selectedPaths.value.has(i.path))
)

const someSelected = computed(() =>
  currentItems.value.some(i => selectedPaths.value.has(i.path))
)

const sortedItems = computed(() => {
  const items = [...currentItems.value]
  items.sort((a, b) => {
    // Folders always first
    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1
    let cmp = 0
    if (sortBy.value === 'name') {
      cmp = a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    } else if (sortBy.value === 'size') {
      cmp = (a.size || 0) - (b.size || 0)
    } else {
      cmp = (a.modified || 0) - (b.modified || 0)
    }
    return sortDesc.value ? -cmp : cmp
  })
  return items
})

const breadcrumbItems = computed(() => {
  if (!currentPath.value) return [{ title: 'Root', path: '', disabled: true }]
  // Find which root this path is under
  const root = rootFolders.value.find(r =>
    currentPath.value === r.path ||
    currentPath.value.startsWith(r.path + '/') ||
    currentPath.value.startsWith(r.path + '\\')
  )
  const crumbs: { title: string; path: string; disabled: boolean }[] = []
  if (root) {
    crumbs.push({ title: root.name, path: root.path, disabled: currentPath.value === root.path })
    const rel = currentPath.value.slice(root.path.length).replace(/^[/\\]+/, '')
    if (rel) {
      const parts = rel.split(/[/\\]+/).filter(Boolean)
      let acc = root.path
      const sep = root.path.includes('\\') ? '\\' : '/'
      for (let i = 0; i < parts.length; i++) {
        acc = acc + sep + parts[i]
        crumbs.push({
          title: parts[i],
          path: acc,
          disabled: i === parts.length - 1,
        })
      }
    }
  } else {
    crumbs.push({ title: currentPath.value, path: currentPath.value, disabled: true })
  }
  return crumbs
})

// ---- Watchers ----------------------------------------------------------
watch(() => props.modelValue, async (open) => {
  if (open) {
    anyOperationHappened.value = false
    await loadRoots()
    if (props.initialPath) {
      currentPath.value = props.initialPath
    } else if (rootFolders.value.length > 0) {
      currentPath.value = rootFolders.value[0].path
    }
    if (currentPath.value) await loadCurrent()
  }
})

// ---- Load --------------------------------------------------------------
async function loadRoots() {
  loadingRoots.value = true
  try {
    const resp = await api.get('/api/data/user-folders')
    rootFolders.value = resp.data.folders || []
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load folders')
  } finally {
    loadingRoots.value = false
  }
}

async function loadCurrent() {
  if (!currentPath.value) return
  loadingItems.value = true
  try {
    const resp = await api.post('/api/data/browse', { path: currentPath.value })
    currentItems.value = resp.data.items || []
    currentPath.value = resp.data.current_path || currentPath.value
    // Clear stale selection (items no longer visible)
    const visiblePaths = new Set(currentItems.value.map((i: FileItem) => i.path))
    selectedPaths.value = new Set(
      [...selectedPaths.value].filter(p => visiblePaths.has(p))
    )
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load folder contents')
  } finally {
    loadingItems.value = false
  }
}

function reloadCurrent() {
  selectedPaths.value = new Set()
  loadCurrent()
}

// ---- Navigation --------------------------------------------------------
function onTreeSelect(path: string) {
  navigateTo(path)
}

function navigateTo(path: string) {
  currentPath.value = path
  selectedPaths.value = new Set()
  lastAnchorPath.value = null
  renamingPath.value = null
  loadCurrent()
}

// ---- Selection ---------------------------------------------------------
function isSelected(path: string) {
  return selectedPaths.value.has(path)
}

function toggleSelect(item: FileItem) {
  const next = new Set(selectedPaths.value)
  if (next.has(item.path)) next.delete(item.path)
  else next.add(item.path)
  selectedPaths.value = next
  lastAnchorPath.value = item.path
}

function toggleSelectAll(v: boolean | null) {
  if (v) {
    selectedPaths.value = new Set(currentItems.value.map(i => i.path))
  } else {
    selectedPaths.value = new Set()
  }
}

function onRowClick(evt: MouseEvent, item: FileItem) {
  if (renamingPath.value) return
  if (evt.shiftKey && lastAnchorPath.value) {
    // Range select
    const visible = sortedItems.value
    const a = visible.findIndex(i => i.path === lastAnchorPath.value)
    const b = visible.findIndex(i => i.path === item.path)
    if (a >= 0 && b >= 0) {
      const [lo, hi] = a < b ? [a, b] : [b, a]
      const next = new Set(selectedPaths.value)
      for (let k = lo; k <= hi; k++) next.add(visible[k].path)
      selectedPaths.value = next
      return
    }
  }
  if (evt.ctrlKey || evt.metaKey) {
    toggleSelect(item)
    return
  }
  // Plain click: single-select
  selectedPaths.value = new Set([item.path])
  lastAnchorPath.value = item.path
}

function onRowDblClick(item: FileItem) {
  if (renamingPath.value) return
  if (item.is_dir) navigateTo(item.path)
}

function setSort(col: 'name' | 'size' | 'modified') {
  if (sortBy.value === col) {
    sortDesc.value = !sortDesc.value
  } else {
    sortBy.value = col
    sortDesc.value = false
  }
}

// ---- New Folder --------------------------------------------------------
function openNewFolderDialog() {
  newFolderDialog.value = { open: true, name: '', error: '', submitting: false }
}

async function submitNewFolder() {
  const name = newFolderDialog.value.name.trim()
  if (!name) return
  newFolderDialog.value.submitting = true
  newFolderDialog.value.error = ''
  try {
    await api.post('/api/data/mkdir', {
      folder: currentPath.value,
      name,
    })
    newFolderDialog.value.open = false
    anyOperationHappened.value = true
    notify.showSuccess(`Created folder: ${name}`)
    await loadCurrent()
  } catch (e: any) {
    newFolderDialog.value.error = e.response?.data?.error || 'Failed to create folder'
  } finally {
    newFolderDialog.value.submitting = false
  }
}

// ---- Upload ------------------------------------------------------------
function triggerUpload() {
  uploadInput.value?.click()
}

async function handleUploadFiles(evt: Event) {
  const target = evt.target as HTMLInputElement
  const files = target.files ? Array.from(target.files) : []
  if (files.length === 0) return
  uploading.value = true
  uploadProgress.value = 0
  let done = 0
  try {
    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('folder', currentPath.value)
      try {
        await api.post('/api/data/upload', fd, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      } catch (e: any) {
        notify.showError(`Upload failed for ${file.name}: ${e.response?.data?.error || e.message}`)
      }
      done++
      uploadProgress.value = Math.round((done / files.length) * 100)
    }
    anyOperationHappened.value = true
    notify.showSuccess(`Uploaded ${done} file(s)`)
    await loadCurrent()
  } finally {
    uploading.value = false
    if (uploadInput.value) uploadInput.value.value = ''
  }
}

// ---- Rename ------------------------------------------------------------
function beginRename() {
  if (selectedItems.value.length !== 1) return
  const item = selectedItems.value[0]
  renamingPath.value = item.path
  renameValue.value = item.name
  nextTick()
}

function cancelRename() {
  renamingPath.value = null
  renameValue.value = ''
}

async function commitRename() {
  const src = renamingPath.value
  if (!src) return
  const newName = renameValue.value.trim()
  const item = currentItems.value.find(i => i.path === src)
  if (!item || !newName || newName === item.name) {
    cancelRename()
    return
  }
  try {
    await api.post('/api/data/rename', {
      path: src,
      new_name: newName,
    })
    anyOperationHappened.value = true
    notify.showSuccess(`Renamed to ${newName}`)
    cancelRename()
    await loadCurrent()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Rename failed')
  }
}

// ---- Move / Copy -------------------------------------------------------
function openMoveDialog() {
  pickerDialog.value = {
    open: true,
    mode: 'move',
    sources: [...selectedItems.value],
    destination: '',
    error: '',
    submitting: false,
  }
}

function openCopyDialog() {
  pickerDialog.value = {
    open: true,
    mode: 'copy',
    sources: [...selectedItems.value],
    destination: '',
    error: '',
    submitting: false,
  }
}

async function submitPicker() {
  const dest = pickerDialog.value.destination
  if (!dest) return
  const sources = pickerDialog.value.sources.map(s => s.path)

  // Refuse if destination equals or is a child of any source (invalid move)
  const invalid = pickerDialog.value.sources.find(s =>
    s.is_dir && (dest === s.path || dest.startsWith(s.path + '/') || dest.startsWith(s.path + '\\'))
  )
  if (invalid) {
    pickerDialog.value.error = `Cannot place ${invalid.name} into itself or a subfolder`
    return
  }

  pickerDialog.value.submitting = true
  pickerDialog.value.error = ''
  try {
    const url = pickerDialog.value.mode === 'move' ? '/api/data/move' : '/api/data/copy'
    const resp = await api.post(url, {
      sources,
      destination: dest,
    })
    const key = pickerDialog.value.mode === 'move' ? 'moved' : 'copied'
    const okCount = (resp.data?.[key] || []).length
    const errs = resp.data?.errors || []
    anyOperationHappened.value = true
    if (errs.length === 0) {
      notify.showSuccess(`${pickerDialog.value.mode === 'move' ? 'Moved' : 'Copied'} ${okCount} item(s)`)
    } else {
      notify.showWarning(`${okCount} succeeded, ${errs.length} failed. First error: ${errs[0]?.reason || 'unknown'}`)
    }
    pickerDialog.value.open = false
    await loadCurrent()
  } catch (e: any) {
    pickerDialog.value.error = e.response?.data?.error || 'Operation failed'
  } finally {
    pickerDialog.value.submitting = false
  }
}

// ---- Delete ------------------------------------------------------------
function openDeleteConfirm() {
  deleteDialog.value = {
    open: true,
    items: [...selectedItems.value],
    confirmText: '',
    submitting: false,
  }
}

async function submitDelete() {
  if (deleteDialog.value.items.length === 0) return
  if (deleteDialog.value.items.length > 10 && deleteDialog.value.confirmText !== 'DELETE') return

  deleteDialog.value.submitting = true
  const endpoint = authStore.isAdmin
    ? '/api/data/admin/delete'
    : '/api/data/delete-upload'
  let ok = 0
  let fail = 0
  try {
    for (const item of deleteDialog.value.items) {
      try {
        await api.post(endpoint, { file_path: item.path })
        ok++
      } catch (e: any) {
        fail++
        // Continue on individual errors
        // eslint-disable-next-line no-console
        console.warn(`Delete failed for ${item.path}:`, e.response?.data?.error || e.message)
      }
    }
    anyOperationHappened.value = true
    if (fail === 0) {
      notify.showSuccess(`Deleted ${ok} item(s)`)
    } else {
      notify.showWarning(`Deleted ${ok}, ${fail} failed`)
    }
    deleteDialog.value.open = false
    await loadCurrent()
  } finally {
    deleteDialog.value.submitting = false
  }
}

// ---- Download ----------------------------------------------------------
async function downloadSelected() {
  if (selectedItems.value.length !== 1) return
  const item = selectedItems.value[0]
  if (item.is_dir) return
  try {
    const resp = await api.get('/api/data/download', {
      params: { path: item.path },
      responseType: 'blob',
    })
    const blobUrl = window.URL.createObjectURL(new Blob([resp.data]))
    const a = document.createElement('a')
    a.href = blobUrl
    a.download = item.name
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(blobUrl)
    document.body.removeChild(a)
  } catch {
    notify.showError('Download failed')
  }
}

// ---- Close -------------------------------------------------------------
function onDialogUpdate(v: boolean) {
  emit('update:modelValue', v)
  if (!v && anyOperationHappened.value) {
    emit('refreshRequested')
  }
}

function close() {
  onDialogUpdate(false)
}

// ---- Formatting --------------------------------------------------------
function formatFileSize(bytes: number | null | undefined) {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatModified(item: FileItem) {
  if (!item.modified) return '—'
  try {
    return new Date(item.modified * 1000).toLocaleString()
  } catch {
    return '—'
  }
}

function fileIcon(ext: string | null | undefined) {
  switch ((ext || '').toLowerCase()) {
    case '.csv': return 'mdi-file-delimited-outline'
    case '.json': return 'mdi-code-json'
    case '.cbor': return 'mdi-file-code-outline'
    case '.txt':
    case '.tsv':
    case '.dat':
    case '.log': return 'mdi-file-document-outline'
    default: return 'mdi-file-outline'
  }
}

function iconColor(ext: string | null | undefined) {
  switch ((ext || '').toLowerCase()) {
    case '.csv': return 'green'
    case '.json': return 'blue'
    case '.cbor': return 'purple'
    default: return 'grey'
  }
}
</script>

<style scoped lang="scss">
.fm-left-pane {
  width: 280px;
  min-width: 280px;
  display: flex;
  flex-direction: column;
  background: rgba(0, 0, 0, 0.02);
  overflow: hidden;
}

.fm-tree-scroll {
  flex-grow: 1;
  overflow-y: auto;
  padding: 4px 0;
}

.fm-right-pane {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.fm-breadcrumbs {
  padding: 8px 12px;
  border-bottom: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}

.fm-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  flex-wrap: wrap;
}

.fm-table-scroll {
  flex-grow: 1;
  overflow-y: auto;
}

.fm-table {
  :deep(th) {
    font-weight: 600;
    user-select: none;
  }
  :deep(tbody tr) {
    cursor: pointer;
  }
  :deep(tbody tr:hover) {
    background: rgba(99, 102, 241, 0.05);
  }
  :deep(tbody tr.fm-row-selected) {
    background: rgba(99, 102, 241, 0.15);
  }
  :deep(tbody tr:focus) {
    outline: 2px solid rgba(99, 102, 241, 0.6);
    outline-offset: -2px;
  }
}

.fm-picker-tree {
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 4px;
  padding: 4px 0;
}
</style>
