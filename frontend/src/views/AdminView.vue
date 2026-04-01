<template>
  <v-container fluid class="pa-6">
    <h1 class="text-h4 font-weight-bold mb-2">Admin Panel</h1>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Manage users and system settings
    </p>

    <v-row>
      <!-- User Management -->
      <v-col cols="12" lg="8">
        <v-card>
          <v-card-title class="d-flex align-center">
            <span>User Management</span>
            <v-spacer />
            <v-btn color="primary" prepend-icon="mdi-plus" @click="openCreateDialog">
              Add User
            </v-btn>
          </v-card-title>

          <v-data-table
            :headers="userHeaders"
            :items="users"
            :loading="loadingUsers"
            density="comfortable"
          >
            <template #item.role="{ item }">
              <v-chip
                :color="item.role === 'admin' ? 'primary' : 'secondary'"
                size="small"
                variant="flat"
              >
                {{ item.role }}
              </v-chip>
            </template>

            <template #item.is_active="{ item }">
              <v-icon :color="item.is_active ? 'success' : 'error'">
                {{ item.is_active ? 'mdi-check-circle' : 'mdi-close-circle' }}
              </v-icon>
            </template>

            <template #item.actions="{ item }">
              <v-btn
                icon="mdi-pencil"
                variant="text"
                size="small"
                @click="openEditDialog(item)"
              />
              <v-btn
                icon="mdi-key"
                variant="text"
                size="small"
                @click="openPasswordDialog(item)"
              />
              <v-btn
                v-if="item.username !== 'admin'"
                icon="mdi-delete"
                variant="text"
                size="small"
                color="error"
                @click="confirmDelete(item)"
              />
            </template>
          </v-data-table>
        </v-card>
      </v-col>

      <!-- Folder Management -->
      <v-col cols="12" lg="4">
        <v-card class="pa-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">Folder Management</h3>

          <v-list density="compact">
            <v-list-item
              v-for="folder in folders"
              :key="folder.name"
            >
              <template #prepend>
                <v-icon :color="folder.type === 'shared' ? 'info' : 'warning'">
                  {{ folder.type === 'shared' ? 'mdi-folder-account' : 'mdi-folder' }}
                </v-icon>
              </template>
              <v-list-item-title>{{ folder.name }}</v-list-item-title>
              <template #append>
                <v-chip size="x-small" :color="folder.type === 'shared' ? 'info' : 'default'" class="mr-2">
                  {{ folder.type }}
                </v-chip>
                <v-btn
                  v-if="folder.type !== 'shared'"
                  icon="mdi-delete"
                  variant="text"
                  size="x-small"
                  color="error"
                  @click="confirmDeleteFolder(folder)"
                />
              </template>
            </v-list-item>
          </v-list>

          <v-alert v-if="folders.length === 0" type="info" variant="tonal" density="compact" class="my-2">
            No folders found. Create one to get started.
          </v-alert>

          <v-divider class="my-4" />

          <v-text-field
            v-model="newFolderName"
            label="New Folder Name"
            density="compact"
            hide-details
            hint="Only letters, numbers, underscore, hyphen allowed"
            class="mb-2"
          />
          <v-btn
            color="primary"
            block
            :loading="creatingFolder"
            :disabled="!newFolderName"
            @click="createFolder"
          >
            <v-icon start>mdi-folder-plus</v-icon>
            Create Folder
          </v-btn>
        </v-card>

        <!-- System Info -->
        <v-card class="pa-4 mt-4">
          <h3 class="text-subtitle-1 font-weight-bold mb-4">System Settings</h3>

          <v-list-item>
            <v-list-item-title class="text-caption">Datasets Root</v-list-item-title>
            <v-list-item-subtitle class="font-weight-medium">
              {{ datasetsRoot || 'Loading...' }}
            </v-list-item-subtitle>
          </v-list-item>

          <v-list-item>
            <v-list-item-title class="text-caption">Session Timeout</v-list-item-title>
            <v-list-item-subtitle class="font-weight-medium">
              8 hours
            </v-list-item-subtitle>
          </v-list-item>

          <v-list-item>
            <v-list-item-title class="text-caption">Version</v-list-item-title>
            <v-list-item-subtitle class="font-weight-medium">
              CiRA ME v1.0.0
            </v-list-item-subtitle>
          </v-list-item>
        </v-card>

        <!-- Storage Volumes -->
        <v-card class="pa-4 mt-4">
          <div class="d-flex align-center mb-4">
            <h3 class="text-subtitle-1 font-weight-bold">Storage Volumes</h3>
            <v-spacer />
            <v-btn size="x-small" variant="text" icon="mdi-refresh" :loading="loadingVolumes" @click="loadStorageVolumes" />
          </div>

          <v-alert v-if="!storageVolumes.length && !loadingVolumes" type="info" variant="tonal" density="compact" class="mb-3">
            Loading volume information...
          </v-alert>

          <div v-for="vol in storageVolumes" :key="vol.container_path" class="mb-4">
            <div class="d-flex align-center mb-1">
              <v-icon size="18" color="primary" class="mr-2">
                {{ vol.name.includes('Model') ? 'mdi-cube-outline' : vol.name.includes('Dataset') ? 'mdi-folder-open' : 'mdi-database' }}
              </v-icon>
              <span class="font-weight-bold text-body-2">{{ vol.name }}</span>
              <v-chip size="x-small" color="info" variant="tonal" class="ml-2">{{ vol.size_mb }} MB</v-chip>
            </div>

            <div class="text-caption text-medium-emphasis ml-7 mb-1">
              <div>Container: <code>{{ vol.container_path }}</code></div>
              <div>Host: <code>{{ vol.host_hint }}</code></div>
              <div>Disk: {{ vol.disk_free_gb }} GB free / {{ vol.disk_total_gb }} GB total</div>
            </div>

            <!-- File tree -->
            <div v-if="vol.contents && vol.contents.length > 0" class="ml-7 mt-1">
              <div
                class="volume-toggle text-caption"
                @click="toggleVolume(vol.container_path)"
                style="cursor: pointer; user-select: none;"
              >
                <v-icon size="12">{{ expandedVolumes[vol.container_path] ? 'mdi-chevron-down' : 'mdi-chevron-right' }}</v-icon>
                <span class="text-medium-emphasis">{{ vol.contents.length }} items</span>
              </div>
              <div v-if="expandedVolumes[vol.container_path]" class="file-tree mt-1">
                <template v-for="item in vol.contents" :key="item.name">
                  <div class="file-tree-item">
                    <v-icon size="14" :color="item.is_dir ? 'amber' : 'grey'" class="mr-1">
                      {{ item.is_dir ? 'mdi-folder' : 'mdi-file-outline' }}
                    </v-icon>
                    <span class="text-caption">{{ item.name }}</span>
                    <span v-if="item.size_mb != null" class="text-caption text-medium-emphasis ml-1">
                      ({{ item.size_mb < 0.01 ? '<0.01' : item.size_mb }} MB)
                    </span>
                  </div>
                  <!-- Children (depth 1) -->
                  <template v-if="item.children && item.children.length > 0">
                    <div v-for="child in item.children" :key="child.name" class="file-tree-item" style="padding-left: 20px;">
                      <v-icon size="12" :color="child.is_dir ? 'amber' : 'grey'" class="mr-1">
                        {{ child.is_dir ? 'mdi-folder' : 'mdi-file-outline' }}
                      </v-icon>
                      <span class="text-caption">{{ child.name }}</span>
                      <span v-if="child.size_mb != null" class="text-caption text-medium-emphasis ml-1">
                        ({{ child.size_mb < 0.01 ? '<0.01' : child.size_mb }} MB)
                      </span>
                    </div>
                  </template>
                </template>
              </div>
            </div>

            <v-divider v-if="storageVolumes.indexOf(vol) < storageVolumes.length - 1" class="mt-3" />
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Create/Edit User Dialog -->
    <v-dialog v-model="userDialog" max-width="500">
      <v-card>
        <v-card-title>
          {{ editingUser ? 'Edit User' : 'Create User' }}
        </v-card-title>
        <v-card-text>
          <v-text-field
            v-model="userForm.username"
            label="Username"
            :disabled="!!editingUser"
          />
          <v-text-field
            v-if="!editingUser"
            v-model="userForm.password"
            label="Password"
            type="password"
          />
          <v-text-field
            v-model="userForm.display_name"
            label="Display Name"
          />
          <v-select
            v-model="userForm.role"
            label="Role"
            :items="['admin', 'annotator']"
          />
          <v-autocomplete
            v-if="userForm.role !== 'admin'"
            v-model="userForm.private_folder"
            label="Private Folder"
            :items="privateFolderOptions"
            hint="Annotators can only access their private folder + shared folder"
            clearable
          >
            <template #no-data>
              <v-list-item>
                <v-list-item-title class="text-caption text-medium-emphasis">
                  No folders available. Create one first.
                </v-list-item-title>
              </v-list-item>
            </template>
          </v-autocomplete>
          <v-alert
            v-else
            type="info"
            variant="tonal"
            density="compact"
            class="mb-4"
          >
            Admin users have full access to all folders.
          </v-alert>
          <v-switch
            v-if="editingUser"
            v-model="userForm.is_active"
            label="Active"
            color="success"
          />

          <!-- User Quotas -->
          <v-divider class="my-3" />
          <div class="text-subtitle-2 font-weight-bold mb-2">Quotas</div>
          <v-row dense>
            <v-col cols="4">
              <v-text-field
                v-model.number="userForm.max_folder_mb"
                label="Folder Size (MB)"
                type="number"
                :min="50"
                :max="10000"
                density="compact"
                variant="outlined"
                hint="Max private folder size"
                persistent-hint
              />
            </v-col>
            <v-col cols="4">
              <v-text-field
                v-model.number="userForm.max_endpoints"
                label="Max Endpoints"
                type="number"
                :min="1"
                :max="100"
                density="compact"
                variant="outlined"
                hint="ME-LAB endpoints"
                persistent-hint
              />
            </v-col>
            <v-col cols="4">
              <v-text-field
                v-model.number="userForm.max_apps"
                label="Max Apps"
                type="number"
                :min="1"
                :max="100"
                density="compact"
                variant="outlined"
                hint="Published web apps"
                persistent-hint
              />
            </v-col>
          </v-row>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="userDialog = false">Cancel</v-btn>
          <v-btn color="primary" @click="saveUser" :loading="savingUser">
            Save
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Password Reset Dialog -->
    <v-dialog v-model="passwordDialog" max-width="400">
      <v-card>
        <v-card-title>Reset Password</v-card-title>
        <v-card-text>
          <p class="mb-4">Reset password for: <strong>{{ selectedUser?.username }}</strong></p>
          <v-text-field
            v-model="newPassword"
            label="New Password"
            type="password"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="passwordDialog = false">Cancel</v-btn>
          <v-btn color="primary" @click="resetPassword" :loading="resettingPassword">
            Reset
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete User Confirmation -->
    <v-dialog v-model="deleteDialog" max-width="400">
      <v-card>
        <v-card-title>Confirm Delete</v-card-title>
        <v-card-text>
          Are you sure you want to delete user <strong>{{ selectedUser?.username }}</strong>?
          This action cannot be undone.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteDialog = false">Cancel</v-btn>
          <v-btn color="error" @click="deleteUser" :loading="deletingUser">
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete Folder Confirmation -->
    <v-dialog v-model="deleteFolderDialog" max-width="450">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="error" class="mr-2">mdi-folder-remove</v-icon>
          Confirm Delete Folder
        </v-card-title>
        <v-card-text>
          <v-alert type="warning" variant="tonal" density="compact" class="mb-4">
            This will permanently delete the folder and all its contents!
          </v-alert>
          <p>
            Are you sure you want to delete folder <strong>{{ selectedFolder?.name }}</strong>?
          </p>
          <p class="text-caption text-medium-emphasis mt-2">
            Path: {{ selectedFolder?.path }}
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteFolderDialog = false">Cancel</v-btn>
          <v-btn color="error" @click="deleteFolder" :loading="deletingFolder">
            Delete Folder
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'

interface User {
  id: number
  username: string
  display_name: string
  role: string
  is_active: boolean
  private_folder?: string
}

const notificationStore = useNotificationStore()

const users = ref<User[]>([])
const folders = ref<{ name: string; path: string; type: string }[]>([])
const datasetsRoot = ref('')
const loadingUsers = ref(false)
const storageVolumes = ref<any[]>([])
const loadingVolumes = ref(false)
const expandedVolumes = reactive<Record<string, boolean>>({})
const newFolderName = ref('')
const creatingFolder = ref(false)

// Computed: Private folder options (exclude 'shared' folder)
const privateFolderOptions = computed(() => {
  return folders.value
    .filter(f => f.type !== 'shared')
    .map(f => f.name)
})

interface Folder {
  name: string
  path: string
  type: string
}

// Dialogs
const userDialog = ref(false)
const passwordDialog = ref(false)
const deleteDialog = ref(false)
const deleteFolderDialog = ref(false)
const editingUser = ref<User | null>(null)
const selectedUser = ref<User | null>(null)
const selectedFolder = ref<Folder | null>(null)
const savingUser = ref(false)
const resettingPassword = ref(false)
const deletingUser = ref(false)
const deletingFolder = ref(false)
const newPassword = ref('')

const userForm = reactive({
  username: '',
  password: '',
  display_name: '',
  role: 'annotator',
  private_folder: '',
  is_active: true,
  max_folder_mb: 500,
  max_endpoints: 5,
  max_apps: 10,
})

const userHeaders = [
  { title: 'Username', key: 'username' },
  { title: 'Display Name', key: 'display_name' },
  { title: 'Role', key: 'role' },
  { title: 'Private Folder', key: 'private_folder' },
  { title: 'Folder MB', key: 'max_folder_mb' },
  { title: 'Endpoints', key: 'max_endpoints' },
  { title: 'Apps', key: 'max_apps' },
  { title: 'Active', key: 'is_active' },
  { title: 'Actions', key: 'actions', sortable: false }
]

async function loadUsers() {
  try {
    loadingUsers.value = true
    const response = await api.get('/api/admin/users')
    users.value = response.data.users
  } catch (e) {
    notificationStore.showError('Failed to load users')
  } finally {
    loadingUsers.value = false
  }
}

async function loadFolders() {
  try {
    const response = await api.get('/api/data/user-folders')
    folders.value = response.data.folders
  } catch (e) {
    console.error('Failed to load folders')
  }
}

async function loadDatasetsRoot() {
  try {
    const response = await api.get('/api/data/datasets-root')
    datasetsRoot.value = response.data.path
  } catch (e) {
    console.error('Failed to load datasets root')
  }
}

async function loadStorageVolumes() {
  loadingVolumes.value = true
  try {
    const response = await api.get('/api/admin/storage-volumes')
    storageVolumes.value = response.data.volumes || []
  } catch (e) {
    console.error('Failed to load storage volumes')
  } finally {
    loadingVolumes.value = false
  }
}

function toggleVolume(path: string) {
  expandedVolumes[path] = !expandedVolumes[path]
}

function openCreateDialog() {
  editingUser.value = null
  Object.assign(userForm, {
    username: '',
    password: '',
    display_name: '',
    role: 'annotator',
    private_folder: '',
    is_active: true
  })
  userDialog.value = true
}

function openEditDialog(user: User) {
  editingUser.value = user
  Object.assign(userForm, {
    username: user.username,
    display_name: user.display_name,
    role: user.role,
    private_folder: user.private_folder || '',
    is_active: user.is_active,
    max_folder_mb: user.max_folder_mb || 500,
    max_endpoints: user.max_endpoints || 5,
    max_apps: user.max_apps || 10,
  })
  userDialog.value = true
}

function openPasswordDialog(user: User) {
  selectedUser.value = user
  newPassword.value = ''
  passwordDialog.value = true
}

function confirmDelete(user: User) {
  selectedUser.value = user
  deleteDialog.value = true
}

async function saveUser() {
  try {
    savingUser.value = true

    if (editingUser.value) {
      await api.put(`/api/admin/users/${editingUser.value.id}`, userForm)
      notificationStore.showSuccess('User updated')
    } else {
      await api.post('/api/admin/users', userForm)
      notificationStore.showSuccess('User created')
    }

    userDialog.value = false
    await loadUsers()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to save user')
  } finally {
    savingUser.value = false
  }
}

async function resetPassword() {
  if (!selectedUser.value || !newPassword.value) return

  try {
    resettingPassword.value = true

    await api.put(`/api/admin/users/${selectedUser.value.id}/password`, {
      new_password: newPassword.value
    })

    notificationStore.showSuccess('Password reset')
    passwordDialog.value = false
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to reset password')
  } finally {
    resettingPassword.value = false
  }
}

async function deleteUser() {
  if (!selectedUser.value) return

  try {
    deletingUser.value = true

    await api.delete(`/api/admin/users/${selectedUser.value.id}`)

    notificationStore.showSuccess('User deleted')
    deleteDialog.value = false
    await loadUsers()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to delete user')
  } finally {
    deletingUser.value = false
  }
}

async function createFolder() {
  if (!newFolderName.value) return

  try {
    creatingFolder.value = true

    await api.post('/api/admin/create-folder', {
      folder_name: newFolderName.value
    })

    notificationStore.showSuccess('Folder created')
    newFolderName.value = ''
    await loadFolders()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to create folder')
  } finally {
    creatingFolder.value = false
  }
}

function confirmDeleteFolder(folder: Folder) {
  selectedFolder.value = folder
  deleteFolderDialog.value = true
}

async function deleteFolder() {
  if (!selectedFolder.value) return

  try {
    deletingFolder.value = true

    await api.post('/api/admin/delete-folder', {
      folder_name: selectedFolder.value.name
    })

    notificationStore.showSuccess('Folder deleted')
    deleteFolderDialog.value = false
    await loadFolders()
  } catch (e: any) {
    notificationStore.showError(e.response?.data?.error || 'Failed to delete folder')
  } finally {
    deletingFolder.value = false
  }
}

onMounted(() => {
  loadUsers()
  loadFolders()
  loadDatasetsRoot()
  loadStorageVolumes()
})
</script>

<style scoped>
.file-tree-item {
  display: flex;
  align-items: center;
  padding: 1px 0;
  font-family: monospace;
}
.file-tree-item code {
  font-size: 11px;
}
</style>
