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
  is_active: true
})

const userHeaders = [
  { title: 'Username', key: 'username' },
  { title: 'Display Name', key: 'display_name' },
  { title: 'Role', key: 'role' },
  { title: 'Private Folder', key: 'private_folder' },
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
    is_active: user.is_active
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
})
</script>
