<template>
  <v-container fluid class="pa-6">
    <div class="d-flex align-center flex-wrap mb-4 ga-2">
      <div>
        <h1 class="text-h4 font-weight-bold">Machine Groups</h1>
        <p class="text-body-2 text-medium-emphasis mb-0">
          Ad-hoc collections of machines used for cross-machine training and
          fleet-wide model deployment. Groups are logical only — they don't
          appear in the asset tree.
        </p>
      </div>
      <v-spacer />
      <v-chip
        v-if="!isAdmin"
        color="warning"
        size="small"
        variant="tonal"
        prepend-icon="mdi-eye"
      >
        Read-only — admins can edit
      </v-chip>
      <v-btn
        v-if="isAdmin"
        color="primary"
        prepend-icon="mdi-plus"
        @click="onCreate"
      >
        Create group
      </v-btn>
    </div>

    <v-card>
      <div v-if="loading" class="pa-6 text-center text-caption">
        <v-progress-circular indeterminate size="20" width="2" class="mr-2" />
        Loading groups…
      </div>
      <div
        v-else-if="groups.length === 0"
        class="pa-8 text-center"
      >
        <v-icon size="48" color="grey">mdi-account-group-outline</v-icon>
        <p class="text-body-1 mt-3 mb-1">No groups yet.</p>
        <p class="text-body-2 text-medium-emphasis mb-4">
          Groups are ad-hoc collections of machines for cross-machine training.
        </p>
        <v-btn
          v-if="isAdmin"
          color="primary"
          prepend-icon="mdi-plus"
          @click="onCreate"
        >
          Create your first group
        </v-btn>
      </div>
      <v-table v-else density="comfortable">
        <thead>
          <tr>
            <th style="width: 20%">Name</th>
            <th>Description</th>
            <th class="text-right" style="width: 90px">Members</th>
            <th style="width: 160px">Created</th>
            <th class="text-right" style="width: 180px">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="g in groups" :key="g.id" class="group-row" @click="onView(g)">
            <td class="font-weight-medium">
              <v-icon size="16" color="secondary" class="mr-1">mdi-account-group</v-icon>
              {{ g.name }}
            </td>
            <td class="text-caption text-medium-emphasis">
              {{ g.description || '—' }}
            </td>
            <td class="text-right">
              <v-chip size="x-small" variant="tonal" color="primary">
                {{ g.member_count || 0 }}
              </v-chip>
            </td>
            <td class="text-caption">
              {{ formatTime(g.created_at) }}
            </td>
            <td class="text-right">
              <v-btn
                v-if="isAdmin"
                size="x-small"
                variant="text"
                prepend-icon="mdi-pencil"
                @click.stop="onEdit(g)"
              >
                Edit
              </v-btn>
              <v-btn
                v-if="isAdmin"
                size="x-small"
                variant="text"
                color="error"
                prepend-icon="mdi-delete-outline"
                @click.stop="onDeletePrompt(g)"
              >
                Delete
              </v-btn>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>

    <!-- Create / edit dialog -->
    <MachineGroupEditDialog
      v-model="dialogOpen"
      :group="editingGroup"
      @saved="onSaved"
    />

    <!-- View / expand a single group -->
    <v-dialog v-model="viewDialogOpen" max-width="640" scrollable>
      <v-card v-if="viewingGroup">
        <v-card-title class="d-flex align-center">
          <v-icon start color="secondary">mdi-account-group</v-icon>
          {{ viewingGroup.name }}
          <v-spacer />
          <v-btn icon size="small" variant="text" @click="viewDialogOpen = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>
        <v-card-text>
          <p v-if="viewingGroup.description" class="mb-3">
            {{ viewingGroup.description }}
          </p>
          <p v-else class="text-caption text-medium-emphasis mb-3">
            No description.
          </p>
          <div class="text-subtitle-2 mb-2">
            Members ({{ viewingGroup.members?.length || 0 }})
          </div>
          <div v-if="!viewingGroup.members || viewingGroup.members.length === 0" class="text-caption text-medium-emphasis">
            No machines in this group.
          </div>
          <v-table v-else density="compact">
            <thead>
              <tr>
                <th>Machine</th>
                <th>Topic path</th>
                <th class="text-right">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="m in viewingGroup.members" :key="m.id">
                <td>{{ m.display_name || m.name }}</td>
                <td class="text-caption"><code>{{ m.topic_path }}</code></td>
                <td class="text-right">
                  <v-chip
                    size="x-small"
                    variant="tonal"
                    :color="m.status === 'retired' ? 'grey' : 'success'"
                  >
                    {{ m.status || 'active' }}
                  </v-chip>
                </td>
              </tr>
            </tbody>
          </v-table>

          <!-- Compatibility summary for the group -->
          <div v-if="viewMemberIds.length >= 2" class="mt-3">
            <CompatibilityBadge :machine-ids="viewMemberIds" />
          </div>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="viewDialogOpen = false">Close</v-btn>
          <v-btn
            v-if="isAdmin"
            color="primary"
            variant="tonal"
            prepend-icon="mdi-pencil"
            @click="onEditFromView"
          >
            Edit
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Delete confirmation -->
    <v-dialog v-model="deleteDialogOpen" max-width="480">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="error" class="mr-2">mdi-delete-outline</v-icon>
          Delete group?
        </v-card-title>
        <v-card-text>
          <p class="mb-2">
            This permanently removes the group
            <code>{{ deletingGroup?.name }}</code>. Machines in the group
            are untouched — only the group definition is deleted.
          </p>
          <p class="text-body-2 text-medium-emphasis">
            Any past training runs that referenced this group by name in
            their metadata keep that reference — the audit trail is preserved.
          </p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteDialogOpen = false">Cancel</v-btn>
          <v-btn
            color="error"
            :loading="deleting"
            @click="confirmDelete"
          >
            Delete group
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
/**
 * Phase C.1 — Machine Groups management view.
 * Replaces the "Phase C stub" placeholder that was previously served by the
 * AssetTreeAdminView "groups" tab. Read-open to any authenticated user;
 * write ops (create / edit / delete) admin-only.
 */
import { ref, computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useNotificationStore } from '@/stores/notification'
import api from '@/services/api'
import MachineGroupEditDialog from '@/components/MachineGroupEditDialog.vue'
import CompatibilityBadge from '@/components/CompatibilityBadge.vue'

interface GroupRow {
  id: number
  name: string
  description?: string | null
  member_count?: number
  created_at?: string
  members?: any[]
}

const authStore = useAuthStore()
const notify = useNotificationStore()

const isAdmin = computed(() => authStore.user?.role === 'admin')

const groups = ref<GroupRow[]>([])
const loading = ref(false)

const dialogOpen = ref(false)
const editingGroup = ref<GroupRow | null>(null)

const viewDialogOpen = ref(false)
const viewingGroup = ref<GroupRow | null>(null)
// Filter retired machines out of the compat validator's input — the
// validator treats retired sensor children as "missing", producing a
// misleading red "sensor mismatch" badge on view. Since Phase C QA #1's
// backend fix now blocks new retired members from being added, this
// filter is defense-in-depth for groups created before that fix.
const viewMemberIds = computed(() =>
  (viewingGroup.value?.members || [])
    .filter((m: any) => m && m.status !== 'retired')
    .map((m: any) => m.id)
    .filter((n: any) => Number.isFinite(n)),
)

const deleteDialogOpen = ref(false)
const deletingGroup = ref<GroupRow | null>(null)
const deleting = ref(false)

async function fetchGroups() {
  loading.value = true
  try {
    const r = await api.get('/api/asset-tree/groups')
    groups.value = r.data?.groups || []
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load groups')
  } finally {
    loading.value = false
  }
}

async function fetchGroupDetail(id: number): Promise<GroupRow | null> {
  try {
    const r = await api.get(`/api/asset-tree/groups/${id}`)
    return r.data
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Failed to load group')
    return null
  }
}

function onCreate() {
  editingGroup.value = null
  dialogOpen.value = true
}
async function onEdit(g: GroupRow) {
  const full = await fetchGroupDetail(g.id)
  if (!full) return
  editingGroup.value = full
  dialogOpen.value = true
}
async function onView(g: GroupRow) {
  const full = await fetchGroupDetail(g.id)
  if (!full) return
  viewingGroup.value = full
  viewDialogOpen.value = true
}
async function onEditFromView() {
  if (!viewingGroup.value) return
  editingGroup.value = viewingGroup.value
  viewDialogOpen.value = false
  dialogOpen.value = true
}

async function onSaved() {
  await fetchGroups()
}

function onDeletePrompt(g: GroupRow) {
  deletingGroup.value = g
  deleteDialogOpen.value = true
}
async function confirmDelete() {
  if (!deletingGroup.value) return
  deleting.value = true
  try {
    await api.delete(`/api/asset-tree/groups/${deletingGroup.value.id}`)
    notify.showSuccess('Group deleted')
    deleteDialogOpen.value = false
    deletingGroup.value = null
    await fetchGroups()
  } catch (e: any) {
    notify.showError(e.response?.data?.error || 'Delete failed')
  } finally {
    deleting.value = false
  }
}

function formatTime(s: string | undefined | null): string {
  if (!s) return '—'
  try {
    return new Date(s + (s.endsWith('Z') ? '' : 'Z')).toLocaleString()
  } catch {
    return s
  }
}

onMounted(fetchGroups)
</script>

<style scoped>
.group-row {
  cursor: pointer;
}
.group-row:hover td {
  background: rgba(var(--v-theme-on-surface), 0.04);
}
</style>
