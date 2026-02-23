# CiRA ME - UI Theme & User Auth/Management System

Updated documentation for the UI color scheme, theming system, and user authentication/management.

---

## Table of Contents

1. [UI Theme & Color Scheme](#ui-theme--color-scheme)
2. [Typography & Fonts](#typography--fonts)
3. [Global Styles (SCSS)](#global-styles-scss)
4. [App Layout & Navigation](#app-layout--navigation)
5. [Authentication System](#authentication-system)
6. [User Management (Admin)](#user-management-admin)
7. [Folder Access Control](#folder-access-control)
8. [File Delete Permissions](#file-delete-permissions)
9. [API Endpoints Reference](#api-endpoints-reference)

---

## UI Theme & Color Scheme

### Vuetify Theme Configuration (`main.ts`)

CiRA ME uses a dual-theme system (dark/light) with Indigo + Cyan brand identity.

```typescript
const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        dark: true,
        colors: {
          background: '#0F172A',        // Slate 900 - deep navy
          surface: '#1E293B',           // Slate 800 - card backgrounds
          'surface-bright': '#334155',  // Slate 700
          'surface-light': '#475569',   // Slate 600
          'surface-variant': '#64748B', // Slate 500
          'on-surface-variant': '#E2E8F0', // Slate 200
          primary: '#6366F1',           // Indigo 500 - brand primary
          'primary-darken-1': '#4F46E5', // Indigo 600
          secondary: '#22D3EE',         // Cyan 400 - brand accent
          'secondary-darken-1': '#06B6D4', // Cyan 500
          error: '#EF4444',             // Red 500
          info: '#3B82F6',              // Blue 500
          success: '#10B981',           // Emerald 500
          warning: '#F59E0B',           // Amber 500
        },
      },
      light: {
        dark: false,
        colors: {
          background: '#F8FAFC',        // Slate 50
          surface: '#FFFFFF',           // White
          'surface-bright': '#F1F5F9',  // Slate 100
          'surface-light': '#E2E8F0',   // Slate 200
          'surface-variant': '#CBD5E1',  // Slate 300
          'on-surface-variant': '#334155', // Slate 700
          primary: '#6366F1',           // Indigo 500
          'primary-darken-1': '#4F46E5', // Indigo 600
          secondary: '#06B6D4',         // Cyan 500
          'secondary-darken-1': '#0891B2', // Cyan 600
          error: '#DC2626',             // Red 600
          info: '#2563EB',              // Blue 600
          success: '#059669',           // Emerald 600
          warning: '#D97706',           // Amber 600
        },
      },
    },
  },
  defaults: {
    VBtn: {
      variant: 'flat',
      rounded: 'lg',
    },
    VCard: {
      rounded: 'lg',
      elevation: 0,
    },
    VTextField: {
      variant: 'outlined',
      density: 'comfortable',
    },
    VSelect: {
      variant: 'outlined',
      density: 'comfortable',
    },
  },
})
```

### Color Palette Summary

| Token | Dark Value | Light Value | Usage |
|-------|-----------|-------------|-------|
| `background` | `#0F172A` | `#F8FAFC` | Page background |
| `surface` | `#1E293B` | `#FFFFFF` | Cards, dialogs, app bar |
| `primary` | `#6366F1` | `#6366F1` | Buttons, links, active states |
| `secondary` | `#22D3EE` | `#06B6D4` | Accents, badges, highlights |
| `error` | `#EF4444` | `#DC2626` | Error states, delete actions |
| `success` | `#10B981` | `#059669` | Success feedback, active status |
| `warning` | `#F59E0B` | `#D97706` | Warning alerts |
| `info` | `#3B82F6` | `#2563EB` | Info messages |

### Component Defaults

| Component | Default Setting |
|-----------|----------------|
| `VBtn` | `variant: 'flat'`, `rounded: 'lg'` |
| `VCard` | `rounded: 'lg'`, `elevation: 0` |
| `VTextField` | `variant: 'outlined'`, `density: 'comfortable'` |
| `VSelect` | `variant: 'outlined'`, `density: 'comfortable'` |

### Theme Toggle

Users can toggle between dark and light themes from the user menu:

```typescript
const toggleTheme = () => {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}
```

---

## Typography & Fonts

### Font Stack (`main.scss`)

```scss
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  // Brand colors (CSS variables)
  --cira-primary: #6366F1;
  --cira-secondary: #22D3EE;
  --cira-accent: #A855F7;
  --cira-success: #10B981;
  --cira-warning: #F59E0B;
  --cira-error: #EF4444;
  --cira-info: #3B82F6;
}

html, body {
  font-family: var(--font-sans);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code, pre, .mono {
  font-family: var(--font-mono);
}
```

### Icon System

Material Design Icons via `@mdi/font`:

```typescript
import '@mdi/font/css/materialdesignicons.css'
```

Usage:
```vue
<v-icon icon="mdi-account"></v-icon>
<v-icon icon="mdi-folder"></v-icon>
<v-icon icon="mdi-shield-account"></v-icon>
```

---

## Global Styles (SCSS)

### Custom Scrollbar

```scss
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(99, 102, 241, 0.3);  // primary with opacity
  border-radius: 4px;
  &:hover {
    background: rgba(99, 102, 241, 0.5);
  }
}
```

### Status Indicator

```scss
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;

  &.connected {
    background-color: var(--cira-success);
    box-shadow: 0 0 8px var(--cira-success);
  }
  &.disconnected {
    background-color: var(--cira-error);
  }
  &.loading {
    background-color: var(--cira-warning);
    animation: pulse 1s infinite;
  }
}
```

### Mode Toggle Styling

```scss
.mode-toggle {
  .v-btn {
    transition: all 0.3s ease;
    &.v-btn--active {
      background: linear-gradient(135deg, var(--cira-primary), var(--cira-secondary)) !important;
    }
  }
}
```

### Metric Cards

```scss
.metric-card {
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--cira-primary), var(--cira-secondary));
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--cira-primary);
  }
}
```

---

## App Layout & Navigation

### App Bar (`App.vue`)

```vue
<v-app-bar v-if="authStore.isAuthenticated" elevation="0" color="surface" border="b">
  <!-- Drawer toggle -->
  <template #prepend>
    <v-app-bar-nav-icon @click="drawer = !drawer" />
  </template>

  <LogoFull compact />
  <v-spacer />

  <!-- Mode Toggle: Anomaly / Classification -->
  <v-btn-toggle v-model="pipelineStore.mode" mandatory rounded="lg" density="comfortable">
    <v-btn value="anomaly" size="small">
      <v-icon start>mdi-chart-bell-curve</v-icon> Anomaly
    </v-btn>
    <v-btn value="classification" size="small">
      <v-icon start>mdi-shape</v-icon> Classification
    </v-btn>
  </v-btn-toggle>

  <v-spacer />

  <!-- User Menu -->
  <v-menu>
    <template #activator="{ props }">
      <v-btn v-bind="props" variant="text" class="text-none">
        <v-avatar size="32" color="primary">
          <span class="text-caption">{{ userInitials }}</span>
        </v-avatar>
        <span class="ml-2 d-none d-sm-inline">{{ authStore.user?.display_name }}</span>
        <v-icon end>mdi-chevron-down</v-icon>
      </v-btn>
    </template>
    <v-list density="compact" nav>
      <v-list-item prepend-icon="mdi-account" :subtitle="authStore.user?.role">
        {{ authStore.user?.username }}
      </v-list-item>
      <v-divider />
      <v-list-item prepend-icon="mdi-key" @click="showChangePassword = true">
        Change Password
      </v-list-item>
      <v-list-item prepend-icon="mdi-theme-light-dark" @click="toggleTheme">
        Toggle Theme
      </v-list-item>
      <v-list-item prepend-icon="mdi-logout" @click="logout" class="text-error">
        Logout
      </v-list-item>
    </v-list>
  </v-menu>
</v-app-bar>
```

### Navigation Drawer

```vue
<v-navigation-drawer v-if="authStore.isAuthenticated" v-model="drawer" :rail="rail" permanent @click="rail = false">
  <v-list density="compact" nav>
    <v-list-item prepend-icon="mdi-view-dashboard" title="Dashboard" :to="{ name: 'dashboard' }" rounded="lg" />

    <v-list-subheader v-if="!rail">PIPELINE</v-list-subheader>

    <v-list-item prepend-icon="mdi-database" title="Data Source" :to="{ name: 'pipeline-data' }" rounded="lg" />
    <v-list-item prepend-icon="mdi-tune-vertical" title="Windowing" :to="{ name: 'pipeline-windowing' }" rounded="lg" />
    <v-list-item prepend-icon="mdi-auto-fix" title="Features" :to="{ name: 'pipeline-features' }" rounded="lg" />
    <v-list-item prepend-icon="mdi-brain" title="Training" :to="{ name: 'pipeline-training' }" rounded="lg" />
    <v-list-item prepend-icon="mdi-rocket-launch" title="Deploy" :to="{ name: 'pipeline-deploy' }" rounded="lg" />
  </v-list>

  <template #append>
    <v-list density="compact" nav>
      <v-divider class="mb-2" />
      <!-- Admin link - only visible for admin users -->
      <v-list-item v-if="authStore.isAdmin" prepend-icon="mdi-shield-account" title="Admin" :to="{ name: 'admin' }" rounded="lg" />
      <!-- Collapse toggle -->
      <v-list-item @click.stop="rail = !rail" rounded="lg">
        <template #prepend>
          <v-icon>{{ rail ? 'mdi-chevron-right' : 'mdi-chevron-left' }}</v-icon>
        </template>
      </v-list-item>
    </v-list>
  </template>
</v-navigation-drawer>
```

### Footer with Backend Status

```vue
<v-footer v-if="authStore.isAuthenticated" app height="32" color="surface" border="t">
  <div class="d-flex align-center w-100 px-4">
    <span class="status-dot" :class="backendStatus" />
    <span class="ml-2 text-caption text-medium-emphasis">
      {{ backendStatus === 'connected' ? 'Backend Connected' : 'Connecting...' }}
    </span>
    <v-divider vertical class="mx-4" />
    <span class="text-caption text-medium-emphasis">
      Mode: <strong>{{ pipelineStore.mode === 'anomaly' ? 'Anomaly Detection' : 'Classification' }}</strong>
    </span>
    <v-spacer />
    <span class="text-caption text-medium-emphasis">CiRA ME v1.0.0</span>
  </div>
</v-footer>
```

### Login Page (`LoginView.vue`)

```vue
<v-container fluid class="fill-height login-container">
  <v-row align="center" justify="center">
    <v-col cols="12" sm="8" md="4">
      <v-card class="pa-6" elevation="8">
        <div class="text-center mb-6">
          <LogoFull show-tagline />
        </div>

        <v-form ref="form" @submit.prevent="handleLogin">
          <v-text-field v-model="username" label="Username" prepend-inner-icon="mdi-account"
            :rules="[rules.required]" :disabled="loading" autofocus />

          <v-text-field v-model="password" label="Password" prepend-inner-icon="mdi-lock"
            :type="showPassword ? 'text' : 'password'"
            :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
            :rules="[rules.required]" :disabled="loading"
            @click:append-inner="showPassword = !showPassword" />

          <v-alert v-if="error" type="error" variant="tonal" density="compact" class="mb-4">
            {{ error }}
          </v-alert>

          <v-btn type="submit" color="primary" size="large" block :loading="loading">
            <v-icon start>mdi-login</v-icon> Sign In
          </v-btn>
        </v-form>
      </v-card>
    </v-col>
  </v-row>
</v-container>

<style scoped lang="scss">
.login-container {
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
  min-height: 100vh;
}
</style>
```

### Notification System (`stores/notification.ts`)

```typescript
interface Snackbar {
  show: boolean
  message: string
  color: string
  timeout: number
}

export const useNotificationStore = defineStore('notification', () => {
  const snackbar = ref<Snackbar>({
    show: false, message: '', color: 'info', timeout: 3000
  })

  function showSuccess(message: string, timeout = 3000) { ... }
  function showError(message: string, timeout = 5000) { ... }
  function showWarning(message: string, timeout = 4000) { ... }
  function showInfo(message: string, timeout = 3000) { ... }
  function hide() { snackbar.value.show = false }

  return { snackbar, showSuccess, showError, showWarning, showInfo, hide }
})
```

Global snackbar in `App.vue`:
```vue
<v-snackbar v-model="snackbar.show" :color="snackbar.color" :timeout="snackbar.timeout" location="bottom right">
  {{ snackbar.message }}
  <template #actions>
    <v-btn variant="text" @click="snackbar.show = false">Close</v-btn>
  </template>
</v-snackbar>
```

---

## Authentication System

### Auth Store (`stores/auth.ts`)

```typescript
interface User {
  id: number
  username: string
  display_name: string
  role: string
  private_folder?: string  // Assigned private folder name
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const initialized = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const isAuthenticated = computed(() => !!user.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function initialize() { /* GET /api/auth/me */ }
  async function login(username: string, password: string) { /* POST /api/auth/login */ }
  async function logout() { /* POST /api/auth/logout */ }
  async function changePassword(currentPassword: string, newPassword: string) { /* POST /api/auth/change-password */ }

  return { user, initialized, loading, error, isAuthenticated, isAdmin, initialize, login, logout, changePassword }
})
```

### API Service (`services/api.ts`)

```typescript
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' }
})

// Response interceptor: redirect to /login on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)
```

### Router Guards (`router/index.ts`)

```typescript
const routes = [
  { path: '/login', name: 'login', meta: { requiresGuest: true } },
  { path: '/', name: 'dashboard', meta: { requiresAuth: true } },
  { path: '/pipeline/data', name: 'pipeline-data', meta: { requiresAuth: true } },
  { path: '/pipeline/windowing', name: 'pipeline-windowing', meta: { requiresAuth: true } },
  { path: '/pipeline/features', name: 'pipeline-features', meta: { requiresAuth: true } },
  { path: '/pipeline/training', name: 'pipeline-training', meta: { requiresAuth: true } },
  { path: '/pipeline/deploy', name: 'pipeline-deploy', meta: { requiresAuth: true } },
  { path: '/admin', name: 'admin', meta: { requiresAuth: true, requiresAdmin: true } },
  { path: '/:pathMatch(.*)*', redirect: '/' }  // Catch-all
]

router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()

  if (!authStore.initialized) {
    await authStore.initialize()
  }

  if (to.meta.requiresGuest && authStore.isAuthenticated) return next({ name: 'dashboard' })
  if (to.meta.requiresAuth && !authStore.isAuthenticated) return next({ name: 'login' })
  if (to.meta.requiresAdmin && !authStore.isAdmin) return next({ name: 'dashboard' })

  next()
})
```

### Backend Login Flow (`routes/auth.py`)

```python
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.get_by_username(username)

    if not user or not user.get('is_active'):
        return jsonify({'error': 'Invalid credentials'}), 401

    if not User.verify_password(user, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    User.update_last_login(user['id'])

    session['user_id'] = user['id']
    session['login_time'] = datetime.utcnow().isoformat()

    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user['id'],
            'username': user['username'],
            'display_name': user['display_name'],
            'role': user['role'],
            'private_folder': user.get('private_folder')
        }
    })
```

### Auth Decorators (`auth.py`)

```python
def login_required(f):
    """Checks session, expiration, user active status. Attaches user to request.current_user."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        # Check session expiration (default 8 hours)
        login_time = session.get('login_time')
        if login_time:
            login_dt = datetime.fromisoformat(login_time)
            lifetime = current_app.config.get('SESSION_LIFETIME_HOURS', 8)
            if datetime.utcnow() - login_dt > timedelta(hours=lifetime):
                session.clear()
                return jsonify({'error': 'Session expired'}), 401

        user = User.get_by_id(user_id)
        if not user or not user.get('is_active'):
            session.clear()
            return jsonify({'error': 'User not found or inactive'}), 401

        request.current_user = user
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Chains login_required + admin role check."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if request.current_user.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function
```

### Session Configuration (`__init__.py`)

```python
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'cira-me-dev-secret-key-change-in-production'),
    SESSION_LIFETIME_HOURS=8,
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max upload
)

CORS(app, supports_credentials=True, origins=['http://localhost:3030', 'http://127.0.0.1:3030'])
```

---

## User Management (Admin)

### Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'annotator',   -- 'admin' or 'annotator'
    is_active INTEGER NOT NULL DEFAULT 1,      -- 0 = disabled
    created_at TEXT NOT NULL,
    last_login TEXT,
    private_folder TEXT                        -- assigned folder name
);
```

### User Model (`models.py`)

```python
class User:
    @staticmethod
    def get_by_id(user_id: int) -> dict: ...

    @staticmethod
    def get_by_username(username: str) -> dict: ...

    @staticmethod
    def verify_password(user: dict, password: str) -> bool:
        return check_password_hash(user['password_hash'], password)

    @staticmethod
    def create(username, password, display_name, role, private_folder=None) -> int:
        # Hashes password with werkzeug.security.generate_password_hash
        ...

    @staticmethod
    def get_all() -> list:
        # Returns all users (without password_hash)
        ...

    @staticmethod
    def update(user_id: int, **kwargs):
        # Allowed fields: display_name, role, is_active, private_folder
        ...

    @staticmethod
    def change_password(user_id: int, new_password: str): ...

    @staticmethod
    def delete(user_id: int): ...
```

### Admin Routes (`routes/admin.py`)

```python
# List all users
@admin_bp.route('/users', methods=['GET'])
@admin_required
def list_users(): ...

# Create new user (auto-creates private folder if specified)
@admin_bp.route('/users', methods=['POST'])
@admin_required
def create_user():
    # Creates private folder on disk if private_folder is provided
    if private_folder:
        folder_path = os.path.join(datasets_root, private_folder)
        os.makedirs(folder_path, exist_ok=True)
    ...

# Update user (auto-creates private folder if changed)
@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    # Cannot modify main admin user by other admins
    # Creates private folder on disk if changed
    if private_folder and private_folder != user.get('private_folder'):
        folder_path = os.path.join(datasets_root, private_folder)
        os.makedirs(folder_path, exist_ok=True)
    ...

# Delete user (cannot delete 'admin' or self)
@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id): ...

# Reset user password (admin only)
@admin_bp.route('/users/<int:user_id>/password', methods=['PUT'])
@admin_required
def reset_user_password(user_id): ...

# Create dataset folder
@admin_bp.route('/create-folder', methods=['POST'])
@admin_required
def create_folder():
    # Sanitizes name: only alphanumeric, underscore, hyphen
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))
    ...

# Delete dataset folder (cannot delete 'shared')
@admin_bp.route('/delete-folder', methods=['POST'])
@admin_required
def delete_folder(): ...
```

### Admin Panel UI (`AdminView.vue`)

The Admin Panel has two sections:

**Left (8 cols): User Management Table**
- Data table showing: Username, Display Name, Role, Private Folder, Active status
- Actions per user: Edit, Reset Password, Delete (except 'admin')
- Role shown as colored chip: `admin` = primary, `annotator` = secondary

**Right (4 cols): Folder Management**
- List of existing folders with type chips (shared/private)
- Delete folder button (except shared folder)
- Create new folder input + button

**Dialogs:**
- Create/Edit User: username, password (create only), display_name, role selector, private_folder autocomplete (only for annotator role), active toggle (edit only)
- Password Reset: new password for selected user
- Delete User Confirmation
- Delete Folder Confirmation with warning alert

```typescript
// Private folder options exclude 'shared' folder
const privateFolderOptions = computed(() => {
  return folders.value
    .filter(f => f.type !== 'shared')
    .map(f => f.name)
})
```

### Default Credentials

- **Username:** `admin`
- **Password:** `admin123`

---

## Folder Access Control

### User Roles

| Role | Access Level |
|------|--------------|
| **admin** | Full access to all folders in datasets root, user management |
| **annotator** | Access to assigned private folder + shared folder only |

### Folder Structure

```
datasets/
├── shared/                    # Accessible by ALL users
│   └── uploads/
│       └── user_{id}/         # Per-user upload subdirectory
├── user1_folder/              # Private folder assigned to user1
├── user2_folder/              # Private folder assigned to user2
└── ...
```

### Path Validation (`auth.py`)

```python
def validate_path(path: str, user: dict, datasets_root: str, shared_folder: str) -> bool:
    """Prevents directory traversal. Validates user access to path."""
    path = os.path.normpath(os.path.abspath(path))
    datasets_root = os.path.normpath(os.path.abspath(datasets_root))

    # Must be within datasets root
    if not path.startswith(datasets_root):
        return False

    # Admin: full access within datasets root
    if user.get('role') == 'admin':
        return True

    # Annotator: shared folder access
    shared_path = os.path.normpath(os.path.join(datasets_root, shared_folder))
    if path.startswith(shared_path):
        return True

    # Annotator: private folder access
    private_folder = user.get('private_folder')
    if private_folder:
        private_path = os.path.normpath(os.path.join(datasets_root, private_folder))
        if path.startswith(private_path):
            return True

    return False
```

### User Folder Discovery (`auth.py`)

```python
def get_user_folders(user: dict, datasets_root: str, shared_folder: str) -> list:
    """Get list of folders accessible to a user."""
    folders = []

    # Shared folder - always visible
    shared_path = os.path.join(datasets_root, shared_folder)
    if os.path.exists(shared_path):
        folders.append({'name': shared_folder, 'path': shared_path, 'type': 'shared'})

    # Admin: gets all folders
    if user.get('role') == 'admin':
        for item in os.listdir(datasets_root):
            item_path = os.path.join(datasets_root, item)
            if os.path.isdir(item_path) and item != shared_folder:
                folders.append({'name': item, 'path': item_path, 'type': 'private'})
    else:
        # Annotator: gets only their private folder
        private_folder = user.get('private_folder')
        if private_folder:
            private_path = os.path.join(datasets_root, private_folder)
            # Auto-create if doesn't exist
            if not os.path.exists(private_path):
                try:
                    os.makedirs(private_path, exist_ok=True)
                except Exception:
                    pass
            if os.path.exists(private_path):
                folders.append({'name': private_folder, 'path': private_path, 'type': 'private'})

    return folders
```

---

## File Delete Permissions

### Permission Logic (Frontend)

Users see a delete button based on context:

```typescript
function canDeleteItem(item: FileItem): boolean {
  // Admins can delete anything
  if (authStore.isAdmin) return true

  const user = authStore.user
  if (!user) return false

  // Normalize paths for cross-platform comparison
  const itemPath = item.path.toLowerCase().replace(/\\/g, '/')
  const currPath = currentPath.value.toLowerCase().replace(/\\/g, '/')

  // User can delete from their private folder
  if (user.private_folder) {
    const privateFolderLower = user.private_folder.toLowerCase()
    if (itemPath.includes(`/${privateFolderLower}/`) || itemPath.endsWith(`/${privateFolderLower}`)) {
      return true
    }
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
```

### Delete Endpoints (Backend)

**User delete** (`POST /api/data/delete-upload`):
- Users can delete from: their uploads folder, their private folder
- Admins can delete from anywhere
- Path must be within datasets root

**Admin delete** (`POST /api/data/admin/delete`):
- Admin-only endpoint
- Can delete any file/folder within datasets root
- Cannot delete datasets root itself or shared folder root

```typescript
// Frontend routes to correct endpoint
async function executeDelete() {
  const endpoint = authStore.isAdmin
    ? '/api/data/admin/delete'
    : '/api/data/delete-upload'

  await api.post(endpoint, { file_path: itemToDelete.value.path })
}
```

---

## API Endpoints Reference

### Authentication (`/api/auth`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/login` | User login (returns user with private_folder) | No |
| POST | `/api/auth/logout` | Clear session | Yes |
| GET | `/api/auth/me` | Get current user info (includes private_folder) | Yes |
| POST | `/api/auth/change-password` | Change own password (min 6 chars) | Yes |

### Admin - User Management (`/api/admin`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/admin/users` | List all users | Admin |
| POST | `/api/admin/users` | Create user (auto-creates private folder) | Admin |
| PUT | `/api/admin/users/:id` | Update user (auto-creates private folder if changed) | Admin |
| DELETE | `/api/admin/users/:id` | Delete user (not 'admin' or self) | Admin |
| PUT | `/api/admin/users/:id/password` | Reset password (min 6 chars) | Admin |

### Admin - Folder Management (`/api/admin`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/admin/create-folder` | Create folder (sanitized name) | Admin |
| POST | `/api/admin/delete-folder` | Delete folder (not 'shared') | Admin |

### Data Source & File Operations (`/api/data`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/data/datasets-root` | Get datasets root path | Yes |
| GET | `/api/data/user-folders` | Get user's accessible folders | Yes |
| POST | `/api/data/browse` | Browse directory contents | Yes |
| POST | `/api/data/upload` | Upload single file | Yes |
| POST | `/api/data/upload-multiple` | Upload multiple files | Yes |
| POST | `/api/data/delete-upload` | Delete file (user's own folders) | Yes |
| POST | `/api/data/admin/delete` | Delete any file (admin only) | Admin |

### Health Check

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Backend health check | No |

---

## Key Architecture Patterns

1. **Composition API** - Vue 3 `<script setup>` with TypeScript
2. **Pinia Stores** - `auth`, `pipeline`, `notification` stores
3. **Session-based Auth** - Flask server-side sessions with cookie
4. **Axios Interceptors** - Auto-redirect to login on 401
5. **Route Guards** - `requiresAuth`, `requiresGuest`, `requiresAdmin`
6. **Decorator Chains** - `@admin_required` chains `@login_required`
7. **Path Validation** - `os.path.normpath` + `startswith` for directory traversal prevention
8. **Auto-folder Creation** - Private folders created on user assignment and on first access
9. **Role-based UI** - Admin-only elements hidden via `v-if="authStore.isAdmin"`
10. **Cross-platform Paths** - Normalize backslashes to forward slashes for path comparison

---

*Updated from CiRA ME project - Machine Intelligence for Edge Computing*
