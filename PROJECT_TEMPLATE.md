# CiRA Oculus - Project Template Documentation

A comprehensive template for building Vue.js + Flask web applications with authentication, multi-user support, and folder-based access control.

---

## Table of Contents

1. [Tech Stack Overview](#tech-stack-overview)
2. [Project Structure](#project-structure)
3. [Frontend Architecture](#frontend-architecture)
4. [Backend Architecture](#backend-architecture)
5. [Authentication System](#authentication-system)
6. [Multi-User & Folder Access Control](#multi-user--folder-access-control)
7. [Configuration](#configuration)
8. [API Endpoints Reference](#api-endpoints-reference)
9. [Getting Started](#getting-started)
10. [Typography & Fonts](#typography--fonts)

---

## Tech Stack Overview

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Vue.js 3** | ^3.4.0 | Reactive UI framework (Composition API) |
| **Vuetify 3** | ^3.5.0 | Material Design component library |
| **Vue Router** | ^4.3.0 | Client-side routing with navigation guards |
| **Pinia** | ^2.1.7 | State management (replaces Vuex) |
| **Axios** | ^1.6.0 | HTTP client for API requests |
| **Vite** | ^5.0.0 | Build tool and dev server |
| **TypeScript** | ^5.3.0 | Type-safe JavaScript |
| **@mdi/font** | ^7.4.0 | Material Design Icons |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Programming language |
| **Flask** | Latest | Web framework |
| **SQLite** | Built-in | Database (lightweight, file-based) |
| **Werkzeug** | Latest | Password hashing & security utilities |

---

## Project Structure

```
project-root/
├── frontend/                    # Vue.js frontend application
│   ├── src/
│   │   ├── components/         # Reusable Vue components
│   │   ├── views/              # Page components (routed)
│   │   ├── stores/             # Pinia stores (state management)
│   │   ├── router/             # Vue Router configuration
│   │   ├── App.vue             # Root component
│   │   └── main.ts             # Application entry point
│   ├── package.json
│   ├── vite.config.ts          # Vite configuration with proxy
│   └── tsconfig.json
│
├── backend/                     # Flask backend application
│   ├── app/
│   │   ├── __init__.py         # Flask app factory
│   │   ├── routes.py           # API endpoints
│   │   ├── models.py           # Database models & functions
│   │   ├── auth.py             # Authentication utilities
│   │   └── config.py           # Configuration settings
│   ├── data/                   # SQLite database storage
│   ├── datasets/               # User data folders
│   │   └── shared/             # Shared folder for all users
│   └── run.py                  # Application entry point
│
└── PROJECT_TEMPLATE.md         # This documentation
```

---

## Frontend Architecture

### Main Entry Point (`main.ts`)

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

// Vuetify setup
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        dark: true,
        colors: {
          primary: '#2196F3',
          secondary: '#424242',
          accent: '#FFC107',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FB8C00',
          background: '#121212',
          surface: '#1E1E1E'
        }
      }
    }
  }
})

const pinia = createPinia()
const app = createApp(App)

app.use(pinia)
app.use(router)
app.use(vuetify)
app.mount('#app')
```

### Router Configuration (`router/index.ts`)

```typescript
import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginPage.vue'),
      meta: { requiresGuest: true }
    },
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/MainApp.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/admin',
      name: 'admin',
      component: () => import('@/views/AdminPage.vue'),
      meta: { requiresAuth: true, requiresAdmin: true }
    }
  ]
})

// Navigation guards
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()

  // Check authentication on first load
  if (!authStore.isInitialized) {
    await authStore.checkAuth()
  }

  // Route protection logic
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    next('/login')
  } else if (to.meta.requiresGuest && authStore.isAuthenticated) {
    next('/')
  } else if (to.meta.requiresAdmin && !authStore.isAdmin) {
    next('/')
  } else {
    next()
  }
})

export default router
```

### Auth Store (`stores/auth.ts`)

```typescript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

interface User {
  id: number
  username: string
  display_name: string
  role: string
  private_folder?: string
}

export const useAuthStore = defineStore('auth', () => {
  // State
  const user = ref<User | null>(null)
  const isInitialized = ref(false)

  // Computed
  const isAuthenticated = computed(() => !!user.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  // Actions
  async function login(username: string, password: string) {
    const response = await axios.post('/api/auth/login',
      { username, password },
      { withCredentials: true }
    )
    if (response.data.success) {
      user.value = response.data.user
    }
    return response.data
  }

  async function logout() {
    await axios.post('/api/auth/logout', {}, { withCredentials: true })
    user.value = null
  }

  async function checkAuth() {
    try {
      const response = await axios.get('/api/auth/me', { withCredentials: true })
      if (response.data.success) {
        user.value = response.data.user
      }
    } catch {
      user.value = null
    } finally {
      isInitialized.value = true
    }
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    const response = await axios.post('/api/auth/change-password',
      { current_password: currentPassword, new_password: newPassword },
      { withCredentials: true }
    )
    return response.data
  }

  return {
    user,
    isInitialized,
    isAuthenticated,
    isAdmin,
    login,
    logout,
    checkAuth,
    changePassword
  }
})
```

### Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import path from 'path'

export default defineConfig({
  plugins: [
    vue(),
    vuetify({ autoImport: true })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    port: 3010,
    proxy: {
      '/api': {
        target: 'http://localhost:5008',
        changeOrigin: true
      }
    }
  }
})
```

### App Layout (`App.vue`)

```vue
<template>
  <v-app>
    <!-- App Bar - Only show when authenticated -->
    <v-app-bar v-if="authStore.isAuthenticated" color="surface" elevation="2">
      <v-app-bar-title>
        <v-icon icon="mdi-eye" class="mr-2"></v-icon>
        Application Name
      </v-app-bar-title>

      <v-spacer></v-spacer>

      <!-- User Menu -->
      <v-menu>
        <template v-slot:activator="{ props }">
          <v-btn v-bind="props" variant="tonal">
            <v-icon icon="mdi-account-circle" class="mr-2"></v-icon>
            {{ authStore.user?.display_name }}
            <v-icon icon="mdi-chevron-down" class="ml-1"></v-icon>
          </v-btn>
        </template>
        <v-list>
          <v-list-item v-if="authStore.isAdmin" to="/admin">
            <v-list-item-title>User Management</v-list-item-title>
          </v-list-item>
          <v-list-item @click="showChangePassword = true">
            <v-list-item-title>Change Password</v-list-item-title>
          </v-list-item>
          <v-divider></v-divider>
          <v-list-item @click="handleLogout">
            <v-list-item-title>Logout</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
    </v-app-bar>

    <!-- Main Content -->
    <v-main>
      <router-view />
    </v-main>

    <!-- Footer -->
    <v-footer v-if="authStore.isAuthenticated" app>
      <span>© 2024 Your Company</span>
    </v-footer>
  </v-app>
</template>

<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

const authStore = useAuthStore()
const router = useRouter()

const handleLogout = async () => {
  await authStore.logout()
  router.push('/login')
}
</script>
```

---

## Backend Architecture

### Flask App Factory (`__init__.py`)

```python
from flask import Flask
from flask_cors import CORS
from datetime import timedelta
import os

def create_app():
    app = Flask(__name__)

    # CORS configuration
    CORS(app, supports_credentials=True, origins=['http://localhost:3010', 'http://localhost:3011'])

    # Session configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS

    # Initialize database
    from app.models import init_db
    with app.app_context():
        init_db()

    # Register blueprints
    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
```

### Configuration (`config.py`)

```python
import os
from datetime import timedelta

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'app.db')

# Session
SECRET_KEY = os.environ.get('SECRET_KEY', 'change-in-production')
SESSION_LIFETIME = timedelta(hours=8)

# Folder paths
DATASETS_ROOT_PATH = os.environ.get('DATASETS_ROOT_PATH', os.path.join(BASE_DIR, 'datasets'))
SHARED_FOLDER_PATH = os.environ.get('SHARED_FOLDER_PATH', os.path.join(DATASETS_ROOT_PATH, 'shared'))

# Ensure folders exist
os.makedirs(DATASETS_ROOT_PATH, exist_ok=True)
os.makedirs(SHARED_FOLDER_PATH, exist_ok=True)

# Default admin credentials
DEFAULT_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = 'admin123'

# User roles
ROLE_ADMIN = 'admin'
ROLE_ANNOTATOR = 'annotator'
```

### Database Models (`models.py`)

```python
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.config import DATABASE_PATH, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, ROLE_ADMIN

def get_db_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            role TEXT NOT NULL DEFAULT 'annotator',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login TEXT,
            private_folder TEXT
        )
    ''')
    conn.commit()

    # Create default admin if no users exist
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        create_user(DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, 'Administrator', ROLE_ADMIN)

    conn.close()

def create_user(username, password, display_name=None, role='annotator', private_folder=None):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO users (username, password_hash, display_name, role, created_at, private_folder)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, generate_password_hash(password), display_name or username, role,
          datetime.utcnow().isoformat(), private_folder))

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    return {'id': user_id, 'username': username, 'display_name': display_name or username, 'role': role}

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()

    if user and user['is_active'] and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None
```

### Authentication Utilities (`auth.py`)

```python
import os
from functools import wraps
from flask import session, jsonify
from app.config import DATASETS_ROOT_PATH, ROLE_ADMIN, SHARED_FOLDER_PATH

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'code': 'AUTH_REQUIRED'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'code': 'AUTH_REQUIRED'}), 401
        if session.get('user_role') != ROLE_ADMIN:
            return jsonify({'error': 'Admin access required', 'code': 'ADMIN_REQUIRED'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    if 'user_id' not in session:
        return None
    return {
        'id': session.get('user_id'),
        'username': session.get('username'),
        'display_name': session.get('display_name'),
        'role': session.get('user_role'),
        'private_folder': session.get('private_folder')
    }

def is_path_allowed_for_user(path):
    """Check if path is accessible for current user."""
    if not path:
        return False

    user = get_current_user()
    if not user:
        return False

    requested_path = os.path.abspath(path)

    # Admins can access anything within DATASETS_ROOT_PATH
    if user.get('role') == ROLE_ADMIN:
        root_path = os.path.abspath(DATASETS_ROOT_PATH)
        common = os.path.commonpath([requested_path, root_path])
        return common == root_path

    # Regular users: check shared folder
    shared_path = os.path.abspath(SHARED_FOLDER_PATH)
    try:
        if os.path.commonpath([requested_path, shared_path]) == shared_path:
            return True
    except ValueError:
        pass

    # Regular users: check private folder
    private_folder = user.get('private_folder')
    if private_folder:
        private_path = os.path.abspath(private_folder)
        try:
            if os.path.commonpath([requested_path, private_path]) == private_path:
                return True
        except ValueError:
            pass

    return False
```

---

## Authentication System

### Login Flow

1. User submits credentials to `/api/auth/login`
2. Backend verifies credentials against database
3. On success, session data is stored:
   ```python
   session['user_id'] = user['id']
   session['username'] = user['username']
   session['display_name'] = user['display_name']
   session['user_role'] = user['role']
   session['private_folder'] = user.get('private_folder')
   session.permanent = True
   ```
4. Frontend stores user info in Pinia store
5. Vue Router redirects to home page

### Session Management

- Sessions expire after 8 hours (configurable)
- Frontend uses `withCredentials: true` for all API requests
- Backend uses Flask session with secure cookies

### Route Protection

**Frontend (Vue Router guards):**
- `requiresAuth`: Redirects to login if not authenticated
- `requiresGuest`: Redirects to home if already authenticated
- `requiresAdmin`: Redirects to home if not admin

**Backend (Decorators):**
- `@login_required`: Returns 401 if not authenticated
- `@admin_required`: Returns 403 if not admin

---

## Multi-User & Folder Access Control

### User Roles

| Role | Access Level |
|------|--------------|
| **admin** | Full access to all datasets, user management |
| **annotator** | Access to assigned private folder + shared folder |

### Folder Structure

```
datasets/
├── shared/           # Accessible by ALL users
├── user1_folder/     # Private folder for user1
├── user2_folder/     # Private folder for user2
└── ...
```

### Access Control Logic

```python
# Admin: Can access anything within DATASETS_ROOT_PATH
# Annotator: Can only access:
#   1. Their assigned private_folder
#   2. The shared folder (SHARED_FOLDER_PATH)
```

### Admin Folder Management

Admins can:
- Create folders within datasets root
- Delete folders (empty or with contents)
- Assign/reassign private folders to users

---

## API Endpoints Reference

### Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/login` | User login | No |
| POST | `/api/auth/logout` | User logout | Yes |
| GET | `/api/auth/me` | Get current user | Yes |
| POST | `/api/auth/change-password` | Change password | Yes |

### Admin - User Management

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/admin/users` | List all users | Admin |
| POST | `/api/admin/users` | Create user | Admin |
| PUT | `/api/admin/users/:id` | Update user | Admin |
| DELETE | `/api/admin/users/:id` | Delete user | Admin |
| PUT | `/api/admin/users/:id/password` | Reset password | Admin |

### Admin - Folder Management

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/admin/create-folder` | Create folder | Admin |
| POST | `/api/admin/delete-folder` | Delete folder | Admin |

### User Folders

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/user-folders` | Get accessible folders | Yes |
| GET | `/api/datasets-root` | Get datasets root path | Yes |
| POST | `/api/browse-directories` | Browse directories | Yes |

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- npm or yarn

### Installation

**Frontend:**
```bash
cd frontend
npm install
```

**Backend:**
```bash
cd backend
pip install flask flask-cors werkzeug
```

### Development

**Start Backend:**
```bash
cd backend
python run.py
# Runs on http://localhost:5008
```

**Start Frontend:**
```bash
cd frontend
npm run dev
# Runs on http://localhost:3010
```

### Production Build

```bash
cd frontend
npm run build
# Output in dist/ folder
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-in-production` | Flask session secret |
| `DATASETS_ROOT_PATH` | `backend/datasets` | Root folder for data |
| `SHARED_FOLDER_PATH` | `datasets/shared` | Shared folder path |

---

## Default Credentials

- **Username:** `admin`
- **Password:** `admin123`

**Important:** Change the default admin password after first login!

---

## Typography & Fonts

### Default Font Stack

Vuetify 3 uses **Roboto** as its default font family, automatically loaded via the framework.

```css
/* Vuetify's default font stack */
font-family: 'Roboto', sans-serif;
```

### Adding Custom Fonts (Optional)

To use custom fonts like Google Fonts:

**1. Add to `index.html`:**
```html
<head>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
```

**2. Override in Vuetify config (`main.ts`):**
```typescript
const vuetify = createVuetify({
  defaults: {
    global: {
      font: {
        family: 'Inter, sans-serif'
      }
    }
  },
  theme: {
    // ... theme config
  }
})
```

**3. Or use CSS variables:**
```css
/* In App.vue or global styles */
:root {
  --v-font-family: 'Inter', sans-serif;
}

body {
  font-family: var(--v-font-family);
}
```

### Icon Font

The project uses **Material Design Icons** via `@mdi/font`:

```typescript
// main.ts
import '@mdi/font/css/materialdesignicons.css'
```

**Usage in templates:**
```vue
<v-icon icon="mdi-account"></v-icon>
<v-icon icon="mdi-folder"></v-icon>
<v-icon icon="mdi-cog"></v-icon>
```

Browse all icons at: https://materialdesignicons.com/

---

## Key Patterns Used

1. **Composition API** - Vue 3 `<script setup>` syntax
2. **Pinia Stores** - Reactive state management with computed properties
3. **Route Guards** - Authentication checks before navigation
4. **Decorators** - Python decorators for auth on endpoints
5. **Session-based Auth** - Server-side sessions with secure cookies
6. **Role-based Access** - Admin vs Annotator permissions
7. **Path Validation** - Prevent directory traversal attacks

---

*Generated from CiRA Oculus project - A SAM Dataset Annotation Platform*
