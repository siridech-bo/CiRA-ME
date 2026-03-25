"""
CiRA ME - Database Models and Functions
"""

import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from contextlib import contextmanager


DATABASE_PATH = None


def init_db(db_path: str):
    """Initialize the database with required tables."""
    global DATABASE_PATH
    DATABASE_PATH = db_path

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()

        # Users table
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

        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                mode TEXT NOT NULL DEFAULT 'anomaly',
                user_id INTEGER NOT NULL,
                config JSON,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                hyperparameters JSON,
                metrics JSON,
                model_path TEXT,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')

        # Saved models table (user-saved benchmarks)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                mode TEXT NOT NULL,
                metrics JSON,
                model_path TEXT,
                training_session_id TEXT,
                pipeline_config JSON,
                dataset_info JSON,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # ME-LAB: Inference endpoints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS melab_endpoints (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                saved_model_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                mode TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                feature_names JSON,
                n_features INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                last_inference_at TEXT,
                inference_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (saved_model_id) REFERENCES saved_models(id)
            )
        ''')

        # ME-LAB: API keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS melab_api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                name TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                expires_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # ME-LAB: Usage log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS melab_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint_id TEXT NOT NULL,
                api_key_id INTEGER,
                request_size INTEGER,
                latency_ms REAL,
                status_code INTEGER,
                created_at TEXT NOT NULL
            )
        ''')

        # App Builder: apps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_builder_apps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL DEFAULT 'Untitled App',
                slug TEXT UNIQUE,
                status TEXT NOT NULL DEFAULT 'draft',
                access TEXT NOT NULL DEFAULT 'private',
                nodes TEXT NOT NULL DEFAULT '[]',
                edges TEXT NOT NULL DEFAULT '[]',
                calls INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                published_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Create default admin user if not exists
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO users (username, password_hash, display_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                'admin',
                generate_password_hash('admin123'),
                'Administrator',
                'admin',
                datetime.utcnow().isoformat()
            ))

        conn.commit()


@contextmanager
def get_db():
    """Get database connection context manager."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class User:
    """User model operations."""

    @staticmethod
    def get_by_id(user_id: int) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_by_username(username: str) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def verify_password(user: dict, password: str) -> bool:
        return check_password_hash(user['password_hash'], password)

    @staticmethod
    def create(username: str, password: str, display_name: str, role: str, private_folder: str = None) -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password_hash, display_name, role, private_folder, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                username,
                generate_password_hash(password),
                display_name,
                role,
                private_folder,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def update_last_login(user_id: int):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE id = ?',
                (datetime.utcnow().isoformat(), user_id)
            )
            conn.commit()

    @staticmethod
    def get_all() -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, display_name, role, is_active, created_at, last_login, private_folder FROM users')
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def update(user_id: int, **kwargs):
        with get_db() as conn:
            cursor = conn.cursor()
            allowed_fields = ['display_name', 'role', 'is_active', 'private_folder']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

            if updates:
                set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
                cursor.execute(
                    f'UPDATE users SET {set_clause} WHERE id = ?',
                    (*updates.values(), user_id)
                )
                conn.commit()

    @staticmethod
    def change_password(user_id: int, new_password: str):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET password_hash = ? WHERE id = ?',
                (generate_password_hash(new_password), user_id)
            )
            conn.commit()

    @staticmethod
    def delete(user_id: int):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()


class Project:
    """Project model operations."""

    @staticmethod
    def create(name: str, user_id: int, mode: str = 'anomaly', description: str = None, config: dict = None) -> int:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO projects (name, description, mode, user_id, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                name,
                description,
                mode,
                user_id,
                json.dumps(config) if config else None,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(project_id: int) -> dict:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
            row = cursor.fetchone()
            if row:
                project = dict(row)
                if project.get('config'):
                    project['config'] = json.loads(project['config'])
                return project
            return None

    @staticmethod
    def get_by_user(user_id: int) -> list:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC', (user_id,))
            projects = []
            for row in cursor.fetchall():
                project = dict(row)
                if project.get('config'):
                    project['config'] = json.loads(project['config'])
                projects.append(project)
            return projects

    @staticmethod
    def update(project_id: int, **kwargs):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            allowed_fields = ['name', 'description', 'mode', 'config']
            updates = {}

            for k, v in kwargs.items():
                if k in allowed_fields:
                    if k == 'config' and isinstance(v, dict):
                        updates[k] = json.dumps(v)
                    else:
                        updates[k] = v

            updates['updated_at'] = datetime.utcnow().isoformat()

            if updates:
                set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
                cursor.execute(
                    f'UPDATE projects SET {set_clause} WHERE id = ?',
                    (*updates.values(), project_id)
                )
                conn.commit()


class TrainingSession:
    """Training session model operations."""

    @staticmethod
    def create(project_id: int, algorithm: str, hyperparameters: dict = None) -> int:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_sessions (project_id, algorithm, hyperparameters, status, started_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                project_id,
                algorithm,
                json.dumps(hyperparameters) if hyperparameters else None,
                'running',
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def update_metrics(session_id: int, metrics: dict, model_path: str = None):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE training_sessions
                SET metrics = ?, model_path = ?, status = ?, completed_at = ?
                WHERE id = ?
            ''', (
                json.dumps(metrics),
                model_path,
                'completed',
                datetime.utcnow().isoformat(),
                session_id
            ))
            conn.commit()

    @staticmethod
    def get_by_project(project_id: int) -> list:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM training_sessions WHERE project_id = ? ORDER BY started_at DESC',
                (project_id,)
            )
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session.get('hyperparameters'):
                    session['hyperparameters'] = json.loads(session['hyperparameters'])
                if session.get('metrics'):
                    session['metrics'] = json.loads(session['metrics'])
                sessions.append(session)
            return sessions


class SavedModel:
    """Saved model (benchmark) operations."""

    @staticmethod
    def save(name: str, algorithm: str, mode: str, metrics: dict,
             model_path: str, training_session_id: str,
             pipeline_config: dict, dataset_info: dict, user_id: int) -> int:
        import json
        import math

        def _sanitize(v):
            """Replace NaN/Inf with 0.0 to ensure valid JSON."""
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return 0.0
            if isinstance(v, dict):
                return {k: _sanitize(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_sanitize(item) for item in v]
            return v

        metrics = _sanitize(metrics)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO saved_models
                (name, algorithm, mode, metrics, model_path, training_session_id,
                 pipeline_config, dataset_info, user_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, algorithm, mode,
                json.dumps(metrics),
                model_path,
                training_session_id,
                json.dumps(pipeline_config),
                json.dumps(dataset_info),
                user_id,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_all(user_id: int) -> list:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM saved_models WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,)
            )
            models = []
            for row in cursor.fetchall():
                m = dict(row)
                for field in ('metrics', 'pipeline_config', 'dataset_info'):
                    if m.get(field):
                        m[field] = json.loads(m[field])
                models.append(m)
            return models

    @staticmethod
    def get_by_id(model_id: int) -> dict:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM saved_models WHERE id = ?', (model_id,))
            row = cursor.fetchone()
            if not row:
                return None
            m = dict(row)
            for field in ('metrics', 'pipeline_config', 'dataset_info'):
                if m.get(field):
                    m[field] = json.loads(m[field])
            return m

    @staticmethod
    def delete(model_id: int, user_id: int) -> bool:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM saved_models WHERE id = ? AND user_id = ?',
                (model_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0


class MeLabEndpoint:
    """ME-LAB inference endpoint operations."""

    @staticmethod
    def create(endpoint_id, user_id, saved_model_id, name, mode, algorithm,
               feature_names=None, n_features=0, description=''):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO melab_endpoints
                (id, user_id, saved_model_id, name, description, status, mode, algorithm,
                 feature_names, n_features, created_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
            ''', (
                endpoint_id, user_id, saved_model_id, name, description,
                mode, algorithm,
                json.dumps(feature_names) if feature_names else None,
                n_features,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return endpoint_id

    @staticmethod
    def get_by_id(endpoint_id):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM melab_endpoints WHERE id = ?', (endpoint_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ('feature_names',):
                if d.get(k) and isinstance(d[k], str):
                    try: d[k] = json.loads(d[k])
                    except: pass
            return d

    @staticmethod
    def get_all(user_id=None):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute('SELECT * FROM melab_endpoints WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
            else:
                cursor.execute('SELECT * FROM melab_endpoints ORDER BY created_at DESC')
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                for k in ('feature_names',):
                    if d.get(k) and isinstance(d[k], str):
                        try: d[k] = json.loads(d[k])
                        except: pass
                results.append(d)
            return results

    @staticmethod
    def update_model(endpoint_id, saved_model_id, algorithm=None):
        with get_db() as conn:
            cursor = conn.cursor()
            if algorithm:
                cursor.execute(
                    'UPDATE melab_endpoints SET saved_model_id=?, algorithm=?, updated_at=? WHERE id=?',
                    (saved_model_id, algorithm, datetime.utcnow().isoformat(), endpoint_id)
                )
            else:
                cursor.execute(
                    'UPDATE melab_endpoints SET saved_model_id=?, updated_at=? WHERE id=?',
                    (saved_model_id, datetime.utcnow().isoformat(), endpoint_id)
                )
            conn.commit()

    @staticmethod
    def update_status(endpoint_id, status):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE melab_endpoints SET status=?, updated_at=? WHERE id=?',
                (status, datetime.utcnow().isoformat(), endpoint_id)
            )
            conn.commit()

    @staticmethod
    def record_inference(endpoint_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE melab_endpoints SET inference_count=inference_count+1, last_inference_at=? WHERE id=?',
                (datetime.utcnow().isoformat(), endpoint_id)
            )
            conn.commit()

    @staticmethod
    def delete(endpoint_id, user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM melab_endpoints WHERE id=? AND user_id=?',
                (endpoint_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def count_active(user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM melab_endpoints WHERE user_id=? AND status='active'",
                (user_id,)
            )
            return cursor.fetchone()[0]


class MeLabApiKey:
    """ME-LAB API key operations."""

    @staticmethod
    def create(user_id, name='default'):
        import secrets, hashlib
        raw_key = 'melab_' + secrets.token_hex(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:14]

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO melab_api_keys (key_hash, key_prefix, user_id, name, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (key_hash, key_prefix, user_id, name, datetime.utcnow().isoformat()))
            conn.commit()
            return {'id': cursor.lastrowid, 'key': raw_key, 'prefix': key_prefix}

    @staticmethod
    def validate(raw_key):
        """Validate an API key. Returns user_id and key_id if valid."""
        import hashlib
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, user_id, is_active, expires_at FROM melab_api_keys WHERE key_hash=?',
                (key_hash,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            if not d['is_active']:
                return None
            if d.get('expires_at'):
                if datetime.fromisoformat(d['expires_at']) < datetime.utcnow():
                    return None
            # Update last_used_at
            cursor.execute(
                'UPDATE melab_api_keys SET last_used_at=? WHERE id=?',
                (datetime.utcnow().isoformat(), d['id'])
            )
            conn.commit()
            return {'key_id': d['id'], 'user_id': d['user_id']}

    @staticmethod
    def get_all(user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, key_prefix, name, is_active, created_at, last_used_at FROM melab_api_keys WHERE user_id=? ORDER BY created_at DESC',
                (user_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def revoke(key_id, user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE melab_api_keys SET is_active=0 WHERE id=? AND user_id=?',
                (key_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0


class AppBuilderApp:
    """App Builder app operations."""

    @staticmethod
    def create(user_id, name='Untitled App', nodes=None, edges=None, access='private'):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO app_builder_apps (user_id, name, nodes, edges, access, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, name,
                json.dumps(nodes or []),
                json.dumps(edges or []),
                access,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(app_id):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM app_builder_apps WHERE id = ?', (app_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ('nodes', 'edges'):
                if d.get(k) and isinstance(d[k], str):
                    try: d[k] = json.loads(d[k])
                    except: pass
            return d

    @staticmethod
    def get_all(user_id):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM app_builder_apps WHERE user_id = ? ORDER BY updated_at DESC, created_at DESC',
                (user_id,)
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                for k in ('nodes', 'edges'):
                    if d.get(k) and isinstance(d[k], str):
                        try: d[k] = json.loads(d[k])
                        except: pass
                results.append(d)
            return results

    @staticmethod
    def update(app_id, **kwargs):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            allowed_fields = ['name', 'nodes', 'edges', 'access', 'status', 'slug',
                              'published_at']
            updates = {}
            for k, v in kwargs.items():
                if k in allowed_fields:
                    if k in ('nodes', 'edges') and isinstance(v, (list, dict)):
                        updates[k] = json.dumps(v)
                    else:
                        updates[k] = v
            updates['updated_at'] = datetime.utcnow().isoformat()
            if updates:
                set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
                cursor.execute(
                    f'UPDATE app_builder_apps SET {set_clause} WHERE id = ?',
                    (*updates.values(), app_id)
                )
                conn.commit()

    @staticmethod
    def delete(app_id, user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM app_builder_apps WHERE id = ? AND user_id = ?',
                (app_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def get_by_slug(slug):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM app_builder_apps WHERE slug = ?', (slug,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ('nodes', 'edges'):
                if d.get(k) and isinstance(d[k], str):
                    try: d[k] = json.loads(d[k])
                    except: pass
            return d

    @staticmethod
    def publish(app_id, slug):
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute(
                'UPDATE app_builder_apps SET status=?, slug=?, published_at=?, updated_at=? WHERE id=?',
                ('published', slug, now, now, app_id)
            )
            conn.commit()

    @staticmethod
    def increment_calls(app_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE app_builder_apps SET calls = calls + 1 WHERE id = ?',
                (app_id,)
            )
            conn.commit()
