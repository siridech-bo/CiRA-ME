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

        # Folder Watchers: per-user daemons that poll a folder and run ML inference
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS folder_watchers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                endpoint_id TEXT NOT NULL,
                input_folder TEXT NOT NULL,
                output_folder TEXT NOT NULL,
                poll_interval_s INTEGER NOT NULL DEFAULT 60,
                file_glob TEXT NOT NULL DEFAULT '*.txt',
                header_mode TEXT NOT NULL DEFAULT 'auto',
                parse_mode TEXT NOT NULL DEFAULT 'csv',
                parse_regex TEXT,
                parse_columns TEXT,
                mqtt_enabled INTEGER NOT NULL DEFAULT 0,
                mqtt_topic TEXT,
                daily_csv_enabled INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'stopped',
                last_run_at TEXT,
                last_error TEXT,
                files_processed INTEGER DEFAULT 0,
                rows_processed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (endpoint_id) REFERENCES melab_endpoints(id)
            )
        ''')

        # Log Watcher additions (migration-safe): parse_mode / regex + MQTT + daily CSV.
        # Existing installs get these columns added with the same defaults as CREATE.
        for _alter in (
            "ALTER TABLE folder_watchers ADD COLUMN parse_mode TEXT NOT NULL DEFAULT 'csv'",
            "ALTER TABLE folder_watchers ADD COLUMN parse_regex TEXT",
            "ALTER TABLE folder_watchers ADD COLUMN parse_columns TEXT",
            "ALTER TABLE folder_watchers ADD COLUMN mqtt_enabled INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE folder_watchers ADD COLUMN mqtt_topic TEXT",
            "ALTER TABLE folder_watchers ADD COLUMN daily_csv_enabled INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                cursor.execute(_alter)
            except Exception:
                pass  # column already exists

        # Add quota columns to users table (migration-safe)
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN max_folder_mb INTEGER DEFAULT 500')
        except Exception:
            pass  # Column already exists
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN max_endpoints INTEGER DEFAULT 5')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN max_apps INTEGER DEFAULT 10')
        except Exception:
            pass

        # ── F4: Project Status feature — pipeline-stage persistence tables ─────
        # Data / Windowing / Features sessions are persisted on Apply boundaries
        # (Q4). One project = one dataset (F4.3); dataset swap = clone.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                format TEXT NOT NULL,
                session_id TEXT,
                sensor_columns TEXT,
                label_column TEXT,
                labels TEXT,
                total_rows INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS windowed_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                data_session_id INTEGER NOT NULL,
                config TEXT,
                num_windows INTEGER,
                window_shape TEXT,
                normalization TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (data_session_id) REFERENCES data_sessions(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                windowed_session_id INTEGER NOT NULL,
                method TEXT,
                feature_names TEXT,
                num_features INTEGER,
                selection TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (windowed_session_id) REFERENCES windowed_sessions(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL UNIQUE,
                ordered_feature_names TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deploy_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                saved_model_id INTEGER,
                target TEXT NOT NULL,
                ref_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')

        # F4 ALTERs — cross-table project linkage (all migration-safe)
        try:
            cursor.execute('ALTER TABLE saved_models ADD COLUMN project_id INTEGER REFERENCES projects(id)')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE melab_endpoints ADD COLUMN project_id INTEGER REFERENCES projects(id)')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE app_builder_apps ADD COLUMN project_id INTEGER REFERENCES projects(id)')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE folder_watchers ADD COLUMN project_id INTEGER REFERENCES projects(id)')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE projects ADD COLUMN current_stage TEXT')
        except Exception:
            pass
        # Persist in-memory session_id UUIDs on windowed/feature rows so the
        # hydrate endpoint can restore the corresponding pickle blobs after a
        # backend restart (Approach 2b — session persistence to disk).
        try:
            cursor.execute('ALTER TABLE windowed_sessions ADD COLUMN session_id TEXT')
        except Exception:
            pass
        try:
            cursor.execute('ALTER TABLE feature_sessions ADD COLUMN session_id TEXT')
        except Exception:
            pass
        # Q3: Legacy project's mode column must permit NULL to render as "mixed".
        # The original schema declared `mode NOT NULL DEFAULT 'anomaly'`, which
        # would reject our Legacy row. SQLite can't drop NOT NULL in-place, so
        # if the constraint is present we just insert 'mixed' as a mode string
        # (chip logic still renders it distinctly).

        # ── F4: Legacy adoption migration (Q3, Watch-out 3, 4) ────────────────
        # Create one Legacy project per user who owns orphan resources, then
        # adopt their orphans. Idempotent: skips users that already have Legacy.
        try:
            orphan_users = set()
            for tbl in ('saved_models', 'melab_endpoints', 'app_builder_apps', 'training_sessions'):
                try:
                    if tbl == 'training_sessions':
                        # training_sessions has project_id NOT NULL — no orphans exist here
                        # by construction. Adopt only rows whose project row has been deleted.
                        cursor.execute(
                            f'''SELECT DISTINCT p.user_id
                                FROM training_sessions t
                                LEFT JOIN projects p ON t.project_id = p.id
                                WHERE p.id IS NULL AND p.user_id IS NOT NULL'''
                        )
                    else:
                        cursor.execute(
                            f'SELECT DISTINCT user_id FROM {tbl} '
                            f'WHERE project_id IS NULL AND user_id IS NOT NULL'
                        )
                    for row in cursor.fetchall():
                        if row[0] is not None:
                            orphan_users.add(row[0])
                except Exception:
                    # Table might not exist yet in super-fresh setups
                    pass

            for uid in orphan_users:
                cursor.execute(
                    "SELECT id FROM projects WHERE name = 'Legacy' AND user_id = ?",
                    (uid,)
                )
                existing = cursor.fetchone()
                if existing:
                    legacy_id = existing[0]
                else:
                    # `mode` column is NOT NULL from original schema; store 'mixed'
                    # sentinel so the UI can render as "mixed" (Q3). The plan
                    # says `mode = NULL` renders as mixed, but the NOT NULL
                    # constraint means we can't. Use 'mixed' string instead.
                    cursor.execute(
                        '''INSERT INTO projects (name, description, mode, user_id, created_at)
                           VALUES (?, ?, ?, ?, ?)''',
                        (
                            'Legacy',
                            'Auto-created project holding pre-F4 resources',
                            'mixed',
                            uid,
                            datetime.utcnow().isoformat()
                        )
                    )
                    legacy_id = cursor.lastrowid

                # Adopt orphans (skip folder_watchers per Q5)
                for tbl in ('saved_models', 'melab_endpoints', 'app_builder_apps'):
                    try:
                        cursor.execute(
                            f'UPDATE {tbl} SET project_id = ? '
                            f'WHERE user_id = ? AND project_id IS NULL',
                            (legacy_id, uid)
                        )
                    except Exception:
                        pass
                # training_sessions has project_id NOT NULL — re-parent rows
                # whose project row has been deleted for this user
                try:
                    cursor.execute(
                        '''UPDATE training_sessions
                           SET project_id = ?
                           WHERE project_id IN (
                             SELECT t.project_id FROM training_sessions t
                             LEFT JOIN projects p ON t.project_id = p.id
                             WHERE p.id IS NULL
                           ) AND EXISTS (
                             SELECT 1 FROM saved_models sm WHERE sm.training_session_id = training_sessions.id AND sm.user_id = ?
                           )''',
                        (legacy_id, uid)
                    )
                except Exception:
                    pass
        except Exception:
            # Migration must never break app startup
            import logging
            logging.getLogger(__name__).exception('F4 Legacy adoption migration failed')

        # ── Asset Tree tables (Phase A, 2026-07-18) ───────────────────────────
        # Physical asset hierarchy replacing "Project" abstraction. Strictly
        # additive — does NOT alter or touch any existing table's schema or
        # existing endpoint contracts. See docs/PLAN_2026-07-18_asset-tree.md.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_tree_config (
                id INTEGER PRIMARY KEY,
                level_names TEXT NOT NULL,
                root_name TEXT NOT NULL,
                topic_mode TEXT NOT NULL,
                meta_prefixes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER REFERENCES asset_nodes(id) ON DELETE CASCADE,
                level INTEGER NOT NULL,
                name TEXT NOT NULL,
                display_name TEXT,
                description TEXT,
                location_tag TEXT,
                topic_path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                retired_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(parent_id, name)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_sensor_meta (
                asset_id INTEGER PRIMARY KEY REFERENCES asset_nodes(id) ON DELETE CASCADE,
                unit TEXT,
                sample_rate_hz REAL,
                expected_min REAL,
                expected_max REAL,
                data_type TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS machine_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_by INTEGER NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS machine_group_members (
                group_id INTEGER NOT NULL REFERENCES machine_groups(id) ON DELETE CASCADE,
                machine_asset_id INTEGER NOT NULL REFERENCES asset_nodes(id) ON DELETE CASCADE,
                added_at TEXT NOT NULL,
                PRIMARY KEY(group_id, machine_asset_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_machine_bindings (
                saved_model_id INTEGER NOT NULL REFERENCES saved_models(id) ON DELETE CASCADE,
                machine_asset_id INTEGER NOT NULL REFERENCES asset_nodes(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                trained_via_group TEXT,
                bound_at TEXT NOT NULL,
                PRIMARY KEY(saved_model_id, machine_asset_id, role)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_tree_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id INTEGER NOT NULL REFERENCES users(id),
                event_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id INTEGER,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # ── Phase D (2026-07-19) — MQTT ingest router config ──────────────
        # Two new columns on asset_tree_config:
        #   ingest_enabled          BOOL — 1 = router routes to CSV, 0 = idle
        #   ingest_retention_days   INT  — janitor deletes CSV/log files
        #                                  older than this. Default 30.
        # Idempotent ALTER TABLE — silently ignored if already applied so a
        # second boot on the same DB doesn't fail. All Phase D features are
        # strictly additive; older columns / endpoints stay untouched.
        for _alter in (
            "ALTER TABLE asset_tree_config ADD COLUMN ingest_enabled INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE asset_tree_config ADD COLUMN ingest_retention_days INTEGER NOT NULL DEFAULT 30",
        ):
            try:
                cursor.execute(_alter)
            except Exception:
                # Column already exists — expected on second-and-later boots.
                pass

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
            allowed_fields = ['display_name', 'role', 'is_active', 'private_folder',
                              'max_folder_mb', 'max_endpoints', 'max_apps']
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
            allowed_fields = ['name', 'description', 'mode', 'config', 'current_stage']
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

    # ── F4 additions ──────────────────────────────────────────────────────
    @staticmethod
    def touch(project_id: int, stage: str):
        """Update current_stage + updated_at. Q2: latest-apply wins."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE projects SET current_stage = ?, updated_at = ? WHERE id = ?',
                (stage, datetime.utcnow().isoformat(), project_id)
            )
            conn.commit()

    @staticmethod
    def delete(project_id: int):
        """Explicit cascade per Watch-out 1 (FKs are off in SQLite).

        Deletes child stage rows; DETACHES saved_models / melab_endpoints /
        app_builder_apps / folder_watchers by setting project_id = NULL.
        """
        with get_db() as conn:
            cursor = conn.cursor()
            for tbl in ('deploy_records', 'feature_sessions',
                        'windowed_sessions', 'data_sessions',
                        'feature_templates'):
                try:
                    cursor.execute(f'DELETE FROM {tbl} WHERE project_id = ?', (project_id,))
                except Exception:
                    pass
            # training_sessions is NOT NULL — hard-delete rather than orphan
            try:
                cursor.execute('DELETE FROM training_sessions WHERE project_id = ?', (project_id,))
            except Exception:
                pass
            # Detach external refs (saved_models, endpoints, apps, watchers)
            for tbl in ('saved_models', 'melab_endpoints',
                        'app_builder_apps', 'folder_watchers'):
                try:
                    cursor.execute(f'UPDATE {tbl} SET project_id = NULL WHERE project_id = ?', (project_id,))
                except Exception:
                    pass
            cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
            conn.commit()

    @staticmethod
    def get_all_with_status(user_id: int = None) -> list:
        """Return per-project rows enriched with stage summaries and deploy
        breakdown. user_id=None means admin view (see all projects).
        """
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute(
                    '''SELECT p.*, u.username AS owner_username
                       FROM projects p
                       LEFT JOIN users u ON p.user_id = u.id
                       ORDER BY COALESCE(p.updated_at, p.created_at) DESC'''
                )
            else:
                cursor.execute(
                    '''SELECT p.*, u.username AS owner_username
                       FROM projects p
                       LEFT JOIN users u ON p.user_id = u.id
                       WHERE p.user_id = ?
                       ORDER BY COALESCE(p.updated_at, p.created_at) DESC''',
                    (user_id,)
                )
            base_rows = cursor.fetchall()
            projects = []
            for row in base_rows:
                proj = dict(row)
                pid = proj['id']
                c2 = conn.cursor()
                # Per-stage rows
                data_row = c2.execute(
                    'SELECT * FROM data_sessions WHERE project_id = ? ORDER BY id DESC LIMIT 1', (pid,)
                ).fetchone()
                win_row = c2.execute(
                    'SELECT * FROM windowed_sessions WHERE project_id = ? ORDER BY id DESC LIMIT 1', (pid,)
                ).fetchone()
                feat_row = c2.execute(
                    'SELECT * FROM feature_sessions WHERE project_id = ? ORDER BY id DESC LIMIT 1', (pid,)
                ).fetchone()
                # Trained models: saved_models attached to project
                sm_row = c2.execute(
                    '''SELECT id, name, algorithm, metrics FROM saved_models
                       WHERE project_id = ? ORDER BY created_at DESC LIMIT 1''', (pid,)
                ).fetchone()
                sm_count = c2.execute(
                    'SELECT COUNT(*) FROM saved_models WHERE project_id = ?', (pid,)
                ).fetchone()[0]

                # Deploy breakdown
                melab_rows = c2.execute(
                    '''SELECT id, name FROM melab_endpoints
                       WHERE project_id = ? AND status = 'active' ''', (pid,)
                ).fetchall()
                app_rows = c2.execute(
                    '''SELECT id, name FROM app_builder_apps
                       WHERE project_id = ? AND status = 'published' ''', (pid,)
                ).fetchall()
                ti_rows = c2.execute(
                    '''SELECT id, ref_id, metadata FROM deploy_records
                       WHERE project_id = ? AND target = 'ti_mcu' AND status = 'active' ''', (pid,)
                ).fetchall()
                jet_rows = c2.execute(
                    '''SELECT id, ref_id, metadata FROM deploy_records
                       WHERE project_id = ? AND target = 'jetson' AND status = 'active' ''', (pid,)
                ).fetchall()

                def _stage(present, summary):
                    return {'status': 'complete' if present else 'not_started',
                            'summary': summary if present else None}

                stages = {
                    'data': _stage(
                        data_row is not None,
                        f"{dict(data_row).get('format') if data_row else ''} · {dict(data_row).get('total_rows') if data_row else ''} rows"
                            if data_row else None
                    ),
                    'windowing': _stage(
                        win_row is not None,
                        f"{dict(win_row).get('num_windows') if win_row else ''} windows"
                            if win_row else None
                    ),
                    'features': _stage(
                        feat_row is not None,
                        f"{dict(feat_row).get('num_features') if feat_row else ''} features"
                            if feat_row else None
                    ),
                    'training': _stage(
                        sm_row is not None,
                        f"{sm_count} model{'s' if sm_count != 1 else ''} saved"
                    ),
                }
                if data_row:
                    stages['data']['id'] = dict(data_row)['id']
                if win_row:
                    stages['windowing']['id'] = dict(win_row)['id']
                if feat_row:
                    stages['features']['id'] = dict(feat_row)['id']

                deploy_targets = [
                    ('melab', melab_rows),
                    ('app_builder', app_rows),
                    ('ti_mcu', ti_rows),
                    ('jetson', jet_rows),
                ]
                total_deploys = sum(len(rows) for _, rows in deploy_targets)
                targets_with_any = sum(1 for _, rows in deploy_targets if len(rows) > 0)
                if total_deploys == 0:
                    deploy_status = 'not_started'
                elif targets_with_any >= 3:
                    deploy_status = 'complete'
                else:
                    deploy_status = 'in_progress'

                def _items(rows):
                    result = []
                    for r in rows:
                        d = dict(r)
                        result.append({
                            'id': d.get('id') or d.get('ref_id'),
                            'name': d.get('name') or d.get('ref_id') or f"#{d.get('id')}",
                        })
                    return result

                deploy_breakdown = {
                    'melab':       {'count': len(melab_rows), 'items': _items(melab_rows)},
                    'app_builder': {'count': len(app_rows),   'items': _items(app_rows)},
                    'ti_mcu':      {'count': len(ti_rows),    'items': _items(ti_rows)},
                    'jetson':      {'count': len(jet_rows),   'items': _items(jet_rows)},
                }

                stages['deploy'] = {'status': deploy_status,
                                    'summary': f"{total_deploys} target{'s' if total_deploys != 1 else ''}" if total_deploys else None}

                # Best metric — pick first available from newest saved_model
                best_metric = None
                if sm_row:
                    m = dict(sm_row).get('metrics')
                    if isinstance(m, str):
                        try: m = json.loads(m)
                        except Exception: m = {}
                    m = m or {}
                    for key in ('f1', 'F1', 'accuracy', 'Accuracy', 'auc_roc', 'AUROC',
                                'r2', 'R2', 'mae', 'MAE'):
                        if key in m and isinstance(m[key], (int, float)):
                            best_metric = {'name': key, 'value': m[key]}
                            break

                proj['stages'] = stages
                proj['deploy_breakdown'] = deploy_breakdown
                proj['best_metric'] = best_metric
                if proj.get('config'):
                    try:
                        proj['config'] = json.loads(proj['config']) if isinstance(proj['config'], str) else proj['config']
                    except Exception:
                        pass
                projects.append(proj)
            return projects


class DataSession:
    """Persisted data session (Q4: created on Windowing apply, not ingest)."""

    @staticmethod
    def create(project_id: int, file_path: str, format: str, session_id: str = None,
               sensor_columns=None, label_column=None, labels=None,
               total_rows=None) -> int:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO data_sessions
                   (project_id, file_path, format, session_id,
                    sensor_columns, label_column, labels, total_rows, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (project_id, file_path, format, session_id,
                 json.dumps(sensor_columns) if sensor_columns else None,
                 label_column,
                 json.dumps(labels) if labels else None,
                 total_rows,
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_project(project_id: int) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM data_sessions WHERE project_id = ? ORDER BY id DESC',
                (project_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_by_id(id: int) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM data_sessions WHERE id = ?', (id,))
            row = cursor.fetchone()
            return dict(row) if row else None


class WindowedSession:
    @staticmethod
    def create(project_id: int, data_session_id: int, config: dict = None,
               num_windows: int = None, window_shape=None, normalization=None,
               session_id: str = None) -> int:
        """Insert a windowed_sessions row.

        session_id is the in-memory UUID handed back by DataLoader.apply_windowing.
        Persisted so the hydrate endpoint can locate the matching pickle blob
        after a backend restart (Approach 2b — session persistence to disk).
        """
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO windowed_sessions
                   (project_id, data_session_id, config, num_windows,
                    window_shape, normalization, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (project_id, data_session_id,
                 json.dumps(config) if config else None,
                 num_windows,
                 json.dumps(window_shape) if window_shape else None,
                 json.dumps(normalization) if normalization else None,
                 session_id,
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_project(project_id: int) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM windowed_sessions WHERE project_id = ? ORDER BY id DESC',
                (project_id,)
            )
            return [dict(r) for r in cursor.fetchall()]


class FeatureSession:
    @staticmethod
    def create(project_id: int, windowed_session_id: int, method: str = None,
               feature_names=None, num_features: int = None,
               selection=None, session_id: str = None) -> int:
        """Insert a feature_sessions row.

        session_id is the in-memory key for _feature_sessions (e.g.
        ``features_<uuid>`` or ``features_fast_<uuid>``). Persisted so the
        hydrate endpoint can locate the matching pickle blob after a backend
        restart (Approach 2b — session persistence to disk).
        """
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO feature_sessions
                   (project_id, windowed_session_id, method, feature_names,
                    num_features, selection, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (project_id, windowed_session_id, method,
                 json.dumps(feature_names) if feature_names else None,
                 num_features,
                 json.dumps(selection) if selection else None,
                 session_id,
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_project(project_id: int) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM feature_sessions WHERE project_id = ? ORDER BY id DESC',
                (project_id,)
            )
            return [dict(r) for r in cursor.fetchall()]


class FeatureTemplate:
    """Per-project ordered feature list — stabilizes the ME-LAB /
    App Builder payload contract across retrainings.
    """
    @staticmethod
    def get(project_id: int) -> dict:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM feature_templates WHERE project_id = ?', (project_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get('ordered_feature_names'):
                try:
                    d['ordered_feature_names'] = json.loads(d['ordered_feature_names'])
                except Exception:
                    d['ordered_feature_names'] = []
            return d

    @staticmethod
    def upsert(project_id: int, ordered_feature_names: list) -> dict:
        """Insert or update. Bumps version on update."""
        import json
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            existing = cursor.execute(
                'SELECT id, version FROM feature_templates WHERE project_id = ?',
                (project_id,)
            ).fetchone()
            if existing:
                new_version = int(existing['version']) + 1
                cursor.execute(
                    '''UPDATE feature_templates
                       SET ordered_feature_names = ?, version = ?, updated_at = ?
                       WHERE project_id = ?''',
                    (json.dumps(ordered_feature_names), new_version, now, project_id)
                )
            else:
                cursor.execute(
                    '''INSERT INTO feature_templates
                       (project_id, ordered_feature_names, version, updated_at)
                       VALUES (?, ?, 1, ?)''',
                    (project_id, json.dumps(ordered_feature_names), now)
                )
            conn.commit()
        return FeatureTemplate.get(project_id)


class DeployRecord:
    """Rows for TI MCU / Jetson deploy targets. ME-LAB and App Builder
    counts are still read from their native tables.
    """
    @staticmethod
    def create(project_id: int, target: str, saved_model_id: int = None,
               ref_id: str = None, metadata: dict = None,
               status: str = 'active') -> int:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO deploy_records
                   (project_id, saved_model_id, target, ref_id, status,
                    metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (project_id, saved_model_id, target, ref_id, status,
                 json.dumps(metadata) if metadata else None,
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_project(project_id: int, target: str = None) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            if target:
                cursor.execute(
                    'SELECT * FROM deploy_records WHERE project_id = ? AND target = ? ORDER BY id DESC',
                    (project_id, target)
                )
            else:
                cursor.execute(
                    'SELECT * FROM deploy_records WHERE project_id = ? ORDER BY id DESC',
                    (project_id,)
                )
            return [dict(r) for r in cursor.fetchall()]


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


class FolderWatcher:
    """Folder Watcher operations. Owns a per-user polling worker that reads
    files from an input folder, runs each row through a ME-LAB endpoint,
    and writes results to an output folder as CSV.
    """

    _ALLOWED_UPDATE_FIELDS = (
        'name', 'input_folder', 'output_folder', 'poll_interval_s',
        'file_glob', 'header_mode', 'parse_mode', 'parse_regex',
        'parse_columns',
        'mqtt_enabled', 'mqtt_topic', 'daily_csv_enabled',
        'status', 'last_run_at', 'last_error',
        'files_processed', 'rows_processed',
    )

    @staticmethod
    def create(user_id, name, endpoint_id, input_folder, output_folder, **kwargs):
        poll_interval_s = int(kwargs.get('poll_interval_s', 60) or 60)
        file_glob = kwargs.get('file_glob') or '*.txt'
        header_mode = kwargs.get('header_mode') or 'auto'
        parse_mode = kwargs.get('parse_mode') or 'csv'
        parse_regex = kwargs.get('parse_regex')
        parse_columns = kwargs.get('parse_columns')
        mqtt_enabled = 1 if kwargs.get('mqtt_enabled') else 0
        mqtt_topic = kwargs.get('mqtt_topic')
        daily_csv_enabled = 1 if kwargs.get('daily_csv_enabled') else 0
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO folder_watchers
                (user_id, name, endpoint_id, input_folder, output_folder,
                 poll_interval_s, file_glob, header_mode,
                 parse_mode, parse_regex, parse_columns,
                 mqtt_enabled, mqtt_topic, daily_csv_enabled,
                 status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stopped', ?)
            ''', (
                user_id, name, endpoint_id, input_folder, output_folder,
                poll_interval_s, file_glob, header_mode,
                parse_mode, parse_regex, parse_columns,
                mqtt_enabled, mqtt_topic, daily_csv_enabled,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(watcher_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM folder_watchers WHERE id = ?', (watcher_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_by_user(user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM folder_watchers WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_all_running():
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM folder_watchers WHERE status = 'running'"
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def update(watcher_id, **kwargs):
        updates = {
            k: v for k, v in kwargs.items()
            if k in FolderWatcher._ALLOWED_UPDATE_FIELDS
        }
        if not updates:
            return
        # Coerce bool → int for the flag columns stored as INTEGER in SQLite.
        for _bool_col in ('mqtt_enabled', 'daily_csv_enabled'):
            if _bool_col in updates:
                updates[_bool_col] = 1 if updates[_bool_col] else 0
        with get_db() as conn:
            cursor = conn.cursor()
            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            cursor.execute(
                f'UPDATE folder_watchers SET {set_clause} WHERE id = ?',
                (*updates.values(), watcher_id)
            )
            conn.commit()

    @staticmethod
    def delete(watcher_id, user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM folder_watchers WHERE id = ? AND user_id = ?',
                (watcher_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def increment_counters(watcher_id, files_delta=0, rows_delta=0):
        """Atomic counter update — avoids read-modify-write race between worker ticks."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE folder_watchers '
                'SET files_processed = COALESCE(files_processed, 0) + ?, '
                '    rows_processed = COALESCE(rows_processed, 0) + ? '
                'WHERE id = ?',
                (int(files_delta), int(rows_delta), watcher_id)
            )
            conn.commit()


# ────────────────────────────────────────────────────────────────────────────
# Asset Tree models (Phase A, 2026-07-18)
# ────────────────────────────────────────────────────────────────────────────

class AssetTreeConfig:
    """Single-row config. Enforced at app layer (id=1 always)."""

    @staticmethod
    def get() -> dict:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM asset_tree_config WHERE id = 1')
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ('level_names', 'meta_prefixes'):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        d[k] = []
            return d

    @staticmethod
    def upsert(level_names: list, root_name: str, topic_mode: str,
               meta_prefixes: list) -> dict:
        import json
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            existing = cursor.execute(
                'SELECT id, created_at FROM asset_tree_config WHERE id = 1'
            ).fetchone()
            if existing:
                cursor.execute(
                    '''UPDATE asset_tree_config
                       SET level_names = ?, root_name = ?, topic_mode = ?,
                           meta_prefixes = ?, updated_at = ?
                       WHERE id = 1''',
                    (json.dumps(level_names), root_name, topic_mode,
                     json.dumps(meta_prefixes), now)
                )
            else:
                cursor.execute(
                    '''INSERT INTO asset_tree_config
                       (id, level_names, root_name, topic_mode,
                        meta_prefixes, created_at, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?)''',
                    (json.dumps(level_names), root_name, topic_mode,
                     json.dumps(meta_prefixes), now, now)
                )
            conn.commit()
        return AssetTreeConfig.get()

    @staticmethod
    def patch(**fields) -> dict:
        """Partial update of the single config row (Phase D).

        Accepts any subset of columns and updates them in place. Row must
        already exist (config is created by wizard / template apply); if
        it doesn't, returns None so callers can 404. Uses the same JSON
        encoding rule as `upsert` for list-valued columns.

        Ignores unknown fields silently — safe to feed a mixed dict.
        """
        import json
        allowed = {
            'level_names', 'root_name', 'topic_mode', 'meta_prefixes',
            'ingest_enabled', 'ingest_retention_days',
        }
        json_cols = {'level_names', 'meta_prefixes'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return AssetTreeConfig.get()
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            existing = cursor.execute(
                'SELECT id FROM asset_tree_config WHERE id = 1'
            ).fetchone()
            if not existing:
                return None
            set_parts = []
            values = []
            for k, v in updates.items():
                set_parts.append(f'{k} = ?')
                if k in json_cols:
                    values.append(json.dumps(v))
                elif k == 'ingest_enabled':
                    # Coerce truthy → 1, falsy → 0. Guards against JS 'true'/'false'
                    # strings arriving from the frontend PATCH.
                    values.append(1 if v and v != 'false' else 0)
                else:
                    values.append(v)
            set_parts.append('updated_at = ?')
            values.append(now)
            values.append(1)
            cursor.execute(
                f"UPDATE asset_tree_config SET {', '.join(set_parts)} WHERE id = ?",
                tuple(values),
            )
            conn.commit()
        return AssetTreeConfig.get()


class AssetNode:
    """Nodes in the asset hierarchy tree."""

    @staticmethod
    def get_by_id(node_id: int) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM asset_nodes WHERE id = ?', (node_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_by_topic_path(topic_path: str) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM asset_nodes WHERE topic_path = ?',
                (topic_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_children(parent_id) -> list:
        """Children of a given parent. parent_id=None returns root(s)."""
        with get_db() as conn:
            cursor = conn.cursor()
            if parent_id is None:
                cursor.execute(
                    'SELECT * FROM asset_nodes WHERE parent_id IS NULL ORDER BY name'
                )
            else:
                cursor.execute(
                    'SELECT * FROM asset_nodes WHERE parent_id = ? ORDER BY name',
                    (parent_id,)
                )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_sibling_by_name(parent_id, name: str) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            if parent_id is None:
                cursor.execute(
                    'SELECT * FROM asset_nodes WHERE parent_id IS NULL AND name = ?',
                    (name,)
                )
            else:
                cursor.execute(
                    'SELECT * FROM asset_nodes WHERE parent_id = ? AND name = ?',
                    (parent_id, name)
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def create(parent_id, level: int, name: str, topic_path: str,
               display_name: str = None, description: str = None,
               location_tag: str = None, status: str = 'active') -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO asset_nodes
                   (parent_id, level, name, display_name, description,
                    location_tag, topic_path, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (parent_id, level, name, display_name, description,
                 location_tag, topic_path, status,
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def update(node_id: int, **kwargs):
        allowed = ('name', 'display_name', 'description', 'location_tag',
                   'topic_path', 'parent_id', 'status', 'retired_at')
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        with get_db() as conn:
            cursor = conn.cursor()
            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            cursor.execute(
                f'UPDATE asset_nodes SET {set_clause} WHERE id = ?',
                (*updates.values(), node_id)
            )
            conn.commit()

    @staticmethod
    def get_descendants(node_id: int) -> list:
        """Recursive descendants (breadth-first)."""
        result = []
        with get_db() as conn:
            cursor = conn.cursor()
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                cursor.execute(
                    'SELECT * FROM asset_nodes WHERE parent_id = ?',
                    (current,)
                )
                for row in cursor.fetchall():
                    d = dict(row)
                    result.append(d)
                    queue.append(d['id'])
        return result

    @staticmethod
    def reactivate(node_id: int) -> bool:
        """Flip a single retired node back to status='active'.

        Used by the simulator's change-profile flow so a name that was
        retired in an earlier swap can be reused when the new profile
        happens to declare a sensor with the same name (e.g. two profiles
        that both have `vibration` or `temperature`). Returns True if the
        node existed and was flipped, False otherwise.
        """
        with get_db() as conn:
            cursor = conn.execute(
                "UPDATE asset_nodes SET status='active', retired_at=NULL "
                "WHERE id = ?", (node_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def retire_cascade(node_id: int) -> list:
        """Set node + all descendants to status='retired'. Returns list of affected ids."""
        now = datetime.utcnow().isoformat()
        affected = [node_id]
        for d in AssetNode.get_descendants(node_id):
            affected.append(d['id'])
        with get_db() as conn:
            cursor = conn.cursor()
            for nid in affected:
                cursor.execute(
                    '''UPDATE asset_nodes
                       SET status = 'retired', retired_at = ?
                       WHERE id = ?''',
                    (now, nid)
                )
            conn.commit()
        return affected

    @staticmethod
    def tree_as_nested_json(include_retired: bool = False) -> list:
        """Build the full tree as a list of root dicts with nested children.

        Sensor-level nodes (leaves at max depth) get their sensor_meta
        merged in. Retired nodes are excluded by default.
        """
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            if include_retired:
                cursor.execute('SELECT * FROM asset_nodes ORDER BY level, name')
            else:
                cursor.execute(
                    "SELECT * FROM asset_nodes WHERE status = 'active' ORDER BY level, name"
                )
            all_nodes = [dict(r) for r in cursor.fetchall()]

            # Attach sensor_meta to any node that has one (usually leaves)
            cursor.execute('SELECT * FROM asset_sensor_meta')
            meta_by_id = {r['asset_id']: dict(r) for r in cursor.fetchall()}

        by_id = {n['id']: n for n in all_nodes}
        for n in all_nodes:
            n['children'] = []
            if n['id'] in meta_by_id:
                meta = meta_by_id[n['id']]
                # asset_id is the FK; expose as sensor_meta dict without the FK
                sensor_meta = {k: v for k, v in meta.items() if k != 'asset_id'}
                n['sensor_meta'] = sensor_meta

        roots = []
        for n in all_nodes:
            pid = n.get('parent_id')
            if pid is None:
                roots.append(n)
            elif pid in by_id:
                by_id[pid]['children'].append(n)
            else:
                # Parent was retired and we're not including retireds; treat as root
                if include_retired:
                    roots.append(n)
        return roots


class AssetSensorMeta:
    """Sensor metadata (unit, sample rate, etc.) for leaf asset nodes."""

    @staticmethod
    def get(asset_id: int) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM asset_sensor_meta WHERE asset_id = ?',
                (asset_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def upsert(asset_id: int, unit: str = None, sample_rate_hz: float = None,
               expected_min: float = None, expected_max: float = None,
               data_type: str = None):
        with get_db() as conn:
            cursor = conn.cursor()
            existing = cursor.execute(
                'SELECT asset_id FROM asset_sensor_meta WHERE asset_id = ?',
                (asset_id,)
            ).fetchone()
            if existing:
                cursor.execute(
                    '''UPDATE asset_sensor_meta
                       SET unit = ?, sample_rate_hz = ?, expected_min = ?,
                           expected_max = ?, data_type = ?
                       WHERE asset_id = ?''',
                    (unit, sample_rate_hz, expected_min, expected_max,
                     data_type, asset_id)
                )
            else:
                cursor.execute(
                    '''INSERT INTO asset_sensor_meta
                       (asset_id, unit, sample_rate_hz, expected_min,
                        expected_max, data_type)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (asset_id, unit, sample_rate_hz, expected_min,
                     expected_max, data_type)
                )
            conn.commit()


class MachineGroup:
    """Groups of machine-level asset nodes."""

    @staticmethod
    def create(name: str, description: str, created_by: int) -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO machine_groups
                   (name, description, created_by, created_at)
                   VALUES (?, ?, ?, ?)''',
                (name, description, created_by, datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_by_id(group_id: int) -> dict:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM machine_groups WHERE id = ?', (group_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_all_with_counts() -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT g.*, COUNT(m.machine_asset_id) AS member_count
                   FROM machine_groups g
                   LEFT JOIN machine_group_members m ON g.id = m.group_id
                   GROUP BY g.id
                   ORDER BY g.name'''
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def update(group_id: int, **kwargs):
        allowed = ('name', 'description')
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        with get_db() as conn:
            cursor = conn.cursor()
            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            cursor.execute(
                f'UPDATE machine_groups SET {set_clause} WHERE id = ?',
                (*updates.values(), group_id)
            )
            conn.commit()

    @staticmethod
    def delete(group_id: int):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM machine_group_members WHERE group_id = ?', (group_id,))
            cursor.execute('DELETE FROM machine_groups WHERE id = ?', (group_id,))
            conn.commit()

    @staticmethod
    def get_members(group_id: int) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT n.* FROM asset_nodes n
                   INNER JOIN machine_group_members m ON n.id = m.machine_asset_id
                   WHERE m.group_id = ?
                   ORDER BY n.topic_path''',
                (group_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def set_members(group_id: int, machine_asset_ids: list):
        """Replace all group members with the given ids."""
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM machine_group_members WHERE group_id = ?',
                (group_id,)
            )
            for aid in machine_asset_ids:
                cursor.execute(
                    '''INSERT INTO machine_group_members
                       (group_id, machine_asset_id, added_at)
                       VALUES (?, ?, ?)''',
                    (group_id, aid, now)
                )
            conn.commit()


class AssetTreeAudit:
    """Audit log for asset-tree state changes."""

    @staticmethod
    def log(actor_user_id: int, event_type: str, target_type: str,
            target_id, payload: dict):
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO asset_tree_audit
                   (actor_user_id, event_type, target_type, target_id,
                    payload, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (actor_user_id, event_type, target_type, target_id,
                 json.dumps(payload, default=str),
                 datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def list(limit: int = 100, offset: int = 0) -> list:
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT a.*, u.username AS actor_username
                   FROM asset_tree_audit a
                   LEFT JOIN users u ON a.actor_user_id = u.id
                   ORDER BY a.id DESC
                   LIMIT ? OFFSET ?''',
                (limit, offset)
            )
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get('payload'):
                    try:
                        d['payload'] = json.loads(d['payload'])
                    except Exception:
                        pass
                rows.append(d)
            return rows

    @staticmethod
    def count() -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM asset_tree_audit')
            return cursor.fetchone()[0]


class ModelMachineBinding:
    """Model ↔ machine binding join table.

    Phase C (2026-07-19). One row per (saved_model_id, machine_asset_id, role)
    tuple. Roles in current use:
      - 'trained_on'   → the machine's data went into the training set.
      - 'deployed_to'  → the model is authorised to serve this machine.

    `trained_via_group` captures the group name at training time so we can
    audit even after the group is renamed or a machine is retired.
    """

    @staticmethod
    def bind(saved_model_id: int, machine_asset_id: int, role: str,
             trained_via_group: str = None) -> None:
        """Idempotent insert of a single binding row."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT OR REPLACE INTO model_machine_bindings
                   (saved_model_id, machine_asset_id, role,
                    trained_via_group, bound_at)
                   VALUES (?, ?, ?, ?, ?)''',
                (saved_model_id, machine_asset_id, role,
                 trained_via_group, datetime.utcnow().isoformat())
            )
            conn.commit()

    @staticmethod
    def bind_bulk(saved_model_id: int, machine_asset_ids: list, role: str,
                  trained_via_group: str = None) -> int:
        """Insert many binding rows in a single transaction. Returns count."""
        now = datetime.utcnow().isoformat()
        count = 0
        with get_db() as conn:
            cursor = conn.cursor()
            for aid in machine_asset_ids:
                cursor.execute(
                    '''INSERT OR REPLACE INTO model_machine_bindings
                       (saved_model_id, machine_asset_id, role,
                        trained_via_group, bound_at)
                       VALUES (?, ?, ?, ?, ?)''',
                    (saved_model_id, aid, role, trained_via_group, now)
                )
                count += 1
            conn.commit()
        return count

    @staticmethod
    def replace_role(saved_model_id: int, role: str,
                     machine_asset_ids: list,
                     trained_via_group: str = None) -> int:
        """Delete all rows for (model, role) and re-insert. Atomic."""
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''DELETE FROM model_machine_bindings
                   WHERE saved_model_id = ? AND role = ?''',
                (saved_model_id, role)
            )
            for aid in machine_asset_ids:
                cursor.execute(
                    '''INSERT INTO model_machine_bindings
                       (saved_model_id, machine_asset_id, role,
                        trained_via_group, bound_at)
                       VALUES (?, ?, ?, ?, ?)''',
                    (saved_model_id, aid, role, trained_via_group, now)
                )
            conn.commit()
        return len(machine_asset_ids)

    @staticmethod
    def get_for_model(saved_model_id: int) -> list:
        """Return every binding for a model, joined with the asset node."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT b.*, n.name AS asset_name, n.topic_path,
                          n.status AS asset_status, n.display_name
                   FROM model_machine_bindings b
                   INNER JOIN asset_nodes n ON n.id = b.machine_asset_id
                   WHERE b.saved_model_id = ?
                   ORDER BY b.role, n.topic_path''',
                (saved_model_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_by_role(saved_model_id: int, role: str) -> list:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT b.*, n.name AS asset_name, n.topic_path,
                          n.status AS asset_status, n.display_name
                   FROM model_machine_bindings b
                   INNER JOIN asset_nodes n ON n.id = b.machine_asset_id
                   WHERE b.saved_model_id = ? AND b.role = ?
                   ORDER BY n.topic_path''',
                (saved_model_id, role)
            )
            return [dict(r) for r in cursor.fetchall()]
