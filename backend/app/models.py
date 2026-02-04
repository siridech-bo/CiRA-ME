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
