"""
CiRA ME - Machine Intelligence for Edge Computing
Flask Application Factory
"""

from flask import Flask
from flask_cors import CORS
import os


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Default configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'cira-me-dev-secret-key-change-in-production'),
        DATASETS_ROOT_PATH=os.environ.get('DATASETS_ROOT_PATH', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')),
        SHARED_FOLDER_PATH='shared',
        SESSION_LIFETIME_HOURS=8,
        DATABASE_PATH=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cirame.db'),
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max upload
    )

    # Override with custom config if provided
    if config:
        app.config.update(config)

    # Enable CORS
    CORS(app, supports_credentials=True, origins=['http://localhost:3030', 'http://127.0.0.1:3030'])

    # Ensure required directories exist
    os.makedirs(app.config['DATASETS_ROOT_PATH'], exist_ok=True)
    os.makedirs(os.path.join(app.config['DATASETS_ROOT_PATH'], app.config['SHARED_FOLDER_PATH']), exist_ok=True)
    os.makedirs(os.path.dirname(app.config['DATABASE_PATH']), exist_ok=True)

    # Initialize database
    from .models import init_db
    init_db(app.config['DATABASE_PATH'])

    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.admin import admin_bp
    from .routes.data_sources import data_sources_bp
    from .routes.features import features_bp
    from .routes.training import training_bp
    from .routes.deployment import deployment_bp
    from .routes.sensor_recording import sensor_bp
    from .routes.ti_tinyml import ti_bp
    from .routes.melab import melab_bp
    from .routes.app_builder import app_builder_bp
    from .routes.mqtt_publisher import mqtt_bp
    from .routes.folder_watchers import folder_watchers_bp
    from .routes.wizard import wizard_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(data_sources_bp, url_prefix='/api/data')
    app.register_blueprint(features_bp, url_prefix='/api/features')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(deployment_bp, url_prefix='/api/deployment')
    app.register_blueprint(sensor_bp, url_prefix='/api/sensors')
    app.register_blueprint(ti_bp, url_prefix='/api/ti')
    app.register_blueprint(melab_bp, url_prefix='/api/melab')
    app.register_blueprint(app_builder_bp, url_prefix='/api/app-builder')
    app.register_blueprint(mqtt_bp, url_prefix='/api/mqtt')
    app.register_blueprint(folder_watchers_bp, url_prefix='/api/folder-watchers')
    app.register_blueprint(wizard_bp, url_prefix='/api/wizard')

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return {'status': 'healthy', 'app': 'CiRA ME', 'version': '1.0.0'}

    # Rehydrate any folder watchers whose persisted status is 'running'.
    # Skip in the Flask dev-reloader's *monitor* process (both parent and
    # child call create_app; only the child gets WERKZEUG_RUN_MAIN set),
    # so we don't spawn duplicate worker threads in the parent that die
    # instantly on fork. Production gunicorn doesn't set this variable —
    # in that mode we only rehydrate once per process; run gunicorn with
    # --workers 1 to avoid cross-process watcher duplication.
    # Wrapped so a failure here can't kill app startup.
    _is_reloader_child = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    _reloader_active = os.environ.get('FLASK_DEBUG') in ('1', 'true')
    if (not _reloader_active) or _is_reloader_child:
        try:
            from .services import folder_watcher_service
            folder_watcher_service.rehydrate_running_watchers()
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(
                f"Folder Watcher rehydration failed at startup: {e}"
            )

    return app
