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

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(data_sources_bp, url_prefix='/api/data')
    app.register_blueprint(features_bp, url_prefix='/api/features')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(deployment_bp, url_prefix='/api/deployment')

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return {'status': 'healthy', 'app': 'CiRA ME', 'version': '1.0.0'}

    return app
