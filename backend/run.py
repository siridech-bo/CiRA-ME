#!/usr/bin/env python3
"""
CiRA ME - Application Entry Point
Machine Intelligence for Edge Computing
"""

import os
from app import create_app

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5100))

    print(f"""
    +-----------------------------------------------------------+
    |                                                           |
    |              ___                                          |
    |             /   \\      ___                                |
    |   _________/     \\____/   \\________                       |
    |                   \\   /                                   |
    |                    ---                                    |
    |                                                           |
    |              CiRA ME v1.0.0                               |
    |       Machine Intelligence for Edge                       |
    |                                                           |
    |   Backend API running at http://{host}:{port}              |
    |                                                           |
    |   Endpoints:                                              |
    |   - /api/health - Health check                            |
    |   - /api/auth/* - Authentication                          |
    |   - /api/data/* - Data sources                            |
    |   - /api/features/* - Feature engineering                 |
    |   - /api/training/* - ML training                         |
    |   - /api/deployment/* - Edge deployment                   |
    |                                                           |
    +-----------------------------------------------------------+
    """)

    app.run(host=host, port=port, debug=debug)
