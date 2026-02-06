#!/usr/bin/env python3
"""
CiRA ME - Application Entry Point
Machine Intelligence for Edge Computing
"""

import os
import sys

# Set environment variables BEFORE importing anything else
# This helps with PyTorch DLL loading on Windows when other CUDA apps are running
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Add CUDA to PATH if not already there (Windows)
if sys.platform == 'win32':
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    ]
    current_path = os.environ.get('PATH', '')
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path) and cuda_path not in current_path:
            os.environ['PATH'] = cuda_path + ';' + current_path
            break

from app import create_app

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = 5100  # Fixed port for CiRA ME backend

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
