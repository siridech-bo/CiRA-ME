#!/bin/bash
# CiRA ME - Installation Script (Linux)
# Loads Docker images from .tar files and sets up the application

set -e

echo "============================================"
echo "  CiRA ME - Installation Script"
echo "  Machine Intelligence for Edge Computing"
echo "============================================"
echo

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running or not installed."
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

echo "[1/4] Docker is running..."
echo

# Load backend image
echo "[2/4] Loading backend image (this may take a few minutes)..."
if [ -f "cirame-backend.tar" ]; then
    docker load -i cirame-backend.tar
    echo "Backend image loaded successfully."
else
    echo "ERROR: cirame-backend.tar not found!"
    echo "Please ensure the file is in the same directory as this script."
    exit 1
fi
echo

# Load frontend image
echo "[3/4] Loading frontend image..."
if [ -f "cirame-frontend.tar" ]; then
    docker load -i cirame-frontend.tar
    echo "Frontend image loaded successfully."
else
    echo "ERROR: cirame-frontend.tar not found!"
    echo "Please ensure the file is in the same directory as this script."
    exit 1
fi
echo

# Create shared folder
echo "[4/4] Creating shared folder for datasets..."
mkdir -p shared
echo "Shared folder ready."
echo

echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo
echo "Installed images:"
docker images | grep cirame
echo
echo "Next steps:"
echo "  With GPU    : bash start.sh"
echo "  Without GPU : bash start-no-gpu.sh"
echo "  Access at   : http://localhost:3030"
echo "  Login       : admin / admin123"
echo
echo "NOTE: Change the admin password after first login!"
echo
