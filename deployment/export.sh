#!/bin/bash
# CiRA ME - Build and Export Script (Linux/macOS)
# Run this from the PROJECT ROOT (parent of deployment/)
# Builds Docker images and saves them as .tar files ready for transfer
#
# Usage:  cd /path/to/CiRA-ME
#         bash deployment/export.sh

set -e

echo "============================================"
echo "  CiRA ME - Build and Export Images"
echo "============================================"
echo

# Must be run from project root
if [ ! -f "docker-compose.yml" ]; then
    echo "ERROR: Run this script from the project root directory,"
    echo "       not from inside the deployment folder."
    echo
    echo "Usage:  cd /path/to/CiRA-ME"
    echo "        bash deployment/export.sh"
    exit 1
fi

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "[1/4] Building backend image..."
docker compose build backend
echo

echo "[2/4] Building frontend image..."
docker compose build frontend
echo

echo "[3/4] Saving backend image (this may take a few minutes)..."
docker save cirame-backend:latest -o deployment/cirame-backend.tar
echo "  Saved: deployment/cirame-backend.tar  ($(du -sh deployment/cirame-backend.tar | cut -f1))"

echo "[4/4] Saving frontend image..."
docker save cirame-frontend:latest -o deployment/cirame-frontend.tar
echo "  Saved: deployment/cirame-frontend.tar  ($(du -sh deployment/cirame-frontend.tar | cut -f1))"
echo

echo "============================================"
echo "  Export Complete!"
echo "============================================"
echo
echo "Transfer the entire 'deployment/' folder to the customer server."
echo "On the customer server run:"
echo "  Windows : install.bat"
echo "  Linux   : bash install.sh"
echo
