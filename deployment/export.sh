#!/bin/bash
# CiRA ME - Build and Export Script (Linux/macOS)
# Run from anywhere — the script finds the project root automatically.

set -e

echo "============================================"
echo "  CiRA ME - Build and Export Images"
echo "============================================"
echo

# This script lives in deployment/; project root is one level up.
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Verify project root
if [ ! -d "$PROJECT_ROOT/backend" ]; then
    echo "ERROR: Cannot locate project root."
    echo "Expected backend/ directory at: $PROJECT_ROOT"
    exit 1
fi

echo "Building from: $PROJECT_ROOT"
echo "Saving to:     $DEPLOY_DIR"
echo

echo "[1/4] Building backend image..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" build backend
echo

echo "[2/4] Building frontend image..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" build frontend
echo

echo "[3/4] Saving backend image (this may take a few minutes)..."
docker save cirame-backend:latest -o "$DEPLOY_DIR/cirame-backend.tar"
echo "  Saved: $DEPLOY_DIR/cirame-backend.tar  ($(du -sh "$DEPLOY_DIR/cirame-backend.tar" | cut -f1))"

echo "[4/4] Saving frontend image..."
docker save cirame-frontend:latest -o "$DEPLOY_DIR/cirame-frontend.tar"
echo "  Saved: $DEPLOY_DIR/cirame-frontend.tar  ($(du -sh "$DEPLOY_DIR/cirame-frontend.tar" | cut -f1))"
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
