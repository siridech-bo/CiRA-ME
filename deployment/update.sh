#!/bin/bash
# CiRA ME - Update Script (Linux)
# Loads new image versions while preserving all customer data
#
# Run from the deployment folder after copying new .tar files here.

echo "============================================"
echo "  CiRA ME - Update Script"
echo "============================================"
echo
echo "This will:"
echo "  1. Stop the running application"
echo "  2. Load the new Docker images"
echo "  3. Clean up old image layers"
echo
echo "Your data (database, models, datasets) will be preserved."
echo
read -p "Continue with update? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Update cancelled."
    exit 0
fi
echo

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running."
    exit 1
fi

# Check new tar files exist
if [ ! -f "cirame-backend.tar" ]; then
    echo "ERROR: cirame-backend.tar not found."
    echo "Copy the new .tar files here before running update."
    exit 1
fi
if [ ! -f "cirame-frontend.tar" ]; then
    echo "ERROR: cirame-frontend.tar not found."
    echo "Copy the new .tar files here before running update."
    exit 1
fi

echo "[1/4] Stopping current application (data is preserved)..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-no-gpu.yml down 2>/dev/null || true
echo "Done."
echo

echo "[2/4] Loading new backend image..."
docker load -i cirame-backend.tar
echo

echo "[3/4] Loading new frontend image..."
docker load -i cirame-frontend.tar
echo

# Load optional images if present
if [ -f "cirame-ti-modelmaker.tar" ]; then
    echo "Loading TI ModelMaker image..."
    docker load -i cirame-ti-modelmaker.tar
    echo
else
    echo "Skipped: cirame-ti-modelmaker.tar not found (optional)"
fi

if [ -f "cirame-mosquitto.tar" ]; then
    echo "Loading Mosquitto MQTT broker..."
    docker load -i cirame-mosquitto.tar
    echo
else
    echo "Skipped: cirame-mosquitto.tar not found (optional)"
fi
echo

echo "[4/4] Cleaning up old image layers..."
docker image prune -f >/dev/null 2>&1
echo "Done."
echo

echo "============================================"
echo "  Update Complete!"
echo "============================================"
echo
echo "Restart the application:"
echo "  With GPU    : bash start.sh"
echo "  Without GPU : bash start-no-gpu.sh"
echo
