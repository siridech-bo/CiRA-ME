#!/bin/bash
# CiRA ME - Installation Script (Linux/macOS)
# Loads Docker images from .tar files and sets up the application
# Safe to re-run — stops old version, removes old images, loads new ones

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

echo "Docker is running."
echo

# Stop previous version if running
echo "[1/4] Stopping previous version (if running)..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-no-gpu.yml down 2>/dev/null || true
# Stop individual containers by name (in case compose file changed)
docker stop cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>/dev/null || true
docker rm cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>/dev/null || true
echo "  Previous containers stopped and removed."
echo

# Remove old images to avoid conflicts
echo "[2/4] Removing old images..."
docker rmi cirame-backend:latest 2>/dev/null || true
docker rmi cirame-frontend:latest 2>/dev/null || true

echo "  Old images removed."
echo

# Load new images
echo "[3/4] Loading new images..."

if [ -f "cirame-backend.tar" ]; then
    echo "  Loading backend image (this may take several minutes)..."
    docker load -i cirame-backend.tar
    echo "  Backend loaded."
else
    echo "ERROR: cirame-backend.tar not found!"
    exit 1
fi

if [ -f "cirame-frontend.tar" ]; then
    echo "  Loading frontend image..."
    docker load -i cirame-frontend.tar
    echo "  Frontend loaded."
else
    echo "ERROR: cirame-frontend.tar not found!"
    exit 1
fi

# Optional images
if [ -f "cirame-ti-modelmaker.tar" ]; then
    echo "  Loading TI ModelMaker image (this may take several minutes)..."
    docker load -i cirame-ti-modelmaker.tar
    echo "  TI ModelMaker loaded."
else
    echo "  Skipped: TI ModelMaker (cirame-ti-modelmaker.tar not found)"
fi

if [ -f "cirame-mosquitto.tar" ]; then
    echo "  Loading Mosquitto MQTT broker..."
    docker load -i cirame-mosquitto.tar
    echo "  Mosquitto loaded."
else
    echo "  Skipped: Mosquitto MQTT (cirame-mosquitto.tar not found)"
fi

echo

# Create folders and config
echo "[4/4] Setting up folders and configuration..."
mkdir -p shared
mkdir -p data/database
mkdir -p data/models
mkdir -p data/ti-projects
mkdir -p data/mosquitto
mkdir -p mosquitto

if [ ! -f "mosquitto/mosquitto.conf" ]; then
    cat > mosquitto/mosquitto.conf << 'MQTTCONF'
listener 1883
protocol mqtt
listener 9001
protocol websockets
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest stdout
log_type warning
log_type error
MQTTCONF
    echo "  Created mosquitto/mosquitto.conf"
fi

echo "  Folders ready."
echo

echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo
echo "Installed images:"
docker images | grep -E "cirame|mosquitto" | head -10
echo
echo "Next steps:"
echo "  With GPU    : bash start.sh"
echo "  Without GPU : bash start-no-gpu.sh"
echo "  Access at   : http://localhost:3030"
echo "  Login       : admin / admin123"
echo
echo "NOTE: Your data (models, database, datasets) is preserved"
echo "      in Docker volumes across updates."
echo
