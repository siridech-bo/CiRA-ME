#!/bin/bash
# CiRA ME - Installation Script (Linux/macOS)
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

echo "Docker is running."
echo

# Load required images
echo "[1/2] Loading required images..."

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

echo

# Load optional images
echo "[2/2] Loading optional images..."

if [ -f "cirame-ti-modelmaker.tar" ]; then
    echo "  Loading TI ModelMaker image (this may take several minutes)..."
    docker load -i cirame-ti-modelmaker.tar
    echo "  TI ModelMaker loaded."
else
    echo "  Skipped: cirame-ti-modelmaker.tar not found (TI MCU features disabled)"
fi

if [ -f "cirame-mosquitto.tar" ]; then
    echo "  Loading Mosquitto MQTT broker..."
    docker load -i cirame-mosquitto.tar
    echo "  Mosquitto loaded."
else
    echo "  Skipped: cirame-mosquitto.tar not found (MQTT live streaming disabled)"
fi

echo

# Create folders
mkdir -p shared
mkdir -p mosquitto

# Copy mosquitto config if not exists
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
    echo "Created mosquitto/mosquitto.conf"
fi

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
echo "NOTE: Change the admin password after first login!"
echo
