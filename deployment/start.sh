#!/bin/bash
# CiRA ME - Start Script with GPU support (Linux)
# Starts available services — skips optional ones if image not installed

echo "============================================"
echo "  CiRA ME - Starting Application (GPU)"
echo "============================================"
echo

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running."
    exit 1
fi

if ! docker images | grep -q cirame-backend; then
    echo "ERROR: Docker images not found. Run 'bash install.sh' first."
    exit 1
fi

echo "Starting core services..."
if ! docker compose -f docker-compose.yml up -d backend frontend; then
    echo "ERROR: Failed to start core services."
    echo "Check logs: docker compose -f docker-compose.yml logs"
    exit 1
fi

# Start optional services if images are available
if docker images | grep -q cirame-ti-modelmaker; then
    echo "Starting TI ModelMaker..."
    docker compose -f docker-compose.yml up -d ti-modelmaker
else
    echo "Skipped: TI ModelMaker (image not installed)"
fi

if docker images | grep -q mosquitto; then
    echo "Starting MQTT Broker..."
    docker compose -f docker-compose.yml up -d mosquitto
else
    echo "Skipped: MQTT Broker (image not installed)"
fi

echo
echo "============================================"
echo "  Application Started!"
echo "============================================"
echo
echo "Services running:"
docker compose -f docker-compose.yml ps 2>/dev/null
echo
echo "  Access at : http://localhost:3030"
echo "  Login     : admin / admin123"
echo
echo "  Logs   : docker compose logs -f"
echo "  Stop   : bash stop.sh"
echo
