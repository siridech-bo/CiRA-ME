#!/bin/bash
# CiRA ME - Start Script without GPU (Linux)
# Starts available services — skips optional ones if image not installed

echo "============================================"
echo "  CiRA ME - Starting Application (CPU Only)"
echo "============================================"
echo

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running or not reachable."
    echo "       Either the daemon is stopped or your user cannot access it."
    echo
    echo "If the daemon is stopped:  sudo systemctl start docker"
    echo "If 'permission denied':    sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# docker compose v2 (plugin) or v1 (standalone)
DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "ERROR: docker compose is not installed. Re-run install.sh."
    exit 1
fi

if ! docker images | grep -q cirame-backend; then
    echo "ERROR: Docker images not found. Run 'bash install.sh' first."
    exit 1
fi

# Port collision check
port_owner() {
    if command -v ss >/dev/null 2>&1; then
        ss -tulnp 2>/dev/null | awk -v p=":$1 " '$0 ~ p'
    else
        netstat -tulnp 2>/dev/null | awk -v p=":$1 " '$0 ~ p'
    fi
}
port_conflict=""
for port in 3030 5100 5200 1883 9001; do
    if [ -n "$(port_owner "$port")" ]; then
        port_conflict="$port_conflict $port"
    fi
done
if [ -n "$port_conflict" ]; then
    echo "ERROR: Ports already in use:$port_conflict"
    echo
    echo "Common culprits:"
    echo "  1883  — host mosquitto service. Stop:  sudo systemctl stop mosquitto"
    echo "  3030  — another web app. Change our port: edit docker-compose-no-gpu.yml"
    echo "  5100  — another Flask app."
    exit 1
fi

echo "Starting core services (CPU mode)..."
if ! $DOCKER_COMPOSE -f docker-compose-no-gpu.yml up -d backend frontend; then
    echo "ERROR: Failed to start core services."
    echo "Check logs: $DOCKER_COMPOSE -f docker-compose-no-gpu.yml logs"
    exit 1
fi

if docker images | grep -q cirame-ti-modelmaker; then
    echo "Starting TI ModelMaker..."
    $DOCKER_COMPOSE -f docker-compose-no-gpu.yml up -d ti-modelmaker
else
    echo "Skipped: TI ModelMaker (image not installed)"
fi

if docker images | grep -q mosquitto; then
    echo "Starting MQTT Broker..."
    $DOCKER_COMPOSE -f docker-compose-no-gpu.yml up -d mosquitto
else
    echo "Skipped: MQTT Broker (image not installed)"
fi

echo
echo "============================================"
echo "  Application Started (CPU Only)!"
echo "============================================"
echo
echo "Services running:"
$DOCKER_COMPOSE -f docker-compose-no-gpu.yml ps 2>/dev/null
echo
echo "  Access at : http://localhost:3030"
echo "  Login     : admin / admin123"
echo
echo "  Logs   : $DOCKER_COMPOSE logs -f"
echo "  Stop   : bash stop.sh"
echo
