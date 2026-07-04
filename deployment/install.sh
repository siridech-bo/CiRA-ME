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

# ─── Pre-flight checks ──────────────────────────────────────────────
# Fail fast with actionable messages instead of letting docker fail cryptically
# later. Each check maps to a real field failure we've seen.

# 1. Don't run as root. sudo creates root-owned folders that regular users
#    can't touch, breaking every subsequent invocation.
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    echo "ERROR: Do not run this script as root or with sudo."
    echo "       Root-owned files break subsequent runs."
    echo
    echo "Fix: add your user to the docker group, then re-run as regular user:"
    echo "     sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# 2. Docker installed + daemon reachable.
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed."
    echo "Install: https://docs.docker.com/engine/install/"
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is installed but the daemon is not reachable."
    echo "       Either Docker isn't running, or your user cannot access it."
    echo
    echo "If the daemon is stopped:  sudo systemctl start docker"
    echo "If you get 'permission denied' errors:"
    echo "     sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# 3. docker compose v2 (plugin) or v1 (docker-compose). Pick whichever is
#    available. Ubuntu focal's docker.io package ships neither by default.
DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "ERROR: docker compose is not installed."
    echo "       Docker itself is running, but neither 'docker compose' (v2 plugin)"
    echo "       nor 'docker-compose' (v1 standalone) is on the PATH."
    echo
    echo "Fix (Ubuntu / Debian, recommended):"
    echo "     sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \\"
    echo "         -o /usr/local/lib/docker/cli-plugins/docker-compose \\"
    echo "     && sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose"
    exit 1
fi
echo "  docker compose: $DOCKER_COMPOSE"

# 4. Free disk space — need ~25 GB unpacked for the 4 images.
FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
if [ "$FREE_GB" -lt 30 ]; then
    echo "ERROR: Only ${FREE_GB} GB free at $(pwd). Need at least 30 GB."
    echo "       docker load will fail partway through with a cryptic 'write error'."
    exit 1
fi
echo "  Free space: ${FREE_GB} GB — ok."

# 5. Verify tarballs are present and non-empty BEFORE we start tearing down
#    the previous install. A missing tarball halfway through leaves the box
#    in a broken state.
missing_required=""
for tar in cirame-backend.tar cirame-frontend.tar; do
    if [ ! -f "$tar" ]; then
        missing_required="$missing_required $tar"
    elif [ ! -s "$tar" ]; then
        echo "ERROR: $tar is empty (0 bytes). Re-download it."
        exit 1
    fi
done
if [ -n "$missing_required" ]; then
    echo "ERROR: Required tarballs missing:$missing_required"
    echo "       Make sure you extracted the full release ZIP and are running"
    echo "       install.sh from the deployment/ folder."
    exit 1
fi

echo "Docker is running."
echo

# Stop previous version if running
echo "[1/4] Stopping previous version (if running)..."
$DOCKER_COMPOSE -f docker-compose.yml down 2>/dev/null || true
$DOCKER_COMPOSE -f docker-compose-no-gpu.yml down 2>/dev/null || true
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

# Load new images — verify each one actually appeared as a tagged image.
# docker load can silently no-op on some kinds of corrupt tarballs; without
# the verify step you'd only find out at start time with "image not found".
echo "[3/4] Loading new images..."

load_and_verify() {
    # $1 = tarball path, $2 = expected repo:tag, $3 = human label
    local tar="$1" tag="$2" label="$3"
    if [ ! -f "$tar" ]; then
        return 1  # caller decides if this is fatal
    fi
    echo "  Loading $label image..."
    docker load -i "$tar"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo "ERROR: $tar loaded without error but $tag is not present."
        echo "       Tarball is likely truncated or malformed. Re-download it."
        exit 1
    fi
    echo "  $label loaded ($tag)."
}

load_and_verify cirame-backend.tar cirame-backend:latest "backend" || {
    echo "ERROR: cirame-backend.tar missing"; exit 1;
}
load_and_verify cirame-frontend.tar cirame-frontend:latest "frontend" || {
    echo "ERROR: cirame-frontend.tar missing"; exit 1;
}
# Optional
if [ -f cirame-ti-modelmaker.tar ]; then
    load_and_verify cirame-ti-modelmaker.tar cirame-ti-modelmaker:latest "TI ModelMaker"
else
    echo "  Skipped: TI ModelMaker (cirame-ti-modelmaker.tar not found)"
fi
if [ -f cirame-mosquitto.tar ]; then
    load_and_verify cirame-mosquitto.tar eclipse-mosquitto:2 "Mosquitto MQTT"
else
    echo "  Skipped: Mosquitto MQTT (cirame-mosquitto.tar not found)"
fi

echo

# Create folders and config
echo "[4/4] Setting up folders and configuration..."

# Migration: legacy ./shared/ -> ./datasets/shared/ (for upgrades from old layout)
if [ -d "shared" ] && [ ! -d "datasets/shared" ]; then
    echo "  Migrating legacy shared/ folder to datasets/shared/..."
    mkdir -p datasets
    mv shared datasets/shared
    echo "  Migration complete: ./shared/ moved to ./datasets/shared/"
fi

mkdir -p datasets/shared
mkdir -p data/database
mkdir -p data/models
mkdir -p data/ti-projects
mkdir -p data/mosquitto
mkdir -p mosquitto
# Folder Watcher input/output. If Docker creates this on first `up`, it
# owns the folder as root — subsequent writes from within the container
# hit "Permission denied". Pre-create as the invoking user so ownership
# matches the container's UID 1000 mapping.
mkdir -p watcher-data

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
if command -v nvidia-smi >/dev/null 2>&1; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    driver_major=$(echo "$driver" | cut -d. -f1)
    echo "  GPU detected (driver $driver) — run: bash start.sh"
    # The stock backend tarball ships torch+cu128 which needs driver 550+.
    # Warn upfront so the customer does not start it, hit a CUDA crash, and
    # then try to debug backwards.
    if [ -n "$driver_major" ] && [ "$driver_major" -lt 550 ] 2>/dev/null; then
        echo "  WARNING: driver $driver is older than 550. The shipped backend"
        echo "           (built for CUDA 12.8) will crash on GPU calls."
        echo "           Options:"
        echo "             (a) update NVIDIA driver to 550+ (best for GPU)"
        echo "             (b) rebuild backend with cu117:"
        echo "                 docker build --build-arg TORCH_VERSION=2.0.1 \\"
        echo "                              --build-arg TORCH_CUDA=cu117 \\"
        echo "                              -t cirame-backend:latest backend/"
        echo "             (c) run in CPU-only mode: bash start-no-gpu.sh"
    fi
else
    echo "  No GPU detected — run: bash start-no-gpu.sh"
fi
echo "  (Force CPU mode any time: bash start-no-gpu.sh)"
echo "  Access at   : http://localhost:3030"
echo "  Login       : admin / admin123"
echo
echo "NOTE: Your data (models, database, datasets, watcher-data)"
echo "      is preserved on the host disk across updates."
echo
