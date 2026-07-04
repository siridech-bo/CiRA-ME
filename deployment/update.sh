#!/bin/bash
# CiRA ME - Update Script (Linux)
# Loads new image versions while preserving all customer data.
# Automatically snapshots the SQLite database before touching anything so a
# broken new image can be rolled back cleanly.
#
# Run from the deployment folder after copying new .tar files here.

echo "============================================"
echo "  CiRA ME - Update Script"
echo "============================================"
echo
echo "This will:"
echo "  1. Snapshot ./data/database/ to a timestamped backup folder"
echo "  2. Stop the running application (data on host disk is preserved)"
echo "  3. Load the new Docker images from .tar files"
echo "  4. Clean up old image layers"
echo
echo "IMPORTANT: Run this script in the SAME folder as your existing"
echo "           installation. Your data in ./data/ and ./datasets/"
echo "           is automatically preserved (bind-mounted)."
echo
echo "If you extracted a new release to a DIFFERENT folder,"
echo "run 'bash migrate.sh <old_folder>' first to copy your data over."
echo
read -p "Continue with update? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Update cancelled."
    exit 0
fi
echo

# ─── Pre-flight ─────────────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running or not reachable."
    exit 1
fi

DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "ERROR: docker compose is not installed. Re-run install.sh first."
    exit 1
fi

# Verify the new tar files exist AND are non-empty. A truncated download
# would otherwise take down the running app and leave nothing to boot with.
for tar in cirame-backend.tar cirame-frontend.tar; do
    if [ ! -f "$tar" ]; then
        echo "ERROR: $tar not found. Copy the new .tar files here before running update."
        exit 1
    fi
    if [ ! -s "$tar" ]; then
        echo "ERROR: $tar is 0 bytes. Re-download and try again."
        exit 1
    fi
done

# ─── [1/5] Snapshot the DB before we touch anything ─────────────────
echo "[1/5] Snapshotting database..."
if [ -d "data/database" ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    BACKUP_DIR="data/database.backup.${STAMP}"
    cp -aR data/database "$BACKUP_DIR" || {
        echo "ERROR: failed to snapshot data/database. Aborting update."
        echo "       Check disk space and permissions on ./data/"
        exit 1
    }
    echo "  Snapshot: $BACKUP_DIR"
    echo "  To roll back later:"
    echo "     ./stop.sh && rm -rf data/database && mv $BACKUP_DIR data/database"
else
    echo "  No existing data/database/ — first-time run? Skipping snapshot."
fi
echo

# ─── [2/5] Stop containers ─────────────────────────────────────────
echo "[2/5] Stopping current application (data on disk is preserved)..."
$DOCKER_COMPOSE -f docker-compose.yml down 2>/dev/null || true
$DOCKER_COMPOSE -f docker-compose-no-gpu.yml down 2>/dev/null || true

# Legacy shared/ -> datasets/shared/ migration for v1.0 upgrades.
if [ -d "shared" ] && [ ! -d "datasets/shared" ]; then
    echo "  Migrating legacy shared/ -> datasets/shared/..."
    mkdir -p datasets
    mv shared datasets/shared
    echo "  Migration complete."
fi
echo "  Stopped."
echo

# ─── [3/5] Load new backend + verify ───────────────────────────────
load_and_verify() {
    local tar="$1" tag="$2" label="$3"
    echo "  Loading $label image..."
    docker load -i "$tar"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo "ERROR: $tar loaded without error but $tag is not present."
        echo "       Tarball is likely truncated or corrupt. Re-download."
        echo "       Your DB snapshot at $BACKUP_DIR is intact — you can"
        echo "       restart the OLD version by re-loading its tarballs."
        exit 1
    fi
    echo "  $label loaded ($tag)."
}

echo "[3/5] Loading new backend image..."
load_and_verify cirame-backend.tar cirame-backend:latest "backend"
echo

echo "[4/5] Loading new frontend image..."
load_and_verify cirame-frontend.tar cirame-frontend:latest "frontend"
echo

# Load optional images if present
if [ -f "cirame-ti-modelmaker.tar" ]; then
    load_and_verify cirame-ti-modelmaker.tar cirame-ti-modelmaker:latest "TI ModelMaker"
    echo
fi
if [ -f "cirame-mosquitto.tar" ]; then
    load_and_verify cirame-mosquitto.tar eclipse-mosquitto:2 "Mosquitto MQTT"
    echo
fi

# ─── [5/5] Cleanup ─────────────────────────────────────────────────
echo "[5/5] Cleaning up old image layers..."
docker image prune -f >/dev/null 2>&1
echo "  Done."
echo

echo "============================================"
echo "  Update Complete!"
echo "============================================"
echo
if [ -n "${BACKUP_DIR:-}" ]; then
    echo "  DB snapshot: $BACKUP_DIR (safe to delete once new version is verified)"
fi
echo
echo "Restart the application:"
echo "  With GPU    : bash start.sh"
echo "  Without GPU : bash start-no-gpu.sh"
echo
echo "If the new version fails to start, roll back with:"
echo "  bash stop.sh"
if [ -n "${BACKUP_DIR:-}" ]; then
    echo "  rm -rf data/database && mv $BACKUP_DIR data/database"
fi
echo
