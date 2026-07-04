#!/bin/bash
# CiRA ME - Migrate Data Script (Linux/macOS)
# Copies user data from an old deployment folder to this folder.
#
# Usage:  bash migrate.sh /path/to/old/deployment
#
# What is migrated:
#   - data/database/         (SQLite database: users, models, endpoints, apps)
#   - data/models/           (Trained model files)
#   - data/ti-projects/      (TI ModelMaker projects)
#   - data/mosquitto/        (MQTT broker persistent data)
#   - datasets/              (User-uploaded datasets, including shared/)
#   - watcher-data/          (Folder Watcher input/output history)
#   - shared/                (Legacy: auto-migrated to datasets/shared/)

echo "============================================"
echo "  CiRA ME - Data Migration Script"
echo "============================================"
echo

if [ -z "$1" ]; then
    echo "ERROR: Please provide the path to your old deployment folder."
    echo
    echo "Usage:  bash migrate.sh /path/to/old/CiRA-ME-deployment"
    echo
    echo "Example:"
    echo "  bash migrate.sh ~/cirame-v1.0/deployment"
    echo
    exit 1
fi

OLD_DIR="$1"

if [ ! -d "$OLD_DIR" ]; then
    echo "ERROR: Old folder does not exist: $OLD_DIR"
    exit 1
fi

# Refuse to migrate from the same folder into itself. cp -aR src/ .
# where src == cwd would copy every file back over itself — at best a
# no-op, at worst a partial mangle if any file changes mid-copy.
OLD_ABS="$(cd "$OLD_DIR" && pwd)"
NEW_ABS="$(pwd)"
if [ "$OLD_ABS" = "$NEW_ABS" ]; then
    echo "ERROR: Source and target are the same folder ($OLD_ABS)."
    echo "       migrate.sh copies from another install, not into itself."
    exit 1
fi

echo "Source (old):  $OLD_ABS"
echo "Target (this): $NEW_ABS"
echo

# Verify source has at least one expected folder
if [ ! -d "$OLD_DIR/data" ] && [ ! -d "$OLD_DIR/datasets" ] && \
   [ ! -d "$OLD_DIR/shared" ] && [ ! -d "$OLD_DIR/watcher-data" ]; then
    echo "ERROR: Source does not look like a CiRA ME deployment folder."
    echo "Expected at least one of: data/, datasets/, shared/, watcher-data/"
    exit 1
fi

# Stop running containers so files aren't locked / SQLite isn't mid-write
DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
fi
if [ -n "$DOCKER_COMPOSE" ]; then
    echo "Stopping running containers (if any)..."
    $DOCKER_COMPOSE -f docker-compose.yml down 2>/dev/null || true
    $DOCKER_COMPOSE -f docker-compose-no-gpu.yml down 2>/dev/null || true
    echo
fi

# Confirm
echo "This will COPY user data from the old folder into this folder."
echo "Existing files in this folder will be OVERWRITTEN."
echo
read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Migration cancelled."
    exit 0
fi
echo

# ─── Safe-copy helper ───────────────────────────────────────────────
# Fatal on failure. Copying half a database and continuing is exactly
# how customers end up with unbootable installs. Any error here aborts
# the migration so the customer knows to intervene before the new
# version starts.
safe_copy() {
    local src="$1" dst="$2" label="$3"
    if [ ! -d "$src" ]; then
        echo "  Skipped: $src not found."
        return 0
    fi
    mkdir -p "$dst" || {
        echo "ERROR: cannot create $dst"
        exit 1
    }
    if cp -aR "$src/." "$dst/"; then
        echo "  $label copied successfully."
    else
        echo "ERROR: failed to copy $label ($src -> $dst)."
        echo "       Migration aborted. Fix disk space / permissions and re-run."
        echo "       Partial copy at $dst may be inconsistent — delete before retrying."
        exit 1
    fi
}

echo "[1/4] Migrating data/ folder (database, models, ti-projects, mosquitto)..."
safe_copy "$OLD_DIR/data" "data" "data/"
echo

echo "[2/4] Migrating datasets/ folder (user uploads & shared)..."
safe_copy "$OLD_DIR/datasets" "datasets" "datasets/"
echo

echo "[3/4] Migrating watcher-data/ folder (Folder Watcher history)..."
safe_copy "$OLD_DIR/watcher-data" "watcher-data" "watcher-data/"
echo

echo "[4/4] Migrating legacy shared/ folder (if present)..."
if [ -d "$OLD_DIR/shared" ]; then
    safe_copy "$OLD_DIR/shared" "datasets/shared" "Legacy shared/ (-> datasets/shared/)"
else
    echo "  Skipped: $OLD_DIR/shared not found."
fi
echo

# Mosquitto config
if [ -f "$OLD_DIR/mosquitto/mosquitto.conf" ] && [ ! -f "mosquitto/mosquitto.conf" ]; then
    mkdir -p mosquitto
    cp "$OLD_DIR/mosquitto/mosquitto.conf" "mosquitto/mosquitto.conf" || {
        echo "ERROR: failed to copy mosquitto.conf"
        exit 1
    }
    echo "  Copied mosquitto/mosquitto.conf"
fi

echo "============================================"
echo "  Migration Complete!"
echo "============================================"
echo
echo "Next steps:"
echo "  1. Run 'bash install.sh' if you have not loaded the new images yet"
echo "     (or 'bash update.sh' if images are already installed)"
echo "  2. Run 'bash start.sh' (or 'bash start-no-gpu.sh') to launch CiRA ME"
echo "  3. Verify your data appears at http://localhost:3030"
echo
