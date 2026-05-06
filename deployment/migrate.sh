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

echo "Source (old):  $OLD_DIR"
echo "Target (this): $(pwd)"
echo

# Verify source has at least one expected folder
if [ ! -d "$OLD_DIR/data" ] && [ ! -d "$OLD_DIR/datasets" ] && [ ! -d "$OLD_DIR/shared" ]; then
    echo "ERROR: Source does not look like a CiRA ME deployment folder."
    echo "Expected at least one of: data/, datasets/, shared/"
    exit 1
fi

# Stop running containers so files aren't locked
echo "Stopping running containers (if any)..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-no-gpu.yml down 2>/dev/null || true
echo

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

# Database & models & TI projects & mosquitto
echo "[1/3] Migrating data/ folder (database, models, ti-projects, mosquitto)..."
if [ -d "$OLD_DIR/data" ]; then
    mkdir -p data
    cp -aR "$OLD_DIR/data/." "data/" && \
      echo "  data/ copied successfully." || \
      echo "  WARNING: Some files in data/ could not be copied."
else
    echo "  Skipped: $OLD_DIR/data not found."
fi
echo

# Datasets (new layout)
echo "[2/3] Migrating datasets/ folder (user uploads & shared)..."
if [ -d "$OLD_DIR/datasets" ]; then
    mkdir -p datasets
    cp -aR "$OLD_DIR/datasets/." "datasets/" && \
      echo "  datasets/ copied successfully." || \
      echo "  WARNING: Some files in datasets/ could not be copied."
else
    echo "  Skipped: $OLD_DIR/datasets not found (may use legacy shared/ layout)."
fi
echo

# Legacy shared (old layout) -> datasets/shared (new layout)
echo "[3/3] Migrating legacy shared/ folder (if present)..."
if [ -d "$OLD_DIR/shared" ]; then
    mkdir -p datasets/shared
    cp -aR "$OLD_DIR/shared/." "datasets/shared/" && \
      echo "  Legacy shared/ copied to datasets/shared/." || \
      echo "  WARNING: Some files in shared/ could not be copied."
else
    echo "  Skipped: $OLD_DIR/shared not found."
fi
echo

# Mosquitto config
if [ -f "$OLD_DIR/mosquitto/mosquitto.conf" ] && [ ! -f "mosquitto/mosquitto.conf" ]; then
    mkdir -p mosquitto
    cp "$OLD_DIR/mosquitto/mosquitto.conf" "mosquitto/mosquitto.conf"
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
