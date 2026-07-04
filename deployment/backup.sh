#!/bin/bash
# CiRA ME - Backup Script (Linux/macOS)
# Creates a compressed tarball of all customer data so it can be restored
# to any CiRA ME deployment. Safe to run at any time (data is copied, not moved).
#
# Usage:  bash backup.sh                       # default: ./backups/cirame-YYYYMMDD-HHMMSS.tar.gz
#         bash backup.sh /path/to/dest.tar.gz  # explicit destination

set -e

echo "============================================"
echo "  CiRA ME - Backup Script"
echo "============================================"
echo

# Figure out where to write the tarball
if [ -n "$1" ]; then
    DEST="$1"
else
    STAMP=$(date +%Y%m%d-%H%M%S)
    mkdir -p backups
    DEST="backups/cirame-${STAMP}.tar.gz"
fi

# Refuse to overwrite an existing backup — otherwise a bad flag turns a
# safety net into an accident.
if [ -e "$DEST" ]; then
    echo "ERROR: $DEST already exists. Pick a different name or delete it first."
    exit 1
fi

# Verify at least one data folder is present. If neither exists, we are
# almost certainly running from the wrong folder.
if [ ! -d "data" ] && [ ! -d "datasets" ]; then
    echo "ERROR: Neither ./data nor ./datasets exist here."
    echo "       Run this script from the deployment folder that contains"
    echo "       the docker-compose.yml + your data/ and datasets/ folders."
    exit 1
fi

# Optional: stop containers first so the SQLite DB is quiesced. Without
# this, backup may capture a partially-written page. Prompt so the user
# can choose depending on whether they can afford downtime.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^cirame-'; then
    echo "CiRA ME containers are currently running."
    echo "For a consistent backup of the SQLite DB, they should be stopped first."
    echo
    read -p "Stop containers now? (yes/no, default yes): " ans
    if [ "$ans" != "no" ]; then
        docker compose -f docker-compose.yml down 2>/dev/null \
            || docker-compose -f docker-compose.yml down 2>/dev/null \
            || true
        docker compose -f docker-compose-no-gpu.yml down 2>/dev/null \
            || docker-compose -f docker-compose-no-gpu.yml down 2>/dev/null \
            || true
        echo "  Containers stopped."
    else
        echo "  Continuing with containers running — DB may be inconsistent."
    fi
    echo
fi

echo "Backing up to: $DEST"
# --ignore-failed-read: skip files a container may have chmod'd 000
# We include mosquitto/ (config), data/ (DB, models, TI projects), datasets/
# (user uploads), watcher-data/ (Folder Watcher history), and the compose
# files so a restore can bring the same customization back.
tar czf "$DEST" \
    --ignore-failed-read \
    data 2>/dev/null || true

# Append other folders if they exist. `tar --append` (-r) requires a plain
# tar, not gzipped. Easier: re-run tar with all present dirs at once.
sources=""
[ -d "data" ] && sources="$sources data"
[ -d "datasets" ] && sources="$sources datasets"
[ -d "watcher-data" ] && sources="$sources watcher-data"
[ -d "mosquitto" ] && sources="$sources mosquitto"
[ -f "docker-compose.yml" ] && sources="$sources docker-compose.yml"
[ -f "docker-compose-no-gpu.yml" ] && sources="$sources docker-compose-no-gpu.yml"

if [ -z "$sources" ]; then
    echo "ERROR: nothing to back up."
    exit 1
fi

tar czf "$DEST" --ignore-failed-read $sources

# Report size — customer wants to know how big before shipping to another box.
SIZE=$(du -h "$DEST" | awk '{print $1}')

echo
echo "============================================"
echo "  Backup Complete"
echo "============================================"
echo
echo "  File: $DEST"
echo "  Size: $SIZE"
echo
echo "To restore later:  bash restore.sh $DEST"
echo "Or manually:       tar xzf $DEST -C /desired/target/folder"
echo
