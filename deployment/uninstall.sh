#!/bin/bash
# CiRA ME - Uninstall Script (Linux)
# Removes Docker containers, images, and (optionally) named volumes.
# By default, your bind-mounted data on the host disk is PRESERVED.
#
# Usage:  bash uninstall.sh                # remove containers + images only (safe)
#         bash uninstall.sh --purge-data   # ALSO deletes ./data and ./datasets

PURGE_DATA=0
for arg in "$@"; do
    case "$arg" in
        --purge-data|--purge) PURGE_DATA=1 ;;
        -h|--help)
            echo "Usage: bash uninstall.sh [--purge-data]"
            echo "  --purge-data   Also delete ./data and ./datasets (DESTRUCTIVE)"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  CiRA ME - Uninstall Script"
echo "============================================"
echo

# Refuse to run as root — same reasoning as install.sh.
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    echo "ERROR: Do not run this script as root or with sudo."
    exit 1
fi

echo "This will remove:"
echo "  - Running CiRA ME containers"
echo "  - CiRA ME Docker images (backend, frontend, ti-modelmaker)"
echo "  - Named Docker volumes (usually empty — data is on host disk)"
echo
if [ "$PURGE_DATA" -eq 1 ]; then
    echo "This will ALSO permanently delete:"
    echo "  - ./data/          (SQLite database, trained models, TI projects)"
    echo "  - ./datasets/      (user-uploaded CSVs and CBOR files)"
    echo "  - ./watcher-data/  (Folder Watcher input/output history)"
    echo
    echo "  ==> THIS IS UNRECOVERABLE. Consider running 'bash backup.sh' first."
else
    echo "Your data is PRESERVED (this is the default):"
    echo "  ./data/          -- database, models, TI projects (kept)"
    echo "  ./datasets/      -- user-uploaded files (kept)"
    echo "  ./watcher-data/  -- Folder Watcher data (kept)"
    echo
    echo "To also delete your data, re-run with:  bash uninstall.sh --purge-data"
fi
echo

# Typed-phrase confirmation. "yes/no" prompts are answered on autopilot;
# a typed phrase forces the user to slow down.
if [ "$PURGE_DATA" -eq 1 ]; then
    echo "Type PURGE (exact case) to confirm data deletion, or anything else to cancel:"
else
    echo "Type REMOVE (exact case) to confirm, or anything else to cancel:"
fi
read -r confirm
if [ "$PURGE_DATA" -eq 1 ] && [ "$confirm" != "PURGE" ]; then
    echo "Cancelled."
    exit 0
fi
if [ "$PURGE_DATA" -eq 0 ] && [ "$confirm" != "REMOVE" ]; then
    echo "Cancelled."
    exit 0
fi
echo

# Pick whichever compose is available (does not fail if missing).
DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
fi

echo "[1/3] Stopping containers..."
if [ -n "$DOCKER_COMPOSE" ]; then
    $DOCKER_COMPOSE -f docker-compose.yml down 2>/dev/null || true
    $DOCKER_COMPOSE -f docker-compose-no-gpu.yml down 2>/dev/null || true
fi
docker stop cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>/dev/null || true
docker rm cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>/dev/null || true

echo "[2/3] Removing images..."
docker rmi cirame-backend:latest 2>/dev/null || true
docker rmi cirame-frontend:latest 2>/dev/null || true
docker rmi cirame-ti-modelmaker:latest 2>/dev/null || true
docker rmi eclipse-mosquitto:2 2>/dev/null || true

echo "[3/3] Removing named volumes (usually empty — data is bind-mounted)..."
docker volume rm deployment_backend-data 2>/dev/null || true
docker volume rm deployment_backend-models 2>/dev/null || true

if [ "$PURGE_DATA" -eq 1 ]; then
    echo
    echo "[+] Deleting host data folders..."
    rm -rf data datasets watcher-data
    echo "  ./data, ./datasets, ./watcher-data deleted."
fi

echo
echo "============================================"
echo "  Uninstall Complete"
echo "============================================"
echo
if [ "$PURGE_DATA" -eq 1 ]; then
    echo "All CiRA ME data, containers, images, and volumes have been removed."
else
    echo "Containers and images removed. Your data at ./data, ./datasets, and"
    echo "./watcher-data was preserved."
    echo "  To also delete data, re-run with:  bash uninstall.sh --purge-data"
fi
echo
