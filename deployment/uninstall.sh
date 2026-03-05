#!/bin/bash
# CiRA ME - Uninstall Script (Linux)
# Removes all CiRA ME containers, images, and volumes

echo "============================================"
echo "  CiRA ME - Uninstall Script"
echo "============================================"
echo
echo "WARNING: This will remove all CiRA ME data including:"
echo "  - Docker containers"
echo "  - Docker images"
echo "  - Data volumes (database and uploaded files)"
echo
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Uninstall cancelled."
    exit 0
fi

echo
echo "[1/3] Stopping containers..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-no-gpu.yml down 2>/dev/null || true

echo "[2/3] Removing images..."
docker rmi cirame-backend:latest 2>/dev/null || true
docker rmi cirame-frontend:latest 2>/dev/null || true

echo "[3/3] Removing volumes..."
docker volume rm deployment_backend-data 2>/dev/null || true
docker volume rm deployment_backend-models 2>/dev/null || true

echo
echo "============================================"
echo "  Uninstall Complete"
echo "============================================"
echo
echo "Note: The 'shared' folder was not deleted."
echo "Remove it manually if needed: rm -rf shared/"
echo
