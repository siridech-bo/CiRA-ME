#!/bin/bash
# CiRA ME - Start Script without GPU (Linux)

echo "============================================"
echo "  CiRA ME - Starting Application (No GPU)"
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

echo "Starting CiRA ME services (CPU only)..."
docker compose -f docker-compose-no-gpu.yml up -d

echo
echo "============================================"
echo "  Application Started!"
echo "============================================"
echo
echo "  Access at : http://localhost:3030"
echo "  Login     : admin / admin123"
echo
echo "  Logs   : docker compose -f docker-compose-no-gpu.yml logs -f"
echo "  Stop   : bash stop.sh"
echo
