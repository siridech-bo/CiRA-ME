#!/bin/bash
# CiRA ME - Stop Script (Linux)

echo "============================================"
echo "  CiRA ME - Stopping Application"
echo "============================================"
echo

echo "Stopping services..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-no-gpu.yml down 2>/dev/null || true

echo
echo "Application stopped."
echo "To restart: bash start.sh  or  bash start-no-gpu.sh"
echo
