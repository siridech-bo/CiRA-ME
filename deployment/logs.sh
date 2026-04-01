#!/bin/bash
# CiRA ME - View Logs Script
# Shows live logs from all containers
# Press Ctrl+C to exit

echo "============================================"
echo "  CiRA ME - Application Logs"
echo "  Press Ctrl+C to exit"
echo "============================================"
echo

docker compose -f docker-compose.yml logs -f 2>/dev/null || docker compose -f docker-compose-no-gpu.yml logs -f
