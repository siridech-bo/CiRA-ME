#!/bin/bash
# CiRA ME - Status Script
# Shows the status of running containers

echo "============================================"
echo "  CiRA ME - Application Status"
echo "============================================"
echo

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running."
    exit 1
fi

echo "Running Containers:"
echo "-------------------"
docker ps --filter "name=cirame" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

echo "Docker Images:"
echo "--------------"
docker images --filter "reference=cirame*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker images --filter "reference=eclipse-mosquitto" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null
echo

echo "Volume Usage:"
echo "-------------"
docker volume ls --filter "name=deployment" --format "table {{.Name}}"
echo

echo "Disk Usage:"
echo "-----------"
if [ -d "./data" ]; then
    du -sh ./data/*/ 2>/dev/null
fi
if [ -d "./shared" ]; then
    du -sh ./shared/ 2>/dev/null
fi
echo
