#!/bin/bash
# CiRA ME - Export Script (Linux/macOS)
# Saves currently running Docker images as .tar files for customer deployment
# Does NOT rebuild — exports exactly what is running now

echo "============================================"
echo "  CiRA ME - Export Docker Images"
echo "============================================"
echo

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "Saving to: $DEPLOY_DIR"
echo

# Save required images
echo "[1/4] Saving backend image (this may take a few minutes)..."
if ! docker save cirame-backend:latest -o "$DEPLOY_DIR/cirame-backend.tar"; then
    echo "  ERROR: Failed to save backend image. Is it built?"
    exit 1
fi
echo "  Saved: cirame-backend.tar  ($(du -sh "$DEPLOY_DIR/cirame-backend.tar" | cut -f1))"
echo

echo "[2/4] Saving frontend image..."
if ! docker save cirame-frontend:latest -o "$DEPLOY_DIR/cirame-frontend.tar"; then
    echo "  ERROR: Failed to save frontend image. Is it built?"
    exit 1
fi
echo "  Saved: cirame-frontend.tar  ($(du -sh "$DEPLOY_DIR/cirame-frontend.tar" | cut -f1))"
echo

# Save optional images
echo "[3/4] Saving TI ModelMaker image..."
docker save cirame-ti-modelmaker:latest -o "$DEPLOY_DIR/cirame-ti-modelmaker.tar" 2>/dev/null && \
  echo "  Saved: cirame-ti-modelmaker.tar  ($(du -sh "$DEPLOY_DIR/cirame-ti-modelmaker.tar" | cut -f1))" || \
  echo "  Skipped: cirame-ti-modelmaker not available"
echo

echo "[4/4] Saving Mosquitto MQTT broker..."
docker save eclipse-mosquitto:2 -o "$DEPLOY_DIR/cirame-mosquitto.tar" 2>/dev/null && \
  echo "  Saved: cirame-mosquitto.tar  ($(du -sh "$DEPLOY_DIR/cirame-mosquitto.tar" | cut -f1))" || \
  echo "  Skipped: mosquitto not available"
echo

echo "============================================"
echo "  Export Complete!"
echo "============================================"
echo
echo "Packages:"
echo "  Required:  cirame-backend.tar + cirame-frontend.tar"
echo "  Optional:  cirame-ti-modelmaker.tar (TI MCU support)"
echo "  Optional:  cirame-mosquitto.tar (MQTT live streaming)"
echo
echo "Transfer the entire 'deployment/' folder to the customer server."
echo "On the customer server run:"
echo "  Windows : install.bat"
echo "  Linux   : bash install.sh"
echo
