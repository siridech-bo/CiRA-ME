@echo off
REM CiRA ME - Installation Script for Windows
REM Loads Docker images from .tar files and sets up the application
REM Safe to re-run — stops old version, removes old images, loads new ones

echo ============================================
echo   CiRA ME - Installation Script
echo   Machine Intelligence for Edge Computing
echo ============================================
echo.

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed.
    echo Please install Docker Desktop and start it before running this script.
    pause
    exit /b 1
)

echo Docker is running.
echo.

REM Stop previous version if running
echo [1/4] Stopping previous version (if running)...
docker compose -f docker-compose.yml down 2>nul
docker compose -f docker-compose-no-gpu.yml down 2>nul
docker stop cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>nul
docker rm cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>nul
echo   Previous containers stopped and removed.
echo.

REM Remove old images to avoid conflicts
echo [2/4] Removing old images...
docker rmi cirame-backend:latest 2>nul
docker rmi cirame-frontend:latest 2>nul

echo   Old images removed.
echo.

REM Load new images
echo [3/4] Loading new images...

if exist "cirame-backend.tar" (
    echo   Loading backend image (this may take several minutes)...
    docker load -i cirame-backend.tar
    if %errorlevel% neq 0 (
        echo ERROR: Failed to load backend image.
        pause
        exit /b 1
    )
    echo   Backend loaded.
) else (
    echo ERROR: cirame-backend.tar not found!
    pause
    exit /b 1
)

if exist "cirame-frontend.tar" (
    echo   Loading frontend image...
    docker load -i cirame-frontend.tar
    if %errorlevel% neq 0 (
        echo ERROR: Failed to load frontend image.
        pause
        exit /b 1
    )
    echo   Frontend loaded.
) else (
    echo ERROR: cirame-frontend.tar not found!
    pause
    exit /b 1
)

REM Optional images
if exist "cirame-ti-modelmaker.tar" (
    echo   Loading TI ModelMaker image (this may take several minutes)...
    docker load -i cirame-ti-modelmaker.tar
    echo   TI ModelMaker loaded.
) else (
    echo   Skipped: TI ModelMaker (cirame-ti-modelmaker.tar not found)
)

if exist "cirame-mosquitto.tar" (
    echo   Loading Mosquitto MQTT broker...
    docker load -i cirame-mosquitto.tar
    echo   Mosquitto loaded.
) else (
    echo   Skipped: Mosquitto MQTT (cirame-mosquitto.tar not found)
)
echo.

REM Create folders and config
echo [4/4] Setting up folders and configuration...
if not exist "shared" mkdir shared
if not exist "data\database" mkdir data\database
if not exist "data\models" mkdir data\models
if not exist "data\ti-projects" mkdir data\ti-projects
if not exist "data\mosquitto" mkdir data\mosquitto
if not exist "mosquitto" mkdir mosquitto

if not exist "mosquitto\mosquitto.conf" (
    (
        echo listener 1883
        echo protocol mqtt
        echo listener 9001
        echo protocol websockets
        echo allow_anonymous true
        echo persistence true
        echo persistence_location /mosquitto/data/
        echo log_dest stdout
        echo log_type warning
        echo log_type error
    ) > mosquitto\mosquitto.conf
    echo   Created mosquitto\mosquitto.conf
)

echo   Folders ready.
echo.

echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo Installed images:
docker images | findstr /i "cirame mosquitto"
echo.
echo Next steps:
echo   1. Run 'start.bat' to start the application
echo   2. Access at http://localhost:3030
echo   3. Login: admin / admin123
echo.
echo NOTE: Your data (models, database, datasets) is preserved
echo       in Docker volumes across updates.
echo.
pause
