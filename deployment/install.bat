@echo off
REM CiRA ME - Installation Script for Windows
REM Loads Docker images from .tar files and sets up the application

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

REM Load required images
echo [1/2] Loading required images...

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
echo.

REM Load optional images
echo [2/2] Loading optional images...

if exist "cirame-ti-modelmaker.tar" (
    echo   Loading TI ModelMaker image (this may take several minutes)...
    docker load -i cirame-ti-modelmaker.tar
    echo   TI ModelMaker loaded.
) else (
    echo   Skipped: cirame-ti-modelmaker.tar not found (TI MCU features disabled)
)

if exist "cirame-mosquitto.tar" (
    echo   Loading Mosquitto MQTT broker...
    docker load -i cirame-mosquitto.tar
    echo   Mosquitto loaded.
) else (
    echo   Skipped: cirame-mosquitto.tar not found (MQTT live streaming disabled)
)
echo.

REM Create folders
if not exist "shared" mkdir shared
if not exist "mosquitto" mkdir mosquitto

REM Create mosquitto config if not exists
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
    echo Created mosquitto\mosquitto.conf
)

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
echo NOTE: If your server has NVIDIA GPU, the application will use it automatically.
echo       If no GPU is available, use 'start-no-gpu.bat' instead.
echo.
pause
