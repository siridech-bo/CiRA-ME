@echo off
REM CiRA ME - Export Script (Windows)
REM Saves currently running Docker images as .tar files for customer deployment
REM Does NOT rebuild — exports exactly what is running now

echo ============================================
echo   CiRA ME - Export Docker Images
echo ============================================
echo.

set DEPLOY_DIR=%~dp0

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Saving to: %DEPLOY_DIR%
echo.

REM Save required images
echo [1/4] Saving backend image (this may take a few minutes)...
docker save cirame-backend:latest -o "%DEPLOY_DIR%cirame-backend.tar"
if %errorlevel% neq 0 (
    echo ERROR: Backend image not found. Is CiRA ME running?
    pause
    exit /b 1
)
echo   Saved: cirame-backend.tar
echo.

echo [2/4] Saving frontend image...
docker save cirame-frontend:latest -o "%DEPLOY_DIR%cirame-frontend.tar"
if %errorlevel% neq 0 (
    echo ERROR: Frontend image not found.
    pause
    exit /b 1
)
echo   Saved: cirame-frontend.tar
echo.

REM Save optional images
echo [3/4] Saving TI ModelMaker image...
docker save cirame-ti-modelmaker:latest -o "%DEPLOY_DIR%cirame-ti-modelmaker.tar" 2>nul
if %errorlevel% equ 0 (
    echo   Saved: cirame-ti-modelmaker.tar
) else (
    echo   Skipped: cirame-ti-modelmaker not available
)
echo.

echo [4/4] Saving Mosquitto MQTT broker...
docker save eclipse-mosquitto:2 -o "%DEPLOY_DIR%cirame-mosquitto.tar" 2>nul
if %errorlevel% equ 0 (
    echo   Saved: cirame-mosquitto.tar
) else (
    echo   Skipped: mosquitto not available
)
echo.

echo ============================================
echo   Export Complete!
echo ============================================
echo.
echo Packages:
echo   Required:  cirame-backend.tar + cirame-frontend.tar
echo   Optional:  cirame-ti-modelmaker.tar (TI MCU support)
echo   Optional:  cirame-mosquitto.tar (MQTT live streaming)
echo.
echo Transfer the entire 'deployment' folder to the customer server.
echo On the customer server run: install.bat
echo.
pause
