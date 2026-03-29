@echo off
REM CiRA ME - Build and Export Script (Windows)
REM Builds Docker images and saves as .tar files for customer deployment

echo ============================================
echo   CiRA ME - Build and Export Images
echo ============================================
echo.

set DEPLOY_DIR=%~dp0
set PROJECT_ROOT=%~dp0..

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%\backend" (
    echo ERROR: Cannot locate project root.
    pause
    exit /b 1
)

echo Building from: %PROJECT_ROOT%
echo Saving to:     %DEPLOY_DIR%
echo.

REM Build images
echo [1/6] Building backend image...
docker compose -f "%PROJECT_ROOT%\docker-compose.yml" build backend
if %errorlevel% neq 0 ( echo ERROR: Backend build failed. & pause & exit /b 1 )
echo.

echo [2/6] Building frontend image...
docker compose -f "%PROJECT_ROOT%\docker-compose.yml" build frontend
if %errorlevel% neq 0 ( echo ERROR: Frontend build failed. & pause & exit /b 1 )
echo.

echo [3/6] Building TI ModelMaker image...
docker compose -f "%PROJECT_ROOT%\docker-compose.yml" build ti-modelmaker
if %errorlevel% neq 0 ( echo WARNING: TI ModelMaker build failed, skipping. )
echo.

REM Save images
echo [4/6] Saving backend image (this may take a few minutes)...
docker save cirame-backend:latest -o "%DEPLOY_DIR%cirame-backend.tar"
echo   Saved: cirame-backend.tar
echo.

echo [5/6] Saving frontend image...
docker save cirame-frontend:latest -o "%DEPLOY_DIR%cirame-frontend.tar"
echo   Saved: cirame-frontend.tar
echo.

echo [6/6] Saving optional images...
docker save cirame-ti-modelmaker:latest -o "%DEPLOY_DIR%cirame-ti-modelmaker.tar" 2>nul
if %errorlevel% equ 0 ( echo   Saved: cirame-ti-modelmaker.tar ) else ( echo   Skipped: ti-modelmaker )

docker pull eclipse-mosquitto:2 >nul 2>&1
docker save eclipse-mosquitto:2 -o "%DEPLOY_DIR%cirame-mosquitto.tar" 2>nul
if %errorlevel% equ 0 ( echo   Saved: cirame-mosquitto.tar ) else ( echo   Skipped: mosquitto )
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
