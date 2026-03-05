@echo off
REM CiRA ME - Build and Export Script
REM Double-click or run from anywhere — the script finds the project root automatically.

echo ============================================
echo   CiRA ME - Build and Export Images
echo ============================================
echo.

REM This script lives in deployment/; project root is one level up.
set DEPLOY_DIR=%~dp0
set PROJECT_ROOT=%~dp0..

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Verify we found the project root (should have backend/ and frontend/ dirs)
if not exist "%PROJECT_ROOT%\backend" (
    echo ERROR: Cannot locate project root.
    echo Expected backend/ directory at: %PROJECT_ROOT%
    pause
    exit /b 1
)

echo Building from: %PROJECT_ROOT%
echo Saving to:     %DEPLOY_DIR%
echo.

REM Build from project root using the dev docker-compose.yml
echo [1/4] Building backend image...
docker compose -f "%PROJECT_ROOT%\docker-compose.yml" build backend
if %errorlevel% neq 0 (
    echo ERROR: Backend build failed.
    pause
    exit /b 1
)
echo.

echo [2/4] Building frontend image...
docker compose -f "%PROJECT_ROOT%\docker-compose.yml" build frontend
if %errorlevel% neq 0 (
    echo ERROR: Frontend build failed.
    pause
    exit /b 1
)
echo.

echo [3/4] Saving backend image (this may take a few minutes)...
docker save cirame-backend:latest -o "%DEPLOY_DIR%cirame-backend.tar"
if %errorlevel% neq 0 (
    echo ERROR: Failed to save backend image.
    pause
    exit /b 1
)
echo   Saved: %DEPLOY_DIR%cirame-backend.tar

echo [4/4] Saving frontend image...
docker save cirame-frontend:latest -o "%DEPLOY_DIR%cirame-frontend.tar"
if %errorlevel% neq 0 (
    echo ERROR: Failed to save frontend image.
    pause
    exit /b 1
)
echo   Saved: %DEPLOY_DIR%cirame-frontend.tar
echo.

echo ============================================
echo   Export Complete!
echo ============================================
echo.
echo Transfer the entire 'deployment' folder to the customer server.
echo On the customer server run:
echo   Windows : install.bat
echo   Linux   : bash install.sh
echo.
pause
