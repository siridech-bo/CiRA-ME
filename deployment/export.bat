@echo off
REM CiRA ME - Build and Export Script
REM Run this from the PROJECT ROOT (parent of deployment/)
REM Builds Docker images and saves them as .tar files ready for transfer
REM
REM Usage:  cd C:\path\to\CiRA-ME
REM         deployment\export.bat

echo ============================================
echo   CiRA ME - Build and Export Images
echo ============================================
echo.

REM Must be run from project root (where docker-compose.yml lives)
if not exist "docker-compose.yml" (
    echo ERROR: Run this script from the project root directory,
    echo        not from inside the deployment folder.
    echo.
    echo Usage:  cd C:\path\to\CiRA-ME
    echo         deployment\export.bat
    pause
    exit /b 1
)

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [1/4] Building backend image...
docker compose build backend
if %errorlevel% neq 0 (
    echo ERROR: Backend build failed.
    pause
    exit /b 1
)
echo.

echo [2/4] Building frontend image...
docker compose build frontend
if %errorlevel% neq 0 (
    echo ERROR: Frontend build failed.
    pause
    exit /b 1
)
echo.

echo [3/4] Saving backend image (this may take a few minutes)...
docker save cirame-backend:latest -o deployment\cirame-backend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to save backend image.
    pause
    exit /b 1
)
echo   Saved: deployment\cirame-backend.tar

echo [4/4] Saving frontend image...
docker save cirame-frontend:latest -o deployment\cirame-frontend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to save frontend image.
    pause
    exit /b 1
)
echo   Saved: deployment\cirame-frontend.tar
echo.

echo ============================================
echo   Export Complete!
echo ============================================
echo.
echo Files ready for transfer:
dir /b deployment\*.tar deployment\*.bat deployment\*.yml deployment\README.md 2>nul
echo.
echo Transfer the entire 'deployment' folder to the customer server.
echo On the customer server, run 'install.bat' (Windows) or 'install.sh' (Linux).
echo.
pause
