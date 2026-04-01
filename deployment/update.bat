@echo off
REM CiRA ME - Update Script (Windows)
REM Loads new image versions while preserving all customer data
REM
REM Run from the deployment folder after copying new .tar files here.

echo ============================================
echo   CiRA ME - Update Script
echo ============================================
echo.
echo This will:
echo   1. Stop the running application
echo   2. Load the new Docker images
echo   3. Restart the application
echo.
echo Your data (database, models, datasets) will be preserved.
echo.
set /p confirm="Continue with update? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Update cancelled.
    pause
    exit /b 0
)

echo.

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running.
    pause
    exit /b 1
)

REM Check new tar files exist
if not exist "cirame-backend.tar" (
    echo ERROR: cirame-backend.tar not found.
    echo Copy the new .tar files here before running update.
    pause
    exit /b 1
)
if not exist "cirame-frontend.tar" (
    echo ERROR: cirame-frontend.tar not found.
    echo Copy the new .tar files here before running update.
    pause
    exit /b 1
)

echo [1/4] Stopping current application (data is preserved)...
docker compose -f docker-compose.yml down 2>nul
docker compose -f docker-compose-no-gpu.yml down 2>nul
echo Done.
echo.

echo [2/4] Loading new backend image...
docker load -i cirame-backend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to load backend image.
    pause
    exit /b 1
)
echo.

echo [3/4] Loading new frontend image...
docker load -i cirame-frontend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to load frontend image.
    pause
    exit /b 1
)
echo.

REM Load optional images if present
if exist "cirame-ti-modelmaker.tar" (
    echo Loading TI ModelMaker image...
    docker load -i cirame-ti-modelmaker.tar
    echo.
) else (
    echo Skipped: cirame-ti-modelmaker.tar not found ^(optional^)
)

if exist "cirame-mosquitto.tar" (
    echo Loading Mosquitto MQTT broker...
    docker load -i cirame-mosquitto.tar
    echo.
) else (
    echo Skipped: cirame-mosquitto.tar not found ^(optional^)
)
echo.

REM Remove dangling old image layers to free disk space
echo [4/4] Cleaning up old image layers...
docker image prune -f >nul 2>&1
echo Done.
echo.

echo ============================================
echo   Update Complete!
echo ============================================
echo.
echo Restart the application:
echo   With GPU    : start.bat
echo   Without GPU : start-no-gpu.bat
echo.
pause
