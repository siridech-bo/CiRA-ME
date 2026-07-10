@echo off
REM CiRA ME - Update Script (Windows)
REM Loads new image versions while preserving all customer data.
REM Automatically snapshots the SQLite database before touching anything so a
REM broken new image can be rolled back cleanly.
REM
REM Run from the deployment folder after copying new .tar files here.

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Update Script
echo ============================================
echo.
echo This will:
echo   1. Snapshot .\data\database\ to a timestamped backup folder
echo   2. Stop the running application (data on host disk is preserved)
echo   3. Load the new Docker images from .tar files
echo   4. Clean up old image layers
echo.
echo IMPORTANT: Run this script in the SAME folder as your existing
echo            installation. Your data in .\data\ and .\datasets\
echo            is automatically preserved (bind-mounted).
echo.
echo If you extracted a new release to a DIFFERENT folder,
echo run 'migrate.bat ^<old_folder^>' first to copy your data over.
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

REM docker compose v2/v1 detection
set "DOCKER_COMPOSE="
docker compose version >nul 2>&1
if !errorlevel! equ 0 (
    set "DOCKER_COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>&1
    if !errorlevel! equ 0 set "DOCKER_COMPOSE=docker-compose"
)
if not defined DOCKER_COMPOSE (
    echo ERROR: docker compose is not installed. Re-run install.bat first.
    pause
    exit /b 1
)

REM Verify tarballs present + non-empty BEFORE tearing down the current app.
REM All 4 are required (see install.bat for the rationale on why TI + Mosquitto
REM are no longer optional).
for %%T in (cirame-backend.tar cirame-frontend.tar cirame-ti-modelmaker.tar cirame-mosquitto.tar) do (
    if not exist "%%T" (
        echo ERROR: %%T not found. Copy the new .tar files here before running update.
        pause
        exit /b 1
    )
    for %%A in ("%%T") do if "%%~zA"=="0" (
        echo ERROR: %%T is 0 bytes. Re-download and try again.
        pause
        exit /b 1
    )
)

REM ─── [1/5] Snapshot the DB before we touch anything ─────────────
echo [1/5] Snapshotting database...
set "BACKUP_DIR="
if exist "data\database" (
    for /f "usebackq" %%A in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"`) do set "STAMP=%%A"
    set "BACKUP_DIR=data\database.backup.!STAMP!"
    xcopy /E /I /Y /Q "data\database" "!BACKUP_DIR!" >nul
    if !errorlevel! neq 0 (
        echo ERROR: failed to snapshot data\database. Aborting update.
        echo Check disk space and permissions on .\data\
        pause
        exit /b 1
    )
    echo   Snapshot: !BACKUP_DIR!
    echo   To roll back later:
    echo     stop.bat
    echo     rmdir /s /q data\database
    echo     move "!BACKUP_DIR!" data\database
) else (
    echo   No existing data\database\ - first-time run? Skipping snapshot.
)
echo.

REM ─── [2/5] Stop containers ─────────────────────────────────────
echo [2/5] Stopping current application (data on disk is preserved)...
%DOCKER_COMPOSE% -f docker-compose.yml down 2>nul
%DOCKER_COMPOSE% -f docker-compose-no-gpu.yml down 2>nul

REM Legacy shared/ -> datasets/shared/ migration for v1.0 upgrades
if exist "shared" if not exist "datasets\shared" (
    echo   Migrating legacy shared\ -^> datasets\shared\...
    if not exist "datasets" mkdir datasets
    move /Y "shared" "datasets\shared" >nul 2>&1
    echo   Migration complete.
)
echo   Stopped.
echo.

REM ─── [3/5] Load new backend + verify ───────────────────────────
echo [3/5] Loading new backend image...
docker load -i cirame-backend.tar
if !errorlevel! neq 0 (
    echo ERROR: Failed to load backend image.
    pause
    exit /b 1
)
docker image inspect cirame-backend:latest >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: cirame-backend.tar loaded but no image is present.
    echo Tarball may be corrupt. Re-download.
    if defined BACKUP_DIR (
        echo Your DB snapshot at !BACKUP_DIR! is intact.
    )
    pause
    exit /b 1
)
echo   Backend loaded.
echo.

REM ─── [4/5] Load new frontend + verify ──────────────────────────
echo [4/5] Loading new frontend image...
docker load -i cirame-frontend.tar
if !errorlevel! neq 0 (
    echo ERROR: Failed to load frontend image.
    pause
    exit /b 1
)
docker image inspect cirame-frontend:latest >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: frontend image did not land after load.
    pause
    exit /b 1
)
echo   Frontend loaded.
echo.

REM TI ModelMaker and Mosquitto are both required (see pre-flight check above).
echo Loading TI ModelMaker image...
docker load -i cirame-ti-modelmaker.tar
if !errorlevel! neq 0 (
    echo ERROR: Failed to load TI ModelMaker image.
    pause
    exit /b 1
)
docker image inspect cirame-ti-modelmaker:latest >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: TI ModelMaker image did not land after load.
    pause
    exit /b 1
)
echo   TI ModelMaker loaded.

echo Loading Mosquitto MQTT broker...
docker load -i cirame-mosquitto.tar
if !errorlevel! neq 0 (
    echo ERROR: Failed to load Mosquitto image.
    pause
    exit /b 1
)
docker image inspect eclipse-mosquitto:2 >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Mosquitto image did not land after load.
    pause
    exit /b 1
)
echo   Mosquitto loaded.
echo.

REM ─── [5/5] Cleanup ─────────────────────────────────────────────
echo [5/5] Cleaning up old image layers...
docker image prune -f >nul 2>&1
echo   Done.
echo.

echo ============================================
echo   Update Complete!
echo ============================================
echo.
if defined BACKUP_DIR (
    echo   DB snapshot: !BACKUP_DIR! ^(safe to delete once new version is verified^)
    echo.
)
echo Restart the application:
echo   With GPU    : start.bat
echo   Without GPU : start-no-gpu.bat
echo.
echo If the new version fails to start, roll back with:
echo   stop.bat
if defined BACKUP_DIR (
    echo   rmdir /s /q data\database
    echo   move "!BACKUP_DIR!" data\database
)
echo.
pause
