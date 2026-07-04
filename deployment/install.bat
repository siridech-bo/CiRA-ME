@echo off
REM CiRA ME - Installation Script for Windows
REM Loads Docker images from .tar files and sets up the application
REM Safe to re-run — stops old version, removes old images, loads new ones

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Installation Script
echo   Machine Intelligence for Edge Computing
echo ============================================
echo.

REM ─── Pre-flight checks ──────────────────────────────────────────────

REM 1. Docker installed + daemon running.
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not on PATH.
    echo Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Desktop is installed but not running.
    echo Start Docker Desktop from the Start menu and wait for the whale icon
    echo in the tray to say "Docker Desktop is running", then re-run install.bat.
    pause
    exit /b 1
)

REM 2. docker compose v2 (plugin) or v1 (standalone).
set "DOCKER_COMPOSE="
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    set "DOCKER_COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>&1
    if %errorlevel% equ 0 (
        set "DOCKER_COMPOSE=docker-compose"
    )
)
if not defined DOCKER_COMPOSE (
    echo ERROR: docker compose is not installed.
    echo Docker Desktop should include it. Try Docker Desktop v20.10+.
    pause
    exit /b 1
)
echo   docker compose: %DOCKER_COMPOSE%

REM 3. Free disk space — ~25 GB unpacked for the 4 images.
for /f "tokens=3" %%A in ('dir /-C ^| findstr "bytes free"') do set "FREE_BYTES=%%A"
if not defined FREE_BYTES (
    echo   WARNING: could not determine free disk space, continuing anyway
) else (
    REM Convert to GB by dividing 1073741824. cmd.exe can't handle big ints so
    REM we approximate: if first 10 digits >= 3, that's roughly 30+ GB.
    set "FB=!FREE_BYTES!"
    set "FB_LEN=0"
    for /l %%i in (0,1,20) do if defined FB (
        set "FB=!FB:~1!"
        set /a "FB_LEN+=1"
    )
    REM Length >= 11 means >= 10 GB. Length >= 12 means >= 100 GB. Length 10 = 1 GB.
    REM We want at least 30 GB — reject if length < 11.
    if !FB_LEN! lss 11 (
        echo ERROR: Less than 10 GB free on current drive. Need at least 30 GB.
        echo docker load will fail partway through with a cryptic error.
        pause
        exit /b 1
    )
    echo   Free space: ok
)

REM 4. Verify tarballs are present before we tear down the previous install.
set "MISSING="
if not exist "cirame-backend.tar" set "MISSING=!MISSING! cirame-backend.tar"
if not exist "cirame-frontend.tar" set "MISSING=!MISSING! cirame-frontend.tar"
if defined MISSING (
    echo ERROR: Required tarballs missing:!MISSING!
    echo Make sure you extracted the full release ZIP and are running install.bat
    echo from the deployment\ folder.
    pause
    exit /b 1
)

echo Docker is running.
echo.

REM Stop previous version if running
echo [1/4] Stopping previous version (if running)...
%DOCKER_COMPOSE% -f docker-compose.yml down 2>nul
%DOCKER_COMPOSE% -f docker-compose-no-gpu.yml down 2>nul
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

REM Load new images and verify each landed
echo [3/4] Loading new images...

echo   Loading backend image (this may take several minutes)...
docker load -i cirame-backend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to load backend image.
    pause
    exit /b 1
)
docker image inspect cirame-backend:latest >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cirame-backend.tar loaded without error but the image is not present.
    echo Tarball is likely truncated or malformed. Re-download it.
    pause
    exit /b 1
)
echo   Backend loaded.

echo   Loading frontend image...
docker load -i cirame-frontend.tar
if %errorlevel% neq 0 (
    echo ERROR: Failed to load frontend image.
    pause
    exit /b 1
)
docker image inspect cirame-frontend:latest >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cirame-frontend.tar loaded without error but the image is not present.
    pause
    exit /b 1
)
echo   Frontend loaded.

REM Optional images
if exist "cirame-ti-modelmaker.tar" (
    echo   Loading TI ModelMaker image (this may take several minutes)...
    docker load -i cirame-ti-modelmaker.tar
    if %errorlevel% neq 0 (
        echo ERROR: Failed to load TI ModelMaker image.
        pause
        exit /b 1
    )
    docker image inspect cirame-ti-modelmaker:latest >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: cirame-ti-modelmaker.tar corrupt. Re-download.
        pause
        exit /b 1
    )
    echo   TI ModelMaker loaded.
) else (
    echo   Skipped: TI ModelMaker (cirame-ti-modelmaker.tar not found)
)

if exist "cirame-mosquitto.tar" (
    echo   Loading Mosquitto MQTT broker...
    docker load -i cirame-mosquitto.tar
    docker image inspect eclipse-mosquitto:2 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Mosquitto image did not land after load.
        pause
        exit /b 1
    )
    echo   Mosquitto loaded.
) else (
    echo   Skipped: Mosquitto MQTT (cirame-mosquitto.tar not found)
)
echo.

REM Create folders and config
echo [4/4] Setting up folders and configuration...

REM Migration: legacy .\shared\ -> .\datasets\shared\ (for upgrades from old layout)
if exist "shared" if not exist "datasets\shared" (
    echo   Migrating legacy shared/ folder to datasets/shared/...
    if not exist "datasets" mkdir datasets
    move /Y "shared" "datasets\shared" >nul 2>&1
    if exist "datasets\shared" (
        echo   Migration complete: .\shared\ moved to .\datasets\shared\
    )
)

if not exist "datasets" mkdir datasets
if not exist "datasets\shared" mkdir datasets\shared
if not exist "data\database" mkdir data\database
if not exist "data\models" mkdir data\models
if not exist "data\ti-projects" mkdir data\ti-projects
if not exist "data\mosquitto" mkdir data\mosquitto
if not exist "mosquitto" mkdir mosquitto
REM Folder Watcher input/output — pre-create so Docker doesn't own it as
REM root on first `up` (Windows Docker Desktop maps this less painfully
REM than Linux, but pre-creating keeps behavior consistent).
if not exist "watcher-data" mkdir watcher-data

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
echo NOTE: Your data (models, database, datasets, watcher-data) is preserved
echo       on the host disk across updates.
echo.
pause
