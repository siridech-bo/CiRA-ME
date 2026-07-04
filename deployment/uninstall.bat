@echo off
REM CiRA ME - Uninstall Script (Windows)
REM Removes Docker containers, images, and (optionally) named volumes.
REM By default, your bind-mounted data on the host disk is PRESERVED.
REM
REM Usage:  uninstall.bat                (remove containers + images only)
REM         uninstall.bat --purge-data   (ALSO deletes .\data and .\datasets)

setlocal enabledelayedexpansion

set "PURGE_DATA=0"
if /i "%~1"=="--purge-data" set "PURGE_DATA=1"
if /i "%~1"=="--purge" set "PURGE_DATA=1"
if /i "%~1"=="-h" (
    echo Usage: uninstall.bat [--purge-data]
    echo   --purge-data   Also delete .\data and .\datasets ^(DESTRUCTIVE^)
    exit /b 0
)
if /i "%~1"=="--help" (
    echo Usage: uninstall.bat [--purge-data]
    echo   --purge-data   Also delete .\data and .\datasets ^(DESTRUCTIVE^)
    exit /b 0
)

echo ============================================
echo   CiRA ME - Uninstall Script
echo ============================================
echo.

echo This will remove:
echo   - Running CiRA ME containers
echo   - CiRA ME Docker images (backend, frontend, ti-modelmaker)
echo   - Named Docker volumes (usually empty — data is on host disk)
echo.

if "%PURGE_DATA%"=="1" (
    echo This will ALSO permanently delete:
    echo   - .\data\          (SQLite database, trained models, TI projects)
    echo   - .\datasets\      (user-uploaded CSVs and CBOR files)
    echo   - .\watcher-data\  (Folder Watcher input/output history)
    echo.
    echo   ==^> THIS IS UNRECOVERABLE. Consider running 'backup.bat' first.
) else (
    echo Your data is PRESERVED (this is the default):
    echo   .\data\          -- database, models, TI projects (kept)
    echo   .\datasets\      -- user-uploaded files (kept)
    echo   .\watcher-data\  -- Folder Watcher data (kept)
    echo.
    echo To also delete your data, re-run with:  uninstall.bat --purge-data
)
echo.

REM Typed-phrase confirmation
if "%PURGE_DATA%"=="1" (
    set /p confirm="Type PURGE (exact case) to confirm data deletion, or anything else to cancel: "
    if not "!confirm!"=="PURGE" (
        echo Cancelled.
        pause
        exit /b 0
    )
) else (
    set /p confirm="Type REMOVE (exact case) to confirm, or anything else to cancel: "
    if not "!confirm!"=="REMOVE" (
        echo Cancelled.
        pause
        exit /b 0
    )
)
echo.

REM Pick whichever compose is available (do not fail if missing)
set "DOCKER_COMPOSE="
docker compose version >nul 2>&1
if !errorlevel! equ 0 (
    set "DOCKER_COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>&1
    if !errorlevel! equ 0 set "DOCKER_COMPOSE=docker-compose"
)

echo [1/3] Stopping containers...
if defined DOCKER_COMPOSE (
    %DOCKER_COMPOSE% -f docker-compose.yml down 2>nul
    %DOCKER_COMPOSE% -f docker-compose-no-gpu.yml down 2>nul
)
docker stop cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>nul
docker rm cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto 2>nul

echo [2/3] Removing images...
docker rmi cirame-backend:latest 2>nul
docker rmi cirame-frontend:latest 2>nul
docker rmi cirame-ti-modelmaker:latest 2>nul
docker rmi eclipse-mosquitto:2 2>nul

echo [3/3] Removing named volumes (usually empty — data is bind-mounted)...
docker volume rm deployment_backend-data 2>nul
docker volume rm deployment_backend-models 2>nul

if "%PURGE_DATA%"=="1" (
    echo.
    echo [+] Deleting host data folders...
    if exist data rmdir /s /q data
    if exist datasets rmdir /s /q datasets
    if exist watcher-data rmdir /s /q watcher-data
    echo   .\data, .\datasets, .\watcher-data deleted.
)

echo.
echo ============================================
echo   Uninstall Complete
echo ============================================
echo.
if "%PURGE_DATA%"=="1" (
    echo All CiRA ME data, containers, images, and volumes have been removed.
) else (
    echo Containers and images removed. Your data at .\data, .\datasets, and
    echo .\watcher-data was preserved.
    echo   To also delete data, re-run with:  uninstall.bat --purge-data
)
echo.
pause
