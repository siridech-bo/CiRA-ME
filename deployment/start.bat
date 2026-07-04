@echo off
REM CiRA ME - Start Script (with GPU support)
REM Starts available services — skips optional ones if image not installed

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Starting Application (GPU)
echo ============================================
echo.

REM Check Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM docker compose v2 (plugin) or v1 (standalone).
set "DOCKER_COMPOSE="
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    set "DOCKER_COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>&1
    if %errorlevel% equ 0 set "DOCKER_COMPOSE=docker-compose"
)
if not defined DOCKER_COMPOSE (
    echo ERROR: docker compose is not installed. Re-run install.bat.
    pause
    exit /b 1
)

REM Check required images
docker images | findstr cirame-backend >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker images not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

REM Port collision check — fail fast naming the port instead of the raw
REM docker "Address already in use" pile.
set "PORT_CONFLICT="
for %%P in (3030 5100 5200 1883 9001) do (
    netstat -an | findstr ":%%P " | findstr "LISTENING" >nul 2>&1
    if !errorlevel! equ 0 set "PORT_CONFLICT=!PORT_CONFLICT! %%P"
)
if defined PORT_CONFLICT (
    echo ERROR: Ports already in use:!PORT_CONFLICT!
    echo.
    echo Common culprits:
    echo   1883 — host mosquitto or another MQTT broker
    echo   3030 — another web application
    echo   5100 — another Flask app
    echo.
    echo See what's using a port: netstat -ano ^| findstr ":^<port^>"
    pause
    exit /b 1
)

echo Starting CiRA ME services...
echo.

REM Start required services first
%DOCKER_COMPOSE% -f docker-compose.yml up -d backend frontend
if %errorlevel% neq 0 (
    echo ERROR: Failed to start core services.
    echo If you see GPU-related errors, try 'start-no-gpu.bat' instead.
    pause
    exit /b 1
)

REM Start optional services (skip if image not available)
docker images | findstr cirame-ti-modelmaker >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting TI ModelMaker...
    %DOCKER_COMPOSE% -f docker-compose.yml up -d ti-modelmaker
) else (
    echo Skipped: TI ModelMaker (image not installed)
)

docker images | findstr mosquitto >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting MQTT Broker...
    %DOCKER_COMPOSE% -f docker-compose.yml up -d mosquitto
) else (
    echo Skipped: MQTT Broker (image not installed)
)

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo Services running:
%DOCKER_COMPOSE% -f docker-compose.yml ps --format "table {{.Name}}\t{{.Status}}" 2>nul
echo.
echo Access at: http://localhost:3030
echo Login: admin / admin123
echo.
echo To view logs: %DOCKER_COMPOSE% logs -f
echo To stop: run 'stop.bat'
echo.
pause
