@echo off
REM CiRA ME - Start Script (with GPU support)
REM Starts available services — skips optional ones if image not installed

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

REM Check required images
docker images | findstr cirame-backend >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker images not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

echo Starting CiRA ME services...
echo.

REM Start required services first
docker compose -f docker-compose.yml up -d backend frontend
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
    docker compose -f docker-compose.yml up -d ti-modelmaker
) else (
    echo Skipped: TI ModelMaker (image not installed)
)

docker images | findstr mosquitto >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting MQTT Broker...
    docker compose -f docker-compose.yml up -d mosquitto
) else (
    echo Skipped: MQTT Broker (image not installed)
)

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo Services running:
docker compose -f docker-compose.yml ps --format "table {{.Name}}\t{{.Status}}" 2>nul
echo.
echo Access at: http://localhost:3030
echo Login: admin / admin123
echo.
echo To view logs: docker compose logs -f
echo To stop: run 'stop.bat'
echo.
pause
