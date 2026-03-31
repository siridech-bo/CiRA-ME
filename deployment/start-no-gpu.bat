@echo off
REM CiRA ME - Start Script (No GPU / CPU Only)
REM Starts available services — skips optional ones if image not installed

echo ============================================
echo   CiRA ME - Starting Application (CPU Only)
echo ============================================
echo.

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running.
    pause
    exit /b 1
)

docker images | findstr cirame-backend >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker images not found. Run 'install.bat' first.
    pause
    exit /b 1
)

echo Starting core services (CPU mode)...
docker compose -f docker-compose-no-gpu.yml up -d backend frontend

docker images | findstr cirame-ti-modelmaker >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting TI ModelMaker...
    docker compose -f docker-compose-no-gpu.yml up -d ti-modelmaker
) else (
    echo Skipped: TI ModelMaker (image not installed)
)

docker images | findstr mosquitto >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting MQTT Broker...
    docker compose -f docker-compose-no-gpu.yml up -d mosquitto
) else (
    echo Skipped: MQTT Broker (image not installed)
)

echo.
echo ============================================
echo   Application Started (CPU Only)!
echo ============================================
echo.
echo Access at: http://localhost:3030
echo Login: admin / admin123
echo.
pause
