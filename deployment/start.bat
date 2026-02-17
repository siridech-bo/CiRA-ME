@echo off
REM CiRA ME - Start Script (with GPU support)
REM Starts the application using Docker Compose

echo ============================================
echo   CiRA ME - Starting Application (GPU)
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if images are loaded
docker images | findstr cirame-backend >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker images not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

echo Starting CiRA ME services...
echo.

docker compose -f docker-compose.yml up -d

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start services.
    echo If you see GPU-related errors, try 'start-no-gpu.bat' instead.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo Access the application at: http://localhost:3030
echo Default login: admin / admin123
echo.
echo To view logs: docker compose logs -f
echo To stop: run 'stop.bat'
echo.
pause
