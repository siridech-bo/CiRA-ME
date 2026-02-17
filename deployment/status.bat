@echo off
REM CiRA ME - Status Script
REM Shows the status of running containers

echo ============================================
echo   CiRA ME - Application Status
echo ============================================
echo.

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running.
    pause
    exit /b 1
)

echo Running Containers:
echo -------------------
docker ps --filter "name=cirame" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo Docker Images:
echo --------------
docker images --filter "reference=cirame*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo.

echo Volume Usage:
echo -------------
docker volume ls --filter "name=deployment_backend" --format "table {{.Name}}"
echo.

pause
