@echo off
REM CiRA ME - Uninstall Script
REM Removes all CiRA ME containers, images, and volumes

echo ============================================
echo   CiRA ME - Uninstall Script
echo ============================================
echo.
echo WARNING: This will remove all CiRA ME data including:
echo   - Docker containers
echo   - Docker images
echo   - Data volumes (all uploaded files and database)
echo.
set /p confirm="Are you sure you want to continue? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

echo.
echo [1/3] Stopping containers...
docker compose -f docker-compose.yml down 2>nul
docker compose -f docker-compose-no-gpu.yml down 2>nul

echo [2/3] Removing images...
docker rmi cirame-backend:latest 2>nul
docker rmi cirame-frontend:latest 2>nul

echo [3/3] Removing volumes...
docker volume rm deployment_backend-data 2>nul
docker volume rm deployment_backend-models 2>nul

echo.
echo ============================================
echo   Uninstall Complete
echo ============================================
echo.
echo Note: The 'shared' folder was not deleted.
echo You can manually delete it if needed.
echo.
pause
