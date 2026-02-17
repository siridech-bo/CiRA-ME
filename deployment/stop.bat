@echo off
REM CiRA ME - Stop Script
REM Stops all running CiRA ME services

echo ============================================
echo   CiRA ME - Stopping Application
echo ============================================
echo.

REM Try stopping with both compose files
echo Stopping services...
docker compose -f docker-compose.yml down 2>nul
docker compose -f docker-compose-no-gpu.yml down 2>nul

echo.
echo ============================================
echo   Application Stopped
echo ============================================
echo.
echo To restart, run 'start.bat' or 'start-no-gpu.bat'
echo.
pause
