@echo off
REM CiRA ME - View Logs Script
REM Shows live logs from all containers

echo ============================================
echo   CiRA ME - Application Logs
echo   Press Ctrl+C to exit
echo ============================================
echo.

docker compose -f docker-compose.yml logs -f 2>nul || docker compose -f docker-compose-no-gpu.yml logs -f
