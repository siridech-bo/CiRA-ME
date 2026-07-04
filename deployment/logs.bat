@echo off
REM CiRA ME - View Logs Script
REM Shows live logs from all containers.
REM Press Ctrl+C to exit.
REM
REM Usage:  logs.bat              (follow all CiRA ME container logs)
REM         logs.bat backend      (follow only one service)
REM         logs.bat --no-follow  (dump recent logs and exit)

setlocal enabledelayedexpansion

set "FOLLOW=-f"
set "SERVICE="
if /i "%~1"=="--no-follow" set "FOLLOW="
if /i "%~1"=="-n" set "FOLLOW="
if /i "%~1"=="backend" set "SERVICE=backend"
if /i "%~1"=="frontend" set "SERVICE=frontend"
if /i "%~1"=="ti-modelmaker" set "SERVICE=ti-modelmaker"
if /i "%~1"=="mosquitto" set "SERVICE=mosquitto"

echo ============================================
echo   CiRA ME - Application Logs
if defined FOLLOW echo   Press Ctrl+C to exit
echo ============================================
echo.

REM Fail early if nothing is running.
docker ps --format "{{.Names}}" 2>nul | findstr /b "cirame-" >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: no CiRA ME containers are running.
    echo Start the app first: start.bat (or start-no-gpu.bat)
    pause
    exit /b 1
)

REM docker compose v2/v1 detection
set "DOCKER_COMPOSE="
docker compose version >nul 2>&1
if !errorlevel! equ 0 (
    set "DOCKER_COMPOSE=docker compose"
) else (
    where docker-compose >nul 2>&1
    if !errorlevel! equ 0 set "DOCKER_COMPOSE=docker-compose"
)
if not defined DOCKER_COMPOSE (
    echo docker compose not installed — falling back to per-container logs
    if defined SERVICE (
        docker logs %FOLLOW% cirame-!SERVICE!
    ) else (
        for %%N in (cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto) do (
            docker ps --format "{{.Names}}" | findstr /b "%%N$" >nul 2>&1
            if !errorlevel! equ 0 (
                echo [%%N]
                docker logs --tail 20 %%N
            )
        )
    )
    exit /b 0
)

for %%C in (docker-compose.yml docker-compose-no-gpu.yml) do (
    if exist "%%C" (
        %DOCKER_COMPOSE% -f "%%C" ps -q 2>nul | findstr /r "." >nul 2>&1
        if !errorlevel! equ 0 (
            %DOCKER_COMPOSE% -f "%%C" logs %FOLLOW% !SERVICE!
            exit /b !errorlevel!
        )
    )
)

echo ERROR: docker compose files present, but no services are running under them.
echo containers may have been started outside of compose — try:
echo   docker logs %FOLLOW% cirame-backend
pause
exit /b 1
