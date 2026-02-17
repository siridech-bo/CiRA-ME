@echo off
REM CiRA ME - Installation Script for Windows
REM This script loads Docker images and sets up the application

echo ============================================
echo   CiRA ME - Installation Script
echo   Machine Intelligence for Edge Computing
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed.
    echo Please install Docker Desktop and start it before running this script.
    pause
    exit /b 1
)

echo [1/4] Docker is running...
echo.

REM Load backend image
echo [2/4] Loading backend image (this may take a few minutes)...
if exist "cirame-backend.tar" (
    docker load -i cirame-backend.tar
    if %errorlevel% neq 0 (
        echo ERROR: Failed to load backend image.
        pause
        exit /b 1
    )
    echo Backend image loaded successfully.
) else (
    echo ERROR: cirame-backend.tar not found!
    echo Please ensure the file is in the same directory as this script.
    pause
    exit /b 1
)
echo.

REM Load frontend image
echo [3/4] Loading frontend image...
if exist "cirame-frontend.tar" (
    docker load -i cirame-frontend.tar
    if %errorlevel% neq 0 (
        echo ERROR: Failed to load frontend image.
        pause
        exit /b 1
    )
    echo Frontend image loaded successfully.
) else (
    echo ERROR: cirame-frontend.tar not found!
    echo Please ensure the file is in the same directory as this script.
    pause
    exit /b 1
)
echo.

REM Create shared folder
echo [4/4] Creating shared folder for datasets...
if not exist "shared" mkdir shared
echo Shared folder created.
echo.

echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo Installed images:
docker images | findstr cirame
echo.
echo Next steps:
echo   1. Run 'start.bat' to start the application
echo   2. Access the application at http://localhost:3030
echo   3. Default login: admin / admin123
echo.
echo NOTE: If your server has NVIDIA GPU, the application will use it automatically.
echo       If no GPU is available, use 'start-no-gpu.bat' instead.
echo.
pause
