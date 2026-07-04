@echo off
REM CiRA ME - Backup Script (Windows)
REM Creates a compressed ZIP of all customer data so it can be restored
REM to any CiRA ME deployment. Safe to run at any time (data is copied, not moved).
REM
REM Usage:  backup.bat                        (default: .\backups\cirame-YYYYMMDD-HHMMSS.zip)
REM         backup.bat C:\path\to\dest.zip    (explicit destination)

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Backup Script
echo ============================================
echo.

REM Figure out where to write the ZIP
if not "%~1"=="" (
    set "DEST=%~1"
) else (
    REM YYYYMMDD-HHMMSS timestamp (avoid locale-dependent %date%/%time% parsing).
    for /f "usebackq" %%A in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"`) do set "STAMP=%%A"
    if not exist backups mkdir backups
    set "DEST=backups\cirame-!STAMP!.zip"
)

if exist "!DEST!" (
    echo ERROR: !DEST! already exists. Pick a different name or delete it first.
    pause
    exit /b 1
)

if not exist data if not exist datasets (
    echo ERROR: Neither .\data nor .\datasets exist here.
    echo Run this script from the deployment folder that contains
    echo the docker-compose.yml + your data\ and datasets\ folders.
    pause
    exit /b 1
)

REM Stop containers if running so SQLite DB is quiesced
docker ps --format "{{.Names}}" 2>nul | findstr /b "cirame-" >nul 2>&1
if !errorlevel! equ 0 (
    echo CiRA ME containers are currently running.
    echo For a consistent backup of the SQLite DB, they should be stopped first.
    echo.
    set /p ANS="Stop containers now? (yes/no, default yes): "
    if not "!ANS!"=="no" (
        docker compose -f docker-compose.yml down 2>nul
        docker compose -f docker-compose-no-gpu.yml down 2>nul
        echo   Containers stopped.
    ) else (
        echo   Continuing with containers running — DB may be inconsistent.
    )
    echo.
)

echo Backing up to: !DEST!

REM Build a PowerShell Compress-Archive command with all present sources.
REM Compress-Archive is available on Windows 10+ / Server 2016+.
set "SOURCES="
if exist data           set "SOURCES=!SOURCES!,'data'"
if exist datasets       set "SOURCES=!SOURCES!,'datasets'"
if exist watcher-data   set "SOURCES=!SOURCES!,'watcher-data'"
if exist mosquitto      set "SOURCES=!SOURCES!,'mosquitto'"
if exist docker-compose.yml       set "SOURCES=!SOURCES!,'docker-compose.yml'"
if exist docker-compose-no-gpu.yml set "SOURCES=!SOURCES!,'docker-compose-no-gpu.yml'"
if not defined SOURCES (
    echo ERROR: nothing to back up.
    pause
    exit /b 1
)
REM Strip leading comma
set "SOURCES=!SOURCES:~1!"

powershell -NoProfile -Command "Compress-Archive -Path !SOURCES! -DestinationPath '!DEST!' -CompressionLevel Optimal -Force"
if %errorlevel% neq 0 (
    echo ERROR: backup failed. See PowerShell error above.
    pause
    exit /b 1
)

for %%A in ("!DEST!") do set "SIZE=%%~zA"
echo.
echo ============================================
echo   Backup Complete
echo ============================================
echo.
echo   File: !DEST!
echo   Size: !SIZE! bytes
echo.
echo To restore later:  restore.bat "!DEST!"
echo Or manually:       Expand-Archive -Path "!DEST!" -DestinationPath C:\target\folder
echo.
pause
