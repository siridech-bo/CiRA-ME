@echo off
REM CiRA ME - Migrate Data Script (Windows)
REM Copies user data from an old deployment folder to this folder.
REM
REM Usage:  migrate.bat "C:\path\to\old\deployment"
REM
REM What is migrated:
REM   - data\database\          (SQLite database: users, models, endpoints, apps)
REM   - data\models\            (Trained model files)
REM   - data\ti-projects\       (TI ModelMaker projects)
REM   - data\mosquitto\         (MQTT broker persistent data)
REM   - datasets\               (User-uploaded datasets, including shared/)
REM   - watcher-data\           (Folder Watcher input/output history)
REM   - shared\                 (Legacy: auto-migrated to datasets\shared\)

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Data Migration Script
echo ============================================
echo.

if "%~1"=="" (
    echo ERROR: Please provide the path to your old deployment folder.
    echo.
    echo Usage:  migrate.bat "C:\path\to\old\CiRA-ME-deployment"
    echo.
    echo Example:
    echo   migrate.bat "D:\CiRA ME v1.0\deployment"
    echo.
    pause
    exit /b 1
)

set "OLD_DIR=%~1"

if not exist "%OLD_DIR%" (
    echo ERROR: Old folder does not exist:
    echo   %OLD_DIR%
    pause
    exit /b 1
)

REM Refuse source == target. xcopy src\*  target\ where src == cwd would
REM copy everything back over itself — at best a no-op, at worst a partial
REM mangle if any file changes mid-copy.
for %%A in ("%OLD_DIR%") do set "OLD_ABS=%%~fA"
set "NEW_ABS=%CD%"
if /i "!OLD_ABS!"=="!NEW_ABS!" (
    echo ERROR: Source and target are the same folder ^(!OLD_ABS!^).
    echo migrate.bat copies from another install, not into itself.
    pause
    exit /b 1
)

echo Source ^(old^):  !OLD_ABS!
echo Target ^(this^): !NEW_ABS!
echo.

REM Verify source has at least one expected folder
set "FOUND=0"
if exist "%OLD_DIR%\data" set "FOUND=1"
if exist "%OLD_DIR%\datasets" set "FOUND=1"
if exist "%OLD_DIR%\shared" set "FOUND=1"
if exist "%OLD_DIR%\watcher-data" set "FOUND=1"
if "!FOUND!"=="0" (
    echo ERROR: Source does not look like a CiRA ME deployment folder.
    echo Expected at least one of: data\, datasets\, shared\, watcher-data\
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

REM Stop running containers so files aren't locked
if defined DOCKER_COMPOSE (
    echo Stopping running containers ^(if any^)...
    %DOCKER_COMPOSE% -f docker-compose.yml down 2>nul
    %DOCKER_COMPOSE% -f docker-compose-no-gpu.yml down 2>nul
    echo.
)

REM Confirm
echo This will COPY user data from the old folder into this folder.
echo Existing files in this folder will be OVERWRITTEN.
echo.
set /p confirm="Continue? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Migration cancelled.
    pause
    exit /b 0
)
echo.

REM ─── Safe copy: fatal on failure ─────────────────────────────────
REM Copying half a database and continuing is exactly how customers end
REM up with unbootable installs. Any xcopy error aborts the migration.

echo [1/4] Migrating data\ folder (database, models, ti-projects, mosquitto)...
if exist "%OLD_DIR%\data" (
    if not exist "data" mkdir data
    xcopy /E /I /Y /Q "%OLD_DIR%\data\*" "data\" >nul
    if errorlevel 1 (
        echo ERROR: failed to copy data\. Migration aborted.
        echo Fix disk space / permissions and re-run. Delete partial data\
        echo copy before retrying so nothing is left inconsistent.
        pause
        exit /b 1
    )
    echo   data\ copied successfully.
) else (
    echo   Skipped: %OLD_DIR%\data not found.
)
echo.

echo [2/4] Migrating datasets\ folder (user uploads ^& shared)...
if exist "%OLD_DIR%\datasets" (
    if not exist "datasets" mkdir datasets
    xcopy /E /I /Y /Q "%OLD_DIR%\datasets\*" "datasets\" >nul
    if errorlevel 1 (
        echo ERROR: failed to copy datasets\. Migration aborted.
        pause
        exit /b 1
    )
    echo   datasets\ copied successfully.
) else (
    echo   Skipped: %OLD_DIR%\datasets not found.
)
echo.

echo [3/4] Migrating watcher-data\ folder (Folder Watcher history)...
if exist "%OLD_DIR%\watcher-data" (
    if not exist "watcher-data" mkdir watcher-data
    xcopy /E /I /Y /Q "%OLD_DIR%\watcher-data\*" "watcher-data\" >nul
    if errorlevel 1 (
        echo ERROR: failed to copy watcher-data\. Migration aborted.
        pause
        exit /b 1
    )
    echo   watcher-data\ copied successfully.
) else (
    echo   Skipped: %OLD_DIR%\watcher-data not found.
)
echo.

echo [4/4] Migrating legacy shared\ folder (if present)...
if exist "%OLD_DIR%\shared" (
    if not exist "datasets\shared" mkdir datasets\shared
    xcopy /E /I /Y /Q "%OLD_DIR%\shared\*" "datasets\shared\" >nul
    if errorlevel 1 (
        echo ERROR: failed to copy legacy shared\. Migration aborted.
        pause
        exit /b 1
    )
    echo   Legacy shared\ copied to datasets\shared\.
) else (
    echo   Skipped: %OLD_DIR%\shared not found.
)
echo.

REM Mosquitto config
if exist "%OLD_DIR%\mosquitto\mosquitto.conf" (
    if not exist "mosquitto" mkdir mosquitto
    if not exist "mosquitto\mosquitto.conf" (
        copy /Y "%OLD_DIR%\mosquitto\mosquitto.conf" "mosquitto\mosquitto.conf" >nul
        if errorlevel 1 (
            echo ERROR: failed to copy mosquitto.conf. Migration aborted.
            pause
            exit /b 1
        )
        echo   Copied mosquitto\mosquitto.conf
    )
)

echo ============================================
echo   Migration Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Run 'install.bat' if you have not loaded the new images yet
echo      (or 'update.bat' if images are already installed)
echo   2. Run 'start.bat' (or 'start-no-gpu.bat') to launch CiRA ME
echo   3. Verify your data appears at http://localhost:3030
echo.
pause
