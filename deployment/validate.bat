@echo off
REM CiRA ME - Pre-install Validation Script (Windows)
REM
REM Runs every pre-flight check that install.bat / start.bat will make, with
REM ZERO side effects. Customer runs this first to know whether the box is
REM ready — no partial installs, no half-loaded images.

setlocal enabledelayedexpansion

echo ============================================
echo   CiRA ME - Pre-install Validation
echo ============================================
echo.

set /a FAIL_COUNT=0
set /a WARN_COUNT=0
set /a PASS_COUNT=0

goto :main

:pass
    echo   PASS  %~1
    set /a PASS_COUNT+=1
    goto :eof

:warn
    echo   WARN  %~1
    set /a WARN_COUNT+=1
    goto :eof

:fail
    echo   FAIL  %~1
    set /a FAIL_COUNT+=1
    goto :eof

:main

REM ─── Docker daemon ─────────────────────────────────────────────
echo == Docker ==
where docker >nul 2>&1
if !errorlevel! neq 0 (
    call :fail "docker command not found. Install Docker Desktop for Windows."
) else (
    docker info >nul 2>&1
    if !errorlevel! neq 0 (
        call :fail "docker daemon not reachable — start Docker Desktop and wait for the whale icon."
    ) else (
        for /f "tokens=3" %%V in ('docker --version') do set "DVER=%%V"
        set "DVER=!DVER:,=!"
        call :pass "docker daemon reachable (!DVER!)"
    )
)

REM ─── docker compose v2 / v1 ───────────────────────────────────
docker compose version >nul 2>&1
if !errorlevel! equ 0 (
    call :pass "docker compose v2 plugin present"
) else (
    where docker-compose >nul 2>&1
    if !errorlevel! equ 0 (
        call :warn "using docker-compose v1 (works, but v2 plugin is recommended)"
    ) else (
        call :fail "docker compose is not installed. Docker Desktop v20.10+ ships it."
    )
)

REM ─── GPU + driver ─────────────────────────────────────────────
echo.
echo == GPU ==
where nvidia-smi >nul 2>&1
if !errorlevel! neq 0 (
    call :warn "no NVIDIA GPU or driver detected — plan to use start-no-gpu.bat (CPU mode)"
) else (
    for /f "tokens=*" %%L in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader') do (
        if not defined GPU_NAME set "GPU_NAME=%%L"
    )
    for /f "tokens=*" %%L in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader') do (
        if not defined DRIVER set "DRIVER=%%L"
    )
    call :pass "GPU detected: !GPU_NAME! (driver !DRIVER!)"

    REM Rough driver-line check: shipped backend (cu128) needs driver 550+
    for /f "tokens=1 delims=." %%A in ("!DRIVER!") do set "DRIVER_MAJOR=%%A"
    if defined DRIVER_MAJOR (
        if !DRIVER_MAJOR! lss 550 (
            call :warn "driver !DRIVER! is older than 550 — shipped backend (cu128) will FAIL."
            call :warn "  Either rebuild backend with cu117 or deploy in CPU mode (start-no-gpu.bat)."
        )
    )

    REM nvidia container runtime bridged into Docker Desktop is automatic
    REM with WSL2 backend + a Windows GPU driver >= r460. Detect via docker info.
    docker info 2>nul | findstr /i "nvidia" >nul 2>&1
    if !errorlevel! equ 0 (
        call :pass "nvidia container runtime available to Docker"
    ) else (
        call :warn "nvidia runtime not detected in docker info — GPU may still work under Docker Desktop WSL2."
        call :warn "  Verify: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu20.04 nvidia-smi"
    )
)

REM ─── Disk space ───────────────────────────────────────────────
echo.
echo == Disk ==
for /f "tokens=3" %%A in ('dir /-C ^| findstr "bytes free"') do set "FB=%%A"
if not defined FB (
    call :warn "could not determine free disk space"
) else (
    REM Length-based comparison: 11 digits ~= 10 GB, 12 = 100 GB.
    REM Need at least 30 GB, so require length >= 11 (10+ GB) with warn under 60 GB.
    set "FBLEN=0"
    set "TMP=!FB!"
    for /l %%i in (0,1,20) do if defined TMP (
        set "TMP=!TMP:~1!"
        set /a "FBLEN+=1"
    )
    if !FBLEN! lss 11 (
        call :fail "less than 10 GB free — install needs at least 30 GB"
    ) else (
        call :pass "disk space appears sufficient (!FB! bytes free)"
    )
)

REM ─── Port collisions ──────────────────────────────────────────
echo.
echo == Ports ==
for %%P in (3030:frontend 5100:backend 5200:ti-modelmaker 1883:mosquitto-mqtt 9001:mosquitto-ws) do (
    for /f "tokens=1,2 delims=:" %%A in ("%%P") do (
        set "PORT=%%A"
        set "SVC=%%B"
        netstat -an | findstr ":!PORT! " | findstr "LISTENING" >nul 2>&1
        if !errorlevel! equ 0 (
            call :fail "port !PORT! already in use — will block !SVC!"
        ) else (
            call :pass "port !PORT! free (for !SVC!)"
        )
    )
)

REM ─── Release tarballs ─────────────────────────────────────────
echo.
echo == Release tarballs ==
set /a TARS_FOUND=0
for %%T in (cirame-backend.tar cirame-frontend.tar cirame-ti-modelmaker.tar cirame-mosquitto.tar) do (
    if exist "%%T" (
        set /a TARS_FOUND+=1
        for %%A in ("%%T") do (
            if "%%~zA"=="0" (
                call :fail "%%T exists but is 0 bytes"
            ) else (
                call :pass "%%T present"
            )
        )
    )
)
if !TARS_FOUND! equ 0 (
    call :warn "no release tarballs in this folder — this is fine if you plan to build from source"
) else (
    if not exist cirame-backend.tar (
        call :fail "cirame-backend.tar missing — required"
    )
    if not exist cirame-frontend.tar (
        call :fail "cirame-frontend.tar missing — required"
    )
)

REM ─── Summary ─────────────────────────────────────────────────
echo.
echo ============================================
echo   Summary: !PASS_COUNT! PASS, !WARN_COUNT! WARN, !FAIL_COUNT! FAIL
echo ============================================

if !FAIL_COUNT! gtr 0 (
    echo.
    echo Fix the FAIL items above before running install.bat.
    pause
    exit /b 1
)
if !WARN_COUNT! gtr 0 (
    echo.
    echo You can install now, but consider addressing WARN items first.
    pause
    exit /b 0
)
echo.
echo Ready for install. Next: install.bat
pause
exit /b 0
