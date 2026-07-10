#!/bin/bash
# CiRA ME - Pre-install Validation Script (Linux/macOS)
#
# Runs every pre-flight check that install.sh / start.sh will make, with
# ZERO side effects. Customer runs this on their box first to know whether
# it's ready — no partial installs, no half-loaded images.
#
# Exit codes:
#   0  everything green (PASS or WARN only)
#   1  at least one FAIL — install.sh would refuse to proceed

echo "============================================"
echo "  CiRA ME - Pre-install Validation"
echo "============================================"
echo

FAIL_COUNT=0
WARN_COUNT=0
PASS_COUNT=0

# Colored output when the terminal supports it. Fall back to plain ASCII
# so log files stay readable and putty sessions do not fill with escapes.
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    C_PASS=$(tput setaf 2)
    C_WARN=$(tput setaf 3)
    C_FAIL=$(tput setaf 1)
    C_OFF=$(tput sgr0)
else
    C_PASS="" C_WARN="" C_FAIL="" C_OFF=""
fi

pass() { echo "  ${C_PASS}PASS${C_OFF}  $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn() { echo "  ${C_WARN}WARN${C_OFF}  $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { echo "  ${C_FAIL}FAIL${C_OFF}  $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

# ─── User environment ──────────────────────────────────────────────
echo "== User environment =="
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    fail "running as root — install.sh will refuse. Run as a regular user."
else
    pass "not running as root"
fi

# ─── Docker daemon ─────────────────────────────────────────────────
echo
echo "== Docker =="
if ! command -v docker >/dev/null 2>&1; then
    fail "docker command not found. Install: https://docs.docker.com/engine/install/"
elif ! docker info >/dev/null 2>&1; then
    fail "docker daemon not reachable — either stopped, or your user lacks access."
    fail "  Fix (daemon down):  sudo systemctl start docker"
    fail "  Fix (permissions):  sudo usermod -aG docker \$USER && newgrp docker"
else
    ver=$(docker --version | awk '{print $3}' | tr -d ,)
    pass "docker daemon reachable ($ver)"
fi

# ─── docker compose v2 / v1 detection ──────────────────────────────
if docker compose version >/dev/null 2>&1; then
    pass "docker compose v2 plugin present"
elif command -v docker-compose >/dev/null 2>&1; then
    warn "using docker-compose v1 (works, but v2 plugin is recommended)"
else
    fail "docker compose is not installed."
    fail "  Fix: sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \\"
    fail "         -o /usr/local/lib/docker/cli-plugins/docker-compose \\"
    fail "       && sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose"
fi

# ─── GPU + driver + CUDA capability ────────────────────────────────
echo
echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no NVIDIA GPU or driver detected — plan to use start-no-gpu.sh (CPU mode)"
else
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    # nvidia-smi's "CUDA Version" is the max CUDA runtime this driver
    # supports (not the CUDA toolkit installed). This determines which
    # torch cu build works: cu128 needs driver >= 550, cu117 needs >= 515.
    cuda_max=$(nvidia-smi 2>/dev/null | awk -F'CUDA Version:' 'NR<=3 {print $2}' | tr -d ' |' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    pass "GPU detected: $gpu (driver $driver, max CUDA $cuda_max)"

    # Rough driver-line check. The stock backend tarball ships with
    # torch+cu128 which needs driver 550+. Anything below and the customer
    # will need a rebuilt backend with cu117.
    driver_major=$(echo "$driver" | cut -d. -f1)
    if [ -n "$driver_major" ] && [ "$driver_major" -lt 550 ] 2>/dev/null; then
        warn "driver $driver is older than 550 — the shipped backend (cu128) will FAIL."
        warn "  You need a backend image built with cu117:"
        warn "    docker build --build-arg TORCH_VERSION=2.0.1 --build-arg TORCH_CUDA=cu117 \\"
        warn "                 -t cirame-backend:latest backend/"
        warn "  Or deploy in CPU mode: bash start-no-gpu.sh"
    fi

    # nvidia-container-toolkit — Docker's runtime bridge to the GPU.
    if docker info 2>/dev/null | grep -qE 'Runtimes:.* nvidia'; then
        pass "nvidia container runtime registered with Docker"
    else
        fail "nvidia-container-toolkit is not configured on the Docker daemon."
        fail "  Fix (Ubuntu):"
        fail "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
        fail "         | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        fail "    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\"
        fail "         | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\"
        fail "         | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        fail "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        fail "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
fi

# ─── Disk space ────────────────────────────────────────────────────
echo
echo "== Disk =="
free_kb=$(df -k . | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
if [ "$free_gb" -lt 30 ]; then
    fail "only ${free_gb} GB free at $(pwd) — install needs at least 30 GB"
elif [ "$free_gb" -lt 60 ]; then
    warn "${free_gb} GB free — enough to install but leaves little room for data growth"
else
    pass "${free_gb} GB free at $(pwd)"
fi

# ─── Port collisions ───────────────────────────────────────────────
echo
echo "== Ports =="
port_free() {
    if command -v ss >/dev/null 2>&1; then
        ! ss -tuln 2>/dev/null | awk -v p=":$1 " '$0 ~ p {found=1} END {exit !found}'
    else
        ! netstat -tuln 2>/dev/null | awk -v p=":$1 " '$0 ~ p {found=1} END {exit !found}'
    fi
}
for port_and_svc in "3030:frontend" "5100:backend" "5200:ti-modelmaker" "1883:mosquitto-mqtt" "9001:mosquitto-ws"; do
    port=${port_and_svc%:*}
    svc=${port_and_svc#*:}
    if port_free "$port"; then
        pass "port $port free (for $svc)"
    else
        fail "port $port already in use — will block $svc"
        if [ "$port" = "1883" ]; then
            fail "  Common: host mosquitto service. Stop: sudo systemctl stop mosquitto && sudo systemctl disable mosquitto"
        fi
    fi
done

# ─── Tarballs — all 4 are required for a tarball install ────────────
# TI + Mosquitto used to be optional. Customers reported the silent-skip
# guard was leaving MQTT and TI features broken with no error to point at,
# so both are now mandatory. If you're building from source, ignore this
# section — no tarballs is a valid state for developers.
echo
echo "== Release tarballs =="
tarballs_found=0
required_missing=""
for tar in cirame-backend.tar cirame-frontend.tar cirame-ti-modelmaker.tar cirame-mosquitto.tar; do
    if [ -f "$tar" ]; then
        tarballs_found=$((tarballs_found + 1))
        if [ ! -s "$tar" ]; then
            fail "$tar exists but is 0 bytes"
        else
            size=$(du -h "$tar" | awk '{print $1}')
            pass "$tar present ($size)"
        fi
    else
        required_missing="$required_missing $tar"
    fi
done
if [ "$tarballs_found" -eq 0 ]; then
    warn "no release tarballs in this folder — this is fine if you plan to build from source"
elif [ -n "$required_missing" ]; then
    fail "required tarballs missing:$required_missing"
fi

# ─── Summary ───────────────────────────────────────────────────────
echo
echo "============================================"
echo "  Summary: ${C_PASS}${PASS_COUNT} PASS${C_OFF}, ${C_WARN}${WARN_COUNT} WARN${C_OFF}, ${C_FAIL}${FAIL_COUNT} FAIL${C_OFF}"
echo "============================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo
    echo "Fix the FAIL items above before running install.sh."
    exit 1
elif [ "$WARN_COUNT" -gt 0 ]; then
    echo
    echo "You can install now, but consider addressing WARN items first."
    exit 0
else
    echo
    echo "Ready for install. Next: bash install.sh"
    exit 0
fi
