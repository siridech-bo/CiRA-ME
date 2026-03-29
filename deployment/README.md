# CiRA ME — Deployment Package

## Machine Intelligence for Edge Computing

---

## Package Contents

| File | Size | Required |
|---|---|---|
| `cirame-backend.tar` | ~14 GB | Yes |
| `cirame-frontend.tar` | ~93 MB | Yes |
| `cirame-ti-modelmaker.tar` | ~11.5 GB | Optional (TI MCU) |
| `cirame-mosquitto.tar` | ~36 MB | Optional (MQTT) |
| `docker-compose.yml` | — | GPU servers |
| `docker-compose-no-gpu.yml` | — | CPU-only servers |
| `shared/` | — | Customer datasets |
| `mosquitto/` | — | MQTT broker config |

## Requirements

- **Docker** 24.0+ with Docker Compose v2
- **NVIDIA GPU** (optional) with nvidia-container-toolkit for GPU acceleration
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 30 GB free space
- **Ports**: 3030 (web), 5100 (API), 1883 (MQTT TCP), 9001 (MQTT WebSocket)

Step 1: install.bat (or bash install.sh)
  ├── [1/4] Stop old containers (if any running)
  ├── [2/4] Remove old images (prevents version conflicts)
  ├── [3/4] Load new images from .tar files
  │         ├── cirame-backend.tar     (required)
  │         ├── cirame-frontend.tar    (required)
  │         ├── cirame-ti-modelmaker.tar (optional, skip if missing)
  │         └── cirame-mosquitto.tar   (optional, skip if missing)
  └── [4/4] Create folders + mosquitto config

Step 2: start.bat (or bash start.sh)
  └── docker compose up -d

Customer data preserved:
  ├── Docker volumes (database, models) survive image replacement
  ├── shared/ folder (datasets) on host disk
  └── mosquitto/ folder (broker config) on host disk



## Installation (Fresh Install or Upgrade)

The install script handles both fresh installation and upgrades.
It is safe to re-run — it stops the old version first, then loads the new images.

**Your data (trained models, database, datasets) is preserved across upgrades.**

### What the installer does:

```
Step 1: Stop old containers (if any previous version is running)
         ├── docker compose down (both GPU and no-GPU variants)
         ├── docker stop cirame-backend cirame-frontend ...
         └── docker rm cirame-backend cirame-frontend ...

Step 2: Remove old images (prevents version conflicts)
         ├── docker rmi cirame-backend:latest
         ├── docker rmi cirame-frontend:latest
         └── docker rmi cirame-ti-modelmaker:latest

Step 3: Load new images from .tar files
         ├── cirame-backend.tar      (required — fails if missing)
         ├── cirame-frontend.tar     (required — fails if missing)
         ├── cirame-ti-modelmaker.tar (optional — skips if missing)
         └── cirame-mosquitto.tar    (optional — skips if missing)

Step 4: Create folders and configuration
         ├── shared/                 (dataset folder)
         └── mosquitto/mosquitto.conf (MQTT broker config)
```

### Windows

```
1. Install Docker Desktop (https://docker.com)
2. Double-click install.bat     ← runs steps 1-4 above
3. Double-click start.bat       ← starts all services
4. Open http://localhost:3030
```

### Linux

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sh

# Install CiRA ME
cd deployment
bash install.sh       # runs steps 1-4 above
bash start.sh         # starts all services

# Access at http://localhost:3030
```

## Data Preservation

Docker volumes store your data separately from images:

| Data | Storage | Survives upgrade? |
|---|---|---|
| Database (users, models, endpoints) | `backend-data` volume | Yes |
| Trained model files (.pkl, .onnx) | `backend-models` volume | Yes |
| TI training projects | `ti-projects` volume | Yes |
| MQTT broker data | `mosquitto-data` volume | Yes |
| Datasets (CSV, CBOR files) | `shared/` folder on host | Yes |
| MQTT config | `mosquitto/` folder on host | Yes |

To **reset all data** (fresh start), remove volumes:
```bash
docker compose down -v    # WARNING: deletes all data
```

## Default Login

```
Username: admin
Password: admin123
```

**Change the password after first login!**

## Management Commands

| Action | Windows | Linux |
|---|---|---|
| Start (GPU) | `start.bat` | `bash start.sh` |
| Start (CPU) | `start-no-gpu.bat` | `bash start-no-gpu.sh` |
| Stop | `stop.bat` | `bash stop.sh` |
| View logs | `logs.bat` | `docker compose logs -f` |
| Check status | `status.bat` | `docker compose ps` |
| Uninstall | `uninstall.bat` | `bash uninstall.sh` |
| Update images | Place new .tar files, run `install.bat` again |

## Network Ports

| Port | Service | Purpose |
|---|---|---|
| **3030** | Frontend (nginx) | Web application |
| 5100 | Backend (Flask) | REST API |
| 5200 | TI ModelMaker | TI MCU training (optional) |
| **1883** | Mosquitto | MQTT TCP (for sensors/devices) |
| **9001** | Mosquitto | MQTT WebSocket (for browsers) |

Ports in **bold** should be accessible from the local network for other machines to connect.

## Datasets

Place CSV or CBOR datasets in the `shared/` folder. They will appear in CiRA ME under "Browse Files > shared".

## MQTT Live Streaming

If `cirame-mosquitto.tar` is installed:

1. IoT devices/sensors connect to `mqtt://server-ip:1883`
2. Published apps connect to `ws://server-ip:9001/mqtt` (auto-resolved in browser)
3. Manage broker at the "MQTT Broker" page in CiRA ME
4. Mobile sensor apps (e.g., SensorSpot) can stream data to the broker

## TI MCU Support

If `cirame-ti-modelmaker.tar` is installed:

1. Train TI model zoo models (Conv1D, MLP) in the Training page
2. Export C code packages for Code Composer Studio
3. Deploy to TMS320 F28379D, F280049C, F28P55x

## Troubleshooting

### Docker images not loading
```bash
# Check disk space (need ~30 GB free)
df -h                                    # Linux
wmic logicaldisk get size,freespace      # Windows

# Check Docker is running
docker info
```

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If GPU not needed, use no-GPU variant:
# Windows: start-no-gpu.bat
# Linux:   bash start-no-gpu.sh
```

### Services won't start
```bash
# Check logs for errors
docker compose logs backend
docker compose logs frontend

# Restart everything
bash stop.sh
bash start.sh
```

### Cannot access from other machines
1. Check firewall allows ports **3030**, **1883**, **9001**
2. Use server IP instead of localhost: `http://192.168.x.x:3030`
3. On Windows: check Docker Desktop network settings

### Old version still showing after upgrade
```bash
# Force remove old containers and restart
docker compose down
docker compose up -d --force-recreate
```

### Database or models lost after upgrade
This should NOT happen — data is stored in Docker volumes.
If it does, check that you did NOT run `docker compose down -v` (the `-v` flag deletes volumes).

## Support

CiRA — Center of Innovative Robotics and Automation
