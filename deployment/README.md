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

## Installation

### Windows

```
1. Install Docker Desktop (https://docker.com)
2. Double-click install.bat
3. Double-click start.bat
4. Open http://localhost:3030
```

### Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install and start CiRA ME
cd deployment
bash install.sh
bash start.sh

# Access at http://localhost:3030
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
| Update images | `update.bat` | `bash update.sh` |

## Network Ports

| Port | Service | Purpose |
|---|---|---|
| **3030** | Frontend (nginx) | Web application |
| 5100 | Backend (Flask) | REST API |
| 5200 | TI ModelMaker | TI MCU training (optional) |
| **1883** | Mosquitto | MQTT TCP (for sensors/devices) |
| **9001** | Mosquitto | MQTT WebSocket (for browsers) |

Ports in **bold** should be accessible from the local network.

## Datasets

Place CSV or CBOR datasets in the `shared/` folder. They will appear in CiRA ME under "Browse Files > shared".

## MQTT Live Streaming

If `cirame-mosquitto.tar` is installed:

1. IoT devices/sensors connect to `mqtt://server-ip:1883`
2. Published apps connect to `ws://server-ip:9001/mqtt`
3. Manage broker at the "MQTT Broker" page in CiRA ME

## TI MCU Support

If `cirame-ti-modelmaker.tar` is installed:

1. Train TI model zoo models (Conv1D, MLP) in the Training page
2. Export C code packages for Code Composer Studio
3. Deploy to TMS320 F28379D, F280049C, F28P55x

## Troubleshooting

### Docker images not loading
```
# Check disk space
df -h                          # Linux
wmic logicaldisk get size,freespace  # Windows

# Check Docker is running
docker info
```

### GPU not detected
```
# Check NVIDIA driver
nvidia-smi

# Check nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Services won't start
```
# Check logs
docker compose logs backend
docker compose logs frontend

# Restart
bash stop.sh
bash start.sh
```

### Cannot access from other machines
- Check firewall allows ports 3030, 1883, 9001
- Use server IP instead of localhost: `http://192.168.x.x:3030`

## Support

CiRA — Center of Innovative Robotics and Automation
