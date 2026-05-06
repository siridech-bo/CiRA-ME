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
| `datasets/` | — | User-uploaded datasets (created on first run) |
| `data/` | — | Database, models, MQTT data (created on first run) |
| `mosquitto/` | — | MQTT broker config |
| `MIGRATION.md` | — | Update & data-migration guide |

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

Customer data preserved on host disk (bind-mounted):
  ├── data/database/    (SQLite database)
  ├── data/models/      (trained model files)
  ├── data/ti-projects/ (TI training projects)
  ├── data/mosquitto/   (MQTT broker persistent data)
  ├── datasets/         (user-uploaded datasets, shared + private folders)
  └── mosquitto/        (broker config)



## Installation (Fresh Install)

For a fresh installation on a new machine, run `install.bat` (Windows) or `bash install.sh` (Linux).

**To upgrade an existing installation, see the [Updating to a New Version](#updating-to-a-new-version) section below.**

The install script is safe to re-run — it stops the old version first, then loads the new images.
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
         ├── data/database/          (SQLite database)
         ├── data/models/            (trained model files)
         ├── data/ti-projects/       (TI ModelMaker projects)
         ├── data/mosquitto/         (MQTT broker persistent data)
         ├── datasets/shared/        (shared datasets folder)
         └── mosquitto/mosquitto.conf (MQTT broker config)
         (auto-migrates legacy ./shared/ -> ./datasets/shared/ if found)
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

## Updating to a New Version

You will receive a new package containing updated `.tar` image files.
There are **two ways** to update — both preserve all your data
(users, models, datasets, endpoints, apps, MQTT history):

### Option A — In-place update (recommended, simplest)

Drop the new `.tar` files into your **existing** deployment folder and run:

```
update.bat              (Windows)
bash update.sh          (Linux / macOS)
```

That's it. Then run `start.bat` (or `start-no-gpu.bat`) to launch the new version.

### Option B — Extract new release to a new folder

If you prefer to extract each release to its own folder for safety:

```
1. Extract the new release into a new folder
2. Open a terminal in the NEW folder
3. Copy data from the OLD folder:
       migrate.bat "C:\path\to\old\deployment"        (Windows)
       bash migrate.sh /path/to/old/deployment        (Linux / macOS)
4. Load the new images:
       install.bat   (or update.bat if images are already loaded)
5. Start:
       start.bat
```

`migrate` copies `data/`, `datasets/`, legacy `shared/` (auto-converts), and
`mosquitto.conf` from the old folder. The old folder is **not deleted** — it
remains as a backup.

### Detailed guide

See **`MIGRATION.md`** for the full update & migration guide, including
backup checklist and troubleshooting.

## Data Preservation

All user data lives on the host disk (bind-mounted into containers):

| Data | Host folder | Survives upgrade? |
|---|---|---|
| Database (users, models, endpoints, apps) | `data/database/` | Yes |
| Trained model files (.pkl, .onnx) | `data/models/` | Yes |
| TI training projects | `data/ti-projects/` | Yes |
| MQTT broker data | `data/mosquitto/` | Yes |
| Datasets (shared + user private folders) | `datasets/` | Yes |
| MQTT broker config | `mosquitto/mosquitto.conf` | Yes |

To **reset all data** (fresh start), delete the `data/` and `datasets/` folders
manually, then re-run `install.bat`.

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
| View logs | `logs.bat` | `bash logs.sh` |
| Check status | `status.bat` | `bash status.sh` |
| **Update** (new .tar files in same folder) | `update.bat` | `bash update.sh` |
| **Migrate** (copy data from another deployment folder) | `migrate.bat <old_folder>` | `bash migrate.sh <old_folder>` |
| Fresh install / reinstall | `install.bat` | `bash install.sh` |
| Uninstall | `uninstall.bat` | `bash uninstall.sh` |

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

Place CSV or CBOR datasets in the `datasets/shared/` folder. They will appear in CiRA ME under "Browse Files > shared".

User private folders are stored in `datasets/<folder_name>/` — each user with a private folder will have their own subfolder there.

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
This should NOT happen — data is stored on the host disk in `./data/` and `./datasets/`.
Common causes:
1. You extracted the new release to a **different folder**. Run `migrate.bat <old_folder>`
   to copy your data over (see the **Updating to a New Version** section above).
2. You ran `install.bat` in an empty new folder instead of `update.bat`.
   `install.bat` is also safe — but only if data folders already exist there.
3. You accidentally deleted the `data/` or `datasets/` folder.

See **`MIGRATION.md`** for the full migration guide.

## Support

CiRA — Center of Innovative Robotics and Automation
