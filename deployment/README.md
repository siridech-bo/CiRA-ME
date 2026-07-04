# CiRA ME — Deployment Package

## Machine Intelligence for Edge Computing

---

## Package Contents

| File / Folder | Size | Required | Purpose |
|---|---|---|---|
| `cirame-backend.tar` | ~14 GB | Yes | Flask API + ML pipeline + F1-F4 features |
| `cirame-frontend.tar` | ~93 MB | Yes | Vue 3 web UI |
| `cirame-ti-modelmaker.tar` | ~11.5 GB | Optional | TI TMS320 MCU export |
| `cirame-mosquitto.tar` | ~36 MB | Optional | MQTT broker for live streaming |
| `docker-compose.yml` | — | GPU | Deployment config with NVIDIA passthrough |
| `docker-compose-no-gpu.yml` | — | CPU | Deployment config, CPU-only |
| `data/` | — | | User data (created on first run) |
| `datasets/` | — | | Uploaded CSVs / CBOR files (created on first run) |
| `watcher-data/` | — | | Folder Watcher input/output (created on first run) |
| `mosquitto/` | — | | MQTT broker config |
| `MIGRATION.md` | — | | Update & data-migration guide |

### Scripts included

| Script | Purpose |
|---|---|
| `validate.sh` / `.bat` | **Run this first.** Pre-flight checks with zero side effects. |
| `install.sh` / `.bat` | Fresh install: load images, create folders, set up config. |
| `start.sh` / `.bat` | Start with GPU support. |
| `start-no-gpu.sh` / `.bat` | Start CPU-only. |
| `stop.sh` / `.bat` | Stop all containers. |
| `status.sh` / `.bat` | Show container health. |
| `logs.sh` / `.bat` | View live logs (accepts service name + `--no-follow`). |
| `update.sh` / `.bat` | In-place update. **Auto-snapshots the database first.** |
| `migrate.sh` / `.bat` | Copy data from a different deployment folder. |
| `backup.sh` / `.bat` | One-shot tarball of all customer data. |
| `uninstall.sh` / `.bat` | Remove containers/images. Data preserved by default. |
| `export.sh` / `.bat` | Rebuild + save tarballs from source (developer use). |

## Requirements

- **Docker** 24.0+ with **Docker Compose v2** (the install script installs v2 as a static binary if missing on Linux)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 30 GB free
- **Ports**: 3030 (web), 5100 (API), 5200 (TI), 1883 (MQTT TCP), 9001 (MQTT WebSocket) — all must be free
- **User** (Linux): must be in the `docker` group (`sudo usermod -aG docker $USER && newgrp docker`). **Do NOT run scripts with `sudo`** — they refuse.

### GPU requirements (optional)

If you want GPU acceleration:

- **NVIDIA GPU** with `nvidia-container-toolkit` installed on the host
- **NVIDIA driver 550+** for the shipped backend (which is built with CUDA 12.8)
- **If your driver is 515.x–545.x**, you have two options:
  - Upgrade the driver to 550+ (requires reboot), OR
  - Rebuild the backend with CUDA 11.7 support (see [Rebuilding the Backend for an Older Driver](#rebuilding-the-backend-for-an-older-driver) below)
- **If no GPU / any driver**: use `start-no-gpu.sh` / `start-no-gpu.bat`. Everything works except deep-learning training runs slower on CPU.

`validate.sh` / `validate.bat` reports driver version and warns about mismatches automatically.

---

## Installation

### Step 0 — Validate (recommended)

Run `validate.sh` / `validate.bat` first to check that your box is ready:

```bash
# Linux / macOS
cd deployment
bash validate.sh

# Windows
cd deployment
validate.bat
```

The output is a PASS/WARN/FAIL report:

```
== User environment ==   PASS  not running as root
== Docker ==             PASS  docker daemon reachable (24.0.7)
                         PASS  docker compose v2 plugin present
== GPU ==                PASS  GPU detected: RTX 5070 Ti (driver 576.88, max CUDA 12.9)
                         PASS  nvidia container runtime registered with Docker
== Disk ==               PASS  1570 GB free
== Ports ==              PASS  port 3030 free
                         ... etc
== Release tarballs ==   PASS  cirame-backend.tar present (4.5G)
                         PASS  cirame-frontend.tar present (26M)

Summary: 12 PASS, 0 WARN, 0 FAIL
Ready for install. Next: bash install.sh
```

If any `FAIL` appears, fix it before running `install.sh` — the install would refuse anyway.

### Step 1 — Install

Fresh install of a new machine, or reinstall over an existing one:

```bash
# Linux / macOS
bash install.sh

# Windows
install.bat
```

The install script performs pre-flight checks (docker running, compose available, disk space, port availability, tarball integrity), stops any previous version, loads all four images, and creates the data folders:

```
[Pre-flight]
  - Not running as root
  - Docker daemon reachable
  - docker compose v2 or v1 available
  - ≥ 30 GB free
  - All required tarballs present and non-empty

[1/4] Stop previous version (if any)
[2/4] Remove old images
[3/4] Load new images (verifies each landed via docker image inspect)
[4/4] Create data/ , datasets/ , watcher-data/ , mosquitto/mosquitto.conf
      (auto-migrates legacy ./shared/ → ./datasets/shared/ if found)
```

**Your data is preserved on host disk** across upgrades — the install script doesn't touch `data/`, `datasets/`, or `watcher-data/` after first creation.

### Step 2 — Start

Pick GPU or CPU based on your hardware:

```bash
# GPU (requires driver 550+ for shipped backend)
bash start.sh          # or: start.bat

# CPU-only
bash start-no-gpu.sh   # or: start-no-gpu.bat
```

`start.sh` first probes port availability and (in GPU mode) verifies the nvidia container runtime is registered. It'll bail with a specific error if anything looks wrong.

Then open **http://localhost:3030** and log in with `admin` / `admin123` (change immediately).

---

## Updating to a New Version

Both update paths **auto-snapshot the database** before touching anything, so a broken new image is a two-command rollback away.

### Option A — In-place update (recommended)

Drop the new `.tar` files into your **existing** deployment folder:

```bash
bash update.sh         # Linux
update.bat             # Windows
```

The update script:

1. **Snapshots `data/database/` to `data/database.backup.<timestamp>/`**
2. Stops the running app (data on disk preserved)
3. Loads the new images and verifies each landed
4. Prunes old image layers

Rollback (if the new version fails to boot):

```bash
bash stop.sh
rm -rf data/database
mv data/database.backup.<timestamp> data/database
bash start.sh
```

### Option B — Extract to a new folder + migrate

```bash
# 1. Extract the new release to a new folder
# 2. Copy your data over
bash migrate.sh /path/to/old/deployment
# 3. Load images
bash install.sh
# 4. Start
bash start.sh
```

`migrate.sh` refuses if source and target are the same folder (would copy files over themselves), and **aborts on any copy failure** so you don't end up with a half-migrated database. It copies `data/`, `datasets/`, `watcher-data/`, and legacy `shared/` (auto-converts to `datasets/shared/`).

### Detailed guide

See **[MIGRATION.md](./MIGRATION.md)** for the full guide, including checksums and troubleshooting.

---

## Backing Up Data

Before any risky operation, run:

```bash
bash backup.sh                          # writes to backups/cirame-YYYYMMDD-HHMMSS.tar.gz
bash backup.sh /path/to/dest.tar.gz     # explicit destination
```

The backup includes everything under `data/`, `datasets/`, `watcher-data/`, `mosquitto/`, and both compose files. It optionally stops containers first so the SQLite DB is quiesced (recommended).

To restore, extract the tarball into a deployment folder:

```bash
tar xzf cirame-20260704-201530.tar.gz -C /target/deployment/folder
```

---

## Uninstalling

**By default, uninstall preserves your data on disk.** Only containers and images are removed:

```bash
bash uninstall.sh      # or: uninstall.bat
```

You'll be asked to type `REMOVE` (exact case) as confirmation — a typed phrase avoids "yes"-autopilot.

To **also delete your data** (unrecoverable — run `backup.sh` first!):

```bash
bash uninstall.sh --purge-data       # or: uninstall.bat --purge-data
```

You'll be asked to type `PURGE` for confirmation.

---

## Data Preservation

All user data lives on the host disk (bind-mounted into containers):

| Data | Host folder | Survives upgrade? |
|---|---|---|
| Database (users, models, endpoints, apps, projects) | `data/database/` | Yes |
| Trained model files (.pkl, .onnx) | `data/models/` | Yes |
| TI training projects | `data/ti-projects/` | Yes |
| MQTT broker data | `data/mosquitto/` | Yes |
| Datasets (shared + user private folders) | `datasets/` | Yes |
| Folder Watcher input/output/errors | `watcher-data/` | Yes |
| MQTT broker config | `mosquitto/mosquitto.conf` | Yes |

To **reset all data** (fresh start): `bash uninstall.sh --purge-data`, then re-run `install.sh`.

---

## Default Login

```
Username: admin
Password: admin123
```

**Change the password after first login!**

---

## Management Commands

| Action | Windows | Linux |
|---|---|---|
| Validate box before install | `validate.bat` | `bash validate.sh` |
| **Backup data** | `backup.bat` | `bash backup.sh` |
| Start (GPU) | `start.bat` | `bash start.sh` |
| Start (CPU) | `start-no-gpu.bat` | `bash start-no-gpu.sh` |
| Stop | `stop.bat` | `bash stop.sh` |
| Check status | `status.bat` | `bash status.sh` |
| View live logs | `logs.bat` | `bash logs.sh` |
| View logs, one service | `logs.bat backend` | `bash logs.sh backend` |
| Dump recent logs and exit | `logs.bat --no-follow` | `bash logs.sh --no-follow` |
| **Update** (new .tar in same folder) | `update.bat` | `bash update.sh` |
| **Migrate** (copy from another folder) | `migrate.bat <old_folder>` | `bash migrate.sh <old_folder>` |
| Fresh install / reinstall | `install.bat` | `bash install.sh` |
| Uninstall (preserve data) | `uninstall.bat` | `bash uninstall.sh` |
| Uninstall + delete data | `uninstall.bat --purge-data` | `bash uninstall.sh --purge-data` |

---

## Network Ports

| Port | Service | Purpose |
|---|---|---|
| **3030** | Frontend (nginx) | Web application |
| 5100 | Backend (Flask) | REST API |
| 5200 | TI ModelMaker | TI MCU training (optional) |
| **1883** | Mosquitto | MQTT TCP (for sensors/devices) |
| **9001** | Mosquitto | MQTT WebSocket (for browsers) |

Ports in **bold** should be reachable from the local network for other machines to connect. `validate.sh` and `start.sh` both check that these are free before proceeding.

---

## Datasets

Place CSV or CBOR datasets in the `datasets/shared/` folder. They'll appear in CiRA ME under "Browse Files → shared".

User private folders are stored in `datasets/<folder_name>/` — each user with a private folder gets their own subfolder.

---

## MQTT Live Streaming

If `cirame-mosquitto.tar` is installed:

1. IoT devices/sensors connect to `mqtt://server-ip:1883`
2. Published apps connect to `ws://server-ip:9001/mqtt` (auto-resolved in browser)
3. Manage broker at the "MQTT Broker" page in CiRA ME
4. Mobile sensor apps (e.g., SensorSpot) can stream data to the broker

---

## TI MCU Support

If `cirame-ti-modelmaker.tar` is installed:

1. Train TI model zoo models (Conv1D, MLP) in the Training page
2. Export C code packages for Code Composer Studio
3. Deploy to TMS320 F28379D, F280049C, F28P55x

---

## Rebuilding the Backend for an Older Driver

The shipped backend is built with PyTorch 2.10 + CUDA 12.8, which requires **NVIDIA driver 550+**. If your host is stuck on driver 515.x–545.x, you can rebuild the backend against CUDA 11.7 without upgrading the driver:

```bash
# From the repo root (not the deployment folder — this needs the source)
docker build \
    --network=host \
    --build-arg TORCH_VERSION=2.0.1 \
    --build-arg TORCH_CUDA=cu117 \
    -t cirame-backend:latest \
    backend/

# Then save it out for deployment
docker save cirame-backend:latest -o deployment/cirame-backend.tar
```

The `--network=host` flag is a workaround for hosts whose Docker default bridge network can't reach external DNS during `apt-get update` inside the build. It's harmless on hosts where the bridge is fine.

You get identical functionality — TimesNet training, GPU inference, TI export all work. Only the CUDA runtime version differs.

---

## Troubleshooting

### First thing to try

Run `validate.sh` / `validate.bat`. It reports every common misconfiguration with the exact fix command.

### Docker images not loading

```bash
# Check disk space (need ~30 GB free)
df -h                                    # Linux
wmic logicaldisk get size,freespace      # Windows

# Check Docker is running
docker info
```

### GPU not detected / "could not select device driver"

```bash
# Confirm host driver
nvidia-smi

# Confirm nvidia-container-toolkit is registered
docker info | grep -i nvidia
# → should show:  Runtimes: io.containerd.runc.v2 nvidia runc

# Test GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu20.04 nvidia-smi

# If it fails: reinstall nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Or run CPU-only:
bash start-no-gpu.sh
```

### Backend crashes with CUDA error on first inference

Your NVIDIA driver is too old for the shipped backend (needs 550+). See [Rebuilding the Backend for an Older Driver](#rebuilding-the-backend-for-an-older-driver) above, or run in CPU mode.

### Port 1883 already in use

Common on Ubuntu / Debian — a host `mosquitto` package is installed and running as a systemd service. To free it for our container:

```bash
sudo systemctl stop mosquitto
sudo systemctl disable mosquitto
```

### Services won't start / backend unhealthy

```bash
# See what's happening
bash logs.sh backend --no-follow | tail -50

# Or restart
bash stop.sh
bash start.sh
```

The backend `start_period` is 120s — first boot on a fresh install takes 60–90s (SQLite migrations + torch import). Give it 2 minutes before assuming it's stuck.

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

This should **not** happen — data is stored on the host disk in `./data/` and `./datasets/`. Common causes:

1. You extracted the new release to a **different folder**. Run `migrate.sh <old_folder>` to copy data over.
2. You ran `install.sh` in an empty new folder instead of `update.sh`. `install.sh` is safe too, but only if data folders already exist.
3. You accidentally deleted the `data/` or `datasets/` folder. Restore from your most recent `backup.sh` output (`backups/cirame-*.tar.gz`).

See [MIGRATION.md](./MIGRATION.md) for the full migration guide.

---

## Support

CiRA — Center of Innovative Robotics and Automation
