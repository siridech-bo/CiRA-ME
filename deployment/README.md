# CiRA ME - Deployment Guide

Machine Intelligence for Edge Computing

---

## Overview

CiRA ME is distributed as pre-built Docker images. No build tools or internet
access are required on the customer server — just Docker.

**Workflow:**
```
[Vendor] export.bat / export.sh  →  cirame-backend.tar + cirame-frontend.tar
            ↓  transfer deployment/ folder
[Customer] install.bat / install.sh  →  start.bat / start.sh
```

---

## For Vendors: Building and Exporting Images

Run from the **project root** (where `docker-compose.yml` lives):

**Windows:**
```cmd
deployment\export.bat
```

**Linux / macOS:**
```bash
bash deployment/export.sh
```

This builds both images and saves them as:
- `deployment/cirame-backend.tar` (~600 MB)
- `deployment/cirame-frontend.tar` (~50 MB)

Then transfer the entire `deployment/` folder to the customer server.

---

## For Customers: Installation

### Prerequisites

- Docker installed and running
  - Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: `curl -fsSL https://get.docker.com | sh`
- Minimum 8 GB RAM, 20 GB free disk space

### Step 1: Install Images

Copy the `deployment` folder to the server, then run:

**Windows:**
```cmd
cd C:\path\to\deployment
install.bat
```

**Linux:**
```bash
cd /path/to/deployment
bash install.sh
```

### Step 2: Start the Application

| Scenario | Windows | Linux |
|----------|---------|-------|
| With NVIDIA GPU | `start.bat` | `bash start.sh` |
| CPU only (no GPU) | `start-no-gpu.bat` | `bash start-no-gpu.sh` |

### Step 3: Access

Open a browser and go to:
```
http://localhost:3030
```

**Default login:** `admin` / `admin123`
> Change the admin password after first login!

---

## Management Scripts

| Task | Windows | Linux |
|------|---------|-------|
| Install images | `install.bat` | `bash install.sh` |
| **Update to new version** | **`update.bat`** | **`bash update.sh`** |
| Start (GPU) | `start.bat` | `bash start.sh` |
| Start (no GPU) | `start-no-gpu.bat` | `bash start-no-gpu.sh` |
| Stop | `stop.bat` | `bash stop.sh` |
| View logs | `logs.bat` | `docker compose logs -f` |
| Check status | `status.bat` | `docker compose ps` |
| Uninstall | `uninstall.bat` | `bash uninstall.sh` |

## Updating an Existing Installation

When shipping a new version to a customer who already has CiRA ME installed:

1. Copy the **new** `cirame-backend.tar` and `cirame-frontend.tar` into the customer's `deployment/` folder (replacing the old ones)
2. Run the update script:
   - Windows: `update.bat`
   - Linux: `bash update.sh`
3. Then restart: `start.bat` / `bash start.sh`

**What is preserved:** All customer data (database, accounts, trained models, datasets) lives in Docker named volumes — they survive the update untouched.

**What is replaced:** The application code (backend + frontend images). Old image layers are pruned automatically to free disk space.

---

## Configuration

### Change Port

Edit `docker-compose.yml` (or `docker-compose-no-gpu.yml`):
```yaml
ports:
  - "3030:80"   # change 3030 to your desired port
```
Then restart the application.

### Add Datasets

Place CSV files in the `shared/` folder. They will appear in the application
for all users automatically.

### Adjust Resource Limits

Edit the compose files:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'   # max CPU cores
      memory: 8G    # max RAM
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Docker is not running` | Start Docker Desktop (Windows) or `sudo systemctl start docker` (Linux) |
| `Images not found` | Run `install.bat` / `install.sh` first |
| `GPU-related error on start` | Use `start-no-gpu.bat` / `bash start-no-gpu.sh` |
| `Port 3030 already in use` | Change port in compose file or stop the conflicting service |
| Backend health check failing | Wait 60 s for backend to fully start, then check `status.bat` |

### View Logs
```cmd
docker compose logs -f backend
docker compose logs -f frontend
```

---

## Data Persistence

Application data is stored in Docker named volumes:

| Volume | Contents |
|--------|----------|
| `deployment_backend-data` | Database, user accounts, training sessions |
| `deployment_backend-models` | Saved ML/DL models |

Volumes persist across restarts. Only removed when running `uninstall`.

### Backup
```bash
# Backup database
docker run --rm -v deployment_backend-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/data-backup.tar.gz /data

# Backup models
docker run --rm -v deployment_backend-models:/models -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /models
```

---

## Deployment Package Contents

```
deployment/
├── cirame-backend.tar       # Backend Docker image  (~600 MB)
├── cirame-frontend.tar      # Frontend Docker image (~50 MB)
├── docker-compose.yml       # Compose config (with GPU)
├── docker-compose-no-gpu.yml# Compose config (CPU only)
├── export.bat               # [Vendor] Build & export on Windows
├── export.sh                # [Vendor] Build & export on Linux/macOS
├── install.bat              # [Customer] Install on Windows
├── install.sh               # [Customer] Install on Linux
├── start.bat                # Start with GPU (Windows)
├── start.sh                 # Start with GPU (Linux)
├── start-no-gpu.bat         # Start CPU only (Windows)
├── start-no-gpu.sh          # Start CPU only (Linux)
├── stop.bat / stop.sh       # Stop application
├── update.bat / update.sh   # Update to new version (preserves data)
├── uninstall.bat / uninstall.sh  # Remove everything
├── logs.bat                 # View logs (Windows)
├── status.bat               # Check status (Windows)
├── shared/                  # Datasets folder (editable by customer)
└── README.md                # This file
```

---

CiRA ME — Machine Intelligence for Edge Computing
