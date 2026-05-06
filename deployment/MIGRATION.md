# CiRA ME — Update & Migration Guide

This guide explains how to update CiRA ME to a new version **without losing your data** (users, models, datasets, MQTT history, etc.).

---

## What data is preserved?

All user data lives outside the Docker images, in folders next to `docker-compose.yml`:

| Folder | Contents |
|---|---|
| `data/database/` | SQLite database — users, saved models, ME-LAB endpoints, App Builder apps |
| `data/models/` | Trained model files (`.pkl`, `.onnx`) |
| `data/ti-projects/` | TI ModelMaker training projects |
| `data/mosquitto/` | MQTT broker persistent data |
| `datasets/` | User-uploaded datasets (includes `datasets/shared/` and per-user folders) |
| `mosquitto/mosquitto.conf` | MQTT broker configuration |

These folders are bind-mounted into the containers, so they survive every container/image update — **as long as the deployment folder stays the same**.

---

## Two ways to update

### Option A — In-place update (recommended)

If you keep your CiRA ME deployment in the same folder, just drop in the new `.tar` files and run the update:

```bash
# 1. Copy new image files into your existing deployment folder:
#    cirame-backend.tar
#    cirame-frontend.tar
#    cirame-ti-modelmaker.tar  (optional)
#    cirame-mosquitto.tar      (optional)

# 2. Run update:
update.bat                 # Windows
bash update.sh             # Linux / macOS
```

`update.bat` / `update.sh` will:
1. Stop the running containers (data folders are untouched).
2. Migrate the legacy `./shared/` folder to `./datasets/shared/` if needed.
3. Load the new Docker images.
4. Clean up old image layers.

After `update.bat` finishes, run `start.bat` (or `start-no-gpu.bat`) to launch the new version with all your data intact.

---

### Option B — New-folder update (with `migrate.bat`)

If you receive a fresh deployment package and prefer to extract it to a new folder (e.g. for backup safety), you can pull data from the old folder using the migrate script.

```bash
# 1. Extract the new release to a new folder, e.g. D:\CiRA ME v1.1\deployment\
# 2. Open a terminal in the NEW folder
# 3. Pull data from the OLD folder:

migrate.bat "D:\CiRA ME v1.0\deployment"             # Windows
bash migrate.sh ~/cirame-v1.0/deployment             # Linux / macOS

# 4. Load images and start:
install.bat   # or update.bat if images already loaded
start.bat
```

`migrate.bat` / `migrate.sh` copies:

- `data/` folder (database, models, TI projects, mosquitto persistence)
- `datasets/` folder (user uploads + shared datasets)
- Legacy `shared/` folder → `datasets/shared/` (auto-migrated to new layout)
- `mosquitto/mosquitto.conf`

**It does not delete the old folder** — your old installation remains untouched as a backup.

---

## Volume layout change (v1.0 → v1.1+)

In v1.0 the deployment compose only bind-mounted the `shared/` subfolder:

```yaml
# OLD (v1.0)
- ./shared:/app/datasets/shared
```

This caused user **private folders** (anything other than `shared/`) to live inside the container only, and they were wiped on every update.

From v1.1 the entire `datasets/` root is bind-mounted:

```yaml
# NEW (v1.1+)
- ./datasets:/app/datasets
```

So all user-uploaded data (shared + private folders) now persists across updates.

`install.bat`, `update.bat`, and `migrate.bat` automatically move any existing `./shared/` folder into `./datasets/shared/` so you keep your shared datasets without manual steps.

---

## Backup checklist (before any update)

To be safe, keep a copy of these folders:

```
deployment/
├── data/        ← whole folder
├── datasets/    ← whole folder (or legacy shared/ in old installs)
└── mosquitto/mosquitto.conf
```

A single zip of the deployment folder (excluding the `.tar` image files) is enough.

---

## Troubleshooting

### After update, my users / models / datasets are missing

- Verify you ran `update.bat` (not `install.bat`) in the **same folder** as your existing installation.
- If you extracted the new release to a different folder, run `migrate.bat <old_folder>` first.
- Confirm the data folders are present:
  ```
  data\database\cirame.db
  data\models\*.pkl
  datasets\shared\
  datasets\<your_private_folders>\
  ```

### Containers fail to start after migration

- Make sure no other CiRA ME containers are running from the old folder.
  Run `stop.bat` in the old folder first.
- Use `status.bat` (Windows) or `bash status.sh` (Linux) to check container health.

### Legacy `./shared/` is still there after update

- The migration step only runs if `./shared/` exists AND `./datasets/shared/` does NOT exist yet.
- If both exist, the new layout takes precedence — you can delete the old `./shared/` manually after verifying data is intact in `./datasets/shared/`.
