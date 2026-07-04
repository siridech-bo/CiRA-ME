# CiRA ME — User Manual

**CiRA ME (Machine Intelligence for Edge)** is an end-to-end no-code AI platform for time-series anomaly detection, classification, and regression. Train ML and deep-learning models from sensor data, deploy them as REST endpoints, build live dashboards, and stream predictions from MQTT — all from a web browser.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Getting Started](#2-getting-started)
3. [Default Login & First Steps](#3-default-login--first-steps)
4. [The Training Pipeline](#4-the-training-pipeline)
   - [4.1 Data Source](#41-data-source)
   - [4.2 Windowing & Preprocessing](#42-windowing--preprocessing)
   - [4.3 Feature Engineering](#43-feature-engineering)
   - [4.4 Training](#44-training)
   - [4.5 Deployment](#45-deployment)
5. [ME-LAB — Model-as-a-Service](#5-me-lab--model-as-a-service)
6. [App Builder — Visual No-Code Apps](#6-app-builder--visual-no-code-apps)
7. [MQTT Live Streaming](#7-mqtt-live-streaming)
8. [Signal Recorder](#8-signal-recorder)
9. [Multi-Model Comparison](#9-multi-model-comparison)
10. [TI TinyML — MCU Deployment](#10-ti-tinyml--mcu-deployment)
11. [Admin Panel](#11-admin-panel)
12. [Updating to a New Version](#12-updating-to-a-new-version)
13. [Troubleshooting](#13-troubleshooting)
14. [Glossary](#14-glossary)

---

## 1. System Overview

### What CiRA ME Does

CiRA ME guides you through five steps that turn raw sensor data into a working AI model:

```
  Data → Windowing → Features → Training → Deployment
```

Every step is visual: upload a CSV, click a button, see a chart. No code required.

After training, you can:

- Call your model via **REST API** (ME-LAB endpoints)
- Build a **live dashboard** by dragging blocks together (App Builder)
- Stream real-time predictions from an **MQTT broker**
- Compare up to **5 models side-by-side** on the same data
- Export to **C code** for TI MSP/TMS320 microcontrollers

### Architecture (for IT teams)

Four Docker containers, all managed by `docker compose`:

| Container | Purpose | Port |
|---|---|---|
| `cirame-backend` | Flask REST API + ML training | 5100 |
| `cirame-frontend` | Vue 3 web UI (nginx) | 3030 |
| `cirame-ti-modelmaker` | TI TinyML training (Python 3.10) | 5200 |
| `cirame-mosquitto` | MQTT broker | 1883 (TCP), 9001 (WebSocket) |

All user data is stored on the host disk in `data/` and `datasets/` folders — survives every update.

### Modes Supported

| Mode | Use Case | Algorithms |
|---|---|---|
| **Classification** | Identify motion gestures, machine states | Random Forest, XGBoost, LightGBM, SVM, KNN, MLP, Logistic Regression, Decision Tree, Naive Bayes |
| **Regression** | Predict continuous values (temperature, pressure) | Random Forest, XGBoost, LightGBM, KNN, SVR, Linear |
| **Anomaly Detection** | Spot machine failures, outliers | Isolation Forest, LOF, OCSVM, AutoEncoder, KNN, COPOD, ECOD |
| **Deep Learning** | TimesNet for complex temporal patterns | TimesNet (PyTorch) |
| **TI TinyML** | On-MCU inference (TMS320, F28379D, F280049C) | TI Model Zoo: MLP, Conv1D |

---

## 2. Getting Started

### Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Docker**: Docker Desktop 24.0+ with Docker Compose v2
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 30 GB free
- **GPU** (optional): NVIDIA GPU with `nvidia-container-toolkit` for faster deep learning

### Installation (Fresh)

You will receive a `deployment/` folder containing `.tar` image files and scripts.

**Windows:**
1. Install Docker Desktop and start it
2. Double-click `install.bat`
3. Wait for the script to load all images (~15–20 minutes)
4. Double-click `start.bat` (or `start-no-gpu.bat` if no NVIDIA GPU)
5. Open `http://localhost:3030`

**Linux / macOS:**
```bash
cd deployment
bash install.sh
bash start.sh         # or bash start-no-gpu.sh
```

Access at `http://localhost:3030` (or `http://<server-ip>:3030` from other machines on your LAN).

### Network Ports

| Port | Service | Should be accessible from LAN? |
|---|---|---|
| **3030** | Web app | Yes |
| 5100 | Backend API | Internal use |
| 5200 | TI ModelMaker | Internal use |
| **1883** | MQTT TCP (for sensors) | Yes |
| **9001** | MQTT WebSocket (for browsers) | Yes |

---

## 3. Default Login & First Steps

```
Username: admin
Password: cira123
```

**Change the password immediately after first login** (Admin → Users → Edit → Reset Password).

### Recommended First Actions

1. **Create your first dataset folder** (Admin → Folder Management → New Folder Name)
2. **Add team members** (Admin → Add User) with their own private folders
3. **Upload your first dataset** (Data Source → Upload, or drop files into `datasets/shared/`)

---

## 4. The Training Pipeline

Click **Pipeline** (or the "Train New Model" tile on the Dashboard) to start. The stepper at the top guides you through five steps.

### 4.1 Data Source

Select one of four input formats:

| Format | When to use |
|---|---|
| **CSV File** | Standard tabular data with headers in row 1. Optional `label` column for classification. |
| **Edge Impulse JSON** | Exported from Edge Impulse Studio (single recording per file). |
| **Edge Impulse CBOR** | Binary CBOR format, faster loading. Supports signed (JWS) and unsigned CBOR. |
| **CiRA CBOR** | CiRA's native recording format with multiple samples per file. |

**Browse Files:**
- Navigate folders with breadcrumbs
- **Upload** button: drag-drop CSV/JSON/CBOR files into your folder
- **Refresh** button: reload after manually copying files into `datasets/`
- **Select All (N)** button (CSV mode only): selects every CSV file in the current folder for multi-file training
- **Live counter chip**: "5 / 192 selected"
- **Record Sensors** (admin only): collect data live from a connected sensor

**Dataset folder detection (CBOR):**
- If the folder contains `training/` and `testing/` subfolders, CiRA ME auto-detects it as a *dataset* and offers a **Scan Dataset** button.
- After scanning, you see counts per partition (e.g., `training: 85 files`, `testing: 16 files`) and per label (e.g., `wave: 4`, `idle: 4`).

**Preview Panel** (right):
- Shows column names, row count, and the first 100 rows
- For multi-CSV selection: validates that all files share the same columns (column-mismatch error if not)

---

### 4.2 Windowing & Preprocessing

Configure how the time series is segmented into fixed-length windows.

**Raw Mode (no windowing) toggle:**
- For pre-processed tabular data where **each row is already a complete feature vector** (e.g., one row = one measurement with computed features)
- Skips windowing AND feature extraction
- Click *"Prepare Data & Go to Training"* — it auto-runs both steps and jumps to Training

**Time-Series Mode (default):**

| Setting | What it does |
|---|---|
| **Window Size** (samples) | Number of consecutive readings per window. E.g., 128 samples at 100 Hz = 1.28 seconds |
| **Stride** | Step between window starts. Stride < window size = overlapping windows (more data) |
| **Test Ratio** | Fraction of data reserved for evaluation (default 0.2 = 20%) |

**Split Strategy** (3 options for single CSV files):

| Strategy | How it works | Best for |
|---|---|---|
| **End Block** (default) | Test = last N% of signal, gap = window size | Standard time-series, prevents future leakage |
| **Interleaved** | Signal split into blocks; test blocks distributed evenly | Better coverage of the full signal range |
| **Random** | Windows shuffled and randomly assigned to train/test | Maximum variety (may leak temporal patterns) |

The chart on the right shows a window-preview animation so you see how `window_size` and `stride` affect segmentation.

**Smart Recommendation:**
- If your dataset is too small for the current settings, an alert suggests better values (e.g., "Try window_size=64, stride=32 for at least 10 training windows").

**Regression Mode — Prediction Target:**
- Pick which sensor column to predict. The remaining columns become input features.
- The test set is split temporally so future predictions are evaluated on truly unseen data.

**Classification Mode — Label Preservation:**
- **Majority Voting** (default): window label = most frequent label in the window
- **First / Last**: window label = first/last sample's label
- **Threshold**: window labeled as anomaly if any sample exceeds threshold

---

### 4.3 Feature Engineering

Extract numerical features from each window. Three workflows:

**1. Extract tab** — choose extraction method:
- **Lightweight (DSP + statistics)** — fast, ~20 features per channel (mean, std, RMS, peak frequency, FFT energy, kurtosis, skewness, etc.)
- **TSFresh** — comprehensive, ~700 features per channel (heavier, longer training time)
- **Both** — concatenated feature set

**2. Select tab** — feature ranking and selection:
- Mutual Information, Chi-squared, ANOVA F-test
- Recursive feature elimination (RFE) with cross-validation
- Click "Auto-select top N" to keep only the most informative features (reduces overfitting and training time)

**3. Visualize tab** — explore feature distributions:
- Per-class violin plots
- Correlation matrix (drop redundant features manually)
- 2D/3D scatter via PCA or UMAP

> **Raw Mode users**: this entire step is skipped. The CSV columns become the input features directly.

---

### 4.4 Training

The page is split by mode (tabs at the top: **Anomaly**, **Classification**, **Regression**).

For each mode you see a table of algorithms with **Train** buttons. Click any algorithm to start training; results appear with all metrics:

**Classification metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
**Regression metrics:** R², RMSE, MAE, residuals histogram, **Actual vs Predicted time-series chart** with TEST ONLY / TRAIN+TEST toggle
**Anomaly metrics:** Precision, Recall, F1 (treating anomaly as positive class), anomaly-score distribution

**Test region visualization** (regression):
- The Actual vs Predicted chart shows test regions as **highlighted orange bands** matching your selected split strategy
  - End Block → one band at the end
  - Interleaved → multiple bands scattered through the signal
  - Random → many small bands wherever test windows fell

**Deep Learning tab (TimesNet):**
- Enter epochs, batch size, learning rate
- Trains using PyTorch (uses GPU if available)
- Live loss curve updates during training

**TI TinyML tab:**
- Forwarded to the TI ModelMaker container (Python 3.10)
- Models export as C code for TMS320, F28379D, F280049C MCUs

**Save the model:**
- Once you choose the best algorithm, click **Save Model**
- Give it a name and description
- It appears in **Dashboard → Saved Models** and can be deployed via ME-LAB

**Test with New Data:**
- After saving, you can upload a NEW dataset and replay the full pipeline (normalization → windowing → features → predict) to verify the model on fresh data.

---

### 4.5 Deployment

Three deployment paths:

1. **ME-LAB endpoint** (most common) — see [Section 5](#5-me-lab--model-as-a-service)
2. **SSH deploy to Jetson** — package the model + a Python inference script and copy to a remote NVIDIA Jetson over SSH
3. **TI MCU C code export** — see [Section 10](#10-ti-tinyml--mcu-deployment)

---

## 5. ME-LAB — Model-as-a-Service

**ME-LAB** turns any saved model into a callable REST endpoint with per-endpoint API keys.

### Creating an Endpoint

1. **Dashboard → ME-LAB** (or the ME-LAB tile)
2. Click **+ New Endpoint**
3. Pick a saved model, give the endpoint a name (e.g., `motor-anomaly-prod`)
4. Endpoint becomes **Active** with a unique ID (e.g., `e3a8b9c5d1f2`)

### Calling an Endpoint

**Two authentication options:**

**A) Session auth (browser):**
```javascript
fetch('http://localhost:5100/api/melab/predict/e3a8b9c5d1f2', {
  method: 'POST',
  credentials: 'include',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ data: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...] })
})
```

**B) API key (for IoT devices, scripts):**
```bash
curl -X POST http://server-ip:5100/api/melab/predict/e3a8b9c5d1f2 \
  -H "X-API-Key: cira_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.1, 0.2, 0.3]]}'
```

Generate API keys: **ME-LAB → Endpoint → API Keys → + Generate**. Each key has an optional expiry date and can be revoked anytime.

### Endpoint Page Shows

- **Live metrics** — inference count, last used time, average latency
- **Required features** — list of feature names the model expects
- **Target column** — the column being predicted (regression)
- **Algorithm + Mode** — what type of model
- **Try it** widget — paste raw data and test predictions instantly

### Quotas

Each user has limits set by admin:
- `max_endpoints` (default 5)
- `max_folder_mb` (default 1000 MB)
- `max_apps` (default 10)

---

## 6. App Builder — Visual No-Code Apps

The **App Builder** lets you compose inference apps by dragging blocks onto a canvas. Each app gets a unique URL (`/apps/<slug>`) that you can share or embed.

### Creating an App

1. **Dashboard → App Builder → + New App**
2. Pick a template:

| Template | What it does |
|---|---|
| **Regression Monitor** | CSV → predict continuous value, show line chart |
| **Anomaly Detector** | CSV → flag anomalies, show alert badges |
| **Classifier** | CSV → predict class label, show prediction table |
| **Live Regression** | MQTT stream → live continuous predictions |
| **Live Anomaly Detector** | MQTT stream → live alerts |
| **Live Classifier** | MQTT stream → live class predictions |
| **MQTT Recorder** | Record labeled sensor data via MQTT (no model needed) |
| **Multi-Model Regression** (CSV / Live) | Compare up to 5 regression models on the same data |
| **Multi-Model Classification** (CSV / Live) | Compare up to 5 classification models |
| **Blank** | Start from scratch — full creative control |

3. App opens in the **Editor** with 3 tabs:

### BUILD Tab

Drag nodes from the left palette onto the canvas:

- **Inputs**: CSV Upload, Live Stream (MQTT)
- **Transforms**: Normalize, Windowing, Fill Missing, Feature Extract
- **Models**: any ME-LAB endpoint (filtered by mode)
- **Outputs**: Line Chart, Alert Badge, Table View, Signal Recorder, Multi-Model Compare

Connect node outputs to inputs by dragging from `→` to `←` ports. The right panel shows the selected node's config.

**Auto-Configure Feature Extract** button: when you select a model node, click this to auto-fill the feature extract node with the exact features the model was trained on.

**Raw Mode models**: if you select a model trained with raw mode, the windowing node auto-sets `window_size=1, stride=1` (effectively pass-through).

### PREVIEW Tab

Shows a browser-mockup of what the published app will look like — including a simulated chart with sample data. Useful for sharing screenshots before publishing.

### PUBLISH Tab

- **Access control**: Public / Team / Private
- Click **Publish** → get a shareable URL like `http://server-ip:3030/apps/my-classifier-a1b2c3`
- Anyone with the URL can use the app (subject to access policy)
- Stats: number of calls, last used time

---

## 7. MQTT Live Streaming

The built-in Mosquitto broker accepts sensor data over MQTT and feeds it into Live App Builder apps.

### Publishing Sensor Data

**From IoT devices / gateways:**
- Broker URL: `mqtt://<server-ip>:1883`
- Publish JSON like `{"accX": 1.2, "accY": 3.4, "accZ": 5.6}` to any topic (e.g., `sensors/machine1`)

**From a browser-based app (e.g., SensorSpot mobile app):**
- WebSocket URL: `ws://<server-ip>:9001/mqtt`

**Supported payload formats** (auto-detected):
1. `{"accX": 1.2, "accY": 3.4, "accZ": 5.6}` — flat object
2. `{"values": {"v0": 1.2, "v1": 3.4}}` — SensorSpot format
3. `[1.2, 3.4, 5.6]` — bare array (channels named `ch0`, `ch1`, `ch2`)

### Subscribing in an App Builder App

1. Choose a **Live …** template
2. In the Live Stream node:
   - **Broker URL** (auto-detected when accessed from a remote browser): `ws://<server-ip>:9001/mqtt`
   - **Topic**: `sensors/#` (the `#` is an MQTT wildcard)
   - **Channels**: leave blank to auto-detect from the first message

3. Publish the app → open the URL → click **Connect**
4. Live stats panel shows: messages received, rate (msg/sec), buffer fill, inference count
5. Predictions accumulate on the chart (up to 200 most recent)

### MQTT Test Publisher (built-in)

For testing without real sensors:
- **Dashboard → MQTT Broker → Test Publisher**
- Pick a CSV file from your datasets folder
- Pick a topic and rate (msg/sec)
- Click **Start Publishing** — the CSV streams to the broker row-by-row

### MQTT Broker Monitoring

**Dashboard → MQTT Broker**:
- Live connection status, broker stats ($SYS)
- **Discover Topics** — captures all active topics for 5 seconds
- **Subscribe** widget — view raw messages from any topic for debugging

### Live Comparison with Ground Truth (Regression)

For Live Regression apps, a **"Compare with column (actual)"** dropdown lets you pick which MQTT field to treat as the ground-truth value. The app then plots actual vs predicted lines in real-time and computes live R²/RMSE/MAE.

---

## 8. Signal Recorder

Collect labeled sensor data via MQTT — useful for building your training dataset before you have any model.

### Creating a Recorder

1. App Builder → **MQTT Recorder** template
2. Configure:
   - **Labels** (comma-separated): e.g., `idle, wave, snake, updown`
   - **Target Sample Rate (Hz)**: e.g., 62.5
   - **Max Duration (seconds)**: safety cap per recording
   - **File Name Prefix**: e.g., `motion_recording`

3. Publish the app
4. Connect to MQTT, then:
   - Click a label button to mark the **current label**
   - Click **REC** to start recording
   - Click **STOP** to save
5. The CSV downloads to your computer with a `label` column populated per the active label at each moment

This gives you a perfectly-labeled dataset for the training pipeline.

---

## 9. Multi-Model Comparison

Run up to **5 different models on the same input data** and compare their predictions side-by-side.

### Creating a Comparison App

1. App Builder → Pick one of:
   - **Multi-Model Regression** (CSV)
   - **Multi-Model Classification** (CSV)
   - **Live Multi-Model Regression** (MQTT)
   - **Live Multi-Model Classification** (MQTT)

2. In the **Multi-Model Compare** node config:
   - **Model Mode**: regression / classification / anomaly (filters the endpoint dropdown)
   - **Model Endpoints**: pick up to 5 active endpoints
   - **Target Column**: which column is the ground truth (auto-filled from the first model)

3. Click **Auto-Configure Feature Extract** to set up windowing/features that satisfy all selected models

4. Publish → open the URL

### What You See

**Metrics Table** (with ground truth):
- Per-model R²/RMSE/MAE (regression) or Accuracy/Precision/F1 (classification)
- 🏆 trophy icon on the best-performing model
- Red row + error message for failed models (e.g., broken pickle, missing features)

**Without ground truth** (live MQTT without target column):
- Latest Prediction + Window count per model

**Chart**:
- **Regression** → line chart with actual (cyan) and one line per model (different colors)
- **Classification** → timeline chart with horizontal bands:
  - One band per model + Actual (if present)
  - Each window = colored rectangle by predicted class
  - Red border on rectangles that disagree with actual
  - Sensor signal plotted below for context

**Per-Window Predictions Table** (collapsible):
- Side-by-side predictions for every window
- For classification: green text if matches actual, red if mismatches

**Download Comparison CSV** button:
- Exports all predictions + actuals + metrics as a single CSV

### Live MQTT Multi-Model

- Predictions accumulate over time (up to 200 windows)
- If the MQTT data includes the target column, ground-truth comparison is enabled automatically
- Metrics recompute every inference

---

## 10. TI TinyML — MCU Deployment

For deploying models to **TI MSP** and **TMS320** microcontrollers (no OS, no Docker, just bare-metal C).

### Workflow

1. Run the standard training pipeline up to **Feature Engineering**
2. **Training → TI TinyML tab**:
   - Pick a model architecture from TI Model Zoo (MLP, Conv1D)
   - Choose target MCU (TMS320 F28379D, F280049C, F28P55x)
   - Configure: epochs, batch size, quantization (INT8 / float32)
3. Click **Train** — request is forwarded to the `cirame-ti-modelmaker` container
4. Once trained, click **Export CCS Package**
5. ZIP downloads containing:
   - `model.c` / `model.h` — compiled inference code
   - `cira_main.c` — application template with serial test tool
   - `README.md` — integration instructions
   - Code Composer Studio (CCS) project files

### On the MCU

Open the project in CCS, build, flash. Predictions stream over UART. Use the included serial test tool on your PC to verify.

---

## 11. Admin Panel

**Dashboard → Admin** (admin role only).

### User Management

| Field | Notes |
|---|---|
| **Username** | Unique login name |
| **Display Name** | Shown in UI |
| **Role** | `admin` (full access) or `annotator` (own data only) |
| **Private Folder** | Folder under `datasets/` that only this user can write to |
| **Active** | Disable login without deleting the account |
| **Quotas** | `max_folder_mb` (storage), `max_endpoints` (ME-LAB), `max_apps` (App Builder) |

- Edit / Delete users (cannot delete the main admin or yourself)
- Reset password (6 character minimum)

### Folder Management

- Create new dataset folders (alphanumeric, underscore, hyphen only)
- Cannot delete the `shared` folder
- Folders appear under Data Source → Browse Files

### System Settings

- **Datasets Root**: path inside the container (`/app/datasets`)
- **Session Timeout**: 8 hours
- **Version**: CiRA ME version

### Storage Volumes

Shows all mounted Docker volumes with:
- **Volume name** + total size
- **Container path** (e.g., `/app/data`)
- **Host path** (e.g., `./data/database` — open this folder directly in Windows File Explorer to manage files manually)
- **Disk space**: free / total GB
- **File tree** (click to expand): see actual files in each volume

This is the easiest way to know where your data lives on disk and back it up.

### Dashboard Stats (any user)

- Total users / models / endpoints
- Models by mode and algorithm (pie charts)
- Recent models (last 10)
- System: CPU usage, memory, GPU, disk, uptime, Python/PyTorch versions

---

## 12. Updating to a New Version

You will receive a new package containing updated `.tar` image files.

**Two ways to update** — both preserve all your data (users, models, datasets, endpoints, apps, MQTT history):

### Option A — In-place update (recommended)

Drop the new `.tar` files into your existing deployment folder and run:
```
update.bat              (Windows)
bash update.sh          (Linux / macOS)
```
Then run `start.bat` (or `start-no-gpu.bat`).

### Option B — New folder

If you extracted the new release to a different folder, copy data from the old folder first:
```
migrate.bat "C:\path\to\old\deployment"        (Windows)
bash migrate.sh /path/to/old/deployment        (Linux / macOS)
```
Then `install.bat` → `start.bat`.

**See `deployment/MIGRATION.md` for the full guide** with backup checklist and troubleshooting.

### Data folders that survive every update

```
deployment/
├── data/database/      ← SQLite database (users, models, endpoints, apps)
├── data/models/        ← Trained model files (.pkl, .onnx)
├── data/ti-projects/   ← TI ModelMaker training projects
├── data/mosquitto/     ← MQTT broker persistent data
├── datasets/           ← User-uploaded datasets (shared + per-user)
└── mosquitto/mosquitto.conf
```

**To back up everything**: zip the entire deployment folder excluding `.tar` files.

---

## 13. Troubleshooting

### Cannot access from other machines

1. Check Windows Firewall allows ports **3030**, **1883**, **9001**
2. Use server IP (not `localhost`) in the browser: `http://192.168.x.x:3030`
3. On Windows Server: Docker Desktop → Settings → Resources → Network → check WSL integration

### "Object of type int64 is not JSON serializable" when applying windowing

Resolved in v1.1+ — update to the latest version.

### Backend container unhealthy after install (Windows)

- **Old image version**: rebuild with updated `requirements.txt` (Werkzeug<3.1, pandas<2.2.3, numpy<2.0 pinned)
- **GPU compose without NVIDIA toolkit**: use `start-no-gpu.bat` instead of `start.bat`

### CBOR file load fails with "PanicException"

Resolved in v1.1+ — `cbor2` library pinned to `<6.0` (avoids Rust panic on signed CBOR files).

### Multi-model classification shows accuracy 0%

- Check the dataset's `label` column matches the model's training classes
- Edge Impulse CBOR files use the folder name as label — make sure the file is from a folder whose name matches one of the trained classes

### MQTT live predictions are flat (constant value)

If using a model trained in **Raw Mode**: the App Builder needs to know — auto-detected as long as the model was trained with the `no_windowing` flag set. Retrain the model and select the latest endpoint.

### Charts not updating in live mode

- Check the buffer bar at the top — needs to reach 100% (one window's worth) before the first inference fires
- Lower the publish rate or window size if the buffer fills too slowly

### Database / models lost after upgrade

- You probably ran `install.bat` in a new folder. Run `migrate.bat <old_folder>` first, then `start.bat`.
- See `deployment/MIGRATION.md`.

### Logs

| What | How |
|---|---|
| All services | `docker compose logs -f` (or `logs.bat` on Windows) |
| Just backend | `docker logs cirame-backend --tail 100` |
| Container status | `status.bat` / `bash status.sh` |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Window** | A contiguous chunk of consecutive sensor readings used as one sample for the model |
| **Stride** | Number of samples to advance between window starts. Stride < window size = overlapping windows |
| **Feature** | A single numeric value computed from a window (e.g., mean, peak frequency) |
| **ME-LAB** | CiRA's Model-as-a-Service module — exposes trained models as REST endpoints |
| **Endpoint** | A deployed model with its own URL + API keys |
| **App Builder** | Visual editor for composing inference apps from blocks |
| **Pipeline** | The five-step training workflow: Data → Windowing → Features → Training → Deploy |
| **Raw Mode** | A mode that skips windowing and feature extraction — each CSV row is one sample |
| **Split Strategy** | How train/test data is divided: End Block, Interleaved, or Random |
| **Target Column** | The column to predict (for regression) or the ground-truth label column |
| **TimesNet** | A deep-learning model architecture for time-series, trained with PyTorch |
| **TI TinyML** | Texas Instruments toolkit for running tiny models on microcontrollers |
| **Multi-Model Compare** | An App Builder feature for running 2–5 models on the same data and comparing outputs |
| **CBOR** | Concise Binary Object Representation — a compact binary alternative to JSON |
| **MQTT** | Lightweight pub-sub messaging protocol commonly used for IoT sensor data |

---

## Support

**CiRA — Center of Innovative Robotics and Automation**

For bug reports, feature requests, or questions, contact your CiRA representative.

---

*This manual covers CiRA ME v1.1+ — last updated based on the latest deployment release.*
