# CiRA ME

<p align="center">
  <img src="frontend/public/logo.svg" alt="CiRA ME Logo" width="80" height="80">
</p>

<h3 align="center">Machine Intelligence for Edge Computing</h3>

<p align="center">
  An end-to-end AI platform for time-series anomaly detection, classification, and regression — from data ingestion to edge deployment and live inference.
</p>

---

## Overview

CiRA ME is a web-based ML platform that enables engineers to build, train, deploy, and serve machine learning models for industrial time-series data. It supports the full lifecycle:

1. **Ingest** sensor data (CSV, CBOR, MQTT live stream)
2. **Preprocess** with windowing, normalization, and DSP feature extraction
3. **Train** using traditional ML, deep learning, TI TinyML, or custom Python models
4. **Deploy** to edge devices (Jetson, Raspberry Pi, TI MCU) or as API endpoints
5. **Serve** as published web apps with live MQTT streaming

## Features

### ML Pipeline

| Feature | Details |
|---|---|
| **Three Modes** | Anomaly Detection, Classification, Regression |
| **Data Sources** | CSV, Edge Impulse JSON/CBOR, CiRA CBOR, MQTT live stream |
| **Windowing** | Time-series segmentation with configurable window size, stride, overlap |
| **Feature Extraction** | 138+ DSP/statistical features (TSFresh + custom spectral analysis) |
| **Feature Selection** | Mutual information, F-regression, user toggle with raw signal pass-through |
| **Column Selection** | Checkboxes on data preview to include/exclude sensor channels |
| **Smart Recommendations** | Auto-suggest window size, stride, and features based on dataset |

### Training Approaches

| Approach | Algorithms |
|---|---|
| **Traditional ML** | Random Forest, XGBoost, LightGBM, Decision Tree, KNN, SVR, SVM, Naive Bayes, Logistic Regression |
| **Deep Learning** | TimesNet (time-series transformer) |
| **TI TinyML** | TI model zoo (Conv1D, MLP) + Traditional ML via emlearn for TMS320 MCUs |
| **Custom Model** | Python editor (CodeMirror 6) with 6 starter templates, any library (sklearn, PyTorch, XGBoost) |

### Deployment Targets

| Target | Method | Format |
|---|---|---|
| **Linux Devices** | SSH + Docker or Python files | ONNX, Pickle, Joblib |
| **TI TMS320 MCU** | Download CCS project package | emlearn C code + firmware template |
| **CiRA CLAW Edge** | ONNX + manifest package | CiRA CLAW C runtime |
| **ME-LAB API** | REST endpoint with API key auth | In-memory model serving |
| **App Builder** | Published web app (standalone URL) | Visual pipeline + live inference |

### ME-LAB (Inference Endpoints)

- Create REST API endpoints from saved models
- API key authentication with rate limiting
- Supports all three modes (anomaly, classification, regression)
- Usage tracking and logging

### App Builder

- **Visual pipeline editor** with drag-and-drop nodes
- **8 templates**: Regression Monitor, Anomaly Detector, Classifier, Live variants (MQTT), Signal Recorder, Blank
- **Node types**: CSV Upload, Live Stream (MQTT), Windowing, Normalize, Fill Missing, Feature Extract, Model Endpoint, Line Chart, Alert Badge, Table View, Signal Recorder
- **Pipeline validation** with real-time feature count/name matching
- **Auto-configure** Feature Extract from model's required features
- **Publish** as standalone web app with shareable URL
- **Access control**: Public, Team (logged-in), Private (API key)

### MQTT Live Streaming

- **Mosquitto broker** included in Docker deployment
- **Browser MQTT client** (mqtt.js) connects via WebSocket
- **Sensor buffer** accumulates window_size samples, auto-triggers inference
- **Live prediction display** with scrolling chart
- **Signal Recorder** mode for labeled data collection from mobile sensors
- **Auto-detect** payload formats (Android SensorSpot, named values, flat objects)
- **MQTT management page** with broker status, topic discovery, test publisher

### Dashboard

- Real-time system stats (GPU/CPU/memory/disk)
- Model counts by mode and algorithm
- Recent models overview
- User count and system info

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Frontend    │  │   Backend    │  │  TI ModelMaker       │   │
│  │  Vue 3 +      │  │  Flask +     │  │  Python 3.10 +       │   │
│  │  Vuetify 3    │  │  PyTorch     │  │  tinyml-modelmaker   │   │
│  │  nginx        │  │  sklearn     │  │  emlearn             │   │
│  │  :3030        │  │  :5100       │  │  :5200               │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────┐                                               │
│  │  Mosquitto   │  MQTT broker for live sensor streaming         │
│  │  :1883 (TCP) │  :9001 (WebSocket)                            │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Container Sizes

| Image | Size | Required |
|---|---|---|
| `cirame-backend` | ~14 GB | Yes |
| `cirame-frontend` | ~93 MB | Yes |
| `cirame-ti-modelmaker` | ~11.5 GB | Optional (TI MCU) |
| `eclipse-mosquitto` | ~36 MB | Optional (MQTT) |

## Tech Stack

### Frontend
- Vue.js 3 (Composition API + `<script setup>`)
- Vuetify 3 (Material Design components)
- Pinia (state management)
- CodeMirror 6 (Python editor)
- mqtt.js (browser MQTT client)
- Chart.js / SVG charts

### Backend
- Flask (Python 3.11)
- PyTorch 2.10.0+cu128 (GPU accelerated)
- scikit-learn, XGBoost, LightGBM, PyOD
- ONNX Runtime (model export/inference)
- Paho MQTT (test publisher)
- SQLite (database)

### TI ModelMaker
- Python 3.10 (required by tinyml-modelmaker)
- TI tinyml-modelmaker (model zoo)
- emlearn (sklearn to C code)
- onnxmltools (XGBoost/LightGBM to ONNX)

## Quick Start

### Development

```bash
# Clone
git clone https://github.com/siridech-bo/CiRA-ME.git
cd CiRA-ME

# Start all services
docker compose up -d

# Access
open http://localhost:3030
# Login: admin / admin123
```

### Production Deployment

```bash
# On build machine: export images
cd deployment
bash export.sh

# Transfer deployment/ folder to customer server

# On customer server: install and start
bash install.sh
bash start.sh

# Access at http://server-ip:3030
```

## Project Structure

```
CiRA ME/
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   │   ├── auth.py              # Authentication
│   │   │   ├── admin.py             # Dashboard stats, user management
│   │   │   ├── data_sources.py      # Data ingestion, windowing
│   │   │   ├── features.py          # Feature extraction, selection
│   │   │   ├── training.py          # ML/DL training, custom models
│   │   │   ├── deployment.py        # Edge deployment (SSH, Docker)
│   │   │   ├── ti_tinyml.py         # TI MCU integration
│   │   │   ├── melab.py             # Inference endpoints, API keys
│   │   │   ├── app_builder.py       # App CRUD, publish, pipeline runner
│   │   │   └── mqtt_publisher.py    # MQTT test publisher, broker management
│   │   ├── services/
│   │   │   ├── data_loader.py       # CSV/CBOR loading, windowing engine
│   │   │   ├── feature_extractor.py # DSP + TSFresh feature extraction
│   │   │   ├── ml_trainer.py        # sklearn/XGBoost/LightGBM training
│   │   │   ├── timesnet_trainer.py  # TimesNet deep learning
│   │   │   ├── custom_model_runner.py # Custom Python model execution
│   │   │   ├── deployer.py          # SSH deployment, Dockerfile generation
│   │   │   ├── melab_service.py     # Model loading, inference, caching
│   │   │   ├── pipeline_replay.py   # Full pipeline replay for evaluation
│   │   │   ├── ti_integration.py    # TI ModelMaker bridge
│   │   │   └── cira_claw_exporter.py # CiRA CLAW ONNX export
│   │   └── models.py               # SQLite models (User, SavedModel, MeLabEndpoint, AppBuilderApp)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── DashboardView.vue
│   │   │   ├── pipeline/
│   │   │   │   ├── DataSourceView.vue
│   │   │   │   ├── WindowingView.vue
│   │   │   │   ├── FeaturesView.vue      # Feature extraction + selection + toggle
│   │   │   │   ├── TrainingView.vue      # ML/DL/TI/Custom training
│   │   │   │   └── DeployView.vue        # Edge deployment + model history
│   │   │   ├── MeLabView.vue             # Inference endpoint management
│   │   │   ├── MqttManagementView.vue    # MQTT broker management
│   │   │   ├── AppBuilderListView.vue    # App list + templates + MQTT publisher
│   │   │   ├── AppBuilderEditorView.vue  # Visual pipeline editor (3-panel)
│   │   │   ├── PublishedAppView.vue      # Standalone published app + live stream
│   │   │   └── AdminView.vue             # User management
│   │   ├── components/
│   │   │   └── CodeEditor.vue            # CodeMirror 6 Python editor
│   │   ├── composables/
│   │   │   ├── useMqtt.ts                # MQTT connection composable
│   │   │   └── useSensorBuffer.ts        # Sensor windowing composable
│   │   ├── stores/
│   │   │   ├── pipeline.ts               # Pipeline state management
│   │   │   ├── auth.ts                   # Authentication state
│   │   │   └── notification.ts           # Toast notifications
│   │   └── services/
│   │       └── api.ts                    # Axios instance with interceptors
│   ├── Dockerfile
│   └── nginx.conf
├── ti-modelmaker/
│   ├── server.py                         # TI ModelMaker Flask API
│   ├── ccs_templates/                    # CCS project templates for TMS320
│   │   └── common/
│   │       ├── cira_main.c               # MCU firmware template
│   │       └── cira_serial_test.py       # Serial test tool
│   ├── Dockerfile
│   └── requirements.txt
├── mosquitto/
│   └── mosquitto.conf                    # MQTT broker config
├── deployment/
│   ├── docker-compose.yml                # Production compose
│   ├── export.sh / export.bat            # Image export scripts
│   ├── install.sh / install.bat          # Customer installation
│   ├── start.sh / start.bat              # Service management
│   └── shared/                           # Customer datasets
├── docker-compose.yml                    # Development compose
└── shared/                               # Shared datasets
```

## Database Schema

| Table | Purpose |
|---|---|
| `users` | User accounts with roles (admin, annotator) |
| `saved_models` | Trained model benchmarks with pipeline config |
| `melab_endpoints` | Inference API endpoints |
| `melab_api_keys` | API key management (hashed) |
| `melab_usage_log` | Inference usage tracking |
| `app_builder_apps` | Published web applications |

## API Endpoints

### Pipeline
- `POST /api/data/ingest/*` — Load CSV/CBOR/folder datasets
- `POST /api/data/windowing` — Apply time-series windowing
- `POST /api/features/extract` — Extract DSP features
- `POST /api/training/train/*` — Train models (ML/DL/TI/Custom)

### Deployment
- `POST /api/deployment/deploy` — SSH deploy to edge devices
- `GET /api/deployment/package/:id` — Download model package
- `GET /api/ti/export-mcu/:id` — Download TI MCU C code package

### ME-LAB
- `CRUD /api/melab/endpoints` — Manage inference endpoints
- `POST /api/melab/v1/:id/predict` — Run inference (API key auth)
- `CRUD /api/melab/keys` — Manage API keys

### App Builder
- `CRUD /api/app-builder/apps` — Manage published apps
- `POST /api/app-builder/apps/:id/publish` — Publish app
- `POST /api/app-builder/run/:slug` — Execute app pipeline
- `GET /api/app-builder/capabilities` — Node catalog + endpoints

### MQTT
- `GET /api/mqtt/broker-info` — Broker status and stats
- `GET /api/mqtt/topics` — Discover active topics
- `POST /api/mqtt/publish` — Start test publisher
- `POST /api/mqtt/topics/subscribe-test` — Test subscribe

## License

Proprietary - CiRA (Center of Innovative Robotics and Automation)
