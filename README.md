# CiRA ME

<p align="center">
  <img src="frontend/public/logo.svg" alt="CiRA ME Logo" width="80" height="80">
</p>

<h3 align="center">Machine Intelligence for Edge Computing</h3>

<p align="center">
  A modern web-based ML platform for building anomaly detection and classification models for edge deployment.
</p>

---

## Features

- **Dual-Mode ML Pipeline**: Seamlessly switch between Anomaly Detection and Classification
- **Multiple Data Sources**: CSV, Edge Impulse JSON, Edge Impulse CBOR, CiRA CBOR
- **Advanced Windowing**: Time-series segmentation with label preservation via majority voting
- **40+ Feature Extraction**: TSFresh statistical features + Custom DSP features
- **10 Anomaly Detection Algorithms**: Isolation Forest, LOF, OCSVM, HBOS, KNN, COPOD, ECOD, SUOD, AutoEncoder, Deep SVDD (via PyOD)
- **8 Classification Algorithms**: Random Forest, Gradient Boosting, SVM, MLP, KNN, Decision Tree, Naive Bayes, Logistic Regression (via Scikit-learn)
- **LLM-Powered Feature Selection**: Local Llama 3.2 integration for intelligent recommendations
- **Edge Deployment**: SSH deployment to NVIDIA Jetson and other Linux devices
- **Role-Based Access Control**: Admin and Annotator roles with folder-based security

## Tech Stack

### Frontend
- Vue.js 3 (Composition API)
- Vuetify 3 (Material Design)
- Pinia (State Management)
- Vue Router 4
- Chart.js (Visualizations)
- TypeScript
- Vite

### Backend
- Python 3.10+
- Flask
- SQLite
- Pandas / NumPy / SciPy
- Scikit-learn
- PyOD
- TSFresh
- CBOR2

## Project Structure

```
CiRA ME/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Flask app factory
│   │   ├── config.py            # Configuration settings
│   │   ├── models.py            # Database models
│   │   ├── auth.py              # Authentication utilities
│   │   ├── routes/
│   │   │   ├── auth.py          # Authentication endpoints
│   │   │   ├── admin.py         # Admin endpoints
│   │   │   ├── data_sources.py  # Data ingestion endpoints
│   │   │   ├── features.py      # Feature extraction endpoints
│   │   │   ├── training.py      # ML training endpoints
│   │   │   └── deployment.py    # Deployment endpoints
│   │   └── services/
│   │       ├── data_loader.py   # Data loading service
│   │       ├── feature_extractor.py  # Feature extraction service
│   │       ├── ml_trainer.py    # ML training service
│   │       └── deployer.py      # Deployment service
│   ├── data/                    # SQLite database
│   ├── datasets/                # Data storage
│   │   └── shared/              # Shared folder
│   ├── run.py                   # Application entry point
│   └── requirements.txt         # Python dependencies
│
└── frontend/
    ├── public/
    │   └── logo.svg             # App logo
    ├── src/
    │   ├── assets/
    │   │   └── LogoFull.vue     # Logo component
    │   ├── components/
    │   │   └── PipelineStepper.vue
    │   ├── views/
    │   │   ├── LoginView.vue
    │   │   ├── DashboardView.vue
    │   │   ├── AdminView.vue
    │   │   └── pipeline/
    │   │       ├── DataSourceView.vue
    │   │       ├── WindowingView.vue
    │   │       ├── FeaturesView.vue
    │   │       ├── TrainingView.vue
    │   │       └── DeployView.vue
    │   ├── stores/
    │   │   ├── auth.ts
    │   │   ├── pipeline.ts
    │   │   └── notification.ts
    │   ├── router/
    │   │   └── index.ts
    │   ├── services/
    │   │   └── api.ts
    │   ├── styles/
    │   │   └── main.scss
    │   ├── App.vue
    │   └── main.ts
    ├── package.json
    ├── vite.config.ts
    └── tsconfig.json
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py
```

The backend will start at `http://localhost:5000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will start at `http://localhost:5173`

### Default Credentials

- **Username**: admin
- **Password**: admin123

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user
- `POST /api/auth/change-password` - Change password

### Data Sources
- `GET /api/data/datasets-root` - Get datasets root path
- `GET /api/data/user-folders` - Get accessible folders
- `POST /api/data/browse` - Browse directories
- `POST /api/data/ingest/csv` - Ingest CSV file
- `POST /api/data/ingest/ei-json` - Ingest Edge Impulse JSON
- `POST /api/data/ingest/ei-cbor` - Ingest Edge Impulse CBOR
- `POST /api/data/ingest/cira-cbor` - Ingest CiRA CBOR
- `POST /api/data/windowing` - Apply windowing

### Features
- `GET /api/features/available` - Get available features
- `POST /api/features/extract` - Extract features
- `POST /api/features/recommend` - Get LLM recommendations

### Training
- `GET /api/training/algorithms` - Get available algorithms
- `POST /api/training/train/anomaly` - Train anomaly model
- `POST /api/training/train/classification` - Train classification model
- `POST /api/training/predict` - Make predictions
- `GET /api/training/metrics/{session_id}` - Get metrics

### Deployment
- `GET /api/deployment/targets` - Get deployment targets
- `POST /api/deployment/test-connection` - Test SSH connection
- `POST /api/deployment/deploy` - Deploy model

## Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#6366F1` | Indigo - Neural/AI |
| Secondary | `#22D3EE` | Cyan - Edge/Tech |
| Accent | `#A855F7` | Purple - Highlights |
| Success | `#10B981` | Green - Normal/Success |
| Warning | `#F59E0B` | Amber - Warnings |
| Error | `#EF4444` | Red - Anomaly/Error |
| Background | `#0F172A` | Slate - Dark mode |
| Surface | `#1E293B` | Card backgrounds |

## License

MIT License - See LICENSE for details.

---

<p align="center">
  Built with ❤️ for Edge AI
</p>
