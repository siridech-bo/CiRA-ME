# CiRA ME - Deployment Guide

Machine Intelligence for Edge Computing

## Prerequisites

1. **Windows Server** with:
   - Windows 10/11 or Windows Server 2019/2022
   - Minimum 8GB RAM (16GB recommended)
   - Minimum 20GB free disk space
   - Docker Desktop installed and running

2. **Optional: NVIDIA GPU** for accelerated machine learning:
   - NVIDIA GPU with CUDA support
   - NVIDIA Container Toolkit installed

## Deployment Package Contents

```
deployment/
├── cirame-backend.tar      # Backend Docker image
├── cirame-frontend.tar     # Frontend Docker image
├── docker-compose.yml      # Docker Compose config (with GPU)
├── docker-compose-no-gpu.yml   # Docker Compose config (CPU only)
├── install.bat             # Installation script
├── start.bat               # Start with GPU support
├── start-no-gpu.bat        # Start without GPU
├── stop.bat                # Stop application
├── status.bat              # Check application status
├── logs.bat                # View application logs
├── uninstall.bat           # Remove application
├── shared/                 # Shared datasets folder
└── README.md               # This file
```

## Quick Start

### Step 1: Install Docker Images

1. Copy the entire `deployment` folder to the target server
2. Open Command Prompt as Administrator
3. Navigate to the deployment folder:
   ```cmd
   cd C:\path\to\deployment
   ```
4. Run the installation script:
   ```cmd
   install.bat
   ```

### Step 2: Start the Application

**With GPU support:**
```cmd
start.bat
```

**Without GPU (CPU only):**
```cmd
start-no-gpu.bat
```

### Step 3: Access the Application

Open a web browser and navigate to:
```
http://localhost:3030
```

**Default Login:**
- Username: `admin`
- Password: `admin123`

> **Important:** Change the admin password after first login!

## Management Commands

| Script | Description |
|--------|-------------|
| `start.bat` | Start application with GPU support |
| `start-no-gpu.bat` | Start application without GPU |
| `stop.bat` | Stop the application |
| `status.bat` | Check container and image status |
| `logs.bat` | View live application logs |
| `uninstall.bat` | Remove all containers, images, and data |

## Configuration

### Changing the Port

By default, the application runs on port 3030. To change this:

1. Edit `docker-compose.yml` (or `docker-compose-no-gpu.yml`)
2. Find the frontend ports section:
   ```yaml
   ports:
     - "3030:80"
   ```
3. Change `3030` to your desired port
4. Restart the application

### Adding Datasets

Place dataset files in the `shared` folder. These will be accessible to all users in the application.

```
deployment/
└── shared/
    └── your-dataset.csv
```

### Resource Limits

Edit the docker-compose files to adjust resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum CPUs
      memory: 8G       # Maximum memory
```

## Troubleshooting

### Docker Not Running
```
ERROR: Docker is not running.
```
**Solution:** Start Docker Desktop and wait for it to fully initialize.

### GPU Not Found
```
ERROR: Failed to start services (GPU related error)
```
**Solution:** Use `start-no-gpu.bat` instead, or install NVIDIA Container Toolkit.

### Port Already in Use
```
ERROR: Port 3030 is already in use.
```
**Solution:** Change the port in docker-compose.yml or stop the conflicting service.

### Application Not Accessible
1. Check if containers are running: `status.bat`
2. Check firewall settings
3. View logs: `logs.bat`

### Backend Health Check Failing
The backend takes up to 60 seconds to start. Wait and check status again.

## Data Persistence

All application data is stored in Docker volumes:
- `deployment_backend-data`: Database and user data
- `deployment_backend-models`: Machine learning models

These volumes persist even after stopping the application. They are only removed when running `uninstall.bat`.

## Backup

To backup application data:

```cmd
docker run --rm -v deployment_backend-data:/data -v %cd%:/backup alpine tar cvf /backup/data-backup.tar /data
docker run --rm -v deployment_backend-models:/models -v %cd%:/backup alpine tar cvf /backup/models-backup.tar /models
```

## Support

For technical support or issues, please contact the system administrator.

---
CiRA ME - Machine Intelligence for Edge Computing
