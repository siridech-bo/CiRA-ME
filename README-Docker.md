# CiRA ME - Docker Deployment Guide

This guide provides comprehensive instructions for deploying CiRA ME (Machine Intelligence for Edge Computing) using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Volume Management](#volume-management)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)
- [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Required Software

- **Docker**: Version 20.10.0 or higher
- **Docker Compose**: Version 2.0.0 or higher

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB free space

**Recommended:**
- CPU: 8 cores
- RAM: 16 GB
- Disk: 50 GB free space (for models and datasets)

### Installation

#### Linux (Ubuntu/Debian)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Windows
Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)

#### macOS
Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)

---

## Quick Start

### 1. Clone or Extract the Repository

```bash
cd /path/to/cirame
```

### 2. Create Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred settings
# nano .env  # Linux/Mac
# notepad .env  # Windows
```

### 3. Create Volume Directories

The application requires persistent storage for data, models, and datasets.

**Linux/Mac:**
```bash
mkdir -p docker-volumes/data docker-volumes/models docker-volumes/datasets
chmod -R 755 docker-volumes/
```

**Windows (PowerShell):**
```powershell
New-Item -Path "docker-volumes\data" -ItemType Directory -Force
New-Item -Path "docker-volumes\models" -ItemType Directory -Force
New-Item -Path "docker-volumes\datasets" -ItemType Directory -Force
```

### 4. Build and Start Services

```bash
# Build images and start all services
docker compose up -d

# View logs
docker compose logs -f

# Check service status
docker compose ps
```

### 5. Access the Application

- **Frontend**: http://localhost:3030
- **Backend API**: http://localhost:5100
- **Health Check**: http://localhost:5100/api/health

Default credentials (if authentication is enabled):
- Username: `admin`
- Password: Check application documentation

---

## Configuration

### Environment Variables

Edit the `.env` file to customize your deployment:

```bash
# Flask Backend
FLASK_DEBUG=False

# Frontend Port
FRONTEND_PORT=3030

# Volume Paths (can be absolute or relative)
CIRAME_DATA_PATH=./docker-volumes/data
CIRAME_MODELS_PATH=./docker-volumes/models
CIRAME_DATASETS_PATH=./docker-volumes/datasets
```

### Port Configuration

To change the frontend port, update `FRONTEND_PORT` in `.env`:

```bash
FRONTEND_PORT=8080
```

Then restart the services:

```bash
docker compose down
docker compose up -d
```

---

## Deployment

### Development Deployment

For development with hot-reload and debugging:

```bash
# Set debug mode in .env
FLASK_DEBUG=True

# Start services
docker compose up
```

### Production Deployment

For production use:

1. **Update Environment Variables**

```bash
# In .env file
FLASK_DEBUG=False
FRONTEND_PORT=80  # or 443 with SSL
```

2. **Use Production Volume Paths**

```bash
# Linux/Mac
CIRAME_DATA_PATH=/var/lib/cirame/data
CIRAME_MODELS_PATH=/var/lib/cirame/models
CIRAME_DATASETS_PATH=/var/lib/cirame/datasets

# Windows
CIRAME_DATA_PATH=C:/CiRAME/data
CIRAME_MODELS_PATH=C:/CiRAME/models
CIRAME_DATASETS_PATH=C:/CiRAME/datasets
```

3. **Create Production Directories**

```bash
# Linux/Mac
sudo mkdir -p /var/lib/cirame/{data,models,datasets}
sudo chown -R 1000:1000 /var/lib/cirame

# Windows (run as Administrator)
New-Item -Path "C:\CiRAME\data" -ItemType Directory -Force
New-Item -Path "C:\CiRAME\models" -ItemType Directory -Force
New-Item -Path "C:\CiRAME\datasets" -ItemType Directory -Force
```

4. **Start Services**

```bash
docker compose up -d
```

### SSL/TLS Configuration (Production)

For HTTPS support, use a reverse proxy like Nginx or Traefik in front of the frontend service.

**Example Nginx Configuration:**

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3030;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Volume Management

### Data Persistence

CiRA ME uses three persistent volumes:

1. **backend-data**: SQLite database and application state
2. **backend-models**: Trained ML models (PKL files)
3. **backend-datasets**: Training and inference datasets

### Backup Strategy

**Automated Backup Script (Linux/Mac):**

```bash
#!/bin/bash
# backup-cirame.sh

BACKUP_DIR="/backup/cirame/$(date +%Y%m%d_%H%M%S)"
VOLUME_BASE="./docker-volumes"

mkdir -p "$BACKUP_DIR"

# Backup volumes
cp -r "$VOLUME_BASE/data" "$BACKUP_DIR/"
cp -r "$VOLUME_BASE/models" "$BACKUP_DIR/"
cp -r "$VOLUME_BASE/datasets" "$BACKUP_DIR/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

**Manual Backup:**

```bash
# Stop services
docker compose down

# Create backup
tar -czf cirame-backup-$(date +%Y%m%d).tar.gz docker-volumes/

# Restart services
docker compose up -d
```

### Restore from Backup

```bash
# Stop services
docker compose down

# Extract backup
tar -xzf cirame-backup-YYYYMMDD.tar.gz

# Restart services
docker compose up -d
```

### Migrating Existing Data

If you have existing data/models from a non-Docker installation:

```bash
# Copy existing data
cp -r /path/to/old/backend/data/* docker-volumes/data/
cp -r /path/to/old/backend/models/* docker-volumes/models/

# Set proper permissions (Linux/Mac)
chmod -R 755 docker-volumes/

# Start services
docker compose up -d
```

---

## Monitoring & Maintenance

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend

# Last 100 lines
docker compose logs --tail=100

# Since timestamp
docker compose logs --since 2024-01-01T10:00:00
```

### Check Health Status

```bash
# Service status
docker compose ps

# Health checks
docker inspect --format='{{.State.Health.Status}}' cirame-backend
docker inspect --format='{{.State.Health.Status}}' cirame-frontend
```

### Resource Monitoring

```bash
# Real-time stats
docker stats

# Specific container
docker stats cirame-backend
```

### Updating the Application

```bash
# Pull latest changes (if from git repository)
git pull

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d

# Or use the shorthand
docker compose up -d --build
```

### Scaling Services

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # Increase CPU
      memory: 16G      # Increase memory
    reservations:
      cpus: '4.0'
      memory: 8G
```

Then restart:

```bash
docker compose down
docker compose up -d
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error:** `Bind for 0.0.0.0:3030 failed: port is already allocated`

**Solution:**
```bash
# Check what's using the port
# Linux/Mac
sudo lsof -i :3030

# Windows
netstat -ano | findstr :3030

# Change port in .env
FRONTEND_PORT=8080
```

#### 2. Permission Denied (Volume Mounts)

**Error:** `Permission denied` when accessing volumes

**Solution (Linux/Mac):**
```bash
# Set ownership to Docker user (UID 1000)
sudo chown -R 1000:1000 docker-volumes/

# Or use current user
sudo chown -R $USER:$USER docker-volumes/
```

#### 3. Container Fails Health Check

**Check logs:**
```bash
docker compose logs backend
docker inspect cirame-backend
```

**Common causes:**
- Missing dependencies (rebuild image)
- Database corruption (restore from backup)
- Resource constraints (increase limits)

#### 4. Out of Memory

**Solution:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings -> Resources -> Memory -> Increase allocation

# Or reduce resource allocation in docker-compose.yml
```

#### 5. Build Failures

**Clear build cache:**
```bash
docker compose down
docker system prune -a
docker compose build --no-cache
docker compose up -d
```

### Advanced Debugging

#### Access Container Shell

```bash
# Backend
docker compose exec backend /bin/bash

# Frontend
docker compose exec frontend /bin/sh
```

#### Check Network Connectivity

```bash
# Test backend from frontend container
docker compose exec frontend curl http://backend:5100/api/health

# Test from host
curl http://localhost:5100/api/health
```

#### Inspect Volumes

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect cirame_backend-data

# Access volume data
docker run --rm -v cirame_backend-data:/data alpine ls -la /data
```

---

## Production Considerations

### Security Best Practices

1. **Network Security**
   - Use firewall rules to restrict access
   - Place behind reverse proxy with SSL/TLS
   - Implement rate limiting

2. **Container Security**
   - Regularly update base images
   - Scan for vulnerabilities: `docker scan cirame-backend`
   - Run containers as non-root (already configured)

3. **Data Security**
   - Encrypt volumes at rest
   - Regular backups with encryption
   - Implement access controls

### High Availability

For production deployments with high availability:

1. **Use Docker Swarm or Kubernetes** for orchestration
2. **Implement load balancing** with multiple frontend replicas
3. **Set up monitoring** with Prometheus/Grafana
4. **Configure log aggregation** with ELK stack or similar

### Performance Optimization

1. **Adjust Worker Processes**

Add Gunicorn to `backend/requirements.txt`:
```
gunicorn>=21.0.0
```

Update backend `CMD` in `backend/Dockerfile`:
```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:5100", "--workers", "4", "--timeout", "300", "run:app"]
```

2. **Enable Caching**
   - Redis for session storage
   - CDN for static assets
   - Database query caching

3. **Database Optimization**
   - Regular VACUUM for SQLite
   - Consider PostgreSQL for larger datasets

---

## Advanced Usage

### Custom Datasets

Mount custom datasets at runtime:

```bash
# In docker-compose.yml, add under backend volumes:
volumes:
  - ./my-dataset:/app/datasets/custom:ro
```

### Environment-Specific Builds

Create multiple compose files:

```bash
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    environment:
      - FLASK_DEBUG=False
    deploy:
      replicas: 2
```

Run with:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### CI/CD Integration

**Example GitLab CI:**

```yaml
build:
  stage: build
  script:
    - docker compose build
    - docker compose push

deploy:
  stage: deploy
  script:
    - docker compose pull
    - docker compose up -d
```

### Multi-Stage Development

For development with code hot-reload:

```yaml
# docker-compose.dev.yml
services:
  backend:
    volumes:
      - ./backend:/app
    environment:
      - FLASK_DEBUG=True

  frontend:
    volumes:
      - ./frontend/src:/app/src
```

Run with:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review container logs: `docker compose logs`
3. Consult the main [README.md](./README.md)
4. Contact support with log files and configuration details

---

## License

See [LICENSE](./LICENSE) file for details.

---

**CiRA ME - Machine Intelligence for Edge Computing**

Powered by Docker
