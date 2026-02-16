# CiRA ME - Docker Architecture

This document describes the Docker architecture for the CiRA ME application.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host Machine                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Docker Network                           │ │
│  │                  (cirame-network)                           │ │
│  │                  172.20.0.0/16                              │ │
│  │                                                             │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐       │ │
│  │  │   Frontend Container │  │   Backend Container   │       │ │
│  │  │   (cirame-frontend)  │  │   (cirame-backend)    │       │ │
│  │  │                      │  │                       │       │ │
│  │  │  ┌────────────────┐  │  │  ┌─────────────────┐ │       │ │
│  │  │  │  Nginx Server  │  │  │  │  Flask API      │ │       │ │
│  │  │  │  Port: 80      │  │  │  │  Port: 5100     │ │       │ │
│  │  │  │                │  │  │  │                 │ │       │ │
│  │  │  │  Vue.js SPA    │◄─┼──┼──┤  /api/health    │ │       │ │
│  │  │  │  Static Files  │  │  │  │  /api/data/*    │ │       │ │
│  │  │  │                │  │  │  │  /api/features/*│ │       │ │
│  │  │  └────────────────┘  │  │  │  /api/training/*│ │       │ │
│  │  │         │            │  │  │                 │ │       │ │
│  │  │         │ Proxy      │  │  └─────────────────┘ │       │ │
│  │  │         │ /api/* ────┼──┼──►     │            │       │ │
│  │  │                      │  │       │            │       │ │
│  │  └──────────────────────┘  └───────┼────────────┘       │ │
│  │           │                         │                    │ │
│  └───────────┼─────────────────────────┼────────────────────┘ │
│              │                         │                      │
│         Port 3030                  Port 5100                  │
│              │                         │                      │
│              ▼                         ▼                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 Volume Mounts (Persistent)               │ │
│  │                                                          │ │
│  │  ./docker-volumes/data     ──► SQLite DB, App State     │ │
│  │  ./docker-volumes/models   ──► ML Models (*.pkl)        │ │
│  │  ./docker-volumes/datasets ──► Training Data            │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

External Access:
  http://localhost:3030  ──► Frontend (Nginx serves Vue.js app)
  http://localhost:5100  ──► Backend API (direct access, optional)
```

## Container Architecture

### Frontend Container

**Base Image:** `nginx:1.25-alpine`

**Build Process:**
1. **Stage 1 (Builder)**:
   - Base: `node:20-alpine`
   - Install npm dependencies
   - Build Vue.js application with Vite
   - Output: Static files in `/app/dist`

2. **Stage 2 (Production)**:
   - Base: `nginx:1.25-alpine`
   - Copy built files from Stage 1
   - Copy custom Nginx configuration
   - Run as non-root user (nginx)

**Key Features:**
- Multi-stage build reduces image size (~150MB final)
- Nginx serves static files with gzip compression
- API requests proxied to backend container
- Client-side routing support for Vue Router
- Security headers enabled
- Health checks every 30 seconds

**Exposed Ports:**
- 80 (container) → 3030 (host)

**Resource Limits:**
- CPU: 0.5-1.0 cores
- Memory: 256MB-512MB

### Backend Container

**Base Image:** `python:3.11-slim`

**Build Process:**
1. **Stage 1 (Base)**:
   - Install system dependencies (gcc, g++, gfortran)
   - Install OpenBLAS and LAPACK for numpy/scipy
   - Create non-root user (cirame)

2. **Stage 2 (Builder)**:
   - Install Python packages from requirements.txt
   - Use --user flag for security

3. **Stage 3 (Runtime)**:
   - Copy Python packages from builder
   - Copy application code
   - Set environment variables
   - Configure health checks

**Key Features:**
- Optimized for ML dependencies (pandas, numpy, scikit-learn)
- CPU-only deployment (CUDA disabled in container)
- Non-root user execution
- Python optimization enabled
- Health check endpoint monitoring
- Auto-restart on failure

**Exposed Ports:**
- 5100 (container) → 5100 (host)

**Resource Limits:**
- CPU: 2.0-4.0 cores (8.0 in production)
- Memory: 4GB-8GB (16GB in production)

**Environment Variables:**
- `FLASK_DEBUG`: False (production)
- `FLASK_HOST`: 0.0.0.0
- `PYTHONOPTIMIZE`: 1
- `CUDA_VISIBLE_DEVICES`: "" (CPU only)

## Network Architecture

### Docker Bridge Network

**Network Name:** `cirame-network`
**Subnet:** 172.20.0.0/16
**Driver:** bridge

**Service Communication:**
- Frontend → Backend: `http://backend:5100`
- Host → Frontend: `http://localhost:3030`
- Host → Backend: `http://localhost:5100` (optional)

**DNS Resolution:**
Docker's embedded DNS server resolves container names to IP addresses, allowing the frontend to communicate with the backend using the service name `backend`.

## Data Persistence

### Volume Strategy

CiRA ME uses bind mounts for transparent data access and easy backup:

```yaml
volumes:
  backend-data:
    type: bind
    source: ./docker-volumes/data
    target: /app/data

  backend-models:
    type: bind
    source: ./docker-volumes/models
    target: /app/models

  backend-datasets:
    type: bind
    source: ./docker-volumes/datasets
    target: /app/datasets
```

### Data Flow

```
User Upload → Frontend → Backend API → SQLite DB
                                    → Pickle Models
                                    → CSV Datasets

Training Process → Model Files (.pkl) → /app/models → Host Volume
Results → Database Records → /app/data/cirame.db → Host Volume
```

### Backup & Recovery

**Backup:**
```bash
./deploy.sh backup
# Creates: backups/YYYYMMDD_HHMMSS.tar.gz
```

**Restore:**
```bash
./deploy.sh restore backups/YYYYMMDD_HHMMSS.tar.gz
```

## Request Flow

### Typical API Request Flow

```
1. User Browser
   │
   ▼
2. http://localhost:3030/api/data/sources
   │
   ▼
3. Frontend Container (Nginx)
   │ (Nginx proxy_pass)
   ▼
4. http://backend:5100/api/data/sources
   │
   ▼
5. Backend Container (Flask)
   │ (Flask route handler)
   ▼
6. Database Query / File Operation
   │
   ▼
7. JSON Response
   │
   ▼
8. Frontend (Nginx) ─► User Browser
```

### Static File Request Flow

```
1. User Browser
   │
   ▼
2. http://localhost:3030/assets/index.js
   │
   ▼
3. Frontend Container (Nginx)
   │ (Nginx static file serving)
   ▼
4. /usr/share/nginx/html/assets/index.js
   │
   ▼
5. File Response (with gzip) ─► User Browser
```

## Health Monitoring

### Health Check Configuration

**Backend:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3
  CMD python -c "import requests; requests.get('http://localhost:5100/api/health', timeout=5)"
```

**Frontend:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3
  CMD curl -f http://localhost:80/
```

### Health States

- **starting**: Container starting, grace period active
- **healthy**: Health check passing
- **unhealthy**: Health check failing (3 consecutive failures)

### Monitoring Commands

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' cirame-backend
docker inspect --format='{{.State.Health.Status}}' cirame-frontend

# View health check logs
docker inspect --format='{{json .State.Health}}' cirame-backend | jq

# Monitor resource usage
docker stats cirame-backend cirame-frontend
```

## Security Architecture

### Container Security

**1. Non-Root Users**
- Backend runs as user `cirame` (UID 1000)
- Frontend runs as user `nginx`

**2. Read-Only Root Filesystem**
- Writable volumes mounted only where needed
- Application code is read-only

**3. Resource Limits**
- CPU and memory caps prevent resource exhaustion
- Prevents DoS attacks

**4. Network Isolation**
- Containers communicate via private bridge network
- Only necessary ports exposed to host

**5. Minimal Base Images**
- Alpine Linux for frontend (small attack surface)
- Debian Slim for backend (balance of size and compatibility)

### Application Security

**1. Nginx Security Headers**
```nginx
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: no-referrer-when-downgrade
```

**2. CORS Configuration**
- Configured in Flask backend
- Restricts cross-origin requests

**3. File Upload Limits**
```nginx
client_max_body_size 100M;
```

**4. Timeout Protection**
```nginx
proxy_read_timeout 300s;  # 5 minutes for ML operations
```

## Deployment Modes

### Development Mode

```bash
docker compose up -d
```

**Characteristics:**
- Logs visible in terminal
- Debug mode available
- Lower resource limits
- Faster startup

### Production Mode

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Characteristics:**
- Python optimization enabled (`PYTHONOPTIMIZE=2`)
- Higher resource limits (8 CPU, 16GB RAM)
- Structured logging with rotation
- Auto-restart always
- No debug mode

## Scaling Considerations

### Horizontal Scaling

To scale the application:

**Option 1: Docker Compose (Limited)**
```bash
docker compose up -d --scale backend=3
```

**Option 2: Kubernetes (Recommended for Production)**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cirame-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cirame-backend
  template:
    metadata:
      labels:
        app: cirame-backend
    spec:
      containers:
      - name: backend
        image: ghcr.io/yourorg/cirame-backend:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Load Balancing

For production deployments, use a load balancer:

**Nginx Load Balancer Example:**
```nginx
upstream cirame_backend {
    least_conn;
    server backend1:5100;
    server backend2:5100;
    server backend3:5100;
}

upstream cirame_frontend {
    server frontend1:80;
    server frontend2:80;
}
```

### Database Considerations

Current setup uses SQLite (single file). For scaling:

1. **Option 1**: Shared volume (NFS/EFS) for SQLite
2. **Option 2**: Migrate to PostgreSQL/MySQL (recommended)
3. **Option 3**: Use separate database container

## Troubleshooting Architecture Issues

### Container Cannot Connect to Backend

**Symptom:** Frontend shows "Network Error"

**Check:**
1. Both containers on same network: `docker network inspect cirame_cirame-network`
2. Backend is healthy: `docker inspect cirame-backend | grep Health`
3. Network connectivity: `docker compose exec frontend curl http://backend:5100/api/health`

### Volume Permission Issues

**Symptom:** "Permission denied" in logs

**Fix:**
```bash
# Linux/Mac
sudo chown -R 1000:1000 docker-volumes/

# Windows
# Run Docker Desktop as Administrator
```

### Out of Memory

**Symptom:** Container killed by OOM

**Fix:**
1. Increase Docker Desktop memory allocation
2. Reduce resource limits in compose file
3. Optimize ML model size

## Performance Optimization

### Image Size Optimization

**Current Sizes:**
- Frontend: ~150MB (multi-stage build)
- Backend: ~1.2GB (Python + ML libraries)

**Further Optimization:**
- Use `.dockerignore` (already implemented)
- Multi-stage builds (already implemented)
- Layer caching (already implemented)

### Build Time Optimization

**Use Build Cache:**
```bash
docker compose build --parallel
```

**Use BuildKit:**
```bash
DOCKER_BUILDKIT=1 docker compose build
```

### Runtime Performance

**Backend:**
- Consider using Gunicorn with multiple workers
- Implement request caching (Redis)
- Use connection pooling for database

**Frontend:**
- Static file compression (already enabled)
- Browser caching (already enabled)
- CDN for static assets (for public deployment)

## CI/CD Integration

The included GitHub Actions workflow provides:

1. **Automated builds** on push to main/master
2. **Multi-platform support** (amd64, arm64)
3. **Security scanning** with Trivy
4. **Container registry** push to GitHub Container Registry
5. **Semantic versioning** from git tags

See `.github/workflows/docker-build.yml` for details.

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Vue.js Documentation](https://vuejs.org/)

---

**CiRA ME - Machine Intelligence for Edge Computing**

For deployment instructions, see [README-Docker.md](./README-Docker.md)
