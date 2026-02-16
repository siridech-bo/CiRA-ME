# CiRA ME - Docker Quick Reference Guide

Quick reference for common Docker operations with CiRA ME.

## Quick Start

```bash
# Linux/Mac
./deploy.sh setup    # First time setup
./deploy.sh build    # Build images
./deploy.sh start    # Start services

# Windows
.\deploy.ps1 setup   # First time setup
.\deploy.ps1 build   # Build images
.\deploy.ps1 start   # Start services
```

## Common Commands

### Basic Operations

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Restart services
docker compose restart

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Development

```bash
# Start with live logs
docker compose up

# Rebuild and start
docker compose up -d --build

# View backend logs only
docker compose logs -f backend

# Access backend shell
docker compose exec backend /bin/bash

# Access frontend shell
docker compose exec frontend /bin/sh
```

### Production

```bash
# Start in production mode
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check health status
docker inspect --format='{{.State.Health.Status}}' cirame-backend
docker inspect --format='{{.State.Health.Status}}' cirame-frontend

# View resource usage
docker stats
```

### Maintenance

```bash
# Update images
docker compose pull
docker compose up -d

# Rebuild everything
docker compose build --no-cache
docker compose up -d

# Clean up
docker compose down -v              # Remove volumes too
docker system prune -a             # Clean all unused resources
```

### Data Management

```bash
# Backup data (manual)
tar -czf backup-$(date +%Y%m%d).tar.gz docker-volumes/

# Restore data
tar -xzf backup-YYYYMMDD.tar.gz

# View volume data
docker run --rm -v cirame_backend-data:/data alpine ls -la /data
```

### Debugging

```bash
# Check logs with timestamps
docker compose logs -f --timestamps

# Last 100 log lines
docker compose logs --tail=100

# Inspect container
docker inspect cirame-backend

# Test network connectivity
docker compose exec frontend curl http://backend:5100/api/health

# Check disk usage
docker system df
```

### URLs

- **Frontend**: http://localhost:3030
- **Backend**: http://localhost:5100
- **Health Check**: http://localhost:5100/api/health

### Helper Scripts

#### Linux/Mac

```bash
./deploy.sh setup       # Initial setup
./deploy.sh start dev   # Start development
./deploy.sh start prod  # Start production
./deploy.sh logs        # View logs
./deploy.sh status      # Check status
./deploy.sh backup      # Backup data
./deploy.sh update      # Update application
```

#### Windows

```powershell
.\deploy.ps1 setup       # Initial setup
.\deploy.ps1 start dev   # Start development
.\deploy.ps1 start prod  # Start production
.\deploy.ps1 logs        # View logs
.\deploy.ps1 status      # Check status
.\deploy.ps1 backup      # Backup data
.\deploy.ps1 update      # Update application
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port (Linux/Mac)
sudo lsof -i :3030

# Find process using port (Windows)
netstat -ano | findstr :3030

# Change port in .env
FRONTEND_PORT=8080
```

### Permission Issues (Linux/Mac)

```bash
# Fix volume permissions
sudo chown -R 1000:1000 docker-volumes/
```

### Container Won't Start

```bash
# Check logs
docker compose logs backend

# Rebuild from scratch
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Out of Disk Space

```bash
# Clean up unused resources
docker system prune -a
docker volume prune
```

## Environment Variables

Edit `.env` file:

```bash
# Frontend port
FRONTEND_PORT=3030

# Volume paths
CIRAME_DATA_PATH=./docker-volumes/data
CIRAME_MODELS_PATH=./docker-volumes/models
CIRAME_DATASETS_PATH=./docker-volumes/datasets

# Debug mode (False for production)
FLASK_DEBUG=False
```

## File Structure

```
CiRA ME/
├── docker-compose.yml           # Main compose file
├── docker-compose.prod.yml      # Production overrides
├── .env                         # Environment variables
├── deploy.sh                    # Linux/Mac helper script
├── deploy.ps1                   # Windows helper script
├── backend/
│   ├── Dockerfile              # Backend image
│   └── .dockerignore
├── frontend/
│   ├── Dockerfile              # Frontend image
│   ├── nginx.conf              # Nginx configuration
│   └── .dockerignore
└── docker-volumes/             # Persistent data
    ├── data/                   # Database
    ├── models/                 # ML models
    └── datasets/               # Training data
```

## Performance Tips

1. **Allocate sufficient resources** in Docker Desktop (Settings > Resources)
   - Recommended: 8GB RAM, 4 CPUs

2. **Use production mode** for better performance
   ```bash
   ./deploy.sh start prod
   ```

3. **Monitor resource usage**
   ```bash
   docker stats
   ```

4. **Clean up regularly**
   ```bash
   docker system prune
   ```

## Security Checklist

- [ ] Change default passwords (if any)
- [ ] Set `FLASK_DEBUG=False` in production
- [ ] Use HTTPS with reverse proxy (nginx/traefik)
- [ ] Implement firewall rules
- [ ] Regular security updates
- [ ] Backup data regularly
- [ ] Monitor container logs

## Support

For detailed documentation, see [README-Docker.md](./README-Docker.md)

---

**Quick help:** Run `./deploy.sh help` or `.\deploy.ps1 help` for available commands.
