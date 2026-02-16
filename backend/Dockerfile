# CiRA ME - Backend Dockerfile
# Production-ready Python Flask API with ML dependencies

# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash cirame && \
    mkdir -p /app/data /app/models /app/datasets && \
    chown -R cirame:cirame /app

WORKDIR /app

# Stage 2: Dependencies installation
FROM base as builder

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --user to install in user directory for better security
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Runtime image
FROM base as runtime

# Copy Python packages from builder
COPY --from=builder --chown=cirame:cirame /root/.local /home/cirame/.local

# Update PATH to include user-installed packages
ENV PATH=/home/cirame/.local/bin:$PATH

# Copy application code
COPY --chown=cirame:cirame . .

# Create directories for persistent data
RUN mkdir -p /app/data /app/models /app/datasets && \
    chown -R cirame:cirame /app/data /app/models /app/datasets

# Switch to non-root user
USER cirame

# Set Flask environment variables
ENV FLASK_APP=run.py \
    FLASK_HOST=0.0.0.0 \
    FLASK_DEBUG=False \
    # Python optimizations
    PYTHONOPTIMIZE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5100/api/health || exit 1

# Expose Flask port
EXPOSE 5100

# Run the application using gunicorn for production
# Fallback to Flask development server if gunicorn not installed
CMD ["python", "run.py"]
