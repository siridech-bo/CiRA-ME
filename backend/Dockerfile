# syntax=docker/dockerfile:1
# CiRA ME - Backend Dockerfile
# Production-ready Python Flask API with ML dependencies

# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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

# PyTorch build args. Default is cu128 to match local dev boxes
# (NVIDIA driver 550+ / CUDA 12.8-12.9). For hosts with older drivers,
# override at build time:
#   docker build --build-arg TORCH_VERSION=2.0.1 \
#                --build-arg TORCH_CUDA=cu117 ...
# Currently supported combos:
#   cu128 -> torch==2.10.0  (driver 550+)
#   cu117 -> torch==2.0.1   (driver 515.x)
ARG TORCH_VERSION=2.10.0
ARG TORCH_CUDA=cu128

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --user to install in user directory for better security
# BuildKit cache mount keeps downloaded wheels across rebuilds
# Install PyTorch first from the correct CUDA index (separate URL).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Install remaining dependencies from PyPI (respect the torch pin already
# installed above — requirements.txt loosens the pin to a range so it
# does not fight with the version we just installed).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

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
