# syntax=docker/dockerfile:1
# CiRA ME - TI TinyML ModelMaker Service
# Separate container with Python 3.10 (required by tinyml-modelmaker)

FROM python:3.10-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Flask API dependencies first (small, cached)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Install tinyml-modelmaker from TI's GitHub (large, cached separately)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "tinyml_modelmaker @ git+https://github.com/TexasInstruments/tinyml-tensorlab.git@r1.3#subdirectory=tinyml-modelmaker"

# Copy API server code
COPY . .

# Expose API port
# Force CPU-only PyTorch (avoids CUDA driver/compute capability issues)
ENV CUDA_VISIBLE_DEVICES="" \
    TORCH_DEVICE="cpu" \
    USE_CUDA="0"

EXPOSE 5200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5200/health || exit 1

CMD ["python", "server.py"]
