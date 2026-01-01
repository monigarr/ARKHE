# ARKHE Framework Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder stage for installing dependencies
FROM base as builder

# Set working directory
WORKDIR /build

# Copy dependency files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user -r requirements.txt

# Stage 3: Production image
FROM base as production

# Create non-root user for security
RUN useradd -m -u 1000 arkhe && \
    mkdir -p /app /data /checkpoints && \
    chown -R arkhe:arkhe /app /data /checkpoints

# Copy installed packages from builder
COPY --from=builder /root/.local /home/arkhe/.local

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=arkhe:arkhe . .

# Install the package in development mode
RUN pip install --no-cache-dir -e . && \
    chown -R arkhe:arkhe /home/arkhe/.local

# Add local bin to PATH
ENV PATH=/home/arkhe/.local/bin:$PATH

# Switch to non-root user
USER arkhe

# Expose Streamlit default port
EXPOSE 8501

# Set default command (can be overridden)
CMD ["python", "-m", "src.apps.cli", "--help"]

# Stage 4: Development image with dev dependencies
FROM production as development

USER root

# Install development dependencies
COPY --from=builder /root/.local /root/.local
RUN pip install --no-cache-dir -r requirements-dev.txt && \
    chown -R arkhe:arkhe /root/.local

USER arkhe

# Default to development mode
ENV ENVIRONMENT=development

# Stage 5: Streamlit-specific image
FROM production as streamlit

# Install streamlit if not already in requirements
RUN pip install --no-cache-dir --user streamlit || true

# Set Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default command for Streamlit
CMD ["streamlit", "run", "src/apps/streamlit_demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 6: CUDA-enabled image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as cuda

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd -m -u 1000 arkhe && \
    mkdir -p /app /data /checkpoints && \
    chown -R arkhe:arkhe /app /data /checkpoints

WORKDIR /app

# Copy project files
COPY --chown=arkhe:arkhe . .

# Install dependencies (PyTorch with CUDA support)
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -e . && \
    chown -R arkhe:arkhe /app

USER arkhe

EXPOSE 8501

CMD ["python3", "-m", "src.apps.cli", "--help"]

