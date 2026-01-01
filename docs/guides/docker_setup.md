# Docker Setup Guide

This guide explains how to use Docker and Docker Compose with the ARKHE Framework.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+ (included with Docker Desktop)
- For GPU support: NVIDIA Docker runtime (nvidia-docker2)

## Quick Start

### Build the Docker Image

```bash
# Build production image
docker build -t arkhe:latest .

# Build Streamlit image
docker build --target streamlit -t arkhe:streamlit .

# Build development image
docker build --target development -t arkhe:dev .
```

### Run with Docker

```bash
# Run CLI help
docker run --rm arkhe:latest python -m src.apps.cli --help

# Generate a sequence
docker run --rm -v $(pwd)/data:/data arkhe:latest python -m src.apps.cli generate --start 27

# Run Streamlit (expose port 8501)
docker run --rm -p 8501:8501 -v $(pwd)/data:/data arkhe:streamlit
```

## Using Docker Compose

Docker Compose provides multiple service profiles for different use cases.

### Streamlit Web Interface

```bash
# Start Streamlit service
docker-compose --profile streamlit up

# Access at http://localhost:8501
```

### CLI Commands

```bash
# Run CLI commands
docker-compose --profile cli run arkhe-cli python -m src.apps.cli generate --start 27

# Train a model
docker-compose --profile cli run arkhe-cli python -m src.apps.cli train --num-samples 10000 --epochs 10
```

### Development Environment

```bash
# Start development container (stays running)
docker-compose --profile dev up -d

# Execute commands in dev container
docker-compose --profile dev exec arkhe-dev python -m src.apps.cli generate --start 27

# View logs
docker-compose --profile dev logs -f

# Stop dev container
docker-compose --profile dev down
```

### GPU Support (CUDA)

```bash
# Build GPU image
docker build --target cuda -t arkhe:cuda .

# Run with GPU
docker-compose --profile gpu run arkhe-gpu python -m src.apps.cli train --num-samples 10000 --epochs 10
```

**Note:** GPU support requires:
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- Docker configured for GPU access

## Volume Mounts

The docker-compose.yml mounts the following directories:

- `./data:/data` - Data files
- `./checkpoints:/checkpoints` - Model checkpoints
- `./configs:/app/configs` - Configuration files

## Environment Variables

You can set environment variables in docker-compose.yml or via `.env` file:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - ENVIRONMENT=production
```

## Building for Different Targets

The Dockerfile supports multiple build targets:

- `base` - Base image with Python
- `builder` - Dependency installation stage
- `production` - Optimized production image (default)
- `development` - Development image with dev dependencies
- `streamlit` - Streamlit-specific image
- `cuda` - GPU-enabled image

## Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER data checkpoints

# Or run container with your user ID
docker run --rm -u $(id -u):$(id -g) arkhe:latest
```

### Out of Memory

For large models, increase Docker memory limits:
- Docker Desktop: Settings → Resources → Memory
- Or use `--memory` flag: `docker run --memory="4g" ...`

### GPU Not Detected

Verify GPU access:

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Best Practices

1. **Use .dockerignore** - Excludes unnecessary files from build context
2. **Multi-stage builds** - Reduces final image size
3. **Non-root user** - Containers run as non-root for security
4. **Volume mounts** - Persist data and checkpoints outside container
5. **Profile-based services** - Only start what you need

## Examples

### Training a Model

```bash
docker-compose --profile cli run arkhe-cli \
  python -m src.apps.cli train \
  --num-samples 50000 \
  --epochs 20 \
  --batch-size 32 \
  --checkpoint-dir /checkpoints
```

### Running Analysis

```bash
docker-compose --profile cli run arkhe-cli \
  python -m src.apps.cli analyze \
  --start 1 \
  --end 1000 \
  --output /data/analysis.json
```

### Interactive Development

```bash
# Start dev container
docker-compose --profile dev up -d

# Run Python interactively
docker-compose --profile dev exec arkhe-dev python

# Run tests
docker-compose --profile dev exec arkhe-dev pytest tests/
```

## Production Deployment

For production deployments:

1. Use the `production` target
2. Set appropriate resource limits
3. Use secrets management for sensitive data
4. Enable health checks
5. Set up logging aggregation
6. Use orchestration (Kubernetes, Docker Swarm)

Example production docker-compose:

```yaml
services:
  arkhe:
    build:
      context: .
      target: production
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "python", "-m", "src.apps.cli", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

**Last Updated:** 2025-01-09  
**Version:** 0.1.0

