# Production Deployment Guide

This guide covers deploying the ARKHE Framework in production environments with best practices for security, monitoring, and scalability.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Deployment](#docker-deployment)
4. [Configuration Management](#configuration-management)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Considerations](#security-considerations)
7. [Scaling & Performance](#scaling--performance)
8. [Backup & Recovery](#backup--recovery)

## Overview

The ARKHE Framework can be deployed in various production scenarios:
- **CLI Application**: Batch processing and scheduled tasks
- **Streamlit Web Application**: Interactive web interface
- **API Service**: RESTful API (future enhancement)
- **Training Pipeline**: Distributed model training

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for training)
- **Storage**: 10GB+ free space for models and data
- **GPU**: Optional but recommended for training (CUDA 11.8+)

### Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Docker Deployment

### Quick Start

```bash
# Build production image
docker build -t arkhe:latest --target production .

# Run container
docker run -d \
  --name arkhe \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  arkhe:latest
```

### Docker Compose

For multi-service deployments:

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d streamlit

# View logs
docker-compose logs -f streamlit
```

See [Docker Setup Guide](docker_setup.md) for detailed instructions.

## Configuration Management

### Environment Variables

Create a `.env` file for production:

```bash
# Application Settings
ARKHE_ENV=production
ARKHE_LOG_LEVEL=INFO
ARKHE_DATA_DIR=/app/data
ARKHE_CHECKPOINT_DIR=/app/checkpoints

# Streamlit Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Security
ARKHE_SECRET_KEY=your-secret-key-here
ARKHE_ALLOWED_HOSTS=yourdomain.com,*.yourdomain.com

# Monitoring
ARKHE_METRICS_ENABLED=true
ARKHE_METRICS_PORT=9090
```

### Production Configuration File

Create `configs/production.yaml`:

```yaml
# Production Configuration
environment: production

# Logging
logging:
  level: INFO
  format: json  # JSON format for log aggregation
  file: /app/logs/arkhe.log
  max_size_mb: 100
  backup_count: 10

# Data Management
data:
  directory: /app/data
  checkpoint_directory: /app/checkpoints
  backup_enabled: true
  backup_interval_hours: 24

# Model Training
training:
  device: cuda  # or cpu
  batch_size: 32
  num_workers: 4
  pin_memory: true

# Security
security:
  enable_cors: true
  allowed_origins:
    - https://yourdomain.com
  rate_limiting:
    enabled: true
    requests_per_minute: 60

# Monitoring
monitoring:
  health_check_interval: 30
  metrics_enabled: true
  metrics_port: 9090
```

## Monitoring & Observability

### Health Checks

The framework includes built-in health checks:

```python
from math_research.utils.health import get_health_status, is_healthy

# Check overall health
if is_healthy():
    print("System is healthy")

# Get detailed status
status = get_health_status()
print(status)
```

### Metrics Export

Enable Prometheus-compatible metrics:

```python
from math_research.utils.metrics import MetricsExporter

# Initialize metrics exporter
exporter = MetricsExporter(port=9090)

# Start metrics server
exporter.start()

# Export custom metrics
exporter.record_training_metric("loss", 0.5, step=100)
exporter.record_inference_latency(0.025)
```

### Logging

Configure structured logging for production:

```python
from math_research.utils.logging import setup_logging

setup_logging(
    level="INFO",
    format="json",  # JSON format for log aggregation
    log_file="/app/logs/arkhe.log"
)
```

### Log Aggregation

For production, integrate with log aggregation services:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki** (Grafana Loki)
- **CloudWatch** (AWS)
- **Datadog**

Example log format:
```json
{
  "timestamp": "2025-01-09T10:30:00Z",
  "level": "INFO",
  "logger": "math_research.ml.training",
  "message": "Training epoch 10 completed",
  "metrics": {
    "loss": 0.45,
    "accuracy": 0.92
  }
}
```

## Security Considerations

### Secrets Management

**Never commit secrets to version control!**

Use environment variables or secret management services:

```bash
# Use environment variables
export ARKHE_API_KEY=your-api-key
export ARKHE_DB_PASSWORD=your-password

# Or use secret management
# AWS Secrets Manager
# HashiCorp Vault
# Kubernetes Secrets
```

### Input Validation

Always validate user inputs:

```python
from math_research.utils.validators import validate_positive_int

# Validate input
try:
    start_value = validate_positive_int(user_input, "start_value")
except ValueError as e:
    # Handle validation error
    pass
```

### Rate Limiting

For web applications, implement rate limiting:

```python
# Example with Flask (if using Flask API)
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

### CORS Configuration

Configure CORS for web applications:

```yaml
# configs/production.yaml
security:
  enable_cors: true
  allowed_origins:
    - https://yourdomain.com
    - https://app.yourdomain.com
```

## Scaling & Performance

### Horizontal Scaling

For Streamlit applications, use multiple instances behind a load balancer:

```yaml
# docker-compose.yml
services:
  streamlit:
    deploy:
      replicas: 3
    environment:
      - STREAMLIT_SERVER_ENABLE_CORS=true
```

### Caching

Implement caching for frequently accessed data:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_sequence_stats(start_value: int):
    # Cached sequence statistics
    pass
```

### Database (Future Enhancement)

For production, consider using a database for:
- Sequence storage
- Model metadata
- Experiment tracking

Options:
- **PostgreSQL**: Relational database
- **MongoDB**: Document database
- **Redis**: Caching and session storage

## Backup & Recovery

### Model Checkpoints

Regularly backup model checkpoints:

```bash
# Backup checkpoints
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz checkpoints/

# Restore from backup
tar -xzf checkpoints_backup_20250109.tar.gz
```

### Automated Backups

Set up automated backups:

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/arkhe"
mkdir -p $BACKUP_DIR

# Backup checkpoints
tar -czf $BACKUP_DIR/checkpoints_$DATE.tar.gz checkpoints/

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Keep only last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

1. **Regular Backups**: Daily automated backups
2. **Off-site Storage**: Store backups in cloud storage (S3, GCS)
3. **Recovery Testing**: Regularly test recovery procedures
4. **Documentation**: Document recovery procedures

## Deployment Checklist

Before deploying to production:

- [ ] All environment variables configured
- [ ] Secrets stored securely (not in code)
- [ ] Health checks configured
- [ ] Monitoring and alerting set up
- [ ] Logging configured and aggregated
- [ ] Backup strategy implemented
- [ ] Security review completed
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Rollback plan prepared

## Troubleshooting

### High Memory Usage

```bash
# Monitor memory usage
docker stats arkhe

# Reduce batch size in training config
# configs/production.yaml
training:
  batch_size: 16  # Reduce from 32
```

### Slow Performance

1. Check GPU availability
2. Verify data loading is optimized
3. Enable caching where appropriate
4. Review logging overhead

### Connection Issues

```bash
# Check container logs
docker logs arkhe

# Check network connectivity
docker exec arkhe ping google.com

# Verify port mappings
docker port arkhe
```

## Additional Resources

- [Docker Setup Guide](docker_setup.md)
- [Security Policy](../../SECURITY.md)
- [API Documentation](../api/)
- [Training Guide](training_guide.md)

