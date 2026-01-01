# Observability Guide

This guide covers distributed tracing and enhanced observability features in the ARKHE Framework.

## Table of Contents

1. [Overview](#overview)
2. [Distributed Tracing](#distributed-tracing)
3. [Request/Response Logging](#requestresponse-logging)
4. [Performance Profiling](#performance-profiling)
5. [Error Tracking](#error-tracking)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)

## Overview

The ARKHE Framework provides comprehensive observability features:

- **Distributed Tracing**: Track requests and operations across services
- **Request/Response Logging**: Log all incoming requests and outgoing responses
- **Performance Profiling**: Profile function execution times
- **Error Tracking**: Integrate with error tracking services (Sentry)

## Distributed Tracing

### Overview

Distributed tracing allows you to track requests and operations as they flow through your application. This is especially useful for:
- Understanding request flows
- Identifying performance bottlenecks
- Debugging distributed systems
- Monitoring service dependencies

### Setup

#### Install Dependencies

```bash
# Install OpenTelemetry
pip install opentelemetry-api opentelemetry-sdk

# For OTLP exporter (recommended for production)
pip install opentelemetry-exporter-otlp

# For Jaeger exporter
pip install opentelemetry-exporter-jaeger

# For Zipkin exporter
pip install opentelemetry-exporter-zipkin
```

#### Basic Usage

```python
from math_research.utils.tracing import get_tracing_manager, trace_function

# Initialize tracing manager
tracer = get_tracing_manager(
    service_name="arkhe-framework",
    enabled=True,
    exporter="console",  # or "otlp", "jaeger", "zipkin"
    endpoint="http://localhost:4317"  # For OTLP
)

# Use context manager
with tracer.start_span("my_operation", attributes={"key": "value"}):
    # Your code here
    result = perform_operation()

# Use decorator
@trace_function(name="compute_sequence", attributes={"module": "sequences"})
def compute_sequence(start: int):
    # Function is automatically traced
    return sequence
```

### Exporters

#### Console Exporter (Development)

```python
tracer = get_tracing_manager(exporter="console")
```

#### OTLP Exporter (Production)

```python
tracer = get_tracing_manager(
    exporter="otlp",
    endpoint="http://jaeger:4317"  # OTLP endpoint
)
```

#### Jaeger Exporter

```python
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure Jaeger exporter
tracer = get_tracing_manager(
    exporter="jaeger",
    endpoint="http://localhost:14268/api/traces"
)
```

### Span Attributes

Add custom attributes to spans:

```python
with tracer.start_span("training_epoch") as span:
    span.set_attribute("epoch", 10)
    span.set_attribute("batch_size", 32)
    span.set_attribute("learning_rate", 0.001)
    
    # Your training code
    train_epoch()
```

## Request/Response Logging

### Overview

Request/response logging automatically logs all incoming requests and outgoing responses with timing information.

### Usage

```python
from math_research.utils.observability import RequestLogger

# Initialize logger
request_logger = RequestLogger(enabled=True, log_level="INFO")

# Log a request
context = request_logger.log_request(
    method="POST",
    path="/api/train",
    params={"epochs": 10, "batch_size": 32},
    headers={"Content-Type": "application/json"}
)

# Perform operation
try:
    result = train_model(epochs=10, batch_size=32)
    duration = time.time() - start_time
    
    # Log successful response
    request_logger.log_response(
        context,
        status="success",
        duration=duration,
        response_data={"model_id": "model_123"}
    )
except Exception as e:
    # Log error response
    request_logger.log_response(
        context,
        status="error",
        duration=time.time() - start_time,
        error=e
    )
```

### Log Format

Request logs include:
- Request ID (for correlation)
- Method and path
- Parameters
- Timestamp
- Duration
- Response status
- Error information (if any)

## Performance Profiling

### Overview

Performance profiling tracks execution times for functions to identify performance bottlenecks.

### Usage

```python
from math_research.utils.observability import PerformanceProfiler

# Initialize profiler
profiler = PerformanceProfiler(enabled=True)

# Use decorator
@profiler.profile_function()
def expensive_operation():
    # This function's execution time will be tracked
    return result

# Get statistics
stats = profiler.get_statistics("expensive_operation")
print(f"Average time: {stats['mean']:.4f}s")
print(f"Min: {stats['min']:.4f}s, Max: {stats['max']:.4f}s")
print(f"Total calls: {stats['count']}")

# Get all statistics
all_stats = profiler.get_all_statistics()
for func_name, stats in all_stats.items():
    print(f"{func_name}: {stats['mean']:.4f}s average")
```

### Statistics

The profiler tracks:
- **count**: Number of calls
- **total**: Total execution time
- **mean**: Average execution time
- **min**: Minimum execution time
- **max**: Maximum execution time

## Error Tracking

### Overview

Error tracking integrates with services like Sentry to capture and track errors in production.

### Setup

#### Install Sentry SDK

```bash
pip install sentry-sdk
```

#### Configuration

```python
from math_research.utils.observability import ErrorTracker

# Initialize error tracker
error_tracker = ErrorTracker(
    enabled=True,
    sentry_dsn="https://your-sentry-dsn@sentry.io/project-id",
    environment="production"
)
```

### Usage

```python
# Capture exceptions
try:
    risky_operation()
except Exception as e:
    error_tracker.capture_exception(
        e,
        context={
            "user_id": user_id,
            "operation": "train_model",
            "config": config_dict
        }
    )

# Capture messages
error_tracker.capture_message(
    "Model training started",
    level="info",
    context={"model_id": "model_123"}
)
```

### Environment Variables

Set Sentry DSN via environment variable:

```bash
export SENTRY_DSN="https://your-sentry-dsn@sentry.io/project-id"
```

```python
import os
from math_research.utils.observability import ErrorTracker

error_tracker = ErrorTracker(
    enabled=True,
    sentry_dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "development")
)
```

## Unified Observability Manager

### Overview

The `ObservabilityManager` combines all observability features into a single interface.

### Usage

```python
from math_research.utils.observability import get_observability_manager

# Initialize manager
obs = get_observability_manager(
    request_logging=True,
    profiling=True,
    error_tracking=True,
    sentry_dsn=os.getenv("SENTRY_DSN"),
    environment="production"
)

# Trace a request
context = obs.trace_request("POST", "/api/train", params={"epochs": 10})

try:
    start_time = time.time()
    result = train_model(epochs=10)
    duration = time.time() - start_time
    
    obs.trace_response(context, status="success", duration=duration)
except Exception as e:
    obs.error_tracker.capture_exception(e, context={"request": context})
    obs.trace_response(context, status="error", error=e)
```

## Configuration

### Environment Variables

```bash
# Tracing
OTEL_SERVICE_NAME=arkhe-framework
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Error Tracking
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
ENVIRONMENT=production

# Observability
OBSERVABILITY_ENABLED=true
REQUEST_LOGGING_ENABLED=true
PROFILING_ENABLED=true
ERROR_TRACKING_ENABLED=true
```

### Configuration File

```yaml
# configs/observability.yaml
observability:
  enabled: true
  request_logging:
    enabled: true
    log_level: INFO
  profiling:
    enabled: true
  error_tracking:
    enabled: true
    sentry_dsn: ${SENTRY_DSN}
    environment: ${ENVIRONMENT}
  tracing:
    enabled: true
    service_name: arkhe-framework
    exporter: otlp
    endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}
```

## Best Practices

### 1. Use Tracing for Critical Operations

```python
@trace_function(name="model_training", attributes={"model_type": "transformer"})
def train_model(config):
    with tracer.start_span("data_loading"):
        data = load_data()
    
    with tracer.start_span("training_loop"):
        model = train(data, config)
    
    return model
```

### 2. Profile Performance-Critical Functions

```python
@profiler.profile_function()
def expensive_computation(input_data):
    # Automatically profiled
    return result
```

### 3. Capture Errors with Context

```python
try:
    result = operation()
except Exception as e:
    error_tracker.capture_exception(
        e,
        context={
            "input": input_data,
            "config": config,
            "user_id": user_id
        }
    )
    raise
```

### 4. Use Request IDs for Correlation

```python
# Request ID is automatically generated and logged
context = request_logger.log_request("POST", "/api/train")
# Use context['request_id'] for correlation across logs
```

### 5. Monitor Performance Trends

```python
# Regularly check profiling statistics
stats = profiler.get_all_statistics()
for func_name, func_stats in stats.items():
    if func_stats['mean'] > threshold:
        logger.warning(f"Slow function detected: {func_name}")
```

## Integration Examples

### Streamlit Application

```python
import streamlit as st
from math_research.utils.observability import get_observability_manager

obs = get_observability_manager()

@st.cache_data
@obs.profiler.profile_function()
def cached_computation(input_data):
    return expensive_operation(input_data)
```

### CLI Application

```python
from math_research.utils.tracing import trace_function

@trace_function(name="cli_generate")
def generate_sequence(start: int):
    # Traced operation
    return sequence
```

### Training Pipeline

```python
from math_research.utils.observability import get_observability_manager

obs = get_observability_manager()

def train_with_observability(config):
    context = obs.trace_request("TRAIN", "training_pipeline", params=config)
    
    try:
        with obs.tracer.start_span("dataset_creation"):
            dataset = create_dataset(config)
        
        with obs.tracer.start_span("model_training"):
            model = train_model(dataset, config)
        
        obs.trace_response(context, status="success")
        return model
    except Exception as e:
        obs.error_tracker.capture_exception(e, context={"config": config})
        obs.trace_response(context, status="error", error=e)
        raise
```

## Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Sentry Python SDK](https://docs.sentry.io/platforms/python/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Production Deployment Guide](production_deployment.md)

