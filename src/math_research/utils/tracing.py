"""
Distributed tracing utilities for ARKHE Framework.

This module provides OpenTelemetry-based distributed tracing for
monitoring request flows and operation performance across the application.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import functools
import time
from typing import Optional, Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, but make it optional
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Distributed tracing will be disabled.")


class TracingManager:
    """
    Manager for distributed tracing.
    
    Provides a unified interface for creating spans, tracking operations,
    and exporting traces to various backends (Jaeger, Zipkin, etc.).
    """
    
    def __init__(
        self,
        service_name: str = "arkhe-framework",
        enabled: bool = True,
        exporter: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """
        Initialize tracing manager.
        
        Args:
            service_name: Name of the service for tracing
            enabled: Whether tracing is enabled
            exporter: Exporter type ('otlp', 'console', 'jaeger', 'zipkin')
            endpoint: Endpoint URL for trace exporter
        """
        self.service_name = service_name
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        
        if not self.enabled:
            logger.info("Distributed tracing is disabled")
            self.tracer = None
            return
        
        # Initialize OpenTelemetry
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        
        # Add span processor based on exporter type
        if exporter == "otlp" and endpoint:
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            processor = BatchSpanProcessor(span_exporter)
            provider.add_span_processor(processor)
            logger.info(f"Initialized OTLP exporter to {endpoint}")
        elif exporter == "console" or exporter is None:
            # Default to console exporter for development
            console_exporter = ConsoleSpanExporter()
            processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(processor)
            logger.info("Initialized console span exporter")
        else:
            logger.warning(f"Unknown exporter type: {exporter}, using console")
            console_exporter = ConsoleSpanExporter()
            processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)
        logger.info(f"Tracing manager initialized for service: {service_name}")
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ):
        """
        Start a new span.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
        
        Returns:
            Span context manager
        """
        if not self.enabled or self.tracer is None:
            return _NoOpSpan()
        
        span_kind = getattr(trace.SpanKind, kind, trace.SpanKind.INTERNAL)
        span = self.tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes or {}
        )
        return span
    
    def trace_function(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Decorator to trace a function.
        
        Args:
            name: Optional span name (defaults to function name)
            attributes: Optional span attributes
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or self.tracer is None:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                    
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        span.set_attribute("duration", duration)
                        span.set_attribute("success", True)
                        
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.record_exception(e)
                        raise
            
            return wrapper
        return decorator


class _NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def record_exception(self, exception: Exception):
        pass


# Global tracing manager instance
_global_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager(
    service_name: str = "arkhe-framework",
    enabled: bool = True,
    exporter: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> TracingManager:
    """
    Get or create global tracing manager instance.
    
    Args:
        service_name: Name of the service
        enabled: Whether tracing is enabled
        exporter: Exporter type
        endpoint: Endpoint URL for exporter
    
    Returns:
        Tracing manager instance
    """
    global _global_tracing_manager
    if _global_tracing_manager is None:
        _global_tracing_manager = TracingManager(
            service_name=service_name,
            enabled=enabled,
            exporter=exporter,
            endpoint=endpoint,
        )
    return _global_tracing_manager


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Convenience decorator to trace a function using global tracer.
    
    Args:
        name: Optional span name
        attributes: Optional span attributes
    
    Returns:
        Decorator function
    """
    manager = get_tracing_manager()
    return manager.trace_function(name=name, attributes=attributes)

