"""
Enhanced observability utilities for ARKHE Framework.

This module provides enhanced observability features including:
- Request/response logging
- Performance profiling
- Error tracking integration
- Application Performance Monitoring (APM)

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import functools
import time
import traceback
import logging
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import optional dependencies
SENTRY_AVAILABLE = False
try:
    import sentry_sdk  # type: ignore
    from sentry_sdk.integrations.logging import LoggingIntegration  # type: ignore
    SENTRY_AVAILABLE = True
except ImportError:
    logger.debug("Sentry SDK not available. Error tracking will be limited.")


class RequestLogger:
    """
    Request/response logging middleware.
    
    Logs incoming requests and outgoing responses with timing information.
    """
    
    def __init__(self, enabled: bool = True, log_level: str = "INFO"):
        """
        Initialize request logger.
        
        Args:
            enabled: Whether request logging is enabled
            log_level: Logging level for requests
        """
        self.enabled = enabled
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    def log_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Log an incoming request.
        
        Args:
            method: HTTP method or operation type
            path: Request path or operation name
            params: Request parameters
            headers: Request headers
        
        Returns:
            Request context dictionary
        """
        if not self.enabled:
            return {}
        
        request_id = f"{int(time.time() * 1000)}"
        context = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "timestamp": datetime.now().isoformat(),
            "params": params or {},
        }
        
        log_data = {
            "type": "request",
            **context
        }
        
        logger.log(
            self.log_level,
            f"Request: {method} {path}",
            extra={"request_context": log_data}
        )
        
        return context
    
    def log_response(
        self,
        context: Dict[str, Any],
        status: str = "success",
        duration: Optional[float] = None,
        response_data: Optional[Any] = None,
        error: Optional[Exception] = None,
    ):
        """
        Log a response.
        
        Args:
            context: Request context from log_request
            status: Response status
            duration: Request duration in seconds
            response_data: Response data (will be truncated if too large)
            error: Optional error exception
        """
        if not self.enabled:
            return
        
        response_log = {
            "type": "response",
            "request_id": context.get("request_id"),
            "status": status,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
        
        if error:
            response_log["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None,
            }
        
        if response_data is not None:
            # Truncate large responses
            response_str = json.dumps(response_data) if not isinstance(response_data, str) else response_data
            if len(response_str) > 1000:
                response_log["response_preview"] = response_str[:1000] + "... (truncated)"
            else:
                response_log["response"] = response_data
        
        log_level = logging.ERROR if error else self.log_level
        logger.log(
            log_level,
            f"Response: {context.get('method')} {context.get('path')} - {status} ({duration:.3f}s)" if duration else f"Response: {context.get('method')} {context.get('path')} - {status}",
            extra={"response_context": response_log}
        )


class PerformanceProfiler:
    """
    Performance profiling utilities.
    
    Provides timing and profiling capabilities for code execution.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.profiles: Dict[str, List[float]] = {}
    
    def profile_function(self, func_name: Optional[str] = None):
        """
        Decorator to profile a function's execution time.
        
        Args:
            func_name: Optional function name (defaults to function name)
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    
                    if name not in self.profiles:
                        self.profiles[name] = []
                    self.profiles[name].append(duration)
                    
                    logger.debug(f"Profiled {name}: {duration:.4f}s")
                    
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.warning(f"Profiled {name} (failed): {duration:.4f}s - {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_statistics(self, func_name: str) -> Dict[str, float]:
        """
        Get profiling statistics for a function.
        
        Args:
            func_name: Function name
        
        Returns:
            Dictionary with statistics (count, total, mean, min, max)
        """
        if func_name not in self.profiles:
            return {}
        
        times = self.profiles[func_name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all profiled functions.
        
        Returns:
            Dictionary mapping function names to their statistics
        """
        return {name: self.get_statistics(name) for name in self.profiles.keys()}
    
    def reset(self):
        """Reset all profiling data."""
        self.profiles.clear()


class ErrorTracker:
    """
    Error tracking integration.
    
    Provides integration with error tracking services like Sentry.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        sentry_dsn: Optional[str] = None,
        environment: str = "development",
    ):
        """
        Initialize error tracker.
        
        Args:
            enabled: Whether error tracking is enabled
            sentry_dsn: Sentry DSN (Data Source Name)
            environment: Environment name (development, staging, production)
        """
        self.enabled = enabled
        self.sentry_dsn = sentry_dsn
        self.environment = environment
        self.sentry_initialized = False
        
        if enabled and SENTRY_AVAILABLE and sentry_dsn:
            try:
                import sentry_sdk  # type: ignore
                from sentry_sdk.integrations.logging import LoggingIntegration  # type: ignore
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    environment=environment,
                    integrations=[
                        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
                    ],
                    traces_sample_rate=0.1,  # Sample 10% of transactions
                )
                self.sentry_initialized = True
                logger.info("Sentry error tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Sentry: {e}")
    
    def capture_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Capture an exception for error tracking.
        
        Args:
            exception: Exception to capture
            context: Optional context information
        """
        if not self.enabled:
            return
        
        if self.sentry_initialized and SENTRY_AVAILABLE:
            try:
                import sentry_sdk  # type: ignore
                with sentry_sdk.push_scope() as scope:
                    if context:
                        for key, value in context.items():
                            scope.set_context(key, value)
                    sentry_sdk.capture_exception(exception)
            except Exception as e:
                logger.warning(f"Failed to capture exception in Sentry: {e}")
        
        # Always log locally
        logger.error(
            f"Exception captured: {type(exception).__name__}: {exception}",
            exc_info=True,
            extra={"error_context": context or {}}
        )
    
    def capture_message(self, message: str, level: str = "info", context: Optional[Dict[str, Any]] = None):
        """
        Capture a message for error tracking.
        
        Args:
            message: Message to capture
            level: Message level (debug, info, warning, error)
            context: Optional context information
        """
        if not self.enabled:
            return
        
        if self.sentry_initialized and SENTRY_AVAILABLE:
            try:
                import sentry_sdk  # type: ignore
                with sentry_sdk.push_scope() as scope:
                    if context:
                        for key, value in context.items():
                            scope.set_context(key, value)
                    sentry_sdk.capture_message(message, level=level)
            except Exception as e:
                logger.warning(f"Failed to capture message in Sentry: {e}")
        
        # Always log locally
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message, extra={"message_context": context or {}})


class ObservabilityManager:
    """
    Unified observability manager.
    
    Combines request logging, profiling, and error tracking.
    """
    
    def __init__(
        self,
        request_logging: bool = True,
        profiling: bool = True,
        error_tracking: bool = True,
        sentry_dsn: Optional[str] = None,
        environment: str = "development",
    ):
        """
        Initialize observability manager.
        
        Args:
            request_logging: Enable request/response logging
            profiling: Enable performance profiling
            error_tracking: Enable error tracking
            sentry_dsn: Sentry DSN for error tracking
            environment: Environment name
        """
        self.request_logger = RequestLogger(enabled=request_logging) if request_logging else None
        self.profiler = PerformanceProfiler(enabled=profiling) if profiling else None
        self.error_tracker = ErrorTracker(
            enabled=error_tracking,
            sentry_dsn=sentry_dsn,
            environment=environment,
        ) if error_tracking else None
    
    def trace_request(self, method: str, path: str, **kwargs):
        """
        Trace a request with full observability.
        
        Args:
            method: Request method
            path: Request path
            **kwargs: Additional arguments for request logger
        
        Returns:
            Request context
        """
        if self.request_logger:
            return self.request_logger.log_request(method, path, **kwargs)
        return {}
    
    def trace_response(self, context: Dict[str, Any], **kwargs):
        """
        Trace a response with full observability.
        
        Args:
            context: Request context
            **kwargs: Additional arguments for response logger
        """
        if self.request_logger:
            self.request_logger.log_response(context, **kwargs)


# Global observability manager instance
_global_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager(**kwargs) -> ObservabilityManager:
    """
    Get or create global observability manager instance.
    
    Args:
        **kwargs: Arguments to pass to ObservabilityManager
    
    Returns:
        Observability manager instance
    """
    global _global_observability_manager
    if _global_observability_manager is None:
        _global_observability_manager = ObservabilityManager(**kwargs)
    return _global_observability_manager

