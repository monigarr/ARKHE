"""
Metrics export utilities for production monitoring.

This module provides Prometheus-compatible metrics export for monitoring
application performance, training metrics, and system health.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import time
import threading
from typing import Dict, Optional, Any
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Prometheus-compatible metrics exporter.
    
    Provides functionality to export application metrics in Prometheus format
    for monitoring and alerting.
    """
    
    def __init__(self, port: int = 9090, enabled: bool = True):
        """
        Initialize metrics exporter.
        
        Args:
            port: Port for metrics HTTP server
            enabled: Whether metrics export is enabled
        """
        self.port = port
        self.enabled = enabled
        self.metrics: Dict[str, Any] = defaultdict(dict)
        self.server_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        if enabled:
            logger.info(f"Metrics exporter initialized on port {port}")
    
    def record_training_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a training metric.
        
        Args:
            name: Metric name (e.g., "loss", "accuracy")
            value: Metric value
            step: Training step/epoch
            labels: Additional labels (e.g., {"model": "collatz_transformer"})
        """
        if not self.enabled:
            return
        
        with self._lock:
            metric_key = f"training_{name}"
            if step is not None:
                self.metrics[metric_key][step] = {
                    "value": value,
                    "timestamp": time.time(),
                    "labels": labels or {}
                }
            else:
                self.metrics[metric_key]["latest"] = {
                    "value": value,
                    "timestamp": time.time(),
                    "labels": labels or {}
                }
    
    def record_inference_latency(self, latency_seconds: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record inference latency.
        
        Args:
            latency_seconds: Inference latency in seconds
            labels: Additional labels
        """
        if not self.enabled:
            return
        
        with self._lock:
            if "inference_latency" not in self.metrics:
                self.metrics["inference_latency"] = {
                    "values": [],
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": 0.0
                }
            
            metric = self.metrics["inference_latency"]
            metric["values"].append(latency_seconds)
            metric["count"] += 1
            metric["sum"] += latency_seconds
            metric["min"] = min(metric["min"], latency_seconds)
            metric["max"] = max(metric["max"], latency_seconds)
            
            # Keep only last 1000 values
            if len(metric["values"]) > 1000:
                metric["values"] = metric["values"][-1000:]
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a counter metric.
        
        Args:
            name: Counter name
            value: Increment value (default: 1)
            labels: Additional labels
        """
        if not self.enabled:
            return
        
        with self._lock:
            metric_key = f"counter_{name}"
            if metric_key not in self.metrics:
                self.metrics[metric_key] = {"value": 0, "labels": labels or {}}
            
            self.metrics[metric_key]["value"] += value
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a gauge metric.
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Additional labels
        """
        if not self.enabled:
            return
        
        with self._lock:
            metric_key = f"gauge_{name}"
            self.metrics[metric_key] = {
                "value": value,
                "timestamp": time.time(),
                "labels": labels or {}
            }
    
    def get_metrics_prometheus_format(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Metrics in Prometheus exposition format
        """
        if not self.enabled:
            return "# Metrics disabled\n"
        
        lines = []
        lines.append("# HELP arkhe_metrics ARKHE Framework metrics")
        lines.append("# TYPE arkhe_metrics gauge")
        lines.append("")
        
        with self._lock:
            for metric_name, metric_data in self.metrics.items():
                if isinstance(metric_data, dict):
                    if "value" in metric_data:
                        # Simple metric
                        labels_str = self._format_labels(metric_data.get("labels", {}))
                        value = metric_data["value"]
                        lines.append(f"{metric_name}{labels_str} {value}")
                    elif "latest" in metric_data:
                        # Training metric with latest value
                        labels_str = self._format_labels(metric_data["latest"].get("labels", {}))
                        value = metric_data["latest"]["value"]
                        lines.append(f"{metric_name}{labels_str} {value}")
                    elif "values" in metric_data:
                        # Latency metric
                        labels_str = self._format_labels({})
                        count = metric_data["count"]
                        sum_val = metric_data["sum"]
                        avg = sum_val / count if count > 0 else 0.0
                        min_val = metric_data["min"] if metric_data["min"] != float("inf") else 0.0
                        max_val = metric_data["max"]
                        
                        lines.append(f"{metric_name}_count{labels_str} {count}")
                        lines.append(f"{metric_name}_sum{labels_str} {sum_val}")
                        lines.append(f"{metric_name}_avg{labels_str} {avg}")
                        lines.append(f"{metric_name}_min{labels_str} {min_val}")
                        lines.append(f"{metric_name}_max{labels_str} {max_val}")
        
        return "\n".join(lines) + "\n"
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """
        Format labels for Prometheus format.
        
        Args:
            labels: Dictionary of label key-value pairs
        
        Returns:
            Formatted label string
        """
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"
    
    def start(self) -> None:
        """
        Start metrics HTTP server.
        
        Note: This requires an HTTP server library. For production,
        use a proper HTTP server or integrate with existing web framework.
        """
        if not self.enabled:
            logger.warning("Metrics export is disabled")
            return
        
        logger.info(f"Starting metrics server on port {self.port}")
        # In a real implementation, you would start an HTTP server here
        # For now, metrics can be accessed via get_metrics_prometheus_format()
        logger.warning("HTTP server not implemented. Use get_metrics_prometheus_format() to get metrics.")
    
    def stop(self) -> None:
        """Stop metrics HTTP server."""
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Stopping metrics server")
            # Stop server implementation would go here
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            logger.info("Metrics reset")


# Global metrics exporter instance
_global_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter(port: int = 9090, enabled: bool = True) -> MetricsExporter:
    """
    Get or create global metrics exporter instance.
    
    Args:
        port: Port for metrics server
        enabled: Whether metrics are enabled
    
    Returns:
        Metrics exporter instance
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter(port=port, enabled=enabled)
    return _global_exporter


def record_training_metric(name: str, value: float, step: Optional[int] = None) -> None:
    """Convenience function to record training metric."""
    exporter = get_metrics_exporter()
    exporter.record_training_metric(name, value, step)


def record_inference_latency(latency_seconds: float) -> None:
    """Convenience function to record inference latency."""
    exporter = get_metrics_exporter()
    exporter.record_inference_latency(latency_seconds)

