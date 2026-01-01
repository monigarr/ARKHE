"""
Health check utilities for production monitoring.

This module provides health check functionality for monitoring
the application's status, dependencies, and system resources.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import sys
import platform
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Health check utility for monitoring application status.
    
    Provides methods to check:
    - System dependencies (PyTorch, CUDA, etc.)
    - File system access
    - Memory availability
    - Module imports
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Dict[str, Any]] = {}
    
    def check_pytorch(self) -> Dict[str, Any]:
        """
        Check PyTorch installation and CUDA availability.
        
        Returns:
            Dictionary with status, version, and CUDA info
        """
        try:
            import torch
            
            result = {
                "status": "healthy",
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": None,
                "device_count": 0,
            }
            
            if result["cuda_available"]:
                result["cuda_version"] = torch.version.cuda
                result["device_count"] = torch.cuda.device_count()
                result["current_device"] = torch.cuda.current_device()
                result["device_name"] = torch.cuda.get_device_name(0)
            
            return result
            
        except ImportError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "PyTorch not installed"
            }
    
    def check_numpy(self) -> Dict[str, Any]:
        """
        Check NumPy installation.
        
        Returns:
            Dictionary with status and version
        """
        try:
            import numpy as np
            return {
                "status": "healthy",
                "version": np.__version__
            }
        except ImportError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "NumPy not installed"
            }
    
    def check_pandas(self) -> Dict[str, Any]:
        """
        Check Pandas installation.
        
        Returns:
            Dictionary with status and version
        """
        try:
            import pandas as pd
            return {
                "status": "healthy",
                "version": pd.__version__
            }
        except ImportError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Pandas not installed"
            }
    
    def check_filesystem(self, paths: Optional[list] = None) -> Dict[str, Any]:
        """
        Check file system access for critical directories.
        
        Args:
            paths: List of paths to check (defaults to common project paths)
        
        Returns:
            Dictionary with status and path checks
        """
        if paths is None:
            paths = [
                "checkpoints",
                "data",
                "configs",
                "logs",
            ]
        
        results = {}
        all_healthy = True
        
        for path_str in paths:
            path = Path(path_str)
            try:
                # Check if path exists or can be created
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                
                # Check write access
                test_file = path / ".health_check"
                test_file.write_text("test")
                test_file.unlink()
                
                results[path_str] = {
                    "status": "healthy",
                    "exists": True,
                    "writable": True
                }
            except Exception as e:
                all_healthy = False
                results[path_str] = {
                    "status": "unhealthy",
                    "exists": False,
                    "writable": False,
                    "error": str(e)
                }
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "paths": results
        }
    
    def check_imports(self) -> Dict[str, Any]:
        """
        Check critical module imports.
        
        Returns:
            Dictionary with import status for each module
        """
        modules = [
            "math_research.sequences",
            "math_research.analysis",
            "math_research.ml",
            "math_research.utils",
        ]
        
        results = {}
        all_healthy = True
        
        for module_name in modules:
            try:
                __import__(module_name)
                results[module_name] = {
                    "status": "healthy",
                    "imported": True
                }
            except ImportError as e:
                all_healthy = False
                results[module_name] = {
                    "status": "unhealthy",
                    "imported": False,
                    "error": str(e)
                }
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "modules": results
        }
    
    def check_system(self) -> Dict[str, Any]:
        """
        Check system information.
        
        Returns:
            Dictionary with system info
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": sys.version,
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_used": disk.percent
                }
            }
        except ImportError:
            # psutil not available, return basic info
            return {
                "status": "healthy",
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": sys.version,
                "note": "psutil not installed, limited system info"
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Comprehensive health status dictionary
        """
        logger.info("Running health checks...")
        
        checks = {
            "pytorch": self.check_pytorch(),
            "numpy": self.check_numpy(),
            "pandas": self.check_pandas(),
            "filesystem": self.check_filesystem(),
            "imports": self.check_imports(),
            "system": self.check_system(),
        }
        
        # Determine overall status
        overall_status = "healthy"
        for check_name, check_result in checks.items():
            if isinstance(check_result, dict) and check_result.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
        
        return {
            "status": overall_status,
            "checks": checks,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_health_summary(self) -> str:
        """
        Get a human-readable health summary.
        
        Returns:
            Formatted string summary
        """
        results = self.run_all_checks()
        
        lines = [
            f"Health Status: {results['status'].upper()}",
            f"Timestamp: {results['timestamp']}",
            ""
        ]
        
        for check_name, check_result in results["checks"].items():
            status = check_result.get("status", "unknown")
            status_icon = "✅" if status == "healthy" else "❌"
            lines.append(f"{status_icon} {check_name}: {status}")
            
            if status == "unhealthy" and "error" in check_result:
                lines.append(f"   Error: {check_result['error']}")
        
        return "\n".join(lines)


def get_health_status() -> Dict[str, Any]:
    """
    Convenience function to get health status.
    
    Returns:
        Health status dictionary
    """
    checker = HealthChecker()
    return checker.run_all_checks()


def is_healthy() -> bool:
    """
    Convenience function to check if system is healthy.
    
    Returns:
        True if all checks pass, False otherwise
    """
    checker = HealthChecker()
    results = checker.run_all_checks()
    return results["status"] == "healthy"

