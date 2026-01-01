"""
Module: config
Package: math_research.utils

Description:
    Configuration management utilities. Provides functions to load and manage
    YAML configuration files with support for environment variables and defaults.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Author Research Interests:
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages (spoken in English communities)

Usage:
    >>> from math_research.utils import load_config, Config
    >>> config = load_config("configs/default.yaml")
    >>> logging_level = config.get("logging.level", "INFO")
    >>> 
    >>> # Or use Config class
    >>> cfg = Config("configs/default.yaml")
    >>> print(cfg.logging.level)

Dependencies:
    - yaml (pyyaml)
    - pathlib (standard library)
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Configuration files should be in YAML format. Supports nested dictionaries.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


class Config:
    """
    Configuration class that provides dictionary-like access with dot notation.
    
    Example:
        >>> config = Config("configs/default.yaml")
        >>> value = config.logging.level
        >>> nested = config.ml.training.batch_size
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize Config object.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Optional dictionary to initialize from (overrides config_path)
        """
        if config_dict is not None:
            self._config = config_dict
        elif config_path:
            self._config = load_config(config_path)
        else:
            self._config = {}
    
    def __getattr__(self, key: str) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value or Config object for nested dictionaries
        """
        if key.startswith('_'):
            return super().__getattribute__(key)
        
        value = self._config.get(key)
        
        if isinstance(value, dict):
            return Config(config_dict=value)
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.__getattr__(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        if isinstance(value, dict):
            return Config(config_dict=value)
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config object to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to merge
        """
        self._config.update(updates)

