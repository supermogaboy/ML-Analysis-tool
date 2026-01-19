#!/usr/bin/env python3

"""
Configuration loader for ML Trading System
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_file = Path(__file__).parent.parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def get_paths() -> Dict[str, str]:
    """Get all file paths from config"""
    config = load_config()
    
    paths = {}
    paths.update(config["data"])
    paths.update(config["models"])
    
    return paths

def get_features() -> list:
    """Get feature list from config"""
    config = load_config()
    return config["features"]

def get_training_params() -> Dict[str, Any]:
    """Get training parameters from config"""
    config = load_config()
    return config["training"]

def get_testing_params() -> Dict[str, Any]:
    """Get testing parameters from config"""
    config = load_config()
    return config["testing"]
