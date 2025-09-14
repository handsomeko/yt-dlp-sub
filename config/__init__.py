"""
Configuration module for yt-dl-sub.

Provides centralized configuration management with environment variable support,
validation, and deployment mode handling.

Usage:
    from config import settings
    from config import get_settings, is_development, is_production
    
    # Access configuration
    print(settings.deployment_mode)
    print(settings.storage_path)
    
    # Check deployment mode
    if is_development():
        print("Running in development mode")
        
    # Get fresh settings instance
    fresh_settings = get_settings()
"""

from .settings import (
    Settings,
    DeploymentMode,
    QueueType,
    LogLevel,
    StorageBackend,
    settings,
    get_settings,
    reload_settings,
    is_development,
    is_production,
    get_storage_path,
    get_database_url,
)

__all__ = [
    "Settings",
    "DeploymentMode", 
    "QueueType",
    "LogLevel",
    "StorageBackend",
    "settings",
    "get_settings",
    "reload_settings",
    "is_development", 
    "is_production",
    "get_storage_path",
    "get_database_url",
]