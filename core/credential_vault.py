"""
Centralized Credential Vault for managing all service credentials.

This module provides a unified credential management system that supports:
- Multiple credential profiles (personal, work, client, etc.)
- All service types (storage, AI, image generation, etc.)
- Easy profile switching
- Secure storage with optional encryption
- Environment variable overrides
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceCategory(Enum):
    """Service categories for organizing credentials."""
    STORAGE = "storage"
    AI_TEXT = "ai_text"
    AI_IMAGE = "ai_image"
    AI_VIDEO = "ai_video"
    AI_AUDIO = "ai_audio"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"


class CredentialVault:
    """
    Unified credential management for all services.
    
    Handles credential storage, retrieval, and profile management
    for all integrated services including storage, AI, and future services.
    """
    
    def __init__(self, vault_path: Optional[Path] = None, profile: Optional[str] = None):
        """
        Initialize the credential vault.
        
        Args:
            vault_path: Path to vault JSON file (default: credentials/vault.json)
            profile: Active profile name (default: from env or 'default')
        """
        from config.settings import get_settings
        settings = get_settings()
        
        # Set vault path
        self.vault_path = vault_path or Path(settings.credential_vault_path)
        
        # Set active profile
        self.profile = (
            profile or 
            os.environ.get('CREDENTIAL_PROFILE') or 
            settings.credential_profile or
            'default'
        )
        
        # Load vault data
        self.vault_data = self._load_vault()
        
        # Cache for environment overrides
        self._env_overrides = {}
        self._load_env_overrides()
        
        logger.info(f"Credential vault initialized with profile: {self.profile}")
    
    def _load_vault(self) -> Dict[str, Any]:
        """Load vault data from JSON file."""
        if not self.vault_path.exists():
            # Create default vault structure if doesn't exist
            default_vault = {
                "profiles": {
                    "default": {
                        "storage": {},
                        "ai_text": {},
                        "ai_image": {},
                        "ai_video": {},
                        "ai_audio": {}
                    }
                },
                "service_registry": {
                    "storage": ["gdrive", "airtable", "s3", "azure", "dropbox"],
                    "ai_text": ["claude", "openai", "gemini", "cohere", "anthropic", "groq"],
                    "ai_image": ["dalle", "midjourney", "stable_diffusion", "leonardo", "ideogram"],
                    "ai_video": ["runway", "pika", "synthesia", "heygen"],
                    "ai_audio": ["elevenlabs", "murf", "descript", "play.ht"]
                }
            }
            
            # Create directory if needed
            self.vault_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default vault
            with open(self.vault_path, 'w') as f:
                json.dump(default_vault, f, indent=2)
            
            return default_vault
        
        try:
            with open(self.vault_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vault file: {e}")
            raise ValueError(f"Invalid vault file format: {self.vault_path}")
    
    def _load_env_overrides(self) -> None:
        """Load environment variable overrides."""
        # Check for override environment variables
        # Format: OVERRIDE_{SERVICE}_{FIELD}
        # Example: OVERRIDE_CLAUDE_API_KEY, OVERRIDE_GDRIVE_FOLDER_ID
        
        for key, value in os.environ.items():
            if key.startswith('OVERRIDE_'):
                parts = key[9:].lower().split('_', 1)  # Remove 'OVERRIDE_' prefix
                if len(parts) == 2:
                    service, field = parts
                    if service not in self._env_overrides:
                        self._env_overrides[service] = {}
                    self._env_overrides[service][field] = value
                    logger.debug(f"Loaded env override: {service}.{field}")
    
    def get_credentials(
        self, 
        service_category: Union[str, ServiceCategory], 
        service_name: str
    ) -> Dict[str, Any]:
        """
        Get credentials for a specific service.
        
        Args:
            service_category: Category of service (storage, ai_text, etc.)
            service_name: Name of service (gdrive, claude, etc.)
            
        Returns:
            Dictionary of credentials for the service
            
        Raises:
            KeyError: If service or profile not found
        """
        # Convert enum to string if needed
        if isinstance(service_category, ServiceCategory):
            service_category = service_category.value
        
        # Check if profile exists
        if self.profile not in self.vault_data.get("profiles", {}):
            logger.warning(f"Profile '{self.profile}' not found, using default")
            profile_data = self.vault_data.get("profiles", {}).get("default", {})
        else:
            profile_data = self.vault_data["profiles"][self.profile]
        
        # Get base credentials from vault
        credentials = profile_data.get(service_category, {}).get(service_name, {}).copy()
        
        # Apply environment overrides if present
        if service_name in self._env_overrides:
            credentials.update(self._env_overrides[service_name])
            logger.debug(f"Applied env overrides for {service_name}")
        
        # Check for service-specific env vars (e.g., CLAUDE_API_KEY)
        service_env_vars = {
            'claude': ['CLAUDE_API_KEY'],
            'openai': ['OPENAI_API_KEY', 'OPENAI_ORGANIZATION'],
            'gemini': ['GEMINI_API_KEY'],
            'gdrive': ['GDRIVE_CREDENTIALS_FILE', 'GDRIVE_FOLDER_ID'],
            'airtable': ['AIRTABLE_API_KEY', 'AIRTABLE_BASE_ID', 'AIRTABLE_TABLE_NAME']
        }
        
        if service_name in service_env_vars:
            for env_var in service_env_vars[service_name]:
                if env_var in os.environ:
                    field_name = env_var.lower().replace(f"{service_name.upper()}_", "")
                    credentials[field_name] = os.environ[env_var]
        
        return credentials
    
    def set_credentials(
        self,
        service_category: Union[str, ServiceCategory],
        service_name: str,
        credentials: Dict[str, Any],
        profile: Optional[str] = None
    ) -> None:
        """
        Set credentials for a specific service.
        
        Args:
            service_category: Category of service
            service_name: Name of service
            credentials: Credential dictionary
            profile: Profile to update (default: current profile)
        """
        if isinstance(service_category, ServiceCategory):
            service_category = service_category.value
        
        profile = profile or self.profile
        
        # Ensure profile exists
        if profile not in self.vault_data["profiles"]:
            self.vault_data["profiles"][profile] = {
                "storage": {},
                "ai_text": {},
                "ai_image": {},
                "ai_video": {},
                "ai_audio": {}
            }
        
        # Ensure service category exists
        if service_category not in self.vault_data["profiles"][profile]:
            self.vault_data["profiles"][profile][service_category] = {}
        
        # Set credentials
        self.vault_data["profiles"][profile][service_category][service_name] = credentials
        
        # Save vault
        self._save_vault()
        
        logger.info(f"Updated credentials for {service_name} in profile {profile}")
    
    def _save_vault(self) -> None:
        """Save vault data to JSON file."""
        with open(self.vault_path, 'w') as f:
            json.dump(self.vault_data, f, indent=2)
    
    def list_profiles(self) -> List[str]:
        """Get list of available profiles."""
        return list(self.vault_data.get("profiles", {}).keys())
    
    def switch_profile(self, profile_name: str) -> None:
        """
        Switch to a different credential profile.
        
        Args:
            profile_name: Name of profile to switch to
            
        Raises:
            KeyError: If profile doesn't exist
        """
        if profile_name not in self.vault_data.get("profiles", {}):
            raise KeyError(f"Profile '{profile_name}' not found. Available: {self.list_profiles()}")
        
        self.profile = profile_name
        logger.info(f"Switched to profile: {profile_name}")
    
    def create_profile(self, profile_name: str, copy_from: Optional[str] = None) -> None:
        """
        Create a new profile.
        
        Args:
            profile_name: Name of new profile
            copy_from: Optional profile to copy from
        """
        if profile_name in self.vault_data.get("profiles", {}):
            raise ValueError(f"Profile '{profile_name}' already exists")
        
        if copy_from:
            if copy_from not in self.vault_data.get("profiles", {}):
                raise KeyError(f"Source profile '{copy_from}' not found")
            
            # Deep copy profile
            import copy
            self.vault_data["profiles"][profile_name] = copy.deepcopy(
                self.vault_data["profiles"][copy_from]
            )
        else:
            # Create empty profile
            self.vault_data["profiles"][profile_name] = {
                "storage": {},
                "ai_text": {},
                "ai_image": {},
                "ai_video": {},
                "ai_audio": {}
            }
        
        self._save_vault()
        logger.info(f"Created profile: {profile_name}")
    
    def delete_profile(self, profile_name: str) -> None:
        """
        Delete a profile.
        
        Args:
            profile_name: Name of profile to delete
        """
        if profile_name == "default":
            raise ValueError("Cannot delete default profile")
        
        if profile_name not in self.vault_data.get("profiles", {}):
            raise KeyError(f"Profile '{profile_name}' not found")
        
        del self.vault_data["profiles"][profile_name]
        self._save_vault()
        
        # Switch to default if we deleted current profile
        if self.profile == profile_name:
            self.profile = "default"
        
        logger.info(f"Deleted profile: {profile_name}")
    
    def list_services(self, service_category: Optional[Union[str, ServiceCategory]] = None) -> Dict[str, List[str]]:
        """
        List registered services.
        
        Args:
            service_category: Optional category to filter by
            
        Returns:
            Dictionary of service categories and their services
        """
        registry = self.vault_data.get("service_registry", {})
        
        if service_category:
            if isinstance(service_category, ServiceCategory):
                service_category = service_category.value
            return {service_category: registry.get(service_category, [])}
        
        return registry
    
    def validate_credentials(
        self,
        service_category: Union[str, ServiceCategory],
        service_name: str
    ) -> bool:
        """
        Validate credentials for a service (basic check).
        
        Args:
            service_category: Category of service
            service_name: Name of service
            
        Returns:
            True if credentials exist and have required fields
        """
        try:
            creds = self.get_credentials(service_category, service_name)
            
            # Basic validation - check for common required fields
            required_fields = {
                'claude': ['api_key'],
                'openai': ['api_key'],
                'gemini': ['api_key'],
                'gdrive': ['credentials_file', 'folder_id'],
                'airtable': ['api_key', 'base_id'],
                'dalle': ['api_key'],
                'stable_diffusion': ['api_key']
            }
            
            if service_name in required_fields:
                for field in required_fields[service_name]:
                    if field not in creds or not creds[field]:
                        logger.warning(f"Missing required field '{field}' for {service_name}")
                        return False
            
            return bool(creds)
            
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False
    
    def export_profile(self, profile_name: str, output_path: Path) -> None:
        """Export a profile to JSON file."""
        if profile_name not in self.vault_data.get("profiles", {}):
            raise KeyError(f"Profile '{profile_name}' not found")
        
        export_data = {
            "profile": profile_name,
            "credentials": self.vault_data["profiles"][profile_name]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported profile '{profile_name}' to {output_path}")
    
    def import_profile(self, input_path: Path, profile_name: Optional[str] = None) -> None:
        """Import a profile from JSON file."""
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        profile_name = profile_name or import_data.get("profile", "imported")
        
        # Add to vault
        self.vault_data["profiles"][profile_name] = import_data["credentials"]
        self._save_vault()
        
        logger.info(f"Imported profile as '{profile_name}'")


# Singleton instance
_credential_vault = None


def get_credential_vault(vault_path: Optional[Path] = None, profile: Optional[str] = None) -> CredentialVault:
    """
    Get or create the credential vault singleton.
    
    Args:
        vault_path: Optional vault file path
        profile: Optional profile name
        
    Returns:
        CredentialVault instance
    """
    global _credential_vault
    if _credential_vault is None or vault_path is not None or profile is not None:
        _credential_vault = CredentialVault(vault_path, profile)
    return _credential_vault