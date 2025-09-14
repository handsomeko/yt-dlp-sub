"""
Service-specific credential wrappers using the centralized CredentialVault.

These wrapper classes provide clean interfaces for accessing service credentials
while handling validation and fallback to environment variables.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.credential_vault import CredentialVault, ServiceCategory, get_credential_vault

logger = logging.getLogger(__name__)


class BaseServiceCredentials:
    """Base class for service credential wrappers."""
    
    def __init__(self, service_category: ServiceCategory, service_name: str):
        """Initialize base service credentials."""
        self.service_category = service_category
        self.service_name = service_name
        self.vault = get_credential_vault()
        self._credentials = None
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from vault."""
        try:
            self._credentials = self.vault.get_credentials(
                self.service_category,
                self.service_name
            )
        except Exception as e:
            logger.warning(f"Failed to load credentials for {self.service_name}: {e}")
            self._credentials = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a credential value."""
        return self._credentials.get(key, default)
    
    def update(self, **kwargs) -> None:
        """Update credentials in vault."""
        self._credentials.update(kwargs)
        self.vault.set_credentials(
            self.service_category,
            self.service_name,
            self._credentials
        )
    
    def validate(self) -> bool:
        """Validate that required credentials are present."""
        return self.vault.validate_credentials(
            self.service_category,
            self.service_name
        )


# Storage Service Credentials

class GoogleDriveCredentials(BaseServiceCredentials):
    """Google Drive credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.STORAGE, "gdrive")
    
    @property
    def credentials_file(self) -> Optional[str]:
        """Get Google Drive credentials file path."""
        return self.get("credentials_file")
    
    @property
    def folder_id(self) -> Optional[str]:
        """Get Google Drive folder ID."""
        return self.get("folder_id")
    
    @property
    def shared_drive_id(self) -> Optional[str]:
        """Get shared drive ID if using Team Drive."""
        return self.get("shared_drive_id")


class AirtableCredentials(BaseServiceCredentials):
    """Airtable credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.STORAGE, "airtable")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Airtable API key."""
        return self.get("api_key")
    
    @property
    def base_id(self) -> Optional[str]:
        """Get Airtable base ID."""
        return self.get("base_id")
    
    @property
    def table_name(self) -> str:
        """Get Airtable table name."""
        return self.get("table_name", "Videos")


class S3Credentials(BaseServiceCredentials):
    """AWS S3 credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.STORAGE, "s3")
    
    @property
    def access_key_id(self) -> Optional[str]:
        """Get AWS access key ID."""
        return self.get("access_key_id")
    
    @property
    def secret_access_key(self) -> Optional[str]:
        """Get AWS secret access key."""
        return self.get("secret_access_key")
    
    @property
    def bucket_name(self) -> Optional[str]:
        """Get S3 bucket name."""
        return self.get("bucket_name")
    
    @property
    def region(self) -> str:
        """Get AWS region."""
        return self.get("region", "us-east-1")


# AI Text Service Credentials

class ClaudeCredentials(BaseServiceCredentials):
    """Claude API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_TEXT, "claude")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Claude API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get default Claude model."""
        return self.get("model", "claude-3-haiku-20240307")
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens."""
        return self.get("max_tokens", 1000)


class OpenAICredentials(BaseServiceCredentials):
    """OpenAI API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_TEXT, "openai")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get("api_key")
    
    @property
    def organization(self) -> Optional[str]:
        """Get OpenAI organization ID."""
        return self.get("organization")
    
    @property
    def model(self) -> str:
        """Get default model."""
        return self.get("model", "gpt-3.5-turbo")
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens."""
        return self.get("max_tokens", 1000)


class GeminiCredentials(BaseServiceCredentials):
    """Google Gemini API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_TEXT, "gemini")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Gemini API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get default Gemini model."""
        return self.get("model", "gemini-pro")
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens."""
        return self.get("max_tokens", 1000)


class GroqCredentials(BaseServiceCredentials):
    """Groq API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_TEXT, "groq")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Groq API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get default Groq model."""
        return self.get("model", "mixtral-8x7b-32768")


# AI Image Service Credentials

class DalleCredentials(BaseServiceCredentials):
    """DALL-E API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_IMAGE, "dalle")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get OpenAI API key for DALL-E."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get DALL-E model version."""
        return self.get("model", "dall-e-3")
    
    @property
    def size(self) -> str:
        """Get default image size."""
        return self.get("size", "1024x1024")
    
    @property
    def quality(self) -> str:
        """Get image quality."""
        return self.get("quality", "standard")


class MidjourneyCredentials(BaseServiceCredentials):
    """Midjourney API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_IMAGE, "midjourney")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Midjourney API key."""
        return self.get("api_key")
    
    @property
    def webhook_url(self) -> Optional[str]:
        """Get webhook URL for notifications."""
        return self.get("webhook_url")


class StableDiffusionCredentials(BaseServiceCredentials):
    """Stable Diffusion API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_IMAGE, "stable_diffusion")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Stable Diffusion API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get model version."""
        return self.get("model", "stable-diffusion-xl-1024-v1-0")
    
    @property
    def steps(self) -> int:
        """Get inference steps."""
        return self.get("steps", 50)


class IdeogramCredentials(BaseServiceCredentials):
    """Ideogram API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_IMAGE, "ideogram")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Ideogram API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get model version."""
        return self.get("model", "V_2")


class LeonardoCredentials(BaseServiceCredentials):
    """Leonardo.ai API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_IMAGE, "leonardo")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Leonardo API key."""
        return self.get("api_key")
    
    @property
    def model_id(self) -> Optional[str]:
        """Get default model ID."""
        return self.get("model_id")


# AI Video Service Credentials

class RunwayCredentials(BaseServiceCredentials):
    """Runway API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_VIDEO, "runway")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Runway API key."""
        return self.get("api_key")
    
    @property
    def model(self) -> str:
        """Get default model."""
        return self.get("model", "gen-2")


class PikaCredentials(BaseServiceCredentials):
    """Pika Labs API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_VIDEO, "pika")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Pika API key."""
        return self.get("api_key")


class HeygenCredentials(BaseServiceCredentials):
    """HeyGen API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_VIDEO, "heygen")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get HeyGen API key."""
        return self.get("api_key")
    
    @property
    def avatar_id(self) -> Optional[str]:
        """Get default avatar ID."""
        return self.get("avatar_id")


# AI Audio Service Credentials

class ElevenLabsCredentials(BaseServiceCredentials):
    """ElevenLabs API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_AUDIO, "elevenlabs")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get ElevenLabs API key."""
        return self.get("api_key")
    
    @property
    def voice_id(self) -> Optional[str]:
        """Get default voice ID."""
        return self.get("voice_id")
    
    @property
    def model_id(self) -> str:
        """Get model ID."""
        return self.get("model_id", "eleven_monolingual_v1")


class MurfCredentials(BaseServiceCredentials):
    """Murf.ai API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_AUDIO, "murf")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Murf API key."""
        return self.get("api_key")
    
    @property
    def voice_id(self) -> Optional[str]:
        """Get default voice ID."""
        return self.get("voice_id")


class PlayHTCredentials(BaseServiceCredentials):
    """Play.ht API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.AI_AUDIO, "playht")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Play.ht API key."""
        return self.get("api_key")
    
    @property
    def user_id(self) -> Optional[str]:
        """Get Play.ht user ID."""
        return self.get("user_id")
    
    @property
    def voice(self) -> str:
        """Get default voice."""
        return self.get("voice", "larry")


# Monitoring Service Credentials

class DatadogCredentials(BaseServiceCredentials):
    """Datadog API credential wrapper."""
    
    def __init__(self):
        super().__init__(ServiceCategory.MONITORING, "datadog")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Datadog API key."""
        return self.get("api_key")
    
    @property
    def app_key(self) -> Optional[str]:
        """Get Datadog application key."""
        return self.get("app_key")


# Utility functions

def get_storage_credentials(backend: str) -> BaseServiceCredentials:
    """Get credentials for a storage backend."""
    backends = {
        "gdrive": GoogleDriveCredentials,
        "airtable": AirtableCredentials,
        "s3": S3Credentials
    }
    
    credential_class = backends.get(backend)
    if not credential_class:
        raise ValueError(f"Unknown storage backend: {backend}")
    
    return credential_class()


def get_ai_text_credentials(provider: str) -> BaseServiceCredentials:
    """Get credentials for an AI text provider."""
    providers = {
        "claude": ClaudeCredentials,
        "openai": OpenAICredentials,
        "gemini": GeminiCredentials,
        "groq": GroqCredentials
    }
    
    credential_class = providers.get(provider)
    if not credential_class:
        raise ValueError(f"Unknown AI text provider: {provider}")
    
    return credential_class()


def get_ai_image_credentials(provider: str) -> BaseServiceCredentials:
    """Get credentials for an AI image provider."""
    providers = {
        "dalle": DalleCredentials,
        "midjourney": MidjourneyCredentials,
        "stable_diffusion": StableDiffusionCredentials,
        "ideogram": IdeogramCredentials,
        "leonardo": LeonardoCredentials
    }
    
    credential_class = providers.get(provider)
    if not credential_class:
        raise ValueError(f"Unknown AI image provider: {provider}")
    
    return credential_class()


def validate_all_credentials() -> Dict[str, bool]:
    """Validate all configured credentials."""
    results = {}
    
    # Check storage credentials
    for backend in ["gdrive", "airtable", "s3"]:
        try:
            creds = get_storage_credentials(backend)
            results[f"storage.{backend}"] = creds.validate()
        except Exception:
            results[f"storage.{backend}"] = False
    
    # Check AI text credentials
    for provider in ["claude", "openai", "gemini", "groq"]:
        try:
            creds = get_ai_text_credentials(provider)
            results[f"ai_text.{provider}"] = creds.validate()
        except Exception:
            results[f"ai_text.{provider}"] = False
    
    # Check AI image credentials
    for provider in ["dalle", "midjourney", "stable_diffusion"]:
        try:
            creds = get_ai_image_credentials(provider)
            results[f"ai_image.{provider}"] = creds.validate()
        except Exception:
            results[f"ai_image.{provider}"] = False
    
    return results