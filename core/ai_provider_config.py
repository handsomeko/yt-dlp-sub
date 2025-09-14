"""
AI Provider Configuration System

Manages configuration for different AI providers including models, pricing,
rate limits, and capabilities.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

from config.settings import get_settings
from core.credential_vault import CredentialVault

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """AI model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "qa"
    CREATIVE_WRITING = "creative"
    STRUCTURED_OUTPUT = "structured"


@dataclass
class ModelConfig:
    """Configuration for a specific AI model."""
    model_id: str
    display_name: str
    provider: str
    
    # Pricing
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    
    # Limits
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    
    # Performance
    typical_latency_ms: float  # Typical response time
    timeout_seconds: int = 30
    
    # Capabilities
    capabilities: List[ModelCapability] = field(default_factory=list)
    
    # Quality ratings (0-100)
    quality_ratings: Dict[str, int] = field(default_factory=dict)
    
    # Availability
    is_available: bool = True
    requires_approval: bool = False
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


@dataclass 
class ProviderConfig:
    """Configuration for an AI provider."""
    provider_id: str
    display_name: str
    
    # API configuration
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    concurrent_requests: int = 5
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Error handling
    fallback_provider: Optional[str] = None
    circuit_breaker_threshold: int = 5  # Consecutive failures before circuit break
    circuit_breaker_reset_minutes: int = 5
    
    # Models
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    default_model: Optional[str] = None
    
    # Features
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_batch: bool = False
    
    def get_model(self, model_id: Optional[str] = None) -> Optional[ModelConfig]:
        """Get model configuration."""
        if model_id:
            return self.models.get(model_id)
        elif self.default_model:
            return self.models.get(self.default_model)
        return None


class AIProviderConfigManager:
    """
    Manages AI provider configurations and model selection.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.credential_vault = CredentialVault()
        self.providers: Dict[str, ProviderConfig] = {}
        self._load_default_configs()
        self._load_custom_configs()
    
    def _load_default_configs(self):
        """Load default provider configurations."""
        
        # Claude Configuration
        claude_config = ProviderConfig(
            provider_id="claude",
            display_name="Anthropic Claude",
            base_url="https://api.anthropic.com",
            api_version="2023-06-01",
            requests_per_minute=1000,
            supports_streaming=True,
            supports_vision=True,
            models={
                "claude-3-opus-20240229": ModelConfig(
                    model_id="claude-3-opus-20240229",
                    display_name="Claude 3 Opus",
                    provider="claude",
                    input_cost_per_1k=0.015,
                    output_cost_per_1k=0.075,
                    max_input_tokens=200000,
                    max_output_tokens=4096,
                    max_total_tokens=200000,
                    typical_latency_ms=2000,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.ANALYSIS,
                        ModelCapability.CREATIVE_WRITING
                    ],
                    quality_ratings={
                        "accuracy": 95,
                        "coherence": 95,
                        "creativity": 90,
                        "speed": 70
                    }
                ),
                "claude-3-sonnet-20240229": ModelConfig(
                    model_id="claude-3-sonnet-20240229",
                    display_name="Claude 3 Sonnet",
                    provider="claude",
                    input_cost_per_1k=0.003,
                    output_cost_per_1k=0.015,
                    max_input_tokens=200000,
                    max_output_tokens=4096,
                    max_total_tokens=200000,
                    typical_latency_ms=1500,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.ANALYSIS
                    ],
                    quality_ratings={
                        "accuracy": 90,
                        "coherence": 90,
                        "creativity": 85,
                        "speed": 80
                    }
                ),
                "claude-3-haiku-20240307": ModelConfig(
                    model_id="claude-3-haiku-20240307",
                    display_name="Claude 3 Haiku",
                    provider="claude",
                    input_cost_per_1k=0.00025,
                    output_cost_per_1k=0.00125,
                    max_input_tokens=200000,
                    max_output_tokens=4096,
                    max_total_tokens=200000,
                    typical_latency_ms=800,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.SUMMARIZATION,
                        ModelCapability.QUESTION_ANSWERING
                    ],
                    quality_ratings={
                        "accuracy": 85,
                        "coherence": 85,
                        "creativity": 75,
                        "speed": 95
                    }
                )
            },
            default_model="claude-3-haiku-20240307"
        )
        self.providers["claude"] = claude_config
        
        # OpenAI Configuration
        openai_config = ProviderConfig(
            provider_id="openai",
            display_name="OpenAI",
            base_url="https://api.openai.com/v1",
            requests_per_minute=500,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            models={
                "gpt-4-turbo-preview": ModelConfig(
                    model_id="gpt-4-turbo-preview",
                    display_name="GPT-4 Turbo",
                    provider="openai",
                    input_cost_per_1k=0.01,
                    output_cost_per_1k=0.03,
                    max_input_tokens=128000,
                    max_output_tokens=4096,
                    max_total_tokens=128000,
                    typical_latency_ms=2500,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.ANALYSIS,
                        ModelCapability.CREATIVE_WRITING
                    ],
                    quality_ratings={
                        "accuracy": 93,
                        "coherence": 92,
                        "creativity": 88,
                        "speed": 75
                    }
                ),
                "gpt-3.5-turbo": ModelConfig(
                    model_id="gpt-3.5-turbo",
                    display_name="GPT-3.5 Turbo",
                    provider="openai",
                    input_cost_per_1k=0.0005,
                    output_cost_per_1k=0.0015,
                    max_input_tokens=16385,
                    max_output_tokens=4096,
                    max_total_tokens=16385,
                    typical_latency_ms=1000,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.SUMMARIZATION,
                        ModelCapability.QUESTION_ANSWERING
                    ],
                    quality_ratings={
                        "accuracy": 85,
                        "coherence": 85,
                        "creativity": 80,
                        "speed": 90
                    }
                )
            },
            default_model="gpt-3.5-turbo"
        )
        self.providers["openai"] = openai_config
        
        # Google Gemini Configuration
        gemini_config = ProviderConfig(
            provider_id="gemini",
            display_name="Google Gemini",
            base_url="https://generativelanguage.googleapis.com",
            api_version="v1",
            requests_per_minute=1000,
            supports_streaming=True,
            supports_vision=True,
            models={
                "gemini-pro": ModelConfig(
                    model_id="gemini-pro",
                    display_name="Gemini Pro",
                    provider="gemini",
                    input_cost_per_1k=0.0005,
                    output_cost_per_1k=0.0015,
                    max_input_tokens=30720,
                    max_output_tokens=2048,
                    max_total_tokens=32768,
                    typical_latency_ms=1200,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.ANALYSIS,
                        ModelCapability.QUESTION_ANSWERING
                    ],
                    quality_ratings={
                        "accuracy": 88,
                        "coherence": 87,
                        "creativity": 82,
                        "speed": 85
                    }
                ),
                "gemini-pro-vision": ModelConfig(
                    model_id="gemini-pro-vision",
                    display_name="Gemini Pro Vision",
                    provider="gemini",
                    input_cost_per_1k=0.0005,
                    output_cost_per_1k=0.0015,
                    max_input_tokens=16384,
                    max_output_tokens=2048,
                    max_total_tokens=16384,
                    typical_latency_ms=1500,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.ANALYSIS
                    ],
                    quality_ratings={
                        "accuracy": 87,
                        "coherence": 86,
                        "creativity": 80,
                        "speed": 82
                    }
                )
            },
            default_model="gemini-pro"
        )
        self.providers["gemini"] = gemini_config
        
        logger.info(f"Loaded {len(self.providers)} default provider configurations")
    
    def _load_custom_configs(self):
        """Load custom provider configurations from file."""
        config_path = Path("config/ai_providers.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    custom_configs = json.load(f)
                    
                for provider_id, config_data in custom_configs.items():
                    if provider_id in self.providers:
                        # Update existing provider
                        self._update_provider_config(provider_id, config_data)
                    else:
                        # Add new provider
                        self._add_custom_provider(provider_id, config_data)
                        
                logger.info(f"Loaded custom configurations from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load custom configs: {str(e)}")
    
    def _update_provider_config(self, provider_id: str, config_data: Dict[str, Any]):
        """Update existing provider configuration."""
        provider = self.providers[provider_id]
        
        # Update rate limits
        if 'rate_limits' in config_data:
            for key, value in config_data['rate_limits'].items():
                setattr(provider, key, value)
        
        # Update models
        if 'models' in config_data:
            for model_id, model_data in config_data['models'].items():
                if model_id in provider.models:
                    # Update existing model
                    model = provider.models[model_id]
                    for key, value in model_data.items():
                        setattr(model, key, value)
    
    def _add_custom_provider(self, provider_id: str, config_data: Dict[str, Any]):
        """Add a custom provider configuration."""
        # Implementation for adding completely new providers
        pass
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration."""
        return self.providers.get(provider_id)
    
    def get_model(self, provider_id: str, model_id: Optional[str] = None) -> Optional[ModelConfig]:
        """Get model configuration."""
        provider = self.get_provider(provider_id)
        if provider:
            return provider.get_model(model_id)
        return None
    
    def get_cheapest_model(
        self,
        capability: Optional[ModelCapability] = None,
        min_quality: int = 80
    ) -> Optional[tuple[str, ModelConfig]]:
        """
        Get the cheapest model that meets requirements.
        
        Args:
            capability: Required capability
            min_quality: Minimum quality rating
            
        Returns:
            Tuple of (provider_id, model_config)
        """
        candidates = []
        
        for provider_id, provider in self.providers.items():
            for model_id, model in provider.models.items():
                if not model.is_available:
                    continue
                
                # Check capability
                if capability and capability not in model.capabilities:
                    continue
                
                # Check quality
                avg_quality = sum(model.quality_ratings.values()) / len(model.quality_ratings) if model.quality_ratings else 0
                if avg_quality < min_quality:
                    continue
                
                # Calculate average cost
                avg_cost = (model.input_cost_per_1k + model.output_cost_per_1k) / 2
                candidates.append((provider_id, model, avg_cost))
        
        if candidates:
            candidates.sort(key=lambda x: x[2])
            return candidates[0][0], candidates[0][1]
        
        return None
    
    def get_fastest_model(
        self,
        capability: Optional[ModelCapability] = None,
        min_quality: int = 80
    ) -> Optional[tuple[str, ModelConfig]]:
        """
        Get the fastest model that meets requirements.
        
        Args:
            capability: Required capability
            min_quality: Minimum quality rating
            
        Returns:
            Tuple of (provider_id, model_config)
        """
        candidates = []
        
        for provider_id, provider in self.providers.items():
            for model_id, model in provider.models.items():
                if not model.is_available:
                    continue
                
                # Check capability
                if capability and capability not in model.capabilities:
                    continue
                
                # Check quality
                avg_quality = sum(model.quality_ratings.values()) / len(model.quality_ratings) if model.quality_ratings else 0
                if avg_quality < min_quality:
                    continue
                
                candidates.append((provider_id, model, model.typical_latency_ms))
        
        if candidates:
            candidates.sort(key=lambda x: x[2])
            return candidates[0][0], candidates[0][1]
        
        return None
    
    def get_best_model(
        self,
        capability: Optional[ModelCapability] = None,
        quality_metric: str = "accuracy"
    ) -> Optional[tuple[str, ModelConfig]]:
        """
        Get the best model for a specific quality metric.
        
        Args:
            capability: Required capability
            quality_metric: Metric to optimize for
            
        Returns:
            Tuple of (provider_id, model_config)
        """
        candidates = []
        
        for provider_id, provider in self.providers.items():
            for model_id, model in provider.models.items():
                if not model.is_available:
                    continue
                
                # Check capability
                if capability and capability not in model.capabilities:
                    continue
                
                # Get quality score
                score = model.quality_ratings.get(quality_metric, 0)
                candidates.append((provider_id, model, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[0][0], candidates[0][1]
        
        return None
    
    def estimate_cost(
        self,
        provider_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        model = self.get_model(provider_id, model_id)
        if model:
            return model.calculate_cost(input_tokens, output_tokens)
        return 0.0
    
    def get_fallback_chain(
        self,
        primary_provider: str,
        task_type: str = "general"
    ) -> List[str]:
        """
        Get fallback provider chain.
        
        Args:
            primary_provider: Primary provider ID
            task_type: Type of task
            
        Returns:
            Ordered list of provider IDs for fallback
        """
        # Define fallback chains per task type
        chains = {
            "transcript_quality": ["claude", "openai", "gemini"],
            "content_quality": ["claude", "openai", "gemini"],
            "content_generation": ["claude", "openai", "gemini"],
            "general": ["claude", "openai", "gemini"]
        }
        
        chain = chains.get(task_type, chains["general"])
        
        # Move primary to front
        if primary_provider in chain:
            chain.remove(primary_provider)
        chain.insert(0, primary_provider)
        
        # Filter to available providers
        return [p for p in chain if p in self.providers]
    
    def save_custom_config(self, config_path: Optional[Path] = None):
        """Save current configuration to file."""
        if not config_path:
            config_path = Path("config/ai_providers.json")
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        config_data = {}
        for provider_id, provider in self.providers.items():
            config_data[provider_id] = {
                "display_name": provider.display_name,
                "rate_limits": {
                    "requests_per_minute": provider.requests_per_minute,
                    "requests_per_hour": provider.requests_per_hour,
                    "requests_per_day": provider.requests_per_day
                },
                "models": {}
            }
            
            for model_id, model in provider.models.items():
                config_data[provider_id]["models"][model_id] = {
                    "display_name": model.display_name,
                    "input_cost_per_1k": model.input_cost_per_1k,
                    "output_cost_per_1k": model.output_cost_per_1k,
                    "is_available": model.is_available
                }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")


# Singleton instance
_config_manager = None


def get_ai_config_manager() -> AIProviderConfigManager:
    """Get or create the AI provider config manager singleton."""
    global _config_manager
    if _config_manager is None:
        _config_manager = AIProviderConfigManager()
    return _config_manager