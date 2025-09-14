"""
AI Provider A/B Testing Manager

Manages experiments comparing different AI providers (Claude, OpenAI, Gemini)
for quality checks and content generation. Tracks performance, cost, and reliability.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ProviderExperimentType(Enum):
    """Types of provider experiments."""
    TRANSCRIPT_QUALITY = "transcript_quality"
    CONTENT_QUALITY = "content_quality"
    CONTENT_GENERATION = "content_generation"
    ALL_TASKS = "all_tasks"


class ProviderSelectionStrategy(Enum):
    """Strategies for selecting AI providers."""
    ROUND_ROBIN = "round_robin"  # Rotate through providers
    WEIGHTED_RANDOM = "weighted_random"  # Random based on weights
    PERFORMANCE_BASED = "performance_based"  # Best performing provider
    COST_OPTIMIZED = "cost_optimized"  # Lowest cost provider
    LATENCY_OPTIMIZED = "latency_optimized"  # Fastest provider
    EPSILON_GREEDY = "epsilon_greedy"  # Explore vs exploit
    MULTI_ARMED_BANDIT = "multi_armed_bandit"  # Advanced optimization


class AIProviderMetrics:
    """Metrics tracked for each AI provider."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.quality_scores = []
        self.error_types = {}
        self.last_used = None
        self.consecutive_failures = 0
    
    def add_result(
        self,
        success: bool,
        tokens: int = 0,
        cost: float = 0.0,
        latency: float = 0.0,
        quality_score: Optional[float] = None,
        error_type: Optional[str] = None
    ):
        """Record a result from using this provider."""
        self.total_requests += 1
        self.last_used = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
            self.total_tokens += tokens
            self.total_cost += cost
            self.total_latency += latency
            if quality_score is not None:
                self.quality_scores.append(quality_score)
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return float('inf')
        return self.total_latency / self.successful_requests
    
    @property
    def average_cost(self) -> float:
        """Calculate average cost per request."""
        if self.successful_requests == 0:
            return float('inf')
        return self.total_cost / self.successful_requests
    
    @property
    def average_quality(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return np.mean(self.quality_scores)
    
    @property
    def quality_std(self) -> float:
        """Calculate quality score standard deviation."""
        if len(self.quality_scores) < 2:
            return 0.0
        return np.std(self.quality_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'average_cost': self.average_cost,
            'average_latency': self.average_latency,
            'average_quality': self.average_quality,
            'quality_std': self.quality_std,
            'error_types': self.error_types,
            'consecutive_failures': self.consecutive_failures,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


class AIProviderABTestManager:
    """
    Manages A/B testing between different AI providers.
    
    Features:
    - Compare providers across different task types
    - Track performance, cost, and reliability metrics
    - Automatic provider selection based on strategy
    - Fallback chains for reliability
    - Cost optimization
    - Statistical analysis of provider performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Available providers and their configurations
        self.providers = {
            'claude_cli': {
                'name': 'Claude CLI',
                'enabled': True,
                'cost_per_1k_tokens': 0.003,  # Haiku pricing
                'max_tokens': 1000,
                'timeout': 30,
                'rate_limit': 100  # requests per minute
            },
            'claude_api': {
                'name': 'Claude API',
                'enabled': True,
                'cost_per_1k_tokens': 0.003,  # Haiku pricing
                'max_tokens': 1000,
                'timeout': 30,
                'rate_limit': 1000
            },
            'openai_api': {
                'name': 'OpenAI API',
                'enabled': True,
                'cost_per_1k_tokens': 0.002,  # GPT-3.5 pricing
                'max_tokens': 1000,
                'timeout': 30,
                'rate_limit': 500
            },
            'gemini_api': {
                'name': 'Gemini API',
                'enabled': True,
                'cost_per_1k_tokens': 0.001,  # Gemini Pro pricing
                'max_tokens': 1000,
                'timeout': 30,
                'rate_limit': 1000
            }
        }
        
        # Metrics tracking per provider per task type
        self.metrics = {}
        for task_type in ProviderExperimentType:
            self.metrics[task_type.value] = {}
            for provider in self.providers:
                self.metrics[task_type.value][provider] = AIProviderMetrics()
        
        # Active experiments
        self.active_experiments = {}
        
        # Fallback chains for reliability
        self.fallback_chains = {
            ProviderExperimentType.TRANSCRIPT_QUALITY: ['claude_api', 'openai_api', 'gemini_api', 'claude_cli'],
            ProviderExperimentType.CONTENT_QUALITY: ['claude_api', 'openai_api', 'gemini_api'],
            ProviderExperimentType.CONTENT_GENERATION: ['claude_api', 'openai_api', 'gemini_api']
        }
        
        # Provider rotation state
        self.rotation_index = {}
        
        logger.info("AI Provider A/B Test Manager initialized")
    
    async def select_provider(
        self,
        task_type: ProviderExperimentType,
        strategy: ProviderSelectionStrategy = ProviderSelectionStrategy.WEIGHTED_RANDOM,
        content_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select an AI provider for a task.
        
        Args:
            task_type: Type of task (transcript quality, content quality, generation)
            strategy: Selection strategy to use
            content_metadata: Optional metadata about the content
            
        Returns:
            Tuple of (provider_id, provider_config)
        """
        # Check if there's an active experiment
        if task_type.value in self.active_experiments:
            experiment = self.active_experiments[task_type.value]
            if experiment['end_date'] > datetime.utcnow():
                return self._select_for_experiment(experiment, task_type)
        
        # Select based on strategy
        if strategy == ProviderSelectionStrategy.ROUND_ROBIN:
            provider = self._select_round_robin(task_type)
        elif strategy == ProviderSelectionStrategy.WEIGHTED_RANDOM:
            provider = self._select_weighted_random(task_type)
        elif strategy == ProviderSelectionStrategy.PERFORMANCE_BASED:
            provider = self._select_performance_based(task_type)
        elif strategy == ProviderSelectionStrategy.COST_OPTIMIZED:
            provider = self._select_cost_optimized(task_type)
        elif strategy == ProviderSelectionStrategy.LATENCY_OPTIMIZED:
            provider = self._select_latency_optimized(task_type)
        elif strategy == ProviderSelectionStrategy.EPSILON_GREEDY:
            provider = self._select_epsilon_greedy(task_type)
        else:
            provider = self._select_weighted_random(task_type)
        
        # Check if provider is healthy
        if not self._is_provider_healthy(provider, task_type):
            provider = self._get_fallback_provider(provider, task_type)
        
        return provider, self.providers[provider]
    
    async def record_result(
        self,
        task_type: ProviderExperimentType,
        provider: str,
        success: bool,
        execution_time: float,
        tokens_used: int = 0,
        quality_score: Optional[float] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record the result of using a provider.
        
        Args:
            task_type: Type of task
            provider: Provider that was used
            success: Whether the request succeeded
            execution_time: Time taken in seconds
            tokens_used: Number of tokens consumed
            quality_score: Quality score if applicable
            error_type: Type of error if failed
            metadata: Additional metadata
        """
        # Calculate cost
        cost = (tokens_used / 1000) * self.providers[provider]['cost_per_1k_tokens']
        
        # Update metrics
        self.metrics[task_type.value][provider].add_result(
            success=success,
            tokens=tokens_used,
            cost=cost,
            latency=execution_time,
            quality_score=quality_score,
            error_type=error_type
        )
        
        # Log for monitoring
        if success:
            logger.info(
                f"Provider {provider} succeeded for {task_type.value}: "
                f"latency={execution_time:.2f}s, tokens={tokens_used}, cost=${cost:.4f}"
            )
        else:
            logger.warning(
                f"Provider {provider} failed for {task_type.value}: "
                f"error={error_type}"
            )
        
        # Check if we need to trigger a fallback
        metrics = self.metrics[task_type.value][provider]
        if metrics.consecutive_failures >= 3:
            logger.error(
                f"Provider {provider} has {metrics.consecutive_failures} consecutive failures. "
                f"Consider using fallback."
            )
    
    async def create_experiment(
        self,
        name: str,
        task_type: ProviderExperimentType,
        providers: List[str],
        duration_days: int = 7,
        traffic_split: Optional[Dict[str, float]] = None,
        min_samples: int = 100,
        confidence_level: float = 0.95
    ) -> str:
        """
        Create a new provider comparison experiment.
        
        Args:
            name: Experiment name
            task_type: Type of task to test
            providers: List of providers to compare
            duration_days: Duration of experiment
            traffic_split: Traffic distribution (auto if None)
            min_samples: Minimum samples per provider
            confidence_level: Statistical confidence level
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid4())[:8]
        
        # Auto-calculate traffic split if not provided
        if not traffic_split:
            split = 1.0 / len(providers)
            traffic_split = {p: split for p in providers}
        
        self.active_experiments[task_type.value] = {
            'id': experiment_id,
            'name': name,
            'providers': providers,
            'traffic_split': traffic_split,
            'start_date': datetime.utcnow(),
            'end_date': datetime.utcnow() + timedelta(days=duration_days),
            'min_samples': min_samples,
            'confidence_level': confidence_level,
            'results': {p: [] for p in providers}
        }
        
        logger.info(
            f"Created provider experiment '{name}' ({experiment_id}) "
            f"for {task_type.value} with providers: {providers}"
        )
        
        return experiment_id
    
    async def analyze_experiment(
        self,
        task_type: ProviderExperimentType
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze experiment results and determine winner.
        
        Args:
            task_type: Type of task
            
        Returns:
            Analysis results with winner if determined
        """
        if task_type.value not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[task_type.value]
        metrics = self.metrics[task_type.value]
        
        results = []
        for provider in experiment['providers']:
            provider_metrics = metrics[provider]
            
            if provider_metrics.total_requests < experiment['min_samples']:
                logger.info(
                    f"Insufficient samples for {provider}: "
                    f"{provider_metrics.total_requests}/{experiment['min_samples']}"
                )
                return None
            
            results.append({
                'provider': provider,
                'success_rate': provider_metrics.success_rate,
                'average_quality': provider_metrics.average_quality,
                'quality_std': provider_metrics.quality_std,
                'average_cost': provider_metrics.average_cost,
                'average_latency': provider_metrics.average_latency,
                'total_requests': provider_metrics.total_requests
            })
        
        # Statistical comparison
        if len(results) >= 2:
            # Compare quality scores using ANOVA
            quality_scores = [metrics[r['provider']].quality_scores for r in results]
            f_stat, p_value = stats.f_oneway(*quality_scores)
            
            # Find best provider
            best = max(results, key=lambda x: x['average_quality'])
            
            return {
                'experiment_id': experiment['id'],
                'winner': best['provider'] if p_value < (1 - experiment['confidence_level']) else None,
                'p_value': p_value,
                'results': results,
                'recommendation': self._get_recommendation(results)
            }
        
        return None
    
    def _select_round_robin(self, task_type: ProviderExperimentType) -> str:
        """Select provider using round-robin."""
        enabled = [p for p, c in self.providers.items() if c['enabled']]
        if not enabled:
            return 'claude_cli'
        
        key = task_type.value
        if key not in self.rotation_index:
            self.rotation_index[key] = 0
        
        provider = enabled[self.rotation_index[key] % len(enabled)]
        self.rotation_index[key] += 1
        
        return provider
    
    def _select_weighted_random(self, task_type: ProviderExperimentType) -> str:
        """Select provider using weighted random."""
        enabled = [p for p, c in self.providers.items() if c['enabled']]
        if not enabled:
            return 'claude_cli'
        
        # Use success rate as weight
        weights = []
        for provider in enabled:
            metrics = self.metrics[task_type.value][provider]
            weight = metrics.success_rate if metrics.total_requests > 0 else 0.5
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1/len(enabled)] * len(enabled)
        
        return random.choices(enabled, weights=weights)[0]
    
    def _select_performance_based(self, task_type: ProviderExperimentType) -> str:
        """Select best performing provider."""
        enabled = [p for p, c in self.providers.items() if c['enabled']]
        if not enabled:
            return 'claude_cli'
        
        best_score = -1
        best_provider = enabled[0]
        
        for provider in enabled:
            metrics = self.metrics[task_type.value][provider]
            if metrics.total_requests > 10:  # Minimum samples
                score = metrics.average_quality * metrics.success_rate
                if score > best_score:
                    best_score = score
                    best_provider = provider
        
        return best_provider
    
    def _select_cost_optimized(self, task_type: ProviderExperimentType) -> str:
        """Select lowest cost provider."""
        enabled = [p for p, c in self.providers.items() if c['enabled']]
        if not enabled:
            return 'claude_cli'
        
        min_cost = float('inf')
        best_provider = enabled[0]
        
        for provider in enabled:
            metrics = self.metrics[task_type.value][provider]
            if metrics.successful_requests > 0:
                if metrics.average_cost < min_cost:
                    min_cost = metrics.average_cost
                    best_provider = provider
        
        return best_provider
    
    def _select_latency_optimized(self, task_type: ProviderExperimentType) -> str:
        """Select fastest provider."""
        enabled = [p for p, c in self.providers.items() if c['enabled']]
        if not enabled:
            return 'claude_cli'
        
        min_latency = float('inf')
        best_provider = enabled[0]
        
        for provider in enabled:
            metrics = self.metrics[task_type.value][provider]
            if metrics.successful_requests > 0:
                if metrics.average_latency < min_latency:
                    min_latency = metrics.average_latency
                    best_provider = provider
        
        return best_provider
    
    def _select_epsilon_greedy(
        self,
        task_type: ProviderExperimentType,
        epsilon: float = 0.1
    ) -> str:
        """Select using epsilon-greedy strategy."""
        if random.random() < epsilon:
            # Explore: random selection
            enabled = [p for p, c in self.providers.items() if c['enabled']]
            return random.choice(enabled) if enabled else 'claude_cli'
        else:
            # Exploit: best performer
            return self._select_performance_based(task_type)
    
    def _select_for_experiment(
        self,
        experiment: Dict[str, Any],
        task_type: ProviderExperimentType
    ) -> Tuple[str, Dict[str, Any]]:
        """Select provider for active experiment."""
        # Weighted random based on traffic split
        providers = list(experiment['traffic_split'].keys())
        weights = list(experiment['traffic_split'].values())
        
        selected = random.choices(providers, weights=weights)[0]
        return selected, self.providers[selected]
    
    def _is_provider_healthy(self, provider: str, task_type: ProviderExperimentType) -> bool:
        """Check if provider is healthy."""
        metrics = self.metrics[task_type.value][provider]
        
        # Check consecutive failures
        if metrics.consecutive_failures >= 3:
            return False
        
        # Check overall success rate
        if metrics.total_requests > 10 and metrics.success_rate < 0.5:
            return False
        
        return True
    
    def _get_fallback_provider(
        self,
        failed_provider: str,
        task_type: ProviderExperimentType
    ) -> str:
        """Get fallback provider."""
        chain = self.fallback_chains.get(task_type, ['claude_cli'])
        
        for provider in chain:
            if provider != failed_provider and self._is_provider_healthy(provider, task_type):
                logger.info(f"Using fallback provider {provider} instead of {failed_provider}")
                return provider
        
        # Last resort
        return 'claude_cli'
    
    def _get_recommendation(self, results: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on results."""
        # Sort by quality
        by_quality = sorted(results, key=lambda x: x['average_quality'], reverse=True)
        
        # Sort by cost
        by_cost = sorted(results, key=lambda x: x['average_cost'])
        
        # Sort by latency
        by_latency = sorted(results, key=lambda x: x['average_latency'])
        
        best_quality = by_quality[0]['provider']
        cheapest = by_cost[0]['provider']
        fastest = by_latency[0]['provider']
        
        if best_quality == cheapest == fastest:
            return f"{best_quality} is the best overall choice"
        elif best_quality == cheapest:
            return f"{best_quality} offers best quality and cost"
        elif best_quality == fastest:
            return f"{best_quality} offers best quality and speed"
        else:
            return (
                f"Trade-offs detected: {best_quality} has best quality, "
                f"{cheapest} is most cost-effective, {fastest} is fastest"
            )
    
    def get_provider_stats(
        self,
        task_type: Optional[ProviderExperimentType] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for all providers.
        
        Args:
            task_type: Optional task type filter
            
        Returns:
            Provider statistics
        """
        stats = {}
        
        task_types = [task_type.value] if task_type else list(self.metrics.keys())
        
        for task in task_types:
            stats[task] = {}
            for provider, metrics in self.metrics[task].items():
                stats[task][provider] = metrics.to_dict()
        
        return stats
    
    def reset_metrics(
        self,
        task_type: Optional[ProviderExperimentType] = None,
        provider: Optional[str] = None
    ) -> None:
        """Reset metrics for providers."""
        if task_type and provider:
            self.metrics[task_type.value][provider] = AIProviderMetrics()
        elif task_type:
            for p in self.providers:
                self.metrics[task_type.value][p] = AIProviderMetrics()
        elif provider:
            for t in ProviderExperimentType:
                self.metrics[t.value][provider] = AIProviderMetrics()
        else:
            # Reset all
            for t in ProviderExperimentType:
                for p in self.providers:
                    self.metrics[t.value][p] = AIProviderMetrics()


# Singleton instance
_provider_ab_manager = None


def get_provider_ab_manager() -> AIProviderABTestManager:
    """Get or create the provider A/B test manager singleton."""
    global _provider_ab_manager
    if _provider_ab_manager is None:
        _provider_ab_manager = AIProviderABTestManager()
    return _provider_ab_manager