"""
A/B Testing Manager for prompt experiments.

Handles experiment lifecycle, variant selection, and statistical analysis
for prompt optimization.
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
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import db_manager, PromptExperiment, PromptAnalytics
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle states."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantSelectionMethod(Enum):
    """Methods for selecting variants."""
    RANDOM = "random"  # Pure random selection
    WEIGHTED = "weighted"  # Weighted by configured split
    EPSILON_GREEDY = "epsilon_greedy"  # Exploration vs exploitation
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian approach


class ABTestManager:
    """
    Manages A/B testing experiments for prompts.
    
    Features:
    - Experiment lifecycle management
    - Multiple variant selection algorithms
    - Statistical significance testing
    - Automatic winner selection
    - Performance tracking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = db_manager
        self._active_experiments = {}  # Cache for active experiments
        
    async def create_experiment(
        self,
        name: str,
        content_type: str,
        control_prompt_id: int,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        duration_days: int = 14,
        min_samples: int = 100,
        confidence_level: float = 0.95,
        metrics: Optional[List[str]] = None
    ) -> int:
        """
        Create a new A/B testing experiment.
        
        Args:
            name: Unique experiment name
            content_type: Type of content being tested
            control_prompt_id: ID of control prompt
            variants: List of variant configurations
            traffic_split: Traffic distribution (auto-calculated if None)
            duration_days: Experiment duration
            min_samples: Minimum samples per variant
            confidence_level: Statistical confidence level
            metrics: Metrics to track
            
        Returns:
            Experiment ID
        """
        # Prepare variants with IDs
        for variant in variants:
            if 'id' not in variant:
                variant['id'] = str(uuid4())[:8]
        
        # Auto-calculate equal traffic split if not provided
        if not traffic_split:
            num_variants = len(variants) + 1  # +1 for control
            split = 1.0 / num_variants
            traffic_split = {
                'control': split,
                **{v['id']: split for v in variants}
            }
        
        # Default metrics
        if not metrics:
            metrics = ['quality_score', 'tokens_used', 'execution_time']
        
        # Winner criteria
        winner_criteria = {
            'min_samples': min_samples,
            'confidence_level': confidence_level,
            'primary_metric': metrics[0],
            'improvement_threshold': 0.05  # 5% improvement required
        }
        
        async with self.db_manager.get_session() as session:
            experiment = PromptExperiment(
                name=name,
                content_type=content_type,
                prompt_id=control_prompt_id,
                variants=variants,
                traffic_split=traffic_split,
                metrics=metrics,
                winner_criteria=winner_criteria,
                status=ExperimentStatus.DRAFT.value,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=duration_days)
            )
            session.add(experiment)
            await session.commit()
            
            logger.info(f"Created experiment '{name}' with {len(variants)} variants")
            return experiment.id
    
    async def start_experiment(self, experiment_id: int) -> bool:
        """
        Start an experiment.
        
        Args:
            experiment_id: ID of experiment to start
            
        Returns:
            Success status
        """
        async with self.db_manager.get_session() as session:
            stmt = select(PromptExperiment).where(PromptExperiment.id == experiment_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                logger.error(f"Experiment {experiment_id} not found")
                return False
            
            if experiment.status != ExperimentStatus.DRAFT.value:
                logger.warning(f"Cannot start experiment in status {experiment.status}")
                return False
            
            # Check for conflicting active experiments
            stmt = select(PromptExperiment).where(
                and_(
                    PromptExperiment.content_type == experiment.content_type,
                    PromptExperiment.status == ExperimentStatus.ACTIVE.value,
                    PromptExperiment.id != experiment_id
                )
            )
            result = await session.execute(stmt)
            conflicts = result.scalars().all()
            
            if conflicts:
                logger.error(f"Active experiment already exists for {experiment.content_type}")
                return False
            
            experiment.status = ExperimentStatus.ACTIVE.value
            experiment.start_date = datetime.utcnow()
            await session.commit()
            
            # Clear cache
            self._clear_cache(experiment.content_type)
            
            logger.info(f"Started experiment '{experiment.name}'")
            return True
    
    async def pause_experiment(self, experiment_id: int) -> bool:
        """Pause an active experiment."""
        return await self._update_status(
            experiment_id,
            ExperimentStatus.PAUSED
        )
    
    async def resume_experiment(self, experiment_id: int) -> bool:
        """Resume a paused experiment."""
        return await self._update_status(
            experiment_id,
            ExperimentStatus.ACTIVE
        )
    
    async def stop_experiment(
        self,
        experiment_id: int,
        select_winner: bool = True
    ) -> Optional[str]:
        """
        Stop an experiment and optionally select winner.
        
        Args:
            experiment_id: Experiment to stop
            select_winner: Whether to analyze and select winner
            
        Returns:
            Winner variant ID if selected
        """
        async with self.db_manager.get_session() as session:
            stmt = select(PromptExperiment).where(PromptExperiment.id == experiment_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
            
            winner = None
            if select_winner:
                winner = await self.analyze_experiment(experiment_id)
                if winner:
                    experiment.winner_variant = winner['variant_id']
            
            experiment.status = ExperimentStatus.COMPLETED.value
            experiment.end_date = datetime.utcnow()
            await session.commit()
            
            self._clear_cache(experiment.content_type)
            
            logger.info(f"Stopped experiment '{experiment.name}', winner: {winner}")
            return winner['variant_id'] if winner else None
    
    async def select_variant(
        self,
        content_type: str,
        method: VariantSelectionMethod = VariantSelectionMethod.WEIGHTED
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select a variant for the active experiment.
        
        Args:
            content_type: Content type to get variant for
            method: Selection method to use
            
        Returns:
            Tuple of (variant_id, variant_config)
        """
        # Check cache
        if content_type in self._active_experiments:
            experiment = self._active_experiments[content_type]
            if experiment['expires'] > datetime.utcnow():
                return self._select_variant_internal(experiment, method)
        
        # Load from database
        async with self.db_manager.get_session() as session:
            stmt = select(PromptExperiment).where(
                and_(
                    PromptExperiment.content_type == content_type,
                    PromptExperiment.status == ExperimentStatus.ACTIVE.value
                )
            )
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return "control", {}
            
            # Cache experiment
            self._active_experiments[content_type] = {
                'id': experiment.id,
                'variants': experiment.variants,
                'traffic_split': experiment.traffic_split,
                'expires': datetime.utcnow() + timedelta(minutes=5)
            }
            
            return self._select_variant_internal(
                self._active_experiments[content_type],
                method
            )
    
    async def record_result(
        self,
        experiment_id: int,
        variant_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Record experiment result.
        
        Args:
            experiment_id: Experiment ID
            variant_id: Variant that was used
            metrics: Performance metrics
        """
        async with self.db_manager.get_session() as session:
            analytics = PromptAnalytics(
                prompt_id=None,  # Will be set based on variant
                experiment_id=experiment_id,
                variant_id=variant_id,
                quality_score=metrics.get('quality_score'),
                tokens_used=metrics.get('tokens_used'),
                execution_time=metrics.get('execution_time'),
                user_feedback=metrics.get('user_feedback'),
                model_used=metrics.get('model_used'),
                error_occurred=metrics.get('error_occurred', False),
                error_message=metrics.get('error_message')
            )
            session.add(analytics)
            await session.commit()
    
    async def analyze_experiment(
        self,
        experiment_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze experiment results and determine winner.
        
        Args:
            experiment_id: Experiment to analyze
            
        Returns:
            Analysis results with winner if determined
        """
        async with self.db_manager.get_session() as session:
            # Get experiment
            stmt = select(PromptExperiment).where(PromptExperiment.id == experiment_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
            
            # Get all analytics
            stmt = select(PromptAnalytics).where(
                PromptAnalytics.experiment_id == experiment_id
            )
            result = await session.execute(stmt)
            analytics = result.scalars().all()
            
            if not analytics:
                return None
            
            # Group by variant
            variant_data = {}
            for record in analytics:
                variant_id = record.variant_id or 'control'
                if variant_id not in variant_data:
                    variant_data[variant_id] = []
                
                variant_data[variant_id].append({
                    'quality_score': record.quality_score,
                    'tokens_used': record.tokens_used,
                    'execution_time': record.execution_time
                })
            
            # Statistical analysis
            criteria = experiment.winner_criteria or {}
            min_samples = criteria.get('min_samples', 100)
            confidence_level = criteria.get('confidence_level', 0.95)
            primary_metric = criteria.get('primary_metric', 'quality_score')
            
            # Check sample sizes
            for variant_id, data in variant_data.items():
                if len(data) < min_samples:
                    logger.info(f"Insufficient samples for {variant_id}: {len(data)}")
                    return None
            
            # Perform statistical tests
            control_data = variant_data.get('control', [])
            if not control_data:
                return None
            
            control_scores = [d[primary_metric] for d in control_data if d[primary_metric] is not None]
            
            results = []
            for variant_id, data in variant_data.items():
                if variant_id == 'control':
                    continue
                
                variant_scores = [d[primary_metric] for d in data if d[primary_metric] is not None]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(variant_scores, control_scores)
                
                # Calculate effect size (Cohen's d)
                effect_size = (np.mean(variant_scores) - np.mean(control_scores)) / np.sqrt(
                    (np.var(variant_scores) + np.var(control_scores)) / 2
                )
                
                results.append({
                    'variant_id': variant_id,
                    'mean_score': np.mean(variant_scores),
                    'std_score': np.std(variant_scores),
                    'sample_size': len(variant_scores),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < (1 - confidence_level),
                    'improvement': (np.mean(variant_scores) - np.mean(control_scores)) / np.mean(control_scores)
                })
            
            # Select winner
            significant_improvements = [
                r for r in results
                if r['significant'] and r['improvement'] > criteria.get('improvement_threshold', 0.05)
            ]
            
            if significant_improvements:
                winner = max(significant_improvements, key=lambda x: x['improvement'])
                return {
                    'winner': True,
                    'variant_id': winner['variant_id'],
                    'improvement': winner['improvement'],
                    'p_value': winner['p_value'],
                    'effect_size': winner['effect_size'],
                    'all_results': results
                }
            
            return {
                'winner': False,
                'message': 'No significant winner found',
                'all_results': results
            }
    
    def _select_variant_internal(
        self,
        experiment: Dict[str, Any],
        method: VariantSelectionMethod
    ) -> Tuple[str, Dict[str, Any]]:
        """Internal variant selection logic."""
        variants = experiment.get('variants', [])
        traffic_split = experiment.get('traffic_split', {})
        
        if method == VariantSelectionMethod.RANDOM:
            # Pure random selection
            all_variants = ['control'] + [v['id'] for v in variants]
            selected = random.choice(all_variants)
            
        elif method == VariantSelectionMethod.WEIGHTED:
            # Weighted random selection
            choices = []
            weights = []
            
            choices.append('control')
            weights.append(traffic_split.get('control', 0.5))
            
            for variant in variants:
                choices.append(variant['id'])
                weights.append(traffic_split.get(variant['id'], 0.5 / len(variants)))
            
            selected = random.choices(choices, weights=weights)[0]
            
        else:
            # Default to weighted
            selected = self._select_variant_internal(
                experiment,
                VariantSelectionMethod.WEIGHTED
            )[0]
        
        # Get variant config
        if selected == 'control':
            return 'control', {'is_control': True}
        
        for variant in variants:
            if variant['id'] == selected:
                return selected, variant
        
        return 'control', {}
    
    async def _update_status(
        self,
        experiment_id: int,
        status: ExperimentStatus
    ) -> bool:
        """Update experiment status."""
        async with self.db_manager.get_session() as session:
            stmt = select(PromptExperiment).where(PromptExperiment.id == experiment_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return False
            
            experiment.status = status.value
            experiment.updated_at = datetime.utcnow()
            await session.commit()
            
            self._clear_cache(experiment.content_type)
            return True
    
    def _clear_cache(self, content_type: Optional[str] = None):
        """Clear experiment cache."""
        if content_type:
            self._active_experiments.pop(content_type, None)
        else:
            self._active_experiments.clear()


# Singleton instance
_ab_test_manager = None


def get_ab_test_manager() -> ABTestManager:
    """Get or create the A/B test manager singleton."""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager