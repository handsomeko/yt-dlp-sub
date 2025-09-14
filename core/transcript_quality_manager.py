"""
Transcript Quality Management System.

This module provides specialized management for transcript quality evaluation prompts,
separate from content generation and content quality prompts. It focuses on
evaluating the accuracy and completeness of extracted transcripts.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import (
    db_manager,
    TranscriptQualityPrompt,
    TranscriptQualityVersion,
    TranscriptQualityExperiment,
    TranscriptQualityAnalytics
)
from core.prompt_templates import get_template_engine
from config.settings import get_settings

logger = logging.getLogger(__name__)


class StrictnessLevel:
    """Transcript evaluation strictness levels."""
    LENIENT = "lenient"  # Basic checks, higher pass rate
    STANDARD = "standard"  # Balanced evaluation
    STRICT = "strict"  # Rigorous checks, lower pass rate
    CUSTOM = "custom"  # Custom criteria


class TranscriptQualityManager:
    """
    Manages prompts specifically for transcript quality evaluation.
    
    Features:
    - Strictness-based prompt selection
    - Duration-aware prompt selection
    - Extraction method specific prompts
    - Specialized A/B testing for accuracy optimization
    - False positive/negative tracking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = db_manager
        self.template_engine = get_template_engine()
        self._cache = {}
        self._active_experiments = {}
        
    async def get_evaluation_prompt(
        self,
        transcript: str,
        video_duration: Optional[int] = None,
        extraction_method: Optional[str] = None,
        strictness: str = StrictnessLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the appropriate transcript evaluation prompt.
        
        Args:
            transcript: The transcript to evaluate
            video_duration: Video duration in seconds
            extraction_method: How transcript was extracted (whisper, ffmpeg, etc.)
            strictness: Evaluation strictness level
            metadata: Additional context (title, channel, etc.)
            
        Returns:
            Tuple of (rendered_prompt, prompt_metadata)
        """
        # Check for active experiments first
        experiment_prompt = await self._check_experiment(strictness, extraction_method)
        if experiment_prompt:
            prompt_template, prompt_metadata = experiment_prompt
        else:
            # Get the appropriate prompt based on criteria
            prompt_template, prompt_metadata = await self._get_prompt_by_criteria(
                strictness=strictness,
                duration=video_duration,
                extraction_method=extraction_method
            )
        
        # Prepare variables for template
        variables = {
            'transcript': transcript,
            'video_duration': video_duration,
            'extraction_method': extraction_method or 'unknown',
            'video_title': metadata.get('title') if metadata else 'Unknown',
            'channel_name': metadata.get('channel_name') if metadata else None,
            'expected_word_count': self._estimate_word_count(video_duration) if video_duration else None,
            'strictness_level': strictness
        }
        
        # Add metadata-specific variables
        if metadata:
            variables.update({
                'language': metadata.get('language', 'en'),
                'has_captions': metadata.get('has_captions', False),
                'is_auto_generated': metadata.get('is_auto_generated', False)
            })
        
        # Render template
        rendered_prompt = self.template_engine.render(
            prompt_template,
            variables,
            safe_mode=True
        )
        
        return rendered_prompt, prompt_metadata
    
    async def create_prompt(
        self,
        name: str,
        template: str,
        strictness: str = StrictnessLevel.STANDARD,
        extraction_method: Optional[str] = None,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None,
        description: Optional[str] = None,
        evaluation_criteria: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> int:
        """
        Create a new transcript quality evaluation prompt.
        
        Args:
            name: Unique name for the prompt
            template: Prompt template with variables
            strictness: Strictness level for evaluation
            extraction_method: Specific extraction method or None for any
            min_duration: Minimum video duration for this prompt
            max_duration: Maximum video duration for this prompt
            description: Optional description
            evaluation_criteria: Specific evaluation criteria
            created_by: Creator identifier
            
        Returns:
            ID of created prompt
        """
        async with self.db_manager.get_session() as session:
            # Create prompt
            prompt = TranscriptQualityPrompt(
                name=name,
                description=description,
                strictness_level=strictness,
                extraction_method=extraction_method,
                min_duration=min_duration,
                max_duration=max_duration,
                current_version=1,
                is_active=True,
                is_default=False
            )
            session.add(prompt)
            await session.flush()
            
            # Create initial version
            version = TranscriptQualityVersion(
                prompt_id=prompt.id,
                version=1,
                template=template,
                evaluation_criteria=evaluation_criteria or self._default_criteria(strictness),
                changelog="Initial version",
                created_by=created_by
            )
            session.add(version)
            
            await session.commit()
            
            logger.info(f"Created transcript quality prompt '{name}' with strictness '{strictness}'")
            return prompt.id
    
    async def update_prompt(
        self,
        prompt_id: int,
        template: str,
        changelog: str,
        evaluation_criteria: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> int:
        """
        Update a transcript quality prompt by creating a new version.
        
        Args:
            prompt_id: ID of prompt to update
            template: New template content
            changelog: Description of changes
            evaluation_criteria: Updated evaluation criteria
            created_by: User making the update
            
        Returns:
            New version number
        """
        async with self.db_manager.get_session() as session:
            # Get prompt
            stmt = select(TranscriptQualityPrompt).where(
                TranscriptQualityPrompt.id == prompt_id
            )
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            if not prompt:
                raise ValueError(f"Transcript quality prompt {prompt_id} not found")
            
            # Create new version
            new_version = prompt.current_version + 1
            version = TranscriptQualityVersion(
                prompt_id=prompt_id,
                version=new_version,
                template=template,
                evaluation_criteria=evaluation_criteria,
                changelog=changelog,
                created_by=created_by
            )
            session.add(version)
            
            # Update prompt's current version
            prompt.current_version = new_version
            prompt.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Clear cache
            self._clear_cache(prompt.strictness_level)
            
            logger.info(f"Updated transcript quality prompt {prompt_id} to version {new_version}")
            return new_version
    
    async def record_evaluation_result(
        self,
        prompt_metadata: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        transcript_id: Optional[int] = None,
        execution_time: float = 0.0,
        model_used: str = "unknown"
    ) -> None:
        """
        Record analytics for a transcript quality evaluation.
        
        Args:
            prompt_metadata: Metadata from prompt selection
            evaluation_result: The evaluation results
            transcript_id: ID of evaluated transcript
            execution_time: Time taken for evaluation
            model_used: AI model used
        """
        async with self.db_manager.get_session() as session:
            # Extract scores from result
            analytics = TranscriptQualityAnalytics(
                prompt_id=prompt_metadata.get('prompt_id'),
                prompt_version=prompt_metadata.get('version'),
                experiment_id=prompt_metadata.get('experiment_id'),
                variant_id=prompt_metadata.get('variant_id'),
                transcript_id=transcript_id,
                quality_score=evaluation_result.get('score'),
                completeness_score=evaluation_result.get('completeness'),
                coherence_score=evaluation_result.get('coherence'),
                accuracy_score=evaluation_result.get('accuracy'),
                issues_detected=len(evaluation_result.get('issues', [])),
                tokens_used=prompt_metadata.get('tokens_used', 0),
                execution_time=execution_time,
                model_used=model_used
            )
            session.add(analytics)
            await session.commit()
            
            # Update accuracy score for this version if we have ground truth
            if prompt_metadata.get('has_ground_truth'):
                await self._update_version_accuracy(
                    prompt_metadata['prompt_id'],
                    prompt_metadata['version'],
                    evaluation_result.get('accuracy_vs_ground_truth')
                )
    
    async def create_experiment(
        self,
        name: str,
        strictness_level: str,
        control_prompt_id: int,
        variants: List[Dict[str, Any]],
        duration_days: int = 7,
        min_samples: int = 50,
        traffic_split: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Create an A/B test for transcript quality evaluation.
        
        Args:
            name: Experiment name
            strictness_level: Strictness level being tested
            control_prompt_id: Control prompt ID
            variants: List of variant templates
            duration_days: Experiment duration
            min_samples: Minimum samples per variant
            traffic_split: Traffic distribution
            
        Returns:
            Experiment ID
        """
        # Prepare variants
        for variant in variants:
            if 'id' not in variant:
                variant['id'] = f"v{len(variants)}"
        
        # Auto-calculate traffic split
        if not traffic_split:
            num_variants = len(variants) + 1
            split = 1.0 / num_variants
            traffic_split = {
                'control': split,
                **{v['id']: split for v in variants}
            }
        
        async with self.db_manager.get_session() as session:
            experiment = TranscriptQualityExperiment(
                name=name,
                prompt_id=control_prompt_id,
                strictness_level=strictness_level,
                variants=variants,
                traffic_split=traffic_split,
                success_metrics={
                    'min_samples': min_samples,
                    'primary_metric': 'accuracy_score',
                    'secondary_metrics': ['false_positive_rate', 'false_negative_rate']
                },
                status='draft',
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=duration_days)
            )
            session.add(experiment)
            await session.commit()
            
            logger.info(f"Created transcript quality experiment '{name}'")
            return experiment.id
    
    async def _get_prompt_by_criteria(
        self,
        strictness: str,
        duration: Optional[int] = None,
        extraction_method: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get prompt based on evaluation criteria.
        
        Args:
            strictness: Strictness level
            duration: Video duration in seconds
            extraction_method: Extraction method used
            
        Returns:
            Tuple of (template, metadata)
        """
        async with self.db_manager.get_session() as session:
            # Build query
            stmt = select(TranscriptQualityPrompt).where(
                and_(
                    TranscriptQualityPrompt.strictness_level == strictness,
                    TranscriptQualityPrompt.is_active == True
                )
            )
            
            # Filter by duration if provided
            if duration is not None:
                stmt = stmt.where(
                    and_(
                        or_(
                            TranscriptQualityPrompt.min_duration.is_(None),
                            TranscriptQualityPrompt.min_duration <= duration
                        ),
                        or_(
                            TranscriptQualityPrompt.max_duration.is_(None),
                            TranscriptQualityPrompt.max_duration >= duration
                        )
                    )
                )
            
            # Filter by extraction method if provided
            if extraction_method:
                stmt = stmt.where(
                    or_(
                        TranscriptQualityPrompt.extraction_method == extraction_method,
                        TranscriptQualityPrompt.extraction_method.is_(None)
                    )
                )
            
            # Try to get custom prompt first, then default
            stmt = stmt.order_by(
                TranscriptQualityPrompt.is_default.asc(),
                TranscriptQualityPrompt.updated_at.desc()
            )
            
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            if not prompt:
                # Fallback to any prompt for this strictness
                stmt = select(TranscriptQualityPrompt).where(
                    and_(
                        TranscriptQualityPrompt.strictness_level == strictness,
                        TranscriptQualityPrompt.is_default == True
                    )
                )
                result = await session.execute(stmt)
                prompt = result.scalar_one_or_none()
            
            if not prompt:
                # Use hardcoded fallback
                return self._get_fallback_prompt(strictness)
            
            # Get current version
            stmt = select(TranscriptQualityVersion).where(
                and_(
                    TranscriptQualityVersion.prompt_id == prompt.id,
                    TranscriptQualityVersion.version == prompt.current_version
                )
            )
            result = await session.execute(stmt)
            version = result.scalar_one_or_none()
            
            if not version:
                return self._get_fallback_prompt(strictness)
            
            metadata = {
                'prompt_id': prompt.id,
                'prompt_name': prompt.name,
                'version': version.version,
                'strictness': strictness,
                'extraction_method': extraction_method,
                'is_experiment': False
            }
            
            return version.template, metadata
    
    async def _check_experiment(
        self,
        strictness: str,
        extraction_method: Optional[str] = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Check for active experiments and select variant.
        
        Args:
            strictness: Strictness level
            extraction_method: Extraction method
            
        Returns:
            Tuple of (template, metadata) or None
        """
        # Check cache
        cache_key = f"{strictness}_{extraction_method}"
        if cache_key in self._active_experiments:
            exp = self._active_experiments[cache_key]
            if exp['expires'] > datetime.utcnow():
                return self._select_experiment_variant(exp)
        
        # Load from database
        async with self.db_manager.get_session() as session:
            stmt = select(TranscriptQualityExperiment).where(
                and_(
                    TranscriptQualityExperiment.strictness_level == strictness,
                    TranscriptQualityExperiment.status == 'active',
                    TranscriptQualityExperiment.start_date <= datetime.utcnow(),
                    or_(
                        TranscriptQualityExperiment.end_date.is_(None),
                        TranscriptQualityExperiment.end_date > datetime.utcnow()
                    )
                )
            )
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
            
            # Cache experiment
            self._active_experiments[cache_key] = {
                'id': experiment.id,
                'variants': experiment.variants,
                'traffic_split': experiment.traffic_split,
                'expires': datetime.utcnow() + timedelta(minutes=5)
            }
            
            return self._select_experiment_variant(self._active_experiments[cache_key])
    
    def _select_experiment_variant(
        self,
        experiment: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select a variant from an experiment."""
        traffic_split = experiment.get('traffic_split', {})
        variants = experiment.get('variants', [])
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        
        for variant in variants:
            cumulative += traffic_split.get(variant['id'], 0)
            if rand < cumulative:
                metadata = {
                    'experiment_id': experiment['id'],
                    'variant_id': variant['id'],
                    'is_experiment': True
                }
                return variant['template'], metadata
        
        # Shouldn't reach here, but fallback to control
        return None
    
    def _estimate_word_count(self, duration_seconds: int) -> int:
        """
        Estimate expected word count based on video duration.
        
        Assumes average speaking rate of 150 words per minute.
        """
        minutes = duration_seconds / 60
        return int(minutes * 150)
    
    def _default_criteria(self, strictness: str) -> Dict[str, Any]:
        """Get default evaluation criteria for a strictness level."""
        criteria = {
            StrictnessLevel.LENIENT: {
                'min_score': 60,
                'completeness_weight': 0.3,
                'coherence_weight': 0.4,
                'accuracy_weight': 0.3,
                'allow_minor_issues': True
            },
            StrictnessLevel.STANDARD: {
                'min_score': 70,
                'completeness_weight': 0.35,
                'coherence_weight': 0.35,
                'accuracy_weight': 0.3,
                'allow_minor_issues': False
            },
            StrictnessLevel.STRICT: {
                'min_score': 85,
                'completeness_weight': 0.4,
                'coherence_weight': 0.3,
                'accuracy_weight': 0.3,
                'allow_minor_issues': False,
                'require_speaker_labels': True,
                'require_punctuation': True
            }
        }
        return criteria.get(strictness, criteria[StrictnessLevel.STANDARD])
    
    def _get_fallback_prompt(self, strictness: str) -> Tuple[str, Dict[str, Any]]:
        """Get hardcoded fallback prompt."""
        templates = {
            StrictnessLevel.LENIENT: """Evaluate this transcript for basic quality.

Video Duration: {{ video_duration | format_duration }}
Transcript ({{ transcript | word_count }} words):
{{ transcript | truncate_words(1500) }}

Respond with JSON:
{
  "score": <0-100>,
  "pass": <true if score >= 60>,
  "issues": [<major issues only>],
  "recommendations": [<key improvements>]
}""",
            
            StrictnessLevel.STANDARD: """Analyze this transcript for quality and completeness.

Video Duration: {{ video_duration | format_duration }}
Expected Words: ~{{ expected_word_count }}
Actual Words: {{ transcript | word_count }}

Transcript:
{{ transcript | truncate_words(2000) }}

Respond with JSON:
{
  "score": <0-100>,
  "pass": <true if score >= 70>,
  "completeness": <0-100>,
  "coherence": <0-100>,
  "accuracy": <0-100>,
  "issues": [<specific issues>],
  "recommendations": [<improvements>]
}""",
            
            StrictnessLevel.STRICT: """Perform rigorous quality analysis of this transcript.

Video: {{ video_title }}
Duration: {{ video_duration | format_duration }}
Extraction: {{ extraction_method }}
Expected Words: {{ expected_word_count }}
Actual Words: {{ transcript | word_count }}

Transcript:
{{ transcript | truncate_words(3000) }}

Respond with JSON:
{
  "score": <0-100>,
  "pass": <true if score >= 85>,
  "completeness": <0-100>,
  "coherence": <0-100>,
  "accuracy": <0-100>,
  "has_speaker_labels": <boolean>,
  "has_timestamps": <boolean>,
  "sentence_structure": <"good"/"fair"/"poor">,
  "technical_accuracy": <0-100>,
  "issues": {
    "critical": [],
    "major": [],
    "minor": []
  },
  "recommendations": [],
  "requires_retranscription": <boolean>
}"""
        }
        
        template = templates.get(strictness, templates[StrictnessLevel.STANDARD])
        metadata = {
            'prompt_id': None,
            'prompt_name': f'fallback_{strictness}',
            'version': 0,
            'strictness': strictness,
            'is_fallback': True
        }
        
        return template, metadata
    
    async def _update_version_accuracy(
        self,
        prompt_id: int,
        version: int,
        accuracy: float
    ) -> None:
        """Update historical accuracy score for a version."""
        async with self.db_manager.get_session() as session:
            stmt = select(TranscriptQualityVersion).where(
                and_(
                    TranscriptQualityVersion.prompt_id == prompt_id,
                    TranscriptQualityVersion.version == version
                )
            )
            result = await session.execute(stmt)
            version_obj = result.scalar_one_or_none()
            
            if version_obj:
                # Update with weighted average
                if version_obj.accuracy_score:
                    version_obj.accuracy_score = (
                        version_obj.accuracy_score * 0.9 + accuracy * 0.1
                    )
                else:
                    version_obj.accuracy_score = accuracy
                
                await session.commit()
    
    def _clear_cache(self, strictness: Optional[str] = None):
        """Clear experiment cache."""
        if strictness:
            keys_to_clear = [k for k in self._active_experiments if k.startswith(f"{strictness}_")]
            for key in keys_to_clear:
                del self._active_experiments[key]
        else:
            self._active_experiments.clear()


# Singleton instance
_transcript_quality_manager = None


def get_transcript_quality_manager() -> TranscriptQualityManager:
    """Get or create the transcript quality manager singleton."""
    global _transcript_quality_manager
    if _transcript_quality_manager is None:
        _transcript_quality_manager = TranscriptQualityManager()
    return _transcript_quality_manager