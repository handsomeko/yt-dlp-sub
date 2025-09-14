"""
Content Quality Management System.

This module provides specialized management for content quality evaluation prompts,
separate from transcript quality and content generation prompts. It focuses on
evaluating the quality of generated content like blog posts, social media, summaries, etc.
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
    ContentQualityPrompt,
    ContentQualityVersion,
    ContentQualityExperiment,
    ContentQualityAnalytics
)
from core.prompt_templates import get_template_engine
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ContentPlatform:
    """Supported content platforms."""
    BLOG = "blog"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"
    NEWSLETTER = "newsletter"
    GENERIC = "generic"


class ContentType:
    """Content types for evaluation."""
    BLOG_POST = "blog_post"
    SOCIAL_POST = "social_post"
    SUMMARY = "summary"
    NEWSLETTER = "newsletter"
    SCRIPT = "script"
    THREAD = "thread"  # Twitter/social threads


class ContentQualityManager:
    """
    Manages prompts specifically for content quality evaluation.
    
    Features:
    - Content-type specific evaluation
    - Platform-aware quality checks
    - SEO and engagement optimization
    - User feedback integration
    - Brand voice consistency checks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = db_manager
        self.template_engine = get_template_engine()
        self._cache = {}
        self._active_experiments = {}
        
    async def get_evaluation_prompt(
        self,
        content: str,
        content_type: str,
        platform: Optional[str] = None,
        source_transcript: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the appropriate content evaluation prompt.
        
        Args:
            content: The generated content to evaluate
            content_type: Type of content (blog, social, summary, etc.)
            platform: Target platform (twitter, linkedin, etc.)
            source_transcript: Original transcript for accuracy check
            metadata: Additional context (word_limit, tone, requirements, etc.)
            
        Returns:
            Tuple of (rendered_prompt, prompt_metadata)
        """
        # Normalize platform
        platform = platform or ContentPlatform.GENERIC
        
        # Check for active experiments first
        experiment_prompt = await self._check_experiment(content_type, platform)
        if experiment_prompt:
            prompt_template, prompt_metadata = experiment_prompt
        else:
            # Get the appropriate prompt based on criteria
            prompt_template, prompt_metadata = await self._get_prompt_by_criteria(
                content_type=content_type,
                platform=platform,
                word_count=len(content.split()) if content else 0
            )
        
        # Prepare variables for template
        variables = {
            'content': content,
            'content_type': content_type,
            'platform': platform,
            'source_transcript': source_transcript,
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0,
            'video_title': metadata.get('video_title') if metadata else None,
            'target_word_count': metadata.get('word_limit', 500) if metadata else 500,
            'tone': metadata.get('tone', 'professional') if metadata else 'professional'
        }
        
        # Add platform-specific variables
        if platform == ContentPlatform.TWITTER:
            variables['tweet_count'] = len(content.split('\n\n')) if '\n\n' in content else 1
            variables['char_limit'] = 280
        elif platform == ContentPlatform.LINKEDIN:
            variables['char_limit'] = 3000
        elif platform == ContentPlatform.INSTAGRAM:
            variables['char_limit'] = 2200
            variables['hashtag_limit'] = 30
        
        # Add metadata-specific variables
        if metadata:
            variables.update({
                'brand_voice': metadata.get('brand_voice'),
                'target_audience': metadata.get('target_audience'),
                'seo_keywords': metadata.get('seo_keywords', []),
                'requirements': metadata.get('requirements', []),
                'call_to_action': metadata.get('call_to_action')
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
        content_type: str,
        template: str,
        platform: Optional[str] = None,
        evaluation_focus: Optional[List[str]] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None,
        description: Optional[str] = None,
        evaluation_rubric: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> int:
        """
        Create a new content quality evaluation prompt.
        
        Args:
            name: Unique name for the prompt
            content_type: Type of content this evaluates
            template: Prompt template with variables
            platform: Specific platform or None for generic
            evaluation_focus: List of focus areas (engagement, seo, accuracy, etc.)
            min_word_count: Minimum word count for this prompt
            max_word_count: Maximum word count for this prompt
            description: Optional description
            evaluation_rubric: Detailed scoring rubric
            created_by: Creator identifier
            
        Returns:
            ID of created prompt
        """
        async with self.db_manager.get_session() as session:
            # Create prompt
            prompt = ContentQualityPrompt(
                name=name,
                content_type=content_type,
                platform=platform,
                description=description,
                evaluation_focus=evaluation_focus or ['quality', 'accuracy', 'engagement'],
                min_word_count=min_word_count,
                max_word_count=max_word_count,
                current_version=1,
                is_active=True,
                is_default=False
            )
            session.add(prompt)
            await session.flush()
            
            # Create initial version
            version = ContentQualityVersion(
                prompt_id=prompt.id,
                version=1,
                template=template,
                evaluation_rubric=evaluation_rubric or self._default_rubric(content_type, platform),
                platform_requirements=self._platform_requirements(platform),
                changelog="Initial version",
                created_by=created_by
            )
            session.add(version)
            
            await session.commit()
            
            logger.info(f"Created content quality prompt '{name}' for {content_type}/{platform}")
            return prompt.id
    
    async def update_prompt(
        self,
        prompt_id: int,
        template: str,
        changelog: str,
        evaluation_rubric: Optional[Dict[str, Any]] = None,
        platform_requirements: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> int:
        """
        Update a content quality prompt by creating a new version.
        
        Args:
            prompt_id: ID of prompt to update
            template: New template content
            changelog: Description of changes
            evaluation_rubric: Updated scoring rubric
            platform_requirements: Updated platform requirements
            created_by: User making the update
            
        Returns:
            New version number
        """
        async with self.db_manager.get_session() as session:
            # Get prompt
            stmt = select(ContentQualityPrompt).where(
                ContentQualityPrompt.id == prompt_id
            )
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            if not prompt:
                raise ValueError(f"Content quality prompt {prompt_id} not found")
            
            # Create new version
            new_version = prompt.current_version + 1
            version = ContentQualityVersion(
                prompt_id=prompt_id,
                version=new_version,
                template=template,
                evaluation_rubric=evaluation_rubric,
                platform_requirements=platform_requirements or self._platform_requirements(prompt.platform),
                changelog=changelog,
                created_by=created_by
            )
            session.add(version)
            
            # Update prompt's current version
            prompt.current_version = new_version
            prompt.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Clear cache
            self._clear_cache(prompt.content_type, prompt.platform)
            
            logger.info(f"Updated content quality prompt {prompt_id} to version {new_version}")
            return new_version
    
    async def record_evaluation_result(
        self,
        prompt_metadata: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        content_id: Optional[int] = None,
        execution_time: float = 0.0,
        model_used: str = "unknown",
        user_feedback: Optional[int] = None
    ) -> None:
        """
        Record analytics for a content quality evaluation.
        
        Args:
            prompt_metadata: Metadata from prompt selection
            evaluation_result: The evaluation results
            content_id: ID of evaluated content
            execution_time: Time taken for evaluation
            model_used: AI model used
            user_feedback: Optional user rating (1-5)
        """
        async with self.db_manager.get_session() as session:
            # Extract scores from result
            analytics = ContentQualityAnalytics(
                prompt_id=prompt_metadata.get('prompt_id'),
                prompt_version=prompt_metadata.get('version'),
                experiment_id=prompt_metadata.get('experiment_id'),
                variant_id=prompt_metadata.get('variant_id'),
                content_id=content_id,
                overall_score=evaluation_result.get('score'),
                engagement_score=evaluation_result.get('engagement_score'),
                seo_score=evaluation_result.get('seo_score'),
                readability_score=evaluation_result.get('readability_score'),
                accuracy_to_source=evaluation_result.get('accuracy_to_source'),
                platform_appropriateness=evaluation_result.get('platform_appropriateness'),
                format_compliance=evaluation_result.get('format_compliance'),
                tokens_used=prompt_metadata.get('tokens_used', 0),
                execution_time=execution_time,
                model_used=model_used,
                user_feedback=user_feedback
            )
            session.add(analytics)
            await session.commit()
            
            # Update average quality score for this version
            await self._update_version_quality_score(
                prompt_metadata['prompt_id'],
                prompt_metadata['version'],
                evaluation_result.get('score', 0)
            )
    
    async def create_experiment(
        self,
        name: str,
        content_type: str,
        platform: Optional[str],
        control_prompt_id: int,
        variants: List[Dict[str, Any]],
        duration_days: int = 14,
        min_samples: int = 100,
        traffic_split: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Create an A/B test for content quality evaluation.
        
        Args:
            name: Experiment name
            content_type: Content type being tested
            platform: Platform being tested
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
            experiment = ContentQualityExperiment(
                name=name,
                content_type=content_type,
                platform=platform,
                prompt_id=control_prompt_id,
                variants=variants,
                traffic_split=traffic_split,
                success_metrics={
                    'min_samples': min_samples,
                    'primary_metric': 'overall_score',
                    'secondary_metrics': ['engagement_score', 'seo_score', 'user_feedback']
                },
                status='draft',
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=duration_days)
            )
            session.add(experiment)
            await session.commit()
            
            logger.info(f"Created content quality experiment '{name}'")
            return experiment.id
    
    async def _get_prompt_by_criteria(
        self,
        content_type: str,
        platform: Optional[str] = None,
        word_count: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get prompt based on evaluation criteria.
        
        Args:
            content_type: Type of content
            platform: Target platform
            word_count: Content word count
            
        Returns:
            Tuple of (template, metadata)
        """
        async with self.db_manager.get_session() as session:
            # Build query
            stmt = select(ContentQualityPrompt).where(
                and_(
                    ContentQualityPrompt.content_type == content_type,
                    ContentQualityPrompt.is_active == True
                )
            )
            
            # Filter by platform
            if platform and platform != ContentPlatform.GENERIC:
                stmt = stmt.where(
                    or_(
                        ContentQualityPrompt.platform == platform,
                        ContentQualityPrompt.platform.is_(None),
                        ContentQualityPrompt.platform == ContentPlatform.GENERIC
                    )
                )
            
            # Filter by word count if provided
            if word_count is not None:
                stmt = stmt.where(
                    and_(
                        or_(
                            ContentQualityPrompt.min_word_count.is_(None),
                            ContentQualityPrompt.min_word_count <= word_count
                        ),
                        or_(
                            ContentQualityPrompt.max_word_count.is_(None),
                            ContentQualityPrompt.max_word_count >= word_count
                        )
                    )
                )
            
            # Prioritize platform-specific over generic
            stmt = stmt.order_by(
                ContentQualityPrompt.platform != platform,  # Platform match first
                ContentQualityPrompt.is_default.asc(),
                ContentQualityPrompt.updated_at.desc()
            )
            
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            if not prompt:
                # Fallback to default for content type
                stmt = select(ContentQualityPrompt).where(
                    and_(
                        ContentQualityPrompt.content_type == content_type,
                        ContentQualityPrompt.is_default == True
                    )
                )
                result = await session.execute(stmt)
                prompt = result.scalar_one_or_none()
            
            if not prompt:
                # Use hardcoded fallback
                return self._get_fallback_prompt(content_type, platform)
            
            # Get current version
            stmt = select(ContentQualityVersion).where(
                and_(
                    ContentQualityVersion.prompt_id == prompt.id,
                    ContentQualityVersion.version == prompt.current_version
                )
            )
            result = await session.execute(stmt)
            version = result.scalar_one_or_none()
            
            if not version:
                return self._get_fallback_prompt(content_type, platform)
            
            metadata = {
                'prompt_id': prompt.id,
                'prompt_name': prompt.name,
                'version': version.version,
                'content_type': content_type,
                'platform': platform,
                'is_experiment': False
            }
            
            return version.template, metadata
    
    async def _check_experiment(
        self,
        content_type: str,
        platform: Optional[str] = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Check for active experiments and select variant.
        
        Args:
            content_type: Content type
            platform: Target platform
            
        Returns:
            Tuple of (template, metadata) or None
        """
        # Check cache
        cache_key = f"{content_type}_{platform}"
        if cache_key in self._active_experiments:
            exp = self._active_experiments[cache_key]
            if exp['expires'] > datetime.utcnow():
                return self._select_experiment_variant(exp)
        
        # Load from database
        async with self.db_manager.get_session() as session:
            stmt = select(ContentQualityExperiment).where(
                and_(
                    ContentQualityExperiment.content_type == content_type,
                    or_(
                        ContentQualityExperiment.platform == platform,
                        ContentQualityExperiment.platform.is_(None)
                    ),
                    ContentQualityExperiment.status == 'active',
                    ContentQualityExperiment.start_date <= datetime.utcnow(),
                    or_(
                        ContentQualityExperiment.end_date.is_(None),
                        ContentQualityExperiment.end_date > datetime.utcnow()
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
    
    def _default_rubric(self, content_type: str, platform: Optional[str]) -> Dict[str, Any]:
        """Get default evaluation rubric for content type and platform."""
        base_rubric = {
            'quality_weight': 0.3,
            'accuracy_weight': 0.3,
            'engagement_weight': 0.2,
            'format_weight': 0.2
        }
        
        # Adjust for content type
        if content_type == ContentType.BLOG_POST:
            base_rubric.update({
                'seo_weight': 0.2,
                'readability_weight': 0.2,
                'min_paragraphs': 5,
                'require_headers': True
            })
        elif content_type == ContentType.SOCIAL_POST:
            base_rubric.update({
                'engagement_weight': 0.4,
                'shareability_weight': 0.2,
                'hashtag_quality': True
            })
        elif content_type == ContentType.SUMMARY:
            base_rubric.update({
                'completeness_weight': 0.3,
                'conciseness_weight': 0.2
            })
        
        # Adjust for platform
        if platform == ContentPlatform.TWITTER:
            base_rubric['char_limit_strict'] = True
            base_rubric['thread_structure'] = True
        elif platform == ContentPlatform.LINKEDIN:
            base_rubric['professional_tone'] = True
            base_rubric['cta_required'] = True
        
        return base_rubric
    
    def _platform_requirements(self, platform: Optional[str]) -> Dict[str, Any]:
        """Get platform-specific requirements."""
        requirements = {
            ContentPlatform.TWITTER: {
                'max_chars': 280,
                'max_thread_length': 25,
                'hashtag_limit': 2,
                'mention_limit': 10
            },
            ContentPlatform.LINKEDIN: {
                'max_chars': 3000,
                'min_chars': 100,
                'hashtag_limit': 5,
                'professional_language': True
            },
            ContentPlatform.INSTAGRAM: {
                'max_chars': 2200,
                'hashtag_limit': 30,
                'emoji_encouraged': True
            },
            ContentPlatform.FACEBOOK: {
                'max_chars': 63206,
                'optimal_chars': 40,
                'link_preview': True
            },
            ContentPlatform.NEWSLETTER: {
                'subject_line_max': 50,
                'preview_text_max': 90,
                'sections_required': True
            }
        }
        return requirements.get(platform, {})
    
    def _get_fallback_prompt(self, content_type: str, platform: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Get hardcoded fallback prompt."""
        # Generic template that works for any content type
        template = """Evaluate this {{ content_type }} content for quality.

Content Type: {{ content_type }}
Platform: {{ platform }}
Word Count: {{ word_count }}
Target: {{ target_word_count }} words

Content to evaluate:
{{ content | truncate_words(2000) }}

{% if source_transcript %}
Source reference (first 500 words):
{{ source_transcript | truncate_words(500) }}
{% endif %}

Respond with JSON:
{
  "score": <0-100>,
  "pass": <true if score >= 70>,
  "format_compliance": <true/false>,
  "engagement_score": <0-100>,
  "seo_score": <0-100>,
  "readability_score": <0-100>,
  "accuracy_to_source": <0-100 if source provided>,
  "platform_appropriateness": <0-100>,
  "issues": [<specific issues>],
  "improvements": [<suggestions>],
  "strengths": [<what works well>]
}"""
        
        metadata = {
            'prompt_id': None,
            'prompt_name': f'fallback_{content_type}_{platform}',
            'version': 0,
            'content_type': content_type,
            'platform': platform,
            'is_fallback': True
        }
        
        return template, metadata
    
    async def _update_version_quality_score(
        self,
        prompt_id: int,
        version: int,
        score: float
    ) -> None:
        """Update average quality score for a version."""
        async with self.db_manager.get_session() as session:
            stmt = select(ContentQualityVersion).where(
                and_(
                    ContentQualityVersion.prompt_id == prompt_id,
                    ContentQualityVersion.version == version
                )
            )
            result = await session.execute(stmt)
            version_obj = result.scalar_one_or_none()
            
            if version_obj:
                # Update with weighted average
                if version_obj.avg_quality_score:
                    version_obj.avg_quality_score = (
                        version_obj.avg_quality_score * 0.95 + score * 0.05
                    )
                else:
                    version_obj.avg_quality_score = score
                
                await session.commit()
    
    def _clear_cache(self, content_type: Optional[str] = None, platform: Optional[str] = None):
        """Clear experiment cache."""
        if content_type:
            pattern = f"{content_type}_"
            if platform:
                pattern = f"{content_type}_{platform}"
            
            keys_to_clear = [k for k in self._active_experiments if k.startswith(pattern)]
            for key in keys_to_clear:
                del self._active_experiments[key]
        else:
            self._active_experiments.clear()


# Singleton instance
_content_quality_manager = None


def get_content_quality_manager() -> ContentQualityManager:
    """Get or create the content quality manager singleton."""
    global _content_quality_manager
    if _content_quality_manager is None:
        _content_quality_manager = ContentQualityManager()
    return _content_quality_manager