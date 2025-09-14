"""
Prompt Management System for content generation.

This module provides centralized prompt template management with versioning,
A/B testing support, and performance tracking.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import (
    db_manager, 
    Prompt, 
    PromptVersion, 
    PromptExperiment, 
    PromptAnalytics
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates for content generation.
    
    Features:
    - CRUD operations for prompts
    - Version control with rollback
    - A/B testing support
    - Performance tracking
    - Default prompt fallback
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = db_manager
        self._cache = {}  # In-memory cache for frequently used prompts
        self._experiments = {}  # Active experiments cache
        
    async def get_prompt(
        self,
        content_type: str,
        experiment_aware: bool = True,
        variables: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the appropriate prompt for a content type.
        
        Args:
            content_type: Type of content (blog_post, summary, etc.)
            experiment_aware: Whether to consider active A/B tests
            variables: Variables to substitute in template
            
        Returns:
            Tuple of (rendered_prompt, metadata)
        """
        # Check for active experiments first
        if experiment_aware:
            experiment_prompt = await self._get_experiment_prompt(content_type)
            if experiment_prompt:
                prompt_template, metadata = experiment_prompt
                return self._render_template(prompt_template, variables), metadata
        
        # Get the active prompt for this content type
        async with self.db_manager.get_session() as session:
            # Try to get custom active prompt
            stmt = select(Prompt).where(
                and_(
                    Prompt.content_type == content_type,
                    Prompt.is_active == True,
                    Prompt.is_default == False
                )
            )
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            # Fallback to default prompt
            if not prompt:
                stmt = select(Prompt).where(
                    and_(
                        Prompt.content_type == content_type,
                        Prompt.is_default == True
                    )
                )
                result = await session.execute(stmt)
                prompt = result.scalar_one_or_none()
            
            if not prompt:
                # No prompt found, use hardcoded fallback
                logger.warning(f"No prompt found for content_type: {content_type}")
                return self._get_fallback_prompt(content_type, variables)
            
            # Get the current version
            stmt = select(PromptVersion).where(
                and_(
                    PromptVersion.prompt_id == prompt.id,
                    PromptVersion.version == prompt.current_version
                )
            )
            result = await session.execute(stmt)
            version = result.scalar_one_or_none()
            
            if not version:
                logger.error(f"No version found for prompt {prompt.id} v{prompt.current_version}")
                return self._get_fallback_prompt(content_type, variables)
            
            metadata = {
                "prompt_id": prompt.id,
                "prompt_name": prompt.name,
                "version": version.version,
                "content_type": content_type,
                "is_experiment": False
            }
            
            return self._render_template(version.template, variables), metadata
    
    async def create_prompt(
        self,
        name: str,
        content_type: str,
        template: str,
        description: Optional[str] = None,
        variables: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> int:
        """
        Create a new prompt template.
        
        Args:
            name: Unique name for the prompt
            content_type: Type of content this prompt generates
            template: The prompt template with {variables}
            description: Optional description
            variables: List of variables used in template
            created_by: User or system creating the prompt
            
        Returns:
            ID of created prompt
        """
        async with self.db_manager.get_session() as session:
            # Create prompt
            prompt = Prompt(
                name=name,
                content_type=content_type,
                description=description,
                current_version=1,
                is_active=True,
                is_default=False
            )
            session.add(prompt)
            await session.flush()
            
            # Create initial version
            version = PromptVersion(
                prompt_id=prompt.id,
                version=1,
                template=template,
                variables=variables or self._extract_variables(template),
                changelog="Initial version",
                created_by=created_by
            )
            session.add(version)
            
            await session.commit()
            
            logger.info(f"Created prompt '{name}' for {content_type}")
            return prompt.id
    
    async def update_prompt(
        self,
        prompt_id: int,
        template: str,
        changelog: str,
        created_by: str = "system"
    ) -> int:
        """
        Update a prompt by creating a new version.
        
        Args:
            prompt_id: ID of prompt to update
            template: New template content
            changelog: Description of changes
            created_by: User making the update
            
        Returns:
            New version number
        """
        async with self.db_manager.get_session() as session:
            # Get prompt
            stmt = select(Prompt).where(Prompt.id == prompt_id)
            result = await session.execute(stmt)
            prompt = result.scalar_one_or_none()
            
            if not prompt:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Create new version
            new_version = prompt.current_version + 1
            version = PromptVersion(
                prompt_id=prompt_id,
                version=new_version,
                template=template,
                variables=self._extract_variables(template),
                changelog=changelog,
                created_by=created_by
            )
            session.add(version)
            
            # Update prompt's current version
            prompt.current_version = new_version
            prompt.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Clear cache
            self._clear_cache(prompt.content_type)
            
            logger.info(f"Updated prompt {prompt_id} to version {new_version}")
            return new_version
    
    async def rollback_prompt(
        self,
        prompt_id: int,
        target_version: int,
        reason: str,
        created_by: str = "system"
    ) -> int:
        """
        Rollback a prompt to a previous version.
        
        Args:
            prompt_id: ID of prompt to rollback
            target_version: Version to rollback to
            reason: Reason for rollback
            created_by: User performing rollback
            
        Returns:
            New version number (copy of target)
        """
        async with self.db_manager.get_session() as session:
            # Get target version
            stmt = select(PromptVersion).where(
                and_(
                    PromptVersion.prompt_id == prompt_id,
                    PromptVersion.version == target_version
                )
            )
            result = await session.execute(stmt)
            target = result.scalar_one_or_none()
            
            if not target:
                raise ValueError(f"Version {target_version} not found for prompt {prompt_id}")
            
            # Create new version as copy
            stmt = select(Prompt).where(Prompt.id == prompt_id)
            result = await session.execute(stmt)
            prompt = result.scalar_one()
            
            new_version = prompt.current_version + 1
            version = PromptVersion(
                prompt_id=prompt_id,
                version=new_version,
                template=target.template,
                variables=target.variables,
                changelog=f"Rollback to v{target_version}: {reason}",
                created_by=created_by
            )
            session.add(version)
            
            # Update prompt
            prompt.current_version = new_version
            prompt.updated_at = datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Rolled back prompt {prompt_id} to v{target_version} (new v{new_version})")
            return new_version
    
    async def list_prompts(
        self,
        content_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List available prompts.
        
        Args:
            content_type: Filter by content type
            active_only: Only show active prompts
            
        Returns:
            List of prompt details
        """
        async with self.db_manager.get_session() as session:
            stmt = select(Prompt)
            
            if content_type:
                stmt = stmt.where(Prompt.content_type == content_type)
            
            if active_only:
                stmt = stmt.where(Prompt.is_active == True)
            
            result = await session.execute(stmt)
            prompts = result.scalars().all()
            
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "content_type": p.content_type,
                    "description": p.description,
                    "current_version": p.current_version,
                    "is_active": p.is_active,
                    "is_default": p.is_default,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None
                }
                for p in prompts
            ]
    
    async def get_prompt_versions(self, prompt_id: int) -> List[Dict[str, Any]]:
        """
        Get all versions of a prompt.
        
        Args:
            prompt_id: Prompt ID
            
        Returns:
            List of version details
        """
        async with self.db_manager.get_session() as session:
            stmt = select(PromptVersion).where(
                PromptVersion.prompt_id == prompt_id
            ).order_by(PromptVersion.version.desc())
            
            result = await session.execute(stmt)
            versions = result.scalars().all()
            
            return [
                {
                    "version": v.version,
                    "template": v.template,
                    "variables": v.variables,
                    "changelog": v.changelog,
                    "performance_score": v.performance_score,
                    "created_by": v.created_by,
                    "created_at": v.created_at.isoformat() if v.created_at else None
                }
                for v in versions
            ]
    
    async def _get_experiment_prompt(
        self,
        content_type: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get prompt variant if there's an active experiment.
        
        Args:
            content_type: Content type to check
            
        Returns:
            Tuple of (template, metadata) or None
        """
        # Check cache first
        if content_type in self._experiments:
            experiment = self._experiments[content_type]
            if experiment['expires'] > datetime.utcnow():
                return self._select_variant(experiment)
        
        # Load active experiment from database
        async with self.db_manager.get_session() as session:
            stmt = select(PromptExperiment).where(
                and_(
                    PromptExperiment.content_type == content_type,
                    PromptExperiment.status == 'active',
                    PromptExperiment.start_date <= datetime.utcnow(),
                    or_(
                        PromptExperiment.end_date.is_(None),
                        PromptExperiment.end_date > datetime.utcnow()
                    )
                )
            )
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
            
            # Cache experiment
            self._experiments[content_type] = {
                'id': experiment.id,
                'variants': experiment.variants,
                'traffic_split': experiment.traffic_split,
                'expires': datetime.utcnow().replace(second=0, microsecond=0)  # Cache for 1 minute
            }
            
            return self._select_variant(self._experiments[content_type])
    
    def _select_variant(self, experiment: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select a variant based on traffic split.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            Selected variant template and metadata
        """
        traffic_split = experiment.get('traffic_split', {})
        variants = experiment.get('variants', [])
        
        # Simple weighted random selection
        rand = random.random()
        cumulative = 0
        
        for variant in variants:
            cumulative += traffic_split.get(variant['id'], 0)
            if rand < cumulative:
                metadata = {
                    "experiment_id": experiment['id'],
                    "variant_id": variant['id'],
                    "is_experiment": True
                }
                return variant['template'], metadata
        
        # Fallback to first variant
        if variants:
            metadata = {
                "experiment_id": experiment['id'],
                "variant_id": variants[0]['id'],
                "is_experiment": True
            }
            return variants[0]['template'], metadata
        
        return None
    
    def _render_template(
        self,
        template: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render a template with variable substitution.
        
        Args:
            template: Template string with {variables}
            variables: Variable values to substitute
            
        Returns:
            Rendered template
        """
        if not variables:
            return template
        
        # Simple string format for now
        # In production, use Jinja2 or similar
        try:
            # Add default values for common variables
            defaults = {
                'timestamp': datetime.utcnow().isoformat(),
                'word_limit': 500,
                'tone': 'professional'
            }
            render_vars = {**defaults, **variables}
            
            return template.format(**render_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template
    
    def _extract_variables(self, template: str) -> List[str]:
        """
        Extract variable names from a template.
        
        Args:
            template: Template string
            
        Returns:
            List of variable names
        """
        import re
        # Find all {variable} patterns
        pattern = r'\{([^}]+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))
    
    def _clear_cache(self, content_type: Optional[str] = None):
        """Clear prompt cache."""
        if content_type:
            self._cache.pop(content_type, None)
            self._experiments.pop(content_type, None)
        else:
            self._cache.clear()
            self._experiments.clear()
    
    def _get_fallback_prompt(
        self,
        content_type: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get hardcoded fallback prompt.
        
        Args:
            content_type: Content type
            variables: Variables for template
            
        Returns:
            Fallback prompt and metadata
        """
        # These match the original hardcoded prompts
        fallbacks = {
            "summary": """Create a comprehensive summary of this video transcript.
            
Video Title: {title}
Transcript:
{transcript}

Please provide:
1. A one-paragraph executive summary
2. 3-5 key points
3. Main insights or takeaways

Format the response in clear, readable sections.""",
            
            "blog_post": """Transform this video transcript into an engaging blog post.
            
Video Title: {title}
Transcript:
{transcript}

Create a 500-800 word blog post that:
1. Has an attention-grabbing title
2. Includes an engaging introduction
3. Breaks down the main points with headers
4. Ends with a strong conclusion
5. Maintains the original message while improving readability""",
            
            "social_media": """Create social media posts from this video transcript.
            
Video Title: {title}
Key Content:
{transcript}

Generate:
1. A Twitter/X thread (3-5 tweets)
2. A LinkedIn post (150-200 words)
3. Key hashtags (5-7 relevant ones)

Make it engaging and shareable while preserving the main message.""",
            
            "newsletter": """Create newsletter content from this video.
            
Video Title: {title}
Transcript:
{transcript}

Structure:
1. Catchy subject line
2. Brief introduction (2-3 sentences)
3. Main points with bullet points
4. Call to action
5. Sign-off

Keep it concise and actionable.""",
            
            "scripts": """Create a short-form video script from this content.
            
Original Video: {title}
Content:
{transcript}

Create a 60-second script that:
1. Has a strong hook (first 3 seconds)
2. Delivers the main value quickly
3. Includes a clear call to action
4. Is optimized for TikTok/YouTube Shorts format"""
        }
        
        template = fallbacks.get(content_type, fallbacks["summary"])
        metadata = {
            "prompt_id": None,
            "prompt_name": f"fallback_{content_type}",
            "version": 0,
            "content_type": content_type,
            "is_experiment": False,
            "is_fallback": True
        }
        
        if variables:
            template = self._render_template(template, variables)
        
        return template, metadata


# Singleton instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get or create the prompt manager singleton."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager