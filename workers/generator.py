"""
Content Generation Orchestrator Worker for the YouTube Content Intelligence Platform.

This module implements the GeneratorWorker that acts as the orchestrator for distributing
content generation tasks to specialized sub-generators. It coordinates parallel content
generation across multiple content types and manages the aggregation of results.

Based on PRD section 4.3 Content Generator Architecture.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from workers.base import BaseWorker, WorkerStatus


class GenerationError(Exception):
    """Base exception for content generation errors."""
    pass


class TranscriptMissingError(GenerationError):
    """Exception raised when no transcript is available for generation."""
    pass


class AIModelError(GenerationError):
    """Exception raised when AI model fails or returns invalid response."""
    pass


class ContentValidationError(GenerationError):
    """Exception raised when generated content fails validation."""
    pass


class SubGeneratorError(GenerationError):
    """Exception raised when sub-generator fails to load or execute."""
    pass


class ContentTooShortError(ContentValidationError):
    """Exception raised when generated content is too short to be useful."""
    pass


class ContentQualityError(ContentValidationError):
    """Exception raised when generated content quality is below threshold."""
    pass


class PromptTemplateError(GenerationError):
    """Exception raised when prompt template fails to render."""
    pass
from workers.generators.base_generator import ContentFormat
from core.database import DatabaseManager, Video, Transcript, GeneratedContent, Job, db_manager, PromptAnalytics
from core.queue import JobQueue, JobType, get_job_queue
from core.storage_paths_v2 import get_storage_paths_v2
from core.filename_sanitizer import sanitize_filename
from core.prompt_manager import get_prompt_manager
from core.prompt_templates import get_template_engine
from core.ab_testing import get_ab_test_manager
from config.settings import get_settings
from workers.ai_backend import AIBackend


class ContentType(Enum):
    """Supported content generation types."""
    SUMMARY = "summary"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    NEWSLETTER = "newsletter"
    SCRIPTS = "scripts"


class SummaryVariant(Enum):
    """Summary length variants."""
    SHORT = "short"      # 1-2 sentences
    MEDIUM = "medium"    # 1-2 paragraphs
    DETAILED = "detailed" # 3-5 paragraphs


class SocialPlatform(Enum):
    """Social media platform variants."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"


class ScriptFormat(Enum):
    """Script format variants."""
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    PODCAST = "podcast"


class GenerationStatus(Enum):
    """Content generation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some sub-generators failed


class GeneratorWorker(BaseWorker):
    """
    Content Generation Orchestrator Worker.
    
    Acts as the main orchestrator that distributes work to sub-generators,
    coordinates parallel content generation across multiple content types,
    and aggregates results from all sub-generators.
    
    Features:
    - Validates input transcript and content requirements
    - Creates jobs for each requested content type
    - Tracks generation progress across multiple sub-generators
    - Handles partial failures gracefully
    - Aggregates and stores final results
    - Provides detailed generation metadata
    
    For Phase 1: Implements orchestration structure without actual AI generation.
    Phase 2 will add the actual AI-powered sub-generators.
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        job_queue: Optional[JobQueue] = None,
        max_concurrent_generations: int = 5,
        generation_timeout_minutes: int = 15
    ):
        """
        Initialize the Generator Worker.
        
        Args:
            db_manager: Database manager instance
            job_queue: Job queue for sub-generator tasks
            max_concurrent_generations: Max parallel sub-generator jobs
            generation_timeout_minutes: Timeout for individual generation tasks
        """
        super().__init__(
            name="GeneratorWorker",
            max_retries=2,  # Less aggressive retries for generation tasks
            retry_delay=2.0,
            log_level="INFO"
        )
        
        self.db_manager = db_manager or globals()['db_manager']
        self.job_queue = job_queue or get_job_queue()
        self.max_concurrent_generations = max_concurrent_generations
        self.generation_timeout_minutes = generation_timeout_minutes
        self.storage_paths = get_storage_paths_v2()
        self.settings = get_settings()
        self.ai_backend = AIBackend()
        
        # Content type configurations
        self.content_type_configs = {
            ContentType.SUMMARY.value: {
                "variants": [v.value for v in SummaryVariant],
                "default_variants": [SummaryVariant.MEDIUM.value],
                "estimated_duration": 60  # seconds
            },
            ContentType.BLOG_POST.value: {
                "variants": ["500_words", "1000_words", "2000_words"],
                "default_variants": ["1000_words"],
                "estimated_duration": 180
            },
            ContentType.SOCIAL_MEDIA.value: {
                "variants": [p.value for p in SocialPlatform],
                "default_variants": [SocialPlatform.TWITTER.value, SocialPlatform.LINKEDIN.value],
                "estimated_duration": 90
            },
            ContentType.NEWSLETTER.value: {
                "variants": ["intro", "main_points", "takeaways"],
                "default_variants": ["main_points"],
                "estimated_duration": 120
            },
            ContentType.SCRIPTS.value: {
                "variants": [s.value for s in ScriptFormat],
                "default_variants": [ScriptFormat.YOUTUBE_SHORTS.value],
                "estimated_duration": 150
            }
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for content generation.
        
        Args:
            input_data: Input data containing video_id, transcript, and content_types
            
        Returns:
            True if input is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["video_id", "content_types"]
            for field in required_fields:
                if field not in input_data:
                    self.log_with_context(
                        f"Missing required field: {field}",
                        level="ERROR",
                        extra_context={"available_keys": list(input_data.keys())}
                    )
                    return False
            
            video_id = input_data["video_id"]
            content_types = input_data["content_types"]
            
            # Validate video_id format
            if not isinstance(video_id, str) or len(video_id) < 5:
                self.log_with_context(
                    f"Invalid video_id format: {video_id}",
                    level="ERROR"
                )
                return False
            
            # Validate content_types
            if not isinstance(content_types, list) or len(content_types) == 0:
                self.log_with_context(
                    "content_types must be a non-empty list",
                    level="ERROR",
                    extra_context={"provided_type": type(content_types).__name__}
                )
                return False
            
            # Check content types are supported
            supported_types = set(ct.value for ct in ContentType)
            requested_types = set(content_types)
            unsupported = requested_types - supported_types
            
            if unsupported:
                self.log_with_context(
                    f"Unsupported content types: {list(unsupported)}",
                    level="ERROR",
                    extra_context={
                        "supported_types": list(supported_types),
                        "requested_types": list(requested_types)
                    }
                )
                return False
            
            # Validate optional transcript data
            transcript_text = input_data.get("transcript_text")
            if transcript_text is not None and not isinstance(transcript_text, str):
                self.log_with_context(
                    "transcript_text must be a string if provided",
                    level="ERROR"
                )
                return False
            
            # Validate optional generation options
            generation_options = input_data.get("generation_options", {})
            if not isinstance(generation_options, dict):
                self.log_with_context(
                    "generation_options must be a dictionary if provided",
                    level="ERROR"
                )
                return False
            
            self.log_with_context(
                "Input validation passed",
                extra_context={
                    "video_id": video_id,
                    "content_types": content_types,
                    "has_transcript": transcript_text is not None,
                    "options_count": len(generation_options)
                }
            )
            
            return True
            
        except Exception as e:
            self.log_with_context(
                f"Input validation error: {str(e)}",
                level="ERROR"
            )
            return False
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content generation orchestration.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Dictionary containing generation results and metadata
        """
        video_id = input_data["video_id"]
        content_types = input_data["content_types"]
        generation_options = input_data.get("generation_options", {})
        
        self.log_with_context(
            "Starting content generation orchestration",
            extra_context={
                "video_id": video_id,
                "content_types": content_types,
                "total_types": len(content_types)
            }
        )
        
        try:
            # Step 1: Get video information for V2 storage
            try:
                video_metadata = await self._get_video_metadata(video_id)
                self._current_video_title = video_metadata.get('title', 'Unknown')
                self._current_channel_id = video_metadata.get('channel_id', 'unknown')
            except Exception as e:
                self.log_with_context(
                    f"Failed to get video metadata: {str(e)}",
                    level="WARNING",
                    extra_context={"video_id": video_id}
                )
                self._current_video_title = "Unknown"
                self._current_channel_id = "unknown"
            
            # Step 2: Get or validate transcript
            try:
                transcript_data = await self._get_transcript_data(video_id, input_data)
            except TranscriptMissingError as e:
                self.log_with_context(
                    f"Transcript validation failed: {str(e)}",
                    level="ERROR",
                    extra_context={"video_id": video_id}
                )
                raise
            
            # Step 2: Parse requested content types and variants
            generation_plan = await self._create_generation_plan(
                content_types, generation_options
            )
            
            # Step 3: Create sub-generator jobs
            sub_jobs = await self._create_sub_generator_jobs(
                video_id, generation_plan, transcript_data
            )
            
            # Step 4: Track generation progress
            results = await self._track_generation_progress(
                video_id, 
                self._current_channel_id, 
                sub_jobs,
                transcript_data
            )
            
            # Step 5: Aggregate and store results
            final_results = await self._aggregate_results(
                video_id, results, generation_plan
            )
            
            # Step 6: Update database with generation metadata
            await self._store_generation_metadata(video_id, final_results)
            
            generation_status = self._determine_final_status(results)
            
            self.log_with_context(
                "Content generation orchestration completed",
                extra_context={
                    "video_id": video_id,
                    "status": generation_status.value,
                    "successful_types": len([r for r in results.values() if r["success"]]),
                    "total_types": len(results),
                    "execution_time": self.get_execution_time()
                }
            )
            
            return {
                "video_id": video_id,
                "status": generation_status.value,
                "generated_content": final_results,
                "generation_metadata": {
                    "total_content_types": len(content_types),
                    "successful_generations": len([r for r in results.values() if r["success"]]),
                    "failed_generations": len([r for r in results.values() if not r["success"]]),
                    "generation_plan": generation_plan,
                    "execution_time_seconds": self.get_execution_time(),
                    "timestamp": datetime.utcnow().isoformat()
                },
                "sub_job_results": results
            }
            
        except Exception as e:
            self.log_with_context(
                f"Content generation orchestration failed: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
            raise
    
    async def _get_transcript_data(
        self, 
        video_id: str, 
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get transcript data from input or database.
        
        Args:
            video_id: Video ID to get transcript for
            input_data: Input data that may contain transcript
            
        Returns:
            Dictionary with transcript data
            
        Raises:
            TranscriptMissingError: When no transcript is available
        """
        # Check if transcript provided in input
        transcript_text = input_data.get("transcript_text")
        if transcript_text:
            if not isinstance(transcript_text, str) or len(transcript_text.strip()) == 0:
                raise TranscriptMissingError(f"Invalid transcript text provided for video {video_id}")
            
            word_count = len(transcript_text.split())
            if word_count < 10:
                raise TranscriptMissingError(f"Transcript too short ({word_count} words) for meaningful content generation")
            
            self.log_with_context(
                "Using transcript from input data",
                extra_context={"video_id": video_id, "text_length": len(transcript_text), "word_count": word_count}
            )
            return {
                "video_id": video_id,
                "content_text": transcript_text,
                "word_count": word_count,
                "source": "input"
            }
        
        # Get transcript from database
        try:
            async with self.db_manager.get_session() as session:
                stmt = select(Transcript).where(Transcript.video_id == video_id)
                result = await session.execute(stmt)
                transcript = result.scalar_one_or_none()
                
                if not transcript:
                    raise TranscriptMissingError(f"No transcript found in database for video {video_id}")
                
                if not transcript.content_text or len(transcript.content_text.strip()) == 0:
                    raise TranscriptMissingError(f"Transcript exists but has no text content for video {video_id}")
                
                word_count = transcript.word_count or len(transcript.content_text.split())
                if word_count < 10:
                    raise TranscriptMissingError(f"Transcript too short ({word_count} words) for meaningful content generation")
                
                # Check transcript quality if available
                if transcript.quality_score is not None and transcript.quality_score < 0.3:
                    self.log_with_context(
                        f"Warning: Low quality transcript (score: {transcript.quality_score}) may affect generation quality",
                        level="WARNING",
                        extra_context={"video_id": video_id}
                    )
                
                self.log_with_context(
                    "Retrieved transcript from database",
                    extra_context={
                        "video_id": video_id,
                        "word_count": word_count,
                        "quality_score": transcript.quality_score,
                        "extraction_method": transcript.extraction_method
                    }
                )
                
                return {
                    "video_id": video_id,
                    "content_text": transcript.content_text,
                    "content_srt": transcript.content_srt,
                    "word_count": word_count,
                    "language": transcript.language,
                    "quality_score": transcript.quality_score,
                    "extraction_method": transcript.extraction_method,
                    "source": "database"
                }
                
        except TranscriptMissingError:
            raise
        except Exception as e:
            self.log_with_context(
                f"Database error retrieving transcript: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
            raise TranscriptMissingError(f"Failed to retrieve transcript for video {video_id}: {str(e)}")
    
    async def _create_generation_plan(
        self, 
        content_types: List[str], 
        generation_options: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a detailed generation plan with variants for each content type.
        
        Args:
            content_types: List of requested content types
            generation_options: Generation options and preferences
            
        Returns:
            Dictionary mapping content types to their generation configurations
        """
        generation_plan = {}
        
        for content_type in content_types:
            config = self.content_type_configs[content_type]
            
            # Get requested variants or use defaults
            type_options = generation_options.get(content_type, {})
            requested_variants = type_options.get("variants", config["default_variants"])
            
            # Ensure requested variants are supported
            supported_variants = config["variants"]
            valid_variants = [v for v in requested_variants if v in supported_variants]
            
            if not valid_variants:
                self.log_with_context(
                    f"No valid variants for {content_type}, using defaults",
                    level="WARNING",
                    extra_context={
                        "requested": requested_variants,
                        "supported": supported_variants
                    }
                )
                valid_variants = config["default_variants"]
            
            generation_plan[content_type] = {
                "variants": valid_variants,
                "estimated_duration": config["estimated_duration"],
                "priority": type_options.get("priority", 5),
                "custom_options": type_options.get("custom_options", {}),
                "sub_jobs": []  # Will be populated when jobs are created
            }
        
        self.log_with_context(
            "Created generation plan",
            extra_context={
                "total_content_types": len(generation_plan),
                "total_variants": sum(len(plan["variants"]) for plan in generation_plan.values()),
                "plan_summary": {ct: len(plan["variants"]) for ct, plan in generation_plan.items()}
            }
        )
        
        return generation_plan
    
    async def _create_sub_generator_jobs(
        self, 
        video_id: str, 
        generation_plan: Dict[str, Dict[str, Any]], 
        transcript_data: Dict[str, Any]
    ) -> Dict[str, List[int]]:
        """
        Create sub-generator jobs for each content type and variant.
        
        Args:
            video_id: Video ID being processed
            generation_plan: Generation plan with content types and variants
            transcript_data: Transcript data for generation
            
        Returns:
            Dictionary mapping content types to lists of job IDs
        """
        sub_jobs = {}
        
        for content_type, plan in generation_plan.items():
            job_ids = []
            
            for variant in plan["variants"]:
                try:
                    # For Phase 1: Create placeholder jobs that will be processed by mock sub-generators
                    job_id = await self.job_queue.enqueue(
                        job_type=f"generate_{content_type}",
                        target_id=f"{video_id}:{variant}",
                        priority=plan["priority"],
                        max_retries=2
                        # In Phase 2, metadata would include:
                        # metadata={
                        #     "video_id": video_id,
                        #     "content_type": content_type,
                        #     "variant": variant,
                        #     "transcript_data": transcript_data,
                        #     "custom_options": plan["custom_options"]
                        # }
                    )
                    
                    job_ids.append(job_id)
                    
                    self.log_with_context(
                        f"Created sub-generator job {job_id}",
                        extra_context={
                            "video_id": video_id,
                            "content_type": content_type,
                            "variant": variant,
                            "priority": plan["priority"]
                        }
                    )
                    
                except Exception as e:
                    self.log_with_context(
                        f"Failed to create sub-generator job for {content_type}:{variant}: {str(e)}",
                        level="ERROR"
                    )
                    continue
            
            sub_jobs[content_type] = job_ids
            generation_plan[content_type]["sub_jobs"] = job_ids
        
        total_jobs = sum(len(jobs) for jobs in sub_jobs.values())
        self.log_with_context(
            f"Created {total_jobs} sub-generator jobs",
            extra_context={
                "video_id": video_id,
                "jobs_per_type": {ct: len(jobs) for ct, jobs in sub_jobs.items()}
            }
        )
        
        return sub_jobs
    
    async def _track_generation_progress(
        self, 
        video_id: str,
        channel_id: str,
        sub_jobs: Dict[str, List[int]],
        transcript_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Track progress of all sub-generator jobs until completion.
        
        Args:
            sub_jobs: Dictionary mapping content types to job IDs
            
        Returns:
            Dictionary with results for each content type
        """
        results = {}
        all_job_ids = []
        job_to_content_type = {}
        
        # Build mapping of job IDs to content types
        for content_type, job_ids in sub_jobs.items():
            all_job_ids.extend(job_ids)
            for job_id in job_ids:
                job_to_content_type[job_id] = content_type
        
        if not all_job_ids:
            self.log_with_context("No sub-generator jobs to track", level="WARNING")
            return results
        
        # Initialize results structure
        for content_type in sub_jobs.keys():
            results[content_type] = {
                "success": False,
                "completed_jobs": 0,
                "total_jobs": len(sub_jobs[content_type]),
                "generated_content": [],
                "errors": [],
                "execution_time": None
            }
        
        # Track job completion (Phase 1: Simulate tracking)
        start_time = time.time()
        timeout_seconds = self.generation_timeout_minutes * 60
        
        self.log_with_context(
            f"Tracking {len(all_job_ids)} sub-generator jobs",
            extra_context={
                "timeout_minutes": self.generation_timeout_minutes,
                "content_types": list(sub_jobs.keys())
            }
        )
        
        # Process generation jobs with actual AI if enabled, else simulate
        if self.settings.ai_backend != 'disabled':
            await self._process_generation_jobs(video_id, channel_id, all_job_ids, results, job_to_content_type, transcript_data)
        else:
            # Fallback to simulation if AI is disabled
            await self._simulate_job_progress(video_id, channel_id, all_job_ids, results, job_to_content_type)
        
        # Calculate final results
        for content_type, result in results.items():
            success_rate = result["completed_jobs"] / result["total_jobs"] if result["total_jobs"] > 0 else 0
            result["success"] = success_rate >= 0.5  # Consider successful if >=50% jobs completed
            result["success_rate"] = success_rate
            result["execution_time"] = time.time() - start_time
        
        self.log_with_context(
            "Generation progress tracking completed",
            extra_context={
                "successful_types": len([r for r in results.values() if r["success"]]),
                "total_types": len(results),
                "total_execution_time": time.time() - start_time
            }
        )
        
        return results
    
    async def _process_generation_jobs(
        self,
        video_id: str,
        channel_id: str,
        job_ids: List[int],
        results: Dict[str, Dict[str, Any]],
        job_to_content_type: Dict[int, str],
        transcript_data: Dict[str, Any]
    ) -> None:
        """
        Process generation jobs with actual AI calls and enhanced error handling.
        
        Args:
            video_id: Video ID being processed
            channel_id: Channel ID
            job_ids: List of job IDs to process
            results: Results dictionary to update
            job_to_content_type: Mapping of job IDs to content types
            transcript_data: Transcript data for generation
        """
        transcript_text = transcript_data.get('content_text', '')
        
        for job_id in job_ids:
            content_type = job_to_content_type[job_id]
            
            try:
                # Prepare generation prompt using PromptManager
                try:
                    prompt, prompt_metadata = await self._prepare_generation_prompt(content_type, transcript_text, transcript_data)
                except Exception as e:
                    raise PromptTemplateError(f"Failed to prepare generation prompt for {content_type}: {str(e)}")
                
                # Generate content using AI backend
                self.log_with_context(
                    f"Generating {content_type} content with AI",
                    extra_context={
                        "job_id": job_id,
                        "ai_backend": self.settings.ai_backend,
                        "prompt_id": prompt_metadata.get('prompt_id'),
                        "prompt_version": prompt_metadata.get('version')
                    }
                )
                
                # Track start time for analytics
                start_time = time.time()
                
                # Get video metadata from database
                video_metadata = await self._get_video_metadata(video_id)
                
                generated_content = None
                generation_source = None
                
                # Try to use sub-generator first
                try:
                    generator_result = await self._generate_with_sub_generator(
                        content_type=content_type,
                        transcript=transcript_data.get("content_text", ""),
                        metadata={
                            'title': video_metadata.get('title', ''),
                            'channel_name': video_metadata.get('channel_name', ''),
                            'video_id': video_id,
                            'duration': video_metadata.get('duration_seconds', 0)
                        }
                    )
                    
                    if generator_result:
                        generated_content = generator_result.get('content', '')
                        generation_source = "sub_generator"
                        self.log_with_context(
                            f"Generated {content_type} using sub-generator",
                            extra_context={"job_id": job_id, "generator": generator_result.get('generator')}
                        )
                        
                except SubGeneratorError as e:
                    self.log_with_context(
                        f"Sub-generator failed for {content_type}, falling back to AI: {str(e)}",
                        level="WARNING",
                        extra_context={"job_id": job_id}
                    )
                
                # Fallback to direct AI generation if sub-generator failed
                if not generated_content:
                    try:
                        generated_content = await self._generate_with_ai(prompt, content_type)
                        generation_source = "ai_backend"
                        self.log_with_context(
                            f"Generated {content_type} using AI backend",
                            extra_context={"job_id": job_id, "backend": self.settings.ai_backend}
                        )
                    except AIModelError as e:
                        self.log_with_context(
                            f"AI generation failed for {content_type}: {str(e)}",
                            level="ERROR",
                            extra_context={"job_id": job_id}
                        )
                        raise
                
                execution_time = time.time() - start_time
                
                # Validate generated content
                if not generated_content:
                    raise AIModelError("Generated content is empty")
                
                try:
                    self._validate_generated_content(generated_content, content_type)
                except ContentValidationError as e:
                    self.log_with_context(
                        f"Generated content failed validation: {str(e)}",
                        level="ERROR",
                        extra_context={"job_id": job_id, "content_type": content_type}
                    )
                    raise
                
                # Save generated content to file with readable name
                try:
                    # Get video title for readable filename
                    video_title = "Unknown"
                    if 'video_title' in locals():
                        video_title = locals()['video_title'] 
                    elif hasattr(self, '_current_video_title'):
                        video_title = self._current_video_title
                    
                    content_path = self.storage_paths.get_content_file(
                        channel_id=channel_id,
                        video_id=video_id,
                        content_type=content_type,
                        job_id=job_id,
                        video_title=video_title,
                        format="txt"
                    )
                    content_path.parent.mkdir(parents=True, exist_ok=True)
                    content_path.write_text(generated_content, encoding='utf-8')
                except Exception as e:
                    self.log_with_context(
                        f"Failed to save generated content to file: {str(e)}",
                        level="ERROR",
                        extra_context={"job_id": job_id, "path": str(content_path)}
                    )
                    # Continue without failing - content is still in memory
                    content_path = None
                
                # Track analytics if part of experiment
                if prompt_metadata.get('is_experiment'):
                    try:
                        ab_test_manager = get_ab_test_manager()
                        await ab_test_manager.record_result(
                            experiment_id=prompt_metadata.get('experiment_id'),
                            variant_id=prompt_metadata.get('variant_id'),
                            metrics={
                                'quality_score': None,  # Will be set by quality check
                                'tokens_used': len(prompt.split()) + len(generated_content.split()),
                                'execution_time': execution_time,
                                'model_used': self.settings.ai_model,
                                'error_occurred': False
                            }
                        )
                    except Exception as e:
                        self.log_with_context(
                            f"Failed to record A/B test result: {str(e)}",
                            level="WARNING",
                            extra_context={"job_id": job_id}
                        )
                
                # Update results
                results[content_type]["completed_jobs"] += 1
                results[content_type]["generated_content"].append({
                    "job_id": job_id,
                    "content_type": content_type,
                    "content": generated_content,
                    "prompt_metadata": prompt_metadata,
                    "metadata": {
                        "word_count": len(generated_content.split()),
                        "generation_time": execution_time,
                        "model": self.settings.ai_model,
                        "backend": self.settings.ai_backend,
                        "generation_source": generation_source
                    },
                    "storage_path": str(content_path) if content_path else None,
                    "created_at": datetime.utcnow().isoformat()
                })
                
                self.log_with_context(
                    f"Successfully generated {content_type} content",
                    extra_context={
                        "job_id": job_id, 
                        "word_count": len(generated_content.split()),
                        "generation_source": generation_source,
                        "execution_time": execution_time
                    }
                )
                
            except (PromptTemplateError, AIModelError, ContentValidationError, SubGeneratorError) as e:
                # Handle specific generation errors
                error_type = type(e).__name__
                results[content_type]["errors"].append({
                    "job_id": job_id,
                    "error": str(e),
                    "error_type": error_type,
                    "content_type": content_type,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self.log_with_context(
                    f"Generation failed for {content_type} with {error_type}: {str(e)}",
                    level="ERROR",
                    extra_context={"job_id": job_id, "content_type": content_type, "error_type": error_type}
                )
                
            except Exception as e:
                # Handle unexpected errors
                error_type = type(e).__name__
                results[content_type]["errors"].append({
                    "job_id": job_id,
                    "error": f"Unexpected generation error: {str(e)}",
                    "error_type": error_type,
                    "content_type": content_type,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self.log_with_context(
                    f"Unexpected generation error for {content_type}: {str(e)}",
                    level="ERROR",
                    extra_context={"job_id": job_id, "content_type": content_type, "error_type": error_type}
                )
    
    async def _get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Get video metadata from database.
        
        Args:
            video_id: Video ID to get metadata for
            
        Returns:
            Dictionary with video metadata
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = select(Video).where(Video.video_id == video_id)
                result = await session.execute(stmt)
                video = result.scalar_one_or_none()
                
                if video:
                    return {
                        'title': video.title or '',
                        'channel_id': video.channel_id,
                        'channel_name': video.channel_id,  # Will be replaced with actual channel name in Phase 2
                        'duration_seconds': video.duration or 0,
                        'view_count': video.view_count or 0,
                        'published_date': video.published_at.isoformat() if video.published_at else None
                    }
                else:
                    return {
                        'title': '',
                        'channel_name': '',
                        'duration_seconds': 0,
                        'view_count': 0,
                        'published_date': None
                    }
        except Exception as e:
            self.log_with_context(
                f"Error getting video metadata: {str(e)}",
                level="WARNING",
                extra_context={"video_id": video_id}
            )
            return {
                'title': '',
                'channel_name': '',
                'duration_seconds': 0,
                'view_count': 0,
                'published_date': None
            }
    
    async def _prepare_generation_prompt(self, content_type: str, transcript: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare generation prompt using the PromptManager.
        
        Args:
            content_type: Type of content to generate
            transcript: Video transcript text
            metadata: Additional metadata
            
        Returns:
            Tuple of (formatted prompt, prompt metadata)
        """
        # Initialize managers
        prompt_manager = get_prompt_manager()
        template_engine = get_template_engine()
        ab_test_manager = get_ab_test_manager()
        
        # Prepare variables for template
        variables = {
            'title': metadata.get('title', 'Video'),
            'transcript': transcript,
            'channel_name': metadata.get('channel_name'),
            'duration': metadata.get('duration'),
            'view_count': metadata.get('view_count'),
            'published_date': metadata.get('published_date'),
            'word_limit': metadata.get('word_limit', 500),
            'tone': metadata.get('tone', 'professional')
        }
        
        # Get prompt from manager (handles A/B testing automatically)
        prompt_template, prompt_metadata = await prompt_manager.get_prompt(
            content_type=content_type,
            experiment_aware=True,
            variables=variables
        )
        
        # Track if this is part of an experiment
        if prompt_metadata.get('is_experiment'):
            self.log_with_context(
                f"Using experimental prompt variant {prompt_metadata.get('variant_id')} for {content_type}",
                extra_context=prompt_metadata
            )
        
        return prompt_template, prompt_metadata
    
    async def _generate_with_sub_generator(
        self, 
        content_type: str, 
        transcript: str, 
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate content using appropriate sub-generator with enhanced error handling.
        
        Args:
            content_type: Type of content to generate
            transcript: Input transcript text
            metadata: Video metadata
            
        Returns:
            Generated content dictionary
            
        Raises:
            SubGeneratorError: When sub-generator fails to load or execute
        """
        try:
            # Import sub-generators with specific error handling
            try:
                from workers.generators.blog import BlogGenerator
                from workers.generators.social import SocialMediaGenerator
                from workers.generators.newsletter import NewsletterGenerator
                from workers.generators.summary import SummaryGenerator
                from workers.generators.scripts import ScriptGenerator
            except ImportError as e:
                raise SubGeneratorError(f"Failed to import sub-generators: {str(e)}")
            
            # Map content types to generators
            generator_map = {
                'blog': BlogGenerator,
                'blog_post': BlogGenerator,
                'social': SocialMediaGenerator,
                'social_media': SocialMediaGenerator,
                'newsletter': NewsletterGenerator,
                'summary': SummaryGenerator,
                'script': ScriptGenerator,
                'scripts': ScriptGenerator,
            }
            
            # Get the appropriate generator class
            generator_class = generator_map.get(content_type.lower())
            if not generator_class:
                raise SubGeneratorError(f"No generator found for content type: {content_type}")
            
            # Initialize the generator
            try:
                generator = generator_class()
            except Exception as e:
                raise SubGeneratorError(f"Failed to initialize {generator_class.__name__}: {str(e)}")
            
            # Validate transcript input
            if not transcript or len(transcript.strip()) < 10:
                raise SubGeneratorError(f"Transcript too short for {content_type} generation")
            
            # Generate content based on type with error handling
            try:
                if content_type.lower() in ['blog', 'blog_post']:
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.MARKDOWN,
                        length="medium",
                        style="conversational",
                        title=metadata.get('title', 'Blog Post')
                    )
                elif content_type.lower() in ['social', 'social_media']:
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.PLAIN_TEXT,
                        platforms=['twitter', 'linkedin'],
                        video_title=metadata.get('title', ''),
                        channel_name=metadata.get('channel_name', '')
                    )
                elif content_type.lower() == 'newsletter':
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.HTML,
                        style='informative'
                    )
                elif content_type.lower() == 'summary':
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.PLAIN_TEXT,
                        variant='medium'
                    )
                elif content_type.lower() in ['script', 'scripts']:
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.PLAIN_TEXT,
                        script_type='youtube_shorts'
                    )
                else:
                    content = generator.generate_content(
                        transcript=transcript,
                        format_type=ContentFormat.PLAIN_TEXT
                    )
            except Exception as e:
                raise SubGeneratorError(f"Content generation failed in {generator_class.__name__}: {str(e)}")
            
            # Validate generated content
            if not content or len(content.strip()) == 0:
                raise SubGeneratorError(f"Generator {generator_class.__name__} returned empty content")
            
            self.log_with_context(
                f"Successfully generated {content_type} using {generator_class.__name__}",
                extra_context={
                    "generator": generator_class.__name__,
                    "content_length": len(content),
                    "content_type": content_type
                }
            )
            
            return {
                'content': content,
                'content_type': content_type,
                'generator': generator_class.__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except SubGeneratorError:
            raise
        except Exception as e:
            raise SubGeneratorError(f"Unexpected error in sub-generator for {content_type}: {str(e)}")
    
    async def _generate_with_ai(self, prompt: str, content_type: str) -> str:
        """
        Generate content using AI backend (fallback method) with enhanced error handling.
        
        Args:
            prompt: Generation prompt
            content_type: Type of content being generated
            
        Returns:
            Generated content
            
        Raises:
            AIModelError: When AI generation fails
        """
        if not prompt or len(prompt.strip()) == 0:
            raise AIModelError("Cannot generate content with empty prompt")
        
        try:
            if self.settings.ai_backend == 'claude_cli':
                # Use Claude CLI
                import subprocess
                
                # Validate Claude CLI is available
                try:
                    subprocess.run(['claude', '--version'], capture_output=True, check=True, timeout=10)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    raise AIModelError("Claude CLI is not available or not working")
                
                cmd = [
                    'claude',
                    prompt,
                    '--model', self.settings.ai_model,
                    '--max-tokens', str(self.settings.ai_max_tokens)
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120  # Increased timeout for content generation
                    )
                except subprocess.TimeoutExpired:
                    raise AIModelError(f"Claude CLI timed out for {content_type} generation")
                
                if result.returncode == 0:
                    generated_content = result.stdout.strip()
                    if not generated_content:
                        raise AIModelError("Claude CLI returned empty response")
                    return generated_content
                else:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    raise AIModelError(f"Claude CLI failed for {content_type}: {error_msg}")
                    
            elif self.settings.ai_backend in ['claude_api', 'openai_api', 'gemini_api']:
                # For Phase 2: Use API backends
                # This would call the respective API clients
                raise AIModelError(f"API backend {self.settings.ai_backend} not yet implemented in Phase 1")
            elif self.settings.ai_backend == 'disabled':
                raise AIModelError("AI backend is disabled")
            else:
                raise AIModelError(f"Unknown AI backend: {self.settings.ai_backend}")
                
        except AIModelError:
            raise
        except Exception as e:
            self.log_with_context(
                f"Unexpected AI generation error: {str(e)}",
                level="ERROR",
                extra_context={"backend": self.settings.ai_backend, "content_type": content_type}
            )
            raise AIModelError(f"Unexpected error during AI generation for {content_type}: {str(e)}")
    
    async def _simulate_job_progress(
        self, 
        video_id: str,
        channel_id: str,
        job_ids: List[int], 
        results: Dict[str, Dict[str, Any]], 
        job_to_content_type: Dict[int, str]
    ) -> None:
        """
        Simulate job progress for Phase 1 implementation.
        
        In Phase 2, this will be replaced with actual job queue monitoring.
        
        Args:
            job_ids: List of job IDs to simulate
            results: Results dictionary to update
            job_to_content_type: Mapping of job IDs to content types
        """
        # Simulate job processing with some randomness
        import random
        
        for job_id in job_ids:
            content_type = job_to_content_type[job_id]
            
            # Simulate processing time
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Simulate success/failure (90% success rate for simulation)
            if random.random() < 0.9:
                # Simulate successful generation
                results[content_type]["completed_jobs"] += 1
                results[content_type]["generated_content"].append({
                    "job_id": job_id,
                    "content_type": content_type,
                    "variant": "simulated_variant",
                    "content": f"[PHASE 1 SIMULATION] Generated {content_type} content for job {job_id}",
                    "metadata": {
                        "word_count": random.randint(50, 500),
                        "generation_time": random.uniform(30, 120),
                        "model": "simulation",
                        "quality_score": random.uniform(0.7, 1.0)
                    },
                    "storage_path": str(self.storage_paths.get_content_file(
                        channel_id=channel_id,
                        video_id=video_id,
                        content_type=content_type,
                        job_id=job_id,
                        video_title=getattr(self, '_current_video_title', 'Unknown'),
                        format="txt"
                    )),
                    "created_at": datetime.utcnow().isoformat()
                })
                
                self.log_with_context(
                    f"Simulated successful generation for job {job_id}",
                    extra_context={"content_type": content_type}
                )
            else:
                # Simulate failure
                results[content_type]["errors"].append({
                    "job_id": job_id,
                    "error": f"Simulated generation failure for job {job_id}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self.log_with_context(
                    f"Simulated generation failure for job {job_id}",
                    level="WARNING",
                    extra_context={"content_type": content_type}
                )
    
    async def _aggregate_results(
        self, 
        video_id: str, 
        results: Dict[str, Dict[str, Any]], 
        generation_plan: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from all sub-generators.
        
        Args:
            video_id: Video ID being processed
            results: Results from sub-generator tracking
            generation_plan: Original generation plan
            
        Returns:
            Aggregated results dictionary
        """
        aggregated = {
            "video_id": video_id,
            "content_by_type": {},
            "generation_summary": {
                "total_content_types": len(results),
                "successful_types": 0,
                "failed_types": 0,
                "partial_types": 0,
                "total_generated_items": 0
            },
            "storage_paths": [],
            "errors": []
        }
        
        for content_type, result in results.items():
            type_summary = {
                "content_type": content_type,
                "success": result["success"],
                "success_rate": result.get("success_rate", 0.0),
                "total_variants": result["total_jobs"],
                "completed_variants": result["completed_jobs"],
                "generated_content": result["generated_content"],
                "errors": result["errors"],
                "execution_time": result.get("execution_time"),
                "plan": generation_plan.get(content_type, {})
            }
            
            aggregated["content_by_type"][content_type] = type_summary
            
            # Update summary counts
            if result["success"]:
                aggregated["generation_summary"]["successful_types"] += 1
            elif result["completed_jobs"] > 0:
                aggregated["generation_summary"]["partial_types"] += 1
            else:
                aggregated["generation_summary"]["failed_types"] += 1
            
            aggregated["generation_summary"]["total_generated_items"] += len(result["generated_content"])
            
            # Collect storage paths
            for content in result["generated_content"]:
                if content.get("storage_path"):
                    aggregated["storage_paths"].append(content["storage_path"])
            
            # Collect errors
            aggregated["errors"].extend(result["errors"])
        
        self.log_with_context(
            "Aggregated generation results",
            extra_context={
                "video_id": video_id,
                "successful_types": aggregated["generation_summary"]["successful_types"],
                "total_items": aggregated["generation_summary"]["total_generated_items"],
                "total_errors": len(aggregated["errors"])
            }
        )
        
        return aggregated
    
    async def _store_generation_metadata(
        self, 
        video_id: str, 
        final_results: Dict[str, Any]
    ) -> None:
        """
        Store generation results and metadata in the database.
        
        Args:
            video_id: Video ID being processed
            final_results: Aggregated generation results
        """
        try:
            async with self.db_manager.get_session() as session:
                # Store each generated content item
                for content_type, type_data in final_results["content_by_type"].items():
                    for content_item in type_data["generated_content"]:
                        generated_content = GeneratedContent(
                            video_id=video_id,
                            content_type=content_type,
                            content=content_item["content"],
                            content_metadata=content_item.get("metadata", {}),
                            quality_score=content_item.get("metadata", {}).get("quality_score"),
                            generation_model=content_item.get("metadata", {}).get("model"),
                            prompt_template="phase1_simulation",  # Placeholder for Phase 1
                            storage_path=content_item.get("storage_path"),
                            created_at=datetime.utcnow()
                        )
                        
                        session.add(generated_content)
                
                await session.flush()
                
                self.log_with_context(
                    "Stored generation metadata in database",
                    extra_context={
                        "video_id": video_id,
                        "total_items": final_results["generation_summary"]["total_generated_items"]
                    }
                )
                
        except Exception as e:
            self.log_with_context(
                f"Error storing generation metadata: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
            # Don't raise here - metadata storage failure shouldn't fail the whole operation
    
    def _determine_final_status(self, results: Dict[str, Dict[str, Any]]) -> GenerationStatus:
        """
        Determine the final generation status based on sub-generator results.
        
        Args:
            results: Results from sub-generator tracking
            
        Returns:
            Final generation status
        """
        if not results:
            return GenerationStatus.FAILED
        
        successful_count = len([r for r in results.values() if r["success"]])
        partial_count = len([r for r in results.values() if not r["success"] and r["completed_jobs"] > 0])
        total_count = len(results)
        
        if successful_count == total_count:
            return GenerationStatus.COMPLETED
        elif successful_count > 0 or partial_count > 0:
            return GenerationStatus.PARTIAL
        else:
            return GenerationStatus.FAILED
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle execution errors with detailed context.
        
        Args:
            error: Exception that occurred during execution
            
        Returns:
            Error handling result with context and recovery information
        """
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "worker_name": self.name,
            "execution_time": self.get_execution_time(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Categorize error types for better handling
        if isinstance(error, ValueError):
            error_context["category"] = "input_validation"
            error_context["recoverable"] = True
            error_context["suggested_action"] = "Verify input data format and content"
        elif isinstance(error, asyncio.TimeoutError):
            error_context["category"] = "timeout"
            error_context["recoverable"] = True
            error_context["suggested_action"] = "Increase timeout or reduce generation complexity"
        elif "transcript" in str(error).lower():
            error_context["category"] = "transcript_missing"
            error_context["recoverable"] = False
            error_context["suggested_action"] = "Ensure transcript is available before generation"
        else:
            error_context["category"] = "unknown"
            error_context["recoverable"] = True
            error_context["suggested_action"] = "Check logs and retry with same parameters"
        
        self.log_with_context(
            "Error handling completed",
            level="INFO",
            extra_context=error_context
        )
        
        return error_context
    
    def _validate_generated_content(self, content: str, content_type: str) -> None:
        """
        Validate generated content meets basic quality requirements.
        
        Args:
            content: Generated content to validate
            content_type: Type of content being validated
            
        Raises:
            ContentValidationError: When content fails validation
        """
        if not content or not isinstance(content, str):
            raise ContentValidationError("Content must be a non-empty string")
        
        content = content.strip()
        if len(content) == 0:
            raise ContentValidationError("Content cannot be empty or whitespace only")
        
        word_count = len(content.split())
        
        # Content type specific validation
        if content_type.lower() in ['blog', 'blog_post']:
            if word_count < 50:
                raise ContentTooShortError(f"Blog content too short: {word_count} words (minimum 50)")
            if word_count > 3000:
                self.log_with_context(
                    f"Blog content very long: {word_count} words",
                    level="WARNING",
                    extra_context={"content_type": content_type}
                )
        elif content_type.lower() in ['social', 'social_media']:
            if word_count < 5:
                raise ContentTooShortError(f"Social media content too short: {word_count} words (minimum 5)")
            # Check for common social media issues
            if len(content) > 1000:  # Roughly Twitter's expanded limit
                self.log_with_context(
                    f"Social media content may be too long: {len(content)} characters",
                    level="WARNING",
                    extra_context={"content_type": content_type}
                )
        elif content_type.lower() == 'summary':
            if word_count < 10:
                raise ContentTooShortError(f"Summary too short: {word_count} words (minimum 10)")
            if word_count > 500:
                self.log_with_context(
                    f"Summary quite long: {word_count} words",
                    level="WARNING",
                    extra_context={"content_type": content_type}
                )
        elif content_type.lower() == 'newsletter':
            if word_count < 30:
                raise ContentTooShortError(f"Newsletter content too short: {word_count} words (minimum 30)")
        elif content_type.lower() in ['script', 'scripts']:
            if word_count < 20:
                raise ContentTooShortError(f"Script too short: {word_count} words (minimum 20)")
        else:
            # Generic validation for unknown content types
            if word_count < 5:
                raise ContentTooShortError(f"{content_type} content too short: {word_count} words (minimum 5)")
        
        # Check for common generation issues
        lower_content = content.lower()
        
        # Check for incomplete generation indicators
        incomplete_indicators = [
            "[truncated]", "[incomplete]", "...", "[continue]",
            "i cannot", "i can't", "as an ai", "i'm sorry",
            "placeholder", "[todo]", "coming soon"
        ]
        
        for indicator in incomplete_indicators:
            if indicator in lower_content:
                raise ContentQualityError(f"Generated content appears incomplete or contains: '{indicator}'")
        
        # Check for excessive repetition (simple check)
        words = content.split()
        if len(words) > 20:  # Only check for longer content
            word_freq = {}
            for word in words:
                word = word.lower().strip('.,!?;:')
                if len(word) > 3:  # Only count meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Flag if any word appears more than 10% of total words
            max_occurrences = max(word_freq.values()) if word_freq else 0
            if max_occurrences > len(words) * 0.1:
                self.log_with_context(
                    f"Generated content has excessive word repetition",
                    level="WARNING",
                    extra_context={"content_type": content_type, "max_word_occurrences": max_occurrences}
                )
        
        self.log_with_context(
            f"Content validation passed for {content_type}",
            extra_context={
                "content_type": content_type,
                "word_count": word_count,
                "char_count": len(content)
            }
        )


# Convenience function for external usage
async def generate_content(
    video_id: str,
    content_types: List[str],
    transcript_text: Optional[str] = None,
    generation_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate content for a video.
    
    Args:
        video_id: Video ID to generate content for
        content_types: List of content types to generate
        transcript_text: Optional transcript text (will fetch from DB if not provided)
        generation_options: Optional generation preferences
        
    Returns:
        Generation results dictionary
        
    Raises:
        Exception: On generation failure
    """
    worker = GeneratorWorker()
    
    input_data = {
        "video_id": video_id,
        "content_types": content_types,
        "generation_options": generation_options or {}
    }
    
    if transcript_text:
        input_data["transcript_text"] = transcript_text
    
    result = worker.run(input_data)
    
    if result["status"] == WorkerStatus.SUCCESS.value:
        return result["data"]
    else:
        raise Exception(f"Content generation failed: {result.get('error', 'Unknown error')}")