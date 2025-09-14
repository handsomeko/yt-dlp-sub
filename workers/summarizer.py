"""
AI-powered video summary generation worker.

Generates concise summaries of video transcripts to enable topic filtering
and provide preview content for the review checkpoint system. Summary length
is based on video duration:
- Shorts (< 1 min): 1 sentence
- Medium videos (1-10 min): 3 sentences  
- Long videos (> 10 min): 5 sentences

This step is optional to control AI costs while maintaining functionality.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from config.settings import get_settings
from core.database import db_manager, Video
from core.ai_backend import AIBackend
from core.queue import JobQueue, JobType, JobStatus
from workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SummarizerWorker(BaseWorker):
    """Worker for generating AI summaries of video transcripts."""
    
    def __init__(self):
        super().__init__(name="summarizer")
        self.settings = get_settings()
        self.ai_backend = AIBackend()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for summary generation."""
        return 'video_id' in input_data or 'target_id' in input_data
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summary generation synchronously."""
        # Use asyncio to run the async method
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.process_job(input_data))
            return {
                "status": "success" if result else "failed",
                "result": result
            }
        finally:
            loop.close()
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors during summary generation."""
        logger.error(f"Summary generation error: {error}")
        return {
            "status": "failed",
            "error": str(error)
        }
        
    async def process_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Process a single summary generation job.
        
        Args:
            job_data: Job data containing video_id and options
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            video_id = job_data.get('video_id')
            if not video_id:
                logger.error("No video_id provided in job data")
                return False
                
            logger.info(f"Starting summary generation for video {video_id}")
            
            async with db_manager.get_session() as session:
                result = await session.execute(select(Video).where(Video.video_id == video_id))
                video = result.scalar_one_or_none()
                if not video:
                    logger.error(f"Video {video_id} not found")
                    return False
                
                # Check if summary already exists
                if video.ai_summary:
                    logger.info(f"Summary already exists for video {video_id}")
                    return True
                
                # Verify transcript exists
                if not video.transcript_text:
                    logger.error(f"No transcript available for video {video_id}")
                    return False
                
                # Generate summary based on duration
                summary = self._generate_summary(video)
                if not summary:
                    logger.error(f"Failed to generate summary for video {video_id}")
                    return False
                
                # Extract topics from summary
                topics = self._extract_topics(summary)
                
                # Update video record
                video.ai_summary = summary
                video.extracted_topics = ','.join(topics) if topics else None
                video.summary_generated_at = datetime.now(timezone.utc)
                video.summary_cost = self._calculate_cost(video.transcript_text, summary)
                
                await session.commit()
                logger.info(f"Successfully generated summary for video {video_id}")
                
                # Create generation jobs if auto-approved
                if self.settings.auto_approve_generation:
                    await self._create_generation_jobs(video_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return False
    
    def _generate_summary(self, video: Video) -> Optional[str]:
        """
        Generate AI summary based on video duration.
        
        Args:
            video: Video database record
            
        Returns:
            Generated summary text or None if failed
        """
        try:
            # Determine summary length based on duration
            duration = video.duration_seconds or 0
            if duration < 60:  # Shorts
                sentence_count = 1
                max_tokens = 50
            elif duration < 600:  # Medium (< 10 min)
                sentence_count = 3
                max_tokens = 150
            else:  # Long videos
                sentence_count = 5
                max_tokens = 250
            
            # Create prompt for summary generation
            prompt = f"""Please create a concise summary of this video transcript in exactly {sentence_count} sentence(s).
Focus on the main topics, key points, and actionable insights.

Title: {video.title}
Duration: {duration // 60}:{duration % 60:02d}
Channel: {video.channel_name}

Transcript:
{video.transcript_text[:4000]}  # Limit to avoid token overflow

Summary ({sentence_count} sentence(s)):"""

            # Generate summary using AI backend
            response = self.ai_backend.generate_content(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for consistent summaries
            )
            
            if response and response.get('content'):
                summary = response['content'].strip()
                logger.info(f"Generated {sentence_count}-sentence summary ({len(summary)} chars)")
                return summary
            else:
                logger.error("AI backend returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def _extract_topics(self, summary: str) -> List[str]:
        """
        Extract key topics from the summary.
        
        Args:
            summary: Generated summary text
            
        Returns:
            List of extracted topics
        """
        try:
            # Use AI to extract topics from summary
            prompt = f"""Extract 3-5 key topics from this video summary. Return only the topics, one per line, no bullets or numbers.
Focus on the main subjects, themes, or categories discussed.

Summary: {summary}

Topics:"""
            
            response = self.ai_backend.generate_content(
                prompt=prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            if response and response.get('content'):
                topics = [
                    topic.strip() 
                    for topic in response['content'].split('\n') 
                    if topic.strip() and not topic.strip().startswith(('•', '-', '*'))
                ]
                return topics[:5]  # Limit to 5 topics max
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _calculate_cost(self, transcript: str, summary: str) -> float:
        """
        Calculate estimated cost for summary generation.
        
        Args:
            transcript: Input transcript text
            summary: Generated summary text
            
        Returns:
            Estimated cost in dollars
        """
        try:
            # Rough token estimation (1 token ≈ 4 characters)
            input_tokens = len(transcript) / 4
            output_tokens = len(summary) / 4
            
            # Haiku pricing: $0.25/1M input, $1.25/1M output tokens
            input_cost = (input_tokens / 1_000_000) * 0.25
            output_cost = (output_tokens / 1_000_000) * 1.25
            
            return round(input_cost + output_cost, 6)
            
        except Exception:
            return 0.0
    
    async def _create_generation_jobs(self, video_id: str) -> None:
        """
        Create content generation jobs for auto-approved videos.
        
        Args:
            video_id: Video ID to create jobs for
        """
        try:
            async with db_manager.get_session() as session:
                result = await session.execute(select(Video).where(Video.video_id == video_id))
                video = result.scalar_one_or_none()
                if not video:
                    return
                
                # Mark as approved (if these fields exist)
                if hasattr(video, 'generation_review_status'):
                    video.generation_review_status = 'approved'
                    video.generation_approved_at = datetime.now(timezone.utc)
                    video.generation_review_notes = 'Auto-approved via configuration'
                await session.commit()
            
            # Create generation jobs
            job_queue = JobQueue()
            
            # Create jobs for each enabled generator
            generators = self.settings.content_generators
            if isinstance(generators, str):
                generators = [g.strip() for g in generators.split(',')]
            
            for generator in generators:
                job_queue.add_job(
                    job_type=JobType.GENERATE,
                    data={
                        'video_id': video_id,
                        'generator_type': generator
                    },
                    priority=3  # Lower priority than download/transcript
                )
            
            logger.info(f"Created {len(generators)} generation jobs for auto-approved video {video_id}")
            
        except Exception as e:
            logger.error(f"Error creating generation jobs: {e}")


def should_auto_summarize(video: Video, settings) -> bool:
    """
    Check if video should be automatically summarized based on configuration.
    
    Args:
        video: Video database record
        settings: Application settings
        
    Returns:
        True if should auto-summarize, False otherwise
    """
    if not settings.enable_ai_summaries:
        return False
    
    # Check if channel is in auto-summarize list
    auto_channels = settings.auto_summarize_channels
    if isinstance(auto_channels, str):
        auto_channels = [c.strip() for c in auto_channels.split(',')]
    
    if video.channel_name in auto_channels:
        return True
    
    return False


def create_summary_job(video_id: str, priority: int = 2) -> bool:
    """
    Create a summary generation job.
    
    Args:
        video_id: Video ID to summarize
        priority: Job priority (1=highest, 5=lowest)
        
    Returns:
        True if job created successfully
    """
    try:
        job_queue = JobQueue()
        job_queue.add_job(
            job_type=JobType.SUMMARIZE,
            data={'video_id': video_id},
            priority=priority
        )
        logger.info(f"Created summary job for video {video_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating summary job: {e}")
        return False