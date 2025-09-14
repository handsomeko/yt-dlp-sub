"""
Enhanced Monitor Worker with Complete Channel Enumeration

This enhanced monitor worker integrates:
- Advanced rate limiting with exponential backoff and circuit breakers
- Comprehensive channel enumeration (RSS + yt-dlp + API)
- Video discovery verification to ensure no videos are missed
- Incremental discovery for continuous monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import feedparser
from sqlalchemy import select, and_

from workers.base import BaseWorker, WorkerStatus
from core.database import (
    DatabaseManager, Channel, Video, Job,
    StorageVersion, TranscriptStatus
)
from core.filename_sanitizer import sanitize_filename
from core.error_handling import network_operation, database_operation, with_error_handling

# Import new systems
from core.rate_limit_manager import get_rate_limit_manager, RateLimitState
from core.channel_enumerator import ChannelEnumerator, EnumerationStrategy
from core.video_discovery_verifier import VideoDiscoveryVerifier, VerificationStatus


class EnhancedMonitorWorker(BaseWorker):
    """
    Enhanced monitor worker with complete channel enumeration and rate limiting.
    
    Key improvements:
    - Uses multiple enumeration strategies (not just RSS)
    - Implements advanced rate limiting with circuit breakers
    - Verifies discovery completeness
    - Handles 429 errors gracefully
    - Discovers ALL videos, not just recent ones
    """
    
    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        log_level: str = "INFO",
        enumeration_strategy: EnumerationStrategy = EnumerationStrategy.HYBRID,
        force_complete_enumeration: bool = False,
        youtube_api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the enhanced monitor worker.
        
        Args:
            database_manager: Database manager instance
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries
            log_level: Logging level
            enumeration_strategy: Strategy for discovering videos
            force_complete_enumeration: If True, always get ALL videos
            youtube_api_key: Optional YouTube Data API key
        """
        super().__init__(
            name="monitor_enhanced",
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_level=log_level
        )
        
        # Database
        if database_manager is None:
            from core.database import db_manager
            self.db_manager = db_manager
        else:
            self.db_manager = database_manager
        
        # Configuration
        self.enumeration_strategy = enumeration_strategy
        self.force_complete_enumeration = force_complete_enumeration
        
        # Initialize new systems
        self.rate_limiter = get_rate_limit_manager()
        self.enumerator = ChannelEnumerator(youtube_api_key=youtube_api_key)
        self.verifier = VideoDiscoveryVerifier(youtube_api_key=youtube_api_key)
        
        # Track discovered videos per channel
        self.discovered_videos: Dict[str, Set[str]] = {}
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for monitor worker."""
        if "channel_id" in input_data:
            return isinstance(input_data["channel_id"], str) and input_data["channel_id"]
        elif "check_all" in input_data:
            return isinstance(input_data["check_all"], bool)
        else:
            self.logger.error("Input must contain either 'channel_id' or 'check_all'")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process monitoring request with enhanced enumeration.
        
        Args:
            input_data: Contains either channel_id or check_all flag
            
        Returns:
            Result dictionary with discovered videos and statistics
        """
        self.logger.info(f"[{self.name}] Starting enhanced monitoring")
        
        try:
            # Determine which channels to check
            if "channel_id" in input_data:
                channels = await self._get_channel(input_data["channel_id"])
            else:
                channels = await self._get_active_channels()
            
            if not channels:
                return {
                    "status": "no_channels",
                    "message": "No channels to monitor",
                    "channels_checked": 0,
                    "new_videos": 0
                }
            
            # Process each channel with rate limiting
            total_new_videos = 0
            results_by_channel = {}
            
            for channel in channels:
                # Check rate limiter before processing
                allowed, wait_time = self.rate_limiter.should_allow_request('youtube.com')
                
                if not allowed:
                    self.logger.info(f"Rate limited, waiting {wait_time:.1f}s before checking {channel.channel_name}")
                    await asyncio.sleep(wait_time)
                
                # Process channel with enhanced enumeration
                result = await self._process_channel_enhanced(channel)
                results_by_channel[channel.channel_id] = result
                total_new_videos += result["new_videos_count"]
                
                # Record successful request
                self.rate_limiter.record_request('youtube.com', success=True)
            
            # Verify completeness for all channels
            verification_reports = {}
            for channel in channels:
                if channel.channel_id in self.discovered_videos:
                    report = self.verifier.verify_channel_completeness(
                        channel.channel_url,
                        self.discovered_videos[channel.channel_id],
                        force_deep_check=self.force_complete_enumeration
                    )
                    verification_reports[channel.channel_id] = {
                        'status': report.verification_status.value,
                        'total_discovered': report.total_discovered,
                        'total_expected': report.total_expected,
                        'missing_count': report.missing_count,
                        'confidence': report.confidence_score,
                        'recommendations': report.recommendations
                    }
            
            return {
                "status": "success",
                "channels_checked": len(channels),
                "new_videos": total_new_videos,
                "results_by_channel": results_by_channel,
                "verification_reports": verification_reports,
                "rate_limit_stats": self.rate_limiter.get_stats('youtube.com')
            }
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Enhanced monitoring failed: {str(e)}")
            
            # Check if it's a rate limit error
            if '429' in str(e) or 'rate' in str(e).lower():
                self.rate_limiter.record_request('youtube.com', success=False, is_429=True)
            
            return {
                "status": "error",
                "error": str(e),
                "channels_checked": 0,
                "new_videos": 0
            }
    
    async def _process_channel_enhanced(self, channel: Channel) -> Dict[str, Any]:
        """
        Process a single channel with enhanced enumeration.
        
        Args:
            channel: Channel object to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"[{self.name}] Processing channel: {channel.channel_name} ({channel.channel_id})")
        
        # Get previously discovered videos for this channel
        known_videos = self.discovered_videos.get(channel.channel_id, set())
        
        # Enumerate videos using configured strategy
        enumeration_result = self.enumerator.enumerate_channel(
            channel.channel_url,
            strategy=self.enumeration_strategy,
            force_complete=self.force_complete_enumeration
        )
        
        if enumeration_result.error:
            self.logger.error(f"Enumeration failed for {channel.channel_name}: {enumeration_result.error}")
            
            # Check for rate limiting
            if '429' in enumeration_result.error:
                self.rate_limiter.record_request('youtube.com', success=False, is_429=True)
            
            return {
                "channel_id": channel.channel_id,
                "channel_name": channel.channel_name,
                "status": "error",
                "error": enumeration_result.error,
                "new_videos_count": 0
            }
        
        # Update discovered videos
        current_videos = {v.video_id for v in enumeration_result.videos}
        self.discovered_videos[channel.channel_id] = current_videos
        
        # Find new videos
        new_video_ids = current_videos - known_videos
        new_videos = [v for v in enumeration_result.videos if v.video_id in new_video_ids]
        
        self.logger.info(
            f"Channel {channel.channel_name}: "
            f"Found {len(current_videos)} total videos, "
            f"{len(new_videos)} new, "
            f"Method: {enumeration_result.enumeration_method}, "
            f"Complete: {enumeration_result.is_complete}"
        )
        
        # Create jobs for new videos
        if new_videos:
            await self._create_video_jobs(channel, new_videos)
        
        # Update channel's last check time and video count
        await self._update_channel_stats(
            channel, 
            len(current_videos),
            enumeration_result.is_complete
        )
        
        return {
            "channel_id": channel.channel_id,
            "channel_name": channel.channel_name,
            "status": "success",
            "total_videos": len(current_videos),
            "new_videos_count": len(new_videos),
            "new_video_ids": list(new_video_ids),
            "enumeration_method": enumeration_result.enumeration_method,
            "is_complete": enumeration_result.is_complete,
            "estimated_missing": enumeration_result.estimated_missing
        }
    
    async def _create_video_jobs(self, channel: Channel, new_videos: List) -> None:
        """Create jobs for newly discovered videos."""
        async with self.db_manager.get_session() as session:
            for video_info in new_videos:
                # Check if video already exists
                result = await session.execute(
                    select(Video).where(Video.video_id == video_info.video_id)
                )
                existing_video = result.scalar_one_or_none()
                
                if not existing_video:
                    # Create video record
                    video = Video(
                        video_id=video_info.video_id,
                        channel_id=channel.channel_id,
                        title=video_info.title,
                        published_at=video_info.upload_date,
                        transcript_status=TranscriptStatus.PENDING
                    )
                    session.add(video)
                    
                    # Create download job
                    job = Job(
                        job_type="download_transcript",
                        target_id=video_info.video_id,
                        priority=5,
                        metadata={
                            "channel_id": channel.channel_id,
                            "channel_name": channel.channel_name,
                            "video_title": video_info.title
                        }
                    )
                    session.add(job)
                    
                    self.logger.info(f"Created job for new video: {video_info.title} ({video_info.video_id})")
            
            await session.commit()
    
    async def _update_channel_stats(
        self, 
        channel: Channel, 
        total_videos: int,
        is_complete: bool
    ) -> None:
        """Update channel statistics after enumeration."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.channel_id == channel.channel_id)
            )
            db_channel = result.scalar_one_or_none()
            
            if db_channel:
                db_channel.last_checked = datetime.utcnow()
                
                # Store enumeration metadata
                if not db_channel.metadata:
                    db_channel.metadata = {}
                
                db_channel.metadata.update({
                    "total_videos": total_videos,
                    "enumeration_complete": is_complete,
                    "last_enumeration_strategy": self.enumeration_strategy.value,
                    "last_check_timestamp": datetime.utcnow().isoformat()
                })
                
                await session.commit()
    
    async def _get_channel(self, channel_id: str) -> List[Channel]:
        """Get a specific channel from database."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(
                    and_(
                        Channel.channel_id == channel_id,
                        Channel.is_active == True
                    )
                )
            )
            channel = result.scalar_one_or_none()
            return [channel] if channel else []
    
    async def _get_active_channels(self) -> List[Channel]:
        """Get all active channels from database."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.is_active == True)
            )
            return list(result.scalars().all())
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring."""
        return self.rate_limiter.get_stats('youtube.com')
    
    def reset_rate_limits(self) -> None:
        """Reset rate limit statistics (useful for testing)."""
        self.rate_limiter.reset_domain_stats('youtube.com')