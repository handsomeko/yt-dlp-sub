"""
Monitor worker for the YouTube Content Intelligence & Repurposing Platform.

This worker monitors YouTube channel RSS feeds to detect new videos and
creates jobs for downstream processing (transcript extraction, etc.).

Key Features:
- RSS feed parsing with feedparser
- New video detection using last_video_id tracking
- Batch job creation for efficiency
- Rate limiting to respect YouTube's limits
- Comprehensive error handling for network issues
- Proper logging and metrics collection
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import feedparser
import httpx
from sqlalchemy import select, and_

from workers.base import BaseWorker, WorkerStatus
from core.database import (
    DatabaseManager, Channel, Video, Job,
    StorageVersion, TranscriptStatus
)
from core.rate_limit_manager import get_rate_limit_manager
from core.filename_sanitizer import sanitize_filename
from core.error_handling import network_operation, database_operation, with_error_handling


class MonitorWorker(BaseWorker):
    """
    Worker that monitors YouTube channel RSS feeds for new videos.
    
    This worker checks RSS feeds for monitored channels, detects new videos
    since the last check, and creates download_transcript jobs for processing.
    
    Features:
    - Rate-limited RSS feed checking
    - New video detection via last_video_id comparison
    - Batch job creation
    - Error categorization (network, parse, rate limit)
    - Configurable check intervals
    """
    
    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rate_limit_delay: float = 1.0,
        request_timeout: float = 30.0,
        log_level: str = "INFO"
    ) -> None:
        """
        Initialize the monitor worker.
        
        Args:
            database_manager: Database manager instance (uses global if None)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            rate_limit_delay: Delay between RSS requests in seconds
            request_timeout: HTTP request timeout in seconds
            log_level: Logging level
        """
        super().__init__(
            name="monitor",
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_level=log_level
        )
        
        # Import global db_manager if not provided
        if database_manager is None:
            from core.database import db_manager
            self.db_manager = db_manager
        else:
            self.db_manager = database_manager
            
        self.rate_limit_delay = rate_limit_delay
        self.request_timeout = request_timeout
        self._last_request_time = 0.0
        
        # RSS feed URL template
        self.rss_url_template = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for monitor worker.
        
        Args:
            input_data: Input data containing either:
                - channel_id: Specific channel to check
                - check_all: Boolean flag to check all active channels
                
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, dict):
            self.log_with_context("Input must be a dictionary", level="ERROR")
            return False
            
        channel_id = input_data.get("channel_id")
        check_all = input_data.get("check_all", False)
        
        # If check_all is provided, must be a boolean
        if "check_all" in input_data and not isinstance(check_all, bool):
            self.log_with_context(
                "check_all must be a boolean value",
                level="ERROR"
            )
            return False
            
        # Must have either channel_id or check_all flag
        if not channel_id and not check_all:
            self.log_with_context(
                "Must provide either 'channel_id' or 'check_all=True'",
                level="ERROR"
            )
            return False
            
        # If channel_id provided, must be non-empty string
        if channel_id and (not isinstance(channel_id, str) or not channel_id.strip()):
            self.log_with_context(
                "channel_id must be a non-empty string",
                level="ERROR"
            )
            return False
            
        return True
        
    async def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of run method that properly handles async execution.
        
        Args:
            input_data: Input data for the worker
            
        Returns:
            Standardized result dictionary with status, data, and metadata
        """
        self.log_with_context("Starting execution", extra_context={"input_keys": list(input_data.keys())})
        
        try:
            with self._execution_timer():
                # Validate input
                if not self.validate_input(input_data):
                    return self._create_result(
                        status=WorkerStatus.FAILED,
                        error="Input validation failed",
                        input_data=input_data
                    )
                
                # Execute main logic
                result = await self.execute(input_data)
                
                # Ensure result follows expected format
                if not isinstance(result, dict):
                    self.log_with_context("Worker returned non-dict result, wrapping", level="WARNING")
                    result = {"data": result}
                
                return self._create_result(
                    status=WorkerStatus.SUCCESS,
                    data=result,
                    input_data=input_data
                )
                
        except Exception as e:
            self.log_with_context(f"Execution failed: {str(e)}", level="ERROR")
            
            try:
                error_result = self.handle_error(e)
                return self._create_result(
                    status=WorkerStatus.FAILED,
                    error=str(e),
                    error_details=error_result,
                    input_data=input_data
                )
            except Exception as handler_error:
                self.log_with_context(
                    f"Error handler also failed: {str(handler_error)}", 
                    level="CRITICAL"
                )
                return self._create_result(
                    status=WorkerStatus.FAILED,
                    error=f"Primary error: {str(e)}. Handler error: {str(handler_error)}",
                    input_data=input_data
                )
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main monitor logic.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Dictionary containing:
            - channels_checked: Number of channels checked
            - new_videos_found: Number of new videos detected  
            - jobs_created: Number of jobs created
            - errors: List of errors encountered
            - processing_time: Time taken for execution
            
        Raises:
            Exception: On execution failure
        """
        start_time = time.time()
        
        channel_id = input_data.get("channel_id")
        check_all = input_data.get("check_all", False)
        
        self.log_with_context(
            f"Starting monitor execution",
            extra_context={
                "channel_id": channel_id,
                "check_all": check_all
            }
        )
        
        # Get channels to process
        if channel_id:
            channels = await self._get_channel_by_id(channel_id)
            if not channels:
                return {
                    "channels_checked": 0,
                    "new_videos_found": 0,
                    "jobs_created": 0,
                    "errors": [f"Channel {channel_id} not found or inactive"],
                    "processing_time": time.time() - start_time
                }
        else:
            channels = await self._get_active_channels()
            
        if not channels:
            self.log_with_context("No channels to monitor", level="WARNING")
            return {
                "channels_checked": 0,
                "new_videos_found": 0,
                "jobs_created": 0,
                "errors": [],
                "processing_time": time.time() - start_time
            }
            
        self.log_with_context(f"Processing {len(channels)} channels")
        
        # Process each channel
        total_new_videos = 0
        total_jobs_created = 0
        errors = []
        
        for channel in channels:
            try:
                new_videos, jobs_created = await self._process_channel(channel)
                total_new_videos += new_videos
                total_jobs_created += jobs_created
                
                self.log_with_context(
                    f"Channel processed successfully",
                    extra_context={
                        "channel_id": channel.channel_id,
                        "channel_name": channel.channel_name,
                        "new_videos": new_videos,
                        "jobs_created": jobs_created
                    }
                )
                
                # Rate limiting between channels
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                error_msg = f"Channel {channel.channel_id} ({channel.channel_name}): {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Error processing channel",
                    level="WARNING",
                    extra_context={
                        "channel_id": channel.channel_id,
                        "error": str(e)
                    }
                )
                
        processing_time = time.time() - start_time
        
        result = {
            "channels_checked": len(channels),
            "new_videos_found": total_new_videos,
            "jobs_created": total_jobs_created,
            "errors": errors,
            "processing_time": processing_time
        }
        
        self.log_with_context(
            "Monitor execution completed",
            extra_context=result
        )
        
        return result
        
    async def _get_active_channels(self) -> List[Channel]:
        """Get all active channels from the database."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.is_active == True)
            )
            return result.scalars().all()
            
    async def _get_channel_by_id(self, channel_id: str) -> List[Channel]:
        """Get a specific channel by ID."""
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
            
    async def _process_channel(self, channel: Channel) -> Tuple[int, int]:
        """
        Process a single channel to find new videos.
        
        Args:
            channel: Channel object to process
            
        Returns:
            Tuple of (new_videos_found, jobs_created)
        """
        # Fetch RSS feed with rate limiting
        await self._rate_limit()
        
        rss_url = self.rss_url_template.format(channel_id=channel.channel_id)
        
        try:
            # Fetch RSS feed
            feed_data = await self._fetch_rss_feed(rss_url)
            
            # Parse feed and extract video info
            videos = self._parse_feed_data(feed_data, channel.channel_id)
            
            if not videos:
                self.log_with_context(
                    f"No videos found in RSS feed",
                    extra_context={"channel_id": channel.channel_id}
                )
                return 0, 0
                
            # Find new videos since last check
            new_videos = self._filter_new_videos(videos, channel.last_video_id)
            
            if not new_videos:
                self.log_with_context(
                    f"No new videos since last check",
                    extra_context={
                        "channel_id": channel.channel_id,
                        "last_video_id": channel.last_video_id
                    }
                )
                # Update last_checked timestamp even if no new videos
                await self._update_channel_last_check(channel.channel_id)
                return 0, 0
                
            # Create video records and jobs
            jobs_created = await self._process_new_videos(channel, new_videos)
            
            # Update channel's last_video_id and last_checked
            latest_video_id = new_videos[0]["video_id"]  # First video is newest
            await self._update_channel_last_check(channel.channel_id, latest_video_id)
            
            return len(new_videos), jobs_created
            
        except Exception as e:
            self.log_with_context(
                f"Failed to process channel RSS feed",
                level="ERROR",
                extra_context={
                    "channel_id": channel.channel_id,
                    "rss_url": rss_url,
                    "error": str(e)
                }
            )
            raise
            
    async def _rate_limit(self) -> None:
        """Enforce rate limiting between RSS requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
            
        self._last_request_time = time.time()
        
    async def _fetch_rss_feed(self, rss_url: str) -> str:
        """
        Fetch RSS feed content with proper error handling.
        
        Args:
            rss_url: URL of the RSS feed
            
        Returns:
            Raw RSS feed content as string
            
        Raises:
            Exception: On fetch failure
        """
        # HIGH PRIORITY FIX: Add rate limiting protection to regular MonitorWorker
        rate_manager = get_rate_limit_manager()
        allowed, wait_time = rate_manager.should_allow_request('youtube.com')
        if not allowed:
            self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before RSS feed request")
            await asyncio.sleep(wait_time)
            
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            try:
                response = await client.get(
                    rss_url,
                    headers={
                        "User-Agent": "yt-dl-sub/1.0 (RSS Feed Monitor)"
                    }
                )
                response.raise_for_status()
                
                self.log_with_context(
                    f"Successfully fetched RSS feed",
                    extra_context={
                        "url": rss_url,
                        "status_code": response.status_code,
                        "content_length": len(response.content)
                    }
                )
                
                # Record successful request
                rate_manager.record_request('youtube.com', success=True, is_429=False)
                
                return response.text
                
            except httpx.TimeoutException:
                # Record timeout as failed request
                rate_manager.record_request('youtube.com', success=False, is_429=False)
                raise Exception(f"Request timeout after {self.request_timeout}s")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Record 404 as failed request (not rate limiting)
                    rate_manager.record_request('youtube.com', success=False, is_429=False)
                    raise Exception("Channel RSS feed not found (channel may not exist)")
                elif e.response.status_code == 403:
                    # Record 403 as failed request (not rate limiting)
                    rate_manager.record_request('youtube.com', success=False, is_429=False)
                    self.log_with_context(
                        "YouTube IP block detected - consider using residential IP/seedbox",
                        level="ERROR",
                        extra_context={"status_code": 403}
                    )
                    raise Exception("Access forbidden (YouTube IP block detected - use residential IP)")
                elif e.response.status_code == 429:
                    # CRITICAL: Record 429 as rate limiting error
                    rate_manager.record_request('youtube.com', success=False, is_429=True)
                    self.log_with_context(
                        "YouTube rate limit hit - will retry with backoff",
                        level="WARNING",
                        extra_context={"status_code": 429}
                    )
                    raise Exception("Rate limited by YouTube - retry with exponential backoff")
                else:
                    # Record other HTTP errors as failed requests
                    rate_manager.record_request('youtube.com', success=False, is_429=False)
                    raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                # Record network errors as failed requests
                rate_manager.record_request('youtube.com', success=False, is_429=False)
                raise Exception(f"Network error: {str(e)}")
                
    def _parse_feed_data(self, feed_content: str, channel_id: str) -> List[Dict[str, Any]]:
        """
        Parse RSS feed content and extract video information.
        
        Args:
            feed_content: Raw RSS feed content
            channel_id: Channel ID for context
            
        Returns:
            List of video dictionaries with extracted metadata
        """
        try:
            feed = feedparser.parse(feed_content)
            
            if feed.bozo:
                self.log_with_context(
                    f"RSS feed has parsing issues",
                    level="WARNING",
                    extra_context={
                        "channel_id": channel_id,
                        "bozo_exception": str(feed.bozo_exception) if feed.bozo_exception else None
                    }
                )
                
            videos = []
            
            for entry in feed.entries:
                try:
                    # Extract video ID from link
                    video_id = self._extract_video_id(entry.link)
                    if not video_id:
                        continue
                        
                    # Parse published date
                    published_at = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])
                    
                    video_info = {
                        "video_id": video_id,
                        "title": entry.get("title", "").strip(),
                        "description": entry.get("summary", "").strip(),
                        "link": entry.get("link", ""),
                        "published_at": published_at,
                        "channel_id": channel_id
                    }
                    
                    videos.append(video_info)
                    
                except Exception as e:
                    self.log_with_context(
                        f"Error parsing RSS entry",
                        level="WARNING",
                        extra_context={
                            "channel_id": channel_id,
                            "entry_title": getattr(entry, 'title', 'unknown'),
                            "error": str(e)
                        }
                    )
                    continue
                    
            self.log_with_context(
                f"Parsed RSS feed",
                extra_context={
                    "channel_id": channel_id,
                    "total_entries": len(feed.entries),
                    "valid_videos": len(videos)
                }
            )
            
            return videos
            
        except Exception as e:
            self.log_with_context(
                f"Failed to parse RSS feed",
                level="ERROR",
                extra_context={
                    "channel_id": channel_id,
                    "error": str(e)
                }
            )
            raise Exception(f"RSS parsing error: {str(e)}")
            
    def _extract_video_id(self, video_url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Video ID string or None if not found
        """
        if not video_url:
            return None
            
        # Handle different YouTube URL formats
        # https://www.youtube.com/watch?v=VIDEO_ID
        # https://youtu.be/VIDEO_ID
        # https://m.youtube.com/watch?v=VIDEO_ID
        
        from core.youtube_validators import is_valid_youtube_id
        
        try:
            video_id = None
            
            if "watch?v=" in video_url:
                video_id = video_url.split("watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[1].split("?")[0]
            else:
                # Try to extract 11-character video ID pattern from YouTube URLs only
                import re
                if "youtube.com" in video_url or "youtu.be" in video_url:
                    match = re.search(r"[a-zA-Z0-9_-]{11}", video_url)
                    video_id = match.group(0) if match else None
                else:
                    return None
            
            # Validate before returning
            if video_id and is_valid_youtube_id(video_id):
                return video_id
            elif video_id:
                self.logger.warning(f"Extracted invalid video ID: {video_id} from URL: {video_url}")
                return None
            else:
                return None
                
        except Exception:
            return None
            
    def _filter_new_videos(
        self, 
        videos: List[Dict[str, Any]], 
        last_video_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter videos to find only new ones since last check.
        
        Args:
            videos: List of video dictionaries from RSS feed
            last_video_id: ID of the last processed video for this channel
            
        Returns:
            List of new video dictionaries (newest first)
        """
        if not last_video_id:
            # No previous videos, all are new
            return videos
            
        # Find the index of the last known video
        new_videos = []
        for video in videos:
            if video["video_id"] == last_video_id:
                # Found the last known video, stop here
                break
            new_videos.append(video)
            
        return new_videos
        
    async def _process_new_videos(
        self, 
        channel: Channel, 
        new_videos: List[Dict[str, Any]]
    ) -> int:
        """
        Process new videos by creating database records and jobs.
        
        Args:
            channel: Channel object
            new_videos: List of new video dictionaries
            
        Returns:
            Number of jobs created
        """
        jobs_created = 0
        
        async with self.db_manager.get_session() as session:
            for video_info in new_videos:
                try:
                    # Create video record with V2 storage fields
                    video_title = video_info["title"]
                    title_sanitized = sanitize_filename(video_title, video_info["video_id"])
                    
                    video = Video(
                        video_id=video_info["video_id"],
                        channel_id=channel.channel_id,
                        title=video_title,
                        description=video_info["description"],
                        published_at=video_info.get("published_at") or datetime.now(),
                        # FIX Issue #23: Use enum values for consistent data integrity
                        transcript_status=TranscriptStatus.PENDING.value,
                        # V2 storage fields - Use enum values
                        video_title_snapshot=video_title,
                        title_sanitized=title_sanitized,
                        storage_version=StorageVersion.V2.value
                    )
                    session.add(video)
                    
                    # Create audio download job (workflow step 1)
                    # Note: Transcription job will be created by orchestrator when audio download completes
                    audio_job = Job(
                        job_type="download_audio",
                        target_id=video_info["video_id"],
                        status="pending",
                        priority=5  # Default priority
                    )
                    session.add(audio_job)
                    jobs_created += 1
                    
                    self.log_with_context(
                        f"Created video record and job",
                        extra_context={
                            "video_id": video_info["video_id"],
                            "title": video_info["title"][:50] + "..." if len(video_info["title"]) > 50 else video_info["title"],
                            "channel_id": channel.channel_id
                        }
                    )
                    
                except Exception as e:
                    self.log_with_context(
                        f"Failed to create video record",
                        level="ERROR",
                        extra_context={
                            "video_id": video_info.get("video_id", "unknown"),
                            "error": str(e)
                        }
                    )
                    # Continue processing other videos
                    continue
                    
        return jobs_created
        
    async def _update_channel_last_check(
        self, 
        channel_id: str, 
        last_video_id: Optional[str] = None
    ) -> None:
        """
        Update channel's last_checked timestamp and optionally last_video_id.
        
        Args:
            channel_id: Channel ID to update
            last_video_id: Latest video ID (optional)
        """
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.channel_id == channel_id)
            )
            channel = result.scalar_one_or_none()
            
            if channel:
                channel.last_checked = datetime.now()
                if last_video_id:
                    channel.last_video_id = last_video_id
                    
                self.log_with_context(
                    f"Updated channel last check",
                    extra_context={
                        "channel_id": channel_id,
                        "last_video_id": last_video_id
                    }
                )
                
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle and categorize execution errors.
        
        Args:
            error: Exception that occurred during execution
            
        Returns:
            Error handling result with categorization and recovery info
        """
        error_str = str(error).lower()
        error_category = "unknown"
        recovery_action = "retry"
        should_retry = True
        
        # Network-related errors
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "network", "dns", "resolve"
        ]):
            error_category = "network"
            recovery_action = "retry_with_backoff"
            
        # Rate limiting errors
        elif any(keyword in error_str for keyword in [
            "rate limit", "forbidden", "403", "429", "too many requests"
        ]):
            error_category = "rate_limit"
            recovery_action = "exponential_backoff"
            
        # Parse errors
        elif any(keyword in error_str for keyword in [
            "parse", "xml", "feed", "format", "invalid"
        ]):
            error_category = "parse_error"
            recovery_action = "skip_and_continue"
            should_retry = False
            
        # Channel not found errors
        elif any(keyword in error_str for keyword in [
            "not found", "404", "channel", "does not exist"
        ]):
            error_category = "channel_not_found"
            recovery_action = "mark_inactive"
            should_retry = False
            
        # Database errors
        elif any(keyword in error_str for keyword in [
            "database", "sqlite", "constraint", "integrity"
        ]):
            error_category = "database"
            recovery_action = "retry_with_backoff"
            
        self.log_with_context(
            f"Error categorized as {error_category}",
            level="ERROR",
            extra_context={
                "error_message": str(error),
                "recovery_action": recovery_action,
                "should_retry": should_retry
            }
        )
        
        return {
            "error_category": error_category,
            "error_message": str(error),
            "recovery_action": recovery_action,
            "should_retry": should_retry,
            "retry_delay_seconds": self._get_retry_delay(error_category),
            "error_context": {
                "worker_name": self.name,
                "timestamp": datetime.now().isoformat(),
                "execution_time": self.get_execution_time()
            }
        }
        
    def _get_retry_delay(self, error_category: str) -> float:
        """
        Get appropriate retry delay based on error category.
        
        Args:
            error_category: Categorized error type
            
        Returns:
            Delay in seconds before retry
        """
        delays = {
            "network": 5.0,
            "rate_limit": 60.0,  # Wait longer for rate limits
            "database": 2.0,
            "parse_error": 0.0,  # Don't retry parse errors
            "channel_not_found": 0.0,  # Don't retry missing channels
            "unknown": self.retry_delay
        }
        
        return delays.get(error_category, self.retry_delay)