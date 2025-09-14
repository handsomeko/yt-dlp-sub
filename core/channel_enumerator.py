"""
Comprehensive Channel Video Enumeration System

This module provides multiple strategies to enumerate ALL videos from a channel:
- RSS feed (recent videos only - usually last 15)
- YouTube Data API (complete listing with pagination)
- yt-dlp channel dumps (full archive)
- Playlist enumeration (uploads playlist)
- Web scraping fallback
- Incremental discovery
"""

import re
import json
import time
import logging
import subprocess
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import feedparser
import asyncio
from enum import Enum

from core.rate_limit_manager import get_rate_limit_manager
from config.settings import get_settings
from core.youtube_validators import is_valid_youtube_id

logger = logging.getLogger(__name__)


class EnumerationStrategy(Enum):
    """Available strategies for video enumeration."""
    RSS_FEED = "rss_feed"           # Quick but limited (recent only)
    YT_DLP_DUMP = "yt_dlp_dump"      # Complete but slower
    YOUTUBE_API = "youtube_api"      # Complete with quota limits
    PLAYLIST = "playlist"            # Via uploads playlist
    WEB_SCRAPE = "web_scrape"        # Fallback method
    HYBRID = "hybrid"                # Combine multiple methods


@dataclass
class VideoInfo:
    """Basic video information from enumeration."""
    video_id: str
    title: str
    upload_date: Optional[datetime] = None
    duration: Optional[int] = None  # seconds
    view_count: Optional[int] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None


@dataclass
class ChannelEnumerationResult:
    """Result of channel video enumeration."""
    channel_id: str
    channel_name: str
    total_videos: int
    videos: List[VideoInfo]
    enumeration_method: str
    is_complete: bool  # True if we got ALL videos
    estimated_missing: int = 0  # Estimated videos we might have missed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChannelEnumerator:
    """Comprehensive channel video enumeration with multiple strategies."""
    
    def __init__(self, 
                 youtube_api_key: Optional[str] = None,
                 cache_dir: Optional[Path] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.youtube_api_key = youtube_api_key
        self.cache_dir = cache_dir or Path.home() / '.yt-dl-sub' / 'channel_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiter
        self.rate_limiter = get_rate_limit_manager()
        
        # Enumeration history for incremental discovery
        self.enumeration_history: Dict[str, Set[str]] = {}
    
    def enumerate_channel(
        self,
        channel_url: str,
        strategy: Optional[EnumerationStrategy] = None,
        force_complete: Optional[bool] = None,
        max_videos: Optional[int] = None
    ) -> ChannelEnumerationResult:
        """
        Enumerate all videos from a channel.
        
        Args:
            channel_url: YouTube channel URL
            strategy: Enumeration strategy to use
            force_complete: If True, ensure we get ALL videos
            max_videos: Maximum number of videos to enumerate
            
        Returns:
            ChannelEnumerationResult with all discovered videos
        """
        # Use centralized settings for defaults
        settings = get_settings()
        
        if strategy is None:
            default_strategy = settings.default_enumeration_strategy.upper()
            strategy = EnumerationStrategy(default_strategy) if hasattr(EnumerationStrategy, default_strategy) else EnumerationStrategy.HYBRID
        
        if force_complete is None:
            force_complete = settings.force_complete_enumeration
        
        if max_videos is None:
            max_videos = settings.max_videos_per_channel
        
        channel_id = self._extract_channel_id(channel_url)
        
        if strategy == EnumerationStrategy.HYBRID:
            return self._enumerate_hybrid(channel_id, channel_url, force_complete, max_videos)
        elif strategy == EnumerationStrategy.RSS_FEED:
            return self._enumerate_rss(channel_id, channel_url, max_videos)
        elif strategy == EnumerationStrategy.YT_DLP_DUMP:
            return self._enumerate_yt_dlp(channel_id, channel_url, max_videos)
        elif strategy == EnumerationStrategy.YOUTUBE_API:
            return self._enumerate_youtube_api(channel_id, max_videos)
        elif strategy == EnumerationStrategy.PLAYLIST:
            return self._enumerate_playlist(channel_id, channel_url, max_videos)
        else:
            return self._enumerate_web_scrape(channel_id, channel_url, max_videos)
    
    def get_all_videos(
        self, 
        channel_url: str, 
        limit: Optional[int] = None
    ) -> List[VideoInfo]:
        """
        BACKWARD COMPATIBILITY METHOD
        
        Legacy method for compatibility with existing tests and code.
        Uses the hybrid enumeration strategy by default.
        
        Args:
            channel_url: YouTube channel URL
            limit: Maximum number of videos to return
            
        Returns:
            List of VideoInfo objects
        """
        result = self.enumerate_channel(
            channel_url=channel_url,
            strategy=EnumerationStrategy.HYBRID,
            max_videos=limit
        )
        return result.videos
    
    def _enumerate_hybrid(
        self,
        channel_id: str,
        channel_url: str,
        force_complete: bool,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """
        Hybrid enumeration using multiple strategies.
        
        1. Start with RSS for quick recent videos
        2. Use yt-dlp for complete enumeration
        3. Cross-reference and merge results
        4. Verify completeness
        """
        self.logger.info(f"Starting hybrid enumeration for channel {channel_id}")
        
        all_videos: Dict[str, VideoInfo] = {}
        methods_used = []
        
        # Step 1: Quick RSS enumeration for recent videos
        rss_result = self._enumerate_rss(channel_id, channel_url, max_videos=None)
        if not rss_result.error:
            for video in rss_result.videos:
                all_videos[video.video_id] = video
            methods_used.append("RSS")
            self.logger.info(f"RSS found {len(rss_result.videos)} recent videos")
        
        # Step 2: Complete enumeration with yt-dlp (if needed)
        if force_complete or len(all_videos) < 50:  # Threshold for suspecting more videos
            self.logger.info("Attempting complete enumeration with yt-dlp...")
            
            yt_dlp_result = self._enumerate_yt_dlp(channel_id, channel_url, max_videos)
            if not yt_dlp_result.error:
                for video in yt_dlp_result.videos:
                    if video.video_id not in all_videos:
                        all_videos[video.video_id] = video
                methods_used.append("yt-dlp")
                self.logger.info(f"yt-dlp found {len(yt_dlp_result.videos)} total videos")
        
        # Step 3: YouTube API for verification (if available and needed)
        if self.youtube_api_key and (force_complete or len(all_videos) < 100):
            api_result = self._enumerate_youtube_api(channel_id, max_videos)
            if not api_result.error:
                for video in api_result.videos:
                    if video.video_id not in all_videos:
                        all_videos[video.video_id] = video
                methods_used.append("YouTube API")
        
        # Step 4: Verify completeness
        channel_name = self._get_channel_name(channel_id, channel_url)
        is_complete = len(methods_used) > 1 or "yt-dlp" in methods_used
        
        # Estimate missing videos
        estimated_missing = 0
        if not is_complete and len(all_videos) % 15 == 0:
            # RSS typically shows 15 videos, if we have exactly 15, 30, etc, we might be missing some
            estimated_missing = 10  # Conservative estimate
        
        # Apply max_videos limit if specified
        video_list = list(all_videos.values())
        if max_videos and len(video_list) > max_videos:
            video_list = video_list[:max_videos]
        
        return ChannelEnumerationResult(
            channel_id=channel_id,
            channel_name=channel_name,
            total_videos=len(video_list),
            videos=video_list,
            enumeration_method="+".join(methods_used),
            is_complete=is_complete,
            estimated_missing=estimated_missing,
            metadata={
                'methods_used': methods_used,
                'total_discovered': len(all_videos)
            }
        )
    
    def _enumerate_rss(
        self,
        channel_id: str,
        channel_url: str,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """Enumerate videos using RSS feed (limited to recent videos)."""
        self.logger.info(f"Enumerating via RSS for channel {channel_id}")
        
        videos = []
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        
        try:
            # Use rate limiter
            def fetch_rss():
                return feedparser.parse(rss_url)
            
            feed, success = self.rate_limiter.execute_with_rate_limit(
                fetch_rss, 
                domain='youtube.com',
                max_retries=3
            )
            
            if not success or not feed:
                return ChannelEnumerationResult(
                    channel_id=channel_id,
                    channel_name="Unknown",
                    total_videos=0,
                    videos=[],
                    enumeration_method="RSS",
                    is_complete=False,
                    error="RSS feed fetch failed"
                )
            
            channel_name = feed.feed.get('title', 'Unknown')
            
            for entry in feed.entries:
                if max_videos and len(videos) >= max_videos:
                    break
                
                video_id = entry.yt_videoid if hasattr(entry, 'yt_videoid') else \
                          entry.link.split('v=')[1] if 'v=' in entry.link else None
                
                if video_id and is_valid_youtube_id(video_id):
                    videos.append(VideoInfo(
                        video_id=video_id,
                        title=entry.title,
                        upload_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None,
                        description=entry.summary if hasattr(entry, 'summary') else None
                    ))
                elif video_id:
                    self.logger.warning(f"Rejected invalid video ID from RSS: {video_id}")
            
            # RSS is never complete (only shows recent)
            is_complete = False
            estimated_missing = 0 if len(videos) < 15 else 10  # Guess
            
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name=channel_name,
                total_videos=len(videos),
                videos=videos,
                enumeration_method="RSS",
                is_complete=is_complete,
                estimated_missing=estimated_missing
            )
            
        except Exception as e:
            self.logger.error(f"RSS enumeration failed: {e}")
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name="Unknown",
                total_videos=0,
                videos=[],
                enumeration_method="RSS",
                is_complete=False,
                error=str(e)
            )
    
    def _enumerate_yt_dlp(
        self,
        channel_id: str,
        channel_url: str,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """
        Enumerate ALL videos using yt-dlp.
        This is the most complete method but slower.
        """
        self.logger.info(f"Enumerating via yt-dlp for channel {channel_id}")
        
        videos = []
        
        try:
            # Build yt-dlp command for dumping channel info
            cmd = [
                'yt-dlp',
                '--flat-playlist',  # Don't download, just list
                '--dump-json',      # Output JSON
                '--no-warnings',
                '--quiet',
                '--no-check-certificate',
                '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            ]
            
            if max_videos:
                cmd.extend(['--playlist-end', str(max_videos)])
            
            cmd.append(channel_url)
            
            # CRITICAL FIX: Check rate limiting BEFORE making subprocess call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp enumeration")
                time.sleep(wait_time)
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for large channels
            )
            
            # Record the request result for rate limiting stats
            rate_manager.record_request('youtube.com', success=(result.returncode == 0), 
                                      is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
            
            if result.returncode != 0:
                error_msg = result.stderr or "yt-dlp enumeration failed"
                
                # CRITICAL FIX: Removed duplicate 429 recording - already handled at line 360-361
                # Old code: self.rate_limiter.record_request('youtube.com', success=False, is_429=True)
                    
                return ChannelEnumerationResult(
                    channel_id=channel_id,
                    channel_name="Unknown",
                    total_videos=0,
                    videos=[],
                    enumeration_method="yt-dlp",
                    is_complete=False,
                    error=error_msg
                )
            
            # Parse JSON output (one JSON object per line)
            channel_name = "Unknown"
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract channel name if available
                    if 'channel' in data:
                        channel_name = data['channel']
                    
                    # Extract video info
                    video_id = data.get('id') or data.get('video_id')
                    if video_id and is_valid_youtube_id(video_id):
                        videos.append(VideoInfo(
                            video_id=video_id,
                            title=data.get('title', 'Unknown'),
                            upload_date=self._parse_date(data.get('upload_date')),
                            duration=data.get('duration'),
                            view_count=data.get('view_count'),
                            description=data.get('description'),
                            thumbnail_url=data.get('thumbnail')
                        ))
                    elif video_id:
                        self.logger.warning(f"Rejected invalid video ID from yt-dlp: {video_id}")
                        
                except json.JSONDecodeError:
                    continue
            
            if videos:
                self.logger.info(f"yt-dlp enumerated {len(videos)} valid videos from {channel_name}")
            else:
                self.logger.warning(f"yt-dlp found no valid videos from {channel_name}")
            
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name=channel_name,
                total_videos=len(videos),
                videos=videos,
                enumeration_method="yt-dlp",
                is_complete=True,  # yt-dlp gets everything
                estimated_missing=0
            )
            
        except subprocess.TimeoutExpired:
            self.logger.error("yt-dlp enumeration timed out")
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name="Unknown",
                total_videos=len(videos),
                videos=videos,
                enumeration_method="yt-dlp",
                is_complete=False,
                error="Enumeration timed out"
            )
            
        except Exception as e:
            self.logger.error(f"yt-dlp enumeration failed: {e}")
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name="Unknown",
                total_videos=0,
                videos=[],
                enumeration_method="yt-dlp",
                is_complete=False,
                error=str(e)
            )
    
    def _enumerate_youtube_api(
        self,
        channel_id: str,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """
        Enumerate videos using YouTube Data API v3.
        Requires API key and has quota limits.
        """
        if not self.youtube_api_key:
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name="Unknown",
                total_videos=0,
                videos=[],
                enumeration_method="YouTube API",
                is_complete=False,
                error="No YouTube API key configured"
            )
        
        self.logger.info(f"Enumerating via YouTube API for channel {channel_id}")
        
        # This would require google-api-python-client
        # Implementation placeholder - would use pagination to get all videos
        
        return ChannelEnumerationResult(
            channel_id=channel_id,
            channel_name="Unknown",
            total_videos=0,
            videos=[],
            enumeration_method="YouTube API",
            is_complete=False,
            error="YouTube API enumeration not yet implemented"
        )
    
    def _enumerate_playlist(
        self,
        channel_id: str,
        channel_url: str,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """Enumerate videos via uploads playlist."""
        # YouTube channels have an uploads playlist with ID UU + channel_id[2:]
        uploads_playlist_id = f"UU{channel_id[2:]}" if channel_id.startswith("UC") else None
        
        if not uploads_playlist_id:
            return ChannelEnumerationResult(
                channel_id=channel_id,
                channel_name="Unknown",
                total_videos=0,
                videos=[],
                enumeration_method="Playlist",
                is_complete=False,
                error="Could not determine uploads playlist ID"
            )
        
        playlist_url = f"https://www.youtube.com/playlist?list={uploads_playlist_id}"
        
        # Use yt-dlp to enumerate the playlist
        return self._enumerate_yt_dlp(channel_id, playlist_url, max_videos)
    
    def _enumerate_web_scrape(
        self,
        channel_id: str,
        channel_url: str,
        max_videos: Optional[int]
    ) -> ChannelEnumerationResult:
        """Web scraping fallback (not recommended)."""
        return ChannelEnumerationResult(
            channel_id=channel_id,
            channel_name="Unknown",
            total_videos=0,
            videos=[],
            enumeration_method="Web Scrape",
            is_complete=False,
            error="Web scraping not implemented (not recommended)"
        )
    
    def verify_completeness(
        self,
        channel_id: str,
        discovered_videos: List[str]
    ) -> Tuple[bool, int]:
        """
        Verify if we have discovered all videos from a channel.
        
        Returns:
            Tuple of (is_complete, estimated_missing_count)
        """
        # Try to get channel statistics
        try:
            cmd = [
                'yt-dlp',
                '--dump-single-json',
                '--playlist-items', '0',  # Don't get videos, just channel info
                f"https://www.youtube.com/channel/{channel_id}"
            ]
            
            # CRITICAL FIX: Check rate limiting BEFORE making subprocess call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp verification")
                time.sleep(wait_time)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Record the request result for rate limiting stats
            rate_manager.record_request('youtube.com', success=(result.returncode == 0),
                                      is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                total_videos = data.get('playlist_count') or data.get('n_entries')
                
                if total_videos:
                    missing = max(0, total_videos - len(discovered_videos))
                    is_complete = missing == 0
                    return is_complete, missing
                    
        except Exception as e:
            self.logger.warning(f"Could not verify completeness: {e}")
        
        # Heuristic: if we have less than 15 videos, we probably have them all
        # If we have exactly 15, 30, etc., we might be missing some
        if len(discovered_videos) < 15:
            return True, 0
        elif len(discovered_videos) % 15 == 0:
            return False, 10  # Estimate
        else:
            return True, 0  # Assume complete if not a multiple of 15
    
    def incremental_discovery(
        self,
        channel_id: str,
        channel_url: str,
        known_videos: Set[str]
    ) -> List[VideoInfo]:
        """
        Discover new videos incrementally.
        Compare against known videos to find new ones.
        """
        self.logger.info(f"Incremental discovery for channel {channel_id}")
        
        # Enumerate with hybrid strategy
        result = self.enumerate_channel(
            channel_url,
            strategy=EnumerationStrategy.HYBRID,
            force_complete=False
        )
        
        # Find new videos
        new_videos = []
        for video in result.videos:
            if video.video_id not in known_videos:
                new_videos.append(video)
                self.logger.info(f"Discovered new video: {video.title} ({video.video_id})")
        
        return new_videos
    
    def _extract_channel_id(self, channel_url: str) -> str:
        """Extract channel ID from various URL formats."""
        patterns = [
            r'youtube\.com/channel/(UC[\w-]+)',
            r'youtube\.com/c/([\w-]+)',
            r'youtube\.com/@([\w-]+)',
            r'youtube\.com/user/([\w-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, channel_url)
            if match:
                channel_identifier = match.group(1)
                
                # If it's already a channel ID, return it
                if channel_identifier.startswith('UC'):
                    return channel_identifier
                
                # Otherwise, need to resolve it to channel ID
                # This would require an API call or yt-dlp
                return self._resolve_channel_id(channel_identifier)
        
        return "unknown"
    
    def _resolve_channel_id(self, identifier: str) -> str:
        """Resolve various channel identifiers to channel ID."""
        # This would use yt-dlp or API to resolve
        # For now, return as-is
        return identifier
    
    def get_channel_info(self, channel_url: str) -> Dict[str, Any]:
        """Get channel metadata information.
        
        Args:
            channel_url: YouTube channel URL
            
        Returns:
            Dict with channel metadata including name, ID, subscriber count, etc.
        """
        try:
            # Use yt-dlp to get channel info
            cmd = [
                'yt-dlp',
                '--dump-single-json',
                '--playlist-items', '0',
                channel_url
            ]
            
            # CRITICAL FIX: Check rate limiting BEFORE making subprocess call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp channel info lookup")
                time.sleep(wait_time)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Record the request result for rate limiting stats
            rate_manager.record_request('youtube.com', success=(result.returncode == 0),
                                      is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract channel info
                channel_info = {
                    'channel_name': data.get('channel') or data.get('uploader') or 'Unknown',
                    'channel_id': data.get('channel_id') or self._extract_channel_id(channel_url),
                    'channel_url': data.get('channel_url') or channel_url,
                    'subscriber_count': data.get('channel_follower_count'),
                    'description': data.get('description'),
                    'channel': data.get('channel') or data.get('uploader'),
                    'uploader': data.get('uploader'),
                    'uploader_id': data.get('uploader_id'),
                    'uploader_url': data.get('uploader_url'),
                    'thumbnail': data.get('thumbnail'),
                    'playlist_count': data.get('playlist_count', 0)
                }
                
                return channel_info
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout getting channel info for: {channel_url}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from yt-dlp: {e}")
        except Exception as e:
            self.logger.error(f"Error getting channel info: {e}")
        
        # Return minimal info on error
        return {
            'channel_name': 'Unknown',
            'channel_id': self._extract_channel_id(channel_url),
            'channel_url': channel_url
        }
    
    def _get_channel_name(self, channel_id: str, channel_url: str) -> str:
        """Get channel name."""
        try:
            # Quick method using yt-dlp
            cmd = [
                'yt-dlp',
                '--dump-single-json',
                '--playlist-items', '0',
                channel_url
            ]
            
            # CRITICAL FIX: Check rate limiting BEFORE making subprocess call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp channel name lookup")
                time.sleep(wait_time)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Record the request result for rate limiting stats
            rate_manager.record_request('youtube.com', success=(result.returncode == 0),
                                      is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('channel') or data.get('uploader') or "Unknown"
                
        except Exception:
            pass
        
        return "Unknown"
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
            
        try:
            # YYYYMMDD format
            if len(date_str) == 8 and date_str.isdigit():
                return datetime.strptime(date_str, '%Y%m%d')
            # ISO format
            elif 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(date_str, '%Y-%m-%d')
        except Exception:
            return None