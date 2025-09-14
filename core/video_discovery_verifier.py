"""
Video Discovery Verification System

This module ensures we discover ALL videos from a channel by:
- Cross-referencing multiple sources
- Tracking discovery history
- Detecting missing videos
- Continuous monitoring for new content
- Validation of completeness
"""

import json
import logging
import subprocess
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from enum import Enum

from core.channel_enumerator import ChannelEnumerator, EnumerationStrategy, VideoInfo
from core.rate_limit_manager import get_rate_limit_manager

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of video discovery verification."""
    COMPLETE = "complete"          # All videos discovered
    PARTIAL = "partial"            # Some videos missing
    UNCERTAIN = "uncertain"        # Cannot verify completeness
    FAILED = "failed"              # Verification failed


@dataclass
class VideoDiscoveryReport:
    """Report of video discovery verification."""
    channel_id: str
    channel_name: str
    verification_status: VerificationStatus
    total_discovered: int
    total_expected: Optional[int]  # From channel metadata
    missing_count: int
    missing_videos: List[str]      # IDs of missing videos if known
    discovery_methods_used: List[str]
    confidence_score: float         # 0.0 to 1.0
    recommendations: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelSnapshot:
    """Snapshot of a channel's video state at a point in time."""
    channel_id: str
    timestamp: datetime
    video_ids: Set[str]
    video_count: int
    latest_video_date: Optional[datetime]
    hash: str  # Hash of video IDs for quick comparison


class VideoDiscoveryVerifier:
    """Verifies completeness of video discovery from YouTube channels."""
    
    def __init__(self, 
                 storage_dir: Optional[Path] = None,
                 youtube_api_key: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.storage_dir = storage_dir or Path.home() / '.yt-dl-sub' / 'discovery_verification'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.enumerator = ChannelEnumerator(youtube_api_key=youtube_api_key)
        self.rate_limiter = get_rate_limit_manager()
        
        # Channel history tracking
        self.channel_history: Dict[str, List[ChannelSnapshot]] = self._load_history()
    
    def verify_channel_completeness(
        self,
        channel_url: str,
        discovered_videos: Set[str],
        force_deep_check: bool = False
    ) -> VideoDiscoveryReport:
        """
        Verify if we have discovered all videos from a channel.
        
        Args:
            channel_url: YouTube channel URL
            discovered_videos: Set of video IDs we have discovered
            force_deep_check: If True, use all available methods for verification
            
        Returns:
            VideoDiscoveryReport with verification results
        """
        channel_id = self._extract_channel_id(channel_url)
        self.logger.info(f"Verifying completeness for channel {channel_id}")
        
        # Step 1: Get channel metadata for expected video count
        expected_count = self._get_expected_video_count(channel_id, channel_url)
        
        # Step 2: Enumerate videos using multiple strategies
        enumeration_strategy = EnumerationStrategy.HYBRID if force_deep_check else EnumerationStrategy.RSS_FEED
        enumeration_result = self.enumerator.enumerate_channel(
            channel_url,
            strategy=enumeration_strategy,
            force_complete=force_deep_check
        )
        
        # Step 3: Cross-reference discovered videos
        all_known_videos = set(discovered_videos)
        for video in enumeration_result.videos:
            all_known_videos.add(video.video_id)
        
        # Step 4: Check for missing videos
        missing_videos = []
        missing_count = 0
        
        if expected_count:
            missing_count = max(0, expected_count - len(all_known_videos))
            
            # Try to identify specific missing videos
            if force_deep_check and missing_count > 0:
                missing_videos = self._identify_missing_videos(
                    channel_id, 
                    channel_url,
                    all_known_videos,
                    expected_count
                )
        
        # Step 5: Determine verification status
        verification_status = self._determine_verification_status(
            len(all_known_videos),
            expected_count,
            missing_count,
            enumeration_result.is_complete
        )
        
        # Step 6: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            verification_status,
            enumeration_result.enumeration_method,
            expected_count is not None
        )
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            verification_status,
            missing_count,
            enumeration_result.enumeration_method
        )
        
        # Step 8: Update history
        self._update_channel_history(channel_id, all_known_videos)
        
        return VideoDiscoveryReport(
            channel_id=channel_id,
            channel_name=enumeration_result.channel_name,
            verification_status=verification_status,
            total_discovered=len(all_known_videos),
            total_expected=expected_count,
            missing_count=missing_count,
            missing_videos=missing_videos,
            discovery_methods_used=enumeration_result.metadata.get('methods_used', []),
            confidence_score=confidence_score,
            recommendations=recommendations,
            timestamp=datetime.now(),
            metadata={
                'enumeration_complete': enumeration_result.is_complete,
                'estimated_missing': enumeration_result.estimated_missing
            }
        )
    
    def monitor_channel_changes(
        self,
        channel_id: str,
        channel_url: str,
        known_videos: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Monitor a channel for new or removed videos.
        
        Returns:
            Tuple of (new_video_ids, removed_video_ids)
        """
        self.logger.info(f"Monitoring channel {channel_id} for changes")
        
        # Get current state
        current_result = self.enumerator.enumerate_channel(
            channel_url,
            strategy=EnumerationStrategy.RSS_FEED  # Quick check
        )
        
        current_video_ids = {v.video_id for v in current_result.videos}
        
        # Find changes
        new_videos = list(current_video_ids - known_videos)
        removed_videos = list(known_videos - current_video_ids)
        
        if new_videos:
            self.logger.info(f"Found {len(new_videos)} new videos")
        if removed_videos:
            self.logger.warning(f"Detected {len(removed_videos)} removed videos")
        
        return new_videos, removed_videos
    
    def validate_discovery_methods(
        self,
        channel_url: str
    ) -> Dict[str, Any]:
        """
        Validate different discovery methods by comparing their results.
        
        Returns:
            Dictionary with validation results for each method
        """
        self.logger.info(f"Validating discovery methods for {channel_url}")
        
        results = {}
        all_videos_by_method = {}
        
        # Test each enumeration strategy
        strategies = [
            EnumerationStrategy.RSS_FEED,
            EnumerationStrategy.YT_DLP_DUMP,
            EnumerationStrategy.PLAYLIST
        ]
        
        for strategy in strategies:
            try:
                result = self.enumerator.enumerate_channel(
                    channel_url,
                    strategy=strategy,
                    max_videos=1000  # Reasonable limit for testing
                )
                
                video_ids = {v.video_id for v in result.videos}
                all_videos_by_method[strategy.value] = video_ids
                
                results[strategy.value] = {
                    'success': not result.error,
                    'video_count': len(video_ids),
                    'is_complete': result.is_complete,
                    'error': result.error
                }
                
            except Exception as e:
                results[strategy.value] = {
                    'success': False,
                    'video_count': 0,
                    'is_complete': False,
                    'error': str(e)
                }
        
        # Compare results
        if len(all_videos_by_method) > 1:
            # Find union of all discovered videos
            all_videos = set()
            for videos in all_videos_by_method.values():
                all_videos.update(videos)
            
            # Calculate overlap between methods
            for method1, videos1 in all_videos_by_method.items():
                for method2, videos2 in all_videos_by_method.items():
                    if method1 != method2:
                        overlap = len(videos1 & videos2)
                        only_in_1 = len(videos1 - videos2)
                        only_in_2 = len(videos2 - videos1)
                        
                        results[f"{method1}_vs_{method2}"] = {
                            'overlap': overlap,
                            f'only_in_{method1}': only_in_1,
                            f'only_in_{method2}': only_in_2
                        }
        
        return results
    
    def _get_expected_video_count(
        self,
        channel_id: str,
        channel_url: str
    ) -> Optional[int]:
        """Get expected video count from channel metadata."""
        try:
            # Use yt-dlp to get channel metadata
            cmd = [
                'yt-dlp',
                '--dump-single-json',
                '--playlist-items', '0',  # Don't download videos
                channel_url
            ]
            
            # CRITICAL FIX: Add rate limiting protection before yt-dlp subprocess call
            allowed, wait_time = self.rate_limiter.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp video count lookup")
                time.sleep(wait_time)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Record the request result for rate limiting stats
            self.rate_limiter.record_request('youtube.com', success=(result.returncode == 0),
                                           is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Different fields might contain video count
                video_count = (
                    data.get('playlist_count') or
                    data.get('n_entries') or
                    data.get('video_count') or
                    data.get('entries_count')
                )
                
                if video_count:
                    self.logger.info(f"Channel metadata shows {video_count} videos")
                    return int(video_count)
                    
        except Exception as e:
            self.logger.warning(f"Could not get expected video count: {e}")
        
        return None
    
    def _identify_missing_videos(
        self,
        channel_id: str,
        channel_url: str,
        known_videos: Set[str],
        expected_count: int
    ) -> List[str]:
        """Try to identify specific missing videos."""
        missing_videos = []
        
        # Use complete enumeration
        complete_result = self.enumerator.enumerate_channel(
            channel_url,
            strategy=EnumerationStrategy.YT_DLP_DUMP,
            force_complete=True
        )
        
        for video in complete_result.videos:
            if video.video_id not in known_videos:
                missing_videos.append(video.video_id)
                self.logger.info(f"Identified missing video: {video.title} ({video.video_id})")
        
        return missing_videos
    
    def _determine_verification_status(
        self,
        discovered_count: int,
        expected_count: Optional[int],
        missing_count: int,
        enumeration_complete: bool
    ) -> VerificationStatus:
        """Determine verification status based on discovery results."""
        if expected_count is None:
            # Can't verify without expected count
            if enumeration_complete:
                return VerificationStatus.COMPLETE
            else:
                return VerificationStatus.UNCERTAIN
        
        if missing_count == 0:
            return VerificationStatus.COMPLETE
        elif missing_count > 0 and discovered_count > 0:
            return VerificationStatus.PARTIAL
        elif discovered_count == 0:
            return VerificationStatus.FAILED
        else:
            return VerificationStatus.UNCERTAIN
    
    def _calculate_confidence_score(
        self,
        status: VerificationStatus,
        enumeration_method: str,
        has_expected_count: bool
    ) -> float:
        """Calculate confidence score for verification."""
        score = 0.0
        
        # Base score from status
        if status == VerificationStatus.COMPLETE:
            score = 0.9
        elif status == VerificationStatus.PARTIAL:
            score = 0.6
        elif status == VerificationStatus.UNCERTAIN:
            score = 0.3
        else:
            score = 0.1
        
        # Adjust based on enumeration method
        if "yt-dlp" in enumeration_method:
            score += 0.05
        if "YouTube API" in enumeration_method:
            score += 0.03
        if "+" in enumeration_method:  # Multiple methods
            score += 0.02
        
        # Adjust based on having expected count
        if has_expected_count:
            score += 0.05
        else:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _generate_recommendations(
        self,
        status: VerificationStatus,
        missing_count: int,
        enumeration_method: str
    ) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []
        
        if status == VerificationStatus.PARTIAL:
            recommendations.append(f"Missing {missing_count} videos - use yt-dlp for complete enumeration")
            
            if "RSS" in enumeration_method and missing_count > 15:
                recommendations.append("RSS only shows recent videos - switch to HYBRID or YT_DLP_DUMP strategy")
        
        elif status == VerificationStatus.UNCERTAIN:
            recommendations.append("Unable to verify completeness - try force_deep_check=True")
            recommendations.append("Consider using YouTube Data API for accurate video count")
        
        elif status == VerificationStatus.FAILED:
            recommendations.append("Discovery failed - check channel URL and network connectivity")
            recommendations.append("Verify rate limiting is not blocking requests")
        
        if missing_count > 100:
            recommendations.append("Large number of missing videos - channel may have many unlisted/private videos")
        
        return recommendations
    
    def _update_channel_history(
        self,
        channel_id: str,
        video_ids: Set[str]
    ):
        """Update channel history with current snapshot."""
        # Create snapshot
        snapshot = ChannelSnapshot(
            channel_id=channel_id,
            timestamp=datetime.now(),
            video_ids=video_ids,
            video_count=len(video_ids),
            latest_video_date=None,  # Would need to get from video metadata
            hash=hashlib.md5(''.join(sorted(video_ids)).encode()).hexdigest()
        )
        
        # Add to history
        if channel_id not in self.channel_history:
            self.channel_history[channel_id] = []
        
        self.channel_history[channel_id].append(snapshot)
        
        # Keep only last 100 snapshots per channel
        if len(self.channel_history[channel_id]) > 100:
            self.channel_history[channel_id] = self.channel_history[channel_id][-100:]
        
        # Save history
        self._save_history()
    
    def _load_history(self) -> Dict[str, List[ChannelSnapshot]]:
        """Load channel history from storage."""
        history_file = self.storage_dir / "channel_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                # Convert JSON to ChannelSnapshot objects
                history = {}
                for channel_id, snapshots in data.items():
                    history[channel_id] = []
                    for snapshot_data in snapshots:
                        snapshot = ChannelSnapshot(
                            channel_id=snapshot_data['channel_id'],
                            timestamp=datetime.fromisoformat(snapshot_data['timestamp']),
                            video_ids=set(snapshot_data['video_ids']),
                            video_count=snapshot_data['video_count'],
                            latest_video_date=datetime.fromisoformat(snapshot_data['latest_video_date']) 
                                            if snapshot_data.get('latest_video_date') else None,
                            hash=snapshot_data['hash']
                        )
                        history[channel_id].append(snapshot)
                
                return history
                
            except Exception as e:
                self.logger.warning(f"Could not load history: {e}")
        
        return {}
    
    def _save_history(self):
        """Save channel history to storage."""
        history_file = self.storage_dir / "channel_history.json"
        
        try:
            # Convert ChannelSnapshot objects to JSON
            data = {}
            for channel_id, snapshots in self.channel_history.items():
                data[channel_id] = []
                for snapshot in snapshots:
                    data[channel_id].append({
                        'channel_id': snapshot.channel_id,
                        'timestamp': snapshot.timestamp.isoformat(),
                        'video_ids': list(snapshot.video_ids),
                        'video_count': snapshot.video_count,
                        'latest_video_date': snapshot.latest_video_date.isoformat() 
                                           if snapshot.latest_video_date else None,
                        'hash': snapshot.hash
                    })
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save history: {e}")
    
    def _extract_channel_id(self, channel_url: str) -> str:
        """Extract channel ID from URL."""
        # Delegate to enumerator
        return self.enumerator._extract_channel_id(channel_url)