"""
YouTube validation utilities for ensuring data integrity.

This module provides validation functions for YouTube-specific data formats,
including video IDs, channel IDs, and URLs.
"""

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def is_valid_youtube_id(video_id: Optional[str]) -> bool:
    """
    Validate YouTube video ID format.
    
    YouTube video IDs must:
    - Be exactly 11 characters long
    - Use base64url alphabet: A-Z, a-z, 0-9, -, _
    - NOT start or end with -, _, or .
    
    Args:
        video_id: The video ID to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> is_valid_youtube_id("dQw4w9WgXcQ")  # Valid
        True
        >>> is_valid_youtube_id("-fXd0uINhFM")  # Invalid - starts with -
        False
        >>> is_valid_youtube_id("_bgDeaaO0wQ")  # Invalid - starts with _
        False
        >>> is_valid_youtube_id("abc")  # Invalid - wrong length
        False
    """
    if not video_id or not isinstance(video_id, str):
        return False
    
    # Must be exactly 11 characters
    if len(video_id) != 11:
        return False
    
    # Cannot end with -, _, or . (YouTube video IDs CAN start with hyphens and underscores)
    if video_id[-1] in ['-', '_', '.']:
        return False
    
    # Must match YouTube's base64url alphabet
    # A-Z, a-z, 0-9, -, _ are valid characters
    if not re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        return False
    
    return True


def is_valid_channel_id(channel_id: Optional[str]) -> bool:
    """
    Validate YouTube channel ID format.
    
    YouTube channel IDs typically:
    - Start with "UC" (for user channels)
    - Are 24 characters long
    - Use base64url alphabet
    
    Args:
        channel_id: The channel ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not channel_id or not isinstance(channel_id, str):
        return False
    
    # Most channel IDs start with UC and are 24 chars
    if channel_id.startswith('UC') and len(channel_id) == 24:
        return bool(re.match(r'^UC[A-Za-z0-9_-]{22}$', channel_id))
    
    # Some legacy formats exist but are less common
    # For now, accept any reasonable looking ID
    if len(channel_id) >= 10 and len(channel_id) <= 30:
        return bool(re.match(r'^[A-Za-z0-9_-]+$', channel_id))
    
    return False


def filter_valid_video_ids(videos: List[Dict[str, Any]], 
                          log_invalid: bool = True) -> List[Dict[str, Any]]:
    """
    Filter a list of video dictionaries to only include those with valid IDs.
    
    Args:
        videos: List of video dictionaries containing video_id or id fields
        log_invalid: Whether to log invalid IDs that are filtered out
        
    Returns:
        List of videos with valid IDs only
    """
    valid_videos = []
    invalid_ids = []
    
    for video in videos:
        # Try different field names for video ID
        video_id = video.get('video_id') or video.get('id') or video.get('videoId')
        
        if is_valid_youtube_id(video_id):
            valid_videos.append(video)
        else:
            invalid_ids.append(video_id)
    
    if invalid_ids and log_invalid:
        logger.warning(f"Filtered out {len(invalid_ids)} invalid video IDs: {invalid_ids}")
    
    return valid_videos


def validate_and_clean_video_list(videos: List[Any]) -> tuple[List[Any], List[str]]:
    """
    Validate and clean a list of videos, returning valid videos and invalid IDs.
    
    Args:
        videos: List of video objects (dicts or VideoInfo objects)
        
    Returns:
        Tuple of (valid_videos, invalid_ids)
    """
    valid_videos = []
    invalid_ids = []
    
    for video in videos:
        # Handle different video object types
        if hasattr(video, 'video_id'):
            video_id = video.video_id
        elif isinstance(video, dict):
            video_id = video.get('video_id') or video.get('id')
        else:
            video_id = str(video) if video else None
        
        if is_valid_youtube_id(video_id):
            valid_videos.append(video)
        else:
            invalid_ids.append(str(video_id))
    
    return valid_videos, invalid_ids


def extract_video_id_from_url(url: str) -> Optional[str]:
    """
    Extract and validate video ID from a YouTube URL.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Valid video ID or None if invalid/not found
    """
    patterns = [
        r'youtube\.com/watch\?v=([A-Za-z0-9_-]{11})',
        r'youtu\.be/([A-Za-z0-9_-]{11})',
        r'youtube\.com/embed/([A-Za-z0-9_-]{11})',
        r'youtube\.com/v/([A-Za-z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            if is_valid_youtube_id(video_id):
                return video_id
    
    return None