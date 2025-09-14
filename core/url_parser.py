"""
YouTube URL Parser Utility

Parses and identifies different types of YouTube URLs:
- Video URLs (regular and shorts)
- Channel URLs (various formats)
- Playlist URLs

Phase 1 implementation for the YouTube Content Intelligence Platform.
"""

import re
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs


class URLType(Enum):
    """Types of YouTube URLs"""
    VIDEO = "video"          # youtube.com/watch?v=XXX
    SHORTS = "shorts"        # youtube.com/shorts/XXX
    CHANNEL = "channel"      # @username, /channel/ID, /c/name, /user/name
    PLAYLIST = "playlist"    # youtube.com/playlist?list=XXX
    INVALID = "invalid"      # Not a valid YouTube URL


class YouTubeURLParser:
    """
    Parses YouTube URLs and extracts relevant information.
    
    Supports:
    - Video URLs: youtube.com/watch?v=VIDEO_ID, youtu.be/VIDEO_ID
    - Shorts: youtube.com/shorts/VIDEO_ID
    - Channels: youtube.com/@username, /channel/ID, /c/name, /user/name
    - Playlists: youtube.com/playlist?list=PLAYLIST_ID
    """
    
    # Regular expressions for different URL patterns
    VIDEO_PATTERNS = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    
    SHORTS_PATTERN = r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
    
    CHANNEL_PATTERNS = [
        r'youtube\.com/@([\w.-]+)(?:/videos|/featured|/)?',  # @username format with suffixes (Unicode support)
        r'youtube\.com/channel/(UC[a-zA-Z0-9_-]{22})',       # Channel ID format
        r'youtube\.com/c/([\w-]+)',                          # Custom channel name (Unicode support)
        r'youtube\.com/user/([\w-]+)',                       # Legacy user format (Unicode support)
        r'youtube\.com/([\w-]+)(?:/featured|/videos)?$',     # Direct channel name (Unicode support)
        r'^@([\w.-]+)$',                                     # Bare @ handle (Unicode support)
        r'^([\w-]+)$'                                        # Plain channel name (Unicode support)
    ]
    
    PLAYLIST_PATTERN = r'youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
    
    def __init__(self):
        """Initialize the URL parser"""
        self.video_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.VIDEO_PATTERNS]
        self.shorts_regex = re.compile(self.SHORTS_PATTERN, re.IGNORECASE)
        self.channel_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.CHANNEL_PATTERNS]
        self.playlist_regex = re.compile(self.PLAYLIST_PATTERN, re.IGNORECASE)
    
    def parse(self, url: str) -> Tuple[URLType, Optional[str], Dict[str, Any]]:
        """
        Parse a YouTube URL and return its type and identifier.
        
        Args:
            url: The YouTube URL to parse
            
        Returns:
            Tuple of (URLType, identifier, metadata)
            - URLType: The type of URL (VIDEO, SHORTS, CHANNEL, etc.)
            - identifier: The extracted ID (video_id, channel_id, etc.)
            - metadata: Additional information extracted from the URL
        """
        if not url:
            return URLType.INVALID, None, {}
        
        # Clean URL
        url = url.strip()
        original_url = url  # Keep original for special pattern matching
        
        # First, try to match bare patterns against original URL (before normalization)
        # This handles @TCM-Chan and TCM-Chan cases
        bare_handle_pattern = re.compile(r'^@([a-zA-Z0-9_.-]+)$')
        plain_name_pattern = re.compile(r'^([a-zA-Z0-9_-]+)$')
        
        # Check for bare @ handle first
        match = bare_handle_pattern.match(original_url)
        if match:
            identifier = match.group(1)
            identifier = identifier if identifier.startswith('@') else f'@{identifier}'
            return URLType.CHANNEL, identifier, {
                'channel_identifier': identifier,
                'channel_type': 'bare_handle',
                'url_type': 'channel',
                'original_url': original_url
            }
        
        # Check for plain channel name
        match = plain_name_pattern.match(original_url)
        if match:
            identifier = match.group(1)
            # Exclude common non-channel words
            if identifier not in ['watch', 'playlist', 'results', 'feed', 'trending', 'shorts']:
                return URLType.CHANNEL, identifier, {
                    'channel_identifier': identifier,
                    'channel_type': 'plain_name',
                    'url_type': 'channel',
                    'original_url': original_url
                }
        
        # Now normalize URL for standard pattern matching
        if url.startswith('@'):
            # Bare @ handle (e.g., @TCM-Chan) -> https://www.youtube.com/@TCM-Chan
            url = f'https://www.youtube.com/{url}'
        elif not url.startswith(('http://', 'https://')) and not ('youtube.com' in url or 'youtu.be' in url):
            # Plain channel name (e.g., TCM-Chan) -> https://www.youtube.com/TCM-Chan
            # But only if it doesn't already contain youtube domains
            if re.match(r'^[a-zA-Z0-9_.-]+$', url):
                url = f'https://www.youtube.com/{url}'
            else:
                # Fallback: just add https://
                url = 'https://' + url
        elif not url.startswith(('http://', 'https://')):
            # Standard case: just add https://
            url = 'https://' + url
        
        # Check for Shorts
        match = self.shorts_regex.search(url)
        if match:
            video_id = match.group(1)
            return URLType.SHORTS, video_id, {
                'video_id': video_id,
                'url_type': 'shorts',
                'original_url': url
            }
        
        # Check for regular video
        for regex in self.video_regex:
            match = regex.search(url)
            if match:
                video_id = match.group(1)
                return URLType.VIDEO, video_id, {
                    'video_id': video_id,
                    'url_type': 'video',
                    'original_url': url
                }
        
        # Check for playlist
        match = self.playlist_regex.search(url)
        if match:
            playlist_id = match.group(1)
            return URLType.PLAYLIST, playlist_id, {
                'playlist_id': playlist_id,
                'url_type': 'playlist',
                'original_url': url
            }
        
        # Check for channel
        for i, regex in enumerate(self.channel_regex):
            match = regex.search(url)
            if match:
                identifier = match.group(1)
                
                # Determine channel identifier type
                if i == 0:  # @username format with suffixes
                    channel_type = 'handle'
                    identifier = identifier if identifier.startswith('@') else f'@{identifier}'
                elif i == 1:  # Channel ID format
                    channel_type = 'channel_id'
                elif i == 2:  # Custom channel name
                    channel_type = 'custom'
                elif i == 3:  # Legacy user format
                    channel_type = 'user'
                elif i == 4:  # Direct channel name
                    channel_type = 'direct'
                    # Filter out common non-channel paths
                    if identifier in ['watch', 'playlist', 'results', 'feed', 'trending']:
                        continue
                elif i == 5:  # Bare @ handle (e.g., @TCM-Chan)
                    channel_type = 'bare_handle'
                    identifier = identifier if identifier.startswith('@') else f'@{identifier}'
                elif i == 6:  # Plain channel name (e.g., TCM-Chan)
                    channel_type = 'plain_name'
                else:
                    channel_type = 'unknown'
                
                return URLType.CHANNEL, identifier, {
                    'channel_identifier': identifier,
                    'channel_type': channel_type,
                    'url_type': 'channel',
                    'original_url': url
                }
        
        # If no patterns match, it's invalid
        return URLType.INVALID, None, {'original_url': url}
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from a URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID if found and valid, None otherwise
        """
        from core.youtube_validators import is_valid_youtube_id
        
        url_type, identifier, _ = self.parse(url)
        if url_type in [URLType.VIDEO, URLType.SHORTS]:
            # Validate before returning
            if is_valid_youtube_id(identifier):
                return identifier
            else:
                # Log warning but don't raise exception
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Extracted invalid video ID: {identifier} from URL: {url}")
                return None
        return None
    
    def extract_channel_identifier(self, url: str) -> Optional[str]:
        """
        Extract channel identifier from a URL.
        
        Args:
            url: YouTube channel URL
            
        Returns:
            Channel identifier if found, None otherwise
        """
        url_type, identifier, _ = self.parse(url)
        if url_type == URLType.CHANNEL:
            return identifier
        return None
    
    def normalize_channel_url(self, url: str) -> Optional[str]:
        """
        Convert any channel URL format to a standardized format.
        
        Args:
            url: YouTube channel URL in any format
            
        Returns:
            Normalized channel URL or None if invalid
        """
        url_type, identifier, metadata = self.parse(url)
        
        if url_type != URLType.CHANNEL:
            return None
        
        channel_type = metadata.get('channel_type')
        
        # Return appropriate normalized URL based on type
        if channel_type == 'handle':
            return f"https://www.youtube.com/{identifier}"
        elif channel_type == 'channel_id':
            return f"https://www.youtube.com/channel/{identifier}"
        elif channel_type == 'custom':
            return f"https://www.youtube.com/c/{identifier}"
        elif channel_type == 'user':
            return f"https://www.youtube.com/user/{identifier}"
        else:
            return f"https://www.youtube.com/{identifier}"
    
    def get_video_url(self, video_id: str) -> str:
        """
        Construct a standard video URL from a video ID.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Full YouTube video URL
        """
        return f"https://www.youtube.com/watch?v={video_id}"
    
    def is_youtube_url(self, url: str) -> bool:
        """
        Check if a URL is a valid YouTube URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if it's a YouTube URL, False otherwise
        """
        if not url:
            return False
        
        # Check for common YouTube domains
        youtube_domains = [
            'youtube.com',
            'www.youtube.com',
            'youtu.be',
            'm.youtube.com',
            'music.youtube.com'
        ]
        
        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
            domain = parsed.netloc.lower()
            return any(domain == d or domain.endswith(f'.{d}') for d in youtube_domains)
        except:
            return False
    
    def get_url_info(self, url: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a YouTube URL.
        
        Args:
            url: YouTube URL to analyze
            
        Returns:
            Dictionary with URL information
        """
        url_type, identifier, metadata = self.parse(url)
        
        return {
            'url': url,
            'type': url_type.value,
            'identifier': identifier,
            'is_valid': url_type != URLType.INVALID,
            'is_video': url_type in [URLType.VIDEO, URLType.SHORTS],
            'is_channel': url_type == URLType.CHANNEL,
            'is_playlist': url_type == URLType.PLAYLIST,
            'metadata': metadata
        }


# Convenience functions
_parser = YouTubeURLParser()

def parse_youtube_url(url: str) -> Tuple[URLType, Optional[str], Dict[str, Any]]:
    """Parse a YouTube URL and return its type and identifier."""
    return _parser.parse(url)

def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from a YouTube URL."""
    return _parser.extract_video_id(url)

def get_channel_identifier(url: str) -> Optional[str]:
    """Extract channel identifier from a YouTube URL."""
    return _parser.extract_channel_identifier(url)

def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL."""
    return _parser.is_youtube_url(url)

def get_url_info(url: str) -> Dict[str, Any]:
    """Get comprehensive information about a YouTube URL."""
    return _parser.get_url_info(url)


# Example usage and testing
if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/abc123def45",
        "https://www.youtube.com/@ColeMedin",
        "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
        "https://www.youtube.com/c/MrBeast",
        "https://www.youtube.com/user/PewDiePie",
        "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
        "youtube.com/@username",
        "invalid-url",
    ]
    
    parser = YouTubeURLParser()
    
    print("YouTube URL Parser Test Results:\n")
    print("-" * 80)
    
    for url in test_urls:
        url_type, identifier, metadata = parser.parse(url)
        print(f"URL: {url}")
        print(f"  Type: {url_type.value}")
        print(f"  Identifier: {identifier}")
        print(f"  Metadata: {metadata}")
        print("-" * 80)