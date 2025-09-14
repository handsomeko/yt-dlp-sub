"""
Download Worker for YouTube Content Intelligence & Repurposing Platform.

This worker downloads video transcripts using a fallback chain approach:
1. Primary: yt-dlp with subtitle extraction
2. Fallback: youtube-transcript-api

Features:
- Automatic SRT to plain text conversion
- File path sanitization and proper storage organization
- Metadata extraction (duration, views, etc.)
- Progress logging and error categorization
- Integration with database for result storage
"""

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from workers.base import BaseWorker, WorkerStatus
from core.database import DatabaseManager, Transcript, Video
from config.settings import get_settings
from core.storage_paths_v2 import get_storage_paths_v2

# CRITICAL FIX: Import rate limiting to prevent 429 errors
from core.rate_limit_manager import get_rate_limit_manager


class DownloadError(Exception):
    """Base exception for download-related errors."""
    
    def __init__(self, message: str, error_code: str, recoverable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable


class DownloadWorker(BaseWorker):
    """
    Worker for downloading YouTube video transcripts with fallback chain.
    
    This worker implements a robust transcript download system that tries
    multiple methods to ensure maximum success rate:
    
    1. yt-dlp with subtitle extraction (preferred)
    2. youtube-transcript-api as fallback
    
    The worker also handles:
    - File path sanitization
    - SRT to plain text conversion
    - Metadata extraction and storage
    - Progress tracking and error handling
    """
    
    # Error codes for different failure types
    ERROR_CODES = {
        "NETWORK": "E001",
        "RATE_LIMIT": "E002", 
        "PRIVATE_VIDEO": "E003",
        "NO_TRANSCRIPT": "E004",
        "INVALID_URL": "E005",
        "FILESYSTEM": "E006",
        "DATABASE": "E007",
    }
    
    def __init__(self, **kwargs):
        super().__init__(
            name="DownloadWorker",
            max_retries=3,
            retry_delay=2.0,
            **kwargs
        )
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.storage = get_storage_paths_v2()  # Use V2 storage
        
        # Configure yt-dlp options
        self.ytdl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'srt',
            'skip_download': True,  # We only want transcripts
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,
            'ignoreerrors': True,
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for transcript download.
        
        Required fields:
        - video_id: YouTube video ID 
        - video_url: Full YouTube URL
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ['video_id', 'video_url']
        
        for field in required_fields:
            if field not in input_data:
                self.log_with_context(
                    f"Missing required field: {field}",
                    level="ERROR",
                    extra_context={"available_fields": list(input_data.keys())}
                )
                return False
        
        # Validate video_id format
        from core.youtube_validators import is_valid_youtube_id
        
        video_id = input_data['video_id']
        if not is_valid_youtube_id(video_id):
            self.log_with_context(
                f"Invalid video_id format: {video_id}",
                level="ERROR"
            )
            return False
        
        # Validate URL format
        video_url = input_data['video_url']
        if not self._is_valid_youtube_url(video_url):
            self.log_with_context(
                f"Invalid YouTube URL: {video_url}",
                level="ERROR"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcript download with fallback chain.
        
        Process:
        1. Try yt-dlp subtitle extraction
        2. Fallback to youtube-transcript-api
        3. Convert SRT to plain text
        4. Store files in organized directory structure
        5. Update database with transcript data
        
        Args:
            input_data: Validated input data
            
        Returns:
            Dict containing download results and file paths
        """
        video_id = input_data['video_id']
        video_url = input_data['video_url']
        channel_id = input_data.get('channel_id', 'unknown')
        
        self.log_with_context(
            f"Starting transcript download",
            extra_context={
                "video_id": video_id,
                "channel_id": channel_id,
                "url": video_url
            }
        )
        
        # Create storage directory using V2
        storage_dir = self._create_storage_directory(channel_id, video_id)
        
        # Get video title if available from input
        video_title = input_data.get('title', None)
        
        # Try primary method: yt-dlp
        transcript_data = None
        try:
            transcript_data = self._download_with_ytdlp(video_url, channel_id, video_id, video_title)
            self.log_with_context("Successfully downloaded transcript with yt-dlp")
        except DownloadError as e:
            if not e.recoverable:
                raise
            self.log_with_context(
                f"yt-dlp failed: {e}. Trying fallback method...",
                level="WARNING"
            )
        
        # Fallback method: youtube-transcript-api
        if transcript_data is None:
            try:
                transcript_data = self._download_with_transcript_api(video_id, channel_id, video_title)
                self.log_with_context("Successfully downloaded transcript with fallback API")
            except DownloadError as e:
                self.log_with_context(f"All download methods failed: {e}", level="ERROR")
                raise
        
        # Extract video metadata
        metadata = self._extract_video_metadata(video_url)
        
        # Store results in database
        asyncio.create_task(self._update_database(video_id, transcript_data, metadata))
        
        result = {
            "video_id": video_id,
            "channel_id": channel_id,
            "storage_dir": str(storage_dir),
            "srt_path": transcript_data.get("srt_path"),
            "txt_path": transcript_data.get("txt_path"),
            "word_count": transcript_data.get("word_count", 0),
            "extraction_method": transcript_data.get("method"),
            "language": transcript_data.get("language", "en"),
            "metadata": metadata,
        }
        
        self.log_with_context(
            "Transcript download completed successfully",
            extra_context={
                "word_count": result["word_count"],
                "method": result["extraction_method"]
            }
        )
        
        return result
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle and categorize download errors.
        
        Args:
            error: Exception that occurred during download
            
        Returns:
            Dict with error categorization and recovery info
        """
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "recoverable": True,
            "suggested_retry_delay": self.retry_delay,
        }
        
        # Categorize specific error types
        if isinstance(error, DownloadError):
            error_details.update({
                "error_code": error.error_code,
                "recoverable": error.recoverable,
            })
        elif "HTTP Error 429" in str(error):
            error_details.update({
                "error_code": self.ERROR_CODES["RATE_LIMIT"],
                "suggested_retry_delay": 300,  # 5 minutes for rate limit
            })
        elif "private" in str(error).lower() or "unavailable" in str(error).lower():
            error_details.update({
                "error_code": self.ERROR_CODES["PRIVATE_VIDEO"],
                "recoverable": False,
            })
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            error_details.update({
                "error_code": self.ERROR_CODES["NETWORK"],
                "suggested_retry_delay": 60,  # 1 minute for network issues
            })
        else:
            error_details["error_code"] = "UNKNOWN"
        
        return error_details
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        youtube_patterns = [
            r'^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}',
            r'^https?://m\.youtube\.com/watch\?v=[a-zA-Z0-9_-]{11}'
        ]
        
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:v\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _create_storage_directory(self, channel_id: str, video_id: str) -> Path:
        """
        Create organized storage directory structure using V2.
        
        Uses V2 structure: {storage_path}/{channel_id}/{video_id}/
        
        Args:
            channel_id: YouTube channel ID
            video_id: YouTube video ID
            
        Returns:
            Path to the video directory
        """
        # Use V2 storage paths
        storage_dir = self.storage.get_video_dir(channel_id, video_id)
        
        try:
            self.log_with_context(f"Using V2 storage directory: {storage_dir}")
            return storage_dir
        except OSError as e:
            raise DownloadError(
                f"Failed to create storage directory: {e}",
                self.ERROR_CODES["FILESYSTEM"],
                recoverable=False
            )
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe filesystem storage.
        
        Args:
            filename: Raw filename
            
        Returns:
            Sanitized filename safe for all filesystems
        """
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename.strip())
        filename = re.sub(r'[^\w\-_.]', '', filename)
        
        # Limit length to avoid filesystem limits
        if len(filename) > 200:
            filename = filename[:200]
        
        # Ensure it's not empty
        if not filename:
            filename = "untitled"
        
        return filename
    
    def _download_with_ytdlp(self, video_url: str, channel_id: str, video_id: str, video_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Download transcript using yt-dlp.
        
        Args:
            video_url: YouTube video URL
            storage_dir: Directory to store downloaded files
            
        Returns:
            Dict with transcript data and file paths
            
        Raises:
            DownloadError: If download fails
        """
        self.log_with_context("Attempting download with yt-dlp using V2 storage")
        
        # Get V2 transcript directory
        transcript_dir = self.storage.get_transcript_dir(channel_id, video_id)
        
        # Configure output paths for V2 structure
        opts = self.ytdl_opts.copy()
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Extract info first
                info = ydl.extract_info(video_url, download=False)
                if not info:
                    raise DownloadError(
                        "Could not extract video information",
                        self.ERROR_CODES["INVALID_URL"]
                    )
                
                # Get video title if not provided
                if not video_title:
                    video_title = info.get('title', 'untitled')
                
                # Use V2 storage method to get transcript file path
                # This will create {video_title}.srt in the transcripts directory
                srt_file_path = self.storage.get_transcript_file(
                    channel_id, video_id, video_title, format='srt'
                )
                
                # Update template to use the V2 path
                opts['outtmpl'] = str(srt_file_path.parent / f"{srt_file_path.stem}.%(ext)s")
                
                # Download subtitles
                with yt_dlp.YoutubeDL(opts) as ydl_download:
                    ydl_download.download([video_url])
                
                # Look for downloaded SRT files in V2 transcript directory
                srt_files = list(transcript_dir.glob("*.srt"))
                if not srt_files:
                    raise DownloadError(
                        "No subtitle files were downloaded",
                        self.ERROR_CODES["NO_TRANSCRIPT"]
                    )
                
                # Use the first available SRT file
                srt_path = srt_files[0]
                
                # Convert SRT to plain text
                txt_path = self._convert_srt_to_text(srt_path)
                
                # Count words in transcript
                word_count = self._count_words(txt_path)
                
                return {
                    "method": "yt-dlp",
                    "srt_path": str(srt_path),
                    "txt_path": str(txt_path),
                    "word_count": word_count,
                    "language": "en",  # yt-dlp typically gets English
                }
                
        except yt_dlp.DownloadError as e:
            error_msg = str(e).lower()
            if "private" in error_msg or "unavailable" in error_msg:
                raise DownloadError(
                    f"Video is private or unavailable: {e}",
                    self.ERROR_CODES["PRIVATE_VIDEO"],
                    recoverable=False
                )
            elif "429" in error_msg:
                raise DownloadError(
                    f"Rate limited: {e}",
                    self.ERROR_CODES["RATE_LIMIT"]
                )
            else:
                raise DownloadError(
                    f"yt-dlp download failed: {e}",
                    self.ERROR_CODES["NETWORK"]
                )
        
        except Exception as e:
            raise DownloadError(
                f"Unexpected error in yt-dlp: {e}",
                self.ERROR_CODES["NETWORK"]
            )
    
    def _download_with_transcript_api(self, video_id: str, storage_dir: Path) -> Dict[str, Any]:
        """
        Download transcript using youtube-transcript-api as fallback.
        
        Args:
            video_id: YouTube video ID
            storage_dir: Directory to store downloaded files
            
        Returns:
            Dict with transcript data and file paths
            
        Raises:
            DownloadError: If download fails
        """
        self.log_with_context("Attempting download with transcript API fallback")
        
        # CRITICAL FIX: Add rate limiting protection before YouTube API call
        rate_manager = get_rate_limit_manager()
        allowed, wait_time = rate_manager.should_allow_request('youtube.com')
        if not allowed:
            self.log_with_context(f"Rate limited - waiting {wait_time:.1f}s before transcript API fallback for {video_id}")
            time.sleep(wait_time)
        
        try:
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # CRITICAL FIX: Record successful API call for rate limiting stats
            rate_manager.record_request('youtube.com', success=True)
            
            # Prefer English transcripts
            transcript = None
            for lang_code in ['en', 'en-US', 'en-GB']:
                try:
                    transcript = transcript_list.find_transcript([lang_code])
                    break
                except:
                    continue
            
            # Fallback to any available transcript
            if not transcript:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Get any available transcript
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0]
            
            if not transcript:
                raise DownloadError(
                    "No transcripts available for this video",
                    self.ERROR_CODES["NO_TRANSCRIPT"],
                    recoverable=False
                )
            
            # Fetch transcript data
            transcript_data = transcript.fetch()
            
            # Use video title for filename if available
            if not video_title:
                video_title = f"transcript_{video_id}"  # Fallback name
            
            # Get V2 file paths
            srt_path = self.storage.get_transcript_file(
                channel_id, video_id, video_title, format='srt'
            )
            
            # Convert to SRT format
            srt_content = self._convert_transcript_to_srt(transcript_data)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # Convert to plain text
            txt_path = self._convert_srt_to_text(srt_path)
            
            # Count words
            word_count = self._count_words(txt_path)
            
            return {
                "method": "youtube-transcript-api",
                "srt_path": str(srt_path),
                "txt_path": str(txt_path),
                "word_count": word_count,
                "language": transcript.language_code,
            }
            
        except NoTranscriptFound:
            # CRITICAL FIX: Record API failure for rate limiting stats
            rate_manager = get_rate_limit_manager()
            rate_manager.record_request('youtube.com', success=False, is_429=False)
            raise DownloadError(
                "No transcript found for this video",
                self.ERROR_CODES["NO_TRANSCRIPT"],
                recoverable=False
            )
        except Exception as e:
            # CRITICAL FIX: Record API failure with 429 detection for rate limiting stats
            rate_manager = get_rate_limit_manager()
            error_str = str(e).lower()
            is_429 = '429' in error_str or 'too many requests' in error_str
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            raise DownloadError(
                f"Transcript API failed: {e}",
                self.ERROR_CODES["NETWORK"]
            )
    
    def _convert_transcript_to_srt(self, transcript_data: List[Dict]) -> str:
        """
        Convert transcript API data to SRT format.
        
        Args:
            transcript_data: List of transcript entries
            
        Returns:
            SRT formatted string
        """
        srt_content = []
        
        for i, entry in enumerate(transcript_data, 1):
            start_time = entry['start']
            duration = entry.get('duration', 3.0)
            text = entry['text'].strip()
            
            # Convert seconds to SRT time format
            start_hours, remainder = divmod(int(start_time), 3600)
            start_minutes, start_seconds = divmod(remainder, 60)
            start_ms = int((start_time % 1) * 1000)
            
            end_time = start_time + duration
            end_hours, remainder = divmod(int(end_time), 3600)
            end_minutes, end_seconds = divmod(remainder, 60)
            end_ms = int((end_time % 1) * 1000)
            
            srt_content.append(f"{i}")
            srt_content.append(
                f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},{start_ms:03d} --> "
                f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d},{end_ms:03d}"
            )
            srt_content.append(text)
            srt_content.append("")  # Empty line between entries
        
        return "\n".join(srt_content)
    
    def _convert_srt_to_text(self, srt_path: Path) -> Path:
        """
        Convert SRT file to plain text transcript.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            Path to created TXT file
        """
        txt_path = srt_path.with_suffix('.txt')
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as srt_file:
                content = srt_file.read()
            
            # Remove SRT formatting (timestamps, sequence numbers)
            # SRT format: number, timestamp, text, empty line
            lines = content.split('\n')
            text_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip sequence numbers (just digits)
                if line.isdigit():
                    i += 1
                    continue
                
                # Skip timestamp lines (contains -->)
                if '-->' in line:
                    i += 1
                    continue
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # This is a text line
                text_lines.append(line)
                i += 1
            
            # Join all text and clean up
            plain_text = ' '.join(text_lines)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            
            # Write plain text file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(plain_text)
            
            self.log_with_context(f"Converted SRT to text: {txt_path}")
            return txt_path
            
        except Exception as e:
            raise DownloadError(
                f"Failed to convert SRT to text: {e}",
                self.ERROR_CODES["FILESYSTEM"]
            )
    
    def _count_words(self, txt_path: Path) -> int:
        """Count words in a text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return len(content.split())
        except Exception:
            return 0
    
    def _extract_video_metadata(self, video_url: str) -> Dict[str, Any]:
        """
        Extract video metadata using yt-dlp.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict with video metadata
        """
        metadata = {}
        
        try:
            opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                if info:
                    metadata = {
                        'title': info.get('title'),
                        'description': info.get('description'),
                        'duration': info.get('duration'),
                        'view_count': info.get('view_count'),
                        'like_count': info.get('like_count'),
                        'upload_date': info.get('upload_date'),
                        'uploader': info.get('uploader'),
                        'uploader_id': info.get('uploader_id'),
                    }
                    
        except Exception as e:
            self.log_with_context(
                f"Could not extract metadata: {e}",
                level="WARNING"
            )
        
        return metadata
    
    async def _update_database(
        self, 
        video_id: str, 
        transcript_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update database with transcript data and video metadata.
        
        Args:
            video_id: YouTube video ID
            transcript_data: Transcript download results
            metadata: Video metadata
        """
        try:
            async with self.db_manager.get_session() as session:
                # Update video record
                video_stmt = (
                    update(Video)
                    .where(Video.video_id == video_id)
                    .values(
                        title=metadata.get('title'),
                        description=metadata.get('description'),
                        duration=metadata.get('duration'),
                        view_count=metadata.get('view_count'),
                        like_count=metadata.get('like_count'),
                        transcript_status='completed'
                    )
                )
                await session.execute(video_stmt)
                
                # Create or update transcript record
                transcript = Transcript(
                    video_id=video_id,
                    content_srt=self._read_file_content(transcript_data.get('srt_path')),
                    content_text=self._read_file_content(transcript_data.get('txt_path')),
                    word_count=transcript_data.get('word_count', 0),
                    language=transcript_data.get('language', 'en'),
                    extraction_method=transcript_data.get('method'),
                    srt_path=transcript_data.get('srt_path'),
                    transcript_path=transcript_data.get('txt_path'),
                )
                
                # Check if transcript already exists
                existing = await session.execute(
                    select(Transcript).where(Transcript.video_id == video_id)
                )
                existing_transcript = existing.scalar_one_or_none()
                
                if existing_transcript:
                    # Update existing transcript
                    for key, value in transcript.__dict__.items():
                        if key.startswith('_'):
                            continue
                        setattr(existing_transcript, key, value)
                else:
                    # Add new transcript
                    session.add(transcript)
                
                await session.commit()
                
                self.log_with_context(
                    "Successfully updated database with transcript data",
                    extra_context={"video_id": video_id}
                )
                
        except SQLAlchemyError as e:
            self.log_with_context(
                f"Database update failed: {e}",
                level="ERROR"
            )
            # Don't raise here as the download was successful
    
    def _read_file_content(self, file_path: Optional[str]) -> Optional[str]:
        """Safely read file content."""
        if not file_path:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None


# Convenience function for direct usage
def download_transcript(video_id: str, video_url: str, channel_id: str = "unknown") -> Dict[str, Any]:
    """
    Download transcript for a single video.
    
    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
        channel_id: Channel ID for organization
        
    Returns:
        Download result dictionary
    """
    worker = DownloadWorker()
    input_data = {
        'video_id': video_id,
        'video_url': video_url,
        'channel_id': channel_id,
    }
    
    return worker.run(input_data)


if __name__ == "__main__":
    # Test with a sample video
    test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    test_url = f"https://www.youtube.com/watch?v={test_video_id}"
    
    result = download_transcript(test_video_id, test_url, "test_channel")
    print(f"Download result: {result}")