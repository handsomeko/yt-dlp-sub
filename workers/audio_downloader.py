"""
AudioDownloadWorker - Downloads audio from YouTube videos for transcription
Handles only the download part, transcription is done by TranscribeWorker
Now uses YouTubeDownloader for full V2 feature support
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import re
from datetime import datetime

from workers.base import BaseWorker, WorkerStatus
from config.settings import get_settings
from core.storage_paths_v2 import get_storage_paths_v2
from core.filename_sanitizer import sanitize_filename
from core.downloader import YouTubeDownloader
from core.youtube_validators import is_valid_youtube_id

logger = logging.getLogger(__name__)


class AudioDownloadWorker(BaseWorker):
    """
    Worker that downloads audio from YouTube videos
    Uses yt-dlp to extract audio in Opus format for efficient storage
    """
    
    def __init__(self):
        super().__init__("audio_downloader")
        self.settings = get_settings()
        self.storage_paths = get_storage_paths_v2()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ['video_id', 'video_url']
        optional = ['channel_id', 'format', 'quality']
        
        if not all(field in input_data for field in required):
            self.log_with_context(
                logging.ERROR,
                f"Missing required fields. Required: {required}",
                extra={"input_fields": list(input_data.keys())}
            )
            return False
        
        # Validate video ID format (11 characters)
        video_id = input_data['video_id']
        if not is_valid_youtube_id(video_id):
            self.log_with_context(
                logging.ERROR,
                f"Invalid video ID format: {video_id}"
            )
            return False
        
        # Validate YouTube URL
        video_url = input_data['video_url']
        if not re.match(r'https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)', video_url):
            self.log_with_context(
                logging.ERROR,
                f"Invalid YouTube URL: {video_url}"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download audio from YouTube video using YouTubeDownloader for V2 features
        
        Args:
            input_data: Must contain video_id, video_url, optionally channel_id
        
        Returns:
            Dict with download status and file paths
        """
        video_id = input_data['video_id']
        video_url = input_data['video_url']
        channel_id = input_data.get('channel_id', 'unknown')
        audio_format = input_data.get('format', 'opus')  # Default to opus for efficiency
        
        self.log_with_context(
            logging.INFO,
            f"Starting audio download for video {video_id}"
        )
        
        # Use YouTubeDownloader with settings for full V2 feature support
        # Check if job metadata includes translation settings
        translate_enabled = input_data.get('subtitle_translation_enabled', False)
        target_language = input_data.get('subtitle_target_language', 'en')
        
        # Create downloader with explicit settings if provided, otherwise use defaults
        from core.downloader import YouTubeDownloader, create_downloader_with_settings
        if 'subtitle_translation_enabled' in input_data:
            # Use explicit settings from job metadata
            downloader = YouTubeDownloader(
                enable_translation=translate_enabled,
                target_language=target_language
            )
        else:
            # Use default settings from environment
            downloader = create_downloader_with_settings()
        
        try:
            # Download with all V2 features:
            # - Comprehensive metadata (~60 fields)
            # - Correct file naming ({video_title}.ext)
            # - video_url.txt creation
            # - Markdown reports
            # - Automatic subtitles (writeautomaticsub)
            result = downloader.download_video(
                url=video_url,
                download_audio_only=True,
                audio_format=audio_format,
                channel_name=input_data.get('channel_name')
            )
            
            if result.get('status') == 'success':
                # Extract video title for response
                video_title = result.get('title', 'Unknown')
                
                # Find the audio file from the downloaded files
                audio_files = [f for f in result.get('files', []) 
                              if any(f.endswith(ext) for ext in ['.opus', '.mp3', '.m4a'])]
                audio_path = audio_files[0] if audio_files else None
                
                self.log_with_context(
                    logging.INFO,
                    f"Successfully downloaded audio for {video_id}: {video_title}",
                    extra={
                        'output_dir': result.get('output_dir'),
                        'files_count': len(result.get('files', [])),
                        'video_title': video_title
                    }
                )
                
                # Note: YouTubeDownloader already handles all metadata saving,
                # video_url.txt creation, markdown reports, etc.
                # No need to duplicate that work here
                
                return {
                    'status': WorkerStatus.SUCCESS,
                    'video_id': video_id,
                    'channel_id': result.get('channel_id', channel_id),
                    'audio_path': audio_path,
                    'output_dir': result.get('output_dir'),
                    'video_title': video_title,  # Pass title to next worker
                    'title_sanitized': sanitize_filename(video_title, video_id),  # For database
                    'files': result.get('files', []),
                    'download_method': 'YouTubeDownloader',
                    'storage_structure': 'v2',
                    'subtitle_result': result.get('subtitle_result'),  # Pass subtitle extraction info
                    'languages_found': result.get('subtitle_result', {}).get('languages_found', []),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                raise Exception(result.get('error', 'Download failed'))
                
        except Exception as e:
            self.log_with_context(
                logging.ERROR,
                f"Failed to download audio for {video_id}: {str(e)}"
            )
            return {
                'status': WorkerStatus.FAILED,
                'video_id': video_id,
                'error': str(e),
                'error_details': self.handle_error(e)
            }
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Categorize and handle different error types"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str:
            return {
                'error_code': 'E001',
                'error_type': 'timeout',
                'message': 'Download timeout',
                'recoverable': True,
                'retry_delay': 60
            }
        elif '429' in error_str or 'rate' in error_str:
            return {
                'error_code': 'E002',
                'error_type': 'rate_limit',
                'message': 'YouTube rate limit hit',
                'recoverable': True,
                'retry_delay': 300  # 5 minutes
            }
        elif 'private' in error_str or 'not available' in error_str:
            return {
                'error_code': 'E003',
                'error_type': 'video_unavailable',
                'message': 'Video is private or unavailable',
                'recoverable': False
            }
        elif 'age-restricted' in error_str:
            return {
                'error_code': 'E008',
                'error_type': 'age_restricted',
                'message': 'Video is age-restricted',
                'recoverable': False
            }
        elif 'no space left' in error_str or 'disk full' in error_str:
            return {
                'error_code': 'E006',
                'error_type': 'storage_full',
                'message': 'Storage space full',
                'recoverable': False,
                'action_required': 'Free up disk space'
            }
        else:
            return {
                'error_code': 'E999',
                'error_type': 'unknown',
                'message': str(error),
                'recoverable': True,
                'retry_delay': 120
            }


# Convenience function for direct use
async def download_audio(video_id: str, video_url: str, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to download audio from a video
    
    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
        channel_id: Optional channel ID for organization
    
    Returns:
        Dict with download result, audio path, and video title
    """
    worker = AudioDownloadWorker()
    return worker.run({
        'video_id': video_id,
        'video_url': video_url,
        'channel_id': channel_id or 'unknown'
    })