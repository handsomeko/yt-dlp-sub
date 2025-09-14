"""Transcript Extractor Module - Fast subtitle/transcript extraction"""

import re
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter, SRTFormatter

# CRITICAL FIX: Import rate limiting to prevent 429 errors
from .rate_limit_manager import get_rate_limit_manager

logger = logging.getLogger(__name__)


class TranscriptExtractor:
    """Modular transcript extractor using youtube-transcript-api"""
    
    def __init__(self, base_path: str = None):
        """Initialize with configurable output path"""
        if base_path is None:
            if os.path.exists('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub'):
                base_path = '/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads'
            else:
                base_path = os.path.join(os.path.dirname(__file__), '../downloads')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        from core.youtube_validators import is_valid_youtube_id
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                # Validate before returning
                if is_valid_youtube_id(video_id):
                    return video_id
                else:
                    logger.warning(f"Extracted invalid video ID: {video_id} from URL: {url}")
        return None
    
    def get_available_languages(self, video_id: str) -> List[str]:
        """Get list of available transcript languages"""
        # CRITICAL FIX: Add rate limiting protection before YouTube API call
        rate_manager = get_rate_limit_manager()
        allowed, wait_time = rate_manager.should_allow_request('youtube.com')
        if not allowed:
            logger.warning(f"Rate limited - waiting {wait_time:.1f}s before transcript language list for {video_id}")
            time.sleep(wait_time)
        
        try:
            transcript_list = YouTubeTranscriptApi.list(video_id)
            
            # Record successful API call for rate limiting stats
            rate_manager.record_request('youtube.com', success=True)
            languages = []
            
            # Manual transcripts
            for transcript in transcript_list._manually_created_transcripts.values():
                languages.append({
                    'code': transcript.language_code,
                    'name': transcript.language,
                    'is_generated': False
                })
            
            # Auto-generated transcripts
            for transcript in transcript_list._generated_transcripts.values():
                languages.append({
                    'code': transcript.language_code,
                    'name': transcript.language,
                    'is_generated': True
                })
                
            return languages
        except Exception as e:
            # CRITICAL FIX: Record failed API call with 429 detection
            rate_manager = get_rate_limit_manager()
            error_str = str(e).lower()
            is_429 = '429' in error_str or 'too many requests' in error_str
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            return []
    
    def get_transcript(self, 
                      video_url_or_id: str,
                      languages: List[str] = None,
                      save_to_file: bool = True,
                      video_title: Optional[str] = None) -> Dict[str, Any]:
        """Get transcript in multiple formats"""
        
        # Extract video ID if URL provided
        if 'youtube.com' in video_url_or_id or 'youtu.be' in video_url_or_id:
            video_id = self.extract_video_id(video_url_or_id)
        else:
            video_id = video_url_or_id
            
        if not video_id:
            return {'status': 'error', 'error': 'Invalid YouTube URL or video ID'}
        
        if languages is None:
            languages = ['en']
        
        # CRITICAL FIX: Add rate limiting protection before YouTube API call
        rate_manager = get_rate_limit_manager()
        allowed, wait_time = rate_manager.should_allow_request('youtube.com')
        if not allowed:
            logger.warning(f"Rate limited - waiting {wait_time:.1f}s before transcript fetch for {video_id}")
            time.sleep(wait_time)
            
        try:
            # Get transcript (using fetch for new API version)
            transcript_data = YouTubeTranscriptApi.fetch(video_id, languages=languages)
            
            # CRITICAL FIX: Record successful API call for rate limiting stats
            rate_manager.record_request('youtube.com', success=True)
            
            # Format as plain text
            text_formatter = TextFormatter()
            plain_text = text_formatter.format_transcript(transcript_data)
            
            # Format as SRT
            srt_formatter = SRTFormatter()
            srt_text = srt_formatter.format_transcript(transcript_data)
            
            result = {
                'status': 'success',
                'video_id': video_id,
                'language': languages[0] if languages else 'en',
                'transcript_entries': len(transcript_data),
                'plain_text': plain_text,
                'srt_text': srt_text,
                'raw_data': transcript_data
            }
            
            # Save to files if requested
            if save_to_file:
                if not video_title:
                    video_title = f"video_{video_id}"
                else:
                    video_title = self._sanitize_filename(video_title)
                    
                output_dir = self.base_path / video_title
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save plain text
                txt_file = output_dir / f"{video_title}.txt"
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                result['text_file'] = str(txt_file)
                
                # Save SRT
                srt_file = output_dir / f"{video_title}.en.srt"
                with open(srt_file, 'w', encoding='utf-8') as f:
                    f.write(srt_text)
                result['srt_file'] = str(srt_file)
                
            return result
            
        except TranscriptsDisabled:
            # CRITICAL FIX: Record API failure for rate limiting stats  
            rate_manager = get_rate_limit_manager()
            rate_manager.record_request('youtube.com', success=False, is_429=False)
            return {
                'status': 'error',
                'error': 'Transcripts are disabled for this video',
                'video_id': video_id
            }
        except NoTranscriptFound:
            # CRITICAL FIX: Record API failure for rate limiting stats
            rate_manager = get_rate_limit_manager()
            rate_manager.record_request('youtube.com', success=False, is_429=False)
            return {
                'status': 'error',
                'error': f'No transcript found for languages: {languages}',
                'video_id': video_id,
                'available_languages': self.get_available_languages(video_id)
            }
        except Exception as e:
            # CRITICAL FIX: Record API failure with 429 detection for rate limiting stats
            rate_manager = get_rate_limit_manager()
            error_str = str(e).lower()
            is_429 = '429' in error_str or 'too many requests' in error_str
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            return {
                'status': 'error',
                'error': str(e),
                'video_id': video_id
            }
    
    def convert_srt_to_text(self, srt_file_path: str) -> str:
        """Convert existing SRT file to plain text"""
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            
        # Remove timestamps and numbers
        lines = srt_content.split('\n')
        text_lines = []
        
        for line in lines:
            # Skip numbers and timestamps
            if not line.strip():
                continue
            if line.strip().isdigit():
                continue
            if '-->' in line:
                continue
            text_lines.append(line.strip())
            
        # Join lines
        text = ' '.join(text_lines)
        
        # Restore punctuation if enabled - using consolidated system
        import os
        if os.getenv('RESTORE_PUNCTUATION', 'true').lower() == 'true':
            from core.chinese_punctuation import ChinesePunctuationRestorer
            restorer = ChinesePunctuationRestorer()
            text, _ = restorer.restore_punctuation_sync(text)
            
        return text
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]