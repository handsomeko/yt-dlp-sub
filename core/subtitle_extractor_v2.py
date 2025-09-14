"""
Language-Agnostic Subtitle Extractor with Optional Translation

Extracts subtitles in ANY available language with comprehensive fallback methods.
Optionally translates to target language when requested by user.
"""

import os
import logging
import yt_dlp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
import asyncio
from contextlib import asynccontextmanager
try:
    from .transcript_cleaner import TranscriptCleaner
except ImportError:
    from transcript_cleaner import TranscriptCleaner


@dataclass
class SubtitleExtractionResult:
    """Result of subtitle extraction"""
    success: bool
    original_files: List[str]  # Original subtitle files created
    translated_files: List[str]  # Translated files (if translation enabled)
    languages_found: List[str]  # Languages detected
    methods_used: List[str]  # Extraction methods that worked
    translation_enabled: bool = False
    target_language: str = 'en'
    error_messages: List[str] = None


class LanguageAgnosticSubtitleExtractor:
    """
    Extracts subtitles in ANY available language using multiple fallback methods.
    Optionally translates to target language when requested.
    
    Extraction priority:
    1. yt-dlp with comprehensive language list
    2. youtube-transcript-api as fallback
    3. Alternative yt-dlp configurations
    
    Always preserves original language files.
    """
    
    def __init__(self, translate_enabled: bool = False, target_language: str = 'en', skip_punctuation: bool = False):
        self.logger = logging.getLogger('SubtitleExtractor')
        self.translate_enabled = translate_enabled
        self.target_language = target_language
        self.skip_punctuation = skip_punctuation  # For two-phase download strategy
        self.cleaner = TranscriptCleaner()
        
        # Initialize YouTube rate limiter
        try:
            from core.youtube_rate_limiter_unified import get_rate_limiter
            self.rate_limiter = get_rate_limiter()
        except ImportError:
            self.rate_limiter = None
            self.logger.warning("YouTube rate limiter not available")
        
        # Comprehensive language list - try to get ANY language
        self.all_languages = [
            'en', 'zh-CN', 'zh-TW', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'ru', 
            'pt', 'it', 'ar', 'hi', 'th', 'vi', 'nl', 'pl', 'tr', 'sv', 
            'da', 'no', 'fi', 'he', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk',
            'sl', 'et', 'lv', 'lt', 'mt', 'ga', 'cy', 'is', 'mk', 'sq',
            'eu', 'ca', 'gl', 'ast', 'oc', 'br', 'co', 'rm', 'sc', 'fur'
        ]
        
    def _detect_video_language(self, video_title: str, video_description: str = None) -> str:
        """
        Detect the primary language of a video based on its metadata.
        
        Args:
            video_title: Video title
            video_description: Video description (optional)
            
        Returns:
            Detected language code ('zh' for Chinese, 'en' for English, etc.)
        """
        import unicodedata
        
        # Check for Chinese characters in title and description
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
        
        # Check title
        if chinese_pattern.search(video_title):
            self.logger.info(f"Detected Chinese language from title: {video_title[:50]}...")
            return 'zh'
        
        # Check description if provided
        if video_description and chinese_pattern.search(video_description):
            self.logger.info("Detected Chinese language from description")
            return 'zh'
        
        # Check for Japanese
        japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]+')
        if japanese_pattern.search(video_title):
            self.logger.info("Detected Japanese language from title")
            return 'ja'
        
        # Check for Korean
        korean_pattern = re.compile(r'[\uac00-\ud7af\u1100-\u11ff]+')
        if korean_pattern.search(video_title):
            self.logger.info("Detected Korean language from title")
            return 'ko'
        
        # Check for Arabic
        arabic_pattern = re.compile(r'[\u0600-\u06ff\u0750-\u077f]+')
        if arabic_pattern.search(video_title):
            self.logger.info("Detected Arabic language from title")
            return 'ar'
        
        # Default to English
        return 'en'
    
    def extract_subtitles(self, video_url: str, output_dir: Path, 
                         video_id: str, video_title: str,
                         video_description: str = None) -> SubtitleExtractionResult:
        """
        Extract subtitles using comprehensive fallback methods.
        
        Args:
            video_url: YouTube video URL
            output_dir: Directory to save subtitle files (transcripts/)
            video_id: YouTube video ID
            video_title: Video title for filename
            video_description: Video description for language detection (optional)
            
        Returns:
            SubtitleExtractionResult with extraction details
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect video language
        detected_language = self._detect_video_language(video_title, video_description)
        
        result = SubtitleExtractionResult(
            success=False,
            original_files=[],
            translated_files=[],
            languages_found=[],
            methods_used=[],
            translation_enabled=self.translate_enabled,
            target_language=self.target_language,
            error_messages=[]
        )
        
        self.logger.info(f"Starting language-agnostic subtitle extraction for {video_id} (detected language: {detected_language})")
        
        # Method 1: yt-dlp with comprehensive language support
        if not result.original_files:
            self._try_ytdlp_comprehensive(video_url, output_dir, video_title, result, detected_language)
        
        # Method 2: youtube-transcript-api fallback
        if not result.original_files:
            self._try_transcript_api(video_url, video_id, output_dir, video_title, result, detected_language)
            
        # Method 3: Alternative yt-dlp configurations
        if not result.original_files:
            self._try_ytdlp_alternatives(video_url, output_dir, video_title, result, detected_language)
        
        # Process results
        if result.original_files:
            result.success = True
            result.languages_found = self._detect_languages_from_files(result.original_files)
            
            # Create TXT versions of original SRT files
            txt_files = self._create_txt_from_srt(result.original_files, output_dir)
            result.original_files.extend(txt_files)
            
            # Optional translation
            if self.translate_enabled and result.languages_found:
                self._translate_if_needed(result, output_dir, video_title)
            
            self.logger.info(f"Successfully extracted subtitles for {video_id} in languages: {result.languages_found}")
            
            # Check if we got Chinese subtitles for a Chinese video
            if detected_language == 'zh' and not any('zh' in lang for lang in result.languages_found):
                self.logger.warning(f"Chinese video but no Chinese subtitles found. Consider using Whisper transcription with Chinese language setting.")
                result.error_messages.append("Chinese subtitles unavailable - recommend Whisper transcription")
        else:
            self.logger.warning(f"All subtitle extraction methods failed for {video_id}")
            
            # For Chinese videos without any subtitles, recommend Whisper
            if detected_language == 'zh':
                self.logger.warning(f"No subtitles available for Chinese video. Recommend using Whisper with Chinese language setting for accurate transcription.")
                result.error_messages.append("No subtitles found - recommend Whisper Chinese transcription")
            
        return result
    
    def _try_ytdlp_comprehensive(self, video_url: str, output_dir: Path, 
                               video_title: str, result: SubtitleExtractionResult,
                               detected_language: str = 'en'):
        """Try yt-dlp with comprehensive language support"""
        # Apply rate limiting if available
        if self.rate_limiter:
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                import time
                time.sleep(wait_time)
        
        # Build language list based on detected language
        if detected_language == 'zh':
            # For Chinese videos, prioritize ORIGINAL Chinese subtitles FIRST!
            # NEVER prioritize English translations over native Chinese!
            priority_langs = [
                'zh-Hans',     # Native Simplified Chinese (PRIORITY #1)
                'zh-Hant',     # Native Traditional Chinese (PRIORITY #2)
                'zh-CN',       # China Chinese (PRIORITY #3)
                'zh-TW',       # Taiwan Chinese (PRIORITY #4)
                'zh',          # Generic Chinese (PRIORITY #5)
                # English translations ONLY as LAST resort:
                'zh-Hans-en',  # English translation (AVOID if Chinese available)
                'zh-Hant-en',  # English translation (AVOID if Chinese available)
                'en'           # English as absolute final fallback
            ]
        elif detected_language == 'ja':
            priority_langs = ['ja', 'ja-en', 'en']  # Japanese first, then translations
        elif detected_language == 'ko':
            priority_langs = ['ko', 'ko-en', 'en']  # Korean first, then translations
        else:
            priority_langs = ['en', detected_language] if detected_language != 'en' else ['en']
        
        configs = [
            # Try priority languages first for detected language
            {
                'writesubtitles': True,
                'writeautomaticsub': True, 
                'subtitleslangs': priority_langs,
                'subtitlesformat': 'srt'
            },
            # Try all languages with both manual and auto subtitles
            {
                'writesubtitles': True,
                'writeautomaticsub': True, 
                'subtitleslangs': ['all'],
                'subtitlesformat': 'srt'
            },
            # Try specific languages if 'all' doesn't work
            {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': self.all_languages,
                'subtitlesformat': 'srt'
            },
            # Try VTT format as fallback
            {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['all'],
                'subtitlesformat': 'vtt'
            }
        ]
        
        for i, config in enumerate(configs):
            try:
                ydl_opts = {
                    'skip_download': True,
                    'outtmpl': str(output_dir / f'{video_title}.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'retries': 3,  # Add retries for 429 errors
                    'retry_sleep_functions': {
                        'http': lambda n: 5 * (2 ** n),  # Exponential backoff: 5s, 10s, 20s
                    },
                    **config
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                # CRITICAL FIX: Track successful yt-dlp request for rate limiting stats
                if self.rate_limiter:
                    self.rate_limiter.track_request(success=True)
                
                # Check for created subtitle files
                subtitle_files = []
                for ext in ['srt', 'vtt']:
                    subtitle_files.extend(list(output_dir.glob(f'{video_title}*.{ext}')))
                
                # Filter non-empty files
                valid_files = [f for f in subtitle_files if f.stat().st_size > 0]
                
                if valid_files:
                    result.original_files.extend([str(f) for f in valid_files])
                    result.methods_used.append(f'yt-dlp-config-{i+1}')
                    self.logger.info(f"yt-dlp method {i+1} found {len(valid_files)} subtitle files")
                    return  # Success, stop trying more configs
                    
            except Exception as e:
                # CRITICAL FIX: Track failed yt-dlp request with 429 detection
                if self.rate_limiter:
                    error_str = str(e).lower()
                    is_429 = '429' in error_str or 'too many requests' in error_str
                    self.rate_limiter.track_request(success=False)
                    if is_429:
                        # Record as 429 error for circuit breaker
                        from core.rate_limit_manager import get_rate_limit_manager
                        rate_manager = get_rate_limit_manager()
                        rate_manager.record_request('youtube.com', success=False, is_429=True)
                result.error_messages.append(f"yt-dlp-config-{i+1}: {str(e)}")
                continue
    
    def _try_transcript_api(self, video_url: str, video_id: str, output_dir: Path,
                          video_title: str, result: SubtitleExtractionResult,
                          detected_language: str = 'en'):
        """Try youtube-transcript-api as fallback"""
        # Apply rate limiting if available
        if self.rate_limiter:
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                import time
                time.sleep(wait_time)
        
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import SRTFormatter
            
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # CRITICAL FIX: Track successful API call for rate limiting stats
            if self.rate_limiter:
                self.rate_limiter.track_request(success=True)
            
            # Try to get any available transcript (don't prioritize language)
            for transcript in transcript_list:
                try:
                    transcript_data = transcript.fetch()
                    language_code = transcript.language_code
                    
                    # Format as SRT
                    formatter = SRTFormatter()
                    srt_content = formatter.format_transcript(transcript_data)
                    
                    # Save with language code and _auto suffix
                    srt_file = output_dir / f'{video_title}_auto.{language_code}.srt'
                    if self._safe_write_file(srt_file, srt_content, backup=False):
                        result.original_files.append(str(srt_file))
                        result.methods_used.append('youtube-transcript-api')
                        self.logger.info(f"youtube-transcript-api found subtitle in {language_code}")
                        return  # Got one, that's enough
                    
                except Exception as e:
                    result.error_messages.append(f"transcript-api-{transcript.language_code}: {str(e)}")
                    continue
                    
        except ImportError:
            result.error_messages.append("youtube-transcript-api not installed")
        except Exception as e:
            # CRITICAL FIX: Track failed API call with 429 detection
            if self.rate_limiter:
                error_str = str(e).lower()
                is_429 = '429' in error_str or 'too many requests' in error_str
                self.rate_limiter.track_request(success=False)
                if is_429:
                    # Record as 429 error for circuit breaker
                    from core.rate_limit_manager import get_rate_limit_manager
                    rate_manager = get_rate_limit_manager()
                    rate_manager.record_request('youtube.com', success=False, is_429=True)
            result.error_messages.append(f"transcript-api-setup: {str(e)}")
    
    def _try_ytdlp_alternatives(self, video_url: str, output_dir: Path,
                              video_title: str, result: SubtitleExtractionResult,
                              detected_language: str = 'en'):
        """Try alternative yt-dlp configurations as last resort"""
        alt_configs = [
            # Minimal config - just get whatever is available
            {
                'skip_download': True,
                'writeautomaticsub': True,
                'outtmpl': str(output_dir / f'{video_title}_alt1.%(ext)s'),
                'quiet': True
            },
            # Try with different user agent
            {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'vtt',
                'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                'outtmpl': str(output_dir / f'{video_title}_alt2.%(ext)s'),
                'quiet': True
            },
            # Try JSON format as absolute last resort
            {
                'skip_download': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'json3',
                'outtmpl': str(output_dir / f'{video_title}_alt3.%(ext)s'),
                'quiet': True
            }
        ]
        
        for i, config in enumerate(alt_configs):
            try:
                # Apply rate limiting if available before alternative attempts
                if self.rate_limiter:
                    wait_time = self.rate_limiter.wait_if_needed()
                    if wait_time > 0:
                        import time
                        time.sleep(wait_time)
                
                with yt_dlp.YoutubeDL(config) as ydl:
                    ydl.download([video_url])
                
                # CRITICAL FIX: Track successful alternative yt-dlp request
                if self.rate_limiter:
                    self.rate_limiter.track_request(success=True)
                
                # Check for any subtitle files
                patterns = [f'{video_title}_alt{i+1}*']
                subtitle_files = []
                for pattern in patterns:
                    subtitle_files.extend(list(output_dir.glob(pattern)))
                
                # Filter for subtitle extensions and non-empty files
                valid_files = []
                for f in subtitle_files:
                    if f.suffix in ['.srt', '.vtt', '.json', '.json3'] and f.stat().st_size > 0:
                        valid_files.append(f)
                
                if valid_files:
                    # Convert non-SRT formats to SRT
                    converted_files = []
                    for file in valid_files:
                        if file.suffix == '.srt':
                            converted_files.append(str(file))
                        else:
                            srt_file = self._convert_to_srt(file, output_dir, video_title)
                            if srt_file:
                                converted_files.append(srt_file)
                    
                    if converted_files:
                        result.original_files.extend(converted_files)
                        result.methods_used.append(f'yt-dlp-alt-{i+1}')
                        self.logger.info(f"yt-dlp alternative {i+1} found subtitles")
                        return
                        
            except Exception as e:
                # CRITICAL FIX: Track failed alternative yt-dlp request with 429 detection
                if self.rate_limiter:
                    error_str = str(e).lower()
                    is_429 = '429' in error_str or 'too many requests' in error_str
                    self.rate_limiter.track_request(success=False)
                    if is_429:
                        # Record as 429 error for circuit breaker
                        from core.rate_limit_manager import get_rate_limit_manager
                        rate_manager = get_rate_limit_manager()
                        rate_manager.record_request('youtube.com', success=False, is_429=True)
                result.error_messages.append(f"yt-dlp-alt-{i+1}: {str(e)}")
                continue
    
    def _detect_languages_from_files(self, files: List[str]) -> List[str]:
        """Detect languages from subtitle filenames"""
        languages = set()
        
        for file_path in files:
            filename = Path(file_path).name
            
            # Look for language codes in filename (e.g., video.en.srt, video.zh-CN.srt)
            lang_pattern = r'\.([a-z]{2}(?:-[A-Z]{2})?)\.[^.]+$'
            match = re.search(lang_pattern, filename)
            
            if match:
                languages.add(match.group(1))
            else:
                # Try to detect from content or default to 'unknown'
                detected_lang = self._detect_language_from_content(file_path)
                languages.add(detected_lang)
        
        return list(languages)
    
    def _detect_language_from_content(self, file_path: str) -> str:
        """Simple language detection from subtitle content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()[:1000]  # First 1000 chars
            
            # Simple heuristics
            if re.search(r'[\u4e00-\u9fff]', content):  # Chinese characters
                return 'zh'
            elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):  # Japanese
                return 'ja'
            elif re.search(r'[\uac00-\ud7af]', content):  # Korean
                return 'ko'
            else:
                return 'unknown'
                
        except:
            return 'unknown'
    
    def _create_txt_from_srt(self, srt_files: List[str], output_dir: Path) -> List[str]:
        """Create TXT versions of SRT files"""
        txt_files = []
        
        for srt_file_path in srt_files:
            if not srt_file_path.endswith('.srt'):
                continue
                
            try:
                with open(srt_file_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                # Convert SRT to plain text
                txt_content = self._srt_to_txt(srt_content)
                
                # Create TXT filename
                srt_path = Path(srt_file_path)
                txt_file = output_dir / srt_path.name.replace('.srt', '.txt')
                
                if self._safe_write_file(txt_file, txt_content, backup=False):
                    txt_files.append(str(txt_file))
                else:
                    txt_files.append(str(txt_file))  # Still add if it exists
                
            except Exception as e:
                self.logger.warning(f"Error creating TXT from {srt_file_path}: {e}")
        
        return txt_files
    
    def _srt_to_txt(self, srt_content: str) -> str:
        """Convert SRT content to plain text"""
        # Import consolidated punctuation restorer
        from core.chinese_punctuation import ChinesePunctuationRestorer
        
        # Check if content is auto-generated and needs cleaning
        if self.cleaner.is_auto_generated(srt_content):
            # For auto-generated, use the cleaner's TXT method directly
            text = self.cleaner.clean_auto_txt(srt_content)
        else:
            # For already clean content, use standard extraction
            lines = srt_content.split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip sequence numbers, timestamps, and empty lines
                if line and not line.isdigit() and '-->' not in line:
                    text_lines.append(line)
            
            text = ' '.join(text_lines)
        
        # Check if punctuation restoration is enabled (default: True)
        import os
        if os.getenv('RESTORE_PUNCTUATION', 'true').lower() == 'true':
            # Restore punctuation if missing using consolidated system
            restorer = ChinesePunctuationRestorer()
            text, _ = restorer.restore_punctuation_sync(text)
        
        return text
    
    def _safe_write_file(self, file_path: Path, content: str, backup: bool = True) -> bool:
        """Safely write file with overwrite protection
        
        Args:
            file_path: Path to write to
            content: Content to write
            backup: If True, create backup if file exists
        
        Returns:
            True if file was written, False if skipped
        """
        try:
            if file_path.exists():
                # Check if the existing file has different content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    
                    if existing_content == content:
                        self.logger.debug(f"File {file_path} already has identical content, skipping")
                        return False
                    
                    if backup:
                        # Create backup with timestamp
                        import datetime
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        backup_path = file_path.with_suffix(f'.backup_{timestamp}{file_path.suffix}')
                        import shutil
                        shutil.copy2(file_path, backup_path)
                        self.logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Could not check existing file {file_path}: {e}")
            
            # Clean content if it's auto-generated before writing
            final_content = content
            if file_path.suffix == '.srt' and self.cleaner.is_auto_generated(content):
                final_content = self.cleaner.clean_auto_srt(content)
                self.logger.info(f"Cleaned auto-generated SRT before saving: {file_path.name}")
            elif file_path.suffix == '.txt' and self.cleaner.is_auto_generated(content):
                final_content = self.cleaner.clean_auto_txt(content)
                self.logger.info(f"Cleaned auto-generated TXT before saving: {file_path.name}")
            
            # Apply Chinese punctuation restoration if enabled for TXT and SRT files
            if file_path.suffix in ['.txt', '.srt']:
                # Note: The method will log if restoration is actually attempted
                final_content = self._apply_chinese_punctuation_restoration(final_content, file_path)
            
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            return True
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    def _apply_chinese_punctuation_restoration(self, content: str, file_path: Path) -> str:
        """
        Apply Chinese punctuation restoration if enabled and content needs it.
        
        Args:
            content: Text content to potentially restore punctuation for
            file_path: Path to file being processed (for logging)
        
        Returns:
            str: Content with restored punctuation (if applicable) or original content
        """
        try:
            # Check skip flag first (for two-phase download strategy)
            if self.skip_punctuation:
                self.logger.debug(f"Skipping Chinese punctuation restoration due to skip_punctuation flag: {file_path.name}")
                return content
            
            # Check if Chinese punctuation restoration is enabled
            from config.settings import get_settings
            settings = get_settings()
            
            if not getattr(settings, 'chinese_punctuation_enabled', False):
                self.logger.debug(f"Chinese punctuation restoration disabled")
                return content
            
            self.logger.info(f"Attempting Chinese punctuation restoration for: {file_path.name}")
            self.logger.info(f"Chinese punctuation restoration is enabled")
            
            # Use the enhanced async implementation with Claude CLI fallback
            from core.chinese_punctuation import ChinesePunctuationRestorer
            
            # Create async restorer instance (will try AI backend, fallback to Claude CLI)
            restorer = ChinesePunctuationRestorer()
            
            # Check if content needs punctuation restoration
            if not restorer.detect_chinese_text(content):
                self.logger.debug(f"No Chinese text detected in {file_path.name}")
                return content
                
            if restorer.has_punctuation(content):
                self.logger.debug(f"Chinese text already has punctuation in {file_path.name}")
                return content
            
            self.logger.info(f"Chinese text needs punctuation restoration in {file_path.name}")
            
            # Handle SRT files differently - only process subtitle text lines
            if file_path.suffix == '.srt':
                lines = content.split('\n')
                restored_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    # Copy index and timestamp lines as-is
                    if i < len(lines) and line.strip().isdigit():
                        restored_lines.append(line)  # Index
                        i += 1
                        if i < len(lines):
                            restored_lines.append(lines[i])  # Timestamp
                            i += 1
                        # Process subtitle text lines
                        subtitle_text = []
                        while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                            subtitle_text.append(lines[i])
                            i += 1
                        
                        if subtitle_text:
                            # Restore punctuation for subtitle text using async system
                            text_to_restore = ' '.join(subtitle_text)
                            if restorer.detect_chinese_text(text_to_restore) and not restorer.has_punctuation(text_to_restore):
                                # Run async restoration in sync context
                                # Use consolidated SRT-aware system with original content as hint
                                restored_text, _ = asyncio.run(restorer.restore_punctuation(text_to_restore, content))
                                if restored_text and restored_text != text_to_restore:
                                    restored_lines.append(restored_text)
                                else:
                                    restored_lines.extend(subtitle_text)
                            else:
                                restored_lines.extend(subtitle_text)
                        
                        # Add empty line between subtitles
                        if i < len(lines) and not lines[i].strip():
                            restored_lines.append(lines[i])
                            i += 1
                    else:
                        restored_lines.append(line)
                        i += 1
                
                restored_content = '\n'.join(restored_lines)
                if restored_content != content:
                    self.logger.info(f"Applied Chinese punctuation restoration to SRT: {file_path.name}")
                    return restored_content
                return content
            else:
                # For TXT files, process the entire content using consolidated system
                restored_content, _ = asyncio.run(restorer.restore_punctuation(content))
                
                if restored_content and restored_content != content:
                    self.logger.info(f"Applied Chinese punctuation restoration to: {file_path.name}")
                    return restored_content
                else:
                    return content
                
        except ImportError as e:
            self.logger.error(f"Failed to import required modules for punctuation restoration: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return content
        except Exception as e:
            self.logger.error(f"Chinese punctuation restoration failed for {file_path.name}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return content
    
    # _restore_punctuation_sync and _restore_single_chunk_sync removed
    # Now using consolidated async implementation from chinese_punctuation.py with Claude CLI fallback
    
    def _convert_to_srt(self, file_path: Path, output_dir: Path, video_title: str) -> Optional[str]:
        """Convert non-SRT subtitle formats to SRT"""
        try:
            if file_path.suffix == '.vtt':
                return self._vtt_to_srt(file_path, output_dir, video_title)
            elif file_path.suffix in ['.json', '.json3']:
                return self._json_to_srt(file_path, output_dir, video_title)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Error converting {file_path} to SRT: {e}")
            return None
    
    def _vtt_to_srt(self, vtt_file: Path, output_dir: Path, video_title: str) -> Optional[str]:
        """Convert VTT to SRT format"""
        try:
            with open(vtt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic VTT to SRT conversion
            lines = content.split('\n')
            srt_lines = []
            counter = 1
            
            for line in lines:
                if '-->' in line and 'WEBVTT' not in line:
                    # Convert timestamp format
                    timestamp = line.replace('.', ',')
                    srt_lines.append(str(counter))
                    srt_lines.append(timestamp)
                    counter += 1
                elif line.strip() and not line.startswith('WEBVTT') and not line.startswith('NOTE'):
                    srt_lines.append(line)
            
            # Detect language from original filename
            lang = self._extract_language_from_filename(vtt_file.name)
            srt_file = output_dir / f'{video_title}_auto.{lang}.srt'
            
            if self._safe_write_file(srt_file, '\n'.join(srt_lines), backup=False):
                return str(srt_file)
            return None
            
        except Exception as e:
            self.logger.warning(f"Error converting VTT to SRT: {e}")
            return None
    
    def _json_to_srt(self, json_file: Path, output_dir: Path, video_title: str) -> Optional[str]:
        """Convert JSON subtitle format to SRT"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            srt_lines = []
            counter = 1
            
            if 'events' in data:
                for event in data['events']:
                    if 'tStartMs' in event and 'dDurationMs' in event:
                        start_ms = event['tStartMs']
                        duration_ms = event['dDurationMs']
                        end_ms = start_ms + duration_ms
                        
                        start_time = self._ms_to_srt_time(start_ms)
                        end_time = self._ms_to_srt_time(end_ms)
                        
                        text = ''
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    text += seg['utf8']
                        
                        if text.strip():
                            srt_lines.extend([
                                str(counter),
                                f"{start_time} --> {end_time}",
                                text.strip(),
                                ""
                            ])
                            counter += 1
            
            if srt_lines:
                lang = self._extract_language_from_filename(json_file.name)
                srt_file = output_dir / f'{video_title}_auto.{lang}.srt'
                
                if self._safe_write_file(srt_file, '\n'.join(srt_lines), backup=False):
                    return str(srt_file)
                return None
            
        except Exception as e:
            self.logger.warning(f"Error converting JSON to SRT: {e}")
            
        return None
    
    def _ms_to_srt_time(self, ms: int) -> str:
        """Convert milliseconds to SRT time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _extract_language_from_filename(self, filename: str) -> str:
        """Extract language code from filename"""
        # Look for language codes like .en.srt, .zh-CN.vtt, etc.
        pattern = r'\.([a-z]{2}(?:-[A-Z]{2})?)\.[^.]+$'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        
        # Default fallbacks
        if 'zh' in filename.lower():
            return 'zh'
        elif 'en' in filename.lower():
            return 'en'
        else:
            return 'auto'  # Auto-generated
    
    def _translate_if_needed(self, result: SubtitleExtractionResult, 
                           output_dir: Path, video_title: str):
        """Translate subtitles if translation is enabled and needed"""
        if not self.translate_enabled:
            return
            
        # Check if we already have target language
        if self.target_language in result.languages_found:
            self.logger.info(f"Target language {self.target_language} already available, skipping translation")
            return
        
        # Find best SRT file to translate
        srt_files = [f for f in result.original_files if f.endswith('.srt')]
        if not srt_files:
            return
        
        # Use the first SRT file found
        source_srt = srt_files[0]
        
        try:
            # Create output files
            translated_srt = output_dir / f'{video_title}_en.srt'
            translated_txt = output_dir / f'{video_title}_en.txt'
            
            # Load source content
            with open(source_srt, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            # Translate using AI backend
            translated_content = self._translate_with_ai(srt_content, self.target_language)
            
            if translated_content:
                # Save translated SRT
                if self._safe_write_file(translated_srt, translated_content, backup=True):
                    # Create translated TXT
                    txt_content = self._srt_to_txt(translated_content)
                    self._safe_write_file(translated_txt, txt_content, backup=True)
                
                result.translated_files.extend([str(translated_srt), str(translated_txt)])
                self.logger.info(f"Successfully translated subtitles to {self.target_language}")
            else:
                # Fallback: copy original if translation fails
                import shutil
                shutil.copy2(source_srt, translated_srt)
                txt_content = self._srt_to_txt(srt_content)
                self._safe_write_file(translated_txt, txt_content, backup=False)
                
                result.translated_files.extend([str(translated_srt), str(translated_txt)])
                result.error_messages.append("Translation failed, using original content")
                self.logger.warning("Translation failed, using original content")
            
        except Exception as e:
            result.error_messages.append(f"Translation failed: {str(e)}")
            self.logger.warning(f"Translation failed: {e}")
    
    def _translate_with_ai(self, srt_content: str, target_language: str) -> Optional[str]:
        """Translate SRT content using AI backend"""
        try:
            # Import here to avoid circular imports
            from workers.ai_backend import AIBackend
            
            # Run async translation in sync context
            return asyncio.run(self._async_translate_with_ai(srt_content, target_language))
            
        except ImportError:
            self.logger.warning("AI backend not available for translation")
            return None
        except Exception as e:
            self.logger.warning(f"Translation error: {e}")
            return None
    
    async def _async_translate_with_ai(self, srt_content: str, target_language: str) -> Optional[str]:
        """Async translation method"""
        try:
            from workers.ai_backend import AIBackend
            
            ai_backend = AIBackend()
            
            # Use the new translate_content method
            result = await ai_backend.translate_content(
                content=srt_content[:3000],  # Limit to avoid token limits
                target_language=target_language,
                source_language=None  # Auto-detect
            )
            
            if result and result.get('success'):
                return result.get('translated_content', '')
            else:
                self.logger.warning(f"Translation failed: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Async translation failed: {e}")
            return None


# Convenience function
def extract_subtitles_any_language(video_url: str, output_dir: Path, video_id: str, 
                                 video_title: str, translate: bool = False, 
                                 target_lang: str = 'en') -> SubtitleExtractionResult:
    """Extract subtitles in any available language with optional translation"""
    extractor = LanguageAgnosticSubtitleExtractor(
        translate_enabled=translate, 
        target_language=target_lang
    )
    return extractor.extract_subtitles(video_url, output_dir, video_id, video_title)