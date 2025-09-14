"""
TranscribeWorker - Handles transcript extraction from downloaded audio/video
Uses multiple methods with fallback chain for reliability
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from workers.base import BaseWorker, WorkerStatus
from config.settings import get_settings
from core.storage_paths_v2 import get_storage_paths_v2
from core.filename_sanitizer import sanitize_filename
from core.rate_limit_manager import get_rate_limit_manager

logger = logging.getLogger(__name__)


# Custom Exception Classes for Better Error Handling
class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class WhisperNotInstalledError(TranscriptionError):
    """Raised when Whisper is not installed."""
    pass


class AudioFileError(TranscriptionError):
    """Raised when there's an issue with the audio file."""
    pass


class WhisperModelError(TranscriptionError):
    """Raised when Whisper model fails to load or process."""
    pass


class FFmpegError(TranscriptionError):
    """Raised when FFmpeg command fails."""
    pass


class APITimeoutError(TranscriptionError):
    """Raised when external API calls timeout."""
    pass


class TranscriptQualityError(TranscriptionError):
    """Raised when transcript quality is too low."""
    pass


class TranscribeWorker(BaseWorker):
    """
    Worker that handles transcript extraction from audio/video files
    
    Fallback chain:
    1. FFmpeg af_whisper (if FFmpeg 8.0+ available) - Future implementation
    2. Whisper local (Python) - Future implementation  
    3. yt-dlp subtitle extraction
    4. youtube-transcript-api
    """
    
    def __init__(self):
        super().__init__("transcriber")
        self.settings = get_settings()
        self.storage_paths = get_storage_paths_v2()
        self.supported_methods = [
            'whisper_local',      # Primary: Local Whisper transcription
            'ffmpeg',             # Secondary: FFmpeg (if available)
            'yt_dlp_subs',        # Fallback: Extract existing subtitles
            'youtube_transcript_api',  # Last resort: API-based
        ]
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ['video_id', 'video_url']
        
        if not all(field in input_data for field in required):
            self.log_with_context(
                f"Missing required fields. Required: {required}",
                level="ERROR",
                extra_context={"input_fields": list(input_data.keys())}
            )
            return False
        
        # Validate video ID format
        video_id = input_data['video_id']
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            self.log_with_context(
                f"Invalid video ID format: {video_id}",
                level="ERROR"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcript extraction with hybrid strategy:
        1. Check for existing high-quality auto-generated subtitles first
        2. Use Whisper as fallback only when auto-generated subtitles unavailable/poor quality
        
        Args:
            input_data: Must contain video_id, video_url, and optionally audio_path
        
        Returns:
            Dict with transcript content and metadata
        """
        video_id = input_data['video_id']
        video_url = input_data['video_url']
        channel_id = input_data.get('channel_id', 'unknown')
        audio_path = input_data.get('audio_path')  # Optional, from DownloadWorker
        video_title = input_data.get('video_title', 'Unknown')  # From DownloadWorker
        
        self.log_with_context(
            f"Starting hybrid transcript extraction for video {video_id}: {video_title}",
            level="INFO"
        )
        
        # STEP 1: Check for existing high-quality auto-generated subtitles
        existing_subtitles = self._check_existing_subtitles(channel_id, video_id, video_title)
        if existing_subtitles:
            self.log_with_context(
                f"Found high-quality auto-generated subtitles, skipping Whisper transcription",
                level="INFO",
                extra_context={
                    "subtitle_files": existing_subtitles['files'],
                    "languages": existing_subtitles.get('languages_found', []),
                    "extraction_method": "auto_generated_cleaned"
                }
            )
            return existing_subtitles
        
        # STEP 2: No suitable auto-generated subtitles found, use Whisper as fallback
        self.log_with_context(
            f"No suitable auto-generated subtitles found, falling back to Whisper transcription",
            level="INFO"
        )
        
        # If no audio_path provided, try to find existing audio files (V2 storage compatibility)
        if not audio_path:
            audio_files = self.storage_paths.find_media_files(channel_id, video_id)
            if audio_files:
                audio_path = str(audio_files[0])  # Use first found
                self.log_with_context(
                    f"Found existing audio file: {audio_path}",
                    level="INFO"
                )
        
        # Try each method in order
        errors = []
        
        for method in self.supported_methods:
            try:
                self.log_with_context(
                    f"Attempting transcript extraction with {method}",
                    level="INFO"
                )
                
                if method == 'whisper_local' and audio_path:
                    result = self._extract_with_whisper_local(audio_path, video_id)
                elif method == 'ffmpeg' and audio_path:
                    result = self._extract_with_ffmpeg(audio_path, video_id)
                elif method == 'yt_dlp_subs':
                    result = self._extract_with_yt_dlp(video_url, video_id)
                elif method == 'youtube_transcript_api':
                    result = self._extract_with_transcript_api(video_id)
                else:
                    continue
                
                if result and result.get('transcript'):
                    # Save transcript files with readable filenames (V2 storage)
                    transcript_dir = self.storage_paths.get_transcript_dir(channel_id, video_id)
                    
                    # Generate readable filenames
                    safe_title = sanitize_filename(video_title, video_id)
                    
                    # Save SRT file with readable name and _whisper suffix
                    srt_path = transcript_dir / f"{safe_title}_whisper.srt"
                    srt_path.write_text(result.get('transcript_srt', ''), encoding='utf-8')
                    
                    # Save plain text file with readable name and _whisper suffix
                    txt_path = transcript_dir / f"{safe_title}_whisper.txt"
                    original_text = result.get('transcript', '')
                    
                    # Apply Chinese punctuation restoration if enabled
                    settings = get_settings()
                    if getattr(settings, 'chinese_punctuation_enabled', False):
                        try:
                            from core.chinese_punctuation import ChinesePunctuationRestorer
                            from workers.ai_backend import AIBackend
                            import asyncio
                            
                            self.log_with_context(
                                f"Applying Chinese punctuation restoration to transcript",
                                level="INFO"
                            )
                            
                            # Initialize punctuation restorer
                            ai_backend = AIBackend()
                            restorer = ChinesePunctuationRestorer(ai_backend)
                            
                            # Run async punctuation restoration in sync context using consolidated SRT-aware system
                            async def restore_punctuation_async():
                                # Get SRT content if available for SRT-aware processing
                                srt_content = result.get('transcript_srt', '')
                                if not srt_content and srt_path.exists():
                                    srt_content = srt_path.read_text(encoding='utf-8')
                                
                                # Use consolidated restore_punctuation with SRT-aware support
                                restored_text, success = await restorer.restore_punctuation(original_text, srt_content)
                                
                                if success:
                                    txt_path.write_text(restored_text, encoding='utf-8')
                                    self.log_with_context(
                                        f"Successfully restored Chinese punctuation for {video_title} using {'SRT-aware' if srt_content else 'text-only'} processing",
                                        level="INFO"
                                    )
                                    
                                    # If we used SRT-aware processing, also update SRT file with punctuation
                                    if srt_content:
                                        try:
                                            restored_srt, srt_success = await restorer.restore_srt_punctuation_aware(srt_content)
                                            if srt_success:
                                                srt_path.write_text(restored_srt, encoding='utf-8')
                                                self.log_with_context(f"Updated SRT file with punctuation for {video_title}", level="INFO")
                                        except Exception as e:
                                            self.log_with_context(f"SRT update failed but TXT successful: {e}", level="WARNING")
                                else:
                                    txt_path.write_text(original_text, encoding='utf-8')
                                    self.log_with_context(f"Punctuation restoration unsuccessful for {video_title}", level="WARNING")
                                
                                return success
                            
                            # Run the async function
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # If we're already in an async context, create a new task
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, restore_punctuation_async())
                                    future.result()
                            else:
                                asyncio.run(restore_punctuation_async())
                        
                        except Exception as e:
                            self.log_with_context(
                                f"Error during Chinese punctuation restoration: {str(e)}",
                                level="WARNING"
                            )
                            # Fallback to original text
                            txt_path.write_text(original_text, encoding='utf-8')
                    else:
                        txt_path.write_text(original_text, encoding='utf-8')
                    
                    # Update channel index (V2 storage)
                    if hasattr(self.storage_paths, 'update_video_index'):
                        self.storage_paths.update_video_index(
                            channel_id=channel_id,
                            video_id=video_id,
                            video_info={
                                'title': video_title,
                                'has_transcript': True
                            }
                        )
                    
                    # Add paths and metadata to result
                    result.update({
                        'status': WorkerStatus.SUCCESS,
                        'video_id': video_id,
                        'channel_id': channel_id,
                        'video_title': video_title,  # Pass title to next worker
                        'title_sanitized': safe_title,  # For database
                        'extraction_method': method,
                        'srt_path': str(srt_path),
                        'txt_path': str(txt_path),
                        'transcript_dir': str(transcript_dir),
                        'language': result.get('language', 'en'),  # Pass detected language
                        'languages_found': [result.get('language', 'en')]  # For consistency
                    })
                    
                    self.log_with_context(
                        f"Successfully extracted and saved transcript with {method}",
                        level="INFO",
                        extra_context={
                            "word_count": result.get('word_count', 0),
                            "srt_path": str(srt_path),
                            "txt_path": str(txt_path)
                        }
                    )
                    return result
                    
            except (WhisperNotInstalledError, FFmpegError) as e:
                # Critical errors that suggest method is not available
                error_msg = f"{method} unavailable: {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Method {method} unavailable, continuing with fallback",
                    level="WARNING",
                    extra_context={"method": method, "error": str(e), "error_type": type(e).__name__}
                )
                continue
                
            except (AudioFileError, APITimeoutError) as e:
                # Retry-able errors
                error_msg = f"{method} failed (retryable): {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Retryable error in {method}, continuing with fallback", 
                    level="WARNING",
                    extra_context={"method": method, "error": str(e), "error_type": type(e).__name__}
                )
                continue
                
            except TranscriptQualityError as e:
                # Quality issues - log but continue trying other methods
                error_msg = f"{method} quality issue: {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Quality issue with {method}, trying next fallback",
                    level="WARNING", 
                    extra_context={"method": method, "error": str(e)}
                )
                continue
                
            except TranscriptionError as e:
                # Generic transcription errors
                error_msg = f"{method} transcription error: {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Transcription error in {method}, trying fallback",
                    level="ERROR",
                    extra_context={"method": method, "error": str(e)}
                )
                continue
                
            except Exception as e:
                # Unexpected errors
                error_msg = f"{method} unexpected error: {str(e)}"
                errors.append(error_msg)
                self.log_with_context(
                    f"Unexpected error in {method}",
                    level="ERROR", 
                    extra_context={"method": method, "error": str(e), "error_type": type(e).__name__}
                )
                continue
        
        # All methods failed
        return {
            'status': WorkerStatus.FAILED,
            'error': 'No transcript extraction method succeeded',
            'errors': errors,
            'video_id': video_id
        }
    
    def _check_existing_subtitles(self, channel_id: str, video_id: str, video_title: str) -> Optional[Dict[str, Any]]:
        """
        Check for existing high-quality auto-generated subtitles from LanguageAgnosticSubtitleExtractor.
        
        Args:
            channel_id: YouTube channel ID
            video_id: YouTube video ID  
            video_title: Video title for filename generation
            
        Returns:
            Dict with subtitle info if high-quality subtitles exist, None otherwise
        """
        try:
            # Get transcript directory
            transcript_dir = self.storage_paths.get_transcript_dir(channel_id, video_id)
            if not transcript_dir.exists():
                return None
            
            # CRITICAL: Use same naming patterns as LanguageAgnosticSubtitleExtractor
            # The subtitle extractor uses RAW video_title, not sanitized version
            
            # Look for auto-generated subtitle files (not _whisper suffixed files)
            # Patterns from LanguageAgnosticSubtitleExtractor:
            # 1. {video_title}.{lang}.srt (direct yt-dlp)  
            # 2. {video_title}_auto.{lang}.srt (transcript-api and converted files)
            # 3. {video_title}_alt1.{lang}.srt, etc. (alternative methods)
            # 4. {video_title}_en.srt (translated files)
            
            subtitle_files = []
            languages_found = []
            
            # Search for subtitle files using the same patterns as LanguageAgnosticSubtitleExtractor
            search_patterns = [
                f"{video_title}*.srt",        # Direct yt-dlp pattern  
                f"{video_title}_auto*.srt",   # Auto-generated pattern
                f"{video_title}_alt*.srt",    # Alternative pattern
                f"{video_title}_en.srt"       # Translated pattern
            ]
            
            for pattern in search_patterns:
                for file_path in transcript_dir.glob(pattern):
                    file_name = file_path.name
                    
                    # Skip Whisper-generated files  
                    if '_whisper' in file_name:
                        continue
                
                    # Extract language code using the same logic as LanguageAgnosticSubtitleExtractor
                    lang_code = self._extract_language_from_filename(file_name)
                    
                    if lang_code:
                        languages_found.append(lang_code)
                        subtitle_files.append(str(file_path))
                        
                        # Also check for corresponding TXT file
                        txt_path = file_path.with_suffix('.txt')
                        if txt_path.exists():
                            subtitle_files.append(str(txt_path))
            
            # Validate subtitle quality
            if not subtitle_files:
                self.log_with_context(
                    f"No auto-generated subtitle files found for {video_id}",
                    level="DEBUG"
                )
                return None
                
            # Check file sizes (cleaned files should be reasonable size)
            for file_path in subtitle_files:
                file_size = Path(file_path).stat().st_size
                if file_size < 50:  # Too small, likely empty or corrupted
                    self.log_with_context(
                        f"Subtitle file too small ({file_size} bytes), quality check failed: {file_path}",
                        level="DEBUG"
                    )
                    return None
            
            # Return existing subtitle information in same format as Whisper results
            # NOTE: We use sanitized title for consistency with Whisper output format
            safe_title = sanitize_filename(video_title, video_id)
            
            return {
                'success': True,  # CRITICAL: Add success field for BatchTranscriber compatibility
                'status': WorkerStatus.SUCCESS,
                'video_id': video_id,
                'channel_id': channel_id,
                'video_title': video_title,
                'title_sanitized': safe_title,
                'extraction_method': 'auto_generated_cleaned',
                'transcript_dir': str(transcript_dir),
                'languages_found': languages_found,
                'files': subtitle_files,
                'transcript': self._read_first_txt_file(subtitle_files),  # For compatibility
                'language': languages_found[0] if languages_found else 'unknown'
            }
            
        except Exception as e:
            self.log_with_context(
                f"Error checking existing subtitles for {video_id}: {str(e)}",
                level="WARNING"
            )
            return None
    
    def _read_first_txt_file(self, file_list: List[str]) -> str:
        """Read content of first .txt file for compatibility with existing workflow"""
        try:
            for file_path in file_list:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read txt file: {e}")
        return ""
    
    def _extract_language_from_filename(self, filename: str) -> str:
        """
        Extract language code from filename using same logic as LanguageAgnosticSubtitleExtractor.
        
        Handles patterns:
        - video_title.lang.srt (e.g., "Video.en.srt")  
        - video_title_auto.lang.srt (e.g., "Video_auto.zh.srt")
        - video_title_alt1.lang.srt (e.g., "Video_alt1.es.srt") 
        - video_title_en.srt (translated files)
        """
        import re
        
        # Pattern 1: Standard language codes like .en.srt, .zh-CN.srt, etc.
        pattern = r'\.([a-z]{2}(?:-[A-Z]{2})?)\.[^.]+$'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        
        # Pattern 2: _auto.lang.srt format
        if '_auto.' in filename:
            parts = filename.split('_auto.', 1)
            if len(parts) == 2:
                lang_part = parts[1].replace('.srt', '').replace('.txt', '')
                # Validate it looks like a language code
                if len(lang_part) >= 2 and len(lang_part) <= 10 and lang_part.isalpha():
                    return lang_part
        
        # Pattern 3: _altN.lang.srt format  
        alt_pattern = r'_alt\d+\.([a-z]{2}(?:-[A-Z]{2})?)\.'
        match = re.search(alt_pattern, filename)
        if match:
            return match.group(1)
        
        # Pattern 4: Translated files _en.srt
        if filename.endswith('_en.srt') or filename.endswith('_en.txt'):
            return 'en'
        
        # Fallback patterns for common languages (use word boundaries to avoid false matches)
        import re
        filename_lower = filename.lower()
        
        # Use word boundaries to avoid false matches like "video" containing "de"
        if 'chinese' in filename_lower or re.search(r'\bzh\b', filename_lower):
            return 'zh'
        elif 'spanish' in filename_lower or re.search(r'\bes\b', filename_lower):
            return 'es'
        elif 'french' in filename_lower or re.search(r'\bfr\b', filename_lower):
            return 'fr'
        elif 'german' in filename_lower or re.search(r'\bde\b', filename_lower):
            return 'de'
        elif 'english' in filename_lower or re.search(r'\ben\b', filename_lower):
            return 'en'
        else:
            return 'auto'  # Auto-generated, language unknown
    
    def _extract_with_whisper_local(self, audio_path: str, video_id: str, 
                                   video_title: str = None, video_description: str = None) -> Dict[str, Any]:
        """
        Extract transcript using local Whisper installation with timeout protection.
        
        Args:
            audio_path: Path to audio file
            video_id: Video ID for reference
            video_title: Video title for language detection (optional)
            video_description: Video description for language detection (optional)
            
        Returns:
            Dict with transcript and metadata
        """
        try:
            import whisper
            from core.audio_analyzer import AudioAnalyzer
            from core.whisper_timeout_manager import WhisperTimeoutManager
            
            # Pre-flight audio analysis
            analyzer = AudioAnalyzer()
            analysis = analyzer.analyze_audio(
                audio_path,
                whisper_timeout_base=self.settings.whisper_timeout_base,
                whisper_timeout_per_minute=self.settings.whisper_timeout_per_minute,
                whisper_max_duration=self.settings.whisper_max_duration,
                whisper_chunk_duration=self.settings.whisper_chunk_duration,
                whisper_model=self.settings.whisper_model
            )
            
            # Validate audio is suitable for processing
            is_valid, error_msg = analyzer.validate_for_whisper(analysis)
            if not is_valid:
                self.log_with_context(
                    f"Audio validation failed: {error_msg}",
                    level="ERROR",
                    extra_context={"audio_path": audio_path, "analysis": analysis.__dict__}
                )
                raise AudioFileError(f"Audio validation failed: {error_msg}")
            
            self.log_with_context(
                f"Pre-flight analysis complete: {analysis.duration_seconds:.1f}s audio, "
                f"timeout: {analysis.recommended_timeout}s, "
                f"memory estimate: {analysis.estimated_memory_mb}MB, "
                f"chunks needed: {analysis.requires_chunking}",
                level="INFO"
            )
            
            # Initialize timeout manager
            timeout_manager = WhisperTimeoutManager(
                max_concurrent_jobs=self.settings.whisper_max_concurrent,
                memory_limit_mb=self.settings.whisper_memory_limit_mb
            )
            
            # Check if system can handle this job
            can_start, resource_error = timeout_manager.can_start_new_job()
            if not can_start:
                self.log_with_context(
                    f"Cannot start Whisper job: {resource_error}",
                    level="ERROR",
                    extra_context={"resource_status": timeout_manager.get_system_resource_status()}
                )
                raise WhisperModelError(f"Resource limit exceeded: {resource_error}")
            
            # Handle chunking for long audio
            if analysis.requires_chunking and self.settings.whisper_enable_chunking:
                return self._transcribe_with_chunking(
                    audio_path, video_id, analysis, timeout_manager
                )
            
            # Regular transcription for shorter audio
            self.log_with_context(
                f"Loading Whisper model: {self.settings.whisper_model}",
                level="INFO"
            )
            
            # Detect video language from title/description if provided
            detected_language = None
            if video_title:
                # Check for Chinese characters
                chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
                if chinese_pattern.search(video_title):
                    detected_language = 'zh'
                    self.log_with_context(f"Detected Chinese language from title", level="INFO")
                elif video_description and chinese_pattern.search(video_description):
                    detected_language = 'zh'
                    self.log_with_context(f"Detected Chinese language from description", level="INFO")
                else:
                    # Check for other languages
                    japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]+')
                    korean_pattern = re.compile(r'[\uac00-\ud7af\u1100-\u11ff]+')
                    
                    if japanese_pattern.search(video_title):
                        detected_language = 'ja'
                        self.log_with_context(f"Detected Japanese language from title", level="INFO")
                    elif korean_pattern.search(video_title):
                        detected_language = 'ko'
                        self.log_with_context(f"Detected Korean language from title", level="INFO")
            
            # Load the model (will download on first use)
            model = whisper.load_model(self.settings.whisper_model)
            
            self.log_with_context(
                f"Starting transcription with timeout: {analysis.recommended_timeout}s"
                f"{f' (language: {detected_language})' if detected_language else ' (auto-detect language)'}",
                level="INFO"
            )
            
            # Define transcription function for timeout wrapper
            def transcribe_with_whisper():
                return model.transcribe(
                    audio_path,
                    language=detected_language,  # Use detected language or None for auto-detect
                    task='transcribe',
                    verbose=False
                )
            
            # Execute with timeout and resource monitoring
            result, monitoring = timeout_manager.execute_with_timeout(
                func=transcribe_with_whisper,
                timeout_seconds=analysis.recommended_timeout,
                job_id=f"whisper_{video_id}",
                monitor_resources=True
            )
            
            self.log_with_context(
                f"Transcription completed successfully: {monitoring.execution_time_seconds:.1f}s, "
                f"max memory: {monitoring.max_memory_mb:.1f}MB, "
                f"max CPU: {monitoring.max_cpu_percent:.1f}%",
                level="INFO"
            )
            
            # Extract detected language from Whisper
            detected_language = result.get('language', 'en')
            
            # Extract segments and create SRT format
            srt_lines = []
            text_lines = []
            
            for i, segment in enumerate(result['segments'], 1):
                # Create SRT timestamp
                start = self._seconds_to_srt_time(segment['start'])
                end = self._seconds_to_srt_time(segment['end'])
                text = segment['text'].strip()
                
                # Add to SRT
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start} --> {end}")
                srt_lines.append(text)
                srt_lines.append("")
                
                # Add to plain text
                text_lines.append(text)
            
            srt_content = '\n'.join(srt_lines)
            text_content = ' '.join(text_lines)
            
            return {
                'transcript_srt': srt_content,
                'transcript': text_content,
                'word_count': len(text_content.split()),
                'language': result.get('language', 'en'),
                'duration': result['segments'][-1]['end'] if result['segments'] else 0,
                'is_auto_generated': False,
                'extraction_method': 'whisper_local_timeout_protected',
                'model_used': self.settings.whisper_model,
                'processing_stats': {
                    'execution_time': monitoring.execution_time_seconds,
                    'max_memory_mb': monitoring.max_memory_mb,
                    'max_cpu_percent': monitoring.max_cpu_percent,
                    'timeout_used': analysis.recommended_timeout,
                    'audio_duration': analysis.duration_seconds
                }
            }
            
        except ImportError as e:
            self.log_with_context(
                "Whisper not available for transcription",
                level="ERROR",
                extra_context={"error": str(e), "solution": "pip install openai-whisper"}
            )
            raise WhisperNotInstalledError("Whisper not installed. Run: pip install openai-whisper")
        
        except TimeoutError as e:
            self.log_with_context(
                "Whisper transcription timed out - trying fallback strategy",
                level="WARNING",
                extra_context={"timeout": analysis.recommended_timeout, "audio_duration": analysis.duration_seconds}
            )
            # Try fallback with smaller model
            return self._try_fallback_transcription(audio_path, video_id, timeout_manager)
        
        except Exception as e:
            # Check if it's a resource-related error
            error_str = str(e).lower()
            if "memory" in error_str or "out of memory" in error_str:
                self.log_with_context(
                    "Memory error during Whisper transcription - trying fallback",
                    level="WARNING", 
                    extra_context={"error": str(e)}
                )
                return self._try_fallback_transcription(audio_path, video_id, timeout_manager)
            else:
                # Re-raise other exceptions
                raise
        
        except FileNotFoundError as e:
            self.log_with_context(
                f"Audio file not found: {audio_path}",
                level="ERROR",
                extra_context={"audio_path": audio_path, "error": str(e)}
            )
            raise AudioFileError(f"Audio file not found: {audio_path}")
        
        except MemoryError as e:
            self.log_with_context(
                "Insufficient memory for Whisper transcription",
                level="ERROR", 
                extra_context={"audio_path": audio_path, "model": self.settings.whisper_model}
            )
            raise WhisperModelError(f"Insufficient memory for Whisper model {self.settings.whisper_model}")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                self.log_with_context(
                    "GPU/CUDA error during transcription",
                    level="ERROR",
                    extra_context={"error": str(e), "fallback": "CPU processing"}
                )
                raise WhisperModelError(f"GPU error in Whisper: {str(e)}")
            elif "timeout" in error_msg or "timed out" in error_msg:
                self.log_with_context(
                    "Whisper transcription timed out",
                    level="ERROR",
                    extra_context={"audio_path": audio_path, "timeout": "300s"}
                )
                raise APITimeoutError(f"Whisper transcription timed out: {str(e)}")
            else:
                self.log_with_context(
                    "Unexpected Whisper transcription error",
                    level="ERROR",
                    extra_context={"error": str(e), "type": type(e).__name__}
                )
                raise TranscriptionError(f"Whisper transcription failed: {str(e)}")
    
    def _try_fallback_transcription(
        self, 
        audio_path: str, 
        video_id: str, 
        timeout_manager: 'WhisperTimeoutManager'
    ) -> Dict[str, Any]:
        """
        Try fallback transcription with smaller/faster models on timeout/memory errors.
        """
        try:
            import whisper
            from core.audio_analyzer import AudioAnalyzer
            
            # Get fallback models from settings
            fallback_models = self.settings.whisper_fallback_models
            original_model = self.settings.whisper_model
            
            # Remove original model from fallbacks if present
            available_fallbacks = [m for m in fallback_models if m != original_model]
            
            if not available_fallbacks:
                self.log_with_context(
                    "No fallback models available",
                    level="ERROR"
                )
                raise WhisperModelError("No fallback models available after timeout")
            
            for fallback_model in available_fallbacks:
                try:
                    self.log_with_context(
                        f"Trying fallback model: {fallback_model}",
                        level="INFO"
                    )
                    
                    # Re-analyze audio with smaller model for new timeout
                    analyzer = AudioAnalyzer()
                    analysis = analyzer.analyze_audio(
                        audio_path,
                        whisper_timeout_base=self.settings.whisper_timeout_base // 2,  # Shorter timeout
                        whisper_timeout_per_minute=self.settings.whisper_timeout_per_minute * 0.8,
                        whisper_max_duration=self.settings.whisper_max_duration,
                        whisper_chunk_duration=self.settings.whisper_chunk_duration,
                        whisper_model=fallback_model
                    )
                    
                    # Load smaller model
                    model = whisper.load_model(fallback_model)
                    
                    # Define transcription function
                    def fallback_transcribe():
                        return model.transcribe(
                            audio_path,
                            language=None,
                            task='transcribe',
                            verbose=False
                        )
                    
                    # Execute with shorter timeout
                    result, monitoring = timeout_manager.execute_with_timeout(
                        func=fallback_transcribe,
                        timeout_seconds=analysis.recommended_timeout,
                        job_id=f"whisper_fallback_{video_id}_{fallback_model}",
                        monitor_resources=True
                    )
                    
                    # Process result same as main method
                    srt_lines = []
                    text_lines = []
                    
                    for i, segment in enumerate(result['segments'], 1):
                        start = self._seconds_to_srt_time(segment['start'])
                        end = self._seconds_to_srt_time(segment['end'])
                        text = segment['text'].strip()
                        
                        srt_lines.extend([f"{i}", f"{start} --> {end}", text, ""])
                        text_lines.append(text)
                    
                    srt_content = '\n'.join(srt_lines)
                    text_content = ' '.join(text_lines)
                    
                    self.log_with_context(
                        f"Fallback transcription successful with {fallback_model}",
                        level="INFO",
                        extra_context={
                            "execution_time": monitoring.execution_time_seconds,
                            "max_memory_mb": monitoring.max_memory_mb
                        }
                    )
                    
                    return {
                        'transcript_srt': srt_content,
                        'transcript': text_content,
                        'word_count': len(text_content.split()),
                        'language': result.get('language', 'en'),
                        'duration': result['segments'][-1]['end'] if result['segments'] else 0,
                        'is_auto_generated': False,
                        'extraction_method': f'whisper_fallback_{fallback_model}',
                        'model_used': fallback_model,
                        'processing_stats': {
                            'execution_time': monitoring.execution_time_seconds,
                            'max_memory_mb': monitoring.max_memory_mb,
                            'max_cpu_percent': monitoring.max_cpu_percent,
                            'timeout_used': analysis.recommended_timeout,
                            'audio_duration': analysis.duration_seconds,
                            'fallback_used': True,
                            'original_model': original_model
                        }
                    }
                    
                except Exception as fallback_error:
                    self.log_with_context(
                        f"Fallback model {fallback_model} also failed: {fallback_error}",
                        level="WARNING"
                    )
                    continue
            
            # All fallbacks failed
            raise WhisperModelError("All fallback models failed")
            
        except Exception as e:
            raise TranscriptionError(f"Fallback transcription failed: {str(e)}")
    
    def _transcribe_with_chunking(
        self, 
        audio_path: str, 
        video_id: str, 
        analysis: 'AudioAnalysisResult',
        timeout_manager: 'WhisperTimeoutManager'
    ) -> Dict[str, Any]:
        """
        Transcribe long audio files using chunking strategy.
        """
        try:
            import whisper
            import subprocess
            import tempfile
            from pathlib import Path
            from core.audio_analyzer import AudioAnalyzer
            
            self.log_with_context(
                f"Starting chunked transcription for {analysis.duration_seconds:.1f}s audio",
                level="INFO",
                extra_context={"chunk_count": analysis.chunk_count}
            )
            
            # Get chunking plan
            analyzer = AudioAnalyzer()
            chunking_plan = analyzer.get_chunking_plan(
                analysis, 
                chunk_duration=self.settings.whisper_chunk_duration,
                overlap_seconds=60  # 1 minute overlap
            )
            
            # Load Whisper model once for all chunks
            model = whisper.load_model(self.settings.whisper_model)
            
            all_segments = []
            chunk_results = []
            total_processing_time = 0
            max_memory_used = 0
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                for chunk_info in chunking_plan['chunks']:
                    chunk_index = chunk_info['index']
                    start_time = chunk_info['start']
                    end_time = chunk_info['end']
                    chunk_duration = end_time - start_time
                    
                    self.log_with_context(
                        f"Processing chunk {chunk_index + 1}/{len(chunking_plan['chunks'])}: "
                        f"{start_time:.1f}s - {end_time:.1f}s ({chunk_duration:.1f}s)",
                        level="INFO"
                    )
                    
                    # Extract audio chunk using ffmpeg
                    chunk_file = temp_path / f"chunk_{chunk_index:03d}.opus"
                    
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',
                        '-i', audio_path,
                        '-ss', str(start_time),
                        '-t', str(chunk_duration),
                        '-c:a', 'libopus',
                        '-b:a', '64k',
                        str(chunk_file)
                    ]
                    
                    result = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60  # 1 minute timeout for chunk extraction
                    )
                    
                    if result.returncode != 0:
                        raise FFmpegError(f"Failed to extract chunk {chunk_index}: {result.stderr}")
                    
                    # Transcribe chunk with timeout
                    def transcribe_chunk():
                        return model.transcribe(
                            str(chunk_file),
                            language=None,
                            task='transcribe',
                            verbose=False
                        )
                    
                    # Calculate timeout for this chunk
                    chunk_timeout = max(120, int(chunk_duration * 2 + 60))  # 2x duration + 1 minute
                    
                    chunk_result, monitoring = timeout_manager.execute_with_timeout(
                        func=transcribe_chunk,
                        timeout_seconds=chunk_timeout,
                        job_id=f"whisper_chunk_{video_id}_{chunk_index}",
                        monitor_resources=True
                    )
                    
                    # Adjust segment timestamps to global timeline
                    adjusted_segments = []
                    for segment in chunk_result['segments']:
                        adjusted_segment = segment.copy()
                        adjusted_segment['start'] += start_time
                        adjusted_segment['end'] += start_time
                        adjusted_segments.append(adjusted_segment)
                    
                    all_segments.extend(adjusted_segments)
                    chunk_results.append({
                        'chunk_index': chunk_index,
                        'start_time': start_time,
                        'end_time': end_time,
                        'processing_time': monitoring.execution_time_seconds,
                        'memory_used': monitoring.max_memory_mb,
                        'segment_count': len(adjusted_segments)
                    })
                    
                    total_processing_time += monitoring.execution_time_seconds
                    max_memory_used = max(max_memory_used, monitoring.max_memory_mb)
                    
                    # Clean up chunk file
                    chunk_file.unlink()
            
            # Merge overlapping segments and create final transcript
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            # Generate SRT and text content
            srt_lines = []
            text_lines = []
            
            for i, segment in enumerate(merged_segments, 1):
                start = self._seconds_to_srt_time(segment['start'])
                end = self._seconds_to_srt_time(segment['end'])
                text = segment['text'].strip()
                
                srt_lines.extend([f"{i}", f"{start} --> {end}", text, ""])
                text_lines.append(text)
            
            srt_content = '\n'.join(srt_lines)
            text_content = ' '.join(text_lines)
            
            self.log_with_context(
                f"Chunked transcription completed: {len(chunking_plan['chunks'])} chunks, "
                f"total time: {total_processing_time:.1f}s, "
                f"max memory: {max_memory_used:.1f}MB",
                level="INFO"
            )
            
            return {
                'transcript_srt': srt_content,
                'transcript': text_content,
                'word_count': len(text_content.split()),
                'language': chunk_results[0]['chunk_index'] if chunk_results else 'en',  # Use first chunk's language
                'duration': analysis.duration_seconds,
                'is_auto_generated': False,
                'extraction_method': 'whisper_local_chunked',
                'model_used': self.settings.whisper_model,
                'processing_stats': {
                    'execution_time': total_processing_time,
                    'max_memory_mb': max_memory_used,
                    'max_cpu_percent': 0,  # Not tracked across chunks
                    'timeout_used': 'dynamic_per_chunk',
                    'audio_duration': analysis.duration_seconds,
                    'chunked': True,
                    'chunk_count': len(chunking_plan['chunks']),
                    'chunk_results': chunk_results
                }
            }
            
        except Exception as e:
            self.log_with_context(
                f"Chunked transcription failed: {str(e)}",
                level="ERROR"
            )
            raise TranscriptionError(f"Chunked transcription failed: {str(e)}")
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge overlapping segments from chunked transcription to avoid duplication.
        """
        if not segments:
            return []
        
        # Sort segments by start time
        segments_sorted = sorted(segments, key=lambda x: x['start'])
        merged = []
        
        for segment in segments_sorted:
            if not merged:
                merged.append(segment)
                continue
            
            last_segment = merged[-1]
            
            # Check for overlap (with small tolerance)
            if segment['start'] <= last_segment['end'] + 1.0:  # 1 second tolerance
                # Merge segments - extend end time and combine text
                if segment['end'] > last_segment['end']:
                    last_segment['end'] = segment['end']
                    # Only append text if it's significantly different to avoid duplication
                    if not self._text_similarity_high(last_segment['text'], segment['text']):
                        last_segment['text'] = last_segment['text'].strip() + ' ' + segment['text'].strip()
            else:
                # No overlap, add as new segment
                merged.append(segment)
        
        return merged
    
    def _text_similarity_high(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """
        Check if two text segments are highly similar (to avoid duplication in overlaps).
        """
        # Simple word-based similarity check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    def _extract_with_ffmpeg(self, audio_path: str, video_id: str) -> Dict[str, Any]:
        """
        Extract transcript using FFmpeg/Whisper CLI with comprehensive timeout protection.
        
        Args:
            audio_path: Path to audio file
            video_id: Video ID for reference
            
        Returns:
            Dict with transcript and metadata
        """
        import tempfile
        
        try:
            from core.audio_analyzer import AudioAnalyzer
            from core.whisper_timeout_manager import WhisperTimeoutManager
            
            # Pre-flight audio analysis (same as local method)
            analyzer = AudioAnalyzer()
            analysis = analyzer.analyze_audio(
                audio_path,
                whisper_timeout_base=self.settings.whisper_timeout_base,
                whisper_timeout_per_minute=self.settings.whisper_timeout_per_minute,
                whisper_max_duration=self.settings.whisper_max_duration,
                whisper_chunk_duration=self.settings.whisper_chunk_duration,
                whisper_model=self.settings.whisper_model
            )
            
            # Validate audio is suitable for processing
            is_valid, error_msg = analyzer.validate_for_whisper(analysis)
            if not is_valid:
                self.log_with_context(
                    f"Audio validation failed for FFmpeg method: {error_msg}",
                    level="ERROR",
                    extra_context={"audio_path": audio_path, "analysis": analysis.__dict__}
                )
                raise AudioFileError(f"Audio validation failed: {error_msg}")
            
            # Initialize timeout manager
            timeout_manager = WhisperTimeoutManager(
                max_concurrent_jobs=self.settings.whisper_max_concurrent,
                memory_limit_mb=self.settings.whisper_memory_limit_mb
            )
            
            # Check system resources
            can_start, resource_error = timeout_manager.can_start_new_job()
            if not can_start:
                raise WhisperModelError(f"FFmpeg method resource limit: {resource_error}")
            
            self.log_with_context(
                f"FFmpeg/Whisper CLI with timeout protection: {analysis.recommended_timeout}s",
                level="INFO",
                extra_context={"audio_duration": analysis.duration_seconds}
            )
            
            # Check FFmpeg version to see if af_whisper is available
            version_cmd = ['ffmpeg', '-version']
            version_result = subprocess.run(version_cmd, capture_output=True, text=True)
            
            # For now, we'll use whisper CLI as a bridge since FFmpeg 8.0 is not yet available
            # When FFmpeg 8.0 is available, this would use: ffmpeg -i audio.opus -af whisper output.srt
            
            with tempfile.TemporaryDirectory() as temp_dir:
                srt_path = Path(temp_dir) / f"{video_id}.srt"
                
                # Use whisper CLI if available
                cmd = [
                    'whisper',
                    audio_path,
                    '--model', self.settings.whisper_model,
                    # Let Whisper auto-detect language, don't force English
                    '--output_format', 'srt',
                    '--output_dir', temp_dir,
                    '--verbose', 'False'
                ]
                
                # Define command execution for timeout wrapper
                def run_whisper_cli():
                    return subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=None  # Let timeout manager handle this
                    )
                
                # Execute with timeout and resource monitoring
                result, monitoring = timeout_manager.execute_with_timeout(
                    func=run_whisper_cli,
                    timeout_seconds=analysis.recommended_timeout,
                    job_id=f"whisper_cli_{video_id}",
                    monitor_resources=True
                )
                
                if result.returncode != 0:
                    self.log_with_context(
                        "Whisper CLI command failed",
                        level="ERROR",
                        extra_context={"return_code": result.returncode, "stderr": result.stderr, "stdout": result.stdout}
                    )
                    if "permission denied" in result.stderr.lower():
                        raise AudioFileError(f"Permission denied accessing audio file: {audio_path}")
                    elif "no such file" in result.stderr.lower():
                        raise AudioFileError(f"Audio file not found: {audio_path}")
                    else:
                        raise FFmpegError(f"Whisper CLI failed: {result.stderr}")
                
                # Find the generated SRT file
                srt_files = list(Path(temp_dir).glob('*.srt'))
                if not srt_files:
                    self.log_with_context(
                        "No SRT file generated by Whisper CLI",
                        level="ERROR", 
                        extra_context={"temp_dir": temp_dir, "files": list(Path(temp_dir).glob('*'))}
                    )
                    raise WhisperModelError("No SRT file generated - model may have failed silently")
                
                # Read the SRT file
                try:
                    srt_content = srt_files[0].read_text(encoding='utf-8')
                    text_content = self._srt_to_text(srt_content)
                    
                    # Basic quality check
                    if len(text_content.strip()) < 10:
                        self.log_with_context(
                            "Generated transcript too short, may be poor quality",
                            level="WARNING",
                            extra_context={"text_length": len(text_content), "audio_path": audio_path}
                        )
                        raise TranscriptQualityError(f"Generated transcript too short ({len(text_content)} chars)")
                    
                    self.log_with_context(
                        f"FFmpeg/CLI transcription completed successfully: {monitoring.execution_time_seconds:.1f}s, "
                        f"max memory: {monitoring.max_memory_mb:.1f}MB",
                        level="INFO"
                    )
                    
                    return {
                        'transcript_srt': srt_content,
                        'transcript': text_content,
                        'word_count': len(text_content.split()),
                        'language': 'unknown',  # FFmpeg/Whisper CLI doesn't return language info easily
                        'is_auto_generated': False,
                        'extraction_method': 'ffmpeg_whisper_timeout_protected',
                        'model_used': self.settings.whisper_model,
                        'processing_stats': {
                            'execution_time': monitoring.execution_time_seconds,
                            'max_memory_mb': monitoring.max_memory_mb,
                            'max_cpu_percent': monitoring.max_cpu_percent,
                            'timeout_used': analysis.recommended_timeout,
                            'audio_duration': analysis.duration_seconds
                        }
                    }
                    
                except UnicodeDecodeError as e:
                    self.log_with_context(
                        "Failed to decode SRT file",
                        level="ERROR",
                        extra_context={"srt_file": str(srt_files[0]), "error": str(e)}
                    )
                    raise FFmpegError(f"Generated SRT file has encoding issues: {str(e)}")
                
        except TimeoutError as e:
            self.log_with_context(
                "FFmpeg/Whisper CLI transcription timed out - trying fallback strategy",
                level="WARNING",
                extra_context={"timeout": analysis.recommended_timeout, "audio_duration": analysis.duration_seconds}
            )
            # Try fallback with smaller model (reuse the fallback method)
            return self._try_fallback_transcription(audio_path, video_id, timeout_manager)
            
        except subprocess.TimeoutExpired as e:
            # Legacy subprocess timeout (shouldn't happen with timeout manager)
            self.log_with_context(
                "FFmpeg/Whisper transcription legacy timeout",
                level="ERROR",
                extra_context={"audio_path": audio_path}
            )
            raise APITimeoutError("FFmpeg/Whisper legacy timeout - audio may be too long or complex")
        
        except FileNotFoundError as e:
            self.log_with_context(
                "Whisper CLI or FFmpeg not found",
                level="ERROR",
                extra_context={"error": str(e), "solution": "Install whisper: pip install openai-whisper"}
            )
            raise WhisperNotInstalledError("Whisper CLI not found. Install with: pip install openai-whisper")
            
        except PermissionError as e:
            self.log_with_context(
                "Permission denied during FFmpeg transcription",
                level="ERROR",
                extra_context={"audio_path": audio_path, "error": str(e)}
            )
            raise AudioFileError(f"Permission denied: {str(e)}")
            
        except Exception as e:
            self.log_with_context(
                "Unexpected FFmpeg transcription error",
                level="ERROR", 
                extra_context={"error": str(e), "type": type(e).__name__}
            )
            raise FFmpegError(f"FFmpeg transcription failed: {str(e)}")
    
    def _extract_with_yt_dlp(self, video_url: str, video_id: str) -> Dict[str, Any]:
        """Extract subtitles using yt-dlp"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = str(Path(temp_dir) / '%(title)s.%(ext)s')
            
            cmd = [
                'yt-dlp',
                '--write-subs',
                '--write-auto-subs',
                '--sub-format', 'srt',
                '--sub-langs', 'all',  # Get all available languages, not just English
                '--skip-download',
                '--output', output_template,
                '--quiet',
                '--no-warnings',
                '--print-json',
                video_url
            ]
            
            # CRITICAL FIX: Add rate limiting protection before yt-dlp subprocess call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before yt-dlp subtitle extraction")
                time.sleep(wait_time)
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Record the request result for rate limiting stats
                rate_manager.record_request('youtube.com', success=(result.returncode == 0),
                                          is_429=('429' in (result.stderr or '') or 'Too Many Requests' in (result.stderr or '')))
                
                if result.returncode != 0:
                    raise Exception(f"yt-dlp failed: {result.stderr}")
                
                # Parse metadata
                metadata = json.loads(result.stdout)
                
                # Find the SRT file
                srt_files = list(Path(temp_dir).glob('*.srt'))
                if not srt_files:
                    raise Exception("No subtitle file generated")
                
                # Read and parse SRT
                srt_content = srt_files[0].read_text(encoding='utf-8')
                text_content = self._srt_to_text(srt_content)
                
                return {
                    'transcript_srt': srt_content,
                    'transcript': text_content,
                    'word_count': len(text_content.split()),
                    'language': metadata.get('language', 'en'),
                    'duration': metadata.get('duration'),
                    'title': metadata.get('title'),
                    'is_auto_generated': 'auto' in str(srt_files[0])
                }
                
            except subprocess.TimeoutExpired:
                raise Exception("yt-dlp timeout")
            except json.JSONDecodeError:
                raise Exception("Failed to parse yt-dlp output")
    
    def _extract_with_transcript_api(self, video_id: str) -> Dict[str, Any]:
        """Extract transcript using youtube-transcript-api"""
        from youtube_transcript_api import YouTubeTranscriptApi
        import time
        
        try:
            # CRITICAL FIX: Check rate limiting BEFORE making YouTube API call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before YouTube transcript API call")
                time.sleep(wait_time)
            
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Record successful API call
            rate_manager.record_request('youtube.com', success=True, is_429=False)
            
            # Try to get any available transcript, preferring manual over auto-generated
            transcript = None
            is_auto = False
            
            # First try to get any manually created transcript
            try:
                # Get all available manual transcripts
                manual_transcripts = [t for t in transcript_list if not t.is_generated]
                if manual_transcripts:
                    transcript = manual_transcripts[0]  # Use first available
                else:
                    raise Exception("No manual transcript")
            except:
                # Fall back to auto-generated transcripts
                try:
                    generated_transcripts = [t for t in transcript_list if t.is_generated]
                    if generated_transcripts:
                        transcript = generated_transcripts[0]  # Use first available
                        is_auto = True
                    else:
                        raise Exception("No transcripts available")
                except:
                    raise Exception("No transcripts available in any language")
            
            # CRITICAL FIX: Check rate limiting BEFORE fetching transcript data
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                self.logger.warning(f"Rate limited - waiting {wait_time:.1f}s before fetching transcript data")
                time.sleep(wait_time)
            
            # Fetch the transcript
            transcript_data = transcript.fetch()
            
            # Record successful transcript fetch
            rate_manager.record_request('youtube.com', success=True, is_429=False)
            
            # Convert to text
            text_parts = []
            srt_parts = []
            
            for i, entry in enumerate(transcript_data, 1):
                text_parts.append(entry['text'])
                
                # Create SRT format
                start = self._seconds_to_srt_time(entry['start'])
                end = self._seconds_to_srt_time(entry['start'] + entry['duration'])
                srt_parts.append(f"{i}\n{start} --> {end}\n{entry['text']}\n")
            
            text_content = ' '.join(text_parts)
            srt_content = '\n'.join(srt_parts)
            
            return {
                'transcript_srt': srt_content,
                'transcript': text_content,
                'word_count': len(text_content.split()),
                'language': transcript.language_code,
                'is_auto_generated': is_auto
            }
            
        except Exception as e:
            # CRITICAL FIX: Record failed request for rate limiting and detect 429 errors
            error_msg = str(e)
            is_429 = ('429' in error_msg or 'Too Many Requests' in error_msg or 'rate limit' in error_msg.lower())
            
            # Record the failed request
            rate_manager = get_rate_limit_manager()
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            
            if is_429:
                self.logger.error(f"YouTube transcript API rate limited: {error_msg}")
                raise APITimeoutError(f"YouTube transcript API rate limited: {error_msg}")
            else:
                raise Exception(f"youtube-transcript-api failed: {error_msg}")
    
    def _srt_to_text(self, srt_content: str) -> str:
        """Convert SRT format to plain text"""
        lines = srt_content.strip().split('\n')
        text_lines = []
        
        for line in lines:
            # Skip subtitle numbers and timestamps
            if (not line.strip() or 
                line.strip().isdigit() or 
                '-->' in line):
                continue
            
            # Clean and add text
            text = line.strip()
            if text:
                text_lines.append(text)
        
        # Join lines
        text = ' '.join(text_lines)
        
        # Restore punctuation if enabled - using consolidated SRT-aware system
        import os
        if os.getenv('RESTORE_PUNCTUATION', 'true').lower() == 'true':
            from core.chinese_punctuation import restore_punctuation_for_file_sync
            # This context doesn't have file paths, so use direct restorer
            from core.chinese_punctuation import ChinesePunctuationRestorer
            restorer = ChinesePunctuationRestorer()
            text, _ = restorer.restore_punctuation_sync(text)
            
        return text
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Categorize and handle different error types"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str:
            return {
                'error_code': 'E001',
                'error_type': 'timeout',
                'message': 'Transcript extraction timeout',
                'recoverable': True
            }
        elif 'private' in error_str or 'unavailable' in error_str:
            return {
                'error_code': 'E003',
                'error_type': 'video_unavailable',
                'message': 'Video is private or unavailable',
                'recoverable': False
            }
        elif 'no transcript' in error_str or 'no english' in error_str:
            return {
                'error_code': 'E004',
                'error_type': 'no_transcript',
                'message': 'No transcript available for this video',
                'recoverable': False
            }
        else:
            return {
                'error_code': 'E999',
                'error_type': 'unknown',
                'message': str(error),
                'recoverable': True
            }


# Convenience function for direct use
async def transcribe_video(video_id: str, video_url: str, channel_id: Optional[str] = None, audio_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to transcribe a video
    
    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
        channel_id: Optional channel ID for organization
        audio_path: Optional path to downloaded audio file
    
    Returns:
        Dict with transcript and metadata
    """
    worker = TranscribeWorker()
    result = worker.run({
        'video_id': video_id,
        'video_url': video_url,
        'channel_id': channel_id or 'unknown',
        'audio_path': audio_path
    })
    
    # Create summary job if transcription successful and summaries enabled
    if result.get('success') and result.get('transcript_text'):
        from workers.summarizer import should_auto_summarize, create_summary_job
        from core.database import get_db_session, Video
        
        try:
            settings = get_settings()
            
            # Check if we should create a summary job
            if settings.enable_ai_summaries:
                with get_db_session() as db:
                    video = db.query(Video).filter(Video.video_id == video_id).first()
                    if video and should_auto_summarize(video, settings):
                        create_summary_job(video.id, priority=2)
                        logger.info(f"Created summary job for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error creating summary job for {video_id}: {e}")
    
    return result