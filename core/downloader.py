"""YouTube Downloader Module - Handles all video/audio downloading"""

import os
import json
import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import yt_dlp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from .storage_paths_v2 import get_storage_paths_v2
from .youtube_rate_limiter_unified import (
    YouTubeRateLimiterUnified as YouTubeRateLimiter, 
    ErrorAction, ErrorType, get_rate_limiter
)
from .subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
from .youtube_validators import is_valid_youtube_id, filter_valid_video_ids


class YouTubeDownloader:
    """Modular YouTube downloader using yt-dlp with V2 storage structure"""
    
    def __init__(self, base_path: str = None, enable_translation: bool = False, target_language: str = 'en', 
                 skip_transcription: bool = False, skip_punctuation: bool = False):
        """Initialize downloader with V2 storage paths and rate limiter"""
        # ALWAYS use V2 storage structure
        self.storage = get_storage_paths_v2()
        self.base_path = self.storage.base_path
        
        # Store skip flags for two-phase download strategy
        self.skip_transcription = skip_transcription
        self.skip_punctuation = skip_punctuation
        
        # Initialize rate limiter
        self.rate_limiter = get_rate_limiter()
        
        # Initialize subtitle extractor with translation settings
        self.subtitle_extractor = LanguageAgnosticSubtitleExtractor(
            translate_enabled=enable_translation,
            target_language=target_language,
            skip_punctuation=skip_punctuation  # Pass skip flag to subtitle extractor
        )
        
        # Logger
        self.logger = logging.getLogger('YouTubeDownloader')
        
        # Video-level progress tracking
        self.video_progress_file = Path(self.storage.base_path) / '.video_progress.json'
        self.video_progress = self._load_video_progress()
        
        # CRITICAL: Clean up stale states on startup
        self._cleanup_stale_states_on_startup()
        
        # Dead letter queue for permanent failures
        self.dead_letter_file = Path(self.storage.base_path) / '.dead_letter_queue.json'
        self.dead_letter_queue = self._load_dead_letter_queue()
        
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Extract ALL video metadata without downloading
        
        Returns the complete info dict from yt-dlp with ~60 fields
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Return the FULL info dict with all available fields
                info = ydl.extract_info(url, download=False)
                
                # Ensure critical fields exist for V2 structure
                if not info.get('id'):
                    return {'error': 'No video ID found'}
                if not info.get('channel_id') and not info.get('uploader_id'):
                    info['channel_id'] = 'unknown_channel'
                    
                return info  # Return complete info dict with ~60 fields
            except Exception as e:
                return {'error': str(e)}
    
    def _convert_srt_to_txt(self, srt_file: Path, output_dir: Path) -> Optional[Path]:
        """Convert SRT subtitle file to plain text transcript
        
        Args:
            srt_file: Path to the SRT file
            output_dir: Directory to save the transcript
            
        Returns:
            Path to the created transcript file or None if failed
        """
        try:
            # Read SRT file
            with open(srt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract only the text lines (skip timestamps and numbers)
            transcript_lines = []
            for line in lines:
                line = line.strip()
                # Skip empty lines, numbers, and timestamp lines
                if line and not line.isdigit() and '-->' not in line:
                    transcript_lines.append(line)
            
            # Join all text into a single paragraph
            transcript_text = ' '.join(transcript_lines)
            
            # Restore punctuation if enabled - using consolidated system
            import os
            if os.getenv('RESTORE_PUNCTUATION', 'true').lower() == 'true':
                from core.chinese_punctuation import ChinesePunctuationRestorer
                restorer = ChinesePunctuationRestorer()
                transcript_text, _ = restorer.restore_punctuation_sync(transcript_text)
            
            # Save as plain text file with same name but .txt extension
            txt_filename = srt_file.stem.replace('.en', '') + '.txt'
            txt_file = output_dir / txt_filename
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            return txt_file
        except Exception as e:
            print(f"Error converting SRT to TXT: {e}")
            return None
    
    def download_video(self, 
                      url: str, 
                      quality: str = '1080p',
                      download_audio_only: bool = True,
                      audio_format: str = 'opus',
                      video_format: str = 'mp4',
                      channel_name: Optional[str] = None) -> Dict[str, Any]:
        """Download audio/video with specified quality and format (defaults to audio-only Opus)
        
        Args:
            url: YouTube URL
            quality: Video quality - '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p', 'best', 'worst'
            download_audio_only: If True, download only audio (default: True for audio-first platform)
            audio_format: Audio format - 'opus', 'mp3', 'm4a', 'wav', 'flac', 'best' (default: 'opus')
            video_format: Video container format - 'mp4', 'mkv', 'webm', 'avi'
            channel_name: Optional channel name for organization
        """
        
        # Check rate limit before attempting download
        if not self.rate_limiter.check_rate_limit():
            wait_time = self.rate_limiter.wait_if_needed()
            self.logger.info(f"Rate limited, waited {wait_time:.1f}s")
        
        # Check if we should stop all downloads
        if self.rate_limiter.should_stop_all():
            return {
                'status': 'error',
                'error': 'Rate limiter has stopped all downloads due to errors',
                'error_type': 'rate_limit_stop'
            }
        
        # Get video info first
        info = self.get_video_info(url)
        if 'error' in info:
            return info
            
        # Create V2 storage structure: channel_id/video_id/
        video_id = info['id']
        channel_id = info.get('channel_id') or info.get('uploader_id') or 'unknown_channel'
        title = info.get('title', 'untitled')
        
        # Sanitize title for filename use
        sanitized_title = self._sanitize_filename(title)
        
        # Use V2 storage paths: {base_path}/{channel_id}/{video_id}/
        video_base_dir = self.storage.get_video_dir(channel_id, video_id)
        video_base_dir.mkdir(parents=True, exist_ok=True)
        
        # TRACKING: Record processing attempt
        from core.processing_tracker import track_video_stage, ProcessingStage
        track_video_stage(channel_id, video_id, ProcessingStage.QUEUED, f"Processing started for: {title}")

        # Check if already fully processed
        processing_complete_marker = video_base_dir / '.processing_complete'
        if processing_complete_marker.exists():
            self.logger.info(f"⏭️  Skipping already processed video: {title}")
            # TRACKING: Record skip
            track_video_stage(channel_id, video_id, ProcessingStage.COMPLETED, "Already processed - skipped")
            return {
                'status': 'skipped',
                'reason': 'already_processed',
                'video_id': video_id,
                'channel_id': channel_id,
                'title': title,
                'output_dir': str(video_base_dir)
            }
        
        # Check if video is in dead letter queue
        if self._is_in_dead_letter_queue(info['id']):
            self.logger.info(f"📪 Skipping {title} - in dead letter queue")
            return {
                'status': 'skipped',
                'reason': 'in_dead_letter_queue',
                'video_id': info['id']
            }
        
        # Check current video state
        current_state = self._get_video_state(info['id'])
        if current_state == 'completed':
            self.logger.info(f"✅ Skipping {title} - already completed")
            return {
                'status': 'skipped', 
                'reason': 'already_completed',
                'video_id': info['id']
            }
        elif current_state == 'downloading':
            # Check if stale (older than 1 hour)
            state_info = self.video_progress.get('downloading', {}).get(info['id'], {})
            if state_info:
                state_time = datetime.fromisoformat(state_info['timestamp'])
                age_seconds = (datetime.now() - state_time).total_seconds()
                if age_seconds > 3600:  # 1 hour
                    self.logger.warning(f"🧹 Clearing stale downloading state for {info['id']} (age: {age_seconds/60:.1f} minutes)")
                    self._update_video_state(info['id'], 'pending')
                else:
                    self.logger.info(f"⏳ {title} is already being downloaded")
                    return {
                        'status': 'skipped',
                        'reason': 'download_in_progress',
                        'state_age_minutes': age_seconds / 60
                    }
        
        # Update state to downloading
        self._update_video_state(info['id'], 'downloading', {'title': info['title']})
        
        # Check if download is in progress (atomic operation)
        downloading_marker = video_base_dir / '.downloading'
        if downloading_marker.exists():
            # Check if marker is stale (older than 1 hour)
            import time
            marker_age = time.time() - downloading_marker.stat().st_mtime
            if marker_age > 3600:  # 1 hour
                self.logger.warning(f"🧹 Removing stale .downloading marker (age: {marker_age/60:.1f} min)")
                downloading_marker.unlink()
            else:
                self.logger.info(f"⏳ Download in progress by another process: {title}")
                # CRITICAL: Reset state to pending since we're not downloading
                self._update_video_state(info['id'], 'pending')
                return {
                    'status': 'skipped',
                    'reason': 'download_in_progress',
                    'video_id': video_id,
                    'channel_id': channel_id,
                    'title': title,
                    'output_dir': str(video_base_dir)
                }
        
        # Create downloading marker (atomic operation)
        try:
            downloading_marker.touch(exist_ok=False)  # Fail if already exists
        except FileExistsError:
            self.logger.info(f"⏳ Another process started downloading: {title}")
            # CRITICAL: Reset state to pending since we're not downloading  
            self._update_video_state(info['id'], 'pending')
            return {
                'status': 'skipped',
                'reason': 'concurrent_download',
                'video_id': video_id,
                'channel_id': channel_id,
                'title': title,
                'output_dir': str(video_base_dir)
            }
        
        # Create V2 subdirectories
        media_dir = video_base_dir / 'media'
        transcripts_dir = video_base_dir / 'transcripts' 
        content_dir = video_base_dir / 'content'
        metadata_dir = video_base_dir / 'metadata'
        
        media_dir.mkdir(exist_ok=True)
        transcripts_dir.mkdir(exist_ok=True)
        content_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)
        
        # Channel directory for later use
        channel_dir = video_base_dir.parent
        # Note: Channel metadata will be created after download with full info
        
        # Sanitized filename for actual file naming (not directory structure)
        video_title = self._sanitize_filename(info['title'])
        
        # Create video-level tracking files
        self._create_video_tracking_files(video_base_dir, info, video_title)
        
        # Check disk space before download (1GB minimum)
        import shutil
        stat = shutil.disk_usage(self.storage.base_path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < 1.0:
            # Remove downloading marker on failure
            downloading_marker.unlink(missing_ok=True)
            self.logger.error(f"❌ Insufficient disk space: {free_gb:.2f}GB free (need 1GB minimum)")
            
            # Update video state to failed
            self._update_video_state(info['id'], 'failed', {
                'error': f'Insufficient disk space: {free_gb:.2f}GB free',
                'error_type': 'disk_full',
                'title': info['title'],
                'retry_count': 1
            })
            
            # Don't add to dead letter queue for disk space - this can be retried when space is available
            
            return {
                'status': 'error',
                'error': f'Insufficient disk space: {free_gb:.2f}GB free',
                'error_type': 'disk_full',
                'video_id': video_id,
                'channel_id': channel_id,
                'title': title
            }
        
        # Configure download options
        if download_audio_only:
            # Audio only download
            audio_quality_map = {
                'mp3': 'bestaudio/best',
                'm4a': 'bestaudio[ext=m4a]/bestaudio',
                'opus': 'bestaudio/best',  # Get best audio and convert to Opus
                'wav': 'bestaudio/best',
                'flac': 'bestaudio/best',
                'best': 'bestaudio/best'
            }
            format_str = audio_quality_map.get(audio_format, 'bestaudio/best')
            # Download to temp file first (atomic operation)
            temp_output = str(media_dir / f'{sanitized_title}.tmp.%(ext)s')
            final_output = str(media_dir / f'{sanitized_title}.%(ext)s')
            output_template = temp_output
            
            # Add postprocessor for audio conversion
            postprocessors = []
            if audio_format in ['mp3', 'wav', 'flac', 'opus']:
                postprocessors.append({
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_format,
                    'preferredquality': '192' if audio_format == 'mp3' else '0',
                })
        else:
            # Video download with custom resolution
            quality_map = {
                '2160p': 'bestvideo[height<=2160]+bestaudio/best[height<=2160]',
                '1440p': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
                '1080p': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                '360p': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
                '240p': 'bestvideo[height<=240]+bestaudio/best[height<=240]',
                '144p': 'bestvideo[height<=144]+bestaudio/best[height<=144]',
                'best': 'bestvideo+bestaudio/best',
                'worst': 'worstvideo+worstaudio/worst'
            }
            format_str = quality_map.get(quality, quality_map['1080p'])
            # Download to temp file first (atomic operation)
            temp_output = str(media_dir / f'{sanitized_title}.tmp.{video_format}')
            final_output = str(media_dir / f'{sanitized_title}.{video_format}')
            output_template = temp_output
            
            # Add postprocessor for video conversion if needed
            postprocessors = []
            if video_format != 'webm':  # yt-dlp native format is often webm
                postprocessors.append({
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': video_format,
                })
        
        ydl_opts = {
            'format': format_str,
            'outtmpl': output_template,
            'merge_output_format': video_format if not download_audio_only else None,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': postprocessors,
            'nooverwrites': True,  # Skip download if file exists
        }
        
        # Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                
                # Move temp files to final location (atomic operation)
                import glob
                temp_files = glob.glob(str(media_dir / '*.tmp.*'))
                for temp_file in temp_files:
                    final_file = temp_file.replace('.tmp.', '.')
                    Path(temp_file).rename(final_file)
                    self.logger.info(f"✅ Moved {Path(temp_file).name} to final location")
                
                # Re-extract full metadata after download (includes format details)
                full_info = ydl.extract_info(url, download=False)
                
                # Update channel metadata with full post-download info
                self._create_channel_metadata(channel_dir, channel_id, video_base_dir, full_info, download_audio_only)
                
                # Extract subtitles using language-agnostic extractor
                print("Extracting subtitles using language-agnostic extractor...")
                subtitle_result = self.subtitle_extractor.extract_subtitles(
                    video_url=url,
                    output_dir=transcripts_dir,
                    video_id=info['id'],
                    video_title=info['title'],
                    video_description=info.get('description', '')
                )
                
                if subtitle_result.success:
                    print(f"✅ Successfully extracted subtitles in languages: {subtitle_result.languages_found}")
                    print(f"📄 Original files: {len(subtitle_result.original_files)}")
                    if subtitle_result.translated_files:
                        print(f"🌐 Translated files: {len(subtitle_result.translated_files)}")
                    
                    # Check if Chinese video needs Whisper transcription
                    # This happens when we only got English subtitles for a Chinese video
                    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
                    is_chinese_video = chinese_pattern.search(info['title'])
                    has_chinese_subs = any('zh' in lang or 'chinese' in lang.lower() 
                                         for lang in subtitle_result.languages_found)
                    
                    if is_chinese_video and not has_chinese_subs:
                        if self.skip_transcription:
                            print("⏭️  Skipping Whisper transcription (Phase 1: download only)")
                        else:
                            print("🎙️  Chinese video with no Chinese subtitles - using Whisper for Chinese transcription...")
                            audio_files = list(media_dir.glob('*.opus')) + list(media_dir.glob('*.mp3'))
                            if audio_files:
                                self._transcribe_with_whisper_chinese(
                                    audio_path=str(audio_files[0]),
                                    video_id=info['id'],
                                    video_title=info['title'],
                                    video_description=info.get('description', ''),
                                    transcripts_dir=transcripts_dir
                                )
                else:
                    print("⚠️  Subtitle extraction failed:")
                    for error in subtitle_result.error_messages:
                        print(f"   • {error}")
                    
                    # For Chinese videos, automatically use Whisper
                    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
                    if chinese_pattern.search(info['title']):
                        if self.skip_transcription:
                            print("⏭️  Skipping Whisper transcription (Phase 1: download only)")
                        else:
                            print("🎙️  Chinese video detected - using Whisper for transcription...")
                            audio_files = list(media_dir.glob('*.opus')) + list(media_dir.glob('*.mp3'))
                            if audio_files:
                                self._transcribe_with_whisper_chinese(
                                    audio_path=str(audio_files[0]),
                                    video_id=info['id'],
                                    video_title=info['title'],
                                    video_description=info.get('description', ''),
                                    transcripts_dir=transcripts_dir
                                )
                
                # Find all downloaded files across V2 structure
                all_files = []
                for subdir in [media_dir, transcripts_dir, content_dir, metadata_dir]:
                    all_files.extend(list(subdir.glob('*')))
                
                result = {
                    'status': 'success',
                    'video_id': info['id'],
                    'channel_id': channel_id,
                    'title': info['title'],
                    'output_dir': str(video_base_dir),  # Use base video dir
                    'files': [str(f) for f in all_files],
                    'download_time': datetime.now().isoformat(),
                    'storage_structure': 'v2',  # Explicitly mark as V2
                    'subtitle_result': {
                        'success': subtitle_result.success,
                        'languages_found': subtitle_result.languages_found,
                        'original_files': subtitle_result.original_files,
                        'translated_files': subtitle_result.translated_files,
                        'methods_used': subtitle_result.methods_used,
                        'translation_enabled': subtitle_result.translation_enabled,
                        'error_messages': subtitle_result.error_messages
                    }
                }
                
                # Save metadata to metadata subdirectory with video_title prefix
                sanitized_title = self._sanitize_filename(info['title'])
                metadata_file = metadata_dir / f'{sanitized_title}_metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Convert Opus to MP3 if requested or for lightweight alternative
                if audio_format == 'opus':
                    self._convert_opus_to_mp3(media_dir, sanitized_title)
                
                # Mark processing as complete
                self._mark_processing_complete(video_base_dir)
                
                # CRITICAL: Update video state to completed
                self._update_video_state(info['id'], 'completed', {
                    'title': info['title'],
                    'completion_time': datetime.now().isoformat()
                })
                
                # Remove downloading marker after successful completion
                downloading_marker.unlink(missing_ok=True)
                
                # Track successful download with rate limiter
                self.rate_limiter.track_request(success=True)
                    
                return result
                
            except Exception as e:
                # Clean up downloading marker on error
                downloading_marker.unlink(missing_ok=True)
                
                # Clean up any temp files
                import glob
                temp_files = glob.glob(str(media_dir / '*.tmp.*'))
                for temp_file in temp_files:
                    try:
                        Path(temp_file).unlink()
                        self.logger.info(f"🧹 Cleaned up temp file: {Path(temp_file).name}")
                    except:
                        pass
                
                # Handle YouTube-specific errors
                error_info = self.rate_limiter.handle_youtube_error(e)
                self.rate_limiter.track_request(success=False)
                
                # Categorize the error
                error_category = self._categorize_error(e)
                self.logger.error(f"Download failed [{error_category}]: {error_info.message}")
                
                # Determine if error is permanent
                is_permanent = error_category in ['not_found', 'private', 'deleted', 'copyright']
                
                # Update video state with failure details
                retry_count = self.video_progress.get('failed', {}).get(info['id'], {}).get('details', {}).get('retry_count', 0) + 1
                
                self._update_video_state(info['id'], 'failed', {
                    'error': str(e),
                    'error_type': error_info.error_type.value if hasattr(error_info, 'error_type') else 'unknown',
                    'error_category': error_category,
                    'is_permanent': is_permanent,
                    'retry_count': retry_count,
                    'title': info.get('title')
                })
                
                # Add to dead letter queue if permanent failure or too many retries
                if is_permanent or retry_count >= 5:
                    self._add_to_dead_letter_queue(
                        info['id'], 
                        str(e),
                        error_category
                    )
                
                return {
                    'status': 'error',
                    'error': str(e),
                    'error_type': error_info.error_type.value if hasattr(error_info, 'error_type') else 'unknown',
                    'error_category': error_category,
                    'is_permanent': is_permanent,
                    'action': error_info.action.value if hasattr(error_info, 'action') else 'retry',
                    'wait_seconds': error_info.wait_seconds if hasattr(error_info, 'wait_seconds') else 0,
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'retry_count': retry_count
                }
    
    def download_subtitles_only(self, url: str) -> Dict[str, Any]:
        """Download only subtitles without video"""
        info = self.get_video_info(url)
        if 'error' in info:
            return info
            
        # Create V2 storage structure: channel_id/video_id/
        video_id = info['id']
        channel_id = info.get('channel_id') or info.get('uploader_id') or 'unknown_channel'
        
        # Use V2 storage paths for transcripts directory
        video_dir = self.storage.get_video_dir(channel_id, video_id)
        transcripts_dir = video_dir / 'transcripts'
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Use language-agnostic subtitle extractor
        print("Extracting subtitles (subtitles-only mode)...")
        subtitle_result = self.subtitle_extractor.extract_subtitles(
            video_url=url,
            output_dir=transcripts_dir,
            video_id=video_id,
            video_title=info['title'],
            video_description=info.get('description', '')
        )
        
        if subtitle_result.success:
            print(f"✅ Successfully extracted subtitles in languages: {subtitle_result.languages_found}")
            
            # Validate downloaded files
            validation_results = self._validate_downloaded_files(video_base_dir, info['title'])
            
            if not validation_results['audio_valid']:
                self.logger.error(f"❌ Audio validation failed for {info['id']}")
                self._update_video_state(info['id'], 'failed', {'reason': 'invalid_audio_file'})
                self._add_to_dead_letter_queue(info['id'], 'Invalid audio file after download', 'validation_failed')
                
                # Clean up downloading marker on validation failure
                if downloading_marker.exists():
                    downloading_marker.unlink()
                
                return {
                    'status': 'error',
                    'error': 'Audio validation failed',
                    'video_id': info['id']
                }
            
            # Mark as completed with validation results
            self._update_video_state(info['id'], 'completed', {
                'title': info['title'],
                'validation': validation_results,
                'subtitle_languages': subtitle_result.languages_found,
                'methods_used': subtitle_result.methods_used
            })
            
            # Clean up downloading marker
            if downloading_marker.exists():
                downloading_marker.unlink()
            
            # Mark processing as complete
            processing_complete = video_base_dir / '.processing_complete'
            processing_complete.touch()
            
            return {
                'status': 'success',
                'video_id': info['id'],
                'channel_id': channel_id,
                'title': info['title'],
                'subtitle_files': subtitle_result.original_files,
                'translated_files': subtitle_result.translated_files,
                'languages_found': subtitle_result.languages_found,
                'methods_used': subtitle_result.methods_used,
                'storage_structure': 'v2',
                'validation': validation_results
            }
        else:
            print("⚠️  Subtitle extraction failed:")
            for error in subtitle_result.error_messages:
                print(f"   • {error}")
            
            # Update state to failed and check if permanent
            error_type = 'subtitle_extraction_failed'
            retry_count = self.video_progress.get('failed', {}).get(info['id'], {}).get('details', {}).get('retry_count', 0) + 1
            
            self._update_video_state(info['id'], 'failed', {
                'error': 'All subtitle extraction methods failed',
                'error_type': error_type,
                'retry_count': retry_count,
                'error_details': subtitle_result.error_messages
            })
            
            # Add to dead letter queue if too many retries
            if retry_count >= 3:
                self._add_to_dead_letter_queue(
                    info['id'], 
                    'All subtitle extraction methods failed after 3 retries',
                    error_type
                )
            
            # Clean up downloading marker
            if downloading_marker.exists():
                downloading_marker.unlink()
            
            return {
                'status': 'error', 
                'error': 'All subtitle extraction methods failed',
                'error_details': subtitle_result.error_messages,
                'retry_count': retry_count
            }
    
    def _create_channel_metadata(self, channel_dir: Path, channel_id: str, video_dir: Path, video_info: dict, download_audio_only: bool = False):
        """Create comprehensive channel and video-level metadata files with ALL available data"""
        import json
        from datetime import datetime
        
        # .channel_info.json - Basic channel tracking
        channel_info_file = channel_dir / '.channel_info.json'
        channel_info = {
            'channel_id': channel_id,
            'channel_name': video_info.get('channel', video_info.get('uploader', 'Unknown')),
            'channel_url': video_info.get('channel_url', video_info.get('uploader_url', '')),
            'channel_follower_count': video_info.get('channel_follower_count'),
            'uploader_id': video_info.get('uploader_id'),
            'last_updated': datetime.now().isoformat(),
            'total_videos': len(list(channel_dir.glob('*/')) if channel_dir.exists() else []),
            'created': datetime.now().isoformat()
        }
        
        if channel_info_file.exists():
            with open(channel_info_file, 'r') as f:
                existing = json.load(f)
            channel_info['created'] = existing.get('created', channel_info['created'])
            channel_info['total_videos'] = len(list(channel_dir.glob('*/')))
        
        with open(channel_info_file, 'w') as f:
            json.dump(channel_info, f, indent=2)
        
        # channel_url.txt - Simple text file with channel URL for easy reference
        channel_url_file = channel_dir / 'channel_url.txt'
        channel_url = video_info.get('channel_url', video_info.get('uploader_url', ''))
        if not channel_url:
            # Fallback: construct from channel_id if available
            if channel_id and channel_id != 'unknown_channel':
                if channel_id.startswith('UC'):
                    # Standard channel ID format
                    channel_url = f'https://www.youtube.com/channel/{channel_id}'
                else:
                    # Might be a handle/username
                    channel_url = f'https://www.youtube.com/@{channel_id}'
        
        if channel_url and not channel_url_file.exists():
            # Only create if we have a URL and file doesn't exist yet
            with open(channel_url_file, 'w') as f:
                f.write(channel_url)
        
        # {channel_title}.txt - Human-readable channel identification file
        channel_name = channel_info.get('channel_name', '')
        if channel_name:
            # Sanitize the channel name for use as filename
            safe_channel_name = self._sanitize_filename(channel_name)
            channel_title_file = channel_dir / f'{safe_channel_name}.txt'
            
            # Prepare content for the title file
            content_lines = [
                channel_name,
                '=' * len(channel_name),
                f'Channel ID: {channel_id}',
                f'Channel URL: {channel_url if channel_url else "Unknown"}',
                f'Videos: {channel_info.get("total_videos", 0)}'
            ]
            
            # Write or update the file
            with open(channel_title_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines) + '\n')
        
        # {@handle}.txt - Channel handle identification file (e.g., @TCM-Chan.txt)
        channel_handle = channel_info.get('uploader_id', '')
        if channel_handle:
            # Ensure handle starts with @ for consistency
            if not channel_handle.startswith('@'):
                channel_handle = f'@{channel_handle}'
            
            # Keep @ symbol in filename - it's the identifier
            channel_handle_file = channel_dir / f'{channel_handle}.txt'
            
            # Simple content - the handle with @
            with open(channel_handle_file, 'w', encoding='utf-8') as f:
                f.write(channel_handle + '\n')
            
        # .video_index.json - Video catalog with enhanced metadata
        video_index_file = channel_dir / '.video_index.json'
        video_entry = {
            'video_id': video_info['id'],
            'title': video_info['title'],
            'duration': video_info.get('duration'),
            'duration_string': video_info.get('duration_string'),
            'upload_date': video_info.get('upload_date'),
            'view_count': video_info.get('view_count'),
            'like_count': video_info.get('like_count'),
            'comment_count': video_info.get('comment_count'),
            'downloaded_at': datetime.now().isoformat()
        }
        
        video_index = []
        if video_index_file.exists():
            with open(video_index_file, 'r') as f:
                video_index = json.load(f)
        
        existing_entry = next((v for v in video_index if v['video_id'] == video_info['id']), None)
        if existing_entry:
            existing_entry.update(video_entry)
        else:
            video_index.append(video_entry)
            
        with open(video_index_file, 'w') as f:
            json.dump(video_index, f, indent=2)
            
        # {video_title}_video_info.json - COMPREHENSIVE video intelligence report (at video level)
        video_title = self._sanitize_filename(video_info['title'])
        video_info_file = video_dir / f'{video_title}_video_info.json'
        
        # Extract ALL available metadata
        comprehensive_data = {
            # Core identifiers
            'video_id': video_info.get('id'),
            'title': video_info.get('title'),
            'fulltitle': video_info.get('fulltitle'),
            'display_id': video_info.get('display_id'),
            
            # Channel information
            'channel': {
                'channel_id': channel_id,
                'channel_name': video_info.get('channel', video_info.get('uploader')),
                'channel_url': video_info.get('channel_url'),
                'uploader': video_info.get('uploader'),
                'uploader_id': video_info.get('uploader_id'),
                'uploader_url': video_info.get('uploader_url'),
                'channel_follower_count': video_info.get('channel_follower_count')
            },
            
            # Video metrics
            'metrics': {
                'view_count': video_info.get('view_count'),
                'like_count': video_info.get('like_count'),
                'comment_count': video_info.get('comment_count'),
                'average_rating': video_info.get('average_rating')
            },
            
            # Content information
            'content': {
                'description': video_info.get('description'),
                'duration': video_info.get('duration'),
                'duration_string': video_info.get('duration_string'),
                'upload_date': video_info.get('upload_date'),
                'release_date': video_info.get('release_date'),
                'modified_date': video_info.get('modified_date'),
                'language': video_info.get('language'),
                'categories': video_info.get('categories', []),
                'tags': video_info.get('tags', []),
                'age_limit': video_info.get('age_limit'),
                'live_status': video_info.get('live_status'),
                'is_live': video_info.get('is_live'),
                'was_live': video_info.get('was_live'),
                'playable_in_embed': video_info.get('playable_in_embed'),
                'availability': video_info.get('availability')
            },
            
            # Chapters/Timestamps
            'chapters': video_info.get('chapters', []),
            
            # Technical details
            'technical': {
                'format': video_info.get('format'),
                'format_id': video_info.get('format_id'),
                'format_note': video_info.get('format_note'),
                'ext': video_info.get('ext'),
                'protocol': video_info.get('protocol'),
                'width': video_info.get('width'),
                'height': video_info.get('height'),
                'resolution': video_info.get('resolution'),
                'fps': video_info.get('fps'),
                'dynamic_range': video_info.get('dynamic_range'),
                'vcodec': video_info.get('vcodec'),
                'vbr': video_info.get('vbr'),
                'acodec': video_info.get('acodec'),
                'abr': video_info.get('abr'),
                'asr': video_info.get('asr'),
                'audio_channels': video_info.get('audio_channels'),
                'filesize_approx': video_info.get('filesize_approx'),
                'tbr': video_info.get('tbr')
            },
            
            # URLs and references
            'urls': {
                'original_url': video_info.get('original_url'),
                'webpage_url': video_info.get('webpage_url'),
                'webpage_url_domain': video_info.get('webpage_url_domain'),
                'thumbnail': video_info.get('thumbnail')
            },
            
            # Subtitles/Captions availability
            'subtitles_available': list(video_info.get('subtitles', {}).keys()),
            'automatic_captions_available': list(video_info.get('automatic_captions', {}).keys())[:10],  # Limit to 10
            
            # Processing metadata
            'processing': {
                'processed_at': datetime.now().isoformat(),
                'extractor': video_info.get('extractor'),
                'extractor_key': video_info.get('extractor_key'),
                'download_audio_only': download_audio_only
            }
        }
        
        # Save comprehensive JSON
        with open(video_info_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        # Create Markdown report
        self._create_markdown_report(video_dir, video_title, comprehensive_data)
    
    def _create_video_tracking_files(self, video_dir: Path, video_info: dict, video_title: str):
        """Create video-level tracking files"""
        import json
        from datetime import datetime
        
        # .metadata.json at video root
        metadata_file = video_dir / '.metadata.json'
        metadata = {
            'video_id': video_info['id'],
            'title': video_info['title'],
            'channel_id': video_info.get('channel_id', video_info.get('uploader_id')),
            'duration': video_info.get('duration'),
            'upload_date': video_info.get('upload_date'),
            'description': video_info.get('description', ''),
            'processing_started': datetime.now().isoformat(),
            'storage_structure': 'v2'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # video_url.txt - Simple text file with video URL for easy reference
        video_url_file = video_dir / 'video_url.txt'
        video_url = video_info.get('webpage_url') or video_info.get('original_url') or f'https://www.youtube.com/watch?v={video_info["id"]}'
        with open(video_url_file, 'w') as f:
            f.write(video_url)
    
    def _convert_opus_to_mp3(self, media_dir: Path, video_title: str):
        """Convert Opus to MP3 for lightweight alternative"""
        import subprocess
        
        opus_file = media_dir / f'{video_title}.opus'
        mp3_file = media_dir / f'{video_title}.mp3'
        
        if opus_file.exists() and not mp3_file.exists():
            try:
                # Use FFmpeg to convert Opus to MP3
                subprocess.run([
                    'ffmpeg', '-i', str(opus_file),
                    '-codec:a', 'libmp3lame',
                    '-b:a', '192k',
                    str(mp3_file)
                ], check=True, capture_output=True)
                print(f"Converted to MP3: {mp3_file.name}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"MP3 conversion failed: {e}")
    
    def _mark_processing_complete(self, video_dir: Path):
        """Mark processing as complete"""
        from datetime import datetime

        complete_file = video_dir / '.processing_complete'
        with open(complete_file, 'w') as f:
            f.write(datetime.now().isoformat())

        # TRACKING: Record successful completion
        try:
            # Extract video_id and channel_id from path structure
            video_id = video_dir.name
            channel_id = video_dir.parent.name

            from core.processing_tracker import track_video_stage, ProcessingStage
            track_video_stage(channel_id, video_id, ProcessingStage.COMPLETED, "Processing completed successfully")
        except Exception as e:
            self.logger.warning(f"Failed to track completion: {e}")

    def _create_markdown_report(self, video_dir: Path, video_title: str, data: dict):
        """Create comprehensive markdown report with all video metadata"""
        from datetime import datetime
        
        # Create markdown file
        md_file = video_dir / f'{video_title}_video_info.md'
        
        # Format helper functions
        def format_number(num):
            if num is None:
                return 'N/A'
            if isinstance(num, int) and num >= 1000:
                return f'{num:,}'
            return str(num)
        
        def format_duration(seconds):
            if not seconds:
                return 'N/A'
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            if hours > 0:
                return f'{hours}h {minutes}m {secs}s'
            elif minutes > 0:
                return f'{minutes}m {secs}s'
            else:
                return f'{secs}s'
        
        def format_date(date_str):
            if not date_str:
                return 'N/A'
            try:
                # Parse YYYYMMDD format
                if len(date_str) == 8:
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    return f'{year}-{month}-{day}'
                return date_str
            except:
                return date_str
        
        def format_filesize(bytes_size):
            if not bytes_size:
                return 'N/A'
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024.0:
                    return f'{bytes_size:.2f} {unit}'
                bytes_size /= 1024.0
            return f'{bytes_size:.2f} TB'
        
        # Build markdown content
        md_content = []
        
        # Header
        md_content.append(f'# {data.get("title", "Video Report")}')
        md_content.append(f'\n*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*\n')
        
        # Video Overview
        md_content.append('## 📹 Video Overview\n')
        md_content.append(f'- **Video ID**: `{data.get("video_id", "N/A")}`')
        md_content.append(f'- **Title**: {data.get("title", "N/A")}')
        
        # Get content section for nested fields
        content = data.get('content', {})
        md_content.append(f'- **Duration**: {format_duration(content.get("duration"))}')
        md_content.append(f'- **Upload Date**: {format_date(content.get("upload_date"))}')
        md_content.append(f'- **Age Limit**: {content.get("age_limit", 0)}+')
        
        if content.get('is_live'):
            md_content.append('- **Type**: 🔴 Live Stream')
        elif content.get('was_live'):
            md_content.append('- **Type**: 📼 Recorded Live Stream')
        
        # Channel Information
        channel = data.get('channel', {})
        md_content.append('\n## 📺 Channel Information\n')
        md_content.append(f'- **Channel Name**: {channel.get("channel_name", "N/A")}')
        md_content.append(f'- **Channel ID**: `{channel.get("channel_id", "N/A")}`')
        md_content.append(f'- **Channel URL**: {channel.get("channel_url", "N/A")}')
        md_content.append(f'- **Subscribers**: {format_number(channel.get("channel_follower_count"))}')
        md_content.append(f'- **Verified**: {channel.get("channel_is_verified", False) and "✅ Yes" or "❌ No"}')
        
        # Metrics
        metrics = data.get('metrics', {})
        md_content.append('\n## 📊 Engagement Metrics\n')
        md_content.append(f'- **Views**: {format_number(metrics.get("view_count"))}')
        md_content.append(f'- **Likes**: {format_number(metrics.get("like_count"))}')
        md_content.append(f'- **Comments**: {format_number(metrics.get("comment_count"))}')
        if metrics.get('view_count') and metrics.get('like_count'):
            engagement_rate = (metrics['like_count'] / metrics['view_count']) * 100
            md_content.append(f'- **Engagement Rate**: {engagement_rate:.2f}%')
        md_content.append(f'- **Average Rating**: {metrics.get("average_rating", "N/A")}')
        
        # Description (from content section)
        if content.get('description'):
            md_content.append('\n## 📝 Description\n')
            md_content.append('```')
            md_content.append(content['description'][:1000])
            if len(content.get('description', '')) > 1000:
                md_content.append('... [truncated]')
            md_content.append('```')
        
        # Tags (from content section)
        if content.get('tags'):
            md_content.append('\n## 🏷️ Tags\n')
            md_content.append(', '.join([f'`{tag}`' for tag in content['tags'][:20]]))
            if len(content.get('tags', [])) > 20:
                md_content.append(f' ... and {len(content["tags"]) - 20} more')
        
        # Categories (from content section)
        if content.get('categories'):
            md_content.append('\n## 📁 Categories\n')
            md_content.append('- ' + '\n- '.join(content['categories']))
        
        # Chapters
        if data.get('chapters'):
            md_content.append('\n## 📚 Chapters\n')
            md_content.append('| Time | Title |')
            md_content.append('|------|-------|')
            for chapter in data['chapters'][:10]:
                start_time = format_duration(chapter.get('start_time', 0))
                title = chapter.get('title', 'Untitled')
                md_content.append(f'| {start_time} | {title} |')
            if len(data.get('chapters', [])) > 10:
                md_content.append(f'| ... | *{len(data["chapters"]) - 10} more chapters* |')
        
        # Technical Details
        technical = data.get('technical', {})
        processing = data.get('processing', {})
        is_audio_only = processing.get('download_audio_only', False)
        
        if technical:
            md_content.append('\n## ⚙️ Technical Details\n')
            
            # For audio-only downloads, clarify why video fields are N/A
            if is_audio_only:
                md_content.append('*Note: Audio-only download - video details not applicable*\n')
                md_content.append(f'- **Format**: {technical.get("format", "N/A")}')
                md_content.append(f'- **Resolution**: N/A (audio-only)')
                md_content.append(f'- **FPS**: N/A (audio-only)')
                md_content.append(f'- **Video Codec**: N/A (audio-only)')
                md_content.append(f'- **Audio Codec**: {technical.get("acodec", "N/A")}')
                md_content.append(f'- **Audio Bitrate**: {technical.get("abr", "N/A")}')
                md_content.append(f'- **Audio Sample Rate**: {technical.get("asr", "N/A")}')
                md_content.append(f'- **Audio Channels**: {technical.get("audio_channels", "N/A")}')
            else:
                md_content.append(f'- **Format**: {technical.get("format", "N/A")}')
                md_content.append(f'- **Resolution**: {technical.get("resolution", "N/A")}')
                md_content.append(f'- **FPS**: {technical.get("fps", "N/A")}')
                md_content.append(f'- **Video Codec**: {technical.get("vcodec", "N/A")}')
                md_content.append(f'- **Audio Codec**: {technical.get("acodec", "N/A")}')
                md_content.append(f'- **Audio Bitrate**: {technical.get("abr", "N/A")}')
            
            md_content.append(f'- **Filesize (Approx)**: {format_filesize(technical.get("filesize_approx"))}')
            md_content.append(f'- **Protocol**: {technical.get("protocol", "N/A")}')
            md_content.append(f'- **Extractor**: {technical.get("extractor", "N/A")}')
        
        # Subtitles/Captions
        subtitles = data.get('subtitles', {})
        if subtitles:
            md_content.append('\n## 📄 Available Subtitles\n')
            languages = list(subtitles.keys())[:10]
            md_content.append('- ' + ', '.join([f'`{lang}`' for lang in languages]))
            if len(subtitles) > 10:
                md_content.append(f' ... and {len(subtitles) - 10} more languages')
        
        # Automatic Captions
        auto_captions = data.get('automatic_captions', {})
        if auto_captions:
            md_content.append('\n## 🤖 Automatic Captions\n')
            languages = list(auto_captions.keys())[:10]
            md_content.append('- ' + ', '.join([f'`{lang}`' for lang in languages]))
            if len(auto_captions) > 10:
                md_content.append(f' ... and {len(auto_captions) - 10} more languages')
        
        # URLs
        urls = data.get('urls', {})
        if urls:
            md_content.append('\n## 🔗 Links\n')
            md_content.append(f'- **Video URL**: {urls.get("webpage_url", "N/A")}')
            if urls.get('thumbnail'):
                md_content.append(f'- **Thumbnail**: [View Image]({urls["thumbnail"]})')
            if urls.get('channel_url'):
                md_content.append(f'- **Channel**: [Visit Channel]({urls["channel_url"]})')
        
        # Download Information
        md_content.append('\n## 💾 Download Information\n')
        md_content.append(f'- **Downloaded At**: {data.get("downloaded_at", datetime.now().isoformat())}')
        md_content.append(f'- **Storage Structure**: V2')
        md_content.append(f'- **Audio Format**: Opus (primary), MP3 (converted)')
        md_content.append(f'- **Video Format**: MP4 (optional)')
        md_content.append(f'- **Transcript**: SRT + Plain Text')
        
        # Processing Status
        md_content.append('\n## ✅ Processing Status\n')
        md_content.append('- [x] Video metadata extracted')
        md_content.append('- [x] Audio downloaded (Opus)')
        md_content.append('- [x] MP3 conversion completed')
        md_content.append('- [ ] Transcript extraction pending')
        md_content.append('- [ ] Content generation pending')
        
        # Footer
        md_content.append('\n---')
        md_content.append('\n*This report was generated by the YouTube Content Intelligence Platform*')
        md_content.append(f'*Report Version: 2.0 | Structure: V2*')
        
        # Write to file
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        print(f"Created markdown report: {md_file.name}")
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]  # Limit length
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for smart retry logic"""
        error_str = str(error).lower()
        
        # Permanent errors (don't retry)
        if 'video unavailable' in error_str or 'private video' in error_str:
            return 'private'
        if 'deleted' in error_str or 'removed' in error_str:
            return 'deleted'
        if 'copyright' in error_str or 'claimed' in error_str:
            return 'copyright'
        if '404' in error_str or 'not found' in error_str:
            return 'not_found'
        if '410' in error_str:
            return 'gone'
        
        # Temporary errors (retry with backoff)
        if '429' in error_str or 'rate limit' in error_str:
            return 'rate_limit'
        if 'connection' in error_str or 'timeout' in error_str:
            return 'network'
        if '500' in error_str or '502' in error_str or '503' in error_str:
            return 'server_error'
        
        # Resource errors
        if 'disk' in error_str or 'space' in error_str:
            return 'disk_full'
        if 'memory' in error_str:
            return 'out_of_memory'
        
        return 'unknown'
    
    def _load_video_progress(self) -> Dict[str, Any]:
        """Load video-level progress tracking"""
        if self.video_progress_file.exists():
            try:
                with open(self.video_progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load video progress: {e}")
        return {
            'pending': {},
            'downloading': {},
            'transcribing': {},
            'completed': {},
            'failed': {}
        }
    
    def _save_video_progress(self):
        """Save video-level progress tracking with atomic writes"""
        try:
            self.video_progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use atomic write: temp file + rename to prevent corruption
            temp_file = self.video_progress_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(self.video_progress, f, indent=2)
            
            # Atomic rename - this is atomic on most filesystems
            temp_file.rename(self.video_progress_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save video progress: {e}")
            # Clean up temp file if it exists
            temp_file = self.video_progress_file.with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
    
    def _update_video_state(self, video_id: str, state: str, details: Dict[str, Any] = None):
        """Update video state in progress tracking
        
        Args:
            video_id: YouTube video ID
            state: One of: pending, downloading, transcribing, completed, failed
            details: Additional details about the state
        """
        # Remove from all other states
        for s in ['pending', 'downloading', 'transcribing', 'completed', 'failed']:
            if s != state and video_id in self.video_progress.get(s, {}):
                del self.video_progress[s][video_id]
        
        # Add to new state
        self.video_progress.setdefault(state, {})[video_id] = {
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        # Save immediately for crash recovery
        self._save_video_progress()
    
    def _get_video_state(self, video_id: str) -> Optional[str]:
        """Get current state of a video"""
        for state in ['completed', 'downloading', 'transcribing', 'failed', 'pending']:
            if video_id in self.video_progress.get(state, {}):
                return state
        return None
    
    def _load_dead_letter_queue(self) -> Dict[str, Any]:
        """Load dead letter queue for permanent failures"""
        if self.dead_letter_file.exists():
            try:
                with open(self.dead_letter_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load dead letter queue: {e}")
        return {'videos': {}, 'total_count': 0}
    
    def _add_to_dead_letter_queue(self, video_id: str, error: str, error_type: str):
        """Add permanently failed video to dead letter queue
        
        Args:
            video_id: YouTube video ID
            error: Error message
            error_type: Type of error that caused permanent failure
        """
        self.dead_letter_queue.setdefault('videos', {})[video_id] = {
            'added_at': datetime.now().isoformat(),
            'error': error,
            'error_type': error_type,
            'retry_count': self.video_progress.get('failed', {}).get(video_id, {}).get('details', {}).get('retry_count', 0)
        }
        self.dead_letter_queue['total_count'] = len(self.dead_letter_queue['videos'])
        
        # Save immediately with atomic writes
        try:
            self.dead_letter_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use atomic write: temp file + rename to prevent corruption
            temp_file = self.dead_letter_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(self.dead_letter_queue, f, indent=2)
            
            # Atomic rename - this is atomic on most filesystems
            temp_file.rename(self.dead_letter_file)
            
            self.logger.warning(f"📪 Added {video_id} to dead letter queue: {error_type}")
        except Exception as e:
            self.logger.error(f"Failed to save dead letter queue: {e}")
    
    def _is_in_dead_letter_queue(self, video_id: str) -> bool:
        """Check if video is in dead letter queue"""
        return video_id in self.dead_letter_queue.get('videos', {})
    
    def _cleanup_stale_states_on_startup(self):
        """Clean up stale downloading/transcribing states on system startup
        
        This prevents accumulation of stale states from system crashes.
        Videos stuck in 'downloading' or 'transcribing' for >1 hour are reset to 'pending'.
        """
        from datetime import datetime
        
        stale_threshold = 3600  # 1 hour in seconds
        current_time = datetime.now()
        cleaned_count = 0
        
        # Check downloading states
        downloading_states = self.video_progress.get('downloading', {}).copy()
        for video_id, state_info in downloading_states.items():
            try:
                state_time = datetime.fromisoformat(state_info['timestamp'])
                age_seconds = (current_time - state_time).total_seconds()
                
                if age_seconds > stale_threshold:
                    self.logger.warning(f"🧹 Cleaning stale 'downloading' state for {video_id} (age: {age_seconds/60:.1f} min)")
                    self._update_video_state(video_id, 'pending')
                    cleaned_count += 1
                    
            except (KeyError, ValueError, TypeError) as e:
                # Handle malformed timestamps
                self.logger.warning(f"🧹 Cleaning malformed 'downloading' state for {video_id}: {e}")
                self._update_video_state(video_id, 'pending')
                cleaned_count += 1
        
        # Check transcribing states  
        transcribing_states = self.video_progress.get('transcribing', {}).copy()
        for video_id, state_info in transcribing_states.items():
            try:
                state_time = datetime.fromisoformat(state_info['timestamp'])
                age_seconds = (current_time - state_time).total_seconds()
                
                if age_seconds > stale_threshold:
                    self.logger.warning(f"🧹 Cleaning stale 'transcribing' state for {video_id} (age: {age_seconds/60:.1f} min)")
                    self._update_video_state(video_id, 'pending')
                    cleaned_count += 1
                    
            except (KeyError, ValueError, TypeError) as e:
                # Handle malformed timestamps
                self.logger.warning(f"🧹 Cleaning malformed 'transcribing' state for {video_id}: {e}")
                self._update_video_state(video_id, 'pending')
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned {cleaned_count} stale state(s) on startup")
            self._save_video_progress()  # Save the cleaned states
    
    def _validate_downloaded_files(self, video_dir: Path, video_title: str) -> Dict[str, bool]:
        """Validate that downloaded files are not corrupted
        
        Args:
            video_dir: Path to video directory
            video_title: Title of the video
            
        Returns:
            Dictionary with validation results for each file type
        """
        validation_results = {
            'audio_valid': False,
            'transcript_valid': False,
            'metadata_valid': False
        }
        
        # Check audio file
        audio_extensions = ['.opus', '.mp3', '.mp4']
        for ext in audio_extensions:
            audio_file = video_dir / 'media' / f"{video_title}{ext}"
            if audio_file.exists():
                # Check file size (should be > 1KB)
                if audio_file.stat().st_size > 1024:
                    validation_results['audio_valid'] = True
                    break
        
        # Check transcript files
        transcript_dir = video_dir / 'transcripts'
        if transcript_dir.exists():
            # Check for any .srt or .txt files
            srt_files = list(transcript_dir.glob('*.srt'))
            txt_files = list(transcript_dir.glob('*.txt'))
            if srt_files or txt_files:
                # Check at least one file has content
                for file in srt_files + txt_files:
                    if file.stat().st_size > 100:  # At least 100 bytes
                        validation_results['transcript_valid'] = True
                        break
        
        # Check metadata file
        metadata_files = list(video_dir.glob('*_video_info.json'))
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                    if 'id' in metadata:  # Basic validation
                        validation_results['metadata_valid'] = True
            except Exception:
                pass
        
        return validation_results
    
    async def download_batch(self, 
                            videos: List[Dict[str, Any]], 
                            max_concurrent: int = 3,
                            download_audio_only: bool = True,
                            audio_format: str = 'opus',
                            video_format: str = 'mp4',
                            quality: str = '1080p',
                            progress_callback: Optional[Callable] = None,
                            rate_limit_delay: float = 0.5) -> Dict[str, Any]:
        """Download multiple videos concurrently with rate limiting
        
        Args:
            videos: List of video info dicts (must contain 'id' or 'url')
            max_concurrent: Maximum concurrent downloads (default: 3, max: 10)
            download_audio_only: If True, download only audio
            audio_format: Audio format for audio-only downloads
            video_format: Video format for full video downloads
            quality: Video quality for full video downloads
            progress_callback: Optional callback function(video_id, status, message)
            rate_limit_delay: Delay between starting new downloads (seconds)
            
        Returns:
            Dict with results for each video and statistics
        """
        # Validate and limit concurrency
        max_concurrent = min(max(1, max_concurrent), 10)
        
        # Filter out invalid video IDs first
        valid_videos = filter_valid_video_ids(videos, log_invalid=True)
        invalid_count = len(videos) - len(valid_videos)
        
        if invalid_count > 0:
            self.logger.warning(f"Filtered out {invalid_count} invalid video IDs before processing")
        
        # Create a semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Results tracking
        results = {
            'successful': [],
            'failed': [],
            'skipped': [],  # Track already processed videos
            'total': len(valid_videos),  # Only count valid videos
            'invalid_skipped': invalid_count,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': None
        }
        
        async def download_with_semaphore(video_info: Dict[str, Any], index: int) -> Dict[str, Any]:
            """Download a single video with semaphore control and rate limiting"""
            async with semaphore:
                video_id = video_info.get('id', video_info.get('video_id'))
                video_url = video_info.get('url')
                
                # Build URL if not provided
                if not video_url and video_id:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                if not video_url:
                    return {
                        'status': 'error',
                        'video_id': video_id,
                        'error': 'No URL or video ID provided'
                    }
                
                # Check if rate limiter says we should stop
                if self.rate_limiter.should_stop_all():
                    return {
                        'status': 'error',
                        'video_id': video_id,
                        'error': 'Rate limiter stopped all downloads',
                        'error_type': 'rate_limit_stop'
                    }
                
                # Check rate limit and wait if needed
                if not self.rate_limiter.check_rate_limit():
                    wait_time = self.rate_limiter.wait_if_needed()
                    if progress_callback:
                        await self._call_async_callback(
                            progress_callback, video_id, 'waiting',
                            f'[{index+1}/{len(videos)}] Rate limited, waiting {wait_time:.1f}s'
                        )
                    await asyncio.sleep(wait_time)
                
                # Apply adaptive delay based on rate limiter
                delay = self.rate_limiter.get_current_delay()
                if index > 0 and delay > 0:
                    await asyncio.sleep(delay)
                
                # Retry logic with exponential backoff
                max_retries = 3
                retry_count = 0
                
                while retry_count <= max_retries:
                    try:
                        # Progress callback
                        if progress_callback:
                            retry_msg = f' (retry {retry_count})' if retry_count > 0 else ''
                            await self._call_async_callback(
                                progress_callback, 
                                video_id, 
                                'starting', 
                                f'[{index+1}/{len(videos)}] Starting download{retry_msg}'
                            )
                        
                        # Run download in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            result = await loop.run_in_executor(
                                executor,
                                self.download_video,
                                video_url,
                                quality,
                                download_audio_only,
                                audio_format,
                                video_format,
                                video_info.get('channel_name')
                            )
                        
                        # Check if download was successful
                        if result.get('status') == 'success':
                            # Track successful request
                            self.rate_limiter.track_request(success=True)
                            
                            # Progress callback
                            if progress_callback:
                                await self._call_async_callback(
                                    progress_callback, video_id, 'completed',
                                    f'[{index+1}/{len(videos)}] Completed'
                                )
                            
                            return {
                                'video_id': video_id,
                                'index': index,
                                **result
                            }
                        elif result.get('status') == 'skipped':
                            # Video was already processed
                            if progress_callback:
                                await self._call_async_callback(
                                    progress_callback, video_id, 'skipped',
                                    f'[{index+1}/{len(videos)}] Already processed'
                                )
                            
                            return {
                                'video_id': video_id,
                                'index': index,
                                **result
                            }
                        else:
                            # Handle download error
                            raise Exception(result.get('error', 'Download failed'))
                        
                    except Exception as e:
                        # Handle error with rate limiter
                        error_info = self.rate_limiter.handle_youtube_error(e, retry_count)
                        self.rate_limiter.track_request(success=False)
                        
                        # Determine action based on error
                        if error_info.action == ErrorAction.SKIP_VIDEO:
                            if progress_callback:
                                await self._call_async_callback(
                                    progress_callback, video_id, 'skipped',
                                    f'[{index+1}/{len(videos)}] Skipped: {error_info.message}'
                                )
                            return {
                                'status': 'error',
                                'video_id': video_id,
                                'index': index,
                                'error': str(e),
                                'error_type': error_info.error_type.value,
                                'skipped': True
                            }
                        
                        elif error_info.action == ErrorAction.STOP_ALL:
                            if progress_callback:
                                await self._call_async_callback(
                                    progress_callback, video_id, 'stopped',
                                    f'[{index+1}/{len(videos)}] Stopped: {error_info.message}'
                                )
                            return {
                                'status': 'error',
                                'video_id': video_id,
                                'index': index,
                                'error': str(e),
                                'error_type': error_info.error_type.value,
                                'stop_all': True
                            }
                        
                        elif error_info.action in [ErrorAction.RETRY_WITH_BACKOFF, ErrorAction.COOLDOWN]:
                            retry_count += 1
                            if retry_count > max_retries:
                                if progress_callback:
                                    await self._call_async_callback(
                                        progress_callback, video_id, 'failed',
                                        f'[{index+1}/{len(videos)}] Failed after {max_retries} retries'
                                    )
                                return {
                                    'status': 'error',
                                    'video_id': video_id,
                                    'index': index,
                                    'error': str(e),
                                    'error_type': error_info.error_type.value,
                                    'retries_exhausted': True
                                }
                            
                            # Wait before retry
                            if error_info.wait_seconds > 0:
                                if progress_callback:
                                    await self._call_async_callback(
                                        progress_callback, video_id, 'waiting',
                                        f'[{index+1}/{len(videos)}] Waiting {error_info.wait_seconds:.1f}s before retry'
                                    )
                                await asyncio.sleep(error_info.wait_seconds)
                            continue  # Retry the download
                        
                        else:
                            # Unknown action, fail the download
                            if progress_callback:
                                await self._call_async_callback(
                                    progress_callback, video_id, 'error',
                                    f'[{index+1}/{len(videos)}] Error: {str(e)}'
                                )
                            return {
                                'status': 'error',
                                'video_id': video_id,
                                'index': index,
                                'error': str(e)
                            }
                
                # If we get here, all retries were exhausted
                return {
                    'status': 'error',
                    'video_id': video_id,
                    'index': index,
                    'error': 'All retries exhausted'
                }
        
        # Create download tasks for valid videos only
        start_time = datetime.now()
        tasks = [
            download_with_semaphore(video, i) 
            for i, video in enumerate(valid_videos)
        ]
        
        # Run all downloads concurrently
        download_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and check for stop_all signal
        should_stop_all = False
        for result in download_results:
            if isinstance(result, Exception):
                results['failed'].append({
                    'error': str(result),
                    'type': 'exception'
                })
            elif isinstance(result, dict):
                if result.get('status') == 'success':
                    results['successful'].append(result)
                elif result.get('status') == 'skipped':
                    results['skipped'].append(result)
                else:
                    results['failed'].append(result)
                    # Check if we should stop all downloads
                    if result.get('stop_all'):
                        should_stop_all = True
                        results['stopped_early'] = True
                        results['stop_reason'] = result.get('error_type', 'unknown')
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration
        results['success_rate'] = len(results['successful']) / len(valid_videos) if valid_videos else 0
        results['downloads_per_minute'] = (len(results['successful']) / duration * 60) if duration > 0 else 0
        
        return results
    
    def _transcribe_with_whisper_chinese(self, audio_path: str, video_id: str, 
                                        video_title: str, video_description: str,
                                        transcripts_dir: Path) -> bool:
        """
        Transcribe Chinese audio using Whisper with Chinese language setting.
        
        Args:
            audio_path: Path to audio file
            video_id: YouTube video ID
            video_title: Video title for language detection
            video_description: Video description
            transcripts_dir: Directory to save transcripts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if Whisper is available
            try:
                import whisper
            except ImportError:
                print("⚠️  Whisper not installed. Install with: pip install openai-whisper")
                return False
            
            print(f"🎙️  Loading Whisper model for Chinese transcription...")
            
            # Use a smaller model for Chinese to reduce memory usage
            model = whisper.load_model("base")
            
            print(f"🎙️  Transcribing Chinese audio (this may take a few minutes)...")
            
            # Transcribe with Chinese language
            result = model.transcribe(
                audio_path,
                language='zh',  # Force Chinese language
                task='transcribe',
                verbose=False
            )
            
            # Extract text and segments
            text = result.get('text', '')
            segments = result.get('segments', [])
            
            if not text or not segments:
                print("⚠️  Whisper transcription produced no results")
                return False
            
            # Create SRT content
            srt_lines = []
            for i, segment in enumerate(segments, 1):
                start = self._seconds_to_srt_time(segment['start'])
                end = self._seconds_to_srt_time(segment['end'])
                text_line = segment['text'].strip()
                
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start} --> {end}")
                srt_lines.append(text_line)
                srt_lines.append("")
            
            srt_content = '\n'.join(srt_lines)
            
            # Save Chinese transcript files
            safe_title = self._sanitize_filename(video_title)
            
            # Save as .zh.srt (Chinese SRT)
            srt_path = transcripts_dir / f"{safe_title}.zh.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"📄 Saved Chinese SRT: {srt_path.name}")
            
            # Apply Chinese punctuation restoration if enabled and not skipped
            # NOTE: Whisper actually DOES add punctuation to Chinese transcripts!
            # This restoration is mainly for edge cases or older transcripts
            final_text = text
            
            if self.skip_punctuation:
                print(f"⏭️  Skipping Chinese punctuation restoration (skip_punctuation=True)")
            else:
                print(f"🔍 Checking Chinese punctuation...")
            
            if not self.skip_punctuation:
                try:
                    # Use the consolidated async implementation with Claude CLI fallback
                    from core.chinese_punctuation import ChinesePunctuationRestorer
                    import asyncio
                
                    restorer = ChinesePunctuationRestorer()
                    is_chinese = restorer.detect_chinese_text(text)
                    has_punct = restorer.has_punctuation(text)
                    
                    print(f"🔍 Text analysis - Is Chinese: {is_chinese}, Has punctuation: {has_punct}")
                    
                    if is_chinese and not has_punct:
                        print("🔤 Applying consolidated SRT-aware Chinese punctuation restoration...")
                        
                        # Use consolidated system with SRT-aware processing
                        srt_content = None
                        if srt_path.exists():
                            srt_content = srt_path.read_text(encoding='utf-8')
                            print("📄 Using SRT content for boundary-aware processing")
                        else:
                            print("📄 No SRT available, using text-only processing")
                        
                        # Use simple rule-based punctuation restoration
                        restored_text, success = restorer.restore_punctuation_sync(text)
                        
                        if success:
                            final_text = restored_text
                            print("✅ Chinese punctuation restored")
                        else:
                            print("⚠️  Punctuation restoration unsuccessful")
                    else:
                        if not is_chinese:
                            print("🔍 Text is not Chinese, skipping punctuation restoration")
                        else:
                            print("🔍 Text already has punctuation, skipping restoration")
                        
                except ImportError as e:
                    print(f"⚠️  Failed to import punctuation module: {e}")
                    logging.error(f"Punctuation module import error: {e}", exc_info=True)
                except Exception as e:
                    print(f"⚠️  Chinese punctuation restoration failed: {e}")
                    logging.error(f"Punctuation restoration error: {e}", exc_info=True)
                    # Keep original text if restoration fails
            
            # Save as .zh.txt (Chinese plain text)
            txt_path = transcripts_dir / f"{safe_title}.zh.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            print(f"📄 Saved Chinese text: {txt_path.name}")
            
            print(f"✅ Successfully transcribed Chinese audio using Whisper")
            return True
            
        except Exception as e:
            print(f"❌ Whisper Chinese transcription failed: {e}")
            logging.error(f"Whisper Chinese transcription error: {e}")
            return False
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    async def download_channel_videos(self,
                                     channel_url: str,
                                     limit: Optional[int] = None,
                                     max_concurrent: int = 3,
                                     download_audio_only: bool = True,
                                     audio_format: str = 'opus',
                                     video_format: str = 'mp4',
                                     quality: str = '1080p',
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Download all videos from a channel concurrently
        
        Args:
            channel_url: YouTube channel URL
            limit: Optional limit on number of videos
            max_concurrent: Maximum concurrent downloads
            download_audio_only: If True, download only audio
            audio_format: Audio format for audio-only downloads
            video_format: Video format for full video downloads
            quality: Video quality for full video downloads
            progress_callback: Optional callback function
            
        Returns:
            Dict with download results and statistics
        """
        from core.channel_enumerator import ChannelEnumerator
        
        # Enumerate channel videos
        enumerator = ChannelEnumerator()
        
        # Get channel info
        channel_info = enumerator.get_channel_info(channel_url)
        if not channel_info:
            return {
                'status': 'error',
                'error': 'Could not get channel info'
            }
        
        # Get videos
        videos = enumerator.get_all_videos(channel_url, limit=limit)
        
        if not videos:
            return {
                'status': 'error',
                'error': 'No videos found in channel',
                'channel_info': channel_info
            }
        
        # Convert VideoInfo objects to dictionaries and add channel name
        channel_name = channel_info.get('channel_name')
        video_dicts = []
        for video in videos:
            # Convert VideoInfo to dict
            video_dict = {
                'video_id': video.video_id,
                'title': video.title,
                'upload_date': video.upload_date,
                'duration': video.duration,
                'view_count': video.view_count,
                'channel_name': channel_name
            }
            video_dicts.append(video_dict)
        
        # Download videos in batches
        results = await self.download_batch(
            videos=video_dicts,
            max_concurrent=max_concurrent,
            download_audio_only=download_audio_only,
            audio_format=audio_format,
            video_format=video_format,
            quality=quality,
            progress_callback=progress_callback
        )
        
        # Add channel info to results
        results['channel_info'] = channel_info
        results['channel_url'] = channel_url
        
        return results
    
    async def _call_async_callback(self, callback: Callable, *args, **kwargs):
        """Helper to call callbacks that might be sync or async"""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args, **kwargs)
        else:
            callback(*args, **kwargs)


def create_downloader_with_settings(skip_transcription: bool = False, skip_punctuation: bool = False) -> YouTubeDownloader:
    """Create a YouTubeDownloader configured with current settings
    
    Args:
        skip_transcription: Skip Whisper transcription for faster downloads
        skip_punctuation: Skip Chinese punctuation restoration for faster downloads
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        return YouTubeDownloader(
            enable_translation=settings.subtitle_translation_enabled,
            target_language=settings.subtitle_target_language,
            skip_transcription=skip_transcription,
            skip_punctuation=skip_punctuation
        )
    except ImportError:
        # Fallback if settings aren't available
        return YouTubeDownloader(
            enable_translation=False, 
            target_language='en',
            skip_transcription=skip_transcription,
            skip_punctuation=skip_punctuation
        )