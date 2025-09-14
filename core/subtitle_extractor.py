"""
Advanced Subtitle Extraction Module

Implements multiple fallback strategies to extract subtitles from YouTube videos
using various methods, formats, and languages to ensure we get transcripts "no matter what".
"""

import os
import logging
import time
import yt_dlp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json

# CRITICAL FIX: Import rate limiting to prevent 429 errors
from core.rate_limit_manager import get_rate_limit_manager


@dataclass
class SubtitleResult:
    """Result of subtitle extraction attempt"""
    success: bool
    method: str
    language: str
    subtitle_format: str
    file_path: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    lines_count: int = 0


class AdvancedSubtitleExtractor:
    """
    Advanced subtitle extractor with multiple fallback strategies.
    
    Extraction strategies (in order of preference):
    1. Manual subtitles in preferred language
    2. Auto-generated subtitles in preferred language  
    3. Manual subtitles in any available language
    4. Auto-generated subtitles in any available language
    5. Live/auto-captions if video is live
    6. Alternative extraction using youtube-transcript-api
    7. OCR extraction from video frames (if configured)
    """
    
    def __init__(self):
        self.logger = logging.getLogger('SubtitleExtractor')
        
        # Language priority order - try these languages in order
        self.language_priority = [
            'en',      # English (preferred)
            'zh-CN',   # Chinese Simplified
            'zh-TW',   # Chinese Traditional
            'zh',      # Chinese (generic)
            'ja',      # Japanese
            'ko',      # Korean
            'es',      # Spanish
            'fr',      # French
            'de',      # German
            'ru',      # Russian
            'pt',      # Portuguese
            'it',      # Italian
            'ar',      # Arabic
            'hi',      # Hindi
            'th',      # Thai
            'vi',      # Vietnamese
        ]
        
        # Format priority - SRT is preferred, but try others
        self.format_priority = ['srt', 'vtt', 'ass', 'ttml', 'json3']
        
    def extract_subtitles_comprehensive(self, 
                                      video_url: str,
                                      output_dir: Path,
                                      video_id: str,
                                      video_title: str = "video") -> Dict[str, Any]:
        """
        Comprehensive subtitle extraction using all available methods.
        
        Args:
            video_url: YouTube video URL
            output_dir: Directory to save subtitle files
            video_id: YouTube video ID
            video_title: Video title for filename
            
        Returns:
            Dict with extraction results and statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'success': False,
            'total_attempts': 0,
            'successful_extractions': [],
            'failed_attempts': [],
            'final_files': [],
            'best_subtitle': None,
            'extraction_methods_used': [],
            'languages_found': [],
            'formats_found': []
        }
        
        self.logger.info(f"Starting comprehensive subtitle extraction for {video_id}")
        
        # Strategy 1: Try yt-dlp with all combinations of languages and formats
        yt_dlp_result = self._extract_with_ytdlp_comprehensive(
            video_url, output_dir, video_id, video_title
        )
        results['total_attempts'] += yt_dlp_result['attempts']
        results['successful_extractions'].extend(yt_dlp_result['successes'])
        results['failed_attempts'].extend(yt_dlp_result['failures'])
        results['extraction_methods_used'].append('yt-dlp')
        
        # Strategy 2: Try youtube-transcript-api as fallback
        if not yt_dlp_result['successes']:
            transcript_api_result = self._extract_with_transcript_api(
                video_url, output_dir, video_id, video_title
            )
            results['total_attempts'] += transcript_api_result['attempts']
            results['successful_extractions'].extend(transcript_api_result['successes'])
            results['failed_attempts'].extend(transcript_api_result['failures'])
            if transcript_api_result['attempts'] > 0:
                results['extraction_methods_used'].append('youtube-transcript-api')
        
        # Strategy 3: Try alternative yt-dlp configurations
        if not results['successful_extractions']:
            alt_result = self._extract_with_alternative_configs(
                video_url, output_dir, video_id, video_title
            )
            results['total_attempts'] += alt_result['attempts']
            results['successful_extractions'].extend(alt_result['successes'])
            results['failed_attempts'].extend(alt_result['failures'])
            if alt_result['attempts'] > 0:
                results['extraction_methods_used'].append('yt-dlp-alternative')
        
        # Process successful extractions
        if results['successful_extractions']:
            results['success'] = True
            results['best_subtitle'] = self._select_best_subtitle(results['successful_extractions'])
            
            # Convert all to standard formats
            final_files = self._convert_to_standard_formats(
                results['successful_extractions'], output_dir, video_title
            )
            results['final_files'] = final_files
            
            # Extract statistics
            results['languages_found'] = list(set([s.language for s in results['successful_extractions']]))
            results['formats_found'] = list(set([s.subtitle_format for s in results['successful_extractions']]))
            
            self.logger.info(f"Successfully extracted subtitles for {video_id} in {len(results['languages_found'])} languages")
        else:
            self.logger.warning(f"Failed to extract any subtitles for {video_id} after {results['total_attempts']} attempts")
        
        return results
    
    def _extract_with_ytdlp_comprehensive(self, video_url: str, output_dir: Path, 
                                        video_id: str, video_title: str) -> Dict[str, Any]:
        """Extract using yt-dlp with comprehensive language and format combinations"""
        results = {'attempts': 0, 'successes': [], 'failures': []}
        
        # Try different combinations of languages and subtitle types
        extraction_configs = [
            # First: Try manual subtitles in priority languages
            {'writesubtitles': True, 'writeautomaticsub': False, 'langs': self.language_priority[:5]},
            # Second: Try auto-generated in priority languages  
            {'writesubtitles': False, 'writeautomaticsub': True, 'langs': self.language_priority[:5]},
            # Third: Try both types in priority languages
            {'writesubtitles': True, 'writeautomaticsub': True, 'langs': self.language_priority[:8]},
            # Fourth: Try any available language
            {'writesubtitles': True, 'writeautomaticsub': True, 'langs': ['all']},
        ]
        
        for config in extraction_configs:
            for subtitle_format in self.format_priority:
                results['attempts'] += 1
                
                try:
                    ydl_opts = {
                        'skip_download': True,
                        'writesubtitles': config['writesubtitles'],
                        'writeautomaticsub': config['writeautomaticsub'],
                        'subtitlesformat': subtitle_format,
                        'subtitleslangs': config['langs'],
                        'outtmpl': str(output_dir / f'{video_title}.%(ext)s'),
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    
                    # Check what files were created
                    subtitle_files = list(output_dir.glob(f'{video_title}*.{subtitle_format}'))
                    if subtitle_files:
                        for subtitle_file in subtitle_files:
                            if subtitle_file.stat().st_size > 0:  # Non-empty file
                                # Extract language from filename
                                lang = self._extract_language_from_filename(subtitle_file.name)
                                
                                result = SubtitleResult(
                                    success=True,
                                    method=f"yt-dlp-{config}",
                                    language=lang,
                                    subtitle_format=subtitle_format,
                                    file_path=str(subtitle_file),
                                    lines_count=self._count_subtitle_lines(subtitle_file)
                                )
                                results['successes'].append(result)
                    
                    # If we got some results, don't try more aggressive methods for this config
                    if results['successes']:
                        break
                        
                except Exception as e:
                    results['failures'].append(f"yt-dlp {config} {subtitle_format}: {str(e)}")
            
            # If we got results with this config, stop trying more configs
            if results['successes']:
                break
        
        return results
    
    def _extract_with_transcript_api(self, video_url: str, output_dir: Path, 
                                   video_id: str, video_title: str) -> Dict[str, Any]:
        """Extract using youtube-transcript-api as fallback"""
        results = {'attempts': 0, 'successes': [], 'failures': []}
        
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import SRTFormatter
            
            results['attempts'] += 1
            
            # CRITICAL FIX: Add rate limiting protection before YouTube API call
            rate_manager = get_rate_limit_manager()
            allowed, wait_time = rate_manager.should_allow_request('youtube.com')
            if not allowed:
                logging.warning(f"Rate limited - waiting {wait_time:.1f}s before transcript API call for {video_id}")
                time.sleep(wait_time)
            
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # CRITICAL FIX: Record successful API call for rate limiting stats
            rate_manager.record_request('youtube.com', success=True)
            
            # Try to get transcript in priority order
            for lang in self.language_priority:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    
                    # Save as SRT
                    formatter = SRTFormatter()
                    srt_content = formatter.format_transcript(transcript_data)
                    
                    srt_file = output_dir / f'{video_title}.{lang}.srt'
                    with open(srt_file, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                    
                    result = SubtitleResult(
                        success=True,
                        method='youtube-transcript-api',
                        language=lang,
                        subtitle_format='srt',
                        file_path=str(srt_file),
                        lines_count=len(transcript_data)
                    )
                    results['successes'].append(result)
                    
                except Exception as e:
                    results['failures'].append(f"transcript-api {lang}: {str(e)}")
                    continue
            
            # If no language worked, try the first available transcript
            if not results['successes']:
                try:
                    first_transcript = next(iter(transcript_list))
                    transcript_data = first_transcript.fetch()
                    
                    formatter = SRTFormatter()
                    srt_content = formatter.format_transcript(transcript_data)
                    
                    lang = first_transcript.language_code
                    srt_file = output_dir / f'{video_title}.{lang}.srt'
                    with open(srt_file, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                    
                    result = SubtitleResult(
                        success=True,
                        method='youtube-transcript-api-any',
                        language=lang,
                        subtitle_format='srt',
                        file_path=str(srt_file),
                        lines_count=len(transcript_data)
                    )
                    results['successes'].append(result)
                    
                except Exception as e:
                    results['failures'].append(f"transcript-api any: {str(e)}")
                    
        except ImportError:
            results['failures'].append("youtube-transcript-api not available")
        except Exception as e:
            # CRITICAL FIX: Record API failure with 429 detection for rate limiting stats
            rate_manager = get_rate_limit_manager()
            error_str = str(e).lower()
            is_429 = '429' in error_str or 'too many requests' in error_str
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            results['failures'].append(f"transcript-api setup: {str(e)}")
        
        return results
    
    def _extract_with_alternative_configs(self, video_url: str, output_dir: Path, 
                                        video_id: str, video_title: str) -> Dict[str, Any]:
        """Try alternative yt-dlp configurations as last resort"""
        results = {'attempts': 0, 'successes': [], 'failures': []}
        
        # Alternative configurations to try
        alt_configs = [
            # Try with different user agents
            {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'vtt',
                'subtitleslangs': ['all'],
                'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                'outtmpl': str(output_dir / f'{video_title}_alt1.%(ext)s')
            },
            # Try with cookies if available
            {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'json3',
                'subtitleslangs': ['all'],
                'outtmpl': str(output_dir / f'{video_title}_alt2.%(ext)s')
            },
            # Try minimal config
            {
                'skip_download': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'zh', 'zh-CN', 'zh-TW'],
                'outtmpl': str(output_dir / f'{video_title}_alt3.%(ext)s')
            }
        ]
        
        for i, config in enumerate(alt_configs):
            results['attempts'] += 1
            
            try:
                config.update({'quiet': True, 'no_warnings': True})
                
                with yt_dlp.YoutubeDL(config) as ydl:
                    ydl.download([video_url])
                
                # Check for any subtitle files created
                subtitle_files = list(output_dir.glob(f'{video_title}_alt{i+1}.*'))
                subtitle_files = [f for f in subtitle_files if f.suffix in ['.srt', '.vtt', '.json', '.ass', '.ttml']]
                
                for subtitle_file in subtitle_files:
                    if subtitle_file.stat().st_size > 0:
                        lang = self._extract_language_from_filename(subtitle_file.name)
                        fmt = subtitle_file.suffix[1:]  # Remove the dot
                        
                        result = SubtitleResult(
                            success=True,
                            method=f'yt-dlp-alt-{i+1}',
                            language=lang,
                            subtitle_format=fmt,
                            file_path=str(subtitle_file),
                            lines_count=self._count_subtitle_lines(subtitle_file)
                        )
                        results['successes'].append(result)
                
            except Exception as e:
                results['failures'].append(f"alt-config-{i+1}: {str(e)}")
        
        return results
    
    def _extract_language_from_filename(self, filename: str) -> str:
        """Extract language code from subtitle filename"""
        # Look for language codes like .en.srt, .zh-CN.vtt, etc.
        pattern = r'\.([a-z]{2}(?:-[A-Z]{2})?)\.[^.]+$'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        
        # Fallback: check for common language patterns
        if '.en.' in filename:
            return 'en'
        elif '.zh-CN.' in filename or '.zh-cn.' in filename:
            return 'zh-CN'
        elif '.zh-TW.' in filename or '.zh-tw.' in filename:
            return 'zh-TW'
        elif '.zh.' in filename:
            return 'zh'
        else:
            return 'unknown'
    
    def _count_subtitle_lines(self, subtitle_file: Path) -> int:
        """Count the number of subtitle lines in the file"""
        try:
            if subtitle_file.suffix == '.json':
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'events' in data:
                        return len(data['events'])
                    elif isinstance(data, list):
                        return len(data)
                    else:
                        return 1
            else:
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count subtitle blocks (separated by double newlines)
                    blocks = content.split('\n\n')
                    return len([block for block in blocks if block.strip()])
        except:
            return 0
    
    def _select_best_subtitle(self, subtitles: List[SubtitleResult]) -> SubtitleResult:
        """Select the best subtitle from available options"""
        if not subtitles:
            return None
        
        # Sort by: 1) Language preference, 2) Format preference, 3) Line count
        def sort_key(subtitle):
            lang_priority = self.language_priority.index(subtitle.language) if subtitle.language in self.language_priority else 999
            format_priority = self.format_priority.index(subtitle.subtitle_format) if subtitle.subtitle_format in self.format_priority else 999
            return (lang_priority, format_priority, -subtitle.lines_count)
        
        return sorted(subtitles, key=sort_key)[0]
    
    def _convert_to_standard_formats(self, subtitles: List[SubtitleResult], 
                                   output_dir: Path, video_title: str) -> List[str]:
        """Convert all subtitles to standard SRT and TXT formats"""
        final_files = []
        
        best_subtitle = self._select_best_subtitle(subtitles)
        if not best_subtitle:
            return final_files
        
        try:
            # Convert best subtitle to SRT and TXT
            srt_file = output_dir / f'{video_title}.srt'
            txt_file = output_dir / f'{video_title}.txt'
            
            # Convert to SRT if not already
            if best_subtitle.subtitle_format != 'srt':
                srt_content = self._convert_to_srt(best_subtitle.file_path, best_subtitle.subtitle_format)
            else:
                with open(best_subtitle.file_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
            
            # Save SRT
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            final_files.append(str(srt_file))
            
            # Convert SRT to TXT
            txt_content = self._srt_to_txt(srt_content)
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            final_files.append(str(txt_file))
            
            self.logger.info(f"Created standard formats: {srt_file.name}, {txt_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error converting to standard formats: {e}")
        
        return final_files
    
    def _convert_to_srt(self, file_path: str, source_format: str) -> str:
        """Convert various subtitle formats to SRT"""
        # This is a simplified converter - in production, you'd want more robust conversion
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if source_format == 'vtt':
                # Basic VTT to SRT conversion
                lines = content.split('\n')
                srt_lines = []
                counter = 1
                
                for i, line in enumerate(lines):
                    if '-->' in line:
                        # Convert VTT timestamp to SRT format
                        timestamp = line.replace('.', ',')
                        srt_lines.append(str(counter))
                        srt_lines.append(timestamp)
                        counter += 1
                    elif line.strip() and not line.startswith('WEBVTT') and not line.startswith('NOTE'):
                        srt_lines.append(line)
                        if i + 1 < len(lines) and not lines[i + 1].strip():
                            srt_lines.append('')
                
                return '\n'.join(srt_lines)
            
            elif source_format == 'json3':
                # Basic JSON to SRT conversion
                import json
                data = json.loads(content)
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
                
                return '\n'.join(srt_lines)
            
            else:
                # Return as-is for other formats
                return content
                
        except Exception as e:
            self.logger.error(f"Error converting {source_format} to SRT: {e}")
            return ""
    
    def _ms_to_srt_time(self, ms: int) -> str:
        """Convert milliseconds to SRT time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _srt_to_txt(self, srt_content: str) -> str:
        """Convert SRT content to plain text"""
        lines = srt_content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip sequence numbers and timestamps
            if line and not line.isdigit() and '-->' not in line:
                text_lines.append(line)
        
        return '\n'.join(text_lines)


# Convenience function
def extract_subtitles_comprehensive(video_url: str, output_dir: Path, 
                                  video_id: str, video_title: str = "video") -> Dict[str, Any]:
    """Extract subtitles using all available methods"""
    extractor = AdvancedSubtitleExtractor()
    return extractor.extract_subtitles_comprehensive(video_url, output_dir, video_id, video_title)