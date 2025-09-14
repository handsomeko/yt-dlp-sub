"""YouTube Downloader Module - Handles all video/audio downloading"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yt_dlp
from datetime import datetime


class YouTubeDownloader:
    """Modular YouTube downloader using yt-dlp"""
    
    def __init__(self, base_path: str = None):
        """Initialize downloader with configurable paths"""
        if base_path is None:
            # Use external drive if available, otherwise local
            if os.path.exists('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub'):
                base_path = '/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads'
            else:
                base_path = os.path.join(os.path.dirname(__file__), '../downloads')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Extract video metadata without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'channel': info.get('channel'),
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'description': info.get('description'),
                    'thumbnail': info.get('thumbnail'),
                    'view_count': info.get('view_count'),
                }
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
                      download_audio_only: bool = False,
                      audio_format: str = 'mp3',
                      video_format: str = 'mp4',
                      channel_name: Optional[str] = None) -> Dict[str, Any]:
        """Download video with specified quality and format
        
        Args:
            url: YouTube URL
            quality: Video quality - '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p', 'best', 'worst'
            download_audio_only: If True, download only audio
            audio_format: Audio format - 'mp3', 'm4a', 'wav', 'opus', 'flac', 'best'
            video_format: Video container format - 'mp4', 'mkv', 'webm', 'avi'
            channel_name: Optional channel name for organization
        """
        
        # Get video info first
        info = self.get_video_info(url)
        if 'error' in info:
            return info
            
        # Create folder structure
        video_title = self._sanitize_filename(info['title'])
        if channel_name:
            output_dir = self.base_path / self._sanitize_filename(channel_name) / video_title
        else:
            output_dir = self.base_path / video_title
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure download options
        if download_audio_only:
            # Audio only download
            audio_quality_map = {
                'mp3': 'bestaudio/best',
                'm4a': 'bestaudio[ext=m4a]/bestaudio',
                'opus': 'bestaudio[ext=opus]/bestaudio',
                'wav': 'bestaudio/best',
                'flac': 'bestaudio/best',
                'best': 'bestaudio/best'
            }
            format_str = audio_quality_map.get(audio_format, 'bestaudio/best')
            output_template = str(output_dir / f'{video_title}.%(ext)s')
            
            # Add postprocessor for audio conversion if needed
            postprocessors = []
            if audio_format in ['mp3', 'wav', 'flac']:
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
            output_template = str(output_dir / f'{video_title}.{video_format}')
            
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
            'writesubtitles': True,
            'subtitlesformat': 'srt',
            'subtitleslangs': ['en'],
            'merge_output_format': video_format if not download_audio_only else None,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': postprocessors,
        }
        
        # Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                
                # Convert SRT to plain text transcript if SRT was downloaded
                srt_files = list(output_dir.glob('*.srt'))
                for srt_file in srt_files:
                    txt_file = self._convert_srt_to_txt(srt_file, output_dir)
                    if txt_file:
                        print(f"Created transcript: {txt_file.name}")
                
                # Find all downloaded files
                files = list(output_dir.glob('*'))
                result = {
                    'status': 'success',
                    'video_id': info['id'],
                    'title': info['title'],
                    'output_dir': str(output_dir),
                    'files': [str(f) for f in files],
                    'download_time': datetime.now().isoformat()
                }
                
                # Save metadata
                metadata_file = output_dir / 'metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                return result
                
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'video_id': info.get('id'),
                    'title': info.get('title')
                }
    
    def download_subtitles_only(self, url: str) -> Dict[str, Any]:
        """Download only subtitles without video"""
        info = self.get_video_info(url)
        if 'error' in info:
            return info
            
        video_title = self._sanitize_filename(info['title'])
        output_dir = self.base_path / video_title
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'subtitleslangs': ['en'],
            'outtmpl': str(output_dir / f'{video_title}.%(ext)s'),
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                srt_files = list(output_dir.glob('*.srt'))
                
                # Convert SRT files to plain text transcripts
                txt_files = []
                for srt_file in srt_files:
                    txt_file = self._convert_srt_to_txt(srt_file, output_dir)
                    if txt_file:
                        txt_files.append(str(txt_file))
                        print(f"Created transcript: {txt_file.name}")
                
                return {
                    'status': 'success',
                    'video_id': info['id'],
                    'title': info['title'],
                    'subtitle_files': [str(f) for f in srt_files],
                    'transcript_files': txt_files
                }
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]  # Limit length