"""
Storage Path Management V2 - ID-based structure with human-readable filenames
Implements new structure: {channel_id}/{video_id}/{type}/{readable_filename}
Maintains backward compatibility with V1 structure
"""

import json
import logging
import threading
import tempfile
import shutil
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from config.settings import get_settings
from core.filename_sanitizer import (
    sanitize_filename, sanitize_channel_id, sanitize_video_id, FilenameSanitizer
)
from core.performance_optimizations import BatchFileOperations, cached_async, FileStreamManager
# V1 import removed - V1 is deprecated and archived
# If V1 compatibility is needed, import from archived location:
# from archived.v1_storage_structure.storage_paths_v1 import StoragePaths as StoragePathsV1
StoragePathsV1 = None  # V1 is deprecated

logger = logging.getLogger(__name__)


class StorageVersion(Enum):
    """Storage structure versions."""
    V1 = "v1"  # Type-based: /audio/channel/video/, /transcripts/channel/video/
    V2 = "v2"  # ID-based: /channel_id/video_id/media/, /channel_id/video_id/transcripts/


class StoragePathsV2:
    """
    V2 Storage path management with ID-based structure and readable filenames.
    
    New structure:
    /storage_path/
    ├── {channel_id}/
    │   ├── .channel_info.json           # Channel metadata
    │   ├── .video_index.json            # Video listing
    │   └── {video_id}/
    │       ├── .metadata.json           # Video metadata
    │       ├── .processing_complete     # Completion marker
    │       ├── media/                   # Audio/video files
    │       │   └── {video_title}.{ext}
    │       ├── transcripts/             # Transcript files
    │       │   ├── {video_title}.srt
    │       │   └── {video_title}.txt
    │       ├── content/                 # Generated content
    │       │   ├── blog_{title}_{job_id}.md
    │       │   └── social_{title}_{job_id}.txt
    │       └── metadata/                # Additional metadata
    │           └── yt_metadata.json
    """
    
    def __init__(self, base_path: Optional[Path] = None, version: StorageVersion = StorageVersion.V2):
        """
        Initialize V2 storage paths with compatibility support.
        
        Args:
            base_path: Override default storage path from settings
            version: Storage version to use (V1 or V2)
        """
        settings = get_settings()
        self.base_path = base_path or Path(settings.storage_path)
        self.version = version
        
        # Thread safety for concurrent operations
        self._lock = threading.RLock()
        
        # Initialize sanitizer with cache file for collision persistence
        cache_file = str(self.base_path / ".sanitizer_cache.json")
        self.sanitizer = FilenameSanitizer(strict_mode=True, cache_file=cache_file)
        
        # Performance optimization tools
        self.batch_ops = BatchFileOperations(max_batch_size=50)
        self.file_manager = FileStreamManager()
        
        # V1 compatibility layer - disabled as V1 is deprecated
        self.v1_storage = None  # V1 is deprecated and archived
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Load storage config
        self.config_path = self.base_path / ".storage_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load storage configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load storage config: {e}")
        
        # Default config
        config = {
            "version": self.version.value,
            "created_at": datetime.utcnow().isoformat(),
            "features": {
                "readable_filenames": True,
                "channel_index": True,
                "completion_markers": True
            }
        }
        
        # Save default config
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save storage config: {e}")
        
        return config
    
    def _validate_and_sanitize_ids(self, channel_id: str, video_id: str = None) -> Tuple[str, Optional[str]]:
        """
        Validate and sanitize channel_id and video_id for secure path usage.
        
        Args:
            channel_id: YouTube channel ID to validate and sanitize
            video_id: YouTube video ID to validate and sanitize (optional)
            
        Returns:
            Tuple of (sanitized_channel_id, sanitized_video_id)
            
        Raises:
            ValueError: If IDs are invalid or contain malicious patterns
        """
        # Use robust sanitization from filename_sanitizer module
        try:
            safe_channel_id = sanitize_channel_id(channel_id)
        except ValueError as e:
            logger.error(f"Invalid channel_id '{channel_id}': {e}")
            raise ValueError(f"Invalid channel_id: {e}")
        
        safe_video_id = None
        if video_id:
            try:
                safe_video_id = sanitize_video_id(video_id)
            except ValueError as e:
                logger.error(f"Invalid video_id '{video_id}': {e}")
                raise ValueError(f"Invalid video_id: {e}")
        
        return safe_channel_id, safe_video_id

    def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically write JSON data to file using temp file + rename pattern.
        
        Args:
            file_path: Target file path
            data: Data to write
            
        Raises:
            IOError: If write operation fails
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temporary file in same directory for atomic rename
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f".tmp_{file_path.name}_",
                suffix='.json'
            )
            
            # Close the file descriptor immediately and use the path
            os.close(temp_fd)
            temp_fd = None  # Mark as closed
            
            # Write data to temporary file using path
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (works on all platforms)
            shutil.move(temp_path, str(file_path))
            
        except Exception as e:
            # Cleanup temp file if it exists
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except:
                    pass
            raise IOError(f"Failed to write {file_path}: {e}")
        finally:
            # Close temp file descriptor if still open
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass

    # ========================================
    # Core Path Methods
    # ========================================
    
    def get_video_dir(self, channel_id: str, video_id: str) -> Path:
        """Get base directory for a video."""
        # Input validation and sanitization for security
        safe_channel_id, safe_video_id = self._validate_and_sanitize_ids(channel_id, video_id)
        
        if self.version == StorageVersion.V1:
            # V1 doesn't have a single video dir, return audio path as base
            return self.v1_storage.get_audio_path(safe_channel_id, safe_video_id)
        
        with self._lock:
            # Use sanitized IDs to prevent path traversal attacks
            path = self.base_path / safe_channel_id / safe_video_id
            path.mkdir(parents=True, exist_ok=True)
            return path
    
    def get_channel_dir(self, channel_id: str) -> Path:
        """Get base directory for a channel."""
        # Input validation and sanitization for security
        safe_channel_id, _ = self._validate_and_sanitize_ids(channel_id)
        
        with self._lock:
            # Use sanitized ID to prevent path traversal attacks
            path = self.base_path / safe_channel_id
            path.mkdir(parents=True, exist_ok=True)
            return path
    
    # ========================================
    # Media Files (Audio/Video)
    # ========================================
    
    def get_media_dir(self, channel_id: str, video_id: str) -> Path:
        """Get media directory for audio/video files."""
        if self.version == StorageVersion.V1:
            return self.v1_storage.get_audio_path(channel_id, video_id)
        
        path = self.get_video_dir(channel_id, video_id) / "media"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_media_file(
        self, 
        channel_id: str, 
        video_id: str,
        video_title: Optional[str] = None,
        format: str = "opus"
    ) -> Path:
        """
        Get path for media file with readable name.
        
        Args:
            channel_id: YouTube channel ID
            video_id: YouTube video ID
            video_title: Optional video title for readable filename
            format: File format (opus, mp4, etc.)
        """
        media_dir = self.get_media_dir(channel_id, video_id)
        
        if self.version == StorageVersion.V1 or not video_title:
            # V1 or no title: use video_id
            return media_dir / f"{video_id}.{format}"
        
        # V2 with title: use sanitized title
        safe_title = sanitize_filename(video_title, video_id)
        return media_dir / f"{safe_title}.{format}"
    
    def find_media_files(self, channel_id: str, video_id: str) -> List[Path]:
        """Find all media files for a video (handles any filename)."""
        media_dir = self.get_media_dir(channel_id, video_id)
        if not media_dir.exists():
            return []
        
        # Common media extensions
        extensions = ['opus', 'mp4', 'webm', 'm4a', 'mp3', 'mkv', 'avi']
        files = []
        
        for ext in extensions:
            # Try video_id based name
            id_file = media_dir / f"{video_id}.{ext}"
            if id_file.exists():
                files.append(id_file)
            
            # Try any file with this extension
            files.extend(media_dir.glob(f"*.{ext}"))
        
        # Remove duplicates and return
        return list(set(files))
    
    # ========================================
    # Transcript Files
    # ========================================
    
    def get_transcript_dir(self, channel_id: str, video_id: str) -> Path:
        """Get transcript directory."""
        if self.version == StorageVersion.V1:
            return self.v1_storage.get_transcript_path(channel_id, video_id)
        
        path = self.get_video_dir(channel_id, video_id) / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_transcript_file(
        self,
        channel_id: str,
        video_id: str,
        video_title: Optional[str] = None,
        format: str = "srt"
    ) -> Path:
        """Get path for transcript file with readable name."""
        transcript_dir = self.get_transcript_dir(channel_id, video_id)
        
        if self.version == StorageVersion.V1 or not video_title:
            return transcript_dir / f"{video_id}.{format}"
        
        safe_title = sanitize_filename(video_title, video_id)
        return transcript_dir / f"{safe_title}.{format}"
    
    def find_transcript_files(self, channel_id: str, video_id: str) -> Dict[str, Path]:
        """Find all transcript files, return dict by type."""
        transcript_dir = self.get_transcript_dir(channel_id, video_id)
        if not transcript_dir.exists():
            return {}
        
        files = {}
        
        # Look for SRT files
        srt_files = list(transcript_dir.glob("*.srt"))
        if srt_files:
            files['srt'] = srt_files[0]
        
        # Look for TXT files
        txt_files = list(transcript_dir.glob("*.txt"))
        if txt_files:
            files['txt'] = txt_files[0]
        
        # Look for VTT files (though we prefer SRT)
        vtt_files = list(transcript_dir.glob("*.vtt"))
        if vtt_files:
            files['vtt'] = vtt_files[0]
        
        return files
    
    # ========================================
    # Generated Content
    # ========================================
    
    def get_content_dir(self, channel_id: str, video_id: str) -> Path:
        """Get generated content directory."""
        if self.version == StorageVersion.V1:
            return self.v1_storage.get_content_path(channel_id, video_id)
        
        path = self.get_video_dir(channel_id, video_id) / "content"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_content_file(
        self,
        channel_id: str,
        video_id: str,
        content_type: str,
        job_id: str,
        video_title: Optional[str] = None,
        format: str = "txt"
    ) -> Path:
        """Get path for generated content file."""
        content_dir = self.get_content_dir(channel_id, video_id)
        
        if self.version == StorageVersion.V1 or not video_title:
            return content_dir / f"{content_type}_{job_id}.{format}"
        
        # Use shortened title for content files
        safe_title = sanitize_filename(video_title, video_id, max_length=40)
        return content_dir / f"{content_type}_{safe_title}_{job_id[-6:]}.{format}"
    
    # ========================================
    # Metadata Files
    # ========================================
    
    def get_metadata_dir(self, channel_id: str, video_id: str) -> Path:
        """Get metadata directory."""
        if self.version == StorageVersion.V1:
            return self.v1_storage.get_metadata_path(channel_id, video_id)
        
        path = self.get_video_dir(channel_id, video_id) / "metadata"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_video_metadata_file(self, channel_id: str, video_id: str) -> Path:
        """Get video metadata file path."""
        if self.version == StorageVersion.V1:
            return self.get_metadata_dir(channel_id, video_id) / "metadata.json"
        
        # V2: Store in video root as hidden file
        return self.get_video_dir(channel_id, video_id) / ".metadata.json"
    
    def save_video_metadata(
        self,
        channel_id: str,
        video_id: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """Save video metadata using atomic write operations."""
        # Input validation and sanitization
        safe_channel_id, safe_video_id = self._validate_and_sanitize_ids(channel_id, video_id)
        
        metadata_file = self.get_video_metadata_file(channel_id, video_id)
        
        # Add storage version info and sanitize data
        safe_metadata = {
            **{k: str(v)[:10000] if isinstance(v, str) else v 
               for k, v in metadata.items()},  # Limit string lengths
            'storage_version': self.version.value,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        try:
            with self._lock:
                self._atomic_write_json(metadata_file, safe_metadata)
        except Exception as e:
            logger.error(f"Failed to save metadata for {video_id}: {e}")
            raise
        
        return metadata_file
    
    # ========================================
    # Channel Index (V2 only)
    # ========================================
    
    def get_channel_info_file(self, channel_id: str) -> Path:
        """Get channel info file path."""
        return self.get_channel_dir(channel_id) / ".channel_info.json"
    
    def save_channel_info(self, channel_id: str, info: Dict[str, Any]) -> Path:
        """Save channel information using atomic operations."""
        # Input validation and sanitization
        safe_channel_id, _ = self._validate_and_sanitize_ids(channel_id)
        
        info_file = self.get_channel_info_file(channel_id)
        
        try:
            with self._lock:
                # Load existing if present
                existing = {}
                if info_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        logger.warning(f"Corrupted channel info file: {info_file}")
                        existing = {}
                
                # Sanitize and merge with new info
                safe_info = {k: str(v)[:5000] if isinstance(v, str) else v 
                           for k, v in info.items()}
                existing.update(safe_info)
                existing['updated_at'] = datetime.utcnow().isoformat()
                
                # Atomic save
                self._atomic_write_json(info_file, existing)
                
        except Exception as e:
            logger.error(f"Failed to save channel info for {channel_id}: {e}")
            raise
        
        return info_file
    
    def get_video_index_file(self, channel_id: str) -> Path:
        """Get video index file path."""
        return self.get_channel_dir(channel_id) / ".video_index.json"
    
    def update_video_index(
        self,
        channel_id: str,
        video_id: str,
        video_info: Dict[str, Any]
    ):
        """Update video index for a channel using atomic operations."""
        # Input validation and sanitization
        safe_channel_id, safe_video_id = self._validate_and_sanitize_ids(channel_id, video_id)
        
        index_file = self.get_video_index_file(channel_id)
        
        try:
            with self._lock:
                # Load existing index
                index = {}
                if index_file.exists():
                    try:
                        with open(index_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Handle both list and dict formats
                        if isinstance(data, list):
                            # Convert list format to dict format
                            index = {}
                            for item in data:
                                if isinstance(item, dict) and 'video_id' in item:
                                    vid_id = item.get('video_id')
                                    index[vid_id] = {
                                        'title': item.get('title', 'Unknown'),
                                        'published_at': item.get('upload_date'),
                                        'duration': item.get('duration'),
                                        'added_at': item.get('downloaded_at', datetime.utcnow().isoformat()),
                                        'has_transcript': item.get('has_transcript', False),
                                        'has_media': item.get('has_media', True)
                                    }
                        elif isinstance(data, dict):
                            index = data
                        else:
                            logger.warning(f"Unexpected video index format: {type(data)}")
                            index = {}
                            
                    except (json.JSONDecodeError, IOError):
                        logger.warning(f"Corrupted video index file: {index_file}")
                        index = {}
                
                # Sanitize video info
                safe_info = {}
                for key, value in video_info.items():
                    if isinstance(value, str):
                        safe_info[key] = value[:1000]  # Limit string lengths
                    else:
                        safe_info[key] = value
                
                # Update with new video
                index[video_id] = {
                    'title': safe_info.get('title', 'Unknown'),
                    'published_at': safe_info.get('published_at'),
                    'duration': safe_info.get('duration'),
                    'added_at': datetime.utcnow().isoformat(),
                    'has_transcript': safe_info.get('has_transcript', False),
                    'has_media': safe_info.get('has_media', False)
                }
                
                # Atomic save
                self._atomic_write_json(index_file, index)
                
        except Exception as e:
            logger.error(f"Failed to update video index: {e}")
            raise
    
    # ========================================
    # Completion Markers (V2 only)
    # ========================================
    
    def mark_processing_complete(
        self,
        channel_id: str,
        video_id: str,
        processing_info: Optional[Dict[str, Any]] = None
    ):
        """Mark video processing as complete."""
        if self.version == StorageVersion.V1:
            return  # V1 doesn't have completion markers
        
        marker_file = self.get_video_dir(channel_id, video_id) / ".processing_complete"
        
        info = processing_info or {}
        info['completed_at'] = datetime.utcnow().isoformat()
        info['storage_version'] = self.version.value
        
        try:
            with open(marker_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create completion marker: {e}")
    
    def is_processing_complete(self, channel_id: str, video_id: str) -> bool:
        """Check if video processing is complete."""
        if self.version == StorageVersion.V1:
            # V1: Check if files exist
            transcript_dir = self.v1_storage.get_transcript_path(channel_id, video_id)
            return transcript_dir.exists() and any(transcript_dir.iterdir())
        
        marker_file = self.get_video_dir(channel_id, video_id) / ".processing_complete"
        return marker_file.exists()
    
    # ========================================
    # Migration Support
    # ========================================
    
    def detect_storage_version(self, channel_id: str, video_id: str) -> StorageVersion:
        """Detect which storage version a video uses."""
        # Input validation and sanitization for security
        safe_channel_id, safe_video_id = self._validate_and_sanitize_ids(channel_id, video_id)
        
        # Check for V2 structure using sanitized IDs
        v2_path = self.base_path / safe_channel_id / safe_video_id
        if v2_path.exists() and (v2_path / "media").exists():
            return StorageVersion.V2
        
        # Check for V1 structure using sanitized IDs
        v1_audio = self.base_path / "audio" / safe_channel_id / safe_video_id
        v1_transcript = self.base_path / "transcripts" / safe_channel_id / safe_video_id
        if v1_audio.exists() or v1_transcript.exists():
            return StorageVersion.V1
        
        # Default to configured version
        return self.version
    
    def get_compatible_path(
        self,
        file_type: str,
        channel_id: str,
        video_id: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Get path that works with both V1 and V2 structures.
        Automatically detects which version to use.
        """
        detected_version = self.detect_storage_version(channel_id, video_id)
        
        if detected_version == StorageVersion.V1:
            # Use V1 paths
            if file_type == 'media' or file_type == 'audio':
                base = self.v1_storage.get_audio_path(channel_id, video_id)
            elif file_type == 'transcript':
                base = self.v1_storage.get_transcript_path(channel_id, video_id)
            elif file_type == 'content':
                base = self.v1_storage.get_content_path(channel_id, video_id)
            else:
                base = self.v1_storage.get_metadata_path(channel_id, video_id)
            
            return base / filename if filename else base
        else:
            # Use V2 paths
            if file_type == 'media' or file_type == 'audio':
                base = self.get_media_dir(channel_id, video_id)
            elif file_type == 'transcript':
                base = self.get_transcript_dir(channel_id, video_id)
            elif file_type == 'content':
                base = self.get_content_dir(channel_id, video_id)
            else:
                base = self.get_metadata_dir(channel_id, video_id)
            
            return base / filename if filename else base
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def list_all_channels(self) -> List[str]:
        """List all channel IDs with data."""
        channels = set()
        
        # V2 structure: direct subdirectories
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                # Check if it looks like a channel ID
                if path.name.startswith('UC') or path.name.startswith('@'):
                    channels.add(path.name)
        
        # V1 structure: check type directories
        for type_dir in ['audio', 'transcripts', 'content', 'metadata']:
            type_path = self.base_path / type_dir
            if type_path.exists():
                for channel_path in type_path.iterdir():
                    if channel_path.is_dir():
                        channels.add(channel_path.name)
        
        return sorted(list(channels))
    
    def list_channel_videos(self, channel_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        List all videos for a channel with metadata.
        Returns list of (video_id, metadata) tuples.
        """
        # Input validation and sanitization for security
        safe_channel_id, _ = self._validate_and_sanitize_ids(channel_id)
        
        videos = []
        
        # Try V2 index first - using original channel_id for internal calls
        index_file = self.get_video_index_file(channel_id)
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                    for video_id, info in index.items():
                        videos.append((video_id, info))
                    return sorted(videos, key=lambda x: x[1].get('added_at', ''), reverse=True)
            except Exception:
                pass
        
        # Fall back to directory scanning using sanitized ID
        video_ids = set()
        
        # V2 structure using sanitized channel_id
        channel_path = self.base_path / safe_channel_id
        if channel_path.exists():
            for video_path in channel_path.iterdir():
                if video_path.is_dir() and not video_path.name.startswith('.'):
                    video_ids.add(video_path.name)
        
        # V1 structure using sanitized channel_id  
        for type_dir in ['audio', 'transcripts', 'content']:
            type_channel_path = self.base_path / type_dir / safe_channel_id
            if type_channel_path.exists():
                for video_path in type_channel_path.iterdir():
                    if video_path.is_dir():
                        video_ids.add(video_path.name)
        
        # Return with minimal metadata
        for video_id in sorted(video_ids):
            videos.append((video_id, {'title': 'Unknown', 'has_data': True}))
        
        return videos
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'version': self.version.value,
            'base_path': str(self.base_path),
            'total_channels': 0,
            'total_videos': 0,
            'total_size_mb': 0,
            'by_type': {}
        }
        
        try:
            # Count channels and videos
            channels = self.list_all_channels()
            stats['total_channels'] = len(channels)
            
            total_videos = 0
            for channel_id in channels:
                videos = self.list_channel_videos(channel_id)
                total_videos += len(videos)
            stats['total_videos'] = total_videos
            
            # Calculate sizes by type
            for type_name in ['media', 'transcripts', 'content', 'metadata']:
                type_size = 0
                type_count = 0
                
                # V2 structure
                for channel_path in self.base_path.glob(f"*/*/{type_name}"):
                    if channel_path.is_dir():
                        for file in channel_path.rglob('*'):
                            if file.is_file():
                                type_size += file.stat().st_size
                                type_count += 1
                
                # V1 structure
                type_path = self.base_path / type_name.rstrip('s')  # Remove plural
                if type_path.exists():
                    for file in type_path.rglob('*'):
                        if file.is_file():
                            type_size += file.stat().st_size
                            type_count += 1
                
                stats['by_type'][type_name] = {
                    'files': type_count,
                    'size_mb': round(type_size / (1024 * 1024), 2)
                }
                stats['total_size_mb'] += stats['by_type'][type_name]['size_mb']
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
        
        return stats
    
    # ========================================
    # Performance Optimizations
    # ========================================
    @cached_async(ttl=300)  # 5-minute cache
    async def get_channel_videos_cached(self, channel_id: str) -> List[str]:
        """Get list of video IDs for a channel with caching."""
        # Input validation and sanitization for security
        safe_channel_id, _ = self._validate_and_sanitize_ids(channel_id)
        
        # Use sanitized channel_id for path construction
        channel_path = self.base_path / safe_channel_id
        if not channel_path.exists():
            return []
        
        video_ids = []
        try:
            for item in channel_path.iterdir():
                if item.is_dir() and len(item.name) == 11:  # YouTube video ID length
                    video_ids.append(item.name)
        except Exception as e:
            logger.error(f"Error scanning channel {channel_id}: {e}")
        
        return sorted(video_ids)
    
    async def copy_video_files_batch(self, operations: List[Tuple[str, str, str]]) -> None:
        """
        Batch copy video files for performance.
        
        Args:
            operations: List of (source_channel_id, source_video_id, dest_video_id) tuples
        """
        for source_channel, source_video, dest_video in operations:
            # Input validation and sanitization for security
            safe_source_channel, safe_source_video = self._validate_and_sanitize_ids(source_channel, source_video)
            safe_dest_channel, safe_dest_video = self._validate_and_sanitize_ids(source_channel, dest_video)
            
            # Use sanitized IDs for path construction 
            source_dir = self.base_path / safe_source_channel / safe_source_video
            dest_dir = self.base_path / safe_dest_channel / safe_dest_video
            
            if source_dir.exists():
                # Add all files to batch operation
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(source_dir)
                        dest_file = dest_dir / relative_path
                        await self.batch_ops.add_copy_operation(file_path, dest_file)
        
        # Execute batch
        await self.batch_ops.execute_batch()
    
    async def stream_large_file_write(self, file_path: Path, content_generator) -> None:
        """Write large files using streaming to avoid memory issues."""
        await self.file_manager.stream_write_text(file_path, content_generator)
    
    async def stream_large_file_read(self, file_path: Path, chunk_size: int = 8192):
        """Read large files using streaming."""
        async for chunk in self.file_manager.stream_read_text(file_path, chunk_size):
            yield chunk
    
    async def cleanup_temp_files_batch(self, max_age_hours: int = 24) -> int:
        """Batch cleanup of temporary files older than specified age."""
        import time
        from datetime import timedelta
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        temp_patterns = ['*.tmp', '.tmp_*', '*.partial']
        files_cleaned = 0
        
        for pattern in temp_patterns:
            for temp_file in self.base_path.rglob(pattern):
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        await self.batch_ops.add_delete_operation(temp_file)
                        files_cleaned += 1
                except (OSError, FileNotFoundError):
                    continue
        
        # Execute batch deletion
        await self.batch_ops.execute_batch()
        logger.info(f"Cleaned up {files_cleaned} temporary files")
        return files_cleaned
    
    @cached_async(ttl=3600)  # 1-hour cache
    async def get_directory_size_cached(self, directory_path: Path) -> int:
        """Get directory size with caching for expensive operations."""
        if not directory_path.exists():
            return 0
        
        total_size = 0
        try:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory_path}: {e}")
        
        return total_size


# ========================================
# Singleton and Factory
# ========================================

_storage_v2_instance = None


def get_storage_paths_v2(
    base_path: Optional[Path] = None,
    version: Optional[StorageVersion] = None
) -> StoragePathsV2:
    """
    Get or create StoragePathsV2 singleton.
    
    Args:
        base_path: Optional override for base storage path
        version: Optional storage version override
        
    Returns:
        StoragePathsV2 instance
    """
    global _storage_v2_instance
    
    if _storage_v2_instance is None or base_path is not None or version is not None:
        # Determine version from settings or auto-detect
        if version is None:
            settings = get_settings()
            version_str = getattr(settings, 'storage_version', 'v2')
            version = StorageVersion.V2 if version_str == 'v2' else StorageVersion.V1
        
        _storage_v2_instance = StoragePathsV2(base_path, version)
    
    return _storage_v2_instance


def migrate_to_v2(channel_id: str, video_id: str) -> bool:
    """
    Migrate a single video from V1 to V2 structure.
    
    Args:
        channel_id: Channel ID
        video_id: Video ID
        
    Returns:
        True if migration successful
    """
    storage = get_storage_paths_v2()
    
    # Implementation would go here
    # This is a placeholder for the migration logic
    logger.info(f"Would migrate {channel_id}/{video_id} to V2")
    return True