"""
Storage Path Management - Centralized path generation for consistent storage structure
Ensures all workers use the same directory structure as defined in PRD
"""

from pathlib import Path
from typing import Optional
from config.settings import get_settings


class StoragePaths:
    """
    Centralized storage path management following PRD structure:
    
    /storage_path/
    ├── audio/
    │   └── {channel_id}/
    │       └── {video_id}/
    ├── transcripts/
    │   └── {channel_id}/
    │       └── {video_id}/
    ├── content/
    │   └── {channel_id}/
    │       └── {video_id}/
    └── metadata/
        └── {channel_id}/
            └── {video_id}/
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize storage paths.
        
        Args:
            base_path: Override default storage path from settings
        """
        settings = get_settings()
        self.base_path = base_path or Path(settings.storage_path)
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_audio_path(self, channel_id: str, video_id: str) -> Path:
        """Get path for audio files"""
        path = self.base_path / 'audio' / channel_id / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_transcript_path(self, channel_id: str, video_id: str) -> Path:
        """Get path for transcript files (SRT and TXT)"""
        path = self.base_path / 'transcripts' / channel_id / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_content_path(self, channel_id: str, video_id: str) -> Path:
        """Get path for generated content"""
        path = self.base_path / 'content' / channel_id / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_metadata_path(self, channel_id: str, video_id: str) -> Path:
        """Get path for metadata files"""
        path = self.base_path / 'metadata' / channel_id / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_path_for_type(self, file_type: str, channel_id: str, video_id: str) -> Path:
        """
        Get path based on file type.
        
        Args:
            file_type: One of 'audio', 'transcript', 'content', 'metadata'
            channel_id: YouTube channel ID
            video_id: YouTube video ID
            
        Returns:
            Path object for the specified type
        """
        path_map = {
            'audio': self.get_audio_path,
            'transcript': self.get_transcript_path,
            'transcripts': self.get_transcript_path,  # Allow plural
            'content': self.get_content_path,
            'metadata': self.get_metadata_path
        }
        
        get_path_func = path_map.get(file_type)
        if not get_path_func:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of: {list(path_map.keys())}")
        
        return get_path_func(channel_id, video_id)
    
    def get_file_path(self, file_type: str, channel_id: str, video_id: str, filename: str) -> Path:
        """
        Get full file path including filename.
        
        Args:
            file_type: Type of file
            channel_id: YouTube channel ID
            video_id: YouTube video ID
            filename: Name of the file
            
        Returns:
            Full path to the file
        """
        dir_path = self.get_path_for_type(file_type, channel_id, video_id)
        return dir_path / filename
    
    def list_files(self, file_type: str, channel_id: str, video_id: str) -> list[Path]:
        """
        List all files in a specific directory.
        
        Args:
            file_type: Type of files to list
            channel_id: YouTube channel ID
            video_id: YouTube video ID
            
        Returns:
            List of Path objects for files in the directory
        """
        dir_path = self.get_path_for_type(file_type, channel_id, video_id)
        if dir_path.exists():
            return list(dir_path.iterdir())
        return []
    
    def get_all_channel_videos(self, channel_id: str) -> list[str]:
        """
        Get all video IDs for a channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            List of video IDs that have data stored
        """
        video_ids = set()
        
        # Check each file type directory
        for file_type in ['audio', 'transcripts', 'content', 'metadata']:
            type_path = self.base_path / file_type / channel_id
            if type_path.exists():
                for video_path in type_path.iterdir():
                    if video_path.is_dir():
                        video_ids.add(video_path.name)
        
        return list(video_ids)


# Singleton instance
_storage_paths = None


def get_storage_paths(base_path: Optional[Path] = None) -> StoragePaths:
    """
    Get or create StoragePaths singleton.
    
    Args:
        base_path: Optional override for base storage path
        
    Returns:
        StoragePaths instance
    """
    global _storage_paths
    if _storage_paths is None or base_path is not None:
        _storage_paths = StoragePaths(base_path)
    return _storage_paths