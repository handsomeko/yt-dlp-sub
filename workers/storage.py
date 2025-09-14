"""
Storage Worker for the YouTube Content Intelligence & Repurposing Platform.

This worker handles multi-backend storage synchronization with phase-based configuration:
- Phase 1: Local + Google Drive  
- Phase 2: + Supabase
- Phase 3: + S3/GCS

Storage structure:
/{channel_id}/{video_id}/
  ├── audio/
  ├── transcripts/
  ├── content/
  └── metadata/
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .base import BaseWorker, WorkerStatus
from config.settings import get_settings
from core.database import db_manager, StorageSync
from core.service_credentials import GoogleDriveCredentials, AirtableCredentials
from core.storage_paths_v2 import get_storage_paths_v2


class StorageBackendError(Exception):
    """Base exception for storage backend errors."""
    pass


class LocalStorageError(StorageBackendError):
    """Exception raised for local storage errors."""
    pass


class GoogleDriveError(StorageBackendError):
    """Exception raised for Google Drive storage errors."""
    pass


class AirtableError(StorageBackendError):
    """Exception raised for Airtable storage errors."""
    pass


class StorageCredentialsError(StorageBackendError):
    """Exception raised when storage credentials are invalid or missing."""
    pass


class StorageSpaceError(StorageBackendError):
    """Exception raised when storage space is insufficient."""
    pass


class StoragePermissionError(StorageBackendError):
    """Exception raised for storage permission issues."""
    pass


class NetworkStorageError(StorageBackendError):
    """Exception raised for network-related storage errors."""
    pass


class GoogleDriveBackend:
    """Google Drive storage backend (Phase 1 implementation)."""
    
    def __init__(self, credentials_file: Optional[str] = None, folder_id: Optional[str] = None):
        # Try to get credentials from vault first, fallback to provided params
        gdrive_creds = GoogleDriveCredentials()
        
        # Use provided params if available, otherwise use vault credentials
        self.credentials_file = credentials_file or gdrive_creds.credentials_file
        self.folder_id = folder_id or gdrive_creds.folder_id
        self.enabled = bool(self.credentials_file and self.folder_id)
        self.service = None
        
        if self.enabled:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                
                # Load credentials from service account file
                if Path(self.credentials_file).exists():
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_file,
                        scopes=['https://www.googleapis.com/auth/drive.file']
                    )
                    self.service = build('drive', 'v3', credentials=credentials)
                else:
                    self.enabled = False
                    logging.warning(f"Google Drive credentials file not found: {self.credentials_file}")
                    logging.info("To enable Google Drive storage, set GOOGLE_DRIVE_CREDENTIALS_FILE and GOOGLE_DRIVE_FOLDER_ID in .env")
            except ImportError:
                self.enabled = False
                logging.warning("Google Drive dependencies not installed")
            except Exception as e:
                self.enabled = False
                logging.error(f"Failed to initialize Google Drive: {str(e)}")
        
    async def upload_file(self, local_path: Path, remote_path: str) -> Dict[str, Any]:
        """
        Upload file to Google Drive with enhanced error handling.
        
        Args:
            local_path: Local file path
            remote_path: Remote path structure
            
        Returns:
            Dict with file_id, url, and metadata
            
        Raises:
            GoogleDriveError: When Google Drive operations fail
        """
        if not self.enabled or not self.service:
            raise GoogleDriveError("Google Drive not configured or disabled")
        
        # Validate local file
        if not local_path.exists():
            raise GoogleDriveError(f"Local file does not exist: {local_path}")
        
        if not local_path.is_file():
            raise GoogleDriveError(f"Path is not a file: {local_path}")
        
        try:
            file_size = local_path.stat().st_size
        except OSError as e:
            raise GoogleDriveError(f"Cannot read local file: {str(e)}")
        
        if file_size == 0:
            raise GoogleDriveError(f"Cannot upload empty file: {local_path}")
        
        # Check Google Drive quota (if possible)
        try:
            about = self.service.about().get(fields='storageQuota').execute()
            quota = about.get('storageQuota', {})
            if quota.get('limit') and quota.get('usage'):
                used = int(quota['usage'])
                limit = int(quota['limit'])
                available = limit - used
                
                if available < file_size:
                    raise GoogleDriveError(
                        f"Insufficient Google Drive quota: {available} bytes available, need {file_size} bytes"
                    )
        except Exception as e:
            # Quota check is optional, don't fail upload if it fails
            logging.warning(f"Could not check Google Drive quota: {str(e)}")
        
        try:
            from googleapiclient.http import MediaFileUpload
            from googleapiclient.errors import HttpError
            
            # Prepare file metadata
            file_metadata = {
                'name': local_path.name,
                'parents': [self.folder_id],
                'description': f'Uploaded from yt-dl-sub: {remote_path}'
            }
            
            # Determine MIME type
            mime_types = {
                '.opus': 'audio/opus',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.srt': 'text/plain',
                '.txt': 'text/plain',
                '.json': 'application/json',
                '.md': 'text/markdown',
                '.html': 'text/html'
            }
            mime_type = mime_types.get(local_path.suffix.lower(), 'application/octet-stream')
            
            # Create media upload with proper chunk size for large files
            chunk_size = min(1024 * 1024 * 10, file_size)  # 10MB or file size, whichever is smaller
            media = MediaFileUpload(
                str(local_path),
                mimetype=mime_type,
                resumable=file_size > chunk_size,
                chunksize=chunk_size
            )
            
            # Upload file with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    file = self.service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id,name,size,webViewLink,webContentLink,createdTime'
                    ).execute()
                    
                    # Verify upload
                    uploaded_size = int(file.get('size', 0))
                    if uploaded_size != file_size:
                        self.log_with_context(
                            f"Google Drive upload size mismatch: expected {file_size}, got {uploaded_size}",
                            level="WARNING"
                        )
                    
                    return {
                        "status": "success",
                        "file_id": file.get('id'),
                        "name": file.get('name'),
                        "size": uploaded_size,
                        "url": file.get('webViewLink'),
                        "download_url": file.get('webContentLink'),
                        "created_time": file.get('createdTime'),
                        "message": "File uploaded to Google Drive successfully"
                    }
                    
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504] and attempt < max_retries - 1:
                        # Server error, retry
                        wait_time = 2 ** attempt
                        logging.warning(f"Google Drive upload failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    elif e.resp.status == 403:
                        error_details = e.error_details[0] if e.error_details else {}
                        if error_details.get('reason') == 'storageQuotaExceeded':
                            raise GoogleDriveError("Google Drive storage quota exceeded")
                        elif error_details.get('reason') == 'rateLimitExceeded':
                            raise GoogleDriveError("Google Drive API rate limit exceeded")
                        else:
                            raise GoogleDriveError(f"Google Drive access forbidden: {str(e)}")
                    elif e.resp.status == 401:
                        raise StorageCredentialsError("Google Drive authentication failed - check credentials")
                    else:
                        raise GoogleDriveError(f"Google Drive API error: {str(e)}")
                        
        except GoogleDriveError:
            raise
        except StorageCredentialsError:
            raise
        except ImportError as e:
            raise GoogleDriveError(f"Google Drive dependencies not available: {str(e)}")
        except Exception as e:
            raise GoogleDriveError(f"Unexpected Google Drive upload error: {str(e)}")
    
    async def download_file(self, file_id: str, local_path: Path) -> bool:
        """Download file from Google Drive."""
        if not self.enabled or not self.service:
            return False
        
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Write to local file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(fh.getvalue())
            
            return True
            
        except Exception as e:
            logging.error(f"Google Drive download failed: {str(e)}")
            return False
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file from Google Drive."""
        if not self.enabled or not self.service:
            return False
        
        try:
            self.service.files().delete(fileId=file_id).execute()
            return True
        except Exception as e:
            logging.error(f"Google Drive delete failed: {str(e)}")
            return False


class AirtableBackend:
    """Airtable storage backend for metadata and content management."""
    
    def __init__(self, api_key: Optional[str] = None, base_id: Optional[str] = None, table_name: str = "Videos"):
        # Try to get credentials from vault first, fallback to provided params
        airtable_creds = AirtableCredentials()
        
        # Use provided params if available, otherwise use vault credentials
        self.api_key = api_key or airtable_creds.api_key
        self.base_id = base_id or airtable_creds.base_id
        self.table_name = table_name or airtable_creds.table_name
        self.enabled = bool(self.api_key and self.base_id)
        self.table = None
        
        if self.enabled:
            try:
                from pyairtable import Table
                self.table = Table(self.api_key, self.base_id, self.table_name)
            except ImportError:
                self.enabled = False
                logging.warning("Airtable dependencies not installed. Run: pip install pyairtable")
                logging.info("To enable Airtable storage, set AIRTABLE_API_KEY and AIRTABLE_BASE_ID in .env")
            except Exception as e:
                self.enabled = False
                logging.error(f"Failed to initialize Airtable: {str(e)}")
    
    async def create_or_update_record(self, video_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a record in Airtable.
        
        Args:
            video_id: YouTube video ID
            data: Record data
            
        Returns:
            Dict with record ID and status
        """
        if not self.enabled or not self.table:
            return {"status": "disabled", "message": "Airtable not configured"}
        
        try:
            # Check if record exists
            existing = self.table.all(formula=f"{{video_id}}='{video_id}'")
            
            # Prepare record data
            record_data = {
                'video_id': video_id,
                'channel_id': data.get('channel_id', ''),
                'title': data.get('title', ''),
                'duration': data.get('duration', 0),
                'upload_date': data.get('upload_date', ''),
                'transcript_path': data.get('transcript_path', ''),
                'audio_path': data.get('audio_path', ''),
                'gdrive_audio_id': data.get('gdrive_audio_id', ''),
                'gdrive_transcript_id': data.get('gdrive_transcript_id', ''),
                'transcript_word_count': data.get('word_count', 0),
                'quality_score': data.get('quality_score', 0),
                'processed_date': datetime.utcnow().isoformat(),
                'status': data.get('status', 'processed')
            }
            
            if existing:
                # Update existing record
                record = self.table.update(existing[0]['id'], record_data)
                return {
                    "status": "updated",
                    "record_id": record['id'],
                    "message": "Record updated in Airtable"
                }
            else:
                # Create new record
                record = self.table.create(record_data)
                return {
                    "status": "created",
                    "record_id": record['id'],
                    "message": "Record created in Airtable"
                }
                
        except Exception as e:
            logging.error(f"Airtable operation failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Airtable operation failed: {str(e)}"
            }
    
    async def get_record(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get a record from Airtable by video ID."""
        if not self.enabled or not self.table:
            return None
        
        try:
            records = self.table.all(formula=f"{{video_id}}='{video_id}'")
            return records[0] if records else None
        except Exception as e:
            logging.error(f"Airtable get failed: {str(e)}")
            return None
    
    async def delete_record(self, video_id: str) -> bool:
        """Delete a record from Airtable."""
        if not self.enabled or not self.table:
            return False
        
        try:
            records = self.table.all(formula=f"{{video_id}}='{video_id}'")
            if records:
                self.table.delete(records[0]['id'])
                return True
            return False
        except Exception as e:
            logging.error(f"Airtable delete failed: {str(e)}")
            return False


class SupabaseBackend:
    """Supabase storage backend (Phase 2 - future implementation)."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url
        self.key = key
        self.enabled = False  # Not implemented yet
        
    async def upload_file(self, local_path: Path, remote_path: str) -> Dict[str, Any]:
        """Upload file to Supabase storage."""
        return {"status": "not_implemented", "message": "Supabase backend planned for Phase 2"}


class S3Backend:
    """S3/GCS storage backend (Phase 3 - future implementation)."""
    
    def __init__(self, bucket: Optional[str] = None, region: Optional[str] = None):
        self.bucket = bucket
        self.region = region
        self.enabled = False  # Not implemented yet
        
    async def upload_file(self, local_path: Path, remote_path: str) -> Dict[str, Any]:
        """Upload file to S3/GCS."""
        return {"status": "not_implemented", "message": "S3/GCS backend planned for Phase 3"}


class StorageWorker(BaseWorker):
    """
    Storage worker for multi-backend file synchronization.
    
    Handles the storage strategy from PRD section 4.4:
    - Primary save to local (blocking)
    - Secondary saves to cloud backends (async)
    - Database sync status tracking
    
    Supports phase-based storage expansion:
    - Phase 1: local + gdrive
    - Phase 2: + supabase  
    - Phase 3: + s3/gcs
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="storage", **kwargs)
        self.settings = get_settings()
        
        # Initialize storage backends
        self._init_backends()
        
    def _init_backends(self) -> None:
        """Initialize storage backends based on configuration."""
        self.backends = {}
        
        # Local storage (always enabled)
        self.backends["local"] = {"enabled": True, "instance": None}
        
        # Google Drive backend (Phase 1)
        self.backends["gdrive"] = {
            "enabled": "gdrive" in self.settings.storage_backends,
            "instance": GoogleDriveBackend(
                credentials_file=self.settings.gdrive_credentials_file,
                folder_id=self.settings.gdrive_folder_id
            )
        }
        
        # Airtable backend (Phase 1)
        self.backends["airtable"] = {
            "enabled": "airtable" in self.settings.storage_backends,
            "instance": AirtableBackend(
                api_key=self.settings.airtable_api_key,
                base_id=self.settings.airtable_base_id,
                table_name=self.settings.airtable_table_name
            )
        }
        
        # Supabase backend (Phase 2 - future)
        self.backends["supabase"] = {
            "enabled": False,  # Not implemented yet
            "instance": SupabaseBackend()
        }
        
        # S3/GCS backend (Phase 3 - future)
        self.backends["s3"] = {
            "enabled": False,  # Not implemented yet  
            "instance": S3Backend()
        }
        
        enabled_backends = [name for name, config in self.backends.items() if config["enabled"]]
        self.log_with_context(
            f"Initialized storage backends: {enabled_backends}",
            extra_context={
                "total_backends": len(self.backends),
                "enabled_count": len(enabled_backends)
            }
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate storage worker input.
        
        Required fields:
        - file_path: Path to file to store
        - file_type: Type of file (audio, transcript, content, metadata)
        - channel_id: YouTube channel ID
        - video_id: YouTube video ID
        
        Optional fields:
        - storage_backends: Override default backends
        - metadata: Additional file metadata
        """
        required_fields = ["file_path", "file_type", "channel_id", "video_id"]
        
        for field in required_fields:
            if field not in input_data:
                self.log_with_context(f"Missing required field: {field}", level="ERROR")
                return False
        
        # Validate file exists
        file_path = Path(input_data["file_path"])
        if not file_path.exists():
            self.log_with_context(f"File does not exist: {file_path}", level="ERROR")
            return False
        
        # Validate file type
        valid_file_types = ["audio", "transcript", "content", "metadata"]
        if input_data["file_type"] not in valid_file_types:
            self.log_with_context(
                f"Invalid file_type: {input_data['file_type']}. Must be one of: {valid_file_types}",
                level="ERROR"
            )
            return False
        
        return True
    
    def _get_storage_structure(self, channel_id: str, video_id: str, file_type: str) -> Path:
        """
        Generate storage directory structure based on PRD.
        
        Structure: /{file_type}/{channel_id}/{video_id}/
        Example: /audio/UCxxx/video123/
        """
        base_path = Path(self.settings.storage_path)
        
        # Map file_type to directory name (some need pluralization)
        dir_map = {
            'audio': 'audio',
            'transcript': 'transcripts',
            'content': 'content',
            'metadata': 'metadata'
        }
        
        dir_name = dir_map.get(file_type, file_type)
        return base_path / dir_name / channel_id / video_id
    
    def _save_to_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save file to local storage (primary, blocking) with enhanced error handling.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Dict with local storage result
            
        Raises:
            LocalStorageError: When local storage operations fail
        """
        source_path = Path(input_data["file_path"])
        channel_id = input_data["channel_id"]
        video_id = input_data["video_id"]
        file_type = input_data["file_type"]
        
        # Validate source file exists and is readable
        if not source_path.exists():
            raise LocalStorageError(f"Source file does not exist: {source_path}")
        
        if not source_path.is_file():
            raise LocalStorageError(f"Source path is not a file: {source_path}")
        
        try:
            # Check if source file is readable
            source_size = source_path.stat().st_size
        except (OSError, PermissionError) as e:
            raise LocalStorageError(f"Cannot read source file {source_path}: {str(e)}")
        
        if source_size == 0:
            raise LocalStorageError(f"Source file is empty: {source_path}")
        
        # Generate destination path
        try:
            dest_dir = self._get_storage_structure(channel_id, video_id, file_type)
        except Exception as e:
            raise LocalStorageError(f"Failed to generate storage path: {str(e)}")
        
        # Check storage space before creating directory
        try:
            storage_root = Path(self.settings.storage_path)
            if storage_root.exists():
                stat = shutil.disk_usage(storage_root)
                free_space = stat.free
                
                # Require at least 2x file size in free space for safety
                if free_space < (source_size * 2):
                    raise StorageSpaceError(
                        f"Insufficient disk space: {free_space} bytes available, need {source_size * 2} bytes"
                    )
        except (OSError, AttributeError) as e:
            self.log_with_context(
                f"Could not check disk space: {str(e)}",
                level="WARNING"
            )
        
        # Create destination directory with proper error handling
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise StoragePermissionError(f"Permission denied creating directory {dest_dir}: {str(e)}")
        except OSError as e:
            raise LocalStorageError(f"Failed to create directory {dest_dir}: {str(e)}")
        
        dest_path = dest_dir / source_path.name
        
        # Check if destination already exists and is identical
        if dest_path.exists():
            try:
                if dest_path.stat().st_size == source_size:
                    # File exists with same size, check if it's identical
                    import filecmp
                    if filecmp.cmp(source_path, dest_path, shallow=False):
                        self.log_with_context(
                            f"File already exists and is identical: {dest_path}",
                            extra_context={"source": str(source_path), "dest": str(dest_path)}
                        )
                        return {
                            "status": "success",
                            "path": str(dest_path),
                            "size": dest_path.stat().st_size,
                            "directory": str(dest_dir),
                            "action": "unchanged"
                        }
            except (OSError, PermissionError) as e:
                self.log_with_context(
                    f"Could not compare existing file: {str(e)}",
                    level="WARNING"
                )
        
        # Copy file to destination with detailed error handling
        try:
            if source_path == dest_path:
                self.log_with_context(f"File already in correct location: {dest_path}")
                action = "in_place"
            else:
                shutil.copy2(source_path, dest_path)
                self.log_with_context(
                    f"File copied successfully",
                    extra_context={
                        "source": str(source_path),
                        "dest": str(dest_path),
                        "size": source_size
                    }
                )
                action = "copied"
            
            # Verify the copied file
            try:
                final_size = dest_path.stat().st_size
                if final_size != source_size:
                    raise LocalStorageError(
                        f"File copy verification failed: expected {source_size} bytes, got {final_size} bytes"
                    )
            except OSError as e:
                raise LocalStorageError(f"Cannot verify copied file: {str(e)}")
            
            return {
                "status": "success",
                "path": str(dest_path),
                "size": final_size,
                "directory": str(dest_dir),
                "action": action
            }
            
        except PermissionError as e:
            raise StoragePermissionError(f"Permission denied copying file to {dest_path}: {str(e)}")
        except OSError as e:
            if "No space left on device" in str(e):
                raise StorageSpaceError(f"No space left on device: {str(e)}")
            else:
                raise LocalStorageError(f"File copy failed: {str(e)}")
        except Exception as e:
            raise LocalStorageError(f"Unexpected error during file copy: {str(e)}")
    
    def _sync_to_backends(self, input_data: Dict[str, Any], local_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync file to configured cloud backends with enhanced error handling.
        
        Args:
            input_data: Original input data
            local_result: Result from local storage
            
        Returns:
            Dict with sync results for each backend
        """
        local_path = Path(local_result["path"])
        channel_id = input_data["channel_id"]
        video_id = input_data["video_id"]
        file_type = input_data["file_type"]
        
        # Generate remote path
        remote_path = f"{channel_id}/{video_id}/{file_type}s/{local_path.name}"
        
        sync_results = {}
        
        # Sync to Google Drive with enhanced error handling
        if self.backends["gdrive"]["enabled"]:
            try:
                gdrive = self.backends["gdrive"]["instance"]
                
                # Check if Google Drive instance is properly initialized
                if not gdrive or not gdrive.enabled:
                    sync_results["gdrive"] = {
                        "status": "disabled",
                        "error": "Google Drive backend not properly initialized"
                    }
                else:
                    # Run async upload in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(gdrive.upload_file(local_path, remote_path))
                        sync_results["gdrive"] = result
                        
                        if result.get("status") == "success":
                            self.log_with_context(
                                "Google Drive sync successful",
                                extra_context={
                                    "file_id": result.get("file_id"),
                                    "file_size": result.get("size"),
                                    "remote_path": remote_path
                                }
                            )
                        else:
                            self.log_with_context(
                                f"Google Drive sync failed: {result.get('message', 'Unknown error')}",
                                level="WARNING",
                                extra_context={"remote_path": remote_path}
                            )
                    finally:
                        loop.close()
                
            except (GoogleDriveError, StorageCredentialsError, NetworkStorageError) as e:
                error_type = type(e).__name__
                sync_results["gdrive"] = {
                    "status": "failed",
                    "error": str(e),
                    "error_type": error_type
                }
                self.log_with_context(
                    f"Google Drive sync failed with {error_type}: {str(e)}",
                    level="ERROR",
                    extra_context={"remote_path": remote_path}
                )
                
            except Exception as e:
                sync_results["gdrive"] = {
                    "status": "failed",
                    "error": f"Unexpected error: {str(e)}",
                    "error_type": type(e).__name__
                }
                self.log_with_context(
                    f"Google Drive sync failed with unexpected error: {str(e)}",
                    level="ERROR",
                    extra_context={"remote_path": remote_path, "error_type": type(e).__name__}
                )
        
        # Sync to Airtable with enhanced error handling
        if self.backends["airtable"]["enabled"]:
            try:
                airtable = self.backends["airtable"]["instance"]
                
                if not airtable or not airtable.enabled:
                    sync_results["airtable"] = {
                        "status": "disabled",
                        "error": "Airtable backend not properly initialized"
                    }
                else:
                    # Prepare metadata for Airtable
                    metadata = {
                        "channel_id": channel_id,
                        "video_id": video_id,
                        "file_type": file_type,
                        "local_path": str(local_path),
                        "file_size": local_result.get("size", 0),
                        "gdrive_file_id": sync_results.get("gdrive", {}).get("file_id"),
                        "gdrive_url": sync_results.get("gdrive", {}).get("url")
                    }
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            airtable.create_or_update_record(video_id, metadata)
                        )
                        sync_results["airtable"] = result
                        
                        if result.get("status") in ["created", "updated"]:
                            self.log_with_context(
                                f"Airtable sync successful: {result.get('status')}",
                                extra_context={"record_id": result.get("record_id")}
                            )
                        else:
                            self.log_with_context(
                                f"Airtable sync failed: {result.get('message', 'Unknown error')}",
                                level="WARNING"
                            )
                    finally:
                        loop.close()
                        
            except AirtableError as e:
                sync_results["airtable"] = {
                    "status": "failed",
                    "error": str(e),
                    "error_type": "AirtableError"
                }
                self.log_with_context(
                    f"Airtable sync failed: {str(e)}",
                    level="ERROR"
                )
                
            except Exception as e:
                sync_results["airtable"] = {
                    "status": "failed",
                    "error": f"Unexpected error: {str(e)}",
                    "error_type": type(e).__name__
                }
                self.log_with_context(
                    f"Airtable sync failed with unexpected error: {str(e)}",
                    level="ERROR",
                    extra_context={"error_type": type(e).__name__}
                )
        
        # Future: Supabase sync (Phase 2)
        if self.backends["supabase"]["enabled"]:
            try:
                supabase = self.backends["supabase"]["instance"]
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(supabase.upload_file(local_path, remote_path))
                finally:
                    loop.close()
                sync_results["supabase"] = result
            except Exception as e:
                sync_results["supabase"] = {"status": "failed", "error": str(e)}
        
        # Future: S3/GCS sync (Phase 3)
        if self.backends["s3"]["enabled"]:
            try:
                s3 = self.backends["s3"]["instance"]
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(s3.upload_file(local_path, remote_path))
                finally:
                    loop.close()
                sync_results["s3"] = result
            except Exception as e:
                sync_results["s3"] = {"status": "failed", "error": str(e)}
        
        return sync_results
    
    def _update_storage_sync_table(
        self, 
        input_data: Dict[str, Any], 
        local_result: Dict[str, Any], 
        sync_results: Dict[str, Any]
    ) -> None:
        """
        Update storage_sync table with sync status.
        
        Args:
            input_data: Original input data
            local_result: Local storage result
            sync_results: Cloud backend sync results
        """
        try:
            # Handle database operations in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def update_db():
                    async with db_manager.get_session() as session:
                        # Create storage sync record
                        storage_sync = StorageSync(
                            file_type=input_data["file_type"],
                            local_path=local_result["path"],
                            gdrive_file_id=sync_results.get("gdrive", {}).get("file_id"),
                            gdrive_url=sync_results.get("gdrive", {}).get("url"),
                            sync_status="synced" if any(
                                result.get("status") == "success" or result.get("status") == "stubbed"
                                for result in sync_results.values()
                            ) else "failed",
                            last_synced=datetime.utcnow()
                        )
                        
                        session.add(storage_sync)
                        await session.commit()
                        
                        self.log_with_context(
                            f"Storage sync record created: {storage_sync.sync_status}",
                            extra_context={"sync_id": storage_sync.id}
                        )
                        
                loop.run_until_complete(update_db())
            finally:
                loop.close()
                
        except Exception as e:
            self.log_with_context(f"Failed to update storage_sync table: {e}", level="ERROR")
            # Don't raise - sync table is supplementary, main storage should continue
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute storage synchronization.
        
        Process:
        1. Primary save to local storage (blocking)
        2. Secondary sync to cloud backends (async)
        3. Update storage_sync table with status
        
        Args:
            input_data: Validated input containing file info and storage requirements
            
        Returns:
            Dict with storage paths and sync status
        """
        self.log_with_context(
            "Starting storage synchronization",
            extra_context={
                "file": input_data["file_path"],
                "type": input_data["file_type"],
                "video": f"{input_data['channel_id']}/{input_data['video_id']}"
            }
        )
        
        # Step 1: Primary save to local storage (blocking)
        local_result = self._save_to_local(input_data)
        
        # Step 2: Secondary sync to cloud backends
        sync_results = self._sync_to_backends(input_data, local_result)
        
        # Step 3: Update storage sync table
        self._update_storage_sync_table(input_data, local_result, sync_results)
        
        # Calculate overall sync status
        successful_syncs = [
            name for name, result in sync_results.items()
            if result.get("status") in ["success", "stubbed"]
        ]
        
        failed_syncs = [
            name for name, result in sync_results.items()
            if result.get("status") == "failed"
        ]
        
        result = {
            "local_storage": local_result,
            "cloud_sync": sync_results,
            "sync_summary": {
                "total_backends": len(sync_results),
                "successful": len(successful_syncs),
                "failed": len(failed_syncs),
                "successful_backends": successful_syncs,
                "failed_backends": failed_syncs
            },
            "storage_paths": {
                "local": local_result["path"],
                "gdrive_url": sync_results.get("gdrive", {}).get("url"),
                "gdrive_file_id": sync_results.get("gdrive", {}).get("file_id")
            }
        }
        
        self.log_with_context(
            f"Storage sync completed: {len(successful_syncs)}/{len(sync_results)} backends successful",
            extra_context={
                "local_path": local_result["path"],
                "successful_backends": successful_syncs,
                "failed_backends": failed_syncs
            }
        )
        
        return result
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle storage errors with enhanced categorization and recovery strategies.
        
        Args:
            error: Exception that occurred during storage operations
            
        Returns:
            Dict with error categorization and recovery information
        """
        error_type = type(error).__name__
        error_details = {
            "error_type": error_type,
            "error_message": str(error),
            "recovery_suggestions": [],
            "retry_recommended": False,
            "retry_delay": None
        }
        
        # Categorize specific storage errors
        if isinstance(error, LocalStorageError):
            error_details["category"] = "local_storage"
            error_details["severity"] = "high"  # Local storage is critical
            error_details["recovery_suggestions"].extend([
                "Check local storage path permissions",
                "Verify sufficient disk space",
                "Ensure storage directory is accessible",
                "Verify source file integrity"
            ])
            
        elif isinstance(error, GoogleDriveError):
            error_details["category"] = "google_drive"
            error_details["severity"] = "medium"  # Cloud storage is secondary
            error_details["retry_recommended"] = True
            error_details["retry_delay"] = 60  # 1 minute
            error_details["recovery_suggestions"].extend([
                "Check Google Drive credentials",
                "Verify Google Drive API access",
                "Check storage quota",
                "Verify internet connectivity"
            ])
            
        elif isinstance(error, AirtableError):
            error_details["category"] = "airtable"
            error_details["severity"] = "low"  # Metadata storage is optional
            error_details["retry_recommended"] = True
            error_details["retry_delay"] = 30  # 30 seconds
            error_details["recovery_suggestions"].extend([
                "Check Airtable API credentials",
                "Verify Airtable base and table configuration",
                "Check Airtable API rate limits",
                "Verify internet connectivity"
            ])
            
        elif isinstance(error, StorageCredentialsError):
            error_details["category"] = "credentials"
            error_details["severity"] = "high"
            error_details["retry_recommended"] = False
            error_details["recovery_suggestions"].extend([
                "Check and update storage service credentials",
                "Verify API keys and authentication tokens",
                "Check credential file permissions",
                "Renew expired credentials"
            ])
            
        elif isinstance(error, StorageSpaceError):
            error_details["category"] = "space"
            error_details["severity"] = "high"
            error_details["retry_recommended"] = False
            error_details["recovery_suggestions"].extend([
                "Free up disk space",
                "Check storage quota limits",
                "Configure automatic cleanup",
                "Consider using compression"
            ])
            
        elif isinstance(error, StoragePermissionError):
            error_details["category"] = "permissions"
            error_details["severity"] = "high"
            error_details["retry_recommended"] = False
            error_details["recovery_suggestions"].extend([
                "Check file and directory permissions",
                "Verify user has write access to storage paths",
                "Run with appropriate user privileges",
                "Check SELinux/AppArmor policies"
            ])
            
        elif isinstance(error, NetworkStorageError):
            error_details["category"] = "network"
            error_details["severity"] = "medium"
            error_details["retry_recommended"] = True
            error_details["retry_delay"] = 120  # 2 minutes
            error_details["recovery_suggestions"].extend([
                "Check internet connectivity",
                "Verify DNS resolution",
                "Check firewall settings",
                "Try alternative network connection"
            ])
            
        elif isinstance(error, (OSError, IOError)):
            error_details["category"] = "filesystem"
            error_details["severity"] = "high"
            error_details["retry_recommended"] = False
            
            # Specific OSError handling
            if "No space left on device" in str(error):
                error_details["category"] = "space"
                error_details["recovery_suggestions"].extend([
                    "Free up disk space immediately",
                    "Check for large temporary files",
                    "Configure automatic cleanup"
                ])
            elif "Permission denied" in str(error):
                error_details["category"] = "permissions"
                error_details["recovery_suggestions"].extend([
                    "Check file system permissions",
                    "Verify user has appropriate access",
                    "Check directory ownership"
                ])
            else:
                error_details["recovery_suggestions"].extend([
                    "Check file system integrity",
                    "Verify hardware functionality",
                    "Check system logs for hardware errors"
                ])
                
        elif isinstance(error, PermissionError):
            error_details["category"] = "permissions"
            error_details["severity"] = "high"
            error_details["retry_recommended"] = False
            error_details["recovery_suggestions"].extend([
                "Check file/directory permissions",
                "Verify user has write access to storage path",
                "Run with appropriate user privileges",
                "Check group membership and umask settings"
            ])
            
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            error_details["category"] = "network"
            error_details["severity"] = "medium"
            error_details["retry_recommended"] = True
            error_details["retry_delay"] = 60
            error_details["recovery_suggestions"].extend([
                "Check internet connectivity",
                "Verify cloud storage service status",
                "Check API rate limits",
                "Retry with exponential backoff"
            ])
            
        else:
            error_details["category"] = "unknown"
            error_details["severity"] = "medium"
            error_details["retry_recommended"] = True
            error_details["retry_delay"] = 30
            error_details["recovery_suggestions"].extend([
                "Review error details and logs",
                "Check system resources",
                "Verify configuration settings",
                "Contact support if issue persists"
            ])
        
        # Add general recovery suggestions based on category
        if error_details["category"] in ["google_drive", "airtable", "network"]:
            error_details["recovery_suggestions"].append(
                "Consider disabling problematic backend temporarily"
            )
        
        self.log_with_context(
            f"Storage error categorized and handled: {error_type}",
            level="ERROR",
            extra_context={
                "category": error_details["category"],
                "severity": error_details.get("severity", "unknown"),
                "retry_recommended": error_details["retry_recommended"],
                "suggestions_count": len(error_details["recovery_suggestions"])
            }
        )
        
        return error_details
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all storage backends.
        
        Returns:
            Dict with backend status information
        """
        storage_path = Path(self.settings.storage_path)
        gdrive = self.backends.get("gdrive", {})
        airtable = self.backends.get("airtable", {})
        
        return {
            "backends": {
                "local": {
                    "enabled": True,
                    "path": str(storage_path),
                    "exists": storage_path.exists()
                },
                "google_drive": {
                    "enabled": gdrive.get("enabled", False),
                    "folder_id": gdrive["instance"].folder_id if gdrive.get("instance") and gdrive.get("enabled") else None
                },
                "airtable": {
                    "enabled": airtable.get("enabled", False),
                    "base_id": airtable["instance"].base_id if airtable.get("instance") and airtable.get("enabled") else None
                },
                "supabase": {
                    "enabled": False,
                    "note": "Phase 2 feature"
                },
                "s3": {
                    "enabled": False,
                    "note": "Phase 3 feature"
                }
            },
            "total_enabled": sum([
                1,  # Local always enabled
                gdrive.get("enabled", False),
                airtable.get("enabled", False)
            ])
        }


# Convenience functions for common storage operations

async def store_file(
    file_path: Union[str, Path],
    file_type: str,
    channel_id: str,
    video_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    storage_backends: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to store a file using the StorageWorker.
    
    Args:
        file_path: Path to file to store
        file_type: Type of file (audio, transcript, content, metadata)
        channel_id: YouTube channel ID
        video_id: YouTube video ID
        metadata: Optional additional metadata
        storage_backends: Override default storage backends
        
    Returns:
        Storage result dict
    """
    storage_worker = StorageWorker()
    
    input_data = {
        "file_path": str(file_path),
        "file_type": file_type,
        "channel_id": channel_id,
        "video_id": video_id
    }
    
    if metadata:
        input_data["metadata"] = metadata
    if storage_backends:
        input_data["storage_backends"] = storage_backends
    
    return storage_worker.run(input_data)


async def get_storage_path(channel_id: str, video_id: str, file_type: str) -> Path:
    """
    Get the local storage path for a file.
    
    Args:
        channel_id: YouTube channel ID
        video_id: YouTube video ID  
        file_type: Type of file (audio, transcript, content, metadata)
        
    Returns:
        Path to storage directory
    """
    storage_paths = get_storage_paths_v2()
    
    # Map file types to V2 storage methods
    if file_type == 'audio':
        return storage_paths.get_audio_dir(channel_id, video_id)
    elif file_type == 'transcript':
        return storage_paths.get_transcript_dir(channel_id, video_id)
    elif file_type == 'content':
        return storage_paths.get_content_dir(channel_id, video_id)
    elif file_type == 'metadata':
        return storage_paths.get_metadata_dir(channel_id, video_id)
    else:
        # Fallback to generic media dir
        return storage_paths.get_media_dir(channel_id, video_id)