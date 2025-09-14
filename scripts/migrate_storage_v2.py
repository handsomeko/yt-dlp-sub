#!/usr/bin/env python3
"""
Storage Migration Script V1 -> V2
Migrates existing storage from type-based structure to ID-based structure with readable filenames
"""

import json
import logging
import shutil
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import V1 from archived location
sys.path.insert(0, '/Users/jk/yt-dl-sub/archived/v1_storage_structure')
from storage_paths_v1 import StoragePaths as StoragePathsV1
from core.storage_paths_v2 import StoragePathsV2, StorageVersion, get_storage_paths_v2
from core.filename_sanitizer import sanitize_filename
from core.database import Video, Channel
from config.settings import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageMigrator:
    """Handles migration from V1 to V2 storage structure."""
    
    def __init__(self, dry_run: bool = True):
        """
        Initialize migrator.
        
        Args:
            dry_run: If True, only simulate migration without making changes
        """
        self.dry_run = dry_run
        self.settings = get_settings()
        
        # Initialize storage systems
        self.v1_storage = StoragePathsV1()
        self.v2_storage = StoragePathsV2(version=StorageVersion.V2)
        
        # Migration state
        self.migration_log = []
        self.errors = []
        self.stats = {
            'channels_processed': 0,
            'videos_processed': 0,
            'files_moved': 0,
            'files_failed': 0,
            'total_size_moved': 0
        }
        
        # Create backup directory
        self.backup_dir = self.v1_storage.base_path / f".backup_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Migration initialized - Dry run: {dry_run}")
        logger.info(f"Source: {self.v1_storage.base_path}")
        logger.info(f"Target: {self.v2_storage.base_path}")
    
    def run_migration(self, backup: bool = True) -> bool:
        """
        Run the complete migration process.
        
        Args:
            backup: Create backup before migration
            
        Returns:
            True if successful
        """
        logger.info("🚀 Starting storage migration V1 -> V2")
        
        try:
            # Phase 1: Pre-migration checks
            if not self._pre_migration_checks():
                return False
            
            # Phase 2: Create backup if requested
            if backup and not self.dry_run:
                if not self._create_backup():
                    return False
            
            # Phase 3: Discover existing structure
            v1_structure = self._discover_v1_structure()
            logger.info(f"Discovered {len(v1_structure)} channels with data")
            
            # Phase 4: Migrate each channel
            for channel_id, channel_info in v1_structure.items():
                self._migrate_channel(channel_id, channel_info)
                self.stats['channels_processed'] += 1
            
            # Phase 5: Create indexes and metadata
            if not self.dry_run:
                self._create_indexes()
                self._create_completion_markers()
            
            # Phase 6: Update database schema (if database exists)
            if not self.dry_run:
                self._update_database_schema()
            
            # Phase 7: Validate migration
            if not self._validate_migration():
                logger.error("❌ Migration validation failed")
                return False
            
            logger.info("✅ Migration completed successfully!")
            self._print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            logger.exception("Migration error details:")
            return False
    
    def _pre_migration_checks(self) -> bool:
        """Run pre-migration validation checks."""
        logger.info("🔍 Running pre-migration checks...")
        
        checks_passed = True
        
        # Check source directory exists
        if not self.v1_storage.base_path.exists():
            logger.error(f"Source directory does not exist: {self.v1_storage.base_path}")
            checks_passed = False
        
        # Check for V1 structure
        v1_indicators = ['audio', 'transcripts', 'content', 'metadata']
        v1_dirs_found = sum(1 for indicator in v1_indicators 
                           if (self.v1_storage.base_path / indicator).exists())
        
        if v1_dirs_found == 0:
            logger.error("No V1 structure found (no audio/transcripts/content/metadata directories)")
            checks_passed = False
        
        # Check for existing V2 structure
        v2_channels = list(self.v2_storage.base_path.glob("UC*"))  # Channel IDs start with UC
        if v2_channels and not self.dry_run:
            logger.warning(f"Found {len(v2_channels)} existing V2 channels - migration will merge/overwrite")
        
        # Check available disk space
        if not self.dry_run:
            free_space = shutil.disk_usage(self.v1_storage.base_path).free
            used_space = sum(f.stat().st_size for f in self.v1_storage.base_path.rglob('*') if f.is_file())
            
            if free_space < used_space * 1.5:  # Need 50% extra space for safety
                logger.error(f"Insufficient disk space. Need ~{used_space * 1.5 / (1024**3):.1f} GB, have {free_space / (1024**3):.1f} GB")
                checks_passed = False
        
        if checks_passed:
            logger.info("✅ Pre-migration checks passed")
        else:
            logger.error("❌ Pre-migration checks failed")
        
        return checks_passed
    
    def _create_backup(self) -> bool:
        """Create backup of original structure."""
        logger.info(f"💾 Creating backup at {self.backup_dir}")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy V1 directories
            for v1_dir in ['audio', 'transcripts', 'content', 'metadata']:
                source = self.v1_storage.base_path / v1_dir
                if source.exists():
                    target = self.backup_dir / v1_dir
                    shutil.copytree(source, target)
                    logger.info(f"Backed up {v1_dir}")
            
            # Create backup manifest
            manifest = {
                'created_at': datetime.now().isoformat(),
                'source_path': str(self.v1_storage.base_path),
                'backup_reason': 'V1 to V2 migration',
                'original_structure': list(self.v1_storage.base_path.iterdir())
            }
            
            with open(self.backup_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            logger.info("✅ Backup created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def _discover_v1_structure(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover existing V1 structure.
        
        Returns:
            Dict mapping channel_id to channel info with videos
        """
        logger.info("🔍 Discovering V1 structure...")
        
        structure = {}
        
        # Scan each V1 type directory
        for type_name in ['audio', 'transcripts', 'content', 'metadata']:
            type_dir = self.v1_storage.base_path / type_name
            if not type_dir.exists():
                continue
            
            # Find channels
            for channel_path in type_dir.iterdir():
                if not channel_path.is_dir():
                    continue
                
                channel_id = channel_path.name
                
                # Initialize channel entry
                if channel_id not in structure:
                    structure[channel_id] = {
                        'channel_id': channel_id,
                        'videos': {},
                        'discovered_from': []
                    }
                
                structure[channel_id]['discovered_from'].append(type_name)
                
                # Find videos in this channel
                for video_path in channel_path.iterdir():
                    if not video_path.is_dir():
                        continue
                    
                    video_id = video_path.name
                    
                    # Initialize video entry
                    if video_id not in structure[channel_id]['videos']:
                        structure[channel_id]['videos'][video_id] = {
                            'video_id': video_id,
                            'files': {}
                        }
                    
                    # Find files in this video directory
                    files = []
                    for file_path in video_path.rglob('*'):
                        if file_path.is_file():
                            files.append({
                                'path': file_path,
                                'size': file_path.stat().st_size,
                                'modified': file_path.stat().st_mtime
                            })
                    
                    structure[channel_id]['videos'][video_id]['files'][type_name] = files
        
        return structure
    
    def _migrate_channel(self, channel_id: str, channel_info: Dict[str, Any]):
        """Migrate a single channel from V1 to V2."""
        logger.info(f"📁 Migrating channel: {channel_id}")
        
        videos = channel_info['videos']
        
        # Create channel directory in V2
        if not self.dry_run:
            channel_dir = self.v2_storage.get_channel_dir(channel_id)
            channel_dir.mkdir(parents=True, exist_ok=True)
        
        # Migrate each video
        for video_id, video_info in videos.items():
            self._migrate_video(channel_id, video_id, video_info)
            self.stats['videos_processed'] += 1
    
    def _migrate_video(self, channel_id: str, video_id: str, video_info: Dict[str, Any]):
        """Migrate a single video from V1 to V2."""
        logger.info(f"  📹 Migrating video: {video_id}")
        
        # Try to extract video title from existing files
        video_title = self._extract_video_title(channel_id, video_id, video_info)
        
        # Create V2 video directory
        if not self.dry_run:
            video_dir = self.v2_storage.get_video_dir(channel_id, video_id)
            video_dir.mkdir(parents=True, exist_ok=True)
        
        # Migrate files by type
        file_types = video_info.get('files', {})
        
        for v1_type, files in file_types.items():
            if v1_type == 'audio':
                self._migrate_media_files(channel_id, video_id, video_title, files)
            elif v1_type == 'transcripts':
                self._migrate_transcript_files(channel_id, video_id, video_title, files)
            elif v1_type == 'content':
                self._migrate_content_files(channel_id, video_id, video_title, files)
            elif v1_type == 'metadata':
                self._migrate_metadata_files(channel_id, video_id, files)
    
    def _extract_video_title(self, channel_id: str, video_id: str, video_info: Dict[str, Any]) -> str:
        """Extract video title from existing files or metadata."""
        # Try to find title in metadata files
        metadata_files = video_info.get('files', {}).get('metadata', [])
        for file_info in metadata_files:
            if file_info['path'].name == 'metadata.json':
                try:
                    with open(file_info['path'], 'r') as f:
                        metadata = json.load(f)
                        if 'title' in metadata:
                            return metadata['title']
                except:
                    pass
        
        # Try to extract from audio filenames (may have readable names)
        audio_files = video_info.get('files', {}).get('audio', [])
        for file_info in audio_files:
            filename = file_info['path'].stem
            if filename != video_id:  # If not just the video ID
                return filename
        
        # Try transcript files
        transcript_files = video_info.get('files', {}).get('transcripts', [])
        for file_info in transcript_files:
            filename = file_info['path'].stem
            if filename != video_id:
                return filename
        
        # Default to video ID
        return f"Video {video_id}"
    
    def _migrate_media_files(self, channel_id: str, video_id: str, video_title: str, files: List[Dict]):
        """Migrate audio/video files to media directory."""
        if not files:
            return
        
        target_dir = self.v2_storage.get_media_dir(channel_id, video_id) if not self.dry_run else Path("dry_run")
        
        for file_info in files:
            source_path = file_info['path']
            safe_title = sanitize_filename(video_title, video_id)
            target_path = target_dir / f"{safe_title}{source_path.suffix}"
            
            self._move_file(source_path, target_path, file_info['size'])
    
    def _migrate_transcript_files(self, channel_id: str, video_id: str, video_title: str, files: List[Dict]):
        """Migrate transcript files to transcripts directory."""
        if not files:
            return
        
        target_dir = self.v2_storage.get_transcript_dir(channel_id, video_id) if not self.dry_run else Path("dry_run")
        
        for file_info in files:
            source_path = file_info['path']
            safe_title = sanitize_filename(video_title, video_id)
            target_path = target_dir / f"{safe_title}{source_path.suffix}"
            
            self._move_file(source_path, target_path, file_info['size'])
    
    def _migrate_content_files(self, channel_id: str, video_id: str, video_title: str, files: List[Dict]):
        """Migrate generated content files to content directory."""
        if not files:
            return
        
        target_dir = self.v2_storage.get_content_dir(channel_id, video_id) if not self.dry_run else Path("dry_run")
        
        for file_info in files:
            source_path = file_info['path']
            # Keep original content filenames but ensure they're safe
            filename = sanitize_filename(source_path.name, None)
            target_path = target_dir / filename
            
            self._move_file(source_path, target_path, file_info['size'])
    
    def _migrate_metadata_files(self, channel_id: str, video_id: str, files: List[Dict]):
        """Migrate metadata files."""
        if not files:
            return
        
        target_dir = self.v2_storage.get_metadata_dir(channel_id, video_id) if not self.dry_run else Path("dry_run")
        
        for file_info in files:
            source_path = file_info['path']
            target_path = target_dir / source_path.name
            
            self._move_file(source_path, target_path, file_info['size'])
    
    def _move_file(self, source_path: Path, target_path: Path, file_size: int):
        """Move a single file with error handling."""
        try:
            if self.dry_run:
                logger.info(f"    [DRY RUN] Would move: {source_path.name} -> {target_path}")
                self.stats['files_moved'] += 1
                self.stats['total_size_moved'] += file_size
                return
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle file collision
            if target_path.exists():
                logger.warning(f"Target exists, creating backup: {target_path}")
                backup_path = target_path.with_suffix(f".backup_{int(datetime.now().timestamp())}")
                shutil.move(str(target_path), str(backup_path))
            
            # Move file
            shutil.move(str(source_path), str(target_path))
            
            logger.info(f"    ✅ Moved: {source_path.name} -> {target_path.name}")
            self.stats['files_moved'] += 1
            self.stats['total_size_moved'] += file_size
            
        except Exception as e:
            logger.error(f"    ❌ Failed to move {source_path}: {e}")
            self.errors.append(f"Failed to move {source_path}: {e}")
            self.stats['files_failed'] += 1
    
    def _create_indexes(self):
        """Create channel indexes and video indexes."""
        logger.info("📊 Creating channel and video indexes...")
        
        channels = self.v2_storage.list_all_channels()
        for channel_id in channels:
            videos = self.v2_storage.list_channel_videos(channel_id)
            
            # Create channel info
            channel_info = {
                'channel_id': channel_id,
                'name': channel_id,  # Will be updated when we have real data
                'video_count': len(videos),
                'migrated_at': datetime.now().isoformat(),
                'migration_source': 'v1_to_v2_script'
            }
            self.v2_storage.save_channel_info(channel_id, channel_info)
            
            # Update video index
            for video_id, video_info in videos:
                self.v2_storage.update_video_index(channel_id, video_id, video_info)
    
    def _create_completion_markers(self):
        """Create processing completion markers for migrated videos."""
        logger.info("✅ Creating completion markers...")
        
        channels = self.v2_storage.list_all_channels()
        for channel_id in channels:
            videos = self.v2_storage.list_channel_videos(channel_id)
            for video_id, video_info in videos:
                # Check if video has both media and transcript
                has_media = bool(self.v2_storage.find_media_files(channel_id, video_id))
                has_transcript = bool(self.v2_storage.find_transcript_files(channel_id, video_id))
                
                if has_media and has_transcript:
                    self.v2_storage.mark_processing_complete(
                        channel_id, 
                        video_id,
                        {
                            'migrated_from_v1': True,
                            'migration_date': datetime.now().isoformat(),
                            'has_media': has_media,
                            'has_transcript': has_transcript
                        }
                    )
    
    def _update_database_schema(self):
        """Update database schema if database exists."""
        logger.info("🗄️  Updating database schema...")
        
        try:
            # This is a simplified version - in production, use proper migrations
            db_path = self.settings.database_url.replace('sqlite:///', '')
            if Path(db_path).exists():
                # Add new columns to existing tables
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Add V2 columns to videos table
                try:
                    cursor.execute("ALTER TABLE videos ADD COLUMN video_title_snapshot TEXT")
                    cursor.execute("ALTER TABLE videos ADD COLUMN title_sanitized TEXT")
                    cursor.execute("ALTER TABLE videos ADD COLUMN storage_version TEXT DEFAULT 'v2'")
                    cursor.execute("ALTER TABLE videos ADD COLUMN processing_completed_at TIMESTAMP")
                    conn.commit()
                    logger.info("✅ Database schema updated")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e):
                        logger.info("Database schema already up to date")
                    else:
                        raise
                
                conn.close()
                
        except Exception as e:
            logger.warning(f"Database schema update failed (non-critical): {e}")
    
    def _validate_migration(self) -> bool:
        """Validate that migration completed successfully."""
        logger.info("🔍 Validating migration...")
        
        # Check that V2 structure exists
        v2_channels = self.v2_storage.list_all_channels()
        if not v2_channels:
            logger.error("No channels found in V2 structure")
            return False
        
        # Check file counts match
        v1_file_count = sum(1 for f in self.v1_storage.base_path.rglob('*') if f.is_file())
        v2_file_count = sum(1 for f in self.v2_storage.base_path.rglob('*') if f.is_file())
        
        # Account for new index and config files
        expected_new_files = len(v2_channels) * 2  # .channel_info.json and .video_index.json per channel
        expected_new_files += 1  # .storage_config.json
        
        if not self.dry_run and v2_file_count < (v1_file_count - expected_new_files) * 0.9:
            logger.error(f"File count mismatch: V1 had {v1_file_count}, V2 has {v2_file_count}")
            return False
        
        logger.info("✅ Migration validation passed")
        return True
    
    def _print_summary(self):
        """Print migration summary."""
        print("\n" + "="*60)
        print("📊 MIGRATION SUMMARY")
        print("="*60)
        print(f"Channels processed: {self.stats['channels_processed']}")
        print(f"Videos processed: {self.stats['videos_processed']}")
        print(f"Files moved: {self.stats['files_moved']}")
        print(f"Files failed: {self.stats['files_failed']}")
        print(f"Total size moved: {self.stats['total_size_moved'] / (1024**2):.1f} MB")
        
        if self.errors:
            print(f"\n❌ Errors encountered: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        if self.dry_run:
            print("\n⚠️  This was a DRY RUN - no files were actually moved")
            print("Run with --execute to perform the actual migration")
        else:
            print("\n✅ Migration completed successfully!")
            print(f"Backup created at: {self.backup_dir}")


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate yt-dl-sub storage from V1 to V2 structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (safe - shows what would happen)
  python migrate_storage_v2.py --dry-run
  
  # Execute migration with backup
  python migrate_storage_v2.py --execute --backup
  
  # Execute without backup (faster but risky)
  python migrate_storage_v2.py --execute --no-backup
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Perform dry run without making changes (default)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute actual migration (overrides --dry-run)'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='Create backup before migration (default)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup creation (faster but risky)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt (use with caution)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine if this is a dry run
    dry_run = not args.execute
    
    # Determine backup setting
    backup = args.backup and not args.no_backup
    
    # Show warning for actual execution
    if not dry_run and not args.force:
        print("⚠️  WARNING: This will modify your storage structure!")
        print("Make sure you have a backup of your data.")
        print("Type 'MIGRATE' to confirm:")
        
        confirmation = input().strip()
        if confirmation != 'MIGRATE':
            print("❌ Migration cancelled")
            return 1
    
    # Run migration
    migrator = StorageMigrator(dry_run=dry_run)
    success = migrator.run_migration(backup=backup)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())