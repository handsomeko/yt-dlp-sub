#!/usr/bin/env python3
"""
Migration script to update existing subtitle files with language codes.
This script:
1. Scans existing transcript files without language codes
2. Detects the language from content
3. Renames files with proper language codes
4. Updates database records with detected languages
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from core.database import db_manager, Transcript, Video
from core.storage_paths_v2 import get_storage_paths_v2
from sqlalchemy import select

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LanguageFileMigration:
    """Handle migration of existing files to language-coded format"""
    
    def __init__(self):
        self.storage_paths = get_storage_paths_v2()
        self.migrated_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
    def detect_language_from_content(self, file_path: Path) -> str:
        """Detect language from subtitle file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()[:2000]  # First 2000 chars
            
            # Check for Chinese characters
            if re.search(r'[\u4e00-\u9fff]', content):
                # Further distinguish Chinese variants
                if '繁' in content or '體' in content or '臺' in content:
                    return 'zh-Hant'  # Traditional Chinese
                return 'zh-Hans'  # Simplified Chinese
            
            # Check for Japanese characters
            elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):
                return 'ja'
            
            # Check for Korean characters
            elif re.search(r'[\uac00-\ud7af]', content):
                return 'ko'
            
            # Check for Arabic
            elif re.search(r'[\u0600-\u06ff]', content):
                return 'ar'
            
            # Check for Hebrew
            elif re.search(r'[\u0590-\u05ff]', content):
                return 'he'
            
            # Check for Cyrillic (Russian, etc.)
            elif re.search(r'[\u0400-\u04ff]', content):
                return 'ru'
            
            # Check for Thai
            elif re.search(r'[\u0e00-\u0e7f]', content):
                return 'th'
            
            # Default to English for Latin scripts
            else:
                # Try to detect specific languages by common words
                content_lower = content.lower()
                if any(word in content_lower for word in ['le', 'la', 'les', 'de', 'un', 'une']):
                    return 'fr'
                elif any(word in content_lower for word in ['el', 'la', 'los', 'las', 'de', 'un', 'una']):
                    return 'es'
                elif any(word in content_lower for word in ['der', 'die', 'das', 'ist', 'ein', 'eine']):
                    return 'de'
                elif any(word in content_lower for word in ['il', 'la', 'le', 'di', 'un', 'una']):
                    return 'it'
                elif any(word in content_lower for word in ['o', 'a', 'os', 'as', 'um', 'uma']):
                    return 'pt'
                else:
                    return 'en'
                    
        except Exception as e:
            logger.error(f"Error detecting language from {file_path}: {e}")
            return 'unknown'
    
    def needs_migration(self, file_path: Path) -> bool:
        """Check if file needs language code migration"""
        filename = file_path.name
        # Check if filename already has language code
        # Pattern: filename.{lang}.ext
        pattern = r'\.[a-z]{2}(-[A-Z]{2,4})?\.(srt|txt)$'
        return not bool(re.search(pattern, filename))
    
    async def migrate_file(self, file_path: Path, video_id: str, video_title: str) -> Optional[Path]:
        """Migrate a single file to include language code"""
        try:
            if not self.needs_migration(file_path):
                logger.debug(f"File already has language code: {file_path}")
                self.skipped_count += 1
                return None
            
            # Detect language from content
            detected_lang = self.detect_language_from_content(file_path)
            
            # Create new filename with language code
            extension = file_path.suffix  # .srt or .txt
            new_filename = f"{video_title}.{detected_lang}{extension}"
            new_path = file_path.parent / new_filename
            
            # Check if target file already exists
            if new_path.exists() and new_path != file_path:
                # Create backup of existing file
                backup_path = new_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}{extension}')
                new_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Rename file
            file_path.rename(new_path)
            logger.info(f"Migrated: {file_path.name} -> {new_filename} (detected: {detected_lang})")
            self.migrated_count += 1
            
            return new_path
            
        except Exception as e:
            logger.error(f"Error migrating file {file_path}: {e}")
            self.error_count += 1
            return None
    
    async def migrate_video_files(self, channel_id: str, video_id: str, video_title: str) -> Dict[str, any]:
        """Migrate all transcript files for a video"""
        transcript_dir = self.storage_paths.get_transcript_dir(channel_id, video_id)
        
        if not transcript_dir.exists():
            return {'status': 'skipped', 'reason': 'no transcript directory'}
        
        migrated_files = []
        detected_languages = set()
        
        # Process all SRT and TXT files
        for pattern in ['*.srt', '*.txt']:
            for file_path in transcript_dir.glob(pattern):
                new_path = await self.migrate_file(file_path, video_id, video_title)
                if new_path:
                    migrated_files.append(str(new_path))
                    # Extract language from new filename
                    lang_match = re.search(r'\.([a-z]{2}(?:-[A-Z]{2,4})?)\.(srt|txt)$', new_path.name)
                    if lang_match:
                        detected_languages.add(lang_match.group(1))
        
        return {
            'status': 'success',
            'migrated_files': migrated_files,
            'detected_languages': list(detected_languages)
        }
    
    async def update_database_language(self, video_id: str, language: str) -> bool:
        """Update transcript language in database"""
        try:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(Transcript).where(Transcript.video_id == video_id)
                )
                transcript = result.scalar_one_or_none()
                
                if transcript:
                    old_language = transcript.language
                    transcript.language = language
                    await session.commit()
                    
                    if old_language != language:
                        logger.info(f"Updated database language for {video_id}: {old_language} -> {language}")
                        return True
                    else:
                        logger.debug(f"Database language already correct for {video_id}: {language}")
                        return False
                else:
                    logger.warning(f"No transcript record found for {video_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating database for {video_id}: {e}")
            return False
    
    async def run_migration(self, limit: Optional[int] = None):
        """Run the migration process"""
        logger.info("Starting language file migration...")
        
        try:
            async with db_manager.get_session() as session:
                # Get all videos with transcripts
                query = select(Video, Transcript).join(
                    Transcript, Video.video_id == Transcript.video_id
                )
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                records = result.all()
                
                logger.info(f"Found {len(records)} videos with transcripts to check")
                
                for video, transcript in records:
                    logger.info(f"\nProcessing video: {video.video_id} - {video.title}")
                    
                    # Migrate files
                    migration_result = await self.migrate_video_files(
                        video.channel_id,
                        video.video_id,
                        video.title or video.video_id
                    )
                    
                    # Update database if we detected languages
                    if migration_result['detected_languages']:
                        # Use first detected language (should typically be only one)
                        primary_language = migration_result['detected_languages'][0]
                        await self.update_database_language(video.video_id, primary_language)
                
                logger.info("\n" + "="*50)
                logger.info("Migration Summary:")
                logger.info(f"Files migrated: {self.migrated_count}")
                logger.info(f"Files skipped: {self.skipped_count}")
                logger.info(f"Errors: {self.error_count}")
                logger.info("="*50)
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise


async def main():
    """Main migration entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate transcript files to include language codes')
    parser.add_argument('--limit', type=int, help='Limit number of videos to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run logic
    
    migration = LanguageFileMigration()
    await migration.run_migration(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())