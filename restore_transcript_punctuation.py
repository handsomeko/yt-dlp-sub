#!/usr/bin/env python3
"""
Script to restore punctuation in existing transcript files that lack proper sentence boundaries.
This processes SRT files and regenerates the TXT files with restored punctuation.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
from core.chinese_punctuation import restore_punctuation_for_file_sync, restore_punctuation_for_directory_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_punctuation_density(text: str, language: str = 'zh') -> float:
    """
    Check the density of sentence-ending punctuation in text
    
    Args:
        text: Text to check
        language: Language code
        
    Returns:
        Punctuation density (punctuation marks per 100 characters)
    """
    if not text:
        return 0.0
        
    if language == 'zh':
        punct_count = len(re.findall(r'[。！？]', text))
    else:
        punct_count = len(re.findall(r'[.!?]', text))
    
    text_length = len(text)
    if text_length == 0:
        return 0.0
        
    return (punct_count / text_length) * 100


def find_transcripts_needing_punctuation(base_path: Path, threshold: float = 1.0) -> List[Tuple[Path, Path]]:
    """
    Find transcript files that need punctuation restoration
    
    Args:
        base_path: Base storage path
        threshold: Minimum punctuation density (marks per 100 chars)
        
    Returns:
        List of (SRT file, TXT file) tuples needing restoration
    """
    transcripts_to_fix = []
    
    # Find all transcript directories
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
            
        for video_dir in channel_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            transcript_dir = video_dir / 'transcripts'
            if not transcript_dir.exists():
                continue
                
            # Check all .txt files
            for txt_file in transcript_dir.glob('*.zh.txt'):
                # Find corresponding SRT file
                srt_file = txt_file.with_suffix('.srt')
                if not srt_file.exists():
                    continue
                    
                # Read TXT content and check punctuation
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        txt_content = f.read()
                    
                    density = check_punctuation_density(txt_content, 'zh')
                    
                    if density < threshold:
                        transcripts_to_fix.append((srt_file, txt_file))
                        logger.debug(f"Found low punctuation ({density:.2f}/100): {txt_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Error checking {txt_file}: {e}")
    
    return transcripts_to_fix


def reprocess_transcript(srt_file: Path, txt_file: Path, backup: bool = True) -> bool:
    """
    Reprocess a single transcript to restore punctuation
    
    Args:
        srt_file: Path to SRT file
        txt_file: Path to TXT file to regenerate
        backup: Whether to backup original TXT file
        
    Returns:
        True if successful
    """
    try:
        # Backup original TXT if requested
        if backup and txt_file.exists():
            backup_file = txt_file.with_suffix('.txt.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(txt_file, backup_file)
                logger.debug(f"Created backup: {backup_file}")
        
        # Read SRT content
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Create extractor and convert to TXT with punctuation restoration
        extractor = LanguageAgnosticSubtitleExtractor()
        txt_content = extractor._srt_to_txt(srt_content)
        
        # Write new TXT file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        # Check new punctuation density
        new_density = check_punctuation_density(txt_content, 'zh')
        logger.info(f"✅ Restored punctuation for {txt_file.name} (density: {new_density:.2f}/100)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reprocess {txt_file}: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Restore punctuation in transcript files')
    parser.add_argument(
        '--path',
        type=str,
        default=os.getenv('STORAGE_PATH', '/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads'),
        help='Base storage path'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.0,
        help='Minimum punctuation density (marks per 100 chars) to consider OK'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just list files that would be processed'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Limit number of files to process (0 = no limit)'
    )
    
    args = parser.parse_args()
    
    # Ensure punctuation restoration is enabled
    os.environ['RESTORE_PUNCTUATION'] = 'true'
    
    base_path = Path(args.path)
    if not base_path.exists():
        logger.error(f"Path does not exist: {base_path}")
        return 1
    
    logger.info(f"Scanning for transcripts needing punctuation restoration...")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Threshold: {args.threshold} punctuation marks per 100 characters")
    
    # Find transcripts needing restoration
    transcripts_to_fix = find_transcripts_needing_punctuation(base_path, args.threshold)
    
    if not transcripts_to_fix:
        logger.info("✨ All transcripts have adequate punctuation!")
        return 0
    
    logger.info(f"Found {len(transcripts_to_fix)} transcripts needing punctuation restoration")
    
    if args.dry_run:
        logger.info("\n=== DRY RUN - Files that would be processed ===")
        for srt_file, txt_file in transcripts_to_fix[:args.limit] if args.limit else transcripts_to_fix:
            logger.info(f"  {txt_file.parent.parent.name}/{txt_file.name}")
        return 0
    
    # Process transcripts
    logger.info("\n=== Processing transcripts ===")
    
    success_count = 0
    error_count = 0
    
    to_process = transcripts_to_fix[:args.limit] if args.limit else transcripts_to_fix
    
    for i, (srt_file, txt_file) in enumerate(to_process, 1):
        logger.info(f"\n[{i}/{len(to_process)}] Processing {txt_file.parent.parent.name}/{txt_file.name}")
        
        if reprocess_transcript(srt_file, txt_file, backup=not args.no_backup):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SUMMARY")
    logger.info(f"  ✅ Successfully restored: {success_count}")
    logger.info(f"  ❌ Failed: {error_count}")
    logger.info(f"  📁 Total processed: {success_count + error_count}")
    
    if not args.no_backup:
        logger.info("\n💡 Tip: Original files backed up with .backup extension")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())