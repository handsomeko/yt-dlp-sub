#!/usr/bin/env python3
"""
Batch Chinese Punctuation Restoration Script

This script processes existing Chinese transcripts to add proper punctuation
using AI-powered restoration. It's designed to work with the existing 
storage structure and handle files safely.

Usage:
    python scripts/restore_chinese_punctuation.py [options]

Options:
    --channel CHANNEL_ID    Process specific channel only
    --limit N              Limit number of files to process
    --dry-run             Preview what would be processed without making changes
    --pattern PATTERN     File pattern to match (default: *.zh.txt)
    --backup              Create backups (default: True)
    --force               Process files that already have punctuation
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.chinese_punctuation import ChinesePunctuationRestorer, restore_punctuation_for_directory
from core.storage_paths_v2 import get_storage_paths_v2
from workers.ai_backend import AIBackend
from config.settings import get_settings

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/chinese_punctuation_restore.log', mode='a')
        ]
    )


def find_chinese_transcript_files(
    storage_paths, 
    channel_id: Optional[str] = None,
    pattern: str = "*.zh.txt"
) -> List[Path]:
    """
    Find Chinese transcript files that need punctuation restoration.
    
    Args:
        storage_paths: Storage paths manager
        channel_id: Optional specific channel to process
        pattern: File pattern to search for
    
    Returns:
        List of file paths to process
    """
    base_path = Path(storage_paths.base_path)
    files_to_process = []
    
    if channel_id:
        # Process specific channel
        channel_dirs = [base_path / channel_id]
    else:
        # Process all channels
        channel_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for channel_dir in channel_dirs:
        if not channel_dir.exists():
            continue
            
        logger.info(f"Scanning channel directory: {channel_dir}")
        
        # Find all video directories
        for video_dir in channel_dir.iterdir():
            if not video_dir.is_dir() or video_dir.name.startswith('.'):
                continue
            
            # Look in transcripts subdirectory
            transcript_dir = video_dir / "transcripts"
            if transcript_dir.exists():
                # Find matching files
                matching_files = list(transcript_dir.glob(pattern))
                for file_path in matching_files:
                    # Skip files that already have punctuation (unless --force)
                    if "_punctuated" not in file_path.name:
                        files_to_process.append(file_path)
    
    return files_to_process


def analyze_file_content(file_path: Path) -> Dict[str, any]:
    """
    Analyze a transcript file to determine if it needs punctuation restoration.
    
    Args:
        file_path: Path to transcript file
    
    Returns:
        Dict with analysis results
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Basic analysis
        char_count = len(content)
        chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
        chinese_ratio = chinese_chars / char_count if char_count > 0 else 0
        
        # Check for punctuation
        chinese_punctuation = '。，！？：；""''《》【】（）、'
        has_punctuation = any(punct in content for punct in chinese_punctuation)
        punctuation_count = sum(1 for c in content if c in chinese_punctuation)
        
        return {
            'file_path': str(file_path),
            'char_count': char_count,
            'chinese_chars': chinese_chars,
            'chinese_ratio': chinese_ratio,
            'has_punctuation': has_punctuation,
            'punctuation_count': punctuation_count,
            'needs_restoration': chinese_ratio > 0.5 and not has_punctuation,
            'content_preview': content[:100] + "..." if len(content) > 100 else content
        }
    
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e),
            'needs_restoration': False
        }


async def process_files_batch(
    files_to_process: List[Path],
    ai_backend: AIBackend,
    create_backup: bool = True,
    force: bool = False
) -> Dict[str, List[str]]:
    """
    Process multiple files for punctuation restoration.
    
    Args:
        files_to_process: List of file paths to process
        ai_backend: AI backend instance
        create_backup: Whether to create backups
        force: Process files even if they appear to have punctuation
    
    Returns:
        Dict with processing results
    """
    restorer = ChinesePunctuationRestorer(ai_backend)
    
    results = {
        'successful': [],
        'skipped': [],
        'failed': [],
        'errors': []
    }
    
    for i, file_path in enumerate(files_to_process, 1):
        try:
            logger.info(f"Processing file {i}/{len(files_to_process)}: {file_path}")
            
            # Analyze file first
            analysis = analyze_file_content(file_path)
            
            if not force and not analysis.get('needs_restoration', False):
                logger.info(f"Skipping {file_path} - doesn't need restoration or already has punctuation")
                results['skipped'].append(str(file_path))
                continue
            
            # Process the file
            success = await restorer.restore_file_punctuation(file_path, create_backup)
            
            if success:
                results['successful'].append(str(file_path))
                logger.info(f"✅ Successfully processed: {file_path}")
            else:
                results['skipped'].append(str(file_path))
                logger.info(f"⏭️  Skipped (no changes needed): {file_path}")
            
            # Add small delay between files to avoid overwhelming the AI backend
            await asyncio.sleep(0.5)
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            results['failed'].append(str(file_path))
            results['errors'].append(error_msg)
    
    return results


def print_summary(results: Dict[str, List[str]], dry_run: bool = False):
    """Print processing summary"""
    if dry_run:
        print("\n🔍 DRY RUN SUMMARY:")
    else:
        print("\n📊 PROCESSING SUMMARY:")
    
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")  
    print(f"❌ Failed: {len(results['failed'])}")
    
    if results['successful']:
        print("\n✅ Successfully processed files:")
        for file_path in results['successful']:
            print(f"  - {file_path}")
    
    if results['failed']:
        print("\n❌ Failed files:")
        for file_path in results['failed']:
            print(f"  - {file_path}")
        
        print("\n📋 Error details:")
        for error in results['errors']:
            print(f"  - {error}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Batch restore punctuation for Chinese transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--channel', 
        help='Process specific channel ID only'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        help='Limit number of files to process'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Preview what would be processed without making changes'
    )
    parser.add_argument(
        '--pattern', 
        default='*.zh.txt', 
        help='File pattern to match (default: *.zh.txt)'
    )
    parser.add_argument(
        '--backup', 
        action='store_true', 
        default=True,
        help='Create backups before modification (default: True)'
    )
    parser.add_argument(
        '--no-backup', 
        action='store_true', 
        help='Disable backup creation'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Process files that already appear to have punctuation'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle backup setting
    create_backup = args.backup and not args.no_backup
    
    logger.info("Starting Chinese punctuation restoration batch processing")
    logger.info(f"Channel filter: {args.channel or 'All channels'}")
    logger.info(f"File pattern: {args.pattern}")
    logger.info(f"Limit: {args.limit or 'No limit'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Create backups: {create_backup}")
    logger.info(f"Force processing: {args.force}")
    
    try:
        # Initialize components
        settings = get_settings()
        storage_paths = get_storage_paths_v2()
        
        # Check if Chinese punctuation is enabled
        if not getattr(settings, 'chinese_punctuation_enabled', False):
            logger.warning("Chinese punctuation restoration is disabled in settings")
            logger.info("Enable it by setting CHINESE_PUNCTUATION_ENABLED=true in .env")
            return 1
        
        # Find files to process
        logger.info("Scanning for Chinese transcript files...")
        files_to_process = find_chinese_transcript_files(
            storage_paths, 
            args.channel, 
            args.pattern
        )
        
        if args.limit:
            files_to_process = files_to_process[:args.limit]
        
        logger.info(f"Found {len(files_to_process)} files to analyze")
        
        if not files_to_process:
            print("No Chinese transcript files found to process.")
            return 0
        
        # Analyze files first
        print(f"\n🔍 Analyzing {len(files_to_process)} files...")
        analyses = []
        for file_path in files_to_process:
            analysis = analyze_file_content(file_path)
            analyses.append(analysis)
        
        # Show analysis summary
        needs_restoration = [a for a in analyses if a.get('needs_restoration', False)]
        already_has_punctuation = [a for a in analyses if a.get('has_punctuation', False)]
        
        print(f"📈 Analysis Results:")
        print(f"  - Files needing restoration: {len(needs_restoration)}")
        print(f"  - Files with existing punctuation: {len(already_has_punctuation)}")
        print(f"  - Total files: {len(files_to_process)}")
        
        if args.dry_run:
            print("\n📋 Files that would be processed:")
            for analysis in needs_restoration:
                print(f"  - {analysis['file_path']} ({analysis['chinese_chars']} Chinese chars)")
            return 0
        
        # Process files
        if needs_restoration or args.force:
            ai_backend = AIBackend()
            
            files_to_actually_process = files_to_process if args.force else [
                Path(a['file_path']) for a in needs_restoration
            ]
            
            print(f"\n🔄 Processing {len(files_to_actually_process)} files...")
            results = await process_files_batch(
                files_to_actually_process,
                ai_backend,
                create_backup,
                args.force
            )
            
            print_summary(results)
            
            if results['failed']:
                return 1
        else:
            print("\n✨ All files already have punctuation - nothing to do!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)