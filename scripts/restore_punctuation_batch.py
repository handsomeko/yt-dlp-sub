#!/usr/bin/env python3
"""
Standalone batch punctuation restoration script for Chinese transcripts.
Processes existing Chinese transcript files without punctuation.
Uses the unified synchronous implementation for consistency.

Usage:
    python scripts/restore_punctuation_batch.py                  # Process all .zh.txt files
    python scripts/restore_punctuation_batch.py --file FILE      # Process specific file
    python scripts/restore_punctuation_batch.py --dir DIR        # Process directory
    python scripts/restore_punctuation_batch.py --dry-run        # Preview without changes
    python scripts/restore_punctuation_batch.py --force          # Skip confirmation prompt
    python scripts/restore_punctuation_batch.py --estimate       # Show cost estimate only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.chinese_punctuation import ChinesePunctuationRestorer
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPunctuationProcessor:
    """Batch processor for Chinese punctuation restoration using unified sync implementation."""
    
    def __init__(self, dry_run: bool = False, show_progress: bool = True):
        """Initialize batch processor."""
        self.dry_run = dry_run
        self.show_progress = show_progress
        self.restorer = ChinesePunctuationRestorer()
        self.processed_count = 0
        self.success_count = 0
        self.skip_count = 0
        self.error_count = 0
        self.total_cost = 0.0
        self.total_api_calls = 0
        
    def check_enabled(self) -> bool:
        """Check if Chinese punctuation is enabled in settings."""
        try:
            settings = get_settings()
            if not getattr(settings, 'chinese_punctuation_enabled', False):
                logger.warning("Chinese punctuation restoration is disabled in settings")
                logger.info("Set CHINESE_PUNCTUATION_ENABLED=true in .env to enable")
                return False
        except:
            # If settings not available, assume enabled
            pass
            
        # Check Claude CLI availability
        if not self.restorer.check_claude_cli():
            logger.error("Claude CLI is not available or not working")
            logger.info("Please install Claude CLI: https://docs.anthropic.com/claude/docs/claude-cli")
            return False
            
        logger.info(f"✅ Chinese punctuation restoration is enabled")
        logger.info(f"📊 Configuration:")
        logger.info(f"  - Chunk size: {self.restorer.max_chunk_size} chars")
        logger.info(f"  - Timeout: {self.restorer.timeout} seconds per chunk")
        
        return True
        
    def process_file(self, file_path: Path) -> bool:
        """Process a single file synchronously."""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Check if it's Chinese text
            if not self.restorer.detect_chinese_text(content):
                logger.info(f"⏭️  Skipping non-Chinese file: {file_path.name}")
                self.skip_count += 1
                return False
                
            # Check if already has punctuation
            if self.restorer.has_punctuation(content):
                logger.info(f"✅ Already has punctuation: {file_path.name}")
                self.skip_count += 1
                return False
                
            # Estimate cost
            cost, num_calls = self.restorer.estimate_cost(content)
            self.total_cost += cost
            self.total_api_calls += num_calls
            
            logger.info(f"🔤 Processing: {file_path.name} ({len(content)} chars, {num_calls} API calls, ~${cost:.4f})")
            
            if self.dry_run:
                logger.info(f"  [DRY RUN] Would restore punctuation for {file_path.name}")
                self.processed_count += 1
                return True
                
            # Create backup
            backup_path = self.restorer.create_backup(file_path)
            
            # Restore punctuation using consolidated system
            restored_text, success = self.restorer.restore_punctuation_sync(content)
            
            if success:
                # Write restored content
                file_path.write_text(restored_text, encoding='utf-8')
                logger.info(f"✅ Successfully restored punctuation: {file_path.name}")
                self.success_count += 1
                return True
            else:
                logger.warning(f"⚠️  No changes made to: {file_path.name}")
                # Remove backup if no changes
                if backup_path.exists():
                    backup_path.unlink()
                self.skip_count += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            self.error_count += 1
            return False
    
    def find_chinese_files(self, search_path: Path = None) -> List[Path]:
        """Find all Chinese transcript files without punctuation."""
        if search_path is None:
            # Use default storage path
            from core.storage_paths_v2 import get_storage_paths_v2
            storage = get_storage_paths_v2()
            search_path = storage.base_path
            
        # Find all .zh.txt files
        pattern = "*.zh.txt"
        files = list(search_path.rglob(pattern))
        
        # Also check for other Chinese transcript patterns
        additional_patterns = ["*_zh.txt", "*chinese*.txt", "*中文*.txt"]
        for pattern in additional_patterns:
            files.extend(search_path.rglob(pattern))
            
        # Remove duplicates
        files = list(set(files))
        
        logger.info(f"Found {len(files)} potential Chinese transcript files")
        return sorted(files)  # Sort for consistent ordering
        
    def process_batch(self, files: List[Path]):
        """Process a batch of files."""
        total = len(files)
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"\n[{i}/{total}] Processing file...")
            self.processed_count += 1
            self.process_file(file_path)
                
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("📊 Processing Summary")
        print("="*60)
        print(f"Total files processed: {self.processed_count}")
        print(f"✅ Successfully punctuated: {self.success_count}")
        print(f"⏭️  Skipped (already has punctuation): {self.skip_count}")
        print(f"❌ Errors: {self.error_count}")
        if self.total_api_calls > 0:
            print(f"💰 Total estimated cost: ${self.total_cost:.4f} USD")
            print(f"📡 Total API calls: {self.total_api_calls}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch restore punctuation to Chinese transcripts"
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process a specific file'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Process all Chinese transcripts in directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be processed without making changes'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process'
    )
    parser.add_argument(
        '--estimate',
        action='store_true',
        help='Show cost estimate only (implies --dry-run)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress display for each chunk'
    )
    
    args = parser.parse_args()
    
    # Estimate implies dry-run
    if args.estimate:
        args.dry_run = True
    
    # Initialize processor
    processor = BatchPunctuationProcessor(
        dry_run=args.dry_run,
        show_progress=not args.no_progress
    )
    
    if not processor.check_enabled():
        return 1
        
    # Determine files to process
    files_to_process = []
    
    if args.file:
        # Process specific file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            return 1
        files_to_process = [file_path]
        
    elif args.dir:
        # Process directory
        dir_path = Path(args.dir)
        if not dir_path.exists():
            logger.error(f"Directory not found: {args.dir}")
            return 1
        files_to_process = processor.find_chinese_files(dir_path)
        
    else:
        # Process all Chinese transcripts in storage
        files_to_process = processor.find_chinese_files()
        
    if not files_to_process:
        logger.info("No Chinese transcript files found to process")
        return 0
        
    # Apply limit if specified
    if args.limit:
        files_to_process = files_to_process[:args.limit]
        
    # Show what will be processed
    print("\n" + "="*60)
    print(f"📝 Found {len(files_to_process)} files to process")
    print("="*60)
    
    # For estimate mode, calculate total cost
    if args.estimate:
        total_chars = 0
        total_cost = 0.0
        total_calls = 0
        
        for f in files_to_process:
            try:
                content = f.read_text(encoding='utf-8')
                if processor.restorer.detect_chinese_text(content) and not processor.restorer.has_punctuation(content):
                    cost, calls = processor.restorer.estimate_cost(content)
                    total_chars += len(content)
                    total_cost += cost
                    total_calls += calls
            except:
                pass
                
        print(f"\n💰 Cost Estimate:")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Total API calls: {total_calls}")
        print(f"  Estimated cost: ${total_cost:.4f} USD")
        print(f"  Average per file: ${total_cost/len(files_to_process):.4f} USD" if files_to_process else "")
        return 0
    
    if not args.force and not args.dry_run:
        # Show first few files
        for i, f in enumerate(files_to_process[:5]):
            print(f"  {i+1}. {f.name}")
        if len(files_to_process) > 5:
            print(f"  ... and {len(files_to_process) - 5} more")
            
        # Quick cost estimate for confirmation
        sample_size = min(5, len(files_to_process))
        sample_cost = 0.0
        for f in files_to_process[:sample_size]:
            try:
                content = f.read_text(encoding='utf-8')
                if processor.restorer.detect_chinese_text(content) and not processor.restorer.has_punctuation(content):
                    cost, _ = processor.restorer.estimate_cost(content)
                    sample_cost += cost
            except:
                pass
        
        if sample_cost > 0:
            estimated_total = (sample_cost / sample_size) * len(files_to_process)
            print(f"\n💰 Estimated cost: ~${estimated_total:.2f} USD")
            
        response = input("\nProceed with punctuation restoration? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0
            
    # Process files
    processor.process_batch(files_to_process)
    
    # Print summary
    processor.print_summary()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)