#!/usr/bin/env python3
"""
CLI tool for exporting transcripts and generated content from the YouTube Content Intelligence platform.

This script provides a command-line interface to export data in various formats
with filtering options for channel and date ranges.

Usage examples:
    # Export all transcripts to JSON
    python3 export_cli.py --format json

    # Export specific channel to CSV
    python3 export_cli.py --format csv --channel-id UC123456789

    # Export recent videos to Markdown
    python3 export_cli.py --format markdown --since 2024-01-01

    # Export with generated content included
    python3 export_cli.py --format json --include-content --output exports/full_export.json
"""

import argparse
import asyncio
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from core.export import ExportService, ExportProgress
from core.database import DatabaseManager


def create_progress_callback(verbose: bool = False):
    """Create a progress callback function."""
    def progress_callback(progress: ExportProgress):
        if verbose:
            print(f"\r[{progress.current_phase.upper()}] "
                  f"{progress.progress_percent:.1f}% "
                  f"({progress.current_item}/{progress.total_items})", 
                  end="", flush=True)
        elif progress.current_item % 10 == 0:  # Less frequent updates
            print(f"\r{progress.progress_percent:.0f}%", end="", flush=True)
    
    return progress_callback


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Export transcripts and generated content from YouTube Content Intelligence platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --format json                                    # Export all to JSON
  %(prog)s --format csv --channel-id UC123456789          # Export specific channel to CSV
  %(prog)s --format markdown --since 2024-01-01           # Export recent videos
  %(prog)s --format txt --include-content                 # Export with generated content
  %(prog)s --stats                                        # Show export statistics only
        """
    )
    
    # Main action
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='Show export statistics without performing export'
    )
    
    # Export options
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv', 'txt', 'markdown'],
        default='json',
        help='Export format (default: json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--include-content',
        action='store_true',
        help='Include generated content in export'
    )
    
    # Filtering options
    parser.add_argument(
        '--channel-id',
        type=str,
        help='Filter by specific channel ID'
    )
    
    parser.add_argument(
        '--since',
        type=str,
        help='Include videos published since this date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--until',
        type=str,
        help='Include videos published until this date (YYYY-MM-DD)'
    )
    
    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to process in each batch (default: 100)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress information'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    # Database options
    parser.add_argument(
        '--database-url',
        type=str,
        help='Database URL (uses default if not specified)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.stats and not args.format:
        parser.error("Must specify --format or --stats")
    
    if args.verbose and args.quiet:
        parser.error("Cannot use --verbose and --quiet together")
    
    try:
        # Initialize database
        if args.database_url:
            db_manager = DatabaseManager(args.database_url)
        else:
            from core.database import db_manager
            
        await db_manager.initialize()
        
        # Create export service
        export_service = ExportService(db_manager)
        
        # Show statistics if requested
        if args.stats:
            print("📊 Export Statistics")
            print("=" * 40)
            
            # Overall stats
            stats = await export_service.get_export_stats(
                channel_id=args.channel_id,
                since=args.since,
                until=args.until
            )
            
            print(f"Total transcripts: {stats.total_transcripts}")
            print(f"Total generated content items: {stats.total_content_items}")
            print(f"Total videos: {stats.total_videos}")
            print(f"Total channels: {stats.total_channels}")
            
            if stats.date_range_start:
                print(f"Date range start: {stats.date_range_start.strftime('%Y-%m-%d')}")
            if stats.date_range_end:
                print(f"Date range end: {stats.date_range_end.strftime('%Y-%m-%d')}")
            
            if args.channel_id:
                print(f"Filtered by channel: {args.channel_id}")
            
            # Exit if only showing stats
            if not args.format:
                await db_manager.close()
                return 0
        
        # Perform export
        print(f"\n📤 Exporting transcripts to {args.format.upper()} format")
        
        # Setup progress callback
        progress_callback = None
        if not args.quiet:
            progress_callback = create_progress_callback(args.verbose)
        
        # Run export
        result = await export_service.export_transcripts(
            format=args.format,
            channel_id=args.channel_id,
            since=args.since,
            until=args.until,
            output_path=args.output,
            include_content=args.include_content,
            progress_callback=progress_callback,
            batch_size=args.batch_size
        )
        
        if not args.quiet:
            print()  # New line after progress
        
        # Display results
        print("✅ Export completed successfully!")
        print(f"📄 Output file: {result['output_path']}")
        
        stats = result['stats']
        print(f"📊 Statistics:")
        print(f"   - Records exported: {stats.total_transcripts}")
        print(f"   - File size: {stats.file_size_bytes:,} bytes")
        print(f"   - Processing time: {stats.export_duration_seconds:.2f} seconds")
        
        # Show applied filters
        filters = result['filters']
        if any(filters.values()):
            print(f"🔍 Applied filters:")
            if filters['channel_id']:
                print(f"   - Channel ID: {filters['channel_id']}")
            if filters['since']:
                print(f"   - Since: {filters['since']}")
            if filters['until']:
                print(f"   - Until: {filters['until']}")
            if filters['include_content']:
                print(f"   - Include generated content: Yes")
        
        await db_manager.close()
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Export cancelled by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)