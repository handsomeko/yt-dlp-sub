#!/usr/bin/env python3
"""
Example usage of the export functionality.

This script demonstrates how to use the ExportService class and convenience functions
to export transcripts in different formats with various filtering options.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from core.export import ExportService, ExportProgress, export_transcripts, get_export_stats
from core.database import db_manager


def print_progress(progress: ExportProgress):
    """Progress callback function to display export progress."""
    bar_length = 30
    filled_length = int(bar_length * progress.progress_percent / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{progress.current_phase}: |{bar}| {progress.progress_percent:.1f}% '
          f'({progress.current_item}/{progress.total_items})', end='', flush=True)


async def example_basic_exports():
    """Demonstrate basic export functionality in different formats."""
    print("📤 Basic Export Examples")
    print("=" * 40)
    
    export_service = ExportService()
    
    # Get statistics first
    stats = await export_service.get_export_stats()
    print(f"Available for export: {stats.total_transcripts} transcripts from {stats.total_channels} channels")
    
    if stats.total_transcripts == 0:
        print("⚠️  No transcripts available for export. Please run the downloader first.")
        return
    
    # Define output directory
    output_dir = Path("exports/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export in different formats
    formats = [
        ("json", "Complete data with all metadata"),
        ("csv", "Spreadsheet-friendly tabular format"),
        ("txt", "Plain text for easy reading"),
        ("markdown", "Formatted text with headings")
    ]
    
    for format_name, description in formats:
        print(f"\n📄 Exporting to {format_name.upper()} - {description}")
        
        try:
            result = await export_service.export_transcripts(
                format=format_name,
                output_path=output_dir / f"all_transcripts.{format_name}",
                include_content=True,  # Include generated content
                progress_callback=print_progress
            )
            
            print(f"\n✅ Export completed: {result['output_path']}")
            stats = result['stats']
            print(f"   📊 {stats.total_transcripts} transcripts, "
                  f"{stats.file_size_bytes:,} bytes, "
                  f"{stats.export_duration_seconds:.2f}s")
            
        except Exception as e:
            print(f"\n❌ Export failed: {e}")


async def example_filtered_exports():
    """Demonstrate filtered export functionality."""
    print("\n\n🔍 Filtered Export Examples")
    print("=" * 40)
    
    export_service = ExportService()
    
    # Define output directory
    output_dir = Path("exports/filtered")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example 1: Export by channel
    print("\n📺 Example 1: Export specific channel")
    
    # First, let's see what channels are available
    stats = await export_service.get_export_stats()
    if stats.total_channels > 0:
        # Get first available channel ID (this would be a real channel ID in practice)
        # For demonstration, let's assume a channel ID
        example_channel_id = "UC123456789"  # Replace with actual channel ID
        
        try:
            result = await export_service.export_transcripts(
                format="json",
                channel_id=example_channel_id,
                output_path=output_dir / f"channel_{example_channel_id}.json",
                include_content=False,
                progress_callback=print_progress
            )
            
            print(f"\n✅ Channel export completed: {result['output_path']}")
            
        except Exception as e:
            print(f"\n⚠️  Channel export failed (this is expected with demo data): {e}")
    
    # Example 2: Export by date range
    print("\n📅 Example 2: Export recent videos (last 30 days)")
    
    since_date = datetime.now() - timedelta(days=30)
    
    try:
        result = await export_service.export_transcripts(
            format="csv",
            since=since_date,
            output_path=output_dir / "recent_transcripts.csv",
            include_content=True,
            progress_callback=print_progress
        )
        
        print(f"\n✅ Date-filtered export completed: {result['output_path']}")
        
    except Exception as e:
        print(f"\n❌ Date-filtered export failed: {e}")
    
    # Example 3: Export date range
    print("\n📆 Example 3: Export specific date range")
    
    start_date = datetime.now() - timedelta(days=60)
    end_date = datetime.now() - timedelta(days=30)
    
    try:
        result = await export_service.export_transcripts(
            format="markdown",
            since=start_date,
            until=end_date,
            output_path=output_dir / "date_range_transcripts.md",
            include_content=True,
            progress_callback=print_progress
        )
        
        print(f"\n✅ Date range export completed: {result['output_path']}")
        
    except Exception as e:
        print(f"\n❌ Date range export failed: {e}")


async def example_convenience_functions():
    """Demonstrate convenience functions for quick exports."""
    print("\n\n⚡ Convenience Functions Examples")
    print("=" * 40)
    
    # Define output directory
    output_dir = Path("exports/convenience")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick stats check
    print("\n📊 Quick statistics check:")
    stats = await get_export_stats()
    print(f"   Total transcripts: {stats.total_transcripts}")
    print(f"   Total channels: {stats.total_channels}")
    print(f"   Total videos: {stats.total_videos}")
    
    # Quick export using convenience function
    print("\n⚡ Quick export using convenience function:")
    
    try:
        result = await export_transcripts(
            format="txt",
            output_path=output_dir / "quick_export.txt",
            include_content=False,
            progress_callback=print_progress
        )
        
        print(f"\n✅ Quick export completed: {result['output_path']}")
        
    except Exception as e:
        print(f"\n❌ Quick export failed: {e}")


async def example_custom_progress_tracking():
    """Demonstrate custom progress tracking."""
    print("\n\n📈 Custom Progress Tracking Example")
    print("=" * 40)
    
    # Custom progress callback with more detailed information
    def detailed_progress_callback(progress: ExportProgress):
        print(f"\n📍 Progress Update:")
        print(f"   Phase: {progress.current_phase}")
        print(f"   Items: {progress.current_item}/{progress.total_items}")
        print(f"   Percent: {progress.progress_percent:.2f}%")
        if progress.estimated_completion:
            print(f"   ETA: {progress.estimated_completion}")
    
    export_service = ExportService()
    output_dir = Path("exports/custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = await export_service.export_transcripts(
            format="json",
            output_path=output_dir / "detailed_progress.json",
            include_content=True,
            progress_callback=detailed_progress_callback,
            batch_size=50  # Smaller batch size for more frequent progress updates
        )
        
        print(f"\n✅ Export with detailed progress completed: {result['output_path']}")
        
    except Exception as e:
        print(f"\n❌ Export with detailed progress failed: {e}")


async def example_error_handling():
    """Demonstrate error handling in export operations."""
    print("\n\n⚠️  Error Handling Examples")
    print("=" * 40)
    
    export_service = ExportService()
    
    # Example 1: Invalid format
    print("\n🔥 Example 1: Invalid format")
    try:
        await export_service.export_transcripts(format="xml")  # Unsupported format
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Example 2: Invalid date format
    print("\n🔥 Example 2: Invalid date format")
    try:
        await export_service.export_transcripts(
            format="json",
            since="not-a-date"
        )
    except ValueError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Example 3: Non-existent output directory (should be created automatically)
    print("\n🔧 Example 3: Non-existent output directory (should auto-create)")
    try:
        result = await export_service.export_transcripts(
            format="json",
            output_path="/tmp/yt_dl_sub_test/deep/nested/path/test.json"
        )
        print(f"   ✅ Directory auto-created successfully: {result['output_path']}")
        
        # Clean up
        Path(result['output_path']).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


async def main():
    """Run all export examples."""
    print("🌟 YouTube Content Intelligence - Export Examples")
    print("=" * 60)
    
    try:
        # Initialize database connection
        await db_manager.initialize()
        
        # Run all examples
        await example_basic_exports()
        await example_filtered_exports()
        await example_convenience_functions()
        await example_custom_progress_tracking()
        await example_error_handling()
        
        print("\n\n🎉 All export examples completed!")
        print("\n💡 Tips:")
        print("   - Check the 'exports' directory for generated files")
        print("   - Use different formats based on your needs:")
        print("     • JSON: Complete data for programmatic access")
        print("     • CSV: For spreadsheet analysis")
        print("     • TXT: For reading and text processing")
        print("     • Markdown: For documentation and reports")
        print("   - Use filters to export specific subsets of data")
        print("   - Monitor progress for large exports")
        
    except Exception as e:
        print(f"\n💥 Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up database connection
        await db_manager.close()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)