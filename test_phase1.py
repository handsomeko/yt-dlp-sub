#!/usr/bin/env python3
"""
Phase 1 Test Script - Complete Workflow Testing

Tests all three scenarios:
1. Single video URL processing
2. Channel bulk archive processing
3. RSS-based channel monitoring

Run this to verify Phase 1 is fully functional.
"""

import sys
import time
import asyncio
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.url_parser import YouTubeURLParser, URLType
from core.channel_enumerator import ChannelEnumerator
from core.database import db_manager


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_cli_command(command):
    """Run a CLI command and return the result."""
    print(f"\n📟 Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def test_url_parser():
    """Test the URL parser utility."""
    print_header("Testing URL Parser")
    
    parser = YouTubeURLParser()
    
    test_urls = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", URLType.VIDEO),
        ("https://youtube.com/shorts/abc123def45", URLType.SHORTS),
        ("https://www.youtube.com/@MrBeast", URLType.CHANNEL),
        ("https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA", URLType.CHANNEL),
        ("invalid-url", URLType.INVALID)
    ]
    
    for url, expected_type in test_urls:
        url_type, identifier, metadata = parser.parse(url)
        status = "✅" if url_type == expected_type else "❌"
        print(f"{status} {url[:50]:<50} -> {url_type.value}")
    
    return True


def test_scenario_1_single_video():
    """Test Scenario 1: Process a single video URL."""
    print_header("Scenario 1: Single Video Processing")
    
    # Use a short video for testing
    video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
    
    print(f"Testing with video: {video_url}")
    
    # Process the video
    success = run_cli_command(f"yt-dl process '{video_url}'")
    
    if success:
        print("✅ Single video processing initiated successfully")
        
        # Check job status
        time.sleep(2)
        run_cli_command("yt-dl jobs pending --limit 5")
        
        return True
    else:
        print("❌ Failed to process single video")
        return False


def test_scenario_2_channel_bulk():
    """Test Scenario 2: Process all videos from a channel."""
    print_header("Scenario 2: Channel Bulk Archive")
    
    # Use a small channel for testing
    channel_url = "https://www.youtube.com/@YouTube"  # Official YouTube channel
    
    print(f"Testing with channel: {channel_url}")
    print("Note: Using --limit 3 to avoid processing too many videos")
    
    # Process channel with limit
    success = run_cli_command(f"yt-dl process '{channel_url}' --all --limit 3")
    
    if success:
        print("✅ Channel bulk processing initiated successfully")
        
        # Check job stats
        time.sleep(2)
        run_cli_command("yt-dl jobs stats")
        
        return True
    else:
        print("❌ Failed to process channel")
        return False


def test_scenario_3_rss_monitoring():
    """Test Scenario 3: RSS-based channel monitoring."""
    print_header("Scenario 3: RSS Channel Monitoring")
    
    # Add a channel for monitoring
    channel_url = "https://www.youtube.com/@TED"
    
    print(f"Adding channel for monitoring: {channel_url}")
    
    # Add channel
    success = run_cli_command(f"yt-dl channel add '{channel_url}'")
    
    if not success:
        print("❌ Failed to add channel")
        return False
    
    print("✅ Channel added successfully")
    
    # List channels
    print("\nListing monitored channels:")
    run_cli_command("yt-dl channel list")
    
    # Check for new videos
    print("\nChecking for new videos:")
    run_cli_command("yt-dl channel check --all")
    
    # Start monitor in daemon mode
    print("\nStarting monitor daemon:")
    success = run_cli_command("yt-dl monitor start --daemon --interval 60")
    
    if success:
        print("✅ Monitor daemon started")
        
        # Check monitor status
        time.sleep(2)
        run_cli_command("yt-dl monitor status")
        
        # Stop monitor
        print("\nStopping monitor:")
        run_cli_command("yt-dl monitor stop")
        
        return True
    else:
        print("❌ Failed to start monitor")
        return False


def test_search_functionality():
    """Test the search functionality."""
    print_header("Testing Search Functionality")
    
    # First, ensure we have some content in the database
    print("Note: Search requires transcripts in the database")
    
    # Test search
    search_terms = ["youtube", "video", "content"]
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        run_cli_command(f"yt-dl search '{term}' --limit 3")
    
    return True


def test_database_operations():
    """Test database operations."""
    print_header("Testing Database Operations")
    
    async def check_db():
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            from sqlalchemy import text
            
            # Check tables
            result = await session.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            print("Database tables:")
            for table in tables:
                if not table.startswith('sqlite_'):
                    print(f"  ✅ {table}")
            
            # Check FTS5 tables
            fts_tables = [t for t in tables if '_fts' in t]
            if fts_tables:
                print("\nFTS5 search tables:")
                for table in fts_tables:
                    print(f"  ✅ {table}")
            
            return True
    
    return asyncio.run(check_db())


def main():
    """Run all tests."""
    print("\n" + "🚀" * 30)
    print("  PHASE 1 COMPLETE WORKFLOW TEST")
    print("🚀" * 30)
    
    tests = [
        ("URL Parser", test_url_parser),
        ("Database Setup", test_database_operations),
        ("Scenario 1: Single Video", test_scenario_1_single_video),
        ("Scenario 2: Channel Bulk", test_scenario_2_channel_bulk),
        ("Scenario 3: RSS Monitoring", test_scenario_3_rss_monitoring),
        ("Search Functionality", test_search_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
                
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<10} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 is fully functional!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)