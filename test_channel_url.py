#!/usr/bin/env python3
"""
Test script to verify channel_url.txt creation
"""

from pathlib import Path
from core.storage_paths_v2 import get_storage_paths_v2
from core.downloader import YouTubeDownloader
import sys


def test_channel_url_creation():
    """Test that channel_url.txt is created when downloading videos"""
    
    print("Testing channel_url.txt creation...")
    print("=" * 50)
    
    # Test URL - use a small video for quick testing
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
    
    # Create downloader
    downloader = YouTubeDownloader()
    storage_paths = get_storage_paths_v2()
    
    # Get video info first
    print(f"Getting video info for: {test_url}")
    info = downloader.get_video_info(test_url)
    
    if 'error' in info:
        print(f"❌ Error getting video info: {info['error']}")
        return False
    
    channel_id = info.get('channel_id') or info.get('uploader_id') or 'unknown_channel'
    channel_name = info.get('channel') or info.get('uploader') or 'Unknown'
    channel_url = info.get('channel_url') or info.get('uploader_url') or ''
    
    print(f"Channel: {channel_name}")
    print(f"Channel ID: {channel_id}")
    print(f"Channel URL from metadata: {channel_url}")
    print()
    
    # Download the video (audio only for speed)
    print("Downloading video (audio only)...")
    result = downloader.download_video(
        url=test_url,
        download_audio_only=True,
        audio_format='opus'
    )
    
    if result.get('status') != 'success':
        print(f"❌ Download failed: {result.get('error')}")
        return False
    
    print("✅ Download successful")
    print()
    
    # Check if channel_url.txt was created
    channel_dir = storage_paths.get_channel_dir(channel_id)
    channel_url_file = channel_dir / 'channel_url.txt'
    
    print(f"Channel directory: {channel_dir}")
    print(f"Checking for channel_url.txt...")
    
    if channel_url_file.exists():
        with open(channel_url_file, 'r') as f:
            saved_url = f.read().strip()
        print(f"✅ channel_url.txt exists!")
        print(f"   Content: {saved_url}")
        
        # Verify the URL is valid
        if saved_url.startswith('https://www.youtube.com/'):
            print(f"✅ URL is valid YouTube channel URL")
            return True
        else:
            print(f"⚠️  URL doesn't look like a valid YouTube channel URL")
            return False
    else:
        print(f"❌ channel_url.txt was not created")
        
        # List what files are in the channel directory
        if channel_dir.exists():
            print(f"\nFiles in channel directory:")
            for file in channel_dir.iterdir():
                print(f"   - {file.name}")
        
        return False


def test_utility_script():
    """Test the add_channel_urls.py utility script"""
    
    print("\nTesting add_channel_urls.py utility script...")
    print("=" * 50)
    
    import subprocess
    import os
    
    # Check if the script exists
    script_path = Path(__file__).parent / 'add_channel_urls.py'
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"Running: python {script_path}")
    
    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Utility script ran successfully")
            return True
        else:
            print(f"❌ Utility script failed with exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Utility script timed out")
        return False
    except Exception as e:
        print(f"❌ Error running utility script: {e}")
        return False


def main():
    """Main test runner"""
    
    print("Channel URL Test Suite")
    print("=" * 50)
    print()
    
    tests_passed = []
    
    # Test 1: Channel URL creation during download
    test1_result = test_channel_url_creation()
    tests_passed.append(("Channel URL creation", test1_result))
    
    # Test 2: Utility script
    test2_result = test_utility_script()
    tests_passed.append(("Utility script", test2_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    for test_name, passed in tests_passed:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in tests_passed)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())