#!/usr/bin/env python3
"""Test script to verify everything works locally"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.downloader import YouTubeDownloader
from core.transcript import TranscriptExtractor  
from core.monitor import ChannelMonitor


def test_components():
    """Test each component individually"""
    
    print("="*50)
    print("🧪 Testing YouTube Download & Monitor System")
    print("="*50)
    
    # Test video to use (short video)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
    
    # 1. Test Downloader
    print("\n1️⃣ Testing Downloader...")
    downloader = YouTubeDownloader()
    
    # Get video info
    info = downloader.get_video_info(test_url)
    if 'error' not in info:
        print(f"   ✅ Video info retrieved: {info['title']}")
    else:
        print(f"   ❌ Error: {info['error']}")
    
    # 2. Test Transcript Extractor
    print("\n2️⃣ Testing Transcript Extractor...")
    transcript_ext = TranscriptExtractor()
    
    # Extract video ID
    video_id = transcript_ext.extract_video_id(test_url)
    if video_id:
        print(f"   ✅ Video ID extracted: {video_id}")
        
        # Get transcript
        result = transcript_ext.get_transcript(test_url, save_to_file=False)
        if result['status'] == 'success':
            print(f"   ✅ Transcript extracted: {result['transcript_entries']} entries")
        else:
            print(f"   ⚠️  No transcript available: {result['error']}")
    else:
        print("   ❌ Failed to extract video ID")
    
    # 3. Test Channel Monitor
    print("\n3️⃣ Testing Channel Monitor...")
    monitor = ChannelMonitor()
    
    # Add a test channel (Google Developers)
    test_channel_id = "UC_x5XG1OV2P6uZZ5FSM9Ttw"
    success = monitor.add_channel(test_channel_id, "Google Developers")
    if success:
        print("   ✅ Channel added to monitor")
        
        # Check for recent videos
        videos = monitor.check_channel_for_new_videos(test_channel_id, days_back=30)
        print(f"   ✅ Found {len(videos)} videos in last 30 days")
        
        if videos:
            print(f"   📹 Latest: {videos[0]['title']}")
    else:
        print("   ❌ Failed to add channel")
    
    # 4. Test API Server
    print("\n4️⃣ Testing API Server...")
    print("   ℹ️  To test the API, run in another terminal:")
    print("   cd /Users/jk/yt-dl-sub/api")
    print("   python main.py")
    print("   Then visit: http://localhost:8000/docs")
    
    print("\n" + "="*50)
    print("✨ Basic tests complete! Check the downloads folder for any files.")
    print("="*50)


def quick_download_test():
    """Quick test to download a video"""
    print("\n🎬 Quick Download Test")
    print("-"*30)
    
    url = input("Enter a YouTube URL (or press Enter for default): ").strip()
    if not url:
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        print(f"Using default: {url}")
    
    downloader = YouTubeDownloader()
    
    print("\nDownloading video...")
    result = downloader.download_video(url, quality="720p")  # Use 720p for faster test
    
    if result['status'] == 'success':
        print(f"✅ Downloaded successfully!")
        print(f"📁 Files saved to: {result['output_dir']}")
        print(f"📄 Files: {result['files']}")
    else:
        print(f"❌ Download failed: {result['error']}")
    
    # Try to get transcript
    print("\nExtracting transcript...")
    transcript_ext = TranscriptExtractor()
    transcript = transcript_ext.get_transcript(url, save_to_file=True, video_title=result.get('title'))
    
    if transcript['status'] == 'success':
        print(f"✅ Transcript extracted!")
        if 'text_file' in transcript:
            print(f"📄 Saved to: {transcript['text_file']}")
    else:
        print(f"⚠️ No transcript available: {transcript['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YouTube Download & Monitor System")
    parser.add_argument("--quick", action="store_true", help="Run quick download test")
    args = parser.parse_args()
    
    if args.quick:
        quick_download_test()
    else:
        test_components()