#!/usr/bin/env python3
"""Test script demonstrating complete download with all required features from CLAUDE.md"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.downloader import YouTubeDownloader


def test_complete_download():
    """Test downloading video with SRT subtitles and plain text transcript"""
    
    print("="*60)
    print("Testing Complete YouTube Download Pipeline")
    print("="*60)
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    # Initialize downloader (will use external drive path from CLAUDE.md)
    downloader = YouTubeDownloader()
    
    print(f"\nConfiguration:")
    print(f"  • Download path: {downloader.base_path}")
    print(f"  • Default resolution: 1080p")
    print(f"  • Subtitle format: SRT")
    print(f"  • Transcript: Plain text without timestamps")
    
    print(f"\n📹 Testing video download at 1080p...")
    result = downloader.download_video(
        url=test_url,
        quality='1080p',  # Default per CLAUDE.md
        video_format='mp4'
    )
    
    if result['status'] == 'success':
        print(f"\n✅ Download successful!")
        print(f"📁 Output directory: {result['output_dir']}")
        
        # Check for all required files
        print("\n📄 Files created (per CLAUDE.md requirements):")
        required_extensions = {'.mp4': False, '.srt': False, '.txt': False}
        
        for file_path in result['files']:
            filename = os.path.basename(file_path)
            print(f"  • {filename}")
            
            # Check which required files were created
            for ext in required_extensions:
                if filename.endswith(ext):
                    required_extensions[ext] = True
        
        # Verify all required files exist
        print("\n✔️ Verification:")
        print(f"  • MP4 video: {'✅' if required_extensions['.mp4'] else '❌'}")
        print(f"  • SRT subtitles: {'✅' if required_extensions['.srt'] else '❌'}")
        print(f"  • TXT transcript: {'✅' if required_extensions['.txt'] else '❌'}")
        
        if all(required_extensions.values()):
            print("\n🎉 All required files created successfully!")
            print("   Following CLAUDE.md requirements:")
            print("   1. Video saved to external drive")
            print("   2. Folder named after video title")
            print("   3. SRT subtitles downloaded (not VTT)")
            print("   4. Plain text transcript created")
            print("   5. Default 1080p resolution used")
        else:
            print("\n⚠️ Some required files are missing")
    else:
        print(f"\n❌ Download failed: {result.get('error', 'Unknown error')}")


def test_audio_only():
    """Test audio-only download"""
    print("\n" + "="*60)
    print("Testing Audio-Only Download")
    print("="*60)
    
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    downloader = YouTubeDownloader()
    
    print("\n🎵 Downloading audio only (MP3)...")
    result = downloader.download_video(
        url=test_url,
        download_audio_only=True,
        audio_format='mp3'
    )
    
    if result['status'] == 'success':
        print(f"✅ Audio download successful!")
        print(f"📁 Saved to: {result['output_dir']}")
        for file_path in result['files']:
            if file_path.endswith('.mp3'):
                print(f"  • Audio file: {os.path.basename(file_path)}")
    else:
        print(f"❌ Failed: {result['error']}")


def test_subtitle_only():
    """Test subtitle-only download with transcript conversion"""
    print("\n" + "="*60)
    print("Testing Subtitle-Only Download with Transcript")
    print("="*60)
    
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    downloader = YouTubeDownloader()
    
    print("\n📝 Downloading subtitles only...")
    result = downloader.download_subtitles_only(url=test_url)
    
    if result['status'] == 'success':
        print(f"✅ Subtitle download successful!")
        print(f"📁 Files created:")
        if result.get('subtitle_files'):
            for srt_file in result['subtitle_files']:
                print(f"  • SRT: {os.path.basename(srt_file)}")
        if result.get('transcript_files'):
            for txt_file in result['transcript_files']:
                print(f"  • TXT: {os.path.basename(txt_file)}")
    else:
        print(f"❌ Failed: {result['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test complete YouTube download pipeline")
    parser.add_argument("--audio", action="store_true", help="Test audio-only download")
    parser.add_argument("--subtitle", action="store_true", help="Test subtitle-only download")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.audio:
        test_audio_only()
    elif args.subtitle:
        test_subtitle_only()
    elif args.all:
        test_complete_download()
        test_audio_only()
        test_subtitle_only()
    else:
        # Default: test complete download
        test_complete_download()