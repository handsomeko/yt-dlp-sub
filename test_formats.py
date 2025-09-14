#!/usr/bin/env python3
"""Test script for different video/audio formats and resolutions"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.downloader import YouTubeDownloader
import json


def test_formats():
    """Test various format and quality options"""
    
    print("="*60)
    print("🎬 Testing Video/Audio Format Options")
    print("="*60)
    
    # Test URL - short video
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    downloader = YouTubeDownloader()
    
    # Get video info first
    print("\n📹 Video Info:")
    info = downloader.get_video_info(test_url)
    if 'error' not in info:
        print(f"Title: {info['title']}")
        print(f"Duration: {info.get('duration', 'N/A')} seconds")
        print(f"Channel: {info.get('channel', 'N/A')}")
    
    tests = []
    
    # Test 1: Different video resolutions
    print("\n" + "="*60)
    print("1️⃣  Testing Video Resolutions")
    print("-"*40)
    
    resolutions = ["1080p", "720p", "480p", "360p"]
    
    print("\nAvailable resolutions to test:")
    for i, res in enumerate(resolutions, 1):
        print(f"  {i}. {res}")
    print(f"  {len(resolutions)+1}. Skip video resolution tests")
    
    choice = input("\nSelect resolution to test (1-5): ").strip()
    
    if choice and choice != str(len(resolutions)+1):
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(resolutions):
                selected_res = resolutions[idx]
                print(f"\n🎬 Downloading in {selected_res}...")
                result = downloader.download_video(
                    url=test_url,
                    quality=selected_res,
                    video_format="mp4"
                )
                
                if result['status'] == 'success':
                    print(f"✅ Downloaded {selected_res} successfully!")
                    print(f"📁 Saved to: {result['output_dir']}")
                    tests.append(f"Video {selected_res}: SUCCESS")
                else:
                    print(f"❌ Failed: {result['error']}")
                    tests.append(f"Video {selected_res}: FAILED")
        except:
            print("Invalid selection")
    
    # Test 2: Audio formats
    print("\n" + "="*60)
    print("2️⃣  Testing Audio Formats")
    print("-"*40)
    
    audio_formats = ["mp3", "m4a", "wav"]
    
    print("\nAvailable audio formats to test:")
    for i, fmt in enumerate(audio_formats, 1):
        print(f"  {i}. {fmt}")
    print(f"  {len(audio_formats)+1}. Skip audio format tests")
    
    choice = input("\nSelect audio format to test (1-4): ").strip()
    
    if choice and choice != str(len(audio_formats)+1):
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(audio_formats):
                selected_format = audio_formats[idx]
                print(f"\n🎵 Downloading audio as {selected_format}...")
                result = downloader.download_video(
                    url=test_url,
                    download_audio_only=True,
                    audio_format=selected_format
                )
                
                if result['status'] == 'success':
                    print(f"✅ Downloaded {selected_format} successfully!")
                    print(f"📁 Saved to: {result['output_dir']}")
                    tests.append(f"Audio {selected_format}: SUCCESS")
                else:
                    print(f"❌ Failed: {result['error']}")
                    tests.append(f"Audio {selected_format}: FAILED")
        except:
            print("Invalid selection")
    
    # Test 3: Video container formats
    print("\n" + "="*60)
    print("3️⃣  Testing Video Container Formats")
    print("-"*40)
    
    video_formats = ["mp4", "mkv", "webm"]
    
    print("\nAvailable video container formats:")
    for i, fmt in enumerate(video_formats, 1):
        print(f"  {i}. {fmt}")
    print(f"  {len(video_formats)+1}. Skip container format tests")
    
    choice = input("\nSelect container format to test (1-4): ").strip()
    
    if choice and choice != str(len(video_formats)+1):
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_formats):
                selected_format = video_formats[idx]
                print(f"\n📦 Downloading as {selected_format}...")
                result = downloader.download_video(
                    url=test_url,
                    quality="720p",
                    video_format=selected_format
                )
                
                if result['status'] == 'success':
                    print(f"✅ Downloaded as {selected_format} successfully!")
                    print(f"📁 Saved to: {result['output_dir']}")
                    tests.append(f"Container {selected_format}: SUCCESS")
                else:
                    print(f"❌ Failed: {result['error']}")
                    tests.append(f"Container {selected_format}: FAILED")
        except:
            print("Invalid selection")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    if tests:
        for test in tests:
            print(f"  • {test}")
    else:
        print("  No tests were run")
    
    print("\n✨ Format testing complete!")
    print("Check the downloads folder for output files")


def interactive_download():
    """Interactive download with format selection"""
    
    print("\n" + "="*60)
    print("🎯 Interactive Download with Format Selection")
    print("="*60)
    
    url = input("\nEnter YouTube URL: ").strip()
    if not url:
        print("No URL provided, exiting...")
        return
    
    print("\nWhat would you like to download?")
    print("  1. Video (with audio)")
    print("  2. Audio only")
    
    media_type = input("\nSelect (1 or 2): ").strip()
    
    downloader = YouTubeDownloader()
    
    if media_type == "1":
        # Video download
        print("\nSelect video quality:")
        qualities = ["2160p (4K)", "1440p", "1080p (Full HD)", "720p (HD)", "480p", "360p", "best available", "smallest size"]
        quality_values = ["2160p", "1440p", "1080p", "720p", "480p", "360p", "best", "worst"]
        
        for i, q in enumerate(qualities, 1):
            print(f"  {i}. {q}")
        
        q_choice = input("\nSelect quality (1-8): ").strip()
        try:
            quality = quality_values[int(q_choice) - 1]
        except:
            quality = "1080p"
            print("Using default: 1080p")
        
        print("\nSelect video format:")
        print("  1. MP4 (most compatible)")
        print("  2. MKV (preserves quality)")
        print("  3. WebM (web optimized)")
        
        f_choice = input("\nSelect format (1-3): ").strip()
        format_map = {"1": "mp4", "2": "mkv", "3": "webm"}
        video_format = format_map.get(f_choice, "mp4")
        
        print(f"\n⏬ Downloading video in {quality} as {video_format}...")
        result = downloader.download_video(
            url=url,
            quality=quality,
            video_format=video_format
        )
        
    elif media_type == "2":
        # Audio download
        print("\nSelect audio format:")
        print("  1. MP3 (most compatible)")
        print("  2. M4A (better quality)")
        print("  3. WAV (uncompressed)")
        print("  4. FLAC (lossless)")
        
        a_choice = input("\nSelect format (1-4): ").strip()
        format_map = {"1": "mp3", "2": "m4a", "3": "wav", "4": "flac"}
        audio_format = format_map.get(a_choice, "mp3")
        
        print(f"\n⏬ Downloading audio as {audio_format}...")
        result = downloader.download_video(
            url=url,
            download_audio_only=True,
            audio_format=audio_format
        )
    else:
        print("Invalid selection")
        return
    
    # Show result
    if result['status'] == 'success':
        print("\n✅ Download successful!")
        print(f"📁 Files saved to: {result['output_dir']}")
        print("\nFiles downloaded:")
        for file in result.get('files', []):
            print(f"  • {os.path.basename(file)}")
    else:
        print(f"\n❌ Download failed: {result['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video/audio format options")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive download mode")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Run format tests")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_download()
    else:
        test_formats()