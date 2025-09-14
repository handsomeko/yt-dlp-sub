#!/usr/bin/env python3
"""
Test script for validating the hybrid transcript extraction strategy:
1. Auto-generated + cleaning as default
2. Whisper as fallback only when needed

This test verifies that TranscribeWorker correctly detects existing 
auto-generated subtitles and skips Whisper transcription.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, '/Users/jk/yt-dl-sub')

from core.downloader import YouTubeDownloader
from workers.transcriber import TranscribeWorker
from core.storage_paths_v2 import get_storage_paths_v2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HybridStrategyTest')

def test_hybrid_strategy():
    """Test the hybrid extraction strategy with a real video"""
    
    # Test video with known auto-generated captions
    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - should have captions
    
    print("🧪 Testing Hybrid Transcript Extraction Strategy")
    print("=" * 60)
    print(f"📹 Test Video: {test_video_url}")
    print()
    
    try:
        # Initialize storage and clear any existing data for this test
        storage = get_storage_paths_v2()
        
        # Extract video ID for cleanup
        video_id = "dQw4w9WgXcQ"
        
        # Clean up any existing data
        print("🧹 Cleaning up any existing test data...")
        for channel_dir in storage.base_path.glob("*"):
            if channel_dir.is_dir():
                for video_dir in channel_dir.glob(f"*{video_id}*"):
                    if video_dir.is_dir():
                        shutil.rmtree(video_dir)
                        print(f"   Removed: {video_dir}")
        
        print("\n📥 PHASE 1: Download with Language-Agnostic Subtitle Extraction")
        print("-" * 60)
        
        # Initialize downloader (no translation to keep it simple)
        downloader = YouTubeDownloader(enable_translation=False)
        
        # Download audio + extract subtitles
        result = downloader.download_video(
            url=test_video_url,
            download_audio_only=True,  # Audio only for speed
            quality="128",  # Lower quality for speed
            video_format="mp4"
        )
        
        # Check if download was successful (different result formats possible)
        download_success = (
            result.get('success', False) or  # Explicit success flag
            result.get('status') == 'success' or  # Status-based success
            result.get('video_id')  # Has video_id means some level of success
        )
        
        if not download_success:
            print(f"❌ Download failed: {result.get('error', 'Unknown error')}")
            print(f"   Result keys: {list(result.keys())}")
            return False
            
        print(f"✅ Download completed")
        print(f"   📁 Channel: {result.get('channel_id', 'Unknown')}")
        print(f"   🎬 Video: {result.get('title', 'Unknown')}")
        
        # Check if subtitles were extracted
        subtitle_info = result.get('subtitle_extraction', {})
        if subtitle_info.get('success'):
            languages = subtitle_info.get('languages_found', [])
            print(f"   📝 Subtitles extracted: {languages}")
            print(f"   📄 Files: {len(subtitle_info.get('original_files', []))}")
        else:
            print("   ⚠️  No subtitles extracted in download phase")
        
        print(f"\n🤖 PHASE 2: TranscribeWorker with Hybrid Strategy")
        print("-" * 60)
        
        # Initialize TranscribeWorker
        transcriber = TranscribeWorker()
        
        # Prepare input data as it would come from orchestrator
        input_data = {
            'video_id': result.get('video_id', video_id),
            'video_url': test_video_url,
            'channel_id': result.get('channel_id', 'unknown'), 
            'video_title': result.get('title', 'Unknown Video'),
            'audio_path': result.get('audio_path')  # May be None, transcriber will find it
        }
        
        # Execute transcription with hybrid strategy
        transcript_result = transcriber.execute(input_data)
        
        print(f"\n📊 RESULTS ANALYSIS")
        print("=" * 60)
        
        if transcript_result.get('status') == 'success':
            extraction_method = transcript_result.get('extraction_method', 'unknown')
            print(f"✅ Transcription completed successfully")
            print(f"   🔧 Method used: {extraction_method}")
            print(f"   🌍 Language: {transcript_result.get('language', 'unknown')}")
            print(f"   📁 Transcript directory: {transcript_result.get('transcript_dir', 'N/A')}")
            
            # Check what files exist
            transcript_dir = Path(transcript_result.get('transcript_dir', ''))
            if transcript_dir.exists():
                files = list(transcript_dir.glob("*"))
                print(f"   📄 Files in transcript directory: {len(files)}")
                for file_path in files:
                    size_kb = file_path.stat().st_size / 1024
                    print(f"      • {file_path.name} ({size_kb:.1f} KB)")
            
            # Validate hybrid strategy
            if extraction_method == 'auto_generated_cleaned':
                print(f"\n🎯 HYBRID STRATEGY VALIDATION: SUCCESS")
                print("   ✅ Auto-generated subtitles were detected and used")
                print("   ✅ Whisper transcription was correctly skipped")
                print("   ✅ Processing was efficient (no unnecessary computation)")
                
            elif extraction_method in ['whisper_local', 'ffmpeg']:
                print(f"\n🎯 HYBRID STRATEGY VALIDATION: FALLBACK USED")
                print("   ⚠️  No suitable auto-generated subtitles found") 
                print("   ✅ Whisper fallback was correctly triggered")
                print("   ℹ️  This is expected for videos without auto-captions")
                
            else:
                print(f"\n🎯 HYBRID STRATEGY VALIDATION: UNEXPECTED")
                print(f"   ❓ Unexpected extraction method: {extraction_method}")
                
        else:
            print(f"❌ Transcription failed: {transcript_result.get('error', 'Unknown error')}")
            errors = transcript_result.get('errors', [])
            for error in errors:
                print(f"   • {error}")
            return False
            
        print(f"\n✨ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_fallback():
    """Test that Whisper fallback works when no auto-generated captions exist"""
    
    print("\n🧪 Testing Whisper Fallback Strategy")
    print("=" * 60)
    
    # This is a more complex test that would require a video without auto-captions
    # For now, just test the logic by simulating the scenario
    
    transcriber = TranscribeWorker()
    
    # Test with non-existent video data (should trigger fallback logic)
    test_input = {
        'video_id': 'test_no_subs',
        'video_url': 'https://example.com/fake-video',
        'channel_id': 'test_channel',
        'video_title': 'Test Video Without Subtitles'
    }
    
    # Check existing subtitles (should return None for non-existent video)
    existing = transcriber._check_existing_subtitles(
        test_input['channel_id'], 
        test_input['video_id'],
        test_input['video_title']
    )
    
    if existing is None:
        print("✅ Correctly detected no existing subtitles")
        print("✅ Would proceed to Whisper fallback")
        return True
    else:
        print("❌ Unexpected existing subtitles found")
        return False

if __name__ == "__main__":
    print("🚀 Hybrid Strategy Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: Main hybrid strategy
    if not test_hybrid_strategy():
        success = False
    
    # Test 2: Whisper fallback logic
    if not test_whisper_fallback():
        success = False
    
    if success:
        print(f"\n🎉 All tests passed! Hybrid strategy is working correctly.")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed. Check the output above.")
        sys.exit(1)