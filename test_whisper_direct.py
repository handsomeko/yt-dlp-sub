#!/usr/bin/env python3
"""
Direct test of Whisper transcription on an existing audio file.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from workers.transcriber import TranscribeWorker
from core.storage_paths_v2 import get_storage_paths_v2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_whisper_on_existing_video():
    """Test Whisper transcription on the @mikeynocode video we just downloaded."""
    
    # Video info from the download
    channel_id = "UCde0vB0fTwC8AT3sJofFV0w"
    video_id = "0S6IHp83zws"
    video_title = "How to Build a Simple App for the App Store Using AI (For Beginners)"
    video_url = "https://www.youtube.com/watch?v=0S6IHp83zws"
    
    # Find the audio file
    storage_paths = get_storage_paths_v2()
    media_dir = storage_paths.get_media_dir(channel_id, video_id)
    
    # Look for audio files
    audio_files = list(media_dir.glob("*.opus")) + list(media_dir.glob("*.mp3"))
    
    if not audio_files:
        logger.error(f"No audio files found in {media_dir}")
        return
    
    audio_path = str(audio_files[0])
    logger.info(f"Found audio file: {audio_path}")
    
    # Create TranscribeWorker
    transcriber = TranscribeWorker()
    
    # Prepare input
    transcriber_input = {
        'video_id': video_id,
        'video_url': video_url,
        'channel_id': channel_id,
        'audio_path': audio_path,
        'video_title': video_title
    }
    
    logger.info("Starting Whisper transcription...")
    
    # Execute transcription
    result = transcriber.execute(transcriber_input)
    
    # Check result
    if result.get('status') == 'success':
        logger.info("✅ Whisper transcription successful!")
        logger.info(f"Method used: {result.get('extraction_method')}")
        logger.info(f"Language detected: {result.get('language')}")
        logger.info(f"Word count: {result.get('word_count')}")
        logger.info(f"SRT file: {result.get('srt_path')}")
        logger.info(f"TXT file: {result.get('txt_path')}")
        
        # Check if files exist
        transcript_dir = storage_paths.get_transcript_dir(channel_id, video_id)
        whisper_files = list(transcript_dir.glob("*_whisper.*"))
        
        if whisper_files:
            logger.info(f"\n✅ Found {len(whisper_files)} Whisper files:")
            for f in whisper_files:
                logger.info(f"  - {f.name}")
        else:
            logger.warning("⚠️ No _whisper files found in transcript directory")
            
        # Also check for auto files
        auto_files = list(transcript_dir.glob("*_auto.*"))
        if auto_files:
            logger.info(f"\n✅ Found {len(auto_files)} auto-generated files:")
            for f in auto_files:
                logger.info(f"  - {f.name}")
                
        # Compare file sizes if both exist
        auto_txt = [f for f in auto_files if f.suffix == '.txt']
        whisper_txt = [f for f in whisper_files if f.suffix == '.txt']
        
        if auto_txt and whisper_txt:
            auto_size = auto_txt[0].stat().st_size
            whisper_size = whisper_txt[0].stat().st_size
            
            logger.info(f"\n📊 File size comparison:")
            logger.info(f"  Auto-generated: {auto_size:,} bytes")
            logger.info(f"  Whisper: {whisper_size:,} bytes")
            logger.info(f"  Difference: {whisper_size - auto_size:+,} bytes ({(whisper_size/auto_size - 1)*100:+.1f}%)")
            
    else:
        logger.error(f"❌ Whisper transcription failed: {result.get('error', 'Unknown error')}")
        logger.error(f"Full result: {result}")


if __name__ == "__main__":
    test_whisper_on_existing_video()