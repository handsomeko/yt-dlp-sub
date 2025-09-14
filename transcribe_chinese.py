#!/usr/bin/env python3
"""
Transcribe Chinese YouTube videos using Whisper when subtitles are unavailable.

This script handles the common case where Chinese YouTube videos only have 
English auto-generated captions, not native Chinese ones.
"""

import sys
import logging
from pathlib import Path
from workers.transcriber import TranscribeWorker
from core.storage_paths_v2 import get_storage_paths_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_chinese_video(video_id: str, channel_id: str = None):
    """
    Transcribe a Chinese video using Whisper with Chinese language setting.
    
    Args:
        video_id: YouTube video ID
        channel_id: YouTube channel ID (optional, will search if not provided)
    """
    try:
        # Get storage paths
        storage = get_storage_paths_v2()
        
        # Find the video directory
        if channel_id:
            video_dir = storage.get_video_dir(channel_id, video_id)
        else:
            # Search for video in all channels
            downloads_dir = Path(storage.base_path)
            video_dirs = list(downloads_dir.glob(f"*/{video_id}"))
            if not video_dirs:
                logger.error(f"Video {video_id} not found in downloads")
                return False
            video_dir = video_dirs[0]
            channel_id = video_dir.parent.name
        
        # Check for audio file
        media_dir = video_dir / 'media'
        audio_files = list(media_dir.glob("*.opus")) + list(media_dir.glob("*.mp3"))
        
        if not audio_files:
            logger.error(f"No audio file found for video {video_id}")
            return False
        
        audio_path = str(audio_files[0])
        logger.info(f"Found audio file: {audio_path}")
        
        # Get video metadata for language detection
        video_info_files = list(video_dir.glob("*_video_info.json"))
        video_title = None
        video_description = None
        
        if video_info_files:
            import json
            with open(video_info_files[0], 'r', encoding='utf-8') as f:
                video_info = json.load(f)
                video_title = video_info.get('title', '')
                video_description = video_info.get('description', '')
                logger.info(f"Video title: {video_title[:50]}...")
        
        # Initialize transcription worker
        transcriber = TranscribeWorker()
        
        # Transcribe with Chinese language setting
        logger.info("Starting Whisper transcription with Chinese language...")
        result = transcriber._extract_with_whisper_local(
            audio_path=audio_path,
            video_id=video_id,
            video_title=video_title,
            video_description=video_description
        )
        
        if result and result.get('status') == 'success':
            # Save Chinese transcript
            transcripts_dir = video_dir / 'transcripts'
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as Chinese SRT
            srt_path = transcripts_dir / f"{video_title or video_id}.zh.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(result.get('srt_content', ''))
            
            # Save as Chinese TXT
            txt_path = transcripts_dir / f"{video_title or video_id}.zh.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.get('text_content', ''))
            
            logger.info(f"✅ Successfully transcribed Chinese audio")
            logger.info(f"📄 Saved: {srt_path.name}")
            logger.info(f"📄 Saved: {txt_path.name}")
            
            return True
        else:
            logger.error(f"Transcription failed: {result.get('error_message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Error transcribing Chinese video: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_chinese.py <video_id> [channel_id]")
        print("\nExample:")
        print("  python transcribe_chinese.py SON6hKNHaDM UCYcMQmLxOKd9TMZguFEotww")
        sys.exit(1)
    
    video_id = sys.argv[1]
    channel_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = transcribe_chinese_video(video_id, channel_id)
    sys.exit(0 if success else 1)