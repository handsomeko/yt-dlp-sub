#!/usr/bin/env python3
"""
Batch Transcription Script - Phase 2 of Two-Phase Download Strategy

This script performs Whisper transcription and Chinese punctuation restoration
on videos that were downloaded with --skip-transcription in Phase 1.

Usage:
    python batch_transcribe.py --channel-id UCxxxxxxx --limit 10
    python batch_transcribe.py --all-channels --limit 50
    python batch_transcribe.py --video-path /path/to/video/dir
    python batch_transcribe.py --missing-only --all-channels
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_transcribe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchTranscriber:
    """
    Batch transcription processor for Phase 2 of two-phase download strategy.
    """
    
    def __init__(self):
        """Initialize the batch transcriber."""
        from core.storage_paths_v2 import get_storage_paths_v2
        from workers.transcriber import TranscribeWorker
        
        self.storage = get_storage_paths_v2()
        self.storage_dir = Path(self.storage.base_path)  # Add this for compatibility
        self.transcriber = TranscribeWorker()
        self.stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
    def find_channels(self) -> List[str]:
        """Find all channel directories in storage."""
        downloads_dir = Path(self.storage.base_path)
        channels = []
        
        for item in downloads_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                channels.append(item.name)
                
        logger.info(f"Found {len(channels)} channels in storage")
        return channels
    
    def find_videos_in_channel(self, channel_id: str) -> List[Path]:
        """Find all video directories in a channel."""
        channel_dir = Path(self.storage.base_path) / channel_id
        videos = []
        
        if not channel_dir.exists():
            logger.warning(f"Channel directory not found: {channel_id}")
            return videos
            
        for item in channel_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                videos.append(item)
                
        logger.info(f"Found {len(videos)} videos in channel {channel_id}")
        return videos
    
    def needs_transcription(self, video_dir: Path) -> bool:
        """Check if a video needs transcription."""
        # Check for existing Whisper transcripts
        transcripts_dir = video_dir / 'transcripts'
        if not transcripts_dir.exists():
            return True
            
        # Look for Chinese transcripts that might need Whisper processing
        whisper_files = list(transcripts_dir.glob("*_whisper.json"))
        chinese_txt_files = list(transcripts_dir.glob("*.zh*.txt"))
        
        # If we have auto-generated subtitles but no Whisper files, we might need Whisper
        auto_files = list(transcripts_dir.glob("*.zh-TW.txt")) + list(transcripts_dir.glob("*.zh-CN.txt"))
        
        if auto_files and not whisper_files:
            logger.info(f"Video {video_dir.name} has auto-subtitles but no Whisper - checking if Whisper needed")
            return True
            
        if not chinese_txt_files and not whisper_files:
            logger.info(f"Video {video_dir.name} has no Chinese transcripts - needs transcription")
            return True
            
        return False
    
    async def transcribe_video(self, video_dir: Path) -> bool:
        """Transcribe a single video using TranscribeWorker."""
        try:
            # Check for audio files
            media_dir = video_dir / 'media'
            if not media_dir.exists():
                logger.warning(f"No media directory for {video_dir.name}")
                return False
                
            audio_files = list(media_dir.glob("*.opus")) + list(media_dir.glob("*.mp3"))
            if not audio_files:
                logger.warning(f"No audio files found for {video_dir.name}")
                return False
                
            audio_file = audio_files[0]  # Use first available audio file
            
            # Prepare job data for TranscribeWorker
            job_data = {
                'video_id': video_dir.name,
                'channel_id': video_dir.parent.name,
                'audio_path': str(audio_file),
                'video_dir': str(video_dir),
                'video_title': video_dir.name,  # Use directory name as fallback
                'video_url': f'https://youtube.com/watch?v={video_dir.name}',  # Construct video URL
                'languages': [],  # Will be populated from existing subtitles if available
            }
            
            # Try to get actual video title from metadata
            try:
                metadata_files = list(video_dir.glob("*_video_info.json"))
                if metadata_files:
                    with open(metadata_files[0], 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if 'title' in metadata:
                            job_data['video_title'] = metadata['title']
            except Exception as e:
                logger.warning(f"Could not read metadata for {video_dir.name}: {e}")
            
            # Try to detect existing subtitle languages from transcripts directory
            try:
                transcripts_dir = video_dir / 'transcripts'
                if transcripts_dir.exists():
                    # Look for language-specific subtitle files (e.g., *.zh.srt, *.en.srt, *.zh-Hans-en.srt)
                    srt_files = list(transcripts_dir.glob("*.srt"))
                    detected_languages = []
                    for srt_file in srt_files:
                        # Extract language code from filename pattern
                        parts = srt_file.stem.rsplit('.', 1)
                        if len(parts) == 2:
                            lang_code = parts[1]
                            # Add the language code as-is (handles both simple and compound codes)
                            detected_languages.append(lang_code)
                    
                    # Remove duplicates while preserving order
                    unique_languages = []
                    for lang in detected_languages:
                        if lang not in unique_languages:
                            unique_languages.append(lang)
                    
                    if unique_languages:
                        job_data['languages'] = unique_languages
                        logger.info(f"Detected existing subtitle languages for {video_dir.name}: {unique_languages}")
            except Exception as e:
                logger.warning(f"Could not detect languages for {video_dir.name}: {e}")
            
            logger.info(f"Starting transcription for {job_data['video_title']}")
            
            # Process with TranscribeWorker
            # Note: TranscribeWorker.execute is synchronous, so we run it in executor
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.transcriber.execute, job_data)
            
            if result.get('success', False):
                logger.info(f"✅ Successfully transcribed {job_data['video_title']}")
                return True
            else:
                logger.error(f"❌ Failed to transcribe {job_data['video_title']}: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error transcribing {video_dir.name}: {e}")
            return False
    
    async def process_specific_videos(self, channel_id: str, video_ids: List[str]) -> Dict[str, int]:
        """Process specific videos in a channel by their IDs.
        
        Args:
            channel_id: The channel ID
            video_ids: List of video IDs to process
        
        Returns:
            Dict with processing statistics
        """
        logger.info(f"Processing {len(video_ids)} specific videos from channel: {channel_id}")
        
        channel_dir = self.storage_dir / channel_id
        if not channel_dir.exists():
            logger.error(f"Channel directory not found: {channel_dir}")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
        
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        
        for video_id in video_ids:
            video_dir = channel_dir / video_id
            if not video_dir.exists():
                logger.warning(f"Video directory not found: {video_dir}")
                stats['errors'] += 1
                continue
            
            self.stats['total_videos'] += 1
            
            try:
                success = await self.transcribe_video(video_dir)
                if success:
                    stats['processed'] += 1
                    self.stats['processed'] += 1
                else:
                    stats['errors'] += 1
                    self.stats['errors'] += 1
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                stats['errors'] += 1
                self.stats['errors'] += 1
        
        return stats
    
    async def process_channel(self, channel_id: str, limit: Optional[int] = None, missing_only: bool = False) -> Dict[str, int]:
        """Process all videos in a channel."""
        logger.info(f"Processing channel: {channel_id}")
        
        videos = self.find_videos_in_channel(channel_id)
        if not videos:
            return {'processed': 0, 'skipped': 0, 'errors': 0}
        
        if limit:
            videos = videos[:limit]
            logger.info(f"Limited to first {limit} videos")
        
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        
        for video_dir in videos:
            self.stats['total_videos'] += 1
            
            # Check if transcription is needed
            if missing_only and not self.needs_transcription(video_dir):
                logger.info(f"⏭️ Skipping {video_dir.name} - already has transcripts")
                stats['skipped'] += 1
                self.stats['skipped'] += 1
                continue
            
            try:
                success = await self.transcribe_video(video_dir)
                if success:
                    stats['processed'] += 1
                    self.stats['processed'] += 1
                else:
                    stats['errors'] += 1
                    self.stats['errors'] += 1
            except Exception as e:
                logger.error(f"Error processing {video_dir.name}: {e}")
                stats['errors'] += 1
                self.stats['errors'] += 1
        
        logger.info(f"Channel {channel_id} complete: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")
        return stats
    
    async def process_all_channels(self, limit_per_channel: Optional[int] = None, missing_only: bool = False) -> None:
        """Process all channels in storage."""
        channels = self.find_channels()
        
        logger.info(f"Starting batch transcription for {len(channels)} channels")
        
        for channel_id in channels:
            try:
                await self.process_channel(channel_id, limit_per_channel, missing_only)
            except Exception as e:
                logger.error(f"Error processing channel {channel_id}: {e}")
                continue
    
    async def process_single_video(self, video_path: str) -> None:
        """Process a single video directory."""
        video_dir = Path(video_path)
        if not video_dir.exists() or not video_dir.is_dir():
            logger.error(f"Video directory not found: {video_path}")
            return
        
        self.stats['total_videos'] = 1
        success = await self.transcribe_video(video_dir)
        
        if success:
            self.stats['processed'] = 1
        else:
            self.stats['errors'] = 1
    
    def print_summary(self) -> None:
        """Print processing summary."""
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("📊 BATCH TRANSCRIPTION SUMMARY")
        print("="*60)
        print(f"Total videos found: {self.stats['total_videos']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Skipped (already done): {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Duration: {duration}")
        
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['processed'] / self.stats['total_videos']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print("="*60)

async def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Batch transcription for Phase 2 processing')
    
    # Mutually exclusive group for target selection
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--channel-id', help='Process specific channel by ID')
    target_group.add_argument('--all-channels', action='store_true', help='Process all channels')
    target_group.add_argument('--video-path', help='Process single video directory')
    
    # Options
    parser.add_argument('--limit', type=int, help='Limit number of videos to process per channel')
    parser.add_argument('--missing-only', action='store_true', 
                       help='Only process videos that need transcription (skip those already done)')
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = BatchTranscriber()
    
    try:
        if args.channel_id:
            await transcriber.process_channel(args.channel_id, args.limit, args.missing_only)
        elif args.all_channels:
            await transcriber.process_all_channels(args.limit, args.missing_only)
        elif args.video_path:
            await transcriber.process_single_video(args.video_path)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        transcriber.print_summary()

if __name__ == "__main__":
    asyncio.run(main())