#!/usr/bin/env python3
"""
Batch process multiple videos with dual transcript extraction (auto + whisper).
"""

import sys
import time
from pathlib import Path
import logging
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from workers.transcriber import TranscribeWorker
from core.storage_paths_v2 import get_storage_paths_v2
from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_video_dual_transcripts(
    channel_id: str,
    video_id: str, 
    video_title: str,
    video_url: str
) -> Dict[str, Any]:
    """Process a single video with both auto and whisper transcripts."""
    
    results = {
        'video_id': video_id,
        'video_title': video_title,
        'video_url': video_url,
        'auto_transcript': None,
        'whisper_transcript': None,
        'errors': []
    }
    
    storage_paths = get_storage_paths_v2()
    media_dir = storage_paths.get_media_dir(channel_id, video_id)
    transcript_dir = storage_paths.get_transcript_dir(channel_id, video_id)
    
    # Find audio files
    audio_files = list(media_dir.glob("*.opus")) + list(media_dir.glob("*.mp3"))
    if not audio_files:
        results['errors'].append("No audio files found")
        return results
    
    audio_path = str(audio_files[0])
    logger.info(f"Processing {video_title} ({video_id})")
    logger.info(f"Audio file: {audio_path}")
    
    # Step 1: Extract auto-generated subtitles
    logger.info("Step 1: Extracting auto-generated subtitles...")
    try:
        extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
        auto_result = extractor.extract_subtitles(
            video_url=video_url,
            output_dir=transcript_dir,
            video_id=video_id,
            video_title=video_title.replace('/', '_').replace(':', '_')
        )
        
        if auto_result.success:
            results['auto_transcript'] = {
                'status': 'success',
                'languages': auto_result.languages_found,
                'files': auto_result.original_files,
                'methods_used': auto_result.methods_used
            }
            logger.info(f"✅ Auto-generated subtitles extracted: {auto_result.languages_found}")
        else:
            results['auto_transcript'] = {
                'status': 'failed',
                'error': str(auto_result.error)
            }
            results['errors'].append(f"Auto extraction failed: {auto_result.error}")
            
    except Exception as e:
        results['auto_transcript'] = {'status': 'failed', 'error': str(e)}
        results['errors'].append(f"Auto extraction error: {e}")
    
    # Step 2: Run Whisper transcription
    logger.info("Step 2: Running Whisper transcription...")
    try:
        transcriber = TranscribeWorker()
        transcriber_input = {
            'video_id': video_id,
            'video_url': video_url,
            'channel_id': channel_id,
            'audio_path': audio_path,
            'video_title': video_title
        }
        
        whisper_result = transcriber.execute(transcriber_input)
        
        if whisper_result.get('status') == 'success':
            results['whisper_transcript'] = {
                'status': 'success',
                'language': whisper_result.get('language'),
                'word_count': whisper_result.get('word_count'),
                'srt_path': whisper_result.get('srt_path'),
                'txt_path': whisper_result.get('txt_path'),
                'extraction_method': whisper_result.get('extraction_method')
            }
            logger.info(f"✅ Whisper transcription completed: {whisper_result.get('word_count')} words")
        else:
            results['whisper_transcript'] = {
                'status': 'failed',
                'error': whisper_result.get('error', 'Unknown error')
            }
            results['errors'].append(f"Whisper failed: {whisper_result.get('error')}")
            
    except Exception as e:
        results['whisper_transcript'] = {'status': 'failed', 'error': str(e)}
        results['errors'].append(f"Whisper error: {e}")
    
    return results


def main():
    """Process multiple videos with dual transcript extraction."""
    
    # Videos to process from @grittoglow channel
    videos_to_process = [
        {
            'channel_id': 'UCRdcuVfskYaubL7Ey0xBcfg',
            'video_id': 'xKsJ7BXLs8I',
            'video_title': 'How 1 Solopreneur Turned Excel Pain into $40K MRR (FormulaBot)',
            'video_url': 'https://www.youtube.com/watch?v=xKsJ7BXLs8I',
            'duration': 27
        },
        {
            'channel_id': 'UCRdcuVfskYaubL7Ey0xBcfg', 
            'video_id': 'ZsXmgmYHzxs',
            'video_title': 'How Tony Dinh Turned ChatGPT into a $45K/Month Solo Machine',
            'video_url': 'https://www.youtube.com/watch?v=ZsXmgmYHzxs',
            'duration': 21
        },
        {
            'channel_id': 'UCRdcuVfskYaubL7Ey0xBcfg',
            'video_id': '7fzf0XHiQeQ', 
            'video_title': '$0 to $45M ARR Jasper AI\u2019s First User Strategy Unlocked',
            'video_url': 'https://www.youtube.com/watch?v=7fzf0XHiQeQ',
            'duration': 18
        }
    ]
    
    logger.info("Starting batch dual transcript extraction...")
    logger.info(f"Processing {len(videos_to_process)} videos")
    
    all_results = []
    start_time = time.time()
    
    for i, video in enumerate(videos_to_process, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video {i}/{len(videos_to_process)}")
        logger.info(f"Title: {video['video_title']}")
        logger.info(f"Duration: {video['duration']} seconds")
        logger.info(f"{'='*60}")
        
        video_start_time = time.time()
        
        result = process_video_dual_transcripts(
            channel_id=video['channel_id'],
            video_id=video['video_id'],
            video_title=video['video_title'],
            video_url=video['video_url']
        )
        
        video_end_time = time.time()
        processing_time = video_end_time - video_start_time
        
        result['processing_time'] = processing_time
        result['duration'] = video['duration']
        all_results.append(result)
        
        logger.info(f"Video {i} completed in {processing_time:.1f} seconds")
        
        # Short break between videos
        if i < len(videos_to_process):
            logger.info("Short break before next video...")
            time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Videos processed: {len(all_results)}")
    
    successful_auto = sum(1 for r in all_results if r['auto_transcript'] and r['auto_transcript']['status'] == 'success')
    successful_whisper = sum(1 for r in all_results if r['whisper_transcript'] and r['whisper_transcript']['status'] == 'success')
    
    logger.info(f"Auto-generated successful: {successful_auto}/{len(all_results)}")
    logger.info(f"Whisper successful: {successful_whisper}/{len(all_results)}")
    
    # Save results to file
    import json
    output_file = Path("batch_dual_transcripts_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processing_time': total_time,
            'videos_processed': len(all_results),
            'successful_auto': successful_auto,
            'successful_whisper': successful_whisper,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()