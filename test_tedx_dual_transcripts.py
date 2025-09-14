#!/usr/bin/env python3
"""
Test dual transcript extraction on TEDx video (clear English speech content).
"""

import sys
import time
from pathlib import Path
import logging
from typing import Dict, Any

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


def test_tedx_dual_transcripts():
    """Test dual transcript extraction on TEDx video."""
    
    # TEDx video details
    channel_id = "UCsT0YIqwnpJCM-mx7-gSA4Q"
    video_id = "m8WomdCLBqE" 
    video_title = "Is AI making us dumber? Maybe. | Charlie Gedeon | TEDxSherbrooke Street West"
    video_url = "https://www.youtube.com/watch?v=m8WomdCLBqE"
    duration = 847  # 14 minutes 7 seconds
    
    logger.info("=" * 80)
    logger.info("DUAL TRANSCRIPT TEST: TEDx Video")
    logger.info("=" * 80)
    logger.info(f"Title: {video_title}")
    logger.info(f"Duration: {duration} seconds ({duration//60}:{duration%60:02d})")
    logger.info(f"Video URL: {video_url}")
    
    storage_paths = get_storage_paths_v2()
    media_dir = storage_paths.get_media_dir(channel_id, video_id)
    transcript_dir = storage_paths.get_transcript_dir(channel_id, video_id)
    
    # Find audio files
    audio_files = list(media_dir.glob("*.opus")) + list(media_dir.glob("*.mp3"))
    if not audio_files:
        logger.error("No audio files found!")
        return
    
    audio_path = str(audio_files[0])
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Audio file size: {Path(audio_path).stat().st_size / (1024*1024):.1f} MB")
    
    results = {
        'video_info': {
            'channel_id': channel_id,
            'video_id': video_id,
            'video_title': video_title,
            'video_url': video_url,
            'duration_seconds': duration,
            'audio_path': audio_path
        },
        'auto_transcript': None,
        'whisper_transcript': None,
        'comparison': None,
        'errors': []
    }
    
    total_start_time = time.time()
    
    # Step 1: Extract auto-generated subtitles
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Extracting auto-generated subtitles...")
    logger.info("=" * 60)
    
    auto_start_time = time.time()
    try:
        extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
        auto_result = extractor.extract_subtitles(
            video_url=video_url,
            output_dir=transcript_dir,
            video_id=video_id,
            video_title=video_title.replace('/', '_').replace(':', '_').replace('|', '_')
        )
        
        auto_processing_time = time.time() - auto_start_time
        
        if auto_result.success:
            # Find the created auto files
            auto_files = list(transcript_dir.glob("*_auto.*"))
            auto_txt_files = [f for f in auto_files if f.suffix == '.txt']
            auto_srt_files = [f for f in auto_files if f.suffix == '.srt']
            
            auto_txt_content = ""
            auto_word_count = 0
            auto_file_sizes = {}
            
            if auto_txt_files:
                auto_txt_content = auto_txt_files[0].read_text(encoding='utf-8')
                auto_word_count = len(auto_txt_content.split())
                auto_file_sizes['txt'] = auto_txt_files[0].stat().st_size
            
            if auto_srt_files:
                auto_file_sizes['srt'] = auto_srt_files[0].stat().st_size
                
            results['auto_transcript'] = {
                'status': 'success',
                'languages': auto_result.languages_found,
                'files_created': [f.name for f in auto_files],
                'txt_content_preview': auto_txt_content[:200] + "..." if len(auto_txt_content) > 200 else auto_txt_content,
                'word_count': auto_word_count,
                'file_sizes': auto_file_sizes,
                'processing_time': auto_processing_time,
                'methods_used': auto_result.methods_used
            }
            logger.info(f"✅ Auto-generated subtitles extracted successfully!")
            logger.info(f"   Languages: {auto_result.languages_found}")
            logger.info(f"   Files: {len(auto_files)} files created")
            logger.info(f"   Word count: {auto_word_count}")
            logger.info(f"   Processing time: {auto_processing_time:.1f}s")
            
        else:
            results['auto_transcript'] = {
                'status': 'failed',
                'error': 'Extraction failed',
                'processing_time': auto_processing_time
            }
            results['errors'].append("Auto extraction failed")
            logger.error("❌ Auto-generated subtitle extraction failed")
            
    except Exception as e:
        auto_processing_time = time.time() - auto_start_time
        results['auto_transcript'] = {
            'status': 'failed', 
            'error': str(e),
            'processing_time': auto_processing_time
        }
        results['errors'].append(f"Auto extraction error: {e}")
        logger.error(f"❌ Auto extraction error: {e}")
    
    # Step 2: Run Whisper transcription
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running Whisper transcription...")
    logger.info("=" * 60)
    logger.info("Note: This may take 5-8 minutes for a 14-minute video")
    
    whisper_start_time = time.time()
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
        whisper_processing_time = time.time() - whisper_start_time
        
        if whisper_result.get('status') == 'success':
            # Find created whisper files
            whisper_files = list(transcript_dir.glob("*_whisper.*"))
            whisper_txt_files = [f for f in whisper_files if f.suffix == '.txt']
            whisper_srt_files = [f for f in whisper_files if f.suffix == '.srt']
            
            whisper_txt_content = ""
            whisper_word_count = 0
            whisper_file_sizes = {}
            
            if whisper_txt_files:
                whisper_txt_content = whisper_txt_files[0].read_text(encoding='utf-8')
                whisper_word_count = len(whisper_txt_content.split())
                whisper_file_sizes['txt'] = whisper_txt_files[0].stat().st_size
                
            if whisper_srt_files:
                whisper_file_sizes['srt'] = whisper_srt_files[0].stat().st_size
            
            results['whisper_transcript'] = {
                'status': 'success',
                'language': whisper_result.get('language'),
                'files_created': [f.name for f in whisper_files],
                'txt_content_preview': whisper_txt_content[:200] + "..." if len(whisper_txt_content) > 200 else whisper_txt_content,
                'word_count': whisper_word_count,
                'file_sizes': whisper_file_sizes,
                'processing_time': whisper_processing_time,
                'extraction_method': whisper_result.get('extraction_method')
            }
            logger.info(f"✅ Whisper transcription completed successfully!")
            logger.info(f"   Language: {whisper_result.get('language')}")
            logger.info(f"   Files: {len(whisper_files)} files created")
            logger.info(f"   Word count: {whisper_word_count}")
            logger.info(f"   Processing time: {whisper_processing_time:.1f}s")
            
        else:
            results['whisper_transcript'] = {
                'status': 'failed',
                'error': whisper_result.get('error', 'Unknown error'),
                'processing_time': whisper_processing_time
            }
            results['errors'].append(f"Whisper failed: {whisper_result.get('error')}")
            logger.error(f"❌ Whisper transcription failed: {whisper_result.get('error')}")
            
    except Exception as e:
        whisper_processing_time = time.time() - whisper_start_time
        results['whisper_transcript'] = {
            'status': 'failed',
            'error': str(e),
            'processing_time': whisper_processing_time
        }
        results['errors'].append(f"Whisper error: {e}")
        logger.error(f"❌ Whisper error: {e}")
    
    total_processing_time = time.time() - total_start_time
    
    # Step 3: Compare results (if both succeeded)
    if (results['auto_transcript'] and results['auto_transcript']['status'] == 'success' and
        results['whisper_transcript'] and results['whisper_transcript']['status'] == 'success'):
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Comparing transcripts...")
        logger.info("=" * 60)
        
        auto_data = results['auto_transcript']
        whisper_data = results['whisper_transcript']
        
        comparison = {
            'word_count_difference': whisper_data['word_count'] - auto_data['word_count'],
            'word_count_ratio': whisper_data['word_count'] / auto_data['word_count'] if auto_data['word_count'] > 0 else 0,
            'file_size_comparison': {
                'txt_difference': whisper_data['file_sizes'].get('txt', 0) - auto_data['file_sizes'].get('txt', 0),
                'srt_difference': whisper_data['file_sizes'].get('srt', 0) - auto_data['file_sizes'].get('srt', 0)
            },
            'processing_time_comparison': {
                'auto_time': auto_data['processing_time'],
                'whisper_time': whisper_data['processing_time'],
                'whisper_slower_by': whisper_data['processing_time'] - auto_data['processing_time']
            }
        }
        
        results['comparison'] = comparison
        
        logger.info(f"📊 COMPARISON RESULTS:")
        logger.info(f"   Auto words: {auto_data['word_count']:,}")
        logger.info(f"   Whisper words: {whisper_data['word_count']:,}")
        logger.info(f"   Difference: {comparison['word_count_difference']:+,} ({(comparison['word_count_ratio']-1)*100:+.1f}%)")
        logger.info(f"   Auto processing: {auto_data['processing_time']:.1f}s")
        logger.info(f"   Whisper processing: {whisper_data['processing_time']:.1f}s")
        logger.info(f"   Whisper slower by: {comparison['processing_time_comparison']['whisper_slower_by']:.1f}s")
        
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processing time: {total_processing_time:.1f}s")
    logger.info(f"Auto-generated: {'✅ Success' if results['auto_transcript'] and results['auto_transcript']['status'] == 'success' else '❌ Failed'}")
    logger.info(f"Whisper: {'✅ Success' if results['whisper_transcript'] and results['whisper_transcript']['status'] == 'success' else '❌ Failed'}")
    
    if results['errors']:
        logger.info(f"Errors encountered: {len(results['errors'])}")
        for error in results['errors']:
            logger.info(f"  - {error}")
    
    # Save detailed results
    import json
    output_file = Path("tedx_dual_transcripts_test.json")
    results['total_processing_time'] = total_processing_time
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Detailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    test_tedx_dual_transcripts()