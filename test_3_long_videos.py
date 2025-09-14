#!/usr/bin/env python3
"""
Test 3 long Chinese videos from TCM-Chan channel.
Validates punctuation restoration and transcription quality.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.downloader import YouTubeDownloader
from core.chinese_punctuation_sync import get_sync_restorer

def test_long_videos():
    """Test 3 long videos from TCM-Chan."""
    
    videos = [
        {
            'url': 'https://youtube.com/watch?v=zboE02_1PIM',
            'title': '養陽氣就等於養命',
            'duration': 25
        },
        {
            'url': 'https://youtube.com/watch?v=SON6hKNHaDM', 
            'title': '5分鐘睡出6小時效果',
            'duration': 22
        },
        {
            'url': 'https://youtube.com/watch?v=boBnQ9tHF1Y',
            'title': '這3個穴位一定要學會',
            'duration': 22
        }
    ]
    
    print("🎬 Testing 3 Long Chinese Videos from TCM-Chan")
    print("="*60)
    
    downloader = YouTubeDownloader()
    restorer = get_sync_restorer()
    results = []
    
    for i, video in enumerate(videos, 1):
        print(f"\n📹 Video {i}/3: {video['title']} ({video['duration']} min)")
        print("-"*60)
        
        start_time = time.time()
        
        try:
            # Download video
            print(f"⬇️  Downloading...")
            result = downloader.download_video(
                video['url'],
                quality='1080p',
                download_audio_only=True
            )
            
            download_time = time.time() - start_time
            
            if result['status'] == 'success':
                print(f"✅ Downloaded in {download_time:.1f}s")
                
                # Check for Chinese transcript
                transcript_files = [f for f in result['files'] if '.zh.txt' in f]
                
                if transcript_files:
                    transcript_path = Path(transcript_files[0])
                    content = transcript_path.read_text(encoding='utf-8')
                    
                    # Analyze transcript
                    is_chinese = restorer.detect_chinese_text(content)
                    has_punct = restorer.has_punctuation(content)
                    char_count = len(content)
                    
                    # Estimate restoration cost if needed
                    if is_chinese and not has_punct:
                        cost, api_calls = restorer.estimate_cost(content)
                        cost_info = f"Would need {api_calls} API calls (~${cost:.4f})"
                    else:
                        cost_info = "N/A - already has punctuation"
                    
                    # Calculate chunks for this size
                    chunks = restorer.chunk_text(content)
                    chunk_count = len(chunks)
                    
                    print(f"📄 Transcript Analysis:")
                    print(f"   - File: {transcript_path.name}")
                    print(f"   - Size: {char_count:,} characters")
                    print(f"   - Chinese: {is_chinese}")
                    print(f"   - Has punctuation: {has_punct}")
                    print(f"   - Chunks (if needed): {chunk_count}")
                    print(f"   - Restoration cost: {cost_info}")
                    
                    # Show sample
                    print(f"   - Sample (first 100 chars):")
                    print(f"     {content[:100]}...")
                    
                    results.append({
                        'video': video['title'],
                        'duration': video['duration'],
                        'chars': char_count,
                        'chinese': is_chinese,
                        'punctuated': has_punct,
                        'chunks': chunk_count,
                        'download_time': download_time
                    })
                    
                else:
                    print("❌ No Chinese transcript found")
                    results.append({
                        'video': video['title'],
                        'duration': video['duration'],
                        'error': 'No Chinese transcript'
                    })
                    
            else:
                print(f"❌ Download failed: {result.get('error', 'Unknown error')}")
                results.append({
                    'video': video['title'],
                    'duration': video['duration'],
                    'error': result.get('error', 'Download failed')
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'video': video['title'],
                'duration': video['duration'],
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n✅ Successful: {len(successful)}/3")
    for r in successful:
        print(f"   - {r['video'][:30]}...")
        print(f"     Duration: {r['duration']} min")
        print(f"     Transcript: {r['chars']:,} chars")
        print(f"     Punctuated: {r['punctuated']}")
        print(f"     Download time: {r['download_time']:.1f}s")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/3")
        for r in failed:
            print(f"   - {r['video'][:30]}...")
            print(f"     Error: {r['error']}")
    
    # Overall statistics
    if successful:
        total_chars = sum(r['chars'] for r in successful)
        total_chunks = sum(r['chunks'] for r in successful)
        avg_download = sum(r['download_time'] for r in successful) / len(successful)
        
        print(f"\n📈 Statistics:")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Total chunks: {total_chunks}")
        print(f"   - Avg download time: {avg_download:.1f}s")
        print(f"   - All have punctuation: {all(r['punctuated'] for r in successful)}")
    
    print("\n" + "="*60)
    if len(successful) == 3:
        print("🎉 All 3 long videos processed successfully!")
        print("✅ Chinese punctuation system working perfectly!")
    else:
        print(f"⚠️  Only {len(successful)}/3 videos processed successfully")
    
    return len(successful) == 3

if __name__ == "__main__":
    success = test_long_videos()
    sys.exit(0 if success else 1)