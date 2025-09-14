#!/usr/bin/env python3
"""
Test Chinese subtitle extraction fix on multiple videos.
Verifies that we get Chinese subtitles, not English translations.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.downloader import YouTubeDownloader

def test_chinese_videos():
    """Test multiple Chinese videos to verify the fix."""
    
    # Test videos from TCM-Chan
    test_videos = [
        {
            'url': 'https://www.youtube.com/watch?v=ADZLFh-LZbA',
            'title': '喉嚨有痰總咳不出來',
            'expected': 'Should get Chinese .zh subtitles'
        },
        {
            'url': 'https://www.youtube.com/watch?v=Ok9KSaqbN-A', 
            'title': '脖子、臉上的脂肪粒',
            'expected': 'Should get Chinese .zh subtitles or Whisper'
        },
        {
            'url': 'https://www.youtube.com/watch?v=SON6hKNHaDM',
            'title': '5分鐘睡出6小時效果',
            'expected': 'May need Whisper transcription'
        }
    ]
    
    print("🔍 Testing Chinese Subtitle Fix")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    downloader = YouTubeDownloader()
    results = []
    
    for i, video in enumerate(test_videos, 1):
        print(f"📹 Test {i}/3: {video['title']}")
        print(f"   URL: {video['url']}")
        print(f"   Expected: {video['expected']}")
        print("-"*60)
        
        try:
            result = downloader.download_subtitles_only(video['url'])
            
            if result['status'] == 'success':
                print("✅ Download successful")
                
                # Analyze files
                zh_files = []
                en_files = []
                
                for f in result.get('files', []):
                    if 'transcripts' in f:
                        name = f.split('/')[-1]
                        if '.zh.' in name:
                            zh_files.append(name)
                        elif '.en.' in name:
                            en_files.append(name)
                
                # Check content of files
                if zh_files:
                    print(f"✅ Found {len(zh_files)} Chinese files:")
                    for f in zh_files:
                        print(f"   - {f}")
                        
                        # Check content preview
                        full_path = None
                        for fp in result.get('files', []):
                            if f in fp and '.txt' in f:
                                full_path = fp
                                break
                        
                        if full_path and Path(full_path).exists():
                            with open(full_path, 'r', encoding='utf-8') as file:
                                preview = file.read(100)
                                # Check if it's actually Chinese
                                import re
                                if re.search(r'[\u4e00-\u9fff]', preview):
                                    print(f"     ✅ Confirmed Chinese content: {preview[:30]}...")
                                else:
                                    print(f"     ❌ NOT Chinese content: {preview[:30]}...")
                
                if en_files:
                    print(f"ℹ️  Found {len(en_files)} English files:")
                    for f in en_files:
                        print(f"   - {f}")
                
                if not zh_files and not en_files:
                    print("❌ No subtitle files found")
                
                # Record result
                results.append({
                    'video': video['title'],
                    'success': True,
                    'zh_files': len(zh_files),
                    'en_files': len(en_files),
                    'has_chinese': len(zh_files) > 0
                })
                
            else:
                print(f"❌ Download failed: {result.get('error', 'Unknown error')}")
                results.append({
                    'video': video['title'],
                    'success': False,
                    'error': result.get('error', 'Unknown')
                })
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'video': video['title'],
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('success')]
    with_chinese = [r for r in successful if r.get('has_chinese')]
    
    print(f"Total tested: {len(results)}")
    print(f"Successful downloads: {len(successful)}/{len(results)}")
    print(f"With Chinese subtitles: {len(with_chinese)}/{len(successful)}")
    
    print("\nDetails:")
    for r in results:
        if r.get('success'):
            status = "✅" if r.get('has_chinese') else "⚠️"
            print(f"{status} {r['video']}: {r['zh_files']} Chinese, {r['en_files']} English files")
        else:
            print(f"❌ {r['video']}: {r.get('error', 'Failed')}")
    
    print("\n" + "="*60)
    if len(with_chinese) == len(successful):
        print("🎉 SUCCESS: All videos have Chinese subtitles!")
    elif len(with_chinese) > 0:
        print(f"⚠️  PARTIAL: {len(with_chinese)}/{len(successful)} videos have Chinese subtitles")
    else:
        print("❌ FAILURE: No videos have Chinese subtitles")
    
    return len(with_chinese) > 0

if __name__ == "__main__":
    success = test_chinese_videos()
    sys.exit(0 if success else 1)