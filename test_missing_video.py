#!/usr/bin/env python3
"""Test the new language-agnostic subtitle extractor on the video that was previously missing transcripts"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yt_dlp
from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor

def test_old_english_only_approach():
    """Test the old English-only approach that failed"""
    video_id = "oJsYHAJZlHU"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    print("=== Testing OLD English-only approach (should fail) ===")
    
    output_dir = Path("test_old_approach")
    output_dir.mkdir(exist_ok=True)
    
    # This mimics the old downloader approach: English-only subtitles
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'srt',
        'subtitleslangs': ['en'],  # OLD APPROACH: English only
        'outtmpl': str(output_dir / f'{video_id}.%(ext)s'),
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Check for subtitle files
        srt_files = list(output_dir.glob('*.srt'))
        print(f"📄 SRT files found: {len(srt_files)}")
        
        if srt_files:
            print("❌ Unexpected: Old approach found subtitles (this shouldn't happen)")
            for file in srt_files:
                print(f"   • {file.name}")
        else:
            print("✅ Expected: No English subtitles found (old approach failed as expected)")
            
        return len(srt_files) > 0
        
    except Exception as e:
        print(f"❌ Old approach failed with error: {e}")
        return False

def test_new_language_agnostic_approach():
    """Test the new language-agnostic approach"""
    video_id = "oJsYHAJZlHU" 
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_title = "每天浪費這個時辰，等同於在減壽！一天中最能增壽的時刻，太多人白白錯過了"
    
    print("\n=== Testing NEW Language-agnostic approach (should succeed) ===")
    
    output_dir = Path("test_new_approach")
    output_dir.mkdir(exist_ok=True)
    
    # Use the new language-agnostic extractor
    extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
    result = extractor.extract_subtitles(
        video_url=video_url,
        output_dir=output_dir,
        video_id=video_id,
        video_title=video_title
    )
    
    print(f"✅ Success: {result.success}")
    print(f"🌍 Languages found: {result.languages_found}")
    print(f"📄 Files created: {len(result.original_files)}")
    print(f"🔧 Methods used: {result.methods_used}")
    
    if result.original_files:
        for file in result.original_files:
            print(f"   • {Path(file).name}")
    
    if result.error_messages:
        print(f"⚠️  Errors encountered: {len(result.error_messages)}")
        for error in result.error_messages[:2]:  # Show first 2 errors
            print(f"   • {error}")
    
    return result.success

def main():
    print("🧪 Testing subtitle extraction: OLD vs NEW approach")
    print("=" * 60)
    print(f"📺 Video ID: oJsYHAJZlHU")
    print(f"🎯 This video previously had NO transcripts extracted")
    print()
    
    # Test old approach
    old_success = test_old_english_only_approach()
    
    # Test new approach
    new_success = test_new_language_agnostic_approach()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY:")
    print(f"   Old English-only approach: {'✅ Found subtitles' if old_success else '❌ No subtitles found'}")
    print(f"   New Language-agnostic approach: {'✅ Found subtitles' if new_success else '❌ No subtitles found'}")
    
    if not old_success and new_success:
        print("\n🎉 SUCCESS: New approach solved the missing subtitle problem!")
        print("   The video that previously had NO subtitles now extracts successfully.")
        print("   This proves the language-agnostic extractor works as intended.")
    elif old_success and new_success:
        print("\n🤔 Both approaches worked (unexpected for this video)")
    elif not old_success and not new_success:
        print("\n❌ Both approaches failed - need further investigation")
    else:
        print("\n🤨 Old approach worked but new didn't - investigation needed")

if __name__ == "__main__":
    main()