#!/usr/bin/env python3
"""Test the new language-agnostic subtitle extractor"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor

def test_problematic_video():
    """Test subtitle extraction on the video that previously had no subtitles"""
    
    # Video that was problematic: oJsYHAJZlHU
    video_url = "https://www.youtube.com/watch?v=oJsYHAJZlHU"
    video_id = "oJsYHAJZlHU"
    video_title = "每天浪費這個時辰，等同於在減壽！一天中最能增壽的時刻，太多人白白錯過了"
    
    # Create temporary output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🧪 Testing subtitle extraction on problematic video: {video_id}")
    print(f"📺 Video title: {video_title}")
    print(f"🔗 URL: {video_url}")
    print()
    
    # Test without translation first
    print("=== Test 1: Extract subtitles without translation ===")
    extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
    result = extractor.extract_subtitles(
        video_url=video_url,
        output_dir=output_dir,
        video_id=video_id,
        video_title=video_title
    )
    
    print(f"✅ Success: {result.success}")
    print(f"🌍 Languages found: {result.languages_found}")
    print(f"📄 Original files: {len(result.original_files)}")
    if result.original_files:
        for file in result.original_files:
            print(f"   • {file}")
    print(f"🔧 Methods used: {result.methods_used}")
    if result.error_messages:
        print(f"⚠️  Errors: {result.error_messages}")
    print()
    
    # Test with translation if we got subtitles
    if result.success and result.original_files:
        print("=== Test 2: Extract subtitles with translation enabled ===")
        extractor_with_translation = LanguageAgnosticSubtitleExtractor(
            translate_enabled=True,
            target_language='en'
        )
        result_translated = extractor_with_translation.extract_subtitles(
            video_url=video_url,
            output_dir=output_dir,
            video_id=video_id,
            video_title=video_title + "_translated"
        )
        
        print(f"✅ Success: {result_translated.success}")
        print(f"🌍 Languages found: {result_translated.languages_found}")
        print(f"📄 Original files: {len(result_translated.original_files)}")
        print(f"🌐 Translated files: {len(result_translated.translated_files)}")
        if result_translated.translated_files:
            for file in result_translated.translated_files:
                print(f"   • {file}")
        if result_translated.error_messages:
            print(f"⚠️  Translation errors: {result_translated.error_messages}")
    
    print("\n=== File Contents Preview ===")
    for file_path in output_dir.glob("*.txt"):
        print(f"\n📄 {file_path.name}:")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()[:500]  # First 500 characters
            print(content)
            if len(content) >= 500:
                print("... (truncated)")
    
    return result

if __name__ == "__main__":
    result = test_problematic_video()
    
    if result.success:
        print("\n🎉 SUCCESS: Language-agnostic subtitle extraction worked!")
        print("This video now has subtitles extracted, solving the original problem.")
    else:
        print("\n❌ FAILED: Still unable to extract subtitles")
        print("Need further investigation of the extraction methods.")