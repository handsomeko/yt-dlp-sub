#!/usr/bin/env python3
"""
Simple test for punctuation restoration functionality
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.punctuation_restorer import restore_transcript_punctuation

def main():
    """Run simple tests"""
    print("=" * 60)
    print("SIMPLE PUNCTUATION RESTORATION TEST")
    print("=" * 60)
    
    # Test Chinese text
    chinese_text = "我今天去了超市 买了很多东西 他说他明天会来 但是我不确定"
    print(f"\nOriginal Chinese: {chinese_text}")
    
    restored_chinese = restore_transcript_punctuation(chinese_text)
    print(f"Restored Chinese: {restored_chinese}")
    
    # Check if any Chinese punctuation was added
    if any(p in restored_chinese for p in ['。', '！', '？']):
        print("✅ Chinese punctuation restoration working")
    else:
        print("❌ Chinese punctuation restoration not working")
    
    # Test English text  
    english_text = "hello world this is a test it should work fine the meeting is tomorrow"
    print(f"\nOriginal English: {english_text}")
    
    restored_english = restore_transcript_punctuation(english_text)
    print(f"Restored English: {restored_english}")
    
    # Check if any English punctuation was added
    if any(p in restored_english for p in ['.', '!', '?']):
        print("✅ English punctuation restoration working")
    else:
        print("❌ English punctuation restoration not working")
    
    # Test that it's integrated in subtitle extractor
    print("\n" + "=" * 60)
    print("Testing integration with subtitle_extractor_v2...")
    
    os.environ['RESTORE_PUNCTUATION'] = 'true'
    from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
    
    extractor = LanguageAgnosticSubtitleExtractor()
    
    # Test SRT with Chinese content
    srt_content = """1
00:00:00,000 --> 00:00:05,000
我今天去了超市

2
00:00:05,000 --> 00:00:10,000
买了很多东西

3
00:00:10,000 --> 00:00:15,000
他说他明天会来"""
    
    txt_result = extractor._srt_to_txt(srt_content)
    print(f"SRT to TXT result: {txt_result}")
    
    if '。' in txt_result or '!' in txt_result or '？' in txt_result:
        print("✅ Punctuation restoration integrated in subtitle_extractor_v2")
    else:
        print("⚠️  Punctuation restoration may not be fully integrated")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Basic functionality verified")
    print("=" * 60)

if __name__ == '__main__':
    main()
