#!/usr/bin/env python3
"""
Test punctuation restoration functionality
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.punctuation_restorer import PunctuationRestorer, restore_transcript_punctuation

def test_chinese_punctuation():
    """Test Chinese punctuation restoration"""
    print("Testing Chinese punctuation restoration...")
    
    test_cases = [
        # (input, expected_output)
        (
            "我今天去了超市 买了很多东西 包括苹果香蕉和橙子",
            "我今天去了超市。买了很多东西。包括苹果香蕉和橙子。"
        ),
        (
            "这个方法很好用 你试试看吧 应该能解决问题的",
            "这个方法很好用。你试试看吧。应该能解决问题的。"
        ),
        (
            "他说他明天会来 但是我不确定 可能会有变化吧",
            "他说他明天会来。但是我不确定。可能会有变化吧。"
        ),
        (
            "中医认为这个很重要 要多注意休息 保持良好的生活习惯",
            "中医认为这个很重要。要多注意休息。保持良好的生活习惯。"
        )
    ]
    
    restorer = PunctuationRestorer(language='zh')
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = restorer.restore_punctuation(input_text)
        if result == expected:
            print(f"✅ PASS: {input_text[:30]}...")
            passed += 1
        else:
            print(f"❌ FAIL:")
            print(f"   Input:    {input_text}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
            failed += 1
    
    print(f"\nChinese tests: {passed} passed, {failed} failed")
    return failed == 0

def test_english_punctuation():
    """Test English punctuation restoration"""
    print("\nTesting English punctuation restoration...")
    
    test_cases = [
        (
            "hello world this is a test it should work fine",
            "Hello world this is a test. It should work fine."
        ),
        (
            "I went to the store today I bought many things",
            "I went to the store today. I bought many things."
        ),
        (
            "the meeting is tomorrow we should prepare well",
            "The meeting is tomorrow. We should prepare well."
        ),
    ]
    
    restorer = PunctuationRestorer(language='en')
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = restorer.restore_punctuation(input_text)
        if result == expected:
            print(f"✅ PASS: {input_text[:30]}...")
            passed += 1
        else:
            print(f"❌ FAIL:")
            print(f"   Input:    {input_text}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
            failed += 1
    
    print(f"\nEnglish tests: {passed} passed, {failed} failed")
    return failed == 0

def test_auto_language_detection():
    """Test automatic language detection"""
    print("\nTesting automatic language detection...")
    
    # Chinese text should be detected and restored with Chinese punctuation
    chinese_text = "这是中文文本 应该用中文标点符号"
    result = restore_transcript_punctuation(chinese_text)
    
    if '。' in result:
        print(f"✅ PASS: Chinese text detected and restored correctly")
        print(f"   Result: {result}")
        chinese_pass = True
    else:
        print(f"❌ FAIL: Chinese text not detected properly")
        print(f"   Result: {result}")
        chinese_pass = False
    
    # English text should be detected and restored with English punctuation
    english_text = "this is english text it should use english punctuation"
    result = restore_transcript_punctuation(english_text)
    
    if '.' in result and '。' not in result:
        print(f"✅ PASS: English text detected and restored correctly")
        print(f"   Result: {result}")
        english_pass = True
    else:
        print(f"❌ FAIL: English text not detected properly")
        print(f"   Result: {result}")
        english_pass = False
    
    return chinese_pass and english_pass

def test_punctuation_density():
    """Test punctuation density calculation"""
    print("\nTesting punctuation density calculation...")
    
    restorer = PunctuationRestorer(language='zh')
    
    test_cases = [
        ("这是一个测试。包含两个句子。", 2.0),  # 2 punctuation marks in ~10 chars = ~20/100
        ("没有标点符号的文本", 0.0),
        ("。。。", 100.0),  # All punctuation
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_min in test_cases:
        density = restorer._calculate_punctuation_density(text)
        # Use approximate comparison for density
        if (expected_min == 0 and density == 0) or (expected_min > 0 and density > 0):
            print(f"✅ PASS: Density for '{text[:20]}...' = {density:.2f}/100 chars")
            passed += 1
        else:
            print(f"❌ FAIL: Density for '{text}' = {density:.2f}, expected >= {expected_min}")
            failed += 1
    
    print(f"\nDensity tests: {passed} passed, {failed} failed")
    return failed == 0

def test_environment_variable():
    """Test environment variable control"""
    print("\nTesting environment variable control...")
    
    # Test with restoration enabled (default)
    os.environ['RESTORE_PUNCTUATION'] = 'true'
    from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
    extractor = LanguageAgnosticSubtitleExtractor()
    
    srt_content = """1
00:00:00,000 --> 00:00:05,000
这是第一句话

2
00:00:05,000 --> 00:00:10,000
这是第二句话

3
00:00:10,000 --> 00:00:15,000
这是第三句话"""
    
    txt_result = extractor._srt_to_txt(srt_content)
    
    if '。' in txt_result:
        print("✅ PASS: Punctuation restoration enabled by default")
        print(f"   Result: {txt_result}")
        enabled_pass = True
    else:
        print("❌ FAIL: Punctuation restoration not working when enabled")
        print(f"   Result: {txt_result}")
        enabled_pass = False
    
    # Test with restoration disabled
    os.environ['RESTORE_PUNCTUATION'] = 'false'
    
    # Need to reimport to get new environment setting
    import importlib
    import core.subtitle_extractor_v2
    importlib.reload(core.subtitle_extractor_v2)
    from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
    extractor = LanguageAgnosticSubtitleExtractor()
    
    txt_result = extractor._srt_to_txt(srt_content)
    
    if '。' not in txt_result:
        print("✅ PASS: Punctuation restoration can be disabled")
        print(f"   Result: {txt_result}")
        disabled_pass = True
    else:
        print("❌ FAIL: Punctuation restoration still active when disabled")
        print(f"   Result: {txt_result}")
        disabled_pass = False
    
    # Reset to default
    os.environ['RESTORE_PUNCTUATION'] = 'true'
    
    return enabled_pass and disabled_pass

def main():
    """Run all tests"""
    print("=" * 60)
    print("PUNCTUATION RESTORATION TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Run each test
    all_passed = test_chinese_punctuation() and all_passed
    all_passed = test_english_punctuation() and all_passed
    all_passed = test_auto_language_detection() and all_passed
    all_passed = test_punctuation_density() and all_passed
    all_passed = test_environment_variable() and all_passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
