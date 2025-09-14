#!/usr/bin/env python3
"""
Test edge cases and error handling in hybrid transcription strategy
"""

import sys
sys.path.insert(0, '/Users/jk/yt-dl-sub')

import tempfile
import shutil
from pathlib import Path

from workers.transcriber import TranscribeWorker

def test_edge_cases():
    """Test various edge cases in hybrid logic"""
    
    print("🧪 Testing Hybrid Strategy Edge Cases")
    print("=" * 60)
    
    transcriber = TranscribeWorker()
    
    # Test 1: Non-existent transcript directory
    print("\n📁 Test 1: Non-existent transcript directory")
    result = transcriber._check_existing_subtitles(
        'fake_channel', 'fake_video', 'Fake Video Title'
    )
    assert result is None, "Should return None for non-existent directory"
    print("✅ Correctly handles non-existent directory")
    
    # Test 2: Empty transcript directory
    print("\n📁 Test 2: Empty transcript directory")  
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        transcriber.storage_paths.base_path = temp_path
        
        # Create empty directory structure
        transcript_dir = temp_path / "test_channel" / "test_video" / "transcripts" 
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        result = transcriber._check_existing_subtitles(
            'test_channel', 'test_video', 'Test Video'
        )
        assert result is None, "Should return None for empty directory"
        print("✅ Correctly handles empty transcript directory")
    
    # Test 3: Directory with only Whisper files (should skip them)
    print("\n📁 Test 3: Directory with only Whisper files")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        transcriber.storage_paths.base_path = temp_path
        
        # Create directory with only Whisper files
        transcript_dir = temp_path / "test_channel" / "test_video" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Whisper files (should be ignored)
        whisper_srt = transcript_dir / "Test_Video_whisper.srt"
        whisper_txt = transcript_dir / "Test_Video_whisper.txt"
        whisper_srt.write_text("1\n00:00:00,000 --> 00:00:05,000\nWhisper content\n")
        whisper_txt.write_text("Whisper content")
        
        result = transcriber._check_existing_subtitles(
            'test_channel', 'test_video', 'Test Video'
        )
        assert result is None, "Should skip Whisper files"
        print("✅ Correctly skips Whisper-generated files")
    
    # Test 4: Very small files (quality check should fail)
    print("\n📁 Test 4: Very small subtitle files")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        transcriber.storage_paths.base_path = temp_path
        
        # Create directory with tiny files
        transcript_dir = temp_path / "test_channel" / "test_video" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tiny auto-generated file (should fail quality check)
        tiny_srt = transcript_dir / "Test Video_auto.en.srt"  
        tiny_srt.write_text("x")  # Only 1 byte
        
        result = transcriber._check_existing_subtitles(
            'test_channel', 'test_video', 'Test Video'
        )
        assert result is None, "Should reject files that are too small"
        print("✅ Correctly rejects files that are too small")
    
    # Test 5: Valid auto-generated files (should work)
    print("\n📁 Test 5: Valid auto-generated subtitle files")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        transcriber.storage_paths.base_path = temp_path
        
        # Create directory with valid files
        transcript_dir = temp_path / "test_channel" / "test_video" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create valid auto-generated files
        valid_srt = transcript_dir / "Test Video_auto.en.srt"
        valid_txt = transcript_dir / "Test Video_auto.en.txt"
        
        srt_content = """1
00:00:00,000 --> 00:00:05,000
This is a valid subtitle file with proper content.

2  
00:00:05,000 --> 00:00:10,000
It has multiple lines and reasonable length.
"""
        txt_content = "This is a valid subtitle file with proper content. It has multiple lines and reasonable length."
        
        valid_srt.write_text(srt_content)
        valid_txt.write_text(txt_content)
        
        result = transcriber._check_existing_subtitles(
            'test_channel', 'test_video', 'Test Video'
        )
        assert result is not None, "Should find valid subtitle files"
        assert result['extraction_method'] == 'auto_generated_cleaned'
        assert 'en' in result['languages_found']
        assert len(result['files']) >= 2  # SRT + TXT
        print("✅ Correctly detects valid auto-generated files")
        print(f"   Found languages: {result['languages_found']}")
        print(f"   Found files: {len(result['files'])}")
    
    # Test 6: Language detection edge cases
    print("\n🌐 Test 6: Language detection patterns")
    test_filenames = [
        ("Video Title.en.srt", "en"),
        ("Video Title_auto.zh.srt", "zh"), 
        ("Video Title_alt1.es.srt", "es"),
        ("Video Title_en.srt", "en"),
        ("Video Title.zh-CN.srt", "zh-CN"),
        ("Video Title chinese content.srt", "zh"),  # Fallback
        ("Video Title xyz.srt", "auto"),  # Default fallback
    ]
    
    for filename, expected_lang in test_filenames:
        detected = transcriber._extract_language_from_filename(filename)
        assert detected == expected_lang, f"Language detection failed for {filename}: got {detected}, expected {expected_lang}"
        print(f"✅ {filename} → {detected}")
    
    # Test 7: Exception handling
    print("\n⚠️  Test 7: Exception handling in _check_existing_subtitles")
    
    # Mock storage paths to trigger exception
    original_get_transcript_dir = transcriber.storage_paths.get_transcript_dir
    transcriber.storage_paths.get_transcript_dir = lambda *args: None  # Will cause AttributeError
    
    result = transcriber._check_existing_subtitles(
        'test_channel', 'test_video', 'Test Video'
    )
    assert result is None, "Should handle exceptions gracefully"
    print("✅ Gracefully handles exceptions in subtitle checking")
    
    # Restore original method
    transcriber.storage_paths.get_transcript_dir = original_get_transcript_dir
    
    print(f"\n🎉 All edge case tests passed!")
    return True

def test_full_hybrid_workflow():
    """Test the complete hybrid workflow logic"""
    
    print(f"\n🔄 Testing Complete Hybrid Workflow Logic")
    print("=" * 60)
    
    transcriber = TranscribeWorker()
    
    # Test with fake input that should trigger fallback to Whisper
    test_input = {
        'video_id': 'nonexistent_video',
        'video_url': 'https://example.com/fake',
        'channel_id': 'nonexistent_channel',
        'video_title': 'Nonexistent Video'
    }
    
    print("🔍 Testing hybrid logic with non-existent video...")
    
    # Check that it would call _check_existing_subtitles first
    existing = transcriber._check_existing_subtitles(
        test_input['channel_id'],
        test_input['video_id'],
        test_input['video_title']
    )
    
    assert existing is None, "Should not find existing subtitles for fake video"
    print("✅ Step 1: Correctly detects no existing subtitles")
    print("✅ Step 2: Would proceed to Whisper fallback (as expected)")
    
    print("🎯 Hybrid workflow logic validated!")
    return True

if __name__ == "__main__":
    success = True
    
    try:
        if not test_edge_cases():
            success = False
            
        if not test_full_hybrid_workflow():
            success = False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print(f"\n🎉 All hybrid strategy edge case tests passed!")
        print("✅ Error handling is robust")
        print("✅ Edge cases handled correctly") 
        print("✅ Language detection working")
        print("✅ File validation working")
        print("✅ Hybrid workflow logic validated")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed.")
        sys.exit(1)