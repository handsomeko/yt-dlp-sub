#!/usr/bin/env python3
"""
Unit tests for Chinese punctuation restoration system.
Tests both synchronous and asynchronous implementations.
"""

import unittest
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.chinese_punctuation import ChinesePunctuationRestorer
from core.chinese_punctuation_sync import ChinesePunctuationSync


class TestChinesePunctuation(unittest.TestCase):
    """Test suite for Chinese punctuation restoration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sync_restorer = ChinesePunctuationSync()
        self.async_restorer = ChinesePunctuationRestorer(ai_backend=None)
        
    def test_chinese_detection(self):
        """Test Chinese text detection."""
        # Pure Chinese
        self.assertTrue(self.sync_restorer.detect_chinese_text("你好世界"))
        self.assertTrue(self.async_restorer.detect_chinese_text("你好世界"))
        
        # Mixed Chinese-English
        self.assertTrue(self.sync_restorer.detect_chinese_text("Hello 你好 World"))
        self.assertTrue(self.async_restorer.detect_chinese_text("Hello 你好 World"))
        
        # Pure English
        self.assertFalse(self.sync_restorer.detect_chinese_text("Hello World"))
        self.assertFalse(self.async_restorer.detect_chinese_text("Hello World"))
        
        # Numbers and symbols
        self.assertFalse(self.sync_restorer.detect_chinese_text("123 456"))
        self.assertFalse(self.async_restorer.detect_chinese_text("123 456"))
        
    def test_punctuation_detection(self):
        """Test punctuation detection for various text types."""
        # Chinese with Chinese punctuation
        self.assertTrue(self.sync_restorer.has_punctuation("你好，世界。"))
        self.assertTrue(self.async_restorer.has_punctuation("你好，世界。"))
        
        # Chinese without punctuation
        self.assertFalse(self.sync_restorer.has_punctuation("你好世界"))
        self.assertFalse(self.async_restorer.has_punctuation("你好世界"))
        
        # Chinese with enumeration mark only (should be false)
        self.assertFalse(self.sync_restorer.has_punctuation("苹果、香蕉、橙子"))
        self.assertFalse(self.async_restorer.has_punctuation("苹果、香蕉、橙子"))
        
        # Mixed content with English punctuation
        self.assertTrue(self.sync_restorer.has_punctuation("你好 world."))
        self.assertTrue(self.async_restorer.has_punctuation("你好 world."))
        
        # English with English punctuation
        self.assertTrue(self.sync_restorer.has_punctuation("Hello world."))
        self.assertTrue(self.async_restorer.has_punctuation("Hello world."))
        
        # English without punctuation
        self.assertFalse(self.sync_restorer.has_punctuation("Hello world"))
        self.assertFalse(self.async_restorer.has_punctuation("Hello world"))
        
    def test_text_chunking(self):
        """Test text chunking for long texts."""
        # Short text (no chunking needed)
        short_text = "你好世界" * 10  # 40 characters
        chunks = self.sync_restorer.chunk_text(short_text)
        self.assertEqual(len(chunks), 1)
        
        # Long text (needs chunking - updated for 500 char chunks)
        long_text = "你好世界" * 200  # 800 characters to ensure chunking
        chunks = self.sync_restorer.chunk_text(long_text)
        self.assertGreater(len(chunks), 1)
        
        # Verify chunk sizes (updated for new 500 char default)
        for chunk in chunks[:-1]:  # All but last chunk
            self.assertLessEqual(len(chunk), 500)
        
        # Verify no text is lost
        rejoined = ''.join(chunks)
        self.assertEqual(rejoined.replace(' ', ''), long_text.replace(' ', ''))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty text
        self.assertFalse(self.sync_restorer.detect_chinese_text(""))
        self.assertFalse(self.sync_restorer.has_punctuation(""))
        
        # Only spaces
        self.assertFalse(self.sync_restorer.detect_chinese_text("   "))
        self.assertFalse(self.sync_restorer.has_punctuation("   "))
        
        # Only punctuation
        self.assertFalse(self.sync_restorer.detect_chinese_text("。，！？"))
        self.assertTrue(self.sync_restorer.has_punctuation("。，！？"))
        
        # Special characters
        text_with_special = "你好@世界#测试"
        self.assertTrue(self.sync_restorer.detect_chinese_text(text_with_special))
        self.assertFalse(self.sync_restorer.has_punctuation(text_with_special))
        
    def test_claude_cli_check(self):
        """Test Claude CLI availability check."""
        # This test may pass or fail depending on system
        # Just verify the method runs without error
        result = self.sync_restorer.check_claude_cli()
        self.assertIsInstance(result, bool)
        
        # Second call should use cached result
        result2 = self.sync_restorer.check_claude_cli()
        self.assertEqual(result, result2)
        
    def test_sync_restoration_without_claude(self):
        """Test sync restoration when Claude CLI is unavailable."""
        # Mock Claude CLI as unavailable
        self.sync_restorer.claude_available = False
        
        text = "你好世界测试文本"
        restored, success = self.sync_restorer.restore_punctuation(text)
        
        # Should return original text and False
        self.assertEqual(restored, text)
        self.assertFalse(success)
        
    async def test_async_restoration_basic(self):
        """Test async restoration basic functionality."""
        # Test with already punctuated text
        punctuated = "你好，世界。"
        result, success = await self.async_restorer.restore_punctuation(punctuated)
        self.assertEqual(result, punctuated)
        self.assertFalse(success)  # No change needed
        
        # Test with non-Chinese text
        english = "Hello world"
        result, success = await self.async_restorer.restore_punctuation(english)
        self.assertEqual(result, english)
        self.assertFalse(success)  # Not Chinese
        
    def test_mixed_language_handling(self):
        """Test handling of mixed Chinese-English content."""
        # Mixed with Chinese punctuation
        mixed1 = "这是Chinese和English的混合文本。"
        self.assertTrue(self.sync_restorer.has_punctuation(mixed1))
        
        # Mixed with English punctuation
        mixed2 = "这是Chinese和English的混合文本."
        self.assertTrue(self.sync_restorer.has_punctuation(mixed2))
        
        # Mixed without punctuation
        mixed3 = "这是Chinese和English的混合文本"
        self.assertFalse(self.sync_restorer.has_punctuation(mixed3))
        
    def test_real_whisper_output(self):
        """Test with real Whisper output patterns."""
        # Whisper often outputs with timestamps
        whisper_text = "你好世界 这是测试"
        self.assertTrue(self.sync_restorer.detect_chinese_text(whisper_text))
        self.assertFalse(self.sync_restorer.has_punctuation(whisper_text))
        
        # Whisper with punctuation (newer versions)
        whisper_punct = "你好世界。这是测试。"
        self.assertTrue(self.sync_restorer.detect_chinese_text(whisper_punct))
        self.assertTrue(self.sync_restorer.has_punctuation(whisper_punct))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_sync_async_consistency(self):
        """Verify sync and async implementations behave consistently."""
        sync_restorer = ChinesePunctuationSync()
        async_restorer = ChinesePunctuationRestorer(ai_backend=None)
        
        test_cases = [
            "你好世界",
            "Hello world",
            "你好，世界。",
            "混合text测试",
            "苹果、香蕉、橙子",
            "",
        ]
        
        for text in test_cases:
            # Test detection
            self.assertEqual(
                sync_restorer.detect_chinese_text(text),
                async_restorer.detect_chinese_text(text),
                f"Detection mismatch for: {text}"
            )
            
            # Test punctuation check
            self.assertEqual(
                sync_restorer.has_punctuation(text),
                async_restorer.has_punctuation(text),
                f"Punctuation check mismatch for: {text}"
            )


def run_async_test(test_func):
    """Helper to run async test functions."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_func())
    finally:
        loop.close()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)