#!/usr/bin/env python3
"""
Test script for TranscriptCleaner

Tests cleaning of problematic auto-generated transcript files.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcript_cleaner import TranscriptCleaner


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_cleaner():
    """Test the TranscriptCleaner on problematic files"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    cleaner = TranscriptCleaner()
    
    # Define test files
    test_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/transcript_comparison_samples")
    
    test_files = [
        ("05_TEDx_AUTO.srt", "05_TEDx_AUTO_CLEANED.srt"),
        ("06_TEDx_AUTO.txt", "06_TEDx_AUTO_CLEANED.txt"),
        ("01_MikeyNoCode_AUTO.srt", "01_MikeyNoCode_AUTO_CLEANED.srt"),
        ("02_MikeyNoCode_AUTO.txt", "02_MikeyNoCode_AUTO_CLEANED.txt"),
    ]
    
    print("\n" + "="*80)
    print("TRANSCRIPT CLEANER TEST")
    print("="*80)
    
    for input_file, output_file in test_files:
        input_path = test_dir / input_file
        output_path = test_dir / output_file
        
        if not input_path.exists():
            print(f"\n❌ File not found: {input_path}")
            continue
        
        print(f"\n📄 Testing: {input_file}")
        print("-" * 40)
        
        # Read original content
        with open(input_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Check if auto-generated
        is_auto = cleaner.is_auto_generated(original_content)
        print(f"Auto-generated detected: {is_auto}")
        
        # Get file sizes
        original_size = len(original_content)
        print(f"Original size: {original_size:,} bytes")
        
        # Clean the file
        if input_file.endswith('.srt'):
            cleaned_content = cleaner.clean_auto_srt(original_content)
            
            # Validate SRT format
            is_valid, error = cleaner.validate_srt_format(cleaned_content)
            print(f"Valid SRT format: {is_valid}")
            if error:
                print(f"  Error: {error}")
        else:  # .txt file
            cleaned_content = cleaner.clean_auto_txt(original_content)
        
        # Save cleaned version
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        cleaned_size = len(cleaned_content)
        reduction = ((original_size - cleaned_size) / original_size * 100) if original_size > 0 else 0
        
        print(f"Cleaned size: {cleaned_size:,} bytes")
        print(f"Size reduction: {reduction:.1f}%")
        print(f"✅ Saved to: {output_file}")
        
        # Show sample of cleaned content
        if cleaned_content:
            lines = cleaned_content.split('\n')
            # For SRT, show first few blocks, for TXT show preview
            if input_file.endswith('.srt'):
                blocks = cleaned_content.strip().split('\n\n')
                preview_blocks = min(3, len(blocks))
                print(f"\nFirst {preview_blocks} subtitle blocks of cleaned content:")
                print("-" * 40)
                for block in blocks[:preview_blocks]:
                    print(f"  {block.replace(chr(10), chr(10) + '  ')}")
                    print()
            else:
                # For TXT files, show first 200 chars
                preview = cleaned_content[:200]
                print(f"\nFirst 200 characters of cleaned content:")
                print("-" * 40)
                print(f"  {preview}...")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    # Test validation on Whisper files (should already be clean)
    print("\n📋 Testing Whisper files (should be clean):")
    print("-" * 40)
    
    whisper_files = [
        "07_TEDx_WHISPER.srt",
        "03_MikeyNoCode_WHISPER.srt"
    ]
    
    for file_name in whisper_files:
        file_path = test_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            is_auto = cleaner.is_auto_generated(content)
            is_valid, error = cleaner.validate_srt_format(content)
            
            print(f"{file_name}:")
            print(f"  Auto-generated: {is_auto} (should be False)")
            print(f"  Valid format: {is_valid} (should be True)")
            if error:
                print(f"  Error: {error}")


if __name__ == "__main__":
    test_cleaner()