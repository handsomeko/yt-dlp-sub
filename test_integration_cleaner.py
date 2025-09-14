#!/usr/bin/env python3
"""
Test the integration of TranscriptCleaner with subtitle extraction
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor


def test_integration():
    """Test that the cleaner is properly integrated"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("TESTING TRANSCRIPT CLEANER INTEGRATION")
    print("="*80)
    
    # Test video URL - using one that definitely has auto-generated captions
    test_url = "https://www.youtube.com/watch?v=xKsJ7BXLs8I"  # MikeyNoCode video we tested earlier
    output_dir = Path("/tmp/test_cleaner_integration")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📺 Testing with video: {test_url}")
    print(f"📁 Output directory: {output_dir}")
    print("-" * 40)
    
    # Create extractor
    extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
    
    # Extract subtitles
    print("\n⏳ Extracting subtitles...")
    result = extractor.extract_subtitles(
        video_url=test_url,
        output_dir=output_dir,
        video_id="test_video",
        video_title="test_integration"
    )
    
    print(f"\n✅ Extraction complete!")
    print(f"Success: {result.success}")
    print(f"Languages found: {result.languages_found}")
    print(f"Original files: {result.original_files}")
    print(f"Methods used: {result.methods_used}")
    
    # Check if files were cleaned
    print("\n📋 Checking file formats:")
    print("-" * 40)
    
    for file_path in result.original_files:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for auto-generated artifacts
            has_xml_tags = '<c>' in content or '</c>' in content
            has_timing_tags = bool(re.search(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', content))
            has_metadata = 'Kind: captions' in content or 'Language: en' in content
            has_position = 'align:start' in content or 'position:' in content
            
            print(f"\n{file_path_obj.name}:")
            print(f"  Size: {len(content):,} bytes")
            print(f"  Has XML tags: {has_xml_tags}")
            print(f"  Has timing tags: {has_timing_tags}")
            print(f"  Has metadata headers: {has_metadata}")
            print(f"  Has position attributes: {has_position}")
            
            if file_path_obj.suffix == '.srt':
                # Check sequence numbering
                lines = content.split('\n')
                numbers = [int(line) for line in lines if line.strip().isdigit()][:10]
                if numbers:
                    is_sequential = numbers == list(range(1, len(numbers) + 1))
                    print(f"  First 10 sequence numbers: {numbers}")
                    print(f"  Sequential numbering: {is_sequential}")
            
            # Show preview
            if file_path_obj.suffix == '.txt':
                preview = content[:200]
                print(f"  Preview: {preview}...")
            elif file_path_obj.suffix == '.srt':
                blocks = content.split('\n\n')[:2]
                print(f"  First 2 blocks:")
                for block in blocks:
                    print(f"    {block.replace(chr(10), ' | ')}")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)
    
    # Clean up
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"\n🧹 Cleaned up test directory")


if __name__ == "__main__":
    import re  # Import needed for checking
    test_integration()