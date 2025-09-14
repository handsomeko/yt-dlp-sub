#!/usr/bin/env python3
"""
Test V2 Storage Structure with Comprehensive Metadata Capture

This script tests the complete V2 implementation:
1. Correct file naming (no video_id suffixes)
2. Channel-level metadata files
3. Comprehensive metadata capture (~60 fields)
4. Markdown report generation
5. Opus audio with MP3 conversion
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.downloader import YouTubeDownloader

def verify_v2_structure(base_path, channel_id, video_id, video_title):
    """Verify all V2 storage structure requirements are met"""
    
    print("\n" + "="*60)
    print("VERIFYING V2 STORAGE STRUCTURE")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Sanitize title like the downloader does
    sanitized_title = video_title
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        sanitized_title = sanitized_title.replace(char, '_')
    sanitized_title = sanitized_title[:200]
    
    # Check channel-level files
    channel_dir = Path(base_path) / channel_id
    print(f"\n📁 Checking channel directory: {channel_dir}")
    
    # Required channel files
    channel_files = {
        '.channel_info.json': 'Channel metadata',
        '.video_index.json': 'Video index',
        f'{sanitized_title}_channel_info.json': 'Comprehensive metadata',
        f'{sanitized_title}_channel_info.md': 'Markdown report'
    }
    
    for file_name, description in channel_files.items():
        file_path = channel_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}: {description}")
            
            # Check JSON validity and content
            if file_name.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Check comprehensive metadata fields
                    if 'channel_info.json' in file_name and file_name != '.channel_info.json':
                        field_count = count_fields(data)
                        print(f"     📊 Total fields captured: {field_count}")
                        if field_count < 30:
                            warnings.append(f"Only {field_count} fields in comprehensive metadata (expected ~60)")
                        else:
                            print(f"     ✨ Comprehensive capture successful!")
                            
                        # Check key sections
                        sections = ['channel', 'metrics', 'technical', 'urls', 'tags', 'chapters']
                        for section in sections:
                            if section in data:
                                print(f"     ✓ {section.capitalize()} section present")
                                
                except Exception as e:
                    errors.append(f"Invalid JSON in {file_name}: {e}")
        else:
            errors.append(f"Missing {file_name}: {description}")
    
    # Check video directory structure
    video_dir = channel_dir / video_id
    print(f"\n📁 Checking video directory: {video_dir}")
    
    if not video_dir.exists():
        errors.append(f"Video directory does not exist: {video_dir}")
        return errors, warnings
    
    # Check video-level tracking files
    tracking_files = {
        '.metadata.json': 'Video tracking metadata',
        '.processing_complete': 'Processing marker'
    }
    
    for file_name, description in tracking_files.items():
        file_path = video_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}: {description}")
        else:
            errors.append(f"Missing {file_name}: {description}")
    
    # Check subdirectories
    subdirs = ['media', 'transcripts', 'content', 'metadata']
    for subdir in subdirs:
        subdir_path = video_dir / subdir
        if subdir_path.exists():
            print(f"  ✅ {subdir}/ directory exists")
            
            # Check files in each subdirectory
            if subdir == 'media':
                expected_files = [
                    (f'{sanitized_title}.opus', True),  # Required
                    (f'{sanitized_title}.mp3', False),  # Optional but should exist
                    (f'{sanitized_title}.mp4', False)   # Optional
                ]
                
                for file_name, required in expected_files:
                    file_path = subdir_path / file_name
                    if file_path.exists():
                        print(f"     ✅ {file_name}")
                        
                        # Check for video_id suffix (should NOT exist)
                        bad_name = f'{sanitized_title}_{video_id}.{file_name.split(".")[-1]}'
                        bad_path = subdir_path / bad_name
                        if bad_path.exists():
                            errors.append(f"Found file with video_id suffix: {bad_name} (should be {file_name})")
                    elif required:
                        errors.append(f"Missing required file: {subdir}/{file_name}")
                    else:
                        print(f"     ⏭️  {file_name} (optional)")
            
            elif subdir == 'metadata':
                metadata_file = subdir_path / f'{sanitized_title}_metadata.json'
                if metadata_file.exists():
                    print(f"     ✅ {sanitized_title}_metadata.json")
                else:
                    # Check for old naming
                    old_file = subdir_path / 'metadata.json'
                    if old_file.exists():
                        warnings.append(f"Found old naming: metadata.json (should be {sanitized_title}_metadata.json)")
        else:
            errors.append(f"Missing subdirectory: {subdir}/")
    
    return errors, warnings

def count_fields(obj, depth=0):
    """Recursively count all fields in a nested dictionary"""
    if depth > 10:  # Prevent infinite recursion
        return 0
        
    count = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            count += 1
            if isinstance(value, dict):
                count += count_fields(value, depth+1)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    count += count_fields(value[0], depth+1)
    return count

def test_download():
    """Test downloading a short video with V2 structure"""
    
    print("\n" + "="*60)
    print("TESTING V2 COMPREHENSIVE DOWNLOAD")
    print("="*60)
    
    # Test video (short for quick testing)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video (19 seconds)
    
    print(f"\n🎬 Test video: {test_url}")
    print("   (First YouTube video - 19 seconds)")
    
    # Initialize downloader
    downloader = YouTubeDownloader()
    
    # Download with audio only (default settings)
    print("\n📥 Starting download with V2 structure...")
    try:
        result = downloader.download_video(
            url=test_url,
            download_audio_only=True,  # Default: Opus audio
            audio_format='opus'         # Explicit Opus format
        )
        
        if result.get('status') == 'success':
            print(f"\n✅ Download successful!")
            print(f"📁 Files saved to: {result['output_dir']}")
            
            # Extract info for verification
            channel_id = result.get('channel_id', 'unknown')
            video_id = result.get('video_id', 'unknown')
            video_title = result.get('title', 'unknown')
            
            # Verify V2 structure
            base_path = Path(result['output_dir']).parent.parent  # Go up from video_id/channel_id to base
            errors, warnings = verify_v2_structure(
                base_path,
                channel_id,
                video_id,
                video_title
            )
            
            # Report results
            print("\n" + "="*60)
            print("VERIFICATION RESULTS")
            print("="*60)
            
            if errors:
                print(f"\n❌ Found {len(errors)} errors:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print("\n✅ All V2 requirements met!")
            
            if warnings:
                print(f"\n⚠️  Found {len(warnings)} warnings:")
                for warning in warnings:
                    print(f"   - {warning}")
            
            # Display sample of comprehensive metadata
            channel_dir = Path(result['output_dir']).parent  # Go up from video_id to channel_id
            sanitized_title = video_title
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                sanitized_title = sanitized_title.replace(char, '_')
            sanitized_title = sanitized_title[:200]
            
            comprehensive_file = channel_dir / f'{sanitized_title}_channel_info.json'
            if comprehensive_file.exists():
                print("\n" + "="*60)
                print("SAMPLE COMPREHENSIVE METADATA")
                print("="*60)
                
                with open(comprehensive_file, 'r') as f:
                    data = json.load(f)
                
                # Show key sections
                if 'channel' in data:
                    print("\n📺 Channel Info:")
                    for key, value in list(data['channel'].items())[:5]:
                        print(f"   {key}: {value}")
                
                if 'metrics' in data:
                    print("\n📊 Metrics:")
                    for key, value in list(data['metrics'].items())[:5]:
                        print(f"   {key}: {value}")
                
                if 'technical' in data:
                    print("\n⚙️  Technical:")
                    for key, value in list(data['technical'].items())[:5]:
                        print(f"   {key}: {value}")
            
            # Check if markdown report exists
            markdown_file = channel_dir / f'{sanitized_title}_channel_info.md'
            if markdown_file.exists():
                print("\n" + "="*60)
                print("MARKDOWN REPORT PREVIEW")
                print("="*60)
                
                with open(markdown_file, 'r') as f:
                    lines = f.readlines()[:20]  # First 20 lines
                    for line in lines:
                        print(line.rstrip())
                print("... [report continues]")
            
            return not bool(errors)  # Return True if no errors
            
        else:
            print(f"\n❌ Download failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("V2 STORAGE STRUCTURE COMPREHENSIVE TEST")
    print("="*60)
    print("\nThis test will:")
    print("1. Download a short test video")
    print("2. Verify V2 storage structure")
    print("3. Check comprehensive metadata capture (~60 fields)")
    print("4. Verify markdown report generation")
    print("5. Confirm no video_id suffixes in filenames")
    
    success = test_download()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED - V2 IMPLEMENTATION COMPLETE!")
    else:
        print("❌ TESTS FAILED - Please review errors above")
    print("="*60)