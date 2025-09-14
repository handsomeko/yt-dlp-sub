#!/usr/bin/env python3
"""Fix the missing transcripts by using the new language-agnostic extractor on the actual storage location"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor

def fix_missing_transcripts():
    """Extract subtitles directly into the actual storage transcripts folder"""
    
    # The actual video with missing transcripts
    video_id = "oJsYHAJZlHU"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_title = "每天浪費這個時辰，等同於在減壽！一天中最能增壽的時刻，太多人白白錯過了"
    
    # Actual transcripts directory in storage
    transcripts_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCYcMQmLxOKd9TMZguFEotww/oJsYHAJZlHU/transcripts")
    
    print(f"🎯 Fixing missing transcripts for video: {video_id}")
    print(f"📺 Video title: {video_title}")
    print(f"📁 Target directory: {transcripts_dir}")
    print()
    
    # Check current state
    print("=== Current State ===")
    existing_files = list(transcripts_dir.glob("*"))
    print(f"📄 Current files in transcripts folder: {len(existing_files)}")
    if existing_files:
        for file in existing_files:
            print(f"   • {file.name}")
    else:
        print("   (empty - no transcript files)")
    print()
    
    # Use the new language-agnostic extractor
    print("=== Extracting Subtitles ===")
    extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
    result = extractor.extract_subtitles(
        video_url=video_url,
        output_dir=transcripts_dir,
        video_id=video_id,
        video_title=video_title
    )
    
    print(f"✅ Extraction success: {result.success}")
    print(f"🌍 Languages found: {result.languages_found}")
    print(f"📄 Files created: {len(result.original_files)}")
    print(f"🔧 Methods used: {result.methods_used}")
    
    if result.original_files:
        print("📋 Created files:")
        for file in result.original_files:
            print(f"   • {Path(file).name}")
    
    if result.error_messages:
        print(f"⚠️  Errors encountered:")
        for error in result.error_messages:
            print(f"   • {error}")
    print()
    
    # Verify final state
    print("=== Final State ===")
    final_files = list(transcripts_dir.glob("*"))
    print(f"📄 Final files in transcripts folder: {len(final_files)}")
    
    if final_files:
        for file in final_files:
            file_size = file.stat().st_size
            print(f"   • {file.name} ({file_size:,} bytes)")
            
        # Show preview of content
        txt_files = [f for f in final_files if f.suffix == '.txt']
        if txt_files:
            print(f"\n📖 Content preview from {txt_files[0].name}:")
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                content = f.read()[:300]
                print(content)
                if len(content) >= 300:
                    print("... (truncated)")
    else:
        print("   (still empty)")
    
    return result.success

if __name__ == "__main__":
    success = fix_missing_transcripts()
    
    if success:
        print("\n🎉 SUCCESS: Missing transcripts have been extracted!")
        print("The video now has subtitles in its actual transcripts folder.")
    else:
        print("\n❌ FAILED: Unable to extract subtitles")
        print("Further investigation needed.")