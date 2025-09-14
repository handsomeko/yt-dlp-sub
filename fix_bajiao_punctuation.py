#!/usr/bin/env python3
"""Fix punctuation for the specific 八角放火上烤一烤 file"""

import asyncio
from pathlib import Path
from core.chinese_punctuation import ChinesePunctuationRestorer

async def main():
    file_path = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCSfzaDNaHzOf5i9zAEJ0d6w/mcP4tc2unVs/transcripts/八角放火上烤一烤，竟是「關節疼痛」的剋星？老祖宗的智慧，不花一分錢，解決中老年人三大難題！#健康知识#老年健康#健康养生#醫師健康日記.zh.txt")
    
    print(f"📄 Processing: {file_path.name}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Check punctuation
    sentence_endings = '。！？'
    punct_count = sum(1 for char in text if char in sentence_endings)
    print(f"🔍 Current punctuation marks: {punct_count}")
    
    if punct_count == 0:
        print("🔤 Applying Chinese punctuation restoration...")
        
        # Use our consolidated SRT-aware system with Claude CLI fallback
        restorer = ChinesePunctuationRestorer()
        
        # Check if there's a corresponding SRT file for SRT-aware processing
        srt_content = None
        srt_path = file_path.with_suffix('.srt')
        if srt_path.exists():
            srt_content = srt_path.read_text(encoding='utf-8')
            print(f"📄 Found SRT file, using SRT-aware processing: {srt_path.name}")
        else:
            print("📄 No SRT file found, using text-only processing")
        
        restored_text, success = await restorer.restore_punctuation(text, srt_content)
        
        if success and restored_text != text:
            # Back up original
            backup_path = file_path.with_suffix('.txt.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"💾 Backed up original to: {backup_path.name}")
            
            # Write restored text
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(restored_text)
            
            # Count new punctuation
            new_punct_count = sum(1 for char in restored_text if char in sentence_endings)
            print(f"✅ Punctuation restored! Added {new_punct_count} punctuation marks")
            print(f"📄 File updated: {file_path.name}")
        else:
            print("❌ Punctuation restoration failed")
    else:
        print(f"✅ File already has {punct_count} punctuation marks")

if __name__ == "__main__":
    asyncio.run(main())