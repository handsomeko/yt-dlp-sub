#!/usr/bin/env python3
"""
Continuous monitoring of TCM-Chan download progress.
"""

import time
import re
from pathlib import Path
from datetime import datetime

def monitor_continuous():
    """Monitor download progress continuously."""
    
    log_file = Path("tcm_chan_download.log")
    
    print("📊 TCM-Chan Download Monitor - Continuous Mode")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_completed = 0
    last_status = ""
    
    while True:
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Count completed
                completed = len(re.findall(r'✅ \[(\d+)/59\] Completed', content))
                
                # Count started
                started_matches = re.findall(r'\[(\d+)/59\] Starting download', content)
                started = len(set(started_matches))
                
                # Count Chinese files
                zh_files = len(re.findall(r'\.zh\.(txt|srt)', content))
                en_files = len(re.findall(r'\.en\.(txt|srt)', content))
                
                # Count Whisper transcriptions
                whisper_count = content.count('Transcribing Chinese audio')
                
                # Count rate limit errors
                rate_limits = content.count('HTTP Error 429')
                
                # Get last few operations
                last_operations = re.findall(r'(✅ \[\d+/59\].*|🎬 \[\d+/59\].*|Successfully extracted.*)', content)[-3:]
                
                # Build status
                progress_pct = (completed / 59) * 100
                status = f"Progress: {completed}/59 ({progress_pct:.1f}%) | Started: {started} | "
                status += f"Chinese: {zh_files} | English: {en_files} | "
                status += f"Whisper: {whisper_count} | 429s: {rate_limits}"
                
                # Only update if something changed
                if completed != last_completed or status != last_status:
                    print(f"\r{datetime.now().strftime('%H:%M:%S')} - {status}", end='', flush=True)
                    
                    # Print completion messages
                    if completed > last_completed:
                        print(f"\n✅ Video {completed} completed!")
                        for op in last_operations[-2:]:
                            print(f"   {op[:80]}...")
                    
                    last_completed = completed
                    last_status = status
                
                # Check if complete
                if completed >= 59:
                    print(f"\n\n🎉 All 59 videos downloaded successfully!")
                    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    break
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Monitoring stopped by user")
            print(f"Final status: {completed}/59 completed")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break
    
    # Final summary
    if log_file.exists():
        downloads_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCYcMQmLxOKd9TMZguFEotww")
        if downloads_dir.exists():
            video_dirs = [d for d in downloads_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            zh_txt_files = []
            en_txt_files = []
            
            for video_dir in video_dirs:
                transcripts_dir = video_dir / "transcripts"
                if transcripts_dir.exists():
                    zh_txt_files.extend(list(transcripts_dir.glob("*.zh.txt")))
                    en_txt_files.extend(list(transcripts_dir.glob("*.en.txt")))
            
            print(f"\n📁 Final File Count:")
            print(f"   Video directories: {len(video_dirs)}")
            print(f"   Chinese .txt files: {len(zh_txt_files)}")
            print(f"   English .txt files: {len(en_txt_files)}")
            
            if zh_txt_files:
                print(f"\n📝 Sample Chinese transcripts:")
                for f in zh_txt_files[:3]:
                    print(f"   - {f.parent.parent.name}/{f.name}")

if __name__ == "__main__":
    monitor_continuous()