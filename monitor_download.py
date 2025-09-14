#!/usr/bin/env python3
"""
Monitor TCM-Chan download progress.
"""

import time
import re
from pathlib import Path

def monitor_progress():
    """Monitor download progress from log file."""
    log_file = Path("tcm_chan_download.log")
    
    if not log_file.exists():
        print("Log file not found")
        return
    
    print("📊 TCM-Chan Download Progress Monitor")
    print("="*60)
    
    while True:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Count completed downloads
            completed_pattern = r'\[(\d+)/59\] Download complete'
            completed_matches = re.findall(completed_pattern, content)
            completed = len(set(completed_matches))
            
            # Count started downloads
            started_pattern = r'\[(\d+)/59\] Starting download'
            started_matches = re.findall(started_pattern, content)
            started = len(set(started_matches))
            
            # Count Chinese and English files
            chinese_files = content.count('.zh.txt') + content.count('.zh.srt')
            english_files = content.count('.en.txt') + content.count('.en.srt')
            
            # Count Whisper transcriptions
            whisper_count = content.count('Transcribing Chinese audio')
            
            # Clear screen and show status
            print(f"\r⏳ Progress: {completed}/59 completed, {started}/59 started", end='')
            print(f" | 🇨🇳 Chinese: {chinese_files} | 🇬🇧 English: {english_files}", end='')
            print(f" | 🎙️ Whisper: {whisper_count}", end='')
            
            if completed >= 59:
                print("\n✅ All downloads complete!")
                break
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    monitor_progress()