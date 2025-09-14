#!/usr/bin/env python3
"""
Quick progress check for TCM-Chan download.
"""

from pathlib import Path
import re

# Check log file
log_file = Path("tcm_chan_download.log")
if log_file.exists():
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Count completed
    completed = len(re.findall(r'✅ \[(\d+)/59\] Completed', content))
    
    # Count started
    started = len(set(re.findall(r'\[(\d+)/59\] Starting download', content)))
    
    # Count Chinese and English files
    zh_files = len(re.findall(r'\.zh\.(txt|srt)', content))
    en_files = len(re.findall(r'\.en\.(txt|srt)', content))
    
    print(f"📊 TCM-Chan Download Progress")
    print(f"="*40)
    print(f"Started:   {started}/59 videos")
    print(f"Completed: {completed}/59 videos")
    print(f"Progress:  {completed/59*100:.1f}%")
    print(f"Chinese files: {zh_files}")
    print(f"English files: {en_files}")

# Check actual downloaded files
downloads_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCYcMQmLxOKd9TMZguFEotww")
if downloads_dir.exists():
    video_dirs = [d for d in downloads_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    zh_txt_count = 0
    en_txt_count = 0
    
    for video_dir in video_dirs:
        transcripts_dir = video_dir / "transcripts"
        if transcripts_dir.exists():
            zh_txt_count += len(list(transcripts_dir.glob("*.zh.txt")))
            en_txt_count += len(list(transcripts_dir.glob("*.en.txt")))
    
    print(f"\n📁 Actual Files Downloaded:")
    print(f"Video directories: {len(video_dirs)}")
    print(f"Chinese .txt files: {zh_txt_count}")
    print(f"English .txt files: {en_txt_count}")