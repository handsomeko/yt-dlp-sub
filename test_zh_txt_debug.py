#!/usr/bin/env python3
"""Debug script to test zh.txt file creation issue"""

import sys
sys.path.insert(0, '/Users/jk/yt-dl-sub')

from core.downloader import YouTubeDownloader
from pathlib import Path

# Test single video
downloader = YouTubeDownloader()
url = 'https://www.youtube.com/watch?v=Ok9KSaqbN-A'

# Check if zh.txt exists before
zh_file = Path('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCYcMQmLxOKd9TMZguFEotww/Ok9KSaqbN-A/transcripts/脖子、臉上的「脂肪粒」，是身體濕氣重發出的信號！不用花大錢做雷射，用這2樣天然好物可消除。.zh.txt')
print(f'Before download: zh.txt exists = {zh_file.exists()}')

print("\n" + "="*60)
print("Starting download...")
print("="*60 + "\n")

result = downloader.download_video(url, quality='1080p', download_audio_only=True)

print("\n" + "="*60)
print("After download check:")
print("="*60)

# Check if zh.txt exists after
print(f'zh.txt exists = {zh_file.exists()}')
if zh_file.exists():
    print(f'File size: {zh_file.stat().st_size} bytes')
    with open(zh_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f'First 100 chars: {content[:100]}')
else:
    print('❌ File not found!')
    # Check if parent dir exists
    print(f'Parent dir exists: {zh_file.parent.exists()}')
    if zh_file.parent.exists():
        txt_files = list(zh_file.parent.glob("*.txt"))
        print(f'Number of .txt files in transcripts dir: {len(txt_files)}')
        for f in txt_files:
            print(f'  - {f.name}')
            
print("\n" + "="*60)
print("Files in result:")
print("="*60)
if 'files' in result:
    for f in result['files']:
        if '.txt' in f:
            print(f'  - {f}')
            print(f'    Exists: {Path(f).exists()}')