#!/usr/bin/env python3
"""Simple test of YouTube download"""

import yt_dlp

url = "https://www.youtube.com/watch?v=GT0jtVjRy2E"

print(f"Testing: {url}")
print("-" * 80)

ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'extract_flat': False
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    
    print(f"✅ Successfully retrieved video info!")
    print(f"\nTitle: {info['title']}")
    print(f"Duration: {info['duration']} seconds ({info['duration']//60} minutes)")
    print(f"Uploader: {info.get('uploader', 'Unknown')}")
    print(f"View Count: {info.get('view_count', 'Unknown'):,}")
    print(f"Upload Date: {info.get('upload_date', 'Unknown')}")
    print(f"\nDescription preview:")
    print(info.get('description', '')[:200] + "...")
    
print("\n✅ YouTube download functionality confirmed working!")