#!/usr/bin/env python3
"""Test direct download of the YouTube URL"""

import sys
import asyncio
sys.path.append('/Users/jk/yt-dl-sub')

from workers.downloader import DownloadWorker

async def test_download():
    """Test downloading the video"""
    worker = DownloadWorker()
    
    url = "https://www.youtube.com/watch?v=GT0jtVjRy2E"
    print(f"Testing download of: {url}")
    print("Video Title: How we restructured Airtable's entire org for AI | Howie Liu")
    print("-" * 80)
    
    # Get video info first
    info = await worker.get_video_info(url)
    print(f"\n✅ Video Info Retrieved:")
    print(f"  Title: {info['title']}")
    print(f"  Duration: {info['duration']} seconds ({info['duration']//60} minutes)")
    print(f"  Uploader: {info.get('uploader', 'Unknown')}")
    print(f"  View Count: {info.get('view_count', 'Unknown')}")
    
    print("\n🎯 Download would save to:")
    print(f"  Storage path: {worker.storage_path}")
    print(f"  Video would be: {worker.storage_path}/videos/...")
    
    return info

if __name__ == "__main__":
    result = asyncio.run(test_download())
    print("\n✅ Test completed successfully!")
    print(f"Video is {result['duration']//60} minutes long")