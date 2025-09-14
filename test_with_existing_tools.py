#!/usr/bin/env python3
"""Test using the existing downloader tools"""

import asyncio
import sys
sys.path.append('/Users/jk/yt-dl-sub')

from workers.downloader import DownloadWorker
from core.database import DatabaseManager

async def test_download():
    """Test downloading using existing tools"""
    
    # Initialize database
    db = DatabaseManager()
    await db.initialize()
    
    # Create a job for the video
    job_data = {
        'url': 'https://www.youtube.com/watch?v=GT0jtVjRy2E',
        'video_id': 'GT0jtVjRy2E',
        'channel_id': 'UC6t1O76G0jYXOAoYCm153dA',  # Lenny's Podcast
        'title': 'How we restructured Airtable\'s entire org for AI',
        'job_type': 'download_audio'
    }
    
    # Create job in database
    job_id = await db.create_job(
        job_type='download_audio',
        data=job_data,
        priority=1
    )
    
    print(f"Created job ID: {job_id}")
    print(f"URL: {job_data['url']}")
    print("-" * 80)
    
    # Initialize download worker
    worker = DownloadWorker()
    
    # Process the job
    print("Processing download job...")
    result = await worker.process_job(job_id)
    
    if result:
        print(f"\n✅ Download successful!")
        print(f"Audio path: {result.get('audio_path', 'N/A')}")
        print(f"Metadata saved: {result.get('metadata_path', 'N/A')}")
    else:
        print("❌ Download failed")
    
    await db.close()
    return result

if __name__ == "__main__":
    result = asyncio.run(test_download())
    if result:
        print("\n✅ Test completed successfully using existing tools!")