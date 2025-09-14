#!/usr/bin/env python3
"""
Simple database test without FTS5 search to verify basic functionality.
"""

import asyncio
import json
from datetime import datetime
from sqlalchemy import select

from core.database import (
    DatabaseManager,
    Channel, Video, Transcript, Job, QualityCheck, 
    GeneratedContent, StorageSync
)


async def test_basic_operations():
    """Test basic database operations without FTS5."""
    print("🧪 Testing Basic Database Operations")
    print("=" * 50)
    
    # Remove existing test database
    import os
    if os.path.exists("data/simple-test.db"):
        os.unlink("data/simple-test.db")
    
    # Initialize database
    db_manager = DatabaseManager("sqlite+aiosqlite:///data/simple-test.db")
    await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        # 1. Test Channel operations
        print("\n1. Testing Channel operations...")
        
        channel = Channel(
            channel_id="UC_test_channel_123",
            channel_name="Test Tech Channel",
            channel_url="https://youtube.com/@testtechchannel",
            description="A test channel for technology content",
            subscriber_count=50000,
            video_count=150,
            is_active=True
        )
        
        session.add(channel)
        await session.flush()
        print(f"✅ Created channel: {channel.channel_name} (ID: {channel.id})")
        
        # 2. Test Video operations
        print("\n2. Testing Video operations...")
        
        video = Video(
            video_id="test_video_456",
            channel_id=channel.channel_id,
            title="Advanced Python Async Programming Tutorial",
            description="Learn advanced async/await patterns in Python",
            duration=1800,
            view_count=25000,
            like_count=1200,
            published_at=datetime.now(),
            transcript_status="completed",
            language="en"
        )
        
        session.add(video)
        await session.flush()
        print(f"✅ Created video: {video.title}")
        
        # 3. Test Transcript operations
        print("\n3. Testing Transcript operations...")
        
        transcript = Transcript(
            video_id=video.video_id,
            content_text="Welcome to our advanced Python async programming tutorial. "
                         "Today we'll cover async/await patterns, event loops, and "
                         "concurrent programming with asyncio.",
            word_count=200,
            language="en",
            extraction_method="whisper-local",
            transcription_model="whisper-base",
            quality_score=0.92
        )
        
        session.add(transcript)
        await session.flush()
        print(f"✅ Created transcript: {transcript.word_count} words")
        
        # 4. Test Generated Content
        print("\n4. Testing Generated Content operations...")
        
        blog_content = GeneratedContent(
            video_id=video.video_id,
            content_type="blog",
            content="# Advanced Python Async Programming\n\nPython's asyncio library provides powerful tools.",
            content_metadata={"word_count": 150, "hashtags": ["python", "async"]},
            quality_score=0.88,
            generation_model="claude-3-sonnet"
        )
        
        session.add(blog_content)
        await session.flush()
        print(f"✅ Created blog content: {blog_content.content_type}")
        
        # 5. Test Job queue
        print("\n5. Testing Job operations...")
        
        job = Job(
            job_type="download_transcript",
            target_id=video.video_id,
            status="pending",
            priority=5
        )
        
        session.add(job)
        await session.flush()
        print(f"✅ Created job: {job.job_type}")
        
        # 6. Test Quality Check
        print("\n6. Testing Quality Check operations...")
        
        quality_check = QualityCheck(
            target_id=video.video_id,
            target_type="transcript",
            check_type="completeness",
            score=0.92,
            passed=True,
            details={"missing_segments": 0}
        )
        
        session.add(quality_check)
        await session.flush()
        print(f"✅ Created quality check: passed={quality_check.passed}")
        
        # 7. Test Storage Sync
        print("\n7. Testing Storage Sync operations...")
        
        storage_sync = StorageSync(
            file_type="transcript",
            local_path="/downloads/transcripts/test_video_456.txt",
            sync_status="synced"
        )
        
        session.add(storage_sync)
        await session.flush()
        print(f"✅ Created storage sync: {storage_sync.sync_status}")
        
        await session.commit()
        print("\n✅ All test data committed to database")
        
        # 8. Test queries
        print("\n8. Testing database queries...")
        
        # Test joins and relationships
        result = await session.execute(
            select(Channel).where(Channel.is_active == True)
        )
        channels = result.scalars().all()
        print(f"✅ Found {len(channels)} active channels")
        
        # Test video-transcript relationship
        result = await session.execute(
            select(Video).join(Transcript).where(Transcript.quality_score > 0.9)
        )
        high_quality_videos = result.scalars().all()
        print(f"✅ Found {len(high_quality_videos)} videos with high-quality transcripts")
        
        # Test content by type
        result = await session.execute(
            select(GeneratedContent).where(GeneratedContent.content_type == "blog")
        )
        blog_posts = result.scalars().all()
        print(f"✅ Found {len(blog_posts)} blog posts")
        
        # Test pending jobs
        result = await session.execute(
            select(Job).where(Job.status == "pending").order_by(Job.priority.desc())
        )
        pending_jobs = result.scalars().all()
        print(f"✅ Found {len(pending_jobs)} pending jobs")
    
    await db_manager.close()
    print("\n🎉 Basic database test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_basic_operations())