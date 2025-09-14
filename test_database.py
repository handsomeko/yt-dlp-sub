#!/usr/bin/env python3
"""
Test script for database functionality.
Demonstrates basic CRUD operations and FTS5 search capabilities.
"""

import asyncio
import json
from datetime import datetime
from sqlalchemy import select

from core.database import (
    DatabaseManager, SearchService,
    Channel, Video, Transcript, Job, QualityCheck, 
    GeneratedContent, StorageSync
)


async def test_database_operations():
    """Test basic database operations."""
    print("🧪 Testing Database Operations")
    print("=" * 50)
    
    # Initialize database
    db_manager = DatabaseManager("sqlite+aiosqlite:///data/test-yt-dl-sub.db")
    await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        # 1. Test Channel operations
        print("\n1. Testing Channel operations...")
        
        # Create a test channel
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
        await session.flush()  # Get the ID
        print(f"✅ Created channel: {channel.channel_name} (ID: {channel.id})")
        
        # 2. Test Video operations
        print("\n2. Testing Video operations...")
        
        video = Video(
            video_id="test_video_456",
            channel_id=channel.channel_id,
            title="Advanced Python Async Programming Tutorial",
            description="Learn advanced async/await patterns in Python",
            duration=1800,  # 30 minutes
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
                         "concurrent programming with asyncio. Python's async capabilities "
                         "allow for efficient handling of I/O operations and concurrent tasks.",
            word_count=200,
            language="en",
            extraction_method="whisper-local",
            transcription_model="whisper-base",
            quality_score=0.92,
            quality_details={"confidence": 0.92, "word_confidence_avg": 0.89},
            audio_path="/downloads/audio/test_video_456.opus",
            transcript_path="/downloads/transcripts/test_video_456.txt"
        )
        
        session.add(transcript)
        await session.flush()
        print(f"✅ Created transcript: {transcript.word_count} words")
        
        # 4. Test Generated Content
        print("\n4. Testing Generated Content operations...")
        
        blog_content = GeneratedContent(
            video_id=video.video_id,
            content_type="blog",
            content="# Advanced Python Async Programming\n\n"
                   "Python's asyncio library provides powerful tools for concurrent programming. "
                   "In this comprehensive guide, we explore async/await patterns, event loops, "
                   "and best practices for building efficient async applications.\n\n"
                   "## Key Concepts\n- Event loops\n- Coroutines\n- Tasks and futures",
            content_metadata={"word_count": 150, "reading_time": 3, "hashtags": ["python", "async", "programming"]},
            quality_score=0.88,
            generation_model="claude-3-sonnet",
            prompt_template="blog_technical"
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
            priority=5,
            max_retries=3
        )
        
        session.add(job)
        await session.flush()
        print(f"✅ Created job: {job.job_type} for {job.target_id}")
        
        # 6. Test Quality Check
        print("\n6. Testing Quality Check operations...")
        
        quality_check = QualityCheck(
            target_id=video.video_id,
            target_type="transcript",
            check_type="completeness",
            score=0.92,
            passed=True,
            details={"missing_segments": 0, "confidence_low_count": 2}
        )
        
        session.add(quality_check)
        await session.flush()
        print(f"✅ Created quality check: {quality_check.check_type} (passed: {quality_check.passed})")
        
        # Commit all changes
        await session.commit()
        print("\n✅ All test data committed to database")
        
        # 7. Test queries
        print("\n7. Testing database queries...")
        
        # Query channels with videos
        result = await session.execute(
            select(Channel).where(Channel.is_active == True)
        )
        channels = result.scalars().all()
        print(f"✅ Found {len(channels)} active channels")
        
        # Query videos by channel
        result = await session.execute(
            select(Video).where(Video.channel_id == channel.channel_id)
        )
        videos = result.scalars().all()
        print(f"✅ Found {len(videos)} videos for channel")
        
        # Query jobs by status
        result = await session.execute(
            select(Job).where(Job.status == "pending")
        )
        pending_jobs = result.scalars().all()
        print(f"✅ Found {len(pending_jobs)} pending jobs")
    
    # 8. Test FTS5 search
    print("\n8. Testing FTS5 Search...")
    search_service = SearchService(db_manager)
    
    # Search transcripts
    results = await search_service.search_transcripts("Python async programming")
    print(f"✅ Transcript search returned {len(results)} results")
    
    if results:
        print(f"   First result: {results[0]['title']}")
        print(f"   Snippet: {results[0]['snippet']}")
    
    # Search generated content
    content_results = await search_service.search_content("event loops coroutines")
    print(f"✅ Content search returned {len(content_results)} results")
    
    if content_results:
        print(f"   First result: {content_results[0]['content_type']}")
        print(f"   Snippet: {content_results[0]['snippet']}")
    
    await db_manager.close()
    print("\n🎉 Database test completed successfully!")


async def test_search_performance():
    """Test search performance with larger dataset."""
    print("\n🚀 Testing Search Performance")
    print("=" * 50)
    
    db_manager = DatabaseManager("sqlite+aiosqlite:///data/test-yt-dl-sub.db")
    search_service = SearchService(db_manager)
    
    import time
    
    # Test various search queries
    test_queries = [
        "Python programming",
        "async await",
        "event loops",
        "concurrent programming",
        "asyncio tutorial"
    ]
    
    for query in test_queries:
        start_time = time.time()
        results = await search_service.search_transcripts(query)
        end_time = time.time()
        
        print(f"Query: '{query}' → {len(results)} results in {(end_time - start_time)*1000:.2f}ms")
    
    await db_manager.close()


if __name__ == "__main__":
    print("🧪 YouTube Transcript Database Test Suite")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_database_operations())
    asyncio.run(test_search_performance())
    
    print("\n✅ All tests completed!")