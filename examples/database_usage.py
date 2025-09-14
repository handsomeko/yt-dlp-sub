#!/usr/bin/env python3
"""
Database Usage Examples
Demonstrates common patterns for working with the YouTube transcript database.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, func, and_, or_, desc
from core.database import (
    DatabaseManager, SearchService,
    Channel, Video, Transcript, Job, QualityCheck, 
    GeneratedContent, StorageSync
)


class YouTubeTranscriptService:
    """Service class demonstrating common database operations."""
    
    def __init__(self, database_url: str = "sqlite+aiosqlite:///data/yt-dl-sub.db"):
        self.db_manager = DatabaseManager(database_url)
        self.search_service = SearchService(self.db_manager)
    
    async def initialize(self):
        """Initialize the database."""
        await self.db_manager.initialize()
    
    # ========================================
    # Channel Management
    # ========================================
    
    async def add_channel(
        self,
        channel_id: str,
        channel_name: str,
        channel_url: str = None,
        description: str = None
    ) -> Channel:
        """Add a new channel to monitor."""
        async with self.db_manager.get_session() as session:
            channel = Channel(
                channel_id=channel_id,
                channel_name=channel_name,
                channel_url=channel_url,
                description=description,
                is_active=True
            )
            session.add(channel)
            await session.flush()
            return channel
    
    async def get_active_channels(self) -> List[Channel]:
        """Get all active channels."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.is_active == True)
            )
            return result.scalars().all()
    
    async def update_channel_last_check(self, channel_id: str, last_video_id: str = None):
        """Update channel's last check timestamp and latest video."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.channel_id == channel_id)
            )
            channel = result.scalar_one_or_none()
            if channel:
                channel.last_checked = datetime.now()
                if last_video_id:
                    channel.last_video_id = last_video_id
    
    # ========================================
    # Video Management
    # ========================================
    
    async def add_video(
        self,
        video_id: str,
        channel_id: str,
        title: str,
        description: str = None,
        duration: int = None,
        published_at: datetime = None
    ) -> Video:
        """Add a new video."""
        async with self.db_manager.get_session() as session:
            video = Video(
                video_id=video_id,
                channel_id=channel_id,
                title=title,
                description=description,
                duration=duration,
                published_at=published_at or datetime.now(),
                transcript_status='pending'
            )
            session.add(video)
            await session.flush()
            return video
    
    async def get_videos_by_channel(self, channel_id: str, limit: int = 50) -> List[Video]:
        """Get videos for a specific channel."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Video)
                .where(Video.channel_id == channel_id)
                .order_by(desc(Video.published_at))
                .limit(limit)
            )
            return result.scalars().all()
    
    async def get_videos_needing_transcription(self, limit: int = 10) -> List[Video]:
        """Get videos that need transcription."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Video)
                .where(Video.transcript_status.in_(['pending', 'failed']))
                .order_by(Video.published_at)
                .limit(limit)
            )
            return result.scalars().all()
    
    # ========================================
    # Transcript Management
    # ========================================
    
    async def save_transcript(
        self,
        video_id: str,
        content_text: str,
        content_srt: str = None,
        extraction_method: str = "whisper-local",
        quality_score: float = None,
        audio_path: str = None
    ) -> Transcript:
        """Save a transcript for a video."""
        async with self.db_manager.get_session() as session:
            # Update video status
            video_result = await session.execute(
                select(Video).where(Video.video_id == video_id)
            )
            video = video_result.scalar_one_or_none()
            if video:
                video.transcript_status = 'completed'
            
            # Create transcript
            transcript = Transcript(
                video_id=video_id,
                content_text=content_text,
                content_srt=content_srt,
                word_count=len(content_text.split()) if content_text else 0,
                extraction_method=extraction_method,
                quality_score=quality_score,
                audio_path=audio_path
            )
            session.add(transcript)
            await session.flush()
            return transcript
    
    async def get_transcripts_by_quality(self, min_quality: float = 0.8) -> List[Transcript]:
        """Get high-quality transcripts."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Transcript)
                .where(Transcript.quality_score >= min_quality)
                .order_by(desc(Transcript.quality_score))
            )
            return result.scalars().all()
    
    # ========================================
    # Job Queue Management
    # ========================================
    
    async def queue_transcript_job(self, video_id: str, priority: int = 5) -> Job:
        """Queue a transcript processing job."""
        async with self.db_manager.get_session() as session:
            job = Job(
                job_type='download_transcript',
                target_id=video_id,
                status='pending',
                priority=priority
            )
            session.add(job)
            await session.flush()
            return job
    
    async def get_next_job(self) -> Optional[Job]:
        """Get the next job to process."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Job)
                .where(Job.status == 'pending')
                .order_by(desc(Job.priority), Job.created_at)
                .limit(1)
            )
            return result.scalar_one_or_none()
    
    async def update_job_status(
        self, 
        job_id: int, 
        status: str, 
        error_message: str = None,
        worker_id: str = None
    ):
        """Update job status."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(Job).where(Job.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = status
                if error_message:
                    job.error_message = error_message
                if worker_id:
                    job.worker_id = worker_id
                if status == 'processing':
                    job.started_at = datetime.now()
                elif status in ['completed', 'failed']:
                    job.completed_at = datetime.now()
    
    # ========================================
    # Content Generation
    # ========================================
    
    async def save_generated_content(
        self,
        video_id: str,
        content_type: str,
        content: str,
        metadata: dict = None,
        generation_model: str = "claude-3-sonnet"
    ) -> GeneratedContent:
        """Save AI-generated content."""
        async with self.db_manager.get_session() as session:
            generated_content = GeneratedContent(
                video_id=video_id,
                content_type=content_type,
                content=content,
                content_metadata=metadata or {},
                generation_model=generation_model
            )
            session.add(generated_content)
            await session.flush()
            return generated_content
    
    async def get_content_by_type(self, content_type: str, limit: int = 50) -> List[GeneratedContent]:
        """Get generated content by type."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(GeneratedContent)
                .where(GeneratedContent.content_type == content_type)
                .order_by(desc(GeneratedContent.created_at))
                .limit(limit)
            )
            return result.scalars().all()
    
    # ========================================
    # Analytics and Reports
    # ========================================
    
    async def get_channel_stats(self, channel_id: str) -> dict:
        """Get statistics for a channel."""
        async with self.db_manager.get_session() as session:
            # Video count
            video_count = await session.execute(
                select(func.count(Video.id)).where(Video.channel_id == channel_id)
            )
            total_videos = video_count.scalar()
            
            # Transcribed count
            transcribed_count = await session.execute(
                select(func.count(Video.id))
                .where(and_(Video.channel_id == channel_id, Video.transcript_status == 'completed'))
            )
            transcribed = transcribed_count.scalar()
            
            # Average quality
            avg_quality = await session.execute(
                select(func.avg(Transcript.quality_score))
                .join(Video)
                .where(Video.channel_id == channel_id)
            )
            quality = avg_quality.scalar() or 0.0
            
            # Total word count
            total_words = await session.execute(
                select(func.sum(Transcript.word_count))
                .join(Video)
                .where(Video.channel_id == channel_id)
            )
            words = total_words.scalar() or 0
            
            return {
                "total_videos": total_videos,
                "transcribed_videos": transcribed,
                "transcription_rate": transcribed / total_videos if total_videos > 0 else 0,
                "average_quality_score": float(quality),
                "total_words": words
            }
    
    async def get_processing_stats(self) -> dict:
        """Get overall processing statistics."""
        async with self.db_manager.get_session() as session:
            # Job queue stats
            pending_jobs = await session.execute(
                select(func.count(Job.id)).where(Job.status == 'pending')
            )
            processing_jobs = await session.execute(
                select(func.count(Job.id)).where(Job.status == 'processing')
            )
            failed_jobs = await session.execute(
                select(func.count(Job.id)).where(Job.status == 'failed')
            )
            
            # Content generation stats
            content_stats = await session.execute(
                select(
                    GeneratedContent.content_type,
                    func.count(GeneratedContent.id).label('count')
                )
                .group_by(GeneratedContent.content_type)
            )
            
            content_by_type = {row.content_type: row.count for row in content_stats}
            
            return {
                "queue": {
                    "pending": pending_jobs.scalar() or 0,
                    "processing": processing_jobs.scalar() or 0,
                    "failed": failed_jobs.scalar() or 0
                },
                "content_generated": content_by_type
            }
    
    async def close(self):
        """Close database connections."""
        await self.db_manager.close()


# ========================================
# Usage Examples
# ========================================

async def example_workflow():
    """Example workflow demonstrating the service."""
    print("🚀 YouTube Transcript Service Example")
    print("=" * 50)
    
    service = YouTubeTranscriptService("sqlite+aiosqlite:///data/example.db")
    await service.initialize()
    
    try:
        # 1. Add a channel
        print("\n1. Adding a channel...")
        channel = await service.add_channel(
            channel_id="UCexampletech",
            channel_name="Example Tech Channel",
            channel_url="https://youtube.com/@exampletech",
            description="Technology tutorials and reviews"
        )
        print(f"✅ Added channel: {channel.channel_name}")
        
        # 2. Add videos
        print("\n2. Adding videos...")
        video1 = await service.add_video(
            video_id="video123",
            channel_id=channel.channel_id,
            title="Python Async Programming Guide",
            description="Learn async/await in Python",
            duration=1800
        )
        
        video2 = await service.add_video(
            video_id="video456", 
            channel_id=channel.channel_id,
            title="Database Design Patterns",
            description="Advanced database patterns",
            duration=2400
        )
        print(f"✅ Added {2} videos")
        
        # 3. Queue transcript jobs
        print("\n3. Queueing transcript jobs...")
        await service.queue_transcript_job(video1.video_id, priority=8)
        await service.queue_transcript_job(video2.video_id, priority=5)
        print("✅ Queued transcript jobs")
        
        # 4. Process a job (simulate)
        print("\n4. Processing jobs...")
        job = await service.get_next_job()
        if job:
            print(f"Processing job: {job.job_type} for {job.target_id}")
            await service.update_job_status(job.id, 'processing', worker_id='worker-1')
            
            # Simulate transcript processing
            await service.save_transcript(
                video_id=job.target_id,
                content_text="This is a sample transcript about Python async programming. " * 50,
                extraction_method="whisper-local",
                quality_score=0.89
            )
            
            await service.update_job_status(job.id, 'completed')
            print("✅ Job completed")
        
        # 5. Generate content
        print("\n5. Generating content...")
        blog_post = await service.save_generated_content(
            video_id=video1.video_id,
            content_type="blog",
            content="# Python Async Programming\n\nDetailed explanation of async/await...",
            metadata={"word_count": 500, "hashtags": ["python", "async"]},
            generation_model="claude-3-sonnet"
        )
        print(f"✅ Generated {blog_post.content_type} content")
        
        # 6. Get statistics
        print("\n6. Channel statistics...")
        stats = await service.get_channel_stats(channel.channel_id)
        print(f"✅ Channel stats: {stats}")
        
        processing_stats = await service.get_processing_stats()
        print(f"✅ Processing stats: {processing_stats}")
        
        # 7. Query data
        print("\n7. Querying data...")
        active_channels = await service.get_active_channels()
        print(f"✅ Found {len(active_channels)} active channels")
        
        high_quality = await service.get_transcripts_by_quality(0.8)
        print(f"✅ Found {len(high_quality)} high-quality transcripts")
        
        print("\n🎉 Example workflow completed!")
        
    finally:
        await service.close()


if __name__ == "__main__":
    # Clean up any existing example database
    import os
    if os.path.exists("data/example.db"):
        os.unlink("data/example.db")
    
    asyncio.run(example_workflow())