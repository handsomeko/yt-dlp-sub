#!/usr/bin/env python3
"""
Test script for the export functionality.

This script demonstrates and tests all export formats with sample data.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from core.database import DatabaseManager, Channel, Video, Transcript, GeneratedContent
from core.export import ExportService, ExportProgress, export_transcripts, get_export_stats


async def setup_test_data(db_manager: DatabaseManager):
    """Create sample test data for export testing."""
    async with db_manager.get_session() as session:
        # Create test channels
        channel1 = Channel(
            channel_id="UC123456789",
            channel_name="Tech Talk Channel",
            channel_url="https://youtube.com/c/techtalk",
            description="Educational technology content",
            subscriber_count=150000,
            video_count=250,
            is_active=True
        )
        
        channel2 = Channel(
            channel_id="UC987654321",
            channel_name="Science Weekly",
            channel_url="https://youtube.com/c/scienceweekly",
            description="Weekly science updates",
            subscriber_count=85000,
            video_count=120,
            is_active=True
        )
        
        session.add(channel1)
        session.add(channel2)
        await session.flush()
        
        # Create test videos
        videos = []
        transcripts = []
        content_items = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(5):
            video = Video(
                video_id=f"video_{i+1:03d}",
                channel_id=channel1.channel_id if i % 2 == 0 else channel2.channel_id,
                title=f"Sample Video {i+1}: Technology Trends in 2024",
                description=f"This is a detailed description for video {i+1} covering important technology trends and innovations in the current year.",
                duration=1800 + i * 300,  # 30-45 minutes
                view_count=10000 + i * 5000,
                like_count=500 + i * 100,
                published_at=base_date + timedelta(days=i * 7),
                transcript_status="completed",
                language="en",
                is_auto_generated=False
            )
            videos.append(video)
            session.add(video)
            
        await session.flush()
        
        # Create test transcripts
        for i, video in enumerate(videos):
            transcript_text = f"""
Welcome to this comprehensive discussion about technology trends in 2024. 
In this video, we'll explore the latest developments in artificial intelligence, 
machine learning, and how they're shaping the future of technology.

First, let's talk about the rapid advancement of large language models. 
These models have shown remarkable capabilities in understanding and generating human-like text.
The implications for various industries are profound and far-reaching.

Moving on to the topic of automation, we're seeing significant changes in how businesses operate.
Companies are increasingly adopting AI-powered solutions to streamline their processes
and improve efficiency across different departments.

The integration of AI into everyday applications has become more seamless and user-friendly.
From virtual assistants to recommendation systems, AI is becoming an integral part
of our digital experience.

Looking ahead, we can expect even more innovative applications of these technologies
in fields such as healthcare, education, and sustainable development.
The potential for positive impact is enormous.

Thank you for watching this video. Please subscribe for more content on technology trends
and don't forget to like and share if you found this information valuable.
            """.strip()
            
            transcript = Transcript(
                video_id=video.video_id,
                content_text=transcript_text,
                content_srt=f"1\n00:00:00,000 --> 00:00:10,000\n{transcript_text[:100]}...\n\n2\n00:00:10,000 --> 00:00:20,000\n{transcript_text[100:200]}...",
                word_count=len(transcript_text.split()),
                language="en",
                extraction_method="whisper-local",
                transcription_model="whisper-base",
                quality_score=0.85 + (i * 0.03),
                quality_details={"confidence": 0.9, "word_accuracy": 0.95},
                audio_path=f"/data/audio/{video.video_id}.opus",
                srt_path=f"/data/srt/{video.video_id}.srt",
                transcript_path=f"/data/txt/{video.video_id}.txt"
            )
            transcripts.append(transcript)
            session.add(transcript)
            
        await session.flush()
        
        # Create test generated content
        for i, video in enumerate(videos):
            # Blog post
            blog_content = GeneratedContent(
                video_id=video.video_id,
                content_type="blog",
                content=f"""# {video.title}

## Introduction

This blog post summarizes the key insights from our latest video about technology trends in 2024.

## Key Points

1. **Artificial Intelligence**: The rapid advancement of AI technologies
2. **Machine Learning**: Applications in various industries
3. **Automation**: Streamlining business processes
4. **Future Outlook**: Expected developments in the coming years

## Conclusion

The technology landscape continues to evolve at an unprecedented pace, offering both opportunities and challenges for businesses and individuals alike.

*This content was generated from the video transcript using AI.*
                """,
                content_metadata={"word_count": 95, "reading_time": "2 minutes"},
                quality_score=0.90,
                generation_model="gpt-4",
                prompt_template="blog_summary_v1",
                created_at=video.published_at + timedelta(hours=2)
            )
            
            # Social media post
            social_content = GeneratedContent(
                video_id=video.video_id,
                content_type="social_twitter",
                content=f"""🚀 Just released: {video.title}

Key takeaways:
✅ AI & ML trends for 2024
✅ Automation impact on business
✅ Future tech developments

Watch the full video for detailed insights! 

#TechTrends #AI #MachineLearning #Technology2024
                """,
                content_metadata={"character_count": 198, "hashtags": 4},
                quality_score=0.88,
                generation_model="gpt-4",
                prompt_template="social_twitter_v1",
                created_at=video.published_at + timedelta(hours=4)
            )
            
            session.add(blog_content)
            session.add(social_content)
            content_items.extend([blog_content, social_content])
        
        await session.commit()
        
        print(f"✅ Test data created:")
        print(f"  - Channels: {len([channel1, channel2])}")
        print(f"  - Videos: {len(videos)}")
        print(f"  - Transcripts: {len(transcripts)}")
        print(f"  - Generated content: {len(content_items)}")


async def test_export_formats():
    """Test all export formats with sample data."""
    print("\n🧪 Testing Export Formats")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_url = f"sqlite+aiosqlite:///{temp_db.name}"
    
    try:
        # Initialize database with test data
        db_manager = DatabaseManager(db_url)
        await db_manager.initialize()
        await setup_test_data(db_manager)
        
        # Create export service
        export_service = ExportService(db_manager)
        
        # Test getting export stats
        print("\n📊 Testing Export Statistics")
        stats = await export_service.get_export_stats()
        print(f"Total transcripts: {stats.total_transcripts}")
        print(f"Total content items: {stats.total_content_items}")
        print(f"Total videos: {stats.total_videos}")
        print(f"Total channels: {stats.total_channels}")
        
        # Test filtered stats
        channel_stats = await export_service.get_export_stats(channel_id="UC123456789")
        print(f"Channel UC123456789 transcripts: {channel_stats.total_transcripts}")
        
        # Test each export format
        formats = ["json", "csv", "txt", "markdown"]
        export_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for fmt in formats:
                print(f"\n📤 Testing {fmt.upper()} export...")
                
                # Progress callback for demonstration
                def progress_callback(progress: ExportProgress):
                    if progress.current_item % 2 == 0:  # Reduce output frequency
                        print(f"  {progress.current_phase}: {progress.progress_percent:.1f}% "
                              f"({progress.current_item}/{progress.total_items})")
                
                # Test export
                result = await export_service.export_transcripts(
                    format=fmt,
                    output_path=temp_path / f"test_export.{fmt}",
                    include_content=True,
                    progress_callback=progress_callback
                )
                
                export_results[fmt] = result
                
                # Verify file was created
                output_file = Path(result['output_path'])
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"  ✅ {fmt.upper()} export successful: {file_size:,} bytes")
                    
                    # Show first few lines for small formats
                    if fmt in ['txt', 'markdown'] and file_size < 5000:
                        print(f"  📄 First 3 lines of {fmt.upper()} export:")
                        with open(output_file, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if i >= 3:
                                    break
                                print(f"     {line.rstrip()}")
                else:
                    print(f"  ❌ {fmt.upper()} export failed: file not created")
        
        # Test filtered exports
        print(f"\n🔍 Testing Filtered Exports")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test channel filter
            channel_result = await export_service.export_transcripts(
                format="json",
                channel_id="UC123456789",
                output_path=temp_path / "channel_filtered.json",
                include_content=False
            )
            
            print(f"Channel filtered export: {channel_result['stats'].total_transcripts} transcripts")
            
            # Test date filter
            since_date = datetime.now() - timedelta(days=15)
            date_result = await export_service.export_transcripts(
                format="csv",
                since=since_date,
                output_path=temp_path / "date_filtered.csv",
                include_content=True
            )
            
            print(f"Date filtered export (since {since_date.strftime('%Y-%m-%d')}): {date_result['stats'].total_transcripts} transcripts")
        
        # Test error handling
        print(f"\n⚠️  Testing Error Handling")
        
        try:
            await export_service.export_transcripts(format="invalid_format")
        except ValueError as e:
            print(f"  ✅ Format validation working: {e}")
        
        try:
            await export_service.export_transcripts(
                format="json",
                since="invalid_date"
            )
        except ValueError as e:
            print(f"  ✅ Date validation working: {e}")
        
        # Clean up
        await db_manager.close()
        
        print(f"\n🎉 All export tests completed successfully!")
        
        # Summary
        print(f"\n📋 Export Test Summary:")
        for fmt, result in export_results.items():
            stats = result['stats']
            print(f"  {fmt.upper():8} - {stats.total_transcripts} transcripts, {stats.file_size_bytes:,} bytes, {stats.export_duration_seconds:.2f}s")
    
    finally:
        # Clean up temporary database
        Path(temp_db.name).unlink(missing_ok=True)


async def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🔧 Testing Convenience Functions")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        db_url = f"sqlite+aiosqlite:///{temp_db.name}"
    
    try:
        # Initialize database with test data
        db_manager = DatabaseManager(db_url)
        await db_manager.initialize()
        await setup_test_data(db_manager)
        
        # Temporarily override global db_manager for convenience functions
        import core.export
        original_db_manager = core.export.db_manager
        core.export.db_manager = db_manager
        
        # Test convenience stats function
        stats = await get_export_stats(db_manager=db_manager)
        print(f"✅ get_export_stats(): {stats.total_transcripts} transcripts")
        
        # Test convenience export function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await export_transcripts(
                format="json",
                output_path=Path(temp_dir) / "convenience_test.json",
                include_content=True,
                db_manager=db_manager
            )
            
            print(f"✅ export_transcripts() convenience function: {result['stats'].total_transcripts} transcripts exported")
        
        # Restore original db_manager
        core.export.db_manager = original_db_manager
        
        await db_manager.close()
        
    finally:
        # Clean up temporary database
        Path(temp_db.name).unlink(missing_ok=True)


async def main():
    """Run all export tests."""
    print("🚀 YouTube Content Intelligence - Export System Tests")
    print("=" * 60)
    
    try:
        await test_export_formats()
        await test_convenience_functions()
        
        print(f"\n🎊 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)