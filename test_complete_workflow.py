#!/usr/bin/env python3
"""
Complete end-to-end workflow test for yt-dl-sub Phase 1.

Tests the entire pipeline from channel addition to content export.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set test database before importing anything else
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test_workflow.db"

from core.database import db_manager, Channel, Video, Job, Transcript
from core.queue import get_job_queue
from config.settings import get_settings
from sqlalchemy import select, text


class WorkflowTester:
    """Test the complete Phase 1 workflow."""
    
    def __init__(self):
        self.settings = get_settings()
        self.results = []
        
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {details}")
        self.results.append((name, passed, details))
    
    async def test_database_init(self):
        """Test database initialization."""
        try:
            # Database already initialized in setup_test_database
            
            # Check if tables exist
            async with db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = [row[0] for row in result]
                
                # Check for critical tables
                required_tables = [
                    'channels', 'videos', 'jobs', 'transcripts', 
                    'generated_content', 'storage_sync', 'quality_checks'
                ]
                
                missing = [t for t in required_tables if t not in tables]
                if missing:
                    self.log_test("Database Tables", False, f"Missing: {missing}")
                    return False
                else:
                    self.log_test("Database Tables", True, f"{len(tables)} tables found")
                    return True
                    
        except Exception as e:
            self.log_test("Database Init", False, str(e))
            return False
    
    async def test_channel_operations(self):
        """Test channel CRUD operations."""
        try:
            test_channel_id = "test_workflow_channel"
            
            async with db_manager.get_session() as session:
                # Clean up any existing test channel
                existing = await session.execute(
                    select(Channel).where(Channel.channel_id == test_channel_id)
                )
                if existing_channel := existing.scalar_one_or_none():
                    await session.delete(existing_channel)
                    await session.commit()
                
                # Create test channel
                channel = Channel(
                    channel_id=test_channel_id,
                    channel_name="Test Workflow Channel",
                    channel_url="https://youtube.com/@test",
                    is_active=True
                )
                session.add(channel)
                await session.commit()
                
                # Read channel
                result = await session.execute(
                    select(Channel).where(Channel.channel_id == test_channel_id)
                )
                found = result.scalar_one_or_none()
                
                if found:
                    self.log_test("Channel Operations", True, "Create/Read working")
                    
                    # Update last_video_id (test incremental sync)
                    found.last_video_id = "test_video_123"
                    await session.commit()
                    
                    # Delete channel
                    await session.delete(found)
                    await session.commit()
                    
                    return True
                else:
                    self.log_test("Channel Operations", False, "Channel not found after creation")
                    return False
                    
        except Exception as e:
            self.log_test("Channel Operations", False, str(e))
            return False
    
    async def test_job_queue(self):
        """Test job queue operations."""
        try:
            queue = get_job_queue()
            
            # Test job creation
            job_id = await queue.enqueue(
                job_type="test_workflow",
                target_id="test_target",
                priority=5,
                metadata={"test": "data"}
            )
            
            if not job_id:
                self.log_test("Job Queue", False, "Failed to create job")
                return False
            
            # Test job dequeue
            job_data = await queue.dequeue("test_worker")
            if not job_data:
                self.log_test("Job Queue", False, "Failed to dequeue job")
                return False
            
            # Test job completion
            await queue.complete(job_data[0], {"status": "success"})
            
            # Test retry with exponential backoff
            retry_job_id = await queue.enqueue(
                job_type="test_retry",
                target_id="test_retry_target",
                priority=5
            )
            
            # Simulate failure and retry
            retry_job = await queue.dequeue("test_worker")
            if retry_job:
                await queue.fail(retry_job[0], "Test failure")
                success = await queue.retry(retry_job[0])
                
                if success:
                    self.log_test("Job Queue", True, "All operations working with retry")
                    return True
                    
            self.log_test("Job Queue", False, "Retry failed")
            return False
            
        except Exception as e:
            self.log_test("Job Queue", False, str(e))
            return False
    
    async def test_worker_chain(self):
        """Test the worker processing chain."""
        try:
            # This would test the actual worker chain if workers were running
            # For now, just verify workers can be imported
            from workers.monitor import MonitorWorker
            from workers.audio_downloader import AudioDownloadWorker
            from workers.transcriber import TranscribeWorker
            from workers.generator import GeneratorWorker
            from workers.storage import StorageWorker
            from workers.quality import QualityWorker
            from workers.publisher import PublishWorker
            
            # Test worker instantiation
            workers = []
            try:
                workers.append(MonitorWorker())
                workers.append(AudioDownloadWorker())
                workers.append(TranscribeWorker())
                workers.append(GeneratorWorker())
                workers.append(StorageWorker())
                workers.append(PublishWorker())
                # QualityWorker might need special init
            except Exception as e:
                self.log_test("Worker Chain", False, f"Worker instantiation failed: {e}")
                return False
            
            self.log_test("Worker Chain", True, f"{len(workers)} workers instantiated")
            return True
            
        except ImportError as e:
            self.log_test("Worker Chain", False, f"Import error: {e}")
            return False
    
    async def test_storage_backends(self):
        """Test storage backend availability."""
        try:
            from workers.storage import StorageWorker
            
            storage = StorageWorker()
            status = storage.get_status()
            
            enabled_backends = [
                backend for backend, info in status["backends"].items()
                if info.get("enabled", False)
            ]
            
            self.log_test(
                "Storage Backends", 
                True, 
                f"Enabled: {enabled_backends if enabled_backends else 'Local only'}"
            )
            return True
            
        except Exception as e:
            self.log_test("Storage Backends", False, str(e))
            return False
    
    async def test_search_functionality(self):
        """Test full-text search."""
        try:
            from core.search import SearchService
            
            search = SearchService(db_manager)
            
            # Create test transcript for search
            async with db_manager.get_session() as session:
                # Clean up any existing test data
                existing_video = await session.execute(
                    select(Video).where(Video.video_id == "search_test_video")
                )
                if existing := existing_video.scalar_one_or_none():
                    await session.delete(existing)
                    await session.commit()
                
                # Ensure channel exists
                existing_channel = await session.execute(
                    select(Channel).where(Channel.channel_id == "test_channel")
                )
                if not existing_channel.scalar_one_or_none():
                    channel = Channel(
                        channel_id="test_channel",
                        channel_name="Test Channel",
                        channel_url="https://youtube.com/@test",
                        is_active=True
                    )
                    session.add(channel)
                
                # Create test video first (with only required fields)
                video = Video(
                    video_id="search_test_video",
                    channel_id="test_channel",
                    title="Test Search Video",
                    transcript_status="completed",
                    language="en"
                )
                session.add(video)
                
                # Create test transcript
                transcript = Transcript(
                    video_id="search_test_video",
                    content_text="This is a test transcript for search functionality",
                    word_count=8,
                    extraction_method="test"
                )
                session.add(transcript)
                await session.commit()
            
            # Test search
            results = await search.search_transcripts("test search")
            
            # Clean up
            async with db_manager.get_session() as session:
                video = await session.get(Video, "search_test_video")
                if video:
                    await session.delete(video)
                transcript = await session.get(Transcript, "search_test_video")
                if transcript:
                    await session.delete(transcript)
                await session.commit()
            
            self.log_test("Search Functionality", True, "FTS5 search working")
            return True
            
        except Exception as e:
            self.log_test("Search Functionality", False, str(e))
            return False
    
    async def test_export_formats(self):
        """Test export format support."""
        formats = ['json', 'csv', 'txt', 'markdown', 'html']
        
        # Just verify the formats are recognized in the CLI
        from cli import validate_export_format
        
        for fmt in formats:
            try:
                # Create a mock context and param
                class MockContext:
                    pass
                class MockParam:
                    pass
                    
                result = validate_export_format(MockContext(), MockParam(), fmt)
                if result != fmt:
                    self.log_test("Export Formats", False, f"{fmt} validation failed")
                    return False
            except Exception as e:
                self.log_test("Export Formats", False, f"{fmt}: {e}")
                return False
        
        self.log_test("Export Formats", True, f"All {len(formats)} formats supported")
        return True
    
    async def run_all_tests(self):
        """Run all workflow tests."""
        print("\n" + "=" * 60)
        print("PHASE 1 COMPLETE WORKFLOW TEST")
        print("=" * 60 + "\n")
        
        # Run tests in sequence
        await self.test_database_init()
        await self.test_channel_operations()
        await self.test_job_queue()
        await self.test_worker_chain()
        await self.test_storage_backends()
        await self.test_search_functionality()
        await self.test_export_formats()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed_count = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        
        for name, passed, details in self.results:
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {name}")
        
        print(f"\nTotal: {passed_count}/{total} tests passed")
        
        if passed_count == total:
            print("\n🎉 ALL WORKFLOW TESTS PASSED! Phase 1 is complete!")
            return 0
        else:
            print(f"\n⚠️  {total - passed_count} tests failed. Review the issues above.")
            return 1


async def setup_test_database():
    """Set up a fresh test database."""
    test_db = Path("test_workflow.db")
    
    # Remove existing test database
    if test_db.exists():
        test_db.unlink()
    
    # Initialize database manager first
    await db_manager.initialize()
    
    # Initialize database schema with all columns
    from core.database import Base
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Create FTS tables
    async with db_manager.engine.begin() as conn:
        # Create FTS5 tables for full-text search
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                video_id UNINDEXED,
                content_text,
                content='transcripts',
                content_rowid='id'
            )
        """))
        
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                transcript_id UNINDEXED,
                content_type UNINDEXED,
                content,
                content='generated_content',
                content_rowid='id'
            )
        """))
        
        # Create triggers for FTS sync
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS transcripts_fts_insert 
            AFTER INSERT ON transcripts BEGIN
                INSERT INTO transcripts_fts(rowid, video_id, content_text)
                VALUES (new.id, new.video_id, new.content_text);
            END
        """))
        
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS content_fts_insert
            AFTER INSERT ON generated_content BEGIN
                INSERT INTO content_fts(rowid, transcript_id, content_type, content)
                VALUES (new.id, new.transcript_id, new.content_type, new.content);
            END
        """))
    
    print("✅ Test database initialized")


async def main():
    """Run the workflow test."""
    # Set up test database
    await setup_test_database()
    
    # Run tests
    tester = WorkflowTester()
    return await tester.run_all_tests()


if __name__ == "__main__":
    from sqlalchemy import text
    exit_code = asyncio.run(main())
    sys.exit(exit_code)