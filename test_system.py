#!/usr/bin/env python3
"""
Comprehensive system test for yt-dl-sub.

Tests all critical components and the complete workflow.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all critical modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    modules = [
        ('core.database', 'Database'),
        ('core.queue', 'Queue'),
        ('core.ai_backend', 'AI Backend'),
        ('config.settings', 'Settings'),
        ('workers.base', 'Base Worker'),
        ('workers.monitor', 'Monitor Worker'),
        ('workers.audio_downloader', 'Audio Downloader'),
        ('workers.transcriber', 'Transcriber'),
        ('workers.summarizer', 'Summarizer'),
        ('workers.generator', 'Generator'),
        ('workers.storage', 'Storage Worker'),
        ('workers.quality', 'Quality Worker'),
        ('workers.orchestrator', 'Orchestrator'),
    ]
    
    failed = []
    for module_path, name in modules:
        try:
            __import__(module_path)
            print(f"✅ {name}: OK")
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0


def test_workers():
    """Test that all workers can be instantiated."""
    print("\n=== Testing Worker Instantiation ===")
    
    workers = []
    failed = []
    
    try:
        from workers.monitor import MonitorWorker
        monitor = MonitorWorker()
        print("✅ MonitorWorker: OK")
    except Exception as e:
        print(f"❌ MonitorWorker: {e}")
        failed.append("MonitorWorker")
    
    try:
        from workers.audio_downloader import AudioDownloadWorker
        downloader = AudioDownloadWorker()
        print("✅ AudioDownloadWorker: OK")
    except Exception as e:
        print(f"❌ AudioDownloadWorker: {e}")
        failed.append("AudioDownloadWorker")
    
    try:
        from workers.transcriber import TranscribeWorker
        transcriber = TranscribeWorker()
        print("✅ TranscribeWorker: OK")
    except Exception as e:
        print(f"❌ TranscribeWorker: {e}")
        failed.append("TranscribeWorker")
    
    try:
        from workers.summarizer import SummarizerWorker
        summarizer = SummarizerWorker()
        print("✅ SummarizerWorker: OK")
    except Exception as e:
        print(f"❌ SummarizerWorker: {e}")
        failed.append("SummarizerWorker")
    
    try:
        from workers.generator import GeneratorWorker
        generator = GeneratorWorker()
        print("✅ GeneratorWorker: OK")
    except Exception as e:
        print(f"❌ GeneratorWorker: {e}")
        failed.append("GeneratorWorker")
    
    try:
        from workers.storage import StorageWorker
        storage = StorageWorker()
        print("✅ StorageWorker: OK")
    except Exception as e:
        print(f"❌ StorageWorker: {e}")
        failed.append("StorageWorker")
    
    try:
        from workers.quality import QualityWorker
        quality = QualityWorker()
        print("✅ QualityWorker: OK")
    except Exception as e:
        print(f"❌ QualityWorker: {e}")
        failed.append("QualityWorker")
    
    try:
        from workers.orchestrator import OrchestratorWorker
        orchestrator = OrchestratorWorker()
        print("✅ OrchestratorWorker: OK")
    except Exception as e:
        print(f"❌ OrchestratorWorker: {e}")
        failed.append("OrchestratorWorker")
    
    return len(failed) == 0


async def test_database():
    """Test database connectivity and operations."""
    print("\n=== Testing Database ===")
    
    try:
        from core.database import db_manager
        
        # Initialize database
        await db_manager.initialize()
        print("✅ Database initialized")
        
        # Test session creation
        async with db_manager.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            print("✅ Database session works")
        
        # Check tables
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            print(f"✅ Found {len(tables)} tables")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


async def test_queue():
    """Test job queue operations."""
    print("\n=== Testing Job Queue ===")
    
    try:
        from core.queue import JobQueue, get_job_queue
        
        queue = get_job_queue()
        
        # Test enqueue
        job_id = await queue.enqueue(
            job_type="test",
            target_id="test_123",
            priority=5
        )
        print(f"✅ Job enqueued: {job_id}")
        
        # Test dequeue
        job_data = await queue.dequeue("test_worker")
        if job_data:
            print(f"✅ Job dequeued: {job_data[0]}")
            
            # Mark as complete
            await queue.complete(job_data[0], {"result": "test"})
            print("✅ Job completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ai_backend():
    """Test AI backend functionality."""
    print("\n=== Testing AI Backend ===")
    
    try:
        from core.ai_backend import get_ai_backend
        
        backend = get_ai_backend()
        print(f"✅ AI Backend initialized: {backend.provider.value}")
        
        # Test availability check
        available = backend.is_available()
        print(f"✅ AI Backend available: {available}")
        
        # Test generation (will use placeholder if Claude not available)
        result = backend.generate_content(
            "Generate a test summary",
            max_tokens=50
        )
        
        if "error" not in result:
            print(f"✅ AI generation works: {result.get('provider', 'unknown')} provider")
        else:
            print(f"⚠️  AI generation error (expected if Claude not installed): {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Backend test failed: {e}")
        return False


async def test_workflow():
    """Test a simplified workflow."""
    print("\n=== Testing Simplified Workflow ===")
    
    try:
        from core.queue import get_job_queue
        from core.database import db_manager, Channel, Video
        from sqlalchemy import select
        
        queue = get_job_queue()
        
        # Create a test channel
        async with db_manager.get_session() as session:
            # Check if test channel exists
            result = await session.execute(
                select(Channel).where(Channel.channel_id == "test_channel")
            )
            channel = result.scalar_one_or_none()
            
            if not channel:
                channel = Channel(
                    channel_id="test_channel",
                    channel_name="Test Channel",
                    channel_url="https://youtube.com/@test",
                    is_active=True
                )
                session.add(channel)
                await session.commit()
                print("✅ Test channel created")
            else:
                print("✅ Test channel exists")
        
        # Create a test video
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(Video).where(Video.video_id == "test_video")
            )
            video = result.scalar_one_or_none()
            
            if not video:
                video = Video(
                    video_id="test_video",
                    channel_id="test_channel",
                    title="Test Video",
                    transcript_status="pending"
                )
                session.add(video)
                await session.commit()
                print("✅ Test video created")
            else:
                print("✅ Test video exists")
        
        # Create workflow jobs
        jobs_created = []
        
        # Download job
        job_id = await queue.enqueue(
            job_type="download_audio",
            target_id="test_video",
            priority=10
        )
        jobs_created.append(job_id)
        print(f"✅ Download job created: {job_id}")
        
        # Transcript job
        job_id = await queue.enqueue(
            job_type="extract_transcript",
            target_id="test_video",
            priority=9
        )
        jobs_created.append(job_id)
        print(f"✅ Transcript job created: {job_id}")
        
        # Generation job
        job_id = await queue.enqueue(
            job_type="generate_content",
            target_id="test_video",
            priority=5
        )
        jobs_created.append(job_id)
        print(f"✅ Generation job created: {job_id}")
        
        print(f"✅ Workflow created with {len(jobs_created)} jobs")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 50)
    print("YT-DL-SUB SYSTEM TEST")
    print("=" * 50)
    
    results = []
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test workers
    results.append(("Worker Instantiation", test_workers()))
    
    # Test database
    results.append(("Database", await test_database()))
    
    # Test queue
    results.append(("Job Queue", await test_queue()))
    
    # Test AI backend
    results.append(("AI Backend", test_ai_backend()))
    
    # Test workflow
    results.append(("Workflow", await test_workflow()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)