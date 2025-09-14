#!/usr/bin/env python3
"""
Test script for the job queue system.
Demonstrates basic queue operations and validates functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import create_database, db_manager
from core.queue import JobQueue, JobType, QueueWorker, enqueue_job, get_queue_status


class TestWorker(QueueWorker):
    """Test worker implementation."""
    
    async def process_job(self, job_type: str, target_id: str, metadata: dict) -> bool:
        """Process test jobs with simulated work."""
        self.logger.info(f"Processing {job_type} for {target_id}")
        
        # Simulate different processing times and success rates
        if job_type == JobType.DOWNLOAD_TRANSCRIPT.value:
            await asyncio.sleep(0.5)  # Simulate transcript download
            return True
        elif job_type == JobType.PROCESS_CHANNEL.value:
            await asyncio.sleep(1.0)  # Simulate channel processing
            return target_id != "fail_channel"  # Fail for specific test case
        elif job_type == JobType.TRANSCRIBE_AUDIO.value:
            await asyncio.sleep(0.3)  # Simulate transcription
            return True
        else:
            self.logger.warning(f"Unknown job type: {job_type}")
            return False


async def test_basic_queue_operations():
    """Test basic queue operations."""
    print("\n=== Testing Basic Queue Operations ===")
    
    # Initialize database
    print("Initializing database...")
    await db_manager.initialize()
    
    # Create job queue with initialized db_manager
    queue = JobQueue(db_manager)
    
    # Test enqueuing jobs
    print("Enqueuing test jobs...")
    job1 = await queue.enqueue(
        JobType.DOWNLOAD_TRANSCRIPT.value,
        "video123",
        priority=1
    )
    print(f"Enqueued job {job1}: download_transcript for video123")
    
    job2 = await queue.enqueue(
        JobType.PROCESS_CHANNEL.value,
        "channel456",
        priority=3
    )
    print(f"Enqueued job {job2}: process_channel for channel456")
    
    job3 = await queue.enqueue(
        JobType.TRANSCRIBE_AUDIO.value,
        "audio789",
        priority=2
    )
    print(f"Enqueued job {job3}: transcribe_audio for audio789")
    
    # Test duplicate job prevention
    duplicate_job = await queue.enqueue(
        JobType.DOWNLOAD_TRANSCRIPT.value,
        "video123",
        priority=1
    )
    print(f"Duplicate job ID: {duplicate_job} (should match {job1})")
    
    # Get queue stats
    stats = await queue.get_queue_stats()
    print(f"\nQueue stats: {stats.pending_jobs} pending, {stats.total_jobs} total")
    
    # Test dequeuing (should get highest priority first)
    print("\nDequeuing jobs by priority...")
    worker_id = "test-worker-1"
    
    for i in range(3):
        job_data = await queue.dequeue(worker_id)
        if job_data:
            job_id, job_type, target_id, metadata = job_data
            print(f"Dequeued job {job_id}: {job_type} for {target_id}")
            
            # Complete the job
            await queue.complete(job_id)
            print(f"Completed job {job_id}")
        else:
            print("No more jobs in queue")
    
    # Final stats
    final_stats = await queue.get_queue_stats()
    print(f"\nFinal stats: {final_stats.completed_jobs} completed, {final_stats.pending_jobs} pending")


async def test_retry_logic():
    """Test job retry logic with exponential backoff."""
    print("\n=== Testing Retry Logic ===")
    
    queue = JobQueue(db_manager, retry_base_delay=0.1, retry_backoff_multiplier=1.5)
    
    # Create a job that will fail
    fail_job = await queue.enqueue(
        JobType.PROCESS_CHANNEL.value,
        "fail_channel",
        priority=5,
        max_retries=2
    )
    print(f"Enqueued failing job {fail_job}")
    
    worker_id = "test-worker-retry"
    
    # Process and fail the job multiple times
    for attempt in range(4):  # More than max_retries
        job_data = await queue.dequeue(worker_id)
        if job_data:
            job_id, job_type, target_id, metadata = job_data
            print(f"Attempt {attempt + 1}: Processing job {job_id}")
            
            # Simulate failure
            await queue.fail(job_id, f"Test failure attempt {attempt + 1}")
            print(f"Failed job {job_id}")
            
            # Try to retry
            retried = await queue.retry(job_id)
            if retried:
                print(f"Job {job_id} queued for retry")
            else:
                print(f"Job {job_id} cannot be retried (max retries reached)")
                break
        else:
            print("No jobs available for retry test")
            break
    
    # Check failed jobs
    failed_jobs = await queue.get_failed_jobs()
    print(f"\nFailed jobs: {len(failed_jobs)}")
    for job in failed_jobs:
        print(f"  Job {job['id']}: {job['error_message']} (retries: {job['retry_count']}/{job['max_retries']})")


async def test_worker_integration():
    """Test integration with QueueWorker class."""
    print("\n=== Testing Worker Integration ===")
    
    queue = JobQueue(db_manager)
    
    # Enqueue some jobs
    jobs = []
    for i in range(5):
        job_id = await queue.enqueue(
            JobType.DOWNLOAD_TRANSCRIPT.value,
            f"video{i}",
            priority=i + 1
        )
        jobs.append(job_id)
    
    print(f"Enqueued {len(jobs)} jobs")
    
    # Create and start a worker
    worker = TestWorker(queue, poll_interval=0.1)
    
    # Start worker in background
    worker_task = asyncio.create_task(worker.start())
    
    # Let worker process jobs for a few seconds
    await asyncio.sleep(3.0)
    
    # Stop worker
    worker.stop()
    await worker_task
    
    # Check results
    stats = await queue.get_queue_stats()
    print(f"Worker processed jobs: {stats.completed_jobs} completed, {stats.pending_jobs} pending")


async def test_queue_cleanup():
    """Test queue cleanup functionality."""
    print("\n=== Testing Queue Cleanup ===")
    
    queue = JobQueue(db_manager, cleanup_after_days=0)  # Cleanup immediately for testing
    
    # Create some completed jobs
    for i in range(3):
        job_id = await queue.enqueue(
            JobType.DOWNLOAD_TRANSCRIPT.value,
            f"cleanup_test_{i}",
            priority=5
        )
        
        # Process job
        job_data = await queue.dequeue("cleanup_worker")
        if job_data:
            await queue.complete(job_data[0])
    
    stats_before = await queue.get_queue_stats()
    print(f"Before cleanup: {stats_before.completed_jobs} completed jobs")
    
    # Test dry run
    dry_run_count = await queue.cleanup_old_jobs(dry_run=True)
    print(f"Dry run would cleanup: {dry_run_count} jobs")
    
    # Actually cleanup
    cleaned_count = await queue.cleanup_old_jobs(dry_run=False)
    print(f"Actually cleaned up: {cleaned_count} jobs")
    
    stats_after = await queue.get_queue_stats()
    print(f"After cleanup: {stats_after.completed_jobs} completed jobs")


async def main():
    """Run all queue tests."""
    print("🚀 Starting Job Queue Tests")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        await test_basic_queue_operations()
        await test_retry_logic()
        await test_worker_integration()
        await test_queue_cleanup()
        
        print("\n✅ All tests completed successfully!")
        
        # Show final queue status
        print("\n=== Final Queue Status ===")
        status = await get_queue_status()
        for key, value in status.items():
            print(f"{key}: {value}")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())