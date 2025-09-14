#!/usr/bin/env python3
"""
Example usage of the JobQueue system for the yt-dl-sub project.
Demonstrates how to integrate the queue with existing workers and API endpoints.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db_manager
from core.queue import JobQueue, JobType, QueueWorker, enqueue_job, get_queue_status
from workers.base import BaseWorker, WorkerStatus


class TranscriptDownloadWorker(BaseWorker):
    """Example worker that integrates with the job queue system."""
    
    def __init__(self):
        super().__init__(
            name="transcript_downloader",
            max_retries=3,
            retry_delay=2.0
        )
    
    def validate_input(self, input_data: dict) -> bool:
        """Validate that we have the required video_id."""
        return "video_id" in input_data and input_data["video_id"]
    
    def execute(self, input_data: dict) -> dict:
        """Execute transcript download logic."""
        video_id = input_data["video_id"]
        
        # Simulate transcript download process
        self.log_with_context(f"Starting transcript download for {video_id}")
        
        # Your actual implementation would go here:
        # 1. Download audio using yt-dlp
        # 2. Extract transcript using Whisper
        # 3. Save to storage
        # 4. Update database
        
        # For demo, just simulate success
        import time
        time.sleep(0.5)  # Simulate processing time
        
        return {
            "video_id": video_id,
            "transcript_path": f"/path/to/{video_id}_transcript.txt",
            "audio_path": f"/path/to/{video_id}_audio.opus",
            "quality_score": 0.95
        }
    
    def handle_error(self, error: Exception) -> dict:
        """Handle errors during transcript download."""
        return {
            "error_type": type(error).__name__,
            "error_details": str(error),
            "suggested_action": "retry_with_different_method",
            "fallback_available": True
        }


class QueuedTranscriptWorker(QueueWorker):
    """Queue worker that processes transcript download jobs."""
    
    def __init__(self, queue: JobQueue):
        super().__init__(queue, poll_interval=1.0)
        self.transcript_worker = TranscriptDownloadWorker()
    
    async def process_job(self, job_type: str, target_id: str, metadata: dict) -> bool:
        """Process different job types."""
        if job_type == JobType.DOWNLOAD_TRANSCRIPT.value:
            return await self._process_transcript_job(target_id, metadata)
        elif job_type == JobType.PROCESS_CHANNEL.value:
            return await self._process_channel_job(target_id, metadata)
        else:
            self.logger.warning(f"Unknown job type: {job_type}")
            return False
    
    async def _process_transcript_job(self, video_id: str, metadata: dict) -> bool:
        """Process a transcript download job."""
        try:
            # Use the BaseWorker to process the job
            input_data = {"video_id": video_id, **metadata}
            result = self.transcript_worker.run(input_data)
            
            if result["status"] == WorkerStatus.SUCCESS.value:
                self.logger.info(f"Successfully processed transcript for {video_id}")
                return True
            else:
                self.logger.error(f"Transcript processing failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during transcript processing: {e}")
            return False
    
    async def _process_channel_job(self, channel_id: str, metadata: dict) -> bool:
        """Process a channel monitoring job."""
        # Simulate channel processing
        self.logger.info(f"Processing channel {channel_id}")
        await asyncio.sleep(1.0)  # Simulate RSS checking, video discovery
        return True


async def demonstrate_api_integration():
    """Demonstrate how to integrate queue with API endpoints."""
    print("\n=== API Integration Example ===")
    
    # Simulate API endpoint for downloading transcript
    async def api_download_transcript(video_id: str, priority: int = 5) -> dict:
        """API endpoint to queue transcript download."""
        try:
            job_id = await enqueue_job(
                JobType.DOWNLOAD_TRANSCRIPT.value,
                video_id,
                priority=priority
            )
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Transcript download queued for video {video_id}",
                "estimated_wait": "2-5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Simulate API endpoint to check queue status
    async def api_queue_status() -> dict:
        """API endpoint to get queue status."""
        return await get_queue_status()
    
    # Demo API calls
    print("Queueing transcript downloads via API...")
    for i in range(3):
        video_id = f"demoVid{i}"
        response = await api_download_transcript(video_id, priority=i+1)
        print(f"API Response: {response}")
    
    print("\nQueue status via API:")
    status = await api_queue_status()
    print(f"Status: {status}")


async def demonstrate_worker_lifecycle():
    """Demonstrate the complete worker lifecycle."""
    print("\n=== Worker Lifecycle Example ===")
    
    # Initialize database and queue
    await db_manager.initialize()
    queue = JobQueue(db_manager)
    
    # Queue some jobs
    print("Queueing jobs...")
    jobs = []
    for i in range(5):
        job_id = await queue.enqueue(
            JobType.DOWNLOAD_TRANSCRIPT.value,
            f"video_{i}",
            priority=(i % 3) + 1  # Vary priorities
        )
        jobs.append(job_id)
    
    print(f"Queued {len(jobs)} jobs")
    
    # Start worker
    print("Starting worker...")
    worker = QueuedTranscriptWorker(queue)
    worker_task = asyncio.create_task(worker.start())
    
    # Monitor progress
    for i in range(10):  # Check status every second for 10 seconds
        await asyncio.sleep(1)
        stats = await queue.get_queue_stats()
        print(f"Progress: {stats.completed_jobs}/{stats.total_jobs} completed, {stats.pending_jobs} pending")
        
        if stats.pending_jobs == 0:
            break
    
    # Stop worker
    print("Stopping worker...")
    worker.stop()
    await worker_task
    
    # Final stats
    final_stats = await queue.get_queue_stats()
    print(f"Final: {final_stats.completed_jobs} completed, {final_stats.failed_jobs} failed")


async def demonstrate_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n=== Error Handling Example ===")
    
    queue = JobQueue(db_manager, retry_base_delay=0.5)
    
    # Create a job that will fail
    job_id = await queue.enqueue(
        JobType.PROCESS_CHANNEL.value,
        "problematic_channel",
        priority=1,
        max_retries=2
    )
    
    print(f"Created problematic job {job_id}")
    
    # Manually process and fail it to demonstrate retry
    for attempt in range(4):
        job_data = await queue.dequeue("error_demo_worker")
        if job_data:
            job_id, job_type, target_id, metadata = job_data
            print(f"Attempt {attempt + 1}: Processing {target_id}")
            
            # Simulate failure
            await queue.fail(job_id, f"Simulated error on attempt {attempt + 1}")
            
            # Try to retry
            can_retry = await queue.retry(job_id)
            if can_retry:
                print(f"Job queued for retry")
            else:
                print(f"Job exhausted retries")
                break
        else:
            print("No more jobs to process")
            break
    
    # Check failed jobs
    failed_jobs = await queue.get_failed_jobs()
    print(f"Failed jobs: {len(failed_jobs)}")


async def main():
    """Run all demonstrations."""
    print("🎯 Job Queue Integration Examples")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        await demonstrate_api_integration()
        await demonstrate_worker_lifecycle()
        await demonstrate_error_handling()
        
        print("\n✅ All demonstrations completed!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        raise
    
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())