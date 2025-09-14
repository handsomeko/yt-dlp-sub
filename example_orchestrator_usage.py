#!/usr/bin/env python3
"""
Example usage of the OrchestratorWorker for coordinating YouTube content processing.

This script demonstrates different ways to use the orchestrator worker:
1. Single job processing
2. Batch processing
3. Continuous processing mode
4. Manual job creation and orchestration
"""

import asyncio
import logging
from core.queue import get_job_queue, enqueue_job
from workers.orchestrator import OrchestratorWorker, start_orchestrator, create_orchestrator

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def example_1_single_batch():
    """Example 1: Process a single batch of jobs."""
    print("=== Example 1: Single Batch Processing ===")
    
    # Create some jobs first
    job_queue = get_job_queue()
    
    # Add a few channel check jobs
    channels = ["UC1234567890", "UC0987654321", "UC1122334455"]
    
    print(f"Creating {len(channels)} channel check jobs...")
    for channel_id in channels:
        job_id = await enqueue_job(
            job_type="check_channel",
            target_id=channel_id,
            priority=5
        )
        print(f"  Created job {job_id} for channel {channel_id}")
    
    # Process the jobs in a single batch
    orchestrator = create_orchestrator(
        max_concurrent_jobs=2,
        polling_interval=1.0
    )
    
    print("Processing jobs in single batch mode...")
    result = await orchestrator.execute({
        "continuous_mode": False,
        "max_jobs": len(channels)
    })
    
    print(f"Batch processing completed: {result['processing_stats']}")


async def example_2_continuous_mode():
    """Example 2: Continuous processing mode with job limit."""
    print("\n=== Example 2: Continuous Processing Mode ===")
    
    # Create jobs continuously while orchestrator runs
    job_queue = get_job_queue()
    
    # Create initial batch of jobs
    video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0", "jNQXAC9IVRw"]
    
    print(f"Creating {len(video_ids)} download jobs...")
    for video_id in video_ids:
        job_id = await enqueue_job(
            job_type="download_transcript",
            target_id=video_id,
            priority=3  # Higher priority
        )
        print(f"  Created download job {job_id} for video {video_id}")
    
    # Start orchestrator in continuous mode
    print("Starting continuous processing (max 3 jobs)...")
    result = await start_orchestrator(
        continuous_mode=True,
        max_jobs=3
    )
    
    print(f"Continuous processing completed: {result['processing_stats']}")


async def example_3_filtered_processing():
    """Example 3: Process only specific job types."""
    print("\n=== Example 3: Filtered Job Processing ===")
    
    job_queue = get_job_queue()
    
    # Create mixed job types
    print("Creating mixed job types...")
    
    # Channel check jobs
    await enqueue_job("check_channel", "UC_example_1", priority=5)
    await enqueue_job("check_channel", "UC_example_2", priority=5)
    
    # Download jobs
    await enqueue_job("download_transcript", "example_video_1", priority=3)
    await enqueue_job("download_transcript", "example_video_2", priority=3)
    
    # Process only channel check jobs
    orchestrator = create_orchestrator()
    
    print("Processing only channel check jobs...")
    result = await orchestrator.execute({
        "continuous_mode": False,
        "max_jobs": 10,
        "job_types": ["check_channel"]  # Filter to only process these
    })
    
    print(f"Filtered processing result: {result['processing_stats']}")
    
    # Check remaining jobs in queue
    remaining_jobs = await job_queue.get_pending_jobs(limit=10)
    print(f"Remaining jobs in queue: {len(remaining_jobs)}")
    for job in remaining_jobs:
        print(f"  - {job['job_type']}: {job['target_id']}")


async def example_4_orchestrator_monitoring():
    """Example 4: Monitor orchestrator status during processing."""
    print("\n=== Example 4: Orchestrator Status Monitoring ===")
    
    # Create orchestrator
    orchestrator = create_orchestrator(
        max_concurrent_jobs=1,
        polling_interval=2.0
    )
    
    # Add some jobs
    job_queue = get_job_queue()
    for i in range(5):
        await enqueue_job("check_channel", f"UC_monitor_test_{i}", priority=5)
    
    # Monitor status during processing
    async def monitor_status():
        """Monitor and display orchestrator status."""
        while orchestrator.status.value in ["starting", "running"]:
            status_info = orchestrator.get_status_info()
            print(f"Status: {status_info['status']} | "
                  f"Uptime: {status_info['uptime_seconds']:.1f}s | "
                  f"Jobs processed: {status_info['stats']['total_jobs_processed']}")
            await asyncio.sleep(1)
    
    # Start monitoring and processing concurrently
    print("Starting orchestrator with status monitoring...")
    
    monitor_task = asyncio.create_task(monitor_status())
    
    # Process jobs
    process_task = asyncio.create_task(orchestrator.execute({
        "continuous_mode": True,
        "max_jobs": 5
    }))
    
    # Wait for processing to complete
    result = await process_task
    monitor_task.cancel()
    
    print(f"Final status: {orchestrator.get_status_info()}")


async def example_5_error_recovery():
    """Example 5: Demonstrate error handling and recovery."""
    print("\n=== Example 5: Error Handling and Recovery ===")
    
    orchestrator = create_orchestrator()
    
    # Simulate different types of errors
    test_errors = [
        Exception("Database connection lost"),
        Exception("Queue timeout occurred"),
        Exception("Network unavailable"),
        Exception("Rate limit exceeded")
    ]
    
    print("Testing error handling capabilities...")
    for i, error in enumerate(test_errors, 1):
        print(f"\nTest {i}: Handling '{error}'")
        error_result = orchestrator.handle_error(error)
        
        print(f"  Category: {error_result['error_category']}")
        print(f"  Recovery Action: {error_result['recovery_action']}")
        print(f"  Should Stop: {error_result['should_stop']}")
        print(f"  Retry Delay: {error_result['retry_delay_seconds']}s")


async def example_6_manual_job_creation():
    """Example 6: Manual job creation and processing workflow."""
    print("\n=== Example 6: Manual Job Creation Workflow ===")
    
    # Simulate a typical workflow:
    # 1. Monitor channels for new videos
    # 2. Download transcripts for new videos
    # 3. Process the results
    
    job_queue = get_job_queue()
    
    print("Step 1: Creating channel monitoring jobs...")
    channels = ["UC_tech_news", "UC_education", "UC_entertainment"]
    
    for channel in channels:
        await enqueue_job("check_channel", channel, priority=1)  # High priority
    
    print("Step 2: Creating some download jobs...")
    videos = ["sample_video_1", "sample_video_2", "sample_video_3"]
    
    for video in videos:
        await enqueue_job("download_transcript", video, priority=3)  # Medium priority
    
    print("Step 3: Processing with orchestrator...")
    
    # Create orchestrator with specific configuration
    orchestrator = OrchestratorWorker(
        max_concurrent_jobs=2,
        polling_interval=1.0,
        health_check_interval=30.0,
        worker_id="manual-workflow-orchestrator"
    )
    
    # Get queue stats before processing
    before_stats = await job_queue.get_queue_stats()
    print(f"Queue before processing: {before_stats.pending_jobs} pending, "
          f"{before_stats.total_jobs} total")
    
    # Process the jobs
    result = await orchestrator.execute({
        "continuous_mode": False,
        "max_jobs": len(channels) + len(videos)
    })
    
    # Get queue stats after processing
    after_stats = await job_queue.get_queue_stats()
    print(f"Queue after processing: {after_stats.pending_jobs} pending, "
          f"{after_stats.total_jobs} total")
    
    print(f"Workflow completed: {result['processing_stats']}")


async def main():
    """Run all examples."""
    print("Starting OrchestratorWorker Usage Examples\n")
    
    try:
        await example_1_single_batch()
        await example_2_continuous_mode()
        await example_3_filtered_processing()
        await example_4_orchestrator_monitoring()
        await example_5_error_recovery()
        await example_6_manual_job_creation()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())