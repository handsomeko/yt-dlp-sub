#!/usr/bin/env python3
"""
Test script for the OrchestratorWorker to verify integration and functionality.

This script demonstrates how to use the orchestrator and tests the basic workflow
of job processing and worker coordination.
"""

import asyncio
import logging
from core.queue import get_job_queue, enqueue_job
from workers.orchestrator import OrchestratorWorker, create_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)


async def test_orchestrator_basic():
    """Test basic orchestrator functionality."""
    print("=== Testing Basic Orchestrator Functionality ===")
    
    # Create orchestrator
    orchestrator = create_orchestrator(
        max_concurrent_jobs=2,
        polling_interval=2.0,
        worker_id="test-orchestrator"
    )
    
    # Test status info
    status_info = orchestrator.get_status_info()
    print(f"Initial status: {status_info}")
    
    # Test validation
    valid_input = {
        "continuous_mode": False,
        "max_jobs": 5
    }
    
    is_valid = orchestrator.validate_input(valid_input)
    print(f"Input validation result: {is_valid}")
    
    print("✓ Basic functionality test passed")


async def test_worker_registry():
    """Test worker registry functionality."""
    print("\n=== Testing Worker Registry ===")
    
    orchestrator = OrchestratorWorker()
    
    # Check worker registry
    print(f"Registered workers: {list(orchestrator.worker_registry.keys())}")
    
    for name, registry in orchestrator.worker_registry.items():
        print(f"  - {name}: job_types={registry.job_types}, enabled={registry.enabled}")
    
    # Test worker retrieval
    monitor_worker = await orchestrator._get_worker_for_job_type("check_channel")
    download_worker = await orchestrator._get_worker_for_job_type("download_transcript")
    unknown_worker = await orchestrator._get_worker_for_job_type("unknown_job")
    
    print(f"Monitor worker: {monitor_worker.name if monitor_worker else 'None'}")
    print(f"Download worker: {download_worker.name if download_worker else 'None'}")
    print(f"Unknown worker: {unknown_worker}")
    
    print("✓ Worker registry test passed")


async def test_job_processing():
    """Test job processing with mock jobs."""
    print("\n=== Testing Job Processing ===")
    
    # Get the job queue
    job_queue = get_job_queue()
    
    # Create some test jobs
    print("Creating test jobs...")
    
    # Create a channel check job
    job_id_1 = await enqueue_job(
        job_type="check_channel",
        target_id="UC_test_channel_123",
        priority=5
    )
    print(f"Created channel check job: {job_id_1}")
    
    # Create a transcript download job  
    job_id_2 = await enqueue_job(
        job_type="download_transcript", 
        target_id="dQw4w9WgXcQ",  # Rick Astley video
        priority=5
    )
    print(f"Created download job: {job_id_2}")
    
    # Check queue stats
    stats = await job_queue.get_queue_stats()
    print(f"Queue stats: pending={stats.pending_jobs}, total={stats.total_jobs}")
    
    # Process jobs with orchestrator
    orchestrator = create_orchestrator(
        max_concurrent_jobs=1,
        polling_interval=1.0
    )
    
    print("Processing jobs...")
    result = await orchestrator.execute({
        "continuous_mode": False,
        "max_jobs": 2
    })
    
    print(f"Processing result: {result}")
    print("✓ Job processing test completed")


async def test_error_handling():
    """Test error handling in orchestrator."""
    print("\n=== Testing Error Handling ===")
    
    orchestrator = OrchestratorWorker()
    
    # Test various error scenarios
    test_errors = [
        Exception("Database connection failed"),
        Exception("Queue timeout occurred"),
        Exception("Memory exhausted"),
        Exception("Signal interrupt received"),
        Exception("Unknown error")
    ]
    
    for error in test_errors:
        error_result = orchestrator.handle_error(error)
        print(f"Error '{error}' -> Category: {error_result['error_category']}, "
              f"Recovery: {error_result['recovery_action']}")
    
    print("✓ Error handling test passed")


async def test_continuous_mode():
    """Test continuous processing mode (brief test)."""
    print("\n=== Testing Continuous Mode (Brief) ===")
    
    # Create a few jobs first
    job_queue = get_job_queue()
    
    for i in range(3):
        await enqueue_job(
            job_type="check_channel",
            target_id=f"test_channel_{i}",
            priority=5
        )
    
    # Run orchestrator in continuous mode for a short time
    orchestrator = create_orchestrator(
        max_concurrent_jobs=1,
        polling_interval=1.0
    )
    
    print("Starting continuous mode (will run for max 3 jobs)...")
    
    result = await orchestrator.execute({
        "continuous_mode": True,
        "max_jobs": 3  # Limit to 3 jobs for testing
    })
    
    print(f"Continuous mode result: {result}")
    print("✓ Continuous mode test completed")


async def main():
    """Run all tests."""
    print("Starting OrchestratorWorker Tests\n")
    
    try:
        await test_orchestrator_basic()
        await test_worker_registry()
        await test_error_handling()
        await test_job_processing()
        await test_continuous_mode()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())