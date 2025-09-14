# Job Queue System

A comprehensive SQLite-based job queue system for the yt-dl-sub project that provides reliable, priority-based async job processing with retry logic and error handling.

## Key Features

- **Priority-based processing** (1=highest, 10=lowest)
- **Exponential backoff retry mechanism** with configurable max retries
- **Worker assignment and tracking** with unique worker IDs
- **Comprehensive error handling** with detailed error messages
- **Race condition prevention** using SQLite row-level locking
- **Queue statistics and monitoring** for performance tracking
- **Automatic cleanup** of old completed jobs
- **Stale job recovery** for crash resilience
- **Duplicate job prevention** based on type and target ID

## Quick Start

```python
from core.queue import JobQueue, JobType, enqueue_job, get_queue_status
from core.database import db_manager

# Initialize database
await db_manager.initialize()

# Enqueue a job
job_id = await enqueue_job(
    JobType.DOWNLOAD_TRANSCRIPT.value,
    "video_id_123",
    priority=1,  # Highest priority
    max_retries=3
)

# Check queue status
status = await get_queue_status()
print(f"Pending: {status['pending']}, Processing: {status['processing']}")
```

## Job Types

The system supports these predefined job types:

- `DOWNLOAD_TRANSCRIPT` - Download video and extract transcript
- `PROCESS_CHANNEL` - Monitor channel for new videos
- `TRANSCRIBE_AUDIO` - Convert audio to transcript using Whisper
- `GENERATE_CONTENT` - Generate content from transcripts
- `SYNC_STORAGE` - Sync files to Google Drive/storage backends
- `QUALITY_CHECK` - Validate transcript/content quality
- `CLEANUP` - Cleanup old files and data

## Creating a Custom Worker

```python
from core.queue import QueueWorker, JobType

class MyWorker(QueueWorker):
    async def process_job(self, job_type: str, target_id: str, metadata: dict) -> bool:
        if job_type == JobType.DOWNLOAD_TRANSCRIPT.value:
            # Your processing logic here
            print(f"Processing {target_id}")
            await asyncio.sleep(1)  # Simulate work
            return True  # Success
        return False  # Unknown job type

# Start the worker
queue = JobQueue(db_manager)
worker = MyWorker(queue)
await worker.start()  # Runs until stopped
```

## API Integration

```python
# FastAPI endpoint example
@app.post("/api/download-transcript")
async def download_transcript(video_id: str, priority: int = 5):
    try:
        job_id = await enqueue_job(
            JobType.DOWNLOAD_TRANSCRIPT.value,
            video_id,
            priority=priority
        )
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/queue-status")
async def queue_status():
    return await get_queue_status()
```

## Database Schema

The queue uses the existing `jobs` table from `database.py`:

```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    worker_id VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME
);
```

## Error Handling & Retry Logic

- **Automatic retries** with exponential backoff (configurable)
- **Max retry limits** per job (default: 3)
- **Error message tracking** for debugging
- **Stale job recovery** for crashed workers
- **Graceful worker shutdown** with job completion

```python
# Configure retry behavior
queue = JobQueue(
    db_manager,
    retry_base_delay=1.0,        # Start with 1s delay
    retry_backoff_multiplier=2.0, # Double each retry
    job_timeout_minutes=30,      # Consider job stale after 30min
    max_retries=3               # Default max retries
)
```

## Monitoring & Maintenance

```python
# Get comprehensive queue statistics
stats = await queue.get_queue_stats()
print(f"Total jobs: {stats.total_jobs}")
print(f"Pending: {stats.pending_jobs}")
print(f"Active workers: {stats.active_workers}")

# Get failed jobs for investigation
failed_jobs = await queue.get_failed_jobs()
for job in failed_jobs:
    print(f"Job {job['id']}: {job['error_message']}")

# Cleanup old completed jobs (older than 7 days by default)
cleaned_count = await queue.cleanup_old_jobs()
print(f"Cleaned up {cleaned_count} old jobs")

# Recover stale jobs from crashed workers
recovered_count = await queue.recover_stale_jobs()
print(f"Recovered {recovered_count} stale jobs")
```

## Integration with BaseWorker

The queue system integrates seamlessly with the existing `BaseWorker` class:

```python
from workers.base import BaseWorker

class TranscriptWorker(BaseWorker):
    def validate_input(self, input_data: dict) -> bool:
        return "video_id" in input_data
    
    def execute(self, input_data: dict) -> dict:
        # Your transcript processing logic
        return {"transcript_path": "/path/to/transcript.txt"}
    
    def handle_error(self, error: Exception) -> dict:
        return {"error_type": type(error).__name__}

# Use with queue
class QueuedTranscriptWorker(QueueWorker):
    def __init__(self, queue):
        super().__init__(queue)
        self.transcript_worker = TranscriptWorker()
    
    async def process_job(self, job_type: str, target_id: str, metadata: dict) -> bool:
        result = self.transcript_worker.run({"video_id": target_id})
        return result["status"] == "success"
```

## Environment Variables

```bash
# Queue configuration
QUEUE_RETRY_BASE_DELAY=1.0      # Base retry delay in seconds
QUEUE_RETRY_MULTIPLIER=2.0      # Exponential backoff multiplier
QUEUE_JOB_TIMEOUT_MINUTES=30    # Job timeout for stale detection
QUEUE_CLEANUP_AFTER_DAYS=7      # Days before cleaning completed jobs
QUEUE_MAX_RETRIES=3            # Default max retries per job
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_queue.py
```

See working examples:

```bash
python3 examples/queue_usage.py
```

## Architecture Notes

- **Thread-safe**: Uses SQLite row-level locking to prevent race conditions
- **Fault-tolerant**: Handles worker crashes gracefully with job recovery
- **Scalable**: Supports multiple workers processing different job types
- **Observable**: Comprehensive logging and statistics for monitoring
- **Maintainable**: Clean separation between queue logic and job processing

The queue system is designed to work reliably in production environments while being simple enough for development and testing.