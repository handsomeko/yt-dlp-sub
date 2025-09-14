# YouTube Content Intelligence Workers

This directory contains the worker implementations for the YouTube Content Intelligence & Repurposing Platform. The workers handle different aspects of content processing in a coordinated, queue-based architecture.

## Architecture Overview

```
JobQueue -> OrchestratorWorker -> Specialized Workers
                |
                ├── MonitorWorker (Channel RSS monitoring)
                ├── DownloadWorker (Transcript extraction)
                └── [Future: Other workers...]
```

## Workers

### 1. BaseWorker (`base.py`)

Abstract base class providing:
- Consistent error handling and logging
- Retry logic with exponential backoff  
- Execution timing and metrics
- Standardized result format

**Key Features:**
- `WorkerStatus` enum for execution results
- Context-aware logging with `log_with_context()`
- Automatic retry with `retry_with_backoff()`
- Abstract methods: `validate_input()`, `execute()`, `handle_error()`

### 2. OrchestratorWorker (`orchestrator.py`)

Main coordinator that manages job flow between workers.

**Key Features:**
- Worker registry pattern for dynamic worker management
- Job routing based on `job_type`
- Concurrent job processing with configurable limits
- Continuous and batch processing modes
- Health monitoring and statistics tracking
- Graceful shutdown handling with signal support

**Supported Job Types:**
- `check_channel` → MonitorWorker
- `check_all_channels` → MonitorWorker  
- `download_transcript` → DownloadWorker

### 3. MonitorWorker (`monitor.py`)

Monitors YouTube channel RSS feeds for new videos.

**Key Features:**
- RSS feed parsing with `feedparser`
- New video detection via `last_video_id` tracking
- Rate limiting to respect YouTube's limits
- Automatic job creation for new videos
- Network error handling and categorization

**Input Format:**
```python
{
    "channel_id": "UC...",  # Specific channel
    # OR
    "check_all": True       # Check all active channels
}
```

### 4. DownloadWorker (`downloader.py`) 

Downloads video transcripts using fallback chain.

**Key Features:**
- Primary: `yt-dlp` with subtitle extraction
- Fallback: `youtube-transcript-api`
- Automatic SRT to plain text conversion
- File path sanitization and organized storage
- Metadata extraction and database storage

**Input Format:**
```python
{
    "video_id": "dQw4w9WgXcQ",
    "video_url": "https://youtube.com/watch?v=...",
    "channel_id": "UC..."  # Optional
}
```

## Usage Examples

### Running the Orchestrator

#### 1. Simple Continuous Mode
```bash
python run_orchestrator.py
```

#### 2. Batch Processing
```bash
python run_orchestrator.py --batch --max-jobs 50
```

#### 3. With Configuration File
```bash
python run_orchestrator.py --config-file orchestrator_config.json
```

#### 4. Filtered Job Processing
```bash
python run_orchestrator.py --job-types check_channel download_transcript
```

### Programmatic Usage

#### Basic Orchestrator Setup
```python
import asyncio
from workers.orchestrator import create_orchestrator

async def main():
    orchestrator = create_orchestrator(
        max_concurrent_jobs=3,
        polling_interval=5.0
    )
    
    result = await orchestrator.execute({
        "continuous_mode": True,
        "max_jobs": 100
    })
    
    print(f"Processing completed: {result}")

asyncio.run(main())
```

#### Adding Jobs to Queue
```python
from core.queue import enqueue_job

# Add channel monitoring job
job_id = await enqueue_job(
    job_type="check_channel",
    target_id="UC1234567890",
    priority=5
)

# Add transcript download job
job_id = await enqueue_job(
    job_type="download_transcript", 
    target_id="dQw4w9WgXcQ",
    priority=3
)
```

#### Direct Worker Usage
```python
from workers.monitor import MonitorWorker

worker = MonitorWorker()
result = await worker.run_async({
    "channel_id": "UC1234567890",
    "check_all": False
})
```

## Configuration

### Orchestrator Configuration
```json
{
  "continuous_mode": true,
  "max_jobs": null,
  "worker_id": "prod-orchestrator-1", 
  "max_concurrent_jobs": 5,
  "polling_interval": 3.0,
  "health_check_interval": 30.0,
  "max_retries": 3,
  "retry_delay": 5.0,
  "log_level": "INFO",
  "job_types": []
}
```

### Environment Variables
- `DATABASE_URL`: Database connection string
- `STORAGE_PATH`: Base path for downloaded files
- `LOG_LEVEL`: Global logging level
- `YOUTUBE_RATE_LIMIT`: Requests per minute limit

## Job Flow Examples

### Channel Monitoring Flow
1. `check_channel` job added to queue
2. OrchestratorWorker routes to MonitorWorker
3. MonitorWorker fetches RSS feed
4. New videos detected and `download_transcript` jobs created
5. OrchestratorWorker routes download jobs to DownloadWorker
6. DownloadWorker extracts transcripts and stores results

### Error Handling Flow
1. Job fails during processing
2. Worker categorizes error (network, rate limit, etc.)
3. OrchestratorWorker determines retry strategy
4. Job either retried with backoff or marked as failed
5. Error logged with full context for debugging

## Monitoring and Operations

### Status Monitoring
```bash
# Check orchestrator status
python run_orchestrator.py --status

# Send status signal to running process
kill -USR1 <orchestrator_pid>
```

### Log Analysis
```bash
# Follow orchestrator logs
tail -f logs/orchestrator.log

# Filter specific worker logs
grep "monitor" logs/orchestrator.log
grep "ERROR" logs/orchestrator.log
```

### Queue Management
```python
from core.queue import get_job_queue

queue = get_job_queue()

# Get queue statistics
stats = await queue.get_queue_stats()
print(f"Pending: {stats.pending_jobs}, Processing: {stats.processing_jobs}")

# Get failed jobs
failed_jobs = await queue.get_failed_jobs(limit=10)

# Cleanup old completed jobs
cleaned = await queue.cleanup_old_jobs(dry_run=False)
```

## Testing

### Unit Tests
```bash
# Test orchestrator functionality
python test_orchestrator.py

# Test individual workers  
python -m pytest workers/test_*.py
```

### Integration Tests
```bash
# Test full workflow
python example_orchestrator_usage.py
```

## Extending with New Workers

### 1. Create Worker Class
```python
from workers.base import BaseWorker

class MyCustomWorker(BaseWorker):
    def validate_input(self, input_data):
        # Validate input data
        return True
    
    def execute(self, input_data):
        # Implement worker logic
        return {"result": "success"}
    
    def handle_error(self, error):
        # Handle errors
        return {"error_category": "custom"}
```

### 2. Register with Orchestrator
```python
# In orchestrator.py _initialize_worker_registry()
self.worker_registry["my_worker"] = WorkerRegistry(
    worker_class=MyCustomWorker,
    job_types={"my_job_type"},
    enabled=True
)
```

### 3. Add Job Type Support
```python
# In orchestrator.py _prepare_job_input_data()
elif job_type == "my_job_type":
    return {"custom_data": target_id}
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- Reduce `max_concurrent_jobs` 
- Check for memory leaks in worker implementations
- Monitor database connection pooling

#### Jobs Stuck in Processing
- Check worker timeouts
- Look for deadlocks in async code
- Use `recover_stale_jobs()` to reset stuck jobs

#### Rate Limiting
- Increase `polling_interval`
- Implement exponential backoff in workers
- Monitor YouTube API usage

### Debug Mode
```bash
python run_orchestrator.py --log-level DEBUG --max-jobs 5
```

This enables verbose logging for troubleshooting worker behavior and job processing flow.