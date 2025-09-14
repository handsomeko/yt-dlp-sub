# Export Module Documentation

The export module provides comprehensive functionality for exporting transcripts and generated content from the YouTube Content Intelligence platform in multiple formats.

## Features

- **Multiple Export Formats**: JSON, CSV, TXT, Markdown
- **Flexible Filtering**: By channel, date range, or custom criteria
- **Efficient Streaming**: Handles large datasets with batch processing
- **Progress Tracking**: Real-time progress callbacks for long operations
- **Error Handling**: Comprehensive validation and error recovery
- **Async Support**: Built for high-performance async operations

## Quick Start

### Basic Usage

```python
from core.export import ExportService, export_transcripts, get_export_stats

# Get export statistics
stats = await get_export_stats()
print(f"Available: {stats.total_transcripts} transcripts")

# Export all transcripts to JSON
result = await export_transcripts(format="json", include_content=True)
print(f"Exported to: {result['output_path']}")
```

### With Filters

```python
from datetime import datetime, timedelta

# Export specific channel's recent videos to CSV
result = await export_transcripts(
    format="csv",
    channel_id="UC123456789",
    since=datetime.now() - timedelta(days=30),
    output_path="recent_videos.csv"
)
```

### Using the Service Class

```python
from core.export import ExportService, ExportProgress

def progress_callback(progress: ExportProgress):
    print(f"Progress: {progress.progress_percent:.1f}%")

export_service = ExportService()

result = await export_service.export_transcripts(
    format="markdown",
    include_content=True,
    progress_callback=progress_callback
)
```

## Export Formats

### JSON Format
Complete structured data with all metadata:
```json
{
  "export_metadata": {
    "timestamp": "2024-01-15T10:30:00",
    "format": "json",
    "include_content": true
  },
  "transcripts": [
    {
      "transcript": {
        "id": 1,
        "video_id": "ABC123",
        "content_text": "Full transcript text...",
        "word_count": 1500,
        "quality_score": 0.95
      },
      "video": {
        "title": "Video Title",
        "duration": 1800,
        "view_count": 10000
      },
      "channel": {
        "channel_name": "Tech Channel",
        "subscriber_count": 50000
      },
      "generated_content": [...]
    }
  ]
}
```

### CSV Format
Tabular format for spreadsheet analysis:
```csv
video_id,video_title,channel_name,transcript_text,word_count,quality_score,...
ABC123,"Video Title","Tech Channel","Transcript content...",1500,0.95,...
```

### TXT Format
Human-readable plain text:
```
TRANSCRIPT EXPORT
Generated: 2024-01-15 10:30:00
================================================================================

VIDEO: Sample Video Title
Channel: Tech Channel
Video ID: ABC123
Duration: 30:00
Transcript: Full transcript content here...

================================================================================
```

### Markdown Format
Formatted documentation:
```markdown
# Transcript Export

## Sample Video Title

- **Channel:** [Tech Channel](https://youtube.com/c/tech)
- **Duration:** 30:00
- **Word Count:** 1,500
- **Quality Score:** 0.95/1.0

### Transcript

Full transcript content here...

### Generated Content

#### Blog Post
Generated blog content...

---
```

## API Reference

### ExportService Class

#### Methods

##### `export_transcripts()`
Main export method with full configuration options.

**Parameters:**
- `format` (str): Export format ('json', 'csv', 'txt', 'markdown')
- `channel_id` (Optional[str]): Filter by specific channel ID
- `since` (Optional[datetime/date/str]): Include videos since this date
- `until` (Optional[datetime/date/str]): Include videos until this date
- `output_path` (Optional[Path]): Custom output path (auto-generated if None)
- `include_content` (bool): Include generated content in export
- `progress_callback` (Optional[Callable]): Progress tracking function
- `batch_size` (int): Records per batch (default: 100)

**Returns:** Dictionary with export results and statistics

##### `get_export_stats()`
Get statistics without performing export.

**Parameters:**
- `channel_id` (Optional[str]): Filter by channel
- `since` (Optional[datetime/date/str]): Date range start
- `until` (Optional[datetime/date/str]): Date range end

**Returns:** `ExportStats` object with counts and metadata

### Convenience Functions

#### `export_transcripts()`
Quick export with automatic service initialization.

#### `get_export_stats()`
Quick statistics check.

### Data Classes

#### `ExportStats`
Statistics container:
```python
@dataclass
class ExportStats:
    total_transcripts: int
    total_content_items: int
    total_channels: int
    total_videos: int
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]
    file_size_bytes: int
    export_duration_seconds: float
```

#### `ExportProgress`
Progress tracking:
```python
@dataclass
class ExportProgress:
    current_item: int
    total_items: int
    current_phase: str
    estimated_completion: Optional[datetime]
    
    @property
    def progress_percent(self) -> float
```

## CLI Tool

Use the command-line interface for quick exports:

```bash
# Show statistics
python3 export_cli.py --stats

# Export all transcripts to JSON
python3 export_cli.py --format json

# Export specific channel
python3 export_cli.py --format csv --channel-id UC123456789

# Export with date filter
python3 export_cli.py --format markdown --since 2024-01-01

# Export with generated content
python3 export_cli.py --format json --include-content --output full_export.json
```

### CLI Options

- `--stats`: Show export statistics only
- `--format, -f`: Export format (json, csv, txt, markdown)
- `--output, -o`: Output file path
- `--include-content`: Include generated content
- `--channel-id`: Filter by channel ID
- `--since`: Date range start (YYYY-MM-DD)
- `--until`: Date range end (YYYY-MM-DD)
- `--batch-size`: Processing batch size
- `--verbose, -v`: Detailed progress
- `--quiet, -q`: No progress output

## Performance Considerations

### Large Datasets
- Uses streaming with configurable batch sizes
- Memory-efficient processing
- Progress tracking for long operations

### File Size Estimates
- JSON: ~5KB per transcript (with content)
- CSV: ~1KB per transcript
- TXT: ~3KB per transcript
- Markdown: ~3.5KB per transcript

### Optimization Tips
- Use smaller batch sizes (50-100) for memory-constrained environments
- Use filters to export only needed data
- Consider CSV format for large datasets requiring analysis
- Use progress callbacks to monitor long-running exports

## Error Handling

The export system handles various error conditions:

### Validation Errors
- Invalid format specification
- Invalid date formats
- Missing required parameters

### Runtime Errors
- Database connection issues
- File system permissions
- Disk space limitations
- Memory constraints

### Recovery
- Automatic cleanup of partial files on failure
- Detailed error messages with context
- Graceful handling of interruptions

## Integration Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI, BackgroundTasks
from core.export import export_transcripts

app = FastAPI()

@app.post("/export/transcripts")
async def export_endpoint(
    format: str = "json",
    channel_id: Optional[str] = None,
    include_content: bool = False
):
    result = await export_transcripts(
        format=format,
        channel_id=channel_id,
        include_content=include_content
    )
    return {
        "status": "completed",
        "download_url": f"/downloads/{Path(result['output_path']).name}",
        "stats": result["stats"]
    }
```

### Scheduled Exports

```python
import asyncio
from datetime import datetime, timedelta

async def daily_export():
    """Export yesterday's transcripts daily."""
    yesterday = datetime.now() - timedelta(days=1)
    
    result = await export_transcripts(
        format="json",
        since=yesterday.strftime("%Y-%m-%d"),
        until=yesterday.strftime("%Y-%m-%d"),
        include_content=True,
        output_path=f"exports/daily_{yesterday.strftime('%Y%m%d')}.json"
    )
    
    print(f"Daily export completed: {result['stats'].total_transcripts} transcripts")

# Run with cron or task scheduler
if __name__ == "__main__":
    asyncio.run(daily_export())
```

### Monitoring and Alerts

```python
from core.export import get_export_stats
import smtplib
from email.mime.text import MIMEText

async def export_health_check():
    """Check export system health."""
    stats = await get_export_stats()
    
    if stats.total_transcripts < 100:  # Threshold check
        # Send alert
        msg = MIMEText(f"Low transcript count: {stats.total_transcripts}")
        # ... email sending logic
    
    return stats
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_export.py
```

Run specific examples:

```bash
python3 examples/export_usage.py
```

## Troubleshooting

### Common Issues

1. **"No module named 'aiofiles'"**: The module uses standard library only
2. **Memory errors with large exports**: Reduce batch size or add filters
3. **Permission errors**: Check file system permissions for output directory
4. **Database connection errors**: Verify database URL and initialization

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run export operations
```

### Performance Profiling

Profile export performance:

```python
import cProfile
import asyncio

async def profile_export():
    result = await export_transcripts(format="json")
    return result

# Profile the export
cProfile.run('asyncio.run(profile_export())')
```

## Contributing

When adding new export formats:

1. Add format validation to `export_transcripts()`
2. Implement `_export_transcripts_<format>()` method
3. Add format-specific cleaning/formatting utilities
4. Update tests and documentation
5. Consider performance implications for large datasets

For questions or issues, refer to the main project documentation or create an issue in the repository.