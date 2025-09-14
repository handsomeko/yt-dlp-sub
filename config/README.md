# Configuration System

Centralized configuration management for yt-dl-sub using pydantic for validation and python-dotenv for environment variable loading.

## Quick Start

```python
from config import settings, is_development

# Access configuration
print(f"Running in {settings.deployment_mode} mode")
print(f"Storage path: {settings.storage_path}")
print(f"Database URL: {settings.database_url}")

# Check environment
if is_development():
    print("Development mode active")
```

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Key Configuration Options

- **DEPLOYMENT_MODE**: `LOCAL` (dev), `MONOLITH` (single server), `DISTRIBUTED` (cloud API + local workers)
- **STORAGE_PATH**: Base path for file storage (auto-creates directories)
- **DATABASE_URL**: SQLite, PostgreSQL, or MySQL connection string
- **WORKER_CONCURRENCY**: Number of concurrent workers (1-20)
- **YOUTUBE_RATE_LIMIT**: API requests per minute (1-100)

### List Fields

Comma-separated environment variables are automatically parsed:

```bash
STORAGE_BACKENDS=local,gdrive,s3
CONTENT_GENERATORS=blog,social,summary
API_KEYS=key1,key2,key3
```

## Configuration Classes

### Settings

Main configuration class with validation:

```python
from config import Settings

# Get fresh instance
settings = Settings()

# Access typed fields
storage_path: Path = settings.storage_path
deployment_mode: DeploymentMode = settings.deployment_mode
```

### Enums

Type-safe configuration options:

- `DeploymentMode`: LOCAL, MONOLITH, DISTRIBUTED  
- `QueueType`: sqlite, redis
- `LogLevel`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `StorageBackend`: local, gdrive, s3, gcs

## Validation

Automatic validation with helpful error messages:

```python
# Invalid Whisper model
WHISPER_MODEL=invalid_model  # ❌ Raises ValidationError

# Missing Redis URL for Redis queue
QUEUE_TYPE=redis  # ❌ Raises ValidationError (redis_url required)

# Valid configuration
WHISPER_MODEL=base  # ✅ 
QUEUE_TYPE=sqlite   # ✅
```

## Logging Configuration

Get structured logging config:

```python
import logging.config
from config import settings

log_config = settings.get_log_config()
logging.config.dictConfig(log_config)
logger = logging.getLogger("yt_dl_sub")
```

## Convenience Functions

```python
from config import is_development, is_production, get_storage_path

if is_development():
    # Development-specific logic
    pass

storage_path = get_storage_path()  # Returns Path object
```

## Deployment Mode Behavior

### LOCAL (Development)
- Single-threaded processing
- SQLite database and queue  
- Debug logging enabled
- Local file storage only

### MONOLITH (Production Single Server)
- Multi-process workers
- SQLite or PostgreSQL
- Production logging
- Multiple storage backends

### DISTRIBUTED (Cloud + Workers)  
- Redis queue required
- API on cloud, workers on residential IP
- Horizontal scaling
- Cloud storage integration

## Integration Examples

### YouTube Downloader

```python
from config import settings

yt_config = {
    "format": f"best[height<={settings.default_video_quality[:-1]}]",
    "outtmpl": str(settings.storage_path / "%(title)s/%(title)s.%(ext)s"),
    "concurrent_fragments": settings.youtube_concurrent_downloads,
}
```

### Database Connection

```python
from config import settings

db_config = {
    "url": settings.database_url,
    "echo": settings.debug,  # Log SQL in debug mode
}
```

### Worker Pool

```python
from config import settings, is_development

if is_development():
    workers = 1  # Single-threaded for debugging
else:
    workers = settings.worker_concurrency
```

## File Structure

```
config/
├── __init__.py      # Public API exports
├── settings.py      # Main Settings class
└── README.md        # This file

.env.example         # Environment template
.env                 # Your local settings (git-ignored)
```