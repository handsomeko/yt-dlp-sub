"""
Setup script for YouTube Content Intelligence Platform
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r for r in requirements if r and not r.startswith("#")]

with open("README.md", "w") as f:
    f.write("""# YouTube Content Intelligence & Repurposing Platform

A powerful YouTube content monitoring and repurposing platform that evolves through three phases:
CLI tool → API service → MicroSaaS product.

## Phase 1: CLI Tool (Current)

Monitor YouTube channels, extract transcripts, and search across all content.

### Features

- ✅ Add/remove YouTube channels to monitor
- ✅ Check channels for new videos via RSS
- ✅ Download transcripts using yt-dlp (fallback: youtube-transcript-api)
- ✅ Store transcripts in SQLite with FTS5 full-text search
- ✅ Search across all transcripts
- ✅ Export in multiple formats (JSON, CSV, TXT, Markdown)
- ✅ Modular worker system for scalability

### Installation

```bash
pip install -r requirements.txt
python setup.py install
```

### Quick Start

```bash
# Add a channel to monitor
yt-dl-sub add-channel https://www.youtube.com/@GoogleDevelopers

# Check all channels for new videos
yt-dl-sub sync

# Search across transcripts
yt-dl-sub search "machine learning"

# Export transcripts
yt-dl-sub export --format json --output transcripts.json

# View help
yt-dl-sub --help
```

### Architecture

- **Worker Pattern**: Each functionality is an isolated, invokable worker
- **Job Queue**: SQLite-based queue for async processing
- **Storage**: External drive support with configurable paths
- **Database**: SQLite with FTS5 for full-text search

### Configuration

Create a `.env` file:

```bash
DEPLOYMENT_MODE=LOCAL
DATABASE_URL=sqlite:///data.db
STORAGE_PATH=/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads
YOUTUBE_RATE_LIMIT=10
LOG_LEVEL=INFO
```

### Documentation

- `Product Requirement Prompt.md` - Vision and strategy
- `Product Requirement Document.md` - Technical specifications
- `CLAUDE.md` - Development guidelines

## Phase 2: API Service (Coming Soon)

- FastAPI REST endpoints
- API key authentication
- Webhook notifications
- Background processing

## Phase 3: MicroSaaS (Future)

- Web dashboard
- Subscription billing
- AI content generation
- Multi-channel publishing

## License

MIT
""")

setup(
    name="yt-dl-sub",
    version="1.0.0",
    description="YouTube Content Intelligence & Repurposing Platform",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "yt-dl-sub=cli:cli",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)