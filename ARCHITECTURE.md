# YouTube Content Intelligence Platform - Worker Architecture

## Complete Worker Pipeline

Following the worker/tool pattern where each functionality is an isolated, invokable worker:

```
┌──────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│            (Coordinates all workers and job flow)            │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     WORKER PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. MonitorWorker                                           │
│     └─> Checks RSS feeds for new videos                     │
│     └─> Creates jobs: download_audio, extract_transcript    │
│                                                              │
│  2. AudioDownloadWorker                                      │
│     └─> Downloads audio from YouTube (yt-dlp)               │
│     └─> Saves to: /audio/{channel_id}/{video_id}/          │
│                                                              │
│  3. TranscribeWorker                                         │
│     └─> Extracts transcript (yt-dlp → youtube-transcript-api)│
│     └─> Future: FFmpeg af_whisper → Whisper local → API     │
│                                                              │
│  4. TranscriptQualityWorker                                 │
│     └─> Validates transcript quality (completeness, density) │
│     └─> Checks coherence, language confidence, error rate    │
│                                                              │
│  5. ContentQualityWorker                                     │
│     └─> Validates generated content quality                  │
│     └─> Checks format, length, relevance, readability       │
│                                                              │
│  6. StorageWorker                                           │
│     └─> Syncs files to storage backends                     │
│     └─> Phase 1: Local + Google Drive                       │
│     └─> Phase 2: + Supabase                                 │
│     └─> Phase 3: + S3/GCS                                   │
│                                                              │
│  7. GeneratorWorker (Orchestrator for content)              │
│     └─> Distributes work to sub-generators                  │
│     └─> Manages parallel content generation                 │
│         │                                                    │
│         ├─> SummaryGenerator                                │
│         │   └─> Short/Medium/Detailed summaries             │
│         │                                                    │
│         ├─> BlogGenerator                                   │
│         │   └─> 500/1000/2000 word posts                    │
│         │                                                    │
│         ├─> SocialMediaGenerator                            │
│         │   └─> Twitter/LinkedIn/Facebook posts             │
│         │                                                    │
│         ├─> NewsletterGenerator                             │
│         │   └─> Email sections (headline/summary/CTA)       │
│         │                                                    │
│         └─> ScriptGenerator                                 │
│             └─> YouTube Shorts/TikTok/Podcast scripts       │
│                                                              │
│  8. PublishWorker                                           │
│     └─> Distributes content to platforms                    │
│     └─> Phase 1: Local, Webhook, Email prep, API prep       │
│     └─> Phase 2: Social media platforms                     │
│     └─> Phase 3: Omnichannel publishing                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Worker Responsibilities (Single Responsibility Principle)

| Worker | Single Responsibility | Input | Output |
|--------|----------------------|-------|---------|
| **MonitorWorker** | Check RSS feeds for new videos | channel_id or check_all | New video jobs |
| **AudioDownloadWorker** | Download audio from YouTube | video_id, video_url | Audio file path |
| **TranscribeWorker** | Extract transcript from video | video_id, video_url, audio_path | SRT & text transcript |
| **TranscriptQualityWorker** | Validate transcript quality | transcript_text, video_duration | Quality score & pass/fail |
| **ContentQualityWorker** | Validate generated content | content_text, content_type | Quality score & pass/fail |
| **StorageWorker** | Sync files to storage backends | file_path, storage_backends | Storage URLs |
| **GeneratorWorker** | Orchestrate content generation | transcript, content_types | Generation jobs |
| **SummaryGenerator** | Generate summaries | transcript, length | Summary text |
| **BlogGenerator** | Generate blog posts | transcript, word_count | Blog post |
| **SocialMediaGenerator** | Generate social posts | transcript, platform | Social media posts |
| **NewsletterGenerator** | Generate newsletter | transcript, sections | Newsletter content |
| **ScriptGenerator** | Generate scripts | transcript, duration | Video/podcast scripts |
| **PublishWorker** | Distribute content | content, publish_targets | Publishing results |

## Job Types and Worker Mapping

```python
JOB_TYPE_TO_WORKER = {
    # Monitoring
    'check_channel': MonitorWorker,
    'check_all_channels': MonitorWorker,
    
    # Download & Transcription
    'download_audio': AudioDownloadWorker,
    'extract_transcript': TranscribeWorker,
    
    # Quality & Storage
    'validate_transcript': TranscriptQualityWorker,
    'validate_content': ContentQualityWorker,
    'sync_storage': StorageWorker,
    
    # Content Generation
    'generate_content': GeneratorWorker,
    'generate_summary': SummaryGenerator,
    'generate_blog': BlogGenerator,
    'generate_social': SocialMediaGenerator,
    'generate_newsletter': NewsletterGenerator,
    'generate_script': ScriptGenerator,
    
    # Publishing
    'publish_content': PublishWorker,
}
```

## Phase Implementation Status

### Phase 1 (CLI) - Current ✅
- ✅ All workers created with base functionality
- ✅ Job queue system operational
- ✅ Local storage working
- ✅ Placeholder content generation
- ✅ Basic publishing (local only)

### Phase 2 (API) - Next
- [ ] Add FastAPI endpoints for each worker
- [ ] Implement real AI content generation
- [ ] Add Google Drive sync
- [ ] Enable webhook/email publishing
- [ ] Add authentication and rate limiting

### Phase 3 (MicroSaaS) - Future
- [ ] Web dashboard
- [ ] Multi-user support
- [ ] Social media publishing
- [ ] Advanced analytics
- [ ] Subscription billing

## Benefits of This Architecture

1. **Modularity**: Each worker is independent and replaceable
2. **Testability**: Workers can be tested in isolation
3. **Scalability**: Scale specific workers based on workload
4. **Reliability**: Failure in one worker doesn't affect others
5. **Maintainability**: Clear separation of concerns
6. **Extensibility**: Easy to add new workers or modify existing ones
7. **Observability**: Each worker has its own logging and metrics

## Usage Example

```python
# Each worker can be invoked independently
from workers import (
    MonitorWorker,
    AudioDownloadWorker,
    TranscribeWorker,
    QualityWorker,
    StorageWorker,
    GeneratorWorker,
    PublishWorker
)

# Monitor channels
monitor = MonitorWorker()
result = monitor.run({'check_all': True})

# Download audio
downloader = AudioDownloadWorker()
result = downloader.run({
    'video_id': 'abc123',
    'video_url': 'https://youtube.com/watch?v=abc123'
})

# Extract transcript
transcriber = TranscribeWorker()
result = transcriber.run({
    'video_id': 'abc123',
    'video_url': 'https://youtube.com/watch?v=abc123',
    'audio_path': '/path/to/audio.opus'
})

# Validate transcript quality
transcript_quality = TranscriptQualityWorker()
result = transcript_quality.run({
    'video_id': 'abc123',
    'transcript_text': 'transcript text...',
    'video_duration': 300
})

# Validate content quality
content_quality = ContentQualityWorker()
result = content_quality.run({
    'content_id': 'content123',
    'content_type': 'blog',
    'content_text': 'blog post content...'
})

# Generate content
generator = GeneratorWorker()
result = generator.run({
    'video_id': 'abc123',
    'transcript': 'transcript text...',
    'content_types': ['summary', 'blog', 'social']
})

# Publish content
publisher = PublishWorker()
result = publisher.run({
    'content_id': 'content123',
    'content_type': 'blog',
    'publish_targets': ['local', 'webhook']
})
```

## Worker Communication

Workers communicate through:
1. **Job Queue**: Asynchronous job creation and processing
2. **Database**: Shared state and results storage
3. **File System**: Content and media storage
4. **Return Values**: Direct result passing when chained

This architecture ensures clean separation of concerns while maintaining flexibility for future enhancements.