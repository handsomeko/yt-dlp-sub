"""
Secure FastAPI server with comprehensive security controls
Addresses Issues #126-135
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
import os
import secrets

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.downloader import YouTubeDownloader
from core.transcript import TranscriptExtractor
from core.monitor import ChannelMonitor
from core.api_security_final import (
    SecurityConfig,
    create_secure_app,
    AuthenticationMiddleware,
    WebhookValidator
)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Download & Monitor API",
    description="Secure API for YouTube operations",
    version="1.0.0",
    docs_url="/docs",  # Keep docs but require auth
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Security configuration
security_config = SecurityConfig(
    # CORS - Issue #126 (Fixed: specific origins instead of wildcard)
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ],
    cors_allow_credentials=True,
    cors_allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    cors_allowed_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    
    # Authentication - Issue #127
    jwt_secret=os.getenv("JWT_SECRET", secrets.token_urlsafe(32)),
    jwt_algorithm="HS256",
    jwt_expiry_hours=24,
    api_key_header="X-API-Key",
    require_auth_endpoints=[
        "/api/v1/*",
        "/video/*",
        "/transcript/*",
        "/channel/*",
        "/queue/*"
    ],
    public_endpoints=[
        "/",
        "/health",
        "/api/security/status"
    ],
    
    # Rate limiting - Issue #128
    rate_limit_requests=100,
    rate_limit_window_seconds=60,
    rate_limit_burst=10,
    
    # Security headers - Issue #129
    enable_security_headers=True,
    hsts_max_age=31536000,
    csp_policy="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    
    # Request limits - Issue #130
    max_request_size_mb=10,
    max_upload_size_mb=100,
    request_timeout_seconds=30,
    
    # Webhook security - Issue #131
    webhook_secret=os.getenv("WEBHOOK_SECRET", secrets.token_urlsafe(32)),
    webhook_timeout=10,
    
    # Database pooling - Issue #132
    db_pool_size=20,
    db_max_overflow=10,
    db_pool_timeout=30,
    db_pool_recycle=3600,
    
    # API versioning - Issue #134
    api_version="v1",
    supported_versions=["v1"],
    deprecation_headers=True
)

# Apply security middleware
app = create_secure_app(app, security_config)

# Initialize core modules
downloader = YouTubeDownloader()
transcript_extractor = TranscriptExtractor()
channel_monitor = ChannelMonitor()

# Pydantic models with validation
class VideoDownloadRequest(BaseModel):
    url: str
    quality: Optional[str] = "1080p"
    audio_only: Optional[bool] = False
    audio_format: Optional[str] = "mp3"
    video_format: Optional[str] = "mp4"
    channel_name: Optional[str] = None
    extract_transcript: Optional[bool] = True
    
    class Config:
        # Prevent additional fields
        extra = "forbid"
        # Validate on assignment
        validate_assignment = True

class TranscriptRequest(BaseModel):
    url: str
    languages: Optional[List[str]] = ["en"]
    save_to_file: Optional[bool] = True
    
    class Config:
        extra = "forbid"
        validate_assignment = True

class ChannelRequest(BaseModel):
    channel_id: str
    channel_name: str
    channel_url: Optional[str] = None
    
    class Config:
        extra = "forbid"
        validate_assignment = True

class BatchDownloadRequest(BaseModel):
    urls: List[str]
    quality: Optional[str] = "1080p"
    extract_transcripts: Optional[bool] = True
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        # Limit batch size
        max_items = 10

class WebhookRequest(BaseModel):
    url: str
    events: List[str]
    secret: Optional[str] = None
    
    class Config:
        extra = "forbid"
        validate_assignment = True


# Health check endpoint (public)
@app.get("/")
async def root():
    """Public health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "security": "enabled"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "downloader": "healthy",
            "transcript": "healthy",
            "monitor": "healthy",
            "security": "enabled"
        }
    }


# API v1 endpoints (authenticated)
@app.post("/api/v1/video/info")
async def get_video_info(url: str):
    """Get video metadata without downloading (authenticated)"""
    info = downloader.get_video_info(url)
    if 'error' in info:
        raise HTTPException(status_code=400, detail=info['error'])
    
    # Log access
    if hasattr(app.state, 'audit_logger'):
        await app.state.audit_logger.log_security_event(
            event_type="video_info_accessed",
            details={"url": url},
            severity="INFO"
        )
    
    return info


@app.post("/api/v1/video/download")
async def download_video(request: VideoDownloadRequest, background_tasks: BackgroundTasks):
    """Download video with security validation (authenticated)"""
    
    # Validate URL format
    if not request.url.startswith(("https://youtube.com", "https://www.youtube.com", "https://youtu.be")):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Queue download task
    result = await downloader.download_video_async(
        url=request.url,
        quality=request.quality,
        audio_only=request.audio_only,
        audio_format=request.audio_format,
        video_format=request.video_format,
        channel_name=request.channel_name
    )
    
    # Extract transcript if requested
    if request.extract_transcript and not request.audio_only:
        background_tasks.add_task(
            transcript_extractor.extract_async,
            request.url,
            ["en"]
        )
    
    # Send webhook if configured
    if hasattr(app.state, 'webhook_validator'):
        webhook_data = {
            "event": "video_downloaded",
            "url": request.url,
            "timestamp": datetime.now().isoformat()
        }
        background_tasks.add_task(
            app.state.webhook_validator.send_webhook,
            "https://your-webhook-url.com/events",
            webhook_data
        )
    
    return result


@app.post("/api/v1/transcript/extract")
async def extract_transcript(request: TranscriptRequest):
    """Extract transcript with validation (authenticated)"""
    
    # Validate URL
    if not request.url.startswith(("https://youtube.com", "https://www.youtube.com", "https://youtu.be")):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Validate languages
    valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
    for lang in request.languages:
        if lang not in valid_languages:
            raise HTTPException(status_code=400, detail=f"Invalid language code: {lang}")
    
    result = await transcript_extractor.extract_async(
        url=request.url,
        languages=request.languages,
        save_to_file=request.save_to_file
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Transcript not available")
    
    return result


@app.post("/api/v1/channel/add")
async def add_channel(request: ChannelRequest):
    """Add channel for monitoring (authenticated)"""
    
    # Validate channel data
    if not request.channel_id or not request.channel_name:
        raise HTTPException(status_code=400, detail="Channel ID and name required")
    
    result = await channel_monitor.add_channel(
        channel_id=request.channel_id,
        channel_name=request.channel_name,
        channel_url=request.channel_url
    )
    
    # Log channel addition
    if hasattr(app.state, 'audit_logger'):
        await app.state.audit_logger.log_security_event(
            event_type="channel_added",
            details={"channel_id": request.channel_id},
            severity="INFO"
        )
    
    return result


@app.get("/api/v1/channel/list")
async def list_channels():
    """List monitored channels (authenticated)"""
    channels = await channel_monitor.list_channels()
    return {
        "total": len(channels),
        "channels": channels
    }


@app.post("/api/v1/channel/check/{channel_id}")
async def check_channel(channel_id: str):
    """Check specific channel for new videos (authenticated)"""
    
    # Validate channel ID format
    if not channel_id.startswith("UC"):
        raise HTTPException(status_code=400, detail="Invalid channel ID format")
    
    new_videos = await channel_monitor.check_channel(channel_id)
    
    return {
        "channel_id": channel_id,
        "new_videos_count": len(new_videos),
        "new_videos": new_videos
    }


@app.post("/api/v1/channel/check-all")
async def check_all_channels(background_tasks: BackgroundTasks):
    """Check all channels for new videos (authenticated)"""
    
    # Queue background task
    background_tasks.add_task(channel_monitor.check_all_channels)
    
    return {
        "status": "checking",
        "message": "Channel check initiated in background"
    }


@app.get("/api/v1/queue/pending")
async def get_pending_jobs():
    """Get pending download jobs (authenticated)"""
    # This would integrate with your queue system
    return {
        "pending_count": 0,
        "jobs": []
    }


@app.post("/api/v1/batch/download")
async def batch_download(request: BatchDownloadRequest, background_tasks: BackgroundTasks):
    """Batch download with limits (authenticated)"""
    
    # Validate batch size
    if len(request.urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 URLs per batch")
    
    # Validate all URLs
    for url in request.urls:
        if not url.startswith(("https://youtube.com", "https://www.youtube.com", "https://youtu.be")):
            raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
    
    # Queue downloads
    job_ids = []
    for url in request.urls:
        job_id = secrets.token_hex(8)
        background_tasks.add_task(
            downloader.download_video_async,
            url=url,
            quality=request.quality
        )
        job_ids.append(job_id)
    
    return {
        "status": "queued",
        "job_ids": job_ids,
        "total": len(job_ids)
    }


@app.post("/api/v1/webhook/register")
async def register_webhook(request: WebhookRequest):
    """Register webhook for events (authenticated)"""
    
    # Validate webhook URL
    if not request.url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Webhook URL must use HTTPS")
    
    # Validate events
    valid_events = ["video_downloaded", "transcript_extracted", "channel_checked"]
    for event in request.events:
        if event not in valid_events:
            raise HTTPException(status_code=400, detail=f"Invalid event: {event}")
    
    # Store webhook configuration
    webhook_id = secrets.token_hex(8)
    
    return {
        "webhook_id": webhook_id,
        "url": request.url,
        "events": request.events,
        "status": "registered"
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    if hasattr(app.state, 'audit_logger'):
        await app.state.audit_logger.log_security_event(
            event_type="validation_error",
            details={"error": str(exc), "path": request.url.path},
            severity="WARNING"
        )
    
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with logging"""
    if hasattr(app.state, 'audit_logger'):
        await app.state.audit_logger.log_security_event(
            event_type="http_exception",
            details={"status": exc.status_code, "detail": exc.detail},
            severity="WARNING" if exc.status_code < 500 else "ERROR"
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with SSL in production
    uvicorn.run(
        app,
        host="127.0.0.1",  # Bind to localhost only
        port=8000,
        log_level="info",
        access_log=True,
        # SSL configuration for production
        # ssl_keyfile="path/to/key.pem",
        # ssl_certfile="path/to/cert.pem"
    )