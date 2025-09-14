"""FastAPI server with comprehensive security enhancements"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
import os
import asyncio
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.downloader import YouTubeDownloader
from core.transcript import TranscriptExtractor
from core.monitor import ChannelMonitor
from core.security_fixes import (
    SecurityManager, RateLimiter, APIParameterValidator,
    SecureFileValidator, DataSanitizer, AsyncManager
)

# Initialize security manager
security_manager = SecurityManager()
security_manager.initialize({
    'log_dir': '/var/log/yt-dl-sub',
    'database_url': os.environ.get('DATABASE_URL', 'sqlite:///data/app.db'),
    'key_file': '/etc/yt-dl-sub/vault.key'
})

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    security_manager.system_integration.validate_environment()
    security_manager.system_integration.setup_signal_handlers()
    yield
    # Shutdown
    security_manager.cleanup()

# Initialize FastAPI app with security enhancements
app = FastAPI(
    title="YouTube Download & Monitor API",
    description="Secure API for YouTube operations",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable in production
    redoc_url=None  # Disable in production
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", os.environ.get("ALLOWED_HOST", "*")]
)

# Configure CORS with restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600
)

# Initialize core modules
downloader = YouTubeDownloader()
transcript_extractor = TranscriptExtractor()
channel_monitor = ChannelMonitor()

# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Check rate limit for request"""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# Enhanced Pydantic models with validation
class VideoDownloadRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=500)
    quality: Optional[str] = Field(default="1080p", pattern="^(2160p|1440p|1080p|720p|480p|360p|240p|144p|best|worst)$")
    audio_only: Optional[bool] = False
    audio_format: Optional[str] = Field(default="mp3", pattern="^(mp3|m4a|wav|opus|flac|best)$")
    video_format: Optional[str] = Field(default="mp4", pattern="^(mp4|mkv|webm|avi)$")
    channel_name: Optional[str] = Field(default=None, max_length=100)
    extract_transcript: Optional[bool] = True
    
    @field_validator('url')
    def validate_url(cls, v):
        # Use security manager's URL validation
        from core.input_validation import URLValidator
        validator = URLValidator()
        result = validator.validate_url(v, allow_youtube_only=True)
        if not result.valid:
            raise ValueError(f"Invalid URL: {', '.join(result.errors)}")
        return result.sanitized_value
    
    @field_validator('quality')
    def validate_quality(cls, v):
        return APIParameterValidator.validate_quality(v)
    
    @field_validator('audio_format')
    def validate_audio_format(cls, v):
        return APIParameterValidator.validate_format('audio', v)
    
    @field_validator('video_format')
    def validate_video_format(cls, v):
        return APIParameterValidator.validate_format('video', v)

class TranscriptRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=500)
    languages: Optional[List[str]] = Field(default=["en"], max_items=5)
    save_to_file: Optional[bool] = True

class ChannelRequest(BaseModel):
    channel_id: str = Field(..., min_length=1, max_length=50)
    channel_name: str = Field(..., min_length=1, max_length=100)
    channel_url: Optional[str] = Field(default=None, max_length=500)
    
    @field_validator('channel_id', 'channel_name')
    def sanitize_text(cls, v):
        return DataSanitizer.sanitize_for_display(v)

class BatchDownloadRequest(BaseModel):
    urls: List[str] = Field(..., max_items=10)
    quality: Optional[str] = Field(default="1080p", pattern="^(2160p|1440p|1080p|720p|480p|360p|240p|144p|best|worst)$")
    extract_transcripts: Optional[bool] = True
    
    @field_validator('urls')
    def validate_urls(cls, v):
        return APIParameterValidator.validate_batch(v)


# Health check with rate limiting
@app.get("/health", dependencies=[Depends(check_rate_limit)])
async def health_check():
    """Health check endpoint"""
    # Check worker health
    worker_status = {}
    for worker_id in security_manager.worker_manager.workers:
        is_healthy = await security_manager.worker_manager.health_check(worker_id)
        worker_status[worker_id] = "healthy" if is_healthy else "unhealthy"
    
    # Check database
    db_healthy = False
    try:
        if security_manager.database:
            with security_manager.database.transaction() as conn:
                conn.execute("SELECT 1")
                db_healthy = True
    except:
        pass
    
    return {
        "status": "healthy" if db_healthy and any(worker_status.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "workers": worker_status,
        "database": "connected" if db_healthy else "disconnected",
        "memory_usage_mb": security_manager.resource_manager.check_memory_usage()
    }


# Video endpoints with security
@app.post("/video/info", dependencies=[Depends(check_rate_limit)])
async def get_video_info(url: str = Field(..., min_length=10, max_length=500)):
    """Get video metadata without downloading"""
    try:
        # Validate URL
        from core.input_validation import URLValidator
        validator = URLValidator()
        result = validator.validate_url(url, allow_youtube_only=True)
        if not result.valid:
            raise HTTPException(status_code=400, detail=f"Invalid URL: {', '.join(result.errors)}")
        
        # Get info with timeout
        info = await AsyncManager.with_timeout(
            downloader.get_video_info(result.sanitized_value),
            timeout_seconds=30
        )
        
        if 'error' in info:
            security_manager.logger.log_safe('error', f"Video info error: {info['error']}")
            raise HTTPException(status_code=400, detail=info['error'])
        
        # Sanitize response
        if 'title' in info:
            info['title'] = DataSanitizer.sanitize_for_display(info['title'])
        if 'description' in info:
            info['description'] = DataSanitizer.sanitize_for_display(info['description'])
        
        return info
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        security_manager.logger.log_safe('error', f"Video info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/video/download", dependencies=[Depends(check_rate_limit)])
async def download_video(request: VideoDownloadRequest, background_tasks: BackgroundTasks):
    """Download video with security enhancements"""
    
    # Check if job already in progress (deduplication)
    job_id = f"{request.url}_{request.quality}_{request.audio_only}"
    if not security_manager.worker_manager.acquire_job_lock(job_id):
        raise HTTPException(status_code=409, detail="Download already in progress")
    
    try:
        # Check file size limits
        info = await AsyncManager.with_timeout(
            downloader.get_video_info(request.url),
            timeout_seconds=30
        )
        
        if 'filesize' in info:
            DataSanitizer.validate_file_size(info['filesize'], max_size_mb=5000)
        
        # Start download with resource limits
        with security_manager.concurrency_manager.get_semaphore('downloads', limit=5):
            result = await AsyncManager.with_timeout(
                downloader.download_video(
                    url=request.url,
                    quality=request.quality,
                    download_audio_only=request.audio_only,
                    audio_format=request.audio_format,
                    video_format=request.video_format,
                    channel_name=request.channel_name
                ),
                timeout_seconds=600  # 10 minutes
            )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Extract transcript if requested
        if request.extract_transcript and not request.audio_only:
            transcript_result = await transcript_extractor.get_transcript(
                video_url_or_id=request.url,
                video_title=result.get('title')
            )
            result['transcript'] = transcript_result
        
        # Mark as downloaded in monitor
        if result.get('video_id'):
            channel_monitor.mark_video_downloaded(result['video_id'])
        
        return result
        
    finally:
        # Release job lock
        security_manager.worker_manager.release_job_lock(job_id)


@app.post("/audio/download", dependencies=[Depends(check_rate_limit)])
async def download_audio(
    url: str = Field(..., min_length=10, max_length=500),
    format: str = Field(default="mp3", pattern="^(mp3|m4a|wav|opus|flac|best)$")
):
    """Download only audio in specified format"""
    # Validate parameters
    APIParameterValidator.validate_format('audio', format)
    
    # Check job lock
    job_id = f"audio_{url}_{format}"
    if not security_manager.worker_manager.acquire_job_lock(job_id):
        raise HTTPException(status_code=409, detail="Download already in progress")
    
    try:
        result = await AsyncManager.with_timeout(
            downloader.download_video(
                url=url,
                download_audio_only=True,
                audio_format=format
            ),
            timeout_seconds=300  # 5 minutes for audio
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    finally:
        security_manager.worker_manager.release_job_lock(job_id)


@app.get("/video/formats")
async def get_available_formats():
    """Get list of supported video and audio formats"""
    return {
        "video_qualities": APIParameterValidator.QUALITY_OPTIONS,
        "audio_formats": APIParameterValidator.AUDIO_FORMATS,
        "video_formats": APIParameterValidator.VIDEO_FORMATS,
        "max_batch_size": APIParameterValidator.MAX_BATCH_SIZE
    }


@app.post("/video/download-batch", dependencies=[Depends(check_rate_limit)])
async def download_batch(request: BatchDownloadRequest, background_tasks: BackgroundTasks):
    """Queue multiple videos for download with limits"""
    results = []
    
    # Use semaphore to limit concurrent batch operations
    async with security_manager.concurrency_manager.get_semaphore('batch_downloads', limit=2):
        for url in request.urls:
            try:
                # Add each download as background task with rate limiting
                await asyncio.sleep(0.5)  # Rate limit between batch items
                
                background_tasks.add_task(
                    download_with_limits,
                    url,
                    request.quality,
                    request.extract_transcripts
                )
                results.append({"url": url, "status": "queued"})
                
            except Exception as e:
                security_manager.logger.log_safe('error', f"Batch download error: {str(e)}")
                results.append({"url": url, "status": "error", "error": "Failed to queue"})
    
    return {"queued": len([r for r in results if r["status"] == "queued"]), "results": results}


async def download_with_limits(url: str, quality: str, extract_transcript: bool):
    """Download with resource limits (for background tasks)"""
    try:
        # Check memory before starting
        if security_manager.resource_manager.check_memory_usage() > 800:  # 800MB threshold
            security_manager.logger.log_safe('warning', "Memory limit approaching, deferring download")
            await asyncio.sleep(60)  # Wait before retrying
        
        # Download with timeout
        await AsyncManager.with_timeout(
            downloader.download_video(url, quality),
            timeout_seconds=900  # 15 minutes
        )
        
    except Exception as e:
        security_manager.logger.log_safe('error', f"Background download failed: {str(e)}")


# Transcript endpoints
@app.post("/transcript/extract", dependencies=[Depends(check_rate_limit)])
async def extract_transcript(request: TranscriptRequest):
    """Extract transcript with security"""
    try:
        result = await AsyncManager.with_timeout(
            transcript_extractor.get_transcript(
                video_url_or_id=request.url,
                languages=request.languages[:5],  # Limit languages
                save_to_file=request.save_to_file
            ),
            timeout_seconds=120  # 2 minutes
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Sanitize transcript content
        if 'transcript' in result:
            result['transcript'] = DataSanitizer.sanitize_for_display(result['transcript'])
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Transcript extraction timeout")


@app.get("/transcript/languages", dependencies=[Depends(check_rate_limit)])
async def get_transcript_languages(video_url: str = Field(..., min_length=10, max_length=500)):
    """Get available transcript languages for a video"""
    video_id = transcript_extractor.extract_video_id(video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    languages = await AsyncManager.with_timeout(
        transcript_extractor.get_available_languages(video_id),
        timeout_seconds=30
    )
    
    return {"video_id": video_id, "languages": languages}


# Channel monitoring endpoints
@app.post("/channel/add", dependencies=[Depends(check_rate_limit)])
async def add_channel(request: ChannelRequest):
    """Add a channel to monitor with validation"""
    # Sanitize inputs
    channel_id = DataSanitizer.sanitize_for_display(request.channel_id)
    channel_name = DataSanitizer.sanitize_for_display(request.channel_name)
    
    # Check for duplicates
    with security_manager.concurrency_manager.transaction_lock('channels'):
        success = channel_monitor.add_channel(
            channel_id=channel_id,
            channel_name=channel_name,
            channel_url=request.channel_url
        )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add channel")
    
    security_manager.logger.log_safe('info', f"Channel added: {channel_name}")
    
    return {"status": "success", "channel": channel_name}


@app.get("/channel/list", dependencies=[Depends(check_rate_limit)])
async def list_channels(enabled_only: bool = True):
    """List monitored channels"""
    channels = channel_monitor.get_channels(enabled_only=enabled_only)
    
    # Sanitize channel data
    for channel in channels:
        if 'name' in channel:
            channel['name'] = DataSanitizer.sanitize_for_display(channel['name'])
    
    return {"channels": channels, "count": len(channels)}


@app.get("/channel/check/{channel_id}", dependencies=[Depends(check_rate_limit)])
async def check_channel(
    channel_id: str = Field(..., min_length=1, max_length=50),
    days_back: int = Field(default=7, ge=1, le=30)
):
    """Check a specific channel for new videos"""
    # Sanitize channel_id
    channel_id = DataSanitizer.sanitize_for_display(channel_id)
    
    new_videos = await AsyncManager.with_timeout(
        channel_monitor.check_channel_for_new_videos(channel_id, days_back),
        timeout_seconds=60
    )
    
    return {
        "channel_id": channel_id,
        "new_videos": new_videos,
        "count": len(new_videos)
    }


@app.post("/channel/check-all", dependencies=[Depends(check_rate_limit)])
async def check_all_channels(days_back: int = Field(default=7, ge=1, le=30)):
    """Check all monitored channels for new videos"""
    # Limit concurrent checks
    async with security_manager.concurrency_manager.get_semaphore('channel_checks', limit=3):
        results = await AsyncManager.with_timeout(
            channel_monitor.check_all_channels(days_back),
            timeout_seconds=300  # 5 minutes
        )
    
    total_videos = sum(len(videos) for videos in results.values())
    
    return {
        "channels_checked": len(results),
        "total_new_videos": total_videos,
        "results": results
    }


# Queue management endpoints
@app.get("/queue/pending", dependencies=[Depends(check_rate_limit)])
async def get_pending_videos(limit: Optional[int] = Field(default=50, ge=1, le=100)):
    """Get videos pending download with limits"""
    videos = channel_monitor.get_pending_videos(limit=min(limit, 100))
    
    # Sanitize video data
    for video in videos:
        if 'title' in video:
            video['title'] = DataSanitizer.sanitize_for_display(video['title'])
    
    return {"pending": videos, "count": len(videos)}


@app.post("/queue/process-next", dependencies=[Depends(check_rate_limit)])
async def process_next_video(background_tasks: BackgroundTasks):
    """Process the next pending video with resource checks"""
    # Check system resources
    if security_manager.resource_manager.check_memory_usage() > 900:
        raise HTTPException(status_code=503, detail="System resources exhausted")
    
    videos = channel_monitor.get_pending_videos(limit=1)
    
    if not videos:
        return {"status": "no_pending_videos"}
    
    video = videos[0]
    
    # Sanitize video data
    if 'title' in video:
        video['title'] = DataSanitizer.sanitize_for_display(video['title'])
    
    # Check job lock
    if not security_manager.worker_manager.acquire_job_lock(video['video_id']):
        return {"status": "already_processing", "video": video}
    
    # Queue download
    background_tasks.add_task(
        process_video_with_cleanup,
        video
    )
    
    return {"status": "processing", "video": video}


async def process_video_with_cleanup(video: Dict[str, Any]):
    """Process video with cleanup on completion"""
    try:
        await download_with_limits(
            video['url'],
            "1080p",
            True
        )
        
        # Mark as downloaded
        channel_monitor.mark_video_downloaded(video['video_id'])
        
    finally:
        # Release job lock
        security_manager.worker_manager.release_job_lock(video['video_id'])


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with sanitized responses"""
    # Don't leak internal details in production
    if os.environ.get('ENV') == 'production':
        if exc.status_code >= 500:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": "Internal server error"}
            )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    security_manager.logger.log_safe('error', f"Unhandled exception: {str(exc)}")
    
    # Don't leak stack traces in production
    if os.environ.get('ENV') == 'production':
        return JSONResponse(
            status_code=500,
            content={"detail": "An error occurred"}
        )
    
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with SSL in production
    ssl_keyfile = os.environ.get('SSL_KEYFILE')
    ssl_certfile = os.environ.get('SSL_CERTFILE')
    
    uvicorn.run(
        app,
        host="127.0.0.1",  # Bind to localhost only
        port=8000,
        reload=False,  # Disable in production
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        access_log=False,  # Use custom logging
        log_config=None  # Use custom logging
    )