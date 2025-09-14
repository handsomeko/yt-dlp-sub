"""FastAPI server - Modular API for YouTube operations"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run startup validation before initializing any core modules
from core.startup_validation import run_startup_validation

# Validate at startup - exit if validation fails
if not run_startup_validation(exit_on_error=True):
    print("Startup validation failed. Please fix the issues and restart.")
    sys.exit(1)

from core.downloader import YouTubeDownloader
from core.transcript import TranscriptExtractor
from core.monitor import ChannelMonitor

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Download & Monitor API",
    description="API for downloading YouTube videos, extracting transcripts, and monitoring channels",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core modules
downloader = YouTubeDownloader()
transcript_extractor = TranscriptExtractor()
channel_monitor = ChannelMonitor()

# Pydantic models
class VideoDownloadRequest(BaseModel):
    url: str
    quality: Optional[str] = "1080p"  # 2160p, 1440p, 1080p, 720p, 480p, 360p, 240p, 144p, best, worst
    audio_only: Optional[bool] = False
    audio_format: Optional[str] = "mp3"  # mp3, m4a, wav, opus, flac, best
    video_format: Optional[str] = "mp4"  # mp4, mkv, webm, avi
    channel_name: Optional[str] = None
    extract_transcript: Optional[bool] = True

class TranscriptRequest(BaseModel):
    url: str
    languages: Optional[List[str]] = ["en"]
    save_to_file: Optional[bool] = True

class ChannelRequest(BaseModel):
    channel_id: str
    channel_name: str
    channel_url: Optional[str] = None

class BatchDownloadRequest(BaseModel):
    urls: List[str]
    quality: Optional[str] = "1080p"
    extract_transcripts: Optional[bool] = True


# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/docs",
            "/video/info",
            "/video/download",
            "/transcript/extract",
            "/channel/add",
            "/channel/list",
            "/channel/check",
            "/channel/check-all",
            "/queue/pending"
        ]
    }


# Video endpoints
@app.post("/video/info")
async def get_video_info(url: str):
    """Get video metadata without downloading"""
    info = downloader.get_video_info(url)
    if 'error' in info:
        raise HTTPException(status_code=400, detail=info['error'])
    return info


@app.post("/video/download")
async def download_video(request: VideoDownloadRequest, background_tasks: BackgroundTasks):
    """Download video with optional transcript extraction
    
    Quality options: 2160p, 1440p, 1080p, 720p, 480p, 360p, 240p, 144p, best, worst
    Audio formats: mp3, m4a, wav, opus, flac, best
    Video formats: mp4, mkv, webm, avi
    """
    
    # Start download
    result = downloader.download_video(
        url=request.url,
        quality=request.quality,
        download_audio_only=request.audio_only,
        audio_format=request.audio_format,
        video_format=request.video_format,
        channel_name=request.channel_name
    )
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])
    
    # Extract transcript if requested
    if request.extract_transcript and not request.audio_only:
        transcript_result = transcript_extractor.get_transcript(
            video_url_or_id=request.url,
            video_title=result.get('title')
        )
        result['transcript'] = transcript_result
    
    # Mark as downloaded in monitor
    if result.get('video_id'):
        channel_monitor.mark_video_downloaded(result['video_id'])
    
    return result


@app.post("/audio/download")
async def download_audio(url: str, format: str = "mp3"):
    """Download only audio in specified format"""
    result = downloader.download_video(
        url=url,
        download_audio_only=True,
        audio_format=format
    )
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result


@app.get("/video/formats")
async def get_available_formats():
    """Get list of supported video and audio formats"""
    return {
        "video_qualities": ["2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p", "best", "worst"],
        "audio_formats": ["mp3", "m4a", "wav", "opus", "flac", "best"],
        "video_formats": ["mp4", "mkv", "webm", "avi"]
    }


@app.post("/video/download-batch")
async def download_batch(request: BatchDownloadRequest, background_tasks: BackgroundTasks):
    """Queue multiple videos for download"""
    results = []
    
    for url in request.urls:
        try:
            # Queue each download
            background_tasks.add_task(
                downloader.download_video,
                url,
                request.quality
            )
            results.append({"url": url, "status": "queued"})
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})
    
    return {"queued": len(request.urls), "results": results}


# Transcript endpoints
@app.post("/transcript/extract")
async def extract_transcript(request: TranscriptRequest):
    """Extract transcript without downloading video"""
    result = transcript_extractor.get_transcript(
        video_url_or_id=request.url,
        languages=request.languages,
        save_to_file=request.save_to_file
    )
    
    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result


@app.get("/transcript/languages")
async def get_transcript_languages(video_url: str):
    """Get available transcript languages for a video"""
    video_id = transcript_extractor.extract_video_id(video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    languages = transcript_extractor.get_available_languages(video_id)
    return {"video_id": video_id, "languages": languages}


# Channel monitoring endpoints
@app.post("/channel/add")
async def add_channel(request: ChannelRequest):
    """Add a channel to monitor"""
    success = channel_monitor.add_channel(
        channel_id=request.channel_id,
        channel_name=request.channel_name,
        channel_url=request.channel_url
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add channel")
    
    return {"status": "success", "channel": request.channel_name}


@app.get("/channel/list")
async def list_channels(enabled_only: bool = True):
    """List monitored channels"""
    channels = channel_monitor.get_channels(enabled_only=enabled_only)
    return {"channels": channels, "count": len(channels)}


@app.get("/channel/check/{channel_id}")
async def check_channel(channel_id: str, days_back: int = 7):
    """Check a specific channel for new videos"""
    new_videos = channel_monitor.check_channel_for_new_videos(channel_id, days_back)
    return {
        "channel_id": channel_id,
        "new_videos": new_videos,
        "count": len(new_videos)
    }


@app.post("/channel/check-all")
async def check_all_channels(days_back: int = 7):
    """Check all monitored channels for new videos"""
    results = channel_monitor.check_all_channels(days_back)
    
    total_videos = sum(len(videos) for videos in results.values())
    
    return {
        "channels_checked": len(results),
        "total_new_videos": total_videos,
        "results": results
    }


# Queue management endpoints
@app.get("/queue/pending")
async def get_pending_videos(limit: Optional[int] = 50):
    """Get videos pending download"""
    videos = channel_monitor.get_pending_videos(limit=limit)
    return {"pending": videos, "count": len(videos)}


@app.post("/queue/process-next")
async def process_next_video(background_tasks: BackgroundTasks):
    """Process the next pending video"""
    videos = channel_monitor.get_pending_videos(limit=1)
    
    if not videos:
        return {"status": "no_pending_videos"}
    
    video = videos[0]
    
    # Queue download
    background_tasks.add_task(
        downloader.download_video,
        video['url'],
        "1080p",
        False,
        video.get('channel_name')
    )
    
    # Mark as downloaded
    channel_monitor.mark_video_downloaded(video['video_id'])
    
    return {"status": "processing", "video": video}


# Utility endpoints
@app.post("/channel/load-from-json")
async def load_channels_from_json(json_file_path: str):
    """Load channels from a JSON file"""
    try:
        added = channel_monitor.load_channels_from_json(json_file_path)
        return {"status": "success", "channels_added": added}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)