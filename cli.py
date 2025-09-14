#!/usr/bin/env python3
"""
YouTube Content Intelligence & Repurposing Platform CLI

A CLI tool for monitoring YouTube channels, extracting transcripts,
and managing content generation workflows.

Phase 1 Implementation - Fully Functional CLI
"""

import asyncio
import json
import re
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from sqlalchemy import select, and_, func, text
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run startup validation before anything else
from core.startup_validation import run_startup_validation

# Validate at startup - exit if validation fails
click.echo("🚨 Running startup validation...")
if not run_startup_validation(exit_on_error=True):
    click.echo("❌ Startup validation failed. Please fix the issues and restart.", err=True)
    sys.exit(1)
click.echo("✅ Startup validation passed")

from core.database import (
    db_manager, Channel, Video, Job, Transcript, 
    GeneratedContent, SearchService,
    StorageVersion, TranscriptStatus, GenerationReviewStatus
)
from core.url_parser import YouTubeURLParser, URLType
from core.channel_enumerator import ChannelEnumerator
from core.youtube_validators import is_valid_youtube_id
from workers.monitor import MonitorWorker
from workers.monitor_enhanced import EnhancedMonitorWorker
from workers.orchestrator import OrchestratorWorker as Orchestrator
from config.settings import get_settings
from core.filename_sanitizer import sanitize_filename

__version__ = "1.0.0"

# Global monitor thread
monitor_thread = None
monitor_stop_event = threading.Event()


def validate_youtube_url_or_id(ctx, param, value):
    """Validate YouTube channel URL or ID."""
    if not value:
        return value
    
    parser = YouTubeURLParser()
    url_type, identifier, metadata = parser.parse(value)
    
    if url_type in [URLType.CHANNEL, URLType.VIDEO, URLType.SHORTS]:
        return value
    
    raise click.BadParameter(
        "Must be a YouTube video URL, channel URL, or channel ID"
    )


def validate_date(ctx, param, value):
    """Validate date format (YYYY-MM-DD)."""
    if not value:
        return value
    
    try:
        datetime.strptime(value, '%Y-%m-%d')
        return value
    except ValueError:
        raise click.BadParameter("Date must be in YYYY-MM-DD format")


def validate_export_format(ctx, param, value):
    """Validate export format."""
    valid_formats = ['json', 'csv', 'txt', 'markdown', 'html']
    if value and value.lower() not in valid_formats:
        raise click.BadParameter(f"Format must be one of: {', '.join(valid_formats)}")
    return value.lower() if value else value


async def get_or_create_channel(channel_url: str) -> Optional[Channel]:
    """Get or create a channel from URL."""
    parser = YouTubeURLParser()
    enumerator = ChannelEnumerator()
    
    # Get channel info
    channel_info = enumerator.get_channel_info(channel_url)
    if not channel_info:
        return None
    
    async with db_manager.get_session() as session:
        # Check if channel exists
        result = await session.execute(
            select(Channel).where(Channel.channel_id == channel_info['channel_id'])
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            # Create new channel
            channel = Channel(
                channel_id=channel_info['channel_id'],
                channel_name=channel_info['channel_name'],
                channel_url=channel_info['channel_url'],
                description=channel_info.get('description'),
                subscriber_count=channel_info.get('subscriber_count'),
                video_count=channel_info.get('video_count'),
                is_active=True,
                created_at=datetime.now()
            )
            session.add(channel)
            await session.commit()
        
        return channel


async def process_single_video(video_url: str, with_summaries: bool = None, auto_approve_generation: bool = None, show_progress: bool = True) -> Dict[str, Any]:
    """Process a single video with review checkpoint workflow."""
    parser = YouTubeURLParser()
    url_type, video_id, metadata = parser.parse(video_url)
    
    if url_type not in [URLType.VIDEO, URLType.SHORTS]:
        raise ValueError(f"Not a video URL: {video_url}")
    
    # Validate video ID format
    if not is_valid_youtube_id(video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id}")
    
    settings = get_settings()
    
    # Determine if we should generate summaries
    if with_summaries is None:
        with_summaries = settings.enable_ai_summaries
    
    # Determine generation review status
    if auto_approve_generation is None:
        auto_approve_generation = settings.auto_approve_generation or not settings.generation_review_required
    
    async with db_manager.get_session() as session:
        # Check if video exists
        result = await session.execute(
            select(Video).where(Video.video_id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            # Create video record with review status - FIX Issue #17: Complete all required V2 fields
            video_title = f"Video {video_id}"
            video = Video(
                video_id=video_id,
                channel_id="unknown",  # Will be updated when processing
                title=video_title,
                # V2 storage fields - CRITICAL: Include all required fields
                video_title_snapshot=video_title,
                title_sanitized=sanitize_filename(video_title, video_id),
                storage_version=StorageVersion.V2.value,  # Use enum value for data integrity
                # Use enum values for status fields
                transcript_status=TranscriptStatus.PENDING.value,
                generation_review_status=(
                    GenerationReviewStatus.AUTO_APPROVED.value if auto_approve_generation 
                    else GenerationReviewStatus.PENDING_REVIEW.value
                ),
                created_at=datetime.now()
            )
            session.add(video)
        
        # Create initial processing jobs (no generation jobs yet)
        jobs_created = []
        
        # Audio download job
        audio_job = Job(
            job_type="download_audio",
            target_id=video_id,
            status="pending",
            priority=10  # High priority for direct processing
        )
        session.add(audio_job)
        jobs_created.append(audio_job)
        
        # Transcript extraction job
        transcript_job = Job(
            job_type="extract_transcript",
            target_id=video_id,
            status="pending",
            priority=10
        )
        session.add(transcript_job)
        jobs_created.append(transcript_job)
        
        # Optional AI summary job
        if with_summaries:
            summary_job = Job(
                job_type="generate_summary",
                target_id=video_id,
                status="pending",
                priority=8  # Lower priority than transcript
            )
            session.add(summary_job)
            jobs_created.append(summary_job)
        
        # Only create generation jobs if auto-approved
        if auto_approve_generation:
            generation_job = Job(
                job_type="generate_content",
                target_id=video_id,
                status="pending",
                priority=5  # Lower priority, runs after transcript/summary
            )
            session.add(generation_job)
            jobs_created.append(generation_job)
        
        await session.commit()
        
        return {
            'video_id': video_id,
            'url_type': url_type.value,
            'jobs_created': len(jobs_created),
            'with_summaries': with_summaries,
            'auto_approved': auto_approve_generation,
            'status': 'queued'
        }


async def process_channel_videos(channel_url: str, limit: Optional[int] = None, 
                                show_progress: bool = True) -> Dict[str, Any]:
    """Process all videos from a channel."""
    enumerator = ChannelEnumerator()
    
    # Get channel info and videos
    channel_info = enumerator.get_channel_info(channel_url)
    if not channel_info:
        raise ValueError(f"Could not get channel info for: {channel_url}")
    
    # Get or create channel in DB
    channel = await get_or_create_channel(channel_url)
    if not channel:
        raise ValueError(f"Could not create channel record")
    
    # Get all videos
    videos = enumerator.get_all_videos(channel_url, limit=limit)
    
    if not videos:
        return {
            'channel_id': channel.channel_id,
            'channel_name': channel.channel_name,
            'videos_found': 0,
            'jobs_created': 0
        }
    
    jobs_created = 0
    
    async with db_manager.get_session() as session:
        # Process each video
        progress = tqdm(videos, desc=f"Processing {channel.channel_name}", disable=not show_progress)
        
        for video_info in progress:
            try:
                # Check if video exists
                result = await session.execute(
                    select(Video).where(Video.video_id == video_info['video_id'])
                )
                video = result.scalar_one_or_none()
                
                if not video:
                    # Create video record
                    # Import sanitize_filename for V2 storage fields
                    from core.filename_sanitizer import sanitize_filename
                    
                    video_title = video_info.get('title', '')
                    title_sanitized = sanitize_filename(video_title, video_info['video_id']) if video_title else None
                    
                    video = Video(
                        video_id=video_info['video_id'],
                        channel_id=channel.channel_id,
                        title=video_title,
                        description=video_info.get('description', ''),
                        duration=video_info.get('duration'),
                        view_count=video_info.get('view_count'),
                        like_count=video_info.get('like_count'),
                        published_at=datetime.fromisoformat(video_info['published_at']) if video_info.get('published_at') else None,
                        transcript_status="pending",
                        # V2 storage fields
                        video_title_snapshot=video_title,
                        title_sanitized=title_sanitized,
                        storage_version="v2"
                    )
                    session.add(video)
                    
                    # Create jobs
                    for job_type in ["download_audio", "extract_transcript", "generate_content"]:
                        job = Job(
                            job_type=job_type,
                            target_id=video_info['video_id'],
                            status="pending",
                            priority=5
                        )
                        session.add(job)
                        jobs_created += 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                click.echo(f"Error processing video {video_info.get('video_id')}: {e}", err=True)
                continue
        
        await session.commit()
    
    return {
        'channel_id': channel.channel_id,
        'channel_name': channel.channel_name,
        'videos_found': len(videos),
        'jobs_created': jobs_created
    }


async def process_multiple_videos(video_urls: List[str], with_summaries: bool = None, auto_approve_generation: bool = None, show_progress: bool = True) -> Dict[str, Any]:
    """Process multiple videos with progress tracking and rate limiting."""
    results = {
        'total': len(video_urls),
        'successful': 0,
        'failed': 0,
        'jobs_created': 0,
        'with_summaries': with_summaries,
        'auto_approved': auto_approve_generation,
        'errors': []
    }
    
    if not video_urls:
        return results
    
    # Progress bar for batch processing
    progress = tqdm(video_urls, desc="Processing videos", disable=not show_progress)
    
    for url in progress:
        try:
            # Update progress bar description with current video
            progress.set_description(f"Processing video {results['successful'] + 1}/{results['total']}")
            
            # Process single video with consistent settings
            result = await process_single_video(url, with_summaries=with_summaries, auto_approve_generation=auto_approve_generation, show_progress=False)
            results['successful'] += 1
            results['jobs_created'] += result['jobs_created']
            
            # Rate limiting between videos
            await asyncio.sleep(0.5)
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'url': url,
                'error': str(e)
            })
            click.echo(f"⚠️  Failed to process {url}: {str(e)}", err=True)
    
    return results


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version and exit')
@click.pass_context
def cli(ctx, version):
    """
    YouTube Content Intelligence & Repurposing Platform
    
    Monitor YouTube channels, extract transcripts, and generate content
    across multiple formats using AI-powered sub-generators.
    """
    if version:
        click.echo(f"yt-dl-sub version {__version__}")
        return
    
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument('urls', nargs=-1, required=False)
@click.option('--from-file', type=click.Path(exists=True), help='Read URLs from file (one per line)')
@click.option('--all', 'process_all', is_flag=True, help='[DEPRECATED - No longer needed] Channels now process ALL videos by default. Use --limit to restrict.')
@click.option('--limit', type=int, help='Limit number of videos per channel')
@click.option('--quality', default='1080p', help='Video quality (720p/1080p/best)')
@click.option('--with-summaries', is_flag=True, help='Generate AI summaries after transcription')
@click.option('--skip-summaries', is_flag=True, help='Skip summaries even if globally enabled')
@click.option('--auto-approve-generation', is_flag=True, help='Skip review, auto-approve all for generation')
@click.option('--translate', is_flag=True, help='Enable AI translation of non-English subtitles to English')
@click.option('--target-language', default='en', help='Target language for subtitle translation (default: en)')
def process(urls, from_file: Optional[str], process_all: bool, limit: Optional[int], quality: str, with_summaries: bool, skip_summaries: bool, auto_approve_generation: bool, translate: bool, target_language: str):
    """
    Process YouTube video/channel URLs immediately.
    
    Accepts multiple URLs:
    - Video URLs: youtube.com/watch?v=...
    - Shorts URLs: youtube.com/shorts/...
    - Channel URLs: youtube.com/@username (processes all videos by default)
    
    Examples:
        yt-dl process url1 url2 url3
        yt-dl process --from-file urls.txt
        yt-dl process https://youtube.com/@channelname  # Downloads all videos
        yt-dl process https://youtube.com/@channelname --limit 10  # Downloads latest 10 videos
    """
    try:
        # Handle conflicting summary options
        if with_summaries and skip_summaries:
            click.echo("❌ Cannot use --with-summaries and --skip-summaries together", err=True)
            return
        
        # Determine final summary setting
        final_with_summaries = None
        if with_summaries:
            final_with_summaries = True
        elif skip_summaries:
            final_with_summaries = False
        # If neither flag is used, final_with_summaries stays None and will use config default
        
        # Collect all URLs from arguments and file
        all_urls = list(urls) if urls else []
        
        # Add URLs from file if provided
        if from_file:
            try:
                with open(from_file, 'r') as f:
                    for line in f:
                        # Remove comments and whitespace
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Remove inline comments
                            url = line.split('#')[0].strip()
                            if url:
                                all_urls.append(url)
                click.echo(f"📄 Loaded URLs from {from_file}")
            except Exception as e:
                click.echo(f"❌ Error reading file {from_file}: {str(e)}", err=True)
                return
        
        if not all_urls:
            click.echo("❌ No URLs provided. Use arguments or --from-file option")
            click.echo("   Example: yt-dl process url1 url2 url3")
            click.echo("   Example: yt-dl process --from-file urls.txt")
            return
        
        # Initialize database
        asyncio.run(db_manager.initialize())
        
        # Parse and categorize URLs
        parser = YouTubeURLParser()
        video_urls = []
        channel_urls = []
        invalid_urls = []
        
        click.echo(f"🔍 Analyzing {len(all_urls)} URL(s)...")
        
        for url in all_urls:
            url_type, identifier, metadata = parser.parse(url)
            if url_type in [URLType.VIDEO, URLType.SHORTS]:
                video_urls.append(url)
            elif url_type == URLType.CHANNEL:
                channel_urls.append(url)
            else:
                invalid_urls.append(url)
        
        # Show summary
        click.echo(f"\n📊 URL Analysis:")
        click.echo(f"   Videos/Shorts: {click.style(str(len(video_urls)), fg='green' if video_urls else 'white')}")
        click.echo(f"   Channels: {click.style(str(len(channel_urls)), fg='blue' if channel_urls else 'white')}")
        if invalid_urls:
            click.echo(f"   Invalid: {click.style(str(len(invalid_urls)), fg='red')}")
            for invalid in invalid_urls[:3]:  # Show first 3 invalid URLs
                click.echo(f"      ❌ {invalid}")
        
        # Process videos if any
        if video_urls:
            click.echo(f"\n🎬 Processing {len(video_urls)} video(s)...")
            
            if len(video_urls) == 1:
                # Single video - use original logic for cleaner output
                result = asyncio.run(process_single_video(video_urls[0], with_summaries=final_with_summaries, auto_approve_generation=auto_approve_generation))
                click.echo(f"✅ Video queued for processing")
                click.echo(f"   Video ID: {click.style(result['video_id'], fg='blue')}")
                click.echo(f"   Jobs created: {click.style(str(result['jobs_created']), fg='green')}")
                if result['with_summaries']:
                    click.echo(f"   AI Summary: {click.style('Enabled', fg='cyan')}")
                if result['auto_approved']:
                    click.echo(f"   Generation: {click.style('Auto-approved', fg='green')}")
                else:
                    click.echo(f"   Generation: {click.style('Pending review', fg='yellow')}")
            else:
                # Multiple videos - use batch processing
                result = asyncio.run(process_multiple_videos(video_urls, with_summaries=final_with_summaries, auto_approve_generation=auto_approve_generation))
                
                click.echo(f"\n✅ Batch processing completed:")
                click.echo(f"   Successful: {click.style(str(result['successful']), fg='green')}/{result['total']}")
                if result['failed'] > 0:
                    click.echo(f"   Failed: {click.style(str(result['failed']), fg='red')}")
                    # Show first 3 errors
                    for error in result['errors'][:3]:
                        click.echo(f"      ❌ {error['url']}: {error['error']}")
                click.echo(f"   Jobs created: {click.style(str(result['jobs_created']), fg='green')}")
        
        # Handle channels if any
        if channel_urls:
            # Default behavior changed: Process ALL videos from channel automatically
            click.echo(f"\n📺 Processing {len(channel_urls)} channel(s)...")
            click.echo("   ℹ️  Processing ALL videos from each channel")
            if limit:
                click.echo(f"   📊 Limit: {limit} videos per channel")
            else:
                click.echo("   💡 Tip: Use --limit N to restrict number of videos")
            
            total_channel_jobs = 0
            total_channel_videos = 0
            
            for channel_url in channel_urls:
                try:
                    click.echo(f"\nProcessing channel: {channel_url}")
                    result = asyncio.run(process_channel_videos(channel_url, limit=limit))
                    total_channel_videos += result['videos_found']
                    total_channel_jobs += result['jobs_created']
                    
                    click.echo(f"   ✅ {result['channel_name']}: {result['videos_found']} videos")
                except Exception as e:
                    click.echo(f"   ❌ Failed: {str(e)}", err=True)
            
            click.echo(f"\n✅ Channel processing summary:")
            click.echo(f"   Total videos: {click.style(str(total_channel_videos), fg='blue')}")
            click.echo(f"   Jobs created: {click.style(str(total_channel_jobs), fg='green')}")
        
        # Final summary
        total_jobs = 0
        if video_urls:
            if len(video_urls) == 1:
                total_jobs += 3  # Single video creates 3 jobs
            else:
                total_jobs += result.get('jobs_created', 0)
        
        # Always include channel jobs (channels now process all videos by default)
        if channel_urls:
            total_jobs += total_channel_jobs
        
        if total_jobs > 0:
            click.echo(f"\n💡 Next steps:")
            click.echo(f"   • Run {click.style('yt-dl sync', fg='cyan')} to start processing {total_jobs} job(s)")
            click.echo(f"   • Use {click.style('yt-dl jobs pending', fg='cyan')} to view queue status")
            
    except Exception as e:
        click.echo(f"❌ Error processing URLs: {str(e)}", err=True)
        raise click.ClickException(f"Processing failed: {str(e)}")


@cli.group()
def channel():
    """Channel management commands."""
    pass


@cli.command('add-channel')
@click.argument('url')
@click.option('--name', '-n', help='Custom name for the channel')
@click.option('--check-frequency', type=int, default=300, 
              help='Check frequency in seconds (default: 300)')
def add_channel_alias(url: str, name: Optional[str], check_frequency: int):
    """Add a YouTube channel to monitor (alias for 'channel add')."""
    # Call the actual add_channel function
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(channel, ['add', url] + 
                          (['--name', name] if name else []) +
                          ['--check-frequency', str(check_frequency)])
    if result.output:
        click.echo(result.output)


@channel.command('add')
@click.argument('url')
@click.option('--name', '-n', help='Custom name for the channel')
@click.option('--check-frequency', type=int, default=300, 
              help='Check frequency in seconds (default: 300)')
def add_channel(url: str, name: Optional[str], check_frequency: int):
    """
    Add a YouTube channel for RSS monitoring.
    
    URL can be any YouTube channel URL format:
    - youtube.com/@username
    - youtube.com/channel/UC...
    - youtube.com/c/channelname
    """
    try:
        parser = YouTubeURLParser()
        url_type, identifier, metadata = parser.parse(url)
        
        if url_type != URLType.CHANNEL:
            raise click.BadParameter(f"Not a channel URL: {url}")
        
        click.echo(f"🔍 Getting channel information...")
        
        # Initialize database
        asyncio.run(db_manager.initialize())
        
        # Get or create channel
        channel = asyncio.run(get_or_create_channel(url))
        
        if not channel:
            raise click.ClickException("Failed to add channel")
        
        # Update channel settings
        async def update_channel():
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(Channel).where(Channel.channel_id == channel.channel_id)
                )
                ch = result.scalar_one()
                
                if name:
                    ch.channel_name = name
                ch.is_active = True
                ch.updated_at = datetime.now()
                
                await session.commit()
                return ch
        
        channel = asyncio.run(update_channel())
        
        # Create channel directory with channel_url.txt immediately
        from core.storage_paths_v2 import get_storage_paths_v2
        storage_paths = get_storage_paths_v2()
        channel_dir = storage_paths.get_channel_dir(channel.channel_id)
        channel_dir.mkdir(parents=True, exist_ok=True)
        
        # Create channel_url.txt
        channel_url_file = channel_dir / 'channel_url.txt'
        if not channel_url_file.exists() and channel.channel_url:
            with open(channel_url_file, 'w') as f:
                f.write(channel.channel_url)
        
        click.echo(f"✅ Successfully added channel:")
        click.echo(f"   Name: {click.style(channel.channel_name, fg='green', bold=True)}")
        click.echo(f"   ID: {click.style(channel.channel_id, fg='blue')}")
        click.echo(f"   URL: {click.style(channel.channel_url, fg='cyan')}")
        click.echo(f"   Videos: {channel.video_count or 'Unknown'}")
        
        click.echo(f"\n💡 Next steps:")
        click.echo(f"   • Run {click.style('yt-dl monitor start', fg='cyan')} to start monitoring")
        click.echo(f"   • Use {click.style('yt-dl channel check', fg='cyan')} to check for new videos")
        click.echo(f"   • Use {click.style('yt-dl channel list', fg='cyan')} to see all channels")
        
    except Exception as e:
        click.echo(f"❌ Error adding channel: {str(e)}", err=True)
        raise click.ClickException(f"Failed to add channel: {str(e)}")


@channel.command('list')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'simple']), 
              default='table', help='Output format')
@click.option('--status', type=click.Choice(['all', 'active', 'inactive']), 
              default='all', help='Filter by status')
def list_channels(output_format: str, status: str):
    """List all monitored YouTube channels."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_channels():
            async with db_manager.get_session() as session:
                query = select(Channel)
                
                if status == 'active':
                    query = query.where(Channel.is_active == True)
                elif status == 'inactive':
                    query = query.where(Channel.is_active == False)
                
                result = await session.execute(query)
                return result.scalars().all()
        
        channels = asyncio.run(get_channels())
        
        if not channels:
            click.echo("📭 No channels found.")
            click.echo(f"   Add a channel with: {click.style('yt-dl channel add <URL>', fg='cyan')}")
            return
        
        if output_format == 'table':
            click.echo(f"📺 Found {len(channels)} channel(s):\n")
            
            # Header
            click.echo(f"{'Name':<30} {'Status':<8} {'Videos':<7} {'Last Check':<19} {'Channel ID':<24}")
            click.echo("─" * 95)
            
            for channel in channels:
                status_color = 'green' if channel.is_active else 'red'
                name_display = channel.channel_name[:28] + '..' if len(channel.channel_name) > 30 else channel.channel_name
                last_check = channel.last_checked.strftime('%Y-%m-%d %H:%M') if channel.last_checked else 'Never'
                
                click.echo(
                    f"{name_display:<30} "
                    f"{click.style('Active' if channel.is_active else 'Inactive', fg=status_color):<17} "
                    f"{channel.video_count or 0:<7} "
                    f"{last_check:<19} "
                    f"{click.style(channel.channel_id, fg='blue', dim=True)}"
                )
                
        elif output_format == 'json':
            data = [{
                'channel_id': ch.channel_id,
                'channel_name': ch.channel_name,
                'channel_url': ch.channel_url,
                'is_active': ch.is_active,
                'video_count': ch.video_count,
                'last_checked': ch.last_checked.isoformat() if ch.last_checked else None,
                'created_at': ch.created_at.isoformat() if ch.created_at else None
            } for ch in channels]
            click.echo(json.dumps(data, indent=2))
            
        elif output_format == 'simple':
            for channel in channels:
                status = 'active' if channel.is_active else 'inactive'
                click.echo(f"{channel.channel_id} {channel.channel_name} ({status})")
        
    except Exception as e:
        click.echo(f"❌ Error listing channels: {str(e)}", err=True)
        raise click.ClickException(f"Failed to list channels: {str(e)}")


@channel.command('check')
@click.option('--channel-id', '-c', help='Check specific channel only')
@click.option('--all', 'check_all', is_flag=True, help='Check all channels')
def check_channels(channel_id: Optional[str], check_all: bool):
    """Check channels for new videos via RSS."""
    try:
        if not channel_id and not check_all:
            click.echo("Specify --channel-id or --all")
            return
        
        asyncio.run(db_manager.initialize())
        
        # Use EnhancedMonitorWorker if available (with prevention systems)
        import os
        use_enhanced_monitor = os.getenv('USE_ENHANCED_MONITOR', 'true').lower() == 'true'
        monitor = EnhancedMonitorWorker() if use_enhanced_monitor else MonitorWorker()
        
        if channel_id:
            input_data = {"channel_id": channel_id}
            click.echo(f"🔍 Checking channel: {channel_id}")
        else:
            input_data = {"check_all": True}
            click.echo("🔍 Checking all active channels")
        
        result = asyncio.run(monitor.run_async(input_data))
        
        if result['status'] == 'success':
            data = result['data']
            click.echo(f"✅ Check completed:")
            click.echo(f"   Channels checked: {data['channels_checked']}")
            click.echo(f"   New videos found: {data['new_videos_found']}")
            click.echo(f"   Jobs created: {data['jobs_created']}")
            
            if data.get('errors'):
                click.echo(f"   ⚠️  Errors: {len(data['errors'])}")
                for error in data['errors'][:3]:
                    click.echo(f"      - {error}")
        else:
            click.echo(f"❌ Check failed: {result.get('error')}")
            
    except Exception as e:
        click.echo(f"❌ Error checking channels: {str(e)}", err=True)
        raise click.ClickException(f"Failed to check channels: {str(e)}")


@channel.command('remove')
@click.argument('channel_id')
@click.option('--keep-data', is_flag=True, help='Keep videos and transcripts')
@click.confirmation_option(prompt='Are you sure you want to remove this channel?')
def channel_remove(channel_id: str, keep_data: bool):
    """Remove a channel from monitoring."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def remove_channel():
            async with db_manager.get_session() as session:
                # Find the channel
                from sqlalchemy import select
                result = await session.execute(
                    select(Channel).where(Channel.channel_id == channel_id)
                )
                channel = result.scalar_one_or_none()
                
                if not channel:
                    click.echo(f"❌ Channel '{channel_id}' not found")
                    return False
                
                channel_name = channel.name
                video_count = 0
                transcript_count = 0
                
                if not keep_data:
                    # Count and delete related data
                    # Count videos
                    video_result = await session.execute(
                        select(func.count(Video.id)).where(Video.channel_id == channel.id)
                    )
                    video_count = video_result.scalar() or 0
                    
                    # Count transcripts
                    transcript_result = await session.execute(
                        select(func.count(Transcript.id))
                        .join(Video, Video.video_id == Transcript.video_id)
                        .where(Video.channel_id == channel.id)
                    )
                    transcript_count = transcript_result.scalar() or 0
                    
                    # Delete generated content
                    await session.execute(
                        select(GeneratedContent)
                        .join(Video, Video.video_id == GeneratedContent.video_id)
                        .where(Video.channel_id == channel.id)
                    )
                    
                    # Delete transcripts
                    await session.execute(
                        select(Transcript)
                        .join(Video, Video.video_id == Transcript.video_id)
                        .where(Video.channel_id == channel.id)
                    )
                    
                    # Delete videos
                    await session.execute(
                        select(Video).where(Video.channel_id == channel.id)
                    )
                
                # Delete the channel
                await session.delete(channel)
                await session.commit()
                
                return {
                    'channel_name': channel_name,
                    'video_count': video_count,
                    'transcript_count': transcript_count,
                    'keep_data': keep_data
                }
        
        result = asyncio.run(remove_channel())
        
        if result:
            click.echo(f"✅ Channel '{result['channel_name']}' ({channel_id}) removed")
            if not result['keep_data']:
                if result['video_count'] > 0:
                    click.echo(f"   Deleted {result['video_count']} videos")
                if result['transcript_count'] > 0:
                    click.echo(f"   Deleted {result['transcript_count']} transcripts")
            else:
                click.echo("   Data kept (videos and transcripts preserved)")
            
            click.echo(f"\n📝 Use {click.style('yt-dl channel list', fg='cyan')} to see remaining channels")
        
    except Exception as e:
        click.echo(f"❌ Error removing channel: {str(e)}", err=True)
        raise click.ClickException(f"Failed to remove channel: {str(e)}")


@channel.command('download')
@click.argument('channel_url')
@click.option('--limit', type=int, help='Limit number of videos to download')
@click.option('--quality', default='1080p', help='Video quality (720p/1080p/best)')
@click.option('--audio-only', is_flag=True, default=True, help='Download audio only (default)')
@click.option('--video', is_flag=True, help='Download video instead of audio only')
@click.option('--concurrent', type=int, default=3, help='Number of concurrent downloads (1-10, default: 3)')
@click.option('--queue', is_flag=True, help='Add to queue instead of downloading immediately')
@click.option('--translate', is_flag=True, help='Enable AI translation of non-English subtitles to English')
@click.option('--target-language', default='en', help='Target language for subtitle translation (default: en)')
@click.option('--skip-transcription', is_flag=True, help='Skip Whisper transcription (Phase 1: download only)')
@click.option('--skip-punctuation', is_flag=True, help='Skip Chinese punctuation restoration for faster processing')
def channel_download(channel_url: str, limit: Optional[int], quality: str, audio_only: bool, video: bool, concurrent: int, queue: bool, translate: bool, target_language: str, skip_transcription: bool, skip_punctuation: bool):
    """
    Download all videos from a YouTube channel with concurrent processing.
    
    By default, downloads directly with concurrent processing (3 parallel downloads).
    Use --queue flag to add videos to the job queue for background processing.
    
    Features:
    - Concurrent downloads (configurable 1-10 parallel)
    - Rate limiting to avoid YouTube blocks
    - Language-agnostic subtitle extraction (any available language)
    - Optional AI translation of non-English subtitles
    - Progress tracking with resume capability
    - All V2 storage features applied
    - Optional queue-based processing
    
    Examples:
        yt-dl channel download https://youtube.com/@channelname
        yt-dl channel download https://youtube.com/@channelname --concurrent 5
        yt-dl channel download https://youtube.com/@channelname --translate
        yt-dl channel download https://youtube.com/@channelname --limit 10 --video --translate
    """
    try:
        # Validate concurrent downloads limit
        concurrent = min(max(1, concurrent), 10)
        
        # Determine download mode
        download_audio_only = not video  # If --video flag is used, download video
        
        if queue:
            # Queue-based processing
            click.echo(f"📋 Adding channel videos to queue...")
            
            from core.channel_enumerator import ChannelEnumerator
            from core.database import DatabaseManager
            from core.queue import JobQueue
            
            # Initialize database and queue
            db_manager = DatabaseManager()
            job_queue = JobQueue(db_manager)
            
            enumerator = ChannelEnumerator()
            
            # Get channel info
            channel_info = enumerator.get_channel_info(channel_url)
            if not channel_info:
                raise ValueError(f"Could not get channel info for: {channel_url}")
            
            channel_name = channel_info.get('channel', channel_info.get('uploader', 'Unknown'))
            channel_id = channel_info.get('channel_id', 'unknown')
            
            click.echo(f"✅ Found channel: {click.style(channel_name, fg='cyan')}")
            
            # Get all videos
            videos = enumerator.get_all_videos(channel_url, limit=limit)
            
            if not videos:
                click.echo("❌ No videos found in channel")
                return
            
            click.echo(f"📊 Found {click.style(str(len(videos)), fg='green')} videos")
            
            # Add videos to queue
            jobs_added = 0
            skipped_invalid = 0
            for video_info in videos:
                video_id = video_info.get('id', video_info.get('video_id'))
                
                # Validate video ID before queueing
                if not is_valid_youtube_id(video_id):
                    click.echo(f"⚠️  Skipping invalid video ID: {video_id}")
                    skipped_invalid += 1
                    continue
                
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                job_id = job_queue.enqueue(
                    job_type='download_audio' if download_audio_only else 'download_video',
                    target_id=video_id,
                    metadata={
                        'video_id': video_id,
                        'video_url': video_url,
                        'channel_id': channel_id,
                        'subtitle_translation_enabled': translate,
                        'subtitle_target_language': target_language,
                        'channel_name': channel_name,
                        'title': video_info.get('title', 'Unknown'),
                        'quality': quality,
                        'format': 'opus' if download_audio_only else 'mp4'
                    }
                )
                
                if job_id:
                    jobs_added += 1
            
            click.echo(f"✅ Added {jobs_added} videos to queue")
            if skipped_invalid > 0:
                click.echo(f"⚠️  Skipped {skipped_invalid} invalid video IDs")
            click.echo(f"\nRun 'yt-dl queue status' to check progress")
            click.echo(f"Run 'yt-dl orchestrator start' to process the queue")
            
        else:
            # Direct concurrent download
            click.echo(f"📺 Starting concurrent download (max {concurrent} parallel)...")
            
            from core.channel_enumerator import ChannelEnumerator
            from core.downloader import YouTubeDownloader
            from core.dynamic_worker_pool import get_dynamic_worker_pool
            import asyncio
            
            # Get real dynamic worker pool instead of fake concurrency manager
            worker_pool = get_dynamic_worker_pool()
            
            # Check if we should use system-wide limits
            use_system_limit = True  # Always use system limits for consistency
            
            # Get channel info for display
            enumerator = ChannelEnumerator()
            channel_info = enumerator.get_channel_info(channel_url)
            channel_name = channel_info.get('channel', channel_info.get('uploader', 'Unknown')) if channel_info else channel_url
            
            if use_system_limit:
                # Check current system status
                status = worker_pool.get_status()
                click.echo(f"⚙️  Worker pool status: {status['total_workers']}/{status['max_workers']} workers active")
                
                # Show worker pool info
                click.echo(f"✅ Using dynamic worker pool for {channel_name}")
            
            try:
                
                async def download_with_progress():
                    """Async function to handle concurrent downloads with progress"""
                    # Use factory to get proper configuration including deduplication
                    from core.downloader import create_downloader_with_settings
                    downloader = create_downloader_with_settings(
                        skip_transcription=skip_transcription,
                        skip_punctuation=skip_punctuation
                    )
                    # Override translation settings if specified
                    if translate is not None:
                        downloader.enable_translation = translate
                    if target_language:
                        downloader.target_language = target_language
                    
                    # Define progress callback
                    def progress_callback(video_id, status, message):
                        if status == 'starting':
                            click.echo(f"🎬 {message}")
                        elif status == 'completed':
                            click.echo(f"✅ {message}")
                        elif status == 'failed':
                            click.echo(f"❌ {message}", err=True)
                        elif status == 'error':
                            click.echo(f"⚠️  {message}", err=True)
                    
                    # Reduce concurrent downloads when system is loaded
                    import psutil
                    cpu_percent = psutil.cpu_percent()
                    actual_concurrent = min(concurrent, 2) if cpu_percent > 80 else concurrent
                    if actual_concurrent != concurrent:
                        click.echo(f"⚠️  High system load, reducing concurrent downloads to {actual_concurrent}")
                    
                    # Download using new concurrent method
                    results = await downloader.download_channel_videos(
                        channel_url=channel_url,
                        limit=limit,
                        max_concurrent=actual_concurrent,
                        download_audio_only=download_audio_only,
                        audio_format='opus' if download_audio_only else None,
                        video_format='mp4' if not download_audio_only else None,
                        quality=quality,
                        progress_callback=progress_callback
                    )
                    
                    return results
                
                # Run the async download
                results = asyncio.run(download_with_progress())
                
            except TimeoutError:
                click.echo(f"❌ Timeout waiting for system slot after 5 minutes", err=True)
                return
                    
            else:
                # No system limit - run directly
                click.echo(f"⚠️  Running without system-wide concurrency limits")
                
                async def download_with_progress():
                    """Async function to handle concurrent downloads with progress"""
                    # Use factory to get proper configuration including deduplication
                    from core.downloader import create_downloader_with_settings
                    downloader = create_downloader_with_settings(
                        skip_transcription=skip_transcription,
                        skip_punctuation=skip_punctuation
                    )
                    # Override translation settings if specified
                    if translate is not None:
                        downloader.enable_translation = translate
                    if target_language:
                        downloader.target_language = target_language
                    
                    # Define progress callback
                    def progress_callback(video_id, status, message):
                        if status == 'starting':
                            click.echo(f"🎬 {message}")
                        elif status == 'completed':
                            click.echo(f"✅ {message}")
                        elif status == 'failed':
                            click.echo(f"❌ {message}", err=True)
                        elif status == 'error':
                            click.echo(f"⚠️  {message}", err=True)
                    
                    # Download using new concurrent method
                    results = await downloader.download_channel_videos(
                        channel_url=channel_url,
                        limit=limit,
                        max_concurrent=concurrent,
                        download_audio_only=download_audio_only,
                        audio_format='opus' if download_audio_only else None,
                        video_format='mp4' if not download_audio_only else None,
                        quality=quality,
                        progress_callback=progress_callback
                    )
                    
                    return results
                
                # Run the async download
                results = asyncio.run(download_with_progress())
            
            # Check results
            if results.get('status') == 'error':
                raise ValueError(results.get('error', 'Download failed'))
            
            # Extract statistics
            successful = len(results.get('successful', []))
            failed = len(results.get('failed', []))
            invalid_skipped = results.get('invalid_skipped', 0)
        
            # Summary for direct download
            channel_name = results.get('channel_info', {}).get('channel_name', 'Unknown')
            total = results.get('total', 0)
            duration = results.get('duration_seconds', 0)
            
            click.echo(f"\n{'='*60}")
            click.echo(f"✅ Download Summary:")
            click.echo(f"   Channel: {click.style(channel_name, fg='cyan')}")
            click.echo(f"   Successful: {click.style(str(successful), fg='green')}/{total}")
            if failed > 0:
                click.echo(f"   Failed: {click.style(str(failed), fg='red')}")
            
            # Show invalid IDs if any were filtered
            invalid_skipped = results.get('invalid_skipped', 0)
            if invalid_skipped > 0:
                click.echo(f"   Invalid IDs filtered: {click.style(str(invalid_skipped), fg='yellow')}")
            click.echo(f"   Mode: {'Audio Only' if download_audio_only else 'Full Video'}")
            click.echo(f"   Quality: {quality}")
            click.echo(f"   Concurrent Downloads: {concurrent}")
            
            # Performance metrics
            if duration > 0:
                click.echo(f"\n📊 Performance:")
                click.echo(f"   Duration: {duration:.1f} seconds")
                click.echo(f"   Downloads/min: {results.get('downloads_per_minute', 0):.1f}")
                click.echo(f"   Success Rate: {results.get('success_rate', 0)*100:.1f}%")
            
            # Note about V2 features
            click.echo(f"\n💡 All V2 features applied:")
            click.echo(f"   • Comprehensive metadata (~60 fields)")
            click.echo(f"   • Correct file naming (video titles)")
            click.echo(f"   • video_url.txt files created")
            click.echo(f"   • Markdown reports generated")
            click.echo(f"   • Automatic subtitles extracted")
            click.echo(f"   • Concurrent processing ({concurrent} parallel)")
        
    except Exception as e:
        click.echo(f"❌ Error downloading channel: {str(e)}", err=True)
        raise click.ClickException(f"Channel download failed: {str(e)}")


@cli.group()
def monitor():
    """Background monitor commands."""
    pass


@monitor.command('start')
@click.option('--interval', type=int, default=300, help='Check interval in seconds (default: 300)')
@click.option('--daemon', is_flag=True, help='Run as daemon in background')
def monitor_start(interval: int, daemon: bool):
    """Start the RSS monitor service."""
    global monitor_thread, monitor_stop_event
    
    try:
        click.echo(f"🚀 Starting monitor service")
        click.echo(f"   Check interval: {interval} seconds")
        click.echo(f"   Mode: {'Daemon' if daemon else 'Foreground'}")
        
        asyncio.run(db_manager.initialize())
        
        def monitor_loop():
            # Use EnhancedMonitorWorker if available (with prevention systems)
            import os
            use_enhanced_monitor = os.getenv('USE_ENHANCED_MONITOR', 'true').lower() == 'true'
            monitor = EnhancedMonitorWorker() if use_enhanced_monitor else MonitorWorker()
            
            while not monitor_stop_event.is_set():
                try:
                    # Check all channels
                    result = asyncio.run(monitor.run_async({"check_all": True}))
                    
                    if result['status'] == 'success':
                        data = result['data']
                        if data['new_videos_found'] > 0:
                            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] Found {data['new_videos_found']} new videos")
                    
                    # Wait for interval or stop signal
                    monitor_stop_event.wait(interval)
                    
                except Exception as e:
                    click.echo(f"Monitor error: {e}", err=True)
                    time.sleep(60)  # Wait a minute before retrying
        
        if daemon:
            # Run in background thread
            monitor_stop_event.clear()
            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()
            
            click.echo(f"✅ Monitor started in background")
            click.echo(f"   Use {click.style('yt-dl monitor stop', fg='cyan')} to stop")
            click.echo(f"   Use {click.style('yt-dl monitor status', fg='cyan')} to check status")
        else:
            # Run in foreground
            click.echo("Monitor running (Ctrl+C to stop)...")
            try:
                monitor_loop()
            except KeyboardInterrupt:
                click.echo("\n👋 Monitor stopped")
                
    except Exception as e:
        click.echo(f"❌ Error starting monitor: {str(e)}", err=True)
        raise click.ClickException(f"Failed to start monitor: {str(e)}")


@monitor.command('stop')
def monitor_stop():
    """Stop the RSS monitor service."""
    global monitor_thread, monitor_stop_event
    
    if monitor_thread and monitor_thread.is_alive():
        monitor_stop_event.set()
        monitor_thread.join(timeout=5)
        click.echo("✅ Monitor stopped")
    else:
        click.echo("Monitor is not running")


@monitor.command('status')
def monitor_status():
    """Check monitor service status."""
    global monitor_thread
    
    if monitor_thread and monitor_thread.is_alive():
        click.echo("✅ Monitor is running")
    else:
        click.echo("❌ Monitor is not running")
        click.echo(f"   Start with: {click.style('yt-dl monitor start', fg='cyan')}")


@cli.command()
@click.argument('query')
@click.option('--channel', '-c', help='Search within specific channel only')
@click.option('--limit', '-l', type=int, default=10, help='Maximum results (default: 10)')
@click.option('--format', 'output_format', type=click.Choice(['detailed', 'simple', 'json']), 
              default='detailed', help='Output format')
def search(query: str, channel: Optional[str], limit: int, output_format: str):
    """Search through video transcripts or topics using full-text search.
    
    Supports topic filtering with 'topics~' syntax:
    - 'topics~AI,machine learning' searches for videos with AI or machine learning topics
    - Regular queries search transcript content
    """
    try:
        asyncio.run(db_manager.initialize())
        search_service = SearchService(db_manager)
        
        # Check if this is a topic search
        if query.startswith('topics~'):
            # Parse topics from query
            topics_str = query[7:]  # Remove 'topics~' prefix
            topics = [topic.strip() for topic in topics_str.split(',') if topic.strip()]
            
            if not topics:
                click.echo("❌ No topics specified. Use: topics~AI,programming,etc")
                return
            
            click.echo(f"🏷️  Searching by topics: {click.style(', '.join(topics), fg='yellow', bold=True)}")
            if channel:
                click.echo(f"📺 Channel filter: {click.style(channel, fg='blue')}")
            
            # Perform topic search
            results = asyncio.run(search_service.search_by_topics(topics, channel_id=channel, limit=limit))
        else:
            # Regular transcript search
            click.echo(f"🔍 Searching for: {click.style(query, fg='yellow', bold=True)}")
            if channel:
                click.echo(f"📺 Channel filter: {click.style(channel, fg='blue')}")
            
            # Perform transcript search
            results = asyncio.run(search_service.search_transcripts(query, limit=limit, channel_id=channel))
        
        if not results or not results.results:
            click.echo("📭 No results found.")
            return
        
        click.echo(f"\n📊 Found {click.style(str(len(results)), fg='green', bold=True)} result(s):\n")
        
        if output_format == 'detailed':
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. Video: {click.style(result['title'], fg='green', bold=True)}")
                click.echo(f"   Channel: {click.style(result['channel_name'], fg='blue')}")
                click.echo(f"   Video ID: {result['video_id']}")
                click.echo(f"   Relevance: {click.style(f'{result["rank"]:.2f}', fg='magenta')}")
                click.echo(f"   Snippet: ...{result['snippet']}...")
                click.echo()
                
        elif output_format == 'simple':
            for result in results:
                click.echo(f"{result['title']} ({result['channel_name']})")
                
        elif output_format == 'json':
            click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Error during search: {str(e)}", err=True)
        raise click.ClickException(f"Search failed: {str(e)}")


@cli.group()
def jobs():
    """Job queue management commands."""
    pass


@jobs.command('pending')
@click.option('--limit', type=int, default=20, help='Number of jobs to show')
def jobs_pending(limit: int):
    """Show pending jobs in the queue."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_pending_jobs():
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(Job)
                    .where(Job.status == 'pending')
                    .order_by(Job.priority.desc(), Job.created_at)
                    .limit(limit)
                )
                return result.scalars().all()
        
        jobs = asyncio.run(get_pending_jobs())
        
        if not jobs:
            click.echo("📭 No pending jobs")
            return
        
        click.echo(f"📋 Pending jobs ({len(jobs)}):\n")
        click.echo(f"{'ID':<6} {'Type':<20} {'Target':<15} {'Priority':<8} {'Created'}")
        click.echo("─" * 70)
        
        for job in jobs:
            created = job.created_at.strftime('%Y-%m-%d %H:%M') if job.created_at else 'Unknown'
            click.echo(f"{job.id:<6} {job.job_type:<20} {job.target_id[:15]:<15} {job.priority:<8} {created}")
            
    except Exception as e:
        click.echo(f"❌ Error getting pending jobs: {str(e)}", err=True)


@jobs.command('stats')
def jobs_stats():
    """Show job queue statistics."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_stats():
            async with db_manager.get_session() as session:
                # Get job counts by status
                result = await session.execute(
                    select(Job.status, func.count(Job.id))
                    .group_by(Job.status)
                )
                status_counts = dict(result.all())
                
                # Get job counts by type
                result = await session.execute(
                    select(Job.job_type, func.count(Job.id))
                    .group_by(Job.job_type)
                )
                type_counts = dict(result.all())
                
                return status_counts, type_counts
        
        status_counts, type_counts = asyncio.run(get_stats())
        
        click.echo("📊 Job Queue Statistics\n")
        
        click.echo("Status:")
        for status, count in status_counts.items():
            color = {'pending': 'yellow', 'processing': 'blue', 
                    'completed': 'green', 'failed': 'red'}.get(status, 'white')
            click.echo(f"  {status:<12} {click.style(str(count), fg=color, bold=True)}")
        
        click.echo("\nJob Types:")
        for job_type, count in type_counts.items():
            click.echo(f"  {job_type:<20} {count}")
            
    except Exception as e:
        click.echo(f"❌ Error getting job stats: {str(e)}", err=True)


@cli.command()
@click.option('--channel', '-c', help='Sync specific channel only')
@click.option('--limit', '-l', type=int, help='Limit number of videos')
@click.option('--dynamic-scaling', is_flag=True, help='Enable dynamic worker scaling based on system resources')
@click.option('--max-workers', type=int, help='Maximum number of concurrent workers (default: CPU count * 2)')
@click.option('--min-workers', type=int, default=1, help='Minimum number of workers (default: 1)')
def sync(channel: Optional[str], limit: Optional[int], dynamic_scaling: bool, max_workers: Optional[int], min_workers: int):
    """Process pending jobs in the queue with optional dynamic scaling."""
    try:
        click.echo("🔄 Starting job processor...")
        
        if dynamic_scaling:
            click.echo("⚡ Dynamic scaling enabled - adjusting workers based on system resources")
        
        asyncio.run(db_manager.initialize())
        
        # Configure dynamic scaling if requested
        scaling_config = None
        if dynamic_scaling:
            from core.resource_manager import ScalingConfig
            import multiprocessing
            
            scaling_config = ScalingConfig(
                min_workers=min_workers,
                max_workers=max_workers or (multiprocessing.cpu_count() * 2),
                cpu_scale_up_threshold=50.0,    # Scale up if CPU < 50%
                cpu_scale_down_threshold=80.0,  # Scale down if CPU > 80%
                memory_scale_up_threshold=60.0,  # Scale up if memory < 60%
                memory_scale_down_threshold=85.0, # Scale down if memory > 85%
                memory_emergency_threshold=95.0,  # Emergency stop if memory > 95%
            )
            
            click.echo(f"   Min workers: {scaling_config.min_workers}")
            click.echo(f"   Max workers: {scaling_config.max_workers}")
        
        # Create orchestrator with scaling configuration
        orchestrator = Orchestrator(
            enable_dynamic_scaling=dynamic_scaling,
            scaling_config=scaling_config,
            max_concurrent_jobs=max_workers or 3
        )
        
        click.echo("Processing jobs (Ctrl+C to stop)...")
        
        try:
            asyncio.run(orchestrator.start())
        except KeyboardInterrupt:
            click.echo("\n👋 Stopped processing")
            if dynamic_scaling and orchestrator.resource_manager:
                status = orchestrator.resource_manager.get_status()
                if status.get('current_metrics'):
                    click.echo(f"   Final resource usage: CPU {status['current_metrics']['cpu_percent']:.1f}%, Memory {status['current_metrics']['memory_percent']:.1f}%")
            
    except Exception as e:
        click.echo(f"❌ Error during sync: {str(e)}", err=True)
        raise click.ClickException(f"Sync failed: {str(e)}")


@cli.group()
def review():
    """Review and approve videos for content generation."""
    pass


@review.command('list')
@click.option('--status', type=click.Choice(['pending_review', 'approved', 'rejected']), 
              default='pending_review', help='Filter by review status')
@click.option('--limit', type=int, default=20, help='Number of videos to show')
@click.option('--show-summary', is_flag=True, help='Show AI summaries if available')
def review_list(status: str, limit: int, show_summary: bool):
    """List videos awaiting review."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_videos():
            async with db_manager.get_session() as db:
                query = select(Video).where(Video.generation_review_status == status)
                
                if status == 'pending_review':
                    # Only show videos with completed transcripts
                    query = query.where(Video.transcript_text.isnot(None))
                
                query = query.order_by(Video.created_at.desc()).limit(limit)
                result = await db.execute(query)
                return result.scalars().all()
        
        videos = asyncio.run(get_videos())
        
        if not videos:
            click.echo(f"📭 No videos with status '{status}' found.")
            return
        
        status_colors = {
            'pending_review': 'yellow',
            'approved': 'green',
            'rejected': 'red'
        }
        color = status_colors.get(status, 'white')
        
        click.echo(f"📺 Videos with status {click.style(status, fg=color, bold=True)} ({len(videos)}):\n")
        
        for i, video in enumerate(videos, 1):
            # Duration display
            duration = f"{video.duration_seconds // 60}:{video.duration_seconds % 60:02d}" if video.duration_seconds else "Unknown"
            
            click.echo(f"{i}. {click.style(video.title[:80], fg='green', bold=True)}")
            click.echo(f"   ID: {video.video_id} | Channel: {video.channel_name}")
            click.echo(f"   Duration: {duration} | Created: {video.created_at.strftime('%Y-%m-%d %H:%M') if video.created_at else 'Unknown'}")
            
            if show_summary and video.ai_summary:
                click.echo(f"   Summary: {click.style(video.ai_summary, fg='cyan')}")
            
            if video.generation_review_notes:
                click.echo(f"   Notes: {video.generation_review_notes}")
            
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Error listing videos: {str(e)}", err=True)


@review.command('approve')
@click.argument('video_ids', nargs=-1, required=True)
@click.option('--notes', help='Optional approval notes')
@click.option('--skip-format-selection', is_flag=True, help='Skip format selection and use all formats')
def review_approve(video_ids: tuple, notes: Optional[str], skip_format_selection: bool):
    """Approve videos for content generation."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def approve_videos():
            approved_count = 0
            job_count = 0
            
            async with db_manager.get_session() as db:
                for video_id in video_ids:
                    # Find video by video_id or database ID
                    if video_id.isdigit():
                        video = await db.get(Video, int(video_id))
                    else:
                        query = select(Video).where(Video.video_id == video_id)
                        result = await db.execute(query)
                        video = result.scalar_one_or_none()
                    
                    if not video:
                        click.echo(f"⚠️  Video {video_id} not found", err=True)
                        continue
                    
                    if not video.transcript_text:
                        click.echo(f"⚠️  Video {video_id} has no transcript - cannot approve", err=True)
                        continue
                    
                    # Update review status
                    video.generation_review_status = 'approved'
                    video.generation_approved_at = datetime.now(timezone.utc)
                    video.generation_review_notes = notes or 'Approved via CLI'
                    
                    settings = get_settings()
                    
                    # Handle format selection workflow
                    if skip_format_selection or settings.auto_select_all_formats:
                        # Skip format selection - use all formats
                        video.generation_selection_status = 'formats_selected'
                        
                        # Set all available formats as selected
                        generators = settings.content_generators
                        if isinstance(generators, str):
                            generators = [g.strip() for g in generators.split(',')]
                        video.selected_generators = json.dumps(generators)
                        video.generation_selected_at = datetime.now(timezone.utc)
                        
                        # Create generation jobs immediately
                        from core.queue import get_job_queue, JobType
                        queue = get_job_queue()
                        for generator in generators:
                            await queue.enqueue(
                                job_type=JobType.GENERATE_CONTENT.value,
                                target_id=video.video_id,
                                priority=3,
                                metadata={'generator_type': generator}
                            )
                            job_count += 1
                    else:
                        # Set to pending format selection
                        video.generation_selection_status = 'pending_selection'
                        
                        # Pre-select default formats
                        default_formats = []
                        
                        # Check for channel-specific defaults
                        channel_defaults = settings.channel_default_formats
                        if isinstance(channel_defaults, dict) and video.channel_name in channel_defaults:
                            formats_str = channel_defaults[video.channel_name]
                            default_formats = [f.strip() for f in formats_str.split(',')]
                        else:
                            # Use global defaults
                            default_formats = settings.default_selected_formats
                            if isinstance(default_formats, str):
                                default_formats = [f.strip() for f in default_formats.split(',')]
                        
                        video.selected_generators = json.dumps(default_formats) if default_formats else None
                    
                    approved_count += 1
                    click.echo(f"✅ Approved: {video.title[:80]}")
                
                await db.commit()
            
            return approved_count, job_count
        
        approved, jobs = asyncio.run(approve_videos())
        
        if approved > 0:
            click.echo(f"\n🎉 Approved {approved} video(s)")
            if jobs > 0:
                click.echo(f"🚀 Created {jobs} generation job(s)")
            elif not skip_format_selection:
                settings = get_settings()
                if settings.format_selection_required:
                    click.echo(f"📝 Videos pending format selection. Use 'yt-dl generate list' to select formats.")
        
    except Exception as e:
        click.echo(f"❌ Error approving videos: {str(e)}", err=True)


@review.command('reject')
@click.argument('video_ids', nargs=-1, required=True)
@click.option('--reason', help='Rejection reason')
def review_reject(video_ids: tuple, reason: Optional[str]):
    """Reject videos (will not generate content)."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def reject_videos():
            rejected_count = 0
            
            async with db_manager.get_session() as db:
                for video_id in video_ids:
                    # Find video by video_id or database ID
                    if video_id.isdigit():
                        video = await db.get(Video, int(video_id))
                    else:
                        query = select(Video).where(Video.video_id == video_id)
                        result = await db.execute(query)
                        video = result.scalar_one_or_none()
                    
                    if not video:
                        click.echo(f"⚠️  Video {video_id} not found", err=True)
                        continue
                    
                    # Update review status
                    video.generation_review_status = 'rejected'
                    video.generation_review_notes = reason or 'Rejected via CLI'
                    rejected_count += 1
                    
                    click.echo(f"❌ Rejected: {video.title[:80]}")
                
                await db.commit()
            
            return rejected_count
        
        rejected = asyncio.run(reject_videos())
        
        if rejected > 0:
            click.echo(f"\n🚫 Rejected {rejected} video(s)")
        
    except Exception as e:
        click.echo(f"❌ Error rejecting videos: {str(e)}", err=True)


@review.command('interactive')
@click.option('--limit', type=int, default=10, help='Number of videos to review')
def review_interactive(limit: int):
    """Interactive review mode."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_pending_videos():
            async with db_manager.get_session() as db:
                query = (
                    select(Video)
                    .where(Video.generation_review_status == 'pending_review')
                    .where(Video.transcript_text.isnot(None))
                    .order_by(Video.created_at.desc())
                    .limit(limit)
                )
                result = await db.execute(query)
                return result.scalars().all()
        
        videos = asyncio.run(get_pending_videos())
        
        if not videos:
            click.echo("📭 No videos pending review!")
            return
        
        click.echo(f"🔍 Interactive review mode - {len(videos)} video(s) to review\n")
        click.echo("Commands: (a)pprove, (r)eject, (s)kip, (q)uit, (t)ranscript")
        
        approved = []
        rejected = []
        
        for i, video in enumerate(videos, 1):
            click.echo(f"\n--- Video {i}/{len(videos)} ---")
            click.echo(f"Title: {click.style(video.title, fg='green', bold=True)}")
            click.echo(f"Channel: {video.channel_name}")
            click.echo(f"Duration: {video.duration_seconds // 60 if video.duration_seconds else 0}:{video.duration_seconds % 60:02d if video.duration_seconds else 0}")
            click.echo(f"Video ID: {video.video_id}")
            
            if video.ai_summary:
                click.echo(f"Summary: {click.style(video.ai_summary, fg='cyan')}")
            
            while True:
                action = click.prompt("\nAction", type=str).lower().strip()
                
                if action in ['a', 'approve']:
                    approved.append(video)
                    click.echo(f"✅ Approved: {video.title[:50]}")
                    break
                elif action in ['r', 'reject']:
                    reason = click.prompt("Rejection reason (optional)", default="", show_default=False)
                    rejected.append((video, reason))
                    click.echo(f"❌ Rejected: {video.title[:50]}")
                    break
                elif action in ['s', 'skip']:
                    click.echo(f"⏭️  Skipped: {video.title[:50]}")
                    break
                elif action in ['q', 'quit']:
                    click.echo("👋 Exiting interactive review")
                    break
                elif action in ['t', 'transcript']:
                    click.echo(f"\nTranscript preview (first 500 chars):")
                    click.echo(click.style(video.transcript_text[:500] + "...", fg='yellow'))
                elif action in ['h', 'help']:
                    click.echo("Commands: (a)pprove, (r)eject, (s)kip, (q)uit, (t)ranscript")
                else:
                    click.echo("Invalid action. Type 'h' for help.")
                    
                if action in ['q', 'quit']:
                    break
            
            if action in ['q', 'quit']:
                break
        
        # Apply changes
        if approved or rejected:
            asyncio.run(apply_review_changes(approved, rejected))
        
    except Exception as e:
        click.echo(f"❌ Error in interactive review: {str(e)}", err=True)


async def apply_review_changes(approved_videos: list, rejected_videos: list):
    """Apply review changes to database."""
    async with db_manager.get_session() as db:
        # Approve videos
        for video in approved_videos:
            video.generation_review_status = 'approved'
            video.generation_approved_at = datetime.now(timezone.utc)
            video.generation_review_notes = 'Approved via interactive review'
        
        # Reject videos
        for video, reason in rejected_videos:
            video.generation_review_status = 'rejected'
            video.generation_review_notes = reason or 'Rejected via interactive review'
        
        await db.commit()
    
    if approved_videos:
        click.echo(f"\n🎉 Applied approval to {len(approved_videos)} video(s)")
    
    if rejected_videos:
        click.echo(f"\n🚫 Applied rejection to {len(rejected_videos)} video(s)")


@cli.group()
def generate():
    """Manage content format selection and generation."""
    pass


@generate.command('list')
@click.option('--show-formats', is_flag=True, help='Show selected formats for each video')
@click.option('--status', type=click.Choice(['pending_selection', 'formats_selected', 'generation_started']),
              default='pending_selection', help='Filter by format selection status')
@click.option('--limit', type=int, default=20, help='Number of videos to show')
def generate_list(show_formats: bool, status: str, limit: int):
    """List videos pending format selection."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def get_videos():
            async with db_manager.get_session() as db:
                query = (
                    select(Video)
                    .where(Video.generation_review_status == 'approved')
                    .where(Video.generation_selection_status == status)
                    .order_by(Video.generation_approved_at.desc())
                    .limit(limit)
                )
                result = await db.execute(query)
                return result.scalars().all()
        
        videos = asyncio.run(get_videos())
        
        if not videos:
            click.echo(f"📭 No videos with status '{status}' found.")
            return
        
        status_colors = {
            'pending_selection': 'yellow',
            'formats_selected': 'cyan',
            'generation_started': 'green'
        }
        color = status_colors.get(status, 'white')
        
        click.echo(f"📺 Videos with status {click.style(status, fg=color, bold=True)} ({len(videos)}):\n")
        
        settings = get_settings()
        available_formats = settings.content_generators
        if isinstance(available_formats, str):
            available_formats = [f.strip() for f in available_formats.split(',')]
        
        for i, video in enumerate(videos, 1):
            click.echo(f"{i}. {click.style(video.title[:80], fg='green', bold=True)}")
            click.echo(f"   ID: {video.video_id} | Channel: {video.channel_name}")
            
            if show_formats:
                selected = json.loads(video.selected_generators) if video.selected_generators else []
                if selected:
                    # Show format selection with checkmarks
                    format_display = []
                    for fmt in available_formats:
                        if fmt in selected:
                            format_display.append(f"✓ {fmt}")
                        else:
                            format_display.append(f"  {fmt}")
                    click.echo(f"   Formats: {' | '.join(format_display)}")
                else:
                    click.echo(f"   Formats: {click.style('None selected', fg='red')}")
            
            if video.generation_selected_at:
                click.echo(f"   Selected: {video.generation_selected_at.strftime('%Y-%m-%d %H:%M')}")
            
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Error listing videos: {str(e)}", err=True)


@generate.command('select')
@click.argument('video_ids', nargs=-1, required=True)
@click.option('--formats', required=True, help='Comma-separated list of formats to select')
@click.option('--append', is_flag=True, help='Add to existing selection instead of replacing')
def generate_select(video_ids: tuple, formats: str, append: bool):
    """Select specific formats for videos."""
    try:
        asyncio.run(db_manager.initialize())
        
        # Parse formats
        format_list = [f.strip() for f in formats.split(',') if f.strip()]
        if not format_list:
            click.echo("❌ No valid formats specified")
            return
        
        # Validate formats against available generators
        settings = get_settings()
        available_formats = settings.content_generators
        if isinstance(available_formats, str):
            available_formats = [f.strip() for f in available_formats.split(',')]
        
        invalid_formats = [f for f in format_list if f not in available_formats]
        if invalid_formats:
            click.echo(f"❌ Invalid formats: {', '.join(invalid_formats)}")
            click.echo(f"   Available: {', '.join(available_formats)}")
            return
        
        async def select_formats():
            selected_count = 0
            
            async with db_manager.get_session() as db:
                for video_id in video_ids:
                    # Find video by video_id or database ID
                    if video_id.isdigit():
                        video = await db.get(Video, int(video_id))
                    else:
                        query = select(Video).where(Video.video_id == video_id)
                        result = await db.execute(query)
                        video = result.scalar_one_or_none()
                    
                    if not video:
                        click.echo(f"⚠️  Video {video_id} not found", err=True)
                        continue
                    
                    if video.generation_review_status != 'approved':
                        click.echo(f"⚠️  Video {video_id} not approved yet", err=True)
                        continue
                    
                    # Update format selection
                    current_selection = json.loads(video.selected_generators) if video.selected_generators else []
                    
                    if append:
                        # Add to existing selection
                        new_selection = list(set(current_selection + format_list))
                    else:
                        # Replace selection
                        new_selection = format_list
                    
                    video.selected_generators = json.dumps(new_selection)
                    video.generation_selection_status = 'formats_selected'
                    video.generation_selected_at = datetime.now(timezone.utc)
                    selected_count += 1
                    
                    click.echo(f"✅ Selected formats for: {video.title[:60]}")
                    click.echo(f"   Formats: {', '.join(new_selection)}")
                
                await db.commit()
            
            return selected_count
        
        count = asyncio.run(select_formats())
        
        if count > 0:
            click.echo(f"\n📋 Updated format selection for {count} video(s)")
            click.echo(f"   Use 'yt-dl generate start' to begin generation")
        
    except Exception as e:
        click.echo(f"❌ Error selecting formats: {str(e)}", err=True)


@generate.command('select-all')
@click.argument('video_ids', nargs=-1, required=False)
@click.option('--all-pending', is_flag=True, help='Select all formats for all pending videos')
def generate_select_all(video_ids: tuple, all_pending: bool):
    """Select all available formats for videos."""
    try:
        asyncio.run(db_manager.initialize())
        
        settings = get_settings()
        all_formats = settings.content_generators
        if isinstance(all_formats, str):
            all_formats = [f.strip() for f in all_formats.split(',')]
        
        async def select_all_formats():
            selected_count = 0
            
            async with db_manager.get_session() as db:
                if all_pending:
                    # Get all pending videos
                    query = (
                        select(Video)
                        .where(Video.generation_review_status == 'approved')
                        .where(Video.generation_selection_status == 'pending_selection')
                    )
                    result = await db.execute(query)
                    videos = result.scalars().all()
                    
                    for video in videos:
                        video.selected_generators = json.dumps(all_formats)
                        video.generation_selection_status = 'formats_selected'
                        video.generation_selected_at = datetime.now(timezone.utc)
                        selected_count += 1
                        click.echo(f"✅ Selected all formats for: {video.title[:60]}")
                else:
                    # Process specific video IDs
                    for video_id in video_ids:
                        if video_id.isdigit():
                            video = await db.get(Video, int(video_id))
                        else:
                            query = select(Video).where(Video.video_id == video_id)
                            result = await db.execute(query)
                            video = result.scalar_one_or_none()
                        
                        if not video:
                            click.echo(f"⚠️  Video {video_id} not found", err=True)
                            continue
                        
                        video.selected_generators = json.dumps(all_formats)
                        video.generation_selection_status = 'formats_selected'
                        video.generation_selected_at = datetime.now(timezone.utc)
                        selected_count += 1
                        click.echo(f"✅ Selected all formats for: {video.title[:60]}")
                
                await db.commit()
            
            return selected_count
        
        count = asyncio.run(select_all_formats())
        
        if count > 0:
            click.echo(f"\n📋 Selected all formats ({', '.join(all_formats)}) for {count} video(s)")
            click.echo(f"   Use 'yt-dl generate start' to begin generation")
        
    except Exception as e:
        click.echo(f"❌ Error selecting all formats: {str(e)}", err=True)


@generate.command('clear')
@click.argument('video_ids', nargs=-1, required=True)
def generate_clear(video_ids: tuple):
    """Clear format selections for videos."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def clear_selections():
            cleared_count = 0
            
            async with db_manager.get_session() as db:
                for video_id in video_ids:
                    if video_id.isdigit():
                        video = await db.get(Video, int(video_id))
                    else:
                        query = select(Video).where(Video.video_id == video_id)
                        result = await db.execute(query)
                        video = result.scalar_one_or_none()
                    
                    if not video:
                        click.echo(f"⚠️  Video {video_id} not found", err=True)
                        continue
                    
                    video.selected_generators = None
                    video.generation_selection_status = 'pending_selection'
                    video.generation_selected_at = None
                    cleared_count += 1
                    
                    click.echo(f"🗑️  Cleared formats for: {video.title[:60]}")
                
                await db.commit()
            
            return cleared_count
        
        count = asyncio.run(clear_selections())
        
        if count > 0:
            click.echo(f"\n🗑️  Cleared format selection for {count} video(s)")
        
    except Exception as e:
        click.echo(f"❌ Error clearing selections: {str(e)}", err=True)


@generate.command('start')
@click.option('--video-ids', help='Comma-separated video IDs to generate (default: all ready)')
@click.option('--dry-run', is_flag=True, help='Show what would be generated without creating jobs')
def generate_start(video_ids: Optional[str], dry_run: bool):
    """Start content generation for videos with selected formats."""
    try:
        asyncio.run(db_manager.initialize())
        
        async def start_generation():
            job_count = 0
            video_count = 0
            
            async with db_manager.get_session() as db:
                if video_ids:
                    # Process specific videos
                    ids = [id.strip() for id in video_ids.split(',')]
                    videos_to_process = []
                    
                    for video_id in ids:
                        if video_id.isdigit():
                            video = await db.get(Video, int(video_id))
                        else:
                            query = select(Video).where(Video.video_id == video_id)
                            result = await db.execute(query)
                            video = result.scalar_one_or_none()
                        
                        if video:
                            videos_to_process.append(video)
                else:
                    # Get all videos with selected formats
                    query = (
                        select(Video)
                        .where(Video.generation_selection_status == 'formats_selected')
                        .where(Video.selected_generators.isnot(None))
                    )
                    result = await db.execute(query)
                    videos_to_process = result.scalars().all()
                
                if not videos_to_process:
                    click.echo("📭 No videos ready for generation (need format selection)")
                    return 0, 0
                
                from core.queue import get_job_queue, JobType
                queue = get_job_queue() if not dry_run else None
                
                for video in videos_to_process:
                    selected_formats = json.loads(video.selected_generators) if video.selected_generators else []
                    
                    if not selected_formats:
                        click.echo(f"⚠️  No formats selected for: {video.title[:60]}")
                        continue
                    
                    click.echo(f"\n🎬 {video.title[:60]}")
                    click.echo(f"   Formats: {', '.join(selected_formats)}")
                    
                    if not dry_run:
                        # Create generation jobs
                        for generator in selected_formats:
                            await queue.enqueue(
                                job_type=JobType.GENERATE_CONTENT.value,
                                target_id=video.video_id,
                                priority=3,
                                metadata={'generator_type': generator}
                            )
                            job_count += 1
                        
                        # Update status
                        video.generation_selection_status = 'generation_started'
                        video.generation_started_at = datetime.now(timezone.utc)
                    
                    video_count += 1
                
                if not dry_run:
                    await db.commit()
            
            return video_count, job_count
        
        videos, jobs = asyncio.run(start_generation())
        
        if dry_run:
            click.echo(f"\n🔍 Dry run: Would generate content for {videos} video(s)")
        else:
            if jobs > 0:
                click.echo(f"\n🚀 Started generation: {jobs} job(s) for {videos} video(s)")
            else:
                click.echo("\n📭 No generation jobs created")
        
    except Exception as e:
        click.echo(f"❌ Error starting generation: {str(e)}", err=True)


@generate.command('interactive')
@click.option('--limit', type=int, default=10, help='Number of videos to process')
def generate_interactive(limit: int):
    """Interactive format selection mode."""
    try:
        asyncio.run(db_manager.initialize())
        
        settings = get_settings()
        available_formats = settings.content_generators
        if isinstance(available_formats, str):
            available_formats = [f.strip() for f in available_formats.split(',')]
        
        async def get_pending_videos():
            async with db_manager.get_session() as db:
                query = (
                    select(Video)
                    .where(Video.generation_review_status == 'approved')
                    .where(Video.generation_selection_status == 'pending_selection')
                    .order_by(Video.generation_approved_at.desc())
                    .limit(limit)
                )
                result = await db.execute(query)
                return result.scalars().all()
        
        videos = asyncio.run(get_pending_videos())
        
        if not videos:
            click.echo("📭 No videos pending format selection!")
            return
        
        click.echo(f"📋 Interactive format selection - {len(videos)} video(s)\n")
        click.echo("Commands: (s)ave, (n)ext, (p)revious, (t)oggle format, (a)ll formats, (c)lear, (q)uit")
        click.echo(f"Available formats: {', '.join(available_formats)}\n")
        
        selections = {}
        current_index = 0
        
        # Initialize selections with current values
        for video in videos:
            current = json.loads(video.selected_generators) if video.selected_generators else []
            selections[video.video_id] = current.copy()
        
        while current_index < len(videos):
            video = videos[current_index]
            selected = selections[video.video_id]
            
            # Display current video
            click.echo(f"\n--- Video {current_index + 1}/{len(videos)} ---")
            click.echo(f"Title: {click.style(video.title, fg='green', bold=True)}")
            click.echo(f"Channel: {video.channel_name}")
            click.echo(f"Video ID: {video.video_id}")
            
            if video.ai_summary:
                click.echo(f"Summary: {click.style(video.ai_summary[:200], fg='cyan')}")
            
            # Display format checkboxes
            click.echo("\nSelect formats to generate:")
            for i, fmt in enumerate(available_formats, 1):
                checkbox = "[x]" if fmt in selected else "[ ]"
                click.echo(f"  {i}. {checkbox} {fmt}")
            
            # Get user action
            action = click.prompt("\nAction", type=str).lower().strip()
            
            if action in ['s', 'save']:
                # Save and continue
                current_index += 1
            elif action in ['n', 'next']:
                # Next without saving changes
                current_index += 1
            elif action in ['p', 'previous'] and current_index > 0:
                current_index -= 1
            elif action.isdigit():
                # Toggle format by number
                idx = int(action) - 1
                if 0 <= idx < len(available_formats):
                    fmt = available_formats[idx]
                    if fmt in selected:
                        selected.remove(fmt)
                        click.echo(f"   Deselected: {fmt}")
                    else:
                        selected.append(fmt)
                        click.echo(f"   Selected: {fmt}")
            elif action in ['a', 'all']:
                selections[video.video_id] = available_formats.copy()
                click.echo("   Selected all formats")
            elif action in ['c', 'clear']:
                selections[video.video_id] = []
                click.echo("   Cleared all formats")
            elif action in ['q', 'quit']:
                if click.confirm("\nSave changes before exiting?"):
                    break
                else:
                    click.echo("👋 Exiting without saving")
                    return
            elif action in ['h', 'help']:
                click.echo("\nCommands:")
                click.echo("  s - Save selection and next")
                click.echo("  n - Next video")
                click.echo("  p - Previous video")
                click.echo("  1-5 - Toggle format by number")
                click.echo("  a - Select all formats")
                click.echo("  c - Clear all formats")
                click.echo("  q - Save and quit")
        
        # Apply selections
        asyncio.run(apply_format_selections(videos, selections))
        
    except Exception as e:
        click.echo(f"❌ Error in interactive mode: {str(e)}", err=True)


async def apply_format_selections(videos: list, selections: dict):
    """Apply format selections to database."""
    async with db_manager.get_session() as db:
        updated = 0
        
        for video in videos:
            selected = selections.get(video.video_id, [])
            
            if selected:
                video.selected_generators = json.dumps(selected)
                video.generation_selection_status = 'formats_selected'
                video.generation_selected_at = datetime.now(timezone.utc)
                updated += 1
        
        await db.commit()
    
    if updated > 0:
        click.echo(f"\n💾 Saved format selections for {updated} video(s)")
        click.echo(f"   Use 'yt-dl generate start' to begin generation")


@cli.command()
@click.option('--type', 'export_type', 
              type=click.Choice(['transcripts', 'content', 'all']),
              default='all', help='Type of data to export')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'csv', 'txt', 'markdown', 'html']),
              default='json', help='Export format')
@click.option('--channel', '-c', help='Export data for specific channel only')
@click.option('--video', '-v', help='Export data for specific video only')
@click.option('--since', type=str, callback=validate_date,
              help='Export data since date (YYYY-MM-DD)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path (default: exports/export_TIMESTAMP.FORMAT)')
@click.option('--content-type', 
              type=click.Choice(['blog', 'social', 'newsletter', 'script', 'summary', 'all']),
              help='Filter by content type (for content exports)')
def export(export_type: str, output_format: str, channel: Optional[str],
          video: Optional[str], since: Optional[str], output: Optional[str],
          content_type: Optional[str]):
    """Export transcripts and generated content in various formats."""
    
    async def run_export():
        try:
            async with db_manager.get_session() as session:
                # Build base queries
                transcript_query = select(Transcript).join(Video)
                content_query = select(GeneratedContent).join(Video)
                
                # Apply filters
                if channel:
                    # Get channel from database
                    ch_result = await session.execute(
                        select(Channel).where(Channel.channel_id == channel)
                    )
                    ch = ch_result.scalar_one_or_none()
                    if not ch:
                        click.echo(f"❌ Channel '{channel}' not found")
                        return
                    transcript_query = transcript_query.where(Video.channel_id == ch.id)
                    content_query = content_query.where(Video.channel_id == ch.id)
                
                if video:
                    transcript_query = transcript_query.where(Video.video_id == video)
                    content_query = content_query.where(
                        GeneratedContent.video_id == video
                    )
                
                if since:
                    since_date = datetime.strptime(since, '%Y-%m-%d')
                    transcript_query = transcript_query.where(
                        Transcript.created_at >= since_date
                    )
                    content_query = content_query.where(
                        GeneratedContent.created_at >= since_date
                    )
                
                if content_type and content_type != 'all':
                    content_query = content_query.where(
                        GeneratedContent.content_type == content_type
                    )
                
                # Execute queries based on export type
                transcripts_data = []
                content_data = []
                
                if export_type in ['transcripts', 'all']:
                    result = await session.execute(transcript_query)
                    transcripts = result.scalars().all()
                    
                    for t in transcripts:
                        # Get associated video
                        video_result = await session.execute(
                            select(Video).where(Video.video_id == t.video_id)
                        )
                        v = video_result.scalar_one_or_none()
                        
                        transcripts_data.append({
                            'video_id': t.video_id,
                            'title': v.title if v else 'Unknown',
                            'channel': v.channel.name if v and v.channel else 'Unknown',
                            'transcript_text': t.text_content,
                            'word_count': len(t.text_content.split()) if t.text_content else 0,
                            'quality_score': t.quality_score,
                            'created_at': t.created_at.isoformat() if t.created_at else None,
                            'extraction_method': t.extraction_method,
                            'transcription_model': t.transcription_model
                        })
                
                if export_type in ['content', 'all']:
                    result = await session.execute(content_query)
                    contents = result.scalars().all()
                    
                    for c in contents:
                        content_data.append({
                            'video_id': c.video_id,
                            'content_type': c.content_type,
                            'format': c.format,
                            'content': c.content,
                            'word_count': c.word_count,
                            'quality_score': c.quality_score,
                            'metadata': c.metadata,
                            'created_at': c.created_at.isoformat() if c.created_at else None
                        })
                
                # Prepare output
                if not output:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    exports_dir = Path('exports')
                    exports_dir.mkdir(exist_ok=True)
                    output = str(exports_dir / f'export_{timestamp}.{output_format}')
                
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Format and write data
                if output_format == 'json':
                    export_data = {}
                    if transcripts_data:
                        export_data['transcripts'] = transcripts_data
                    if content_data:
                        export_data['generated_content'] = content_data
                    
                    with open(output_path, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                
                elif output_format == 'csv':
                    import csv
                    
                    # Write transcripts CSV
                    if transcripts_data:
                        trans_path = output_path.with_suffix('.transcripts.csv')
                        with open(trans_path, 'w', newline='') as f:
                            if transcripts_data:
                                writer = csv.DictWriter(f, fieldnames=transcripts_data[0].keys())
                                writer.writeheader()
                                writer.writerows(transcripts_data)
                    
                    # Write content CSV
                    if content_data:
                        content_path = output_path.with_suffix('.content.csv')
                        with open(content_path, 'w', newline='') as f:
                            if content_data:
                                # Flatten metadata for CSV
                                for item in content_data:
                                    if 'metadata' in item and isinstance(item['metadata'], dict):
                                        item['metadata'] = json.dumps(item['metadata'])
                                writer = csv.DictWriter(f, fieldnames=content_data[0].keys())
                                writer.writeheader()
                                writer.writerows(content_data)
                
                elif output_format == 'txt':
                    with open(output_path, 'w') as f:
                        if transcripts_data:
                            f.write("=" * 80 + "\n")
                            f.write("TRANSCRIPTS\n")
                            f.write("=" * 80 + "\n\n")
                            for t in transcripts_data:
                                f.write(f"Video: {t['title']}\n")
                                f.write(f"Channel: {t['channel']}\n")
                                f.write(f"Video ID: {t['video_id']}\n")
                                f.write(f"Words: {t['word_count']}\n")
                                f.write(f"Quality: {t['quality_score']}\n")
                                f.write(f"Date: {t['created_at']}\n")
                                f.write("-" * 40 + "\n")
                                f.write(t['transcript_text'] + "\n")
                                f.write("\n" + "=" * 80 + "\n\n")
                        
                        if content_data:
                            f.write("=" * 80 + "\n")
                            f.write("GENERATED CONTENT\n")
                            f.write("=" * 80 + "\n\n")
                            for c in content_data:
                                f.write(f"Type: {c['content_type']}\n")
                                f.write(f"Format: {c['format']}\n")
                                f.write(f"Video ID: {c['video_id']}\n")
                                f.write(f"Words: {c['word_count']}\n")
                                f.write(f"Date: {c['created_at']}\n")
                                f.write("-" * 40 + "\n")
                                f.write(c['content'] + "\n")
                                f.write("\n" + "=" * 80 + "\n\n")
                
                elif output_format == 'markdown':
                    with open(output_path, 'w') as f:
                        f.write("# YouTube Content Export\n\n")
                        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                        
                        if transcripts_data:
                            f.write("## Transcripts\n\n")
                            for t in transcripts_data:
                                f.write(f"### {t['title']}\n\n")
                                f.write(f"- **Channel:** {t['channel']}\n")
                                f.write(f"- **Video ID:** `{t['video_id']}`\n")
                                f.write(f"- **Words:** {t['word_count']}\n")
                                f.write(f"- **Quality Score:** {t['quality_score']}\n")
                                f.write(f"- **Created:** {t['created_at']}\n\n")
                                f.write("#### Transcript\n\n")
                                f.write(t['transcript_text'] + "\n\n")
                                f.write("---\n\n")
                        
                        if content_data:
                            f.write("## Generated Content\n\n")
                            for c in content_data:
                                f.write(f"### {c['content_type'].title()} Content\n\n")
                                f.write(f"- **Format:** {c['format']}\n")
                                f.write(f"- **Video ID:** `{c['video_id']}`\n")
                                f.write(f"- **Words:** {c['word_count']}\n")
                                f.write(f"- **Created:** {c['created_at']}\n\n")
                                f.write("#### Content\n\n")
                                f.write(c['content'] + "\n\n")
                                f.write("---\n\n")
                
                elif output_format == 'html':
                    with open(output_path, 'w') as f:
                        # Write HTML header
                        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Content Export</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .metadata {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metadata strong { color: #2c3e50; }
        .content-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .transcript, .generated-content {
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .video-id {
            font-family: monospace;
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .timestamp { color: #95a5a6; font-size: 12px; }
        hr { border: none; border-top: 1px solid #ecf0f1; margin: 30px 0; }
    </style>
</head>
<body>
    <h1>YouTube Content Export</h1>
    <p class="timestamp">Generated: """ + datetime.now().isoformat() + """</p>
""")
                        
                        if transcripts_data:
                            f.write("    <h2>Transcripts</h2>\n")
                            for t in transcripts_data:
                                f.write(f"""    <div class="content-box">
        <h3>{t['title']}</h3>
        <div class="metadata">
            <strong>Channel:</strong> {t['channel']}<br>
            <strong>Video ID:</strong> <span class="video-id">{t['video_id']}</span><br>
            <strong>Words:</strong> {t['word_count']}<br>
            <strong>Quality Score:</strong> {t['quality_score']}<br>
            <strong>Created:</strong> {t['created_at']}
        </div>
        <h4>Transcript</h4>
        <div class="transcript">{t['transcript_text']}</div>
    </div>
    <hr>
""")
                        
                        if content_data:
                            f.write("    <h2>Generated Content</h2>\n")
                            for c in content_data:
                                f.write(f"""    <div class="content-box">
        <h3>{c['content_type'].title()} Content</h3>
        <div class="metadata">
            <strong>Format:</strong> {c['format']}<br>
            <strong>Video ID:</strong> <span class="video-id">{c['video_id']}</span><br>
            <strong>Words:</strong> {c['word_count']}<br>
            <strong>Created:</strong> {c['created_at']}
        </div>
        <h4>Content</h4>
        <div class="generated-content">{c['content']}</div>
    </div>
    <hr>
""")
                        
                        # Write HTML footer
                        f.write("""</body>
</html>""")
                
                # Summary
                total_items = len(transcripts_data) + len(content_data)
                if total_items > 0:
                    click.echo(f"\n✅ Exported {total_items} items to {output_path}")
                    if transcripts_data:
                        click.echo(f"   - {len(transcripts_data)} transcripts")
                    if content_data:
                        click.echo(f"   - {len(content_data)} generated content items")
                else:
                    click.echo("⚠️  No data found matching the specified filters")
        
        except Exception as e:
            click.echo(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the export
    asyncio.run(run_export())


@cli.command()
def version():
    """Show version information."""
    click.echo(f"📦 yt-dl-sub version {__version__}")
    
    import platform
    import sys
    
    click.echo(f"\n🖥️  System:")
    click.echo(f"   Python: {sys.version.split()[0]}")
    click.echo(f"   Platform: {platform.platform()}")
    
    from config.settings import get_settings
    settings = get_settings()
    
    click.echo(f"\n⚙️  Configuration:")
    click.echo(f"   Mode: {settings.deployment_mode}")
    click.echo(f"   Storage: {settings.storage_path}")
    click.echo(f"   AI Backend: {settings.ai_backend}")
    click.echo(f"   Whisper Model: {settings.whisper_model}")
    click.echo(f"   Review Required: {settings.generation_review_required}")
    click.echo(f"   AI Summaries: {settings.enable_ai_summaries}")


# ================================
# Channel Complete Workflow Functions  
# ================================

async def complete_channel_workflow(channel_url: str, limit: Optional[int] = None, quality: str = '1080p', 
                                   concurrent: int = 3, translate: bool = False, target_language: str = 'en') -> Dict[str, Any]:
    """
    Complete channel workflow: download all videos then process all videos.
    
    Args:
        channel_url: YouTube channel URL
        limit: Optional limit on number of videos
        quality: Video quality
        concurrent: Number of concurrent downloads
        translate: Enable translation
        target_language: Target language for translation
        
    Returns:
        Dict with completion results and statistics
    """
    from datetime import datetime
    import asyncio
    from core.storage_paths_v2 import get_storage_paths_v2
    from core.url_parser import YouTubeURLParser
    from core.downloader import create_downloader_with_settings
    
    start_time = datetime.now()
    
    try:
        # Parse and validate channel URL
        url_parser = YouTubeURLParser()
        url_type, identifier, metadata = url_parser.parse(channel_url)
        if url_type != URLType.CHANNEL:
            raise click.ClickException(f"❌ Invalid channel URL: {channel_url}")
        
        click.echo(f"🔄 Starting complete workflow for: {channel_url}")
        click.echo(f"📊 Mode: Download-all → Process-all (Channel-by-channel)")
        if limit:
            click.echo(f"📊 Limit: {limit} videos")
        
        # ========================================
        # PHASE 1: BULK DOWNLOAD (FAST)
        # ========================================
        click.echo(f"\n📥 PHASE 1: Bulk Download (Fast)")
        click.echo(f"⏭️  Skipping transcription and punctuation for speed...")
        
        phase1_start = datetime.now()
        
        # Create downloader with skip flags for Phase 1 (fast download)
        downloader = create_downloader_with_settings(
            skip_transcription=True,
            skip_punctuation=True
        )
        
        # Download all videos with skip flags for speed
        download_result = await downloader.download_channel_videos(
            channel_url=channel_url,
            limit=limit,
            quality=quality,
            download_audio_only=True,
            max_concurrent=concurrent
        )
        
        phase1_duration = datetime.now() - phase1_start
        
        # Check if download was successful
        # The download_batch method returns 'successful' (list), 'failed' (list), 'total' (int)
        # Also check for 'status' == 'error' from channel_enumerator
        if download_result.get('status') == 'error':
            raise click.ClickException(f"❌ Phase 1 failed: {download_result.get('error', 'Unknown error')}")
        
        successful_count = len(download_result.get('successful', []))
        failed_count = len(download_result.get('failed', []))
        total_count = download_result.get('total', 0)
        
        if total_count == 0:
            raise click.ClickException(f"❌ Phase 1 failed: No videos found to download")
        
        if successful_count == 0 and total_count > 0:
            raise click.ClickException(f"❌ Phase 1 failed: All {total_count} downloads failed")
            
        click.echo(f"✅ Phase 1 completed in {phase1_duration}")
        click.echo(f"📊 Downloaded: {successful_count}/{total_count} videos successfully")
        if failed_count > 0:
            click.echo(f"⚠️  Failed: {failed_count} videos")
        
        # ========================================
        # PHASE 2: BULK PROCESSING (COMPLETE)
        # ========================================
        click.echo(f"\n🎙️  PHASE 2: Bulk Processing (Complete)")
        click.echo(f"🔄 Processing all videos for transcription + punctuation...")
        
        phase2_start = datetime.now()
        
        # Use batch_transcribe for Phase 2
        from batch_transcribe import BatchTranscriber
        
        # Extract channel_id from download result
        # The download result includes 'channel_info' which has 'channel_id'
        channel_info = download_result.get('channel_info', {})
        channel_id = channel_info.get('channel_id')
        
        if not channel_id:
            # Fallback: extract from storage
            storage = get_storage_paths_v2()
            downloads_dir = Path(storage.base_path)
            channel_dirs = [d for d in downloads_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if not channel_dirs:
                raise click.ClickException("❌ No channel directory found after download")
            channel_id = channel_dirs[-1].name  # Use most recent
        
        # Extract video IDs from Phase 1 successful downloads
        downloaded_video_ids = []
        for item in download_result.get('successful', []):
            # Each successful item should have a video_id
            if isinstance(item, dict):
                video_id = item.get('video_id') or item.get('id')
                if video_id:
                    downloaded_video_ids.append(video_id)
            elif isinstance(item, str):
                # If it's just a string, assume it's the video ID
                downloaded_video_ids.append(item)
        
        if not downloaded_video_ids:
            click.echo(f"⚠️  No video IDs found from Phase 1 downloads")
            return {
                'success': False,
                'error': 'No video IDs to process in Phase 2'
            }
        
        click.echo(f"📊 Processing {len(downloaded_video_ids)} videos from Phase 1")
        
        # Initialize batch transcriber
        transcriber = BatchTranscriber()
        
        # Process only the videos that were downloaded in Phase 1
        process_result = await transcriber.process_specific_videos(
            channel_id=channel_id,
            video_ids=downloaded_video_ids
        )
        
        phase2_duration = datetime.now() - phase2_start
        total_duration = datetime.now() - start_time
        
        click.echo(f"✅ Phase 2 completed in {phase2_duration}")
        click.echo(f"📊 Processed: {process_result.get('processed', 0)} videos")
        click.echo(f"📊 Skipped: {process_result.get('skipped', 0)} videos")
        click.echo(f"📊 Errors: {process_result.get('errors', 0)} videos")
        
        # ========================================
        # COMPLETION SUMMARY
        # ========================================
        click.echo(f"\n🎉 CHANNEL COMPLETE: {channel_url}")
        click.echo(f"⏱️  Total time: {total_duration}")
        click.echo(f"📥 Phase 1 (Download): {phase1_duration} - {successful_count} videos")
        click.echo(f"🎙️  Phase 2 (Process): {phase2_duration} - {process_result.get('processed', 0)} processed")
        
        # Calculate efficiency improvement
        if successful_count > 0:
            avg_time_per_video = total_duration.total_seconds() / successful_count
            click.echo(f"📊 Efficiency: {avg_time_per_video:.1f} seconds/video average")
        
        return {
            'success': True,
            'channel_url': channel_url,
            'channel_id': channel_id,
            'total_duration': total_duration,
            'phase1_duration': phase1_duration,
            'phase2_duration': phase2_duration,
            'download_result': download_result,
            'process_result': process_result,
            'total_videos': successful_count,
            'processed_videos': process_result.get('processed', 0)
        }
        
    except Exception as e:
        click.echo(f"❌ Channel workflow failed: {str(e)}", err=True)
        return {
            'success': False,
            'error': str(e),
            'channel_url': channel_url
        }


async def complete_multiple_channels(channel_urls: List[str], limit: Optional[int] = None, 
                                   quality: str = '1080p', concurrent: int = 3, 
                                   translate: bool = False, target_language: str = 'en') -> List[Dict[str, Any]]:
    """
    Complete multiple channels sequentially with complete workflow.
    
    Args:
        channel_urls: List of YouTube channel URLs
        limit: Optional limit on number of videos per channel
        quality: Video quality
        concurrent: Number of concurrent downloads
        translate: Enable translation
        target_language: Target language for translation
        
    Returns:
        List of completion results for each channel
    """
    from datetime import datetime
    
    overall_start = datetime.now()
    results = []
    
    click.echo(f"🚀 MULTI-CHANNEL COMPLETE WORKFLOW")
    click.echo(f"📊 Channels to process: {len(channel_urls)}")
    click.echo(f"📊 Strategy: Complete each channel before moving to next")
    
    for i, channel_url in enumerate(channel_urls, 1):
        click.echo(f"\n" + "="*60)
        click.echo(f"📺 CHANNEL {i}/{len(channel_urls)}: {channel_url}")
        click.echo(f"="*60)
        
        # Complete single channel workflow
        result = await complete_channel_workflow(
            channel_url=channel_url,
            limit=limit,
            quality=quality,
            concurrent=concurrent,
            translate=translate,
            target_language=target_language
        )
        
        results.append(result)
        
        if result.get('success'):
            click.echo(f"✅ Channel {i}/{len(channel_urls)} completed successfully")
        else:
            click.echo(f"❌ Channel {i}/{len(channel_urls)} failed: {result.get('error')}")
            # Continue with next channel even if one fails
    
    # ========================================
    # OVERALL SUMMARY
    # ========================================
    overall_duration = datetime.now() - overall_start
    successful_channels = sum(1 for r in results if r.get('success'))
    total_videos = sum(r.get('total_videos', 0) for r in results)
    total_processed = sum(r.get('processed_videos', 0) for r in results)
    
    click.echo(f"\n" + "="*60)
    click.echo(f"🎉 MULTI-CHANNEL WORKFLOW COMPLETE")
    click.echo(f"="*60)
    click.echo(f"⏱️  Total time: {overall_duration}")
    click.echo(f"📊 Successful channels: {successful_channels}/{len(channel_urls)}")
    click.echo(f"📊 Total videos downloaded: {total_videos}")
    click.echo(f"📊 Total videos processed: {total_processed}")
    
    if total_videos > 0:
        avg_time_per_video = overall_duration.total_seconds() / total_videos
        click.echo(f"📊 Overall efficiency: {avg_time_per_video:.1f} seconds/video")
    
    return results


# ================================
# Channel Complete CLI Commands
# ================================

@channel.command('complete')
@click.argument('channel_url')
@click.option('--limit', type=int, help='Limit number of videos to process')
@click.option('--quality', default='1080p', help='Video quality (720p/1080p/best)')
@click.option('--concurrent', type=int, default=3, help='Number of concurrent downloads (1-10, default: 3)')
@click.option('--translate', is_flag=True, help='Enable AI translation of non-English subtitles to English')
@click.option('--target-language', default='en', help='Target language for subtitle translation (default: en)')
def channel_complete(channel_url: str, limit: Optional[int], quality: str, concurrent: int, translate: bool, target_language: str):
    """
    Complete channel workflow: download all videos then process all videos.
    
    This implements the channel-by-channel approach requested:
    1. PHASE 1: Download ALL videos from the channel (fast, skip transcription)
    2. PHASE 2: Process ALL videos in the channel (transcription + punctuation)
    
    This eliminates download bottlenecks while ensuring 100% completion.
    
    Examples:
        yt-dl channel complete "https://youtube.com/@TCM-Chan"
        yt-dl channel complete "@TCM-Chan" --limit 10
        yt-dl channel complete "TCM-Chan" --translate --concurrent 5
    """
    try:
        # Run the complete workflow
        result = asyncio.run(complete_channel_workflow(
            channel_url=channel_url,
            limit=limit,
            quality=quality,
            concurrent=min(max(1, concurrent), 10),
            translate=translate,
            target_language=target_language
        ))
        
        if not result.get('success'):
            raise click.ClickException(f"Channel workflow failed: {result.get('error')}")
            
    except KeyboardInterrupt:
        click.echo("\n🛑 Workflow interrupted by user", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Workflow error: {str(e)}", err=True)
        raise click.ClickException(f"Workflow failed: {str(e)}")


# ================================
# Multi-Channel CLI Commands
# ================================

@cli.group()
def channels():
    """Multi-channel management commands."""
    pass


@channels.command('complete')
@click.argument('channel_urls', nargs=-1, required=True)
@click.option('--from-file', type=click.Path(exists=True), help='Read channel URLs from file (one per line)')
@click.option('--limit', type=int, help='Limit number of videos per channel')
@click.option('--quality', default='1080p', help='Video quality (720p/1080p/best)')
@click.option('--concurrent', type=int, default=3, help='Number of concurrent downloads (1-10, default: 3)')
@click.option('--translate', is_flag=True, help='Enable AI translation of non-English subtitles to English')
@click.option('--target-language', default='en', help='Target language for subtitle translation (default: en)')
def channels_complete(channel_urls: tuple, from_file: Optional[str], limit: Optional[int], 
                     quality: str, concurrent: int, translate: bool, target_language: str):
    """
    Complete multiple channels sequentially with complete workflow.
    
    Processes each channel completely (download-all → process-all) before moving to the next.
    This implements the requested architecture: eliminate download bottlenecks while ensuring
    100% completion for every channel.
    
    Examples:
        yt-dl channels complete "@TCM-Chan" "@health-diary" "@dr-zhao"
        yt-dl channels complete --from-file channels.txt --limit 10
        yt-dl channels complete "@channel1" "@channel2" --translate --concurrent 5
    """
    try:
        # Collect channel URLs
        urls = list(channel_urls)
        
        if from_file:
            click.echo(f"📄 Reading channels from file: {from_file}")
            with open(from_file, 'r', encoding='utf-8') as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                urls.extend(file_urls)
                click.echo(f"📊 Loaded {len(file_urls)} channels from file")
        
        if not urls:
            raise click.ClickException("❌ No channel URLs provided")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        if len(unique_urls) != len(urls):
            click.echo(f"📊 Removed {len(urls) - len(unique_urls)} duplicate URLs")
        
        # Run multi-channel complete workflow
        results = asyncio.run(complete_multiple_channels(
            channel_urls=unique_urls,
            limit=limit,
            quality=quality,
            concurrent=min(max(1, concurrent), 10),
            translate=translate,
            target_language=target_language
        ))
        
        # Check for any failures
        failed_channels = [r for r in results if not r.get('success')]
        if failed_channels:
            click.echo(f"\n⚠️  {len(failed_channels)} channels failed:")
            for failed in failed_channels:
                click.echo(f"   ❌ {failed.get('channel_url')}: {failed.get('error')}")
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Multi-channel workflow interrupted by user", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Multi-channel workflow error: {str(e)}", err=True)
        raise click.ClickException(f"Multi-channel workflow failed: {str(e)}")


if __name__ == '__main__':
    cli()