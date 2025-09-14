"""
Export functionality for YouTube Content Intelligence platform.

This module provides comprehensive export capabilities for transcripts and generated content
in multiple formats (JSON, CSV, TXT, Markdown) with support for filtering, streaming,
and progress tracking for efficient handling of large datasets.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from io import StringIO

from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload, joinedload

from .database import (
    Channel, Video, Transcript, GeneratedContent, DatabaseManager, db_manager
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExportStats:
    """Statistics for export operations."""
    total_transcripts: int = 0
    total_content_items: int = 0
    total_channels: int = 0
    total_videos: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    file_size_bytes: int = 0
    export_duration_seconds: float = 0.0


@dataclass
class ExportProgress:
    """Progress tracking for export operations."""
    current_item: int = 0
    total_items: int = 0
    current_phase: str = "initializing"
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.current_item / self.total_items) * 100.0


class ExportService:
    """
    Comprehensive export service for transcripts and generated content.
    
    Supports multiple output formats with efficient streaming for large datasets,
    flexible filtering options, and progress tracking.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize export service with database manager."""
        from .database import db_manager as global_db_manager
        self.db_manager = db_manager or global_db_manager
        self.settings = get_settings()
    
    async def export_transcripts(
        self,
        format: str = "json",
        channel_id: Optional[str] = None,
        since: Optional[Union[datetime, date, str]] = None,
        until: Optional[Union[datetime, date, str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        include_content: bool = False,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Export transcripts in the specified format with filtering options.
        
        Args:
            format: Output format ('json', 'csv', 'txt', 'markdown')
            channel_id: Filter by specific channel ID
            since: Include videos published since this date
            until: Include videos published until this date
            output_path: Output file path (auto-generated if None)
            include_content: Include generated content in export
            progress_callback: Optional progress tracking callback
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with export results and statistics
        """
        start_time = datetime.now()
        
        # Validate format
        supported_formats = ["json", "csv", "txt", "markdown"]
        if format not in supported_formats:
            raise ValueError(f"Unsupported format: {format}. Must be one of: {supported_formats}")
        
        # Parse date filters
        since_date = self._parse_date(since) if since else None
        until_date = self._parse_date(until) if until else None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            channel_suffix = f"_{channel_id}" if channel_id else ""
            filename = f"transcripts_export{channel_suffix}_{timestamp}.{format}"
            output_path = self.settings.storage_path / "exports" / filename
        else:
            output_path = Path(output_path)
        
        # Ensure export directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracking
        progress = ExportProgress(current_phase="counting_records")
        if progress_callback:
            progress_callback(progress)
        
        # Get total count for progress tracking
        total_count = await self._get_transcript_count(channel_id, since_date, until_date)
        progress.total_items = total_count
        progress.current_phase = "exporting_data"
        
        if progress_callback:
            progress_callback(progress)
        
        # Export based on format
        export_methods = {
            "json": self._export_transcripts_json,
            "csv": self._export_transcripts_csv,
            "txt": self._export_transcripts_txt,
            "markdown": self._export_transcripts_markdown
        }
        
        try:
            await export_methods[format](
                output_path=output_path,
                channel_id=channel_id,
                since_date=since_date,
                until_date=until_date,
                include_content=include_content,
                progress=progress,
                progress_callback=progress_callback,
                batch_size=batch_size
            )
            
            # Get final statistics
            file_size = output_path.stat().st_size if output_path.exists() else 0
            export_duration = (datetime.now() - start_time).total_seconds()
            
            stats = ExportStats(
                total_transcripts=total_count,
                date_range_start=since_date,
                date_range_end=until_date,
                file_size_bytes=file_size,
                export_duration_seconds=export_duration
            )
            
            logger.info(f"Export completed: {output_path} ({file_size} bytes, {export_duration:.2f}s)")
            
            return {
                "status": "success",
                "output_path": str(output_path),
                "format": format,
                "stats": stats,
                "filters": {
                    "channel_id": channel_id,
                    "since": since_date.isoformat() if since_date else None,
                    "until": until_date.isoformat() if until_date else None,
                    "include_content": include_content
                }
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            raise
    
    async def get_export_stats(
        self,
        channel_id: Optional[str] = None,
        since: Optional[Union[datetime, date, str]] = None,
        until: Optional[Union[datetime, date, str]] = None
    ) -> ExportStats:
        """
        Get statistics for exportable items without performing export.
        
        Args:
            channel_id: Filter by specific channel ID
            since: Include videos published since this date
            until: Include videos published until this date
            
        Returns:
            ExportStats with counts and metadata
        """
        since_date = self._parse_date(since) if since else None
        until_date = self._parse_date(until) if until else None
        
        async with self.db_manager.get_session() as session:
            # Base query for statistics
            base_conditions = []
            
            if channel_id:
                base_conditions.append(Video.channel_id == channel_id)
            if since_date:
                base_conditions.append(Video.published_at >= since_date)
            if until_date:
                base_conditions.append(Video.published_at <= until_date)
            
            # Count transcripts
            transcript_query = (
                select(func.count(Transcript.id))
                .select_from(Transcript)
                .join(Video, Transcript.video_id == Video.video_id)
            )
            
            if base_conditions:
                transcript_query = transcript_query.where(and_(*base_conditions))
            
            transcript_result = await session.execute(transcript_query)
            total_transcripts = transcript_result.scalar() or 0
            
            # Count generated content
            content_query = (
                select(func.count(GeneratedContent.id))
                .select_from(GeneratedContent)
                .join(Video, GeneratedContent.video_id == Video.video_id)
            )
            
            if base_conditions:
                content_query = content_query.where(and_(*base_conditions))
            
            content_result = await session.execute(content_query)
            total_content = content_result.scalar() or 0
            
            # Count unique videos and channels
            video_query = select(func.count(Video.id.distinct()))
            if base_conditions:
                video_query = video_query.where(and_(*base_conditions))
            
            video_result = await session.execute(video_query)
            total_videos = video_result.scalar() or 0
            
            channel_query = select(func.count(Video.channel_id.distinct()))
            if base_conditions:
                channel_query = channel_query.where(and_(*base_conditions))
            
            channel_result = await session.execute(channel_query)
            total_channels = channel_result.scalar() or 0
            
            return ExportStats(
                total_transcripts=total_transcripts,
                total_content_items=total_content,
                total_videos=total_videos,
                total_channels=total_channels,
                date_range_start=since_date,
                date_range_end=until_date
            )
    
    async def _get_transcript_query(
        self,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        include_content: bool = False
    ):
        """Build base query for transcript export."""
        query = (
            select(Transcript, Video, Channel)
            .select_from(Transcript)
            .join(Video, Transcript.video_id == Video.video_id)
            .join(Channel, Video.channel_id == Channel.channel_id)
        )
        
        # Add content if requested
        if include_content:
            query = query.options(
                selectinload(Video.generated_content)
            )
        
        # Apply filters
        conditions = []
        if channel_id:
            conditions.append(Video.channel_id == channel_id)
        if since_date:
            conditions.append(Video.published_at >= since_date)
        if until_date:
            conditions.append(Video.published_at <= until_date)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        return query.order_by(Video.published_at.desc())
    
    async def _get_transcript_count(
        self,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None
    ) -> int:
        """Get count of transcripts matching filters."""
        async with self.db_manager.get_session() as session:
            query = (
                select(func.count(Transcript.id))
                .select_from(Transcript)
                .join(Video, Transcript.video_id == Video.video_id)
            )
            
            conditions = []
            if channel_id:
                conditions.append(Video.channel_id == channel_id)
            if since_date:
                conditions.append(Video.published_at >= since_date)
            if until_date:
                conditions.append(Video.published_at <= until_date)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await session.execute(query)
            return result.scalar() or 0
    
    async def _export_transcripts_json(
        self,
        output_path: Path,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        include_content: bool = False,
        progress: Optional[ExportProgress] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
        batch_size: int = 100
    ):
        """Export transcripts to JSON format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('{\n  "export_metadata": {\n')
            f.write(f'    "timestamp": "{datetime.now().isoformat()}",\n')
            f.write(f'    "format": "json",\n')
            f.write(f'    "include_content": {json.dumps(include_content)},\n')
            f.write(f'    "filters": {{\n')
            f.write(f'      "channel_id": {json.dumps(channel_id)},\n')
            f.write(f'      "since_date": {json.dumps(since_date.isoformat() if since_date else None)},\n')
            f.write(f'      "until_date": {json.dumps(until_date.isoformat() if until_date else None)}\n')
            f.write('    }\n  },\n')
            f.write('  "transcripts": [\n')
            
            first_item = True
            async for batch in self._get_transcript_batches(
                channel_id, since_date, until_date, include_content, batch_size
            ):
                for transcript, video, channel in batch:
                    if not first_item:
                        f.write(',\n')
                    first_item = False
                    
                    # Build transcript data
                    transcript_data = {
                        "transcript": {
                            "id": transcript.id,
                            "video_id": transcript.video_id,
                            "content_text": transcript.content_text,
                            "content_srt": transcript.content_srt,
                            "word_count": transcript.word_count,
                            "language": transcript.language,
                            "extraction_method": transcript.extraction_method,
                            "transcription_model": transcript.transcription_model,
                            "quality_score": transcript.quality_score,
                            "quality_details": transcript.quality_details,
                            "created_at": transcript.created_at.isoformat() if transcript.created_at else None
                        },
                        "video": {
                            "id": video.id,
                            "video_id": video.video_id,
                            "title": video.title,
                            "description": video.description,
                            "duration": video.duration,
                            "view_count": video.view_count,
                            "like_count": video.like_count,
                            "published_at": video.published_at.isoformat() if video.published_at else None,
                            "language": video.language,
                            "is_auto_generated": video.is_auto_generated
                        },
                        "channel": {
                            "id": channel.id,
                            "channel_id": channel.channel_id,
                            "channel_name": channel.channel_name,
                            "channel_url": channel.channel_url,
                            "subscriber_count": channel.subscriber_count,
                            "video_count": channel.video_count
                        }
                    }
                    
                    # Add generated content if requested
                    if include_content and hasattr(video, 'generated_content'):
                        transcript_data["generated_content"] = [
                            {
                                "id": content.id,
                                "content_type": content.content_type,
                                "content": content.content,
                                "content_metadata": content.content_metadata,
                                "quality_score": content.quality_score,
                                "generation_model": content.generation_model,
                                "created_at": content.created_at.isoformat() if content.created_at else None
                            }
                            for content in video.generated_content
                        ]
                    
                    json_str = json.dumps(transcript_data, indent=4, ensure_ascii=False)
                    # Indent the JSON content to match the array structure
                    indented_json = '\n'.join('    ' + line for line in json_str.split('\n'))
                    f.write(indented_json)
                    
                    # Update progress
                    if progress and progress_callback:
                        progress.current_item += 1
                        progress_callback(progress)
            
            f.write('\n  ]\n}')
    
    async def _export_transcripts_csv(
        self,
        output_path: Path,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        include_content: bool = False,
        progress: Optional[ExportProgress] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
        batch_size: int = 100
    ):
        """Export transcripts to CSV format."""
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # Define CSV headers
            headers = [
                'video_id', 'video_title', 'video_description', 'video_duration',
                'video_view_count', 'video_like_count', 'video_published_at',
                'channel_id', 'channel_name', 'channel_url', 'channel_subscriber_count',
                'transcript_id', 'transcript_text', 'transcript_word_count',
                'transcript_language', 'extraction_method', 'transcription_model',
                'quality_score', 'transcript_created_at'
            ]
            
            if include_content:
                headers.extend([
                    'content_types', 'content_count', 'latest_content_created_at'
                ])
            
            # Write headers
            writer = csv.writer(f)
            writer.writerow(headers)
            
            async for batch in self._get_transcript_batches(
                channel_id, since_date, until_date, include_content, batch_size
            ):
                for transcript, video, channel in batch:
                    # Build row data
                    row = [
                        video.video_id,
                        self._clean_csv_text(video.title),
                        self._clean_csv_text(video.description),
                        video.duration,
                        video.view_count,
                        video.like_count,
                        video.published_at.isoformat() if video.published_at else '',
                        channel.channel_id,
                        self._clean_csv_text(channel.channel_name),
                        channel.channel_url,
                        channel.subscriber_count,
                        transcript.id,
                        self._clean_csv_text(transcript.content_text),
                        transcript.word_count,
                        transcript.language,
                        transcript.extraction_method,
                        transcript.transcription_model,
                        transcript.quality_score,
                        transcript.created_at.isoformat() if transcript.created_at else ''
                    ]
                    
                    if include_content and hasattr(video, 'generated_content'):
                        content_items = video.generated_content
                        content_types = ', '.join([c.content_type for c in content_items])
                        content_count = len(content_items)
                        latest_created = max([c.created_at for c in content_items if c.created_at], default=None)
                        
                        row.extend([
                            content_types,
                            content_count,
                            latest_created.isoformat() if latest_created else ''
                        ])
                    
                    # Write row
                    writer.writerow(row)
                    
                    # Update progress
                    if progress and progress_callback:
                        progress.current_item += 1
                        progress_callback(progress)
    
    async def _export_transcripts_txt(
        self,
        output_path: Path,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        include_content: bool = False,
        progress: Optional[ExportProgress] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
        batch_size: int = 100
    ):
        """Export transcripts to plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"TRANSCRIPT EXPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Filter - Channel ID: {channel_id or 'All channels'}\n")
            f.write(f"Filter - Since: {since_date.strftime('%Y-%m-%d') if since_date else 'No limit'}\n")
            f.write(f"Filter - Until: {until_date.strftime('%Y-%m-%d') if until_date else 'No limit'}\n")
            f.write(f"Include Content: {'Yes' if include_content else 'No'}\n")
            f.write("=" * 80 + "\n\n")
            
            async for batch in self._get_transcript_batches(
                channel_id, since_date, until_date, include_content, batch_size
            ):
                for transcript, video, channel in batch:
                    # Video header
                    f.write(f"VIDEO: {video.title}\n")
                    f.write(f"Channel: {channel.channel_name}\n")
                    f.write(f"Video ID: {video.video_id}\n")
                    f.write(f"Published: {video.published_at.strftime('%Y-%m-%d') if video.published_at else 'Unknown'}\n")
                    f.write(f"Duration: {self._format_duration(video.duration)}\n")
                    f.write(f"Views: {video.view_count:,} | Likes: {video.like_count:,}\n")
                    
                    # Transcript info
                    f.write(f"Transcript Language: {transcript.language}\n")
                    f.write(f"Word Count: {transcript.word_count}\n")
                    f.write(f"Quality Score: {transcript.quality_score:.2f}\n")
                    f.write(f"Method: {transcript.extraction_method}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Transcript content
                    if transcript.content_text:
                        f.write(transcript.content_text)
                        f.write("\n")
                    
                    # Generated content
                    if include_content and hasattr(video, 'generated_content'):
                        for content in video.generated_content:
                            f.write(f"\n[{content.content_type.upper()}]\n")
                            f.write(content.content)
                            f.write("\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
                    
                    # Update progress
                    if progress and progress_callback:
                        progress.current_item += 1
                        progress_callback(progress)
    
    async def _export_transcripts_markdown(
        self,
        output_path: Path,
        channel_id: Optional[str] = None,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        include_content: bool = False,
        progress: Optional[ExportProgress] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
        batch_size: int = 100
    ):
        """Export transcripts to Markdown format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write frontmatter and header
            f.write("# Transcript Export\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Channel Filter:** {channel_id or 'All channels'}  \n")
            f.write(f"**Date Range:** {since_date.strftime('%Y-%m-%d') if since_date else 'No start'} to {until_date.strftime('%Y-%m-%d') if until_date else 'No end'}  \n")
            f.write(f"**Include Content:** {'Yes' if include_content else 'No'}  \n\n")
            f.write("---\n\n")
            
            async for batch in self._get_transcript_batches(
                channel_id, since_date, until_date, include_content, batch_size
            ):
                for transcript, video, channel in batch:
                    # Video title as main heading
                    clean_title = self._clean_markdown_text(video.title)
                    f.write(f"## {clean_title}\n\n")
                    
                    # Video metadata
                    f.write(f"- **Channel:** [{channel.channel_name}]({channel.channel_url})\n")
                    f.write(f"- **Video ID:** `{video.video_id}`\n")
                    f.write(f"- **Published:** {video.published_at.strftime('%Y-%m-%d') if video.published_at else 'Unknown'}\n")
                    f.write(f"- **Duration:** {self._format_duration(video.duration)}\n")
                    f.write(f"- **Views:** {video.view_count:,}\n")
                    f.write(f"- **Likes:** {video.like_count:,}\n")
                    
                    # Transcript metadata
                    f.write(f"- **Language:** {transcript.language}\n")
                    f.write(f"- **Word Count:** {transcript.word_count}\n")
                    f.write(f"- **Quality Score:** {transcript.quality_score:.2f}/1.0\n")
                    f.write(f"- **Extraction Method:** {transcript.extraction_method}\n\n")
                    
                    # Video description if available
                    if video.description:
                        clean_desc = self._clean_markdown_text(video.description)
                        f.write(f"**Description:**  \n{clean_desc}\n\n")
                    
                    # Transcript content
                    f.write("### Transcript\n\n")
                    if transcript.content_text:
                        clean_content = self._clean_markdown_text(transcript.content_text)
                        f.write(f"{clean_content}\n\n")
                    else:
                        f.write("*No transcript content available*\n\n")
                    
                    # Generated content
                    if include_content and hasattr(video, 'generated_content'):
                        f.write("### Generated Content\n\n")
                        for content in video.generated_content:
                            f.write(f"#### {content.content_type.title()}\n\n")
                            if content.quality_score:
                                f.write(f"*Quality Score: {content.quality_score:.2f}/1.0*\n\n")
                            clean_content_text = self._clean_markdown_text(content.content)
                            f.write(f"{clean_content_text}\n\n")
                    
                    f.write("---\n\n")
                    
                    # Update progress
                    if progress and progress_callback:
                        progress.current_item += 1
                        progress_callback(progress)
    
    async def _get_transcript_batches(
        self,
        channel_id: Optional[str],
        since_date: Optional[datetime],
        until_date: Optional[datetime],
        include_content: bool,
        batch_size: int
    ) -> AsyncGenerator[List, None]:
        """Generate batches of transcript data for streaming export."""
        async with self.db_manager.get_session() as session:
            query = await self._get_transcript_query(
                channel_id, since_date, until_date, include_content
            )
            
            offset = 0
            while True:
                batch_query = query.offset(offset).limit(batch_size)
                result = await session.execute(batch_query)
                batch = result.fetchall()
                
                if not batch:
                    break
                    
                yield batch
                offset += batch_size
    
    def _parse_date(self, date_value: Union[datetime, date, str]) -> datetime:
        """Parse various date formats into datetime object."""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        elif isinstance(date_value, str):
            # Try to parse common date formats
            for fmt in [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_value}")
        else:
            raise ValueError(f"Invalid date type: {type(date_value)}")
    
    def _clean_csv_text(self, text: Optional[str]) -> str:
        """Clean text for CSV export."""
        if not text:
            return ""
        # Replace newlines and tabs, limit length
        cleaned = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Collapse multiple spaces
        cleaned = ' '.join(cleaned.split())
        # Limit length for CSV readability
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
        return cleaned
    
    def _clean_markdown_text(self, text: Optional[str]) -> str:
        """Clean text for Markdown export."""
        if not text:
            return ""
        # Escape markdown special characters but preserve basic formatting
        cleaned = text.replace('\\', '\\\\').replace('`', '\\`')
        return cleaned
    
    def _format_duration(self, duration_seconds: Optional[int]) -> str:
        """Format duration in seconds to human-readable format."""
        if not duration_seconds:
            return "Unknown"
        
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


# Convenience functions
async def export_transcripts(
    format: str = "json",
    channel_id: Optional[str] = None,
    since: Optional[Union[datetime, date, str]] = None,
    until: Optional[Union[datetime, date, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    include_content: bool = False,
    progress_callback: Optional[Callable[[ExportProgress], None]] = None,
    db_manager: Optional[DatabaseManager] = None
) -> Dict[str, Any]:
    """Convenience function for exporting transcripts."""
    export_service = ExportService(db_manager)
    return await export_service.export_transcripts(
        format=format,
        channel_id=channel_id,
        since=since,
        until=until,
        output_path=output_path,
        include_content=include_content,
        progress_callback=progress_callback
    )


async def get_export_stats(
    channel_id: Optional[str] = None,
    since: Optional[Union[datetime, date, str]] = None,
    until: Optional[Union[datetime, date, str]] = None,
    db_manager: Optional[DatabaseManager] = None
) -> ExportStats:
    """Convenience function for getting export statistics."""
    export_service = ExportService(db_manager)
    return await export_service.get_export_stats(
        channel_id=channel_id,
        since=since,
        until=until
    )


# CLI usage example
async def main():
    """Example CLI usage of export functionality."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m core.export <format> [channel_id] [since_date]")
        print("Formats: json, csv, txt, markdown")
        return
    
    format_type = sys.argv[1]
    channel_id = sys.argv[2] if len(sys.argv) > 2 else None
    since_date = sys.argv[3] if len(sys.argv) > 3 else None
    
    def progress_callback(progress: ExportProgress):
        print(f"\r{progress.current_phase}: {progress.progress_percent:.1f}% "
              f"({progress.current_item}/{progress.total_items})", end="", flush=True)
    
    try:
        result = await export_transcripts(
            format=format_type,
            channel_id=channel_id,
            since=since_date,
            include_content=True,
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Export completed: {result['output_path']}")
        print(f"📊 Statistics:")
        stats = result['stats']
        print(f"  - Transcripts: {stats.total_transcripts}")
        print(f"  - File size: {stats.file_size_bytes:,} bytes")
        print(f"  - Duration: {stats.export_duration_seconds:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())