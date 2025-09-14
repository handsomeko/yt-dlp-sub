"""
SQLite-based job queue system for the YouTube Content Intelligence Platform.

This module provides a robust job queue implementation that supports priority-based
processing, retry logic with exponential backoff, and comprehensive error handling.
Integrates with the existing database schema and BaseWorker class.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DatabaseManager, Job, db_manager


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(Enum):
    """Supported job types."""
    DOWNLOAD_TRANSCRIPT = "download_transcript"
    PROCESS_CHANNEL = "process_channel"
    TRANSCRIBE_AUDIO = "transcribe_audio"
    SUMMARIZE = "summarize"
    GENERATE_CONTENT = "generate_content"
    SYNC_STORAGE = "sync_storage"
    QUALITY_CHECK = "quality_check"
    CLEANUP = "cleanup"


@dataclass
class QueueStats:
    """Queue statistics data structure."""
    total_jobs: int
    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    active_workers: int
    oldest_pending_age_minutes: Optional[float]
    average_processing_time_seconds: Optional[float]


class JobQueue:
    """
    SQLite-based job queue with priority processing and retry logic.
    
    Features:
    - Priority-based job ordering (higher priority first)
    - Exponential backoff retry mechanism
    - Worker assignment and tracking
    - Comprehensive error handling and logging
    - Queue statistics and monitoring
    - Automatic cleanup of old completed jobs
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        default_max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_backoff_multiplier: float = 2.0,
        job_timeout_minutes: int = 30,
        cleanup_after_days: int = 7
    ):
        """
        Initialize the job queue.
        
        Args:
            db_manager: Database manager instance (creates new if None)
            default_max_retries: Default maximum retries per job
            retry_base_delay: Base delay for retry backoff in seconds
            retry_backoff_multiplier: Multiplier for exponential backoff
            job_timeout_minutes: Timeout for jobs in processing state
            cleanup_after_days: Days after which completed jobs are cleaned up
        """
        from .database import db_manager as global_db_manager
        self.db_manager = db_manager or global_db_manager
        self.default_max_retries = default_max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.job_timeout_minutes = job_timeout_minutes
        self.cleanup_after_days = cleanup_after_days
        
        # Setup logging
        self.logger = logging.getLogger("queue.JobQueue")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def enqueue(
        self,
        job_type: str,
        target_id: str,
        priority: int = 5,
        max_retries: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new job to the queue.
        
        Args:
            job_type: Type of job (see JobType enum)
            target_id: Target identifier (video_id, channel_id, etc.)
            priority: Job priority (1=highest, 10=lowest, default=5)
            max_retries: Maximum retry attempts (uses default if None)
            metadata: Additional job metadata
            
        Returns:
            Job ID of the created job
            
        Raises:
            ValueError: If job parameters are invalid
            SQLAlchemyError: If database operation fails
        """
        if not job_type or not target_id:
            raise ValueError("job_type and target_id are required")
        
        if priority < 1 or priority > 10:
            raise ValueError("priority must be between 1 (highest) and 10 (lowest)")
        
        max_retries = max_retries or self.default_max_retries
        
        try:
            async with self.db_manager.get_session() as session:
                # Check for existing pending/processing job with same type and target
                existing_stmt = select(Job).where(
                    and_(
                        Job.job_type == job_type,
                        Job.target_id == target_id,
                        Job.status.in_([JobStatus.PENDING.value, JobStatus.PROCESSING.value])
                    )
                )
                existing_result = await session.execute(existing_stmt)
                existing_job = existing_result.scalar_one_or_none()
                
                if existing_job:
                    self.logger.info(
                        f"Job already exists: {job_type} for {target_id} "
                        f"(ID: {existing_job.id}, Status: {existing_job.status})"
                    )
                    return existing_job.id
                
                # Create new job
                job = Job(
                    job_type=job_type,
                    target_id=target_id,
                    status=JobStatus.PENDING.value,
                    priority=priority,
                    max_retries=max_retries,
                    retry_count=0,
                    created_at=datetime.utcnow()
                )
                
                session.add(job)
                await session.flush()  # Get job ID
                
                job_id = job.id
                self.logger.info(
                    f"Enqueued job {job_id}: {job_type} for {target_id} "
                    f"(priority: {priority}, max_retries: {max_retries})"
                )
                
                return job_id
                
        except IntegrityError as e:
            self.logger.error(f"Failed to enqueue job due to constraint violation: {e}")
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while enqueuing job: {e}")
            raise
    
    async def dequeue(self, worker_id: str) -> Optional[Tuple[int, str, str, Dict[str, Any]]]:
        """
        Get the next job from the queue based on priority.
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Tuple of (job_id, job_type, target_id, metadata) or None if no jobs
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not worker_id:
            raise ValueError("worker_id is required")
        
        try:
            async with self.db_manager.get_session() as session:
                # Find next pending job by priority and creation time
                stmt = (
                    select(Job)
                    .where(Job.status == JobStatus.PENDING.value)
                    .order_by(Job.priority.asc(), Job.created_at.asc())
                    .limit(1)
                    .with_for_update(skip_locked=True)  # Prevent race conditions
                )
                
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if not job:
                    return None
                
                # Update job to processing status
                job.status = JobStatus.PROCESSING.value
                job.worker_id = worker_id
                job.started_at = datetime.utcnow()
                
                await session.flush()
                
                self.logger.info(
                    f"Dequeued job {job.id}: {job.job_type} for {job.target_id} "
                    f"(worker: {worker_id}, priority: {job.priority})"
                )
                
                return (job.id, job.job_type, job.target_id, {})
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while dequeuing job: {e}")
            raise
    
    async def complete(
        self,
        job_id: int,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a job as completed.
        
        Args:
            job_id: Job ID to complete
            result_data: Optional result data from job execution
            
        Returns:
            True if job was successfully completed, False otherwise
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if not job:
                    self.logger.warning(f"Job {job_id} not found for completion")
                    return False
                
                if job.status != JobStatus.PROCESSING.value:
                    self.logger.warning(
                        f"Job {job_id} is not in processing state (current: {job.status})"
                    )
                    return False
                
                # Update job status
                job.status = JobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                job.error_message = None  # Clear any previous error
                
                await session.flush()
                
                self.logger.info(f"Completed job {job_id}: {job.job_type} for {job.target_id}")
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while completing job {job_id}: {e}")
            raise
    
    async def fail(
        self,
        job_id: int,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a job as failed with error information.
        
        Args:
            job_id: Job ID to mark as failed
            error_message: Error message describing the failure
            error_details: Optional additional error context
            
        Returns:
            True if job was successfully marked as failed, False otherwise
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not error_message:
            raise ValueError("error_message is required")
        
        try:
            async with self.db_manager.get_session() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if not job:
                    self.logger.warning(f"Job {job_id} not found for failure")
                    return False
                
                # Update job status
                job.status = JobStatus.FAILED.value
                job.error_message = error_message[:1000]  # Truncate long messages
                job.completed_at = datetime.utcnow()
                
                await session.flush()
                
                self.logger.error(
                    f"Failed job {job_id}: {job.job_type} for {job.target_id} - {error_message}"
                )
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while failing job {job_id}: {e}")
            raise
    
    async def retry(self, job_id: int, force: bool = False) -> bool:
        """
        Retry a failed job with exponential backoff.
        
        Args:
            job_id: Job ID to retry
            force: Force retry even if max retries exceeded
            
        Returns:
            True if job was successfully queued for retry, False otherwise
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if not job:
                    self.logger.warning(f"Job {job_id} not found for retry")
                    return False
                
                if job.status not in [JobStatus.FAILED.value, JobStatus.PROCESSING.value]:
                    self.logger.warning(
                        f"Job {job_id} cannot be retried (current status: {job.status})"
                    )
                    return False
                
                # Check retry limit
                if not force and job.retry_count >= job.max_retries:
                    self.logger.warning(
                        f"Job {job_id} has reached max retries ({job.max_retries})"
                    )
                    return False
                
                # Calculate exponential backoff delay
                delay_seconds = self.retry_base_delay * (
                    self.retry_backoff_multiplier ** job.retry_count
                )
                
                # Reset job for retry
                job.status = JobStatus.PENDING.value
                job.retry_count += 1
                job.worker_id = None
                job.started_at = None
                job.completed_at = None
                job.error_message = None
                
                await session.flush()
                
                self.logger.info(
                    f"Queued job {job_id} for retry {job.retry_count}/{job.max_retries} "
                    f"(backoff delay: {delay_seconds:.1f}s)"
                )
                
                # Apply exponential backoff delay
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while retrying job {job_id}: {e}")
            raise
    
    async def get_pending_jobs(
        self,
        limit: int = 100,
        job_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of pending jobs.
        
        Args:
            limit: Maximum number of jobs to return
            job_type: Filter by job type (optional)
            
        Returns:
            List of job dictionaries with basic information
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(Job)
                    .where(Job.status == JobStatus.PENDING.value)
                    .order_by(Job.priority.asc(), Job.created_at.asc())
                    .limit(limit)
                )
                
                if job_type:
                    stmt = stmt.where(Job.job_type == job_type)
                
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                
                return [
                    {
                        "id": job.id,
                        "job_type": job.job_type,
                        "target_id": job.target_id,
                        "priority": job.priority,
                        "retry_count": job.retry_count,
                        "max_retries": job.max_retries,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "age_minutes": (
                            (datetime.utcnow() - job.created_at).total_seconds() / 60
                            if job.created_at else None
                        )
                    }
                    for job in jobs
                ]
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while getting pending jobs: {e}")
            raise
    
    async def get_failed_jobs(
        self,
        limit: int = 100,
        include_retryable: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get list of failed jobs.
        
        Args:
            limit: Maximum number of jobs to return
            include_retryable: Include jobs that can still be retried
            
        Returns:
            List of failed job dictionaries with error information
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(Job)
                    .where(Job.status == JobStatus.FAILED.value)
                    .order_by(Job.completed_at.desc())
                    .limit(limit)
                )
                
                if not include_retryable:
                    stmt = stmt.where(Job.retry_count >= Job.max_retries)
                
                result = await session.execute(stmt)
                jobs = result.scalars().all()
                
                return [
                    {
                        "id": job.id,
                        "job_type": job.job_type,
                        "target_id": job.target_id,
                        "priority": job.priority,
                        "retry_count": job.retry_count,
                        "max_retries": job.max_retries,
                        "error_message": job.error_message,
                        "can_retry": job.retry_count < job.max_retries,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "failed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "worker_id": job.worker_id
                    }
                    for job in jobs
                ]
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while getting failed jobs: {e}")
            raise
    
    async def cleanup_old_jobs(self, dry_run: bool = False) -> int:
        """
        Remove old completed jobs from the queue.
        
        Args:
            dry_run: If True, return count without actually deleting
            
        Returns:
            Number of jobs cleaned up (or would be cleaned up in dry run)
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.cleanup_after_days)
            
            async with self.db_manager.get_session() as session:
                # Find old completed jobs
                stmt = select(Job).where(
                    and_(
                        Job.status == JobStatus.COMPLETED.value,
                        Job.completed_at < cutoff_date
                    )
                )
                
                result = await session.execute(stmt)
                jobs_to_cleanup = result.scalars().all()
                count = len(jobs_to_cleanup)
                
                if dry_run:
                    self.logger.info(f"Dry run: Would cleanup {count} old completed jobs")
                    return count
                
                if count > 0:
                    # Delete old jobs
                    for job in jobs_to_cleanup:
                        await session.delete(job)
                    
                    await session.flush()
                    self.logger.info(f"Cleaned up {count} old completed jobs")
                
                return count
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error during cleanup: {e}")
            raise
    
    async def get_queue_stats(self) -> QueueStats:
        """
        Get comprehensive queue statistics.
        
        Returns:
            QueueStats object with current queue metrics
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            async with self.db_manager.get_session() as session:
                # Get job counts by status
                count_stmt = (
                    select(
                        Job.status,
                        func.count(Job.id).label('count')
                    )
                    .group_by(Job.status)
                )
                count_result = await session.execute(count_stmt)
                status_counts = {row.status: row.count for row in count_result}
                
                # Get active worker count
                worker_stmt = (
                    select(func.count(func.distinct(Job.worker_id)))
                    .where(Job.status == JobStatus.PROCESSING.value)
                )
                worker_result = await session.execute(worker_stmt)
                active_workers = worker_result.scalar() or 0
                
                # Get oldest pending job age
                oldest_pending_stmt = (
                    select(func.min(Job.created_at))
                    .where(Job.status == JobStatus.PENDING.value)
                )
                oldest_result = await session.execute(oldest_pending_stmt)
                oldest_pending = oldest_result.scalar()
                oldest_age_minutes = None
                if oldest_pending:
                    oldest_age_minutes = (
                        datetime.utcnow() - oldest_pending
                    ).total_seconds() / 60
                
                # Get average processing time
                avg_time_stmt = (
                    select(func.avg(
                        func.julianday(Job.completed_at) - func.julianday(Job.started_at)
                    ))
                    .where(
                        and_(
                            Job.status == JobStatus.COMPLETED.value,
                            Job.started_at.isnot(None),
                            Job.completed_at.isnot(None)
                        )
                    )
                )
                avg_time_result = await session.execute(avg_time_stmt)
                avg_time_days = avg_time_result.scalar()
                avg_time_seconds = avg_time_days * 24 * 60 * 60 if avg_time_days else None
                
                return QueueStats(
                    total_jobs=sum(status_counts.values()),
                    pending_jobs=status_counts.get(JobStatus.PENDING.value, 0),
                    processing_jobs=status_counts.get(JobStatus.PROCESSING.value, 0),
                    completed_jobs=status_counts.get(JobStatus.COMPLETED.value, 0),
                    failed_jobs=status_counts.get(JobStatus.FAILED.value, 0),
                    active_workers=active_workers,
                    oldest_pending_age_minutes=oldest_age_minutes,
                    average_processing_time_seconds=avg_time_seconds
                )
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while getting queue stats: {e}")
            raise
    
    async def recover_stale_jobs(self) -> int:
        """
        Recover jobs that have been processing too long (likely due to worker crash).
        
        Returns:
            Number of jobs recovered
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            timeout_threshold = datetime.utcnow() - timedelta(minutes=self.job_timeout_minutes)
            
            async with self.db_manager.get_session() as session:
                # Find stale processing jobs
                stmt = select(Job).where(
                    and_(
                        Job.status == JobStatus.PROCESSING.value,
                        Job.started_at < timeout_threshold
                    )
                )
                
                result = await session.execute(stmt)
                stale_jobs = result.scalars().all()
                
                recovered_count = 0
                for job in stale_jobs:
                    # Reset job to pending for retry
                    job.status = JobStatus.PENDING.value
                    job.worker_id = None
                    job.started_at = None
                    job.error_message = f"Recovered from stale processing state (timeout: {self.job_timeout_minutes}min)"
                    recovered_count += 1
                    
                    self.logger.warning(
                        f"Recovered stale job {job.id}: {job.job_type} for {job.target_id}"
                    )
                
                if recovered_count > 0:
                    await session.flush()
                    self.logger.info(f"Recovered {recovered_count} stale jobs")
                
                return recovered_count
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error during stale job recovery: {e}")
            raise


class QueueWorker:
    """
    Base queue worker that processes jobs from the JobQueue.
    Integrates with BaseWorker for consistent execution patterns.
    """
    
    def __init__(
        self,
        queue: JobQueue,
        worker_id: Optional[str] = None,
        poll_interval: float = 5.0,
        max_consecutive_failures: int = 5
    ):
        """
        Initialize queue worker.
        
        Args:
            queue: JobQueue instance to process jobs from
            worker_id: Unique worker identifier (generated if None)
            poll_interval: Seconds to wait between queue polls
            max_consecutive_failures: Max failures before worker stops
        """
        self.queue = queue
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.max_consecutive_failures = max_consecutive_failures
        
        self.is_running = False
        self.consecutive_failures = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"queue.QueueWorker.{self.worker_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def start(self):
        """Start the worker processing loop."""
        self.is_running = True
        self.consecutive_failures = 0
        
        self.logger.info(f"Starting queue worker {self.worker_id}")
        
        try:
            while self.is_running:
                try:
                    # Try to process a job
                    job_processed = await self._process_next_job()
                    
                    if job_processed:
                        # Reset failure counter on successful processing
                        self.consecutive_failures = 0
                    else:
                        # No job available, wait and continue
                        await asyncio.sleep(self.poll_interval)
                    
                except Exception as e:
                    self.consecutive_failures += 1
                    self.logger.error(
                        f"Job processing error (failure {self.consecutive_failures}/"
                        f"{self.max_consecutive_failures}): {e}"
                    )
                    
                    # Stop if too many consecutive failures
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.critical(
                            f"Worker {self.worker_id} stopping due to too many consecutive failures"
                        )
                        break
                    
                    # Wait before retrying
                    await asyncio.sleep(self.poll_interval)
        
        finally:
            self.is_running = False
            self.logger.info(f"Queue worker {self.worker_id} stopped")
    
    def stop(self):
        """Stop the worker processing loop."""
        self.is_running = False
        self.logger.info(f"Stop requested for queue worker {self.worker_id}")
    
    async def _process_next_job(self) -> bool:
        """
        Process the next available job from the queue.
        
        Returns:
            True if a job was processed, False if no job available
        """
        # Get next job
        job_data = await self.queue.dequeue(self.worker_id)
        if not job_data:
            return False
        
        job_id, job_type, target_id, metadata = job_data
        
        try:
            # Process the job (override in subclasses)
            success = await self.process_job(job_type, target_id, metadata)
            
            if success:
                await self.queue.complete(job_id)
                self.logger.info(f"Successfully completed job {job_id}")
            else:
                await self.queue.fail(job_id, "Job processing returned False")
                self.logger.warning(f"Job {job_id} processing failed")
            
            return True
            
        except Exception as e:
            # Mark job as failed
            error_msg = f"Job processing exception: {str(e)}"
            await self.queue.fail(job_id, error_msg)
            self.logger.error(f"Job {job_id} failed with exception: {e}")
            raise  # Re-raise to trigger consecutive failure counting
    
    async def process_job(
        self,
        job_type: str,
        target_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Process a single job. Override in subclasses.
        
        Args:
            job_type: Type of job to process
            target_id: Target identifier for the job
            metadata: Additional job metadata
            
        Returns:
            True if job completed successfully, False otherwise
        """
        # Default implementation - override in subclasses
        self.logger.info(f"Processing job: {job_type} for {target_id}")
        await asyncio.sleep(1)  # Simulate work
        return True


# Global job queue instance (initialized lazily)
job_queue = None


def get_job_queue() -> JobQueue:
    """Get or create the global job queue instance."""
    global job_queue
    if job_queue is None:
        from .database import db_manager
        job_queue = JobQueue(db_manager)
    return job_queue


# Convenience functions for common operations
async def enqueue_job(
    job_type: str,
    target_id: str,
    priority: int = 5,
    max_retries: Optional[int] = None
) -> int:
    """Convenience function to enqueue a job."""
    queue = get_job_queue()
    return await queue.enqueue(job_type, target_id, priority, max_retries)


async def get_queue_status() -> Dict[str, Any]:
    """Convenience function to get queue status."""
    queue = get_job_queue()
    stats = await queue.get_queue_stats()
    return {
        "total_jobs": stats.total_jobs,
        "pending": stats.pending_jobs,
        "processing": stats.processing_jobs,
        "completed": stats.completed_jobs,
        "failed": stats.failed_jobs,
        "active_workers": stats.active_workers,
        "oldest_pending_age_minutes": stats.oldest_pending_age_minutes,
        "average_processing_time_seconds": stats.average_processing_time_seconds
    }


async def cleanup_old_completed_jobs(dry_run: bool = False) -> int:
    """Convenience function to cleanup old jobs."""
    queue = get_job_queue()
    return await queue.cleanup_old_jobs(dry_run)