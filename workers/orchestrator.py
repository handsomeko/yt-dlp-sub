"""
Orchestrator Worker for YouTube Content Intelligence & Repurposing Platform.

This is the main coordinator worker that manages job flow between different workers,
handles worker chaining, and orchestrates the overall workflow. It acts as the central
hub that processes jobs from the queue and routes them to appropriate specialized workers.

Features:
- Worker registry pattern for dynamic worker management
- Job routing logic based on job_type
- Error recovery and retry handling
- Progress tracking and monitoring
- Continuous processing mode with graceful shutdown
- Integration with JobQueue for queue management
"""

import asyncio
import signal
import time
import gc
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Type, Union
from enum import Enum
from dataclasses import dataclass

from workers.base import BaseWorker, WorkerStatus
from workers.monitor import MonitorWorker
from workers.monitor_enhanced import EnhancedMonitorWorker
from workers.audio_downloader import AudioDownloadWorker
from workers.transcriber import TranscribeWorker
from workers.summarizer import SummarizerWorker
from workers.generator import GeneratorWorker
from workers.storage import StorageWorker
from workers.quality import QualityWorker
from workers.publisher import PublishWorker
from core.queue import JobQueue, JobStatus, JobType, get_job_queue
from core.resource_manager import ResourceManager, ScalingAction, ScalingConfig
from core.database import DatabaseManager, db_manager


class OrchestratorStatus(Enum):
    """Orchestrator status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CircuitBreakerStatus(Enum):
    """Circuit breaker status for worker fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for worker fault tolerance (Issue #19)."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.status = CircuitBreakerStatus.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.status == CircuitBreakerStatus.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                self.status = CircuitBreakerStatus.HALF_OPEN
                return False
            return True
        return False
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.is_open():
            raise Exception(f"Circuit breaker is open - too many failures")
        
        try:
            result = await func(*args, **kwargs)
            
            # Reset on success
            if self.status == CircuitBreakerStatus.HALF_OPEN:
                self.status = CircuitBreakerStatus.CLOSED
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.status = CircuitBreakerStatus.OPEN
            
            raise e


@dataclass
class WorkerRegistry:
    """Enhanced registry entry for a worker with health tracking and lifecycle management."""
    worker_class: Type[BaseWorker]
    instance: Optional[BaseWorker] = None
    job_types: Set[str] = None
    enabled: bool = True
    # Issue #25: Worker lifecycle management
    health_status: str = "unknown"  # unknown, healthy, degraded, failed
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    max_failures_before_disable: int = 5
    last_failure_time: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 3
    supports_async: bool = False
    required_methods: Set[str] = None
    
    def __post_init__(self):
        if self.job_types is None:
            self.job_types = set()
        if self.required_methods is None:
            self.required_methods = {"validate_input", "execute"}


@dataclass
class ProcessingStats:
    """Processing statistics for the orchestrator."""
    total_jobs_processed: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    retried_jobs: int = 0
    jobs_per_minute: float = 0.0
    uptime_seconds: float = 0.0
    active_workers: int = 0
    queue_size: int = 0


class OrchestratorWorker(BaseWorker):
    """
    Main orchestrator worker that coordinates job flow between different workers.
    
    This worker acts as the central coordinator for the entire system, managing:
    - Job dequeuing from the main job queue
    - Routing jobs to appropriate specialized workers
    - Worker lifecycle management
    - Error handling and recovery
    - Progress monitoring and statistics
    - Graceful shutdown handling
    
    The orchestrator uses a worker registry pattern to dynamically manage
    different worker types and their capabilities.
    """
    
    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        job_queue: Optional[JobQueue] = None,
        worker_id: str = None,
        max_concurrent_jobs: int = 3,
        polling_interval: float = 5.0,
        health_check_interval: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        log_level: str = "INFO",
        enable_dynamic_scaling: bool = False,
        scaling_config: Optional[ScalingConfig] = None
    ) -> None:
        """
        Initialize the orchestrator worker.
        
        Args:
            database_manager: Database manager instance
            job_queue: Job queue instance
            worker_id: Unique worker identifier
            max_concurrent_jobs: Maximum number of concurrent jobs
            polling_interval: Queue polling interval in seconds
            health_check_interval: Health check interval in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries
            log_level: Logging level
            enable_dynamic_scaling: Enable dynamic worker scaling based on resources
            scaling_config: Configuration for dynamic scaling behavior
        """
        super().__init__(
            name="orchestrator",
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_level=log_level
        )
        
        # Core components
        self.db_manager = database_manager or db_manager
        self.job_queue = job_queue or get_job_queue()
        self.worker_id = worker_id or f"orchestrator-{int(time.time())}"
        
        # Configuration
        self.max_concurrent_jobs = max_concurrent_jobs
        self.polling_interval = polling_interval
        self.health_check_interval = health_check_interval
        
        # State management
        self.status = OrchestratorStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.stop_requested = False
        self.processing_tasks: Set[asyncio.Task] = set()
        self.stats = ProcessingStats()
        
        # Worker registry
        self.worker_registry: Dict[str, WorkerRegistry] = {}
        self._initialize_worker_registry()
        
        # Dynamic scaling
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.resource_manager: Optional[ResourceManager] = None
        if enable_dynamic_scaling:
            self.resource_manager = ResourceManager(scaling_config)
            self.log_with_context("Dynamic scaling enabled", extra_context={
                "min_workers": scaling_config.min_workers if scaling_config else 1,
                "max_workers": scaling_config.max_workers if scaling_config else "auto"
            })
        
        # Signal handling setup
        self._setup_signal_handlers()
        
        # Issue #19: Worker error handling and recovery
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Issue #25: Worker lifecycle management
        self._worker_health_check_interval = 300.0  # 5 minutes
        self._last_worker_health_check = 0.0
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for orchestrator worker.
        
        The orchestrator can be started with various configurations:
        - continuous_mode: Boolean flag for continuous processing
        - max_jobs: Maximum number of jobs to process before stopping
        - job_types: List of job types to process (filters)
        
        Args:
            input_data: Input configuration for orchestrator
            
        Returns:
            True if input is valid, False otherwise
        """
        # Orchestrator can run with minimal or no input
        if not isinstance(input_data, dict):
            self.log_with_context("Input must be a dictionary", level="ERROR")
            return False
        
        # Validate optional parameters
        continuous_mode = input_data.get("continuous_mode", True)
        if not isinstance(continuous_mode, bool):
            self.log_with_context("continuous_mode must be boolean", level="ERROR")
            return False
        
        max_jobs = input_data.get("max_jobs")
        if max_jobs is not None and (not isinstance(max_jobs, int) or max_jobs <= 0):
            self.log_with_context("max_jobs must be positive integer", level="ERROR")
            return False
        
        job_types = input_data.get("job_types")
        if job_types is not None and not isinstance(job_types, list):
            self.log_with_context("job_types must be a list", level="ERROR")
            return False
        
        return True
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute orchestrator main processing loop.
        
        Args:
            input_data: Configuration for orchestrator execution
            
        Returns:
            Dict containing processing results and statistics
        """
        continuous_mode = input_data.get("continuous_mode", True)
        max_jobs = input_data.get("max_jobs")
        job_type_filter = input_data.get("job_types", [])
        
        self.log_with_context(
            "Starting orchestrator execution",
            extra_context={
                "continuous_mode": continuous_mode,
                "max_jobs": max_jobs,
                "job_type_filter": job_type_filter,
                "worker_id": self.worker_id
            }
        )
        
        # Initialize orchestrator
        await self._start_orchestrator()
        
        try:
            if continuous_mode:
                # Run continuous processing until stop is requested
                await self._run_continuous_processing(max_jobs, job_type_filter)
            else:
                # Process a single batch of jobs
                await self._process_job_batch(max_jobs or 10, job_type_filter)
            
            # Get final statistics
            final_stats = await self._get_processing_stats()
            
            return {
                "status": "completed",
                "worker_id": self.worker_id,
                "processing_stats": final_stats.__dict__,
                "uptime": self._calculate_uptime(),
                "workers_registered": len(self.worker_registry),
                "stop_reason": "requested" if self.stop_requested else "completed"
            }
            
        finally:
            await self._stop_orchestrator()
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle and categorize orchestrator errors.
        
        Args:
            error: Exception that occurred during execution
            
        Returns:
            Error handling result with context and recovery information
        """
        error_str = str(error).lower()
        error_category = "unknown"
        recovery_action = "retry"
        should_stop = False
        
        # Database errors
        if any(keyword in error_str for keyword in [
            "database", "sqlite", "connection", "timeout"
        ]):
            error_category = "database"
            recovery_action = "reconnect_and_retry"
            
        # Queue errors
        elif any(keyword in error_str for keyword in [
            "queue", "dequeue", "job", "worker"
        ]):
            error_category = "queue"
            recovery_action = "reset_queue_connection"
            
        # Resource exhaustion
        elif any(keyword in error_str for keyword in [
            "memory", "disk", "space", "resource"
        ]):
            error_category = "resource_exhaustion"
            recovery_action = "cleanup_and_retry"
            should_stop = True
            
        # Signal/shutdown related
        elif any(keyword in error_str for keyword in [
            "signal", "interrupt", "shutdown", "cancelled"
        ]):
            error_category = "shutdown"
            recovery_action = "graceful_shutdown"
            should_stop = True
            
        self.log_with_context(
            f"Orchestrator error categorized as {error_category}",
            level="ERROR",
            extra_context={
                "error_message": str(error),
                "recovery_action": recovery_action,
                "should_stop": should_stop
            }
        )
        
        if should_stop:
            self.stop_requested = True
        
        return {
            "error_category": error_category,
            "error_message": str(error),
            "recovery_action": recovery_action,
            "should_stop": should_stop,
            "retry_delay_seconds": self._get_recovery_delay(error_category),
            "orchestrator_stats": self.stats.__dict__
        }
    
    async def _start_orchestrator(self) -> None:
        """Initialize and start the orchestrator."""
        self.status = OrchestratorStatus.STARTING
        self.start_time = datetime.now()
        self.stop_requested = False
        self.processing_tasks.clear()
        
        # Initialize worker instances
        await self._initialize_workers()
        
        # Recover any stale jobs
        recovered_count = await self.job_queue.recover_stale_jobs()
        if recovered_count > 0:
            self.log_with_context(f"Recovered {recovered_count} stale jobs from previous runs")
        
        self.status = OrchestratorStatus.RUNNING
        self.log_with_context("Orchestrator started successfully")
    
    async def _stop_orchestrator(self) -> None:
        """Gracefully stop the orchestrator with enhanced cleanup."""
        self.status = OrchestratorStatus.STOPPING
        self.stop_requested = True
        
        # Cancel all processing tasks
        if self.processing_tasks:
            self.log_with_context(f"Cancelling {len(self.processing_tasks)} processing tasks")
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.processing_tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                self.log_with_context("Some tasks did not complete within timeout", level="WARNING")
        
        # Issue #25: Clean up worker instances
        await self._cleanup_workers()
        
        self.processing_tasks.clear()
        self.status = OrchestratorStatus.STOPPED
        self.log_with_context("Orchestrator stopped")
    
    async def _cleanup_workers(self) -> None:
        """Clean up all worker instances and resources."""
        self.log_with_context("Cleaning up worker instances")
        
        for worker_name, registry in self.worker_registry.items():
            if registry.instance:
                try:
                    # Call cleanup method if available
                    if hasattr(registry.instance, 'cleanup'):
                        if asyncio.iscoroutinefunction(registry.instance.cleanup):
                            await registry.instance.cleanup()
                        else:
                            registry.instance.cleanup()
                    
                    # Clear instance reference
                    registry.instance = None
                    registry.health_status = "unknown"
                    
                    self.log_with_context(f"Cleaned up worker: {worker_name}")
                    
                except Exception as e:
                    self.log_with_context(
                        f"Error cleaning up worker {worker_name}: {e}",
                        level="WARNING"
                    )
        
        # Clear circuit breakers
        self._circuit_breakers.clear()
    
    async def _run_continuous_processing(
        self, 
        max_jobs: Optional[int] = None, 
        job_type_filter: List[str] = None
    ) -> None:
        """
        Run continuous job processing until stopped.
        
        Args:
            max_jobs: Maximum jobs to process before stopping
            job_type_filter: List of job types to process
        """
        jobs_processed = 0
        last_health_check = time.time()
        memory_threshold_mb = 2048  # 2GB threshold
        last_memory_cleanup = time.time()
        memory_cleanup_interval = 300  # Check memory every 5 minutes
        
        # Check initial memory
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.log_with_context(
            "Starting continuous processing mode",
            extra_context={"initial_memory_mb": f"{initial_memory_mb:.1f}"}
        )
        
        while not self.stop_requested and self.status == OrchestratorStatus.RUNNING:
            try:
                # Process a batch of jobs
                batch_results = await self._process_job_batch(
                    batch_size=min(self.max_concurrent_jobs, 5),
                    job_type_filter=job_type_filter
                )
                
                jobs_processed += batch_results.get("jobs_processed", 0)
                
                # Update statistics
                self.stats.total_jobs_processed = jobs_processed
                self.stats.uptime_seconds = self._calculate_uptime()
                self._update_jobs_per_minute()
                
                # Check memory periodically
                current_time = time.time()
                if current_time - last_memory_cleanup > memory_cleanup_interval:
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    self.log_with_context(
                        f"💾 Memory check: {current_memory_mb:.1f}MB",
                        extra_context={"memory_percent": f"{process.memory_percent():.1f}%"}
                    )
                    
                    # Cleanup if memory usage is high
                    if current_memory_mb > memory_threshold_mb:
                        self.log_with_context(
                            f"⚠️ High memory usage ({current_memory_mb:.1f}MB), running cleanup...",
                            level="WARNING"
                        )
                        gc.collect()
                        gc.collect(2)  # Force collection of highest generation
                        await asyncio.sleep(2)  # Give system time to reclaim memory
                        after_cleanup_mb = process.memory_info().rss / 1024 / 1024
                        self.log_with_context(
                            f"✅ Memory after cleanup: {after_cleanup_mb:.1f}MB"
                        )
                    
                    last_memory_cleanup = current_time
                
                # Force cleanup every 10 jobs to prevent gradual buildup
                if jobs_processed > 0 and jobs_processed % 10 == 0:
                    gc.collect()
                
                # Check if we've hit the job limit
                if max_jobs and jobs_processed >= max_jobs:
                    self.log_with_context(f"Reached maximum job limit: {max_jobs}")
                    break
                
                # Periodic health check
                current_time = time.time()
                if current_time - last_health_check >= self.health_check_interval:
                    await self._perform_health_check()
                    last_health_check = current_time
                
                # Wait before next batch if no jobs were processed
                if batch_results.get("jobs_processed", 0) == 0:
                    await asyncio.sleep(self.polling_interval)
                
            except asyncio.CancelledError:
                self.log_with_context("Processing cancelled")
                break
            except Exception as e:
                self.log_with_context(f"Error in processing loop: {e}", level="ERROR")
                await asyncio.sleep(self.retry_delay)
                continue
    
    async def _process_job_batch(
        self, 
        batch_size: int = 5, 
        job_type_filter: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of jobs from the queue.
        
        Args:
            batch_size: Maximum number of jobs to process in parallel
            job_type_filter: Optional list of job types to process
            
        Returns:
            Dict with batch processing results
        """
        batch_start_time = time.time()
        jobs_processed = 0
        successful_jobs = 0
        failed_jobs = 0
        
        # Dynamic scaling: Adjust batch size based on resources
        if self.enable_dynamic_scaling and self.resource_manager:
            # Get pending job count
            pending_jobs = await self.job_queue.get_queue_size()
            current_workers = len(self.processing_tasks)
            
            # Get scaling decision
            action, recommended_workers = self.resource_manager.get_scaling_decision(
                active_workers=current_workers,
                pending_jobs=pending_jobs
            )
            
            # Adjust batch size based on recommendation
            if action == ScalingAction.EMERGENCY_STOP:
                self.log_with_context(
                    "EMERGENCY: System resources critical - reducing workers",
                    level="ERROR"
                )
                batch_size = 1  # Minimum
            elif action == ScalingAction.SCALE_DOWN:
                batch_size = max(1, recommended_workers)
                self.log_with_context(
                    f"Scaling down: batch_size={batch_size} (was {current_workers})"
                )
            elif action == ScalingAction.SCALE_UP:
                batch_size = min(recommended_workers, self.max_concurrent_jobs)
                self.log_with_context(
                    f"Scaling up: batch_size={batch_size} (was {current_workers})"
                )
            else:  # MAINTAIN
                batch_size = min(current_workers or batch_size, self.max_concurrent_jobs)
        else:
            # Use configured max or provided batch size
            batch_size = min(batch_size, self.max_concurrent_jobs)
        
        # Create tasks for concurrent job processing
        processing_tasks = []
        
        for _ in range(batch_size):
            if self.stop_requested:
                break
                
            # Get next job from queue
            job_data = await self.job_queue.dequeue(self.worker_id)
            if not job_data:
                break  # No more jobs available
            
            job_id, job_type, target_id, metadata = job_data
            
            # Apply job type filter if specified
            if job_type_filter and job_type not in job_type_filter:
                # Re-queue the job as we're not processing this type
                await self.job_queue.retry(job_id, force=True)
                continue
            
            # Create task for job processing
            task = asyncio.create_task(
                self._process_single_job(job_id, job_type, target_id, metadata)
            )
            processing_tasks.append(task)
            self.processing_tasks.add(task)
        
        if not processing_tasks:
            return {
                "jobs_processed": 0,
                "successful_jobs": 0,
                "failed_jobs": 0,
                "processing_time": 0.0
            }
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Clean up completed tasks
        for task in processing_tasks:
            self.processing_tasks.discard(task)
        
        # Process results
        for result in results:
            jobs_processed += 1
            if isinstance(result, Exception):
                failed_jobs += 1
                self.log_with_context(f"Job processing exception: {result}", level="ERROR")
            elif isinstance(result, dict):
                if result.get("status") == "success":
                    successful_jobs += 1
                else:
                    failed_jobs += 1
            else:
                failed_jobs += 1
        
        # Update statistics
        self.stats.successful_jobs += successful_jobs
        self.stats.failed_jobs += failed_jobs
        
        processing_time = time.time() - batch_start_time
        
        self.log_with_context(
            f"Processed job batch",
            extra_context={
                "jobs_processed": jobs_processed,
                "successful": successful_jobs,
                "failed": failed_jobs,
                "processing_time": f"{processing_time:.2f}s"
            }
        )
        
        return {
            "jobs_processed": jobs_processed,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "processing_time": processing_time
        }
    
    async def _process_single_job(
        self, 
        job_id: int, 
        job_type: str, 
        target_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single job using the appropriate worker.
        
        Args:
            job_id: Job ID from queue
            job_type: Type of job to process
            target_id: Target identifier for the job
            metadata: Additional job metadata
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        self.log_with_context(
            f"Processing job {job_id}",
            extra_context={
                "job_type": job_type,
                "target_id": target_id,
                "worker_id": self.worker_id
            }
        )
        
        try:
            # Route job to appropriate worker
            worker = await self._get_worker_for_job_type(job_type)
            if not worker:
                raise ValueError(f"No worker available for job type: {job_type}")
            
            # Prepare input data based on job type
            input_data = await self._prepare_job_input_data(job_type, target_id, metadata)
            
            # Issue #17: Enhanced worker execution with proper async/sync handling
            worker_name = next((name for name, reg in self.worker_registry.items() 
                              if reg.instance == worker), "unknown")
            
            circuit_breaker = self._get_circuit_breaker(worker_name)
            
            try:
                # Execute through circuit breaker for fault tolerance
                result = await circuit_breaker.call(self._execute_worker_safely, worker, input_data)
                
                # Update worker health on successful execution
                if worker_name in self.worker_registry:
                    registry = self.worker_registry[worker_name]
                    registry.consecutive_failures = 0
                    registry.health_status = "healthy"
                    registry.last_health_check = datetime.now()
                    
            except Exception as execution_error:
                # Handle worker execution failure
                if worker_name in self.worker_registry:
                    registry = self.worker_registry[worker_name]
                    registry.consecutive_failures += 1
                    registry.last_failure_time = datetime.now()
                    
                    if registry.consecutive_failures >= registry.max_failures_before_disable:
                        registry.health_status = "failed"
                        registry.enabled = False
                        self.log_with_context(
                            f"Worker {worker_name} disabled after {registry.consecutive_failures} failures",
                            level="ERROR"
                        )
                    else:
                        registry.health_status = "degraded"
                
                raise execution_error
            
            # Check result status
            if result.get("status") == WorkerStatus.SUCCESS.value:
                await self.job_queue.complete(job_id, result.get("data"))
                
                # Handle job chaining if needed
                await self._handle_job_chaining(job_type, target_id, result)
                
                processing_time = time.time() - start_time
                self.log_with_context(
                    f"Job {job_id} completed successfully",
                    extra_context={
                        "processing_time": f"{processing_time:.2f}s",
                        "worker": worker.name
                    }
                )
                
                return {"status": "success", "result": result}
                
            else:
                # Job failed, mark as failed in queue
                error_msg = result.get("error", "Job returned non-success status")
                await self.job_queue.fail(job_id, error_msg, result.get("error_details"))
                
                self.log_with_context(
                    f"Job {job_id} failed",
                    level="WARNING",
                    extra_context={
                        "error": error_msg,
                        "worker": worker.name
                    }
                )
                
                return {"status": "failed", "error": error_msg}
        
        except Exception as e:
            # Issue #19: Enhanced error handling with categorization and recovery
            error_category = self._categorize_worker_error(e)
            
            # Determine if this is a recoverable error
            is_recoverable = error_category in ['network', 'temporary', 'rate_limit']
            
            if is_recoverable and self.stats.retried_jobs < job_id * 2:  # Simple retry limit
                # Mark for retry instead of immediate failure
                retry_delay = self._get_retry_delay_for_error(error_category)
                await self.job_queue.retry(job_id, delay=retry_delay)
                self.stats.retried_jobs += 1
                
                self.log_with_context(
                    f"Job {job_id} scheduled for retry",
                    level="WARNING",
                    extra_context={
                        "job_type": job_type,
                        "target_id": target_id,
                        "error_category": error_category,
                        "retry_delay": retry_delay,
                        "error": str(e)
                    }
                )
                
                return {"status": "retrying", "error_category": error_category}
            else:
                # Mark job as permanently failed
                error_msg = f"Job processing exception ({error_category}): {str(e)}"
                await self.job_queue.fail(job_id, error_msg, {"error_category": error_category})
                
                self.log_with_context(
                    f"Job {job_id} failed permanently",
                    level="ERROR",
                    extra_context={
                        "job_type": job_type,
                        "target_id": target_id,
                        "error_category": error_category,
                        "is_recoverable": is_recoverable,
                        "error": str(e)
                    }
                )
                
                # Don't re-raise - return error status instead to prevent orchestrator crash
                return {"status": "failed", "error": error_msg, "error_category": error_category}
    
    # ========================================
    # Issue #17, #18, #19, #25: Enhanced Worker Management
    # ========================================
    
    def _get_circuit_breaker(self, worker_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for worker."""
        if worker_name not in self._circuit_breakers:
            self._circuit_breakers[worker_name] = CircuitBreaker(
                failure_threshold=3,  # Allow 3 failures before opening
                recovery_timeout=60,  # Try again after 1 minute
                expected_exception=Exception
            )
        return self._circuit_breakers[worker_name]
    
    async def _execute_worker_safely(self, worker: BaseWorker, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute worker with proper async/sync handling and validation."""
        # Issue #17: Proper async/sync coordination
        registry = next((reg for reg in self.worker_registry.values() if reg.instance == worker), None)
        
        if registry and registry.supports_async and hasattr(worker, 'run_async'):
            # Use async version
            result = await worker.run_async(input_data)
        elif hasattr(worker, 'run_async'):
            # Try async first, fallback to sync
            try:
                result = await worker.run_async(input_data)
            except Exception as async_error:
                self.log_with_context(
                    f"Worker async execution failed, trying sync: {async_error}",
                    level="WARNING"
                )
                # Fallback to sync execution in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None, worker.run, input_data
                )
        else:
            # Use sync version in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None, worker.run, input_data
            )
        
        # Issue #18: Validate result format
        if not isinstance(result, dict):
            self.log_with_context(
                "Worker returned non-dict result, wrapping",
                level="WARNING"
            )
            result = {"data": result, "status": WorkerStatus.SUCCESS.value}
        
        return result
    
    def _categorize_worker_error(self, error: Exception) -> str:
        """Categorize worker errors for appropriate recovery strategies."""
        error_str = str(error).lower()
        
        # Network-related errors (recoverable)
        if any(keyword in error_str for keyword in [
            'connection', 'timeout', 'network', 'dns', 'http', 'ssl', 'tls'
        ]):
            return 'network'
        
        # Rate limiting (recoverable with delay)
        elif any(keyword in error_str for keyword in [
            'rate limit', 'too many requests', '429', 'quota exceeded'
        ]):
            return 'rate_limit'
        
        # Temporary service issues (recoverable)
        elif any(keyword in error_str for keyword in [
            'service unavailable', '503', '502', '504', 'temporary'
        ]):
            return 'temporary'
        
        # Input/validation errors (not recoverable)
        elif any(keyword in error_str for keyword in [
            'validation', 'invalid input', 'bad request', '400'
        ]):
            return 'validation'
        
        # Permission/auth errors (not recoverable)
        elif any(keyword in error_str for keyword in [
            'unauthorized', '401', '403', 'permission', 'forbidden', 'access denied'
        ]):
            return 'permission'
        
        # Resource errors (potentially recoverable)
        elif any(keyword in error_str for keyword in [
            'memory', 'disk space', 'resource', 'capacity'
        ]):
            return 'resource'
        
        # Unknown error type
        else:
            return 'unknown'
    
    def _get_retry_delay_for_error(self, error_category: str) -> int:
        """Get appropriate retry delay based on error category."""
        delay_map = {
            'network': 30,      # Network issues - retry after 30s
            'rate_limit': 300,  # Rate limiting - retry after 5 minutes
            'temporary': 60,    # Temporary issues - retry after 1 minute
            'resource': 120,    # Resource issues - retry after 2 minutes
            'unknown': 60       # Unknown errors - retry after 1 minute
        }
        
        return delay_map.get(error_category, 60)
    
    async def _validate_worker_health(self, worker_name: str, registry: WorkerRegistry) -> bool:
        """Validate worker health and compatibility."""
        try:
            # Issue #25: Check if worker needs health check
            now = datetime.now()
            if (registry.last_health_check and 
                now - registry.last_health_check < timedelta(seconds=self._worker_health_check_interval)):
                return registry.health_status in ["healthy", "degraded"]
            
            # Issue #18: Validate worker has required methods
            if registry.instance:
                for method_name in registry.required_methods:
                    if not hasattr(registry.instance, method_name):
                        self.log_with_context(
                            f"Worker {worker_name} missing required method: {method_name}",
                            level="ERROR"
                        )
                        registry.health_status = "failed"
                        return False
            
            # Test basic functionality with minimal input
            if registry.instance:
                try:
                    # Test validate_input method
                    test_input = {"test": True}
                    if hasattr(registry.instance, 'validate_input'):
                        registry.instance.validate_input(test_input)
                    
                    registry.health_status = "healthy"
                    registry.last_health_check = now
                    return True
                    
                except Exception as health_error:
                    self.log_with_context(
                        f"Worker {worker_name} health check failed: {health_error}",
                        level="WARNING"
                    )
                    registry.health_status = "degraded"
                    registry.last_health_check = now
                    return False
            
            return True
            
        except Exception as e:
            self.log_with_context(
                f"Error validating worker {worker_name} health: {e}",
                level="ERROR"
            )
            registry.health_status = "failed"
            return False
    
    async def _restart_worker(self, worker_name: str, registry: WorkerRegistry) -> bool:
        """Restart a failed worker instance."""
        try:
            if registry.restart_count >= registry.max_restarts:
                self.log_with_context(
                    f"Worker {worker_name} exceeded max restart attempts ({registry.max_restarts})",
                    level="ERROR"
                )
                return False
            
            # Clean up old instance
            if registry.instance:
                try:
                    if hasattr(registry.instance, 'cleanup'):
                        await registry.instance.cleanup()
                except Exception as cleanup_error:
                    self.log_with_context(
                        f"Error cleaning up worker {worker_name}: {cleanup_error}",
                        level="WARNING"
                    )
            
            # Create new instance
            registry.instance = None
            registry.instance = await self._initialize_worker_safely(worker_name, registry)
            
            if registry.instance:
                registry.restart_count += 1
                registry.consecutive_failures = 0
                registry.health_status = "healthy"
                registry.last_health_check = datetime.now()
                
                self.log_with_context(
                    f"Successfully restarted worker {worker_name} (restart #{registry.restart_count})"
                )
                return True
            
            return False
            
        except Exception as e:
            self.log_with_context(
                f"Failed to restart worker {worker_name}: {e}",
                level="ERROR"
            )
            return False
    
    async def _initialize_worker_safely(self, worker_name: str, registry: WorkerRegistry) -> Optional[BaseWorker]:
        """Initialize worker with comprehensive validation."""
        try:
            # Create worker instance
            worker_instance = registry.worker_class()
            
            # Issue #18: Validate worker interface compatibility
            required_attributes = ['name', 'validate_input', 'execute']
            for attr in required_attributes:
                if not hasattr(worker_instance, attr):
                    self.log_with_context(
                        f"Worker {worker_name} missing required attribute: {attr}",
                        level="ERROR"
                    )
                    return None
            
            # Check async support
            registry.supports_async = hasattr(worker_instance, 'run_async')
            
            # Validate worker can handle its assigned job types
            for job_type in registry.job_types:
                test_input = self._create_test_input_for_job_type(job_type)
                if not worker_instance.validate_input(test_input):
                    self.log_with_context(
                        f"Worker {worker_name} failed validation for job type {job_type}",
                        level="WARNING"
                    )
            
            self.log_with_context(
                f"Worker {worker_name} initialized successfully",
                extra_context={
                    "supports_async": registry.supports_async,
                    "job_types": list(registry.job_types)
                }
            )
            
            return worker_instance
            
        except Exception as e:
            self.log_with_context(
                f"Failed to initialize worker {worker_name}: {e}",
                level="ERROR"
            )
            return None
    
    def _create_test_input_for_job_type(self, job_type: str) -> Dict[str, Any]:
        """Create minimal test input for job type validation."""
        test_inputs = {
            "check_channel": {"channel_id": "test_channel"},
            "download_audio": {"video_id": "test_video", "video_url": "https://youtube.com/watch?v=test"},
            "extract_transcript": {"video_id": "test_video", "video_url": "https://youtube.com/watch?v=test"},
            "generate_content": {"video_id": "test_video"},
            "check_transcript_quality": {"target_id": "test_video", "content": "test transcript"},
            "check_content_quality": {"target_id": "test_video", "content": "test content"},
            "store_file": {"file_path": "/test/path", "file_type": "test"},
            "publish_content": {"video_id": "test_video"}
        }
        
        return test_inputs.get(job_type, {"target_id": "test"})
    
    async def _prepare_job_input_data(
        self, 
        job_type: str, 
        target_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare input data for a job based on its type.
        
        Args:
            job_type: Type of job
            target_id: Target identifier
            metadata: Additional metadata
            
        Returns:
            Prepared input data for the worker
        """
        base_input = {"target_id": target_id, "metadata": metadata}
        
        if job_type == "check_channel":
            # MonitorWorker input
            return {
                "channel_id": target_id,
                "check_all": False
            }
        
        elif job_type == "download_transcript":
            # DownloadWorker input - need to get video URL from database
            async with self.db_manager.get_session() as session:
                from sqlalchemy import select
                from core.database import Video
                
                result = await session.execute(
                    select(Video).where(Video.video_id == target_id)
                )
                video = result.scalar_one_or_none()
                
                if not video:
                    # Construct URL from video_id if not in database
                    video_url = f"https://www.youtube.com/watch?v={target_id}"
                    channel_id = "unknown"
                else:
                    video_url = f"https://www.youtube.com/watch?v={video.video_id}"
                    channel_id = video.channel_id
                
                return {
                    "video_id": target_id,
                    "video_url": video_url,
                    "channel_id": channel_id
                }
        
        elif job_type == "check_all_channels":
            # MonitorWorker input for checking all channels
            return {
                "check_all": True
            }
        
        # Default fallback
        return base_input
    
    async def _handle_job_chaining(
        self, 
        job_type: str, 
        target_id: str, 
        result: Dict[str, Any]
    ) -> None:
        """
        Handle chaining of jobs based on completion results.
        
        Implements the complete workflow chain:
        Monitor → Download → Store → Transcribe → Store → Generate → Store → Quality → Export
        
        Args:
            job_type: Type of completed job
            target_id: Target identifier
            result: Job execution result
        """
        try:
            # Channel monitoring creates download jobs
            if job_type == "check_channel":
                new_videos_found = result.get("data", {}).get("new_videos_found", 0)
                jobs_created = result.get("data", {}).get("jobs_created", 0)
                
                if new_videos_found > 0:
                    self.log_with_context(
                        f"Channel check created {jobs_created} new download jobs",
                        extra_context={
                            "channel_id": target_id,
                            "new_videos": new_videos_found
                        }
                    )
            
            # Audio download completion → Create transcription job (next step in workflow)
            elif job_type == "download_audio":
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    audio_path = result.get("data", {}).get("audio_path")
                    channel_id = result.get("data", {}).get("channel_id", "unknown")
                    
                    if audio_path:
                        # Create storage job for audio file
                        await self._create_storage_job(target_id, "audio", audio_path, channel_id)
                        
                        # CRITICAL: Create transcription job to continue workflow chain
                        await self._create_transcription_job(target_id, audio_path)
                        
                        self.log_with_context(
                            f"Created storage and transcription jobs for audio: {target_id}",
                            extra_context={"audio_path": audio_path}
                        )
            
            # Transcript extraction completion → Save to database and create storage jobs
            elif job_type in ["extract_transcript", "download_transcript"]:
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    transcript_data = result.get("data", {})
                    channel_id = transcript_data.get("channel_id", "unknown")
                    
                    # CRITICAL: Save transcript to database with detected language
                    await self._save_transcript_to_database(target_id, transcript_data)
                    
                    # Store SRT file
                    srt_path = transcript_data.get("srt_path")
                    if srt_path:
                        await self._create_storage_job(target_id, "transcript", srt_path, channel_id)
                    
                    # Store text file
                    text_path = transcript_data.get("text_path") or transcript_data.get("txt_path")
                    if text_path:
                        await self._create_storage_job(target_id, "transcript", text_path, channel_id)
                    
                    # Create quality check job for transcript
                    await self._create_quality_job(target_id, "check_transcript_quality")
                    
                    self.log_with_context(
                        f"Created storage and quality jobs for transcript: {target_id}",
                        extra_context={"srt_path": srt_path, "text_path": text_path}
                    )
            
            # Content generation completion → Create storage jobs for generated content
            elif job_type in ["generate_content", "generate_blog", "generate_social", 
                            "generate_newsletter", "generate_scripts"]:
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    generation_data = result.get("data", {})
                    channel_id = generation_data.get("channel_id", "unknown")
                    storage_paths = generation_data.get("storage_paths", [])
                    
                    # Create storage jobs for each generated content file
                    for content_path in storage_paths:
                        await self._create_storage_job(target_id, "content", content_path, channel_id)
                    
                    # Create quality check job for generated content
                    await self._create_quality_job(target_id, "check_content_quality")
                    
                    self.log_with_context(
                        f"Created storage and quality jobs for generated content: {target_id}",
                        extra_context={"content_files": len(storage_paths)}
                    )
            
            # Summary generation completion → Create storage job
            elif job_type in ["summarize", "generate_summary"]:
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    summary_data = result.get("data", {})
                    channel_id = summary_data.get("channel_id", "unknown")
                    summary_path = summary_data.get("summary_path")
                    
                    if summary_path:
                        await self._create_storage_job(target_id, "content", summary_path, channel_id)
                        self.log_with_context(
                            f"Created storage job for summary: {target_id}",
                            extra_context={"summary_path": summary_path}
                        )
            
            # Transcript quality check completion → Create content generation job
            elif job_type in ["check_transcript_quality"]:
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    transcript_quality_data = result.get("data", {})
                    
                    # Only proceed with generation if transcript quality passed
                    if transcript_quality_data.get("passed", False):
                        await self._create_content_generation_job(target_id)
                        
                        self.log_with_context(
                            f"Created content generation job for transcript: {target_id}",
                            extra_context={"transcript_quality_passed": True}
                        )
                    else:
                        self.log_with_context(
                            f"Skipping content generation for {target_id} - transcript quality check failed",
                            level="WARNING",
                            extra_context={
                                "quality_score": transcript_quality_data.get("overall_score", 0),
                                "transcript_quality_passed": False
                            }
                        )
            
            # Content quality check completion → Create publishing job (final step in workflow)
            elif job_type in ["check_content_quality", "validate_quality"]:
                if result.get("status") == WorkerStatus.SUCCESS.value:
                    quality_data = result.get("data", {})
                    
                    # Only proceed with publishing if quality check passed
                    if quality_data.get("passed", False):
                        await self._create_publishing_job(target_id)
                        
                        self.log_with_context(
                            f"Created publishing job for content: {target_id}",
                            extra_context={"quality_passed": True}
                        )
                    else:
                        self.log_with_context(
                            f"Skipping publishing for {target_id} - quality check failed",
                            level="WARNING",
                            extra_context={
                                "quality_score": quality_data.get("overall_score", 0),
                                "quality_passed": False
                            }
                        )
            
        except Exception as e:
            self.log_with_context(
                f"Error in job chaining: {e}",
                level="WARNING",
                extra_context={
                    "job_type": job_type,
                    "target_id": target_id
                }
            )
    
    async def _create_storage_job(
        self, 
        video_id: str, 
        file_type: str, 
        file_path: str, 
        channel_id: str
    ) -> None:
        """
        Create a storage job for file synchronization.
        
        Args:
            video_id: Video ID being processed
            file_type: Type of file (audio, transcript, content)
            file_path: Path to file that needs storage
            channel_id: Channel ID for organization
        """
        try:
            job_id = await self.job_queue.enqueue(
                job_type="store_file",
                target_id=f"{video_id}:{file_type}",
                priority=3,  # Lower priority than main processing
                metadata={
                    "video_id": video_id,
                    "file_type": file_type,
                    "file_path": file_path,
                    "channel_id": channel_id
                }
            )
            
            self.log_with_context(
                f"Created storage job {job_id} for {file_type}",
                extra_context={
                    "video_id": video_id,
                    "file_path": file_path
                }
            )
            
        except Exception as e:
            self.log_with_context(
                f"Failed to create storage job: {str(e)}",
                level="ERROR",
                extra_context={
                    "video_id": video_id,
                    "file_type": file_type,
                    "file_path": file_path
                }
            )
    
    async def _create_quality_job(self, video_id: str, quality_job_type: str) -> None:
        """
        Create a quality check job with required content data.
        
        Args:
            video_id: Video ID being processed
            quality_job_type: Type of quality check job
        """
        try:
            # Determine target type and get content based on job type
            if quality_job_type == "check_transcript_quality":
                target_type = "transcript"
                content = await self._get_transcript_content(video_id)
            elif quality_job_type == "check_content_quality":
                target_type = "content"
                content = await self._get_generated_content(video_id)
            else:
                raise Exception(f"Unknown quality job type: {quality_job_type}")
            
            if not content:
                raise Exception(f"No {target_type} content found for video {video_id}")
            
            job_id = await self.job_queue.enqueue(
                job_type=quality_job_type,
                target_id=video_id,
                priority=4,  # Lower priority than main processing
                metadata={
                    "target_id": video_id,
                    "target_type": target_type,
                    "content": content,
                    "quality_check_type": quality_job_type
                }
            )
            
            self.log_with_context(
                f"Created quality job {job_id} ({quality_job_type})",
                extra_context={"video_id": video_id}
            )
            
        except Exception as e:
            self.log_with_context(
                f"Failed to create quality job: {str(e)}",
                level="ERROR",
                extra_context={
                    "video_id": video_id,
                    "quality_job_type": quality_job_type
                }
            )
    
    async def _create_publishing_job(self, video_id: str) -> None:
        """
        Create a publishing job for distributing generated content.
        
        Args:
            video_id: Video ID being processed
        """
        try:
            job_id = await self.job_queue.enqueue(
                job_type="publish_content",
                target_id=video_id,
                priority=5,  # Lowest priority - final step in workflow
                metadata={
                    "video_id": video_id,
                    "content_type": "all",  # Publish all generated content types
                    "publish_targets": ["local", "webhook"],  # Phase 1 targets
                    "publish_options": {}
                }
            )
            
            self.log_with_context(
                f"Created publishing job {job_id} for video {video_id}",
                extra_context={"video_id": video_id}
            )
            
        except Exception as e:
            self.log_with_context(
                f"Failed to create publishing job: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
    
    async def _create_content_generation_job(self, video_id: str) -> None:
        """
        Create a content generation job for generating various content types.
        
        Args:
            video_id: Video ID being processed
        """
        try:
            job_id = await self.job_queue.enqueue(
                job_type="generate_content",
                target_id=video_id,
                priority=3,  # Medium priority - core workflow step
                metadata={
                    "video_id": video_id,
                    "content_types": ["summary", "blog_post", "social_media"],  # Default content types for Phase 1
                    "generation_options": {}
                }
            )
            
            self.log_with_context(
                f"Created content generation job {job_id} for video {video_id}",
                extra_context={"video_id": video_id}
            )
            
        except Exception as e:
            self.log_with_context(
                f"Failed to create content generation job: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
    
    async def _create_transcription_job(self, video_id: str, audio_path: str) -> None:
        """
        Create a transcription job for audio file processing.
        
        Args:
            video_id: Video ID being processed
            audio_path: Path to audio file to transcribe
        """
        try:
            # Look up video URL from database (required by transcriber)
            from core.database import Video, db_manager
            from sqlalchemy import select
            
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(Video).where(Video.video_id == video_id)
                )
                video = result.scalar_one_or_none()
                
                if not video:
                    raise Exception(f"Video {video_id} not found in database")
                
                # Create video_url from video_id (YouTube standard format)
                video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            job_id = await self.job_queue.enqueue(
                job_type="extract_transcript",
                target_id=video_id,
                priority=2,  # High priority - critical workflow step
                metadata={
                    "video_id": video_id,
                    "video_url": video_url,
                    "audio_path": audio_path,
                    "output_format": "srt"  # Default to SRT format as per PRD
                }
            )
            
            self.log_with_context(
                f"Created transcription job {job_id} for video {video_id}",
                extra_context={"video_id": video_id, "audio_path": audio_path}
            )
            
        except Exception as e:
            self.log_with_context(
                f"Failed to create transcription job: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id, "audio_path": audio_path}
            )
    
    async def _get_transcript_content(self, video_id: str) -> Optional[str]:
        """Get transcript content for quality validation."""
        try:
            from core.database import Transcript
            from sqlalchemy import select
            
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(Transcript).where(Transcript.video_id == video_id)
                )
                transcript = result.scalar_one_or_none()
                
                if transcript and transcript.content_text:
                    return transcript.content_text
                else:
                    # Try to read from file system as fallback
                    from core.storage_paths_v2 import get_storage_paths_v2
                    storage_paths = get_storage_paths_v2()
                    
                    # Get channel_id from video record
                    from core.database import Video
                    video_result = await session.execute(
                        select(Video).where(Video.video_id == video_id)
                    )
                    video = video_result.scalar_one_or_none()
                    
                    if video:
                        # Try to find transcript file using V2 storage discovery
                        transcript_files = list(storage_paths.get_transcript_dir(video.channel_id, video_id).glob('*.txt'))
                        if transcript_files:
                            # Use the first txt file found
                            return transcript_files[0].read_text(encoding='utf-8')
                
                return None
                
        except Exception as e:
            self.log_with_context(
                f"Failed to get transcript content: {str(e)}",
                level="ERROR",
                extra_context={"video_id": video_id}
            )
            return None
    
    async def _save_transcript_to_database(self, video_id: str, transcript_data: Dict[str, Any]) -> None:
        """
        Save transcript data to database after extraction.
        
        Args:
            video_id: Video ID
            transcript_data: Transcript data from TranscribeWorker
        """
        try:
            from core.database import Transcript, Video
            from sqlalchemy import select
            
            async with db_manager.get_session() as session:
                # Update video transcript status
                video_result = await session.execute(
                    select(Video).where(Video.video_id == video_id)
                )
                video = video_result.scalar_one_or_none()
                if video:
                    video.transcript_status = 'completed'
                
                # Check if transcript already exists
                existing_result = await session.execute(
                    select(Transcript).where(Transcript.video_id == video_id)
                )
                existing_transcript = existing_result.scalar_one_or_none()
                
                # Read content from files if not in data
                transcript_text = transcript_data.get('transcript', '')
                transcript_srt = transcript_data.get('transcript_srt', '')
                
                if not transcript_text and transcript_data.get('txt_path'):
                    try:
                        with open(transcript_data['txt_path'], 'r', encoding='utf-8') as f:
                            transcript_text = f.read()
                    except:
                        pass
                
                if not transcript_srt and transcript_data.get('srt_path'):
                    try:
                        with open(transcript_data['srt_path'], 'r', encoding='utf-8') as f:
                            transcript_srt = f.read()
                    except:
                        pass
                
                # Prepare transcript data with detected language
                detected_language = transcript_data.get('language', 'unknown')
                languages_found = transcript_data.get('languages_found', [detected_language])
                
                # Log the detected language for debugging
                self.log_with_context(
                    f"Saving transcript with detected language: {detected_language}",
                    extra_context={
                        "video_id": video_id,
                        "languages_found": languages_found,
                        "extraction_method": transcript_data.get('extraction_method')
                    }
                )
                
                if existing_transcript:
                    # Update existing transcript
                    existing_transcript.content_text = transcript_text
                    existing_transcript.content_srt = transcript_srt
                    existing_transcript.word_count = transcript_data.get('word_count', len(transcript_text.split()))
                    existing_transcript.language = detected_language  # Save detected language
                    existing_transcript.extraction_method = transcript_data.get('extraction_method', 'unknown')
                    existing_transcript.srt_path = transcript_data.get('srt_path')
                    existing_transcript.transcript_path = transcript_data.get('txt_path')
                    existing_transcript.audio_path = transcript_data.get('audio_path')
                else:
                    # Create new transcript
                    transcript = Transcript(
                        video_id=video_id,
                        content_text=transcript_text,
                        content_srt=transcript_srt,
                        word_count=transcript_data.get('word_count', len(transcript_text.split())),
                        language=detected_language,  # Save detected language
                        extraction_method=transcript_data.get('extraction_method', 'unknown'),
                        srt_path=transcript_data.get('srt_path'),
                        transcript_path=transcript_data.get('txt_path'),
                        audio_path=transcript_data.get('audio_path')
                    )
                    session.add(transcript)
                
                await session.commit()
                
                self.log_with_context(
                    f"Successfully saved transcript to database for video {video_id}",
                    extra_context={
                        "language": detected_language,
                        "word_count": transcript_data.get('word_count', 0)
                    }
                )
                
        except Exception as e:
            self.log_with_context(
                f"Failed to save transcript to database: {str(e)}",
                level="ERROR",
                extra_context={
                    "video_id": video_id,
                    "error": str(e)
                }
            )
    
    async def _get_generated_content(self, video_id: str) -> Optional[str]:
        """Get generated content for quality validation."""
        try:
            from core.database import GeneratedContent
            from sqlalchemy import select
            
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(GeneratedContent).where(GeneratedContent.transcript_id == video_id)
                    .order_by(GeneratedContent.created_at.desc())
                    .limit(1)
                )
                content = result.scalar_one_or_none()
                
                if content and content.content:
                    return content.content
                
                return None
                
        except Exception as e:
            self.log_with_context(
                f"Failed to get generated content: {str(e)}",
                level="ERROR", 
                extra_context={"video_id": video_id}
            )
            return None
    
    async def _get_worker_for_job_type(self, job_type: str) -> Optional[BaseWorker]:
        """
        Get the appropriate worker instance for a job type with comprehensive validation and health checking.
        
        Args:
            job_type: Type of job to process
            
        Returns:
            Worker instance or None if not found/healthy
        """
        # Issue #17: Worker coordination and communication problems
        for worker_name, registry in self.worker_registry.items():
            if job_type in registry.job_types and registry.enabled:
                
                # Issue #19: Check circuit breaker status
                circuit_breaker = self._get_circuit_breaker(worker_name)
                if circuit_breaker.is_open():
                    self.log_with_context(
                        f"Circuit breaker open for worker {worker_name} - skipping",
                        level="WARNING",
                        extra_context={"job_type": job_type}
                    )
                    continue
                
                # Issue #25: Worker health validation before use
                if not await self._validate_worker_health(worker_name, registry):
                    self.log_with_context(
                        f"Worker {worker_name} failed health check - attempting restart",
                        level="WARNING"
                    )
                    
                    # Attempt to restart unhealthy worker
                    if await self._restart_worker(worker_name, registry):
                        self.log_with_context(f"Successfully restarted worker {worker_name}")
                    else:
                        self.log_with_context(
                            f"Failed to restart worker {worker_name} - disabling",
                            level="ERROR"
                        )
                        registry.enabled = False
                        continue
                
                # Issue #18: Validate worker compatibility and methods
                if not registry.instance:
                    # Enhanced worker initialization with validation
                    try:
                        registry.instance = await self._initialize_worker_safely(worker_name, registry)
                        if not registry.instance:
                            continue
                    except Exception as e:
                        self.log_with_context(
                            f"Failed to initialize worker {worker_name}: {e}",
                            level="ERROR"
                        )
                        registry.enabled = False
                        continue
                
                return registry.instance
        
        self.log_with_context(
            f"No healthy worker found for job type: {job_type}",
            level="WARNING",
            extra_context={
                "available_workers": [
                    f"{name}({'enabled' if reg.enabled else 'disabled'}, health: {reg.health_status})"
                    for name, reg in self.worker_registry.items()
                ]
            }
        )
        return None
    
    def _initialize_worker_registry(self) -> None:
        """Initialize the worker registry with available workers."""
        # Register Enhanced MonitorWorker for channel checking jobs with prevention systems
        # Use environment variable to choose between monitor workers
        import os
        use_enhanced_monitor = os.getenv('USE_ENHANCED_MONITOR', 'true').lower() == 'true'
        
        monitor_worker_class = EnhancedMonitorWorker if use_enhanced_monitor else MonitorWorker
        
        self.worker_registry["monitor"] = WorkerRegistry(
            worker_class=monitor_worker_class,
            job_types={"check_channel", "check_all_channels"},
            enabled=True
        )
        
        # Register AudioDownloadWorker for audio download jobs  
        self.worker_registry["audio_downloader"] = WorkerRegistry(
            worker_class=AudioDownloadWorker,
            job_types={"download_audio"},
            enabled=True
        )
        
        # Register TranscribeWorker for transcript extraction jobs
        self.worker_registry["transcriber"] = WorkerRegistry(
            worker_class=TranscribeWorker,
            job_types={"extract_transcript", "download_transcript"},
            enabled=True
        )
        
        # Register SummarizerWorker for AI summary generation
        self.worker_registry["summarizer"] = WorkerRegistry(
            worker_class=SummarizerWorker,
            job_types={"summarize", "generate_summary"},  # Support both job type names
            enabled=True
        )
        
        # Register GeneratorWorker for content generation
        self.worker_registry["generator"] = WorkerRegistry(
            worker_class=GeneratorWorker,
            job_types={"generate_content", "generate_blog", "generate_social", 
                      "generate_newsletter", "generate_scripts"},
            enabled=True
        )
        
        # Register StorageWorker for file synchronization
        self.worker_registry["storage"] = WorkerRegistry(
            worker_class=StorageWorker,
            job_types={"store_file", "sync_storage", "backup_file"},
            enabled=True
        )
        
        # Register QualityWorker for quality validation
        self.worker_registry["quality"] = WorkerRegistry(
            worker_class=QualityWorker,
            job_types={"check_transcript_quality", "check_content_quality", "validate_quality"},
            enabled=True
        )
        
        # Register PublishWorker for content distribution
        self.worker_registry["publisher"] = WorkerRegistry(
            worker_class=PublishWorker,
            job_types={"publish_content", "distribute_content", "export_content"},
            enabled=True
        )
        
        self.log_with_context(
            f"Initialized worker registry with {len(self.worker_registry)} workers"
        )
    
    async def _initialize_workers(self) -> None:
        """Initialize worker instances that need async setup."""
        for worker_name, registry in self.worker_registry.items():
            if registry.enabled:
                try:
                    # Initialize the worker instance
                    registry.instance = registry.worker_class()
                    self.log_with_context(f"Initialized {worker_name} worker")
                except Exception as e:
                    self.log_with_context(
                        f"Failed to initialize {worker_name} worker: {e}",
                        level="ERROR"
                    )
                    registry.enabled = False
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check on orchestrator and workers."""
        try:
            # Check queue health
            stats = await self.job_queue.get_queue_stats()
            self.stats.queue_size = stats.pending_jobs
            self.stats.active_workers = len([r for r in self.worker_registry.values() if r.enabled])
            
            # Issue #25: Perform worker health checks
            current_time = time.time()
            if current_time - self._last_worker_health_check >= self._worker_health_check_interval:
                await self._perform_worker_health_checks()
                self._last_worker_health_check = current_time
            
            # Log health status with enhanced worker information
            worker_health_summary = {
                name: {
                    'enabled': reg.enabled,
                    'health': reg.health_status,
                    'failures': reg.consecutive_failures,
                    'restarts': reg.restart_count
                }
                for name, reg in self.worker_registry.items()
            }
            
            self.log_with_context(
                "Health check completed",
                extra_context={
                    "queue_pending": stats.pending_jobs,
                    "queue_processing": stats.processing_jobs,
                    "active_workers": self.stats.active_workers,
                    "uptime": f"{self._calculate_uptime():.0f}s",
                    "worker_health": worker_health_summary
                }
            )
            
        except Exception as e:
            self.log_with_context(f"Health check failed: {e}", level="WARNING")
    
    async def _perform_worker_health_checks(self) -> None:
        """Perform health checks on all registered workers."""
        for worker_name, registry in self.worker_registry.items():
            if registry.enabled:
                try:
                    is_healthy = await self._validate_worker_health(worker_name, registry)
                    if not is_healthy and registry.health_status == "failed":
                        # Attempt to restart failed workers
                        if registry.restart_count < registry.max_restarts:
                            self.log_with_context(
                                f"Attempting to restart failed worker: {worker_name}"
                            )
                            await self._restart_worker(worker_name, registry)
                        else:
                            self.log_with_context(
                                f"Worker {worker_name} permanently disabled - exceeded max restarts",
                                level="ERROR"
                            )
                            registry.enabled = False
                            
                except Exception as e:
                    self.log_with_context(
                        f"Error during health check for worker {worker_name}: {e}",
                        level="ERROR"
                    )
                    registry.health_status = "failed"
    
    async def _get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        self.stats.uptime_seconds = self._calculate_uptime()
        self._update_jobs_per_minute()
        
        try:
            queue_stats = await self.job_queue.get_queue_stats()
            self.stats.queue_size = queue_stats.pending_jobs
        except Exception:
            pass
        
        return self.stats
    
    def _calculate_uptime(self) -> float:
        """Calculate orchestrator uptime in seconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
    
    def _update_jobs_per_minute(self) -> None:
        """Update jobs per minute metric."""
        uptime = self._calculate_uptime()
        if uptime > 0:
            self.stats.jobs_per_minute = (self.stats.total_jobs_processed / uptime) * 60
    
    def _get_recovery_delay(self, error_category: str) -> float:
        """Get recovery delay based on error category."""
        delays = {
            "database": 10.0,
            "queue": 5.0,
            "resource_exhaustion": 30.0,
            "network": 15.0,
            "unknown": self.retry_delay
        }
        return delays.get(error_category, self.retry_delay)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            def signal_handler(signum, frame):
                self.log_with_context(f"Received signal {signum}, initiating shutdown")
                self.stop_requested = True
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
        except Exception as e:
            self.log_with_context(f"Could not setup signal handlers: {e}", level="WARNING")
    
    async def stop(self) -> None:
        """Public method to request orchestrator stop."""
        self.log_with_context("Stop requested")
        self.stop_requested = True
    
    async def pause(self) -> None:
        """Pause the orchestrator processing."""
        self.status = OrchestratorStatus.PAUSED
        self.log_with_context("Orchestrator paused")
    
    async def resume(self) -> None:
        """Resume the orchestrator processing."""
        if self.status == OrchestratorStatus.PAUSED:
            self.status = OrchestratorStatus.RUNNING
            self.log_with_context("Orchestrator resumed")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status and statistics."""
        return {
            "status": self.status.value,
            "worker_id": self.worker_id,
            "uptime_seconds": self._calculate_uptime(),
            "stats": self.stats.__dict__,
            "worker_registry": {
                name: {
                    "enabled": reg.enabled,
                    "job_types": list(reg.job_types),
                    "initialized": reg.instance is not None
                }
                for name, reg in self.worker_registry.items()
            }
        }


# Convenience functions for direct usage
async def start_orchestrator(
    continuous_mode: bool = True,
    max_jobs: Optional[int] = None,
    worker_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start the orchestrator with specified configuration.
    
    Args:
        continuous_mode: Run in continuous mode
        max_jobs: Maximum jobs to process
        worker_id: Optional worker ID
        
    Returns:
        Orchestrator execution result
    """
    orchestrator = OrchestratorWorker(worker_id=worker_id)
    input_data = {
        "continuous_mode": continuous_mode,
        "max_jobs": max_jobs
    }
    
    if hasattr(orchestrator, 'run_async'):
        return await orchestrator.run_async(input_data)
    else:
        # Fallback for sync execution
        return orchestrator.run(input_data)


def create_orchestrator(
    max_concurrent_jobs: int = 3,
    polling_interval: float = 5.0,
    worker_id: Optional[str] = None
) -> OrchestratorWorker:
    """
    Create a new orchestrator instance with configuration.
    
    Args:
        max_concurrent_jobs: Maximum concurrent job processing
        polling_interval: Queue polling interval
        worker_id: Optional worker ID
        
    Returns:
        Configured OrchestratorWorker instance
    """
    return OrchestratorWorker(
        max_concurrent_jobs=max_concurrent_jobs,
        polling_interval=polling_interval,
        worker_id=worker_id
    )


if __name__ == "__main__":
    # Example usage for testing
    import asyncio
    
    async def main():
        orchestrator = OrchestratorWorker()
        
        # Start orchestrator in continuous mode
        result = await orchestrator.execute({
            "continuous_mode": True,
            "max_jobs": 10
        })
        
        print(f"Orchestrator result: {result}")
    
    # Run the example
    asyncio.run(main())