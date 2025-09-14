"""
Whisper Timeout Manager

This module provides timeout management and process monitoring for Whisper transcription:
- Timeout wrappers with graceful termination
- Resource monitoring (memory/CPU) 
- Process management and kill switches
- Concurrent job limiting
- Fallback strategy coordination
"""

import asyncio
import threading
import time
import psutil
import signal
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessMonitoringResult:
    """Results from process monitoring during execution."""
    max_memory_mb: float
    max_cpu_percent: float
    execution_time_seconds: float
    killed_due_to_resources: bool
    timeout_occurred: bool
    success: bool
    error: Optional[str] = None


class WhisperTimeoutManager:
    """Manages timeouts and resource monitoring for Whisper operations."""
    
    def __init__(self, max_concurrent_jobs: int = 2, memory_limit_mb: int = 8192):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.memory_limit_mb = memory_limit_mb
        self.active_jobs = {}  # Track active jobs
        self.job_semaphore = threading.Semaphore(max_concurrent_jobs)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track system resources
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.system_cpu_count = psutil.cpu_count()
    
    def execute_with_timeout(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout_seconds: int = 300,
        job_id: str = "unknown",
        monitor_resources: bool = True
    ) -> Tuple[Any, ProcessMonitoringResult]:
        """
        Execute a function with timeout and resource monitoring.
        
        Args:
            func: Function to execute
            args: Function positional arguments
            kwargs: Function keyword arguments
            timeout_seconds: Timeout in seconds
            job_id: Identifier for this job
            monitor_resources: Whether to monitor resource usage
            
        Returns:
            Tuple of (result, monitoring_result)
        """
        if kwargs is None:
            kwargs = {}
        
        # Wait for semaphore (limit concurrent jobs)
        if not self.job_semaphore.acquire(timeout=30):  # 30 second wait max
            raise RuntimeError(f"Could not acquire job semaphore - too many concurrent jobs")
        
        start_time = time.time()
        monitoring_result = ProcessMonitoringResult(
            max_memory_mb=0,
            max_cpu_percent=0,
            execution_time_seconds=0,
            killed_due_to_resources=False,
            timeout_occurred=False,
            success=False
        )
        
        try:
            # Register this job
            self.active_jobs[job_id] = {
                'start_time': start_time,
                'timeout': timeout_seconds,
                'pid': os.getpid()
            }
            
            if monitor_resources:
                # Execute with resource monitoring
                result, monitoring_result = self._execute_with_monitoring(
                    func, args, kwargs, timeout_seconds, job_id
                )
            else:
                # Execute with basic timeout only
                result = self._execute_with_basic_timeout(
                    func, args, kwargs, timeout_seconds
                )
                monitoring_result.success = True
            
            monitoring_result.execution_time_seconds = time.time() - start_time
            return result, monitoring_result
            
        except FutureTimeoutError:
            monitoring_result.timeout_occurred = True
            monitoring_result.execution_time_seconds = time.time() - start_time
            monitoring_result.error = f"Operation timed out after {timeout_seconds} seconds"
            self.logger.error(f"Job {job_id} timed out after {timeout_seconds}s")
            raise TimeoutError(f"Whisper transcription timed out after {timeout_seconds} seconds")
            
        except Exception as e:
            monitoring_result.execution_time_seconds = time.time() - start_time
            monitoring_result.error = str(e)
            self.logger.error(f"Job {job_id} failed: {e}")
            raise
            
        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            self.job_semaphore.release()
    
    def _execute_with_monitoring(
        self, 
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout_seconds: int,
        job_id: str
    ) -> Tuple[Any, ProcessMonitoringResult]:
        """Execute function with detailed resource monitoring."""
        
        monitoring_result = ProcessMonitoringResult(
            max_memory_mb=0,
            max_cpu_percent=0,
            execution_time_seconds=0,
            killed_due_to_resources=False,
            timeout_occurred=False,
            success=False
        )
        
        # Start resource monitoring in separate thread
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_resources():
            """Monitor resource usage in background thread."""
            process = psutil.Process()
            while monitoring_active.is_set():
                try:
                    # Get memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    monitoring_result.max_memory_mb = max(monitoring_result.max_memory_mb, memory_mb)
                    
                    # Get CPU usage
                    cpu_percent = process.cpu_percent()
                    monitoring_result.max_cpu_percent = max(monitoring_result.max_cpu_percent, cpu_percent)
                    
                    # Check if we need to kill due to resource limits
                    if memory_mb > self.memory_limit_mb:
                        self.logger.error(f"Job {job_id} exceeded memory limit: {memory_mb:.1f}MB > {self.memory_limit_mb}MB")
                        monitoring_result.killed_due_to_resources = True
                        monitoring_result.error = f"Memory limit exceeded: {memory_mb:.1f}MB"
                        # Kill the process
                        try:
                            process.terminate()
                            time.sleep(2)
                            if process.is_running():
                                process.kill()
                        except:
                            pass
                        break
                    
                    time.sleep(1)  # Check every second
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process ended or we can't access it
                    break
                except Exception as e:
                    self.logger.warning(f"Resource monitoring error: {e}")
                    break
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        try:
            # Execute the function with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=timeout_seconds)
            
            monitoring_result.success = True
            return result, monitoring_result
            
        except FutureTimeoutError:
            monitoring_result.timeout_occurred = True
            raise
        finally:
            # Stop monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=1)
    
    def _execute_with_basic_timeout(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout_seconds: int
    ) -> Any:
        """Execute function with basic timeout only."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout_seconds)
    
    def calculate_dynamic_timeout(
        self,
        duration_seconds: float,
        base_timeout: int = 300,
        timeout_per_minute: float = 2.0,
        min_timeout: int = 60,
        max_timeout: int = 3600
    ) -> int:
        """
        Calculate dynamic timeout based on audio duration.
        
        Args:
            duration_seconds: Audio duration in seconds
            base_timeout: Base timeout regardless of duration
            timeout_per_minute: Additional timeout per minute of audio
            min_timeout: Minimum timeout value
            max_timeout: Maximum timeout value
            
        Returns:
            Recommended timeout in seconds
        """
        duration_minutes = duration_seconds / 60
        calculated_timeout = base_timeout + (duration_minutes * timeout_per_minute)
        
        # Apply min/max bounds
        timeout = max(min_timeout, min(max_timeout, int(calculated_timeout)))
        
        self.logger.info(
            f"Calculated timeout: {timeout}s for {duration_seconds:.1f}s audio "
            f"({duration_minutes:.1f} minutes)"
        )
        
        return timeout
    
    def get_active_jobs_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active jobs."""
        current_time = time.time()
        status = {}
        
        for job_id, job_info in self.active_jobs.items():
            elapsed = current_time - job_info['start_time']
            remaining = max(0, job_info['timeout'] - elapsed)
            
            status[job_id] = {
                'elapsed_seconds': elapsed,
                'remaining_seconds': remaining,
                'timeout_seconds': job_info['timeout'],
                'pid': job_info['pid'],
                'progress_percent': min(100, (elapsed / job_info['timeout']) * 100)
            }
        
        return status
    
    def kill_job(self, job_id: str) -> bool:
        """Kill a specific job by ID."""
        if job_id not in self.active_jobs:
            return False
        
        try:
            pid = self.active_jobs[job_id]['pid']
            process = psutil.Process(pid)
            
            # Try graceful termination first
            process.terminate()
            time.sleep(2)
            
            # Force kill if still running
            if process.is_running():
                process.kill()
            
            self.logger.info(f"Killed job {job_id} (PID {pid})")
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Could not kill job {job_id}: {e}")
            return False
    
    def kill_all_jobs(self) -> int:
        """Kill all active jobs. Returns number of jobs killed."""
        killed_count = 0
        
        for job_id in list(self.active_jobs.keys()):
            if self.kill_job(job_id):
                killed_count += 1
        
        return killed_count
    
    def get_system_resource_status(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent_used': memory.percent,
                'cpu_percent_used': cpu_percent,
                'cpu_count': self.system_cpu_count,
                'active_whisper_jobs': len(self.active_jobs),
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'memory_limit_per_job_mb': self.memory_limit_mb
            }
        except Exception as e:
            return {'error': str(e)}
    
    def can_start_new_job(self) -> Tuple[bool, Optional[str]]:
        """Check if system can handle a new Whisper job."""
        try:
            # Check job limit
            if len(self.active_jobs) >= self.max_concurrent_jobs:
                return False, f"Maximum concurrent jobs reached ({self.max_concurrent_jobs})"
            
            # Check system memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            required_gb = self.memory_limit_mb / 1024
            
            if available_gb < required_gb:
                return False, f"Insufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
            
            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False, f"High CPU usage: {cpu_percent}%"
            
            return True, None
            
        except Exception as e:
            return False, f"Resource check failed: {e}"
    
    def get_recommended_model_for_resources(self) -> str:
        """Recommend Whisper model based on available system resources."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 8:
                return "large"
            elif available_gb >= 4:
                return "medium"
            elif available_gb >= 2:
                return "small" 
            elif available_gb >= 1:
                return "base"
            else:
                return "tiny"
                
        except Exception:
            return "base"  # Safe default