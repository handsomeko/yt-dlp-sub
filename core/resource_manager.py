"""
Dynamic Resource Manager for worker scaling.

Monitors system resources (CPU, memory) and dynamically adjusts the number of
concurrent workers to maximize throughput while preventing system overload.

This is especially important when running locally on development machines.
"""

import asyncio
import logging
import multiprocessing
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not installed - dynamic scaling disabled")


class ScalingAction(Enum):
    """Actions the resource manager can take."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ResourceMetrics:
    """Current system resource metrics."""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float
    active_workers: int
    pending_jobs: int
    timestamp: float


@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling behavior."""
    # Resource thresholds
    cpu_scale_up_threshold: float = 50.0    # Scale up if CPU < this
    cpu_scale_down_threshold: float = 80.0  # Scale down if CPU > this
    memory_scale_up_threshold: float = 60.0  # Scale up if memory < this
    memory_scale_down_threshold: float = 85.0  # Scale down if memory > this
    memory_emergency_threshold: float = 95.0  # Emergency stop if memory > this
    
    # Worker limits
    min_workers: int = 1
    max_workers: int = multiprocessing.cpu_count() * 2  # Good for I/O bound tasks
    worker_step_size: int = 1  # How many workers to add/remove at once
    
    # Timing
    metrics_interval: float = 5.0  # How often to check metrics (seconds)
    scale_cooldown: float = 30.0  # Wait time after scaling (seconds)
    emergency_cooldown: float = 60.0  # Wait time after emergency stop
    
    # Smoothing
    metrics_window_size: int = 3  # Average over N samples to smooth spikes
    scale_up_consecutive: int = 2  # Need N consecutive scale-up signals
    scale_down_consecutive: int = 1  # Need N consecutive scale-down signals


class ResourceManager:
    """
    Manages system resources and provides scaling recommendations.
    
    This class monitors CPU and memory usage, maintains a history of metrics,
    and provides intelligent scaling decisions to prevent system overload
    while maximizing throughput.
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        """
        Initialize the resource manager.
        
        Args:
            config: Scaling configuration (uses defaults if not provided)
        """
        self.config = config or ScalingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Metrics history for smoothing
        self.metrics_history: List[ResourceMetrics] = []
        self.last_scale_time: float = 0
        self.last_scale_action: Optional[ScalingAction] = None
        
        # Consecutive action counters
        self.scale_up_count: int = 0
        self.scale_down_count: int = 0
        
        # Emergency state
        self.emergency_mode: bool = False
        self.emergency_start_time: float = 0
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - resource monitoring disabled")
    
    def get_current_metrics(self, active_workers: int, pending_jobs: int) -> ResourceMetrics:
        """
        Get current system resource metrics.
        
        Args:
            active_workers: Number of currently active workers
            pending_jobs: Number of jobs waiting in queue
            
        Returns:
            Current resource metrics
        """
        if not PSUTIL_AVAILABLE:
            # Return safe defaults if psutil not available
            return ResourceMetrics(
                cpu_percent=50.0,
                memory_percent=50.0,
                available_memory_gb=4.0,
                active_workers=active_workers,
                pending_jobs=pending_jobs,
                timestamp=time.time()
            )
        
        # Get CPU usage (average over 1 second)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory_gb = memory.available / (1024 ** 3)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            available_memory_gb=available_memory_gb,
            active_workers=active_workers,
            pending_jobs=pending_jobs,
            timestamp=time.time()
        )
    
    def _get_smoothed_metrics(self) -> Optional[ResourceMetrics]:
        """
        Get averaged metrics over the configured window.
        
        Returns:
            Smoothed metrics or None if not enough history
        """
        if len(self.metrics_history) < self.config.metrics_window_size:
            return None
        
        # Get recent metrics
        recent = self.metrics_history[-self.config.metrics_window_size:]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        avg_memory_gb = sum(m.available_memory_gb for m in recent) / len(recent)
        
        # Use latest worker and job counts
        latest = self.metrics_history[-1]
        
        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            available_memory_gb=avg_memory_gb,
            active_workers=latest.active_workers,
            pending_jobs=latest.pending_jobs,
            timestamp=latest.timestamp
        )
    
    def _check_emergency(self, metrics: ResourceMetrics) -> bool:
        """
        Check if system is in emergency state (about to crash).
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            True if emergency action needed
        """
        # Check memory emergency threshold
        if metrics.memory_percent >= self.config.memory_emergency_threshold:
            return True
        
        # Check if available memory is critically low (< 500MB)
        if metrics.available_memory_gb < 0.5:
            return True
        
        # Check CPU (sustained 100% might indicate hanging)
        if metrics.cpu_percent >= 99.0:
            # Check if this is sustained
            if len(self.metrics_history) >= 3:
                recent_cpu = [m.cpu_percent for m in self.metrics_history[-3:]]
                if all(cpu >= 99.0 for cpu in recent_cpu):
                    return True
        
        return False
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're still in cooldown period from last scaling."""
        if self.emergency_mode:
            # Check emergency cooldown
            if time.time() - self.emergency_start_time < self.config.emergency_cooldown:
                return True
            else:
                # Exit emergency mode
                self.emergency_mode = False
                self.logger.info("Exiting emergency mode")
        
        # Check normal cooldown
        if self.last_scale_time > 0:
            elapsed = time.time() - self.last_scale_time
            if elapsed < self.config.scale_cooldown:
                return True
        
        return False
    
    def get_scaling_decision(
        self, 
        active_workers: int, 
        pending_jobs: int
    ) -> Tuple[ScalingAction, int]:
        """
        Determine if workers should be scaled up, down, or maintained.
        
        Args:
            active_workers: Number of currently active workers
            pending_jobs: Number of jobs waiting in queue
            
        Returns:
            Tuple of (action, recommended_worker_count)
        """
        # Get current metrics
        current = self.get_current_metrics(active_workers, pending_jobs)
        
        # Add to history
        self.metrics_history.append(current)
        
        # Limit history size
        max_history = self.config.metrics_window_size * 3
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
        
        # Get smoothed metrics
        metrics = self._get_smoothed_metrics() or current
        
        # Log current state
        self.logger.debug(
            f"Resources: CPU={metrics.cpu_percent:.1f}%, "
            f"Memory={metrics.memory_percent:.1f}%, "
            f"Workers={active_workers}, Queue={pending_jobs}"
        )
        
        # Check for emergency
        if self._check_emergency(metrics):
            if not self.emergency_mode:
                self.logger.error(
                    f"EMERGENCY: Memory={metrics.memory_percent:.1f}%, "
                    f"Available={metrics.available_memory_gb:.1f}GB"
                )
                self.emergency_mode = True
                self.emergency_start_time = time.time()
            return (ScalingAction.EMERGENCY_STOP, self.config.min_workers)
        
        # Check cooldown
        if self._is_in_cooldown():
            return (ScalingAction.MAINTAIN, active_workers)
        
        # Determine action based on thresholds
        action = self._determine_action(metrics)
        
        # Calculate new worker count
        new_workers = self._calculate_worker_count(action, active_workers, pending_jobs)
        
        # Record scaling decision if changing
        if new_workers != active_workers:
            self.last_scale_time = time.time()
            self.last_scale_action = action
            self.logger.info(
                f"Scaling {action.value}: {active_workers} → {new_workers} workers "
                f"(CPU={metrics.cpu_percent:.1f}%, Memory={metrics.memory_percent:.1f}%)"
            )
        
        return (action, new_workers)
    
    def _determine_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """
        Determine scaling action based on metrics.
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            Recommended scaling action
        """
        # Check if we should scale down (high resource usage)
        if (metrics.cpu_percent > self.config.cpu_scale_down_threshold or
            metrics.memory_percent > self.config.memory_scale_down_threshold):
            
            self.scale_down_count += 1
            self.scale_up_count = 0
            
            if self.scale_down_count >= self.config.scale_down_consecutive:
                return ScalingAction.SCALE_DOWN
        
        # Check if we should scale up (low resource usage with pending work)
        elif (metrics.cpu_percent < self.config.cpu_scale_up_threshold and
              metrics.memory_percent < self.config.memory_scale_up_threshold and
              metrics.pending_jobs > 0):
            
            self.scale_up_count += 1
            self.scale_down_count = 0
            
            if self.scale_up_count >= self.config.scale_up_consecutive:
                return ScalingAction.SCALE_UP
        
        # Otherwise maintain current level
        else:
            self.scale_up_count = 0
            self.scale_down_count = 0
        
        return ScalingAction.MAINTAIN
    
    def _calculate_worker_count(
        self, 
        action: ScalingAction, 
        current_workers: int,
        pending_jobs: int
    ) -> int:
        """
        Calculate the new worker count based on action.
        
        Args:
            action: Scaling action to take
            current_workers: Current number of workers
            pending_jobs: Number of pending jobs
            
        Returns:
            Recommended number of workers
        """
        if action == ScalingAction.SCALE_UP:
            # Scale up by step size, but don't exceed max or pending jobs
            new_workers = current_workers + self.config.worker_step_size
            new_workers = min(new_workers, self.config.max_workers)
            new_workers = min(new_workers, current_workers + pending_jobs)
            return new_workers
        
        elif action == ScalingAction.SCALE_DOWN:
            # Scale down by step size, but keep minimum
            new_workers = current_workers - self.config.worker_step_size
            new_workers = max(new_workers, self.config.min_workers)
            return new_workers
        
        elif action == ScalingAction.EMERGENCY_STOP:
            # Drop to minimum immediately
            return self.config.min_workers
        
        else:  # MAINTAIN
            return current_workers
    
    async def monitor_resources(self, callback=None):
        """
        Continuously monitor resources and call callback with decisions.
        
        Args:
            callback: Async function to call with (action, worker_count)
        """
        while True:
            try:
                # Get current worker and job counts (would be passed in real implementation)
                # For now, just monitor
                metrics = self.get_current_metrics(1, 0)
                
                self.logger.debug(
                    f"Resource Monitor: CPU={metrics.cpu_percent:.1f}%, "
                    f"Memory={metrics.memory_percent:.1f}%"
                )
                
                if callback:
                    await callback(metrics)
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.config.metrics_interval)
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current resource manager status.
        
        Returns:
            Status dictionary with current metrics and state
        """
        latest = self.metrics_history[-1] if self.metrics_history else None
        smoothed = self._get_smoothed_metrics()
        
        return {
            "enabled": PSUTIL_AVAILABLE,
            "emergency_mode": self.emergency_mode,
            "in_cooldown": self._is_in_cooldown(),
            "last_scale_action": self.last_scale_action.value if self.last_scale_action else None,
            "last_scale_time": self.last_scale_time,
            "current_metrics": {
                "cpu_percent": latest.cpu_percent if latest else 0,
                "memory_percent": latest.memory_percent if latest else 0,
                "available_memory_gb": latest.available_memory_gb if latest else 0,
            } if latest else None,
            "smoothed_metrics": {
                "cpu_percent": smoothed.cpu_percent if smoothed else 0,
                "memory_percent": smoothed.memory_percent if smoothed else 0,
            } if smoothed else None,
            "thresholds": {
                "cpu_scale_up": self.config.cpu_scale_up_threshold,
                "cpu_scale_down": self.config.cpu_scale_down_threshold,
                "memory_scale_up": self.config.memory_scale_up_threshold,
                "memory_scale_down": self.config.memory_scale_down_threshold,
                "memory_emergency": self.config.memory_emergency_threshold,
            }
        }