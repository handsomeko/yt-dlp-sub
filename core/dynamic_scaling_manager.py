"""
Dynamic Scaling Manager - TRUE Dynamic Scaling Based on System Resources

This module implements genuine dynamic scaling that adjusts processing capacity 
based on real-time system metrics:

- CPU usage and load average
- Memory consumption and availability  
- Network conditions and error rates
- Processing queue depth
- Historical performance metrics

Unlike static resource management, this system continuously adapts
processing parameters to optimize performance and prevent overload.
"""

import psutil
import time
import threading
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ScalingState(Enum):
    """Current scaling state of the system."""
    IDLE = "idle"                    # Low load, can scale up
    NORMAL = "normal"                # Balanced load  
    HIGH_LOAD = "high_load"         # High load, maintain current
    OVERLOADED = "overloaded"       # Critical load, scale down
    CRITICAL = "critical"           # Emergency, minimal processing


class ResourceMetric(Enum):
    """System resource metrics for scaling decisions."""
    CPU_PERCENT = "cpu_percent"
    MEMORY_PERCENT = "memory_percent"
    LOAD_AVERAGE = "load_average"
    DISK_IO_WAIT = "disk_io_wait"
    NETWORK_ERROR_RATE = "network_error_rate"
    PROCESSING_QUEUE_DEPTH = "queue_depth"


@dataclass
class ScalingThresholds:
    """Dynamic thresholds for scaling decisions."""
    # CPU thresholds (percentage)
    cpu_idle: float = 30.0           # Below this: can scale up
    cpu_normal: float = 60.0         # Below this: normal operation
    cpu_high: float = 80.0           # Above this: scale down
    cpu_critical: float = 95.0       # Above this: emergency mode
    
    # Memory thresholds (percentage)
    memory_idle: float = 40.0
    memory_normal: float = 70.0  
    memory_high: float = 85.0
    memory_critical: float = 95.0
    
    # Load average thresholds (relative to CPU count)
    load_normal: float = 0.7         # 70% of CPU cores
    load_high: float = 1.0           # 100% of CPU cores
    load_critical: float = 1.5       # 150% of CPU cores
    
    # Network error rate thresholds (errors per minute)
    network_normal: float = 5.0
    network_high: float = 15.0
    network_critical: float = 30.0
    
    # Queue depth thresholds
    queue_normal: int = 10
    queue_high: int = 50
    queue_critical: int = 100


@dataclass  
class ScalingParams:
    """Dynamic scaling parameters that adjust based on system state."""
    max_concurrent_downloads: int = 3
    max_concurrent_transcriptions: int = 2
    download_batch_size: int = 5
    processing_delay_seconds: float = 2.0
    retry_backoff_multiplier: float = 2.0
    health_check_interval: float = 60.0
    
    # Scaling bounds
    min_concurrent_downloads: int = 1
    max_concurrent_downloads_limit: int = 8
    min_concurrent_transcriptions: int = 1
    max_concurrent_transcriptions_limit: int = 4


@dataclass
class SystemMetrics:
    """Current system resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    load_average: float
    disk_io_wait: float
    network_error_rate: float
    queue_depth: int
    active_downloads: int = 0
    active_transcriptions: int = 0


class DynamicScalingManager:
    """
    TRUE Dynamic Scaling Manager
    
    Continuously monitors system resources and dynamically adjusts
    processing capacity to optimize performance while preventing overload.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Scaling configuration
        self.thresholds = ScalingThresholds()
        self.current_params = ScalingParams()
        
        # System monitoring
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
        # Metrics history for trend analysis
        self.metrics_history: deque = deque(maxlen=100)  # Last 100 measurements
        self.scaling_history: deque = deque(maxlen=50)   # Last 50 scaling decisions
        
        # Current state
        self.current_state = ScalingState.NORMAL
        self.last_scale_time = datetime.now()
        self.scaling_cooldown = 30.0  # Seconds between scaling decisions
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'avg_download_time': deque(maxlen=20),
            'avg_transcription_time': deque(maxlen=20), 
            'error_rate': deque(maxlen=20),
            'throughput': deque(maxlen=20)
        }
        
        self.logger.info("Dynamic Scaling Manager initialized")
        self.logger.info(f"System: {self.cpu_count} CPU cores, {self.total_memory // (1024**3):.1f}GB RAM")
    
    def start_monitoring(self):
        """Start continuous system monitoring and dynamic scaling."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Dynamic scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Dynamic scaling monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_metrics()
                
                # Analyze and make scaling decision
                new_state = self._analyze_scaling_decision(metrics)
                
                # Apply scaling if needed
                if new_state != self.current_state:
                    self._apply_scaling(new_state, metrics)
                
                # Store metrics for history
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Sleep until next check
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30.0)  # Longer sleep on error
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Load average (1-minute)
        try:
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else cpu_percent / 100.0 * self.cpu_count
        except (OSError, AttributeError):
            load_avg = cpu_percent / 100.0 * self.cpu_count
        
        # Disk I/O wait (approximation)
        try:
            disk_io = psutil.disk_io_counters()
            disk_io_wait = 0.0  # Simplified - would need more complex calculation
        except:
            disk_io_wait = 0.0
        
        # Network error rate (from recent history)
        network_error_rate = self._calculate_network_error_rate()
        
        # Queue depth (would be injected from processing system)
        queue_depth = getattr(self, '_current_queue_depth', 0)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent, 
            load_average=load_avg,
            disk_io_wait=disk_io_wait,
            network_error_rate=network_error_rate,
            queue_depth=queue_depth
        )
    
    def _analyze_scaling_decision(self, metrics: SystemMetrics) -> ScalingState:
        """Analyze metrics and determine optimal scaling state."""
        
        # Score each resource dimension
        cpu_score = self._score_cpu_usage(metrics.cpu_percent, metrics.load_average)
        memory_score = self._score_memory_usage(metrics.memory_percent)
        network_score = self._score_network_conditions(metrics.network_error_rate)
        queue_score = self._score_queue_depth(metrics.queue_depth)
        
        # Trend analysis from recent history
        trend_score = self._analyze_trends()
        
        # Composite score (weighted)
        composite_score = (
            cpu_score * 0.35 +
            memory_score * 0.30 +
            network_score * 0.20 +
            queue_score * 0.10 +
            trend_score * 0.05
        )
        
        # Determine scaling state
        if composite_score >= 0.9:
            return ScalingState.CRITICAL
        elif composite_score >= 0.75:
            return ScalingState.OVERLOADED
        elif composite_score >= 0.55:
            return ScalingState.HIGH_LOAD
        elif composite_score >= 0.25:
            return ScalingState.NORMAL
        else:
            return ScalingState.IDLE
    
    def _score_cpu_usage(self, cpu_percent: float, load_avg: float) -> float:
        """Score CPU usage (0.0 = idle, 1.0 = critical)."""
        # CPU percentage score
        if cpu_percent <= self.thresholds.cpu_idle:
            cpu_score = 0.0
        elif cpu_percent <= self.thresholds.cpu_normal:
            cpu_score = (cpu_percent - self.thresholds.cpu_idle) / (self.thresholds.cpu_normal - self.thresholds.cpu_idle) * 0.5
        elif cpu_percent <= self.thresholds.cpu_high:
            cpu_score = 0.5 + (cpu_percent - self.thresholds.cpu_normal) / (self.thresholds.cpu_high - self.thresholds.cpu_normal) * 0.25
        else:
            cpu_score = 0.75 + min((cpu_percent - self.thresholds.cpu_high) / (self.thresholds.cpu_critical - self.thresholds.cpu_high) * 0.25, 0.25)
        
        # Load average score
        load_relative = load_avg / self.cpu_count
        if load_relative <= self.thresholds.load_normal:
            load_score = 0.0
        elif load_relative <= self.thresholds.load_high:
            load_score = (load_relative - self.thresholds.load_normal) / (self.thresholds.load_high - self.thresholds.load_normal) * 0.5
        else:
            load_score = 0.5 + min((load_relative - self.thresholds.load_high) / (self.thresholds.load_critical - self.thresholds.load_high) * 0.5, 0.5)
        
        # Return the higher of the two scores (more conservative)
        return max(cpu_score, load_score)
    
    def _score_memory_usage(self, memory_percent: float) -> float:
        """Score memory usage (0.0 = abundant, 1.0 = critical)."""
        if memory_percent <= self.thresholds.memory_idle:
            return 0.0
        elif memory_percent <= self.thresholds.memory_normal:
            return (memory_percent - self.thresholds.memory_idle) / (self.thresholds.memory_normal - self.thresholds.memory_idle) * 0.5
        elif memory_percent <= self.thresholds.memory_high:
            return 0.5 + (memory_percent - self.thresholds.memory_normal) / (self.thresholds.memory_high - self.thresholds.memory_normal) * 0.25
        else:
            return 0.75 + min((memory_percent - self.thresholds.memory_high) / (self.thresholds.memory_critical - self.thresholds.memory_high) * 0.25, 0.25)
    
    def _score_network_conditions(self, error_rate: float) -> float:
        """Score network conditions (0.0 = excellent, 1.0 = poor)."""
        if error_rate <= self.thresholds.network_normal:
            return 0.0
        elif error_rate <= self.thresholds.network_high:
            return (error_rate - self.thresholds.network_normal) / (self.thresholds.network_high - self.thresholds.network_normal) * 0.5
        else:
            return 0.5 + min((error_rate - self.thresholds.network_high) / (self.thresholds.network_critical - self.thresholds.network_high) * 0.5, 0.5)
    
    def _score_queue_depth(self, queue_depth: int) -> float:
        """Score processing queue depth (0.0 = empty, 1.0 = overflowing)."""
        if queue_depth <= self.thresholds.queue_normal:
            return 0.0
        elif queue_depth <= self.thresholds.queue_high:
            return (queue_depth - self.thresholds.queue_normal) / (self.thresholds.queue_high - self.thresholds.queue_normal) * 0.5
        else:
            return 0.5 + min((queue_depth - self.thresholds.queue_high) / (self.thresholds.queue_critical - self.thresholds.queue_high) * 0.5, 0.5)
    
    def _analyze_trends(self) -> float:
        """Analyze trends from recent metrics (0.0 = improving, 1.0 = degrading)."""
        if len(self.metrics_history) < 5:
            return 0.5  # Neutral when insufficient data
        
        recent_metrics = list(self.metrics_history)[-5:]
        
        # Calculate trends for key metrics
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        error_trend = self._calculate_trend([m.network_error_rate for m in recent_metrics])
        
        # Combine trends (positive trend = worsening)
        avg_trend = (cpu_trend + memory_trend + error_trend) / 3
        
        # Convert to 0-1 score (0 = improving, 1 = degrading)
        return max(0.0, min(1.0, 0.5 + avg_trend * 0.5))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend for a series of values (-1 = decreasing, +1 = increasing)."""
        if len(values) < 2:
            return 0.0
            
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope using least squares
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize slope to -1 to +1 range
        max_possible_slope = (max(values) - min(values)) / (n - 1) if n > 1 else 0
        if max_possible_slope == 0:
            return 0.0
            
        return max(-1.0, min(1.0, slope / max_possible_slope))
    
    def _apply_scaling(self, new_state: ScalingState, metrics: SystemMetrics):
        """Apply scaling changes based on new state."""
        # Check cooldown period
        time_since_last_scale = (datetime.now() - self.last_scale_time).total_seconds()
        if time_since_last_scale < self.scaling_cooldown:
            self.logger.debug(f"Scaling cooldown active ({time_since_last_scale:.1f}s < {self.scaling_cooldown}s)")
            return
        
        old_state = self.current_state
        self.current_state = new_state
        self.last_scale_time = datetime.now()
        
        # Calculate new parameters based on state
        new_params = self._calculate_scaling_params(new_state, metrics)
        
        # Apply changes
        params_changed = self._update_scaling_params(new_params)
        
        # Log scaling decision
        self.logger.info(
            f"Dynamic scaling: {old_state.value} → {new_state.value} "
            f"(CPU: {metrics.cpu_percent:.1f}%, Mem: {metrics.memory_percent:.1f}%, "
            f"Load: {metrics.load_average:.1f}, Queue: {metrics.queue_depth})"
        )
        
        if params_changed:
            self.logger.info(
                f"Scaling params updated: downloads={self.current_params.max_concurrent_downloads}, "
                f"transcriptions={self.current_params.max_concurrent_transcriptions}, "
                f"delay={self.current_params.processing_delay_seconds:.1f}s"
            )
        
        # Record scaling decision
        self.scaling_history.append({
            'timestamp': datetime.now(),
            'old_state': old_state,
            'new_state': new_state,
            'metrics': metrics,
            'params': new_params
        })
    
    def _calculate_scaling_params(self, state: ScalingState, metrics: SystemMetrics) -> ScalingParams:
        """Calculate optimal scaling parameters for the given state."""
        base_params = ScalingParams()
        
        if state == ScalingState.IDLE:
            # System is idle, can scale up
            base_params.max_concurrent_downloads = min(
                self.current_params.max_concurrent_downloads + 1,
                base_params.max_concurrent_downloads_limit
            )
            base_params.max_concurrent_transcriptions = min(
                self.current_params.max_concurrent_transcriptions + 1,
                base_params.max_concurrent_transcriptions_limit
            )
            base_params.processing_delay_seconds = max(1.0, self.current_params.processing_delay_seconds * 0.8)
            base_params.download_batch_size = min(10, self.current_params.download_batch_size + 1)
            
        elif state == ScalingState.NORMAL:
            # Maintain current levels with minor adjustments
            base_params.max_concurrent_downloads = self.current_params.max_concurrent_downloads
            base_params.max_concurrent_transcriptions = self.current_params.max_concurrent_transcriptions
            base_params.processing_delay_seconds = self.current_params.processing_delay_seconds
            base_params.download_batch_size = self.current_params.download_batch_size
            
        elif state == ScalingState.HIGH_LOAD:
            # Moderate scale down
            base_params.max_concurrent_downloads = max(
                base_params.min_concurrent_downloads,
                self.current_params.max_concurrent_downloads - 1
            )
            base_params.max_concurrent_transcriptions = max(
                base_params.min_concurrent_transcriptions,
                self.current_params.max_concurrent_transcriptions
            )
            base_params.processing_delay_seconds = self.current_params.processing_delay_seconds * 1.2
            base_params.download_batch_size = max(1, self.current_params.download_batch_size - 1)
            
        elif state == ScalingState.OVERLOADED:
            # Aggressive scale down
            base_params.max_concurrent_downloads = max(
                base_params.min_concurrent_downloads,
                self.current_params.max_concurrent_downloads - 2
            )
            base_params.max_concurrent_transcriptions = base_params.min_concurrent_transcriptions
            base_params.processing_delay_seconds = self.current_params.processing_delay_seconds * 1.5
            base_params.download_batch_size = 1
            
        elif state == ScalingState.CRITICAL:
            # Emergency mode - minimal processing
            base_params.max_concurrent_downloads = base_params.min_concurrent_downloads
            base_params.max_concurrent_transcriptions = base_params.min_concurrent_transcriptions
            base_params.processing_delay_seconds = max(5.0, self.current_params.processing_delay_seconds * 2.0)
            base_params.download_batch_size = 1
            base_params.retry_backoff_multiplier = 3.0
        
        return base_params
    
    def _update_scaling_params(self, new_params: ScalingParams) -> bool:
        """Update current scaling parameters and return True if any changed."""
        with self._lock:
            changed = (
                self.current_params.max_concurrent_downloads != new_params.max_concurrent_downloads or
                self.current_params.max_concurrent_transcriptions != new_params.max_concurrent_transcriptions or
                abs(self.current_params.processing_delay_seconds - new_params.processing_delay_seconds) > 0.1 or
                self.current_params.download_batch_size != new_params.download_batch_size
            )
            
            if changed:
                self.current_params = new_params
                
            return changed
    
    def _calculate_network_error_rate(self) -> float:
        """Calculate recent network error rate (errors per minute)."""
        if len(self.performance_metrics['error_rate']) == 0:
            return 0.0
        
        # Use recent error rate from performance tracking
        recent_errors = list(self.performance_metrics['error_rate'])
        return statistics.mean(recent_errors) if recent_errors else 0.0
    
    # Public API methods
    
    def get_current_scaling_params(self) -> ScalingParams:
        """Get current scaling parameters."""
        with self._lock:
            return ScalingParams(
                max_concurrent_downloads=self.current_params.max_concurrent_downloads,
                max_concurrent_transcriptions=self.current_params.max_concurrent_transcriptions,
                download_batch_size=self.current_params.download_batch_size,
                processing_delay_seconds=self.current_params.processing_delay_seconds,
                retry_backoff_multiplier=self.current_params.retry_backoff_multiplier,
                health_check_interval=self.current_params.health_check_interval
            )
    
    def update_queue_depth(self, depth: int):
        """Update current processing queue depth."""
        self._current_queue_depth = depth
    
    def record_performance_metric(self, metric_type: str, value: float):
        """Record a performance metric for scaling decisions."""
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self._lock:
            current_metrics = self._collect_metrics() if self._monitoring else None
            
            return {
                'current_state': self.current_state.value,
                'scaling_params': {
                    'max_concurrent_downloads': self.current_params.max_concurrent_downloads,
                    'max_concurrent_transcriptions': self.current_params.max_concurrent_transcriptions,
                    'processing_delay_seconds': self.current_params.processing_delay_seconds,
                    'download_batch_size': self.current_params.download_batch_size
                },
                'system_metrics': {
                    'cpu_percent': current_metrics.cpu_percent if current_metrics else None,
                    'memory_percent': current_metrics.memory_percent if current_metrics else None,
                    'load_average': current_metrics.load_average if current_metrics else None,
                    'queue_depth': current_metrics.queue_depth if current_metrics else 0
                },
                'monitoring_active': self._monitoring,
                'last_scale_time': self.last_scale_time.isoformat(),
                'scaling_history_count': len(self.scaling_history)
            }


# Global instance
_dynamic_scaling_manager = None


def get_dynamic_scaling_manager() -> DynamicScalingManager:
    """Get or create the global dynamic scaling manager instance."""
    global _dynamic_scaling_manager
    if _dynamic_scaling_manager is None:
        _dynamic_scaling_manager = DynamicScalingManager()
    return _dynamic_scaling_manager