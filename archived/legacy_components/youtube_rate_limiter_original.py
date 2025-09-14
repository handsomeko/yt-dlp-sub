"""
YouTube Rate Limiter Module

Provides comprehensive rate limiting, error detection, and recovery strategies
for YouTube downloads to prevent blocks and handle rate limits gracefully.
"""

import time
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque
import random


class ErrorType(Enum):
    """YouTube error types with different handling strategies"""
    RATE_LIMIT = "rate_limit"          # 429 or "Too many requests"
    IP_BLOCKED = "ip_blocked"           # 403 or persistent blocks
    BOT_CHECK = "bot_check"             # CAPTCHA or bot detection
    VIDEO_UNAVAILABLE = "unavailable"   # Video is private/deleted
    NETWORK_ERROR = "network"           # Connection issues
    UNKNOWN = "unknown"                 # Other errors


class ErrorAction(Enum):
    """Actions to take based on error type"""
    RETRY_WITH_BACKOFF = "retry_backoff"    # Wait and retry
    STOP_ALL = "stop_all"                   # Stop all downloads
    SKIP_VIDEO = "skip"                     # Skip this video
    COOLDOWN = "cooldown"                   # Long cooldown period
    RETRY_IMMEDIATE = "retry_now"           # Retry immediately


@dataclass
class ErrorInfo:
    """Information about a YouTube error"""
    error_type: ErrorType
    action: ErrorAction
    wait_seconds: float
    message: str
    retry_count: int = 0
    should_alert: bool = False


class CircuitBreaker:
    """Circuit breaker pattern to prevent hammering YouTube when errors occur"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def record_success(self):
        """Record a successful request"""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            return True
        return False
        
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        if self.state == "open":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False
        
    def reset(self):
        """Reset the circuit breaker"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


class YouTubeRateLimiter:
    """
    Comprehensive rate limiter for YouTube downloads with:
    - Token bucket algorithm for rate limiting
    - Sliding window request tracking
    - Error detection and categorization
    - Circuit breaker pattern
    - Adaptive delay calculation
    - User agent rotation
    """
    
    def __init__(
        self,
        requests_per_minute: int = None,
        burst_size: int = None,
        cooldown_minutes: int = None,
        max_failures: int = None,
        backoff_multiplier: float = None
    ):
        """
        Initialize the rate limiter with configurable parameters.
        
        Args:
            requests_per_minute: Maximum requests per minute (default from env)
            burst_size: Allow burst of requests (default from env)
            cooldown_minutes: Cooldown after errors (default from env)
            max_failures: Max failures before stopping (default from env)
            backoff_multiplier: Exponential backoff multiplier (default from env)
        """
        # Load from environment with defaults
        self.requests_per_minute = requests_per_minute or int(os.getenv('YOUTUBE_RATE_LIMIT', '10'))
        self.burst_size = burst_size or int(os.getenv('YOUTUBE_BURST_SIZE', '3'))
        self.cooldown_minutes = cooldown_minutes or int(os.getenv('YOUTUBE_COOLDOWN_MINUTES', '5'))
        self.max_failures = max_failures or int(os.getenv('YOUTUBE_MAX_FAILURES_BEFORE_STOP', '3'))
        self.backoff_multiplier = backoff_multiplier or float(os.getenv('YOUTUBE_BACKOFF_MULTIPLIER', '2.0'))
        
        # Calculate delays
        self.min_delay = 60.0 / self.requests_per_minute  # Minimum seconds between requests
        self.current_delay = self.min_delay
        
        # Request tracking (sliding window)
        self.request_times = deque(maxlen=self.requests_per_minute)
        self.last_request_time = 0
        
        # Error tracking
        self.consecutive_errors = 0
        self.total_errors = 0
        self.last_error_time = None
        self.error_history: List[ErrorInfo] = []
        
        # Circuit breakers for different error types
        self.circuit_breakers = {
            ErrorType.RATE_LIMIT: CircuitBreaker(failure_threshold=2, recovery_timeout=300),
            ErrorType.IP_BLOCKED: CircuitBreaker(failure_threshold=1, recovery_timeout=3600),
            ErrorType.BOT_CHECK: CircuitBreaker(failure_threshold=1, recovery_timeout=3600),
        }
        
        # Success/failure tracking for adaptive rate limiting
        self.success_count = 0
        self.recent_success_rate = 1.0
        self.success_history = deque(maxlen=20)  # Track last 20 requests
        
        # User agents for rotation
        self.user_agents = self._load_user_agents()
        self.current_ua_index = 0
        
        # Logger
        self.logger = logging.getLogger("youtube_rate_limiter")
        
        # Cooldown tracking
        self.cooldown_until = None
        
    def _load_user_agents(self) -> List[str]:
        """Load user agents from environment or use defaults"""
        ua_env = os.getenv('YOUTUBE_USER_AGENTS', '')
        if ua_env:
            return ua_env.split(',')
        
        # Default user agents that mimic real browsers
        return [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
    
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def check_rate_limit(self) -> bool:
        """
        Check if we can make a request based on rate limits.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        # Check if in cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False
        
        # Check circuit breakers
        for breaker in self.circuit_breakers.values():
            if breaker.is_open():
                self.logger.warning("Circuit breaker is open - blocking requests")
                return False
        
        # Check sliding window rate limit
        current_time = time.time()
        
        # Remove old requests outside the window
        while self.request_times and self.request_times[0] < current_time - 60:
            self.request_times.popleft()
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            # Allow burst if configured
            if len(self.request_times) < self.requests_per_minute + self.burst_size:
                self.logger.info(f"Using burst capacity ({len(self.request_times)}/{self.requests_per_minute + self.burst_size})")
                return True
            return False
        
        return True
    
    def wait_if_needed(self) -> float:
        """
        Calculate and wait for the appropriate delay.
        
        Returns:
            Seconds waited
        """
        # Check if in cooldown
        if self.cooldown_until:
            wait_seconds = (self.cooldown_until - datetime.now()).total_seconds()
            if wait_seconds > 0:
                self.logger.info(f"In cooldown for {wait_seconds:.1f} more seconds")
                time.sleep(wait_seconds)
                self.cooldown_until = None
                return wait_seconds
        
        # Calculate adaptive delay based on success rate
        if self.recent_success_rate < 0.5:
            # Low success rate - increase delay
            self.current_delay = min(self.current_delay * 1.5, 60)
        elif self.recent_success_rate > 0.9 and self.consecutive_errors == 0:
            # High success rate - can decrease delay slightly
            self.current_delay = max(self.current_delay * 0.9, self.min_delay)
        
        # Ensure minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.current_delay:
            wait_time = self.current_delay - time_since_last
            time.sleep(wait_time)
            return wait_time
        
        return 0
    
    def track_request(self, success: bool):
        """
        Track a request for rate limiting and statistics.
        
        Args:
            success: Whether the request was successful
        """
        current_time = time.time()
        self.request_times.append(current_time)
        self.last_request_time = current_time
        
        # Track success/failure
        self.success_history.append(success)
        if success:
            self.success_count += 1
            self.consecutive_errors = 0
            # Reset circuit breakers on success
            for breaker in self.circuit_breakers.values():
                breaker.record_success()
        else:
            self.consecutive_errors += 1
            self.total_errors += 1
            self.last_error_time = current_time
        
        # Calculate recent success rate
        if len(self.success_history) > 0:
            self.recent_success_rate = sum(self.success_history) / len(self.success_history)
    
    def detect_error_type(self, error: Exception) -> ErrorType:
        """
        Detect the type of YouTube error from exception.
        
        Args:
            error: The exception that occurred
            
        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        
        # Check for rate limiting indicators
        rate_limit_patterns = [
            r'429',
            r'too many requests',
            r'rate limit',
            r'quota exceeded',
            r'please try again later',
            r'temporarily unavailable'
        ]
        
        for pattern in rate_limit_patterns:
            if re.search(pattern, error_str):
                return ErrorType.RATE_LIMIT
        
        # Check for IP blocks
        block_patterns = [
            r'403',
            r'forbidden',
            r'blocked',
            r'banned',
            r'access denied'
        ]
        
        for pattern in block_patterns:
            if re.search(pattern, error_str):
                return ErrorType.IP_BLOCKED
        
        # Check for bot detection
        bot_patterns = [
            r'captcha',
            r'bot',
            r'verify',
            r'challenge',
            r'human'
        ]
        
        for pattern in bot_patterns:
            if re.search(pattern, error_str):
                return ErrorType.BOT_CHECK
        
        # Check for video unavailability
        unavailable_patterns = [
            r'video unavailable',
            r'private video',
            r'deleted',
            r'not found',
            r'404'
        ]
        
        for pattern in unavailable_patterns:
            if re.search(pattern, error_str):
                return ErrorType.VIDEO_UNAVAILABLE
        
        # Check for network errors
        network_patterns = [
            r'connection',
            r'timeout',
            r'network',
            r'ssl',
            r'certificate'
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, error_str):
                return ErrorType.NETWORK_ERROR
        
        return ErrorType.UNKNOWN
    
    def handle_youtube_error(self, error: Exception, retry_count: int = 0) -> ErrorInfo:
        """
        Handle a YouTube error and determine the appropriate action.
        
        Args:
            error: The exception that occurred
            retry_count: Number of retries already attempted
            
        Returns:
            ErrorInfo with recommended action
        """
        error_type = self.detect_error_type(error)
        
        # Update circuit breaker
        if error_type in self.circuit_breakers:
            should_stop = self.circuit_breakers[error_type].record_failure()
            if should_stop:
                self.logger.error(f"Circuit breaker tripped for {error_type.value}")
        
        # Determine action based on error type and retry count
        if error_type == ErrorType.RATE_LIMIT:
            # Exponential backoff for rate limits
            wait_seconds = self.min_delay * (self.backoff_multiplier ** retry_count)
            wait_seconds = min(wait_seconds, 300)  # Cap at 5 minutes
            
            action = ErrorAction.RETRY_WITH_BACKOFF if retry_count < 3 else ErrorAction.COOLDOWN
            
            error_info = ErrorInfo(
                error_type=error_type,
                action=action,
                wait_seconds=wait_seconds,
                message=f"Rate limited by YouTube. Waiting {wait_seconds:.1f}s",
                retry_count=retry_count,
                should_alert=retry_count >= 2
            )
            
        elif error_type == ErrorType.IP_BLOCKED:
            # IP block requires long cooldown
            error_info = ErrorInfo(
                error_type=error_type,
                action=ErrorAction.STOP_ALL,
                wait_seconds=self.cooldown_minutes * 60,
                message="IP appears to be blocked by YouTube. Stopping all downloads.",
                retry_count=retry_count,
                should_alert=True
            )
            
        elif error_type == ErrorType.BOT_CHECK:
            # Bot detection requires manual intervention
            error_info = ErrorInfo(
                error_type=error_type,
                action=ErrorAction.STOP_ALL,
                wait_seconds=0,
                message="YouTube bot detection triggered. Manual intervention required.",
                retry_count=retry_count,
                should_alert=True
            )
            
        elif error_type == ErrorType.VIDEO_UNAVAILABLE:
            # Skip unavailable videos
            error_info = ErrorInfo(
                error_type=error_type,
                action=ErrorAction.SKIP_VIDEO,
                wait_seconds=0,
                message="Video is unavailable (private/deleted)",
                retry_count=retry_count,
                should_alert=False
            )
            
        elif error_type == ErrorType.NETWORK_ERROR:
            # Network errors - retry with small delay
            error_info = ErrorInfo(
                error_type=error_type,
                action=ErrorAction.RETRY_WITH_BACKOFF if retry_count < 3 else ErrorAction.SKIP_VIDEO,
                wait_seconds=5 * (retry_count + 1),
                message=f"Network error. Retry {retry_count + 1}/3",
                retry_count=retry_count,
                should_alert=False
            )
            
        else:
            # Unknown errors - be cautious
            error_info = ErrorInfo(
                error_type=error_type,
                action=ErrorAction.RETRY_WITH_BACKOFF if retry_count < 2 else ErrorAction.SKIP_VIDEO,
                wait_seconds=10 * (retry_count + 1),
                message=f"Unknown error: {str(error)[:100]}",
                retry_count=retry_count,
                should_alert=retry_count >= 2
            )
        
        # Store error history
        self.error_history.append(error_info)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Check if we should enter cooldown
        if self.consecutive_errors >= self.max_failures:
            self.logger.warning(f"Max failures reached ({self.max_failures}). Entering cooldown.")
            self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
            error_info.action = ErrorAction.COOLDOWN
            error_info.wait_seconds = self.cooldown_minutes * 60
        
        return error_info
    
    def get_current_delay(self) -> float:
        """Get the current delay between requests"""
        return self.current_delay
    
    def reset(self):
        """Reset the rate limiter state"""
        self.consecutive_errors = 0
        self.current_delay = self.min_delay
        self.cooldown_until = None
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        self.logger.info("Rate limiter reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        return {
            'requests_per_minute': self.requests_per_minute,
            'current_delay': self.current_delay,
            'consecutive_errors': self.consecutive_errors,
            'total_errors': self.total_errors,
            'success_rate': self.recent_success_rate,
            'in_cooldown': self.cooldown_until is not None,
            'cooldown_remaining': (self.cooldown_until - datetime.now()).total_seconds() if self.cooldown_until else 0,
            'circuit_breakers': {
                error_type.value: breaker.state 
                for error_type, breaker in self.circuit_breakers.items()
            }
        }
    
    def should_stop_all(self) -> bool:
        """Check if we should stop all downloads"""
        # Check if any circuit breaker is open
        for breaker in self.circuit_breakers.values():
            if breaker.is_open():
                return True
        
        # Check if in cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return True
        
        # Check consecutive errors
        if self.consecutive_errors >= self.max_failures:
            return True
        
        return False


# Convenience function for global rate limiter instance
_global_rate_limiter = None

def get_rate_limiter() -> YouTubeRateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = YouTubeRateLimiter()
    return _global_rate_limiter