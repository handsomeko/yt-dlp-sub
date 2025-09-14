"""
Advanced Rate Limiting Manager for YouTube API Requests

This module provides comprehensive rate limiting prevention with:
- Exponential backoff for 429 errors
- Request throttling and queuing
- Circuit breaker pattern
- Per-domain rate tracking
- Adaptive rate adjustment
"""

import time
import threading
import logging
import os
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import random
import asyncio

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RateLimitState(Enum):
    """Circuit breaker states for rate limiting."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking all requests
    HALF_OPEN = "half_open"  # Testing recovery


class JitterType(Enum):
    """Jitter algorithms for bulletproof rate limiting."""
    NONE = "none"          # Fixed intervals (backward compatibility)
    FULL = "full"          # Full jitter: random(0, base_interval)
    EQUAL = "equal"        # Equal jitter: base/2 + random(0, base/2)
    DECORRELATED = "decorrelated"  # Decorrelated: based on previous delay


@dataclass
class RateLimitStats:
    """Statistics for rate limit tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    failed_requests: int = 0
    last_429_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_429s: int = 0
    current_delay_seconds: float = 0.0
    success_rate: float = 0.0
    success_history: deque = field(default_factory=lambda: deque(maxlen=20))
    cooldown_until: Optional[datetime] = None


@dataclass 
class DomainRateLimit:
    """Rate limit configuration per domain with bulletproof jittered intervals."""
    domain: str
    requests_per_minute: int = 10
    requests_per_hour: int = 500
    burst_size: int = 5
    min_request_interval: float = 1.0  # DEPRECATED: Use base_interval + jitter instead
    backoff_base: float = 2.0
    backoff_max: float = 300.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    # Bulletproof Jittered Rate Limiting (NEW)
    jitter_type: JitterType = JitterType.FULL
    base_interval: float = 1.5  # Base interval for jittered calculations
    jitter_variance: float = 1.0  # Jitter variance multiplier
    last_decorrelated_delay: float = 0.0  # State for decorrelated jitter


class RateLimitManager:
    """Manages rate limiting with multiple prevention strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get centralized settings
        settings = get_settings()
        
        # Parse jitter type from settings (with backward compatibility)
        jitter_type_str = getattr(settings, 'prevention_jitter_type', 'full').lower()
        jitter_type = JitterType.FULL  # Default
        try:
            jitter_type = JitterType(jitter_type_str)
        except ValueError:
            self.logger.warning(f"Invalid jitter type '{jitter_type_str}', using 'full'")
            jitter_type = JitterType.FULL

        # Domain-specific configurations from centralized settings
        self.domain_configs: Dict[str, DomainRateLimit] = {
            'youtube.com': DomainRateLimit(
                domain='youtube.com',
                requests_per_minute=settings.prevention_rate_limit,
                requests_per_hour=1000,
                burst_size=settings.prevention_burst_size,
                min_request_interval=settings.prevention_min_request_interval,  # Legacy support
                backoff_base=settings.prevention_backoff_base,
                backoff_max=settings.prevention_backoff_max,
                circuit_breaker_threshold=settings.prevention_circuit_breaker_threshold,
                circuit_breaker_timeout=settings.prevention_circuit_breaker_timeout,
                # NEW: Bulletproof jittered rate limiting
                jitter_type=jitter_type,
                base_interval=getattr(settings, 'prevention_base_interval', 1.5),
                jitter_variance=getattr(settings, 'prevention_jitter_variance', 1.0)
            ),
            'googleapis.com': DomainRateLimit(
                domain='googleapis.com', 
                requests_per_minute=60,
                requests_per_hour=10000,
                burst_size=20,
                min_request_interval=1.0,  # Legacy support
                backoff_base=settings.prevention_backoff_base,
                backoff_max=settings.prevention_backoff_max,
                circuit_breaker_threshold=settings.prevention_circuit_breaker_threshold,
                circuit_breaker_timeout=settings.prevention_circuit_breaker_timeout,
                # NEW: Use same jitter config for consistency
                jitter_type=jitter_type,
                base_interval=getattr(settings, 'prevention_base_interval', 1.5),
                jitter_variance=getattr(settings, 'prevention_jitter_variance', 1.0)
            )
        }
        
        # Rate limit tracking
        self.domain_stats: Dict[str, RateLimitStats] = {}
        self.request_history: Dict[str, deque] = {}
        self.circuit_states: Dict[str, RateLimitState] = {}
        self.circuit_open_until: Dict[str, datetime] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Request queue for throttling
        self.request_queues: Dict[str, deque] = {}
        
        # Initialize tracking for each domain
        for domain in self.domain_configs:
            self.domain_stats[domain] = RateLimitStats()
            self.request_history[domain] = deque(maxlen=1000)
            self.circuit_states[domain] = RateLimitState.CLOSED
            self.request_queues[domain] = deque()
        
        # User agent rotation for better scraping success
        self.user_agents = self._load_user_agents()
        self.current_ua_index = 0
    
    def _calculate_jittered_interval(self, config: DomainRateLimit, attempt: int = 0) -> float:
        """
        Calculate bulletproof jittered interval using AWS-recommended algorithms.
        
        Args:
            config: Domain rate limit configuration
            attempt: Number of attempts for exponential backoff (default: 0 for proactive throttling)
            
        Returns:
            Jittered interval in seconds
        """
        base_interval = config.base_interval
        variance = config.jitter_variance
        
        if config.jitter_type == JitterType.NONE:
            # Backward compatibility: use fixed min_request_interval
            return config.min_request_interval
        
        elif config.jitter_type == JitterType.FULL:
            # Full Jitter: random(0, base_interval * variance)
            # AWS formula: random(0, min(base * factor^attempt, max_delay))
            max_interval = base_interval * variance
            if attempt > 0:
                exponential_factor = config.backoff_base ** attempt
                max_interval = min(base_interval * exponential_factor * variance, config.backoff_max)
            
            return random.uniform(0, max_interval)
        
        elif config.jitter_type == JitterType.EQUAL:
            # Equal Jitter: (base/2) + random(0, base/2)
            # AWS formula: (backoff / 2) + random(0, backoff / 2)
            interval = base_interval
            if attempt > 0:
                interval = min(base_interval * (config.backoff_base ** attempt), config.backoff_max)
            
            adjusted_interval = interval * variance
            return (adjusted_interval / 2) + random.uniform(0, adjusted_interval / 2)
        
        elif config.jitter_type == JitterType.DECORRELATED:
            # Decorrelated Jitter: min(base * factor^attempt + random(0, 1), max_delay)
            # Uses last delay as basis for next calculation
            if attempt == 0:
                # First request or proactive throttling
                jitter_amount = random.uniform(0, base_interval * variance)
                config.last_decorrelated_delay = base_interval + jitter_amount
            else:
                # Reactive backoff: base the next delay on the previous one
                base_delay = min(base_interval * (config.backoff_base ** attempt), config.backoff_max)
                jitter_amount = random.uniform(0, base_delay * 0.3)  # 30% jitter for decorrelated
                config.last_decorrelated_delay = min(base_delay + jitter_amount, config.backoff_max)
            
            return config.last_decorrelated_delay
        
        # Default fallback to full jitter
        return random.uniform(0, base_interval * variance)
    
    def should_allow_request(self, domain: str) -> Tuple[bool, float]:
        """
        Check if a request should be allowed based on rate limits.
        
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        with self.lock:
            # Get or create domain config
            if domain not in self.domain_configs:
                self.domain_configs[domain] = DomainRateLimit(domain=domain)
                self.domain_stats[domain] = RateLimitStats()
                self.request_history[domain] = deque(maxlen=1000)
                self.circuit_states[domain] = RateLimitState.CLOSED
                self.request_queues[domain] = deque()
            
            config = self.domain_configs[domain]
            stats = self.domain_stats[domain]
            
            # Check if domain is in cooldown
            if self.check_cooldown(domain):
                wait_time = (stats.cooldown_until - datetime.now()).total_seconds()
                return False, max(wait_time, 0)
            
            # Check circuit breaker state
            circuit_state = self._check_circuit_state(domain)
            if circuit_state == RateLimitState.OPEN:
                wait_time = (self.circuit_open_until[domain] - datetime.now()).total_seconds()
                self.logger.warning(f"Circuit breaker OPEN for {domain}, wait {wait_time:.1f}s")
                return False, max(wait_time, 0)
            
            # Check rate limits
            now = datetime.now()
            
            # Clean old history
            self._clean_request_history(domain, now)
            
            # Check per-minute limit
            minute_ago = now - timedelta(minutes=1)
            recent_minute_requests = sum(1 for req_time in self.request_history[domain] 
                                        if req_time > minute_ago)
            
            if recent_minute_requests >= config.requests_per_minute:
                wait_time = 60.0 - (now - min(self.request_history[domain])).total_seconds()
                self.logger.info(f"Rate limit reached for {domain}: {recent_minute_requests}/{config.requests_per_minute} per minute")
                return False, max(wait_time, 1.0)
            
            # Check jittered interval (replaces fixed min_request_interval)
            if self.request_history[domain]:
                last_request = max(self.request_history[domain])
                time_since_last = (now - last_request).total_seconds()
                
                # Calculate jittered interval instead of using fixed interval
                jittered_interval = self._calculate_jittered_interval(config, attempt=0)
                
                if time_since_last < jittered_interval:
                    wait_time = jittered_interval - time_since_last
                    self.logger.debug(f"Jittered interval {jittered_interval:.2f}s requires wait of {wait_time:.2f}s for {domain}")
                    return False, wait_time
            
            # Check if we're in backoff period
            if stats.current_delay_seconds > 0:
                if stats.last_429_time:
                    time_since_429 = (now - stats.last_429_time).total_seconds()
                    if time_since_429 < stats.current_delay_seconds:
                        wait_time = stats.current_delay_seconds - time_since_429
                        self.logger.info(f"In backoff period for {domain}, wait {wait_time:.1f}s")
                        return False, wait_time
                    else:
                        # Backoff period expired
                        stats.current_delay_seconds = 0
            
            return True, 0.0
    
    def record_request(self, domain: str, success: bool, is_429: bool = False):
        """Record a request and update statistics."""
        with self.lock:
            if domain not in self.domain_stats:
                return
            
            stats = self.domain_stats[domain]
            stats.total_requests += 1
            
            now = datetime.now()
            self.request_history[domain].append(now)
            
            if is_429:
                stats.rate_limited_requests += 1
                stats.consecutive_429s += 1
                stats.last_429_time = now
                
                # Calculate jittered exponential backoff using bulletproof algorithms
                config = self.domain_configs[domain]
                jittered_backoff_delay = self._calculate_jittered_interval(
                    config, 
                    attempt=stats.consecutive_429s
                )
                
                stats.current_delay_seconds = jittered_backoff_delay
                
                self.logger.warning(
                    f"Got 429 for {domain} (#{stats.consecutive_429s}), "
                    f"jittered backoff for {stats.current_delay_seconds:.1f}s "
                    f"(algorithm: {config.jitter_type.value})"
                )
                
                # Check if we should open circuit breaker
                if stats.consecutive_429s >= config.circuit_breaker_threshold:
                    self._open_circuit(domain)
                    
            elif success:
                stats.successful_requests += 1
                stats.last_success_time = now
                stats.consecutive_429s = 0  # Reset on success
                stats.current_delay_seconds = max(0, stats.current_delay_seconds * 0.9)  # Gradual recovery
                
                # Try to close circuit if in half-open state
                if self.circuit_states[domain] == RateLimitState.HALF_OPEN:
                    self._close_circuit(domain)
            else:
                stats.failed_requests += 1
            
            # Update success rate tracking
            self.update_success_rate(domain, success)
    
    def execute_with_rate_limit(
        self, 
        func: Callable,
        domain: str,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0
    ) -> Tuple[Any, bool]:
        """
        Execute a function with rate limiting and retry logic.
        
        Returns:
            Tuple of (result, success)
        """
        retries = 0
        retry_delay = initial_retry_delay
        
        while retries <= max_retries:
            # Check if we should make the request
            allowed, wait_time = self.should_allow_request(domain)
            
            if not allowed:
                if wait_time > 0:
                    self.logger.info(f"Rate limited for {domain}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
            
            try:
                # Execute the function
                result = func()
                
                # Check for rate limit response
                if isinstance(result, dict) and result.get('status_code') == 429:
                    self.record_request(domain, success=False, is_429=True)
                    retries += 1
                    
                    if retries > max_retries:
                        self.logger.error(f"Max retries exceeded for {domain}")
                        return None, False
                    
                    continue
                
                # Success
                self.record_request(domain, success=True)
                return result, True
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if '429' in error_str or 'rate' in error_str or 'too many' in error_str:
                    self.record_request(domain, success=False, is_429=True)
                    retries += 1
                    
                    if retries > max_retries:
                        self.logger.error(f"Max retries exceeded for {domain}: {e}")
                        return None, False
                        
                else:
                    # Non-rate-limit error
                    self.record_request(domain, success=False)
                    self.logger.error(f"Request failed for {domain}: {e}")
                    
                    # Still retry but with different backoff
                    retries += 1
                    if retries > max_retries:
                        return None, False
                    
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Less aggressive backoff for non-429 errors
        
        return None, False
    
    async def execute_with_rate_limit_async(
        self,
        func: Callable,
        domain: str,
        max_retries: int = 5
    ) -> Tuple[Any, bool]:
        """Async version of execute_with_rate_limit."""
        retries = 0
        
        while retries <= max_retries:
            allowed, wait_time = self.should_allow_request(domain)
            
            if not allowed:
                if wait_time > 0:
                    self.logger.info(f"Rate limited for {domain}, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
            
            try:
                result = await func() if asyncio.iscoroutinefunction(func) else func()
                
                if isinstance(result, dict) and result.get('status_code') == 429:
                    self.record_request(domain, success=False, is_429=True)
                    retries += 1
                    continue
                
                self.record_request(domain, success=True)
                return result, True
                
            except Exception as e:
                if '429' in str(e).lower():
                    self.record_request(domain, success=False, is_429=True)
                else:
                    self.record_request(domain, success=False)
                
                retries += 1
                if retries > max_retries:
                    return None, False
        
        return None, False
    
    def _check_circuit_state(self, domain: str) -> RateLimitState:
        """Check and update circuit breaker state."""
        current_state = self.circuit_states[domain]
        
        if current_state == RateLimitState.OPEN:
            # Check if timeout has expired
            if domain in self.circuit_open_until:
                if datetime.now() > self.circuit_open_until[domain]:
                    # Move to half-open to test
                    self.circuit_states[domain] = RateLimitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker for {domain} moved to HALF_OPEN")
                    return RateLimitState.HALF_OPEN
        
        return current_state
    
    def _open_circuit(self, domain: str):
        """Open the circuit breaker for a domain."""
        config = self.domain_configs[domain]
        self.circuit_states[domain] = RateLimitState.OPEN
        self.circuit_open_until[domain] = datetime.now() + timedelta(seconds=config.circuit_breaker_timeout)
        
        self.logger.warning(
            f"Circuit breaker OPENED for {domain}, "
            f"will retry at {self.circuit_open_until[domain].strftime('%H:%M:%S')}"
        )
    
    def _close_circuit(self, domain: str):
        """Close the circuit breaker for a domain."""
        self.circuit_states[domain] = RateLimitState.CLOSED
        self.domain_stats[domain].consecutive_429s = 0
        self.logger.info(f"Circuit breaker CLOSED for {domain}, resuming normal operation")
    
    def _clean_request_history(self, domain: str, now: datetime):
        """Remove old entries from request history."""
        hour_ago = now - timedelta(hours=1)
        
        # Remove requests older than 1 hour
        while self.request_history[domain] and self.request_history[domain][0] < hour_ago:
            self.request_history[domain].popleft()
    
    def get_stats(self, domain: str) -> Dict[str, Any]:
        """Get current statistics for a domain."""
        with self.lock:
            if domain not in self.domain_stats:
                return {}
            
            stats = self.domain_stats[domain]
            config = self.domain_configs[domain]
            
            return {
                'domain': domain,
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'rate_limited_requests': stats.rate_limited_requests,
                'failed_requests': stats.failed_requests,
                'success_rate': (stats.successful_requests / stats.total_requests * 100) 
                               if stats.total_requests > 0 else 0,
                'consecutive_429s': stats.consecutive_429s,
                'current_delay_seconds': stats.current_delay_seconds,
                'circuit_state': self.circuit_states[domain].value,
                'requests_per_minute_limit': config.requests_per_minute,
                'last_429_time': stats.last_429_time.isoformat() if stats.last_429_time else None,
                'last_success_time': stats.last_success_time.isoformat() if stats.last_success_time else None
            }
    
    def reset_domain_stats(self, domain: str):
        """Reset statistics for a domain (useful for testing)."""
        with self.lock:
            if domain in self.domain_stats:
                self.domain_stats[domain] = RateLimitStats()
                self.circuit_states[domain] = RateLimitState.CLOSED
                self.request_history[domain].clear()
                self.logger.info(f"Reset rate limit stats for {domain}")
    
    def _load_user_agents(self) -> List[str]:
        """Load user agents from environment or use defaults."""
        ua_env = os.getenv('YOUTUBE_USER_AGENTS', '')
        if ua_env:
            return ua_env.split(',')
        
        # Default user agents that mimic real browsers
        return [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation."""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def check_cooldown(self, domain: str) -> bool:
        """Check if domain is in cooldown period."""
        if domain not in self.domain_stats:
            return False
        
        stats = self.domain_stats[domain]
        if stats.cooldown_until and datetime.now() < stats.cooldown_until:
            return True
        return False
    
    def set_cooldown(self, domain: str, minutes: int):
        """Set cooldown period for a domain."""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = RateLimitStats()
        
        self.domain_stats[domain].cooldown_until = datetime.now() + timedelta(minutes=minutes)
        self.logger.info(f"Set {minutes}-minute cooldown for {domain}")
    
    def update_success_rate(self, domain: str, success: bool):
        """Update success rate tracking."""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = RateLimitStats()
        
        stats = self.domain_stats[domain]
        stats.success_history.append(success)
        
        # Calculate success rate from recent history
        if len(stats.success_history) > 0:
            stats.success_rate = sum(stats.success_history) / len(stats.success_history)


# Global instance for easy access
_rate_limit_manager = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get or create the global rate limit manager instance."""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager