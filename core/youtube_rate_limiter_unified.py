"""
Unified YouTube Rate Limiter - Backward Compatible Wrapper

This module provides a backward-compatible wrapper around the new RateLimitManager
to replace the old YouTubeRateLimiter while maintaining the same interface.

This allows gradual migration from the old system to the new unified system.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from .rate_limit_manager import get_rate_limit_manager, RateLimitManager
from config.settings import get_settings


class ErrorType(Enum):
    """YouTube error types with different handling strategies"""
    RATE_LIMIT = "rate_limit"          # 429 or "Too many requests"
    IP_BLOCKED = "ip_blocked"           # 403 or persistent blocks
    BOT_CHECK = "bot_check"             # CAPTCHA or bot detection
    VIDEO_UNAVAILABLE = "unavailable"   # Video is private/deleted
    NETWORK_ERROR = "network"           # Connection issues
    UNKNOWN = "unknown"                 # Other errors


class ErrorAction(Enum):
    """Recommended actions for different error types"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RETRY_WITH_NEW_IP = "retry_with_new_ip"
    RETRY_AFTER_DELAY = "retry_after_delay"
    SKIP_VIDEO = "skip_video"
    STOP_PROCESSING = "stop_processing"


@dataclass
class ErrorInfo:
    """Information about a detected error"""
    error_type: ErrorType
    recommended_action: ErrorAction
    wait_time: float
    retry_count: int
    message: str


class YouTubeRateLimiterUnified:
    """
    Unified YouTube Rate Limiter - Backward Compatible Wrapper
    
    This class provides the same interface as the old YouTubeRateLimiter
    but uses the new RateLimitManager underneath for consistency.
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
        Initialize the unified rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (uses unified config if None)
            burst_size: Allow burst of requests (uses unified config if None)
            cooldown_minutes: Cooldown after errors (uses unified config if None)
            max_failures: Max failures before stopping (uses unified config if None)
            backoff_multiplier: Exponential backoff multiplier (uses unified config if None)
        """
        self.logger = logging.getLogger("youtube_rate_limiter_unified")
        
        # Load settings and create unified configuration
        settings = get_settings()
        self._unified_config = self._create_unified_config(settings)
        
        # Use the unified rate limit manager
        self.rate_manager = get_rate_limit_manager()
        
        # Store parameters using unified configuration with override support
        self.requests_per_minute = requests_per_minute or self._unified_config['requests_per_minute']
        self.burst_size = burst_size or self._unified_config['burst_size']
        self.cooldown_minutes = cooldown_minutes or self._unified_config['cooldown_minutes']
        self.max_failures = max_failures or self._unified_config['max_failures']
        self.backoff_multiplier = backoff_multiplier or self._unified_config['backoff_multiplier']
        
        # Domain we're managing
        self.domain = 'youtube.com'
        
        # Backward compatibility state
        self.consecutive_errors = 0
        self.total_errors = 0
        self.last_error_time = None
    
    def _create_unified_config(self, settings) -> Dict[str, Any]:
        """
        Create unified configuration by prioritizing prevention settings with fallback to legacy.
        
        This method implements the standardization by:
        1. Using prevention settings as primary source
        2. Falling back to legacy settings if prevention not available
        3. Logging deprecation warnings for legacy usage
        4. Providing sensible defaults as final fallback
        
        Args:
            settings: Application settings object
            
        Returns:
            Dictionary with unified rate limiting configuration
        """
        config = {}
        
        # Rate limit mapping: prevention_rate_limit (30) vs youtube_rate_limit (10)
        if hasattr(settings, 'prevention_rate_limit'):
            config['requests_per_minute'] = settings.prevention_rate_limit
        elif hasattr(settings, 'youtube_rate_limit'):
            config['requests_per_minute'] = settings.youtube_rate_limit
            self.logger.warning("Using legacy youtube_rate_limit. Please migrate to PREVENTION_RATE_LIMIT")
        else:
            config['requests_per_minute'] = 30  # Default to prevention system default
        
        # Burst size mapping: prevention_burst_size (10) vs youtube_burst_size (10)
        if hasattr(settings, 'prevention_burst_size'):
            config['burst_size'] = settings.prevention_burst_size
        elif hasattr(settings, 'youtube_burst_size'):
            config['burst_size'] = settings.youtube_burst_size
            self.logger.warning("Using legacy youtube_burst_size. Please migrate to PREVENTION_BURST_SIZE")
        else:
            config['burst_size'] = 10
        
        # Cooldown mapping: prevention_circuit_breaker_timeout (60s) vs youtube_cooldown_minutes (5min = 300s)
        if hasattr(settings, 'prevention_circuit_breaker_timeout'):
            config['cooldown_minutes'] = settings.prevention_circuit_breaker_timeout / 60  # Convert seconds to minutes
        elif hasattr(settings, 'youtube_cooldown_minutes'):
            config['cooldown_minutes'] = settings.youtube_cooldown_minutes
            self.logger.warning("Using legacy youtube_cooldown_minutes. Please migrate to PREVENTION_CIRCUIT_BREAKER_TIMEOUT")
        else:
            config['cooldown_minutes'] = 1  # 60 seconds default
        
        # Max failures mapping: prevention_circuit_breaker_threshold (5) vs youtube_max_failures_before_stop (3)
        if hasattr(settings, 'prevention_circuit_breaker_threshold'):
            config['max_failures'] = settings.prevention_circuit_breaker_threshold
        elif hasattr(settings, 'youtube_max_failures_before_stop'):
            config['max_failures'] = settings.youtube_max_failures_before_stop
            self.logger.warning("Using legacy youtube_max_failures_before_stop. Please migrate to PREVENTION_CIRCUIT_BREAKER_THRESHOLD")
        else:
            config['max_failures'] = 5
        
        # Backoff multiplier mapping: prevention_backoff_base (2.0) vs youtube_backoff_multiplier (2.0)
        if hasattr(settings, 'prevention_backoff_base'):
            config['backoff_multiplier'] = settings.prevention_backoff_base
        elif hasattr(settings, 'youtube_backoff_multiplier'):
            config['backoff_multiplier'] = settings.youtube_backoff_multiplier
            self.logger.warning("Using legacy youtube_backoff_multiplier. Please migrate to PREVENTION_BACKOFF_BASE")
        else:
            config['backoff_multiplier'] = 2.0
        
        # Log unified configuration for transparency
        self.logger.info(f"Unified rate limiting config: {config}")
        
        return config
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get the current migration status and recommendations.
        
        Returns:
            Dictionary with migration status and recommendations
        """
        settings = get_settings()
        migration_status = {
            'using_prevention_system': True,
            'legacy_fields_detected': [],
            'recommendations': []
        }
        
        # Check each field mapping
        legacy_checks = [
            ('youtube_rate_limit', 'PREVENTION_RATE_LIMIT'),
            ('youtube_burst_size', 'PREVENTION_BURST_SIZE'), 
            ('youtube_cooldown_minutes', 'PREVENTION_CIRCUIT_BREAKER_TIMEOUT'),
            ('youtube_max_failures_before_stop', 'PREVENTION_CIRCUIT_BREAKER_THRESHOLD'),
            ('youtube_backoff_multiplier', 'PREVENTION_BACKOFF_BASE')
        ]
        
        for legacy_field, prevention_env in legacy_checks:
            if hasattr(settings, legacy_field) and not hasattr(settings, legacy_field.replace('youtube_', 'prevention_')):
                migration_status['legacy_fields_detected'].append(legacy_field)
                migration_status['using_prevention_system'] = False
                migration_status['recommendations'].append(
                    f"Replace {legacy_field.upper()} with {prevention_env}"
                )
        
        if migration_status['using_prevention_system']:
            migration_status['recommendations'].append("✅ All rate limiting is using the prevention system")
        else:
            migration_status['recommendations'].append("⚠️  Consider migrating legacy settings to prevention system")
            migration_status['recommendations'].append("See updated .env for PREVENTION_* variables")
        
        return migration_status
        
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation."""
        return self.rate_manager.get_next_user_agent()
    
    def check_rate_limit(self) -> bool:
        """
        Check if we can make a request based on rate limits.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        allowed, wait_time = self.rate_manager.should_allow_request(self.domain)
        
        if not allowed and wait_time > 0:
            self.logger.warning(f"Rate limited, need to wait {wait_time:.1f}s")
            
        return allowed
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limiting is required.
        
        Returns:
            Time waited in seconds
        """
        allowed, wait_time = self.rate_manager.should_allow_request(self.domain)
        
        if not allowed and wait_time > 0:
            self.logger.info(f"Waiting {wait_time:.1f}s for rate limit")
            time.sleep(wait_time)
            return wait_time
            
        return 0.0
    
    def track_request(self, success: bool):
        """Track a request for rate limiting and statistics."""
        self.rate_manager.record_request(self.domain, success=success)
        
        if success:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            self.total_errors += 1
            self.last_error_time = datetime.now()
    
    def detect_error_type(self, error: Exception) -> ErrorType:
        """
        Detect the type of error based on the exception.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Detected error type
        """
        error_str = str(error).lower()
        
        # Rate limiting errors
        if any(phrase in error_str for phrase in ['429', 'too many requests', 'rate limit']):
            return ErrorType.RATE_LIMIT
        
        # IP blocking
        if any(phrase in error_str for phrase in ['403', 'forbidden', 'blocked']):
            return ErrorType.IP_BLOCKED
        
        # Bot detection
        if any(phrase in error_str for phrase in ['captcha', 'bot', 'automated']):
            return ErrorType.BOT_CHECK
        
        # Video unavailable
        if any(phrase in error_str for phrase in ['unavailable', 'private', 'deleted']):
            return ErrorType.VIDEO_UNAVAILABLE
        
        # Network errors
        if any(phrase in error_str for phrase in ['connection', 'timeout', 'network']):
            return ErrorType.NETWORK_ERROR
        
        return ErrorType.UNKNOWN
    
    def handle_youtube_error(self, error: Exception, retry_count: int = 0) -> ErrorInfo:
        """
        Handle a YouTube error and provide recommendations.
        
        Args:
            error: Exception that occurred
            retry_count: Current retry attempt
            
        Returns:
            ErrorInfo with recommendations
        """
        error_type = self.detect_error_type(error)
        
        # Record the error
        is_429 = error_type == ErrorType.RATE_LIMIT
        self.rate_manager.record_request(self.domain, success=False, is_429=is_429)
        
        # Determine action based on error type
        if error_type == ErrorType.RATE_LIMIT:
            wait_time = min(self.backoff_multiplier ** retry_count, 300)  # Max 5 minutes
            return ErrorInfo(
                error_type=error_type,
                recommended_action=ErrorAction.RETRY_WITH_BACKOFF,
                wait_time=wait_time,
                retry_count=retry_count,
                message=f"Rate limited, wait {wait_time:.1f}s"
            )
        
        elif error_type == ErrorType.IP_BLOCKED:
            # Set longer cooldown for IP blocks
            self.rate_manager.set_cooldown(self.domain, 60)  # 60 minutes
            return ErrorInfo(
                error_type=error_type,
                recommended_action=ErrorAction.RETRY_WITH_NEW_IP,
                wait_time=3600,  # 1 hour
                retry_count=retry_count,
                message="IP blocked, need new IP or long wait"
            )
        
        elif error_type == ErrorType.BOT_CHECK:
            # Set cooldown for bot detection
            self.rate_manager.set_cooldown(self.domain, 30)  # 30 minutes
            return ErrorInfo(
                error_type=error_type,
                recommended_action=ErrorAction.RETRY_AFTER_DELAY,
                wait_time=1800,  # 30 minutes
                retry_count=retry_count,
                message="Bot detection, wait 30 minutes"
            )
        
        elif error_type == ErrorType.VIDEO_UNAVAILABLE:
            return ErrorInfo(
                error_type=error_type,
                recommended_action=ErrorAction.SKIP_VIDEO,
                wait_time=0,
                retry_count=retry_count,
                message="Video unavailable, skip"
            )
        
        else:
            # Generic backoff for other errors
            wait_time = min(retry_count * 5, 60)  # Max 1 minute
            return ErrorInfo(
                error_type=error_type,
                recommended_action=ErrorAction.RETRY_WITH_BACKOFF,
                wait_time=wait_time,
                retry_count=retry_count,
                message=f"Generic error, wait {wait_time}s"
            )
    
    def get_current_delay(self) -> float:
        """Get current delay between requests."""
        stats = self.rate_manager.get_stats(self.domain)
        return stats.get('current_delay_seconds', 0.0)
    
    def reset(self):
        """Reset the rate limiter state."""
        self.rate_manager.reset_domain_stats(self.domain)
        self.consecutive_errors = 0
        self.total_errors = 0
        self.last_error_time = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        stats = self.rate_manager.get_stats(self.domain)
        
        return {
            'requests_per_minute': self.requests_per_minute,
            'total_requests': stats.get('total_requests', 0),
            'successful_requests': stats.get('successful_requests', 0),
            'failed_requests': stats.get('failed_requests', 0),
            'rate_limited_requests': stats.get('rate_limited_requests', 0),
            'consecutive_errors': self.consecutive_errors,
            'current_delay': stats.get('current_delay_seconds', 0.0),
            'success_rate': stats.get('success_rate', 0.0),
            'circuit_state': stats.get('circuit_state', 'closed'),
            'last_error_time': self.last_error_time
        }
    
    def should_stop_all(self) -> bool:
        """Check if we should stop all processing due to too many errors."""
        if self.consecutive_errors >= self.max_failures:
            return True
            
        # Check if circuit breaker is open
        stats = self.rate_manager.get_stats(self.domain)
        return stats.get('circuit_state') == 'open'


# Backward compatibility function
def get_rate_limiter() -> YouTubeRateLimiterUnified:
    """Get a rate limiter instance for backward compatibility."""
    return YouTubeRateLimiterUnified()