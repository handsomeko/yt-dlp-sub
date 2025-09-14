"""
Comprehensive Error Handling System
Provides centralized error management, recovery, and monitoring
"""

import functools
import logging
import traceback
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Enhanced error categories for comprehensive classification and handling."""
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    SECURITY = "security"
    CONCURRENCY = "concurrency"
    EXTERNAL_API = "external_api"
    CONFIGURATION = "configuration"
    
    # Issue #11: Additional categories for complete coverage
    YOUTUBE_API = "youtube_api"
    WHISPER_AI = "whisper_ai"
    WORKER_FAILURE = "worker_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    PARSING_ERROR = "parsing_error"
    STORAGE_SYNC = "storage_sync"
    
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Detailed error information for tracking and recovery."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    traceback: str
    context: Dict[str, Any]
    timestamp: datetime
    recoverable: bool
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'exception_type': self.exception_type,
            'traceback': self.traceback,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'recoverable': self.recoverable,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if this strategy can recover from the error."""
        return False
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy with exponential backoff."""
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if error is retryable."""
        return (error_details.recoverable and 
                error_details.retry_count < error_details.max_retries)
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Retry operation with exponential backoff."""
        wait_time = error_details.retry_delay * (2 ** error_details.retry_count)
        logger.info(f"Retrying operation after {wait_time}s (attempt {error_details.retry_count + 1})")
        
        time.sleep(wait_time)
        error_details.retry_count += 1
        
        return operation(*args, **kwargs)


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if fallback is available."""
        return self.fallback_func is not None
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Use fallback function."""
        logger.info(f"Using fallback for failed operation: {operation.__name__}")
        return self.fallback_func(*args, **kwargs)


class GracefulDegradationStrategy(ErrorRecoveryStrategy):
    """Issue #12: Graceful degradation strategy for service failures."""
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if graceful degradation is applicable."""
        degradable_categories = {
            ErrorCategory.SERVICE_UNAVAILABLE,
            ErrorCategory.EXTERNAL_API,
            ErrorCategory.NETWORK,
            ErrorCategory.STORAGE_SYNC,
            ErrorCategory.WHISPER_AI
        }
        return error_details.category in degradable_categories
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Provide degraded functionality."""
        logger.warning(f"Activating graceful degradation for {error_details.category.value}")
        
        # Return minimal functionality based on error category
        if error_details.category == ErrorCategory.WHISPER_AI:
            # Return empty transcript with metadata indicating AI failure
            return {
                "status": "degraded",
                "data": {
                    "transcript": "",
                    "degradation_reason": "AI transcription unavailable",
                    "fallback_used": True
                }
            }
        elif error_details.category == ErrorCategory.STORAGE_SYNC:
            # Continue with local storage only
            return {
                "status": "degraded",
                "data": {
                    "local_stored": True,
                    "cloud_sync_failed": True,
                    "degradation_reason": "Cloud storage unavailable"
                }
            }
        else:
            # Generic degraded response
            return {
                "status": "degraded",
                "data": {},
                "degradation_reason": f"Service degraded due to {error_details.category.value}"
            }


class ResourceCleanupStrategy(ErrorRecoveryStrategy):
    """Issue #12: Resource cleanup and recovery strategy."""
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if resource cleanup can help."""
        return error_details.category == ErrorCategory.RESOURCE_EXHAUSTION
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Attempt resource cleanup and retry."""
        logger.warning("Attempting resource cleanup due to exhaustion")
        
        import gc
        import threading
        
        # Force garbage collection
        gc.collect()
        
        # Log current resource usage
        thread_count = threading.active_count()
        logger.info(f"Active threads after cleanup: {thread_count}")
        
        # Wait a moment for resources to be freed
        time.sleep(5)
        
        # Retry operation with reduced parameters if possible
        try:
            # Attempt to reduce operation scale
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                logger.info(f"Reduced batch_size to {kwargs['batch_size']}")
            
            return operation(*args, **kwargs)
        except Exception as retry_error:
            logger.error(f"Resource cleanup recovery failed: {retry_error}")
            raise retry_error


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """Issue #12: Circuit breaker pattern for repeated failures."""
    
    def __init__(self):
        self.failure_counts = {}
        self.failure_threshold = 5
        self.recovery_time = 300  # 5 minutes
        self.last_failure_times = {}
    
    def can_recover(self, error_details: ErrorDetails) -> bool:
        """Check if circuit breaker allows recovery."""
        operation_key = error_details.context.get('operation', 'unknown')
        
        # Check if we're in recovery period
        if operation_key in self.last_failure_times:
            last_failure = self.last_failure_times[operation_key]
            if (datetime.utcnow() - last_failure).total_seconds() < self.recovery_time:
                return False  # Still in circuit breaker period
        
        return True
    
    def recover(self, error_details: ErrorDetails, operation: Callable, *args, **kwargs) -> Any:
        """Track failures and implement circuit breaker logic."""
        operation_key = error_details.context.get('operation', 'unknown')
        
        # Increment failure count
        self.failure_counts[operation_key] = self.failure_counts.get(operation_key, 0) + 1
        
        if self.failure_counts[operation_key] >= self.failure_threshold:
            # Circuit breaker triggered
            self.last_failure_times[operation_key] = datetime.utcnow()
            logger.error(f"Circuit breaker activated for {operation_key} - too many failures")
            raise Exception(f"Circuit breaker: {operation_key} temporarily disabled")
        
        # Allow retry
        return operation(*args, **kwargs)


class ErrorAnalyzer:
    """Analyzes exceptions and classifies them."""
    
    ERROR_PATTERNS = {
        # Filesystem errors
        (FileNotFoundError, PermissionError, OSError): {
            'category': ErrorCategory.FILESYSTEM,
            'severity': ErrorSeverity.MEDIUM,
            'recoverable': True
        },
        
        # Database errors
        ('sqlite3.OperationalError', 'sqlalchemy.exc.OperationalError', 'sqlite3.DatabaseError'): {
            'category': ErrorCategory.DATABASE,
            'severity': ErrorSeverity.HIGH,
            'recoverable': True
        },
        
        # Network errors
        ('requests.exceptions.RequestException', 'urllib3.exceptions.HTTPError', 'httpx.RequestError'): {
            'category': ErrorCategory.NETWORK,
            'severity': ErrorSeverity.MEDIUM,
            'recoverable': True
        },
        
        # Validation errors
        (ValueError, TypeError): {
            'category': ErrorCategory.VALIDATION,
            'severity': ErrorSeverity.LOW,
            'recoverable': False
        },
        
        # Security errors
        ('SecurityError', 'PermissionError'): {
            'category': ErrorCategory.SECURITY,
            'severity': ErrorSeverity.CRITICAL,
            'recoverable': False
        },
        
        # Concurrency errors
        ('threading.ThreadError', 'multiprocessing.ProcessError'): {
            'category': ErrorCategory.CONCURRENCY,
            'severity': ErrorSeverity.HIGH,
            'recoverable': True
        },
        
        # Issue #11: Enhanced error patterns for complete coverage
        
        # YouTube/yt-dlp specific errors
        ('yt_dlp.utils.DownloadError', 'youtube_dl.utils.DownloadError'): {
            'category': ErrorCategory.YOUTUBE_API,
            'severity': ErrorSeverity.HIGH,
            'recoverable': True
        },
        
        # Whisper AI errors
        ('whisper.transcribe.TranscriptionError', 'openai.error.RateLimitError'): {
            'category': ErrorCategory.WHISPER_AI,
            'severity': ErrorSeverity.HIGH,
            'recoverable': True
        },
        
        # Worker and async errors
        ('asyncio.TimeoutError', 'concurrent.futures.TimeoutError'): {
            'category': ErrorCategory.WORKER_FAILURE,
            'severity': ErrorSeverity.HIGH,
            'recoverable': True
        },
        
        # Resource exhaustion
        ('MemoryError', 'OSError'): {
            'category': ErrorCategory.RESOURCE_EXHAUSTION,
            'severity': ErrorSeverity.CRITICAL,
            'recoverable': False
        },
        
        # Authentication errors
        ('requests.exceptions.HTTPError', 'urllib3.exceptions.HTTPError'): {
            'category': ErrorCategory.AUTHENTICATION,
            'severity': ErrorSeverity.HIGH,
            'recoverable': False
        },
        
        # Parsing errors
        ('json.JSONDecodeError', 'xml.etree.ElementTree.ParseError', 'feedparser.FeedParserError'): {
            'category': ErrorCategory.PARSING_ERROR,
            'severity': ErrorSeverity.MEDIUM,
            'recoverable': True
        }
    }
    
    def analyze_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorDetails:
        """Analyze exception and create error details."""
        error_info = self._classify_exception(exception)
        
        error_details = ErrorDetails(
            error_id=self._generate_error_id(),
            category=error_info['category'],
            severity=error_info['severity'],
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or {},
            timestamp=datetime.utcnow(),
            recoverable=error_info['recoverable']
        )
        
        # Adjust based on specific error patterns
        self._adjust_error_details(error_details, exception)
        
        return error_details
    
    def _classify_exception(self, exception: Exception) -> Dict[str, Any]:
        """Classify exception into category and severity."""
        exc_type = type(exception)
        exc_name = exc_type.__name__
        
        # Check direct type matches
        for pattern, info in self.ERROR_PATTERNS.items():
            if isinstance(pattern, tuple) and isinstance(pattern[0], type):
                if any(isinstance(exception, exc_type) for exc_type in pattern):
                    return info
            elif isinstance(pattern, tuple):
                if any(exc_name == name or name in str(exc_type) for name in pattern):
                    return info
        
        # Default classification
        return {
            'category': ErrorCategory.UNKNOWN,
            'severity': ErrorSeverity.MEDIUM,
            'recoverable': True
        }
    
    def _adjust_error_details(self, error_details: ErrorDetails, exception: Exception):
        """Adjust error details based on specific patterns with enhanced detection."""
        message_lower = str(exception).lower()
        exception_name = type(exception).__name__.lower()
        
        # Issue #11: Enhanced pattern matching for comprehensive error classification
        
        # Resource exhaustion - critical and non-recoverable
        if any(pattern in message_lower for pattern in [
            'no space', 'disk full', 'quota exceeded', 'memory error', 'out of memory',
            'resource temporarily unavailable', 'too many open files'
        ]):
            error_details.category = ErrorCategory.RESOURCE_EXHAUSTION
            error_details.severity = ErrorSeverity.CRITICAL
            error_details.recoverable = False
        
        # YouTube API specific errors
        elif any(pattern in message_lower for pattern in [
            'video unavailable', 'private video', 'blocked in your country',
            'youtube said:', 'age-restricted', 'signin required'
        ]):
            error_details.category = ErrorCategory.YOUTUBE_API
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = False  # These are permanent failures
        
        # Rate limiting - recoverable with longer delay
        elif any(pattern in message_lower for pattern in [
            'rate limit', 'too many requests', '429', 'quota exceeded',
            'api limit', 'throttle', 'slow down'
        ]):
            error_details.category = ErrorCategory.RATE_LIMITING
            error_details.severity = ErrorSeverity.MEDIUM
            error_details.recoverable = True
            error_details.max_retries = 5
            error_details.retry_delay = 300  # 5 minutes for rate limits
        
        # Network issues - retryable with backoff
        elif any(pattern in message_lower for pattern in [
            'timeout', 'connection', 'network', 'dns', 'ssl', 'tls', 
            'unreachable', 'refused', 'reset by peer'
        ]):
            error_details.category = ErrorCategory.NETWORK
            error_details.severity = ErrorSeverity.MEDIUM
            error_details.recoverable = True
            error_details.max_retries = 3
            error_details.retry_delay = 10
        
        # Authentication and permission errors
        elif any(pattern in message_lower for pattern in [
            'unauthorized', '401', '403', 'forbidden', 'access denied',
            'authentication', 'invalid credentials', 'permission denied'
        ]):
            error_details.category = ErrorCategory.AUTHENTICATION
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = False
        
        # Worker and async failures
        elif any(pattern in message_lower for pattern in [
            'worker failed', 'task cancelled', 'coroutine', 'asyncio',
            'concurrent.futures', 'thread pool', 'process pool'
        ]):
            error_details.category = ErrorCategory.WORKER_FAILURE
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = True
            error_details.max_retries = 2
            error_details.retry_delay = 30
        
        # Database corruption or integrity issues
        elif any(pattern in message_lower for pattern in [
            'corrupt', 'database disk image is malformed', 'integrity constraint',
            'foreign key constraint', 'unique constraint', 'check constraint'
        ]):
            error_details.category = ErrorCategory.DATA_CORRUPTION
            error_details.severity = ErrorSeverity.CRITICAL
            error_details.recoverable = False
        
        # Service unavailability - retryable
        elif any(pattern in message_lower for pattern in [
            'service unavailable', '503', '502', '504', 'bad gateway',
            'gateway timeout', 'temporarily unavailable'
        ]):
            error_details.category = ErrorCategory.SERVICE_UNAVAILABLE
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = True
            error_details.max_retries = 3
            error_details.retry_delay = 60
        
        # Path traversal and security issues
        elif any(pattern in message_lower for pattern in [
            '../', '..\\', 'dangerous pattern', 'path traversal',
            'directory traversal', 'file inclusion', 'security violation'
        ]):
            error_details.category = ErrorCategory.SECURITY
            error_details.severity = ErrorSeverity.CRITICAL
            error_details.recoverable = False
        
        # Configuration errors
        elif any(pattern in message_lower for pattern in [
            'configuration', 'config', 'settings', 'environment variable',
            'missing key', 'invalid setting', 'setup error'
        ]):
            error_details.category = ErrorCategory.CONFIGURATION
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = False
        
        # Whisper/AI specific errors
        elif any(pattern in message_lower for pattern in [
            'whisper', 'transcription failed', 'ai model', 'openai',
            'model not found', 'cuda', 'gpu memory'
        ]):
            error_details.category = ErrorCategory.WHISPER_AI
            error_details.severity = ErrorSeverity.HIGH
            error_details.recoverable = True
            error_details.max_retries = 2
            error_details.retry_delay = 30
        
        # Storage sync errors
        elif any(pattern in message_lower for pattern in [
            'storage sync', 'google drive', 'airtable', 'backup failed',
            'sync failed', 'upload failed', 'cloud storage'
        ]):
            error_details.category = ErrorCategory.STORAGE_SYNC
            error_details.severity = ErrorSeverity.MEDIUM
            error_details.recoverable = True
            error_details.max_retries = 3
            error_details.retry_delay = 60
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"ERR_{int(time.time())}_{uuid.uuid4().hex[:8]}"


class ErrorManager:
    """Enhanced central error management system with monitoring and diagnostics."""
    
    def __init__(self, error_log_path: Optional[Path] = None):
        self.analyzer = ErrorAnalyzer()
        
        # Issue #12: Enhanced recovery strategies
        self.recovery_strategies = [
            RetryStrategy(),
            GracefulDegradationStrategy(),
            ResourceCleanupStrategy(),
            CircuitBreakerStrategy()
        ]
        
        self.error_log_path = error_log_path
        self.error_history: List[ErrorDetails] = []
        self._lock = threading.RLock()
        
        # Issue #13: Enhanced error statistics and monitoring
        self.stats = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'recovered_errors': 0,
            'unrecoverable_errors': 0,
            'degraded_operations': 0,
            'circuit_breaker_activations': 0,
            'resource_cleanup_attempts': 0
        }
        
        # Error pattern detection for Issue #13
        self.error_patterns = {}
        self.alert_thresholds = {
            'critical_errors_per_hour': 5,
            'failed_recovery_rate': 0.8,
            'repeated_error_pattern': 3
        }
        
        # Issue #13: Monitoring and alerting
        self.monitoring_callbacks = []
        self.last_alert_times = {}
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    def handle_error(self, 
                    exception: Exception, 
                    operation: Optional[Callable] = None,
                    context: Dict[str, Any] = None,
                    *args, **kwargs) -> Any:
        """Handle error with recovery attempts."""
        
        # Analyze the error
        error_details = self.analyzer.analyze_error(exception, context)
        
        with self._lock:
            # Update statistics
            self._update_stats(error_details)
            
            # Log error
            self._log_error(error_details)
            
            # Store in history
            self.error_history.append(error_details)
            
            # Attempt recovery if operation is provided
            if operation:
                for strategy in self.recovery_strategies:
                    if strategy.can_recover(error_details):
                        try:
                            logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                            result = strategy.recover(error_details, operation, *args, **kwargs)
                            
                            # Update stats on successful recovery
                            self.stats['recovered_errors'] += 1
                            
                            logger.info(f"Successfully recovered from error: {error_details.error_id}")
                            return result
                            
                        except Exception as recovery_error:
                            logger.warning(f"Recovery strategy failed: {recovery_error}")
                            continue
            
            # No recovery possible
            self.stats['unrecoverable_errors'] += 1
            logger.error(f"Unrecoverable error: {error_details.error_id}")
            
            # Re-raise if critical
            if error_details.severity == ErrorSeverity.CRITICAL:
                raise exception
            
            return None
    
    # Issue #30: Async error handling support
    async def handle_error_async(self, 
                                exception: Exception, 
                                operation: Optional[Callable] = None,
                                context: Dict[str, Any] = None,
                                *args, **kwargs) -> Any:
        """Async version of handle_error for consistency across sync/async operations."""
        
        # Analyze the error
        error_details = self.analyzer.analyze_error(exception, context)
        
        with self._lock:
            # Update statistics
            self._update_stats(error_details)
            
            # Log error
            self._log_error(error_details)
            
            # Store in history
            self.error_history.append(error_details)
            
            # Attempt recovery if operation is provided
            if operation:
                for strategy in self.recovery_strategies:
                    if strategy.can_recover(error_details):
                        try:
                            logger.info(f"Attempting async recovery with {strategy.__class__.__name__}")
                            
                            # Handle async recovery if operation is async
                            if asyncio.iscoroutinefunction(operation):
                                result = await self._async_recovery(strategy, error_details, operation, *args, **kwargs)
                            else:
                                result = strategy.recover(error_details, operation, *args, **kwargs)
                            
                            # Update stats on successful recovery
                            self.stats['recovered_errors'] += 1
                            
                            logger.info(f"Successfully recovered from async error: {error_details.error_id}")
                            return result
                            
                        except Exception as recovery_error:
                            logger.warning(f"Async recovery strategy failed: {recovery_error}")
                            continue
            
            # No recovery possible
            self.stats['unrecoverable_errors'] += 1
            logger.error(f"Unrecoverable async error: {error_details.error_id}")
            
            # Re-raise if critical
            if error_details.severity == ErrorSeverity.CRITICAL:
                raise exception
            
            return None
    
    async def _async_recovery(self, strategy: ErrorRecoveryStrategy, error_details: ErrorDetails, 
                             operation: Callable, *args, **kwargs) -> Any:
        """Handle async recovery operations."""
        if isinstance(strategy, RetryStrategy):
            # Async retry with proper await
            wait_time = error_details.retry_delay * (2 ** error_details.retry_count)
            logger.info(f"Async retry after {wait_time}s (attempt {error_details.retry_count + 1})")
            
            await asyncio.sleep(wait_time)
            error_details.retry_count += 1
            
            return await operation(*args, **kwargs)
        
        elif isinstance(strategy, GracefulDegradationStrategy):
            # Graceful degradation for async operations
            return strategy.recover(error_details, operation, *args, **kwargs)
        
        elif isinstance(strategy, ResourceCleanupStrategy):
            # Async resource cleanup
            logger.warning("Attempting async resource cleanup due to exhaustion")
            
            import gc
            gc.collect()
            
            # Async wait for resources to be freed
            await asyncio.sleep(5)
            
            # Retry operation with reduced parameters if possible
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                logger.info(f"Reduced batch_size to {kwargs['batch_size']}")
            
            return await operation(*args, **kwargs)
        
        else:
            # Fallback to sync recovery for other strategies
            return strategy.recover(error_details, operation, *args, **kwargs)
    
    def _update_stats(self, error_details: ErrorDetails):
        """Update error statistics."""
        self.stats['total_errors'] += 1
        
        category = error_details.category.value
        severity = error_details.severity.value
        
        self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
        self.stats['by_severity'][severity] = self.stats['by_severity'].get(severity, 0) + 1
    
    def _log_error(self, error_details: ErrorDetails):
        """Enhanced error logging with monitoring and pattern detection."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_details.severity, logging.ERROR)
        
        # Issue #13: Enhanced error logging with context
        context_info = ""
        if error_details.context:
            context_parts = [f"{k}={v}" for k, v in error_details.context.items()]
            context_info = f" | Context: {', '.join(context_parts)}"
        
        logger.log(
            log_level,
            f"Error {error_details.error_id} [{error_details.category.value}]: {error_details.message}{context_info}"
        )
        
        # Write to error log file if configured
        if self.error_log_path:
            try:
                self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.error_log_path, 'a', encoding='utf-8') as f:
                    json.dump(error_details.to_dict(), f)
                    f.write('\n')
            except Exception as e:
                logger.warning(f"Failed to write error log: {e}")
        
        # Issue #13: Pattern detection and alerting
        self._detect_error_patterns(error_details)
        self._check_alert_thresholds(error_details)
        
        # Notify monitoring callbacks
        for callback in self.monitoring_callbacks:
            try:
                callback(error_details)
            except Exception as callback_error:
                logger.warning(f"Monitoring callback failed: {callback_error}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and diagnostics."""
        with self._lock:
            # Calculate additional metrics
            total_errors = self.stats['total_errors']
            recovery_rate = 0.0
            if total_errors > 0:
                recovery_rate = self.stats['recovered_errors'] / total_errors
            
            # Recent error trends
            recent_errors = self.error_history[-50:]  # Last 50 errors for trend analysis
            error_trend = self._calculate_error_trend(recent_errors)
            
            # Pattern analysis
            common_patterns = self._get_common_error_patterns()
            
            return {
                **self.stats,
                'error_history_count': len(self.error_history),
                'recovery_rate': recovery_rate,
                'error_trend': error_trend,
                'common_patterns': common_patterns,
                'recent_errors': [
                    {
                        'id': err.error_id,
                        'category': err.category.value,
                        'severity': err.severity.value,
                        'message': err.message[:100],
                        'timestamp': err.timestamp.isoformat(),
                        'recoverable': err.recoverable,
                        'retry_count': err.retry_count
                    }
                    for err in self.error_history[-10:]  # Last 10 errors
                ],
                'health_indicators': self._get_system_health_indicators()
            }
    
    def clear_error_history(self, older_than_days: int = 7):
        """Clear old error history."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        with self._lock:
            old_count = len(self.error_history)
            self.error_history = [
                err for err in self.error_history 
                if err.timestamp > cutoff
            ]
            new_count = len(self.error_history)
            logger.info(f"Cleared {old_count - new_count} old error records")
    
    # Issue #13: Enhanced monitoring and diagnostics methods
    
    def add_monitoring_callback(self, callback: Callable[[ErrorDetails], None]):
        """Add callback for error monitoring and alerting."""
        self.monitoring_callbacks.append(callback)
    
    def _detect_error_patterns(self, error_details: ErrorDetails):
        """Detect recurring error patterns."""
        pattern_key = f"{error_details.category.value}:{error_details.exception_type}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                'count': 0,
                'first_seen': error_details.timestamp,
                'last_seen': error_details.timestamp,
                'messages': []
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern['count'] += 1
        pattern['last_seen'] = error_details.timestamp
        pattern['messages'].append(error_details.message[:100])  # Keep recent messages
        
        # Keep only recent messages
        if len(pattern['messages']) > 10:
            pattern['messages'] = pattern['messages'][-10:]
    
    def _check_alert_thresholds(self, error_details: ErrorDetails):
        """Check if error patterns exceed alert thresholds."""
        now = datetime.utcnow()
        
        # Check critical errors per hour
        if error_details.severity == ErrorSeverity.CRITICAL:
            recent_critical = [
                err for err in self.error_history[-20:] 
                if err.severity == ErrorSeverity.CRITICAL and 
                   (now - err.timestamp).total_seconds() < 3600
            ]
            
            if len(recent_critical) >= self.alert_thresholds['critical_errors_per_hour']:
                self._trigger_alert('critical_errors_per_hour', {
                    'count': len(recent_critical),
                    'threshold': self.alert_thresholds['critical_errors_per_hour']
                })
    
    def _trigger_alert(self, alert_type: str, context: Dict[str, Any]):
        """Trigger system alert for critical error patterns."""
        # Prevent spam alerts
        if alert_type in self.last_alert_times:
            last_alert = self.last_alert_times[alert_type]
            if (datetime.utcnow() - last_alert).total_seconds() < 300:  # 5 minutes
                return
        
        self.last_alert_times[alert_type] = datetime.utcnow()
        
        logger.critical(f"SYSTEM ALERT: {alert_type} - {context}")
        
        # Trigger monitoring callbacks
        alert_details = ErrorDetails(
            error_id=f"ALERT_{alert_type}_{int(time.time())}",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.CRITICAL,
            message=f"System alert: {alert_type}",
            exception_type="SystemAlert",
            traceback="",
            context=context,
            timestamp=datetime.utcnow(),
            recoverable=False
        )
        
        for callback in self.monitoring_callbacks:
            try:
                callback(alert_details)
            except Exception as callback_error:
                logger.error(f"Alert callback failed: {callback_error}")
    
    def _calculate_error_trend(self, recent_errors: List[ErrorDetails]) -> Dict[str, Any]:
        """Calculate error trend analysis."""
        if not recent_errors:
            return {'trend': 'stable', 'change': 0}
        
        # Split into two halves for trend comparison
        mid_point = len(recent_errors) // 2
        first_half = recent_errors[:mid_point]
        second_half = recent_errors[mid_point:]
        
        if not first_half or not second_half:
            return {'trend': 'insufficient_data', 'change': 0}
        
        # Calculate error rates per time period
        first_half_rate = len(first_half)
        second_half_rate = len(second_half)
        
        change_percent = 0
        if first_half_rate > 0:
            change_percent = ((second_half_rate - first_half_rate) / first_half_rate) * 100
        
        if change_percent > 20:
            trend = 'increasing'
        elif change_percent < -20:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_percent': round(change_percent, 2),
            'recent_error_count': len(recent_errors)
        }
    
    def _get_common_error_patterns(self) -> List[Dict[str, Any]]:
        """Get most common error patterns."""
        sorted_patterns = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [
            {
                'pattern': pattern_key,
                'count': pattern_data['count'],
                'first_seen': pattern_data['first_seen'].isoformat(),
                'last_seen': pattern_data['last_seen'].isoformat(),
                'recent_message': pattern_data['messages'][-1] if pattern_data['messages'] else 'N/A'
            }
            for pattern_key, pattern_data in sorted_patterns[:5]  # Top 5 patterns
        ]
    
    def _get_system_health_indicators(self) -> Dict[str, Any]:
        """Get overall system health indicators."""
        total_errors = self.stats['total_errors']
        if total_errors == 0:
            return {'status': 'healthy', 'score': 100}
        
        # Calculate health score based on various factors
        recovery_rate = self.stats['recovered_errors'] / total_errors
        critical_error_rate = self.stats['by_severity'].get('critical', 0) / total_errors
        
        # Health score (0-100)
        health_score = 100
        health_score -= (critical_error_rate * 50)  # Critical errors heavily impact score
        health_score += (recovery_rate * 20)  # Good recovery improves score
        health_score = max(0, min(100, health_score))
        
        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'score': round(health_score, 1),
            'recovery_rate': round(recovery_rate * 100, 1),
            'critical_error_rate': round(critical_error_rate * 100, 1)
        }


# Global error manager instance
_error_manager = None


def get_error_manager() -> ErrorManager:
    """Get or create global error manager."""
    global _error_manager
    if _error_manager is None:
        from config.settings import get_settings
        settings = get_settings()
        error_log_path = Path(settings.storage_path) / "logs" / "errors.jsonl"
        _error_manager = ErrorManager(error_log_path)
    return _error_manager


def with_error_handling(
    operation_name: str = None,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    recoverable: bool = True,
    max_retries: int = 3,
    fallback_func: Callable = None,
    return_format: str = "standard"  # Issue #30: Unified response format
):
    """
    Enhanced decorator for automatic error handling and recovery with async support.
    
    Args:
        operation_name: Name for logging
        category: Error category
        recoverable: Whether errors are recoverable
        max_retries: Maximum retry attempts
        fallback_func: Fallback function to use
        return_format: Response format (standard, dict, status)
    """
    def decorator(func: Callable) -> Callable:
        
        # Issue #30: Support both sync and async functions
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = {
                    'operation': operation_name or func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'async_operation': True
                }
                
                error_manager = get_error_manager()
                
                try:
                    result = await func(*args, **kwargs)
                    return _format_response(result, return_format, success=True)
                except Exception as e:
                    handled_result = await error_manager.handle_error_async(e, func, context, *args, **kwargs)
                    return _format_response(handled_result, return_format, success=False, error=str(e))
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                context = {
                    'operation': operation_name or func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'async_operation': False
                }
                
                error_manager = get_error_manager()
                
                try:
                    result = func(*args, **kwargs)
                    return _format_response(result, return_format, success=True)
                except Exception as e:
                    handled_result = error_manager.handle_error(e, func, context, *args, **kwargs)
                    return _format_response(handled_result, return_format, success=False, error=str(e))
            
            return sync_wrapper
    
    return decorator


def safe_operation(func: Callable, *args, **kwargs) -> Any:
    """
    Execute operation with automatic error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if unrecoverable error
    """
    error_manager = get_error_manager()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = {
            'operation': func.__name__,
            'safe_operation': True
        }
        return error_manager.handle_error(e, func, context, *args, **kwargs)


# Convenience decorators for common patterns
def filesystem_operation(func: Callable) -> Callable:
    """Decorator for filesystem operations."""
    return with_error_handling(
        category=ErrorCategory.FILESYSTEM,
        recoverable=True,
        max_retries=3
    )(func)


def database_operation(func: Callable) -> Callable:
    """Decorator for database operations."""
    return with_error_handling(
        category=ErrorCategory.DATABASE,
        recoverable=True,
        max_retries=2
    )(func)


def network_operation(func: Callable) -> Callable:
    """Decorator for network operations.""" 
    return with_error_handling(
        category=ErrorCategory.NETWORK,
        recoverable=True,
        max_retries=3
    )(func)


def validation_operation(func: Callable) -> Callable:
    """Decorator for validation operations."""
    return with_error_handling(
        category=ErrorCategory.VALIDATION,
        recoverable=False
    )(func)


# Issue #30: Additional consistency decorators for unified error handling
def youtube_api_operation(func: Callable) -> Callable:
    """Decorator for YouTube API operations."""
    return with_error_handling(
        category=ErrorCategory.YOUTUBE_API,
        recoverable=True,
        max_retries=3,
        return_format="dict"
    )(func)


def whisper_ai_operation(func: Callable) -> Callable:
    """Decorator for Whisper AI operations."""
    return with_error_handling(
        category=ErrorCategory.WHISPER_AI,
        recoverable=True,
        max_retries=2,
        return_format="dict"
    )(func)


def worker_operation(func: Callable) -> Callable:
    """Decorator for worker operations."""
    return with_error_handling(
        category=ErrorCategory.WORKER_FAILURE,
        recoverable=True,
        max_retries=2,
        return_format="status"
    )(func)


def storage_sync_operation(func: Callable) -> Callable:
    """Decorator for storage sync operations."""
    return with_error_handling(
        category=ErrorCategory.STORAGE_SYNC,
        recoverable=True,
        max_retries=3,
        return_format="dict"
    )(func)


def rate_limited_operation(func: Callable) -> Callable:
    """Decorator for operations that may hit rate limits."""
    return with_error_handling(
        category=ErrorCategory.RATE_LIMITING,
        recoverable=True,
        max_retries=5,
        return_format="dict"
    )(func)


# Issue #30: Response formatting and consistency utilities
def _format_response(result: Any, return_format: str, success: bool, error: str = None) -> Any:
    """Format response according to specified format for consistency."""
    if return_format == "dict":
        return {
            "success": success,
            "data": result if success else None,
            "error": error if not success else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    elif return_format == "status":
        return {
            "status": "success" if success else "error",
            "result": result,
            "error_message": error if not success else None
        }
    else:  # standard format
        if success:
            return result
        else:
            return None if result is None else result


# Issue #30: Async convenience functions for consistency
async def safe_operation_async(func: Callable, *args, **kwargs) -> Any:
    """
    Execute async operation with automatic error handling.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if unrecoverable error
    """
    error_manager = get_error_manager()
    
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        context = {
            'operation': func.__name__,
            'safe_operation_async': True
        }
        return await error_manager.handle_error_async(e, func, context, *args, **kwargs)


def create_error_response(success: bool, data: Any = None, error: str = None, 
                         error_code: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized error response format for consistency across the system."""
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if success:
        response["data"] = data
        response["status"] = "success"
    else:
        response["error"] = {
            "message": error or "Unknown error occurred",
            "code": error_code or "UNKNOWN_ERROR"
        }
        response["status"] = "error"
        response["data"] = None
    
    if metadata:
        response["metadata"] = metadata
    
    return response


class StandardErrorResponse:
    """Standard error response class for consistent error handling across components."""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 error_code: str = None, metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.error = error
        self.error_code = error_code
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return create_error_response(
            success=self.success,
            data=self.data,
            error=self.error,
            error_code=self.error_code,
            metadata=self.metadata
        )
    
    def __str__(self) -> str:
        if self.success:
            return f"Success: {self.data}"
        else:
            return f"Error [{self.error_code}]: {self.error}"
    
    @classmethod
    def success_response(cls, data: Any, metadata: Dict[str, Any] = None) -> 'StandardErrorResponse':
        """Create successful response."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(cls, error: str, error_code: str = None, metadata: Dict[str, Any] = None) -> 'StandardErrorResponse':
        """Create error response."""
        return cls(success=False, error=error, error_code=error_code or "ERROR", metadata=metadata)