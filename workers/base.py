"""
Base worker class for the YouTube Content Intelligence & Repurposing Platform.

This module provides the abstract base class that all workers must inherit from,
ensuring consistent error handling, logging, and execution patterns across the system.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager


class WorkerStatus(Enum):
    """Status constants for worker execution results."""
    SUCCESS = "success"
    FAILED = "failed" 
    RETRY = "retry"
    SKIPPED = "skipped"


class BaseWorker(ABC):
    """
    Abstract base class for all workers in the system.
    
    Provides common functionality for logging, error handling, retries,
    and execution tracking while enforcing a consistent interface.
    
    Attributes:
        name: Human-readable name for the worker
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
        logger: Configured logger instance
    """
    
    def __init__(
        self, 
        name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        log_level: str = "INFO"
    ) -> None:
        """
        Initialize the base worker.
        
        Args:
            name: Human-readable name for this worker
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds  
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = self._setup_logger(log_level)
        self._execution_start_time: Optional[float] = None
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """
        Set up logger with consistent formatting for this worker.
        
        Args:
            log_level: Logging level as string
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"worker.{self.name}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.setLevel(getattr(logging, log_level.upper()))
        return logger
    
    @contextmanager
    def _execution_timer(self):
        """Context manager to track execution time."""
        self._execution_start_time = time.time()
        try:
            yield
        finally:
            if self._execution_start_time:
                execution_time = time.time() - self._execution_start_time
                self.log_with_context(
                    f"Execution completed in {execution_time:.2f}s",
                    level="INFO"
                )
    
    def log_with_context(
        self, 
        message: str, 
        level: str = "INFO",
        extra_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log message with worker context and optional additional context.
        
        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            extra_context: Additional context to include in log
        """
        context_msg = f"[{self.name}] {message}"
        
        if extra_context:
            context_parts = [f"{k}={v}" for k, v in extra_context.items()]
            context_msg += f" | Context: {', '.join(context_parts)}"
            
        log_method = getattr(self.logger, level.lower())
        log_method(context_msg)
    
    def retry_with_backoff(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        backoff_multiplier: float = 2.0,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_retries: Override default max_retries for this call
            retry_delay: Override default retry_delay for this call
            backoff_multiplier: Multiplier for exponential backoff
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result on success
            
        Raises:
            Exception: Last exception encountered after all retries exhausted
        """
        max_attempts = max_retries if max_retries is not None else self.max_retries
        delay = retry_delay if retry_delay is not None else self.retry_delay
        
        last_exception = None
        
        for attempt in range(max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.log_with_context(
                        f"Operation succeeded on attempt {attempt + 1}",
                        level="INFO"
                    )
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_attempts:
                    self.log_with_context(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s...",
                        level="WARNING",
                        extra_context={"attempt": attempt + 1, "max_attempts": max_attempts}
                    )
                    time.sleep(delay)
                    delay *= backoff_multiplier
                else:
                    self.log_with_context(
                        f"All {max_attempts + 1} attempts failed. Last error: {str(e)}",
                        level="ERROR"
                    )
        
        # Re-raise the last exception after all retries exhausted
        raise last_exception
    
    def get_execution_time(self) -> Optional[float]:
        """
        Get current execution time if timer is active.
        
        Returns:
            Execution time in seconds, or None if timer not started
        """
        if self._execution_start_time:
            return time.time() - self._execution_start_time
        return None
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method with built-in error handling and timing.
        
        Args:
            input_data: Input data for the worker
            
        Returns:
            Standardized result dictionary with status, data, and metadata
        """
        self.log_with_context("Starting execution", extra_context={"input_keys": list(input_data.keys())})
        
        try:
            with self._execution_timer():
                # Validate input
                if not self.validate_input(input_data):
                    return self._create_result(
                        status=WorkerStatus.FAILED,
                        error="Input validation failed",
                        input_data=input_data
                    )
                
                # Execute main logic
                result = self.execute(input_data)
                
                # Ensure result follows expected format
                if not isinstance(result, dict):
                    self.log_with_context("Worker returned non-dict result, wrapping", level="WARNING")
                    result = {"data": result}
                
                return self._create_result(
                    status=WorkerStatus.SUCCESS,
                    data=result,
                    input_data=input_data
                )
                
        except Exception as e:
            self.log_with_context(f"Execution failed: {str(e)}", level="ERROR")
            
            try:
                error_result = self.handle_error(e)
                return self._create_result(
                    status=WorkerStatus.FAILED,
                    error=str(e),
                    error_details=error_result,
                    input_data=input_data
                )
            except Exception as handler_error:
                self.log_with_context(
                    f"Error handler also failed: {str(handler_error)}", 
                    level="CRITICAL"
                )
                return self._create_result(
                    status=WorkerStatus.FAILED,
                    error=f"Primary error: {str(e)}. Handler error: {str(handler_error)}",
                    input_data=input_data
                )
    
    def _create_result(
        self,
        status: WorkerStatus,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized result dictionary.
        
        Args:
            status: Execution status
            data: Result data on success
            error: Error message on failure
            error_details: Additional error context
            input_data: Original input data for context
            
        Returns:
            Standardized result dictionary
        """
        result = {
            "status": status.value,
            "worker": self.name,
            "timestamp": time.time(),
            "execution_time": self.get_execution_time()
        }
        
        if data is not None:
            result["data"] = data
            
        if error is not None:
            result["error"] = error
            
        if error_details is not None:
            result["error_details"] = error_details
            
        if input_data is not None:
            result["input_summary"] = {
                "keys": list(input_data.keys()),
                "size": len(str(input_data))
            }
            
        return result
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before execution.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    @abstractmethod 
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main worker logic.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Execution result data
            
        Raises:
            Exception: On execution failure
        """
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle execution errors and return error context.
        
        Args:
            error: Exception that occurred during execution
            
        Returns:
            Error handling result with context and recovery information
        """
        pass