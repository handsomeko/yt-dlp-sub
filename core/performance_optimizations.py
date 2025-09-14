"""
Performance Optimization Utilities
Provides streaming, caching, and batch operations for improved performance
"""

import asyncio
import hashlib
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from functools import lru_cache, wraps
from contextlib import asynccontextmanager, contextmanager
import logging
import uuid

# Optional dependency - graceful degradation
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    aiofiles = None
    HAS_AIOFILES = False

logger = logging.getLogger(__name__)


class DeadlockPrevention:
    """
    Deadlock prevention utility using lock ordering and timeout mechanisms.
    FIX Issue #38: Prevent circular lock dependencies.
    """
    
    # Global lock ordering - acquire locks in this order to prevent deadlocks
    LOCK_ORDER = {
        'global_cache': 1,
        'filename_sanitizer': 2, 
        'storage_paths': 3,
        'smart_cache': 4,
        'error_manager': 5,
        'connection_pool': 6
    }
    
    _active_locks = {}  # Track active locks per thread
    _lock_registry = {}  # Registry of all named locks
    _registry_lock = threading.RLock()
    
    @classmethod
    def register_lock(cls, name: str, lock: threading.RLock) -> None:
        """Register a named lock for deadlock prevention."""
        with cls._registry_lock:
            cls._lock_registry[name] = lock
            if name not in cls.LOCK_ORDER:
                # Assign a high order number for unregistered locks
                max_order = max(cls.LOCK_ORDER.values()) if cls.LOCK_ORDER else 0
                cls.LOCK_ORDER[name] = max_order + 1
                logger.warning(f"Unregistered lock '{name}' assigned order {cls.LOCK_ORDER[name]}")
    
    @classmethod
    @contextmanager
    def acquire_ordered(cls, name: str, lock: threading.RLock, timeout: float = 30.0):
        """
        Acquire lock with deadlock prevention using ordering and timeout.
        
        Args:
            name: Name of the lock (must be in LOCK_ORDER)
            lock: The actual lock object
            timeout: Maximum time to wait for lock acquisition
            
        Raises:
            RuntimeError: If deadlock is detected or timeout occurs
        """
        thread_id = threading.get_ident()
        current_order = cls.LOCK_ORDER.get(name, 999)
        
        # Check for lock ordering violation
        if thread_id in cls._active_locks:
            active_locks = cls._active_locks[thread_id]
            max_active_order = max(cls.LOCK_ORDER.get(lock_name, 0) for lock_name in active_locks)
            
            if current_order <= max_active_order:
                active_lock_names = ', '.join(active_locks)
                raise RuntimeError(
                    f"Lock ordering violation detected! Attempting to acquire '{name}' (order {current_order}) "
                    f"while holding locks with higher order: {active_lock_names}. "
                    f"This could cause deadlock."
                )
        
        # Attempt to acquire lock with timeout
        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock '{name}' within {timeout} seconds - potential deadlock")
        
        try:
            # Track active lock
            if thread_id not in cls._active_locks:
                cls._active_locks[thread_id] = set()
            cls._active_locks[thread_id].add(name)
            
            yield lock
            
        finally:
            # Release lock and update tracking
            lock.release()
            if thread_id in cls._active_locks:
                cls._active_locks[thread_id].discard(name)
                if not cls._active_locks[thread_id]:
                    del cls._active_locks[thread_id]


class FileStreamManager:
    """Manages streaming file operations to reduce memory usage."""
    
    @staticmethod
    async def stream_read_text(file_path: Path, chunk_size: int = 8192) -> AsyncIterator[str]:
        """Stream read text file in chunks to avoid loading large files into memory."""
        if not HAS_AIOFILES:
            # Fallback: Create proper async generator from sync operations
            import asyncio
            
            def _sync_read_chunks():
                """Synchronous generator for file chunks."""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    return
            
            # Convert sync generator to async generator
            for chunk in _sync_read_chunks():
                yield chunk
                # Yield control to event loop to maintain async behavior
                await asyncio.sleep(0)
            return
            
        # Use aiofiles for truly async operations
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return
    
    @staticmethod
    async def stream_write_text(file_path: Path, content_generator: AsyncIterator[str]) -> None:
        """Stream write text content from generator to avoid memory spikes."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not HAS_AIOFILES:
            # Fallback: Properly handle async generator with sync file operations
            import asyncio
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    async for chunk in content_generator:
                        f.write(chunk)
                        # Yield control to event loop periodically
                        await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Error writing file {file_path}: {e}")
                raise
            return
        
        # Use aiofiles for truly async operations
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                async for chunk in content_generator:
                    await f.write(chunk)
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise
    
    @staticmethod 
    async def stream_copy(source: Path, destination: Path, chunk_size: int = 64 * 1024) -> None:
        """Stream copy large files without loading into memory."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if not HAS_AIOFILES:
            # Fallback: Add async yielding to prevent event loop blocking
            import asyncio
            
            try:
                with open(source, 'rb') as src:
                    with open(destination, 'wb') as dst:
                        chunk_count = 0
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            dst.write(chunk)
                            
                            # Yield control to event loop every 10 chunks to prevent blocking
                            chunk_count += 1
                            if chunk_count % 10 == 0:
                                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Error copying {source} to {destination}: {e}")
                raise
            return
        
        # Use aiofiles for truly async operations
        try:
            async with aiofiles.open(source, 'rb') as src:
                async with aiofiles.open(destination, 'wb') as dst:
                    while True:
                        chunk = await src.read(chunk_size)
                        if not chunk:
                            break
                        await dst.write(chunk)
        except Exception as e:
            logger.error(f"Error copying {source} to {destination}: {e}")
            raise


class BatchFileOperations:
    """Batch file operations for improved I/O performance."""
    
    def __init__(self, max_batch_size: int = 100):
        self.max_batch_size = max_batch_size
        self._pending_operations = []
    
    async def add_copy_operation(self, source: Path, destination: Path) -> None:
        """Add file copy to batch."""
        self._pending_operations.append(('copy', source, destination))
        
        if len(self._pending_operations) >= self.max_batch_size:
            await self.execute_batch()
    
    async def add_delete_operation(self, file_path: Path) -> None:
        """Add file deletion to batch."""
        self._pending_operations.append(('delete', file_path))
        
        if len(self._pending_operations) >= self.max_batch_size:
            await self.execute_batch()
    
    async def execute_batch(self) -> None:
        """Execute all pending batch operations."""
        if not self._pending_operations:
            return
        
        # Group operations by type for efficiency
        copy_ops = []
        delete_ops = []
        
        for op in self._pending_operations:
            if op[0] == 'copy':
                copy_ops.append((op[1], op[2]))
            elif op[0] == 'delete':
                delete_ops.append(op[1])
        
        # Execute operations in parallel
        tasks = []
        
        # Batch copy operations
        for source, dest in copy_ops:
            tasks.append(FileStreamManager.stream_copy(source, dest))
        
        # Batch delete operations  
        for file_path in delete_ops:
            tasks.append(self._async_delete(file_path))
        
        if tasks:
            # Execute tasks and properly handle exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions that occurred
            failed_operations = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_operations += 1
                    # Determine operation type for better logging
                    if i < len(copy_ops):
                        op_type = "copy"
                        op_detail = f"{copy_ops[i][0]} -> {copy_ops[i][1]}"
                    else:
                        op_type = "delete" 
                        delete_index = i - len(copy_ops)
                        op_detail = str(delete_ops[delete_index])
                    
                    logger.error(f"Batch {op_type} operation failed for {op_detail}: {result}")
            
            if failed_operations > 0:
                logger.warning(f"Batch execution completed with {failed_operations} failures out of {len(tasks)} operations")
        
        # Clear pending operations
        self._pending_operations.clear()
        
        successful_ops = len(tasks) - failed_operations if 'failed_operations' in locals() else len(tasks)  
        logger.info(f"Executed batch file operations: {len(copy_ops)} copies, {len(delete_ops)} deletes ({successful_ops} successful)")
    
    async def _async_delete(self, file_path: Path) -> None:
        """Async file deletion."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")


class SmartCache:
    """Intelligent caching system with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._expiry_times = {}
        self._lock = threading.RLock()  # Thread safety for concurrent access
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        return key in self._expiry_times and time.time() > self._expiry_times[key]
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self._expiry_times.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries when cache is full."""
        if len(self._cache) < self.max_size:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all cache structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiry_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache or self._is_expired(key):
                self._remove_key(key)
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            
            current_time = time.time()
            
            self._cache[key] = value
            self._access_times[key] = current_time
            
            if ttl is None:
                ttl = self.default_ttl
            
            self._expiry_times[key] = current_time + ttl
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(1 for expiry in self._expiry_times.values() if current_time > expiry)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_entries': expired_count,
                'hit_rate': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
            }


# Thread-safe global cache instance with lazy initialization
_global_cache = None
_global_cache_lock = threading.RLock()

def _get_global_cache():
    """Get global cache instance with thread-safe lazy initialization."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            # Double-check locking pattern for thread safety
            if _global_cache is None:
                _global_cache = SmartCache()
    return _global_cache


def cached_async(ttl: int = 3600, key_func: Optional[callable] = None):
    """Decorator for caching async function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try cache first (use thread-safe getter)
            cache = _get_global_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class ConnectionPool:
    """Generic connection pool for external services."""
    
    def __init__(self, connection_factory, max_connections: int = 10, timeout: int = 30):
        self.connection_factory = connection_factory
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._current_connections = 0
        self._lock = asyncio.Lock()
    
    async def _create_connection(self):
        """Create a new connection."""
        return await self.connection_factory()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with context manager."""
        connection = None
        
        try:
            # Try to get existing connection from pool
            try:
                connection = await asyncio.wait_for(
                    self._pool.get(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # Create new connection if pool is empty and under limit
                async with self._lock:
                    if self._current_connections < self.max_connections:
                        connection = await self._create_connection()
                        self._current_connections += 1
                    else:
                        # Wait longer for connection to be available
                        connection = await self._pool.get()
            
            yield connection
            
        finally:
            if connection:
                # Return connection to pool
                try:
                    self._pool.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool is full, close the connection
                    if hasattr(connection, 'close'):
                        await connection.close()
                    self._current_connections -= 1
    
    async def close_all(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                if hasattr(connection, 'close'):
                    await connection.close()
                self._current_connections -= 1
            except asyncio.QueueEmpty:
                break


def get_cache() -> SmartCache:
    """Get global cache instance with thread safety."""
    return _get_global_cache()


def clear_cache():
    """Clear global cache with thread safety."""
    cache = _get_global_cache()
    cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Get global cache statistics with thread safety."""
    cache = _get_global_cache()
    return cache.stats()


# Thread-safe global resource registries for cleanup - FIX Issue #41: Resource leak prevention
_connection_pools: List[ConnectionPool] = []
_batch_operations: List[BatchFileOperations] = []
_resource_registry_lock = threading.RLock()

def register_connection_pool(pool: ConnectionPool) -> None:
    """Register a connection pool for automatic cleanup with thread safety."""
    with _resource_registry_lock:
        if pool not in _connection_pools:
            _connection_pools.append(pool)
            logger.debug(f"Registered connection pool for cleanup: {id(pool)}")

def register_batch_operations(batch_ops: BatchFileOperations) -> None:
    """Register batch operations for automatic cleanup with thread safety."""
    with _resource_registry_lock:
        if batch_ops not in _batch_operations:
            _batch_operations.append(batch_ops)  
            logger.debug(f"Registered batch operations for cleanup: {id(batch_ops)}")

def unregister_connection_pool(pool: ConnectionPool) -> None:
    """Unregister a connection pool to prevent cleanup."""
    with _resource_registry_lock:
        if pool in _connection_pools:
            _connection_pools.remove(pool)
            logger.debug(f"Unregistered connection pool: {id(pool)}")

def unregister_batch_operations(batch_ops: BatchFileOperations) -> None:
    """Unregister batch operations to prevent cleanup."""
    with _resource_registry_lock:
        if batch_ops in _batch_operations:
            _batch_operations.remove(batch_ops)
            logger.debug(f"Unregistered batch operations: {id(batch_ops)}")

async def cleanup_all_resources() -> None:
    """Clean up all registered resources with thread safety. Call on application shutdown."""
    logger.info("Starting comprehensive resource cleanup...")
    
    # Get snapshot of registries to avoid modification during iteration
    with _resource_registry_lock:
        connection_pools_snapshot = list(_connection_pools)
        batch_operations_snapshot = list(_batch_operations)
    
    # Close all connection pools
    for i, pool in enumerate(connection_pools_snapshot):
        try:
            await pool.close_all()
            logger.info(f"Connection pool {i+1}/{len(connection_pools_snapshot)} closed successfully")
        except Exception as e:
            logger.error(f"Error closing connection pool {i+1}: {e}")
    
    # Execute pending batch operations
    for i, batch_ops in enumerate(batch_operations_snapshot):
        try:
            await batch_ops.execute_batch()
            logger.info(f"Batch operations {i+1}/{len(batch_operations_snapshot)} completed successfully")
        except Exception as e:
            logger.error(f"Error executing batch operations {i+1}: {e}")
    
    # Clear global cache
    try:
        clear_cache()
        logger.info("Global cache cleared")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
    
    # Clear registries safely
    with _resource_registry_lock:
        _connection_pools.clear()
        _batch_operations.clear()
        logger.info(f"Cleared {len(connection_pools_snapshot)} connection pools and {len(batch_operations_snapshot)} batch operations from registries")
    
    # Clean up deadlock prevention tracking - FIX Issue #41: Prevent memory leaks
    try:
        DeadlockPrevention._active_locks.clear()
        DeadlockPrevention._lock_registry.clear()
        logger.info("Deadlock prevention tracking cleared")
    except Exception as e:
        logger.error(f"Error clearing deadlock prevention tracking: {e}")
    
    logger.info("Comprehensive resource cleanup completed")

import atexit
import signal
import asyncio

def _sync_cleanup():
    """Synchronous cleanup wrapper for atexit."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # Schedule cleanup for later if loop is running
            loop.create_task(cleanup_all_resources())
        else:
            # Run cleanup in new loop
            asyncio.run(cleanup_all_resources())
    except RuntimeError:
        # No event loop, create one
        try:
            asyncio.run(cleanup_all_resources())
        except Exception as e:
            logger.error(f"Failed to run async cleanup: {e}")

# Register cleanup handlers
atexit.register(_sync_cleanup)

def _signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, cleaning up resources...")
    _sync_cleanup()
    
# Register signal handlers for graceful shutdown  
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)