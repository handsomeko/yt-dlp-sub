"""
System-wide Concurrency Manager

Manages concurrent downloads across ALL processes to prevent system overload.
Uses file-based locking for inter-process coordination.
"""

import os
import json
import time
import psutil
import fcntl
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SystemConcurrencyManager:
    """
    Manages system-wide concurrency using file-based coordination.
    
    This ensures that even when multiple CLI processes are running,
    the total system load stays within safe limits.
    """
    
    def __init__(self, 
                 max_system_downloads: int = 3,
                 max_load_average: float = 4.0,
                 lock_dir: Optional[Path] = None):
        """
        Initialize the system-wide concurrency manager.
        
        Args:
            max_system_downloads: Maximum downloads across ALL processes
            max_load_average: Maximum system load average before throttling
            lock_dir: Directory for lock files (default: /tmp/yt-dl-sub-locks)
        """
        self.max_system_downloads = max_system_downloads
        self.max_load_average = max_load_average
        self.lock_dir = lock_dir or Path("/tmp/yt-dl-sub-locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # Files for coordination
        self.state_file = self.lock_dir / "system_state.json"
        self.state_lock = self.lock_dir / "system_state.lock"
        self.download_locks_dir = self.lock_dir / "downloads"
        self.download_locks_dir.mkdir(parents=True, exist_ok=True)
        
        # Process ID for this instance
        self.pid = os.getpid()
        self.hostname = os.uname().nodename
        
    def _read_state(self) -> Dict[str, Any]:
        """Read the current system state."""
        if not self.state_file.exists():
            return {
                "active_downloads": [],
                "last_updated": None,
                "total_active": 0
            }
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {
                "active_downloads": [],
                "last_updated": None,
                "total_active": 0
            }
    
    def _write_state(self, state: Dict[str, Any]):
        """Write the system state."""
        state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _clean_stale_locks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove locks from processes that no longer exist."""
        active_downloads = []
        
        for download in state.get("active_downloads", []):
            lock_file = Path(download["lock_file"])
            
            # Check if lock file still exists
            if not lock_file.exists():
                continue
                
            # Check if process still exists
            try:
                if download["hostname"] == self.hostname:
                    # Local process - check if PID exists
                    if psutil.pid_exists(download["pid"]):
                        active_downloads.append(download)
                    else:
                        # Process dead, remove lock
                        lock_file.unlink(missing_ok=True)
                        logger.info(f"Cleaned stale lock from PID {download['pid']}")
                else:
                    # Remote process - check file age (assume dead after 10 minutes)
                    lock_age = datetime.now() - datetime.fromisoformat(download["started"])
                    if lock_age < timedelta(minutes=10):
                        active_downloads.append(download)
                    else:
                        lock_file.unlink(missing_ok=True)
                        logger.info(f"Cleaned stale remote lock from {download['hostname']}")
            except Exception as e:
                logger.warning(f"Error checking lock {lock_file}: {e}")
                # Be conservative - keep the lock if we can't verify
                active_downloads.append(download)
        
        state["active_downloads"] = active_downloads
        state["total_active"] = len(active_downloads)
        return state
    
    def check_system_load(self) -> bool:
        """Check if system load is acceptable for new downloads."""
        try:
            # Get load average (1-minute)
            load_avg = os.getloadavg()[0]
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Check thresholds
            if load_avg > self.max_load_average:
                logger.warning(f"System load too high: {load_avg:.2f} > {self.max_load_average}")
                return False
            
            if memory.percent > 90:
                logger.warning(f"Memory usage too high: {memory.percent:.1f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system load: {e}")
            # Be conservative - allow if we can't check
            return True
    
    @contextmanager
    def acquire_download_slot(self, channel_name: str, timeout: float = 60.0):
        """
        Acquire a download slot or wait until one is available.
        
        Args:
            channel_name: Name of the channel being downloaded
            timeout: Maximum time to wait for a slot (seconds)
            
        Yields:
            True if slot acquired, raises TimeoutError if timeout
        """
        start_time = time.time()
        lock_file = self.download_locks_dir / f"{self.pid}_{int(time.time())}.lock"
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for download slot after {timeout}s")
            
            # Acquire state lock
            with open(self.state_lock, 'w') as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                
                try:
                    # Read current state
                    state = self._read_state()
                    
                    # Clean stale locks
                    state = self._clean_stale_locks(state)
                    
                    # Check if we can acquire a slot
                    if state["total_active"] < self.max_system_downloads:
                        # Check system load
                        if not self.check_system_load():
                            logger.info("System load too high, waiting...")
                            time.sleep(5)
                            continue
                        
                        # Acquire slot
                        lock_file.touch()
                        state["active_downloads"].append({
                            "pid": self.pid,
                            "hostname": self.hostname,
                            "channel": channel_name,
                            "lock_file": str(lock_file),
                            "started": datetime.now().isoformat()
                        })
                        state["total_active"] = len(state["active_downloads"])
                        
                        # Write updated state
                        self._write_state(state)
                        
                        logger.info(f"Acquired download slot ({state['total_active']}/{self.max_system_downloads}) for {channel_name}")
                        
                        # Release state lock
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        
                        # Yield control with the slot acquired
                        try:
                            yield True
                        finally:
                            # Release the download slot
                            self._release_slot(lock_file)
                        
                        return
                    
                    else:
                        # No slots available
                        logger.info(f"All {self.max_system_downloads} slots in use, waiting...")
                        
                finally:
                    # Release state lock
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            
            # Wait before retrying
            time.sleep(2)
    
    def _release_slot(self, lock_file: Path):
        """Release a download slot."""
        with open(self.state_lock, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            try:
                # Read current state
                state = self._read_state()
                
                # Remove this download
                state["active_downloads"] = [
                    d for d in state["active_downloads"] 
                    if d["lock_file"] != str(lock_file)
                ]
                state["total_active"] = len(state["active_downloads"])
                
                # Write updated state
                self._write_state(state)
                
                # Remove lock file
                lock_file.unlink(missing_ok=True)
                
                logger.info(f"Released download slot ({state['total_active']}/{self.max_system_downloads})")
                
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system concurrency status."""
        with open(self.state_lock, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            try:
                state = self._read_state()
                state = self._clean_stale_locks(state)
                
                # Add system metrics
                state["system_load"] = os.getloadavg()[0]
                state["memory_percent"] = psutil.virtual_memory().percent
                state["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                state["max_allowed"] = self.max_system_downloads
                
                return state
                
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    def wait_for_slot(self, channel_name: str, check_interval: float = 5.0) -> bool:
        """
        Wait until a download slot is available.
        
        Args:
            channel_name: Channel to download
            check_interval: How often to check (seconds)
            
        Returns:
            True when slot is available
        """
        while True:
            status = self.get_status()
            
            if status["total_active"] < self.max_system_downloads:
                if self.check_system_load():
                    return True
                else:
                    logger.info(f"System load too high (load: {status['system_load']:.2f}, mem: {status['memory_percent']:.1f}%)")
            else:
                logger.info(f"Waiting for slot ({status['total_active']}/{self.max_system_downloads} active)")
                
                # Show what's currently downloading
                for download in status["active_downloads"]:
                    logger.info(f"  - {download['channel']} (PID: {download['pid']})")
            
            time.sleep(check_interval)


# Global instance
_manager = None

def get_system_concurrency_manager() -> SystemConcurrencyManager:
    """Get the global system concurrency manager."""
    global _manager
    if _manager is None:
        # Get settings from environment
        max_downloads = int(os.getenv("MAX_SYSTEM_DOWNLOADS", "3"))
        max_load = float(os.getenv("MAX_LOAD_AVERAGE", "4.0"))
        
        _manager = SystemConcurrencyManager(
            max_system_downloads=max_downloads,
            max_load_average=max_load
        )
    return _manager