#!/usr/bin/env python3
"""
Production startup script for the OrchestratorWorker.

This script provides a production-ready way to run the orchestrator worker
as a service or daemon process with proper configuration, logging, and
graceful shutdown handling.

Usage:
    python run_orchestrator.py [options]

Options:
    --continuous          Run in continuous mode (default)
    --batch              Run in batch mode (process available jobs and exit)
    --max-jobs N         Maximum number of jobs to process
    --worker-id ID       Custom worker ID
    --max-concurrent N   Maximum concurrent jobs (default: 3)
    --polling-interval N Polling interval in seconds (default: 5.0)
    --log-level LEVEL    Logging level (default: INFO)
    --config-file FILE   Configuration file path
"""

import argparse
import asyncio
import logging
import signal
import sys
import os
import json
import gc
import time
import psutil
from pathlib import Path

# Run startup validation before anything else
from core.startup_validation import run_startup_validation
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from workers.orchestrator import OrchestratorWorker
from core.queue import get_job_queue
from config.settings import get_settings
from core.dynamic_worker_pool import get_dynamic_worker_pool
from core.dynamic_scaling_manager import get_dynamic_scaling_manager


class OrchestratorService:
    """Service wrapper for the OrchestratorWorker."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service with configuration."""
        self.config = config
        self.orchestrator: Optional[OrchestratorWorker] = None
        self.worker_pool = None
        self.dynamic_scaling = None
        self.running = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                # Add file handler if log_file is specified
                *([logging.FileHandler(self.config['log_file'])] 
                  if self.config.get('log_file') else [])
            ]
        )
        
        return logging.getLogger('orchestrator-service')
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Handle SIGUSR1 for status reporting
        def status_handler(signum, frame):
            if self.orchestrator:
                status = self.orchestrator.get_status_info()
                self.logger.info(f"Orchestrator Status: {json.dumps(status, indent=2)}")
        
        signal.signal(signal.SIGUSR1, status_handler)
    
    async def start(self) -> Dict[str, Any]:
        """Start the orchestrator service."""
        self.logger.info("Starting Orchestrator Service")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize dynamic worker pool
        self.worker_pool = get_dynamic_worker_pool()
        self.logger.info("Initialized dynamic worker pool")
        
        # Initialize TRUE dynamic scaling manager
        self.dynamic_scaling = get_dynamic_scaling_manager()
        self.dynamic_scaling.start_monitoring()
        self.logger.info("✅ TRUE Dynamic Scaling Manager started - monitors CPU, memory, load, and auto-adjusts capacity")
        
        # Get initial scaling parameters (dynamic instead of static)
        scaling_params = self.dynamic_scaling.get_current_scaling_params()
        initial_concurrent_jobs = scaling_params.max_concurrent_downloads
        
        # Create orchestrator with DYNAMIC parameters
        self.orchestrator = OrchestratorWorker(
            worker_id=self.config.get('worker_id'),
            max_concurrent_jobs=initial_concurrent_jobs,  # Dynamic from scaling manager
            polling_interval=self.config.get('polling_interval', 5.0),
            health_check_interval=self.config.get('health_check_interval', 60.0),
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 5.0),
            log_level=self.config.get('log_level', 'INFO')
        )
        
        self.logger.info(f"🎯 Dynamic scaling active: concurrent_jobs={initial_concurrent_jobs} (adjusts automatically based on system load)")
        
        self.running = True
        
        try:
            # Prepare execution input
            execution_input = {
                "continuous_mode": self.config.get('continuous_mode', True),
                "max_jobs": self.config.get('max_jobs'),
                "job_types": self.config.get('job_types', [])
            }
            
            self.logger.info(f"Starting orchestrator with input: {execution_input}")
            
            # Start both orchestrator and dynamic worker pool concurrently
            async def run_orchestrator():
                """Run orchestrator with memory management"""
                jobs_processed = 0
                memory_threshold_mb = 2048  # 2GB threshold
                
                # Check initial memory
                process = psutil.Process()
                initial_memory_mb = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"💾 Initial memory: {initial_memory_mb:.1f}MB")
                
                # Create a wrapper with TRUE dynamic scaling integration
                async def run_with_dynamic_scaling():
                    nonlocal jobs_processed
                    
                    # Update queue depth for scaling decisions
                    if hasattr(self.orchestrator, 'get_queue_size'):
                        queue_depth = self.orchestrator.get_queue_size()
                        self.dynamic_scaling.update_queue_depth(queue_depth)
                    
                    # Get current dynamic scaling parameters
                    scaling_params = self.dynamic_scaling.get_current_scaling_params()
                    
                    # Update orchestrator with new dynamic parameters if changed
                    if hasattr(self.orchestrator, 'update_scaling_params'):
                        self.orchestrator.update_scaling_params(scaling_params)
                    
                    # Execute with monitoring
                    start_time = time.time()
                    result = await self.orchestrator.execute(execution_input)
                    execution_time = time.time() - start_time
                    
                    # Record performance metrics for scaling decisions
                    self.dynamic_scaling.record_performance_metric('throughput', jobs_processed / max(execution_time, 1))
                    
                    # After execution, check memory (legacy monitoring)
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    self.logger.info(f"💾 Memory after execution: {current_memory_mb:.1f}MB")
                    
                    # Get scaling status for logging
                    scaling_status = self.dynamic_scaling.get_scaling_status()
                    self.logger.info(f"🎯 Dynamic scaling status: {scaling_status['current_state']} "
                                   f"(CPU: {scaling_status['system_metrics'].get('cpu_percent', 'N/A')}%, "
                                   f"concurrent: {scaling_status['scaling_params']['max_concurrent_downloads']})")
                    
                    # Legacy memory cleanup (now handled by dynamic scaling)
                    if current_memory_mb > memory_threshold_mb:
                        self.logger.warning(f"⚠️ High memory usage ({current_memory_mb:.1f}MB), running cleanup...")
                        gc.collect()
                        gc.collect(2)  # Force collection of highest generation
                        await asyncio.sleep(2)  # Give system time to reclaim memory
                        after_cleanup_mb = process.memory_info().rss / 1024 / 1024
                        self.logger.info(f"✅ Memory after cleanup: {after_cleanup_mb:.1f}MB")
                    
                    return result
                
                return await run_with_dynamic_scaling()
            
            async def run_worker_pool():
                if self.worker_pool:
                    await self.worker_pool.run_forever()
                return {"status": "worker_pool_stopped"}
            
            # Run both concurrently
            orchestrator_task = asyncio.create_task(run_orchestrator())
            worker_pool_task = asyncio.create_task(run_worker_pool())
            
            try:
                # Wait for either to complete
                done, pending = await asyncio.wait(
                    [orchestrator_task, worker_pool_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Get result from completed task
                completed_task = list(done)[0]
                result = await completed_task
                
                self.logger.info(f"Service completed: {result}")
                return result
                
            except asyncio.CancelledError:
                self.logger.info("Tasks cancelled")
                return {"status": "cancelled"}
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            return {"status": "interrupted", "reason": "keyboard_interrupt"}
            
        except Exception as e:
            self.logger.error(f"Orchestrator service failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
            
        finally:
            self.running = False
            
            # Shutdown worker pool
            if self.worker_pool:
                self.logger.info("Shutting down worker pool...")
                self.worker_pool.shutdown(graceful=True)
            
            # Shutdown dynamic scaling manager
            if self.dynamic_scaling:
                self.logger.info("Shutting down dynamic scaling manager...")
                self.dynamic_scaling.stop_monitoring()
            
            self.logger.info("Orchestrator service stopped")
    
    async def status(self) -> Dict[str, Any]:
        """Get current service status."""
        status_info = {
            "service_running": self.running,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.orchestrator:
            status_info["orchestrator_status"] = self.orchestrator.get_status_info()
            
        if self.worker_pool:
            status_info["worker_pool_status"] = self.worker_pool.get_status()
            
        try:
            queue_stats = await get_job_queue().get_queue_stats()
            status_info["queue_stats"] = queue_stats
        except Exception as e:
            status_info["queue_stats"] = {"error": str(e)}
            
        return status_info


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "continuous_mode": True,
        "max_jobs": None,
        "worker_id": None,
        "max_concurrent_jobs": 3,
        "polling_interval": 5.0,
        "health_check_interval": 60.0,
        "max_retries": 3,
        "retry_delay": 5.0,
        "log_level": "INFO",
        "job_types": []
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the YouTube Content Orchestrator Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run in continuous mode (default)
    python run_orchestrator.py
    
    # Process 100 jobs and exit
    python run_orchestrator.py --batch --max-jobs 100
    
    # Run with custom configuration
    python run_orchestrator.py --config-file orchestrator.json
    
    # Run with specific worker ID and log level
    python run_orchestrator.py --worker-id prod-worker-1 --log-level DEBUG
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--continuous',
        action='store_true',
        default=True,
        help='Run in continuous mode (default)'
    )
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Run in batch mode (process available jobs and exit)'
    )
    
    # Job configuration
    parser.add_argument(
        '--max-jobs',
        type=int,
        metavar='N',
        help='Maximum number of jobs to process'
    )
    parser.add_argument(
        '--job-types',
        nargs='+',
        metavar='TYPE',
        help='Specific job types to process (e.g., check_channel download_transcript)'
    )
    
    # Worker configuration
    parser.add_argument(
        '--worker-id',
        type=str,
        metavar='ID',
        help='Custom worker ID'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        metavar='N',
        help='Maximum concurrent jobs (default: 3)'
    )
    parser.add_argument(
        '--polling-interval',
        type=float,
        default=5.0,
        metavar='N',
        help='Polling interval in seconds (default: 5.0)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        metavar='FILE',
        help='Log to file in addition to stdout'
    )
    
    # Configuration file
    parser.add_argument(
        '--config-file',
        type=str,
        metavar='FILE',
        help='Load configuration from JSON file'
    )
    
    # Service commands
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status and exit'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        metavar='FILE',
        help='Create default configuration file and exit'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up basic logging for main function
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run startup validation
    logger.info("Running startup validation...")
    if not run_startup_validation(exit_on_error=True):
        logger.error("Startup validation failed. Exiting.")
        sys.exit(1)
    
    # Handle special commands
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default configuration file: {args.create_config}")
        return
    
    # Load configuration
    if args.config_file:
        config = load_config_file(args.config_file)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.batch:
        config['continuous_mode'] = False
    if args.max_jobs:
        config['max_jobs'] = args.max_jobs
    if args.worker_id:
        config['worker_id'] = args.worker_id
    if args.max_concurrent:
        config['max_concurrent_jobs'] = args.max_concurrent
    if args.polling_interval:
        config['polling_interval'] = args.polling_interval
    if args.log_level:
        config['log_level'] = args.log_level
    if args.log_file:
        config['log_file'] = args.log_file
    if args.job_types:
        config['job_types'] = args.job_types
    
    # Create and start service
    service = OrchestratorService(config)
    
    if args.status:
        # Show status and exit
        status = await service.status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Start the service
    result = await service.start()
    
    # Print final result
    print(f"\nOrchestrator Service Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Exit with appropriate code
    if result.get('status') == 'failed':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)