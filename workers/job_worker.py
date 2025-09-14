#!/usr/bin/env python3
"""
Independent Job Worker Process

This script runs as an independent subprocess spawned by the DynamicWorkerPool.
It pulls jobs from the queue and processes them based on its worker type.

Key Features:
- Runs as independent OS process (not thread)
- Pulls jobs from database queue
- Handles different worker types (download, transcribe, etc.)
- Graceful shutdown on signals
- Heartbeat to parent process
- Error recovery and retry logic
"""

import sys
import os
import argparse
import asyncio
import logging
import signal
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.queue import get_job_queue
from config.settings import get_settings
from workers.orchestrator import OrchestratorWorker
from workers.audio_downloader import AudioDownloadWorker
from workers.transcriber import TranscribeWorker
from workers.generator import GeneratorWorker
from workers.monitor import MonitorWorker

logger = logging.getLogger("job_worker")


class JobWorker:
    """
    Independent worker process that pulls and processes jobs from the queue
    """
    
    def __init__(self, worker_id: str, worker_type: str):
        """
        Initialize the job worker
        
        Args:
            worker_id: Unique identifier for this worker
            worker_type: Type of worker (download, transcribe, process, etc.)
        """
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.running = False
        self.jobs_processed = 0
        self.start_time = datetime.now()
        self.last_job_time = None
        self.current_job = None
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize job queue and orchestrator
        self.queue = get_job_queue()
        self.orchestrator = OrchestratorWorker()
        
        # Initialize worker based on type
        self.worker = self._create_worker()
        
        logger.info(f"🚀 JobWorker {worker_id} ({worker_type}) initialized")
        
    def _setup_logging(self):
        """Setup logging for this worker process"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - Worker[{self.worker_id}] - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
        
    def _create_worker(self):
        """Create the appropriate worker based on type"""
        if self.worker_type == "download":
            return AudioDownloadWorker()
        elif self.worker_type == "transcribe":
            return TranscribeWorker()
        elif self.worker_type == "process":
            return self.orchestrator  # Use orchestrator for processing
        elif self.worker_type == "monitor":
            return MonitorWorker()
        elif self.worker_type == "generator":
            return GeneratorWorker()
        else:
            logger.warning(f"Unknown worker type: {self.worker_type}, using AudioDownloadWorker")
            return AudioDownloadWorker()
            
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle SIGUSR1 for status reporting
        def status_handler(signum, frame):
            status = self.get_status()
            logger.info(f"Worker Status: {json.dumps(status, indent=2)}")
            
        signal.signal(signal.SIGUSR1, status_handler)
    
    async def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get the next job from the queue based on worker type
        
        Returns:
            Job dictionary if available, None otherwise
        """
        try:
            # Map worker types to job types
            job_type_mapping = {
                "download": "audio_download",
                "transcribe": "transcribe",
                "process": "orchestrate", 
                "monitor": "monitor",
                "generator": "content_generation"
            }
            
            job_type = job_type_mapping.get(self.worker_type, "audio_download")
            
            # Get job from queue using correct method name
            job_result = await self.queue.dequeue(self.worker_id)
            if job_result:
                job_id, job_type_from_queue, target_id, metadata = job_result
                job = {
                    'id': job_id,
                    'type': job_type_from_queue,
                    'target_id': target_id,
                    'metadata': metadata or {}
                }
            else:
                job = None
            
            if job:
                logger.info(f"📥 Got job {job.get('id', 'unknown')} of type {job_type}")
                self.current_job = job
                
            return job
            
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
    
    async def process_job(self, job: Dict[str, Any]) -> bool:
        """
        Process a single job using the appropriate worker
        
        Args:
            job: Job dictionary from the queue
            
        Returns:
            True if successful, False otherwise
        """
        job_id = job.get('id', 'unknown')
        job_type = job.get('type', 'unknown')
        
        try:
            logger.info(f"🔄 Processing job {job_id} ({job_type})")
            
            # Process the job using the worker's execute method (BaseWorker interface)
            if hasattr(self.worker, 'execute'):
                # Convert job format to input_data format expected by BaseWorker
                input_data = {
                    'job_id': job_id,
                    'job_type': job_type,
                    'target_id': job.get('target_id'),
                    **job.get('metadata', {})
                }
                result = await self.worker.execute(input_data) if asyncio.iscoroutinefunction(self.worker.execute) else self.worker.execute(input_data)
            else:
                # Fallback for workers without execute method
                result = {'success': False, 'error': 'Worker does not support execute method'}
            
            if result.get('success', False):
                logger.info(f"✅ Completed job {job_id}")
                self.jobs_processed += 1
                self.last_job_time = datetime.now()
                
                # Mark job as complete in queue
                await self.queue.complete(job_id, result)
                return True
                
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ Failed job {job_id}: {error}")
                
                # Mark job as failed in queue
                await self.queue.fail(job_id, error)
                return False
                
        except Exception as e:
            logger.error(f"Exception processing job {job_id}: {e}")
            await self.queue.fail(job_id, str(e))
            return False
            
        finally:
            self.current_job = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of this worker
        
        Returns:
            Status dictionary
        """
        runtime = datetime.now() - self.start_time
        
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'pid': os.getpid(),
            'running': self.running,
            'jobs_processed': self.jobs_processed,
            'runtime': str(runtime),
            'last_job_time': str(self.last_job_time) if self.last_job_time else None,
            'current_job': self.current_job.get('id') if self.current_job else None,
            'start_time': str(self.start_time)
        }
    
    async def heartbeat(self):
        """Send periodic heartbeat (could be implemented to communicate with parent)"""
        # For now, just log heartbeat
        logger.debug(f"💓 Heartbeat - Processed {self.jobs_processed} jobs")
    
    async def run_forever(self):
        """
        Main worker loop - runs forever pulling and processing jobs
        """
        self.running = True
        self._setup_signal_handlers()
        
        logger.info(f"🔄 Worker {self.worker_id} ({self.worker_type}) starting main loop")
        
        heartbeat_interval = 60  # Send heartbeat every 60 seconds
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Get next job
                job = await self.get_next_job()
                
                if job:
                    # Process the job
                    success = await self.process_job(job)
                    
                    if not success:
                        # Brief pause on failure to avoid rapid retries
                        await asyncio.sleep(1)
                        
                else:
                    # No jobs available, wait and try again
                    await asyncio.sleep(2)
                
                # Send periodic heartbeat
                if time.time() - last_heartbeat > heartbeat_interval:
                    await self.heartbeat()
                    last_heartbeat = time.time()
                    
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Brief pause to avoid rapid error loops
                await asyncio.sleep(5)
        
        logger.info(f"✅ Worker {self.worker_id} shutting down after processing {self.jobs_processed} jobs")
    
    async def run_single_job(self):
        """
        Process a single job and exit (useful for testing)
        """
        logger.info(f"Running single job mode for worker {self.worker_id}")
        
        job = await self.get_next_job()
        if job:
            success = await self.process_job(job)
            exit_code = 0 if success else 1
        else:
            logger.info("No jobs available")
            exit_code = 2
            
        sys.exit(exit_code)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Independent Job Worker Process")
    parser.add_argument("--type", required=True, choices=["download", "transcribe", "process", "monitor", "generator"],
                       help="Type of worker")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID")
    parser.add_argument("--single", action="store_true", help="Process single job and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create worker
    worker = JobWorker(args.worker_id, args.type)
    
    try:
        if args.single:
            await worker.run_single_job()
        else:
            await worker.run_forever()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())