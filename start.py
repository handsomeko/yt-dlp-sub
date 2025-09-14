#!/usr/bin/env python3
"""
Startup script for YouTube Content Intelligence Platform
Initializes database and starts the orchestrator
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run startup validation before anything else
from core.startup_validation import run_startup_validation

# Validate at startup - exit if validation fails
print("🚨 Running startup validation...")
if not run_startup_validation(exit_on_error=True):
    print("❌ Startup validation failed. Please fix the issues and restart.")
    sys.exit(1)
print("✅ Startup validation passed")

from core.database import create_database
from workers.orchestrator import start_orchestrator
from config.settings import get_settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize the system: database, directories, etc."""
    logger.info("Initializing YouTube Content Intelligence Platform...")
    
    # Load settings
    settings = get_settings()
    logger.info(f"Deployment mode: {settings.deployment_mode}")
    logger.info(f"Storage path: {settings.storage_path}")
    
    # Create storage directories
    storage_path = Path(settings.storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage directory ready: {storage_path}")
    
    # Initialize database
    logger.info("Setting up database...")
    await create_database(settings.database_url)
    logger.info("Database initialized successfully")
    
    # Create subdirectories
    subdirs = ['audio', 'transcripts', 'content', 'metadata']
    for subdir in subdirs:
        (storage_path / subdir).mkdir(exist_ok=True)
    logger.info("Storage subdirectories created")
    
    return settings


async def main():
    """Main entry point"""
    try:
        # Initialize system
        settings = await initialize_system()
        
        # Print startup banner
        print("\n" + "="*60)
        print("YouTube Content Intelligence & Repurposing Platform")
        print("Phase 1: CLI Tool")
        print("="*60)
        print(f"Database: {settings.database_url}")
        print(f"Storage: {settings.storage_path}")
        print(f"Mode: {settings.deployment_mode}")
        print("="*60 + "\n")
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "worker":
                # Start orchestrator in continuous mode
                logger.info("Starting orchestrator in continuous mode...")
                result = await start_orchestrator(
                    continuous_mode=True,
                    max_concurrent_jobs=settings.worker_concurrency
                )
                logger.info(f"Orchestrator stopped: {result}")
            elif sys.argv[1] == "batch":
                # Run orchestrator in batch mode
                max_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                logger.info(f"Running orchestrator in batch mode (max {max_jobs} jobs)...")
                result = await start_orchestrator(
                    continuous_mode=False,
                    max_jobs=max_jobs,
                    max_concurrent_jobs=settings.worker_concurrency
                )
                logger.info(f"Batch processing complete: {result}")
            else:
                print("Usage:")
                print("  python start.py          # Show this help")
                print("  python start.py worker   # Start continuous worker")
                print("  python start.py batch [N]  # Process N jobs (default: 10)")
                print("\nFor CLI commands, use:")
                print("  python cli.py --help")
        else:
            print("System initialized successfully!")
            print("\nUsage:")
            print("  python start.py worker    # Start continuous worker")
            print("  python start.py batch [N]   # Process N jobs")
            print("\nFor CLI commands, use:")
            print("  python cli.py --help")
            print("\nQuick start:")
            print("  python cli.py add-channel https://www.youtube.com/@channelname")
            print("  python cli.py sync")
            print("  python cli.py search 'your query'")
            
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())