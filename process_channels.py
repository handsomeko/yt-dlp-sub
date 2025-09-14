#!/usr/bin/env python3
"""
Single Entry Point for Channel Processing

This script provides a unified interface for processing multiple YouTube channels
using the dynamic worker pool system. It can handle 4, 10, or 100+ channels
with automatic resource-based scaling.

Key Features:
- Process any number of channels (4, 10, 100+)
- Multiple input methods: command-line args, JSON config, text files
- Automatic job queuing and distribution
- Dynamic worker scaling based on system resources
- 24/7 autonomous operation
- Progress monitoring and status reporting
"""

import argparse
import asyncio
import json
import logging
import sys
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.url_parser import parse_youtube_url, URLType
from core.startup_validation import run_startup_validation
from core.downloader import create_downloader_with_settings

logger = logging.getLogger("process_channels")


# Removed duplicate functions - will use existing consolidated system
class ChannelProcessor:
    """
    Main controller for processing multiple channels using dynamic workers
    """
    
    def __init__(self):
        """Initialize the channel processor"""
        self.channels = []
        self.start_time = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🚀 ChannelProcessor initialized")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
    
    def _load_progress_state(self, progress_file: Path) -> dict:
        """Load progress state from file"""
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress state: {e}")
        return {'completed_channels': {}, 'failed_channels': {}}
    
    def _update_progress_state(self, progress_file: Path, state: dict, channel_id: str, status: str, details: dict = None):
        """Update and save progress state"""
        if status == 'started':
            state.setdefault('started_channels', {})[channel_id] = datetime.now().isoformat()
        elif status == 'completed':
            state.setdefault('completed_channels', {})[channel_id] = {
                'completed_at': datetime.now().isoformat(),
                'details': details or {}
            }
            # Remove from started if present
            state.get('started_channels', {}).pop(channel_id, None)
        elif status == 'failed':
            state.setdefault('failed_channels', {})[channel_id] = {
                'failed_at': datetime.now().isoformat(),
                'details': details or {}
            }
            # Remove from started if present
            state.get('started_channels', {}).pop(channel_id, None)
        
        # Save state to file
        try:
            with open(progress_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress state: {e}")
        
    def parse_channels_from_args(self, channels: List[str]) -> List[Dict[str, Any]]:
        """
        Parse channels from command line arguments
        
        Args:
            channels: List of channel URLs or handles
            
        Returns:
            List of parsed channel dictionaries
        """
        parsed_channels = []
        
        for channel_input in channels:
            try:
                # Parse the URL/handle
                url_type, identifier, metadata = parse_youtube_url(channel_input)
                
                if url_type == URLType.CHANNEL:
                    channel_info = {
                        'url': channel_input,
                        'channel_id': identifier,
                        'channel_name': metadata.get('channel_name') or metadata.get('handle') or identifier or 'Unknown',
                        'source': 'command_line'
                    }
                    parsed_channels.append(channel_info)
                    logger.info(f"✅ Parsed channel: {channel_info['channel_name']}")
                else:
                    logger.warning(f"⚠️ Not a channel URL: {channel_input}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to parse channel {channel_input}: {e}")
                
        return parsed_channels
        
    def parse_channels_from_config(self, config_path: str) -> List[Dict[str, Any]]:
        """
        Parse channels from JSON configuration file
        
        Args:
            config_path: Path to JSON config file
            
        Returns:
            List of parsed channel dictionaries
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            channels = config.get('channels', [])
            parsed_channels = []
            
            for channel_config in channels:
                if isinstance(channel_config, str):
                    # Simple URL string
                    url_type, identifier, metadata = parse_youtube_url(channel_config)
                    if url_type == URLType.CHANNEL:
                        parsed_channels.append({
                            'url': channel_config,
                            'channel_id': identifier,
                            'channel_name': metadata.get('channel_name') or metadata.get('handle') or identifier or 'Unknown',
                            'source': 'config_file'
                        })
                elif isinstance(channel_config, dict):
                    # Channel with additional config
                    url = channel_config.get('url', channel_config.get('channel'))
                    if url:
                        url_type, identifier, metadata = parse_youtube_url(url)
                        if url_type == URLType.CHANNEL:
                            channel_info = {
                                'url': url,
                                'channel_id': identifier,
                                'channel_name': channel_config.get('name') or metadata.get('channel_name') or metadata.get('handle') or identifier or 'Unknown',
                                'source': 'config_file',
                                'priority': channel_config.get('priority', 'normal'),
                                'limit': channel_config.get('limit'),
                                'translate': channel_config.get('translate', False),
                                'target_language': channel_config.get('target_language', 'en')
                            }
                            parsed_channels.append(channel_info)
                            
            logger.info(f"✅ Loaded {len(parsed_channels)} channels from config file")
            return parsed_channels
            
        except Exception as e:
            logger.error(f"❌ Failed to parse config file {config_path}: {e}")
            return []
            
    def parse_channels_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse channels from text file (one URL per line)
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of parsed channel dictionaries
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            channels = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    channels.append(line)
                    
            return self.parse_channels_from_args(channels)
            
        except Exception as e:
            logger.error(f"❌ Failed to parse file {file_path}: {e}")
            return []
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def _cleanup_memory(self):
        """Force garbage collection to free memory"""
        gc.collect()
        # Force collection of highest generation
        gc.collect(2)
    
    async def process_channels_sequentially(self, channels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process channels sequentially using existing YouTubeDownloader
        """
        results = []
        total_videos_downloaded = 0
        total_videos_failed = 0
        total_videos_skipped = 0
        
        # Load progress state
        progress_file = Path('channel_processing_progress.json')
        progress_state = self._load_progress_state(progress_file)
        
        # Memory monitoring
        memory_threshold_mb = 2048  # 2GB threshold
        initial_memory = self._check_memory_usage()
        
        logger.info(f"🚀 SEQUENTIAL CHANNEL PROCESSING")
        logger.info(f"📊 Processing {len(channels)} channels sequentially")
        logger.info(f"💾 Initial memory: {initial_memory['rss_mb']:.1f}MB ({initial_memory['percent']:.1f}%)")
        
        for i, channel in enumerate(channels, 1):
            try:
                channel_id = channel.get('channel_id', channel['url'])
                
                # Check if channel already completed
                if progress_state.get('completed_channels', {}).get(channel_id):
                    logger.info(f"⏭️  CHANNEL {i}/{len(channels)}: {channel['channel_name']} already completed")
                    results.append({'channel': channel['channel_name'], 'status': 'already_completed'})
                    continue
                
                logger.info(f"\n📺 CHANNEL {i}/{len(channels)}: {channel['channel_name']}")
                
                # Mark channel as started
                self._update_progress_state(progress_file, progress_state, channel_id, 'started')
                
                # Use factory function to respect environment settings
                # Override with channel-specific settings if provided
                downloader = create_downloader_with_settings()
                if channel.get('translate') is not None:
                    downloader.enable_translation = channel.get('translate')
                if channel.get('target_language'):
                    downloader.target_language = channel.get('target_language')
                
                # Download channel using existing consolidated system
                result = await downloader.download_channel_videos(
                    channel_url=channel['url'],
                    limit=channel.get('limit'),
                    max_concurrent=3,
                    download_audio_only=True,
                    audio_format='opus'
                )
                
                successful = len(result.get('successful', []))
                failed = len(result.get('failed', []))
                skipped = len(result.get('skipped', []))
                
                total_videos_downloaded += successful
                total_videos_failed += failed
                total_videos_skipped += skipped
                
                # Mark channel as completed
                self._update_progress_state(progress_file, progress_state, channel_id, 'completed', {
                    'successful': successful,
                    'failed': failed,
                    'skipped': skipped
                })
                
                results.append({
                    'channel': channel['channel_name'],
                    'successful': successful,
                    'failed': failed,
                    'skipped': skipped
                })
                
                logger.info(f"✅ Completed {channel['channel_name']}: {successful} downloaded, {skipped} skipped")
                
                # Check memory after each channel
                current_memory = self._check_memory_usage()
                logger.info(f"💾 Memory after channel: {current_memory['rss_mb']:.1f}MB ({current_memory['percent']:.1f}%)")
                
                # Cleanup if memory usage is high
                if current_memory['rss_mb'] > memory_threshold_mb:
                    logger.warning(f"⚠️ High memory usage detected ({current_memory['rss_mb']:.1f}MB), running cleanup...")
                    self._cleanup_memory()
                    await asyncio.sleep(2)  # Give system time to reclaim memory
                    after_cleanup = self._check_memory_usage()
                    logger.info(f"✅ Memory after cleanup: {after_cleanup['rss_mb']:.1f}MB")
                
                # Force cleanup every 5 channels to prevent gradual buildup
                if i % 5 == 0:
                    self._cleanup_memory()
                
            except Exception as e:
                logger.error(f"❌ Failed: {e}")
                results.append({'channel': channel['channel_name'], 'error': str(e)})
                # Mark channel as failed
                self._update_progress_state(progress_file, progress_state, channel_id, 'failed', {'error': str(e)})
                # Cleanup on error too
                self._cleanup_memory()
        
        # Final memory stats
        final_memory = self._check_memory_usage()
        memory_growth_mb = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        return {
            'total_videos_downloaded': total_videos_downloaded,
            'total_videos_failed': total_videos_failed,
            'total_videos_skipped': total_videos_skipped,
            'results': results,
            'memory_stats': {
                'initial_mb': initial_memory['rss_mb'],
                'final_mb': final_memory['rss_mb'],
                'growth_mb': memory_growth_mb,
                'final_percent': final_memory['percent']
            }
        }
    
    async def process_channels(self, 
                             args_channels: List[str] = None,
                             config_file: str = None,
                             text_file: str = None,
                             continuous: bool = True) -> Dict[str, Any]:
        """
        Main method to process channels from multiple input sources
        
        Args:
            args_channels: Channels from command line arguments
            config_file: Path to JSON config file
            text_file: Path to text file with channels
            continuous: Run continuously or batch mode
            
        Returns:
            Processing results
        """
        # Collect channels from all sources
        all_channels = []
        
        if args_channels:
            channels = self.parse_channels_from_args(args_channels)
            all_channels.extend(channels)
            
        if config_file:
            channels = self.parse_channels_from_config(config_file)
            all_channels.extend(channels)
            
        if text_file:
            channels = self.parse_channels_from_file(text_file)
            all_channels.extend(channels)
            
        if not all_channels:
            logger.error("❌ No valid channels found from any input source")
            return {'status': 'failed', 'error': 'No channels to process'}
            
        self.channels = all_channels
        self.start_time = datetime.now()
        
        logger.info(f"📺 Total channels to process: {len(all_channels)}")
        for channel in all_channels:
            logger.info(f"   - {channel['channel_name']} ({channel['url']})")
        
        # Process channels sequentially with actual downloads
        result = await self.process_channels_sequentially(all_channels)
        
        end_time = datetime.now()
        
        # Determine status
        if result['total_videos_downloaded'] == 0 and result['total_videos_failed'] > 0:
            status = 'failed'
        elif result['total_videos_failed'] > 0:
            status = 'partial'
        else:
            status = 'completed'
        
        return {
            'status': status,
            'channels_processed': len(all_channels),
            'videos_downloaded': result['total_videos_downloaded'],
            'videos_failed': result['total_videos_failed'],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': end_time.isoformat(),
            'duration': str(end_time - self.start_time) if self.start_time else None,
            'results': result['results']
        }


def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "channels": [
            {
                "url": "https://www.youtube.com/@逍遙健康指南",
                "name": "逍遙健康指南",
                "priority": "high",
                "translate": True,
                "target_language": "en",
                "limit": 10
            },
            {
                "url": "https://www.youtube.com/@health-k6s",
                "name": "Health Channel",
                "priority": "normal",
                "translate": True
            },
            "@healthdiary7",
            "@healthyeyes2"
        ],
        "settings": {
            "max_concurrent_workers": 10,
            "min_workers": 2,
            "continuous_mode": True,
            "log_level": "INFO"
        }
    }
    
    config_path = "channels_sample.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
        
    print(f"✅ Created sample config file: {config_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process multiple YouTube channels with dynamic scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process channels from command line
  python process_channels.py @channel1 @channel2 https://youtube.com/@channel3
  
  # Process from JSON config file
  python process_channels.py --config channels.json
  
  # Process from text file
  python process_channels.py --file channels.txt
  
  # Mix multiple input methods
  python process_channels.py @channel1 --config channels.json --file more_channels.txt
  
  # Batch mode (process once and exit)
  python process_channels.py --config channels.json --batch
  
  # Create sample config
  python process_channels.py --create-sample-config
        """
    )
    
    parser.add_argument('channels', nargs='*', help='YouTube channel URLs or handles')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--file', help='Text file with channels (one per line)')
    parser.add_argument('--batch', action='store_true', help='Process jobs and exit (default: continuous)')
    parser.add_argument('--create-sample-config', action='store_true', help='Create sample config file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create sample config if requested
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Run startup validation
    try:
        run_startup_validation()
        logger.info("✅ Startup validation passed")
    except Exception as e:
        logger.error(f"❌ Startup validation failed: {e}")
        sys.exit(1)
    
    # Check if any channels were provided
    if not args.channels and not args.config and not args.file:
        logger.error("❌ No channels specified. Use --help for usage examples.")
        sys.exit(1)
    
    # Create processor and run
    processor = ChannelProcessor()
    
    try:
        result = await processor.process_channels(
            args_channels=args.channels,
            config_file=args.config,
            text_file=args.file,
            continuous=not args.batch
        )
        
        # Print final result
        logger.info("🏁 Processing completed")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        # Exit with appropriate code
        exit_code = 0 if result.get('status') == 'completed' else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())