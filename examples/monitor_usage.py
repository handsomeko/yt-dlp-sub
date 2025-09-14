#!/usr/bin/env python3
"""
MonitorWorker Usage Example
Demonstrates how to use the MonitorWorker to check YouTube channels for new videos.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.monitor import MonitorWorker
from core.database import DatabaseManager, Channel


async def example_usage():
    """Example of how to use MonitorWorker in a real application."""
    print("📡 MonitorWorker Usage Example")
    print("=" * 40)
    
    # Initialize database
    db_manager = DatabaseManager("sqlite+aiosqlite:///data/example_monitor.db")
    await db_manager.initialize()
    
    try:
        # Add some channels to monitor
        print("\n1. Adding channels to monitor...")
        channels_to_add = [
            {
                "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",  # Google for Developers
                "name": "Google for Developers",
                "url": "https://www.youtube.com/@GoogleForDevelopers"
            },
            {
                "channel_id": "UCCezIgC97PvUuR4_gbFUs5g",  # Corey Schafer (Python tutorials)
                "name": "Corey Schafer",
                "url": "https://www.youtube.com/@coreyms"
            }
        ]
        
        async with db_manager.get_session() as session:
            for channel_info in channels_to_add:
                channel = Channel(
                    channel_id=channel_info["channel_id"],
                    channel_name=channel_info["name"],
                    channel_url=channel_info["url"],
                    is_active=True
                )
                session.add(channel)
            await session.flush()
            
        print(f"✅ Added {len(channels_to_add)} channels")
        
        # Initialize monitor worker
        print("\n2. Initializing MonitorWorker...")
        monitor = MonitorWorker(
            database_manager=db_manager,
            rate_limit_delay=1.0,  # 1 second between requests
            request_timeout=30.0,
            log_level="INFO"
        )
        print("✅ MonitorWorker ready")
        
        # Check all channels for new videos
        print("\n3. Checking all channels for new videos...")
        result = await monitor.run_async({"check_all": True})
        
        print("\nMonitoring Results:")
        print(f"  Status: {result['status']}")
        print(f"  Channels checked: {result['data']['channels_checked']}")
        print(f"  New videos found: {result['data']['new_videos_found']}")
        print(f"  Jobs created: {result['data']['jobs_created']}")
        print(f"  Processing time: {result['data']['processing_time']:.2f}s")
        
        if result['data']['errors']:
            print(f"  Errors: {result['data']['errors']}")
        else:
            print("  No errors occurred")
        
        # Check specific channel
        print("\n4. Checking specific channel...")
        result = await monitor.run_async({
            "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw"
        })
        
        print(f"  Single channel check - New videos: {result['data']['new_videos_found']}")
        print(f"  (Should be 0 since we just checked)")
        
        # Show what's in the database now
        print("\n5. Database contents:")
        from sqlalchemy import select, func
        from core.database import Video, Job
        
        async with db_manager.get_session() as session:
            # Count videos and jobs
            video_count = await session.execute(select(func.count(Video.id)))
            job_count = await session.execute(select(func.count(Job.id)))
            
            total_videos = video_count.scalar()
            total_jobs = job_count.scalar()
            
            print(f"  Total videos in database: {total_videos}")
            print(f"  Total jobs in queue: {total_jobs}")
            
            # Show recent videos
            if total_videos > 0:
                recent_videos = await session.execute(
                    select(Video.title, Video.video_id, Video.channel_id)
                    .order_by(Video.created_at.desc())
                    .limit(5)
                )
                
                print(f"\n  Recent videos discovered:")
                for video in recent_videos:
                    title = video.title[:60] + "..." if len(video.title) > 60 else video.title
                    print(f"    - {title} (ID: {video.video_id})")
        
        print("\n🎉 MonitorWorker example completed!")
        
    except Exception as e:
        print(f"\n❌ Error in example: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await db_manager.close()


async def scheduled_monitoring_example():
    """Example of how to run monitoring on a schedule."""
    print("\n🕒 Scheduled Monitoring Example")
    print("=" * 40)
    print("This would typically run as a background task or cron job")
    
    # In a real application, you might use:
    # - asyncio.create_task() for background tasks
    # - APScheduler for more complex scheduling
    # - Celery for distributed task queues
    
    # Simple scheduling loop example (not recommended for production)
    monitoring_interval = 300  # 5 minutes
    
    print(f"Monitoring every {monitoring_interval} seconds...")
    print("(This is just an example - press Ctrl+C to stop)")
    
    try:
        while True:
            print(f"\n⏰ Running scheduled check at {datetime.now()}")
            
            # Initialize fresh components for each run
            db_manager = DatabaseManager("sqlite+aiosqlite:///data/example_monitor.db")
            await db_manager.initialize()
            
            monitor = MonitorWorker(
                database_manager=db_manager,
                rate_limit_delay=2.0,  # Be more conservative with rate limiting
                log_level="INFO"
            )
            
            try:
                result = await monitor.run_async({"check_all": True})
                print(f"✅ Found {result['data']['new_videos_found']} new videos")
                
                if result['data']['errors']:
                    print(f"⚠️  Errors: {result['data']['errors']}")
                    
            except Exception as e:
                print(f"❌ Monitoring failed: {e}")
                
            finally:
                await db_manager.close()
            
            print(f"💤 Sleeping for {monitoring_interval} seconds...")
            await asyncio.sleep(monitoring_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Scheduled monitoring stopped by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MonitorWorker Usage Examples")
    parser.add_argument(
        "--scheduled", 
        action="store_true",
        help="Run scheduled monitoring example (use Ctrl+C to stop)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.scheduled:
            asyncio.run(scheduled_monitoring_example())
        else:
            asyncio.run(example_usage())
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Example stopped by user")
    except Exception as e:
        print(f"\n\n❌ Example failed: {e}")
        sys.exit(1)