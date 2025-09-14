#!/usr/bin/env python3
"""
Add channel_url.txt files to existing channel directories that don't have them.
Reads the URL from the database or constructs it from channel_id.
"""

import asyncio
import json
from pathlib import Path
from core.database import db_manager, Channel
from sqlalchemy import select
from core.storage_paths_v2 import get_storage_paths_v2


async def add_channel_urls():
    """Add channel_url.txt to all existing channel directories"""
    
    # Initialize database with proper async support
    import os
    os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///data.db'
    await db_manager.initialize()
    storage_paths = get_storage_paths_v2()
    
    # Base downloads directory
    base_path = storage_paths.base_path
    
    added = 0
    skipped = 0
    errors = 0
    
    print(f"Base path: {base_path}")
    print("=" * 50)
    
    # Get all channels from database
    async with db_manager.get_session() as session:
        result = await session.execute(select(Channel))
        channels = result.scalars().all()
        
        print(f"Found {len(channels)} channels in database")
        print()
        
        for channel in channels:
            channel_id = channel.channel_id
            channel_url = channel.channel_url
            channel_name = channel.channel_name
            
            # Get channel directory
            channel_dir = storage_paths.get_channel_dir(channel_id)
            
            # Skip if directory doesn't exist
            if not channel_dir.exists():
                print(f"⏭️  No directory for channel: {channel_name} ({channel_id})")
                skipped += 1
                continue
            
            channel_url_file = channel_dir / 'channel_url.txt'
            
            # Skip if already exists
            if channel_url_file.exists():
                print(f"✓ Already has channel_url.txt: {channel_name}")
                skipped += 1
                continue
            
            # Get channel URL from database or construct it
            if not channel_url:
                # Try to construct from channel_id
                if channel_id.startswith('UC'):
                    # Standard channel ID format
                    channel_url = f'https://www.youtube.com/channel/{channel_id}'
                else:
                    # Might be a handle/username
                    channel_url = f'https://www.youtube.com/@{channel_id}'
            
            if channel_url:
                # Write the channel_url.txt file
                try:
                    with open(channel_url_file, 'w') as f:
                        f.write(channel_url)
                    
                    print(f"✓ Added channel_url.txt: {channel_name}")
                    print(f"  URL: {channel_url}")
                    added += 1
                except Exception as e:
                    print(f"✗ Error writing channel_url.txt for {channel_name}: {e}")
                    errors += 1
            else:
                print(f"⚠️  No URL available for channel: {channel_name}")
                errors += 1
    
    # Also check for any channel directories that exist but aren't in the database
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
        
        # Skip non-channel directories (like .DS_Store, etc)
        if channel_dir.name.startswith('.'):
            continue
        
        channel_url_file = channel_dir / 'channel_url.txt'
        
        # Skip if already has channel_url.txt
        if channel_url_file.exists():
            continue
        
        # Check if this channel exists in database
        channel_id = channel_dir.name
        
        # Try to get channel info from .channel_info.json if it exists
        channel_info_file = channel_dir / '.channel_info.json'
        if channel_info_file.exists():
            try:
                with open(channel_info_file, 'r') as f:
                    channel_info = json.load(f)
                    channel_url = channel_info.get('channel_url')
                    
                    if channel_url and not channel_url_file.exists():
                        with open(channel_url_file, 'w') as f:
                            f.write(channel_url)
                        print(f"✓ Added channel_url.txt from metadata: {channel_id}")
                        print(f"  URL: {channel_url}")
                        added += 1
            except Exception as e:
                print(f"⚠️  Could not read channel info for {channel_id}: {e}")
    
    print()
    print("=" * 50)
    print(f"Summary:")
    print(f"  Added: {added}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {added + skipped + errors}")
    print("=" * 50)


if __name__ == "__main__":
    print("Adding channel_url.txt files to channel directories...")
    print()
    asyncio.run(add_channel_urls())