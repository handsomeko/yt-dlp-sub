#!/usr/bin/env python3
"""
Simple script to add channel_url.txt files to existing channel directories.
Works directly with the filesystem without database dependency.
"""

import json
from pathlib import Path
from core.storage_paths_v2 import get_storage_paths_v2


def add_channel_urls():
    """Add channel_url.txt to all existing channel directories"""
    
    storage_paths = get_storage_paths_v2()
    base_path = storage_paths.base_path
    
    added = 0
    skipped = 0
    errors = 0
    
    print(f"Base path: {base_path}")
    print("=" * 50)
    
    # Iterate through all channel directories
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
        
        # Skip non-channel directories (like .DS_Store, etc)
        if channel_dir.name.startswith('.'):
            continue
        
        channel_id = channel_dir.name
        channel_url_file = channel_dir / 'channel_url.txt'
        
        # Skip if already has channel_url.txt
        if channel_url_file.exists():
            print(f"✓ Already has channel_url.txt: {channel_id}")
            skipped += 1
            continue
        
        # Try to get channel URL from .channel_info.json
        channel_info_file = channel_dir / '.channel_info.json'
        channel_url = None
        
        if channel_info_file.exists():
            try:
                with open(channel_info_file, 'r') as f:
                    channel_info = json.load(f)
                    channel_url = channel_info.get('channel_url')
                    channel_name = channel_info.get('channel_name', 'Unknown')
            except Exception as e:
                print(f"⚠️  Could not read channel info for {channel_id}: {e}")
                channel_name = channel_id
        else:
            channel_name = channel_id
        
        # If no URL found in metadata, construct from channel_id
        if not channel_url:
            if channel_id.startswith('UC'):
                # Standard YouTube channel ID format
                channel_url = f'https://www.youtube.com/channel/{channel_id}'
            else:
                # Might be a handle/username
                channel_url = f'https://www.youtube.com/@{channel_id}'
        
        # Write the channel_url.txt file
        try:
            with open(channel_url_file, 'w') as f:
                f.write(channel_url)
            
            print(f"✓ Added channel_url.txt: {channel_name} ({channel_id})")
            print(f"  URL: {channel_url}")
            added += 1
        except Exception as e:
            print(f"✗ Error writing channel_url.txt for {channel_name}: {e}")
            errors += 1
    
    print()
    print("=" * 50)
    print(f"Summary:")
    print(f"  Added: {added}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {added + skipped + errors}")
    print("=" * 50)


if __name__ == "__main__":
    print("Adding channel_url.txt files to existing channel directories...")
    print()
    add_channel_urls()