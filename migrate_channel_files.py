#!/usr/bin/env python3
"""
Migration script to add {channel_title}.txt and {channel_handle}.txt files
to all existing channel directories.
"""

import json
import sys
from pathlib import Path

def sanitize_filename(name: str) -> str:
    """Sanitize filename for filesystem safety"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Limit length
    name = name[:200] if len(name) > 200 else name
    return name.strip()

def main():
    # Get storage path from environment or use default
    import os
    storage_path = os.getenv('STORAGE_PATH', '/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads')
    base_path = Path(storage_path)
    
    if not base_path.exists():
        print(f"❌ Storage path does not exist: {base_path}")
        return 1
    
    print(f"📂 Processing channels in: {base_path}")
    
    created_files = []
    skipped_channels = []
    errors = []
    
    # Iterate through all channel directories
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
            
        # Skip hidden directories
        if channel_dir.name.startswith('.'):
            continue
            
        # Look for .channel_info.json
        channel_info_file = channel_dir / '.channel_info.json'
        if not channel_info_file.exists():
            skipped_channels.append(f"{channel_dir.name} - no .channel_info.json")
            continue
        
        try:
            with open(channel_info_file, 'r', encoding='utf-8') as f:
                channel_info = json.load(f)
            
            # Create {channel_title}.txt
            channel_name = channel_info.get('channel_name', '')
            if channel_name:
                safe_channel_name = sanitize_filename(channel_name)
                channel_title_file = channel_dir / f'{safe_channel_name}.txt'
                
                if not channel_title_file.exists():
                    # Prepare content for the title file
                    content_lines = [
                        channel_name,
                        '=' * len(channel_name),
                        f"Channel ID: {channel_info.get('channel_id', 'Unknown')}",
                        f"Channel URL: {channel_info.get('channel_url', 'Unknown')}",
                        f"Videos: {channel_info.get('total_videos', 0)}",
                        ""
                    ]
                    
                    with open(channel_title_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(content_lines) + '\n')
                    
                    created_files.append(f"✅ Created {channel_title_file.name} in {channel_dir.name}")
            
            # Create {@handle}.txt
            channel_handle = channel_info.get('uploader_id', '')
            if channel_handle:
                # Ensure handle starts with @ for consistency
                if not channel_handle.startswith('@'):
                    channel_handle = f'@{channel_handle}'
                
                # Keep @ symbol in filename - it's the identifier
                channel_handle_file = channel_dir / f'{channel_handle}.txt'
                
                if not channel_handle_file.exists():
                    # Simple content - the handle with @
                    with open(channel_handle_file, 'w', encoding='utf-8') as f:
                        f.write(channel_handle + '\n')
                    
                    created_files.append(f"✅ Created {channel_handle_file.name} in {channel_dir.name}")
                    
        except Exception as e:
            errors.append(f"❌ Error processing {channel_dir.name}: {e}")
    
    # Print summary
    print(f"\n📊 Migration Summary:")
    print(f"   Created files: {len(created_files)}")
    print(f"   Skipped channels: {len(skipped_channels)}")
    print(f"   Errors: {len(errors)}")
    
    if created_files:
        print(f"\n✅ Created Files:")
        for msg in created_files[:10]:  # Show first 10
            print(f"   {msg}")
        if len(created_files) > 10:
            print(f"   ... and {len(created_files) - 10} more")
    
    if errors:
        print(f"\n❌ Errors:")
        for msg in errors:
            print(f"   {msg}")
    
    print(f"\n✨ Migration complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())