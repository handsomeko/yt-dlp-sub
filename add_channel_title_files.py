#!/usr/bin/env python3
"""
Add {channel_title}.txt files to each channel directory.
This provides quick human-readable channel identification.
"""

import json
import os
import sys
from pathlib import Path
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.storage_paths_v2 import get_storage_paths_v2


def sanitize_channel_title(title):
    """Sanitize channel title for use as filename."""
    # Ensure title is a string
    if not isinstance(title, str):
        title = str(title)
    
    # Remove or replace problematic characters
    title = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Remove any leading/trailing whitespace and dots
    title = title.strip('. ')
    # Limit length to avoid filesystem issues
    if len(title) > 200:
        title = title[:200]
    return title


def get_channel_info(channel_dir):
    """Extract channel information from existing files."""
    channel_info = {}
    
    # Try to read .channel_info.json
    channel_info_file = channel_dir / '.channel_info.json'
    if channel_info_file.exists():
        try:
            with open(channel_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Try different field names for channel title
                title = data.get('channel_title') or data.get('title') or data.get('channel_name') or ''
                # Ensure it's a string
                if isinstance(title, dict):
                    title = str(title)
                channel_info['title'] = title
                channel_info['url'] = data.get('channel_url', data.get('url', ''))
                channel_info['id'] = data.get('channel_id', channel_dir.name)
        except Exception as e:
            print(f"  ⚠️ Error reading {channel_info_file}: {e}")
    
    # If no title found, try to get from video metadata
    if not channel_info.get('title'):
        # Look for any video_info.json file
        for video_dir in channel_dir.iterdir():
            if video_dir.is_dir():
                for file in video_dir.glob('*_video_info.json'):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            video_data = json.load(f)
                            uploader = video_data.get('uploader', video_data.get('channel', ''))
                            # Ensure uploader is a string
                            if uploader and isinstance(uploader, str):
                                channel_info['title'] = uploader
                                break
                            elif uploader and isinstance(uploader, dict):
                                # Handle case where uploader might be a dict with name field
                                channel_info['title'] = str(uploader.get('name', uploader))
                                break
                    except:
                        continue
                if channel_info.get('title'):
                    break
    
    # Fallback to channel ID if no title found
    if not channel_info.get('title'):
        channel_info['title'] = channel_dir.name
        channel_info['is_fallback'] = True
    
    return channel_info


def add_channel_title_files():
    """Add channel title files to all channel directories."""
    storage = get_storage_paths_v2()
    downloads_dir = Path(storage.base_path)
    
    if not downloads_dir.exists():
        print(f"❌ Downloads directory not found: {downloads_dir}")
        return
    
    print("=" * 80)
    print("📁 Adding Channel Title Files")
    print("=" * 80)
    print(f"📍 Storage path: {downloads_dir}")
    print()
    
    # Statistics
    stats = {
        'processed': 0,
        'added': 0,
        'skipped': 0,
        'errors': 0,
        'updated': 0
    }
    
    # Process each channel directory
    channel_dirs = [d for d in downloads_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    print(f"📊 Found {len(channel_dirs)} channel directories")
    print()
    
    for channel_dir in sorted(channel_dirs):
        stats['processed'] += 1
        print(f"📺 Processing: {channel_dir.name}")
        
        # Get channel information
        channel_info = get_channel_info(channel_dir)
        channel_title = channel_info.get('title', channel_dir.name)
        
        # Sanitize title for filename
        safe_title = sanitize_channel_title(channel_title)
        title_file_path = channel_dir / f"{safe_title}.txt"
        
        # Prepare content for the file
        content_lines = [
            channel_title,
            "=" * len(channel_title),
            f"Channel ID: {channel_dir.name}"
        ]
        
        if channel_info.get('url'):
            content_lines.append(f"Channel URL: {channel_info['url']}")
        
        # Add video count
        video_dirs = [d for d in channel_dir.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        content_lines.append(f"Videos: {len(video_dirs)}")
        
        content = "\n".join(content_lines) + "\n"
        
        # Check if file already exists
        if title_file_path.exists():
            # Read existing content
            try:
                with open(title_file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                if existing_content == content:
                    print(f"  ✓ Already exists: {safe_title}.txt")
                    stats['skipped'] += 1
                else:
                    # Update the file
                    with open(title_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ↻ Updated: {safe_title}.txt")
                    stats['updated'] += 1
            except Exception as e:
                print(f"  ❌ Error reading existing file: {e}")
                stats['errors'] += 1
        else:
            # Create new file
            try:
                with open(title_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Created: {safe_title}.txt")
                if channel_info.get('is_fallback'):
                    print(f"     ⚠️ Using channel ID as title (no title found)")
                stats['added'] += 1
            except Exception as e:
                print(f"  ❌ Error creating file: {e}")
                stats['errors'] += 1
        
        print()
    
    # Print summary
    print("=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"  Total processed: {stats['processed']}")
    print(f"  ✅ Files added: {stats['added']}")
    print(f"  ↻ Files updated: {stats['updated']}")
    print(f"  ⏭️ Files skipped: {stats['skipped']}")
    print(f"  ❌ Errors: {stats['errors']}")
    print()
    
    # Show example of accessing
    if stats['added'] > 0 or stats['updated'] > 0:
        print("💡 Channel titles are now easily accessible:")
        print("   cat downloads/{channel_id}/{channel_title}.txt")
        print()


def update_download_system():
    """
    Update the download system to automatically create channel title files.
    This shows what needs to be added to the downloader.
    """
    print("📝 To automatically create channel title files in future downloads:")
    print()
    print("Add this to core/downloader.py in the download process:")
    print()
    print("""
    # After creating channel directory and getting channel info:
    def _save_channel_title_file(self, channel_dir: Path, channel_info: dict):
        \"\"\"Save channel title file for easy identification.\"\"\"
        channel_title = channel_info.get('channel_title', channel_info.get('title', ''))
        if channel_title:
            safe_title = self._sanitize_channel_title(channel_title)
            title_file = channel_dir / f"{safe_title}.txt"
            
            content = [
                channel_title,
                "=" * len(channel_title),
                f"Channel ID: {channel_dir.name}",
                f"Channel URL: {channel_info.get('channel_url', '')}",
                f"Downloaded: {datetime.now().isoformat()}"
            ]
            
            with open(title_file, 'w', encoding='utf-8') as f:
                f.write('\\n'.join(content))
    """)


if __name__ == "__main__":
    print()
    print("🚀 Channel Title File Creator")
    print("This will add {channel_title}.txt files to each channel directory")
    print()
    
    # Run the main function
    add_channel_title_files()
    
    # Show how to update the system
    print()
    update_download_system()