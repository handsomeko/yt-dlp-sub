#!/usr/bin/env python3
"""
Add video_url.txt files to existing video directories that don't have them.
Reads the URL from the comprehensive metadata files.
"""

import json
from pathlib import Path

def add_video_urls():
    """Add video_url.txt to all existing video directories"""
    
    # Base downloads directory
    downloads_dir = Path('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads')
    
    if not downloads_dir.exists():
        print(f"Downloads directory not found: {downloads_dir}")
        return
    
    added = 0
    skipped = 0
    errors = 0
    
    # Find all video directories (pattern: channel_id/video_id)
    for channel_dir in downloads_dir.iterdir():
        if not channel_dir.is_dir() or channel_dir.name.startswith('.'):
            continue
        
        for video_dir in channel_dir.iterdir():
            if not video_dir.is_dir() or video_dir.name.startswith('.'):
                continue
            
            # Check if this looks like a video directory (11 char video ID)
            if len(video_dir.name) != 11:
                continue
            
            video_url_file = video_dir / 'video_url.txt'
            
            # Skip if already exists
            if video_url_file.exists():
                print(f"✓ Already has video_url.txt: {channel_dir.name}/{video_dir.name}")
                skipped += 1
                continue
            
            try:
                # Try to find the URL from existing metadata files
                video_url = None
                
                # First try: Look for comprehensive video_info.json
                for info_file in video_dir.glob('*_video_info.json'):
                    with open(info_file, 'r') as f:
                        data = json.load(f)
                        urls = data.get('urls', {})
                        video_url = urls.get('webpage_url') or urls.get('original_url')
                        if video_url:
                            break
                
                # Second try: Look in .metadata.json
                if not video_url:
                    metadata_file = video_dir / '.metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            video_id = data.get('video_id', video_dir.name)
                            # Construct URL from video_id
                            video_url = f'https://www.youtube.com/watch?v={video_id}'
                
                # Third try: Look in metadata subdirectory
                if not video_url:
                    metadata_dir = video_dir / 'metadata'
                    if metadata_dir.exists():
                        for metadata_file in metadata_dir.glob('*_metadata.json'):
                            with open(metadata_file, 'r') as f:
                                data = json.load(f)
                                video_url = data.get('webpage_url') or data.get('original_url')
                                if video_url:
                                    break
                
                # Final fallback: Construct from directory name (video_id)
                if not video_url:
                    video_url = f'https://www.youtube.com/watch?v={video_dir.name}'
                
                # Write the video_url.txt file
                with open(video_url_file, 'w') as f:
                    f.write(video_url)
                
                print(f"✓ Added video_url.txt: {channel_dir.name}/{video_dir.name}")
                print(f"  URL: {video_url}")
                added += 1
                
            except Exception as e:
                print(f"✗ Error processing {channel_dir.name}/{video_dir.name}: {e}")
                errors += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Added:   {added} files")
    print(f"  Skipped: {skipped} files (already exist)")
    print(f"  Errors:  {errors} files")
    print(f"  Total:   {added + skipped + errors} video directories")

if __name__ == "__main__":
    add_video_urls()