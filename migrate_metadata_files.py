#!/usr/bin/env python3
"""
Migrate existing _channel_info files from channel level to video level
and rename them to _video_info
"""

import os
import shutil
import json
from pathlib import Path

def migrate_metadata_files():
    """Move and rename all _channel_info files to video level _video_info files"""
    
    # Base downloads directory
    downloads_dir = Path('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads')
    
    # Find all _channel_info files
    channel_info_files = list(downloads_dir.glob('**/*_channel_info.json'))
    channel_info_files.extend(list(downloads_dir.glob('**/*_channel_info.md')))
    
    if not channel_info_files:
        print("No _channel_info files found to migrate")
        return
    
    print(f"Found {len(channel_info_files)} files to migrate\n")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for file_path in channel_info_files:
        try:
            # Extract the video title from filename
            filename = file_path.name
            if filename.endswith('_channel_info.json'):
                video_title = filename[:-18]  # Remove _channel_info.json
                ext = '.json'
            elif filename.endswith('_channel_info.md'):
                video_title = filename[:-16]  # Remove _channel_info.md
                ext = '.md'
            else:
                print(f"⚠️  Skipping unexpected file: {file_path}")
                skipped += 1
                continue
            
            # Current location is at channel level
            channel_dir = file_path.parent
            
            # For JSON files, read to get video_id
            video_id = None
            if ext == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        video_id = data.get('video_id')
                except:
                    pass
            
            # If we couldn't get video_id from JSON, try to find it from directory structure
            if not video_id:
                # Look for subdirectories that might be video IDs (11 character YouTube IDs)
                for subdir in channel_dir.iterdir():
                    if subdir.is_dir() and len(subdir.name) == 11:
                        # Check if this directory has the same video title in its files
                        video_files = list(subdir.glob(f'**/{video_title}*'))
                        if video_files:
                            video_id = subdir.name
                            break
            
            if not video_id:
                print(f"⚠️  Could not find video_id for: {video_title}")
                print(f"   File: {file_path}")
                skipped += 1
                continue
            
            # Target location at video level
            video_dir = channel_dir / video_id
            
            if not video_dir.exists():
                print(f"⚠️  Video directory doesn't exist: {video_dir}")
                print(f"   For file: {file_path}")
                skipped += 1
                continue
            
            # New filename with _video_info instead of _channel_info
            new_filename = f"{video_title}_video_info{ext}"
            new_path = video_dir / new_filename
            
            # Check if target already exists
            if new_path.exists():
                print(f"✓  Already migrated: {new_filename}")
                # Remove the old file at channel level
                os.remove(file_path)
                migrated += 1
                continue
            
            # Move and rename the file
            shutil.move(str(file_path), str(new_path))
            print(f"✓  Migrated: {channel_dir.name}/{filename}")
            print(f"   → {video_id}/{new_filename}")
            migrated += 1
            
        except Exception as e:
            print(f"✗  Error migrating {file_path}: {e}")
            errors += 1
    
    print(f"\n{'='*60}")
    print(f"Migration Summary:")
    print(f"  Migrated: {migrated} files")
    print(f"  Skipped:  {skipped} files")
    print(f"  Errors:   {errors} files")
    print(f"  Total:    {len(channel_info_files)} files")

if __name__ == "__main__":
    migrate_metadata_files()