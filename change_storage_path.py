#!/usr/bin/env python3
"""
Utility script to change the storage path for yt-dl-sub.

This script demonstrates how easy it is to change the storage location
by simply updating the STORAGE_PATH environment variable.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def update_env_file(new_path: str, env_file: str = ".env") -> bool:
    """
    Update the STORAGE_PATH in the .env file.
    
    Args:
        new_path: New storage path
        env_file: Path to .env file (default: .env in current directory)
    
    Returns:
        True if successful, False otherwise
    """
    env_path = Path(env_file)
    
    # Read existing .env file if it exists
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        # Create from .env.example if .env doesn't exist
        example_path = Path('.env.example')
        if example_path.exists():
            with open(example_path, 'r') as f:
                lines = f.readlines()
        else:
            print(f"Error: Neither {env_file} nor .env.example found")
            return False
    
    # Update or add STORAGE_PATH
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith('STORAGE_PATH='):
            new_lines.append(f'STORAGE_PATH={new_path}\n')
            updated = True
            print(f"Updated: STORAGE_PATH={new_path}")
        else:
            new_lines.append(line)
    
    # If STORAGE_PATH wasn't found, add it
    if not updated:
        new_lines.append(f'\n# Storage path (updated by change_storage_path.py)\n')
        new_lines.append(f'STORAGE_PATH={new_path}\n')
        print(f"Added: STORAGE_PATH={new_path}")
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    return True


def validate_path(path_str: str) -> Optional[Path]:
    """
    Validate and expand the given path.
    
    Args:
        path_str: Path string to validate
    
    Returns:
        Expanded Path object if valid, None otherwise
    """
    try:
        path = Path(path_str).expanduser().absolute()
        
        # Check if parent directory exists (we'll create the final directory)
        if not path.parent.exists():
            print(f"Warning: Parent directory doesn't exist: {path.parent}")
            response = input("Create it? (y/n): ")
            if response.lower() != 'y':
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the storage directory if it doesn't exist
        if not path.exists():
            print(f"Storage directory will be created: {path}")
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            print(f"Error: Path exists but is not a directory: {path}")
            return None
        
        # Test write permissions
        test_file = path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            print(f"Error: Cannot write to directory: {e}")
            return None
        
        return path
    except Exception as e:
        print(f"Error validating path: {e}")
        return None


def main():
    """Main function to change storage path."""
    print("YouTube Content Intelligence Platform - Storage Path Configuration")
    print("=" * 70)
    
    # Show current configuration
    current_path = os.environ.get('STORAGE_PATH')
    if not current_path:
        # Try to read from .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('STORAGE_PATH='):
                        current_path = line.split('=', 1)[1].strip()
                        break
    
    if current_path:
        print(f"\nCurrent storage path: {current_path}")
    else:
        print("\nNo storage path currently configured")
    
    print("\nExamples of valid storage paths:")
    print("  - Local: /home/user/youtube-content")
    print("  - External (Mac): /Volumes/External Drive/youtube-storage")
    print("  - External (Linux): /mnt/external/youtube-storage")
    print("  - External (Windows): D:/youtube-storage")
    print("  - Network: /mnt/nas/youtube-storage")
    
    # Get new path from user
    print("\nEnter new storage path (or 'q' to quit):")
    new_path_str = input("> ").strip()
    
    if new_path_str.lower() == 'q':
        print("Cancelled")
        return 0
    
    # Validate the new path
    new_path = validate_path(new_path_str)
    if not new_path:
        print("Invalid path. Exiting.")
        return 1
    
    # Update .env file
    if update_env_file(str(new_path)):
        print(f"\n✓ Successfully updated storage path to: {new_path}")
        print("\nThe following directory structure will be created:")
        print(f"  {new_path}/")
        print(f"  ├── audio/         # Downloaded audio files")
        print(f"  ├── transcripts/   # SRT and TXT transcript files")
        print(f"  ├── content/       # Generated content")
        print(f"  └── metadata/      # Video and processing metadata")
        print("\nNote: Restart any running workers for the change to take effect.")
        return 0
    else:
        print("Failed to update .env file")
        return 1


if __name__ == "__main__":
    sys.exit(main())