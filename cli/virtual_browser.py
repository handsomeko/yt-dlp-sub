#!/usr/bin/env python3
"""
Virtual Browser CLI - Interactive filesystem browser for yt-dl-sub storage
Provides human-friendly navigation of the ID-based storage structure
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage_paths_v2 import get_storage_paths_v2, StorageVersion
from core.filename_sanitizer import sanitize_filename
from config.settings import get_settings


class VirtualBrowser:
    """
    Interactive browser for yt-dl-sub storage with human-readable navigation.
    Provides a virtual view of the ID-based storage structure.
    """
    
    def __init__(self):
        self.storage = get_storage_paths_v2()
        self.settings = get_settings()
        self.current_channel = None
        self.current_video = None
        self._cache = {}  # Cache for frequently accessed data
    
    def run_interactive(self):
        """Run the interactive browser."""
        print("🎥 YT-DL-Sub Virtual Browser")
        print("=" * 50)
        print("Navigate your video library with human-readable names!")
        print("Commands: channels, videos, files, search, help, quit")
        print()
        
        while True:
            try:
                # Show current location
                location = self._get_current_location()
                prompt = f"[{location}] > "
                
                # Get user input
                try:
                    command = input(prompt).strip().lower()
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                
                if not command:
                    continue
                
                # Parse command
                parts = command.split(maxsplit=1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                
                # Execute command
                if cmd in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif cmd in ['h', 'help']:
                    self._show_help()
                elif cmd in ['c', 'channels']:
                    self._list_channels(args)
                elif cmd in ['v', 'videos']:
                    self._list_videos(args)
                elif cmd in ['f', 'files']:
                    self._list_files(args)
                elif cmd in ['cd', 'goto']:
                    self._change_location(args)
                elif cmd in ['s', 'search']:
                    self._search(args)
                elif cmd in ['info', 'metadata']:
                    self._show_metadata(args)
                elif cmd in ['open', 'play']:
                    self._open_file(args)
                elif cmd in ['copy', 'cp']:
                    self._copy_path(args)
                elif cmd in ['recent']:
                    self._show_recent(args)
                elif cmd in ['stats']:
                    self._show_stats()
                elif cmd in ['clear', 'cls']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _get_current_location(self) -> str:
        """Get current location string for prompt."""
        if self.current_video and self.current_channel:
            # Try to get channel name
            channel_info = self._get_channel_info(self.current_channel)
            channel_name = channel_info.get('name', self.current_channel)
            
            # Try to get video title
            video_info = self._get_video_info(self.current_channel, self.current_video)
            video_title = video_info.get('title', self.current_video)
            
            return f"{channel_name[:20]}.../{video_title[:20]}..."
        elif self.current_channel:
            channel_info = self._get_channel_info(self.current_channel)
            channel_name = channel_info.get('name', self.current_channel)
            return f"{channel_name[:30]}..."
        else:
            return "Home"
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🎥 Virtual Browser Commands:

Navigation:
  channels [filter]    - List all channels (with optional filter)
  videos [channel]     - List videos for current or specified channel
  files [video]        - List files for current or specified video
  cd <target>          - Change to channel or video
  recent [days]        - Show recent videos (default: 7 days)

Information:
  info [target]        - Show metadata for channel/video
  stats                - Show storage statistics
  search <term>        - Search channels and videos by name

File Operations:
  open <file>          - Open file with default application
  copy <path>          - Copy path to clipboard
  
General:
  clear/cls            - Clear screen
  help/h               - Show this help
  quit/q               - Exit browser

Examples:
  > channels music         # Find channels with 'music' in name
  > cd UC123...            # Go to specific channel
  > videos                 # List videos in current channel
  > cd "Video Title"       # Go to specific video
  > files                  # List all files for current video
  > search "python"        # Search for videos about python
  > recent 3               # Show videos from last 3 days
"""
        print(help_text)
    
    def _list_channels(self, filter_term: str = ""):
        """List all channels with optional filtering."""
        channels = self.storage.list_all_channels()
        
        if not channels:
            print("📁 No channels found.")
            return
        
        print(f"📁 Channels ({len(channels)} total):")
        print("-" * 60)
        
        filtered_channels = []
        for channel_id in channels:
            # Get channel info
            info = self._get_channel_info(channel_id)
            channel_name = info.get('name', 'Unknown')
            video_count = len(self.storage.list_channel_videos(channel_id))
            
            # Apply filter
            if filter_term and filter_term.lower() not in channel_name.lower():
                continue
            
            filtered_channels.append((channel_id, channel_name, video_count, info))
        
        # Sort by name
        filtered_channels.sort(key=lambda x: x[1])
        
        for channel_id, name, video_count, info in filtered_channels:
            # Truncate long names
            display_name = name[:40] + "..." if len(name) > 40 else name
            
            # Show status indicator
            current = "👉 " if channel_id == self.current_channel else "   "
            
            print(f"{current}{display_name:<45} {video_count:>3} videos")
            print(f"      ID: {channel_id}")
            
            # Show last activity if available
            if 'last_updated' in info:
                print(f"      Updated: {info['last_updated']}")
            
            print()
    
    def _list_videos(self, channel_filter: str = ""):
        """List videos for current channel or specified channel."""
        target_channel = self.current_channel
        
        # If channel specified, use that
        if channel_filter:
            # Try to find channel by name or ID
            channels = self.storage.list_all_channels()
            for channel_id in channels:
                info = self._get_channel_info(channel_id)
                if (channel_filter.lower() in info.get('name', '').lower() or 
                    channel_filter in channel_id):
                    target_channel = channel_id
                    break
        
        if not target_channel:
            print("❌ No channel selected. Use 'channels' to list available channels.")
            return
        
        videos = self.storage.list_channel_videos(target_channel)
        
        if not videos:
            print(f"📹 No videos found for channel {target_channel}")
            return
        
        channel_info = self._get_channel_info(target_channel)
        channel_name = channel_info.get('name', target_channel)
        
        print(f"📹 Videos in {channel_name} ({len(videos)} total):")
        print("-" * 60)
        
        for video_id, video_info in videos[:20]:  # Limit to 20 for readability
            title = video_info.get('title', 'Unknown Title')
            display_title = title[:50] + "..." if len(title) > 50 else title
            
            # Show status
            current = "👉 " if video_id == self.current_video else "   "
            
            # Show file status
            status_icons = []
            if video_info.get('has_media'):
                status_icons.append("🎵")
            if video_info.get('has_transcript'):
                status_icons.append("📝")
            
            status = "".join(status_icons) or "⏳"
            
            print(f"{current}{display_title:<55} {status}")
            print(f"      ID: {video_id}")
            
            if 'published_at' in video_info:
                print(f"      Published: {video_info['published_at']}")
            
            print()
        
        if len(videos) > 20:
            print(f"... and {len(videos) - 20} more videos. Use search to filter.")
    
    def _list_files(self, video_filter: str = ""):
        """List files for current video or specified video."""
        if not self.current_channel:
            print("❌ No channel selected. Use 'channels' to select a channel first.")
            return
        
        target_video = self.current_video
        
        # If video specified, try to find it
        if video_filter:
            videos = self.storage.list_channel_videos(self.current_channel)
            for video_id, video_info in videos:
                if (video_filter.lower() in video_info.get('title', '').lower() or
                    video_filter in video_id):
                    target_video = video_id
                    break
        
        if not target_video:
            print("❌ No video selected. Use 'videos' to list available videos.")
            return
        
        video_info = self._get_video_info(self.current_channel, target_video)
        title = video_info.get('title', 'Unknown')
        
        print(f"📁 Files for: {title}")
        print("-" * 60)
        
        # Get all file types
        file_types = {
            '🎵 Media': self.storage.find_media_files(self.current_channel, target_video),
            '📝 Transcripts': list(self.storage.find_transcript_files(self.current_channel, target_video).values()),
            '📄 Content': [],  # TODO: Implement content file discovery
            '📊 Metadata': []  # TODO: Implement metadata file discovery
        }
        
        total_files = 0
        for category, files in file_types.items():
            if files:
                print(f"\n{category}:")
                for file_path in files:
                    file_path = Path(file_path)
                    size = self._format_file_size(file_path.stat().st_size)
                    print(f"  📎 {file_path.name} ({size})")
                    total_files += 1
        
        if total_files == 0:
            print("📂 No files found for this video.")
        else:
            print(f"\n📊 Total: {total_files} files")
    
    def _change_location(self, target: str):
        """Change current location to channel or video."""
        if not target:
            # Go to home
            self.current_channel = None
            self.current_video = None
            print("🏠 Moved to home")
            return
        
        # Try to find channel first
        channels = self.storage.list_all_channels()
        for channel_id in channels:
            info = self._get_channel_info(channel_id)
            if (target.lower() in info.get('name', '').lower() or 
                target in channel_id):
                self.current_channel = channel_id
                self.current_video = None
                print(f"📁 Moved to channel: {info.get('name', channel_id)}")
                return
        
        # If we're in a channel, try to find video
        if self.current_channel:
            videos = self.storage.list_channel_videos(self.current_channel)
            for video_id, video_info in videos:
                if (target.lower() in video_info.get('title', '').lower() or
                    target in video_id):
                    self.current_video = video_id
                    print(f"📹 Moved to video: {video_info.get('title', video_id)}")
                    return
        
        print(f"❌ Could not find '{target}'. Use channels/videos to see available options.")
    
    def _search(self, term: str):
        """Search for channels and videos."""
        if not term:
            print("❌ Please provide a search term.")
            return
        
        print(f"🔍 Searching for: '{term}'")
        print("-" * 60)
        
        found_channels = []
        found_videos = []
        
        # Search channels
        channels = self.storage.list_all_channels()
        for channel_id in channels:
            info = self._get_channel_info(channel_id)
            if term.lower() in info.get('name', '').lower():
                found_channels.append((channel_id, info))
        
        # Search videos
        for channel_id in channels:
            videos = self.storage.list_channel_videos(channel_id)
            for video_id, video_info in videos:
                if term.lower() in video_info.get('title', '').lower():
                    found_videos.append((channel_id, video_id, video_info))
        
        # Show results
        if found_channels:
            print(f"📁 Channels ({len(found_channels)}):")
            for channel_id, info in found_channels:
                name = info.get('name', channel_id)
                print(f"  📁 {name}")
                print(f"      ID: {channel_id}")
        
        if found_videos:
            print(f"\n📹 Videos ({len(found_videos)}):")
            for channel_id, video_id, video_info in found_videos[:10]:  # Limit results
                title = video_info.get('title', 'Unknown')
                channel_info = self._get_channel_info(channel_id)
                channel_name = channel_info.get('name', channel_id)
                print(f"  📹 {title}")
                print(f"      Channel: {channel_name}")
                print(f"      ID: {video_id}")
        
        if not found_channels and not found_videos:
            print("❌ No results found.")
    
    def _show_metadata(self, target: str = ""):
        """Show detailed metadata for current or specified item."""
        if not target and self.current_video and self.current_channel:
            # Show video metadata
            metadata_file = self.storage.get_video_metadata_file(self.current_channel, self.current_video)
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print("📊 Video Metadata:")
                print("-" * 40)
                self._print_dict(metadata, indent="  ")
            else:
                print("❌ No metadata file found for current video.")
        
        elif not target and self.current_channel:
            # Show channel metadata
            info_file = self.storage.get_channel_info_file(self.current_channel)
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                print("📊 Channel Metadata:")
                print("-" * 40)
                self._print_dict(info, indent="  ")
            else:
                print("❌ No metadata file found for current channel.")
        else:
            print("❌ Please navigate to a channel or video first, or specify a target.")
    
    def _open_file(self, filename: str):
        """Open a file with the default application."""
        if not self.current_channel or not self.current_video:
            print("❌ Please navigate to a video first.")
            return
        
        # Find the file
        all_files = []
        all_files.extend(self.storage.find_media_files(self.current_channel, self.current_video))
        all_files.extend(self.storage.find_transcript_files(self.current_channel, self.current_video).values())
        
        target_file = None
        for file_path in all_files:
            if filename.lower() in Path(file_path).name.lower():
                target_file = file_path
                break
        
        if not target_file:
            print(f"❌ File '{filename}' not found.")
            return
        
        try:
            # Open with default application
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", target_file])
            elif sys.platform == "linux":  # Linux
                subprocess.run(["xdg-open", target_file])
            else:  # Windows
                os.startfile(target_file)
            
            print(f"📂 Opened: {Path(target_file).name}")
        except Exception as e:
            print(f"❌ Failed to open file: {e}")
    
    def _copy_path(self, target: str = ""):
        """Copy file path to clipboard."""
        if not target and self.current_video and self.current_channel:
            # Copy video directory path
            video_dir = self.storage.get_video_dir(self.current_channel, self.current_video)
            path_to_copy = str(video_dir)
        elif not target and self.current_channel:
            # Copy channel directory path
            channel_dir = self.storage.get_channel_dir(self.current_channel)
            path_to_copy = str(channel_dir)
        else:
            print("❌ Please specify what to copy or navigate to a location first.")
            return
        
        try:
            # Copy to clipboard using platform-specific command
            if sys.platform == "darwin":  # macOS
                subprocess.run(["pbcopy"], input=path_to_copy.encode())
            elif sys.platform == "linux":  # Linux
                subprocess.run(["xclip", "-selection", "clipboard"], input=path_to_copy.encode())
            else:  # Windows
                subprocess.run(["clip"], input=path_to_copy.encode())
            
            print(f"📋 Copied to clipboard: {path_to_copy}")
        except Exception as e:
            print(f"❌ Failed to copy to clipboard: {e}")
            print(f"Path: {path_to_copy}")
    
    def _show_recent(self, days_str: str = "7"):
        """Show recent videos."""
        try:
            days = int(days_str) if days_str else 7
        except ValueError:
            days = 7
        
        print(f"📅 Recent videos (last {days} days):")
        print("-" * 60)
        
        # This is a simplified version - in a full implementation,
        # we would parse dates from metadata
        channels = self.storage.list_all_channels()
        recent_videos = []
        
        for channel_id in channels[:5]:  # Limit to prevent slowness
            videos = self.storage.list_channel_videos(channel_id)
            for video_id, video_info in videos[:5]:  # Limit per channel
                recent_videos.append((channel_id, video_id, video_info))
        
        for channel_id, video_id, video_info in recent_videos[:10]:
            title = video_info.get('title', 'Unknown')
            channel_info = self._get_channel_info(channel_id)
            channel_name = channel_info.get('name', channel_id)
            
            print(f"📹 {title}")
            print(f"    Channel: {channel_name}")
            print()
    
    def _show_stats(self):
        """Show storage statistics."""
        stats = self.storage.get_storage_stats()
        
        print("📊 Storage Statistics:")
        print("-" * 40)
        print(f"Storage Version: {stats['version']}")
        print(f"Base Path: {stats['base_path']}")
        print(f"Total Channels: {stats['total_channels']}")
        print(f"Total Videos: {stats['total_videos']}")
        print(f"Total Size: {stats['total_size_mb']} MB")
        
        print("\nBy File Type:")
        for file_type, type_stats in stats.get('by_type', {}).items():
            print(f"  {file_type.capitalize()}: {type_stats['files']} files, {type_stats['size_mb']} MB")
    
    def _get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get channel information with caching."""
        if channel_id not in self._cache:
            info_file = self.storage.get_channel_info_file(channel_id)
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        self._cache[channel_id] = json.load(f)
                except:
                    self._cache[channel_id] = {'name': channel_id}
            else:
                # Try to extract channel name from directory structure
                self._cache[channel_id] = {'name': channel_id}
        
        return self._cache[channel_id]
    
    def _get_video_info(self, channel_id: str, video_id: str) -> Dict[str, Any]:
        """Get video information."""
        cache_key = f"{channel_id}:{video_id}"
        if cache_key not in self._cache:
            # Try video index first
            index_file = self.storage.get_video_index_file(channel_id)
            if index_file.exists():
                try:
                    with open(index_file, 'r') as f:
                        index = json.load(f)
                        self._cache[cache_key] = index.get(video_id, {'title': video_id})
                except:
                    self._cache[cache_key] = {'title': video_id}
            else:
                self._cache[cache_key] = {'title': video_id}
        
        return self._cache[cache_key]
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def _print_dict(self, data: Dict[str, Any], indent: str = ""):
        """Print dictionary in a formatted way."""
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                self._print_dict(value, indent + "  ")
            elif isinstance(value, list):
                print(f"{indent}{key}: {len(value)} items")
            else:
                print(f"{indent}{key}: {value}")


def main():
    """Main entry point for the virtual browser CLI."""
    parser = argparse.ArgumentParser(
        description="Virtual Browser for yt-dl-sub storage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show storage statistics and exit'
    )
    
    parser.add_argument(
        '--search',
        metavar='TERM',
        help='Search for channels/videos and exit'
    )
    
    parser.add_argument(
        '--list-channels',
        action='store_true',
        help='List all channels and exit'
    )
    
    args = parser.parse_args()
    
    browser = VirtualBrowser()
    
    # Handle non-interactive commands
    if args.stats:
        browser._show_stats()
        return
    
    if args.search:
        browser._search(args.search)
        return
    
    if args.list_channels:
        browser._list_channels()
        return
    
    # Start interactive mode
    browser.run_interactive()


if __name__ == '__main__':
    main()