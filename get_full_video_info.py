#!/usr/bin/env python3
"""Get complete information about the YouTube video"""

import yt_dlp
import json
from datetime import datetime

url = "https://www.youtube.com/watch?v=GT0jtVjRy2E"

print(f"Extracting ALL information for: {url}")
print("=" * 80)

ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'extract_flat': False,
    'getcomments': False,  # Skip comments for speed
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    
    # Basic Information
    print("\n📺 BASIC VIDEO INFO:")
    print(f"  Title: {info.get('title', 'N/A')}")
    print(f"  Video ID: {info.get('id', 'N/A')}")
    print(f"  URL: {info.get('webpage_url', 'N/A')}")
    
    # Channel Information
    print("\n👤 CHANNEL INFO:")
    print(f"  Channel: {info.get('channel', 'N/A')}")
    print(f"  Channel ID: {info.get('channel_id', 'N/A')}")
    print(f"  Channel URL: {info.get('channel_url', 'N/A')}")
    print(f"  Uploader: {info.get('uploader', 'N/A')}")
    print(f"  Uploader ID: {info.get('uploader_id', 'N/A')}")
    
    # Statistics
    print("\n📊 STATISTICS:")
    print(f"  View Count: {info.get('view_count', 0):,}")
    print(f"  Like Count: {info.get('like_count', 0):,}")
    print(f"  Comment Count: {info.get('comment_count', 0):,}")
    print(f"  Duration: {info.get('duration', 0)} seconds ({info.get('duration', 0)//60} minutes)")
    print(f"  Age Limit: {info.get('age_limit', 0)}")
    print(f"  Live Stream: {info.get('is_live', False)}")
    print(f"  Was Live: {info.get('was_live', False)}")
    
    # Dates
    print("\n📅 DATES:")
    upload_date = info.get('upload_date', 'N/A')
    if upload_date != 'N/A':
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    print(f"  Upload Date: {upload_date}")
    print(f"  Release Date: {info.get('release_date', 'N/A')}")
    print(f"  Modified Date: {info.get('modified_date', 'N/A')}")
    
    # Description
    print("\n📝 DESCRIPTION:")
    description = info.get('description', '')
    if description:
        # Show first 500 chars
        if len(description) > 500:
            print(f"  {description[:500]}...")
            print(f"  [Total length: {len(description)} characters]")
        else:
            print(f"  {description}")
    else:
        print("  No description available")
    
    # Tags
    print("\n🏷️ TAGS:")
    tags = info.get('tags', [])
    if tags:
        print(f"  {', '.join(tags[:10])}")
        if len(tags) > 10:
            print(f"  ... and {len(tags)-10} more tags")
    else:
        print("  No tags")
    
    # Categories
    print("\n📂 CATEGORIES:")
    print(f"  Categories: {info.get('categories', [])}")
    
    # Technical Details
    print("\n🎬 TECHNICAL DETAILS:")
    print(f"  Format: {info.get('ext', 'N/A')}")
    print(f"  Resolution: {info.get('resolution', 'N/A')}")
    print(f"  FPS: {info.get('fps', 'N/A')}")
    print(f"  Video Codec: {info.get('vcodec', 'N/A')}")
    print(f"  Audio Codec: {info.get('acodec', 'N/A')}")
    
    # Thumbnail
    print("\n🖼️ THUMBNAILS:")
    thumbnails = info.get('thumbnails', [])
    if thumbnails:
        print(f"  Available: {len(thumbnails)} thumbnails")
        # Show highest quality
        if thumbnails:
            best = max(thumbnails, key=lambda x: x.get('width', 0) * x.get('height', 0))
            print(f"  Best Quality: {best.get('width')}x{best.get('height')}")
            print(f"  URL: {best.get('url', 'N/A')[:100]}...")
    
    # Subtitles/Captions
    print("\n💬 SUBTITLES/CAPTIONS:")
    subtitles = info.get('subtitles', {})
    automatic_captions = info.get('automatic_captions', {})
    if subtitles:
        print(f"  Manual Subtitles: {list(subtitles.keys())}")
    if automatic_captions:
        print(f"  Auto-generated: {list(automatic_captions.keys())[:5]}")
        if len(automatic_captions) > 5:
            print(f"  ... and {len(automatic_captions)-5} more languages")
    if not subtitles and not automatic_captions:
        print("  No subtitles available")
    
    # Chapters
    print("\n📖 CHAPTERS:")
    chapters = info.get('chapters', [])
    if chapters:
        print(f"  Total Chapters: {len(chapters)}")
        for i, chapter in enumerate(chapters[:5], 1):
            start = int(chapter.get('start_time', 0))
            title = chapter.get('title', 'Untitled')
            print(f"  {i}. [{start//60:02d}:{start%60:02d}] {title}")
        if len(chapters) > 5:
            print(f"  ... and {len(chapters)-5} more chapters")
    else:
        print("  No chapters")
    
    # Playlist Info (if part of playlist)
    print("\n📋 PLAYLIST INFO:")
    print(f"  Playlist: {info.get('playlist', 'N/A')}")
    print(f"  Playlist Index: {info.get('playlist_index', 'N/A')}")
    
    # Available Formats
    print("\n📹 AVAILABLE FORMATS:")
    formats = info.get('formats', [])
    if formats:
        print(f"  Total Formats: {len(formats)}")
        # Group by quality
        qualities = {}
        for fmt in formats:
            res = fmt.get('resolution', 'audio only')
            if res not in qualities:
                qualities[res] = []
            qualities[res].append(fmt.get('ext', 'unknown'))
        
        for quality, exts in sorted(qualities.items(), key=lambda x: x[0], reverse=True)[:10]:
            print(f"  {quality}: {', '.join(set(exts))}")
    
    # Additional metadata
    print("\n🔍 ADDITIONAL METADATA:")
    print(f"  Availability: {info.get('availability', 'N/A')}")
    print(f"  License: {info.get('license', 'N/A')}")
    print(f"  Location: {info.get('location', 'N/A')}")
    print(f"  Language: {info.get('language', 'N/A')}")
    print(f"  Playable in embed: {info.get('playable_in_embed', 'N/A')}")
    
    # Save full info to file
    with open('/Users/jk/yt-dl-sub/video_info_GT0jtVjRy2E.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print("\n💾 Full info saved to: video_info_GT0jtVjRy2E.json")

print("\n" + "=" * 80)
print("✅ Information extraction complete!")