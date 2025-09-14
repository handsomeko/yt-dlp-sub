#!/usr/bin/env python3
"""Detailed verification of @healthdiary7 channel"""

import os
from pathlib import Path
from datetime import datetime

channel_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCSfzaDNaHzOf5i9zAEJ0d6w")

video_dirs = sorted([d for d in channel_dir.iterdir() if d.is_dir()])
print(f"📊 Total video directories: {len(video_dirs)}")

categories = {
    "complete": [],
    "has_transcript_only": [],
    "has_media_only": [],
    "empty": [],
    "downloading": []
}

for video_dir in video_dirs:
    video_id = video_dir.name
    
    # Check for .downloading marker
    is_downloading = (video_dir / ".downloading").exists()
    
    # Check for transcript files
    transcript_dir = video_dir / "transcripts"
    has_transcript = False
    transcript_files = []
    if transcript_dir.exists():
        txt_files = list(transcript_dir.glob("*.txt"))
        srt_files = list(transcript_dir.glob("*.srt"))
        transcript_files = txt_files + srt_files
        has_transcript = len(transcript_files) > 0
    
    # Check for media files
    media_dir = video_dir / "media"
    has_media = False
    media_files = []
    if media_dir.exists():
        opus_files = list(media_dir.glob("*.opus"))
        mp3_files = list(media_dir.glob("*.mp3"))
        media_files = opus_files + mp3_files
        has_media = len(media_files) > 0
    
    # Categorize
    if is_downloading:
        categories["downloading"].append(video_id)
    elif has_transcript and has_media:
        categories["complete"].append(video_id)
    elif has_transcript and not has_media:
        categories["has_transcript_only"].append(video_id)
    elif has_media and not has_transcript:
        categories["has_media_only"].append(video_id)
    else:
        categories["empty"].append(video_id)

# Report
print(f"\n✅ Complete (transcript + media): {len(categories['complete'])}")
print(f"📝 Transcript only: {len(categories['has_transcript_only'])}")
print(f"🎵 Media only: {len(categories['has_media_only'])}")
print(f"⏳ Currently downloading: {len(categories['downloading'])}")
print(f"❌ Empty: {len(categories['empty'])}")

# Show incomplete details
if categories["downloading"]:
    print(f"\n⏳ Videos marked as downloading:")
    for vid in categories["downloading"]:
        print(f"  - {vid}")

if categories["has_media_only"]:
    print(f"\n🎵 Videos with media but no transcript:")
    for vid in categories["has_media_only"]:
        print(f"  - {vid}")

# Final calculation
actually_complete = len(categories["complete"]) + len(categories["has_media_only"])  # Media-only might have failed transcript
completion_rate = (len(categories["complete"]) / 106) * 100
print(f"\n📊 Final Status:")
print(f"  Complete with transcripts: {len(categories['complete'])}/106 ({completion_rate:.1f}%)")
print(f"  Total with media: {len(categories['complete']) + len(categories['has_media_only'])}/106")

if len(categories["downloading"]) > 0:
    print(f"\n⚠️  Note: {len(categories['downloading'])} videos are marked as 'downloading' but may be stalled")