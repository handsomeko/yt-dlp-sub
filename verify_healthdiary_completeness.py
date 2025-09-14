#!/usr/bin/env python3
"""Verify @healthdiary7 channel completeness"""

import os
from pathlib import Path

channel_dir = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/UCSfzaDNaHzOf5i9zAEJ0d6w")

# Count video directories
video_dirs = [d for d in channel_dir.iterdir() if d.is_dir()]
print(f"Total video directories: {len(video_dirs)}")

missing_transcripts = []
missing_media = []
complete_videos = []

for video_dir in video_dirs:
    video_id = video_dir.name
    
    # Check for transcript files
    transcript_dir = video_dir / "transcripts"
    has_transcript = False
    if transcript_dir.exists():
        txt_files = list(transcript_dir.glob("*.txt"))
        srt_files = list(transcript_dir.glob("*.srt"))
        has_transcript = len(txt_files) > 0 or len(srt_files) > 0
    
    # Check for media files
    media_dir = video_dir / "media"
    has_media = False
    if media_dir.exists():
        opus_files = list(media_dir.glob("*.opus"))
        mp3_files = list(media_dir.glob("*.mp3"))
        has_media = len(opus_files) > 0 or len(mp3_files) > 0
    
    if not has_transcript:
        missing_transcripts.append(video_id)
    if not has_media:
        missing_media.append(video_id)
    
    if has_transcript and has_media:
        complete_videos.append(video_id)

print(f"\n✅ Complete videos: {len(complete_videos)}/106")
print(f"❌ Missing transcripts: {len(missing_transcripts)}")
print(f"❌ Missing media: {len(missing_media)}")

if missing_transcripts:
    print("\nVideos without transcripts:")
    for vid in missing_transcripts[:5]:  # Show first 5
        print(f"  - {vid}")
    if len(missing_transcripts) > 5:
        print(f"  ... and {len(missing_transcripts) - 5} more")

if missing_media:
    print("\nVideos without media:")
    for vid in missing_media[:5]:  # Show first 5
        print(f"  - {vid}")
    if len(missing_media) > 5:
        print(f"  ... and {len(missing_media) - 5} more")

# Summary
completion_rate = (len(complete_videos) / 106) * 100
print(f"\n📊 Completion Rate: {completion_rate:.1f}%")

if completion_rate == 100:
    print("🎉 100% COMPLETE - All 106 videos have both transcripts and media!")
else:
    print(f"⚠️  {106 - len(complete_videos)} videos are incomplete")