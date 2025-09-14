#!/usr/bin/env python3
"""
Comprehensive verification script for 4 health channels
Verifies that all files are properly processed and meet requirements
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_punctuation_density(text: str) -> float:
    """Calculate punctuation marks per 100 characters"""
    if not text:
        return 0.0
    
    # Chinese and English punctuation marks
    punctuation_marks = set('。！？，、；：""''（）【】《》.!?,;:"\'()[]{}…')
    
    punct_count = sum(1 for char in text if char in punctuation_marks)
    char_count = len(text)
    
    if char_count == 0:
        return 0.0
    
    return (punct_count / char_count) * 100

def verify_video_directory(video_path: Path) -> Dict[str, any]:
    """Verify a single video directory for completeness"""
    result = {
        'video_id': video_path.name,
        'audio_files': [],
        'srt_files': [],
        'txt_files': [],
        'errors': [],
        'punctuation_density': {}
    }
    
    # Check for audio files
    media_dir = video_path / 'media'
    if media_dir.exists():
        for audio_ext in ['.opus', '.mp3', '.m4a', '.wav']:
            audio_files = list(media_dir.glob(f'*{audio_ext}'))
            result['audio_files'].extend([f.name for f in audio_files])
    else:
        result['errors'].append('No media directory')
    
    # Check for transcript files
    transcript_dir = video_path / 'transcripts'
    if transcript_dir.exists():
        # Check SRT files
        srt_files = list(transcript_dir.glob('*.srt'))
        result['srt_files'] = [f.name for f in srt_files]
        
        # Check TXT files and their punctuation
        txt_files = list(transcript_dir.glob('*.txt'))
        for txt_file in txt_files:
            result['txt_files'].append(txt_file.name)
            
            # Read and check punctuation
            try:
                content = txt_file.read_text(encoding='utf-8')
                density = calculate_punctuation_density(content)
                result['punctuation_density'][txt_file.name] = density
                
                # Flag if punctuation is too low
                if density < 0.5:
                    result['errors'].append(f'{txt_file.name}: Low punctuation ({density:.2f}%)')
            except Exception as e:
                result['errors'].append(f'Error reading {txt_file.name}: {e}')
    else:
        result['errors'].append('No transcripts directory')
    
    # Basic completeness check
    if not result['audio_files']:
        result['errors'].append('No audio files found')
    if not result['srt_files']:
        result['errors'].append('No SRT files found')
    if not result['txt_files']:
        result['errors'].append('No TXT files found')
    
    return result

def verify_channel(channel_path: Path, channel_name: str) -> Dict[str, any]:
    """Verify all videos in a channel"""
    print(f"\n{'='*60}")
    print(f"Verifying Channel: {channel_name}")
    print(f"Path: {channel_path}")
    print(f"{'='*60}")
    
    if not channel_path.exists():
        print(f"❌ Channel directory does not exist!")
        return {'error': 'Directory not found'}
    
    # Get all video directories
    video_dirs = [d for d in channel_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    channel_result = {
        'channel_name': channel_name,
        'channel_id': channel_path.name,
        'total_videos': len(video_dirs),
        'videos_verified': [],
        'videos_with_errors': [],
        'summary': {}
    }
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in sorted(video_dirs):
        video_result = verify_video_directory(video_dir)
        
        if video_result['errors']:
            channel_result['videos_with_errors'].append(video_result)
            print(f"❌ {video_dir.name}: {len(video_result['errors'])} errors")
            for error in video_result['errors']:
                print(f"   - {error}")
        else:
            channel_result['videos_verified'].append(video_result)
            avg_punct = sum(video_result['punctuation_density'].values()) / len(video_result['punctuation_density']) if video_result['punctuation_density'] else 0
            print(f"✅ {video_dir.name}: {len(video_result['audio_files'])} audio, {len(video_result['srt_files'])} SRT, {len(video_result['txt_files'])} TXT (punct: {avg_punct:.2f}%)")
    
    # Summary statistics
    channel_result['summary'] = {
        'total_videos': len(video_dirs),
        'videos_complete': len(channel_result['videos_verified']),
        'videos_with_errors': len(channel_result['videos_with_errors']),
        'completion_rate': (len(channel_result['videos_verified']) / len(video_dirs) * 100) if video_dirs else 0
    }
    
    print(f"\n📊 Channel Summary:")
    print(f"   - Total Videos: {channel_result['summary']['total_videos']}")
    print(f"   - Complete: {channel_result['summary']['videos_complete']}")
    print(f"   - With Errors: {channel_result['summary']['videos_with_errors']}")
    print(f"   - Completion Rate: {channel_result['summary']['completion_rate']:.1f}%")
    
    return channel_result

def main():
    """Main verification function"""
    print("="*80)
    print("COMPREHENSIVE VERIFICATION OF 4 HEALTH CHANNELS")
    print("="*80)
    
    base_path = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads")
    
    # Define the 4 health channels
    channels = [
        ("UCuwovWnwZWbusjkHY8OHibA", "逍遙健康指南"),
        ("UCLsJJvIx82JWIG2XXYmRbXA", "health-k6s (智慧之泉)"),
        ("UCSfzaDNaHzOf5i9zAEJ0d6w", "healthdiary7 (醫師健康日記)"),
        ("UCfGGIPQzHZT6dfsCarC-gbA", "healthyeyes2 (健康之眼)")
    ]
    
    all_results = []
    total_videos = 0
    total_complete = 0
    total_errors = 0
    
    for channel_id, channel_name in channels:
        channel_path = base_path / channel_id
        result = verify_channel(channel_path, channel_name)
        all_results.append(result)
        
        if 'summary' in result:
            total_videos += result['summary']['total_videos']
            total_complete += result['summary']['videos_complete']
            total_errors += result['summary']['videos_with_errors']
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL VERIFICATION SUMMARY")
    print("="*80)
    print(f"📊 Overall Statistics:")
    print(f"   - Total Channels: {len(channels)}")
    print(f"   - Total Videos: {total_videos}")
    print(f"   - Complete Videos: {total_complete}")
    print(f"   - Videos with Errors: {total_errors}")
    print(f"   - Overall Completion Rate: {(total_complete/total_videos*100) if total_videos else 0:.1f}%")
    
    if total_errors > 0:
        print(f"\n⚠️  {total_errors} videos need attention!")
        print("Run punctuation restoration script to fix low punctuation issues.")
    else:
        print("\n✅ All videos are properly processed and meet requirements!")
    
    return all_results

if __name__ == "__main__":
    results = main()