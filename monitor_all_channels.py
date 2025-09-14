#!/usr/bin/env python3
"""
Continuous monitoring of all active channel downloads.
"""

import time
import re
from pathlib import Path
from datetime import datetime
import os

def get_channel_status(log_file, channel_name, total_videos):
    """Get status for a specific channel."""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Try to extract total videos from log if not provided
    if total_videos == 0:
        found_match = re.search(r'📊 Found (\d+) videos', content)
        if found_match:
            total_videos = int(found_match.group(1))
    
    # If still 0, return minimal status
    if total_videos == 0:
        return {
            'completed': 0,
            'started': 0,
            'zh_files': 0,
            'en_files': 0,
            'whisper': 0,
            'rate_limits': 0,
            'last_op': "Enumerating...",
            'progress_pct': 0,
            'total_videos': 0
        }
    
    # Count completed
    pattern = f'✅ \\[(\\d+)/{total_videos}\\] Completed'
    completed = len(re.findall(pattern, content))
    
    # Count started
    pattern = f'\\[(\\d+)/{total_videos}\\] Starting download'
    started_matches = re.findall(pattern, content)
    started = len(set(started_matches))
    
    # Count Chinese files
    zh_files = len(re.findall(r'\.zh\.(txt|srt)', content))
    en_files = len(re.findall(r'\.en\.(txt|srt)', content))
    
    # Count Whisper transcriptions
    whisper_count = content.count('Transcribing Chinese audio')
    
    # Count rate limit errors
    rate_limits = content.count('HTTP Error 429')
    
    # Get last operation
    last_operations = re.findall(f'(✅ \\[\\d+/{total_videos}\\].*|🎬 \\[\\d+/{total_videos}\\].*)', content)
    last_op = last_operations[-1][:60] if last_operations else "Waiting..."
    
    return {
        'completed': completed,
        'started': started,
        'zh_files': zh_files,
        'en_files': en_files,
        'whisper': whisper_count,
        'rate_limits': rate_limits,
        'last_op': last_op,
        'progress_pct': (completed / total_videos) * 100,
        'total_videos': total_videos
    }

def monitor_continuous():
    """Monitor all active downloads continuously."""
    
    channels = [
        {
            'name': 'TCM-Chan (中醫陳醫師)',
            'log_file': Path('tcm_chan_download.log'),
            'total_videos': 59
        },
        {
            'name': '養生之道健康長壽',
            'log_file': Path('health_longevity_download.log'),
            'total_videos': 20
        },
        {
            'name': 'dr.zhaopeng',
            'log_file': Path('dr_zhaopeng.log'),
            'total_videos': 0  # Will be updated when enumeration completes
        },
        {
            'name': 'yinfawuyou',
            'log_file': Path('yinfawuyou.log'),
            'total_videos': 0
        },
        {
            'name': '百歲人生的故事1',
            'log_file': Path('baisui_story.log'),
            'total_videos': 0
        },
        {
            'name': '樂享養生-un9dd',
            'log_file': Path('lexiang_yangsheng.log'),
            'total_videos': 0
        },
        {
            'name': '樂齡美活',
            'log_file': Path('leling_meihuo.log'),
            'total_videos': 0
        },
        {
            'name': '逍遙健康指南',
            'log_file': Path('xiaoyao_health.log'),
            'total_videos': 0
        },
        {
            'name': 'health-k6s',
            'log_file': Path('health_k6s.log'),
            'total_videos': 0
        },
        {
            'name': 'healthdiary7',
            'log_file': Path('healthdiary7.log'),
            'total_videos': 0
        },
        {
            'name': 'healthyeyes2',
            'log_file': Path('healthyeyes2.log'),
            'total_videos': 0
        }
    ]
    
    print("📊 Multi-Channel Download Monitor")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop monitoring\n")
    
    # Track last status to detect changes
    last_statuses = {}
    
    while True:
        try:
            # Clear screen for clean display
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("📊 Multi-Channel Download Monitor")
            print("="*80)
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
            
            all_completed = True
            
            for channel in channels:
                status = get_channel_status(
                    channel['log_file'], 
                    channel['name'], 
                    channel['total_videos']
                )
                
                if status:
                    # Update total_videos if discovered
                    if status['total_videos'] > 0:
                        channel['total_videos'] = status['total_videos']
                    
                    # Check if completed
                    if status['completed'] < status['total_videos']:
                        all_completed = False
                    
                    # Build display
                    print(f"📺 {channel['name']}")
                    print(f"   Progress: {status['completed']}/{status['total_videos']} ({status['progress_pct']:.1f}%)")
                    print(f"   Started: {status['started']} | Chinese: {status['zh_files']} | English: {status['en_files']}")
                    print(f"   Whisper: {status['whisper']} | Rate limits: {status['rate_limits']}")
                    print(f"   Last: {status['last_op']}")
                    
                    # Progress bar
                    bar_length = 40
                    filled = int(bar_length * status['progress_pct'] / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"   [{bar}]")
                    
                    # Check for completion milestones
                    channel_key = channel['name']
                    if channel_key in last_statuses:
                        if status['completed'] > last_statuses[channel_key]['completed']:
                            milestone = status['completed']
                            if milestone % 5 == 0 or milestone == channel['total_videos']:
                                print(f"   🎉 Milestone: {milestone} videos completed!")
                    
                    last_statuses[channel_key] = status
                else:
                    print(f"📺 {channel['name']}")
                    print(f"   Status: Not started or log file not found")
                    all_completed = False
                
                print()
            
            # Summary statistics
            total_videos = sum(c['total_videos'] for c in channels)
            total_completed = sum(
                get_channel_status(c['log_file'], c['name'], c['total_videos'])['completed'] 
                for c in channels 
                if get_channel_status(c['log_file'], c['name'], c['total_videos'])
            )
            
            print("="*80)
            print(f"📈 Overall Progress: {total_completed}/{total_videos} ({total_completed/total_videos*100:.1f}%)")
            
            # Estimate time remaining (rough estimate)
            if total_completed > 0:
                elapsed = (datetime.now() - datetime(2025, 9, 10, 16, 58, 0)).total_seconds() / 60
                rate = total_completed / elapsed if elapsed > 0 else 0
                remaining = (total_videos - total_completed) / rate if rate > 0 else 999
                print(f"⏱️  Estimated time remaining: {int(remaining)} minutes")
            
            if all_completed:
                print("\n🎉 🎉 🎉 ALL DOWNLOADS COMPLETED! 🎉 🎉 🎉")
                break
            
            time.sleep(10)  # Update every 10 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Monitoring stopped by user")
            print(f"Final status: {total_completed}/{total_videos} videos completed")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_continuous()