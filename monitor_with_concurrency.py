#!/usr/bin/env python3
"""
Monitor all channel downloads and manage future downloads with SystemConcurrencyManager.
"""

import time
import os
import re
import json
from pathlib import Path
from datetime import datetime
from core.enhanced_system_concurrency_manager import get_enhanced_concurrency_manager, WorkerType

def get_progress_from_log(log_file):
    """Extract progress information from log file."""
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for completion markers
        completed = len(re.findall(r'✅.*Completed', content))
        
        # Look for current video being processed
        current_match = re.findall(r'\[(\d+)/(\d+)\]', content)
        if current_match:
            current, total = current_match[-1]
            return {
                'current': int(current),
                'total': int(total),
                'completed': completed,
                'percent': (int(current) / int(total)) * 100
            }
        
        # Look for enumeration status
        if '📊 Found' in content:
            found_match = re.search(r'📊 Found (\d+) videos', content)
            if found_match:
                total = int(found_match.group(1))
                return {
                    'current': 0,
                    'total': total,
                    'completed': 0,
                    'percent': 0
                }
    except Exception as e:
        pass
    
    return None

def monitor_continuous():
    """Monitor downloads and show enhanced system status."""
    
    # Get enhanced system concurrency manager with all 25 improvements
    sys_manager = get_enhanced_concurrency_manager()
    
    # Channels currently being processed (old method)
    active_channels = [
        {'name': 'dr.zhaopeng', 'log': 'dr_zhaopeng.log'},
        {'name': 'yinfawuyou', 'log': 'yinfawuyou.log'},
        {'name': '百歲人生的故事1', 'log': 'baisui_story.log'},
        {'name': '樂享養生-un9dd', 'log': 'lexiang_yangsheng.log'},
    ]
    
    # Channels pending (will use SystemConcurrencyManager)
    pending_channels = [
        'https://www.youtube.com/@樂齡美活',
        'https://www.youtube.com/@逍遙健康指南',
        'https://www.youtube.com/@health-k6s',
        'https://www.youtube.com/@healthdiary7',
        'https://www.youtube.com/@healthyeyes2'
    ]
    
    print("📊 Channel Download Monitor with System Concurrency Management")
    print("=" * 80)
    
    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Get enhanced system status
        status = sys_manager.get_system_status()
        
        print("📊 Enhanced Channel Download Monitor")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"System Load: {status['system']['cpu_percent']:.1f}% | Memory: {status['system']['memory_percent']:.1f}%")
        
        # Show worker status by type
        print("\n🔧 Worker Status:")
        for worker_type, counts in status['processes'].items():
            print(f"  {worker_type}: {counts['active']} active, {counts['running']} running")
        print("=" * 80)
        print()
        
        # Check active downloads
        all_completed = True
        for channel in active_channels:
            log_file = Path(channel['log'])
            progress = get_progress_from_log(log_file)
            
            if progress:
                if progress['current'] < progress['total']:
                    all_completed = False
                
                print(f"📺 {channel['name']}")
                print(f"   Progress: [{progress['current']}/{progress['total']}] {progress['percent']:.1f}%")
                
                # Progress bar
                bar_length = 40
                filled = int(bar_length * progress['percent'] / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{bar}]")
            else:
                print(f"📺 {channel['name']} - Waiting to start...")
                all_completed = False
            print()
        
        # Show pending channels
        if pending_channels:
            print("\n📋 Pending Channels (will use SystemConcurrencyManager):")
            for url in pending_channels:
                print(f"   • {url}")
        
        # Advice based on enhanced system metrics
        print("\n💡 Enhanced System Advice:")
        cpu_load = status['system']['cpu_percent']
        memory_load = status['system']['memory_percent']
        
        if cpu_load > 80 or memory_load > 85:
            print("   ⚠️  CRITICAL: System overloaded! Waiting for current downloads to complete...")
            print(f"   📊 CPU: {cpu_load:.1f}% | Memory: {memory_load:.1f}%")
            print("   ⏸️  Enhanced scaling will throttle new downloads")
        elif cpu_load > 60 or memory_load > 70:
            print("   ⚠️  High load - dynamic scaling adjusting concurrency")
            print(f"   📊 CPU: {cpu_load:.1f}% | Memory: {memory_load:.1f}%")
        else:
            print("   ✅ System healthy - enhanced scaling will optimize performance")
            print(f"   📊 CPU: {cpu_load:.1f}% | Memory: {memory_load:.1f}%")
        
        # Show scaling features
        if status['features']['gpu_enabled']:
            print("   🎮 GPU acceleration available for Whisper")
        if status['features']['ml_prediction']:
            print("   🔮 ML-based predictive scaling active")
        
        # Check if we should start new downloads with enhanced logic
        if all_completed and pending_channels and cpu_load < 70 and memory_load < 80:
            print("\n🚀 Starting next channel with SystemConcurrencyManager...")
            next_channel = pending_channels[0]
            print(f"   Next: {next_channel}")
            print("\n   Run this command manually:")
            print(f"   python3 cli.py channel download \"{next_channel}\" --audio-only")
            break
        
        time.sleep(10)
    
    print("\n✅ Ready to process remaining channels with proper concurrency control!")

if __name__ == "__main__":
    monitor_continuous()