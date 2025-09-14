#!/usr/bin/env python3
"""
Continuous Channel Processing Monitor
Reports status every 5 minutes with progress tracking
"""

import time
import subprocess
import re
from datetime import datetime

def get_bash_output(bash_id):
    """Get output from a background bash process"""
    try:
        result = subprocess.run(['python3', '-c', f'''
import subprocess
result = subprocess.run(["curl", "-s", "http://localhost:8000/bash/{bash_id}/output"], 
                       capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "")
'''], capture_output=True, text=True, timeout=10)
        return result.stdout
    except:
        return ""

def extract_progress_info(output):
    """Extract current progress from sequential processor output"""
    lines = output.strip().split('\n')
    
    # Look for video completion patterns
    completed_videos = []
    current_video = None
    total_videos = 66  # Default from earlier enumeration
    channel_status = "Starting..."
    
    for line in lines[-50:]:  # Check last 50 lines for recent activity
        # Extract completed video count
        if "✅ [" in line and "/66] Completed" in line:
            match = re.search(r'✅ \[(\d+)/(\d+)\] Completed', line)
            if match:
                completed = int(match.group(1))
                total = int(match.group(2))
                total_videos = total
                completed_videos.append(completed)
        
        # Extract current video being processed
        if "🎬 [" in line and "Starting download" in line:
            match = re.search(r'🎬 \[(\d+)/(\d+)\] Starting download', line)
            if match:
                current_video = int(match.group(1))
                total_videos = int(match.group(2))
        
        # Check for channel completion
        if "✅ SUCCESS:" in line and "completed successfully!" in line:
            channel_status = "Channel completed successfully!"
        elif "❌ ERROR:" in line and "failed with code" in line:
            channel_status = "Channel failed!"
    
    # Get the highest completed video number
    max_completed = max(completed_videos) if completed_videos else 0
    
    return {
        'total_videos': total_videos,
        'completed_videos': max_completed,
        'current_video': current_video or (max_completed + 1 if max_completed < total_videos else max_completed),
        'channel_status': channel_status,
        'progress_percentage': round((max_completed / total_videos) * 100, 1) if total_videos > 0 else 0
    }

def format_eta(completed, total, start_time):
    """Calculate and format ETA based on current progress"""
    if completed <= 0:
        return "Calculating..."
    
    elapsed_time = time.time() - start_time
    rate = completed / elapsed_time  # videos per second
    remaining_videos = total - completed
    eta_seconds = remaining_videos / rate if rate > 0 else 0
    
    if eta_seconds < 3600:  # Less than 1 hour
        return f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
    else:  # More than 1 hour
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def print_status_report(progress_info, start_time):
    """Print a comprehensive status report"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*80}")
    print(f"🔄 CHANNEL PROCESSING STATUS REPORT - {now}")
    print(f"{'='*80}")
    
    # Channel 1 Status
    print(f"📊 CHANNEL 1: @百歲人生的故事1 (Century Life Stories)")
    print(f"   Status: {progress_info['channel_status']}")
    print(f"   Progress: {progress_info['completed_videos']}/{progress_info['total_videos']} videos ({progress_info['progress_percentage']}%)")
    print(f"   Current: Processing video #{progress_info['current_video']}")
    
    if progress_info['completed_videos'] > 0:
        eta = format_eta(progress_info['completed_videos'], progress_info['total_videos'], start_time)
        print(f"   ETA: {eta} remaining")
    
    # Queue Status
    print(f"\n⏸️  QUEUE STATUS:")
    if progress_info['progress_percentage'] < 100:
        print(f"   ⏳ Channel 2: @樂享養生-un9dd (Enjoy Health) - WAITING")
        print(f"   ⏳ Channel 3: @樂齡美活 (Happy Aging) - WAITING")
    else:
        print(f"   ➡️  Processing next channel...")
    
    print(f"{'='*80}")

def main():
    """Main monitoring loop"""
    print("🚀 Starting Continuous Channel Processing Monitor")
    print("📊 Reporting every 5 minutes...")
    print("Press Ctrl+C to stop")
    
    start_time = time.time()
    
    try:
        while True:
            # Get output from the main sequential processor
            output = ""
            try:
                with open("flexible_sequential_health_channels.log", "r", encoding='utf-8') as f:
                    output = f.read()
            except FileNotFoundError:
                output = "Log file not found - processor may not be running"
            
            # Extract progress information
            progress_info = extract_progress_info(output)
            
            # Print status report
            print_status_report(progress_info, start_time)
            
            # Wait 5 minutes
            print(f"⏱️  Next report in 5 minutes... (monitoring continues)")
            time.sleep(300)  # 5 minutes = 300 seconds
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")
        print(f"📊 Final report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
