#!/usr/bin/env python3
"""Monitor sequential processing progress"""

import time
import subprocess
import sys
from datetime import datetime
import os
from pathlib import Path

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def check_progress():
    """Check processing progress"""
    
    # Check for running python processes
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        python_processes = [line for line in result.stdout.split('\n') 
                          if 'python' in line and 'cli.py' in line]
        
        if python_processes:
            log(f"🔄 Active processes: {len(python_processes)}")
            for proc in python_processes[:3]:  # Show first 3
                # Extract command part
                parts = proc.split()
                if len(parts) > 10:
                    cmd = ' '.join(parts[10:])[:100]  # First 100 chars of command
                    log(f"  → {cmd}...")
        else:
            log("⚠️  No active CLI processes found")
            
    except Exception as e:
        log(f"Error checking processes: {e}")
    
    # Check download directory for progress
    download_path = Path("/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads")
    
    # Check Dr. Zhao Peng directory
    dr_zhao_dir = download_path / "UCde0vB0fTwC8AT3sJofFV0w"
    if dr_zhao_dir.exists():
        video_dirs = [d for d in dr_zhao_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        log(f"📁 Dr. Zhao Peng videos: {len(video_dirs)}")
        
        # Check for latest activity
        if video_dirs:
            latest = max(video_dirs, key=lambda d: d.stat().st_mtime)
            log(f"  📝 Latest: {latest.name}")
            
            # Check for opus files
            opus_files = list(latest.glob("media/*.opus"))
            if opus_files:
                log(f"    ✅ Audio downloaded")
            
            # Check for transcripts
            transcript_files = list(latest.glob("transcripts/*.txt"))
            if transcript_files:
                log(f"    ✅ Transcript created")
    
    # Check Yinfawuyou directory
    yinfawuyou_dir = download_path / "UCYcMQmLxOKd9TMZguFEotww"
    if yinfawuyou_dir.exists():
        video_dirs = [d for d in yinfawuyou_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        log(f"📁 Yinfawuyou videos: {len(video_dirs)}")

def main():
    log("🔍 MONITORING SEQUENTIAL PROCESSING")
    log("=" * 60)
    
    while True:
        check_progress()
        log("-" * 40)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()