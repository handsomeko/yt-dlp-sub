#!/usr/bin/env python3
"""
TRUE SEQUENTIAL CHANNEL PROCESSOR
Ensures each channel is 100% complete before starting the next
"""

import subprocess
import time
import sys
from datetime import datetime

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def process_channel(channel_url, channel_name):
    """Process a single channel completely"""
    log(f"{'='*60}")
    log(f"🎯 STARTING: {channel_name}")
    log(f"📍 URL: {channel_url}")
    log(f"{'='*60}")
    
    # Command to process ALL videos from channel
    cmd = [
        "python3", "cli.py", "channel", "download",
        channel_url,
        "--audio-only"
    ]
    
    log(f"📥 Downloading ALL videos from {channel_name}...")
    log(f"⚙️  Command: {' '.join(cmd)}")
    
    try:
        # Run the command and wait for completion
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per channel
        )
        
        if process.returncode == 0:
            log(f"✅ SUCCESS: {channel_name} completed successfully!")
            log(f"📊 Output summary (last 500 chars):")
            print(process.stdout[-500:])
            return True
        else:
            log(f"❌ ERROR: {channel_name} failed with code {process.returncode}")
            log(f"Error output: {process.stderr[-1000:]}")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"⏰ TIMEOUT: {channel_name} exceeded 2 hour limit")
        return False
    except Exception as e:
        log(f"❌ EXCEPTION: {str(e)}")
        return False

def main():
    """Main sequential processor"""
    log("🚀 STARTING TRUE SEQUENTIAL CHANNEL PROCESSOR")
    log("📋 Processing order: TCM-Chan → Dr. Zhao Peng → Yinfawuyou")
    
    channels = [
        # TCM-Chan already completed, commenting out
        # ("https://www.youtube.com/@TCM-Chan", "TCM-Chan"),
        ("https://www.youtube.com/@dr.zhaopeng", "Dr. Zhao Peng"),
        ("https://www.youtube.com/@yinfawuyou", "Yinfawuyou")
    ]
    
    for channel_url, channel_name in channels:
        log(f"\n{'='*60}")
        log(f"📌 CHANNEL {channels.index((channel_url, channel_name)) + 1}/{len(channels)}: {channel_name}")
        log(f"{'='*60}")
        
        success = process_channel(channel_url, channel_name)
        
        if success:
            log(f"✅ {channel_name} complete, moving to next channel...")
            time.sleep(5)  # Brief pause between channels
        else:
            log(f"❌ {channel_name} failed. Stopping sequential processing.")
            sys.exit(1)
    
    log("\n" + "="*60)
    log("🎉 ALL CHANNELS PROCESSED SUCCESSFULLY!")
    log("✅ Sequential processing complete")
    log("="*60)

if __name__ == "__main__":
    main()