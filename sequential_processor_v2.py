#!/usr/bin/env python3
"""
FLEXIBLE SEQUENTIAL CHANNEL PROCESSOR V3
Handles any number of channels dynamically with real-time output
Supports command line arguments, config files, and URL lists
"""

import subprocess
import time
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def extract_channel_name_from_url(url):
    """Extract channel name from URL for display"""
    if '@' in url:
        # Extract @channelname from URLs like https://www.youtube.com/@TCM-Chan
        parts = url.split('@')
        if len(parts) > 1:
            name = parts[1].split('/')[0].split('?')[0]  # Remove path and query params
            return f"@{name}"
    return url.split('/')[-1] or url

def parse_channel_input(channels_input):
    """Parse channels from various input formats"""
    channels = []
    
    for item in channels_input:
        if item.startswith('http') or item.startswith('@'):
            # Single URL or handle
            channel_name = extract_channel_name_from_url(item)
            channels.append((item, channel_name))
        elif item.endswith('.json') and Path(item).exists():
            # JSON config file
            with open(item, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'channels' in config:
                    for channel in config['channels']:
                        if isinstance(channel, dict):
                            url = channel.get('url', '')
                            name = channel.get('name', extract_channel_name_from_url(url))
                            channels.append((url, name))
                        else:
                            # String URL
                            name = extract_channel_name_from_url(channel)
                            channels.append((channel, name))
        elif item.endswith('.txt') and Path(item).exists():
            # Text file with URLs
            with open(item, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        name = extract_channel_name_from_url(line)
                        channels.append((line, name))
        else:
            # Assume it's a channel handle or name
            name = f"@{item}" if not item.startswith('@') else item
            url = f"https://www.youtube.com/{name}"
            channels.append((url, name))
    
    return channels

def process_channel(channel_url, channel_name):
    """Process a single channel completely with real-time output"""
    log(f"{'='*60}")
    log(f"🎯 STARTING: {channel_name}")
    log(f"📍 URL: {channel_url}")
    log(f"{'='*60}")
    
    # EMERGENCY FIX: Use proper phase separation for 10x faster processing
    # Phase 1: Download all videos (fast, skip expensive operations)
    cmd = [
        "python3", "cli.py", "channel", "download",
        channel_url,
        "--audio-only",
        "--skip-transcription",  # Skip Whisper for speed
        "--skip-punctuation",    # Skip punctuation for speed
        "--concurrent", "1"      # Use serial processing for reliability
    ]
    
    log(f"📥 Downloading ALL videos from {channel_name}...")
    log(f"⚙️  Command: {' '.join(cmd)}")
    log(f"📊 Real-time progress below:")
    log(f"{'-'*60}")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        line_count = 0
        for line in process.stdout:
            print(f"  │ {line.rstrip()}")
            sys.stdout.flush()
            line_count += 1
            
            # Show progress indicator every 10 lines
            if line_count % 10 == 0:
                log(f"  ⏳ Processing... ({line_count} lines processed)")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            log(f"{'-'*60}")
            log(f"✅ SUCCESS: {channel_name} completed successfully!")
            return True
        
        else:
            log(f"{'-'*60}")
            log(f"❌ ERROR: {channel_name} failed with code {return_code}")
            return False
            
    except Exception as e:
        log(f"❌ EXCEPTION: {str(e)}")
        return False

def create_sample_config():
    """Create a sample config file for reference"""
    sample_config = {
        "channels": [
            {
                "url": "https://www.youtube.com/@百歲人生的故事1",
                "name": "Century Life Stories"
            },
            {
                "url": "https://www.youtube.com/@樂享養生-un9dd", 
                "name": "Enjoy Health"
            },
            {
                "url": "https://www.youtube.com/@樂齡美活",
                "name": "Happy Aging"
            }
        ]
    }
    
    with open('channels_config_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    log("📝 Created sample config: channels_config_sample.json")

def main():
    """Main flexible sequential processor"""
    parser = argparse.ArgumentParser(
        description="Flexible Sequential Channel Processor - Process any number of YouTube channels sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process channels by URL
  python3 sequential_processor_v2.py https://www.youtube.com/@channel1 @channel2
  
  # Process from config file
  python3 sequential_processor_v2.py --config channels.json
  
  # Process from text file
  python3 sequential_processor_v2.py --file channels.txt
  
  # Mix multiple sources
  python3 sequential_processor_v2.py @channel1 --config channels.json https://youtube.com/@channel2
  
  # Create sample config
  python3 sequential_processor_v2.py --create-sample-config
  
  # The 3 new health channels from user request
  python3 sequential_processor_v2.py https://www.youtube.com/@百歲人生的故事1 https://www.youtube.com/@樂享養生-un9dd https://www.youtube.com/@樂齡美活
        """)
    
    parser.add_argument('channels', nargs='*', 
                       help='YouTube channel URLs, handles, or config files')
    parser.add_argument('--config', '-c', 
                       help='JSON config file with channel list')
    parser.add_argument('--file', '-f',
                       help='Text file with channel URLs (one per line)')
    parser.add_argument('--create-sample-config', action='store_true',
                       help='Create a sample JSON config file')
    parser.add_argument('--pause', '-p', type=int, default=5,
                       help='Seconds to pause between channels (default: 5)')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing even if a channel fails')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Collect all channel inputs
    all_inputs = []
    if args.channels:
        all_inputs.extend(args.channels)
    if args.config:
        all_inputs.append(args.config)
    if args.file:
        all_inputs.append(args.file)
    
    if not all_inputs:
        # Default to the 3 health channels the user requested
        log("⚠️  No channels specified, using the 3 new health and longevity channels")
        all_inputs = [
            "https://www.youtube.com/@百歲人生的故事1",
            "https://www.youtube.com/@樂享養生-un9dd", 
            "https://www.youtube.com/@樂齡美活"
        ]
    
    # Parse all channel inputs
    try:
        channels = parse_channel_input(all_inputs)
    except Exception as e:
        log(f"❌ Error parsing channel inputs: {e}")
        sys.exit(1)
    
    if not channels:
        log("❌ No valid channels found to process")
        sys.exit(1)
    
    # Display processing plan
    log("🚀 STARTING FLEXIBLE SEQUENTIAL CHANNEL PROCESSOR V3")
    log(f"📊 Total channels to process: {len(channels)}")
    log(f"⏸️  Pause between channels: {args.pause} seconds")
    log(f"🔄 Continue on error: {args.continue_on_error}")
    log("🔄 Real-time output enabled")
    log("\n📋 Processing order:")
    for i, (url, name) in enumerate(channels, 1):
        log(f"  {i}. {name} ({url})")
    
    # Process each channel sequentially
    successful = 0
    failed = 0
    
    for i, (channel_url, channel_name) in enumerate(channels, 1):
        log(f"\n{'='*60}")
        log(f"📌 CHANNEL {i}/{len(channels)}: {channel_name}")
        log(f"{'='*60}")
        
        success = process_channel(channel_url, channel_name)
        
        if success:
            successful += 1
            log(f"✅ {channel_name} completed successfully!")
            if i < len(channels):
                log(f"⏸️  Pausing {args.pause} seconds before next channel...")
                time.sleep(args.pause)
        else:
            failed += 1
            log(f"❌ {channel_name} failed!")
            if not args.continue_on_error:
                log(f"🛑 Stopping sequential processing due to failure.")
                break
            elif i < len(channels):
                log(f"⏭️  Continuing to next channel despite failure...")
                time.sleep(args.pause)
    
    # Final summary
    log(f"\n{'='*60}")
    log("📊 SEQUENTIAL PROCESSING SUMMARY")
    log(f"✅ Successful: {successful}")
    log(f"❌ Failed: {failed}")
    log(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%")
    
    if failed == 0:
        log("🎉 ALL CHANNELS PROCESSED SUCCESSFULLY!")
        log("✅ Sequential processing complete")
    elif successful > 0:
        log("⚠️  PARTIAL SUCCESS - Some channels processed")
        log("💡 Check logs above for failure details")
    else:
        log("❌ ALL CHANNELS FAILED")
        log("💡 Check configuration and network connectivity")
        sys.exit(1)
    
    log("="*60)

if __name__ == "__main__":
    main()