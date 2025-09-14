#!/usr/bin/env python3
"""
Simple direct test of Whisper on the downloaded audio file.
"""

import sys
from pathlib import Path

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Find audio file
channel_id = "UCde0vB0fTwC8AT3sJofFV0w"
video_id = "0S6IHp83zws"
audio_path = f"/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/{channel_id}/{video_id}/media/How to Build a Simple App for the App Store Using AI (For Beginners).opus"

print(f"Testing Whisper on: {audio_path}")
print(f"File exists: {Path(audio_path).exists()}")

if not Path(audio_path).exists():
    print("ERROR: Audio file not found!")
    sys.exit(1)

# Test 1: Try importing whisper
try:
    import whisper
    print("✅ Whisper library imported successfully")
except ImportError as e:
    print(f"❌ Whisper not installed: {e}")
    print("Run: pip install openai-whisper")
    sys.exit(1)

# Test 2: Load Whisper model
try:
    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    sys.exit(1)

# Test 3: Transcribe the audio WITH TIMEOUT PROTECTION
try:
    # Add timeout protection for this test file
    sys.path.insert(0, '/Users/jk/yt-dl-sub')
    from core.audio_analyzer import AudioAnalyzer
    from core.whisper_timeout_manager import WhisperTimeoutManager
    
    print(f"Analyzing audio file for timeout protection...")
    analyzer = AudioAnalyzer()
    analysis = analyzer.analyze_audio(audio_path)
    
    if not analysis.is_valid:
        print(f"❌ Audio analysis failed: {analysis.error}")
        sys.exit(1)
    
    print(f"Audio duration: {analysis.duration_seconds:.1f}s, recommended timeout: {analysis.recommended_timeout}s")
    
    # Initialize timeout manager
    timeout_manager = WhisperTimeoutManager(max_concurrent_jobs=1, memory_limit_mb=4096)
    
    # Define transcription function for timeout wrapper
    def transcribe_with_whisper():
        return model.transcribe(
            audio_path,
            language=None,  # Auto-detect language
            task='transcribe',
            verbose=False
        )
    
    print(f"Transcribing audio file with timeout protection ({analysis.recommended_timeout}s)...")
    result, monitoring = timeout_manager.execute_with_timeout(
        func=transcribe_with_whisper,
        timeout_seconds=analysis.recommended_timeout,
        job_id="test_whisper_simple",
        monitor_resources=True
    )
    
    print("✅ Transcription completed with timeout protection!")
    print(f"Detected language: {result.get('language', 'unknown')}")
    print(f"Number of segments: {len(result.get('segments', []))}")
    print(f"Execution time: {monitoring.execution_time_seconds:.1f}s")
    print(f"Max memory used: {monitoring.max_memory_mb:.1f}MB")
    print(f"Max CPU: {monitoring.max_cpu_percent:.1f}%")
    
    # Get transcript text
    text = result.get('text', '')
    print(f"Transcript length: {len(text)} characters")
    print(f"Word count: {len(text.split())} words")
    
    # Show first 500 characters
    print("\nFirst 500 characters of transcript:")
    print("-" * 50)
    print(text[:500])
    print("-" * 50)
    
    # Save to file with _whisper suffix
    output_dir = Path(f"/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads/{channel_id}/{video_id}/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text file
    whisper_txt = output_dir / "How to Build a Simple App for the App Store Using AI (For Beginners)_whisper.txt"
    whisper_txt.write_text(text, encoding='utf-8')
    print(f"\n✅ Saved Whisper transcript to: {whisper_txt.name}")
    
    # Create SRT format
    srt_lines = []
    for i, segment in enumerate(result['segments'], 1):
        start_time = f"{int(segment['start']//3600):02d}:{int(segment['start']%3600//60):02d}:{int(segment['start']%60):02d},{int((segment['start']%1)*1000):03d}"
        end_time = f"{int(segment['end']//3600):02d}:{int(segment['end']%3600//60):02d}:{int(segment['end']%60):02d},{int((segment['end']%1)*1000):03d}"
        
        srt_lines.append(str(i))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(segment['text'].strip())
        srt_lines.append("")
    
    srt_content = '\n'.join(srt_lines)
    
    # Save SRT file
    whisper_srt = output_dir / "How to Build a Simple App for the App Store Using AI (For Beginners)_whisper.srt"
    whisper_srt.write_text(srt_content, encoding='utf-8')
    print(f"✅ Saved Whisper SRT to: {whisper_srt.name}")
    
    # Compare with auto-generated transcript if it exists
    auto_txt = output_dir / "How to Build a Simple App for the App Store Using AI (For Beginners)_auto.en.txt"
    if auto_txt.exists():
        auto_text = auto_txt.read_text(encoding='utf-8')
        print(f"\n📊 Comparison with auto-generated transcript:")
        print(f"  Auto-generated: {len(auto_text)} characters, {len(auto_text.split())} words")
        print(f"  Whisper: {len(text)} characters, {len(text.split())} words")
        print(f"  Difference: {len(text) - len(auto_text):+} characters ({(len(text)/len(auto_text) - 1)*100:+.1f}%)")
        
        # Check punctuation
        auto_punct = sum(1 for c in auto_text if c in '.,!?;:')
        whisper_punct = sum(1 for c in text if c in '.,!?;:')
        print(f"  Punctuation - Auto: {auto_punct}, Whisper: {whisper_punct} ({(whisper_punct/auto_punct - 1)*100:+.1f}%)")
    else:
        print(f"\n⚠️ Auto-generated transcript not found for comparison")
    
except Exception as e:
    print(f"❌ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests completed successfully!")