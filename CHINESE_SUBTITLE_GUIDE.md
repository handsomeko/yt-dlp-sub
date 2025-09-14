# Chinese Subtitle Extraction Guide

## The Problem

Many Chinese YouTube videos have a specific subtitle structure that makes getting native Chinese subtitles challenging:

1. **Original audio**: Chinese spoken language
2. **Auto-generated captions**: YouTube generates ENGLISH captions from the Chinese audio
3. **Auto-translations**: Other languages (including Chinese) are auto-translated FROM the English captions

This creates a problematic workflow:
```
Chinese Audio → English Auto-Captions → Chinese Auto-Translation
```

The result is often poor quality Chinese subtitles due to the double translation.

## Current Implementation

We've implemented comprehensive language detection and Chinese subtitle prioritization:

### 1. Automatic Language Detection
- Detects Chinese characters in video titles and descriptions
- Properly identifies Chinese, Japanese, Korean, and Arabic content
- Passes detected language to all extraction methods

### 2. Chinese Subtitle Priority
For Chinese videos, the system now requests subtitles in this order:
- `zh-Hans-en` (Simplified Chinese auto-translated from English)
- `zh-Hant-en` (Traditional Chinese auto-translated from English)
- `zh-Hans`, `zh-Hant`, `zh-CN`, `zh-TW` (native Chinese if available)
- `en` (English as fallback)

### 3. Whisper Chinese Support
- Whisper now detects Chinese content and transcribes in Chinese
- Provides accurate native Chinese transcripts directly from audio
- Avoids the Chinese→English→Chinese translation problem

## Solutions

### Option 1: Wait for YouTube Rate Limits (Not Recommended)
YouTube heavily rate-limits subtitle downloads. You might get Chinese subtitles by:
- Waiting several minutes between attempts
- Using the retry logic we've implemented
- But these are still re-translations, not native Chinese

### Option 2: Use Whisper Transcription (Recommended)
For accurate Chinese transcripts, use Whisper to transcribe directly from audio:

```bash
# Use the provided helper script
python transcribe_chinese.py <video_id> [channel_id]

# Example:
python transcribe_chinese.py SON6hKNHaDM UCYcMQmLxOKd9TMZguFEotww
```

This will:
- Load the audio file
- Detect Chinese language from metadata
- Transcribe using Whisper with Chinese language setting
- Save as `.zh.srt` and `.zh.txt` files

### Option 3: Manual Download (For Testing)
```bash
# Try to download Chinese subtitles manually
yt-dlp --write-auto-sub --sub-langs "zh-Hans-en,zh-Hant-en" \
       --skip-download --retries 5 --retry-sleep 10 \
       "https://www.youtube.com/watch?v=VIDEO_ID" \
       -o "output.%(ext)s"
```

## Why This Happens

YouTube's subtitle system for non-English content:
1. **Speech Recognition**: YouTube's AI converts Chinese speech to English text first
2. **Translation Service**: Then translates the English back to Chinese
3. **Quality Loss**: Each translation step loses accuracy and context

Native Chinese subtitles are only available when:
- The creator manually uploads Chinese subtitles
- The creator enables community contributions
- YouTube's Chinese speech recognition is activated (rare)

## Recommendations

1. **For Content Creators**: Always upload native language subtitles
2. **For Downloaders**: Use Whisper for accurate Chinese transcription
3. **For Developers**: Implement fallback to Whisper when native subtitles unavailable

## Technical Details

### Files Modified
- `core/subtitle_extractor_v2.py`: Added language detection and Chinese priority
- `workers/transcriber.py`: Added Whisper Chinese language support
- `core/downloader.py`: Passes video metadata for language detection

### Rate Limiting
YouTube aggressively rate-limits subtitle API calls:
- 429 errors are common, especially for auto-translated subtitles
- Exponential backoff implemented: 5s, 10s, 20s retries
- Circuit breaker pattern prevents cascading failures

### Language Detection Regex
```python
# Chinese characters (CJK Unified Ideographs)
chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]+'

# Japanese (Hiragana + Katakana)
japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff]+'

# Korean (Hangul)
korean_pattern = r'[\uac00-\ud7af\u1100-\u11ff]+'
```

## Future Improvements

1. **Caching**: Cache successful Chinese subtitle downloads
2. **Parallel Whisper**: Run Whisper in parallel with subtitle extraction
3. **Language Model**: Use AI to improve Chinese→English→Chinese translations
4. **Direct API**: Investigate YouTube Data API v3 for better subtitle access