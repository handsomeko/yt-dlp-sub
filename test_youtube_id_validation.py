#!/usr/bin/env python3
"""
Test YouTube ID validation system to ensure invalid IDs are properly filtered.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.youtube_validators import (
    is_valid_youtube_id, 
    is_valid_channel_id,
    filter_valid_video_ids,
    validate_and_clean_video_list,
    extract_video_id_from_url
)


def test_valid_youtube_ids():
    """Test valid YouTube IDs are accepted."""
    valid_ids = [
        "dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
        "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
        "9bZkp7q19f0",  # Gangnam Style
        "kJQP7kiw5Fk",  # Despacito
        "1Vc5vAIhHPw",  # TCM-Chan video 58
        "x7Fx8zg8HN4",  # TCM-Chan video 59
    ]
    
    print("Testing VALID YouTube IDs:")
    for video_id in valid_ids:
        result = is_valid_youtube_id(video_id)
        status = "✅" if result else "❌"
        print(f"  {status} {video_id}: {result}")
    
    print()


def test_invalid_youtube_ids():
    """Test invalid YouTube IDs are rejected."""
    invalid_ids = [
        "-fXd0uINhFM",  # Starts with -
        "_bgDeaaO0wQ",  # Starts with _
        ".startwithdot",  # Starts with .
        "endwithdot.",  # Ends with .
        "endwithdash-",  # Ends with -
        "endwithunder_",  # Ends with _
        "tooshort",  # Wrong length (9 chars)
        "waytoolongidhere",  # Wrong length (16 chars)
        "",  # Empty
        None,  # None
        "has spaces",  # Contains spaces
        "has@special",  # Contains invalid character
    ]
    
    print("Testing INVALID YouTube IDs:")
    for video_id in invalid_ids:
        result = is_valid_youtube_id(video_id)
        status = "❌" if result else "✅"  # Should be False for invalid IDs
        print(f"  {status} {video_id!r}: {result}")
    
    print()


def test_filter_video_list():
    """Test filtering a mixed list of videos."""
    videos = [
        {'video_id': 'dQw4w9WgXcQ', 'title': 'Valid Video 1'},
        {'video_id': '-fXd0uINhFM', 'title': 'Invalid Video 1'},
        {'video_id': 'jNQXAC9IVRw', 'title': 'Valid Video 2'},
        {'video_id': '_bgDeaaO0wQ', 'title': 'Invalid Video 2'},
        {'video_id': '1Vc5vAIhHPw', 'title': 'Valid Video 3'},
        {'id': 'x7Fx8zg8HN4', 'title': 'Valid Video 4 (using id field)'},
        {'video_id': 'tooshort', 'title': 'Invalid Video 3'},
    ]
    
    print("Testing filter_valid_video_ids:")
    print(f"  Input: {len(videos)} videos")
    
    valid_videos = filter_valid_video_ids(videos, log_invalid=False)
    invalid_count = len(videos) - len(valid_videos)
    
    print(f"  Output: {len(valid_videos)} valid, {invalid_count} invalid filtered")
    print(f"  Valid IDs: {[v.get('video_id') or v.get('id') for v in valid_videos]}")
    
    print()


def test_channel_ids():
    """Test channel ID validation."""
    test_cases = [
        ("UCYcMQmLxOKd9TMZguFEotww", True),  # TCM-Chan channel ID
        ("UCde0vB0fTwC8AT3sJofFV0w", True),  # Valid channel ID
        ("UC" + "x" * 22, True),  # Valid format
        ("NotAChannelID", False),  # Invalid format
        ("UC", False),  # Too short
        ("", False),  # Empty
        (None, False),  # None
    ]
    
    print("Testing channel ID validation:")
    for channel_id, expected in test_cases:
        result = is_valid_channel_id(channel_id)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {channel_id!r}: {result} (expected {expected})")
    
    print()


def test_url_extraction():
    """Test extracting video IDs from URLs."""
    urls = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/jNQXAC9IVRw", "jNQXAC9IVRw"),
        ("https://youtube.com/embed/9bZkp7q19f0", "9bZkp7q19f0"),
        ("https://www.youtube.com/watch?v=-fXd0uINhFM", None),  # Invalid ID
        ("https://www.youtube.com/watch?v=_bgDeaaO0wQ", None),  # Invalid ID
        ("not a url", None),
        ("https://www.youtube.com/channel/UCYcMQmLxOKd9TMZguFEotww", None),  # Channel URL
    ]
    
    print("Testing URL extraction:")
    for url, expected in urls:
        result = extract_video_id_from_url(url)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {url[:50]}...: {result!r} (expected {expected!r})")
    
    print()


def test_real_world_scenario():
    """Test a real-world scenario with TCM-Chan videos."""
    print("Testing real-world scenario (TCM-Chan channel):")
    
    # Simulate videos from enumeration (including invalid ones)
    enumerated_videos = [
        {'video_id': '1Vc5vAIhHPw', 'title': 'Video 58'},
        {'video_id': 'x7Fx8zg8HN4', 'title': 'Video 59'},
        {'video_id': '-fXd0uINhFM', 'title': 'Invalid 1'},
        {'video_id': '_bgDeaaO0wQ', 'title': 'Invalid 2'},
    ]
    
    print(f"  Enumerated: {len(enumerated_videos)} videos")
    
    # Filter valid videos
    valid_videos, invalid_ids = validate_and_clean_video_list(enumerated_videos)
    
    print(f"  Valid: {len(valid_videos)} videos")
    print(f"  Invalid: {len(invalid_ids)} IDs filtered: {invalid_ids}")
    print(f"  Correct count for summary: {len(valid_videos)}/{len(valid_videos)} (not {len(valid_videos)}/{len(enumerated_videos)})")
    
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("YouTube ID Validation System Test")
    print("=" * 60)
    print()
    
    test_valid_youtube_ids()
    test_invalid_youtube_ids()
    test_filter_video_list()
    test_channel_ids()
    test_url_extraction()
    test_real_world_scenario()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()