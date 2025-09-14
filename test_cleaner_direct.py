#!/usr/bin/env python3
"""
Direct test of transcript cleaner with sample auto-generated content
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcript_cleaner import TranscriptCleaner


def test_direct_cleaning():
    """Test the cleaner directly with sample content"""
    
    print("\n" + "="*80)
    print("DIRECT TRANSCRIPT CLEANER TEST")
    print("="*80)
    
    cleaner = TranscriptCleaner()
    
    # Sample auto-generated SRT content (like YouTube provides)
    auto_srt = """Kind: captions
Language: en
1
00:00:00,160 --> 00:00:05,600 align:start position:0%
Have<00:00:00.480><c> you</c><00:00:00.640><c> ever</c><00:00:00.879><c> wondered</c><00:00:01.360><c> how</c><00:00:01.520><c> apps</c><00:00:01.839><c> like</c>
2
00:00:02,480 --> 00:00:07,680 align:start position:0%
Have you ever wondered how apps like
this<00:00:02.800><c> one</c><00:00:03.120><c> are</c><00:00:03.280><c> making</c>
3
00:00:05,600 --> 00:00:10,480 align:start position:0%
this one are making $500,000
every<00:00:06.240><c> single</c><00:00:06.640><c> month?</c>"""

    # Sample auto-generated TXT content
    auto_txt = """Kind: captions
Language: en
Have<00:00:00.480><c> you</c><00:00:00.640><c> ever</c><00:00:00.879><c> wondered</c>
Have you ever wondered how apps like
Have you ever wondered how apps like
this<00:00:02.800><c> one</c><00:00:03.120><c> are</c><00:00:03.280><c> making</c>
this one are making $500,000
this one are making $500,000
every<00:00:06.240><c> single</c><00:00:06.640><c> month?</c>
every single month?
every single month?"""

    # Test SRT cleaning
    print("\n📄 Testing SRT Cleaning:")
    print("-" * 40)
    print("Original SRT size:", len(auto_srt), "bytes")
    
    is_auto = cleaner.is_auto_generated(auto_srt)
    print(f"Detected as auto-generated: {is_auto}")
    
    cleaned_srt = cleaner.clean_auto_srt(auto_srt)
    print("Cleaned SRT size:", len(cleaned_srt), "bytes")
    print(f"Size reduction: {(1 - len(cleaned_srt)/len(auto_srt)) * 100:.1f}%")
    
    is_valid, error = cleaner.validate_srt_format(cleaned_srt)
    print(f"Valid SRT format: {is_valid}")
    if error:
        print(f"  Error: {error}")
    
    print("\nCleaned SRT content:")
    print("-" * 40)
    print(cleaned_srt)
    
    # Test TXT cleaning
    print("\n📄 Testing TXT Cleaning:")
    print("-" * 40)
    print("Original TXT size:", len(auto_txt), "bytes")
    
    is_auto = cleaner.is_auto_generated(auto_txt)
    print(f"Detected as auto-generated: {is_auto}")
    
    cleaned_txt = cleaner.clean_auto_txt(auto_txt)
    print("Cleaned TXT size:", len(cleaned_txt), "bytes")
    print(f"Size reduction: {(1 - len(cleaned_txt)/len(auto_txt)) * 100:.1f}%")
    
    print("\nCleaned TXT content:")
    print("-" * 40)
    print(cleaned_txt)
    
    # Test with clean content (should not be modified)
    print("\n📄 Testing with already clean content:")
    print("-" * 40)
    
    clean_srt = """1
00:00:00,000 --> 00:00:03,000
This is clean subtitle text.

2
00:00:03,000 --> 00:00:06,000
No XML tags or duplicates here."""

    is_auto = cleaner.is_auto_generated(clean_srt)
    print(f"Clean SRT detected as auto-generated: {is_auto}")
    
    if not is_auto:
        print("✅ Correctly identified as NOT auto-generated")
    else:
        print("❌ Incorrectly identified as auto-generated")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_direct_cleaning()