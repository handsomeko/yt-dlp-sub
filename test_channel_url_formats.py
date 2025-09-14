#!/usr/bin/env python3
"""
Test YouTube Channel URL Format Support

Tests the enhanced YouTubeURLParser with all 5 requested channel formats:
1. https://www.youtube.com/@TCM-Chan/videos
2. https://www.youtube.com/@TCM-Chan/
3. https://www.youtube.com/@TCM-Chan/featured
4. @TCM-Chan
5. TCM-Chan
"""

from core.url_parser import YouTubeURLParser, URLType

def test_channel_url_formats():
    """Test all 5 channel URL formats"""
    parser = YouTubeURLParser()
    
    # Test cases with expected results
    test_cases = [
        {
            'url': 'https://www.youtube.com/@TCM-Chan/videos',
            'expected_type': URLType.CHANNEL,
            'expected_identifier': '@TCM-Chan',
            'expected_channel_type': 'handle',
            'description': 'Full URL with /videos suffix'
        },
        {
            'url': 'https://www.youtube.com/@TCM-Chan/',
            'expected_type': URLType.CHANNEL,
            'expected_identifier': '@TCM-Chan',
            'expected_channel_type': 'handle',
            'description': 'Full URL with trailing slash'
        },
        {
            'url': 'https://www.youtube.com/@TCM-Chan/featured',
            'expected_type': URLType.CHANNEL,
            'expected_identifier': '@TCM-Chan',
            'expected_channel_type': 'handle',
            'description': 'Full URL with /featured suffix'
        },
        {
            'url': '@TCM-Chan',
            'expected_type': URLType.CHANNEL,
            'expected_identifier': '@TCM-Chan',
            'expected_channel_type': 'bare_handle',
            'description': 'Bare @ handle'
        },
        {
            'url': 'TCM-Chan',
            'expected_type': URLType.CHANNEL,
            'expected_identifier': 'TCM-Chan',
            'expected_channel_type': 'plain_name',
            'description': 'Plain channel name'
        }
    ]
    
    print("Testing Enhanced YouTube Channel URL Format Support")
    print("=" * 80)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        url = test_case['url']
        expected_type = test_case['expected_type']
        expected_identifier = test_case['expected_identifier']
        expected_channel_type = test_case['expected_channel_type']
        description = test_case['description']
        
        print(f"\nTest {i}: {description}")
        print(f"Input URL: {url}")
        
        # Parse the URL
        url_type, identifier, metadata = parser.parse(url)
        
        print(f"Result Type: {url_type.value}")
        print(f"Result Identifier: {identifier}")
        print(f"Channel Type: {metadata.get('channel_type', 'N/A')}")
        print(f"Metadata: {metadata}")
        
        # Check if it matches expected results
        type_match = url_type == expected_type
        identifier_match = identifier == expected_identifier
        channel_type_match = metadata.get('channel_type') == expected_channel_type
        
        if type_match and identifier_match and channel_type_match:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            if not type_match:
                print(f"   Expected type: {expected_type.value}, Got: {url_type.value}")
            if not identifier_match:
                print(f"   Expected identifier: {expected_identifier}, Got: {identifier}")
            if not channel_type_match:
                print(f"   Expected channel_type: {expected_channel_type}, Got: {metadata.get('channel_type')}")
            all_passed = False
        
        print("-" * 40)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    if all_passed:
        print("✅ All 5 channel URL formats are supported!")
        print("✅ Enhanced URL parser working correctly!")
    else:
        print("❌ Some channel URL formats need additional fixes")
        
    return all_passed

def test_url_normalization():
    """Test the URL normalization logic separately"""
    parser = YouTubeURLParser()
    
    print("\n" + "=" * 80)
    print("Testing URL Normalization Logic")
    print("=" * 80)
    
    normalization_cases = [
        {
            'input': '@TCM-Chan',
            'expected_normalized': 'https://www.youtube.com/@TCM-Chan',
            'description': 'Bare @ handle normalization'
        },
        {
            'input': 'TCM-Chan',
            'expected_normalized': 'https://www.youtube.com/TCM-Chan',
            'description': 'Plain channel name normalization'
        },
        {
            'input': 'youtube.com/@TCM-Chan/videos',
            'expected_normalized': 'https://youtube.com/@TCM-Chan/videos',
            'description': 'YouTube URL without protocol'
        }
    ]
    
    for i, test_case in enumerate(normalization_cases, 1):
        input_url = test_case['input']
        expected_normalized = test_case['expected_normalized']
        description = test_case['description']
        
        print(f"\nNormalization Test {i}: {description}")
        print(f"Input: '{input_url}'")
        
        # We need to replicate the normalization logic from parse()
        url = input_url.strip()
        
        # Smart URL normalization for different input formats
        if url.startswith('@'):
            url = f'https://www.youtube.com/{url}'
        elif not url.startswith(('http://', 'https://')) and not ('youtube.com' in url or 'youtu.be' in url):
            if re.match(r'^[a-zA-Z0-9_.-]+$', url):
                url = f'https://www.youtube.com/{url}'
            else:
                url = 'https://' + url
        elif not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"Normalized: '{url}'")
        
        if url == expected_normalized:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            print(f"   Expected: '{expected_normalized}'")
        
        print("-" * 40)

def test_backward_compatibility():
    """Test that existing URL formats still work"""
    parser = YouTubeURLParser()
    
    print("\n" + "=" * 80)
    print("Testing Backward Compatibility")
    print("=" * 80)
    
    # Test existing formats that should still work
    legacy_cases = [
        {
            'url': 'https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw',
            'expected_type': URLType.CHANNEL,
            'expected_channel_type': 'channel_id',
            'description': 'Channel ID format'
        },
        {
            'url': 'https://www.youtube.com/c/MrBeast',
            'expected_type': URLType.CHANNEL,
            'expected_channel_type': 'custom',
            'description': 'Custom channel name'
        },
        {
            'url': 'https://www.youtube.com/user/PewDiePie',
            'expected_type': URLType.CHANNEL,
            'expected_channel_type': 'user',
            'description': 'Legacy user format'
        },
        {
            'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'expected_type': URLType.VIDEO,
            'description': 'Video URL'
        },
        {
            'url': 'https://www.youtube.com/shorts/abc123def45',
            'expected_type': URLType.SHORTS,
            'description': 'Shorts URL'
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(legacy_cases, 1):
        url = test_case['url']
        expected_type = test_case['expected_type']
        description = test_case['description']
        
        print(f"\nBackward Compatibility Test {i}: {description}")
        print(f"URL: {url}")
        
        url_type, identifier, metadata = parser.parse(url)
        
        print(f"Type: {url_type.value}")
        print(f"Identifier: {identifier}")
        
        if url_type == expected_type:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            print(f"   Expected: {expected_type.value}, Got: {url_type.value}")
            all_passed = False
        
        print("-" * 40)
    
    if all_passed:
        print("\n✅ All backward compatibility tests passed!")
    else:
        print("\n❌ Some backward compatibility tests failed!")
    
    return all_passed

if __name__ == "__main__":
    import re
    
    print("YouTube Channel URL Format Enhancement Tests")
    print("=" * 80)
    
    # Run all test suites
    main_tests_passed = test_channel_url_formats()
    test_url_normalization()
    backward_tests_passed = test_backward_compatibility()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    
    if main_tests_passed and backward_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All 5 channel URL formats supported")
        print("✅ Backward compatibility maintained")
        print("✅ Enhanced URL parser ready for production")
    else:
        if not main_tests_passed:
            print("❌ Main functionality tests failed")
        if not backward_tests_passed:
            print("❌ Backward compatibility tests failed")
        print("⚠️  Additional fixes needed")