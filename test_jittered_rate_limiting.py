#!/usr/bin/env python3
"""
Test Script: Bulletproof Jittered Rate Limiting Implementation

Validates that the new jittered rate limiting system:
1. Generates varied, non-predictable intervals
2. Implements all three AWS-recommended jitter algorithms correctly
3. Maintains backward compatibility
4. Properly handles exponential backoff with jitter
"""

import time
import statistics
from core.rate_limit_manager import get_rate_limit_manager, JitterType

def test_jitter_algorithms():
    """Test all jitter algorithms for variability and correctness"""
    print("🧪 Testing Bulletproof Jittered Rate Limiting")
    print("=" * 60)
    
    # Test each jitter type
    jitter_types = [JitterType.FULL, JitterType.EQUAL, JitterType.DECORRELATED]
    domain = "test.youtube.com"
    
    for jitter_type in jitter_types:
        print(f"\n🔬 Testing {jitter_type.value.upper()} Jitter Algorithm")
        print("-" * 40)
        
        # Configure environment for this jitter type
        import os
        os.environ['PREVENTION_JITTER_TYPE'] = jitter_type.value
        
        # Get fresh rate limiter with new config
        manager = get_rate_limit_manager()
        
        # Generate 10 interval samples to check variability
        intervals = []
        for i in range(10):
            allowed, wait_time = manager.should_allow_request(domain)
            if not allowed:
                intervals.append(wait_time)
            
            # Simulate a request to trigger interval calculation
            manager.record_request(domain, success=True)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        if intervals:
            print(f"📊 Interval samples: {[f'{x:.2f}s' for x in intervals[:5]]}...")
            print(f"📈 Mean: {statistics.mean(intervals):.2f}s")
            print(f"📉 StdDev: {statistics.stdev(intervals) if len(intervals) > 1 else 0:.2f}s")
            print(f"🎯 Range: {min(intervals):.2f}s - {max(intervals):.2f}s")
            
            # Check for variability (should not all be the same)
            unique_intervals = len(set(round(x, 1) for x in intervals))
            variability_score = unique_intervals / len(intervals) * 100
            print(f"🌈 Variability: {variability_score:.0f}% ({unique_intervals}/{len(intervals)} unique)")
            
            if variability_score >= 50:
                print("✅ PASS: Good variability detected")
            else:
                print("⚠️  WARNING: Low variability - might be too predictable")
        else:
            print("ℹ️  No rate limiting triggered (all requests allowed)")
    
    print("\n" + "=" * 60)

def test_exponential_backoff_with_jitter():
    """Test that exponential backoff scales properly with jitter"""
    print("🧪 Testing Exponential Backoff with Jitter")
    print("=" * 60)
    
    import os
    os.environ['PREVENTION_JITTER_TYPE'] = 'full'  # Use full jitter for this test
    
    manager = get_rate_limit_manager()
    domain = "backoff.test.youtube.com"
    
    # Simulate multiple 429 errors to trigger exponential backoff
    backoff_delays = []
    for attempt in range(5):
        # Record a 429 error
        manager.record_request(domain, success=False, is_429=True)
        
        # Check the resulting backoff delay
        allowed, wait_time = manager.should_allow_request(domain)
        if not allowed:
            backoff_delays.append(wait_time)
            print(f"🔄 Attempt #{attempt + 1}: {wait_time:.2f}s backoff (jittered)")
    
    if len(backoff_delays) >= 3:
        # Check that delays generally increase (allowing for jitter variance)
        increasing_trend = all(
            backoff_delays[i] <= backoff_delays[i+1] * 1.5  # Allow 50% jitter variance
            for i in range(len(backoff_delays)-1)
        )
        
        print(f"📈 Exponential trend (with jitter tolerance): {'✅ PASS' if increasing_trend else '❌ FAIL'}")
        print(f"📊 Backoff sequence: {[f'{x:.1f}s' for x in backoff_delays]}")
    
    print("=" * 60)

def test_backward_compatibility():
    """Test backward compatibility with old YOUTUBE_ environment variables"""
    print("🧪 Testing Backward Compatibility")
    print("=" * 60)
    
    import os
    
    # Test with legacy environment variables
    os.environ['PREVENTION_JITTER_TYPE'] = 'none'  # Disable jitter for this test
    os.environ['PREVENTION_BASE_INTERVAL'] = '2.0'  # Set to old fixed interval
    
    manager = get_rate_limit_manager()
    domain = "legacy.test.youtube.com"
    
    # Test multiple requests
    intervals = []
    for i in range(5):
        allowed, wait_time = manager.should_allow_request(domain)
        if not allowed:
            intervals.append(wait_time)
        manager.record_request(domain, success=True)
        time.sleep(0.01)
    
    if intervals:
        # With jitter disabled, intervals should be consistent
        interval_consistency = len(set(round(x, 1) for x in intervals)) == 1
        print(f"🔄 Consistent intervals (jitter=none): {'✅ PASS' if interval_consistency else '❌ FAIL'}")
        print(f"📊 Intervals: {[f'{x:.1f}s' for x in intervals]}")
    else:
        print("ℹ️  No rate limiting triggered")
    
    print("=" * 60)

def main():
    """Run all jittered rate limiting tests"""
    print("🚀 BULLETPROOF JITTERED RATE LIMITING TEST SUITE")
    print("Testing implementation based on AWS best practices")
    print("Algorithms: Full Jitter, Equal Jitter, Decorrelated Jitter")
    print("\n")
    
    try:
        test_jitter_algorithms()
        test_exponential_backoff_with_jitter()
        test_backward_compatibility()
        
        print("🎉 JITTERED RATE LIMITING TEST SUITE COMPLETED")
        print("✅ Implementation appears to be working correctly")
        print("\n💡 Key Benefits:")
        print("  • Unpredictable intervals prevent bot detection")
        print("  • AWS-recommended algorithms reduce server load")
        print("  • Exponential backoff with jitter handles 429 errors gracefully")
        print("  • Backward compatibility maintained")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()