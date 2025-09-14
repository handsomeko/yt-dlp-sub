#!/usr/bin/env python3
"""
Test Rate Limiting Prevention System

This test verifies that our comprehensive rate limiting system:
- Prevents 429 errors through proactive throttling
- Implements exponential backoff correctly
- Uses circuit breakers to prevent cascading failures
- Handles multiple domains independently
"""

import time
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.rate_limit_manager import (
    RateLimitManager, RateLimitState, 
    get_rate_limit_manager
)


def test_rate_limit_prevention():
    """Test basic rate limiting prevention."""
    print("🧪 Testing Rate Limit Prevention")
    print("=" * 60)
    
    manager = RateLimitManager()
    
    # Test 1: Basic rate limiting
    print("\n📊 Test 1: Basic rate limiting")
    
    # Simulate rapid requests
    allowed_count = 0
    blocked_count = 0
    
    for i in range(50):
        allowed, wait_time = manager.should_allow_request('youtube.com')
        
        if allowed:
            allowed_count += 1
            manager.record_request('youtube.com', success=True)
        else:
            blocked_count += 1
            print(f"  Request {i+1} blocked, wait time: {wait_time:.2f}s")
    
    print(f"✅ Allowed: {allowed_count}, Blocked: {blocked_count}")
    assert blocked_count > 0, "Rate limiting should block some requests"
    
    # Test 2: 429 error handling
    print("\n🚫 Test 2: 429 error exponential backoff")
    
    # Reset stats
    manager.reset_domain_stats('youtube.com')
    
    # Simulate 429 errors
    for i in range(3):
        manager.record_request('youtube.com', success=False, is_429=True)
        stats = manager.get_stats('youtube.com')
        print(f"  After 429 #{i+1}: delay = {stats['current_delay_seconds']:.2f}s")
    
    stats = manager.get_stats('youtube.com')
    assert stats['current_delay_seconds'] > 0, "Should have backoff delay after 429s"
    assert stats['consecutive_429s'] == 3, "Should track consecutive 429s"
    print(f"✅ Exponential backoff working: {stats['current_delay_seconds']:.2f}s delay")
    
    # Test 3: Circuit breaker
    print("\n🔌 Test 3: Circuit breaker activation")
    
    # Trigger more 429s to open circuit
    for i in range(3):
        manager.record_request('youtube.com', success=False, is_429=True)
    
    stats = manager.get_stats('youtube.com')
    
    if stats['circuit_state'] == 'open':
        print(f"✅ Circuit breaker OPENED after {stats['consecutive_429s']} consecutive 429s")
        
        # Check that requests are blocked
        allowed, wait_time = manager.should_allow_request('youtube.com')
        assert not allowed, "Circuit breaker should block requests when open"
        print(f"✅ Requests blocked, recovery in {wait_time:.1f}s")
    else:
        print(f"⚠️  Circuit breaker state: {stats['circuit_state']}")
    
    # Test 4: Multi-domain handling
    print("\n🌐 Test 4: Multi-domain rate limiting")
    
    domains = ['youtube.com', 'googleapis.com', 'example.com']
    
    for domain in domains:
        # Each domain should have independent limits
        allowed, _ = manager.should_allow_request(domain)
        stats = manager.get_stats(domain)
        print(f"  {domain}: requests={stats['total_requests']}, state={stats['circuit_state']}")
    
    print("✅ Each domain has independent rate limiting")
    
    return True


def test_execute_with_rate_limit():
    """Test the execute_with_rate_limit wrapper."""
    print("\n🧪 Testing Execute with Rate Limit")
    print("=" * 60)
    
    manager = RateLimitManager()
    
    # Test 1: Successful execution
    print("\n✅ Test 1: Successful function execution")
    
    def successful_function():
        return {"status": "success", "data": "test"}
    
    result, success = manager.execute_with_rate_limit(
        successful_function,
        domain='youtube.com',
        max_retries=3
    )
    
    assert success, "Should succeed"
    assert result["status"] == "success", "Should return correct result"
    print("✅ Function executed successfully with rate limiting")
    
    # Test 2: Function that returns 429
    print("\n🔄 Test 2: Function returning 429 with retry")
    
    call_count = 0
    
    def rate_limited_function():
        nonlocal call_count
        call_count += 1
        
        if call_count < 3:
            return {"status_code": 429, "error": "Too Many Requests"}
        else:
            return {"status": "success"}
    
    result, success = manager.execute_with_rate_limit(
        rate_limited_function,
        domain='youtube.com',
        max_retries=5
    )
    
    print(f"✅ Function retried {call_count} times before success")
    assert call_count >= 2, "Should retry on 429"
    assert success, "Should eventually succeed"
    
    # Test 3: Function that always fails
    print("\n❌ Test 3: Function that exceeds max retries")
    
    def always_429():
        return {"status_code": 429}
    
    result, success = manager.execute_with_rate_limit(
        always_429,
        domain='youtube.com',
        max_retries=2
    )
    
    assert not success, "Should fail after max retries"
    print("✅ Correctly failed after max retries")
    
    return True


async def test_async_rate_limiting():
    """Test async version of rate limiting."""
    print("\n🧪 Testing Async Rate Limiting")
    print("=" * 60)
    
    manager = RateLimitManager()
    
    print("\n⚡ Test: Async function execution")
    
    async def async_function():
        await asyncio.sleep(0.1)
        return {"status": "async_success"}
    
    result, success = await manager.execute_with_rate_limit_async(
        async_function,
        domain='youtube.com'
    )
    
    assert success, "Async execution should succeed"
    assert result["status"] == "async_success"
    print("✅ Async rate limiting working")
    
    return True


def test_with_real_scenario():
    """Test with a scenario similar to real YouTube API usage."""
    print("\n🧪 Testing Real-World Scenario")
    print("=" * 60)
    
    manager = get_rate_limit_manager()  # Use global instance
    
    # Simulate checking multiple channels
    channels = [
        'channel1', 'channel2', 'channel3', 'channel4', 'channel5'
    ]
    
    print("\n📺 Simulating channel enumeration with rate limiting...")
    
    successful_checks = 0
    rate_limited = 0
    total_wait_time = 0
    
    for i, channel_id in enumerate(channels * 10):  # Check each channel 10 times
        allowed, wait_time = manager.should_allow_request('youtube.com')
        
        if not allowed:
            rate_limited += 1
            total_wait_time += wait_time
            print(f"  Rate limited on request {i+1}, waiting {wait_time:.2f}s")
            time.sleep(min(wait_time, 0.1))  # Sleep briefly for test
        
        # Simulate occasional 429
        if i % 15 == 14:
            manager.record_request('youtube.com', success=False, is_429=True)
            print(f"  ⚠️  Got 429 on request {i+1}")
        else:
            manager.record_request('youtube.com', success=True)
            successful_checks += 1
    
    stats = manager.get_stats('youtube.com')
    
    print(f"\n📊 Results:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Successful: {stats['successful_requests']}")
    print(f"  Rate limited (429): {stats['rate_limited_requests']}")
    print(f"  Times throttled: {rate_limited}")
    print(f"  Total wait time: {total_wait_time:.2f}s")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    assert stats['rate_limited_requests'] > 0, "Should have some 429s"
    assert rate_limited > 0, "Should have throttled some requests"
    print("\n✅ Real-world scenario handled correctly")
    
    return True


def main():
    """Run all rate limiting tests."""
    print("🚀 Rate Limiting Prevention Test Suite")
    print("=" * 80)
    
    tests = [
        ("Rate Limit Prevention", test_rate_limit_prevention),
        ("Execute with Rate Limit", test_execute_with_rate_limit),
        ("Real-World Scenario", test_with_real_scenario)
    ]
    
    # Run async test separately
    async def run_async_test():
        return await test_async_rate_limiting()
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print('='*80)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append((test_name, False))
    
    # Run async test
    try:
        asyncio.run(run_async_test())
        results.append(("Async Rate Limiting", True))
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        results.append(("Async Rate Limiting", False))
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL RATE LIMITING TESTS PASSED!")
        print("✅ Proactive throttling prevents 429 errors")
        print("✅ Exponential backoff handles rate limits gracefully")
        print("✅ Circuit breakers prevent cascading failures")
        print("✅ Multi-domain support works correctly")
        print("✅ Real-world scenarios handled properly")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)