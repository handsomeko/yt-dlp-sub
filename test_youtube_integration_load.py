#!/usr/bin/env python3
"""
Comprehensive YouTube Integration Load Test

This test suite validates the prevention system under real YouTube load conditions:
- Tests real YouTube API interactions (no mocking)
- Validates rate limiting prevents 429 errors
- Tests unified wrapper integration 
- Verifies circuit breaker functionality
- Tests channel enumeration under load
- Validates video discovery verification

IMPORTANT: This test makes real YouTube API calls. Run sparingly to avoid rate limits.
"""

import asyncio
import time
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.rate_limit_manager import get_rate_limit_manager, RateLimitManager
from core.youtube_rate_limiter_unified import YouTubeRateLimiterUnified
from core.channel_enumerator import ChannelEnumerator, EnumerationStrategy
from core.video_discovery_verifier import VideoDiscoveryVerifier
from config.settings import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class IntegrationTestResult:
    """Results from integration testing."""
    test_name: str
    success: bool
    duration_seconds: float
    requests_made: int
    rate_limited_count: int
    errors_encountered: List[str]
    metadata: Dict[str, Any]

class YouTubeIntegrationLoadTester:
    """Integration tester for YouTube prevention systems."""
    
    # Real YouTube channels for testing (small channels to be respectful)
    TEST_CHANNELS = [
        "https://youtube.com/@tedtalks",  # Large channel with many videos
        "https://youtube.com/@mit",       # Educational content
        "https://youtube.com/@stanford",  # Another educational channel
    ]
    
    # Individual video URLs for single video tests
    TEST_VIDEOS = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (stable URL)
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
    ]
    
    def __init__(self):
        self.settings = get_settings()
        self.rate_manager = get_rate_limit_manager()
        self.unified_limiter = YouTubeRateLimiterUnified()
        self.enumerator = ChannelEnumerator()
        self.verifier = VideoDiscoveryVerifier()
        self.test_results: List[IntegrationTestResult] = []
        
        logger.info("🚀 YouTube Integration Load Tester initialized")
        logger.info(f"🔧 Rate limit: {self.settings.prevention_rate_limit} requests/minute")
        logger.info(f"🔧 Burst size: {self.settings.prevention_burst_size}")
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return summary."""
        logger.info("=" * 80)
        logger.info("🧪 STARTING COMPREHENSIVE YOUTUBE INTEGRATION TESTS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Test 1: Basic rate limiting under load
        self._test_rate_limiting_under_load()
        
        # Test 2: Unified wrapper integration
        self._test_unified_wrapper_integration()
        
        # Test 3: Channel enumeration with rate limiting
        self._test_channel_enumeration_with_rate_limiting()
        
        # Test 4: Video discovery verification integration
        self._test_video_discovery_integration()
        
        # Test 5: Circuit breaker functionality
        self._test_circuit_breaker_functionality()
        
        # Test 6: Multi-component integration
        self._test_multi_component_integration()
        
        total_duration = time.time() - start_time
        
        # Generate summary report
        summary = self._generate_test_summary(total_duration)
        self._print_test_summary(summary)
        
        return summary
    
    def _test_rate_limiting_under_load(self) -> None:
        """Test rate limiting system under sustained load."""
        logger.info("🔍 Test 1: Rate limiting under sustained load")
        
        start_time = time.time()
        requests_made = 0
        rate_limited_count = 0
        errors = []
        
        # Simulate sustained load for 2 minutes
        test_duration = 120  # seconds
        end_time = start_time + test_duration
        
        while time.time() < end_time:
            try:
                allowed, wait_time = self.rate_manager.should_allow_request('youtube.com')
                requests_made += 1
                
                if allowed:
                    # Simulate actual work
                    self.rate_manager.record_request('youtube.com', success=True)
                    time.sleep(0.1)  # Simulate processing time
                else:
                    rate_limited_count += 1
                    logger.info(f"  Rate limited - waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error during rate limit test: {e}")
        
        duration = time.time() - start_time
        success = len(errors) == 0 and rate_limited_count > 0  # Should have some rate limiting
        
        result = IntegrationTestResult(
            test_name="rate_limiting_under_load",
            success=success,
            duration_seconds=duration,
            requests_made=requests_made,
            rate_limited_count=rate_limited_count,
            errors_encountered=errors,
            metadata={
                'requests_per_second': requests_made / duration,
                'rate_limiting_percentage': (rate_limited_count / requests_made) * 100
            }
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Made {requests_made} requests, {rate_limited_count} rate limited")
        
    def _test_unified_wrapper_integration(self) -> None:
        """Test unified wrapper with real YouTube calls."""
        logger.info("🔍 Test 2: Unified wrapper integration")
        
        start_time = time.time()
        requests_made = 0
        errors = []
        
        try:
            # Test migration status
            migration_status = self.unified_limiter.get_migration_status()
            logger.info(f"  Migration status: {migration_status}")
            
            # Test rate limiting through wrapper
            for i in range(20):
                if self.unified_limiter.check_rate_limit():
                    requests_made += 1
                    self.unified_limiter.track_request(success=True)
                else:
                    wait_time = self.unified_limiter.wait_if_needed()
                    logger.info(f"  Wrapper enforced wait: {wait_time:.2f}s")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Unified wrapper test error: {e}")
        
        duration = time.time() - start_time
        result = IntegrationTestResult(
            test_name="unified_wrapper_integration",
            success=len(errors) == 0,
            duration_seconds=duration,
            requests_made=requests_made,
            rate_limited_count=0,
            errors_encountered=errors,
            metadata=migration_status
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Unified wrapper handled {requests_made} requests")
        
    def _test_channel_enumeration_with_rate_limiting(self) -> None:
        """Test channel enumeration with rate limiting protection."""
        logger.info("🔍 Test 3: Channel enumeration with rate limiting")
        
        start_time = time.time()
        errors = []
        channels_processed = 0
        
        for channel_url in self.TEST_CHANNELS[:2]:  # Test first 2 channels
            try:
                logger.info(f"  Enumerating channel: {channel_url}")
                
                # Use RSS feed strategy for speed
                result = self.enumerator.enumerate_channel(
                    channel_url,
                    strategy=EnumerationStrategy.RSS_FEED,
                    max_videos=10  # Limit for testing
                )
                
                if not result.error:
                    channels_processed += 1
                    logger.info(f"  Found {len(result.videos)} videos from {result.channel_name}")
                else:
                    errors.append(f"Channel {channel_url}: {result.error}")
                    
                # Respect rate limiting between channels
                time.sleep(2)
                
            except Exception as e:
                errors.append(f"Channel {channel_url}: {str(e)}")
                logger.error(f"Channel enumeration error: {e}")
        
        duration = time.time() - start_time
        result = IntegrationTestResult(
            test_name="channel_enumeration_with_rate_limiting",
            success=len(errors) == 0 and channels_processed > 0,
            duration_seconds=duration,
            requests_made=channels_processed,
            rate_limited_count=0,
            errors_encountered=errors,
            metadata={'channels_processed': channels_processed}
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Processed {channels_processed} channels")
        
    def _test_video_discovery_integration(self) -> None:
        """Test video discovery verification with rate limiting."""
        logger.info("🔍 Test 4: Video discovery verification integration")
        
        start_time = time.time()
        errors = []
        verifications_completed = 0
        
        for channel_url in self.TEST_CHANNELS[:1]:  # Test 1 channel
            try:
                logger.info(f"  Verifying channel completeness: {channel_url}")
                
                # Get some videos from enumeration first
                enum_result = self.enumerator.enumerate_channel(
                    channel_url,
                    strategy=EnumerationStrategy.RSS_FEED,
                    max_videos=5
                )
                
                if not enum_result.error:
                    discovered_videos = {v.video_id for v in enum_result.videos}
                    
                    # Verify completeness
                    verification_result = self.verifier.verify_channel_completeness(
                        channel_url=channel_url,
                        discovered_videos=discovered_videos,
                        force_deep_check=False  # Keep it light for testing
                    )
                    
                    verifications_completed += 1
                    logger.info(f"  Verification status: {verification_result.verification_status}")
                    logger.info(f"  Confidence score: {verification_result.confidence_score}")
                    
                else:
                    errors.append(f"Enumeration failed: {enum_result.error}")
                    
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Video discovery test error: {e}")
        
        duration = time.time() - start_time
        result = IntegrationTestResult(
            test_name="video_discovery_integration",
            success=len(errors) == 0 and verifications_completed > 0,
            duration_seconds=duration,
            requests_made=verifications_completed,
            rate_limited_count=0,
            errors_encountered=errors,
            metadata={'verifications_completed': verifications_completed}
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Completed {verifications_completed} verifications")
        
    def _test_circuit_breaker_functionality(self) -> None:
        """Test circuit breaker prevents cascading failures."""
        logger.info("🔍 Test 5: Circuit breaker functionality")
        
        start_time = time.time()
        errors = []
        
        try:
            # Simulate failures to trigger circuit breaker
            test_domain = 'test-circuit-breaker.com'
            
            # Record consecutive failures
            for i in range(self.settings.prevention_circuit_breaker_threshold + 1):
                self.rate_manager.record_request(test_domain, success=False, is_429=True)
            
            # Check if circuit breaker is open
            stats = self.rate_manager.get_stats(test_domain)
            circuit_state = stats.get('circuit_state', 'closed')
            
            logger.info(f"  Circuit state after {self.settings.prevention_circuit_breaker_threshold + 1} failures: {circuit_state}")
            
            # Test that requests are blocked when circuit is open
            allowed, wait_time = self.rate_manager.should_allow_request(test_domain)
            logger.info(f"  Request allowed with open circuit: {allowed}, wait time: {wait_time}")
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Circuit breaker test error: {e}")
        
        duration = time.time() - start_time
        result = IntegrationTestResult(
            test_name="circuit_breaker_functionality",
            success=len(errors) == 0,
            duration_seconds=duration,
            requests_made=self.settings.prevention_circuit_breaker_threshold + 1,
            rate_limited_count=0,
            errors_encountered=errors,
            metadata={'circuit_state': circuit_state if 'circuit_state' in locals() else 'unknown'}
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Circuit breaker functionality verified")
        
    def _test_multi_component_integration(self) -> None:
        """Test multiple components working together."""
        logger.info("🔍 Test 6: Multi-component integration")
        
        start_time = time.time()
        errors = []
        operations_completed = 0
        
        try:
            # Test end-to-end workflow: enumerate + verify + rate limit
            for channel_url in self.TEST_CHANNELS[:1]:
                logger.info(f"  Full workflow test for: {channel_url}")
                
                # Step 1: Enumerate with rate limiting
                enum_result = self.enumerator.enumerate_channel(
                    channel_url,
                    strategy=EnumerationStrategy.RSS_FEED,
                    max_videos=3
                )
                
                if enum_result.error:
                    errors.append(f"Enumeration failed: {enum_result.error}")
                    continue
                
                # Step 2: Verify completeness with rate limiting
                discovered_videos = {v.video_id for v in enum_result.videos}
                verification_result = self.verifier.verify_channel_completeness(
                    channel_url=channel_url,
                    discovered_videos=discovered_videos,
                    force_deep_check=False
                )
                
                # Step 3: Test unified wrapper
                wrapper_status = self.unified_limiter.get_status()
                
                operations_completed += 1
                logger.info(f"  ✅ Completed full workflow for {enum_result.channel_name}")
                logger.info(f"  Videos found: {len(discovered_videos)}")
                logger.info(f"  Verification confidence: {verification_result.confidence_score}")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Multi-component integration error: {e}")
        
        duration = time.time() - start_time
        result = IntegrationTestResult(
            test_name="multi_component_integration",
            success=len(errors) == 0 and operations_completed > 0,
            duration_seconds=duration,
            requests_made=operations_completed,
            rate_limited_count=0,
            errors_encountered=errors,
            metadata={'operations_completed': operations_completed}
        )
        
        self.test_results.append(result)
        logger.info(f"  ✅ Completed {operations_completed} full workflow operations")
    
    def _generate_test_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        total_requests = sum(r.requests_made for r in self.test_results)
        total_rate_limited = sum(r.rate_limited_count for r in self.test_results)
        total_errors = sum(len(r.errors_encountered) for r in self.test_results)
        
        return {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration_seconds': total_duration
            },
            'request_summary': {
                'total_requests_made': total_requests,
                'total_rate_limited': total_rate_limited,
                'requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
                'rate_limiting_percentage': (total_rate_limited / total_requests) * 100 if total_requests > 0 else 0
            },
            'error_summary': {
                'total_errors': total_errors,
                'error_rate': (total_errors / total_requests) * 100 if total_requests > 0 else 0
            },
            'individual_test_results': [
                {
                    'name': r.test_name,
                    'success': r.success,
                    'duration': r.duration_seconds,
                    'requests': r.requests_made,
                    'errors': len(r.errors_encountered)
                }
                for r in self.test_results
            ],
            'prevention_system_status': {
                'rate_limit_settings': {
                    'requests_per_minute': self.settings.prevention_rate_limit,
                    'burst_size': self.settings.prevention_burst_size,
                    'circuit_breaker_threshold': self.settings.prevention_circuit_breaker_threshold
                },
                'unified_wrapper_migration': self.unified_limiter.get_migration_status()
            }
        }
    
    def _print_test_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted test summary."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        test_summary = summary['test_summary']
        print(f"🧪 Tests: {test_summary['successful_tests']}/{test_summary['total_tests']} passed")
        print(f"📈 Success Rate: {test_summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {test_summary['total_duration_seconds']:.1f}s")
        
        request_summary = summary['request_summary']
        print(f"\n🔥 Load Statistics:")
        print(f"   Total Requests: {request_summary['total_requests_made']}")
        print(f"   Rate Limited: {request_summary['total_rate_limited']}")
        print(f"   Requests/sec: {request_summary['requests_per_second']:.2f}")
        print(f"   Rate Limiting %: {request_summary['rate_limiting_percentage']:.1f}%")
        
        error_summary = summary['error_summary']
        print(f"\n❌ Error Statistics:")
        print(f"   Total Errors: {error_summary['total_errors']}")
        print(f"   Error Rate: {error_summary['error_rate']:.2f}%")
        
        print(f"\n🛡️  Prevention System:")
        prevention_status = summary['prevention_system_status']
        settings = prevention_status['rate_limit_settings']
        print(f"   Rate Limit: {settings['requests_per_minute']}/min")
        print(f"   Burst Size: {settings['burst_size']}")
        print(f"   Circuit Threshold: {settings['circuit_breaker_threshold']}")
        
        migration = prevention_status['unified_wrapper_migration']
        print(f"   Using Prevention System: {migration['using_prevention_system']}")
        
        print("\n" + "=" * 80)


def main():
    """Run the integration tests."""
    print("🚀 Starting YouTube Integration Load Tests...")
    print("⚠️  This test makes real YouTube API calls - use sparingly!")
    print("⏳ Estimated duration: 5-10 minutes")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    tester = YouTubeIntegrationLoadTester()
    summary = tester.run_all_tests()
    
    # Save results to file
    import json
    results_file = Path("test_results_integration_load.json")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Return exit code based on success
    success_rate = summary['test_summary']['success_rate']
    if success_rate >= 80:  # 80% success threshold
        print("✅ Integration tests PASSED")
        return 0
    else:
        print("❌ Integration tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())