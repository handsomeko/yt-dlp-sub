#!/usr/bin/env python3
"""
Integration Test Runner

Quick launcher for all prevention system integration tests.
Provides options for different test levels and configurations.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List

def run_test_file(test_file: str, description: str) -> Dict[str, any]:
    """Run a specific test file and return results."""
    print(f"\n🔍 Running: {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=False, text=True, timeout=600)  # 10 minute timeout
        
        return {
            'test_file': test_file,
            'description': description,
            'success': result.returncode == 0,
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test {test_file} timed out after 10 minutes")
        return {
            'test_file': test_file, 
            'description': description,
            'success': False,
            'return_code': -1,
            'error': 'timeout'
        }
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return {
            'test_file': test_file,
            'description': description, 
            'success': False,
            'return_code': -1,
            'error': str(e)
        }

def run_quick_tests() -> List[Dict]:
    """Run quick unit tests (mocked, fast)."""
    print("🚀 Running QUICK tests (unit tests with mocking)")
    
    tests = [
        ("test_rate_limiting_prevention.py", "Rate Limiting Prevention (Unit)"),
        ("test_channel_enumeration.py", "Channel Enumeration (Unit)"),
        ("test_video_discovery.py", "Video Discovery Verification (Unit)")
    ]
    
    results = []
    for test_file, description in tests:
        if Path(test_file).exists():
            results.append(run_test_file(test_file, description))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                'test_file': test_file,
                'description': description,
                'success': False,
                'return_code': -1,
                'error': 'file_not_found'
            })
    
    return results

def run_integration_tests() -> List[Dict]:
    """Run comprehensive integration tests (real YouTube calls)."""
    print("🚀 Running INTEGRATION tests (real YouTube API calls)")
    print("⚠️  These tests make real API calls and may take 5-10 minutes")
    
    response = input("\nContinue with integration tests? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Integration tests skipped by user")
        return []
    
    tests = [
        ("test_youtube_integration_load.py", "Comprehensive YouTube Integration Load Test")
    ]
    
    results = []
    for test_file, description in tests:
        if Path(test_file).exists():
            results.append(run_test_file(test_file, description))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                'test_file': test_file,
                'description': description,
                'success': False,
                'return_code': -1,
                'error': 'file_not_found'
            })
    
    return results

def print_summary(quick_results: List[Dict], integration_results: List[Dict]) -> None:
    """Print test execution summary."""
    print("\n" + "=" * 80)
    print("📊 TEST EXECUTION SUMMARY")  
    print("=" * 80)
    
    # Quick tests summary
    if quick_results:
        quick_passed = sum(1 for r in quick_results if r['success'])
        print(f"🏃 Quick Tests: {quick_passed}/{len(quick_results)} passed")
        
        for result in quick_results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['description']}")
    
    # Integration tests summary  
    if integration_results:
        int_passed = sum(1 for r in integration_results if r['success'])
        print(f"🔥 Integration Tests: {int_passed}/{len(integration_results)} passed")
        
        for result in integration_results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['description']}")
    
    # Overall summary
    total_tests = len(quick_results) + len(integration_results)
    total_passed = (sum(1 for r in quick_results if r['success']) + 
                   sum(1 for r in integration_results if r['success']))
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("✅ PREVENTION SYSTEM TESTS PASSED")
        else:
            print("❌ PREVENTION SYSTEM TESTS FAILED")
    
    print("=" * 80)

def main():
    """Main test runner."""
    print("🧪 Prevention System Test Runner")
    print("=" * 50)
    print("Available test levels:")
    print("  1. Quick tests only (unit tests, ~1 minute)")
    print("  2. Integration tests only (real YouTube, ~5-10 minutes)")
    print("  3. All tests (quick + integration)")
    print("  4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Quick tests only
            quick_results = run_quick_tests()
            print_summary(quick_results, [])
            break
            
        elif choice == "2":
            # Integration tests only
            integration_results = run_integration_tests()
            print_summary([], integration_results)
            break
            
        elif choice == "3":
            # All tests
            quick_results = run_quick_tests()
            integration_results = run_integration_tests()
            print_summary(quick_results, integration_results)
            break
            
        elif choice == "4":
            print("Exiting test runner")
            return 0
            
        else:
            print("Invalid choice. Please select 1-4.")
    
    return 0

if __name__ == "__main__":
    exit(main())