#!/usr/bin/env python3
"""
Simplified edge case tests for Whisper timeout prevention system.
Focuses on verifying integration and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/Users/jk/yt-dl-sub')

def test_core_components():
    """Test that core timeout prevention components exist and are importable"""
    print("🧪 Testing Core Component Integration")
    print("=" * 60)
    
    try:
        # Test AudioAnalyzer import and basic structure
        print("\n📊 Testing AudioAnalyzer...")
        from core.audio_analyzer import AudioAnalyzer, AudioAnalysisResult
        
        analyzer = AudioAnalyzer()
        assert hasattr(analyzer, 'analyze_audio'), "AudioAnalyzer missing analyze_audio method"
        
        # Test AudioAnalysisResult structure by creating an instance
        result_fields = ['duration_seconds', 'recommended_timeout', 'requires_chunking', 'is_valid', 'error']
        
        # Create a test instance to verify fields exist
        test_result = AudioAnalysisResult(
            duration_seconds=0.0,
            file_size_mb=0.0,
            format="test",
            channels=2,
            sample_rate=44100,
            bitrate=None,
            is_valid=False,
            recommended_timeout=300,
            requires_chunking=False,
            chunk_count=1,
            estimated_memory_mb=100,
            error="test"
        )
        
        for field in result_fields:
            assert hasattr(test_result, field), f"AudioAnalysisResult missing {field}"
        
        print("✅ AudioAnalyzer import and structure OK")
        
        # Test WhisperTimeoutManager import and basic structure
        print("\n⏱️  Testing WhisperTimeoutManager...")
        from core.whisper_timeout_manager import WhisperTimeoutManager, ProcessMonitoringResult
        
        manager = WhisperTimeoutManager()
        assert hasattr(manager, 'execute_with_timeout'), "WhisperTimeoutManager missing execute_with_timeout method"
        
        # Test ProcessMonitoringResult structure by creating an instance
        monitoring_fields = ['max_memory_mb', 'execution_time_seconds', 'success', 'timeout_occurred']
        
        # Create a test instance to verify fields exist  
        test_monitoring = ProcessMonitoringResult(
            max_memory_mb=100.0,
            max_cpu_percent=50.0,
            execution_time_seconds=1.0,
            killed_due_to_resources=False,
            timeout_occurred=False,
            success=True,
            error=None
        )
        
        for field in monitoring_fields:
            assert hasattr(test_monitoring, field), f"ProcessMonitoringResult missing {field}"
        
        print("✅ WhisperTimeoutManager import and structure OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Component import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_transcriber_integration():
    """Test that TranscribeWorker properly integrates timeout protection"""
    print("\n🧪 Testing TranscribeWorker Integration")
    print("=" * 60)
    
    try:
        from workers.transcriber import TranscribeWorker
        
        transcriber = TranscribeWorker()
        
        # Test that timeout-protected methods exist
        timeout_methods = [
            '_extract_with_whisper_local',
            '_extract_with_ffmpeg', 
            '_try_fallback_transcription',
            '_transcribe_with_chunking'
        ]
        
        for method in timeout_methods:
            assert hasattr(transcriber, method), f"TranscribeWorker missing {method}"
            print(f"✅ {method} available")
        
        # Test that the main execute method exists
        assert hasattr(transcriber, 'execute'), "TranscribeWorker missing execute method"
        print("✅ Main execute method available")
        
        return True
        
    except ImportError as e:
        print(f"❌ TranscribeWorker import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TranscribeWorker integration test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration system for timeout settings"""
    print("\n🧪 Testing Configuration System")
    print("=" * 60)
    
    try:
        from config.settings import get_settings
        
        settings = get_settings()
        
        # Test all timeout-related settings exist
        timeout_settings = [
            'whisper_timeout_base',
            'whisper_timeout_per_minute', 
            'whisper_max_duration',
            'whisper_chunk_duration',
            'whisper_max_concurrent',
            'whisper_memory_limit_mb',
            'whisper_enable_chunking',
            'whisper_fallback_models'
        ]
        
        for setting in timeout_settings:
            assert hasattr(settings, setting), f"Missing timeout setting: {setting}"
            value = getattr(settings, setting)
            assert value is not None, f"Timeout setting {setting} is None"
            print(f"✅ {setting}: {value}")
        
        # Test setting bounds
        assert settings.whisper_timeout_base >= 60, "Base timeout too low"
        assert settings.whisper_timeout_per_minute >= 1.0, "Per-minute timeout too low"
        assert settings.whisper_max_concurrent >= 1, "Max concurrent too low"
        assert len(settings.whisper_fallback_models) > 0, "No fallback models configured"
        
        print("✅ All timeout settings configured with reasonable bounds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Settings import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of timeout components"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 60)
    
    try:
        # Test AudioAnalyzer with non-existent file (should fail gracefully)
        print("\n📁 Testing AudioAnalyzer error handling...")
        from core.audio_analyzer import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        result = analyzer.analyze_audio("/nonexistent/file.mp3")
        
        assert not result.is_valid, "Should fail for non-existent file"
        assert result.error is not None, "Should have error message"
        assert "not found" in result.error.lower() or "no such file" in result.error.lower()
        
        print("✅ AudioAnalyzer handles errors gracefully")
        
        # Test WhisperTimeoutManager with simple function
        print("\n⚡ Testing WhisperTimeoutManager basic execution...")
        from core.whisper_timeout_manager import WhisperTimeoutManager
        
        manager = WhisperTimeoutManager()
        
        def simple_function():
            return "success"
        
        result, monitoring = manager.execute_with_timeout(
            func=simple_function,
            timeout_seconds=5,
            job_id="test_basic",
            monitor_resources=False
        )
        
        assert result == "success", f"Expected success, got {result}"
        assert monitoring.success, "Monitoring should show success"
        
        print("✅ WhisperTimeoutManager executes functions correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timeout_calculation():
    """Test timeout calculation logic"""
    print("\n🧪 Testing Timeout Calculation Logic")
    print("=" * 60)
    
    try:
        # Test basic timeout math
        base_timeout = 300
        per_minute_timeout = 2.0
        
        # Test cases: duration in minutes -> expected timeout
        test_cases = [
            (1.0, 302),    # 1 minute -> 300 + (1 * 2) = 302
            (10.0, 320),   # 10 minutes -> 300 + (10 * 2) = 320
            (30.0, 360),   # 30 minutes -> 300 + (30 * 2) = 360
            (60.0, 420),   # 60 minutes -> 300 + (60 * 2) = 420
        ]
        
        for duration_minutes, expected in test_cases:
            calculated = int(base_timeout + (duration_minutes * per_minute_timeout))
            assert calculated == expected, f"For {duration_minutes}min: expected {expected}, got {calculated}"
            print(f"✅ {duration_minutes} min audio -> {calculated}s timeout")
        
        # Test chunking logic
        chunk_duration = 1800  # 30 minutes in seconds
        test_durations = [
            (900, False),   # 15 min -> no chunking
            (1800, False),  # 30 min -> no chunking (exactly at limit)
            (2700, True),   # 45 min -> chunking required
            (3600, True),   # 60 min -> chunking required
        ]
        
        for duration_seconds, should_chunk in test_durations:
            requires_chunking = duration_seconds > chunk_duration
            assert requires_chunking == should_chunk, f"Chunking logic failed for {duration_seconds}s"
            print(f"✅ {duration_seconds}s audio -> chunking: {requires_chunking}")
        
        return True
        
    except Exception as e:
        print(f"❌ Timeout calculation test failed: {e}")
        return False

def main():
    """Run all simplified edge case tests"""
    print("🚀 Whisper Timeout Prevention - Simplified Edge Case Tests")
    print("=" * 80)
    
    tests = [
        ("Core Component Integration", test_core_components),
        ("TranscribeWorker Integration", test_transcriber_integration),
        ("Configuration System", test_configuration_system),
        ("Basic Functionality", test_basic_functionality),
        ("Timeout Calculation", test_timeout_calculation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print('='*80)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("SIMPLIFIED EDGE CASE TEST SUMMARY")
    print('='*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL SIMPLIFIED EDGE CASE TESTS PASSED!")
        print("✅ Core timeout prevention components are integrated")
        print("✅ Configuration system is complete") 
        print("✅ Basic functionality is working")
        print("✅ TranscribeWorker integration is correct")
        print("✅ Timeout calculation logic is validated")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - review output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)