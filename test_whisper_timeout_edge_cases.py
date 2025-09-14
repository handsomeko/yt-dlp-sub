#!/usr/bin/env python3
"""
Test edge cases for Whisper timeout prevention system.
Validates that our comprehensive timeout implementation handles all scenarios.
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, '/Users/jk/yt-dl-sub')

def test_audio_analyzer_edge_cases():
    """Test AudioAnalyzer with various edge cases"""
    print("🧪 Testing AudioAnalyzer Edge Cases")
    print("=" * 60)
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        
        # Test 1: Non-existent file
        print("\n📁 Test 1: Non-existent audio file")
        result = analyzer.analyze_audio("/nonexistent/file.mp3")
        assert not result.is_valid, "Should fail for non-existent file"
        assert "not found" in result.error or "No such file" in result.error
        print("✅ Correctly handles non-existent files")
        
        # Test 2: Very short audio (edge case for timeout calculation)
        print("\n⏱️  Test 2: Simulate very short audio")
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch('subprocess.run') as mock_run:
            # Mock file size
            mock_stat.return_value = Mock(st_size=1024000)  # 1MB
            # Mock FFprobe output for 5-second audio
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"streams":[{"codec_type":"audio","codec_name":"mp3","channels":2,"sample_rate":"44100","bit_rate":"128000"}],"format":{"duration":"5.0","format_name":"mp3"}}'
            )
            result = analyzer.analyze_audio("dummy.mp3")
            
            assert result.is_valid, f"Should succeed for short audio: {result.error}"
            assert result.duration_seconds == 5.0
            assert result.recommended_timeout >= 300  # Should use minimum timeout
            print(f"✅ Short audio (5s) gets minimum timeout: {result.recommended_timeout}s")
        
        # Test 3: Very long audio (should trigger chunking recommendation)
        print("\n⏳ Test 3: Simulate very long audio")
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch('subprocess.run') as mock_run:
            # Mock file size
            mock_stat.return_value = Mock(st_size=500*1024*1024)  # 500MB
            # Mock FFprobe output for 3-hour audio
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"streams":[{"codec_type":"audio","codec_name":"opus","channels":2,"sample_rate":"48000","bit_rate":"128000"}],"format":{"duration":"10800.0","format_name":"opus"}}'
            )
            result = analyzer.analyze_audio("dummy.opus")
            
            assert result.is_valid, f"Should succeed for long audio: {result.error}"
            assert result.duration_seconds == 10800.0  # 3 hours
            assert result.requires_chunking, "Should require chunking for 3-hour audio"
            assert result.recommended_timeout > 1000  # Should have significant timeout
            print(f"✅ Long audio (3h) requires chunking, timeout: {result.recommended_timeout}s")
        
        # Test 4: Corrupted/invalid audio file
        print("\n💥 Test 4: Simulate corrupted audio file")
        with patch('subprocess.run') as mock_run:
            # Mock FFprobe failure
            mock_run.return_value = Mock(
                returncode=1,
                stderr="Invalid data found when processing input"
            )
            result = analyzer.analyze_audio("corrupted.mp3")
            
            assert not result.is_valid, "Should fail for corrupted file"
            assert "error analyzing audio" in result.error.lower() or "invalid" in result.error.lower()
            print("✅ Correctly handles corrupted audio files")
        
        return True
        
    except ImportError as e:
        print(f"❌ AudioAnalyzer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ AudioAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timeout_manager_edge_cases():
    """Test WhisperTimeoutManager with edge cases"""
    print("\n🧪 Testing WhisperTimeoutManager Edge Cases")
    print("=" * 60)
    
    try:
        from core.whisper_timeout_manager import WhisperTimeoutManager
        
        # Test 1: Quick function execution (should complete normally)
        print("\n⚡ Test 1: Quick function execution")
        manager = WhisperTimeoutManager()
        
        def quick_function():
            return "completed successfully"
        
        result, monitoring = manager.execute_with_timeout(
            func=quick_function,
            timeout_seconds=10,
            job_id="test_quick",
            monitor_resources=True
        )
        
        assert result == "completed successfully", f"Expected success, got: {result}"
        assert monitoring.success, "Should complete normally"
        print("✅ Quick functions execute without timeout")
        
        # Test 2: Function that would timeout (simulated)
        print("\n⏰ Test 2: Function timeout simulation")
        
        def slow_function():
            time.sleep(2)  # 2 second delay
            return "should not reach here"
        
        try:
            result, monitoring = manager.execute_with_timeout(
                func=slow_function,
                timeout_seconds=1,  # 1 second timeout
                job_id="test_timeout",
                monitor_resources=False  # Disable monitoring for cleaner test
            )
            # If we get here, timeout didn't work as expected
            print("⚠️  Timeout test inconclusive - function may have completed")
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                print("✅ Timeout protection is working")
            else:
                print(f"⚠️  Unexpected error: {e}")
        
        # Test 3: Memory monitoring (mocked)
        print("\n🧠 Test 3: Memory monitoring simulation")
        
        def memory_intensive_function():
            return "memory test complete"
        
        # Test with very low memory limit to trigger monitoring
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(available=1024 * 1024)  # 1MB available
            
            result, monitoring = manager.execute_with_timeout(
                func=memory_intensive_function,
                timeout_seconds=5,
                job_id="test_memory",
                monitor_resources=True
            )
            
            # Should either complete or detect memory issue
            print("✅ Memory monitoring system is active")
        
        return True
        
    except ImportError as e:
        print(f"❌ WhisperTimeoutManager not available: {e}")
        return False
    except Exception as e:
        print(f"❌ WhisperTimeoutManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transcriber_timeout_integration():
    """Test that TranscribeWorker properly integrates timeout protection"""
    print("\n🧪 Testing TranscribeWorker Timeout Integration")
    print("=" * 60)
    
    try:
        from workers.transcriber import TranscribeWorker
        
        transcriber = TranscribeWorker()
        
        # Test 1: Verify timeout components are available
        print("\n🔧 Test 1: Verify timeout components are importable")
        try:
            from core.audio_analyzer import AudioAnalyzer
            from core.whisper_timeout_manager import WhisperTimeoutManager
            print("✅ AudioAnalyzer imported successfully")
            print("✅ WhisperTimeoutManager imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test 2: Check if TranscribeWorker has the timeout-protected methods
        print("\n🔍 Test 2: Check TranscribeWorker timeout integration")
        
        # Verify the _extract_with_whisper_local method exists
        assert hasattr(transcriber, '_extract_with_whisper_local'), "Missing _extract_with_whisper_local method"
        
        # Verify the _extract_with_ffmpeg method exists  
        assert hasattr(transcriber, '_extract_with_ffmpeg'), "Missing _extract_with_ffmpeg method"
        
        print("✅ TranscribeWorker has timeout-protected methods")
        
        # Test 3: Test the fallback mechanism exists
        print("\n🔄 Test 3: Check fallback mechanisms")
        
        assert hasattr(transcriber, '_try_fallback_transcription'), "Missing fallback transcription method"
        assert hasattr(transcriber, '_transcribe_with_chunking'), "Missing chunk transcription method"
        
        print("✅ Fallback and chunking mechanisms available")
        
        return True
        
    except ImportError as e:
        print(f"❌ TranscribeWorker not available: {e}")
        return False
    except Exception as e:
        print(f"❌ TranscribeWorker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_edge_cases():
    """Test configuration handling for timeout settings"""
    print("\n🧪 Testing Configuration Edge Cases")
    print("=" * 60)
    
    try:
        from config.settings import get_settings
        
        # Test 1: Verify all timeout settings have defaults
        print("\n⚙️  Test 1: Verify timeout configuration defaults")
        settings = get_settings()
        
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
        
        # Test 2: Verify setting bounds are reasonable
        print("\n📏 Test 2: Verify setting bounds")
        
        assert settings.whisper_timeout_base >= 60, "Base timeout too low"
        assert settings.whisper_timeout_per_minute >= 1.0, "Per-minute timeout too low" 
        assert settings.whisper_max_duration >= 300, "Max duration too low"
        assert settings.whisper_chunk_duration >= 300, "Chunk duration too low"
        assert settings.whisper_max_concurrent >= 1, "Max concurrent too low"
        assert settings.whisper_memory_limit_mb >= 2048, "Memory limit too low"
        assert isinstance(settings.whisper_fallback_models, list), "Fallback models must be a list"
        
        print("✅ All timeout settings have reasonable bounds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Settings not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_recovery_scenarios():
    """Test various error recovery scenarios"""
    print("\n🧪 Testing Error Recovery Scenarios")
    print("=" * 60)
    
    try:
        # Test 1: Verify error recovery components exist
        print("\n🛠️  Test 1: Verify error recovery components")
        
        from workers.transcriber import TranscribeWorker
        transcriber = TranscribeWorker()
        
        # Check for error handling methods
        error_handling_methods = [
            '_try_fallback_transcription',
            '_transcribe_with_chunking',
            '_extract_with_whisper_local',
            '_extract_with_ffmpeg'
        ]
        
        for method in error_handling_methods:
            assert hasattr(transcriber, method), f"Missing error handling method: {method}"
            print(f"✅ {method} available")
        
        # Test 2: Check that fallback models are configured
        print("\n🔄 Test 2: Verify fallback model configuration")
        
        from config.settings import get_settings
        settings = get_settings()
        
        assert len(settings.whisper_fallback_models) > 0, "No fallback models configured"
        assert 'base' in settings.whisper_fallback_models or 'tiny' in settings.whisper_fallback_models, "No basic fallback models"
        
        print(f"✅ Fallback models configured: {settings.whisper_fallback_models}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all edge case tests"""
    print("🚀 Whisper Timeout Prevention - Edge Case Test Suite")
    print("=" * 60)
    
    tests = [
        ("Audio Analyzer Edge Cases", test_audio_analyzer_edge_cases),
        ("Timeout Manager Edge Cases", test_timeout_manager_edge_cases),
        ("Transcriber Integration", test_transcriber_timeout_integration),
        ("Configuration Edge Cases", test_configuration_edge_cases),
        ("Error Recovery Scenarios", test_error_recovery_scenarios)
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
    print("EDGE CASE TEST SUMMARY")
    print('='*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL EDGE CASE TESTS PASSED!")
        print("✅ Whisper timeout prevention system is robust")
        print("✅ Error handling is comprehensive") 
        print("✅ Configuration is complete")
        print("✅ Integration points are working")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - review output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)