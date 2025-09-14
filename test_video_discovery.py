#!/usr/bin/env python3
"""
Comprehensive Test Suite for Video Discovery Verification System

Tests completeness verification, discovery methods validation, and confidence scoring.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from core.video_discovery_verifier import (
    VideoDiscoveryVerifier,
    VerificationStatus,
    VideoDiscoveryReport,
    ChannelSnapshot
)
from core.channel_enumerator import VideoInfo, ChannelEnumerationResult, EnumerationStrategy
from config.settings import get_settings


class TestVideoDiscoveryVerifier:
    """Test suite for VideoDiscoveryVerifier class."""
    
    @pytest.fixture
    def temp_history_dir(self):
        """Create temporary history directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def verifier(self, temp_history_dir):
        """Create VideoDiscoveryVerifier instance with temp history."""
        return VideoDiscoveryVerifier(history_dir=temp_history_dir)
    
    @pytest.fixture
    def sample_video_info(self):
        """Sample VideoInfo for testing."""
        return [
            VideoInfo(
                video_id="vid1",
                title="Video 1",
                upload_date=datetime(2023, 1, 1),
                duration=300
            ),
            VideoInfo(
                video_id="vid2", 
                title="Video 2",
                upload_date=datetime(2023, 1, 2),
                duration=400
            ),
            VideoInfo(
                video_id="vid3",
                title="Video 3", 
                upload_date=datetime(2023, 1, 3),
                duration=500
            )
        ]
    
    @pytest.fixture
    def sample_enumeration_result(self, sample_video_info):
        """Sample ChannelEnumerationResult for testing."""
        return ChannelEnumerationResult(
            channel_id="UC123",
            channel_name="Test Channel",
            total_videos=3,
            videos=sample_video_info,
            enumeration_method="hybrid",
            is_complete=True
        )

    def test_initialization(self, temp_history_dir):
        """Test VideoDiscoveryVerifier initialization."""
        # Test with default parameters
        verifier1 = VideoDiscoveryVerifier()
        assert verifier1.history_dir.exists()
        assert verifier1.enumerator is not None
        
        # Test with custom parameters
        verifier2 = VideoDiscoveryVerifier(history_dir=temp_history_dir)
        assert verifier2.history_dir == temp_history_dir
        assert verifier2.history_dir.exists()
    
    def test_verification_status_enum(self):
        """Test VerificationStatus enum values."""
        assert VerificationStatus.COMPLETE.value == "complete"
        assert VerificationStatus.PARTIAL.value == "partial" 
        assert VerificationStatus.UNCERTAIN.value == "uncertain"
        assert VerificationStatus.FAILED.value == "failed"
        
        # Test all statuses are accessible
        all_statuses = list(VerificationStatus)
        assert len(all_statuses) == 4
    
    def test_video_discovery_report_dataclass(self):
        """Test VideoDiscoveryReport dataclass creation and attributes."""
        report = VideoDiscoveryReport(
            channel_id="UC123",
            channel_name="Test Channel",
            verification_status=VerificationStatus.COMPLETE,
            total_discovered=10,
            total_expected=10,
            missing_count=0,
            missing_videos=[],
            discovery_methods_used=["hybrid", "rss"],
            confidence_score=0.95,
            recommendations=[],
            timestamp=datetime.now()
        )
        
        assert report.channel_id == "UC123"
        assert report.channel_name == "Test Channel"
        assert report.verification_status == VerificationStatus.COMPLETE
        assert report.total_discovered == 10
        assert report.total_expected == 10
        assert report.missing_count == 0
        assert len(report.missing_videos) == 0
        assert "hybrid" in report.discovery_methods_used
        assert "rss" in report.discovery_methods_used
        assert report.confidence_score == 0.95
        assert isinstance(report.timestamp, datetime)
        assert report.metadata == {}  # Default factory
    
    def test_channel_snapshot_dataclass(self):
        """Test ChannelSnapshot dataclass."""
        snapshot = ChannelSnapshot(
            timestamp=datetime.now(),
            video_count=100,
            video_ids={"vid1", "vid2", "vid3"},
            enumeration_method="hybrid",
            verification_status=VerificationStatus.COMPLETE
        )
        
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.video_count == 100
        assert len(snapshot.video_ids) == 3
        assert "vid1" in snapshot.video_ids
        assert snapshot.enumeration_method == "hybrid"
        assert snapshot.verification_status == VerificationStatus.COMPLETE
        assert snapshot.metadata == {}  # Default factory
    
    def test_channel_id_extraction(self, verifier):
        """Test channel ID extraction from various URL formats."""
        test_cases = [
            ("https://youtube.com/@channelname", "@channelname"),
            ("https://youtube.com/c/ChannelName", "c/ChannelName"),
            ("https://youtube.com/channel/UC1234567890", "UC1234567890"),
            ("UCabcdef1234567890", "UCabcdef1234567890"),  # Direct channel ID
            ("@channelname", "@channelname"),  # Handle name only
        ]
        
        for url, expected in test_cases:
            result = verifier._extract_channel_id(url)
            assert result == expected, f"Failed for URL: {url}"
    
    @patch('core.video_discovery_verifier.get_settings')
    @patch.object(VideoDiscoveryVerifier, '_get_expected_video_count')
    def test_verify_channel_completeness_complete(self, mock_get_count, mock_get_settings, verifier, sample_enumeration_result):
        """Test verification when channel discovery is complete."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.verify_channel_completeness = True
        mock_settings.deep_check_threshold = 100
        mock_settings.missing_video_confidence = 0.85
        mock_settings.verification_sample_size = 50
        mock_get_settings.return_value = mock_settings
        
        # Mock expected count matches discovered count
        mock_get_count.return_value = 3
        
        # Mock enumerator to return sample result
        with patch.object(verifier.enumerator, 'enumerate_channel') as mock_enumerate:
            mock_enumerate.return_value = sample_enumeration_result
            
            report = verifier.verify_channel_completeness("https://youtube.com/@test")
            
            # Verify complete verification
            assert report.verification_status == VerificationStatus.COMPLETE
            assert report.total_discovered == 3
            assert report.total_expected == 3
            assert report.missing_count == 0
            assert len(report.missing_videos) == 0
            assert report.confidence_score > 0.8  # High confidence
            assert report.channel_id == "UC123"
            assert report.channel_name == "Test Channel"
    
    @patch('core.video_discovery_verifier.get_settings')
    @patch.object(VideoDiscoveryVerifier, '_get_expected_video_count')
    def test_verify_channel_completeness_partial(self, mock_get_count, mock_get_settings, verifier, sample_enumeration_result):
        """Test verification when some videos are missing."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.verify_channel_completeness = True
        mock_settings.deep_check_threshold = 100
        mock_settings.missing_video_confidence = 0.85
        mock_settings.verification_sample_size = 50
        mock_get_settings.return_value = mock_settings
        
        # Mock expected count higher than discovered 
        mock_get_count.return_value = 5  # Expected 5, but only found 3
        
        with patch.object(verifier.enumerator, 'enumerate_channel') as mock_enumerate:
            mock_enumerate.return_value = sample_enumeration_result
            
            report = verifier.verify_channel_completeness("https://youtube.com/@test")
            
            # Verify partial verification
            assert report.verification_status == VerificationStatus.PARTIAL
            assert report.total_discovered == 3
            assert report.total_expected == 5
            assert report.missing_count == 2
            assert len(report.recommendations) > 0  # Should have recommendations
    
    @patch('core.video_discovery_verifier.get_settings')
    def test_verify_channel_completeness_disabled(self, mock_get_settings, verifier):
        """Test verification when disabled in settings."""
        # Mock settings with verification disabled
        mock_settings = Mock()
        mock_settings.verify_channel_completeness = False
        mock_get_settings.return_value = mock_settings
        
        report = verifier.verify_channel_completeness("https://youtube.com/@test")
        
        # Should return uncertain status when disabled
        assert report.verification_status == VerificationStatus.UNCERTAIN
        assert "disabled" in report.recommendations[0].lower()
    
    def test_validate_discovery_methods(self, verifier):
        """Test validation of different discovery methods."""
        channel_url = "https://youtube.com/@test"
        
        # Mock enumerator responses for different methods
        mock_results = {
            EnumerationStrategy.RSS_FEED: ChannelEnumerationResult(
                channel_id="UC123", channel_name="Test", total_videos=2,
                videos=[VideoInfo("vid1", "Video 1"), VideoInfo("vid2", "Video 2")],
                enumeration_method="rss_feed", is_complete=False
            ),
            EnumerationStrategy.YT_DLP_DUMP: ChannelEnumerationResult(
                channel_id="UC123", channel_name="Test", total_videos=3,
                videos=[VideoInfo("vid1", "Video 1"), VideoInfo("vid2", "Video 2"), VideoInfo("vid3", "Video 3")],
                enumeration_method="yt_dlp_dump", is_complete=True
            )
        }
        
        with patch.object(verifier.enumerator, 'enumerate_channel') as mock_enumerate:
            def mock_enumerate_side_effect(url, strategy=None, **kwargs):
                return mock_results.get(strategy, mock_results[EnumerationStrategy.YT_DLP_DUMP])
            
            mock_enumerate.side_effect = mock_enumerate_side_effect
            
            results = verifier.validate_discovery_methods(channel_url, [
                EnumerationStrategy.RSS_FEED,
                EnumerationStrategy.YT_DLP_DUMP
            ])
            
            # Should return results for both methods
            assert len(results) == 2
            assert EnumerationStrategy.RSS_FEED in results
            assert EnumerationStrategy.YT_DLP_DUMP in results
            
            # RSS should find 2 videos, YT_DLP should find 3
            assert results[EnumerationStrategy.RSS_FEED].total_videos == 2
            assert results[EnumerationStrategy.YT_DLP_DUMP].total_videos == 3
    
    def test_history_management(self, verifier, sample_enumeration_result):
        """Test history loading and saving."""
        channel_id = "UC123"
        
        # Initially no history should exist
        history = verifier._load_history()
        assert channel_id not in history
        
        # Update history with a result
        verifier._update_channel_history(channel_id, sample_enumeration_result)
        
        # History should now exist
        history = verifier._load_history()
        assert channel_id in history
        assert len(history[channel_id]) == 1
        
        snapshot = history[channel_id][0]
        assert snapshot.video_count == 3
        assert len(snapshot.video_ids) == 3
        assert "vid1" in snapshot.video_ids
        assert "vid2" in snapshot.video_ids
        assert "vid3" in snapshot.video_ids
    
    def test_confidence_score_calculation(self, verifier):
        """Test confidence score calculation logic."""
        # Test high confidence scenario
        confidence1 = verifier._calculate_confidence_score(
            discovered=100,
            expected=100,
            methods_used=3,
            has_recent_data=True,
            is_complete_method=True
        )
        assert confidence1 > 0.9  # Very high confidence
        
        # Test medium confidence scenario  
        confidence2 = verifier._calculate_confidence_score(
            discovered=80,
            expected=100,
            methods_used=2,
            has_recent_data=True,
            is_complete_method=False
        )
        assert 0.5 < confidence2 < 0.9  # Medium confidence
        
        # Test low confidence scenario
        confidence3 = verifier._calculate_confidence_score(
            discovered=50,
            expected=100,
            methods_used=1,
            has_recent_data=False,
            is_complete_method=False
        )
        assert confidence3 < 0.6  # Low confidence
    
    def test_verification_status_determination(self, verifier):
        """Test verification status determination logic."""
        # Test complete status
        status1 = verifier._determine_verification_status(
            discovered=100, expected=100, confidence=0.95
        )
        assert status1 == VerificationStatus.COMPLETE
        
        # Test partial status
        status2 = verifier._determine_verification_status(
            discovered=80, expected=100, confidence=0.85
        )
        assert status2 == VerificationStatus.PARTIAL
        
        # Test uncertain status (low confidence)
        status3 = verifier._determine_verification_status(
            discovered=50, expected=100, confidence=0.4
        )
        assert status3 == VerificationStatus.UNCERTAIN
    
    def test_recommendations_generation(self, verifier):
        """Test recommendation generation."""
        # Test recommendations for partial discovery
        recommendations = verifier._generate_recommendations(
            status=VerificationStatus.PARTIAL,
            discovered=80,
            expected=100,
            confidence=0.7,
            methods_used=["rss_feed"],
            missing_count=20
        )
        
        assert len(recommendations) > 0
        # Should recommend trying additional methods
        rec_text = " ".join(recommendations).lower()
        assert "additional" in rec_text or "method" in rec_text or "strategy" in rec_text
    
    def test_expected_video_count_estimation(self, verifier):
        """Test expected video count estimation."""
        # Mock subprocess calls for video count estimation
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful yt-dlp output with video count
            mock_subprocess.return_value = Mock(
                returncode=0,
                stdout='{"channel_follower_count": 1000000, "video_count": 500}',
                stderr=""
            )
            
            count = verifier._get_expected_video_count("UC123", "https://youtube.com/@test")
            assert count is not None
            assert count > 0
    
    def test_error_handling(self, verifier):
        """Test error handling in verification."""
        # Test with invalid URL
        with patch.object(verifier.enumerator, 'enumerate_channel') as mock_enumerate:
            # Mock enumeration failure
            mock_enumerate.return_value = ChannelEnumerationResult(
                channel_id="",
                channel_name="",
                total_videos=0,
                videos=[],
                enumeration_method="failed",
                is_complete=False,
                error="Channel not found"
            )
            
            report = verifier.verify_channel_completeness("invalid-url")
            
            # Should handle error gracefully
            assert report.verification_status == VerificationStatus.FAILED
            assert "error" in report.recommendations[0].lower() or "failed" in report.recommendations[0].lower()
    
    def test_monitor_channel_changes_integration(self, verifier):
        """Test channel monitoring integration."""
        # Test that monitor_channel_changes method exists and is callable
        assert hasattr(verifier, 'monitor_channel_changes')
        assert callable(verifier.monitor_channel_changes)
        
        # Basic integration test - should not crash
        try:
            with patch.object(verifier, 'verify_channel_completeness') as mock_verify:
                mock_verify.return_value = VideoDiscoveryReport(
                    channel_id="UC123",
                    channel_name="Test",
                    verification_status=VerificationStatus.COMPLETE,
                    total_discovered=10,
                    total_expected=10,
                    missing_count=0,
                    missing_videos=[],
                    discovery_methods_used=["hybrid"],
                    confidence_score=0.95,
                    recommendations=[],
                    timestamp=datetime.now()
                )
                
                changes = verifier.monitor_channel_changes("https://youtube.com/@test")
                # Should return a result (could be empty if no changes)
                assert isinstance(changes, dict)
        except Exception as e:
            pytest.fail(f"monitor_channel_changes should not raise exception: {e}")


class TestSettingsIntegration:
    """Test integration with centralized settings system."""
    
    def test_verification_settings_exist(self):
        """Test that required verification settings exist."""
        settings = get_settings()
        
        # Verify all required verification fields exist
        assert hasattr(settings, 'verify_channel_completeness')
        assert hasattr(settings, 'deep_check_threshold')
        assert hasattr(settings, 'missing_video_confidence')
        assert hasattr(settings, 'verification_sample_size')
        
        # Verify field types
        assert isinstance(settings.verify_channel_completeness, bool)
        assert isinstance(settings.deep_check_threshold, int)
        assert isinstance(settings.missing_video_confidence, float)
        assert isinstance(settings.verification_sample_size, int)
    
    def test_verification_settings_values(self):
        """Test that verification settings have reasonable values."""
        settings = get_settings()
        
        # Check numeric values are in reasonable ranges
        assert 50 <= settings.deep_check_threshold <= 1000
        assert 0.5 <= settings.missing_video_confidence <= 0.99
        assert 10 <= settings.verification_sample_size <= 200
    
    @patch('core.video_discovery_verifier.get_settings')
    def test_settings_integration_in_verifier(self, mock_get_settings):
        """Test that verifier properly uses centralized settings."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.verify_channel_completeness = False
        mock_settings.deep_check_threshold = 150
        mock_settings.missing_video_confidence = 0.75
        mock_settings.verification_sample_size = 25
        mock_get_settings.return_value = mock_settings
        
        verifier = VideoDiscoveryVerifier()
        
        # Test that verification respects the disabled setting
        report = verifier.verify_channel_completeness("https://youtube.com/@test")
        
        # Should return uncertain when disabled
        assert report.verification_status == VerificationStatus.UNCERTAIN
        mock_get_settings.assert_called()


class TestDiscoveryReportAnalysis:
    """Test analysis and reporting functionality."""
    
    def test_missing_video_identification(self):
        """Test missing video identification logic."""
        verifier = VideoDiscoveryVerifier()
        
        # Mock method to test missing video identification
        discovered_ids = {"vid1", "vid2", "vid3"}
        expected_ids = {"vid1", "vid2", "vid3", "vid4", "vid5"}
        
        missing = verifier._identify_missing_videos(discovered_ids, expected_ids, 0.8)
        
        # Should identify vid4 and vid5 as missing
        assert len(missing) == 2
        assert "vid4" in missing
        assert "vid5" in missing
    
    def test_discovery_completeness_scoring(self):
        """Test completeness scoring algorithms."""
        verifier = VideoDiscoveryVerifier()
        
        # Test perfect match
        score1 = verifier._calculate_confidence_score(100, 100, 3, True, True)
        assert score1 > 0.9
        
        # Test significant mismatch
        score2 = verifier._calculate_confidence_score(50, 100, 1, False, False)
        assert score2 < 0.6
        
        # Scores should be between 0 and 1
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1


if __name__ == "__main__":
    # Run tests if executed directly
    print("Video Discovery Verification Test Suite")
    print("=" * 45)
    
    # Basic smoke test
    try:
        from core.video_discovery_verifier import (
            VideoDiscoveryVerifier, VerificationStatus, 
            VideoDiscoveryReport, ChannelSnapshot
        )
        print("✅ Imports successful")
        
        verifier = VideoDiscoveryVerifier()
        print("✅ Verifier creation successful")
        
        # Test settings integration
        from config.settings import get_settings
        settings = get_settings()
        print(f"✅ Settings loaded - Verification enabled: {settings.verify_channel_completeness}")
        print(f"   - Deep check threshold: {settings.deep_check_threshold}")
        print(f"   - Missing video confidence: {settings.missing_video_confidence}")
        print(f"   - Verification sample size: {settings.verification_sample_size}")
        
        # Test enum access
        statuses = list(VerificationStatus)
        print(f"✅ Found {len(statuses)} verification statuses")
        
        print("\n🎉 All basic tests passed!")
        print("\nRun with: python -m pytest test_video_discovery.py -v")
        
    except Exception as e:
        print(f"❌ Error in basic tests: {e}")
        import traceback
        traceback.print_exc()