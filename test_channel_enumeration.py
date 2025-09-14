#!/usr/bin/env python3
"""
Comprehensive Test Suite for Channel Enumeration System

Tests all enumeration strategies, settings integration, and edge cases.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Optional

from core.channel_enumerator import (
    ChannelEnumerator, 
    EnumerationStrategy, 
    VideoInfo, 
    ChannelEnumerationResult
)
from config.settings import get_settings


class TestChannelEnumerator:
    """Test suite for ChannelEnumerator class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def enumerator(self, temp_cache_dir):
        """Create ChannelEnumerator instance with temp cache."""
        return ChannelEnumerator(cache_dir=temp_cache_dir)
    
    @pytest.fixture
    def sample_video_info(self):
        """Sample VideoInfo for testing."""
        return VideoInfo(
            video_id="dQw4w9WgXcQ",
            title="Rick Astley - Never Gonna Give You Up",
            upload_date=datetime(2009, 10, 25),
            duration=212,
            view_count=1200000000,
            description="Official video",
            thumbnail_url="https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
        )

    def test_initialization(self, temp_cache_dir):
        """Test ChannelEnumerator initialization."""
        # Test with default parameters
        enumerator1 = ChannelEnumerator()
        assert enumerator1.youtube_api_key is None
        assert enumerator1.cache_dir.exists()
        
        # Test with custom parameters
        enumerator2 = ChannelEnumerator(
            youtube_api_key="test_key", 
            cache_dir=temp_cache_dir
        )
        assert enumerator2.youtube_api_key == "test_key"
        assert enumerator2.cache_dir == temp_cache_dir
        assert enumerator2.cache_dir.exists()
    
    def test_channel_id_extraction(self, enumerator):
        """Test channel ID extraction from various URL formats."""
        test_cases = [
            ("https://youtube.com/@channelname", "@channelname"),
            ("https://youtube.com/c/ChannelName", "c/ChannelName"),
            ("https://youtube.com/channel/UC1234567890", "UC1234567890"),
            ("https://youtube.com/user/username", "user/username"),
            ("UCabcdef1234567890", "UCabcdef1234567890"),  # Direct channel ID
            ("@channelname", "@channelname"),  # Handle name only
        ]
        
        for url, expected in test_cases:
            result = enumerator._extract_channel_id(url)
            assert result == expected, f"Failed for URL: {url}"
    
    def test_enumeration_strategy_enum(self):
        """Test EnumerationStrategy enum values."""
        assert EnumerationStrategy.RSS_FEED.value == "rss_feed"
        assert EnumerationStrategy.YT_DLP_DUMP.value == "yt_dlp_dump"
        assert EnumerationStrategy.YOUTUBE_API.value == "youtube_api"
        assert EnumerationStrategy.PLAYLIST.value == "playlist"
        assert EnumerationStrategy.WEB_SCRAPE.value == "web_scrape"
        assert EnumerationStrategy.HYBRID.value == "hybrid"
        
        # Test all strategies are accessible
        all_strategies = list(EnumerationStrategy)
        assert len(all_strategies) == 6
    
    def test_video_info_dataclass(self):
        """Test VideoInfo dataclass creation and attributes."""
        video = VideoInfo(
            video_id="test123",
            title="Test Video",
            upload_date=datetime(2023, 1, 1),
            duration=300,
            view_count=1000
        )
        
        assert video.video_id == "test123"
        assert video.title == "Test Video"
        assert video.upload_date == datetime(2023, 1, 1)
        assert video.duration == 300
        assert video.view_count == 1000
        assert video.description is None  # Default value
        assert video.thumbnail_url is None  # Default value
    
    def test_channel_enumeration_result_dataclass(self, sample_video_info):
        """Test ChannelEnumerationResult dataclass."""
        result = ChannelEnumerationResult(
            channel_id="UC123",
            channel_name="Test Channel",
            total_videos=1,
            videos=[sample_video_info],
            enumeration_method="test",
            is_complete=True
        )
        
        assert result.channel_id == "UC123"
        assert result.channel_name == "Test Channel"
        assert result.total_videos == 1
        assert len(result.videos) == 1
        assert result.videos[0] == sample_video_info
        assert result.enumeration_method == "test"
        assert result.is_complete is True
        assert result.estimated_missing == 0  # Default value
        assert result.error is None  # Default value
        assert result.metadata == {}  # Default factory
    
    @patch('core.channel_enumerator.get_settings')
    def test_settings_integration(self, mock_get_settings, enumerator):
        """Test integration with centralized settings system."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.default_enumeration_strategy = "HYBRID"
        mock_settings.force_complete_enumeration = True
        mock_settings.max_videos_per_channel = 5000
        mock_get_settings.return_value = mock_settings
        
        # Test that settings are used when parameters are None
        with patch.object(enumerator, '_enumerate_hybrid') as mock_hybrid:
            mock_hybrid.return_value = ChannelEnumerationResult(
                channel_id="UC123",
                channel_name="Test",
                total_videos=0,
                videos=[],
                enumeration_method="hybrid",
                is_complete=True
            )
            
            result = enumerator.enumerate_channel(
                channel_url="https://youtube.com/@test",
                strategy=None,  # Should use settings default
                force_complete=None,  # Should use settings default
                max_videos=None  # Should use settings default
            )
            
            # Verify settings were called
            mock_get_settings.assert_called_once()
            
            # Verify the hybrid method was called (default strategy)
            mock_hybrid.assert_called_once()
    
    def test_get_all_videos_backward_compatibility(self, enumerator):
        """Test backward compatibility method get_all_videos()."""
        with patch.object(enumerator, 'enumerate_channel') as mock_enumerate:
            # Mock return value
            mock_result = ChannelEnumerationResult(
                channel_id="UC123",
                channel_name="Test",
                total_videos=2,
                videos=[
                    VideoInfo("vid1", "Video 1"),
                    VideoInfo("vid2", "Video 2")
                ],
                enumeration_method="hybrid",
                is_complete=True
            )
            mock_enumerate.return_value = mock_result
            
            # Test get_all_videos
            videos = enumerator.get_all_videos("https://youtube.com/@test")
            
            # Verify it calls enumerate_channel with correct parameters
            mock_enumerate.assert_called_once_with(
                "https://youtube.com/@test",
                strategy=EnumerationStrategy.HYBRID,
                max_videos=None
            )
            
            # Verify it returns the videos list
            assert len(videos) == 2
            assert videos[0].video_id == "vid1"
            assert videos[1].video_id == "vid2"
    
    def test_get_all_videos_with_limit(self, enumerator):
        """Test get_all_videos with limit parameter."""
        with patch.object(enumerator, 'enumerate_channel') as mock_enumerate:
            mock_result = ChannelEnumerationResult(
                channel_id="UC123",
                channel_name="Test", 
                total_videos=1,
                videos=[VideoInfo("vid1", "Video 1")],
                enumeration_method="hybrid",
                is_complete=True
            )
            mock_enumerate.return_value = mock_result
            
            # Test with limit
            videos = enumerator.get_all_videos(
                "https://youtube.com/@test", 
                limit=10
            )
            
            # Verify limit is passed as max_videos
            mock_enumerate.assert_called_once_with(
                "https://youtube.com/@test",
                strategy=EnumerationStrategy.HYBRID,
                max_videos=10
            )
    
    def test_date_parsing(self, enumerator):
        """Test _parse_date method."""
        # Test valid date formats
        date1 = enumerator._parse_date("2023-01-01")
        assert date1 == datetime(2023, 1, 1)
        
        date2 = enumerator._parse_date("2023-01-01T12:00:00Z")
        assert date2.year == 2023
        assert date2.month == 1
        assert date2.day == 1
        
        # Test invalid/None dates
        assert enumerator._parse_date(None) is None
        assert enumerator._parse_date("") is None
        assert enumerator._parse_date("invalid-date") is None
    
    @patch('subprocess.run')
    def test_yt_dlp_enumeration_error_handling(self, mock_subprocess, enumerator):
        """Test error handling in yt-dlp enumeration."""
        # Mock subprocess failure
        mock_subprocess.return_value = Mock(
            returncode=1,
            stderr="ERROR: Video unavailable",
            stdout=""
        )
        
        result = enumerator._enumerate_yt_dlp("UC123", "https://youtube.com/@test", False, None)
        
        # Should return error result
        assert result.error is not None
        assert "failed" in result.error.lower()
        assert result.total_videos == 0
        assert len(result.videos) == 0
    
    @patch('feedparser.parse')
    def test_rss_enumeration_error_handling(self, mock_feedparser, enumerator):
        """Test error handling in RSS enumeration."""
        # Mock feedparser failure
        mock_feedparser.return_value = Mock(
            entries=[],
            feed=Mock(title="Test Channel"),
            bozo=True,  # Indicates parsing error
            bozo_exception=Exception("Parse error")
        )
        
        result = enumerator._enumerate_rss("UC123", "https://youtube.com/@test", False, None)
        
        # Should handle gracefully and return empty result
        assert result.total_videos == 0
        assert len(result.videos) == 0
        assert result.enumeration_method == "rss_feed"
    
    def test_rate_limit_integration(self, enumerator):
        """Test that rate limiting is integrated properly."""
        # The enumerator should have rate limit manager
        # This is integration test - we just verify the manager is accessible
        from core.rate_limit_manager import get_rate_limit_manager
        
        rate_manager = get_rate_limit_manager()
        assert rate_manager is not None
        
        # Verify it has YouTube configuration
        assert 'youtube.com' in rate_manager.domain_configs
        youtube_config = rate_manager.domain_configs['youtube.com']
        assert youtube_config.requests_per_minute > 0
    
    def test_channel_name_extraction(self, enumerator):
        """Test channel name extraction/fallback."""
        # Test with basic URL - should extract channel identifier
        name1 = enumerator._get_channel_name("UC123", "https://youtube.com/@testchannel")
        # Should contain some form of the channel identifier
        assert name1 is not None
        assert len(name1) > 0
        
        # Test with channel ID
        name2 = enumerator._get_channel_name("UC123", "https://youtube.com/channel/UC123")
        assert name2 is not None
    
    def test_verification_integration(self, enumerator):
        """Test completeness verification is available."""
        # This tests that the verify_completeness method exists and is callable
        # We don't need to test the full implementation here, just integration
        result = ChannelEnumerationResult(
            channel_id="UC123",
            channel_name="Test",
            total_videos=10,
            videos=[],
            enumeration_method="test",
            is_complete=False
        )
        
        # Method should exist and be callable
        assert hasattr(enumerator, 'verify_completeness')
        assert callable(enumerator.verify_completeness)
    
    def test_incremental_discovery_integration(self, enumerator):
        """Test incremental discovery is available."""
        # Test that the incremental_discovery method exists
        assert hasattr(enumerator, 'incremental_discovery')
        assert callable(enumerator.incremental_discovery)


class TestEnumerationStrategies:
    """Test individual enumeration strategies."""
    
    @pytest.fixture
    def enumerator(self):
        temp_dir = tempfile.mkdtemp()
        enum = ChannelEnumerator(cache_dir=Path(temp_dir))
        yield enum
        shutil.rmtree(temp_dir)
    
    def test_strategy_selection(self, enumerator):
        """Test that correct strategy methods are called."""
        test_url = "https://youtube.com/@test"
        
        with patch.object(enumerator, '_enumerate_hybrid') as mock_hybrid, \
             patch.object(enumerator, '_enumerate_rss') as mock_rss, \
             patch.object(enumerator, '_enumerate_yt_dlp') as mock_yt_dlp:
            
            # Mock return values
            base_result = ChannelEnumerationResult(
                channel_id="UC123", channel_name="Test", total_videos=0,
                videos=[], enumeration_method="", is_complete=True
            )
            mock_hybrid.return_value = base_result
            mock_rss.return_value = base_result  
            mock_yt_dlp.return_value = base_result
            
            # Test HYBRID strategy
            enumerator.enumerate_channel(test_url, EnumerationStrategy.HYBRID)
            mock_hybrid.assert_called_once()
            
            # Test RSS strategy
            enumerator.enumerate_channel(test_url, EnumerationStrategy.RSS_FEED)
            mock_rss.assert_called_once()
            
            # Test YT_DLP strategy
            enumerator.enumerate_channel(test_url, EnumerationStrategy.YT_DLP_DUMP)
            mock_yt_dlp.assert_called_once()


class TestSettingsIntegration:
    """Test integration with centralized settings system."""
    
    def test_settings_fields_exist(self):
        """Test that required enumeration settings exist."""
        settings = get_settings()
        
        # Verify all required fields exist
        assert hasattr(settings, 'default_enumeration_strategy')
        assert hasattr(settings, 'force_complete_enumeration')
        assert hasattr(settings, 'max_videos_per_channel')
        assert hasattr(settings, 'enumeration_timeout')
        assert hasattr(settings, 'cache_duration_hours')
        assert hasattr(settings, 'incremental_check_interval')
        
        # Verify field types and values
        assert isinstance(settings.default_enumeration_strategy, str)
        assert isinstance(settings.force_complete_enumeration, bool)
        assert settings.max_videos_per_channel is None or isinstance(settings.max_videos_per_channel, int)
        assert isinstance(settings.enumeration_timeout, int)
        assert isinstance(settings.cache_duration_hours, int)
        assert isinstance(settings.incremental_check_interval, int)
    
    def test_settings_values_reasonable(self):
        """Test that settings have reasonable default values."""
        settings = get_settings()
        
        # Check enumeration strategy is valid
        valid_strategies = ["RSS_FEED", "YT_DLP_DUMP", "YOUTUBE_API", "PLAYLIST", "HYBRID", "WEB_SCRAPE"]
        assert settings.default_enumeration_strategy in valid_strategies
        
        # Check numeric values are in reasonable ranges
        assert 60 <= settings.enumeration_timeout <= 1800  # 1-30 minutes
        assert 1 <= settings.cache_duration_hours <= 168   # 1 hour to 1 week
        assert 300 <= settings.incremental_check_interval <= 86400  # 5 minutes to 1 day
        
        if settings.max_videos_per_channel is not None:
            assert 100 <= settings.max_videos_per_channel <= 50000


if __name__ == "__main__":
    # Run tests if executed directly
    print("Channel Enumeration Test Suite")
    print("=" * 40)
    
    # Basic smoke test
    try:
        from core.channel_enumerator import ChannelEnumerator, EnumerationStrategy
        print("✅ Imports successful")
        
        enumerator = ChannelEnumerator()
        print("✅ Enumerator creation successful")
        
        # Test settings integration
        from config.settings import get_settings
        settings = get_settings()
        print(f"✅ Settings loaded - Default strategy: {settings.default_enumeration_strategy}")
        
        # Test enum access
        strategies = list(EnumerationStrategy)
        print(f"✅ Found {len(strategies)} enumeration strategies")
        
        print("\n🎉 All basic tests passed!")
        print("\nRun with: python -m pytest test_channel_enumeration.py -v")
        
    except Exception as e:
        print(f"❌ Error in basic tests: {e}")
        import traceback
        traceback.print_exc()