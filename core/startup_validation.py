"""
Startup validation to ensure V2 storage structure is enforced.
Prevents accidental V1 usage and validates environment configuration.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class StartupValidator:
    """Validates system configuration at startup."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.base_path = Path(os.getenv('STORAGE_PATH', '~/yt-dl-sub-storage')).expanduser()
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (success, errors, warnings)
        """
        self._check_storage_version()
        self._check_v1_remnants()
        self._check_migration_status()
        self._check_imports()
        self._validate_environment()
        self._validate_rate_limiting_config()
        self._validate_prevention_system_integration()
        self._validate_settings_integration()
        
        success = len(self.errors) == 0
        
        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"❌ {error}")
        
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"⚠️  {warning}")
        
        if success and not self.warnings:
            logger.info("✅ All startup validation checks passed")
        
        return success, self.errors, self.warnings
    
    def _check_storage_version(self):
        """Ensure STORAGE_VERSION is set to v2."""
        storage_version = os.getenv('STORAGE_VERSION', '').lower()
        
        if not storage_version:
            self.errors.append(
                "STORAGE_VERSION not set in .env - must be set to 'v2'"
            )
        elif storage_version != 'v2':
            self.errors.append(
                f"STORAGE_VERSION is '{storage_version}' but must be 'v2'. "
                f"V1 storage is deprecated and no longer supported."
            )
    
    def _check_v1_remnants(self):
        """Check for V1 directory structure remnants."""
        v1_dirs = ['audio', 'transcripts', 'content', 'metadata']
        found_v1_dirs = []
        
        for dir_name in v1_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # Check if it's not empty
                if any(dir_path.iterdir()):
                    found_v1_dirs.append(dir_name)
        
        if found_v1_dirs:
            self.warnings.append(
                f"Found V1 directories with content: {', '.join(found_v1_dirs)}. "
                f"These should be migrated using scripts/migrate_storage_v2.py"
            )
    
    def _check_migration_status(self):
        """Check if migration has been completed."""
        migration_marker = self.base_path / '.migration_complete_v2'
        
        if not migration_marker.exists():
            # Check if there's any data that needs migration
            v1_dirs = ['audio', 'transcripts', 'content', 'metadata']
            has_v1_data = any(
                (self.base_path / dir_name).exists() and 
                any((self.base_path / dir_name).iterdir())
                for dir_name in v1_dirs
                if (self.base_path / dir_name).exists()
            )
            
            if has_v1_data:
                self.warnings.append(
                    "V1 to V2 migration not completed. Run: python scripts/migrate_storage_v2.py"
                )
    
    def _check_imports(self):
        """Verify that V1 imports will fail."""
        try:
            # This should fail
            from core.storage_paths import StoragePaths
            self.errors.append(
                "V1 storage_paths can still be imported! This should raise ImportError."
            )
        except ImportError as e:
            # This is expected - V1 should not be importable
            if "V1 STORAGE IS DEPRECATED" not in str(e):
                self.errors.append(
                    f"V1 import fails but with wrong error: {e}"
                )
    
    def _validate_environment(self):
        """Validate critical environment variables."""
        required_vars = [
            ('STORAGE_PATH', 'Storage path for all downloads'),
            ('DATABASE_URL', 'Database connection string'),
            ('STORAGE_VERSION', 'Must be set to v2'),
        ]
        
        for var_name, description in required_vars:
            if not os.getenv(var_name):
                self.errors.append(
                    f"Missing required environment variable: {var_name} ({description})"
                )
        
        # Check storage path accessibility
        storage_path = os.getenv('STORAGE_PATH')
        if storage_path:
            path = Path(storage_path).expanduser()
            if not path.exists():
                self.warnings.append(
                    f"Storage path does not exist: {storage_path}. It will be created on first use."
                )
            elif not os.access(path, os.W_OK):
                self.errors.append(
                    f"Storage path is not writable: {storage_path}"
                )
    
    def _validate_rate_limiting_config(self):
        """Validate rate limiting configuration for conflicts."""
        # Check for conflicting rate limiting configurations
        prevention_rate = os.getenv('PREVENTION_RATE_LIMIT')
        youtube_rate = os.getenv('YOUTUBE_RATE_LIMIT')
        
        prevention_burst = os.getenv('PREVENTION_BURST_SIZE')
        youtube_burst = os.getenv('YOUTUBE_BURST_SIZE')
        
        # Validate prevention system variables exist
        if not prevention_rate:
            self.errors.append(
                "PREVENTION_RATE_LIMIT not set - prevention system requires this variable"
            )
        
        if not prevention_burst:
            self.errors.append(
                "PREVENTION_BURST_SIZE not set - prevention system requires this variable"
            )
        
        # Validate legacy system variables exist (for backward compatibility)
        if not youtube_rate:
            self.warnings.append(
                "YOUTUBE_RATE_LIMIT not set - legacy systems may not function properly"
            )
        
        if not youtube_burst:
            self.warnings.append(
                "YOUTUBE_BURST_SIZE not set - legacy systems may not function properly"
            )
        
        # Check for reasonable values
        if prevention_rate:
            try:
                rate_val = int(prevention_rate)
                if not (5 <= rate_val <= 100):
                    self.warnings.append(
                        f"PREVENTION_RATE_LIMIT ({rate_val}) outside recommended range 5-100"
                    )
            except ValueError:
                self.errors.append(
                    f"PREVENTION_RATE_LIMIT must be an integer, got: {prevention_rate}"
                )
        
        if prevention_burst:
            try:
                burst_val = int(prevention_burst)
                if not (3 <= burst_val <= 20):
                    self.warnings.append(
                        f"PREVENTION_BURST_SIZE ({burst_val}) outside recommended range 3-20"
                    )
            except ValueError:
                self.errors.append(
                    f"PREVENTION_BURST_SIZE must be an integer, got: {prevention_burst}"
                )
        
        # Validate other prevention variables
        prevention_vars = [
            ('PREVENTION_CIRCUIT_BREAKER_THRESHOLD', '3-10'),
            ('PREVENTION_CIRCUIT_BREAKER_TIMEOUT', '30-300'),
            ('PREVENTION_MIN_REQUEST_INTERVAL', '0.5-10.0'),
            ('PREVENTION_BACKOFF_BASE', '1.5-3.0'),
            ('PREVENTION_BACKOFF_MAX', '60-600')
        ]
        
        for var_name, range_desc in prevention_vars:
            if not os.getenv(var_name):
                self.warnings.append(
                    f"{var_name} not set - using default value (range: {range_desc})"
                )
    
    def _validate_prevention_system_integration(self):
        """Validate prevention system components integration."""
        try:
            # Test rate limit manager
            from core.rate_limit_manager import get_rate_limit_manager
            manager = get_rate_limit_manager()
            
            # Validate it has required domain configs
            if 'youtube.com' not in manager.domain_configs:
                self.errors.append(
                    "Rate limit manager missing YouTube domain configuration"
                )
            else:
                youtube_config = manager.domain_configs['youtube.com']
                if youtube_config.requests_per_minute <= 0:
                    self.errors.append(
                        "YouTube rate limit configuration has invalid requests_per_minute"
                    )
        except ImportError as e:
            self.errors.append(
                f"Cannot import rate limit manager: {e}"
            )
        except Exception as e:
            self.errors.append(
                f"Error validating rate limit manager: {e}"
            )
        
        try:
            # Test channel enumerator
            from core.channel_enumerator import ChannelEnumerator, EnumerationStrategy
            enumerator = ChannelEnumerator()
            
            # Validate enumeration strategies are accessible
            strategies = list(EnumerationStrategy)
            if len(strategies) != 6:
                self.warnings.append(
                    f"Expected 6 enumeration strategies, found {len(strategies)}"
                )
        except ImportError as e:
            self.errors.append(
                f"Cannot import channel enumerator: {e}"
            )
        except Exception as e:
            self.errors.append(
                f"Error validating channel enumerator: {e}"
            )
        
        try:
            # Test video discovery verifier
            from core.video_discovery_verifier import VideoDiscoveryVerifier, VerificationStatus
            verifier = VideoDiscoveryVerifier()
            
            # Validate verification statuses are accessible
            statuses = list(VerificationStatus)
            if len(statuses) != 4:
                self.warnings.append(
                    f"Expected 4 verification statuses, found {len(statuses)}"
                )
        except ImportError as e:
            self.errors.append(
                f"Cannot import video discovery verifier: {e}"
            )
        except Exception as e:
            self.errors.append(
                f"Error validating video discovery verifier: {e}"
            )
    
    def _validate_settings_integration(self):
        """Validate centralized settings system integration."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            
            # Validate prevention system settings
            prevention_fields = [
                'prevention_rate_limit',
                'prevention_burst_size', 
                'prevention_circuit_breaker_threshold',
                'prevention_circuit_breaker_timeout',
                'prevention_min_request_interval',
                'prevention_backoff_base',
                'prevention_backoff_max'
            ]
            
            for field in prevention_fields:
                if not hasattr(settings, field):
                    self.errors.append(
                        f"Settings missing required prevention field: {field}"
                    )
                else:
                    value = getattr(settings, field)
                    if value is None:
                        self.errors.append(
                            f"Settings field {field} is None"
                        )
            
            # Validate enumeration settings
            enumeration_fields = [
                'default_enumeration_strategy',
                'force_complete_enumeration',
                'max_videos_per_channel',
                'enumeration_timeout',
                'cache_duration_hours',
                'incremental_check_interval'
            ]
            
            for field in enumeration_fields:
                if not hasattr(settings, field):
                    self.errors.append(
                        f"Settings missing required enumeration field: {field}"
                    )
            
            # Validate verification settings
            verification_fields = [
                'verify_channel_completeness',
                'deep_check_threshold',
                'missing_video_confidence',
                'verification_sample_size'
            ]
            
            for field in verification_fields:
                if not hasattr(settings, field):
                    self.errors.append(
                        f"Settings missing required verification field: {field}"
                    )
            
            # Test that prevention systems use settings values
            if hasattr(settings, 'prevention_rate_limit'):
                from core.rate_limit_manager import get_rate_limit_manager
                manager = get_rate_limit_manager()
                if 'youtube.com' in manager.domain_configs:
                    youtube_config = manager.domain_configs['youtube.com']
                    if youtube_config.requests_per_minute != settings.prevention_rate_limit:
                        self.warnings.append(
                            f"Rate manager value ({youtube_config.requests_per_minute}) "
                            f"doesn't match settings ({settings.prevention_rate_limit})"
                        )
        
        except ImportError as e:
            self.errors.append(
                f"Cannot import settings system: {e}"
            )
        except Exception as e:
            self.errors.append(
                f"Error validating settings integration: {e}"
            )


def run_startup_validation(exit_on_error: bool = True) -> bool:
    """
    Run startup validation checks.
    
    Args:
        exit_on_error: If True, exit the program on validation errors
        
    Returns:
        True if validation passed, False otherwise
    """
    validator = StartupValidator()
    success, errors, warnings = validator.validate_all()
    
    if not success and exit_on_error:
        print("\n" + "="*70)
        print("❌ STARTUP VALIDATION FAILED")
        print("="*70)
        for error in errors:
            print(f"  • {error}")
        print("="*70)
        print("\nPlease fix these issues before running the application.")
        print("For V1 to V2 migration, run: python scripts/migrate_storage_v2.py")
        sys.exit(1)
    
    return success


# Auto-run validation if imported (can be disabled by setting SKIP_STARTUP_VALIDATION=1)
if not os.getenv('SKIP_STARTUP_VALIDATION'):
    # Only run if this is being imported, not if run directly
    import __main__
    if hasattr(__main__, '__file__') and __main__.__file__ != __file__:
        # Configure basic logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Run validation but don't exit on error during import
        # (let the main application decide what to do)
        run_startup_validation(exit_on_error=False)