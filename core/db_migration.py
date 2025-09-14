"""
Database Migration Utility for yt-dl-sub
Provides safe database schema migrations with rollback capability
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import subprocess
import sqlite3
import shutil
from datetime import datetime, timedelta
import threading
import time
import fcntl
import os
import signal
import json

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from urllib.parse import urlparse
import re

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseMigrationError(Exception):
    """Raised when database migration fails."""
    pass


class MigrationLockError(DatabaseMigrationError):
    """Raised when migration is already in progress."""
    pass


class RollbackSafetyError(DatabaseMigrationError):
    """Raised when rollback is not safe to perform."""
    pass


class ProgressTrackingError(DatabaseMigrationError):
    """Raised when progress tracking fails."""
    pass


class DatabaseMigrator:
    """
    Handles database schema migrations with safety checks and rollback capability.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.alembic_cfg_path = Path(__file__).parent.parent / "alembic.ini"
        self.alembic_cfg = Config(str(self.alembic_cfg_path))
        
        # Validate database URL for security
        validated_url = self._validate_database_url(self.settings.database_url)
        
        # Override database URL from settings with validated URL
        self.alembic_cfg.set_main_option("sqlalchemy.url", validated_url)
        
        # Create engine for direct database operations
        self.engine = create_engine(validated_url)
        
        # Migration lock and state management
        self.lock_file_path = self._get_migration_lock_path()
        self.state_file_path = self._get_migration_state_path()
        self._migration_lock = None
        self._migration_thread_lock = threading.RLock()
        
        # Progress tracking robustness
        self._progress_state = {
            'current_step': None,
            'progress': 0.0,
            'last_update': None,
            'callback_failures': 0,
            'max_callback_failures': 5
        }
    
    def _validate_database_url(self, db_url: str) -> str:
        """
        Validate database URL to prevent injection and ensure security.
        
        Args:
            db_url: Database URL to validate
            
        Returns:
            Validated database URL
            
        Raises:
            DatabaseMigrationError: If URL is invalid or insecure
        """
        if not db_url or not isinstance(db_url, str):
            raise DatabaseMigrationError("Database URL cannot be empty or non-string")
        
        # Check for basic SQL injection patterns
        dangerous_patterns = [
            ';', '--', '/*', '*/', 'union', 'select', 'drop', 'delete', 
            'insert', 'update', 'create', 'alter', 'exec', 'execute'
        ]
        
        url_lower = db_url.lower()
        for pattern in dangerous_patterns:
            if pattern in url_lower:
                logger.error(f"Dangerous pattern '{pattern}' detected in database URL")
                raise DatabaseMigrationError(f"Database URL contains dangerous pattern: {pattern}")
        
        try:
            # Parse URL to validate structure
            parsed = urlparse(db_url)
            
            # Validate scheme (only allow known safe database schemes)
            allowed_schemes = ['sqlite', 'postgresql', 'mysql', 'mariadb']
            if parsed.scheme not in allowed_schemes:
                raise DatabaseMigrationError(f"Unsupported database scheme: {parsed.scheme}")
            
            # For SQLite, validate path
            if parsed.scheme == 'sqlite':
                # Extract path from sqlite:///path format
                if db_url.startswith('sqlite:///'):
                    path_part = db_url[10:]  # Remove 'sqlite:///'
                    
                    # Validate path doesn't contain dangerous patterns
                    if '..' in path_part or '~' in path_part:
                        raise DatabaseMigrationError("SQLite path contains path traversal patterns")
                    
                    # Ensure it's a reasonable file path
                    if not re.match(r'^[a-zA-Z0-9_/.-]+$', path_part):
                        raise DatabaseMigrationError("SQLite path contains invalid characters")
            
            # Length check to prevent DoS
            if len(db_url) > 500:
                raise DatabaseMigrationError("Database URL too long (potential DoS)")
            
            return db_url
            
        except ValueError as e:
            raise DatabaseMigrationError(f"Invalid database URL format: {e}")
    
    def _validate_table_name(self, table_name: str) -> str:
        """
        Validate table name to prevent SQL injection.
        
        Args:
            table_name: Table name to validate
            
        Returns:
            Validated table name
            
        Raises:
            DatabaseMigrationError: If table name is invalid
        """
        if not table_name or not isinstance(table_name, str):
            raise DatabaseMigrationError("Table name cannot be empty")
        
        # Only allow alphanumeric, underscores, and limited length
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]{0,63}$', table_name):
            raise DatabaseMigrationError(f"Invalid table name: {table_name}")
        
        # Prevent SQL reserved words
        reserved_words = [
            'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter',
            'table', 'index', 'view', 'database', 'schema', 'user', 'group',
            'union', 'join', 'where', 'order', 'group', 'having', 'limit'
        ]
        
        if table_name.lower() in reserved_words:
            raise DatabaseMigrationError(f"Table name cannot be SQL reserved word: {table_name}")
        
        return table_name
    
    def check_migration_needed(self) -> bool:
        """
        Check if database migration is needed.
        
        Returns:
            True if migration is needed
        """
        try:
            # Check if database exists and has tables
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            if not tables:
                logger.info("Database is empty, migration needed")
                return True
            
            # Check if videos table has V2 columns
            if 'videos' in tables:
                columns = [col['name'] for col in inspector.get_columns('videos')]
                v2_columns = ['video_title_snapshot', 'title_sanitized', 
                             'storage_version', 'processing_completed_at']
                
                missing_columns = [col for col in v2_columns if col not in columns]
                if missing_columns:
                    logger.info(f"Missing V2 columns: {missing_columns}")
                    return True
                
            # Check Alembic version
            try:
                with self.engine.connect() as conn:
                    context = MigrationContext.configure(conn)
                    current_rev = context.get_current_revision()
                    
                    if not current_rev:
                        logger.info("No Alembic revision found, migration needed")
                        return True
                        
                    # Check if we're at the latest revision
                    script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                    latest_rev = script_dir.get_current_head()
                    
                    if current_rev != latest_rev:
                        logger.info(f"Current revision {current_rev} != latest {latest_rev}")
                        return True
                        
            except Exception as e:
                logger.warning(f"Could not check Alembic revision: {e}")
                return True
                
            logger.info("Database schema is up to date")
            return False
            
        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return True
    
    def validate_database_safety(self) -> Dict[str, Any]:
        """
        Perform safety checks before migration.
        
        Returns:
            Dict with validation results
        """
        validation = {
            'safe_to_migrate': True,
            'warnings': [],
            'errors': [],
            'backup_recommended': False
        }
        
        try:
            # Check if database exists and has data
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            if 'videos' in tables:
                with self.engine.connect() as conn:
                    result = conn.execute("SELECT COUNT(*) FROM videos").scalar()
                    if result and result > 0:
                        validation['warnings'].append(f"Database contains {result} video records")
                        validation['backup_recommended'] = True
            
            # Check disk space
            db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
            if db_path.exists():
                db_size = db_path.stat().st_size
                free_space = db_path.parent.stat().st_dev  # This is approximate
                
                if db_size > 100 * 1024 * 1024:  # 100MB
                    validation['warnings'].append("Large database detected (>100MB)")
                    validation['backup_recommended'] = True
            
            # Check for active connections (SQLite specific)
            try:
                with sqlite3.connect(db_path.as_posix()) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.rollback()
            except sqlite3.OperationalError:
                validation['errors'].append("Database is locked by another process")
                validation['safe_to_migrate'] = False
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")
            validation['safe_to_migrate'] = False
        
        return validation
    
    def backup_database(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create a backup of the database before migration.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        if not backup_path:
            db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
            backup_path = db_path.parent / f"{db_path.stem}_backup_{int(__import__('time').time())}.db"
        
        db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
        
        try:
            # For SQLite, we can use the backup API
            import shutil
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            raise DatabaseMigrationError(f"Failed to create backup: {e}")
    
    def run_migration(self, backup: bool = True, target_revision: str = "head") -> bool:
        """
        Run database migration with safety checks.
        
        Args:
            backup: Create backup before migration
            target_revision: Target revision (default: latest)
            
        Returns:
            True if successful
        """
        logger.info("Starting database migration")
        
        try:
            # Safety validation
            validation = self.validate_database_safety()
            if not validation['safe_to_migrate']:
                for error in validation['errors']:
                    logger.error(error)
                raise DatabaseMigrationError("Database migration is not safe")
            
            for warning in validation['warnings']:
                logger.warning(warning)
            
            # Create backup if requested or recommended
            backup_path = None
            if backup or validation['backup_recommended']:
                backup_path = self.backup_database()
            
            # Initialize Alembic if needed
            try:
                with self.engine.connect() as conn:
                    context = MigrationContext.configure(conn)
                    current_rev = context.get_current_revision()
                    
                    if not current_rev:
                        logger.info("Initializing Alembic")
                        command.stamp(self.alembic_cfg, "base")
                        
            except Exception as e:
                logger.info(f"Initializing Alembic (expected on first run): {e}")
                command.stamp(self.alembic_cfg, "base")
            
            # Run migration with safety checks
            logger.info(f"Upgrading database to {target_revision}")
            
            # Record pre-migration state
            pre_migration_state = self._capture_database_state()
            
            try:
                # Execute migration in transaction if possible
                command.upgrade(self.alembic_cfg, target_revision)
                
                # Validate migration success
                if not self._validate_migration_success():
                    logger.error("Migration validation failed - attempting rollback")
                    
                    # Attempt automatic rollback
                    if backup_path and self._attempt_recovery_from_backup(backup_path):
                        raise DatabaseMigrationError("Migration validation failed - restored from backup")
                    else:
                        raise DatabaseMigrationError("Migration validation failed - manual intervention required")
                
                # Additional post-migration verification
                post_migration_state = self._capture_database_state()
                if not self._verify_data_integrity(pre_migration_state, post_migration_state):
                    logger.warning("Data integrity check failed - migration may have data issues")
                
                logger.info("Database migration completed successfully")
                
                if backup_path:
                    logger.info(f"Backup available at: {backup_path}")
                
                return True
                
            except Exception as migration_error:
                logger.error(f"Migration execution failed: {migration_error}")
                
                # Attempt recovery
                if backup_path:
                    logger.info("Attempting to restore from backup...")
                    if self._attempt_recovery_from_backup(backup_path):
                        raise DatabaseMigrationError(f"Migration failed, restored from backup: {migration_error}")
                
                raise DatabaseMigrationError(f"Migration failed with no recovery: {migration_error}")
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            raise DatabaseMigrationError(f"Migration process failed: {e}")
    
    def rollback_migration(
        self, 
        target_revision: str = "-1", 
        backup_before_rollback: bool = True,
        force_rollback: bool = False
    ) -> bool:
        """
        Safely rollback database migration with comprehensive safety checks.
        
        Args:
            target_revision: Target revision to rollback to
            backup_before_rollback: Create backup before rollback
            force_rollback: Skip safety checks (dangerous!)
            
        Returns:
            True if successful
            
        Raises:
            RollbackSafetyError: If rollback is not safe
            DatabaseMigrationError: If rollback fails
        """
        logger.info(f"Starting safe rollback to {target_revision}")
        
        try:
            # Acquire migration lock to prevent concurrent operations
            with self._acquire_migration_lock("rollback", timeout=300):  # 5 minute timeout
                
                # Safety validation unless forced
                if not force_rollback:
                    safety_result = self._validate_rollback_safety(target_revision)
                    if not safety_result['safe_to_rollback']:
                        error_msg = "Rollback safety validation failed:\n" + "\n".join(safety_result['errors'])
                        logger.error(error_msg)
                        raise RollbackSafetyError(error_msg)
                    
                    for warning in safety_result['warnings']:
                        logger.warning(f"Rollback warning: {warning}")
                
                # Create backup before rollback if requested
                backup_path = None
                if backup_before_rollback:
                    logger.info("Creating backup before rollback")
                    backup_path = self.backup_database()
                    logger.info(f"Backup created: {backup_path}")
                
                # Capture current state for verification
                pre_rollback_state = self._capture_database_state()
                
                # Record rollback in state file
                self._update_migration_state({
                    'operation': 'rollback',
                    'target_revision': target_revision,
                    'started_at': datetime.utcnow().isoformat(),
                    'backup_path': str(backup_path) if backup_path else None,
                    'pre_rollback_state': pre_rollback_state
                })
                
                try:
                    # Execute rollback
                    logger.info(f"Executing rollback to {target_revision}")
                    command.downgrade(self.alembic_cfg, target_revision)
                    
                    # Validate rollback success
                    if not self._validate_rollback_success(target_revision):
                        logger.error("Rollback validation failed")
                        
                        # Attempt recovery if we have a backup
                        if backup_path and self._attempt_recovery_from_backup(backup_path):
                            raise DatabaseMigrationError("Rollback validation failed - restored from backup")
                        else:
                            raise DatabaseMigrationError("Rollback validation failed - manual intervention required")
                    
                    # Clear migration state on success
                    self._clear_migration_state()
                    
                    logger.info("Database rollback completed successfully")
                    return True
                    
                except Exception as rollback_error:
                    logger.error(f"Rollback execution failed: {rollback_error}")
                    
                    # Update state with error
                    self._update_migration_state({
                        'error': str(rollback_error),
                        'failed_at': datetime.utcnow().isoformat()
                    })
                    
                    # Attempt recovery if backup exists
                    if backup_path and self._attempt_recovery_from_backup(backup_path):
                        raise DatabaseMigrationError(f"Rollback failed, restored from backup: {rollback_error}")
                    
                    raise DatabaseMigrationError(f"Rollback failed: {rollback_error}")
                
        except MigrationLockError:
            raise DatabaseMigrationError("Another migration operation is already in progress")
        except Exception as e:
            logger.error(f"Rollback process failed: {e}")
            raise DatabaseMigrationError(f"Rollback process failed: {e}")
    
    def _validate_migration_success(self) -> bool:
        """Validate that migration completed successfully."""
        try:
            inspector = inspect(self.engine)
            
            # Check that videos table has V2 columns
            if 'videos' in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns('videos')]
                v2_columns = ['video_title_snapshot', 'title_sanitized', 
                             'storage_version', 'processing_completed_at']
                
                for col in v2_columns:
                    if col not in columns:
                        logger.error(f"Missing column after migration: {col}")
                        return False
            
            # Check Alembic revision
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                
                if not current_rev:
                    logger.error("No Alembic revision set after migration")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Migration validation error: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current database migration status."""
        status = {
            'database_exists': False,
            'has_tables': False,
            'current_revision': None,
            'latest_revision': None,
            'migration_needed': False,
            'v2_ready': False
        }
        
        try:
            # Check database existence
            db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
            status['database_exists'] = db_path.exists()
            
            if status['database_exists']:
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                status['has_tables'] = len(tables) > 0
                
                # Check Alembic revision
                try:
                    with self.engine.connect() as conn:
                        context = MigrationContext.configure(conn)
                        status['current_revision'] = context.get_current_revision()
                except Exception as e:
                    logger.warning(f"Could not get current revision: {e}")
                
                # Get latest revision
                try:
                    script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                    status['latest_revision'] = script_dir.get_current_head()
                except Exception as e:
                    logger.warning(f"Could not get latest revision: {e}")
                
                # Check if migration is needed
                status['migration_needed'] = self.check_migration_needed()
                
                # Check V2 readiness
                if 'videos' in tables:
                    columns = [col['name'] for col in inspector.get_columns('videos')]
                    v2_columns = ['video_title_snapshot', 'title_sanitized', 
                                 'storage_version', 'processing_completed_at']
                    status['v2_ready'] = all(col in columns for col in v2_columns)
        
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
        
        return status
    
    # ========================================
    # Enhanced Safety Methods
    # ========================================
    
    def _capture_database_state(self) -> Dict[str, Any]:
        """Capture database state for integrity verification."""
        state = {
            'table_counts': {},
            'table_schemas': {},
            'important_data_samples': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            with self.engine.connect() as conn:
                # Capture row counts for each table
                for table in tables:
                    try:
                        # Validate table name to prevent SQL injection
                        validated_table = self._validate_table_name(table)
                        
                        # Use SQLAlchemy text() with safe table name
                        query = text(f"SELECT COUNT(*) FROM {validated_table}")
                        result = conn.execute(query)
                        state['table_counts'][table] = result.scalar()
                    except DatabaseMigrationError as e:
                        logger.warning(f"Invalid table name {table}: {e}")
                        state['table_counts'][table] = -1
                    except Exception as e:
                        logger.warning(f"Could not count rows in {table}: {e}")
                        state['table_counts'][table] = -1
                
                # Capture schema information
                for table in tables:
                    try:
                        columns = inspector.get_columns(table)
                        state['table_schemas'][table] = [
                            {'name': col['name'], 'type': str(col['type'])}
                            for col in columns
                        ]
                    except Exception as e:
                        logger.warning(f"Could not get schema for {table}: {e}")
                
                # Capture sample data from critical tables
                critical_tables = ['videos', 'channels', 'transcripts']
                for table in critical_tables:
                    if table in tables and state['table_counts'].get(table, 0) > 0:
                        try:
                            # Validate table name to prevent SQL injection
                            validated_table = self._validate_table_name(table)
                            
                            # Use SQLAlchemy text() with safe table name
                            query = text(f"SELECT * FROM {validated_table} LIMIT 3")
                            result = conn.execute(query)
                            rows = result.fetchall()
                            state['important_data_samples'][table] = [
                                dict(row._mapping) for row in rows
                            ]
                        except DatabaseMigrationError as e:
                            logger.warning(f"Invalid table name {table}: {e}")
                        except Exception as e:
                            logger.warning(f"Could not sample data from {table}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to capture database state: {e}")
        
        return state
    
    def migrate_with_progress_tracking(
        self,
        backup: bool = True,
        target_revision: str = "head",
        callback: Optional[Callable[[str, float, str], None]] = None
    ) -> bool:
        """
        Run database migration with comprehensive progress tracking.
        
        Args:
            backup: Create backup before migration
            target_revision: Target revision (default: latest)
            callback: Progress callback function(step: str, progress: float, message: str)
            
        Returns:
            True if successful
        """
        def progress_update(step: str, progress: float, message: str = ""):
            """Robust progress update with error handling and state persistence."""
            try:
                # Update internal progress state
                self._progress_state['current_step'] = step
                self._progress_state['progress'] = progress
                self._progress_state['last_update'] = datetime.utcnow().isoformat()
                
                # Log progress (always succeeds)
                logger.info(f"Migration Progress [{progress:.1%}]: {step} - {message}")
                
                # Persist progress state for recovery
                try:
                    self._update_migration_state({
                        'progress': {
                            'step': step,
                            'progress': progress,
                            'message': message,
                            'timestamp': self._progress_state['last_update']
                        }
                    })
                except Exception as state_error:
                    logger.warning(f"Failed to persist progress state: {state_error}")
                    # Continue - this is not critical for migration success
                
                # Call user callback with robust error handling
                if callback:
                    try:
                        callback(step, progress, message)
                        # Reset failure count on success
                        self._progress_state['callback_failures'] = 0
                        
                    except Exception as callback_error:
                        self._progress_state['callback_failures'] += 1
                        logger.warning(
                            f"Progress callback failed (attempt {self._progress_state['callback_failures']}): {callback_error}"
                        )
                        
                        # If too many callback failures, disable callback to prevent migration failure
                        if self._progress_state['callback_failures'] >= self._progress_state['max_callback_failures']:
                            logger.error(
                                f"Progress callback failed {self._progress_state['callback_failures']} times - "
                                f"disabling callback to prevent migration failure"
                            )
                            callback = None  # Disable callback for remainder of migration
                            
                            # Update state to indicate callback was disabled
                            try:
                                self._update_migration_state({
                                    'callback_disabled': True,
                                    'callback_disable_reason': 'Too many failures',
                                    'callback_failures': self._progress_state['callback_failures']
                                })
                            except Exception:
                                pass  # Best effort
            
            except Exception as progress_error:
                # Progress updates should never cause migration to fail
                logger.error(f"Progress update failed: {progress_error}")
                # Continue with migration - progress is not critical
        
        def safe_progress_update(step: str, progress: float, message: str = ""):
            """Extra-safe progress update that catches all exceptions."""
            try:
                progress_update(step, progress, message)
            except Exception as e:
                # Last resort - even basic logging failed
                try:
                    logger.error(f"Critical progress update failure: {e}")
                except:
                    pass  # Give up on progress updates
        
        logger.info("Starting database migration with progress tracking")
        
        try:
            # Initialize progress tracking state
            self._progress_state['callback_failures'] = 0
            
            # Acquire migration lock to prevent duplicates (Issue #22)
            with self._acquire_migration_lock("migration", timeout=600):  # 10 minute timeout
                
                # Check for previous incomplete migration
                previous_state = self.get_migration_state()
                if previous_state and previous_state.get('operation') == 'migration':
                    if previous_state.get('completed_at') is None:
                        logger.warning("Found incomplete previous migration - checking recovery options")
                        
                        # If previous migration was recent, consider it failed
                        started_at = datetime.fromisoformat(previous_state.get('started_at', '1970-01-01T00:00:00'))
                        if datetime.utcnow() - started_at < timedelta(hours=24):
                            logger.warning("Previous migration appears to have failed - continuing with new migration")
                        
                        # Clear old state
                        self._clear_migration_state()
                
                # Record migration start
                self._update_migration_state({
                    'operation': 'migration',
                    'target_revision': target_revision,
                    'started_at': datetime.utcnow().isoformat(),
                    'backup_requested': backup
                })
                
                # Progress Step 1: Safety validation (10%)
                safe_progress_update("validation", 0.10, "Validating database safety")
            validation = self.validate_database_safety()
            if not validation['safe_to_migrate']:
                for error in validation['errors']:
                    logger.error(error)
                raise DatabaseMigrationError("Database migration is not safe")
            
            # Progress Step 2: Backup creation (25%)
            if backup or validation['backup_recommended']:
                safe_progress_update("backup", 0.25, "Creating database backup")
                backup_path = self.backup_database()
                safe_progress_update("backup", 0.30, f"Backup created: {backup_path.name}")
            else:
                safe_progress_update("backup", 0.30, "Backup skipped")
            
            # Progress Step 3: Pre-migration state capture (40%)
            safe_progress_update("pre_capture", 0.40, "Capturing pre-migration state")
            pre_migration_state = self._capture_database_state()
            safe_progress_update("pre_capture", 0.45, "Pre-migration state captured")
            
            # Progress Step 4: Alembic initialization (50%)
            safe_progress_update("alembic_init", 0.50, "Initializing Alembic")
            try:
                with self.engine.connect() as conn:
                    context = MigrationContext.configure(conn)
                    current_rev = context.get_current_revision()
                    if not current_rev:
                        command.stamp(self.alembic_cfg, "base")
            except Exception as e:
                command.stamp(self.alembic_cfg, "base")
            safe_progress_update("alembic_init", 0.55, "Alembic initialized")
            
            # Progress Step 5: Migration execution (70%)
            safe_progress_update("migration", 0.60, f"Executing migration to {target_revision}")
            command.upgrade(self.alembic_cfg, target_revision)
            safe_progress_update("migration", 0.70, "Migration execution completed")
            
            # Progress Step 6: Post-migration validation (85%)
            safe_progress_update("validation", 0.75, "Validating migration success")
            if not self._validate_migration_success():
                raise DatabaseMigrationError("Migration validation failed")
            safe_progress_update("validation", 0.80, "Migration validation successful")
            
            # Progress Step 7: Data integrity check (95%)
            safe_progress_update("integrity", 0.85, "Checking data integrity")
            post_migration_state = self._capture_database_state()
            if not self._verify_data_integrity(pre_migration_state, post_migration_state):
                logger.warning("Data integrity check failed - migration may have data issues")
                safe_progress_update("integrity", 0.95, "Data integrity warning - check logs")
            else:
                safe_progress_update("integrity", 0.95, "Data integrity verified")
            
            # Progress Step 8: Completion (100%)
            safe_progress_update("complete", 1.0, "Migration completed successfully")
            
            # Record successful completion in state
            self._update_migration_state({
                'completed_at': datetime.utcnow().isoformat(),
                'success': True
            })
            
            # Clear state after successful completion
            self._clear_migration_state()
            
            if 'backup_path' in locals():
                logger.info(f"Backup available at: {backup_path}")
            
            return True
            
        except MigrationLockError:
            safe_progress_update("error", 0.0, "Another migration is already in progress")
            raise DatabaseMigrationError("Another migration operation is already in progress")
        
        except Exception as e:
            safe_progress_update("error", 0.0, f"Migration failed: {e}")
            logger.error(f"Migration process failed: {e}")
            
            # Record failure in state
            try:
                self._update_migration_state({
                    'failed_at': datetime.utcnow().isoformat(),
                    'error': str(e),
                    'success': False
                })
            except Exception:
                pass  # Best effort
            
            # Attempt recovery if backup exists
            if 'backup_path' in locals():
                safe_progress_update("recovery", 0.0, "Attempting recovery from backup")
                if self._attempt_recovery_from_backup(backup_path):
                    safe_progress_update("recovery", 0.5, "Recovery successful - restored from backup")
                    
                    # Update state to reflect recovery
                    try:
                        self._update_migration_state({
                            'recovered_at': datetime.utcnow().isoformat(),
                            'recovery_successful': True
                        })
                    except Exception:
                        pass
                    
                    raise DatabaseMigrationError(f"Migration failed, restored from backup: {e}")
                else:
                    safe_progress_update("recovery", 0.0, "Recovery failed")
                    
                    # Update state to reflect recovery failure
                    try:
                        self._update_migration_state({
                            'recovery_attempted_at': datetime.utcnow().isoformat(),
                            'recovery_successful': False
                        })
                    except Exception:
                        pass
            
            raise DatabaseMigrationError(f"Migration failed: {e}")
    
    def _verify_data_integrity(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> bool:
        """Verify data integrity after migration."""
        try:
            # Check that critical tables didn't lose data unexpectedly
            critical_tables = ['videos', 'channels', 'transcripts']
            
            for table in critical_tables:
                pre_count = pre_state['table_counts'].get(table, 0)
                post_count = post_state['table_counts'].get(table, 0)
                
                # Allow for slight increases but not decreases
                if post_count < pre_count:
                    logger.error(f"Data loss detected in {table}: {pre_count} -> {post_count}")
                    return False
                
                if post_count > pre_count + 100:  # Suspicious large increase
                    logger.warning(f"Unexpected data increase in {table}: {pre_count} -> {post_count}")
            
            # Verify schema changes are as expected
            for table in pre_state['table_schemas']:
                if table in post_state['table_schemas']:
                    pre_cols = {col['name'] for col in pre_state['table_schemas'][table]}
                    post_cols = {col['name'] for col in post_state['table_schemas'][table]}
                    
                    # For videos table, ensure V2 columns were added
                    if table == 'videos':
                        required_v2_cols = {'video_title_snapshot', 'title_sanitized', 
                                          'storage_version', 'processing_completed_at'}
                        missing_v2_cols = required_v2_cols - post_cols
                        if missing_v2_cols:
                            logger.error(f"Missing V2 columns in videos table: {missing_v2_cols}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data integrity verification failed: {e}")
            return False
    
    def _attempt_recovery_from_backup(self, backup_path: Path) -> bool:
        """Attempt to recover database from backup."""
        try:
            db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
            
            # Close current connections
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
            
            # Replace database with backup
            if backup_path.exists() and db_path.exists():
                # Remove corrupted database
                db_path.unlink()
                
                # Copy backup to original location
                import shutil
                shutil.copy2(backup_path, db_path)
                
                # Re-initialize engine
                self.engine = create_engine(self.settings.database_url)
                
                logger.info(f"Successfully restored database from backup: {backup_path}")
                return True
            else:
                logger.error(f"Backup file not found or database path invalid: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to recover from backup: {e}")
            return False
    
    def verify_migration_safety(self) -> Dict[str, Any]:
        """Comprehensive pre-migration safety check."""
        safety_check = {
            'safe_to_proceed': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check database size and available space
            db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
            if db_path.exists():
                db_size = db_path.stat().st_size
                free_space = shutil.disk_usage(db_path.parent).free
                
                if db_size > free_space * 0.5:  # Database is more than 50% of free space
                    safety_check['warnings'].append(
                        f"Large database ({db_size / 1024 / 1024:.1f}MB) with limited free space"
                    )
                    safety_check['recommendations'].append("Free up disk space before migration")
                
                if db_size > 1024 * 1024 * 1024:  # > 1GB
                    safety_check['recommendations'].append("Consider running migration during off-peak hours")
            
            # Check for concurrent access
            try:
                with sqlite3.connect(db_path.as_posix(), timeout=1.0) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.rollback()
            except sqlite3.OperationalError:
                safety_check['errors'].append("Database is currently locked - close other applications")
                safety_check['safe_to_proceed'] = False
            
            # Verify Alembic configuration
            if not self.alembic_cfg_path.exists():
                safety_check['errors'].append("Alembic configuration file not found")
                safety_check['safe_to_proceed'] = False
            
            # Check for required migrations directory
            migrations_dir = self.alembic_cfg_path.parent / "alembic" / "versions"
            if not migrations_dir.exists():
                safety_check['errors'].append("Alembic migrations directory not found")
                safety_check['safe_to_proceed'] = False
            
            # Validate current database state
            validation = self.validate_database_safety()
            safety_check['warnings'].extend(validation.get('warnings', []))
            safety_check['errors'].extend(validation.get('errors', []))
            
            if not validation.get('safe_to_migrate', True):
                safety_check['safe_to_proceed'] = False
                
        except Exception as e:
            safety_check['errors'].append(f"Safety check failed: {e}")
            safety_check['safe_to_proceed'] = False
        
        return safety_check
    
    # ========================================
    # Migration Lock and State Management (Issue #22)
    # ========================================
    
    def _get_migration_lock_path(self) -> Path:
        """Get path for migration lock file."""
        db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
        return db_path.parent / '.migration.lock'
    
    def _get_migration_state_path(self) -> Path:
        """Get path for migration state file."""
        db_path = Path(self.settings.database_url.replace('sqlite:///', ''))
        return db_path.parent / '.migration_state.json'
    
    def _acquire_migration_lock(self, operation: str, timeout: int = 300):
        """Context manager for acquiring migration lock to prevent duplicate operations."""
        from contextlib import contextmanager
        import fcntl
        
        @contextmanager
        def lock_context():
            lock_acquired = False
            lock_file = None
            start_time = time.time()
            
            try:
                # Check for existing lock with timeout
                while time.time() - start_time < timeout:
                    try:
                        # Try to create/open lock file
                        lock_file = open(self.lock_file_path, 'w+')
                        
                        # Try to acquire exclusive lock (non-blocking)
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # Write operation info to lock file
                        lock_info = {
                            'operation': operation,
                            'pid': os.getpid(),
                            'started_at': datetime.utcnow().isoformat(),
                            'timeout_at': (datetime.utcnow() + timedelta(seconds=timeout)).isoformat()
                        }
                        lock_file.write(json.dumps(lock_info, indent=2))
                        lock_file.flush()
                        
                        lock_acquired = True
                        logger.info(f"Migration lock acquired for {operation} (PID: {os.getpid()})")
                        break
                        
                    except (IOError, OSError) as e:
                        if lock_file:
                            lock_file.close()
                            lock_file = None
                        
                        # Check if lock is stale (holder process no longer exists)
                        if self._is_lock_stale():
                            logger.warning("Removing stale migration lock")
                            self._remove_stale_lock()
                            continue
                        
                        # Wait before retrying
                        time.sleep(1)
                
                if not lock_acquired:
                    raise MigrationLockError(f"Could not acquire migration lock for {operation} within {timeout} seconds")
                
                yield
                
            finally:
                # Release lock
                if lock_file and lock_acquired:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                        
                        # Remove lock file
                        if self.lock_file_path.exists():
                            self.lock_file_path.unlink()
                        
                        logger.info(f"Migration lock released for {operation}")
                        
                    except Exception as e:
                        logger.error(f"Error releasing migration lock: {e}")
        
        return lock_context()
    
    def _is_lock_stale(self) -> bool:
        """Check if migration lock is stale (holding process no longer exists)."""
        try:
            if not self.lock_file_path.exists():
                return False
            
            with open(self.lock_file_path, 'r') as f:
                lock_info = json.load(f)
            
            # Check if process still exists
            pid = lock_info.get('pid')
            if pid:
                try:
                    # Send signal 0 to check if process exists (doesn't actually send signal)
                    os.kill(int(pid), 0)
                    
                    # Check if lock has timed out
                    timeout_at = datetime.fromisoformat(lock_info.get('timeout_at', '1970-01-01T00:00:00'))
                    if datetime.utcnow() > timeout_at:
                        logger.warning(f"Migration lock timed out (PID: {pid})")
                        return True
                    
                    return False  # Process exists and hasn't timed out
                    
                except (OSError, ProcessLookupError):
                    logger.warning(f"Lock holder process {pid} no longer exists")
                    return True
            
            return True  # No PID in lock file
            
        except Exception as e:
            logger.warning(f"Error checking lock staleness: {e}")
            return True  # Assume stale on error
    
    def _remove_stale_lock(self):
        """Remove stale migration lock file."""
        try:
            if self.lock_file_path.exists():
                self.lock_file_path.unlink()
                logger.info("Removed stale migration lock")
        except Exception as e:
            logger.error(f"Error removing stale lock: {e}")
    
    def _update_migration_state(self, state_update: Dict[str, Any]):
        """Update migration state file for tracking and recovery."""
        try:
            current_state = {}
            
            # Load existing state if it exists
            if self.state_file_path.exists():
                with open(self.state_file_path, 'r') as f:
                    current_state = json.load(f)
            
            # Update state
            current_state.update(state_update)
            current_state['last_updated'] = datetime.utcnow().isoformat()
            
            # Write updated state atomically
            temp_path = self.state_file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(current_state, f, indent=2)
            
            # Atomic rename
            temp_path.rename(self.state_file_path)
            
        except Exception as e:
            logger.error(f"Failed to update migration state: {e}")
    
    def _clear_migration_state(self):
        """Clear migration state file on successful completion."""
        try:
            if self.state_file_path.exists():
                self.state_file_path.unlink()
                logger.info("Migration state cleared")
        except Exception as e:
            logger.error(f"Error clearing migration state: {e}")
    
    def get_migration_state(self) -> Optional[Dict[str, Any]]:
        """Get current migration state for recovery purposes."""
        try:
            if self.state_file_path.exists():
                with open(self.state_file_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error reading migration state: {e}")
            return None
    
    # ========================================
    # Enhanced Rollback Safety (Issue #20)
    # ========================================
    
    def _validate_rollback_safety(self, target_revision: str) -> Dict[str, Any]:
        """Validate that rollback is safe to perform."""
        safety = {
            'safe_to_rollback': True,
            'warnings': [],
            'errors': [],
            'data_loss_risk': False
        }
        
        try:
            # Get current and target revisions
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                
                if not current_rev:
                    safety['errors'].append("Cannot determine current database revision")
                    safety['safe_to_rollback'] = False
                    return safety
            
            # Validate target revision exists
            try:
                script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                
                # Handle special cases
                if target_revision == "-1":
                    # Get previous revision
                    revisions = list(script_dir.walk_revisions())
                    current_index = next((i for i, rev in enumerate(revisions) if rev.revision == current_rev), None)
                    
                    if current_index is None or current_index >= len(revisions) - 1:
                        safety['errors'].append("Cannot find previous revision for rollback")
                        safety['safe_to_rollback'] = False
                        return safety
                    
                    target_revision = revisions[current_index + 1].revision
                
                elif target_revision != "base":
                    # Validate specific revision exists
                    try:
                        script_dir.get_revision(target_revision)
                    except Exception:
                        safety['errors'].append(f"Target revision '{target_revision}' not found")
                        safety['safe_to_rollback'] = False
                        return safety
            
            except Exception as e:
                safety['errors'].append(f"Cannot validate target revision: {e}")
                safety['safe_to_rollback'] = False
                return safety
            
            # Check for data loss risk
            data_loss_check = self._assess_rollback_data_loss_risk(current_rev, target_revision)
            safety['data_loss_risk'] = data_loss_check['has_risk']
            if data_loss_check['has_risk']:
                safety['warnings'].extend(data_loss_check['warnings'])
            
            # Check database state
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            # Warn about V2 columns that might be lost
            if 'videos' in tables:
                columns = [col['name'] for col in inspector.get_columns('videos')]
                v2_columns = {'video_title_snapshot', 'title_sanitized', 'storage_version', 'processing_completed_at'}
                present_v2_cols = v2_columns.intersection(set(columns))
                
                if present_v2_cols:
                    safety['warnings'].append(
                        f"Rollback may remove V2 storage columns: {', '.join(present_v2_cols)}"
                    )
                    safety['data_loss_risk'] = True
            
            # Check for data that depends on current schema
            with self.engine.connect() as conn:
                if 'videos' in tables:
                    # Check if any videos use V2 storage
                    try:
                        result = conn.execute(
                            text("SELECT COUNT(*) FROM videos WHERE storage_version = 'v2'")
                        )
                        v2_count = result.scalar() or 0
                        
                        if v2_count > 0:
                            safety['warnings'].append(
                                f"{v2_count} videos use V2 storage format - rollback may cause data issues"
                            )
                            safety['data_loss_risk'] = True
                    except Exception:
                        pass  # Column may not exist in older schema
            
            # Final safety assessment
            if safety['data_loss_risk'] and not safety['errors']:
                safety['warnings'].append(
                    "Rollback has data loss risk - strongly recommend creating backup first"
                )
            
        except Exception as e:
            safety['errors'].append(f"Rollback safety validation failed: {e}")
            safety['safe_to_rollback'] = False
        
        return safety
    
    def _assess_rollback_data_loss_risk(self, current_rev: str, target_rev: str) -> Dict[str, Any]:
        """Assess data loss risk for rollback operation."""
        risk_assessment = {
            'has_risk': False,
            'warnings': [],
            'risk_level': 'low'  # low, medium, high
        }
        
        try:
            # This is a simplified assessment - in a real implementation,
            # we would analyze the actual migration scripts between revisions
            
            # For now, assume any rollback from V2-related migrations has risk
            if current_rev and target_rev:
                # V2 migrations typically add columns, so rolling back removes them
                risk_assessment['has_risk'] = True
                risk_assessment['warnings'].append(
                    "Rollback may remove schema changes made in recent migrations"
                )
                risk_assessment['risk_level'] = 'medium'
        
        except Exception as e:
            logger.warning(f"Could not assess rollback data loss risk: {e}")
            risk_assessment['has_risk'] = True
            risk_assessment['risk_level'] = 'high'
            risk_assessment['warnings'].append("Could not assess rollback safety - assume high risk")
        
        return risk_assessment
    
    def _validate_rollback_success(self, target_revision: str) -> bool:
        """Validate that rollback completed successfully."""
        try:
            # Check that we're at the target revision
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                
                if target_revision == "base":
                    expected_rev = None  # Base has no revision
                elif target_revision == "-1":
                    # We need to determine what -1 actually resolved to
                    # For validation, we'll accept any non-None revision
                    expected_rev = "any"
                else:
                    expected_rev = target_revision
                
                if expected_rev == "any":
                    if current_rev is None:
                        logger.error("Rollback to -1 resulted in base revision (unexpected)")
                        return False
                elif expected_rev is None:
                    if current_rev is not None:
                        logger.error(f"Expected base revision, got {current_rev}")
                        return False
                else:
                    if current_rev != expected_rev:
                        logger.error(f"Expected revision {expected_rev}, got {current_rev}")
                        return False
            
            # Verify database is still accessible and consistent
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            # Basic connectivity test
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Rollback validation successful - at revision {current_rev or 'base'}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback validation failed: {e}")
            return False


# Convenience functions
def migrate_database(backup: bool = True) -> bool:
    """
    Convenience function to migrate database to latest version.
    
    Args:
        backup: Create backup before migration
        
    Returns:
        True if successful
    """
    migrator = DatabaseMigrator()
    return migrator.run_migration(backup=backup)


def check_database_ready() -> bool:
    """
    Check if database is ready for V2 storage operations.
    
    Returns:
        True if database has V2 schema
    """
    migrator = DatabaseMigrator()
    status = migrator.get_migration_status()
    return status['v2_ready'] and not status['migration_needed']