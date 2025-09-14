#!/usr/bin/env python3
"""
Database migration script for yt-dl-sub.

Handles schema updates and migrations for existing databases.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import db_manager
from sqlalchemy import text, inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handle database schema migrations."""
    
    def __init__(self):
        self.db_manager = db_manager
        
    async def get_current_schema_version(self) -> int:
        """Get the current schema version from the database."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                )
                row = result.fetchone()
                return row[0] if row else 0
        except Exception:
            # Schema version table doesn't exist yet
            return 0
    
    async def create_schema_version_table(self):
        """Create the schema version tracking table."""
        async with self.db_manager.get_session() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """))
            await session.commit()
    
    async def get_existing_columns(self, table_name: str) -> List[str]:
        """Get list of existing columns in a table."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text(f"PRAGMA table_info({table_name})")
                )
                columns = [row[1] for row in result]
                return columns
        except Exception:
            return []
    
    async def run_migrations(self):
        """Run all pending migrations."""
        await self.db_manager.initialize()
        await self.create_schema_version_table()
        
        current_version = await self.get_current_schema_version()
        logger.info(f"Current schema version: {current_version}")
        
        migrations = [
            (1, "Add review checkpoint fields to videos table", self.migration_001),
            (2, "Add last_video_id to channels table", self.migration_002),
            (3, "Add rate limit tracking fields", self.migration_003),
        ]
        
        for version, description, migration_func in migrations:
            if version > current_version:
                logger.info(f"Applying migration {version}: {description}")
                try:
                    await migration_func()
                    
                    # Record migration
                    async with self.db_manager.get_session() as session:
                        await session.execute(text("""
                            INSERT INTO schema_version (version, description)
                            VALUES (:version, :description)
                        """), {"version": version, "description": description})
                        await session.commit()
                    
                    logger.info(f"✅ Migration {version} completed")
                except Exception as e:
                    logger.error(f"❌ Migration {version} failed: {e}")
                    raise
        
        logger.info("All migrations completed!")
    
    async def migration_001(self):
        """Add review checkpoint fields to videos table."""
        columns = await self.get_existing_columns("videos")
        
        async with self.db_manager.get_session() as session:
            # Add generation review fields if they don't exist
            if "generation_review_status" not in columns:
                await session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN generation_review_status TEXT DEFAULT 'pending'
                """))
            
            if "generation_approved_at" not in columns:
                await session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN generation_approved_at TIMESTAMP
                """))
            
            if "generation_review_notes" not in columns:
                await session.execute(text("""
                    ALTER TABLE videos 
                    ADD COLUMN generation_review_notes TEXT
                """))
            
            await session.commit()
    
    async def migration_002(self):
        """Add last_video_id tracking to channels table."""
        columns = await self.get_existing_columns("channels")
        
        async with self.db_manager.get_session() as session:
            if "last_video_id" not in columns:
                await session.execute(text("""
                    ALTER TABLE channels 
                    ADD COLUMN last_video_id TEXT
                """))
            
            if "last_check_at" not in columns:
                await session.execute(text("""
                    ALTER TABLE channels 
                    ADD COLUMN last_check_at TIMESTAMP
                """))
            
            await session.commit()
    
    async def migration_003(self):
        """Add rate limit tracking fields."""
        columns = await self.get_existing_columns("channels")
        
        async with self.db_manager.get_session() as session:
            if "rate_limit_hits" not in columns:
                await session.execute(text("""
                    ALTER TABLE channels 
                    ADD COLUMN rate_limit_hits INTEGER DEFAULT 0
                """))
            
            if "last_rate_limit_at" not in columns:
                await session.execute(text("""
                    ALTER TABLE channels 
                    ADD COLUMN last_rate_limit_at TIMESTAMP
                """))
            
            await session.commit()


async def main():
    """Run database migrations."""
    migrator = DatabaseMigrator()
    
    try:
        await migrator.run_migrations()
        return 0
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)