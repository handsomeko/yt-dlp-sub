#!/usr/bin/env python3
"""
Database initialization script for yt-dl-sub.

Creates all necessary tables including FTS5 search tables.
Run this before first use or to reset the database.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import db_manager, Base
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize the database with all tables."""
    try:
        logger.info("Initializing database...")
        
        # Initialize the database connection
        await db_manager.initialize()
        
        # Create all tables
        async with db_manager.engine.begin() as conn:
            # Create main tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Created main database tables")
            
            # Create FTS5 tables for search
            await conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                    video_id UNINDEXED,
                    content_text,
                    content='transcripts',
                    content_rowid='id'
                )
            """))
            
            await conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    video_id UNINDEXED,
                    content_type UNINDEXED,
                    content,
                    content='generated_content',
                    content_rowid='id'
                )
            """))
            
            # Create triggers to keep FTS tables in sync
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
                    INSERT INTO transcripts_fts(video_id, content_text) 
                    VALUES (new.video_id, new.content_text);
                END
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
                    DELETE FROM transcripts_fts WHERE rowid = old.id;
                END
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
                    DELETE FROM transcripts_fts WHERE rowid = old.id;
                    INSERT INTO transcripts_fts(video_id, content_text) 
                    VALUES (new.video_id, new.content_text);
                END
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS content_ai AFTER INSERT ON generated_content BEGIN
                    INSERT INTO content_fts(video_id, content_type, content) 
                    VALUES (new.video_id, new.content_type, new.content);
                END
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS content_ad AFTER DELETE ON generated_content BEGIN
                    DELETE FROM content_fts WHERE rowid = old.id;
                END
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS content_au AFTER UPDATE ON generated_content BEGIN
                    DELETE FROM content_fts WHERE rowid = old.id;
                    INSERT INTO content_fts(video_id, content_type, content) 
                    VALUES (new.video_id, new.content_type, new.content);
                END
            """))
            
            logger.info("Created FTS5 search tables and triggers")
        
        # Verify tables were created
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            logger.info(f"Database initialized with {len(tables)} tables:")
            for table in sorted(tables):
                logger.info(f"  - {table}")
        
        logger.info("✅ Database initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(init_database())
    sys.exit(0 if success else 1)