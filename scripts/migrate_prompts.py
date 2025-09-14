#!/usr/bin/env python3
"""
Migration script to load prompt templates from YAML files into database.
Separates transcript quality and content quality prompts into respective tables.
"""

import os
import sys
import yaml
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabaseManager
from core.database import TranscriptQualityPrompt, ContentQualityPrompt
from core.database import ContentGenerationPrompt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_yaml_prompts(directory: Path) -> list:
    """Load all YAML prompt files from a directory."""
    prompts = []
    
    for yaml_file in directory.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
                
            # Split on --- separator for multiple prompts in one file
            sections = content.split('\n---\n')
            
            for section in sections:
                if section.strip():
                    prompt_data = yaml.safe_load(section)
                    if prompt_data:
                        prompts.append(prompt_data)
                        logger.info(f"Loaded prompt: {prompt_data.get('name', 'unnamed')}")
                        
        except Exception as e:
            logger.error(f"Error loading {yaml_file}: {str(e)}")
    
    return prompts


async def migrate_transcript_quality_prompts(db_manager: DatabaseManager):
    """Migrate transcript quality prompts to database."""
    prompts_dir = Path("/Users/jk/yt-dl-sub/prompts/transcript_quality")
    
    if not prompts_dir.exists():
        logger.warning(f"Directory not found: {prompts_dir}")
        return
    
    prompts = await load_yaml_prompts(prompts_dir)
    
    async with db_manager.get_session() as session:
        for prompt_data in prompts:
            try:
                # Check if prompt already exists
                existing = session.query(TranscriptQualityPrompt).filter_by(
                    name=prompt_data['name']
                ).first()
                
                if existing:
                    logger.info(f"Updating existing transcript quality prompt: {prompt_data['name']}")
                    existing.template = prompt_data['template']
                    existing.description = prompt_data.get('description', '')
                    existing.strictness_level = prompt_data.get('strictness_level', 'standard')
                    existing.variables = prompt_data.get('variables', [])
                    existing.updated_at = datetime.utcnow()
                else:
                    logger.info(f"Creating new transcript quality prompt: {prompt_data['name']}")
                    new_prompt = TranscriptQualityPrompt(
                        name=prompt_data['name'],
                        template=prompt_data['template'],
                        description=prompt_data.get('description', ''),
                        strictness_level=prompt_data.get('strictness_level', 'standard'),
                        variables=prompt_data.get('variables', []),
                        version=1,
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(new_prompt)
                
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error migrating transcript prompt {prompt_data.get('name')}: {str(e)}")
                await session.rollback()


async def migrate_content_quality_prompts(db_manager: DatabaseManager):
    """Migrate content quality prompts to database."""
    prompts_dir = Path("/Users/jk/yt-dl-sub/prompts/content_quality")
    
    if not prompts_dir.exists():
        logger.warning(f"Directory not found: {prompts_dir}")
        return
    
    prompts = await load_yaml_prompts(prompts_dir)
    
    async with db_manager.get_session() as session:
        for prompt_data in prompts:
            try:
                # Determine content type from the prompt name
                content_type = prompt_data.get('content_type', 'any')
                if 'blog' in prompt_data['name']:
                    content_type = 'blog'
                elif 'social' in prompt_data['name']:
                    content_type = 'social'
                elif 'summary' in prompt_data['name']:
                    content_type = 'summary'
                
                # Check if prompt already exists
                existing = session.query(ContentQualityPrompt).filter_by(
                    name=prompt_data['name']
                ).first()
                
                if existing:
                    logger.info(f"Updating existing content quality prompt: {prompt_data['name']}")
                    existing.template = prompt_data['template']
                    existing.content_type = content_type
                    existing.description = prompt_data.get('description', '')
                    existing.platform = prompt_data.get('platform')
                    existing.variables = prompt_data.get('variables', [])
                    existing.updated_at = datetime.utcnow()
                else:
                    logger.info(f"Creating new content quality prompt: {prompt_data['name']}")
                    new_prompt = ContentQualityPrompt(
                        name=prompt_data['name'],
                        template=prompt_data['template'],
                        content_type=content_type,
                        description=prompt_data.get('description', ''),
                        platform=prompt_data.get('platform'),
                        variables=prompt_data.get('variables', []),
                        version=1,
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(new_prompt)
                
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error migrating content prompt {prompt_data.get('name')}: {str(e)}")
                await session.rollback()


async def migrate_content_generation_prompts(db_manager: DatabaseManager):
    """Migrate content generation prompts to database."""
    prompts_dir = Path("/Users/jk/yt-dl-sub/prompts/content_generation")
    
    if not prompts_dir.exists():
        logger.warning(f"Directory not found: {prompts_dir}")
        return
    
    prompts = await load_yaml_prompts(prompts_dir)
    
    async with db_manager.get_session() as session:
        for prompt_data in prompts:
            try:
                # Check if prompt already exists
                existing = session.query(ContentGenerationPrompt).filter_by(
                    name=prompt_data['name'],
                    content_type=prompt_data['content_type']
                ).first()
                
                if existing:
                    logger.info(f"Updating existing generation prompt: {prompt_data['name']}")
                    existing.template = prompt_data['template']
                    existing.description = prompt_data.get('description', '')
                    existing.variables = prompt_data.get('variables', [])
                    existing.updated_at = datetime.utcnow()
                else:
                    logger.info(f"Creating new generation prompt: {prompt_data['name']}")
                    new_prompt = ContentGenerationPrompt(
                        name=prompt_data['name'],
                        content_type=prompt_data['content_type'],
                        template=prompt_data['template'],
                        description=prompt_data.get('description', ''),
                        variables=prompt_data.get('variables', []),
                        version=1,
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(new_prompt)
                
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error migrating generation prompt {prompt_data.get('name')}: {str(e)}")
                await session.rollback()


async def verify_migration(db_manager: DatabaseManager):
    """Verify that prompts were migrated successfully."""
    async with db_manager.get_session() as session:
        # Count prompts in each table
        transcript_count = session.query(TranscriptQualityPrompt).count()
        content_count = session.query(ContentQualityPrompt).count()
        generation_count = session.query(ContentGenerationPrompt).count()
        
        logger.info("Migration Summary:")
        logger.info(f"  Transcript Quality Prompts: {transcript_count}")
        logger.info(f"  Content Quality Prompts: {content_count}")
        logger.info(f"  Content Generation Prompts: {generation_count}")
        
        # Show active prompts
        logger.info("\nActive Prompts:")
        
        transcript_prompts = session.query(TranscriptQualityPrompt).filter_by(is_active=True).all()
        for prompt in transcript_prompts:
            logger.info(f"  [Transcript] {prompt.name} (v{prompt.version})")
        
        content_prompts = session.query(ContentQualityPrompt).filter_by(is_active=True).all()
        for prompt in content_prompts:
            logger.info(f"  [Content] {prompt.name} - {prompt.content_type} (v{prompt.version})")
        
        generation_prompts = session.query(ContentGenerationPrompt).filter_by(is_active=True).all()
        for prompt in generation_prompts:
            logger.info(f"  [Generation] {prompt.name} - {prompt.content_type} (v{prompt.version})")


async def main():
    """Main migration function."""
    logger.info("Starting prompt migration...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    await db_manager.init_db()
    
    # Run migrations
    logger.info("\nMigrating transcript quality prompts...")
    await migrate_transcript_quality_prompts(db_manager)
    
    logger.info("\nMigrating content quality prompts...")
    await migrate_content_quality_prompts(db_manager)
    
    logger.info("\nMigrating content generation prompts...")
    await migrate_content_generation_prompts(db_manager)
    
    # Verify migration
    logger.info("\n" + "="*50)
    await verify_migration(db_manager)
    
    logger.info("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(main())