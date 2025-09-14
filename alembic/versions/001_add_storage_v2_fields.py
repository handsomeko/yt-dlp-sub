"""Add storage V2 fields for filename management

Revision ID: 001_add_storage_v2_fields
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_add_storage_v2_fields'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add V2 storage fields to videos table."""
    
    # Store current timestamp to identify pre-migration records
    migration_timestamp = sa.text("datetime('now')")
    
    # Add new columns for V2 storage management
    with op.batch_alter_table('videos', schema=None) as batch_op:
        # Storage V2 fields for filename management
        batch_op.add_column(sa.Column('video_title_snapshot', sa.String(500), nullable=True,
                                     comment='Title at time of processing for consistent filenames'))
        batch_op.add_column(sa.Column('title_sanitized', sa.String(200), nullable=True,
                                     comment='Cached sanitized filename'))
        batch_op.add_column(sa.Column('storage_version', sa.String(10), 
                                     nullable=False, server_default='v1',  # Changed default to v1
                                     comment='Storage structure version: v1 or v2'))
        batch_op.add_column(sa.Column('processing_completed_at', sa.DateTime(), nullable=True,
                                     comment='When all processing finished'))
        
        # Add index for storage version queries
        batch_op.create_index('idx_videos_storage_version', ['storage_version'])
    
    # No UPDATE needed - existing records get v1 by default, new records will be explicitly set to v2 by the application


def downgrade() -> None:
    """Remove V2 storage fields from videos table."""
    
    with op.batch_alter_table('videos', schema=None) as batch_op:
        # Drop index first
        batch_op.drop_index('idx_videos_storage_version')
        
        # Remove V2 columns
        batch_op.drop_column('processing_completed_at')
        batch_op.drop_column('storage_version')
        batch_op.drop_column('title_sanitized')
        batch_op.drop_column('video_title_snapshot')