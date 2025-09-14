"""
V1 Storage Paths - DEPRECATED AND REMOVED

This module has been intentionally replaced with an error to prevent accidental usage.
V1 storage structure is deprecated and all code must use V2.

The original V1 code has been archived at: archived/v1_storage_structure/storage_paths_v1.py
"""

raise ImportError(
    "\n" + "="*70 + "\n"
    "❌ V1 STORAGE IS DEPRECATED AND REMOVED\n"
    "="*70 + "\n\n"
    "The V1 storage structure (audio/, transcripts/, etc.) is no longer supported.\n"
    "You MUST use V2 storage structure instead.\n\n"
    "To fix this error:\n"
    "1. Replace: from core.storage_paths import StoragePaths\n"
    "   With:    from core.storage_paths_v2 import get_storage_paths_v2\n\n"
    "2. Update your code:\n"
    "   OLD: storage = StoragePaths()\n"
    "   NEW: storage = get_storage_paths_v2()\n\n"
    "3. Use the new methods:\n"
    "   storage.get_media_dir(channel_id, video_id)\n"
    "   storage.get_transcript_dir(channel_id, video_id)\n"
    "   storage.get_content_dir(channel_id, video_id)\n\n"
    "V1 has been archived at: archived/v1_storage_structure/\n"
    "For migration help, see: scripts/migrate_storage_v2.py\n"
    "="*70
)