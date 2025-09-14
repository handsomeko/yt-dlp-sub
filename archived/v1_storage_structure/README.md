# V1 Storage Structure (DEPRECATED)

**Status:** ARCHIVED - DO NOT USE  
**Archived Date:** 2025-09-02  
**Reason:** Migrated to V2 storage structure  

## ⚠️ WARNING
This code is archived for reference only. DO NOT import or use these modules.
All code must use V2 storage structure from `core.storage_paths_v2`

## Why V1 Was Replaced
- **Poor Organization:** Type-based structure (audio/, transcripts/, etc.) was confusing
- **Filename Issues:** Used video_id instead of human-readable titles
- **No Channel Index:** Lacked proper channel-level organization
- **Missing Metadata:** No completion markers or processing status

## V1 Structure (Deprecated)
```
{STORAGE_PATH}/
├── audio/{channel_id}/{video_id}/
├── transcripts/{channel_id}/{video_id}/
├── content/{channel_id}/{video_id}/
└── metadata/{channel_id}/{video_id}/
```

## V2 Structure (Current)
```
{STORAGE_PATH}/
└── {channel_id}/
    └── {video_id}/
        ├── media/
        ├── transcripts/
        ├── content/
        └── metadata/
```

## Migration
To migrate V1 files to V2:
```bash
python scripts/migrate_storage_v2.py --execute --backup
```

## Files in This Archive
- `storage_paths_v1.py` - Original V1 storage path management
- `downloader_v1.py` - Original downloader using V1 paths
- `example_structure.txt` - Example of V1 directory layout

## Contact
If you need to reference V1 for debugging, check the migration script first.
NEVER reintroduce V1 patterns into the codebase.