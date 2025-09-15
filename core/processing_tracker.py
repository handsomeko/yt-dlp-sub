#!/usr/bin/env python3
"""
Processing Tracker Module

Comprehensive tracking system to eliminate silent processing failures.
Tracks every video from discovery through completion with audit trails.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Video processing stages for tracking."""
    DISCOVERED = "discovered"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    PUNCTUATING = "punctuating"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"

@dataclass
class VideoProcessingEntry:
    """Comprehensive tracking entry for a single video."""
    video_id: str
    channel_id: str
    title: str
    discovery_time: str
    discovery_source: str  # RSS, yt-dlp, API, etc.
    stage: ProcessingStage
    stage_updated: str
    failure_reason: Optional[str] = None
    processing_notes: Optional[List[str]] = None

class ProcessingTracker:
    """
    Comprehensive video processing tracker to eliminate silent failures.
    Tracks every video from discovery through completion.
    """

    def __init__(self, channel_id: str):
        """Initialize tracker for a specific channel."""
        self.channel_id = channel_id
        self.tracker_file = Path(f"processing_tracker_{channel_id}.json")
        self.entries: Dict[str, VideoProcessingEntry] = {}
        self._load_existing_tracker()

        # Statistics
        self.stats = {
            'discovered': 0,
            'queued': 0,
            'completed': 0,
            'failed': 0,
            'dropped': 0
        }
        self._update_stats()

    def _load_existing_tracker(self):
        """Load existing tracking data if available."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)

                for entry_data in data:
                    entry = VideoProcessingEntry(
                        video_id=entry_data['video_id'],
                        channel_id=entry_data['channel_id'],
                        title=entry_data['title'],
                        discovery_time=entry_data['discovery_time'],
                        discovery_source=entry_data['discovery_source'],
                        stage=ProcessingStage(entry_data['stage']),
                        stage_updated=entry_data['stage_updated'],
                        failure_reason=entry_data.get('failure_reason'),
                        processing_notes=entry_data.get('processing_notes', [])
                    )
                    self.entries[entry.video_id] = entry

                logger.info(f"Loaded {len(self.entries)} existing tracking entries for {self.channel_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing tracker: {e}")

    def _save_tracker(self):
        """Save tracking data to file."""
        try:
            tracker_data = []
            for entry in self.entries.values():
                entry_dict = asdict(entry)
                entry_dict['stage'] = entry.stage.value  # Convert enum to string
                tracker_data.append(entry_dict)

            with open(self.tracker_file, 'w') as f:
                json.dump(tracker_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracker: {e}")

    def _update_stats(self):
        """Update statistics from current entries."""
        self.stats = {stage.value: 0 for stage in ProcessingStage}
        for entry in self.entries.values():
            self.stats[entry.stage.value] += 1

    def track_discovery(self, video_id: str, title: str, discovery_source: str):
        """Track video discovery."""
        if video_id in self.entries:
            # Update existing entry
            self.entries[video_id].discovery_source = discovery_source
            self.entries[video_id].stage_updated = datetime.now().isoformat()
        else:
            # Create new entry
            self.entries[video_id] = VideoProcessingEntry(
                video_id=video_id,
                channel_id=self.channel_id,
                title=title,
                discovery_time=datetime.now().isoformat(),
                discovery_source=discovery_source,
                stage=ProcessingStage.DISCOVERED,
                stage_updated=datetime.now().isoformat(),
                processing_notes=[]
            )

        self._update_stats()
        self._save_tracker()
        logger.debug(f"Tracked discovery: {video_id} from {discovery_source}")

    def track_stage_transition(self, video_id: str, new_stage: ProcessingStage, note: str = None):
        """Track transition to new processing stage."""
        if video_id not in self.entries:
            logger.warning(f"Attempting to track unknown video: {video_id}")
            return

        entry = self.entries[video_id]
        old_stage = entry.stage
        entry.stage = new_stage
        entry.stage_updated = datetime.now().isoformat()

        if note:
            if not entry.processing_notes:
                entry.processing_notes = []
            entry.processing_notes.append(f"{datetime.now().isoformat()}: {note}")

        self._update_stats()
        self._save_tracker()

        logger.info(f"Stage transition: {video_id} {old_stage.value} → {new_stage.value}")

    def track_failure(self, video_id: str, reason: str):
        """Track processing failure with reason."""
        if video_id in self.entries:
            self.track_stage_transition(video_id, ProcessingStage.FAILED)
            self.entries[video_id].failure_reason = reason
            self._save_tracker()
            logger.error(f"Processing failed: {video_id} - {reason}")
        else:
            logger.warning(f"Cannot track failure for unknown video: {video_id}")

    def detect_gaps(self) -> Dict[str, List[str]]:
        """Detect gaps in processing pipeline."""
        gaps = {
            'discovered_not_queued': [],
            'queued_not_started': [],
            'started_not_completed': [],
            'silent_drops': []
        }

        for video_id, entry in self.entries.items():
            if entry.stage == ProcessingStage.DISCOVERED:
                # Video discovered but never queued
                gaps['discovered_not_queued'].append(video_id)
            elif entry.stage == ProcessingStage.QUEUED:
                # Video queued but processing never started
                gaps['queued_not_started'].append(video_id)
            elif entry.stage in [ProcessingStage.DOWNLOADING, ProcessingStage.TRANSCRIBING]:
                # Video started but never completed
                gaps['started_not_completed'].append(video_id)

        # Silent drops - videos that disappeared without failure logging
        all_discovered = {vid for vid, entry in self.entries.items() if entry.stage != ProcessingStage.DROPPED}
        completed_or_failed = {vid for vid, entry in self.entries.items()
                              if entry.stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]}
        gaps['silent_drops'] = list(all_discovered - completed_or_failed)

        return gaps

    def generate_completion_report(self) -> Dict[str, any]:
        """Generate comprehensive completion report."""
        gaps = self.detect_gaps()

        # Calculate true success metrics
        discovered_count = len([e for e in self.entries.values()
                               if e.stage != ProcessingStage.DROPPED])
        completed_count = self.stats[ProcessingStage.COMPLETED.value]
        failed_count = self.stats[ProcessingStage.FAILED.value]

        true_success_rate = (completed_count / discovered_count * 100) if discovered_count > 0 else 0

        report = {
            'channel_id': self.channel_id,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'discovered_videos': discovered_count,
            'completed_videos': completed_count,
            'failed_videos': failed_count,
            'true_success_rate': true_success_rate,
            'gaps_detected': {k: len(v) for k, v in gaps.items()},
            'gap_details': gaps,
            'processing_complete': len(gaps['silent_drops']) == 0 and len(gaps['discovered_not_queued']) == 0
        }

        return report

    def log_completion_summary(self):
        """Log comprehensive completion summary."""
        report = self.generate_completion_report()

        logger.info(f"=== PROCESSING COMPLETION REPORT: {self.channel_id} ===")
        logger.info(f"Discovered: {report['discovered_videos']}")
        logger.info(f"Completed: {report['completed_videos']}")
        logger.info(f"Failed: {report['failed_videos']}")
        logger.info(f"True Success Rate: {report['true_success_rate']:.1f}%")

        # Report gaps
        for gap_type, count in report['gaps_detected'].items():
            if count > 0:
                logger.warning(f"GAP DETECTED: {count} videos {gap_type}")

        if report['processing_complete']:
            logger.info("✅ PROCESSING COMPLETE: All discovered videos accounted for")
        else:
            logger.error("❌ PROCESSING INCOMPLETE: Silent failures detected")

        return report

# Global tracker instances
_trackers: Dict[str, ProcessingTracker] = {}

def get_processing_tracker(channel_id: str) -> ProcessingTracker:
    """Get or create processing tracker for channel."""
    if channel_id not in _trackers:
        _trackers[channel_id] = ProcessingTracker(channel_id)
    return _trackers[channel_id]

def track_video_discovery(channel_id: str, video_id: str, title: str, source: str):
    """Convenience function to track video discovery."""
    tracker = get_processing_tracker(channel_id)
    tracker.track_discovery(video_id, title, source)

def track_video_stage(channel_id: str, video_id: str, stage: ProcessingStage, note: str = None):
    """Convenience function to track stage transitions."""
    tracker = get_processing_tracker(channel_id)
    tracker.track_stage_transition(video_id, stage, note)

def generate_channel_report(channel_id: str) -> Dict[str, any]:
    """Generate completion report for channel."""
    tracker = get_processing_tracker(channel_id)
    return tracker.generate_completion_report()