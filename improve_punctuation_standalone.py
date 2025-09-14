#!/usr/bin/env python3
"""
Standalone Punctuation Improvement Script

Safely improves punctuation in inadequate .zh.txt files using mechanical SRT-aware processing.
Does NOT modify existing system code - pure file processing approach.

Usage:
    python3 improve_punctuation_standalone.py --channel-id UCfGGIPQzHZT6dfsCarC-gbA
    python3 improve_punctuation_standalone.py --channel-id UCfGGIPQzHZT6dfsCarC-gbA --dry-run
    python3 improve_punctuation_standalone.py --file /path/to/file.zh.txt
"""

import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandalonePunctuationImprover:
    """
    Standalone punctuation improvement using mechanical SRT-aware approach.
    No dependencies on existing system - pure file processing.
    """

    def __init__(self):
        """Initialize the standalone improver."""
        pass

    def detect_chinese_text(self, text: str) -> bool:
        """Detect if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
        return bool(chinese_pattern.search(text))

    def has_adequate_punctuation(self, text: str) -> bool:
        """Check if text has adequate punctuation (not just ANY punctuation)."""
        if not text or not text.strip():
            return True  # Empty text is "adequate"

        punct_count = sum(1 for c in text if c in '。！？')
        text_length = len(text.strip())

        # Require meaningful punctuation density
        # At least 3 marks or 1 per 200 characters (whichever is higher)
        required_marks = max(3, text_length // 200)

        is_adequate = punct_count >= required_marks
        logger.debug(f"Punctuation check: {punct_count} marks, {text_length} chars, need {required_marks}, adequate: {is_adequate}")

        return is_adequate

    def parse_srt_segments(self, srt_content: str) -> List[Dict[str, Any]]:
        """Parse SRT content into segments with timing information."""
        segments = []
        current_segment = {}
        lines = srt_content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Segment number
            if line.isdigit():
                if current_segment:
                    segments.append(current_segment)
                current_segment = {'number': int(line), 'text': ''}
                i += 1
                continue

            # Timestamp line
            if '-->' in line:
                if current_segment:
                    times = line.split('-->')
                    current_segment['start'] = times[0].strip()
                    current_segment['end'] = times[1].strip()
                i += 1
                continue

            # Text line(s) - collect until next segment or empty line
            text_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit() and '-->' not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1

            if current_segment and text_lines:
                current_segment['text'] = ' '.join(text_lines)

        # Add last segment
        if current_segment and 'text' in current_segment:
            segments.append(current_segment)

        return segments

    def analyze_segment_boundary(self, current_text: str, prev_text: str, next_text: str) -> str:
        """Analyze segment boundary to determine punctuation type."""
        if not current_text:
            return 'continuation'

        # Strong sentence endings
        strong_endings = ['了', '吧', '呢', '啊', '嗎']
        if any(current_text.endswith(ending) for ending in strong_endings):
            if next_text and any(next_text.startswith(starter) for starter in ['我', '你', '他', '這', '那', '醫生', '專家']):
                return 'sentence_end'

        # Question indicators
        question_words = ['什麼', '怎麼', '為什麼', '嗎', '呢', '哪裡', '誰']
        if any(word in current_text for word in question_words):
            return 'question'

        # Continuation indicators (should flow to next segment)
        continuation_endings = ['的', '在', '是', '會', '要', '可以', '應該']
        if any(current_text.endswith(ending) for ending in continuation_endings):
            return 'continuation'

        return 'clause_break'

    def apply_boundary_punctuation(self, text: str, boundary_type: str) -> str:
        """Apply contextual punctuation based on boundary analysis."""
        # Remove existing punctuation first
        text = text.rstrip('。！？，')

        if boundary_type == 'sentence_end':
            return text + '。'
        elif boundary_type == 'question':
            return text + '？'
        elif boundary_type == 'clause_break':
            return text + '，'
        elif boundary_type == 'continuation':
            return text  # No punctuation for continuation
        else:
            return text + '。'  # Default to period

    def improve_punctuation_from_srt(self, srt_content: str) -> Tuple[str, int]:
        """
        Improve punctuation using SRT boundaries.
        Returns (improved_text, punctuation_count)
        """
        # Parse SRT into segments
        segments = self.parse_srt_segments(srt_content)

        if not segments:
            logger.warning("No segments found in SRT content")
            return "", 0

        # Filter Chinese segments
        chinese_segments = [seg for seg in segments if self.detect_chinese_text(seg.get('text', ''))]
        if not chinese_segments:
            logger.info("No Chinese segments found")
            return "", 0

        logger.info(f"Processing {len(chinese_segments)} Chinese segments with mechanical boundary analysis")

        # Process segments with boundary-aware punctuation
        punctuation_added = 0
        for i, segment in enumerate(chinese_segments):
            original_text = segment['text']

            # Analyze boundary context
            prev_text = chinese_segments[i-1]['text'] if i > 0 else None
            next_text = chinese_segments[i+1]['text'] if i < len(chinese_segments)-1 else None

            boundary_type = self.analyze_segment_boundary(original_text, prev_text, next_text)
            punctuated_text = self.apply_boundary_punctuation(original_text, boundary_type)

            # Update segment
            segment['text'] = punctuated_text

            # Track punctuation added
            if punctuated_text != original_text:
                punctuation_added += 1

        # Extract final text
        improved_text = ''.join([seg['text'] for seg in chinese_segments])
        final_punct_count = sum(1 for c in improved_text if c in '。！？')

        logger.info(f"Mechanical SRT-aware: Added punctuation to {punctuation_added}/{len(chinese_segments)} segments")
        logger.info(f"Final punctuation count: {final_punct_count}")

        return improved_text, final_punct_count

    def process_file(self, txt_file: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process a single .zh.txt file."""
        try:
            # Read current content
            original_content = txt_file.read_text(encoding='utf-8')
            original_punct_count = sum(1 for c in original_content if c in '。！？')

            # Check if improvement needed
            if self.has_adequate_punctuation(original_content):
                return {
                    'status': 'skipped',
                    'reason': 'already_adequate',
                    'original_count': original_punct_count,
                    'improved_count': original_punct_count
                }

            # Find corresponding SRT file
            srt_file = txt_file.with_suffix('.srt')
            if not srt_file.exists():
                return {
                    'status': 'failed',
                    'reason': 'no_srt_file',
                    'original_count': original_punct_count,
                    'improved_count': original_punct_count
                }

            # Read SRT content
            srt_content = srt_file.read_text(encoding='utf-8')

            # Apply mechanical SRT-aware improvement
            improved_content, improved_punct_count = self.improve_punctuation_from_srt(srt_content)

            if not improved_content or improved_punct_count <= original_punct_count:
                return {
                    'status': 'failed',
                    'reason': 'no_improvement',
                    'original_count': original_punct_count,
                    'improved_count': improved_punct_count
                }

            if not dry_run:
                # Create backup
                backup_file = txt_file.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
                backup_file.write_text(original_content, encoding='utf-8')

                # Write improved content
                txt_file.write_text(improved_content, encoding='utf-8')

                logger.info(f"✅ Improved: {txt_file.name} ({original_punct_count} → {improved_punct_count} marks)")
            else:
                logger.info(f"🔍 DRY RUN: {txt_file.name} would improve ({original_punct_count} → {improved_punct_count} marks)")

            return {
                'status': 'improved',
                'reason': 'mechanical_srt_aware',
                'original_count': original_punct_count,
                'improved_count': improved_punct_count,
                'improvement_factor': improved_punct_count / max(1, original_punct_count)
            }

        except Exception as e:
            logger.error(f"❌ Error processing {txt_file.name}: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'original_count': 0,
                'improved_count': 0
            }

    def process_channel(self, channel_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Process all inadequate files in a channel."""
        # Find channel directory
        storage_base = Path('/Volumes/Seagate Exp/Mac 2025/code/yt-dl-sub/downloads')
        channel_dir = storage_base / channel_id

        if not channel_dir.exists():
            logger.error(f"Channel directory not found: {channel_dir}")
            return {'error': 'channel_not_found'}

        # Find all Chinese text files
        txt_files = list(channel_dir.rglob('*.zh.txt'))
        logger.info(f"Found {len(txt_files)} Chinese text files in channel {channel_id}")

        if not txt_files:
            return {'error': 'no_chinese_files'}

        # Process each file
        results = {'improved': 0, 'skipped': 0, 'failed': 0, 'errors': 0, 'total_improvement': 0}

        for txt_file in txt_files:
            result = self.process_file(txt_file, dry_run)

            if result['status'] == 'improved':
                results['improved'] += 1
                results['total_improvement'] += result['improvement_factor']
            elif result['status'] == 'skipped':
                results['skipped'] += 1
            elif result['status'] == 'failed':
                results['failed'] += 1
            elif result['status'] == 'error':
                results['errors'] += 1

        # Calculate average improvement
        if results['improved'] > 0:
            results['average_improvement'] = results['total_improvement'] / results['improved']
        else:
            results['average_improvement'] = 0

        results['total_files'] = len(txt_files)
        return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Standalone punctuation improvement using mechanical SRT-aware processing')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--channel-id', help='Channel ID to process (e.g., UCfGGIPQzHZT6dfsCarC-gbA)')
    group.add_argument('--file', help='Single file to process')

    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')

    args = parser.parse_args()

    improver = StandalonePunctuationImprover()

    if args.channel_id:
        logger.info(f"{'DRY RUN: ' if args.dry_run else ''}Processing channel {args.channel_id}")

        results = improver.process_channel(args.channel_id, args.dry_run)

        if 'error' in results:
            logger.error(f"Channel processing failed: {results['error']}")
            return 1

        logger.info(f"📊 RESULTS:")
        logger.info(f"   Total files: {results['total_files']}")
        logger.info(f"   Improved: {results['improved']}")
        logger.info(f"   Skipped (adequate): {results['skipped']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"   Errors: {results['errors']}")

        if results['improved'] > 0:
            logger.info(f"   Average improvement: {results['average_improvement']:.1f}x")
            logger.info(f"✅ SUCCESS: Dramatically improved punctuation in {results['improved']} files")
        else:
            logger.info("ℹ️  No files needed improvement")

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1

        logger.info(f"{'DRY RUN: ' if args.dry_run else ''}Processing single file {file_path}")

        result = improver.process_file(file_path, args.dry_run)

        logger.info(f"📊 RESULT:")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   Original count: {result['original_count']}")
        logger.info(f"   Improved count: {result['improved_count']}")

        if result['status'] == 'improved':
            logger.info(f"   Improvement: {result['improvement_factor']:.1f}x")
            logger.info("✅ SUCCESS: File punctuation dramatically improved")

    return 0

if __name__ == "__main__":
    exit(main())