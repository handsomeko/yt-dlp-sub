"""
Chinese Punctuation Restoration Module - STATE 1 CLEAN BASELINE

Simple, rule-based punctuation restoration for Chinese transcripts.
No experimental features, no AI integration, no complex dependencies.
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class ChinesePunctuationRestorer:
    """
    Simple rule-based punctuation restoration for Chinese text.
    """
    
    def __init__(self):
        """Initialize the punctuation restorer."""
        # Chinese sentence-ending patterns
        self.zh_sentence_endings = [
            r'了$', r'的$', r'吧$', r'呢$', r'啊$', r'嗎$', r'吶$',
            r'啦$', r'唄$', r'咧$', r'哦$', r'喔$', r'哇$', r'呀$',
            r'麼$', r'著$', r'過$', r'來$', r'去$', r'上$', r'下$',
            r'裡$', r'外$', r'中$', r'內$', r'後$', r'前$', r'間$'
        ]
        
        # Common Chinese sentence starters
        self.zh_sentence_starters = [
            '我', '你', '他', '她', '它', '我們', '你們', '他們', '她們',
            '這', '那', '什麼', '怎麼', '為什麼', '如果', '因為', '所以',
            '但是', '可是', '然而', '不過', '而且', '並且', '或者', '還是',
            '醫生', '專家', '研究', '根據', '通過', '經過', '採用'
        ]
    
    def detect_chinese_text(self, text: str) -> bool:
        """Detect if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
        return bool(chinese_pattern.search(text))
    
    def has_punctuation(self, text: str) -> bool:
        """Check if text has adequate punctuation."""
        if not text or not text.strip():
            return False
        
        # Check for Chinese punctuation
        chinese_punct = '。！？'
        return any(punct in text for punct in chinese_punct)
    
    def restore_punctuation_sync(self, text: str, srt_content: str = None) -> Tuple[str, bool]:
        """
        Restore punctuation using mechanical SRT-aware approach when available.

        Args:
            text: Input text without punctuation
            srt_content: Optional SRT content for boundary-aware processing

        Returns:
            Tuple[str, bool]: (Restored text, Success flag)
        """
        if not self.detect_chinese_text(text):
            return text, False

        if self.has_punctuation(text):
            return text, False

        # ENHANCED: Use mechanical SRT-aware approach if SRT content available
        if srt_content and srt_content.strip():
            logger.info("Using mechanical SRT-aware punctuation restoration")
            try:
                restored_srt, success = self.restore_srt_punctuation_mechanical(srt_content)
                if success:
                    # Extract text from punctuated SRT
                    segments = self.parse_srt_segments(restored_srt)
                    restored_text = ''.join([seg['text'] for seg in segments])
                    logger.info(f"Mechanical SRT-aware success: {len(segments)} segments processed")
                    return restored_text, True
                else:
                    logger.warning("SRT-aware processing failed, falling back to simple rules")
            except Exception as e:
                logger.error(f"SRT-aware processing error: {e}, falling back to simple rules")

        # Fallback: Apply basic punctuation rules
        logger.info("Using simple rule-based punctuation")
        restored_text = text

        # Add periods after likely sentence endings
        for ending in self.zh_sentence_endings:
            for starter in self.zh_sentence_starters:
                pattern = f'({ending})({starter})'
                restored_text = re.sub(pattern, r'\1。\2', restored_text)

        # Add final period if text doesn't end with punctuation
        if restored_text and not restored_text[-1] in '。！？':
            restored_text += '。'

        # Check if we actually added punctuation
        success = restored_text != text
        return restored_text, success

    def parse_srt_segments(self, srt_content: str) -> List[Dict[str, Any]]:
        """
        Parse SRT content into segments with timing information.

        Args:
            srt_content: SRT file content

        Returns:
            List[Dict]: List of segments with text and timing
        """
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

    def restore_srt_punctuation_mechanical(self, srt_content: str) -> Tuple[str, bool]:
        """
        MECHANICAL SRT-AWARE: Use segment boundaries for intelligent punctuation placement
        No AI calls - pure mechanical approach using natural speech boundaries

        Args:
            srt_content: SRT file content

        Returns:
            Tuple[str, bool]: (Restored SRT content, Success flag)
        """
        # Parse SRT into segments
        segments = self.parse_srt_segments(srt_content)

        if not segments:
            logger.warning("No segments found in SRT content")
            return srt_content, False

        # Filter Chinese segments
        chinese_segments = [seg for seg in segments if self.detect_chinese_text(seg.get('text', ''))]
        if not chinese_segments:
            logger.info("No Chinese segments found")
            return srt_content, False

        logger.info(f"Processing {len(chinese_segments)} Chinese segments with mechanical boundary analysis")

        # Process segments with boundary-aware punctuation
        punctuation_added = 0
        for i, segment in enumerate(chinese_segments):
            original_text = segment['text']

            # Analyze boundary context
            prev_text = chinese_segments[i-1]['text'] if i > 0 else None
            next_text = chinese_segments[i+1]['text'] if i < len(chinese_segments)-1 else None

            boundary_type = self._analyze_segment_boundary(original_text, prev_text, next_text)
            punctuated_text = self._apply_boundary_punctuation(original_text, boundary_type)

            # Update segment
            segment['text'] = punctuated_text

            # Track punctuation added
            if punctuated_text != original_text:
                punctuation_added += 1

        # Rebuild SRT content
        restored_srt = self._rebuild_srt_content(segments)

        success = punctuation_added > 0
        logger.info(f"Mechanical SRT-aware: Added punctuation to {punctuation_added}/{len(chinese_segments)} segments")

        return restored_srt, success

    def _analyze_segment_boundary(self, current_text: str, prev_text: Optional[str], next_text: Optional[str]) -> str:
        """Analyze segment boundary to determine punctuation type"""
        if not current_text:
            return 'continuation'

        # Strong sentence endings
        strong_endings = ['了', '吧', '呢', '啊', '嗎']
        if any(current_text.endswith(ending) for ending in strong_endings):
            if next_text and any(next_text.startswith(starter) for starter in ['我', '你', '他', '這', '那', '醫生']):
                return 'sentence_end'

        # Question indicators
        if any(word in current_text for word in ['什麼', '怎麼', '為什麼', '嗎', '呢']):
            return 'question'

        # Continuation indicators
        if any(current_text.endswith(ending) for ending in ['的', '在', '是', '會', '要']):
            return 'continuation'

        return 'clause_break'

    def _apply_boundary_punctuation(self, text: str, boundary_type: str) -> str:
        """Apply contextual punctuation based on boundary analysis"""
        text = text.rstrip('。！？，')  # Remove existing punctuation

        if boundary_type == 'sentence_end':
            return text + '。'
        elif boundary_type == 'question':
            return text + '？'
        elif boundary_type == 'clause_break':
            return text + '，'
        elif boundary_type == 'continuation':
            return text  # No punctuation
        else:
            return text + '。'

    def _rebuild_srt_content(self, segments: List[Dict[str, Any]]) -> str:
        """Rebuild SRT content with punctuated text"""
        lines = []
        for seg in segments:
            lines.append(str(seg['number']))
            lines.append(f"{seg['start']} --> {seg['end']}")
            lines.append(seg['text'])
            lines.append('')  # Empty line between segments
        return '\n'.join(lines)

# Convenience function for backward compatibility
def restore_punctuation_for_file_sync(file_path: str) -> bool:
    """Restore punctuation for a single file."""
    try:
        from pathlib import Path
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return False
        
        text = file_path_obj.read_text(encoding='utf-8')
        
        restorer = ChinesePunctuationRestorer()
        restored_text, success = restorer.restore_punctuation_sync(text)
        
        if success:
            # Create backup
            backup_path = file_path_obj.with_suffix('.backup')
            backup_path.write_text(text, encoding='utf-8')
            
            # Write restored content
            file_path_obj.write_text(restored_text, encoding='utf-8')
            logger.info(f"Successfully restored punctuation in: {file_path_obj}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False