"""
Punctuation restoration for transcripts lacking proper sentence boundaries.
Supports Chinese and English text with intelligent punctuation insertion.
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """Restore missing punctuation in transcripts"""
    
    def __init__(self, language: str = 'zh'):
        """
        Initialize punctuation restorer
        
        Args:
            language: Language code ('zh' for Chinese, 'en' for English)
        """
        self.language = language
        
        # Chinese sentence-ending patterns that likely need punctuation
        self.zh_sentence_endings = [
            r'了$', r'的$', r'吧$', r'呢$', r'啊$', r'嗎$', r'吶$',
            r'啦$', r'唄$', r'咧$', r'哦$', r'喔$', r'哇$', r'呀$',
            r'麼$', r'著$', r'過$', r'來$', r'去$', r'上$', r'下$',
            r'裡$', r'外$', r'中$', r'內$', r'後$', r'前$', r'間$'
        ]
        
        # Common Chinese sentence starters (indicates previous sentence should end)
        self.zh_sentence_starters = [
            '我', '你', '他', '她', '它', '我們', '你們', '他們', '她們',
            '這', '那', '什麼', '怎麼', '為什麼', '如果', '因為', '所以',
            '但是', '可是', '然而', '不過', '而且', '並且', '或者', '還是',
            '當', '在', '從', '到', '對', '把', '被', '讓', '叫', '請',
            '第一', '第二', '第三', '首先', '其次', '最後', '另外', '此外',
            '總之', '簡而言之', '換句話說', '也就是說', '比如', '例如',
            '現在', '今天', '明天', '昨天', '以前', '以後', '將來',
            '醫生', '專家', '研究', '根據', '通過', '經過', '採用',
            '很多', '許多', '大家', '各位', '朋友', '同學', '老師'
        ]
        
        # Patterns that indicate a complete thought (for Chinese)
        self.zh_complete_thoughts = [
            r'.{10,}[，,].{10,}',  # Long clause with comma in middle
            r'.{20,}',  # Long segment likely a complete sentence
        ]
        
    def restore_punctuation(self, text: str) -> str:
        """
        Restore punctuation to text lacking sentence boundaries
        
        Args:
            text: Input text with missing punctuation
            
        Returns:
            Text with restored punctuation
        """
        if not text or not text.strip():
            return text
            
        # Check if text already has adequate punctuation
        if self._has_adequate_punctuation(text):
            logger.debug("Text already has adequate punctuation")
            return text
            
        if self.language == 'zh':
            return self._restore_chinese_punctuation(text)
        else:
            return self._restore_english_punctuation(text)
    
    def _has_adequate_punctuation(self, text: str) -> bool:
        """
        Check if text already has adequate punctuation
        
        Args:
            text: Text to check
            
        Returns:
            True if text has adequate punctuation
        """
        # Count sentence-ending punctuation per 100 characters
        text_length = len(text)
        if text_length == 0:
            return True
            
        if self.language == 'zh':
            punct_count = len(re.findall(r'[。！？]', text))
        else:
            punct_count = len(re.findall(r'[.!?]', text))
            
        # Expect at least 1 sentence ending per 50 characters
        expected_punct = text_length / 50
        return punct_count >= expected_punct * 0.5  # Allow some variance
    
    def _restore_chinese_punctuation(self, text: str) -> str:
        """
        Restore punctuation for Chinese text
        
        Args:
            text: Chinese text with missing punctuation
            
        Returns:
            Text with restored punctuation
        """
        # Split by existing commas to preserve natural breaks
        segments = re.split(r'([，,])', text)
        
        result = []
        accumulated = ""
        
        for i, segment in enumerate(segments):
            if segment in '，,':
                # Keep commas
                accumulated += segment
            else:
                segment = segment.strip()
                if not segment:
                    continue
                    
                # Check if we should add a period before this segment
                if accumulated and self._should_end_chinese_sentence(accumulated, segment):
                    # Add period to accumulated text
                    accumulated = accumulated.rstrip('，,') + '。'
                    result.append(accumulated)
                    accumulated = segment
                else:
                    # Continue accumulating
                    if accumulated and not accumulated.endswith('，') and not accumulated.endswith(','):
                        accumulated += segment
                    else:
                        accumulated += segment
        
        # Handle the last segment
        if accumulated:
            # Check if it needs a period
            if not re.search(r'[。！？]$', accumulated):
                if self._looks_like_sentence_end(accumulated):
                    accumulated += '。'
            result.append(accumulated)
        
        # Join and clean up
        final_text = ''.join(result)
        
        # Post-processing: Add periods where obviously missing
        final_text = self._add_obvious_periods(final_text)
        
        # Clean up any double punctuation
        final_text = re.sub(r'([。！？])+', r'\1', final_text)
        final_text = re.sub(r'，+', '，', final_text)
        
        return final_text.strip()
    
    def _should_end_chinese_sentence(self, accumulated: str, next_segment: str) -> bool:
        """
        Determine if accumulated text should end with a period before next segment
        
        Args:
            accumulated: Text accumulated so far
            next_segment: Next text segment
            
        Returns:
            True if sentence should end
        """
        # Check if accumulated ends with sentence-ending patterns
        for pattern in self.zh_sentence_endings:
            if re.search(pattern, accumulated.rstrip('，,')):
                # Check if next segment starts with sentence starter
                for starter in self.zh_sentence_starters:
                    if next_segment.startswith(starter):
                        return True
        
        # Check length - very long accumulated text should end
        if len(accumulated) > 50:
            # Check if next segment starts with common sentence starter
            for starter in self.zh_sentence_starters[:20]:  # Check most common starters
                if next_segment.startswith(starter):
                    return True
        
        # Check for question markers
        if '嗎' in accumulated or '呢' in accumulated or '什麼' in accumulated:
            return True
            
        return False
    
    def _looks_like_sentence_end(self, text: str) -> bool:
        """
        Check if text looks like it should end with punctuation
        
        Args:
            text: Text segment to check
            
        Returns:
            True if it looks like sentence end
        """
        text = text.strip()
        
        # Already has ending punctuation
        if re.search(r'[。！？]$', text):
            return False
            
        # Check for common ending patterns
        for pattern in self.zh_sentence_endings[:10]:  # Check most common
            if re.search(pattern, text):
                return True
                
        # Long text without punctuation likely needs it
        if len(text) > 30 and '，' not in text[-10:]:
            return True
            
        return False
    
    def _add_obvious_periods(self, text: str) -> str:
        """
        Add periods where obviously missing (post-processing)
        
        Args:
            text: Text to process
            
        Returns:
            Text with added periods
        """
        # Add period before obvious sentence starters if missing
        for starter in ['但是', '然而', '因此', '所以', '另外', '此外', '總之', 
                       '第一', '第二', '第三', '首先', '其次', '最後']:
            # Pattern: no punctuation before these starters
            pattern = r'([^。！？，\s])(\s*' + starter + r')'
            text = re.sub(pattern, r'\1。\2', text)
        
        # Add period before "I/you/he/she" patterns if missing
        pattern = r'([^。！？，\s])\s*(我|你|他|她|我們|你們|他們|她們)\s*(說|覺得|認為|想|要|會|能|可以)'
        text = re.sub(pattern, r'\1。\2\3', text)
        
        return text
    
    def _restore_english_punctuation(self, text: str) -> str:
        """
        Restore punctuation for English text (basic implementation)
        
        Args:
            text: English text with missing punctuation
            
        Returns:
            Text with restored punctuation
        """
        # Split by existing punctuation
        segments = re.split(r'([,;:])', text)
        
        result = []
        accumulated = ""
        
        for segment in segments:
            if segment in ',;:':
                accumulated += segment
            else:
                segment = segment.strip()
                if not segment:
                    continue
                    
                # Check if we should add a period
                if accumulated and len(accumulated) > 40:
                    # Check if next segment starts with capital
                    if segment and segment[0].isupper():
                        accumulated += '.'
                        result.append(accumulated)
                        accumulated = segment
                    else:
                        accumulated += ' ' + segment
                else:
                    if accumulated:
                        accumulated += ' ' + segment
                    else:
                        accumulated = segment
        
        if accumulated:
            if not re.search(r'[.!?]$', accumulated):
                accumulated += '.'
            result.append(accumulated)
        
        return ' '.join(result).strip()


def restore_transcript_punctuation(text: str, language: Optional[str] = None) -> str:
    """
    Convenience function to restore punctuation in transcript text
    
    Args:
        text: Input text
        language: Language code (auto-detected if not provided)
        
    Returns:
        Text with restored punctuation
    """
    if not language:
        # Auto-detect language based on characters
        if re.search(r'[\u4e00-\u9fff]', text):
            language = 'zh'
        else:
            language = 'en'
    
    restorer = PunctuationRestorer(language=language)
    return restorer.restore_punctuation(text)