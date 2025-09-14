"""
Transcript Cleaner Module

Cleans and standardizes transcript files, especially YouTube auto-generated captions
that contain XML tags, metadata headers, and duplicate lines.
"""

import re
from typing import List, Tuple, Optional
import logging


class TranscriptCleaner:
    """Clean and standardize transcript files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Regex patterns for cleaning
        self.xml_tag_pattern = re.compile(r'<[^>]+>')  # Matches any XML/HTML tags
        self.timing_tag_pattern = re.compile(r'<\d{2}:\d{2}:\d{2}\.\d{3}>')  # Matches timing tags like <00:00:00.960>
        self.position_attr_pattern = re.compile(r'\s*(align|position):[^%]*%?\s*')  # Matches position attributes
        
    def is_auto_generated(self, content: str) -> bool:
        """
        Detect if content is from YouTube auto-generated captions
        
        Args:
            content: The transcript content to check
            
        Returns:
            True if content appears to be auto-generated
        """
        indicators = [
            'Kind: captions' in content,
            'Language: ' in content,
            '<c>' in content,
            '</c>' in content,
            'align:start' in content,
            'position:0%' in content,
            bool(re.search(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', content))
        ]
        
        # If any 2+ indicators are present, it's likely auto-generated
        matches = sum(1 for ind in indicators if ind)
        return matches >= 2
    
    def clean_auto_srt(self, srt_content: str) -> str:
        """
        Clean auto-generated SRT content
        
        Removes:
        - Metadata headers (Kind: captions, Language: en)
        - XML timing tags and <c> tags
        - Position attributes (align:start position:0%)
        - Duplicate consecutive lines
        
        Args:
            srt_content: Raw SRT content
            
        Returns:
            Cleaned SRT content in standard format
        """
        lines = srt_content.split('\n')
        cleaned_blocks = []
        in_subtitle_block = False
        current_block = []
        last_text = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip metadata headers
            if line.startswith('Kind:') or line.startswith('Language:'):
                continue
            
            # Check if it's a sequence number
            if line.isdigit():
                # Save previous block if exists
                if current_block:
                    cleaned_blocks.append(current_block)
                    current_block = []
                
                in_subtitle_block = True
                continue
            
            # Check if it's a timestamp line
            if '-->' in line:
                # Clean position attributes from timestamp line
                timestamp = self.position_attr_pattern.sub('', line).strip()
                current_block.append(('timestamp', timestamp))
                continue
            
            # If we're in a subtitle block and have non-empty text
            if in_subtitle_block and line:
                # Remove all XML tags
                clean_text = self.remove_xml_tags(line)
                
                # Only add if it's different from the last text (avoid duplicates)
                if clean_text and clean_text != last_text:
                    current_block.append(('text', clean_text))
                    last_text = clean_text
            
            # Empty line marks end of subtitle block
            elif not line and in_subtitle_block:
                in_subtitle_block = False
        
        # Add final block if exists
        if current_block:
            cleaned_blocks.append(current_block)
        
        # Now rebuild with proper sequential numbering
        cleaned_lines = []
        output_block_num = 1
        for block in cleaned_blocks:
            timestamp = None
            texts = []
            
            for block_type, content in block:
                if block_type == 'timestamp':
                    timestamp = content
                elif block_type == 'text':
                    texts.append(content)
            
            # Only add if we have both timestamp and text
            if timestamp and texts:
                cleaned_lines.append(str(output_block_num))
                cleaned_lines.append(timestamp)
                # Combine all text lines (usually just one after deduplication)
                cleaned_lines.append(' '.join(texts))
                cleaned_lines.append('')  # Empty line between blocks
                output_block_num += 1
        
        return '\n'.join(cleaned_lines)
    
    
    def clean_auto_txt(self, txt_content: str) -> str:
        """
        Clean auto-generated TXT content
        
        Removes:
        - Metadata headers
        - All XML/HTML tags
        - Duplicate consecutive lines
        - Extra whitespace
        
        Args:
            txt_content: Raw TXT content
            
        Returns:
            Clean plain text with restored punctuation
        """
        lines = txt_content.split('\n')
        cleaned_lines = []
        last_line = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip metadata headers
            if line.startswith('Kind:') or line.startswith('Language:'):
                continue
            
            # Remove all XML tags
            clean_line = self.remove_xml_tags(line)
            
            # Skip empty lines and duplicates
            if clean_line and clean_line != last_line:
                cleaned_lines.append(clean_line)
                last_line = clean_line
        
        # Join with single spaces for continuous text
        text = ' '.join(cleaned_lines)
        
        # Restore punctuation if enabled
        import os
        if os.getenv('RESTORE_PUNCTUATION', 'true').lower() == 'true':
            from core.chinese_punctuation import ChinesePunctuationRestorer
            restorer = ChinesePunctuationRestorer()
            text, _ = restorer.restore_punctuation_sync(text)
        
        return text
    
    def remove_xml_tags(self, text: str) -> str:
        """
        Remove all XML/HTML-like tags from text
        
        Args:
            text: Text possibly containing XML tags
            
        Returns:
            Text with all tags removed
        """
        # First remove timing tags like <00:00:00.960>
        text = self.timing_tag_pattern.sub('', text)
        
        # Then remove all other XML/HTML tags like <c> and </c>
        text = self.xml_tag_pattern.sub('', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def validate_srt_format(self, srt_content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that SRT content follows standard format
        
        Standard SRT format:
        1. Sequence number
        2. Timestamp (HH:MM:SS,mmm --> HH:MM:SS,mmm)
        3. Subtitle text
        4. Empty line
        
        Args:
            srt_content: SRT content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        lines = srt_content.strip().split('\n')
        
        if not lines:
            return False, "Empty SRT content"
        
        # Check for common issues
        if any(line.startswith('Kind:') or line.startswith('Language:') for line in lines[:5]):
            return False, "Contains metadata headers"
        
        if any('<c>' in line or '</c>' in line for line in lines):
            return False, "Contains XML tags"
        
        if any('align:' in line or 'position:' in line for line in lines):
            return False, "Contains position attributes"
        
        # Basic structure check
        subtitle_blocks = srt_content.strip().split('\n\n')
        for block in subtitle_blocks[:3]:  # Check first 3 blocks
            block_lines = block.strip().split('\n')
            if len(block_lines) < 3:
                continue  # Skip incomplete blocks
            
            # First line should be a number
            if not block_lines[0].isdigit():
                return False, f"Invalid sequence number: {block_lines[0]}"
            
            # Second line should contain timestamp
            if '-->' not in block_lines[1]:
                return False, f"Invalid timestamp format: {block_lines[1]}"
            
            # Should not have XML tags in text
            if any('<' in line and '>' in line for line in block_lines[2:]):
                return False, "Text contains XML/HTML tags"
        
        return True, None
    
    def clean_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """
        Clean a transcript file
        
        Args:
            file_path: Path to the file to clean
            output_path: Optional output path (defaults to overwriting input)
            
        Returns:
            True if cleaning was successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine file type and clean accordingly
            if file_path.endswith('.srt'):
                if self.is_auto_generated(content):
                    cleaned = self.clean_auto_srt(content)
                    self.logger.info(f"Cleaned auto-generated SRT: {file_path}")
                else:
                    cleaned = content  # Already clean
                    self.logger.debug(f"SRT already clean: {file_path}")
            
            elif file_path.endswith('.txt'):
                if self.is_auto_generated(content):
                    cleaned = self.clean_auto_txt(content)
                    self.logger.info(f"Cleaned auto-generated TXT: {file_path}")
                else:
                    # For already clean TXT, just ensure no extra whitespace
                    cleaned = ' '.join(content.split())
                    self.logger.debug(f"TXT already clean: {file_path}")
            
            else:
                self.logger.warning(f"Unsupported file type: {file_path}")
                return False
            
            # Write cleaned content
            output = output_path or file_path
            with open(output, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning file {file_path}: {e}")
            return False