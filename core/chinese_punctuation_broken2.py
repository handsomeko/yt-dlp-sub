"""
Chinese Punctuation Restoration Module

This module provides functionality to add proper punctuation to Chinese transcripts
that lack punctuation marks, commonly found in Whisper-generated Chinese transcripts.

Key Features:
- AI-powered punctuation restoration using existing AI backend
- Text chunking for long transcripts
- Backup creation before modification
- Chinese text detection
- Support for both .txt and .srt file formats
"""

import re
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import asyncio

logger = logging.getLogger(__name__)

class ChinesePunctuationRestorer:
    """
    Restores punctuation to Chinese text using AI backend integration.
    """
    
    def __init__(self, ai_backend=None):
        """Initialize the punctuation restorer with AI backend."""
        self.ai_backend = ai_backend
        self.max_chunk_size = 500  # Balanced for performance (60% fewer API calls than 200)
        self.claude_available = None  # Cache for Claude CLI availability
        self.timeout = 30  # Timeout for Claude CLI calls
        
    def detect_chinese_text(self, text: str) -> bool:
        """
        Detect if text contains Chinese characters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            bool: True if text contains Chinese characters
        """
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
        return bool(chinese_pattern.search(text))
    
    def has_punctuation(self, text: str) -> bool:
        """
        Check if text has adequate sentence punctuation density.
        For Chinese text, uses density-based approach rather than just existence.
        
        Args:
            text: Input text to check
            
        Returns:
            bool: True if text contains adequate sentence punctuation
        """
        if not text or not text.strip():
            return False
        
        text_length = len(text.strip())
        if text_length == 0:
            return True
            
        # Check if this is primarily Chinese text
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        is_chinese_text = chinese_chars > text_length * 0.3  # 30% Chinese chars = Chinese text
        
        if is_chinese_text:
            # For Chinese text, check punctuation density
            chinese_punct = '。！？'
            punct_count = sum(1 for c in text if c in chinese_punct)
            
            # Expect at least 1 sentence ending per 100 characters for Chinese
            expected_punct = max(1, text_length / 100)
            return punct_count >= expected_punct * 0.3  # 30% of expected minimum
        else:
            # For non-Chinese text, check for any sentence-ending punctuation
            english_punct = '.!?'
            return any(punct in text for punct in english_punct)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split long text into chunks for API processing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        # Try to split at sentence boundaries first
        sentences = re.split(r'[。！？]', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by length
                    while len(sentence) > self.max_chunk_size:
                        chunks.append(sentence[:self.max_chunk_size])
                        sentence = sentence[self.max_chunk_size:]
                    current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
        """Check if Claude CLI is available."""
        if self.claude_available is not None:
            return self.claude_available
            
        try:
            result = subprocess.run(
                ['claude', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.claude_available = result.returncode == 0
            if not self.claude_available:
                logger.warning("Claude CLI not available or not working properly")
            return self.claude_available
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Claude CLI check failed: {e}")
            self.claude_available = False
            return False
    
    async def restore_punctuation_chunk(self, text_chunk: str) -> str:
        """
        Restore punctuation for a single text chunk using enhanced mechanical rules.
        NO AI CALLS - Pure mechanical approach for reliability and speed.
        
        Args:
            text_chunk: Text chunk to process
            
        Returns:
            str: Text with restored punctuation
        """
        # Use enhanced mechanical heuristics only - no AI bottlenecks
        logger.debug("Using enhanced mechanical punctuation (no AI)")
        return self._apply_enhanced_mechanical_punctuation(text_chunk)
    
    def _apply_enhanced_mechanical_punctuation(self, text: str) -> str:
        """
        PRODUCTION SOLUTION: Enhanced mechanical punctuation with SRT-awareness
        Combines intelligent boundary detection with reliable rule-based processing
        """
        if not text or not self.detect_chinese_text(text):
            return text
        
        # Enhanced Chinese punctuation rules
        import re
        
        # Sentence ending patterns (stronger detection)
        strong_endings = ['了', '吧', '呢', '啊', '嗎', '的']
        medium_endings = ['著', '過', '來', '去', '上', '下', '裡', '外']
        
        # Enhanced sentence starters (more comprehensive)
        starters = [
            '我們', '你們', '他們', '這個', '那個', '什麼', '怎麼', '為什麼', 
            '如果', '因為', '所以', '但是', '可是', '醫生', '專家', '研究',
            '現在', '今天', '明天', '第一', '第二', '另外', '此外', '還有'
        ]
        
        # Apply contextual punctuation rules
        for ending in strong_endings:
            for starter in starters:
                # Strong endings get periods
                pattern = f'({ending})([{starter[0]}])'
                text = re.sub(pattern, r'\1。\2', text)
        
        for ending in medium_endings:
            for starter in starters:
                # Medium endings get commas for flow
                pattern = f'({ending})([{starter[0]}])'
                text = re.sub(pattern, r'\1，\2', text)
        
        # Question patterns
        question_words = ['什麼', '怎麼', '為什麼', '哪裡', '誰', '嗎', '呢']
        for question in question_words:
            pattern = f'({question})([我你他這那什])'
            text = re.sub(pattern, r'\1？\2', text)
        
        # Add final punctuation if missing
        if text and not text[-1] in '。！？，':
            # Choose appropriate ending based on last words
            if any(q in text[-10:] for q in question_words):
                text += '？'
            elif any(e in text[-5:] for e in ['吧', '呢', '啊']):
                text += '！'
            else:
                text += '。'
        
        return text
    
    def _apply_basic_punctuation_heuristics(self, text: str) -> str:
        """
        EMERGENCY FALLBACK: Apply basic punctuation using rule-based heuristics
        Fast fallback when Claude CLI is overloaded
        """
        if not text or not self.detect_chinese_text(text):
            return text
        
        # Split into sentences using common patterns
        import re
        
        # Chinese sentence ending patterns
        endings = ['了', '的', '吧', '呢', '啊', '嗎', '吶', '吗', '呀']
        starters = ['我們', '你們', '他們', '這', '那', '什麼', '怎麼', '為什麼', '如果', '因為', '所以', '但是', '可是', '醫生', '專家', '研究']
        
        # Add periods after likely sentence endings
        for ending in endings:
            pattern = f'({ending})([{"|".join(starters)}])'
            text = re.sub(pattern, r'\1。\2', text)
        
        # Add periods at natural breaks (long pauses)
        text = re.sub(r'([了的吧呢啊嗎])(\s*[我你他這那什怎為如因所但可醫專研])', r'\1。\2', text)
        
        # Add question marks for obvious questions
        text = re.sub(r'(什麼|怎麼|為什麼|嗎)([我你他這那])', r'\1？\2', text)
        
        # Add final period if text doesn't end with punctuation
        if text and not text[-1] in '。！？，':
            text += '。'
        
        return text
    
    async def _restore_punctuation_chunk_ai_backend(self, text_chunk: str) -> str:
        """Use AI backend for punctuation restoration"""
            
        prompt = f"""请为以下中文文本添加适当的标点符号。要求：
1. 不要改变任何汉字或词语
2. 只添加标点符号（句号、逗号、问号、感叹号等）
3. 保持原文的意思不变
4. 返回添加标点后的文本

原文：{text_chunk}

添加标点后的文本："""

        try:
            # Get the Chinese punctuation model from settings
            from config.settings import get_settings
            settings = get_settings()
            punctuation_model = getattr(settings, 'chinese_punctuation_model', 'sonnet')
            
            # Temporarily override the model for punctuation restoration
            original_model = self.ai_backend.model
            self.ai_backend.model = punctuation_model
            
            # Add timeout handling - 10 seconds per chunk
            import asyncio
            result = await asyncio.wait_for(
                self.ai_backend.generate_content(
                    prompt=prompt,
                    max_tokens=len(text_chunk) + 100,  # Allow some extra tokens for punctuation
                    task_type="punctuation_restoration"
                ),
                timeout=10.0  # 10 second timeout per chunk
            )
            
            # Restore original model
            self.ai_backend.model = original_model
            
            if result and result.get('content'):
                restored_text = result['content'].strip()
                # Remove any prefix like "添加标点后的文本：" if AI includes it
                if '：' in restored_text and restored_text.startswith(('添加标点后的文本', '结果', '答案')):
                    restored_text = restored_text.split('：', 1)[1].strip()
                return restored_text
            else:
                logger.warning(f"AI backend returned empty result for punctuation restoration")
                return text_chunk
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during punctuation restoration for chunk (10s exceeded), returning original")
            return text_chunk
        except Exception as e:
            logger.error(f"Error during punctuation restoration: {e}")
            return text_chunk
    
        
        # Get model from settings
        try:
            from config.settings import get_settings
            settings = get_settings()
            model = getattr(settings, 'chinese_punctuation_model', 'sonnet')
        except:
            model = 'sonnet'
        
        prompt = f"""请为以下中文文本添加适当的标点符号。要求：
1. 不要改变任何汉字或词语
2. 只添加标点符号（句号、逗号、问号、感叹号等）
3. 保持原文的意思不变
4. 只返回添加标点后的文本，不要有任何其他内容

原文：{text_chunk}"""
        
        try:
            # EMERGENCY FIX: Add rate limiting before Claude CLI calls
            
            # Run Claude CLI in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            cmd = ['claude', '--model', model, '--print']
            
            # Use run_in_executor to make subprocess call async-friendly
            result = await loop.run_in_executor(
                None, 
                lambda: subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            )
            
            if result.returncode == 0 and result.stdout:
                restored = result.stdout.strip()
                
                # Remove any prefix if AI includes it
                if '：' in restored and restored.startswith(('添加标点后', '结果', '答案')):
                    restored = restored.split('：', 1)[1].strip()
                
                # EMERGENCY FIX: Better success validation
                if len(restored) >= len(text_chunk) * 0.8 and restored != text_chunk:
                    return restored
                else:
                    logger.warning("Restored text invalid or unchanged, using original")
                    return text_chunk
            else:
                logger.warning(f"Claude CLI returned error: {result.stderr}")
                return text_chunk
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout ({self.timeout}s) during Claude CLI punctuation restoration")
            return text_chunk
        except Exception as e:
            logger.error(f"Error during Claude CLI punctuation restoration: {e}")
            return text_chunk
    
    async def restore_punctuation(self, text: str, srt_content: str = None) -> Tuple[str, bool]:
        """
        Restore punctuation to Chinese text with optional SRT-aware processing.
        
        Args:
            text: Input text without punctuation
            srt_content: Optional SRT content for boundary-aware processing
            
        Returns:
            Tuple[str, bool]: (Restored text, Success flag)
        """
        # Check if text is Chinese
        if not self.detect_chinese_text(text):
            logger.info("Text does not contain Chinese characters, skipping punctuation restoration")
            return text, False
        
        # Check if text already has punctuation
        if self.has_punctuation(text):
            logger.info("Text already contains punctuation, skipping restoration")
            return text, False
        
        # PRODUCTION SOLUTION: Enhanced mechanical punctuation only
        # No Claude CLI calls - pure mechanical approach for reliability and speed
        
        # Strategy 1: Use SRT-aware processing if available (intelligent boundaries)
        if srt_content and srt_content.strip():
            logger.info("Using mechanical SRT-aware punctuation restoration")
            try:
                restored_srt, success = await self.restore_srt_punctuation_aware(srt_content)
                if success:
                    # Extract text from restored SRT
                    segments = self.parse_srt_segments(restored_srt)
                    restored_text = ''.join([seg['text'] for seg in segments if seg.get('text')])
                    return restored_text, True
                else:
                    logger.warning("SRT-aware restoration failed, falling back to enhanced mechanical")
            except Exception as e:
                logger.error(f"SRT-aware restoration error: {e}, falling back to enhanced mechanical")
        
        # Strategy 2: Enhanced mechanical punctuation (always works)
        logger.info("Using enhanced mechanical punctuation restoration")
        
        # Apply enhanced mechanical rules to entire text
        restored_text = self._apply_enhanced_mechanical_punctuation(text)
        
        # Check if enhanced mechanical rules actually added punctuation
        if restored_text != text:
            logger.info("Enhanced mechanical punctuation restoration successful")
            return restored_text, True
        else:
            logger.warning("Enhanced mechanical punctuation restoration failed")
            return text, False
    
    def restore_punctuation_sync(self, text: str, srt_content: str = None) -> Tuple[str, bool]:
        """
        Synchronous wrapper for restore_punctuation.
        
        Args:
            text: Input text without punctuation
            srt_content: Optional SRT content for boundary-aware processing
            
        Returns:
            Tuple[str, bool]: (Restored text, Success flag)
        """
        import asyncio
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.restore_punctuation(text, srt_content))
                    return future.result()
            else:
                # Safe to run in current thread
                return loop.run_until_complete(self.restore_punctuation(text, srt_content))
        except RuntimeError:
            # No event loop, safe to create new one
            return asyncio.run(self.restore_punctuation(text, srt_content))
    
    def restore_srt_punctuation_sync(self, srt_content: str, use_srt_aware: bool = True) -> Tuple[str, bool]:
        """
        Synchronous wrapper for restore_srt_punctuation.
        
        Args:
            srt_content: SRT file content
            use_srt_aware: Use SRT-aware processing that leverages segment boundaries
            
        Returns:
            Tuple[str, bool]: (Restored content, Success flag)
        """
        import asyncio
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.restore_srt_punctuation(srt_content, use_srt_aware))
                    return future.result()
            else:
                # Safe to run in current thread
                return loop.run_until_complete(self.restore_srt_punctuation(srt_content, use_srt_aware))
        except RuntimeError:
            # No event loop, safe to create new one
            return asyncio.run(self.restore_srt_punctuation(srt_content, use_srt_aware))
    
    def create_backup(self, file_path: Path) -> Path:
        """
        Create a backup of the original file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path: Path to backup file
        """
        backup_path = file_path.with_suffix(f'.backup{file_path.suffix}')
        if file_path.exists():
            backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
            logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    async def restore_file_punctuation(self, file_path: Path, create_backup: bool = True) -> bool:
        """
        Restore punctuation in a text or SRT file.
        
        Args:
            file_path: Path to file to process
            create_backup: Whether to create backup before modification
            
        Returns:
            bool: True if file was processed successfully
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            # Read original content
            original_content = file_path.read_text(encoding='utf-8')
            
            # Create backup if requested
            if create_backup:
                self.create_backup(file_path)
            
            # Handle SRT files differently
            if file_path.suffix.lower() == '.srt':
                restored_content, success = await self.restore_srt_punctuation(original_content)
            else:
                restored_content, success = await self.restore_punctuation(original_content)
            
            if success:
                # Write restored content
                file_path.write_text(restored_content, encoding='utf-8')
                logger.info(f"Successfully restored punctuation in: {file_path}")
                return True
            else:
                logger.info(f"No punctuation restoration needed for: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    async def restore_srt_punctuation(self, srt_content: str, use_srt_aware: bool = True) -> Tuple[str, bool]:
        """
        Restore punctuation in SRT subtitle content.
        
        Args:
            srt_content: SRT file content
            use_srt_aware: Use SRT-aware processing that leverages segment boundaries
            
        Returns:
            Tuple[str, bool]: (Restored content, Success flag)
        """
        if use_srt_aware:
            # Use new SRT-aware method that uses segment boundaries as hints
            return await self.restore_srt_punctuation_aware(srt_content)
        else:
            # Legacy method - process line by line
            lines = srt_content.split('\n')
            restored_lines = []
            any_changes = False
            
            for line in lines:
                # Skip SRT metadata lines (numbers, timestamps, empty lines)
                if (line.strip().isdigit() or 
                    '-->' in line or 
                    not line.strip() or
                    not self.detect_chinese_text(line)):
                    restored_lines.append(line)
                    continue
                
                # Process subtitle text lines
                restored_line, success = await self.restore_punctuation(line)
                restored_lines.append(restored_line)
                if success:
                    any_changes = True
            
            return '\n'.join(restored_lines), any_changes
    
    def parse_srt_segments(self, srt_content: str) -> List[dict]:
        """
        Parse SRT content into segments with timing information.
        
        Args:
            srt_content: SRT file content
            
        Returns:
            List[dict]: List of segments with text and timing
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
    
    async def restore_srt_punctuation_aware(self, srt_content: str) -> Tuple[str, bool]:
        """
        PRODUCTION SOLUTION: Mechanical SRT-aware punctuation using natural boundaries
        NO AI CALLS - Uses segment boundaries to intelligently place punctuation
        
        Args:
            srt_content: SRT file content
            
        Returns:
            Tuple[str, bool]: (Restored content with original SRT structure, Success flag)
        """
        # Parse SRT into segments
        segments = self.parse_srt_segments(srt_content)
        
        if not segments:
            logger.warning("No segments found in SRT content")
            return srt_content, False
        
        logger.info(f"Processing {len(segments)} SRT segments with mechanical boundary analysis")
        
        if not segments:
            logger.warning("No segments found in SRT content")
            return srt_content, False
        
        # Check if any segments need punctuation
        chinese_segments = [seg for seg in segments if self.detect_chinese_text(seg['text'])]
        if not chinese_segments:
            logger.info("No Chinese segments found in SRT")
            return srt_content, False
        
        # Extract just the text from Chinese segments
        texts = [seg['text'] for seg in chinese_segments]
        
        # Check if already has punctuation
        combined_text = ' '.join(texts)
        if self.has_punctuation(combined_text):
            logger.info("SRT segments already contain punctuation")
            return srt_content, False
        
        # PRODUCTION SOLUTION: Mechanical boundary-aware punctuation processing
        # Process all segments using mechanical rules with boundary intelligence
        processed_segments = []
        
        for i, segment in enumerate(chinese_segments):
            segment_text = segment['text']
            
            # Apply enhanced mechanical punctuation with boundary context
            boundary_type = self._analyze_segment_boundary(
                segment_text, 
                chinese_segments[i-1]['text'] if i > 0 else None,
                chinese_segments[i+1]['text'] if i < len(chinese_segments)-1 else None
            )
            
            # Apply contextual punctuation based on boundary analysis
            punctuated_text = self._apply_boundary_contextual_punctuation(segment_text, boundary_type)
            segment['text'] = punctuated_text
            
            processed_segments.append(segment)
        
        # Rebuild SRT content with punctuated text
        return self.rebuild_srt_content(segments, processed_segments), True
    
    def _analyze_segment_boundary(self, current_text: str, prev_text: Optional[str], next_text: Optional[str]) -> str:
        """
        Analyze SRT segment boundary to determine appropriate punctuation type.
        Uses natural speech patterns without AI calls.
        
        Returns:
            str: Boundary type ('sentence_end', 'clause_break', 'continuation', 'question')
        """
        if not current_text:
            return 'continuation'
        
        current_lower = current_text.lower()
        
        # Strong sentence ending indicators
        strong_endings = ['了', '吧', '呢', '啊', '嗎']
        if any(current_lower.endswith(ending) for ending in strong_endings):
            # Check if next segment starts new thought
            if next_text and any(next_text.startswith(starter) for starter in ['我', '你', '他', '這', '那', '什麼', '醫生']):
                return 'sentence_end'
        
        # Question indicators
        question_words = ['什麼', '怎麼', '為什麼', '哪裡', '誰', '嗎', '呢']
        if any(word in current_lower for word in question_words):
            return 'question'
        
        # Continuation indicators (should flow to next segment)
        continuation_endings = ['的', '在', '是', '會', '要', '可以', '應該']
        if any(current_lower.endswith(ending) for ending in continuation_endings):
            return 'continuation'
        
        # Default to clause break (comma)
        return 'clause_break'
    
    def _apply_boundary_contextual_punctuation(self, text: str, boundary_type: str) -> str:
        """
        Apply punctuation based on boundary analysis.
        Different punctuation for different boundary types.
        """
        if not text:
            return text
        
        # Remove any existing punctuation first
        text = text.rstrip('。！？，')
        
        # Apply appropriate punctuation based on boundary type
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
    
    async def restore_punctuation_with_boundaries(self, text_segments: List[str]) -> List[str]:
        """
        Restore punctuation for text segments, using boundaries as hints.
        
        Args:
            text_segments: List of text segments from SRT
            
        Returns:
            List[str]: Segments with restored punctuation
        """
        if not text_segments:
            return []
        
        # Prepare prompt that preserves segment structure
        segments_str = '\n'.join([f"[{i+1}] {text}" for i, text in enumerate(text_segments)])
        
        prompt = f"""请为以下分段的中文文本添加适当的标点符号。

重要说明：
1. 每个[数字]标记代表一个语音段落的边界（说话时的自然停顿）
2. 请根据语法和语义判断每个边界处应该使用什么标点：
   - 如果是完整句子的结尾，使用句号（。）、问号（？）或感叹号（！）
   - 如果是句子内的停顿，使用逗号（，）
   - 如果段落应该与下一段连接，可以不加标点
3. 保持每个段落独立，按原始编号返回
4. 不要改变任何汉字或词语
5. 每行返回格式：[数字] 添加标点后的文本

原始分段文本：
{segments_str}

请返回添加标点后的分段文本："""
        
        try:
            # Use Claude CLI if available, otherwise use AI backend
                result = await self._restore_segments_claude_cli(prompt)
            elif self.ai_backend:
                result = await self._restore_segments_ai_backend(prompt, len(text_segments))
            else:
                logger.warning("No AI available for punctuation restoration")
                return text_segments
            
            # Parse the response to extract punctuated segments
            return self.parse_punctuated_segments(result, text_segments)
            
        except Exception as e:
            logger.error(f"Error in boundary-aware punctuation restoration: {e}")
            return text_segments
    
                
        except Exception as e:
            logger.error(f"Claude CLI segment restoration error: {e}")
            return ""
    
    async def _restore_segments_ai_backend(self, prompt: str, num_segments: int) -> str:
        """Call AI backend for segment punctuation."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            model = getattr(settings, 'chinese_punctuation_model', 'sonnet')
            
            original_model = self.ai_backend.model
            self.ai_backend.model = model
            
            result = await asyncio.wait_for(
                self.ai_backend.generate_content(
                    prompt=prompt,
                    max_tokens=num_segments * 100,  # Estimate tokens needed
                    task_type="punctuation_restoration"
                ),
                timeout=15.0
            )
            
            self.ai_backend.model = original_model
            
            if result and result.get('content'):
                return result['content'].strip()
            return ""
            
        except Exception as e:
            logger.error(f"AI backend segment restoration error: {e}")
            return ""
    
    def parse_punctuated_segments(self, ai_response: str, original_segments: List[str]) -> List[str]:
        """
        Parse AI response to extract punctuated segments.
        
        Args:
            ai_response: AI's response with punctuated segments
            original_segments: Original segments as fallback
            
        Returns:
            List[str]: Punctuated segments
        """
        if not ai_response:
            return original_segments
        
        punctuated = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for pattern like "[1] text" or "1. text" or just the text
            if line.startswith('[') and ']' in line:
                text = line.split(']', 1)[1].strip()
                punctuated.append(text)
            elif line[0].isdigit() and ('. ' in line or '、' in line or ' ' in line):
                # Handle "1. text" or "1、text" or "1 text"
                for separator in ['. ', '、', ' ']:
                    if separator in line:
                        text = line.split(separator, 1)[1].strip()
                        punctuated.append(text)
                        break
            elif len(punctuated) < len(original_segments):
                # If no number prefix, assume it's the next segment
                punctuated.append(line)
        
        # If we didn't get all segments, pad with originals
        while len(punctuated) < len(original_segments):
            punctuated.append(original_segments[len(punctuated)])
        
        return punctuated[:len(original_segments)]
    
    def rebuild_srt_content(self, all_segments: List[dict], punctuated_segments: List[dict]) -> str:
        """
        Rebuild SRT content with punctuated text.
        
        Args:
            all_segments: All original SRT segments
            punctuated_segments: Segments that were punctuated
            
        Returns:
            str: Complete SRT content with punctuation
        """
        # Create mapping of segment numbers to punctuated text
        punctuated_map = {seg['number']: seg['text'] for seg in punctuated_segments}
        
        # Build SRT content
        lines = []
        for seg in all_segments:
            lines.append(str(seg['number']))
            lines.append(f"{seg['start']} --> {seg['end']}")
            
            # Use punctuated text if available, otherwise original
            text = punctuated_map.get(seg['number'], seg['text'])
            lines.append(text)
            lines.append('')  # Empty line between segments
        
        return '\n'.join(lines)
    
    def _distribute_punctuation_to_srt(self, segments: List[dict], punctuated_text: str) -> str:
        """
        Distribute punctuation from text back to SRT segments for large transcript optimization.
        
        Args:
            segments: Original SRT segments
            punctuated_text: Text with punctuation added
            
        Returns:
            str: SRT content with punctuation distributed to segments
        """
        # Split punctuated text into sentences
        sentences = []
        current_sentence = ""
        
        for char in punctuated_text:
            current_sentence += char
            if char in '。！？':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text as last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Distribute sentences across segments
        sentence_idx = 0
        char_position = 0
        
        for seg in segments:
            if not seg.get('text') or sentence_idx >= len(sentences):
                continue
                
            seg_text = seg['text']
            seg_length = len(seg_text)
            
            # Find which sentence(s) this segment should contain
            while (sentence_idx < len(sentences) and 
                   char_position < len(punctuated_text) and
                   seg_text in punctuated_text[char_position:char_position + seg_length + 50]):
                
                # Check if this segment should end with punctuation
                sentence = sentences[sentence_idx]
                if sentence.endswith(('。', '！', '？')) and seg_text in sentence:
                    # Add punctuation to segment if it's the end of a sentence
                    for punct in '。！？':
                        if sentence.endswith(punct) and not seg_text.endswith(punct):
                            seg['text'] = seg_text + punct
                            break
                    sentence_idx += 1
                    break
                
                char_position += seg_length
                break
        
        # Rebuild SRT
        return self.rebuild_srt_content(segments, segments)
    
    def validate_punctuation_success(self, text: str, expected_minimum: int = 1) -> bool:
        """
        EMERGENCY FIX: Post-processing validation to verify punctuation was actually added
        
        Args:
            text: Text to validate
            expected_minimum: Minimum expected punctuation marks
            
        Returns:
            bool: True if text has adequate punctuation
        """
        if not text or not text.strip():
            return False
            
        # Count sentence-ending punctuation
        punct_count = sum(1 for c in text if c in '。！？')
        text_length = len(text.strip())
        
        # Calculate expected punctuation based on text length
        expected_for_length = max(expected_minimum, text_length // 200)  # 1 per 200 chars minimum
        
        success = punct_count >= expected_for_length
        
        if success:
            logger.info(f"✅ Punctuation validation PASSED: {punct_count} marks (≥{expected_for_length} expected)")
        else:
            logger.warning(f"❌ Punctuation validation FAILED: {punct_count} marks (<{expected_for_length} expected)")
            
        return success


async def restore_punctuation_for_file(file_path: str, ai_backend=None, use_srt_aware: bool = True) -> bool:
    """
    Convenience function to restore punctuation for a single file.
    
    Args:
        file_path: Path to file to process
        ai_backend: AI backend instance
        use_srt_aware: Use SRT-aware processing if SRT file available
        
    Returns:
        bool: True if processing was successful
    """
    restorer = ChinesePunctuationRestorer(ai_backend)
    file_path_obj = Path(file_path)
    
    # Check if there's a corresponding SRT file for SRT-aware processing
    if use_srt_aware and file_path_obj.suffix.lower() == '.txt':
        srt_path = file_path_obj.with_suffix('.srt')
        if srt_path.exists():
            logger.info(f"Found SRT file for SRT-aware processing: {srt_path.name}")
            try:
                srt_content = srt_path.read_text(encoding='utf-8')
                txt_content = file_path_obj.read_text(encoding='utf-8')
                
                # Use SRT-aware restoration
                restored_text, success = await restorer.restore_punctuation(txt_content, srt_content)
                
                if success:
                    # Create backup and update files
                    restorer.create_backup(file_path_obj)
                    file_path_obj.write_text(restored_text, encoding='utf-8')
                    
                    # Also update the SRT with punctuation
                    restored_srt, _ = await restorer.restore_srt_punctuation_aware(srt_content)
                    restorer.create_backup(srt_path)
                    srt_path.write_text(restored_srt, encoding='utf-8')
                    
                    logger.info(f"Successfully applied SRT-aware punctuation to {file_path_obj.name} and {srt_path.name}")
                    return True
                else:
                    logger.warning(f"SRT-aware punctuation failed, falling back to standard processing")
            except Exception as e:
                logger.error(f"SRT-aware processing error: {e}, falling back to standard processing")
    
    # Fall back to standard file processing
    return await restorer.restore_file_punctuation(file_path_obj)


async def restore_punctuation_for_directory(directory_path: str, pattern: str = "*.zh.txt", ai_backend=None, use_srt_aware: bool = True) -> List[str]:
    """
    Restore punctuation for all matching files in a directory.
    
    Args:
        directory_path: Path to directory to process
        pattern: File pattern to match (e.g., "*.zh.txt", "*.zh.srt")
        ai_backend: AI backend instance
        use_srt_aware: Use SRT-aware processing when available
        
    Returns:
        List[str]: List of successfully processed files
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    processed_files = []
    matching_files = list(directory.glob(pattern))
    
    logger.info(f"Found {len(matching_files)} files matching pattern '{pattern}' in {directory_path}")
    
    for file_path in matching_files:
        success = await restore_punctuation_for_file(str(file_path), ai_backend, use_srt_aware)
        if success:
            processed_files.append(str(file_path))
    
    logger.info(f"Successfully processed {len(processed_files)} files")
    return processed_files


# Synchronous convenience functions for backward compatibility
def restore_punctuation_for_file_sync(file_path: str, ai_backend=None, use_srt_aware: bool = True) -> bool:
    """
    Synchronous wrapper for restore_punctuation_for_file.
    
    Args:
        file_path: Path to file to process
        ai_backend: AI backend instance
        use_srt_aware: Use SRT-aware processing if SRT file available
        
    Returns:
        bool: True if processing was successful
    """
    import asyncio
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, restore_punctuation_for_file(file_path, ai_backend, use_srt_aware))
                return future.result()
        else:
            # Safe to run in current thread
            return loop.run_until_complete(restore_punctuation_for_file(file_path, ai_backend, use_srt_aware))
    except RuntimeError:
        # No event loop, safe to create new one
        return asyncio.run(restore_punctuation_for_file(file_path, ai_backend, use_srt_aware))


def restore_punctuation_for_directory_sync(directory_path: str, pattern: str = "*.zh.txt", ai_backend=None, use_srt_aware: bool = True) -> List[str]:
    """
    Synchronous wrapper for restore_punctuation_for_directory.
    
    Args:
        directory_path: Path to directory to process
        pattern: File pattern to match (e.g., "*.zh.txt", "*.zh.srt")
        ai_backend: AI backend instance
        use_srt_aware: Use SRT-aware processing when available
        
    Returns:
        List[str]: List of successfully processed files
    """
    import asyncio
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, restore_punctuation_for_directory(directory_path, pattern, ai_backend, use_srt_aware))
                return future.result()
        else:
            # Safe to run in current thread
            return loop.run_until_complete(restore_punctuation_for_directory(directory_path, pattern, ai_backend, use_srt_aware))
    except RuntimeError:
        # No event loop, safe to create new one
        return asyncio.run(restore_punctuation_for_directory(directory_path, pattern, ai_backend, use_srt_aware))