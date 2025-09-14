"""
Synchronous wrapper for Chinese punctuation restoration.
Provides a unified interface for both sync and async contexts.
"""

import subprocess
import logging
import re
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ChinesePunctuationSync:
    """Synchronous Chinese punctuation restoration using Claude CLI directly."""
    
    def __init__(self, max_chunk_size: int = 500, timeout: int = 30):
        """Initialize with configurable chunk size and timeout.
        
        Note: 500 chars balances between API call reduction and timeout risk.
        This reduces API calls by 60% compared to 200 char chunks.
        Timeout increased to 30s for long videos with complex content.
        """
        self.max_chunk_size = max_chunk_size
        self.timeout = timeout
        self.claude_available = None
        
        # Cost estimation (approximate)
        self.cost_per_1k_chars = 0.0003  # Claude Haiku cost estimate
        
    def check_claude_cli(self) -> bool:
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
    
    def detect_chinese_text(self, text: str) -> bool:
        """Detect if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
        return bool(chinese_pattern.search(text))
    
    def has_punctuation(self, text: str) -> bool:
        """
        Check if text has adequate sentence punctuation density.
        For Chinese text, uses density-based approach rather than just existence.
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
    
    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks for processing."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        remaining = text
        
        while remaining:
            if len(remaining) <= self.max_chunk_size:
                chunks.append(remaining.strip())
                break
            
            # Find a good split point
            split_pos = self.max_chunk_size
            
            # Try to split at sentence boundaries or spaces
            for i in range(self.max_chunk_size - 20, max(0, self.max_chunk_size - 50), -1):
                if i < len(remaining) and remaining[i] in ' \n。！？':
                    split_pos = i + 1
                    break
            
            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:]
        
        return chunks
    
    def restore_punctuation_chunk(self, text_chunk: str, model: str = 'sonnet') -> str:
        """
        Restore punctuation for a single text chunk using Claude CLI.
        Returns original text if restoration fails.
        """
        if not self.check_claude_cli():
            logger.warning("Claude CLI not available, skipping punctuation restoration")
            return text_chunk
        
        prompt = f"""请为以下中文文本添加适当的标点符号。要求：
1. 不要改变任何汉字或词语
2. 只添加标点符号（句号、逗号、问号、感叹号等）
3. 保持原文的意思不变
4. 只返回添加标点后的文本，不要有任何其他内容

原文：{text_chunk}"""
        
        try:
            cmd = ['claude', '--model', model, '--print']
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0 and result.stdout:
                restored = result.stdout.strip()
                
                # Remove any prefix if AI includes it
                if '：' in restored and restored.startswith(('添加标点后', '结果', '答案')):
                    restored = restored.split('：', 1)[1].strip()
                
                # Sanity check
                if len(restored) >= len(text_chunk) * 0.8:
                    return restored
                else:
                    logger.warning("Restored text too short, using original")
                    return text_chunk
            else:
                logger.warning(f"Claude CLI returned error: {result.stderr}")
                return text_chunk
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout ({self.timeout}s) during punctuation restoration")
            return text_chunk
        except Exception as e:
            logger.error(f"Error during punctuation restoration: {e}")
            return text_chunk
    
    def estimate_cost(self, text: str) -> Tuple[float, int]:
        """
        Estimate the cost and number of API calls for punctuation restoration.
        
        Returns:
            Tuple of (estimated_cost_usd, num_api_calls)
        """
        chunks = self.chunk_text(text)
        num_calls = len(chunks)
        total_chars = len(text)
        estimated_cost = (total_chars / 1000) * self.cost_per_1k_chars * 2  # x2 for prompt overhead
        return estimated_cost, num_calls
    
    def restore_punctuation(self, text: str, model: Optional[str] = None, show_progress: bool = False) -> Tuple[str, bool]:
        """
        Main entry point for synchronous punctuation restoration.
        
        Returns:
            Tuple of (restored_text, success_flag)
        """
        # Get model from settings if not provided
        if model is None:
            try:
                from config.settings import get_settings
                settings = get_settings()
                model = getattr(settings, 'chinese_punctuation_model', 'sonnet')
            except:
                model = 'sonnet'
        
        # Check if enabled
        try:
            from config.settings import get_settings
            settings = get_settings()
            if not getattr(settings, 'chinese_punctuation_enabled', False):
                logger.info("Chinese punctuation restoration is disabled")
                return text, False
        except:
            pass
        
        # Check if text needs punctuation
        if not self.detect_chinese_text(text):
            logger.info("Text does not contain Chinese characters")
            return text, False
        
        if self.has_punctuation(text):
            logger.info("Text already has punctuation")
            return text, False
        
        # Process in chunks
        chunks = self.chunk_text(text)
        restored_chunks = []
        successful_chunks = 0
        total_chunks = len(chunks)
        
        # Show cost warning if significant
        if show_progress and total_chunks > 10:
            cost, _ = self.estimate_cost(text)
            if cost > 0.01:  # More than 1 cent
                logger.warning(f"⚠️  Estimated cost: ${cost:.4f} USD for {total_chunks} API calls")
        
        for i, chunk in enumerate(chunks):
            if show_progress:
                progress_pct = (i / total_chunks) * 100 if total_chunks > 0 else 0
                logger.info(f"[{progress_pct:.1f}%] Processing chunk {i+1}/{total_chunks} ({len(chunk)} chars)")
            else:
                logger.info(f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} chars)")
            
            restored = self.restore_punctuation_chunk(chunk, model)
            
            if restored != chunk:
                successful_chunks += 1
            
            restored_chunks.append(restored)
        
        # Calculate success
        success_rate = successful_chunks / len(chunks) if chunks else 0
        restored_text = ''.join(restored_chunks)
        
        if success_rate == 0:
            logger.warning("No chunks were successfully punctuated")
            return text, False
        elif success_rate < 0.5:
            logger.warning(f"Partial success: {successful_chunks}/{len(chunks)} chunks")
            return restored_text, True
        else:
            logger.info(f"Success: {successful_chunks}/{len(chunks)} chunks punctuated")
            return restored_text, True
    
    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of the original file."""
        backup_path = file_path.with_suffix(f'.backup{file_path.suffix}')
        if file_path.exists():
            backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
            logger.info(f"Created backup: {backup_path}")
        return backup_path


# Global instance for easy access
_sync_restorer = None

def get_sync_restorer() -> ChinesePunctuationSync:
    """Get or create the global synchronous restorer instance."""
    global _sync_restorer
    if _sync_restorer is None:
        _sync_restorer = ChinesePunctuationSync()
    return _sync_restorer