"""
AI Backend system for content generation and analysis.

Supports multiple providers (Claude CLI, Claude API, OpenAI, Gemini) with
fallback chains and cost tracking.
"""

import logging
import subprocess
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from config.settings import get_settings

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers."""
    CLAUDE_CLI = "claude_cli"
    CLAUDE_API = "claude_api"
    OPENAI_API = "openai_api"
    GEMINI_API = "gemini_api"
    DISABLED = "disabled"


@dataclass
class AIResponse:
    """Response from AI provider."""
    content: str
    usage_tokens: Optional[int] = None
    cost: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    error: Optional[str] = None


class AIBackend:
    """AI backend for content generation and analysis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.provider = AIProvider(self.settings.ai_backend)
        
    def generate_content(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content using the configured AI provider.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0-1)
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with content and metadata
        """
        if self.provider == AIProvider.DISABLED:
            return {"error": "AI backend is disabled"}
        
        max_tokens = max_tokens or self.settings.ai_max_tokens
        
        try:
            if self.provider == AIProvider.CLAUDE_CLI:
                return self._generate_claude_cli(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == AIProvider.CLAUDE_API:
                return self._generate_claude_api(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == AIProvider.OPENAI_API:
                return self._generate_openai_api(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == AIProvider.GEMINI_API:
                return self._generate_gemini_api(prompt, max_tokens, temperature, system_prompt)
            else:
                return {"error": f"Unsupported provider: {self.provider}"}
                
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_claude_cli(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate content using Claude CLI."""
        try:
            # Combine system prompt and user prompt if needed
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Try the simpler command format first
            cmd = ["claude", full_prompt]
            
            # Run Claude CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                content = result.stdout.strip()
                if content:
                    return {
                        "content": content,
                        "model": self.settings.ai_model or "claude",
                        "provider": "claude_cli",
                        "usage_tokens": self._estimate_tokens(prompt + content),
                        "cost": self._estimate_cost(prompt, content)
                    }
                else:
                    # Empty response, fallback to placeholder
                    return self._generate_placeholder(prompt, max_tokens)
            else:
                error_msg = result.stderr.strip() or "Claude CLI execution failed"
                logger.warning(f"Claude CLI error: {error_msg}, falling back to placeholder")
                # Fallback to placeholder content
                return self._generate_placeholder(prompt, max_tokens)
                
        except subprocess.TimeoutExpired:
            logger.warning("Claude CLI timed out, falling back to placeholder")
            return self._generate_placeholder(prompt, max_tokens)
        except FileNotFoundError:
            logger.warning("Claude CLI not found, falling back to placeholder")
            return self._generate_placeholder(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Claude CLI execution error: {e}, falling back to placeholder")
            return self._generate_placeholder(prompt, max_tokens)
    
    def _generate_claude_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate content using Claude API (placeholder for Phase 2)."""
        return {"error": "Claude API not yet implemented - use claude_cli for Phase 1"}
    
    def _generate_openai_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate content using OpenAI API (placeholder for Phase 2)."""
        return {"error": "OpenAI API not yet implemented - use claude_cli for Phase 1"}
    
    def _generate_gemini_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Generate content using Gemini API (placeholder for Phase 2)."""
        return {"error": "Gemini API not yet implemented - use claude_cli for Phase 1"}
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _estimate_cost(self, input_text: str, output_text: str) -> float:
        """Estimate cost based on token usage and model pricing."""
        try:
            input_tokens = self._estimate_tokens(input_text)
            output_tokens = self._estimate_tokens(output_text)
            
            # Claude Haiku pricing (rough estimates)
            input_cost = (input_tokens / 1_000_000) * 0.25  # $0.25/1M input tokens
            output_cost = (output_tokens / 1_000_000) * 1.25  # $1.25/1M output tokens
            
            return round(input_cost + output_cost, 6)
            
        except Exception:
            return 0.0
    
    def _generate_placeholder(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Generate placeholder content when AI is unavailable."""
        # Extract key information from prompt for better placeholder
        prompt_lower = prompt.lower()
        
        if "summary" in prompt_lower or "summarize" in prompt_lower:
            content = "[AI Summary Placeholder] This video discusses important topics and provides valuable insights on the subject matter. The content covers key points and offers practical information for viewers."
        elif "blog" in prompt_lower:
            content = "[Blog Post Placeholder]\n\n## Introduction\nThis content explores fascinating topics discussed in the video.\n\n## Main Points\nThe video covers several important aspects worth considering.\n\n## Conclusion\nThe insights shared provide valuable takeaways for the audience."
        elif "social" in prompt_lower or "tweet" in prompt_lower:
            content = "[Social Media Placeholder] 🎥 Check out this amazing video content! Key insights and valuable information shared. #content #video"
        elif "newsletter" in prompt_lower:
            content = "[Newsletter Placeholder]\n\nDear Subscribers,\n\nThis week's featured content provides excellent insights on important topics.\n\nBest regards,\nThe Team"
        elif "script" in prompt_lower:
            content = "[Script Placeholder]\n\n[INTRO]\nWelcome to today's content!\n\n[MAIN CONTENT]\nLet's dive into the key topics...\n\n[OUTRO]\nThanks for watching!"
        else:
            content = f"[Placeholder Content] Generated placeholder for: {prompt[:100]}..."
        
        return {
            "content": content,
            "model": "placeholder",
            "provider": "placeholder",
            "usage_tokens": 0,
            "cost": 0.0,
            "is_placeholder": True
        }
    
    def is_available(self) -> bool:
        """Check if the AI backend is available and configured."""
        if self.provider == AIProvider.DISABLED:
            return False
        
        if self.provider == AIProvider.CLAUDE_CLI:
            try:
                result = subprocess.run(
                    ["claude", "--version"],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            except Exception:
                return False
        
        # For API providers, we'd check API keys and connectivity
        return False


# Global AI backend instance
_ai_backend = None


def get_ai_backend() -> AIBackend:
    """Get the global AI backend instance."""
    global _ai_backend
    if _ai_backend is None:
        _ai_backend = AIBackend()
    return _ai_backend


def generate_summary(
    transcript: str,
    duration_seconds: int,
    title: str = "",
    channel_name: str = ""
) -> Dict[str, Any]:
    """
    Generate a video summary based on transcript and duration.
    
    Args:
        transcript: Video transcript text
        duration_seconds: Video duration in seconds
        title: Video title
        channel_name: Channel name
        
    Returns:
        Dictionary with summary and metadata
    """
    backend = get_ai_backend()
    
    # Determine summary length based on duration
    if duration_seconds < 60:  # Shorts
        sentence_count = 1
        max_tokens = 50
    elif duration_seconds < 600:  # Medium (< 10 min)
        sentence_count = 3
        max_tokens = 150
    else:  # Long videos
        sentence_count = 5
        max_tokens = 250
    
    # Create summary prompt
    system_prompt = """You are a helpful assistant that creates concise, informative summaries of video content. 
Focus on the main topics, key insights, and actionable information."""
    
    prompt = f"""Please create a concise summary of this video transcript in exactly {sentence_count} sentence(s).
Focus on the main topics, key points, and actionable insights.

Title: {title}
Channel: {channel_name}
Duration: {duration_seconds // 60}:{duration_seconds % 60:02d}

Transcript:
{transcript[:4000]}  # Limit to avoid token overflow

Summary ({sentence_count} sentence(s)):"""

    return backend.generate_content(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for consistent summaries
        system_prompt=system_prompt
    )


def extract_topics(summary: str) -> List[str]:
    """
    Extract key topics from a video summary.
    
    Args:
        summary: Video summary text
        
    Returns:
        List of extracted topics
    """
    backend = get_ai_backend()
    
    prompt = f"""Extract 3-5 key topics from this video summary. Return only the topics, one per line, no bullets or numbers.
Focus on the main subjects, themes, or categories discussed.

Summary: {summary}

Topics:"""

    response = backend.generate_content(
        prompt=prompt,
        max_tokens=100,
        temperature=0.2
    )
    
    if response.get('content'):
        topics = [
            topic.strip() 
            for topic in response['content'].split('\n') 
            if topic.strip() and not topic.strip().startswith(('•', '-', '*'))
        ]
        return topics[:5]  # Limit to 5 topics max
    else:
        return []