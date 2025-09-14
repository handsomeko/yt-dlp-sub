"""
AI Backend Manager - Centralized AI interface for all workers
Handles routing to different AI providers based on configuration
Phase 1: Claude CLI
Phase 2: Claude API, OpenAI API, Gemini API
Phase 3: Multiple APIs with load balancing
"""

import json
import logging
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime

from config.settings import get_settings
from core.service_credentials import ClaudeCredentials, OpenAICredentials, GeminiCredentials
from core.transcript_quality_manager import get_transcript_quality_manager
from core.content_quality_manager import get_content_quality_manager
from core.ab_testing import get_ab_test_manager
from core.ai_provider_ab_testing import get_provider_ab_manager, ProviderExperimentType, ProviderSelectionStrategy
from core.ai_provider_config import get_ai_config_manager

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    CLAUDE_CLI = "claude_cli"
    CLAUDE_API = "claude_api" 
    OPENAI_API = "openai_api"
    GEMINI_API = "gemini_api"
    DISABLED = "disabled"  # For testing or when AI not available


class AIBackend:
    """
    Centralized AI backend for quality checking and content generation
    Routes requests to appropriate AI provider based on configuration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.backend = self.settings.ai_backend if hasattr(self.settings, 'ai_backend') else 'disabled'
        self.model = self.settings.ai_model if hasattr(self.settings, 'ai_model') else 'claude-3-haiku-20240307'
        self.max_tokens = self.settings.ai_max_tokens if hasattr(self.settings, 'ai_max_tokens') else 1000
        
        # Initialize provider A/B testing and configuration
        self.provider_ab_manager = get_provider_ab_manager()
        self.config_manager = get_ai_config_manager()
        
        # Provider selection strategy
        self.selection_strategy = ProviderSelectionStrategy.WEIGHTED_RANDOM
        if hasattr(self.settings, 'provider_selection_strategy'):
            self.selection_strategy = ProviderSelectionStrategy[self.settings.provider_selection_strategy.upper()]
        
        # Track usage for cost management
        self.usage_stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        
        logger.info(f"AI Backend initialized with default provider: {self.backend}, model: {self.model}")
    
    async def generate_content(self, 
                             prompt: str,
                             max_tokens: Optional[int] = None,
                             task_type: str = "general") -> Dict[str, Any]:
        """
        General content generation method
        
        Args:
            prompt: The prompt for content generation
            max_tokens: Maximum tokens to generate (optional)
            task_type: Type of task for tracking/analytics
            
        Returns:
            Dict with generated content and metadata
        """
        try:
            # Use provided max_tokens or fall back to instance default
            tokens_to_use = max_tokens or self.max_tokens
            original_max_tokens = self.max_tokens
            
            # Temporarily set max_tokens if different
            if max_tokens:
                self.max_tokens = max_tokens
            
            # Route to appropriate provider
            if self.backend == AIProvider.CLAUDE_CLI.value:
                result = self._claude_cli_evaluate(prompt)
            elif self.backend == AIProvider.CLAUDE_API.value:
                result = self._claude_api_evaluate(prompt) 
            elif self.backend == AIProvider.OPENAI_API.value:
                result = self._openai_api_evaluate(prompt)
            elif self.backend == AIProvider.GEMINI_API.value:
                result = self._gemini_api_evaluate(prompt)
            else:
                result = self._mock_evaluate("content_generation")
            
            # Restore original max_tokens
            self.max_tokens = original_max_tokens
            
            return {
                'content': result,
                'task_type': task_type,
                'provider': self.backend,
                'model': self.model,
                'max_tokens': tokens_to_use,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            return {
                'content': '',
                'error': str(e),
                'task_type': task_type,
                'provider': self.backend,
                'success': False
            }
    
    async def translate_content(self, 
                              content: str, 
                              target_language: str = 'en',
                              source_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate content to target language using configured AI provider
        
        Args:
            content: Text content to translate
            target_language: Target language code (default: 'en')
            source_language: Optional source language (auto-detect if not provided)
        
        Returns:
            Dict with 'success', 'translated_content', and 'error' keys
        """
        if self.backend == 'disabled':
            return {
                'success': False,
                'translated_content': None,
                'error': 'AI backend is disabled'
            }
        
        # Create translation prompt
        source_text = f"from {source_language} " if source_language else ""
        prompt = f"""Translate the following content {source_text}to {target_language}.
Keep the exact formatting intact. Only translate the text content.

Content:
{content[:3000]}"""  # Limit to avoid token limits

        try:
            # Use existing evaluation methods for translation
            if self.backend == 'claude_cli':
                response = self._claude_cli_evaluate(prompt)
            elif self.backend == 'claude_api':
                response = self._claude_api_evaluate(prompt)
            elif self.backend == 'openai_api':
                response = self._openai_api_evaluate(prompt)
            elif self.backend == 'gemini_api':
                response = self._gemini_api_evaluate(prompt)
            else:
                return {
                    'success': False,
                    'translated_content': None,
                    'error': f'Unknown AI backend: {self.backend}'
                }
            
            return {
                'success': True,
                'translated_content': response,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                'success': False,
                'translated_content': None,
                'error': str(e)
            }
    
    async def evaluate_transcript(
        self, 
        transcript: str,
        video_duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate transcript quality using AI with prompt management
        
        Args:
            transcript: The transcript text to evaluate
            video_duration: Optional video duration in seconds
            metadata: Optional additional context
        
        Returns:
            Dict with score, pass/fail, issues, and recommendations
        """
        # Check if quality checks are enabled
        if not self._should_check_quality():
            return self._skip_quality_check("Transcript quality checks disabled")
        
        # Get transcript quality manager
        quality_manager = get_transcript_quality_manager()
        
        # Prepare the evaluation prompt using TranscriptQualityManager
        prompt, prompt_metadata = await quality_manager.get_evaluation_prompt(
            transcript=transcript,
            video_duration=video_duration,
            extraction_method=metadata.get('extraction_method') if metadata else None,
            strictness=metadata.get('strictness', 'standard') if metadata else 'standard'
        )
        
        # Track experiment if applicable
        start_time = datetime.utcnow()
        
        # Select provider using A/B testing
        provider, provider_config = await self.provider_ab_manager.select_provider(
            task_type=ProviderExperimentType.TRANSCRIPT_QUALITY,
            strategy=self.selection_strategy,
            content_metadata=metadata
        )
        
        # Call appropriate backend
        tokens_used = 0
        try:
            if provider == 'claude_cli':
                result = self._claude_cli_evaluate(prompt)
                tokens_used = len(prompt.split()) + len(result.split())  # Approximate
            elif provider == 'claude_api':
                result, tokens_used = await self._claude_api_evaluate_with_tracking(prompt)
            elif provider == 'openai_api':
                result, tokens_used = await self._openai_api_evaluate_with_tracking(prompt)
            elif provider == 'gemini_api':
                result, tokens_used = await self._gemini_api_evaluate_with_tracking(prompt)
            else:
                result = self._mock_evaluate("transcript")
                tokens_used = 100  # Mock token count
            
            # Parse and validate result
            parsed_result = self._parse_evaluation_result(result, "transcript")
            
            # Track provider performance
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self.provider_ab_manager.record_result(
                task_type=ProviderExperimentType.TRANSCRIPT_QUALITY,
                provider=provider,
                success=True,
                execution_time=execution_time,
                tokens_used=tokens_used,
                quality_score=parsed_result.get('score'),
                metadata={'prompt_version': prompt_metadata.get('version')}
            )
            
            # Track analytics for experiments
            await self._track_quality_analytics(
                prompt_metadata, 
                parsed_result,
                execution_time
            )
            
            # Add provider info to result
            parsed_result['provider_used'] = provider
            parsed_result['tokens_used'] = tokens_used
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"AI evaluation failed with provider {provider}: {str(e)}")
            
            # Track provider failure
            await self.provider_ab_manager.record_result(
                task_type=ProviderExperimentType.TRANSCRIPT_QUALITY,
                provider=provider,
                success=False,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                error_type=type(e).__name__
            )
            
            # Track failed experiment
            if prompt_metadata.get('is_experiment'):
                await self._track_quality_analytics(
                    prompt_metadata,
                    {'error': True},
                    (datetime.utcnow() - start_time).total_seconds()
                )
            
            return self._error_result(str(e))
    
    async def evaluate_content(
        self,
        content: str,
        content_type: str,
        source_transcript: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated content quality using AI with prompt management
        
        Args:
            content: The content to evaluate
            content_type: Type of content (blog, social, newsletter, etc.)
            source_transcript: Optional source transcript for relevance check
            metadata: Optional additional context
        
        Returns:
            Dict with score, pass/fail, and improvements
        """
        # Check if quality checks are enabled
        if not self._should_check_quality():
            return self._skip_quality_check("Content quality checks disabled")
        
        # Get content quality manager
        quality_manager = get_content_quality_manager()
        
        # Prepare the evaluation prompt using ContentQualityManager
        prompt, prompt_metadata = await quality_manager.get_evaluation_prompt(
            content=content,
            content_type=content_type,
            platform=metadata.get('platform') if metadata else None,
            target_audience=metadata.get('target_audience') if metadata else None,
            requirements=metadata.get('requirements') if metadata else None,
            source_transcript=source_transcript,
            video_title=metadata.get('video_title') if metadata else None,
            word_limit=metadata.get('word_limit') if metadata else None,
            tone=metadata.get('tone') if metadata else None
        )
        
        # Track experiment if applicable
        start_time = datetime.utcnow()
        
        # Select provider using A/B testing
        provider, provider_config = await self.provider_ab_manager.select_provider(
            task_type=ProviderExperimentType.CONTENT_QUALITY,
            strategy=self.selection_strategy,
            content_metadata={'content_type': content_type, **metadata} if metadata else {'content_type': content_type}
        )
        
        # Call appropriate backend
        tokens_used = 0
        try:
            if provider == 'claude_cli':
                result = self._claude_cli_evaluate(prompt)
                tokens_used = len(prompt.split()) + len(result.split())  # Approximate
            elif provider == 'claude_api':
                result, tokens_used = await self._claude_api_evaluate_with_tracking(prompt)
            elif provider == 'openai_api':
                result, tokens_used = await self._openai_api_evaluate_with_tracking(prompt)
            elif provider == 'gemini_api':
                result, tokens_used = await self._gemini_api_evaluate_with_tracking(prompt)
            else:
                result = self._mock_evaluate("content")
                tokens_used = 100  # Mock token count
            
            # Parse and validate result
            parsed_result = self._parse_evaluation_result(result, "content")
            
            # Track provider performance
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self.provider_ab_manager.record_result(
                task_type=ProviderExperimentType.CONTENT_QUALITY,
                provider=provider,
                success=True,
                execution_time=execution_time,
                tokens_used=tokens_used,
                quality_score=parsed_result.get('score'),
                metadata={'content_type': content_type, 'prompt_version': prompt_metadata.get('version')}
            )
            
            # Track analytics for experiments
            await self._track_quality_analytics(
                prompt_metadata,
                parsed_result,
                execution_time
            )
            
            # Add provider info to result
            parsed_result['provider_used'] = provider
            parsed_result['tokens_used'] = tokens_used
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"AI evaluation failed with provider {provider}: {str(e)}")
            
            # Track provider failure
            await self.provider_ab_manager.record_result(
                task_type=ProviderExperimentType.CONTENT_QUALITY,
                provider=provider,
                success=False,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                error_type=type(e).__name__
            )
            
            # Track failed experiment
            if prompt_metadata.get('is_experiment'):
                await self._track_quality_analytics(
                    prompt_metadata,
                    {'error': True},
                    (datetime.utcnow() - start_time).total_seconds()
                )
            
            return self._error_result(str(e))
    
    async def _track_quality_analytics(
        self,
        prompt_metadata: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: float
    ) -> None:
        """
        Track analytics for quality check experiments.
        
        Args:
            prompt_metadata: Metadata from prompt selection
            result: Evaluation result
            execution_time: Time taken for evaluation
        """
        if not prompt_metadata.get('is_experiment'):
            return
        
        ab_test_manager = get_ab_test_manager()
        
        # Extract quality score from result
        quality_score = result.get('score')
        if quality_score is None and 'pass' in result:
            quality_score = 100 if result['pass'] else 50
        
        await ab_test_manager.record_result(
            experiment_id=prompt_metadata['experiment_id'],
            variant_id=prompt_metadata['variant_id'],
            metrics={
                'quality_score': quality_score,
                'tokens_used': len(str(result).split()),  # Approximate
                'execution_time': execution_time,
                'model_used': self.model,
                'error_occurred': result.get('error', False)
            }
        )
    
    def _should_check_quality(self) -> bool:
        """Check if quality checks are enabled and should run"""
        if self.backend == AIProvider.DISABLED.value:
            return False
        
        # Check if quality checks are enabled
        if hasattr(self.settings, 'quality_checks_enabled'):
            if not self.settings.quality_checks_enabled:
                return False
        
        # Check sample rate
        if hasattr(self.settings, 'quality_check_sample_rate'):
            import random
            if random.random() > self.settings.quality_check_sample_rate:
                logger.info("Skipping quality check based on sample rate")
                return False
        
        return True
    
    
    def _claude_cli_evaluate(self, prompt: str) -> str:
        """Evaluate using Claude CLI (Phase 1)"""
        try:
            # Build command (Claude CLI doesn't have --max-tokens option)
            cmd = [
                'claude',
                '--print',  # Non-interactive output
                '--model', self.model,
                prompt
            ]
            
            logger.info(f"Calling Claude CLI with model {self.model}")
            
            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Claude CLI error: {result.stderr}")
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise Exception("Claude CLI timeout")
        except FileNotFoundError:
            raise Exception("Claude CLI not found. Please install: pip install claude-cli")
        except Exception as e:
            raise Exception(f"Claude CLI error: {str(e)}")
    
    def _claude_api_evaluate(self, prompt: str) -> str:
        """Evaluate using Claude API (Phase 2)"""
        try:
            import anthropic
            
            # Get credentials from vault
            claude_creds = ClaudeCredentials()
            api_key = claude_creds.api_key
            
            if not api_key:
                raise Exception("Claude API key not configured in credential vault")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=claude_creds.model or self.model,
                max_tokens=claude_creds.max_tokens or self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            if response.usage:
                self.usage_stats['total_tokens'] += response.usage.total_tokens
            
            return response.content[0].text
            
        except ImportError:
            raise Exception("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    def _openai_api_evaluate(self, prompt: str) -> str:
        """Evaluate using OpenAI API (Phase 2)"""
        try:
            import openai
            
            # Get credentials from vault
            openai_creds = OpenAICredentials()
            api_key = openai_creds.api_key
            
            if not api_key:
                raise Exception("OpenAI API key not configured in credential vault")
            
            client = openai.OpenAI(
                api_key=api_key,
                organization=openai_creds.organization
            )
            
            response = client.chat.completions.create(
                model=openai_creds.model or self.model,
                max_tokens=openai_creds.max_tokens or self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            if response.usage:
                self.usage_stats['total_tokens'] += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except ImportError:
            raise Exception("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _gemini_api_evaluate(self, prompt: str) -> str:
        """Evaluate using Gemini API (Phase 2)"""
        try:
            import google.generativeai as genai
            
            # Get credentials from vault
            gemini_creds = GeminiCredentials()
            api_key = gemini_creds.api_key
            
            if not api_key:
                raise Exception("Gemini API key not configured in credential vault")
            
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(gemini_creds.model or 'gemini-pro')
            
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=gemini_creds.max_tokens or self.max_tokens
                )
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            
            return response.text
            
        except ImportError:
            raise Exception("Google Generative AI library not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _mock_evaluate(self, eval_type: str) -> str:
        """Mock evaluation for testing when AI is disabled"""
        if eval_type == "transcript":
            return json.dumps({
                "score": 85,
                "pass": True,
                "issues": ["Mock evaluation - AI disabled"],
                "recommendations": ["Enable AI backend for real quality checks"]
            })
        else:
            return json.dumps({
                "score": 80,
                "pass": True,
                "format_correct": True,
                "length_appropriate": True,
                "improvements": ["Mock evaluation - AI disabled"]
            })
    
    async def _claude_api_evaluate_with_tracking(self, prompt: str) -> tuple[str, int]:
        """Evaluate using Claude API with token tracking."""
        try:
            import anthropic
            
            # Get credentials from vault
            claude_creds = ClaudeCredentials()
            api_key = claude_creds.api_key
            
            if not api_key:
                raise Exception("Claude API key not configured in credential vault")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=claude_creds.model or self.model,
                max_tokens=claude_creds.max_tokens or self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.total_tokens
                self.usage_stats['total_tokens'] += tokens_used
            
            return response.content[0].text, tokens_used
            
        except ImportError:
            raise Exception("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")
    
    async def _openai_api_evaluate_with_tracking(self, prompt: str) -> tuple[str, int]:
        """Evaluate using OpenAI API with token tracking."""
        try:
            import openai
            
            # Get credentials from vault
            openai_creds = OpenAICredentials()
            api_key = openai_creds.api_key
            
            if not api_key:
                raise Exception("OpenAI API key not configured in credential vault")
            
            client = openai.OpenAI(
                api_key=api_key,
                organization=openai_creds.organization
            )
            
            response = client.chat.completions.create(
                model=openai_creds.model or self.model,
                max_tokens=openai_creds.max_tokens or self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.total_tokens
                self.usage_stats['total_tokens'] += tokens_used
            
            return response.choices[0].message.content, tokens_used
            
        except ImportError:
            raise Exception("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _gemini_api_evaluate_with_tracking(self, prompt: str) -> tuple[str, int]:
        """Evaluate using Gemini API with token tracking."""
        try:
            import google.generativeai as genai
            
            # Get credentials from vault
            gemini_creds = GeminiCredentials()
            api_key = gemini_creds.api_key
            
            if not api_key:
                raise Exception("Gemini API key not configured in credential vault")
            
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(gemini_creds.model or 'gemini-pro')
            
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=gemini_creds.max_tokens or self.max_tokens
                )
            )
            
            # Track usage
            self.usage_stats['total_calls'] += 1
            # Gemini doesn't provide token counts directly, so we approximate
            tokens_used = len(prompt.split()) + len(response.text.split())
            self.usage_stats['total_tokens'] += tokens_used
            
            return response.text, tokens_used
            
        except ImportError:
            raise Exception("Google Generative AI library not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _parse_evaluation_result(self, result: str, eval_type: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            # Handle case where AI might include markdown or extra text
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if eval_type == "transcript":
                    required = ['score', 'pass', 'issues', 'recommendations']
                else:
                    required = ['score', 'pass', 'improvements']
                
                for field in required:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")
                
                # Add metadata
                parsed['evaluation_timestamp'] = datetime.utcnow().isoformat()
                parsed['ai_backend'] = self.backend
                parsed['ai_model'] = self.model
                
                return parsed
            else:
                raise ValueError("No JSON found in AI response")
                
        except Exception as e:
            logger.error(f"Failed to parse AI response: {str(e)}")
            logger.debug(f"Raw response: {result}")
            
            # Return a default structure with the raw response
            return {
                'score': 50,
                'pass': False,
                'issues': ['Failed to parse AI response'],
                'recommendations': ['Manual review required'],
                'raw_response': result,
                'parse_error': str(e)
            }
    
    def _skip_quality_check(self, reason: str) -> Dict[str, Any]:
        """Return result when quality check is skipped"""
        return {
            'score': None,
            'pass': True,  # Don't block workflow
            'skipped': True,
            'reason': reason,
            'issues': [],
            'recommendations': []
        }
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return result when AI evaluation fails"""
        return {
            'score': None,
            'pass': True,  # Don't block workflow on AI failure
            'error': True,
            'error_message': error,
            'issues': ['AI evaluation failed'],
            'recommendations': ['Manual review required']
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get AI usage statistics for cost tracking"""
        return self.usage_stats.copy()
    
    async def restore_punctuation(self, text: str, language: str = 'zh') -> Tuple[str, bool]:
        """
        Restore punctuation to text using the appropriate punctuation system.
        
        Args:
            text: Input text without punctuation
            language: Language of the text ('zh' for Chinese, 'en' for English)
            
        Returns:
            Tuple[str, bool]: (Restored text, Success flag)
        """
        if language == 'zh':
            from core.chinese_punctuation import ChinesePunctuationRestorer
            restorer = ChinesePunctuationRestorer(ai_backend=self)
            return await restorer.restore_punctuation(text)
        else:
            # For now, only Chinese is supported
            self.logger.info(f"Punctuation restoration not implemented for language: {language}")
            return text, False


# Singleton instance for reuse
_ai_backend_instance = None


def get_ai_backend() -> AIBackend:
    """Get or create AI backend singleton"""
    global _ai_backend_instance
    if _ai_backend_instance is None:
        _ai_backend_instance = AIBackend()
    return _ai_backend_instance