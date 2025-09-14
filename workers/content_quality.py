"""
ContentQualityWorker - Uses AI to validate generated content quality
Delegates quality evaluation to AI backend instead of programmatic checks
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from workers.base import BaseWorker, WorkerStatus
from workers.ai_backend import get_ai_backend

logger = logging.getLogger(__name__)


class ContentQualityWorker(BaseWorker):
    """
    Worker that validates generated content quality using AI evaluation
    
    The AI evaluates:
    - Format compliance: Is it properly formatted for the content type?
    - Quality: Is the content engaging and well-written?
    - Length appropriateness: Is the length suitable for the platform/type?
    - Relevance: Does it relate to the source material?
    """
    
    def __init__(self):
        super().__init__("content_quality")
        self.ai_backend = get_ai_backend()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ['content_id', 'content_type', 'content_text']
        
        if not all(field in input_data for field in required):
            self.log_with_context(
                logging.ERROR,
                f"Missing required fields. Required: {required}",
                extra={"input_fields": list(input_data.keys())}
            )
            return False
        
        # Check content is not empty
        content = input_data.get('content_text', '').strip()
        if not content:
            self.log_with_context(
                logging.ERROR,
                "Content text is empty"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content quality validation using AI
        
        Args:
            input_data: Must contain:
                - content_id: Content identifier
                - content_type: Type of content (summary, blog, social, etc.)
                - content_text: The content to validate
                - source_transcript: Original transcript (optional but recommended)
                - variant: Content variant (short, medium, twitter, etc.)
                - platform: Target platform (optional)
                - target_audience: Intended audience (optional)
        
        Returns:
            Dict with quality scores and recommendations from AI
        """
        content_id = input_data['content_id']
        content_type = input_data['content_type']
        content_text = input_data['content_text']
        source_transcript = input_data.get('source_transcript', '')
        
        # Prepare metadata for AI context
        metadata = {
            'variant': input_data.get('variant', 'default'),
            'platform': input_data.get('platform'),
            'target_audience': input_data.get('target_audience', 'general')
        }
        
        self.log_with_context(
            logging.INFO,
            f"Starting AI content quality validation for {content_id}",
            extra={
                'content_type': content_type,
                'content_length': len(content_text),
                'variant': metadata.get('variant'),
                'ai_backend': self.ai_backend.backend
            }
        )
        
        # Call AI backend for evaluation
        ai_result = self.ai_backend.evaluate_content(
            content=content_text,
            content_type=content_type,
            source_transcript=source_transcript,
            metadata=metadata
        )
        
        # Check if evaluation was skipped or failed
        if ai_result.get('skipped'):
            self.log_with_context(
                logging.INFO,
                f"Quality check skipped: {ai_result.get('reason')}"
            )
            return {
                'status': WorkerStatus.SUCCESS,
                'content_id': content_id,
                'quality_check_skipped': True,
                'reason': ai_result.get('reason'),
                'passed': True  # Don't block workflow
            }
        
        if ai_result.get('error'):
            self.log_with_context(
                logging.WARNING,
                f"AI evaluation failed: {ai_result.get('error_message')}"
            )
            return {
                'status': WorkerStatus.SUCCESS,  # Don't fail the worker
                'content_id': content_id,
                'quality_check_failed': True,
                'error': ai_result.get('error_message'),
                'passed': True  # Don't block workflow on AI failure
            }
        
        # Extract results from AI evaluation
        quality_score = ai_result.get('score', 0)
        passed = ai_result.get('pass', False)
        format_correct = ai_result.get('format_correct', False)
        length_appropriate = ai_result.get('length_appropriate', False)
        improvements = ai_result.get('improvements', [])
        
        # Log the result
        self.log_with_context(
            logging.INFO,
            f"Content quality check completed: {'PASSED' if passed else 'FAILED'} (score: {quality_score})",
            extra={
                'quality_score': quality_score,
                'passed': passed,
                'format_correct': format_correct,
                'length_appropriate': length_appropriate,
                'improvements_count': len(improvements)
            }
        )
        
        # Prepare result
        result = {
            'status': WorkerStatus.SUCCESS,
            'content_id': content_id,
            'quality_check_type': 'content',
            'passed': passed,
            'quality_score': quality_score,
            'format_correct': format_correct,
            'length_appropriate': length_appropriate,
            'improvements': improvements,
            'metadata': {
                'content_type': content_type,
                'variant': metadata.get('variant'),
                'word_count': len(content_text.split()),
                'character_count': len(content_text),
                'ai_backend': ai_result.get('ai_backend'),
                'ai_model': ai_result.get('ai_model'),
                'checked_at': datetime.utcnow().isoformat()
            }
        }
        
        # Include raw AI response if parsing failed
        if ai_result.get('raw_response'):
            result['raw_ai_response'] = ai_result['raw_response']
            result['parse_error'] = ai_result.get('parse_error')
        
        return result
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors during quality checking"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str:
            return {
                'error_code': 'CQ001',
                'error_type': 'timeout',
                'message': 'Quality check timeout',
                'recoverable': True
            }
        elif 'ai' in error_str or 'claude' in error_str or 'openai' in error_str:
            return {
                'error_code': 'CQ002',
                'error_type': 'ai_error',
                'message': 'AI service error',
                'recoverable': True
            }
        else:
            return {
                'error_code': 'CQ999',
                'error_type': 'unknown',
                'message': str(error),
                'recoverable': True
            }


# Convenience function
async def check_content_quality(
    content_id: str,
    content_type: str,
    content_text: str,
    source_transcript: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Check content quality using AI
    
    Args:
        content_id: Content identifier
        content_type: Type of content
        content_text: Content to validate
        source_transcript: Original transcript
        **kwargs: Additional metadata
    
    Returns:
        Quality check results from AI
    """
    worker = ContentQualityWorker()
    return worker.run({
        'content_id': content_id,
        'content_type': content_type,
        'content_text': content_text,
        'source_transcript': source_transcript,
        **kwargs
    })