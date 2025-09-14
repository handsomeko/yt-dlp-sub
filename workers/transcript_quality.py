"""
TranscriptQualityWorker - Uses AI to validate transcript quality
Delegates quality evaluation to AI backend instead of programmatic checks
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from workers.base import BaseWorker, WorkerStatus
from workers.ai_backend import get_ai_backend

logger = logging.getLogger(__name__)


class TranscriptQualityWorker(BaseWorker):
    """
    Worker that validates transcript quality using AI evaluation
    
    The AI evaluates:
    - Completeness: Does the transcript cover the video content?
    - Coherence: Is it understandable and well-structured?
    - Accuracy: Are there obvious errors or missing sections?
    - Overall quality score
    """
    
    def __init__(self):
        super().__init__("transcript_quality")
        self.ai_backend = get_ai_backend()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ['video_id', 'transcript_text']
        
        if not all(field in input_data for field in required):
            self.log_with_context(
                logging.ERROR,
                f"Missing required fields. Required: {required}",
                extra={"input_fields": list(input_data.keys())}
            )
            return False
        
        # Check transcript is not empty
        transcript = input_data.get('transcript_text', '').strip()
        if not transcript:
            self.log_with_context(
                logging.ERROR,
                "Transcript text is empty"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcript quality validation using AI
        
        Args:
            input_data: Must contain:
                - video_id: Video identifier
                - transcript_text: The transcript to validate
                - video_duration: Duration in seconds (optional but recommended)
                - language: Expected language (optional)
                - extraction_method: How transcript was obtained (optional)
        
        Returns:
            Dict with quality scores and recommendations from AI
        """
        video_id = input_data['video_id']
        transcript_text = input_data['transcript_text']
        video_duration = input_data.get('video_duration')
        
        # Prepare metadata for AI context
        metadata = {
            'language': input_data.get('language', 'en'),
            'extraction_method': input_data.get('extraction_method', 'unknown'),
            'is_auto_generated': input_data.get('is_auto_generated', False)
        }
        
        self.log_with_context(
            logging.INFO,
            f"Starting AI transcript quality validation for video {video_id}",
            extra={
                'transcript_length': len(transcript_text),
                'video_duration': video_duration,
                'ai_backend': self.ai_backend.backend
            }
        )
        
        # Call AI backend for evaluation
        ai_result = self.ai_backend.evaluate_transcript(
            transcript=transcript_text,
            video_duration=video_duration,
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
                'video_id': video_id,
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
                'video_id': video_id,
                'quality_check_failed': True,
                'error': ai_result.get('error_message'),
                'passed': True  # Don't block workflow on AI failure
            }
        
        # Extract results from AI evaluation
        quality_score = ai_result.get('score', 0)
        passed = ai_result.get('pass', False)
        issues = ai_result.get('issues', [])
        recommendations = ai_result.get('recommendations', [])
        
        # Log the result
        self.log_with_context(
            logging.INFO,
            f"Transcript quality check completed: {'PASSED' if passed else 'FAILED'} (score: {quality_score})",
            extra={
                'quality_score': quality_score,
                'passed': passed,
                'issues_count': len(issues)
            }
        )
        
        # Prepare result
        result = {
            'status': WorkerStatus.SUCCESS,
            'video_id': video_id,
            'quality_check_type': 'transcript',
            'passed': passed,
            'quality_score': quality_score,
            'issues': issues,
            'recommendations': recommendations,
            'metadata': {
                'word_count': len(transcript_text.split()),
                'character_count': len(transcript_text),
                'extraction_method': metadata.get('extraction_method'),
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
                'error_code': 'TQ001',
                'error_type': 'timeout',
                'message': 'Quality check timeout',
                'recoverable': True
            }
        elif 'ai' in error_str or 'claude' in error_str or 'openai' in error_str:
            return {
                'error_code': 'TQ002',
                'error_type': 'ai_error',
                'message': 'AI service error',
                'recoverable': True
            }
        else:
            return {
                'error_code': 'TQ999',
                'error_type': 'unknown',
                'message': str(error),
                'recoverable': True
            }


# Convenience function
async def check_transcript_quality(
    video_id: str,
    transcript_text: str,
    video_duration: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Check transcript quality using AI
    
    Args:
        video_id: Video identifier
        transcript_text: Transcript to validate
        video_duration: Video duration in seconds
        **kwargs: Additional metadata
    
    Returns:
        Quality check results from AI
    """
    worker = TranscriptQualityWorker()
    return worker.run({
        'video_id': video_id,
        'transcript_text': transcript_text,
        'video_duration': video_duration,
        **kwargs
    })