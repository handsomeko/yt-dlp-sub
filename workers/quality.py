"""
QualityWorker - Validates transcript and generated content quality.

This worker implements the quality validation system from the PRD, providing
two types of quality checks:
1. transcript_quality: Validates transcript completeness, coherence, and language
2. content_quality: Validates generated content format, length, and readability

Quality thresholds are configurable but follow PRD defaults.
"""

import asyncio
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from workers.base import BaseWorker, WorkerStatus
from core.database import db_manager, QualityCheck

# Try to import NLTK, fall back to basic text processing if not available
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # Test if NLTK data is actually available by trying to use it
    try:
        # Try to tokenize a simple test string
        test_sentences = sent_tokenize("Hello world. This is a test.")
        test_words = word_tokenize("Hello world")
        if test_sentences and test_words:
            NLTK_AVAILABLE = True
    except Exception:
        # NLTK import succeeded but data is missing or broken
        NLTK_AVAILABLE = False
except ImportError:
    # NLTK module not installed
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityValidationError(Exception):
    """Base exception for quality validation errors."""
    pass


class ContentValidationError(QualityValidationError):
    """Exception raised when content validation fails."""
    pass


class TranscriptValidationError(QualityValidationError):
    """Exception raised when transcript validation fails."""
    pass


class QualityThresholdError(QualityValidationError):
    """Exception raised when quality thresholds are not met."""
    pass


class QualityDependencyError(QualityValidationError):
    """Exception raised when quality analysis dependencies are missing."""
    pass


class QualityMetricsError(QualityValidationError):
    """Exception raised when quality metrics calculation fails."""
    pass


def basic_sent_tokenize(text: str) -> List[str]:
    """Basic sentence tokenization fallback when NLTK is not available."""
    # Simple sentence splitting on common sentence endings
    sentences = re.split(r'[.!?]+\s+', text.strip())
    # Filter out empty sentences and add back the punctuation
    return [s.strip() for s in sentences if s.strip()]


def basic_word_tokenize(text: str) -> List[str]:
    """Basic word tokenization fallback when NLTK is not available."""
    # Simple word splitting on whitespace and common punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def get_sentences(text: str) -> List[str]:
    """Get sentences using NLTK if available, otherwise use basic tokenization."""
    if NLTK_AVAILABLE:
        return sent_tokenize(text)
    else:
        return basic_sent_tokenize(text)


def get_words(text: str) -> List[str]:
    """Get words using NLTK if available, otherwise use basic tokenization."""
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    else:
        return basic_word_tokenize(text)


class TranscriptQualityMetrics:
    """Quality metrics and thresholds for transcript validation from PRD 4.2"""
    
    THRESHOLDS = {
        'completeness': 0.8,    # Min ratio of transcript duration to video duration
        'coherence': 0.7,       # Min coherence score
        'word_density': 50,     # Min words per minute
        'language_confidence': 0.9  # Min language detection confidence
    }
    
    QUALITY_WEIGHTS = {
        'completeness': 0.4,    # Most important - coverage of video
        'coherence': 0.3,       # Text makes sense
        'word_density': 0.2,    # Reasonable speaking pace
        'language_confidence': 0.1  # Language detection accuracy
    }


class ContentQualityMetrics:
    """Quality metrics for generated content validation"""
    
    MIN_LENGTHS = {
        'summary': 100,      # Minimum words for summary
        'blog_post': 500,    # Minimum words for blog post
        'social_media': 10,  # Minimum words for social posts
        'newsletter': 200,   # Minimum words for newsletter
        'script': 50        # Minimum words for script
    }
    
    MAX_LENGTHS = {
        'summary': 500,      # Maximum words for summary
        'blog_post': 2000,   # Maximum words for blog post
        'social_media': 280, # Twitter character limit considerations
        'newsletter': 1000,  # Maximum words for newsletter
        'script': 1000      # Maximum words for script
    }
    
    READABILITY_THRESHOLD = 60  # Flesch reading ease minimum score


class QualityWorker(BaseWorker):
    """
    Worker that validates transcript and content quality using AI and rule-based checks.
    
    Supports two main validation types:
    1. transcript_quality: Validates transcripts against video metadata
    2. content_quality: Validates generated content format and quality
    """
    
    def __init__(self, 
                 name: str = "quality",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 log_level: str = "INFO"):
        """
        Initialize QualityWorker.
        
        Args:
            max_retries: Maximum retry attempts for quality checks
            retry_delay: Delay between retries in seconds
            log_level: Logging level
        """
        super().__init__(
            name="quality_checker",
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_level=log_level
        )
        
        # Initialize NLTK analyzer for sentiment/readability if available
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                self.log_with_context(
                    f"Failed to initialize NLTK analyzer: {e}. Using basic text processing.",
                    level="WARNING"
                )
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for quality checking.
        
        Required fields:
        - target_id: ID of transcript or content being validated
        - target_type: 'transcript' or 'content'
        - content: Text content to validate
        
        Optional fields:
        - video_duration: Required for transcript validation
        - content_type: Type of content (summary, blog_post, etc.)
        - video_metadata: Additional video context
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ['target_id', 'target_type', 'content']
        
        # Check required fields
        if not all(field in input_data for field in required_fields):
            missing = [f for f in required_fields if f not in input_data]
            self.log_with_context(
                f"Missing required fields: {missing}",
                level="ERROR",
                extra_context={"available_fields": list(input_data.keys())}
            )
            return False
        
        # Validate target_type
        target_type = input_data['target_type']
        if target_type not in ['transcript', 'content']:
            self.log_with_context(
                f"Invalid target_type: {target_type}. Must be 'transcript' or 'content'",
                level="ERROR"
            )
            return False
        
        # For transcript validation, video_duration is required
        if target_type == 'transcript' and 'video_duration' not in input_data:
            self.log_with_context(
                "video_duration is required for transcript quality validation",
                level="ERROR"
            )
            return False
        
        # Validate content is not empty
        content = input_data.get('content', '')
        if not content or not content.strip():
            self.log_with_context(
                "Content is empty or contains only whitespace",
                level="ERROR"
            )
            return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute quality validation based on target_type.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Quality validation results with score, pass/fail, and recommendations
        """
        target_id = input_data['target_id']
        target_type = input_data['target_type']
        content = input_data['content']
        
        self.log_with_context(
            f"Starting quality validation for {target_type} {target_id}",
            extra_context={"content_length": len(content)}
        )
        
        try:
            if target_type == 'transcript':
                result = self._validate_transcript_quality(input_data)
            else:  # content
                result = self._validate_content_quality(input_data)
            
            # Store quality check result in database (skip if no event loop)
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._store_quality_result(target_id, target_type, result))
            except RuntimeError:
                # No event loop running, skip database storage for testing
                self.log_with_context(
                    "Skipping database storage - no event loop running",
                    level="DEBUG"
                )
            
            self.log_with_context(
                f"Quality validation completed - Score: {result['overall_score']:.2f}, "
                f"Passed: {result['passed']}",
                extra_context={"target_id": target_id, "target_type": target_type}
            )
            
            return result
            
        except Exception as e:
            self.log_with_context(
                f"Quality validation failed: {str(e)}",
                level="ERROR",
                extra_context={"target_id": target_id, "target_type": target_type}
            )
            raise
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle quality validation errors and categorize them.
        
        Args:
            error: Exception that occurred during validation
            
        Returns:
            Error context and recovery recommendations
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        # Categorize common errors
        if "NLTK" in error_str or "punkt" in error_str:
            error_category = "nltk_dependency"
            recovery_action = "install_nltk_data"
            is_retryable = True
        elif "database" in error_str.lower():
            error_category = "database_error"
            recovery_action = "check_database_connection"
            is_retryable = True
        elif "content" in error_str.lower() and "empty" in error_str.lower():
            error_category = "invalid_content"
            recovery_action = "provide_valid_content"
            is_retryable = False
        else:
            error_category = "unknown"
            recovery_action = "manual_review"
            is_retryable = False
        
        self.log_with_context(
            f"Quality validation error categorized as {error_category}",
            level="WARNING",
            extra_context={
                "error_type": error_type,
                "is_retryable": is_retryable,
                "recovery_action": recovery_action
            }
        )
        
        return {
            "error_type": error_type,
            "error_category": error_category,
            "recovery_action": recovery_action,
            "is_retryable": is_retryable,
            "details": error_str
        }
    
    def _validate_transcript_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate transcript quality using PRD metrics.
        
        Checks:
        1. Completeness - transcript duration vs video duration
        2. Coherence - text flow and sentence structure
        3. Word density - speaking pace (words per minute)
        4. Language confidence - consistency of language detection
        
        Args:
            input_data: Input containing transcript content and video metadata
            
        Returns:
            Quality validation results
        """
        content = input_data['content']
        video_duration = input_data['video_duration']  # in seconds
        video_metadata = input_data.get('video_metadata', {})
        
        scores = {}
        details = {}
        
        # 1. Completeness Check
        completeness_score, completeness_details = self._check_completeness(
            content, video_duration
        )
        scores['completeness'] = completeness_score
        details['completeness'] = completeness_details
        
        # 2. Coherence Check
        coherence_score, coherence_details = self._check_coherence(content)
        scores['coherence'] = coherence_score
        details['coherence'] = coherence_details
        
        # 3. Word Density Check
        density_score, density_details = self._check_word_density(
            content, video_duration
        )
        scores['word_density'] = density_score
        details['word_density'] = density_details
        
        # 4. Language Confidence Check
        language_score, language_details = self._check_language_confidence(content)
        scores['language_confidence'] = language_score
        details['language_confidence'] = language_details
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(
            scores, TranscriptQualityMetrics.QUALITY_WEIGHTS
        )
        
        # Determine if transcript passes quality checks
        passed = self._passes_transcript_thresholds(scores)
        
        # Generate recommendations
        recommendations = self._generate_transcript_recommendations(scores, details)
        
        return {
            "target_type": "transcript",
            "overall_score": overall_score,
            "individual_scores": scores,
            "passed": passed,
            "details": details,
            "recommendations": recommendations,
            "thresholds_used": TranscriptQualityMetrics.THRESHOLDS,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _validate_content_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated content quality.
        
        Checks:
        1. Format validity - proper structure and formatting
        2. Length requirements - appropriate word/character count
        3. Keyword presence - relevant terms from original content
        4. Readability - Flesch reading ease score
        
        Args:
            input_data: Input containing content and metadata
            
        Returns:
            Quality validation results
        """
        content = input_data['content']
        content_type = input_data.get('content_type', 'unknown')
        source_transcript = input_data.get('source_transcript', '')
        
        scores = {}
        details = {}
        
        # 1. Format Validity Check
        format_score, format_details = self._check_content_format(
            content, content_type
        )
        scores['format_validity'] = format_score
        details['format_validity'] = format_details
        
        # 2. Length Requirements Check
        length_score, length_details = self._check_content_length(
            content, content_type
        )
        scores['length_requirements'] = length_score
        details['length_requirements'] = length_details
        
        # 3. Keyword Relevance Check
        relevance_score, relevance_details = self._check_keyword_relevance(
            content, source_transcript
        )
        scores['keyword_relevance'] = relevance_score
        details['keyword_relevance'] = relevance_details
        
        # 4. Readability Check
        readability_score, readability_details = self._check_readability(content)
        scores['readability'] = readability_score
        details['readability'] = readability_details
        
        # Calculate overall score (equal weights for content validation)
        overall_score = statistics.mean(scores.values())
        
        # Determine if content passes quality checks
        passed = all(score >= 0.6 for score in scores.values())  # 60% threshold for content
        
        # Generate recommendations
        recommendations = self._generate_content_recommendations(scores, details, content_type)
        
        return {
            "target_type": "content",
            "content_type": content_type,
            "overall_score": overall_score,
            "individual_scores": scores,
            "passed": passed,
            "details": details,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_completeness(self, content: str, video_duration: int) -> Tuple[float, Dict]:
        """Check if transcript covers adequate portion of video duration."""
        # Estimate transcript duration based on word count and average speaking pace
        words = len(get_words(content))
        # Average speaking pace: 150-160 words per minute, use 155
        estimated_duration = (words / 155) * 60  # Convert to seconds
        
        # Calculate completeness ratio
        completeness_ratio = min(estimated_duration / video_duration, 1.0) if video_duration > 0 else 0.0
        
        # Score based on threshold
        threshold = TranscriptQualityMetrics.THRESHOLDS['completeness']
        score = min(completeness_ratio / threshold, 1.0)
        
        details = {
            "word_count": words,
            "estimated_transcript_duration_seconds": estimated_duration,
            "video_duration_seconds": video_duration,
            "completeness_ratio": completeness_ratio,
            "threshold": threshold,
            "coverage_percentage": completeness_ratio * 100
        }
        
        return score, details
    
    def _check_coherence(self, content: str) -> Tuple[float, Dict]:
        """Check text coherence using sentence structure analysis."""
        sentences = get_sentences(content)
        
        if not sentences:
            return 0.0, {"error": "No sentences found"}
        
        # Basic coherence metrics
        avg_sentence_length = statistics.mean([len(get_words(s)) for s in sentences])
        sentence_count = len(sentences)
        
        # Check for sentence variety (avoid too many very short or very long sentences)
        sentence_lengths = [len(get_words(s)) for s in sentences]
        length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Coherence score based on reasonable sentence structure
        # Ideal: 10-25 words per sentence, with some variance
        if 10 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif avg_sentence_length < 5 or avg_sentence_length > 40:
            length_score = 0.3  # Too fragmented or too complex
        else:
            length_score = 0.7
        
        # Variance score - some variety is good, too much or too little is bad
        if 10 <= length_variance <= 50:
            variance_score = 1.0
        else:
            variance_score = 0.6
        
        # Check for repeated phrases (sign of poor transcription)
        words = get_words(content.lower())
        unique_words = set(words)
        repetition_score = len(unique_words) / len(words) if words else 0
        
        # Combined coherence score
        coherence_score = (length_score * 0.4 + variance_score * 0.3 + repetition_score * 0.3)
        
        details = {
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_variance": length_variance,
            "unique_word_ratio": repetition_score,
            "length_score": length_score,
            "variance_score": variance_score,
            "repetition_score": repetition_score
        }
        
        return coherence_score, details
    
    def _check_word_density(self, content: str, video_duration: int) -> Tuple[float, Dict]:
        """Check word density (words per minute)."""
        words = len(get_words(content))
        duration_minutes = video_duration / 60 if video_duration > 0 else 1
        
        words_per_minute = words / duration_minutes
        threshold = TranscriptQualityMetrics.THRESHOLDS['word_density']
        
        # Score based on meeting minimum threshold
        if words_per_minute >= threshold:
            # Cap at reasonable maximum (200 wpm) to avoid artificially high scores
            score = min(words_per_minute / threshold, 200 / threshold)
        else:
            score = words_per_minute / threshold
        
        score = min(score, 1.0)  # Cap at 1.0
        
        details = {
            "word_count": words,
            "video_duration_minutes": duration_minutes,
            "words_per_minute": words_per_minute,
            "threshold": threshold,
            "meets_threshold": words_per_minute >= threshold
        }
        
        return score, details
    
    def _check_language_confidence(self, content: str) -> Tuple[float, Dict]:
        """Check language consistency and confidence."""
        # Simple language consistency check based on character patterns
        # This is a basic implementation - could be enhanced with proper language detection
        
        # Check for consistent use of English characters
        total_chars = len(content)
        if total_chars == 0:
            return 0.0, {"error": "No content to analyze"}
        
        # Count ASCII alphabetic characters (English indicator)
        english_chars = sum(1 for c in content if c.isascii() and c.isalpha())
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        # Check for common English words
        words = get_words(content.lower())
        common_english_words = {
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
            'for', 'as', 'was', 'on', 'are', 'you', 'this', 'be', 'at', 'have'
        }
        
        if words:
            common_word_count = sum(1 for w in words if w in common_english_words)
            common_word_ratio = common_word_count / len(words)
        else:
            common_word_ratio = 0
        
        # Combine metrics for language confidence
        confidence_score = (english_ratio * 0.6 + common_word_ratio * 0.4)
        
        details = {
            "total_characters": total_chars,
            "english_character_ratio": english_ratio,
            "common_english_word_ratio": common_word_ratio,
            "confidence_score": confidence_score
        }
        
        return confidence_score, details
    
    def _check_content_format(self, content: str, content_type: str) -> Tuple[float, Dict]:
        """Check if content follows expected format for its type."""
        score = 1.0  # Default to passing
        details = {"format_checks": []}
        
        if content_type == 'blog_post':
            # Check for basic blog structure
            has_title = bool(re.search(r'^#\s+.+', content, re.MULTILINE))
            has_paragraphs = len(content.split('\n\n')) >= 3
            
            details["format_checks"].extend([
                {"check": "has_title", "passed": has_title},
                {"check": "has_paragraphs", "passed": has_paragraphs}
            ])
            
            if not has_title:
                score -= 0.3
            if not has_paragraphs:
                score -= 0.2
        
        elif content_type == 'social_media':
            # Check for appropriate social media formatting
            has_hashtags = bool(re.search(r'#\w+', content))
            reasonable_length = len(content) <= 280  # Twitter-like limit
            
            details["format_checks"].extend([
                {"check": "has_hashtags", "passed": has_hashtags},
                {"check": "reasonable_length", "passed": reasonable_length}
            ])
            
            if not reasonable_length:
                score -= 0.5  # Length is critical for social media
        
        elif content_type in ['summary', 'newsletter']:
            # Check for structured content
            has_structure = len(content.split('\n')) >= 3
            details["format_checks"].append(
                {"check": "has_structure", "passed": has_structure}
            )
            
            if not has_structure:
                score -= 0.3
        
        # Ensure score doesn't go below 0
        score = max(score, 0.0)
        
        details["content_type"] = content_type
        details["format_score"] = score
        
        return score, details
    
    def _check_content_length(self, content: str, content_type: str) -> Tuple[float, Dict]:
        """Check if content meets length requirements for its type."""
        word_count = len(get_words(content))
        char_count = len(content)
        
        # Get length requirements
        min_length = ContentQualityMetrics.MIN_LENGTHS.get(content_type, 50)
        max_length = ContentQualityMetrics.MAX_LENGTHS.get(content_type, 1000)
        
        # Calculate score based on length appropriateness
        if min_length <= word_count <= max_length:
            score = 1.0
        elif word_count < min_length:
            # Too short - score based on how close to minimum
            score = word_count / min_length
        else:
            # Too long - penalize excess length
            excess_ratio = (word_count - max_length) / max_length
            score = max(0.5, 1.0 - (excess_ratio * 0.5))  # Gradual penalty
        
        details = {
            "word_count": word_count,
            "character_count": char_count,
            "content_type": content_type,
            "min_length_requirement": min_length,
            "max_length_requirement": max_length,
            "meets_min_length": word_count >= min_length,
            "meets_max_length": word_count <= max_length,
            "length_score": score
        }
        
        return score, details
    
    def _check_keyword_relevance(self, content: str, source_transcript: str) -> Tuple[float, Dict]:
        """Check if content contains relevant keywords from source transcript."""
        if not source_transcript:
            # No source to compare against - assume relevant
            return 0.8, {"note": "No source transcript provided for comparison"}
        
        # Extract key terms from both content and source
        content_words = set(get_words(content.lower()))
        source_words = set(get_words(source_transcript.lower()))
        
        # Remove common stop words for better relevance matching
        stop_words = {
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
            'for', 'as', 'was', 'on', 'are', 'you', 'this', 'be', 'at', 'have',
            'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what',
            'all', 'were', 'they', 'we', 'when', 'your', 'can', 'said'
        }
        
        content_keywords = content_words - stop_words
        source_keywords = source_words - stop_words
        
        if not source_keywords:
            return 0.5, {"note": "No meaningful keywords found in source"}
        
        # Calculate keyword overlap
        common_keywords = content_keywords & source_keywords
        relevance_ratio = len(common_keywords) / len(source_keywords)
        
        # Score based on reasonable keyword overlap (20-80% is good)
        if 0.2 <= relevance_ratio <= 0.8:
            score = 1.0
        elif relevance_ratio < 0.2:
            score = relevance_ratio / 0.2  # Scale up to 0.2
        else:
            score = 0.9  # Slight penalty for too much overlap (might be copy-paste)
        
        details = {
            "content_keywords_count": len(content_keywords),
            "source_keywords_count": len(source_keywords),
            "common_keywords_count": len(common_keywords),
            "relevance_ratio": relevance_ratio,
            "common_keywords": list(common_keywords)[:20]  # Sample of common keywords
        }
        
        return score, details
    
    def _check_readability(self, content: str) -> Tuple[float, Dict]:
        """Check content readability using Flesch Reading Ease approximation."""
        sentences = get_sentences(content)
        words = get_words(content)
        
        if not sentences or not words:
            return 0.0, {"error": "No sentences or words found"}
        
        # Count syllables (approximation using vowel groups)
        def count_syllables(word):
            word = word.lower()
            syllables = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in 'aeiouy'
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel
            # Handle silent 'e' and ensure at least 1 syllable
            if word.endswith('e') and syllables > 1:
                syllables -= 1
            return max(syllables, 1)
        
        total_syllables = sum(count_syllables(word) for word in words if word.isalpha())
        
        # Flesch Reading Ease formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words) if words else 0
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(flesch_score, 100))  # Clamp between 0-100
        
        # Convert Flesch score to 0-1 scale using threshold
        threshold = ContentQualityMetrics.READABILITY_THRESHOLD
        normalized_score = min(flesch_score / threshold, 1.0)
        
        # Readability level interpretation
        if flesch_score >= 90:
            level = "Very Easy"
        elif flesch_score >= 80:
            level = "Easy"
        elif flesch_score >= 70:
            level = "Fairly Easy"
        elif flesch_score >= 60:
            level = "Standard"
        elif flesch_score >= 50:
            level = "Fairly Difficult"
        elif flesch_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        details = {
            "flesch_reading_ease": flesch_score,
            "readability_level": level,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word,
            "total_words": len(words),
            "total_sentences": len(sentences),
            "total_syllables": total_syllables,
            "meets_threshold": flesch_score >= threshold
        }
        
        return normalized_score, details
    
    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average score."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.25)  # Default weight if not specified
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _passes_transcript_thresholds(self, scores: Dict[str, float]) -> bool:
        """Check if transcript scores pass minimum thresholds."""
        thresholds = TranscriptQualityMetrics.THRESHOLDS
        
        for metric, score in scores.items():
            threshold = thresholds.get(metric, 0.6)  # Default threshold
            if score < threshold:
                return False
        
        return True
    
    def _generate_transcript_recommendations(self, scores: Dict[str, float], details: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for transcript quality improvement."""
        recommendations = []
        thresholds = TranscriptQualityMetrics.THRESHOLDS
        
        # Completeness recommendations
        if scores.get('completeness', 0) < thresholds['completeness']:
            completeness_ratio = details.get('completeness', {}).get('completeness_ratio', 0)
            recommendations.append(
                f"Transcript appears incomplete ({completeness_ratio*100:.1f}% coverage). "
                "Consider re-transcribing with a different method or checking for audio quality issues."
            )
        
        # Coherence recommendations
        if scores.get('coherence', 0) < thresholds['coherence']:
            avg_length = details.get('coherence', {}).get('avg_sentence_length', 0)
            if avg_length < 5:
                recommendations.append(
                    "Transcript has fragmented sentences. This may indicate poor audio quality "
                    "or aggressive automatic punctuation. Consider manual review."
                )
            elif avg_length > 40:
                recommendations.append(
                    "Transcript has overly long sentences. Consider adding punctuation "
                    "or using a different transcription service."
                )
        
        # Word density recommendations
        if scores.get('word_density', 0) < thresholds['word_density']:
            wpm = details.get('word_density', {}).get('words_per_minute', 0)
            recommendations.append(
                f"Low word density ({wpm:.1f} words/minute). This may indicate "
                "significant gaps in transcription or very slow speech. "
                "Verify audio quality and transcription completeness."
            )
        
        # Language confidence recommendations
        if scores.get('language_confidence', 0) < thresholds['language_confidence']:
            recommendations.append(
                "Low language confidence detected. This may indicate mixed languages, "
                "poor audio quality, or transcription errors. Consider manual review."
            )
        
        if not recommendations:
            recommendations.append("Transcript meets all quality thresholds.")
        
        return recommendations
    
    def _generate_content_recommendations(self, scores: Dict[str, float], details: Dict[str, Any], content_type: str) -> List[str]:
        """Generate actionable recommendations for content quality improvement."""
        recommendations = []
        
        # Format recommendations
        if scores.get('format_validity', 0) < 0.8:
            format_checks = details.get('format_validity', {}).get('format_checks', [])
            failed_checks = [check['check'] for check in format_checks if not check.get('passed', True)]
            if failed_checks:
                recommendations.append(
                    f"Content format issues detected: {', '.join(failed_checks)}. "
                    f"Ensure {content_type} follows expected formatting conventions."
                )
        
        # Length recommendations
        if scores.get('length_requirements', 0) < 0.8:
            length_details = details.get('length_requirements', {})
            word_count = length_details.get('word_count', 0)
            min_length = length_details.get('min_length_requirement', 0)
            max_length = length_details.get('max_length_requirement', 0)
            
            if word_count < min_length:
                recommendations.append(
                    f"Content is too short ({word_count} words, minimum {min_length}). "
                    "Add more detail and comprehensive coverage of the topic."
                )
            elif word_count > max_length:
                recommendations.append(
                    f"Content is too long ({word_count} words, maximum {max_length}). "
                    "Consider condensing and focusing on key points."
                )
        
        # Relevance recommendations
        if scores.get('keyword_relevance', 0) < 0.6:
            relevance_ratio = details.get('keyword_relevance', {}).get('relevance_ratio', 0)
            if relevance_ratio < 0.2:
                recommendations.append(
                    "Content appears to have low relevance to source material. "
                    "Ensure generated content directly relates to the original transcript."
                )
        
        # Readability recommendations
        if scores.get('readability', 0) < 0.6:
            flesch_score = details.get('readability', {}).get('flesch_reading_ease', 0)
            if flesch_score < 30:
                recommendations.append(
                    "Content is very difficult to read. Consider simplifying sentence structure "
                    "and using more common vocabulary."
                )
            elif flesch_score < 60:
                recommendations.append(
                    "Content readability could be improved. Consider shortening sentences "
                    "and using simpler language where appropriate."
                )
        
        if not recommendations:
            recommendations.append("Content meets all quality standards.")
        
        return recommendations
    
    async def _store_quality_result(self, target_id: str, target_type: str, result: Dict[str, Any]):
        """Store quality check result in database."""
        try:
            async with db_manager.get_session() as session:
                # Create quality check record for overall result
                quality_check = QualityCheck(
                    target_id=target_id,
                    target_type=target_type,
                    check_type='overall_quality',
                    score=result['overall_score'],
                    passed=result['passed'],
                    details=result
                )
                
                session.add(quality_check)
                
                # Create individual records for each quality metric
                individual_scores = result.get('individual_scores', {})
                for metric, score in individual_scores.items():
                    individual_check = QualityCheck(
                        target_id=target_id,
                        target_type=target_type,
                        check_type=metric,
                        score=score,
                        passed=score >= 0.6,  # Basic threshold for individual metrics
                        details={
                            "metric": metric,
                            "score": score,
                            "details": result.get('details', {}).get(metric, {})
                        }
                    )
                    session.add(individual_check)
                
                await session.commit()
                
                self.log_with_context(
                    f"Quality check results stored for {target_type} {target_id}",
                    extra_context={"overall_score": result['overall_score'], "passed": result['passed']}
                )
                
        except Exception as e:
            self.log_with_context(
                f"Failed to store quality check results: {str(e)}",
                level="ERROR",
                extra_context={"target_id": target_id, "target_type": target_type}
            )
            # Don't re-raise as this shouldn't fail the main quality check