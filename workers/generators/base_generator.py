"""
Base generator class for all content generators.

Provides common functionality for prompt template management, output formatting,
metadata handling, and AI model integration across all sub-generators.
"""

import json
import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from ..base import BaseWorker, WorkerStatus


class ContentFormat(Enum):
    """Standard content format constants."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class GenerationQuality(Enum):
    """Content quality levels for different use cases."""
    DRAFT = "draft"      # Fast generation, may need editing
    STANDARD = "standard"  # Balanced quality and speed
    PREMIUM = "premium"   # High quality, slower generation


class BaseGenerator(BaseWorker):
    """
    Abstract base class for all content generators.
    
    Provides common functionality for:
    - Prompt template management
    - Output formatting and validation
    - Metadata tracking
    - AI model abstraction (Phase 2)
    - Content quality assessment
    
    Attributes:
        supported_formats: List of output formats this generator supports
        max_word_count: Maximum words this generator can produce
        min_word_count: Minimum words this generator should produce
        quality_level: Generation quality setting
    """
    
    def __init__(
        self,
        name: str,
        supported_formats: List[ContentFormat] = None,
        max_word_count: int = 5000,
        min_word_count: int = 50,
        quality_level: GenerationQuality = GenerationQuality.STANDARD,
        **kwargs
    ) -> None:
        """
        Initialize the base generator.
        
        Args:
            name: Human-readable name for this generator
            supported_formats: List of supported output formats
            max_word_count: Maximum words this generator can produce
            min_word_count: Minimum words this generator should produce
            quality_level: Quality level for generation
            **kwargs: Additional arguments for BaseWorker
        """
        super().__init__(name=name, **kwargs)
        
        self.supported_formats = supported_formats or [ContentFormat.PLAIN_TEXT]
        self.max_word_count = max_word_count
        self.min_word_count = min_word_count
        self.quality_level = quality_level
        
        # Template storage for prompt management
        self._prompt_templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load prompt templates for this generator. Override in subclasses."""
        # Base template - subclasses should override
        self._prompt_templates = {
            "base": "Generate content based on the following transcript:\n\n{transcript}\n\nRequirements:\n{requirements}"
        }
    
    def get_template(self, template_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template string
            
        Raises:
            KeyError: If template doesn't exist
        """
        if template_name not in self._prompt_templates:
            available = list(self._prompt_templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available: {available}")
        
        return self._prompt_templates[template_name]
    
    def format_prompt(
        self, 
        template_name: str, 
        transcript: str,
        requirements: str = "",
        **kwargs
    ) -> str:
        """
        Format a prompt template with provided data.
        
        Args:
            template_name: Name of template to use
            transcript: Video transcript content
            requirements: Specific requirements for this generation
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        
        # Standard variables available to all templates
        template_vars = {
            'transcript': transcript,
            'requirements': requirements,
            'max_words': self.max_word_count,
            'min_words': self.min_word_count,
            'quality': self.quality_level.value,
            **kwargs
        }
        
        return template.format(**template_vars)
    
    def count_words(self, text: str) -> int:
        """
        Count words in text content.
        
        Args:
            text: Text to count words in
            
        Returns:
            Word count
        """
        if not text or not text.strip():
            return 0
        
        # Simple word counting - split on whitespace
        return len(text.split())
    
    def validate_word_count(self, content: str, target_count: Optional[int] = None) -> bool:
        """
        Validate that content meets word count requirements.
        
        Args:
            content: Generated content to validate
            target_count: Specific target word count (overrides min/max)
            
        Returns:
            True if word count is acceptable
        """
        word_count = self.count_words(content)
        
        if target_count:
            # Allow 10% variance for target counts
            tolerance = max(10, int(target_count * 0.1))
            return abs(word_count - target_count) <= tolerance
        
        return self.min_word_count <= word_count <= self.max_word_count
    
    def format_output(
        self,
        content: str,
        format_type: ContentFormat,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format generated content in the specified format.
        
        Args:
            content: Raw generated content
            format_type: Desired output format
            metadata: Additional metadata to include
            
        Returns:
            Formatted content with metadata
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Format {format_type.value} not supported by {self.name}")
        
        # Base metadata
        result_metadata = {
            "generator": self.name,
            "format": format_type.value,
            "word_count": self.count_words(content),
            "quality_level": self.quality_level.value,
            "generation_timestamp": datetime.utcnow().isoformat(),
            "processing_time": self.get_execution_time()
        }
        
        # Add custom metadata
        if metadata:
            result_metadata.update(metadata)
        
        # Format content based on type
        formatted_content = self._apply_format(content, format_type)
        
        return {
            "content": formatted_content,
            "metadata": result_metadata,
            "raw_content": content if format_type != ContentFormat.PLAIN_TEXT else None
        }
    
    def _apply_format(self, content: str, format_type: ContentFormat) -> Union[str, Dict[str, Any]]:
        """
        Apply specific formatting to content.
        
        Args:
            content: Raw content to format
            format_type: Target format
            
        Returns:
            Formatted content
        """
        if format_type == ContentFormat.PLAIN_TEXT:
            return content.strip()
        
        elif format_type == ContentFormat.MARKDOWN:
            # Basic markdown formatting - subclasses can override
            return content.strip()
        
        elif format_type == ContentFormat.HTML:
            # Basic HTML formatting - subclasses can override
            paragraphs = content.strip().split('\n\n')
            html_paragraphs = [f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()]
            return '\n'.join(html_paragraphs)
        
        elif format_type == ContentFormat.JSON:
            # JSON structure - subclasses should override for specific structure
            return {
                "content": content.strip(),
                "type": "generic"
            }
        
        return content
    
    def create_generation_metadata(
        self,
        transcript_length: int,
        target_format: ContentFormat,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for a generation request.
        
        Args:
            transcript_length: Length of input transcript in characters
            target_format: Requested output format
            generation_params: Additional generation parameters
            
        Returns:
            Complete metadata dictionary
        """
        return {
            "input": {
                "transcript_length": transcript_length,
                "transcript_word_count": self.count_words("x " * (transcript_length // 5))  # Rough estimate
            },
            "generation": {
                "target_format": target_format.value,
                "quality_level": self.quality_level.value,
                "max_words": self.max_word_count,
                "min_words": self.min_word_count,
                "parameters": generation_params or {}
            },
            "generator": {
                "name": self.name,
                "version": "1.0.0",  # TODO: Dynamic versioning
                "supported_formats": [fmt.value for fmt in self.supported_formats]
            }
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for content generation.
        
        Args:
            input_data: Input data containing transcript and generation parameters
            
        Returns:
            True if input is valid
        """
        # Required fields
        required_fields = ['transcript']
        for field in required_fields:
            if field not in input_data:
                self.log_with_context(f"Missing required field: {field}", level="ERROR")
                return False
        
        # Validate transcript content
        transcript = input_data['transcript']
        if not transcript or not isinstance(transcript, str) or not transcript.strip():
            self.log_with_context("Transcript is empty or invalid", level="ERROR")
            return False
        
        # Validate format if specified
        if 'format' in input_data:
            format_str = input_data['format']
            try:
                requested_format = ContentFormat(format_str)
                if requested_format not in self.supported_formats:
                    self.log_with_context(
                        f"Unsupported format '{format_str}' for {self.name}",
                        level="ERROR"
                    )
                    return False
            except ValueError:
                self.log_with_context(f"Invalid format '{format_str}'", level="ERROR")
                return False
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content generation with common processing logic.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Generation result
        """
        transcript = input_data['transcript']
        target_format = ContentFormat(input_data.get('format', 'plain_text'))
        generation_params = input_data.get('params', {})
        
        self.log_with_context(
            f"Starting generation for {len(transcript)} character transcript",
            extra_context={
                "format": target_format.value,
                "quality": self.quality_level.value
            }
        )
        
        # Create metadata for this generation
        base_metadata = self.create_generation_metadata(
            transcript_length=len(transcript),
            target_format=target_format,
            generation_params=generation_params
        )
        
        # Perform the actual generation (implemented by subclasses)
        generated_content = self.generate_content(
            transcript=transcript,
            format_type=target_format,
            **generation_params
        )
        
        # Validate word count
        if not self.validate_word_count(generated_content, generation_params.get('target_words')):
            word_count = self.count_words(generated_content)
            self.log_with_context(
                f"Generated content word count ({word_count}) outside acceptable range",
                level="WARNING"
            )
        
        # Format and package the output
        formatted_result = self.format_output(
            content=generated_content,
            format_type=target_format,
            metadata=base_metadata
        )
        
        self.log_with_context(
            f"Generation completed: {formatted_result['metadata']['word_count']} words"
        )
        
        return formatted_result
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle generation errors with recovery information.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error context and recovery suggestions
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        recovery_suggestions = []
        
        # Common error recovery patterns
        if "template" in error_message.lower():
            recovery_suggestions.append("Check prompt template formatting")
        
        if "format" in error_message.lower():
            recovery_suggestions.append("Verify output format is supported")
        
        if "word" in error_message.lower() or "length" in error_message.lower():
            recovery_suggestions.append("Adjust word count requirements")
        
        # Default recovery
        if not recovery_suggestions:
            recovery_suggestions.append("Retry with default parameters")
            recovery_suggestions.append("Check input transcript quality")
        
        return {
            "error_type": error_type,
            "error_message": error_message,
            "recovery_suggestions": recovery_suggestions,
            "generator": self.name,
            "supported_formats": [fmt.value for fmt in self.supported_formats]
        }
    
    @abstractmethod
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        **kwargs
    ) -> str:
        """
        Generate content from transcript. Must be implemented by subclasses.
        
        Args:
            transcript: Input transcript text
            format_type: Desired output format
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content as string
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass