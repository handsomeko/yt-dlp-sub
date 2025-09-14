"""
Summary generator for creating concise summaries from video transcripts.

Provides three summary formats:
- Short: ~100 words, key takeaways only
- Medium: ~300 words, structured overview with main points  
- Detailed: ~500 words, comprehensive summary with context
"""

from typing import Dict, Any
from enum import Enum

from .base_generator import BaseGenerator, ContentFormat, GenerationQuality


class SummaryLength(Enum):
    """Summary length options with target word counts."""
    SHORT = ("short", 100)
    MEDIUM = ("medium", 300)
    DETAILED = ("detailed", 500)
    
    def __init__(self, name: str, target_words: int):
        self.summary_name = name
        self.target_words = target_words


class SummaryGenerator(BaseGenerator):
    """
    Generates summaries of varying lengths from video transcripts.
    
    Features:
    - Three length options: short (100w), medium (300w), detailed (500w)
    - Structured output with key points
    - Bullet point and paragraph formats
    - Quality validation for coherence and completeness
    """
    
    def __init__(self, **kwargs):
        """Initialize the summary generator."""
        super().__init__(
            name="SummaryGenerator",
            supported_formats=[
                ContentFormat.PLAIN_TEXT,
                ContentFormat.MARKDOWN,
                ContentFormat.JSON
            ],
            max_word_count=600,  # Allow some buffer over detailed target
            min_word_count=50,   # Minimum for even shortest summaries
            **kwargs
        )
        
        # Summary-specific configuration
        self.default_length = SummaryLength.MEDIUM
        self.supported_lengths = list(SummaryLength)
    
    def _load_templates(self) -> None:
        """Load summary-specific prompt templates."""
        self._prompt_templates = {
            "short": """Create a concise summary of this video transcript in approximately {target_words} words.

Transcript:
{transcript}

Focus on:
- Main topic and key takeaway
- Most important points only
- Clear, direct language

Target: {target_words} words maximum""",

            "medium": """Create a structured summary of this video transcript in approximately {target_words} words.

Transcript:
{transcript}

Include:
- Main topic and context
- 3-5 key points with brief explanations
- Overall conclusion or takeaway
- Logical flow and structure

Target: {target_words} words""",

            "detailed": """Create a comprehensive summary of this video transcript in approximately {target_words} words.

Transcript:
{transcript}

Include:
- Introduction with context and background
- Main arguments or points with supporting details
- Key examples, data, or evidence mentioned
- Logical progression of ideas
- Clear conclusion with implications
- Maintain original tone and style

Target: {target_words} words"""
        }
    
    def get_summary_length(self, length_str: str) -> SummaryLength:
        """
        Get SummaryLength enum from string.
        
        Args:
            length_str: Length identifier ('short', 'medium', 'detailed')
            
        Returns:
            SummaryLength enum
            
        Raises:
            ValueError: If length_str is not recognized
        """
        for length in SummaryLength:
            if length.summary_name == length_str.lower():
                return length
        
        available = [l.summary_name for l in SummaryLength]
        raise ValueError(f"Unknown summary length '{length_str}'. Available: {available}")
    
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        length: str = "medium",
        **kwargs
    ) -> str:
        """
        Generate summary content from transcript.
        
        Args:
            transcript: Input video transcript
            format_type: Desired output format
            length: Summary length ('short', 'medium', 'detailed')
            **kwargs: Additional parameters
            
        Returns:
            Generated summary content
        """
        # Parse summary length
        try:
            summary_length = self.get_summary_length(length)
        except ValueError:
            self.log_with_context(
                f"Invalid summary length '{length}', using default 'medium'",
                level="WARNING"
            )
            summary_length = self.default_length
        
        # Try AI generation first, fallback to placeholder
        try:
            from core.ai_backend import get_ai_backend
            ai_backend = get_ai_backend()
            
            if ai_backend and ai_backend.is_available():
                return self._generate_ai_summary(
                    transcript=transcript,
                    summary_length=summary_length,
                    format_type=format_type
                )
        except Exception as e:
            self.log_with_context(
                f"AI generation failed, using placeholder: {e}",
                level="WARNING"
            )
        
        # Fallback to placeholder content
        return self._generate_placeholder_summary(transcript, summary_length, format_type)
    
    def _generate_ai_summary(
        self,
        transcript: str,
        summary_length: SummaryLength,
        format_type: ContentFormat
    ) -> str:
        """Generate real summary content using AI backend."""
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        # Get the appropriate template
        template_name = summary_length.summary_name
        template = self.get_template(template_name)
        
        # Format the prompt
        prompt = template.format(
            transcript=transcript[:8000],  # Limit transcript length for AI
            target_words=summary_length.target_words
        )
        
        # Generate content
        self.log_with_context(f"Generating {summary_length.summary_name} summary with AI")
        
        # Adjust token limit based on summary length
        max_tokens = summary_length.target_words * 2  # Conservative estimate
        
        response = ai_backend.generate_content(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.5  # Lower temperature for factual summaries
        )
        
        if not response:
            raise ValueError("AI backend returned empty response")
        
        # Parse and structure the response
        summary_data = self._parse_ai_summary(response, summary_length)
        
        # Add metadata
        summary_data["metadata"] = {
            "summary_length": summary_length.summary_name,
            "target_words": summary_length.target_words,
            "actual_words": self.count_words(response),
            "source_words": self.count_words(transcript),
            "compression_ratio": round(self.count_words(transcript) / max(1, self.count_words(response)), 2),
            "ai_generated": True
        }
        
        return self._format_summary_content(summary_data, format_type)
    
    def _parse_ai_summary(self, response: str, summary_length: SummaryLength) -> Dict[str, Any]:
        """Parse AI response into structured summary format."""
        lines = response.strip().split('\n')
        
        if summary_length == SummaryLength.SHORT:
            # Short summary - just extract the main content
            return {
                "main_topic": lines[0] if lines else "Summary of video content",
                "key_takeaway": response.strip(),
                "context": f"AI-generated {summary_length.summary_name} summary",
                "target_words": summary_length.target_words
            }
        
        elif summary_length == SummaryLength.MEDIUM:
            # Medium summary - extract introduction, key points, and conclusion
            introduction = ""
            key_points = []
            conclusion = ""
            
            current_section = "introduction"
            for line in lines:
                line = line.strip()
                
                # Detect section transitions
                if any(marker in line.lower() for marker in ['key point', 'main point', 'important', 'highlight']):
                    current_section = "key_points"
                    continue
                elif any(marker in line.lower() for marker in ['conclusion', 'overall', 'summary', 'in conclusion']):
                    current_section = "conclusion"
                    if not line.lower().startswith(('conclusion', 'overall', 'summary', 'in conclusion')):
                        conclusion += line + " "
                    continue
                
                # Add content to appropriate section
                if current_section == "introduction" and line:
                    introduction += line + " "
                elif current_section == "key_points" and line:
                    # Handle bullet points or numbered lists
                    if line.startswith(('•', '-', '*', '1', '2', '3', '4', '5')):
                        clean_point = line.lstrip('•-*1234567890. ')
                        if clean_point:
                            key_points.append(clean_point)
                    elif len(key_points) > 0 and not line.startswith(('•', '-', '*')):
                        # Continuation of previous point
                        key_points[-1] += " " + line
                    elif line:
                        # New point without bullet
                        key_points.append(line)
                elif current_section == "conclusion" and line:
                    conclusion += line + " "
            
            # Fallback if sections weren't parsed
            if not introduction and lines:
                introduction = lines[0]
            if not key_points and len(lines) > 2:
                # Extract middle lines as key points
                for line in lines[1:-1]:
                    if line.strip():
                        key_points.append(line.strip())
                        if len(key_points) >= 4:
                            break
            if not conclusion and lines:
                conclusion = lines[-1] if len(lines) > 1 else "Key insights from the video content."
            
            return {
                "introduction": introduction.strip() or "Summary of video content",
                "key_points": key_points[:5] if key_points else ["Main point from the video"],
                "conclusion": conclusion.strip() or "Overall takeaway from the discussion.",
                "context": f"AI-generated {summary_length.summary_name} summary",
                "target_words": summary_length.target_words
            }
        
        else:  # DETAILED
            # Detailed summary - extract all sections
            introduction = ""
            background = ""
            main_sections = []
            examples = []
            conclusion = ""
            
            current_section = "introduction"
            current_main_section = None
            
            for line in lines:
                line = line.strip()
                
                # Detect section headers
                if any(marker in line.lower() for marker in ['background', 'context', 'foundation']):
                    current_section = "background"
                    continue
                elif any(marker in line.lower() for marker in ['example', 'case study', 'instance']):
                    current_section = "examples"
                    continue
                elif any(marker in line.lower() for marker in ['conclusion', 'summary', 'implications']):
                    current_section = "conclusion"
                    if not line.lower().startswith(('conclusion', 'summary', 'implications')):
                        conclusion += line + " "
                    continue
                elif line and ':' in line and len(line.split(':')[0]) < 50:
                    # Potential section header
                    title = line.split(':')[0].strip()
                    content = ':'.join(line.split(':')[1:]).strip()
                    current_main_section = {
                        "title": title,
                        "content": content
                    }
                    main_sections.append(current_main_section)
                    current_section = "main"
                    continue
                
                # Add content to appropriate section
                if current_section == "introduction" and line:
                    introduction += line + " "
                elif current_section == "background" and line:
                    background += line + " "
                elif current_section == "main" and current_main_section and line:
                    current_main_section["content"] += " " + line
                elif current_section == "examples" and line:
                    if line.startswith(('•', '-', '*', '1', '2', '3')):
                        examples.append(line.lstrip('•-*1234567890. '))
                    elif examples and not line.startswith(('•', '-', '*')):
                        examples[-1] += " " + line
                    else:
                        examples.append(line)
                elif current_section == "conclusion" and line:
                    conclusion += line + " "
                elif line:  # Unassigned content
                    if not main_sections:
                        main_sections.append({
                            "title": "Main Content",
                            "content": line
                        })
                    else:
                        main_sections[-1]["content"] += " " + line
            
            # Ensure we have content in all sections
            if not introduction:
                introduction = "Comprehensive analysis of the video content"
            if not background:
                background = "Context and foundation for the discussion"
            if not main_sections:
                main_sections = [{
                    "title": "Primary Analysis",
                    "content": response[:500] if len(response) > 500 else response
                }]
            if not examples:
                examples = ["Key example from the content"]
            if not conclusion:
                conclusion = "Summary and implications of the discussion"
            
            return {
                "introduction": introduction.strip(),
                "background": background.strip(),
                "main_sections": main_sections[:3],  # Limit to 3 main sections
                "examples": examples[:3],  # Limit to 3 examples
                "conclusion": conclusion.strip(),
                "context": f"AI-generated {summary_length.summary_name} summary",
                "target_words": summary_length.target_words
            }
    
    def _generate_placeholder_summary(
        self,
        transcript: str,
        summary_length: SummaryLength,
        format_type: ContentFormat
    ) -> str:
        """
        Generate structured placeholder summary for Phase 1.
        
        Args:
            transcript: Original transcript
            summary_length: Target summary length
            format_type: Output format
            
        Returns:
            Placeholder summary content
        """
        # Extract some basic info from transcript
        transcript_words = self.count_words(transcript)
        first_sentence = transcript.split('.')[0][:100] + "..." if transcript else "No content"
        
        # Generate content based on length
        if summary_length == SummaryLength.SHORT:
            content = self._generate_short_placeholder(first_sentence, transcript_words)
        elif summary_length == SummaryLength.MEDIUM:
            content = self._generate_medium_placeholder(first_sentence, transcript_words)
        else:  # DETAILED
            content = self._generate_detailed_placeholder(first_sentence, transcript_words)
        
        # Apply format-specific structuring
        return self._format_summary_content(content, format_type)
    
    def _generate_short_placeholder(self, first_sentence: str, transcript_words: int) -> Dict[str, Any]:
        """Generate short summary placeholder."""
        return {
            "main_topic": f"Video discusses: {first_sentence}",
            "key_takeaway": "Primary insight extracted from the content.",
            "context": f"Based on {transcript_words}-word transcript analysis.",
            "target_words": SummaryLength.SHORT.target_words
        }
    
    def _generate_medium_placeholder(self, first_sentence: str, transcript_words: int) -> Dict[str, Any]:
        """Generate medium summary placeholder."""
        return {
            "introduction": f"This video covers: {first_sentence}",
            "key_points": [
                "First major point discussed in the content",
                "Second important topic or argument presented", 
                "Third significant insight or conclusion",
                "Additional supporting point or example"
            ],
            "conclusion": "Overall takeaway and implications of the discussion.",
            "context": f"Summarized from {transcript_words}-word transcript.",
            "target_words": SummaryLength.MEDIUM.target_words
        }
    
    def _generate_detailed_placeholder(self, first_sentence: str, transcript_words: int) -> Dict[str, Any]:
        """Generate detailed summary placeholder."""
        return {
            "introduction": f"This comprehensive video discusses: {first_sentence}",
            "background": "Context and background information establishing the foundation for the discussion.",
            "main_sections": [
                {
                    "title": "Primary Topic Area",
                    "content": "Detailed exploration of the first major theme with supporting evidence and examples."
                },
                {
                    "title": "Secondary Analysis", 
                    "content": "In-depth examination of related concepts and their implications."
                },
                {
                    "title": "Key Arguments",
                    "content": "Main arguments presented with supporting data and reasoning."
                }
            ],
            "examples": [
                "Specific example or case study mentioned in the content",
                "Additional supporting evidence or data point",
                "Practical application or real-world connection"
            ],
            "conclusion": "Comprehensive conclusion tying together all major themes and their broader implications for the field or topic area.",
            "context": f"Detailed analysis of {transcript_words}-word transcript.",
            "target_words": SummaryLength.DETAILED.target_words
        }
    
    def _format_summary_content(self, content_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """
        Format summary content based on output type.
        
        Args:
            content_data: Structured content data
            format_type: Target format
            
        Returns:
            Formatted summary string
        """
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(content_data, indent=2)
        
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_summary(content_data)
        
        else:  # PLAIN_TEXT
            return self._format_text_summary(content_data)
    
    def _format_markdown_summary(self, content_data: Dict[str, Any]) -> str:
        """Format summary as Markdown."""
        lines = []
        
        # Title based on length
        target_words = content_data.get("target_words", 300)
        if target_words <= 150:
            lines.append("# Quick Summary")
        elif target_words <= 350:
            lines.append("# Summary")
        else:
            lines.append("# Detailed Summary")
        
        lines.append("")
        
        # Main content
        if "main_topic" in content_data:
            lines.append(f"**Main Topic:** {content_data['main_topic']}")
        
        if "introduction" in content_data:
            lines.append(content_data["introduction"])
            lines.append("")
        
        if "background" in content_data:
            lines.append(f"**Background:** {content_data['background']}")
            lines.append("")
        
        if "key_points" in content_data:
            lines.append("## Key Points")
            for point in content_data["key_points"]:
                lines.append(f"- {point}")
            lines.append("")
        
        if "main_sections" in content_data:
            lines.append("## Main Sections")
            for section in content_data["main_sections"]:
                lines.append(f"### {section['title']}")
                lines.append(section["content"])
                lines.append("")
        
        if "examples" in content_data:
            lines.append("## Examples")
            for example in content_data["examples"]:
                lines.append(f"- {example}")
            lines.append("")
        
        if "key_takeaway" in content_data:
            lines.append(f"**Key Takeaway:** {content_data['key_takeaway']}")
        
        if "conclusion" in content_data:
            lines.append(f"**Conclusion:** {content_data['conclusion']}")
        
        # Metadata
        lines.append("")
        lines.append("---")
        lines.append(f"*{content_data.get('context', 'Generated summary')}*")
        
        return "\n".join(lines)
    
    def _format_text_summary(self, content_data: Dict[str, Any]) -> str:
        """Format summary as plain text."""
        lines = []
        
        # Introduction or main topic
        if "main_topic" in content_data:
            lines.append(content_data["main_topic"])
        elif "introduction" in content_data:
            lines.append(content_data["introduction"])
        
        lines.append("")
        
        # Background if available
        if "background" in content_data:
            lines.append(content_data["background"])
            lines.append("")
        
        # Key points
        if "key_points" in content_data:
            for i, point in enumerate(content_data["key_points"], 1):
                lines.append(f"{i}. {point}")
            lines.append("")
        
        # Main sections
        if "main_sections" in content_data:
            for section in content_data["main_sections"]:
                lines.append(f"{section['title']}: {section['content']}")
                lines.append("")
        
        # Examples
        if "examples" in content_data:
            lines.append("Examples:")
            for example in content_data["examples"]:
                lines.append(f"• {example}")
            lines.append("")
        
        # Conclusion or takeaway
        if "key_takeaway" in content_data:
            lines.append(content_data["key_takeaway"])
        elif "conclusion" in content_data:
            lines.append(content_data["conclusion"])
        
        return "\n".join(lines).strip()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate summary generation input.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid
        """
        # Call parent validation first
        if not super().validate_input(input_data):
            return False
        
        # Validate summary length if specified
        if "length" in input_data:
            try:
                self.get_summary_length(input_data["length"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Ensure transcript is long enough for meaningful summary
        transcript = input_data["transcript"]
        if len(transcript.split()) < 20:
            self.log_with_context(
                "Transcript too short for meaningful summary (minimum 20 words)",
                level="ERROR"
            )
            return False
        
        return True
    
    def get_supported_lengths(self) -> Dict[str, int]:
        """
        Get supported summary lengths with target word counts.
        
        Returns:
            Dictionary mapping length names to target word counts
        """
        return {length.summary_name: length.target_words for length in SummaryLength}