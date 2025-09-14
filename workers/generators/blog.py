"""
Blog post generator for transforming video transcripts into structured blog articles.

Provides three article lengths:
- Short: ~500 words, focused on key insights
- Medium: ~1000 words, detailed exploration with examples
- Long: ~2000 words, comprehensive analysis with multiple sections
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

from .base_generator import BaseGenerator, ContentFormat, GenerationQuality


class BlogLength(Enum):
    """Blog post length options with target word counts."""
    SHORT = ("short", 500)
    MEDIUM = ("medium", 1000) 
    LONG = ("long", 2000)
    
    def __init__(self, name: str, target_words: int):
        self.blog_name = name
        self.target_words = target_words


class BlogStyle(Enum):
    """Blog writing styles for different audiences."""
    PROFESSIONAL = "professional"    # Formal, industry-focused
    CONVERSATIONAL = "conversational"  # Casual, engaging tone
    EDUCATIONAL = "educational"      # Tutorial/explainer style
    ANALYTICAL = "analytical"       # Data-driven, research-focused


class BlogGenerator(BaseGenerator):
    """
    Generates blog posts from video transcripts with configurable length and style.
    
    Features:
    - Three length options: short (500w), medium (1000w), long (2000w)
    - Multiple writing styles for different audiences
    - SEO-friendly structure with headers and sections
    - Markdown and HTML output support
    - Automatic excerpt generation
    """
    
    def __init__(self, **kwargs):
        """Initialize the blog generator."""
        super().__init__(
            name="BlogPostGenerator",
            supported_formats=[
                ContentFormat.PLAIN_TEXT,
                ContentFormat.MARKDOWN,
                ContentFormat.HTML,
                ContentFormat.JSON
            ],
            max_word_count=2500,  # Allow buffer over long target
            min_word_count=300,   # Minimum for meaningful blog post
            **kwargs
        )
        
        # Blog-specific configuration
        self.default_length = BlogLength.MEDIUM
        self.default_style = BlogStyle.CONVERSATIONAL
        self.supported_lengths = list(BlogLength)
        self.supported_styles = list(BlogStyle)
    
    def _load_templates(self) -> None:
        """Load blog-specific prompt templates."""
        self._prompt_templates = {
            "short_professional": """Write a professional blog post of approximately {target_words} words based on this video transcript.

Transcript:
{transcript}

Structure:
- Compelling title and introduction
- 2-3 main sections with key insights
- Professional tone suitable for industry audience
- Clear conclusion with actionable takeaways

Target: {target_words} words""",

            "short_conversational": """Write an engaging, conversational blog post of approximately {target_words} words from this video content.

Transcript:
{transcript}

Style:
- Friendly, approachable tone
- Use "you" to engage readers
- Include personal insights and relatable examples
- 2-3 main points with clear explanations

Target: {target_words} words""",

            "medium_educational": """Create an educational blog post of approximately {target_words} words that teaches readers about the topics in this transcript.

Transcript:
{transcript}

Include:
- Clear introduction explaining what readers will learn
- 3-5 main sections breaking down key concepts
- Examples and explanations for clarity
- Step-by-step guidance where applicable
- Summary with next steps

Target: {target_words} words""",

            "long_analytical": """Write a comprehensive analytical blog post of approximately {target_words} words examining the topics in this transcript.

Transcript:
{transcript}

Structure:
- Executive summary
- Detailed analysis of main themes
- Supporting evidence and examples
- Implications and broader context
- Data-driven insights where relevant
- Thorough conclusion with recommendations

Target: {target_words} words"""
        }
    
    def get_blog_length(self, length_str: str) -> BlogLength:
        """
        Get BlogLength enum from string.
        
        Args:
            length_str: Length identifier ('short', 'medium', 'long')
            
        Returns:
            BlogLength enum
            
        Raises:
            ValueError: If length_str is not recognized
        """
        for length in BlogLength:
            if length.blog_name == length_str.lower():
                return length
        
        available = [l.blog_name for l in BlogLength]
        raise ValueError(f"Unknown blog length '{length_str}'. Available: {available}")
    
    def get_blog_style(self, style_str: str) -> BlogStyle:
        """
        Get BlogStyle enum from string.
        
        Args:
            style_str: Style identifier
            
        Returns:
            BlogStyle enum
            
        Raises:
            ValueError: If style_str is not recognized
        """
        try:
            return BlogStyle(style_str.lower())
        except ValueError:
            available = [s.value for s in BlogStyle]
            raise ValueError(f"Unknown blog style '{style_str}'. Available: {available}")
    
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        length: str = "medium",
        style: str = "conversational",
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate blog post content from transcript.
        
        Args:
            transcript: Input video transcript
            format_type: Desired output format
            length: Blog length ('short', 'medium', 'long')
            style: Writing style ('professional', 'conversational', etc.)
            title: Custom title for the blog post
            **kwargs: Additional parameters
            
        Returns:
            Generated blog post content
        """
        # Parse blog parameters
        try:
            blog_length = self.get_blog_length(length)
        except ValueError:
            self.log_with_context(
                f"Invalid blog length '{length}', using default 'medium'",
                level="WARNING"
            )
            blog_length = self.default_length
        
        try:
            blog_style = self.get_blog_style(style)
        except ValueError:
            self.log_with_context(
                f"Invalid blog style '{style}', using default 'conversational'",
                level="WARNING"
            )
            blog_style = self.default_style
        
        # Check if AI backend is available
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        if ai_backend.is_available():
            # Use AI to generate real content
            return self._generate_ai_blog(
                transcript, blog_length, blog_style, format_type, title
            )
        else:
            # Fallback to placeholder content
            return self._generate_placeholder_blog(
                transcript, blog_length, blog_style, format_type, title
            )
    
    def _generate_ai_blog(
        self,
        transcript: str,
        blog_length: BlogLength,
        blog_style: BlogStyle,
        format_type: ContentFormat,
        custom_title: Optional[str] = None
    ) -> str:
        """
        Generate actual blog post using AI backend.
        
        Args:
            transcript: Original transcript
            blog_length: Target blog length
            blog_style: Writing style
            format_type: Output format
            custom_title: Custom title override
            
        Returns:
            AI-generated blog post content
        """
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        # Select appropriate template
        template_key = f"{blog_length.blog_name}_{blog_style.value}"
        if template_key not in self._prompt_templates:
            template_key = f"{blog_length.blog_name}_conversational"
        
        template = self._prompt_templates[template_key]
        
        # Build prompt
        prompt = template.format(
            transcript=transcript[:8000],  # Limit transcript to avoid token overflow
            target_words=blog_length.target_words
        )
        
        # Add custom title instruction if provided
        if custom_title:
            prompt = f"Use this title for the blog post: {custom_title}\n\n{prompt}"
        
        # Add format instruction
        if format_type == ContentFormat.MARKDOWN:
            prompt += "\n\nFormat the output in Markdown with proper headers, bold text, and lists where appropriate."
        elif format_type == ContentFormat.HTML:
            prompt += "\n\nFormat the output in clean HTML with proper tags like <h2>, <p>, <ul>, etc."
        
        # Generate content using AI
        response = ai_backend.generate_content(
            prompt=prompt,
            max_tokens=blog_length.target_words * 2,  # Rough estimate
            temperature=0.7
        )
        
        if response and response.get('content'):
            generated_content = response['content']
            
            # Post-process based on format
            if format_type == ContentFormat.MARKDOWN:
                # Ensure proper markdown formatting
                if not generated_content.startswith('#'):
                    generated_content = f"# {custom_title or 'Blog Post'}\n\n{generated_content}"
            elif format_type == ContentFormat.HTML:
                # Wrap in basic HTML structure if needed
                if not generated_content.startswith('<'):
                    generated_content = f"<article>\n<h1>{custom_title or 'Blog Post'}</h1>\n{generated_content}\n</article>"
            
            return generated_content
        else:
            # Fallback to placeholder if AI generation fails
            self.log_with_context(
                "AI generation failed, falling back to placeholder",
                level="WARNING"
            )
            return self._generate_placeholder_blog(
                transcript, blog_length, blog_style, format_type, custom_title
            )
    
    def _generate_placeholder_blog(
        self,
        transcript: str,
        blog_length: BlogLength,
        blog_style: BlogStyle,
        format_type: ContentFormat,
        custom_title: Optional[str] = None
    ) -> str:
        """
        Generate structured placeholder blog post for Phase 1.
        
        Args:
            transcript: Original transcript
            blog_length: Target blog length
            blog_style: Writing style
            format_type: Output format
            custom_title: Custom title override
            
        Returns:
            Placeholder blog post content
        """
        # Extract basic info from transcript
        transcript_words = self.count_words(transcript)
        first_sentences = '. '.join(transcript.split('.')[:2]) + "." if transcript else "No content available"
        
        # Generate structured content
        blog_data = self._create_blog_structure(
            first_sentences, transcript_words, blog_length, blog_style, custom_title
        )
        
        # Apply format-specific rendering
        return self._format_blog_content(blog_data, format_type)
    
    def _create_blog_structure(
        self,
        content_preview: str,
        transcript_words: int,
        blog_length: BlogLength,
        blog_style: BlogStyle,
        custom_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create structured blog post data."""
        
        # Generate title
        if custom_title:
            title = custom_title
        else:
            title = self._generate_title(content_preview, blog_style)
        
        # Generate excerpt
        excerpt = self._generate_excerpt(content_preview, blog_style)
        
        # Generate sections based on length
        if blog_length == BlogLength.SHORT:
            sections = self._generate_short_sections(content_preview, blog_style)
        elif blog_length == BlogLength.MEDIUM:
            sections = self._generate_medium_sections(content_preview, blog_style)
        else:  # LONG
            sections = self._generate_long_sections(content_preview, blog_style)
        
        return {
            "title": title,
            "excerpt": excerpt,
            "introduction": sections["introduction"],
            "main_sections": sections["main_sections"],
            "conclusion": sections["conclusion"],
            "metadata": {
                "target_words": blog_length.target_words,
                "style": blog_style.value,
                "length": blog_length.blog_name,
                "source_words": transcript_words,
                "generated_date": datetime.now().strftime("%Y-%m-%d"),
                "estimated_reading_time": self._calculate_reading_time(blog_length.target_words)
            }
        }
    
    def _generate_title(self, content_preview: str, style: BlogStyle) -> str:
        """Generate blog post title based on content and style."""
        if style == BlogStyle.PROFESSIONAL:
            return "Industry Analysis: Key Insights from Video Content"
        elif style == BlogStyle.CONVERSATIONAL:
            return "What I Learned from This Video (And Why You Should Care)"
        elif style == BlogStyle.EDUCATIONAL:
            return "Complete Guide: Understanding the Key Concepts"
        else:  # ANALYTICAL
            return "Deep Dive: Comprehensive Analysis of Video Content"
    
    def _generate_excerpt(self, content_preview: str, style: BlogStyle) -> str:
        """Generate blog post excerpt/meta description."""
        if style == BlogStyle.PROFESSIONAL:
            return "Professional analysis of key insights and industry implications from video content."
        elif style == BlogStyle.CONVERSATIONAL:
            return "A friendly breakdown of the main ideas and why they matter to you."
        elif style == BlogStyle.EDUCATIONAL:
            return "Learn the essential concepts and practical applications covered in this comprehensive guide."
        else:  # ANALYTICAL
            return "In-depth examination of themes, evidence, and broader implications of the discussed topics."
    
    def _generate_short_sections(self, content_preview: str, style: BlogStyle) -> Dict[str, Any]:
        """Generate sections for short blog posts."""
        return {
            "introduction": self._get_style_introduction(content_preview, style, "short"),
            "main_sections": [
                {
                    "title": "Key Insight #1",
                    "content": "First major point discussed in the video with supporting explanation and context."
                },
                {
                    "title": "Key Insight #2", 
                    "content": "Second important concept with practical applications and real-world examples."
                }
            ],
            "conclusion": self._get_style_conclusion(style, "short")
        }
    
    def _generate_medium_sections(self, content_preview: str, style: BlogStyle) -> Dict[str, Any]:
        """Generate sections for medium blog posts."""
        return {
            "introduction": self._get_style_introduction(content_preview, style, "medium"),
            "main_sections": [
                {
                    "title": "Understanding the Foundation",
                    "content": "Establishing the core concepts and background information necessary to grasp the main discussion points."
                },
                {
                    "title": "Deep Dive into Key Concepts",
                    "content": "Detailed exploration of the primary themes with examples, evidence, and supporting analysis."
                },
                {
                    "title": "Practical Applications",
                    "content": "How these insights apply to real-world scenarios and what actions readers can take."
                },
                {
                    "title": "Broader Implications",
                    "content": "The wider significance of these concepts and their impact on the field or industry."
                }
            ],
            "conclusion": self._get_style_conclusion(style, "medium")
        }
    
    def _generate_long_sections(self, content_preview: str, style: BlogStyle) -> Dict[str, Any]:
        """Generate sections for long blog posts."""
        return {
            "introduction": self._get_style_introduction(content_preview, style, "long"),
            "main_sections": [
                {
                    "title": "Executive Summary",
                    "content": "High-level overview of all major points and conclusions for quick reference."
                },
                {
                    "title": "Background and Context",
                    "content": "Comprehensive background information, historical context, and current state of the field."
                },
                {
                    "title": "Primary Analysis",
                    "content": "Detailed examination of the main themes with thorough supporting evidence and examples."
                },
                {
                    "title": "Supporting Evidence and Case Studies",
                    "content": "Additional data points, research findings, and real-world case studies that reinforce the main arguments."
                },
                {
                    "title": "Critical Evaluation",
                    "content": "Objective assessment of strengths, limitations, and potential counterarguments to the presented ideas."
                },
                {
                    "title": "Future Implications and Trends",
                    "content": "Analysis of how these concepts may evolve and their potential impact on future developments."
                }
            ],
            "conclusion": self._get_style_conclusion(style, "long")
        }
    
    def _get_style_introduction(self, content_preview: str, style: BlogStyle, length: str) -> str:
        """Generate style-appropriate introduction."""
        if style == BlogStyle.PROFESSIONAL:
            return f"This analysis examines key insights from recent video content, providing professional perspective on {content_preview.split('.')[0].lower()}. The following assessment covers critical points relevant to industry professionals."
        elif style == BlogStyle.CONVERSATIONAL:
            return f"Hey there! I just watched an interesting video about {content_preview.split('.')[0].lower()}, and I wanted to share some thoughts with you. Here's what caught my attention and why I think you'll find it valuable too."
        elif style == BlogStyle.EDUCATIONAL:
            return f"In this comprehensive guide, we'll explore the key concepts presented in a recent video discussion. You'll learn about {content_preview.split('.')[0].lower()} and gain practical understanding of the important principles covered."
        else:  # ANALYTICAL
            return f"This analytical review examines the content, methodology, and implications of recent video material discussing {content_preview.split('.')[0].lower()}. We'll provide data-driven insights and critical evaluation of the presented arguments."
    
    def _get_style_conclusion(self, style: BlogStyle, length: str) -> str:
        """Generate style-appropriate conclusion."""
        if style == BlogStyle.PROFESSIONAL:
            return "These insights provide valuable perspective for industry professionals. Consider how these concepts align with current best practices and strategic initiatives in your organization."
        elif style == BlogStyle.CONVERSATIONAL:
            return "What do you think about these ideas? I'd love to hear your thoughts in the comments. If you found this helpful, don't forget to share it with others who might benefit!"
        elif style == BlogStyle.EDUCATIONAL:
            return "Now you have a solid understanding of these key concepts. Practice applying these principles in your own context, and refer back to this guide as needed for reinforcement."
        else:  # ANALYTICAL
            return "This analysis demonstrates the complexity and nuance of the discussed topics. Further research and longitudinal studies would provide additional validation of these findings and their broader implications."
    
    def _calculate_reading_time(self, word_count: int) -> str:
        """Calculate estimated reading time (assuming 200 words per minute)."""
        minutes = max(1, round(word_count / 200))
        return f"{minutes} min read"
    
    def _format_blog_content(self, blog_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """Format blog content based on output type."""
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(blog_data, indent=2)
        elif format_type == ContentFormat.HTML:
            return self._format_html_blog(blog_data)
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_blog(blog_data)
        else:  # PLAIN_TEXT
            return self._format_text_blog(blog_data)
    
    def _format_markdown_blog(self, blog_data: Dict[str, Any]) -> str:
        """Format blog as Markdown."""
        lines = []
        
        # Title
        lines.append(f"# {blog_data['title']}")
        lines.append("")
        
        # Metadata
        metadata = blog_data['metadata']
        lines.append(f"*{metadata['estimated_reading_time']} • {metadata['style'].title()} style • {metadata['target_words']} words*")
        lines.append("")
        
        # Excerpt
        lines.append(f"**{blog_data['excerpt']}**")
        lines.append("")
        
        # Introduction
        lines.append(blog_data['introduction'])
        lines.append("")
        
        # Main sections
        for section in blog_data['main_sections']:
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(section['content'])
            lines.append("")
        
        # Conclusion
        lines.append("## Conclusion")
        lines.append("")
        lines.append(blog_data['conclusion'])
        
        return "\n".join(lines)
    
    def _format_html_blog(self, blog_data: Dict[str, Any]) -> str:
        """Format blog as HTML."""
        html_parts = []
        
        # Article header
        html_parts.append("<article>")
        html_parts.append(f"<h1>{blog_data['title']}</h1>")
        
        # Metadata
        metadata = blog_data['metadata']
        html_parts.append(f"<p class='meta'><em>{metadata['estimated_reading_time']} • {metadata['style'].title()} style • {metadata['target_words']} words</em></p>")
        
        # Excerpt
        html_parts.append(f"<p class='excerpt'><strong>{blog_data['excerpt']}</strong></p>")
        
        # Introduction
        html_parts.append(f"<p>{blog_data['introduction']}</p>")
        
        # Main sections
        for section in blog_data['main_sections']:
            html_parts.append(f"<h2>{section['title']}</h2>")
            html_parts.append(f"<p>{section['content']}</p>")
        
        # Conclusion
        html_parts.append("<h2>Conclusion</h2>")
        html_parts.append(f"<p>{blog_data['conclusion']}</p>")
        
        html_parts.append("</article>")
        
        return "\n".join(html_parts)
    
    def _format_text_blog(self, blog_data: Dict[str, Any]) -> str:
        """Format blog as plain text."""
        lines = []
        
        # Title
        lines.append(blog_data['title'].upper())
        lines.append("=" * len(blog_data['title']))
        lines.append("")
        
        # Introduction
        lines.append(blog_data['introduction'])
        lines.append("")
        
        # Main sections
        for section in blog_data['main_sections']:
            lines.append(section['title'].upper())
            lines.append("-" * len(section['title']))
            lines.append("")
            lines.append(section['content'])
            lines.append("")
        
        # Conclusion
        lines.append("CONCLUSION")
        lines.append("-" * 10)
        lines.append("")
        lines.append(blog_data['conclusion'])
        
        return "\n".join(lines)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate blog generation input."""
        if not super().validate_input(input_data):
            return False
        
        # Validate blog length if specified
        if "length" in input_data:
            try:
                self.get_blog_length(input_data["length"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate blog style if specified  
        if "style" in input_data:
            try:
                self.get_blog_style(input_data["style"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Ensure transcript is substantial enough for blog post
        transcript = input_data["transcript"]
        if len(transcript.split()) < 50:
            self.log_with_context(
                "Transcript too short for meaningful blog post (minimum 50 words)",
                level="ERROR"
            )
            return False
        
        return True
    
    def get_supported_configurations(self) -> Dict[str, Any]:
        """Get supported blog configurations."""
        return {
            "lengths": {length.blog_name: length.target_words for length in BlogLength},
            "styles": [style.value for style in BlogStyle],
            "formats": [fmt.value for fmt in self.supported_formats]
        }