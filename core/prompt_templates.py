"""
Advanced template engine for prompt management.

Provides Jinja2-style template rendering with custom filters and functions
for prompt generation and variable substitution.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import jinja2
from jinja2 import Environment, Template, meta

class PromptTemplateEngine:
    """
    Advanced template engine for rendering prompts.
    
    Features:
    - Jinja2 template syntax
    - Custom filters and functions
    - Variable validation
    - Safe rendering with defaults
    """
    
    def __init__(self):
        """Initialize the template engine with custom filters."""
        self.env = Environment(
            autoescape=False,  # We're not generating HTML
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined  # Fail on undefined variables
        )
        
        # Add custom filters
        self.env.filters['truncate_words'] = self._truncate_words
        self.env.filters['truncate_chars'] = self._truncate_chars
        self.env.filters['word_count'] = self._word_count
        self.env.filters['sentence_case'] = self._sentence_case
        self.env.filters['title_case'] = self._title_case
        self.env.filters['remove_timestamps'] = self._remove_timestamps
        self.env.filters['extract_key_points'] = self._extract_key_points
        self.env.filters['format_duration'] = self._format_duration
        
        # Add custom functions
        self.env.globals['now'] = datetime.utcnow
        self.env.globals['today'] = datetime.utcnow().date
    
    def render(
        self,
        template_str: str,
        variables: Dict[str, Any],
        safe_mode: bool = True
    ) -> str:
        """
        Render a template with the provided variables.
        
        Args:
            template_str: Jinja2 template string
            variables: Variables to substitute
            safe_mode: Use safe defaults for missing variables
            
        Returns:
            Rendered template string
        """
        try:
            if safe_mode:
                # Use a more forgiving undefined handler
                env = Environment(
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True,
                    undefined=jinja2.ChainableUndefined
                )
                # Copy filters and globals
                env.filters.update(self.env.filters)
                env.globals.update(self.env.globals)
                template = env.from_string(template_str)
            else:
                template = self.env.from_string(template_str)
            
            # Add default variables
            render_vars = self._get_default_variables()
            render_vars.update(variables)
            
            return template.render(**render_vars)
            
        except jinja2.UndefinedError as e:
            raise ValueError(f"Missing required template variable: {e}")
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering error: {e}")
    
    def extract_variables(self, template_str: str) -> Set[str]:
        """
        Extract all variables used in a template.
        
        Args:
            template_str: Jinja2 template string
            
        Returns:
            Set of variable names
        """
        try:
            ast = self.env.parse(template_str)
            return meta.find_undeclared_variables(ast)
        except jinja2.TemplateSyntaxError:
            # Fallback to regex for invalid templates
            pattern = r'\{\{?\s*(\w+)'
            matches = re.findall(pattern, template_str)
            return set(matches)
    
    def validate_template(self, template_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a template for syntax errors.
        
        Args:
            template_str: Template to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.env.from_string(template_str)
            return True, None
        except jinja2.TemplateSyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def preview(
        self,
        template_str: str,
        sample_variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Preview a template with sample data.
        
        Args:
            template_str: Template to preview
            sample_variables: Optional sample variables
            
        Returns:
            Rendered preview
        """
        if not sample_variables:
            sample_variables = self._get_sample_variables()
        
        return self.render(template_str, sample_variables, safe_mode=True)
    
    # Custom Filters
    
    def _truncate_words(self, text: str, count: int = 100, suffix: str = "...") -> str:
        """Truncate text to a specific number of words."""
        words = text.split()
        if len(words) <= count:
            return text
        return ' '.join(words[:count]) + suffix
    
    def _truncate_chars(self, text: str, count: int = 500, suffix: str = "...") -> str:
        """Truncate text to a specific number of characters."""
        if len(text) <= count:
            return text
        return text[:count - len(suffix)] + suffix
    
    def _word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _sentence_case(self, text: str) -> str:
        """Convert text to sentence case."""
        if not text:
            return text
        return text[0].upper() + text[1:].lower()
    
    def _title_case(self, text: str) -> str:
        """Convert text to title case."""
        return text.title()
    
    def _remove_timestamps(self, text: str) -> str:
        """Remove timestamp markers from transcript text."""
        # Remove common timestamp patterns like [00:00:00] or (0:00)
        patterns = [
            r'\[\d{1,2}:\d{2}:\d{2}\]',  # [00:00:00]
            r'\[\d{1,2}:\d{2}\]',         # [00:00]
            r'\(\d{1,2}:\d{2}:\d{2}\)',   # (00:00:00)
            r'\(\d{1,2}:\d{2}\)',         # (0:00)
            r'^\d{1,2}:\d{2}:\d{2}\s+',   # 00:00:00 at line start
            r'^\d{1,2}:\d{2}\s+'          # 00:00 at line start
        ]
        
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.MULTILINE)
        
        return result.strip()
    
    def _extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text (simple version).
        
        In production, this would use NLP techniques.
        """
        # Simple extraction based on sentences ending with periods
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter for substantial sentences (more than 5 words)
        key_sentences = [s for s in sentences if len(s.split()) > 5]
        
        # Return first N sentences as key points
        return key_sentences[:max_points]
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs else f"{minutes} minutes"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def _get_default_variables(self) -> Dict[str, Any]:
        """Get default variables available in all templates."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'date': datetime.utcnow().date().isoformat(),
            'year': datetime.utcnow().year,
            'default_word_limit': 500,
            'default_tone': 'professional'
        }
    
    def _get_sample_variables(self) -> Dict[str, Any]:
        """Get sample variables for template preview."""
        return {
            'title': 'Sample Video Title: How to Build Better Software',
            'transcript': """This is a sample transcript that demonstrates the template engine.
            
            In this video, we'll explore three key concepts:
            1. Understanding user requirements
            2. Building maintainable code
            3. Testing and deployment strategies
            
            Let's start with understanding user requirements. The most important aspect
            of any software project is understanding what the users actually need...""",
            'channel_name': 'Tech Talks Channel',
            'duration': 1234,  # seconds
            'view_count': 10000,
            'published_date': '2024-01-15',
            'word_limit': 500,
            'tone': 'professional',
            'key_points': [
                'Understanding user requirements is crucial',
                'Maintainable code saves time in the long run',
                'Automated testing improves reliability'
            ]
        }


class PromptLibrary:
    """
    Library of pre-built prompt templates.
    
    Provides a collection of tested, optimized prompts for different
    content types and use cases.
    """
    
    # Blog Post Templates
    BLOG_TECHNICAL = """Transform this technical video into an informative blog post.

Video: {{ title }}
Duration: {{ duration | format_duration }}

{{ transcript | truncate_words(800) }}

Create a {{ word_limit }}-word technical blog post that:
1. Opens with a compelling problem statement
2. Explains the solution step-by-step with code examples where relevant
3. Includes practical tips and best practices
4. Ends with actionable next steps for readers

Tone: {{ tone | default('technical but accessible') }}
Target audience: Developers and technical professionals"""
    
    BLOG_TUTORIAL = """Convert this tutorial video into a step-by-step blog guide.

Video: {{ title }}
Channel: {{ channel_name }}

{{ transcript | remove_timestamps | truncate_words(1000) }}

Create a {{ word_limit }}-word tutorial blog post with:
1. Clear introduction explaining what readers will learn
2. Numbered steps with detailed explanations
3. Screenshots or diagram descriptions where helpful
4. Common pitfalls and how to avoid them
5. Summary of key takeaways

Make it practical and easy to follow."""
    
    # Summary Templates
    SUMMARY_EXECUTIVE = """Create an executive summary of this video content.

Title: {{ title }}
Duration: {{ duration | format_duration }}
Views: {{ view_count | default('N/A') }}

{{ transcript | truncate_words(500) }}

Provide:
1. One-paragraph overview (50-75 words)
2. Key findings or insights (3-5 bullet points)
3. Recommended actions (2-3 items)
4. Bottom line conclusion (1-2 sentences)

Focus on business impact and strategic value."""
    
    SUMMARY_ACADEMIC = """Create an academic-style abstract of this video content.

{{ title }}

{{ transcript | truncate_words(600) }}

Structure:
- Background/Context (2-3 sentences)
- Objectives/Research Question
- Methods/Approach
- Key Findings (3-4 points)
- Conclusions/Implications
- Keywords (5-7 relevant terms)

Use formal academic language and passive voice where appropriate."""
    
    # Social Media Templates
    SOCIAL_TWITTER_THREAD = """Create a Twitter/X thread from this video.

{{ title }}

Key points from transcript:
{{ transcript | truncate_words(300) }}

Generate:
1. Hook tweet that grabs attention (280 chars)
2. 3-5 follow-up tweets with key insights
3. Final tweet with call-to-action
4. Relevant hashtags: #tech #learning #development

Make it conversational and engaging. Use emojis sparingly."""
    
    SOCIAL_LINKEDIN = """Create a LinkedIn post from this video content.

Video: {{ title }}
Channel: {{ channel_name }}

{{ transcript | truncate_words(400) }}

Write a 150-200 word LinkedIn post that:
- Starts with a thought-provoking question or statement
- Shares 2-3 key insights
- Includes a personal reflection or industry observation
- Ends with engaging question for comments
- Uses professional but conversational tone

Include 3-5 relevant hashtags."""
    
    @classmethod
    def get_template(cls, template_name: str) -> Optional[str]:
        """Get a template by name."""
        return getattr(cls, template_name.upper(), None)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available template names."""
        return [
            name for name in dir(cls)
            if not name.startswith('_') and 
            isinstance(getattr(cls, name), str)
        ]


# Singleton instance
_template_engine = None


def get_template_engine() -> PromptTemplateEngine:
    """Get or create the template engine singleton."""
    global _template_engine
    if _template_engine is None:
        _template_engine = PromptTemplateEngine()
    return _template_engine