"""
Content generators package for the YouTube Content Intelligence & Repurposing Platform.

This package contains specialized sub-generators that transform video transcripts
into various content formats through AI-powered generation.

Available generators:
- SummaryGenerator: Short, medium, and detailed summaries
- BlogGenerator: Blog posts with configurable word counts
- SocialMediaGenerator: Platform-specific social media content
- NewsletterGenerator: Structured newsletter content with sections
- ScriptGenerator: Video scripts with duration targeting

All generators inherit from BaseGenerator and follow the same interface pattern.
"""

from .base_generator import BaseGenerator
from .summary import SummaryGenerator
from .blog import BlogGenerator
from .social import SocialMediaGenerator
from .newsletter import NewsletterGenerator
from .scripts import ScriptGenerator

__all__ = [
    'BaseGenerator',
    'SummaryGenerator', 
    'BlogGenerator',
    'SocialMediaGenerator',
    'NewsletterGenerator',
    'ScriptGenerator'
]

# Content generator registry for dynamic instantiation
CONTENT_GENERATORS = {
    'summary': {
        'class': SummaryGenerator,
        'name': 'SummaryGenerator',
        'models': ['gpt-4', 'claude-3'],
        'output_formats': ['short', 'medium', 'detailed'],
        'max_tokens': 500
    },
    'blog_post': {
        'class': BlogGenerator,
        'name': 'BlogPostGenerator', 
        'models': ['gpt-4', 'claude-3'],
        'output_formats': ['markdown', 'html'],
        'target_words': [500, 1000, 2000]
    },
    'social_media': {
        'class': SocialMediaGenerator,
        'name': 'SocialMediaGenerator',
        'models': ['gpt-3.5-turbo', 'claude-haiku'],
        'platforms': ['twitter', 'linkedin', 'facebook'],
        'constraints': {
            'twitter': 280,
            'linkedin': 3000,
            'facebook': 63206
        }
    },
    'newsletter': {
        'class': NewsletterGenerator,
        'name': 'NewsletterGenerator',
        'models': ['gpt-4', 'claude-3'],
        'sections': ['headline', 'summary', 'key_points', 'cta'],
        'output_format': 'html'
    },
    'scripts': {
        'class': ScriptGenerator,
        'name': 'ScriptGenerator',
        'models': ['gpt-4'],
        'types': ['youtube_shorts', 'tiktok', 'podcast'],
        'duration_targets': [60, 180, 600]  # seconds
    }
}


def get_generator(generator_type: str, **kwargs):
    """
    Factory function to create generator instances.
    
    Args:
        generator_type: Type of generator ('summary', 'blog_post', etc.)
        **kwargs: Additional arguments to pass to generator constructor
        
    Returns:
        Generator instance
        
    Raises:
        ValueError: If generator_type is not recognized
    """
    if generator_type not in CONTENT_GENERATORS:
        available = list(CONTENT_GENERATORS.keys())
        raise ValueError(f"Unknown generator type '{generator_type}'. Available: {available}")
    
    config = CONTENT_GENERATORS[generator_type]
    generator_class = config['class']
    
    return generator_class(name=config['name'], **kwargs)