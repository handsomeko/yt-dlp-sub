"""
Social media content generator for transforming video transcripts into platform-specific posts.

Supports multiple platforms with their specific constraints:
- Twitter: 280 characters max, hashtags, concise messaging
- LinkedIn: 3000 characters max, professional tone, industry insights
- Facebook: 63206 characters max, engaging narrative, community focus
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import re

from .base_generator import BaseGenerator, ContentFormat, GenerationQuality


class SocialPlatform(Enum):
    """Social media platforms with their character limits."""
    TWITTER = ("twitter", 280)
    LINKEDIN = ("linkedin", 3000)
    FACEBOOK = ("facebook", 63206)
    
    def __init__(self, platform_name: str, char_limit: int):
        self.platform_name = platform_name
        self.char_limit = char_limit


class PostType(Enum):
    """Types of social media posts."""
    SINGLE = "single"        # One standalone post
    THREAD = "thread"        # Multi-part series (Twitter threads, LinkedIn carousels)
    STORY = "story"         # Story format for platforms that support it
    CAROUSEL = "carousel"    # Multi-slide content


class SocialTone(Enum):
    """Tone styles for social media content."""
    PROFESSIONAL = "professional"     # Business/industry focused
    CASUAL = "casual"                 # Friendly, conversational
    EDUCATIONAL = "educational"       # Teaching/explaining
    INSPIRATIONAL = "inspirational"   # Motivational, uplifting
    HUMOROUS = "humorous"            # Light-hearted, entertaining


class SocialMediaGenerator(BaseGenerator):
    """
    Generates platform-specific social media content from video transcripts.
    
    Features:
    - Platform-specific character limits and formatting
    - Multiple post types (single, thread, carousel)
    - Tone adaptation for different audiences
    - Hashtag and mention generation
    - Automatic content truncation and optimization
    """
    
    def __init__(self, **kwargs):
        """Initialize the social media generator."""
        super().__init__(
            name="SocialMediaGenerator",
            supported_formats=[
                ContentFormat.PLAIN_TEXT,
                ContentFormat.JSON,
                ContentFormat.MARKDOWN
            ],
            max_word_count=1000,  # Max for longest platform content
            min_word_count=10,    # Min for shortest tweets
            **kwargs
        )
        
        # Social media specific configuration
        self.supported_platforms = list(SocialPlatform)
        self.supported_post_types = list(PostType)
        self.supported_tones = list(SocialTone)
        self.default_platform = SocialPlatform.TWITTER
        self.default_tone = SocialTone.CASUAL
    
    def _load_templates(self) -> None:
        """Load social media specific prompt templates."""
        self._prompt_templates = {
            "twitter_single": """Create a compelling Twitter post from this video content (max {char_limit} characters).

Transcript:
{transcript}

Requirements:
- {char_limit} characters maximum
- Include 2-3 relevant hashtags
- Engaging hook in first 10 words  
- {tone} tone
- Call to action if appropriate

Focus on the most impactful insight.""",

            "twitter_thread": """Create a Twitter thread from this video content (3-5 tweets, {char_limit} chars each).

Transcript:
{transcript}

Structure:
- Hook tweet (introduce the thread)
- 2-3 content tweets with key points
- Conclusion tweet with CTA
- {tone} tone throughout
- Thread numbering (1/n format)
- Relevant hashtags on appropriate tweets""",

            "linkedin_professional": """Create a professional LinkedIn post from this video content (max {char_limit} characters).

Transcript:
{transcript}

Requirements:
- Professional, industry-relevant tone
- Start with engaging hook or question
- 2-3 key insights with context
- Personal perspective or commentary
- End with discussion prompt
- Include relevant hashtags
- {char_limit} characters maximum""",

            "facebook_community": """Create an engaging Facebook post from this video content (max {char_limit} characters).

Transcript:
{transcript}

Style:
- Community-focused, conversational tone
- Story-driven narrative
- Multiple paragraphs for readability
- Encourage comments and discussion
- Include call-to-action
- {char_limit} characters maximum"""
        }
    
    def get_platform(self, platform_str: str) -> SocialPlatform:
        """Get SocialPlatform enum from string."""
        for platform in SocialPlatform:
            if platform.platform_name == platform_str.lower():
                return platform
        
        available = [p.platform_name for p in SocialPlatform]
        raise ValueError(f"Unknown platform '{platform_str}'. Available: {available}")
    
    def get_tone(self, tone_str: str) -> SocialTone:
        """Get SocialTone enum from string."""
        try:
            return SocialTone(tone_str.lower())
        except ValueError:
            available = [t.value for t in SocialTone]
            raise ValueError(f"Unknown tone '{tone_str}'. Available: {available}")
    
    def get_post_type(self, type_str: str) -> PostType:
        """Get PostType enum from string."""
        try:
            return PostType(type_str.lower())
        except ValueError:
            available = [t.value for t in PostType]
            raise ValueError(f"Unknown post type '{type_str}'. Available: {available}")
    
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        platform: str = "twitter",
        post_type: str = "single",
        tone: str = "casual",
        include_hashtags: bool = True,
        custom_hashtags: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate social media content from transcript.
        
        Args:
            transcript: Input video transcript
            format_type: Desired output format
            platform: Social platform ('twitter', 'linkedin', 'facebook')
            post_type: Type of post ('single', 'thread', 'carousel')
            tone: Content tone ('professional', 'casual', etc.)
            include_hashtags: Whether to include hashtags
            custom_hashtags: Custom hashtags to use
            **kwargs: Additional parameters
            
        Returns:
            Generated social media content
        """
        # Parse parameters
        try:
            social_platform = self.get_platform(platform)
        except ValueError:
            self.log_with_context(
                f"Invalid platform '{platform}', using default 'twitter'",
                level="WARNING"
            )
            social_platform = self.default_platform
        
        try:
            social_tone = self.get_tone(tone)
        except ValueError:
            self.log_with_context(
                f"Invalid tone '{tone}', using default 'casual'",
                level="WARNING"
            )
            social_tone = self.default_tone
        
        try:
            social_post_type = self.get_post_type(post_type)
        except ValueError:
            self.log_with_context(
                f"Invalid post type '{post_type}', using default 'single'",
                level="WARNING"
            )
            social_post_type = PostType.SINGLE
        
        # Try AI generation first, fallback to placeholder
        try:
            from core.ai_backend import get_ai_backend
            ai_backend = get_ai_backend()
            
            if ai_backend and ai_backend.is_available():
                return self._generate_ai_social(
                    transcript=transcript,
                    platform=social_platform,
                    post_type=social_post_type,
                    tone=social_tone,
                    format_type=format_type,
                    include_hashtags=include_hashtags,
                    custom_hashtags=custom_hashtags
                )
        except Exception as e:
            self.log_with_context(
                f"AI generation failed, using placeholder: {e}",
                level="WARNING"
            )
        
        # Fallback to placeholder content
        return self._generate_placeholder_social(
            transcript=transcript,
            platform=social_platform,
            post_type=social_post_type,
            tone=social_tone,
            format_type=format_type,
            include_hashtags=include_hashtags,
            custom_hashtags=custom_hashtags
        )
    
    def _generate_ai_social(
        self,
        transcript: str,
        platform: SocialPlatform,
        post_type: PostType,
        tone: SocialTone,
        format_type: ContentFormat,
        include_hashtags: bool,
        custom_hashtags: Optional[List[str]]
    ) -> str:
        """Generate real social media content using AI backend."""
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        # Select appropriate template based on platform and post type
        template_key = f"{platform.platform_name}_{post_type.value}"
        
        # Build template mapping
        template_map = {
            "twitter_single": "twitter_single",
            "twitter_thread": "twitter_thread",
            "linkedin_single": "linkedin_professional",
            "linkedin_thread": "linkedin_professional",
            "facebook_single": "facebook_community",
            "facebook_thread": "facebook_community"
        }
        
        # Get template or use default
        template_name = template_map.get(template_key, "twitter_single")
        template = self.get_template(template_name)
        
        # Format the prompt
        prompt = template.format(
            transcript=transcript[:4000],  # Limit transcript length for AI
            char_limit=platform.char_limit,
            tone=tone.value
        )
        
        # Add hashtag instructions if needed
        if include_hashtags:
            if custom_hashtags:
                hashtag_str = " ".join([f"#{tag.lstrip('#')}" for tag in custom_hashtags])
                prompt += f"\n\nInclude these hashtags: {hashtag_str}"
            else:
                prompt += "\n\nInclude relevant hashtags based on the content."
        
        # Generate content
        self.log_with_context(f"Generating {platform.platform_name} {post_type.value} content with AI")
        
        # Adjust token limit based on platform
        max_tokens = min(platform.char_limit * 2, 1000)  # Conservative estimate
        
        response = ai_backend.generate_content(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        if not response:
            raise ValueError("AI backend returned empty response")
        
        # Parse and structure the response
        content_data = self._parse_ai_response(
            response, platform, post_type, tone, include_hashtags
        )
        
        # Add metadata
        content_data["metadata"] = {
            "platform": platform.platform_name,
            "post_type": post_type.value,
            "tone": tone.value,
            "char_limit": platform.char_limit,
            "hashtags_included": include_hashtags,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_generated": True
        }
        
        return self._format_social_content(content_data, format_type)
    
    def _parse_ai_response(
        self,
        response: str,
        platform: SocialPlatform,
        post_type: PostType,
        tone: SocialTone,
        include_hashtags: bool
    ) -> Dict[str, Any]:
        """Parse AI response into structured social media content."""
        lines = response.strip().split('\n')
        
        if post_type == PostType.SINGLE:
            # Single post - take the entire response
            content = response.strip()
            
            # Ensure it fits within platform limits
            if len(content) > platform.char_limit:
                # Try to cut at a sentence boundary
                sentences = content.split('. ')
                truncated = ""
                for sentence in sentences:
                    if len(truncated) + len(sentence) + 2 <= platform.char_limit - 3:
                        truncated += sentence + ". "
                    else:
                        break
                content = truncated.strip() + "..."
            
            # Extract hashtags if present
            hashtags = []
            if include_hashtags:
                import re
                hashtags = re.findall(r'#\w+', content)
            
            return {
                "type": "single_post",
                "content": content,
                "character_count": len(content),
                "hashtags": hashtags
            }
        
        elif post_type == PostType.THREAD:
            # Thread - parse numbered posts or split by paragraphs
            posts = []
            current_post = ""
            post_number = 1
            
            for line in lines:
                # Check if this is a new post marker (e.g., "1/4", "Tweet 1:", etc.)
                if any(marker in line for marker in ['1/', '2/', '3/', '4/', 'Tweet', 'Post']):
                    if current_post:
                        posts.append({
                            "number": f"{post_number}/{len(lines)//3 + 1}",
                            "content": current_post.strip(),
                            "character_count": len(current_post.strip())
                        })
                        current_post = ""
                        post_number += 1
                    # Start new post with the line after the marker
                    continue
                current_post += line + " "
            
            # Add last post
            if current_post:
                posts.append({
                    "number": f"{post_number}/{post_number}",
                    "content": current_post.strip(),
                    "character_count": len(current_post.strip())
                })
            
            # If no posts were parsed, split into chunks
            if not posts:
                text_chunks = self._split_into_chunks(response, platform.char_limit - 20)
                for i, chunk in enumerate(text_chunks[:5], 1):  # Max 5 posts
                    posts.append({
                        "number": f"{i}/{len(text_chunks[:5])}",
                        "content": chunk,
                        "character_count": len(chunk)
                    })
            
            # Extract hashtags from all posts
            hashtags = []
            if include_hashtags:
                import re
                for post in posts:
                    hashtags.extend(re.findall(r'#\w+', post["content"]))
                hashtags = list(set(hashtags))  # Remove duplicates
            
            return {
                "type": "thread",
                "posts": posts,
                "total_posts": len(posts),
                "hashtags": hashtags
            }
        
        else:  # CAROUSEL or STORY
            # Parse as slides
            slides = []
            slide_number = 1
            
            # Try to identify slide boundaries
            paragraphs = response.split('\n\n')
            for para in paragraphs[:5]:  # Max 5 slides
                if para.strip():
                    # Extract title if present (first line if it's short)
                    lines = para.strip().split('\n')
                    if lines and len(lines[0]) < 50:
                        title = lines[0]
                        content = '\n'.join(lines[1:]) if len(lines) > 1 else lines[0]
                    else:
                        title = f"Point {slide_number}"
                        content = para.strip()
                    
                    slides.append({
                        "slide_number": slide_number,
                        "title": title,
                        "content": content[:500],  # Limit slide content
                        "character_count": len(content)
                    })
                    slide_number += 1
            
            # Ensure we have at least 3 slides
            while len(slides) < 3:
                slides.append({
                    "slide_number": len(slides) + 1,
                    "title": f"Additional Point {len(slides) + 1}",
                    "content": "Expand on the key concepts discussed.",
                    "character_count": 35
                })
            
            hashtags = []
            if include_hashtags:
                import re
                hashtags = re.findall(r'#\w+', response)
                hashtags = list(set(hashtags))[:5]  # Max 5 unique hashtags
            
            return {
                "type": "carousel",
                "slides": slides,
                "total_slides": len(slides),
                "hashtags": hashtags
            }
    
    def _split_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit within max_length."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (word + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_placeholder_social(
        self,
        transcript: str,
        platform: SocialPlatform,
        post_type: PostType,
        tone: SocialTone,
        format_type: ContentFormat,
        include_hashtags: bool,
        custom_hashtags: Optional[List[str]]
    ) -> str:
        """Generate placeholder social media content for Phase 1."""
        
        # Extract key info from transcript
        transcript_preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
        key_topics = self._extract_key_topics(transcript)
        
        # Generate hashtags
        hashtags = self._generate_hashtags(key_topics, platform, custom_hashtags) if include_hashtags else []
        
        # Create content based on post type
        if post_type == PostType.SINGLE:
            content_data = self._create_single_post(
                transcript_preview, platform, tone, hashtags
            )
        elif post_type == PostType.THREAD:
            content_data = self._create_thread_posts(
                transcript_preview, platform, tone, hashtags
            )
        else:  # CAROUSEL or STORY
            content_data = self._create_carousel_posts(
                transcript_preview, platform, tone, hashtags
            )
        
        # Add metadata
        content_data["metadata"] = {
            "platform": platform.platform_name,
            "post_type": post_type.value,
            "tone": tone.value,
            "char_limit": platform.char_limit,
            "hashtags_included": include_hashtags,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self._format_social_content(content_data, format_type)
    
    def _extract_key_topics(self, transcript: str) -> List[str]:
        """Extract key topics from transcript for hashtag generation."""
        # Simple keyword extraction - in Phase 2, use NLP
        words = transcript.lower().split()
        
        # Filter common words and extract potential topics
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Get longer words that might be topics
        potential_topics = [
            word.strip('.,!?;:"()[]') for word in words
            if len(word) > 3 and word not in common_words and word.isalpha()
        ]
        
        # Return first few unique topics
        unique_topics = list(dict.fromkeys(potential_topics))[:5]
        return unique_topics
    
    def _generate_hashtags(
        self,
        topics: List[str],
        platform: SocialPlatform,
        custom_hashtags: Optional[List[str]]
    ) -> List[str]:
        """Generate relevant hashtags for the content."""
        hashtags = []
        
        # Add custom hashtags first
        if custom_hashtags:
            hashtags.extend([f"#{tag.lstrip('#')}" for tag in custom_hashtags])
        
        # Add topic-based hashtags
        for topic in topics[:3]:  # Max 3 topic hashtags
            hashtags.append(f"#{topic.capitalize()}")
        
        # Add platform-specific generic hashtags
        if platform == SocialPlatform.TWITTER:
            hashtags.extend(["#VideoInsights", "#Learning"])
        elif platform == SocialPlatform.LINKEDIN:
            hashtags.extend(["#ProfessionalDevelopment", "#Industry"])
        else:  # Facebook
            hashtags.extend(["#VideoContent", "#Discussion"])
        
        # Limit hashtags based on platform conventions
        max_hashtags = {
            SocialPlatform.TWITTER: 3,
            SocialPlatform.LINKEDIN: 5,
            SocialPlatform.FACEBOOK: 3
        }
        
        return hashtags[:max_hashtags[platform]]
    
    def _create_single_post(
        self,
        content_preview: str,
        platform: SocialPlatform,
        tone: SocialTone,
        hashtags: List[str]
    ) -> Dict[str, Any]:
        """Create a single social media post."""
        
        # Generate platform-specific content
        if platform == SocialPlatform.TWITTER:
            post_content = self._create_twitter_post(content_preview, tone)
        elif platform == SocialPlatform.LINKEDIN:
            post_content = self._create_linkedin_post(content_preview, tone)
        else:  # Facebook
            post_content = self._create_facebook_post(content_preview, tone)
        
        # Add hashtags if they fit within character limit
        hashtag_text = " " + " ".join(hashtags) if hashtags else ""
        
        # Ensure content fits within platform limits
        available_chars = platform.char_limit - len(hashtag_text)
        if len(post_content) > available_chars:
            post_content = post_content[:available_chars-3] + "..."
        
        final_content = post_content + hashtag_text
        
        return {
            "type": "single_post",
            "content": final_content,
            "character_count": len(final_content),
            "hashtags": hashtags
        }
    
    def _create_thread_posts(
        self,
        content_preview: str,
        platform: SocialPlatform,
        tone: SocialTone,
        hashtags: List[str]
    ) -> Dict[str, Any]:
        """Create a thread of social media posts."""
        
        posts = []
        
        if platform == SocialPlatform.TWITTER:
            # Twitter thread
            posts = [
                {
                    "number": "1/4",
                    "content": f"🧵 Thread: Key insights from video content {hashtags[0] if hashtags else ''}",
                    "character_count": 0  # Will be calculated
                },
                {
                    "number": "2/4", 
                    "content": f"First major point: {content_preview.split('.')[0] if content_preview else 'Key insight discussed in detail.'}",
                    "character_count": 0
                },
                {
                    "number": "3/4",
                    "content": "Second important concept with practical applications and real-world examples.",
                    "character_count": 0
                },
                {
                    "number": "4/4",
                    "content": f"What are your thoughts on this? {' '.join(hashtags[-2:]) if len(hashtags) >= 2 else ''}",
                    "character_count": 0
                }
            ]
        else:
            # LinkedIn carousel-style posts
            posts = [
                {
                    "slide": 1,
                    "content": f"Key insights from recent video content:\n\n{content_preview[:500]}...",
                    "character_count": 0
                },
                {
                    "slide": 2,
                    "content": "Main takeaways and practical applications for your industry.",
                    "character_count": 0
                }
            ]
        
        # Calculate character counts
        for post in posts:
            post["character_count"] = len(post["content"])
        
        return {
            "type": "thread",
            "posts": posts,
            "total_posts": len(posts),
            "hashtags": hashtags
        }
    
    def _create_carousel_posts(
        self,
        content_preview: str,
        platform: SocialPlatform,
        tone: SocialTone,
        hashtags: List[str]
    ) -> Dict[str, Any]:
        """Create carousel-style content."""
        
        slides = [
            {
                "slide_number": 1,
                "title": "Key Video Insights",
                "content": "Main takeaways from the video content",
                "character_count": 0
            },
            {
                "slide_number": 2,
                "title": "Important Points",
                "content": f"{content_preview[:200]}{'...' if len(content_preview) > 200 else ''}",
                "character_count": 0
            },
            {
                "slide_number": 3,
                "title": "Your Takeaway",
                "content": "How can you apply these insights in your context?",
                "character_count": 0
            }
        ]
        
        # Calculate character counts
        for slide in slides:
            slide["character_count"] = len(slide["content"])
        
        return {
            "type": "carousel",
            "slides": slides,
            "total_slides": len(slides),
            "hashtags": hashtags
        }
    
    def _create_twitter_post(self, content_preview: str, tone: SocialTone) -> str:
        """Create Twitter-specific post content."""
        if tone == SocialTone.PROFESSIONAL:
            return f"Key insights from video analysis: {content_preview[:100]}... Industry implications worth considering."
        elif tone == SocialTone.EDUCATIONAL:
            return f"💡 What I learned: {content_preview[:80]}... Here's why it matters:"
        elif tone == SocialTone.INSPIRATIONAL:
            return f"🚀 This video changed my perspective: {content_preview[:90]}... Your thoughts?"
        else:  # CASUAL or HUMOROUS
            return f"Just watched this video and 🤯 {content_preview[:100]}... Anyone else think about this?"
    
    def _create_linkedin_post(self, content_preview: str, tone: SocialTone) -> str:
        """Create LinkedIn-specific post content."""
        if tone == SocialTone.PROFESSIONAL:
            return f"""I recently came across valuable insights that are relevant to our industry.

Key points discussed:
• {content_preview.split('.')[0] if content_preview else 'Primary insight from the content'}
• Practical applications for business strategy
• Long-term implications for the sector

What's your experience with these concepts? I'd love to hear different perspectives."""
        
        else:  # Other tones adapted for LinkedIn
            return f"""Thought-provoking content I wanted to share with my network.

{content_preview[:300]}{'...' if len(content_preview) > 300 else ''}

The discussion around this topic continues to evolve. What are your thoughts?"""
    
    def _create_facebook_post(self, content_preview: str, tone: SocialTone) -> str:
        """Create Facebook-specific post content."""
        return f"""I just watched an interesting video that got me thinking...

{content_preview[:400]}{'...' if len(content_preview) > 400 else ''}

The points raised really resonated with me, especially around [key topic area]. It's fascinating how these ideas connect to what we see happening in our daily lives.

What do you all think? Have you noticed similar patterns or trends? I'd love to start a conversation about this!

Drop your thoughts in the comments below. 👇"""
    
    def _format_social_content(self, content_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """Format social content based on output type."""
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(content_data, indent=2)
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_social(content_data)
        else:  # PLAIN_TEXT
            return self._format_text_social(content_data)
    
    def _format_markdown_social(self, content_data: Dict[str, Any]) -> str:
        """Format social content as Markdown."""
        lines = []
        
        # Header
        metadata = content_data.get("metadata", {})
        platform = metadata.get("platform", "social").title()
        post_type = metadata.get("post_type", "single").title()
        
        lines.append(f"# {platform} {post_type}")
        lines.append("")
        
        # Metadata
        lines.append(f"**Platform:** {platform}")
        lines.append(f"**Type:** {post_type}")
        lines.append(f"**Tone:** {metadata.get('tone', 'casual').title()}")
        lines.append(f"**Character Limit:** {metadata.get('char_limit', 'N/A')}")
        lines.append("")
        
        # Content
        if content_data["type"] == "single_post":
            lines.append("## Content")
            lines.append("")
            lines.append(f"```\n{content_data['content']}\n```")
            lines.append("")
            lines.append(f"**Character Count:** {content_data['character_count']}")
        
        elif content_data["type"] == "thread":
            lines.append("## Thread Posts")
            for post in content_data["posts"]:
                if "number" in post:
                    lines.append(f"### Post {post['number']}")
                else:
                    lines.append(f"### Slide {post.get('slide', 1)}")
                lines.append("")
                lines.append(f"```\n{post['content']}\n```")
                lines.append("")
                lines.append(f"*Characters: {post['character_count']}*")
                lines.append("")
        
        elif content_data["type"] == "carousel":
            lines.append("## Carousel Slides")
            for slide in content_data["slides"]:
                lines.append(f"### Slide {slide['slide_number']}: {slide['title']}")
                lines.append("")
                lines.append(slide['content'])
                lines.append("")
        
        # Hashtags
        if content_data.get("hashtags"):
            lines.append("## Hashtags")
            lines.append("")
            lines.append(" ".join(content_data["hashtags"]))
        
        return "\n".join(lines)
    
    def _format_text_social(self, content_data: Dict[str, Any]) -> str:
        """Format social content as plain text."""
        lines = []
        
        # Header
        metadata = content_data.get("metadata", {})
        platform = metadata.get("platform", "social").upper()
        post_type = metadata.get("post_type", "single").upper()
        
        lines.append(f"{platform} {post_type}")
        lines.append("=" * len(f"{platform} {post_type}"))
        lines.append("")
        
        # Content
        if content_data["type"] == "single_post":
            lines.append(content_data["content"])
            lines.append("")
            lines.append(f"Characters: {content_data['character_count']}")
        
        elif content_data["type"] == "thread":
            for i, post in enumerate(content_data["posts"], 1):
                lines.append(f"POST {i}")
                lines.append("-" * 10)
                lines.append(post["content"])
                lines.append(f"Characters: {post['character_count']}")
                lines.append("")
        
        elif content_data["type"] == "carousel":
            for slide in content_data["slides"]:
                lines.append(f"SLIDE {slide['slide_number']}: {slide['title'].upper()}")
                lines.append("-" * 30)
                lines.append(slide["content"])
                lines.append("")
        
        return "\n".join(lines)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate social media generation input."""
        if not super().validate_input(input_data):
            return False
        
        # Validate platform if specified
        if "platform" in input_data:
            try:
                self.get_platform(input_data["platform"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate tone if specified
        if "tone" in input_data:
            try:
                self.get_tone(input_data["tone"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate post type if specified
        if "post_type" in input_data:
            try:
                self.get_post_type(input_data["post_type"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        return True
    
    def get_platform_limits(self) -> Dict[str, int]:
        """Get character limits for all supported platforms."""
        return {platform.platform_name: platform.char_limit for platform in SocialPlatform}
    
    def get_supported_configurations(self) -> Dict[str, Any]:
        """Get all supported social media configurations."""
        return {
            "platforms": {p.platform_name: p.char_limit for p in SocialPlatform},
            "post_types": [t.value for t in PostType],
            "tones": [t.value for t in SocialTone],
            "formats": [fmt.value for fmt in self.supported_formats]
        }