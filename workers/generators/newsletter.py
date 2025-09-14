"""
Newsletter generator for transforming video transcripts into structured newsletter content.

Creates newsletters with standard sections:
- Headline: Compelling subject line
- Summary: Brief overview paragraph
- Key Points: Bullet points or numbered list
- CTA: Call-to-action section

Supports multiple newsletter formats and styles for different audiences.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

from .base_generator import BaseGenerator, ContentFormat, GenerationQuality


class NewsletterStyle(Enum):
    """Newsletter style options for different audiences."""
    PROFESSIONAL = "professional"     # Business/corporate newsletter
    CASUAL = "casual"                 # Friendly, personal newsletter  
    EDUCATIONAL = "educational"       # Learning-focused content
    PROMOTIONAL = "promotional"       # Marketing/sales focused


class NewsletterLength(Enum):
    """Newsletter length options."""
    BRIEF = ("brief", 200)           # Quick digest format
    STANDARD = ("standard", 400)     # Regular newsletter length
    COMPREHENSIVE = ("comprehensive", 800)  # Detailed analysis
    
    def __init__(self, name: str, target_words: int):
        self.newsletter_name = name
        self.target_words = target_words


class CTAType(Enum):
    """Types of call-to-action options."""
    WATCH_VIDEO = "watch_video"      # Link back to original video
    SUBSCRIBE = "subscribe"          # Newsletter subscription
    ENGAGE = "engage"               # Social media engagement
    LEARN_MORE = "learn_more"       # Additional resources
    CUSTOM = "custom"               # Custom action


class NewsletterGenerator(BaseGenerator):
    """
    Generates structured newsletter content from video transcripts.
    
    Features:
    - Structured sections: headline, summary, key points, CTA
    - Multiple styles for different audiences
    - Length options from brief digest to comprehensive analysis
    - HTML and plain text output formats
    - Configurable call-to-action types
    - Email-ready formatting
    """
    
    def __init__(self, **kwargs):
        """Initialize the newsletter generator."""
        super().__init__(
            name="NewsletterGenerator",
            supported_formats=[
                ContentFormat.PLAIN_TEXT,
                ContentFormat.HTML,
                ContentFormat.MARKDOWN,
                ContentFormat.JSON
            ],
            max_word_count=1000,  # Max for comprehensive newsletters
            min_word_count=150,   # Min for brief newsletters
            **kwargs
        )
        
        # Newsletter-specific configuration
        self.supported_styles = list(NewsletterStyle)
        self.supported_lengths = list(NewsletterLength)
        self.supported_cta_types = list(CTAType)
        self.default_style = NewsletterStyle.PROFESSIONAL
        self.default_length = NewsletterLength.STANDARD
        self.default_cta = CTAType.WATCH_VIDEO
    
    def _load_templates(self) -> None:
        """Load newsletter-specific prompt templates."""
        self._prompt_templates = {
            "professional_standard": """Create a professional newsletter from this video content (approximately {target_words} words).

Transcript:
{transcript}

Structure:
- Compelling subject line/headline
- Executive summary paragraph
- 3-5 key insights with brief explanations
- Professional call-to-action
- Maintain business-appropriate tone

Target: {target_words} words""",

            "casual_brief": """Create a friendly, casual newsletter digest from this video content (approximately {target_words} words).

Transcript:
{transcript}

Style:
- Engaging, conversational headline
- Personal summary paragraph
- Top 3 takeaways in accessible language
- Friendly call-to-action
- Warm, approachable tone

Target: {target_words} words""",

            "educational_comprehensive": """Create an educational newsletter from this video content (approximately {target_words} words).

Transcript:
{transcript}

Include:
- Learning-focused headline
- What readers will discover
- Detailed breakdown of concepts
- Practical applications
- Next steps for continued learning

Target: {target_words} words"""
        }
    
    def get_newsletter_style(self, style_str: str) -> NewsletterStyle:
        """Get NewsletterStyle enum from string."""
        try:
            return NewsletterStyle(style_str.lower())
        except ValueError:
            available = [s.value for s in NewsletterStyle]
            raise ValueError(f"Unknown newsletter style '{style_str}'. Available: {available}")
    
    def get_newsletter_length(self, length_str: str) -> NewsletterLength:
        """Get NewsletterLength enum from string."""
        for length in NewsletterLength:
            if length.newsletter_name == length_str.lower():
                return length
        
        available = [l.newsletter_name for l in NewsletterLength]
        raise ValueError(f"Unknown newsletter length '{length_str}'. Available: {available}")
    
    def get_cta_type(self, cta_str: str) -> CTAType:
        """Get CTAType enum from string."""
        try:
            return CTAType(cta_str.lower())
        except ValueError:
            available = [c.value for c in CTAType]
            raise ValueError(f"Unknown CTA type '{cta_str}'. Available: {available}")
    
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        style: str = "professional",
        length: str = "standard", 
        cta_type: str = "watch_video",
        custom_cta_text: Optional[str] = None,
        custom_headline: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate newsletter content from transcript.
        
        Args:
            transcript: Input video transcript
            format_type: Desired output format
            style: Newsletter style ('professional', 'casual', etc.)
            length: Newsletter length ('brief', 'standard', 'comprehensive')
            cta_type: Call-to-action type ('watch_video', 'subscribe', etc.)
            custom_cta_text: Custom CTA text override
            custom_headline: Custom headline override
            **kwargs: Additional parameters
            
        Returns:
            Generated newsletter content
        """
        # Parse parameters
        try:
            newsletter_style = self.get_newsletter_style(style)
        except ValueError:
            self.log_with_context(
                f"Invalid newsletter style '{style}', using default 'professional'",
                level="WARNING"
            )
            newsletter_style = self.default_style
        
        try:
            newsletter_length = self.get_newsletter_length(length)
        except ValueError:
            self.log_with_context(
                f"Invalid newsletter length '{length}', using default 'standard'",
                level="WARNING"
            )
            newsletter_length = self.default_length
        
        try:
            cta_type_enum = self.get_cta_type(cta_type)
        except ValueError:
            self.log_with_context(
                f"Invalid CTA type '{cta_type}', using default 'watch_video'",
                level="WARNING"
            )
            cta_type_enum = self.default_cta
        
        # Try AI generation first, fallback to placeholder
        try:
            from core.ai_backend import get_ai_backend
            ai_backend = get_ai_backend()
            
            if ai_backend and ai_backend.is_available():
                return self._generate_ai_newsletter(
                    transcript=transcript,
                    style=newsletter_style,
                    length=newsletter_length,
                    cta_type=cta_type_enum,
                    format_type=format_type,
                    custom_cta_text=custom_cta_text,
                    custom_headline=custom_headline
                )
        except Exception as e:
            self.log_with_context(
                f"AI generation failed, using placeholder: {e}",
                level="WARNING"
            )
        
        # Fallback to placeholder content
        return self._generate_placeholder_newsletter(
            transcript=transcript,
            style=newsletter_style,
            length=newsletter_length,
            cta_type=cta_type_enum,
            format_type=format_type,
            custom_cta_text=custom_cta_text,
            custom_headline=custom_headline
        )
    
    def _generate_ai_newsletter(
        self,
        transcript: str,
        style: NewsletterStyle,
        length: NewsletterLength,
        cta_type: CTAType,
        format_type: ContentFormat,
        custom_cta_text: Optional[str],
        custom_headline: Optional[str]
    ) -> str:
        """Generate real newsletter content using AI backend."""
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        # Select appropriate template based on style and length
        template_key = f"{style.value}_{length.newsletter_name}"
        
        # Build template mapping with fallbacks
        template_map = {
            "professional_brief": "professional_standard",
            "professional_standard": "professional_standard",
            "professional_comprehensive": "professional_standard",
            "casual_brief": "casual_brief",
            "casual_standard": "casual_brief",
            "casual_comprehensive": "casual_brief",
            "educational_brief": "educational_comprehensive",
            "educational_standard": "educational_comprehensive",
            "educational_comprehensive": "educational_comprehensive",
            "promotional_brief": "casual_brief",
            "promotional_standard": "professional_standard",
            "promotional_comprehensive": "educational_comprehensive"
        }
        
        # Get template or use default
        template_name = template_map.get(template_key, "professional_standard")
        template = self.get_template(template_name)
        
        # Format the prompt
        prompt = template.format(
            transcript=transcript[:6000],  # Limit transcript length for AI
            target_words=length.target_words
        )
        
        # Add custom instructions if provided
        if custom_headline:
            prompt += f"\n\nUse this headline: {custom_headline}"
        
        if custom_cta_text:
            prompt += f"\n\nInclude this call-to-action: {custom_cta_text}"
        else:
            # Add CTA type instructions
            cta_instructions = {
                CTAType.WATCH_VIDEO: "Include a call-to-action to watch the original video.",
                CTAType.SUBSCRIBE: "Include a call-to-action to subscribe to the newsletter.",
                CTAType.ENGAGE: "Include a call-to-action for social media engagement.",
                CTAType.LEARN_MORE: "Include a call-to-action to explore additional resources.",
                CTAType.CUSTOM: "Include an appropriate call-to-action."
            }
            prompt += f"\n\n{cta_instructions.get(cta_type, cta_instructions[CTAType.WATCH_VIDEO])}"
        
        # Generate content
        self.log_with_context(f"Generating {style.value} {length.newsletter_name} newsletter with AI")
        
        # Adjust token limit based on length
        max_tokens = length.target_words * 2  # Conservative estimate
        
        response = ai_backend.generate_content(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        if not response:
            raise ValueError("AI backend returned empty response")
        
        # Parse and structure the response
        newsletter_data = self._parse_ai_newsletter(
            response, style, length, cta_type, custom_cta_text
        )
        
        # Add metadata
        newsletter_data["metadata"] = {
            "style": style.value,
            "length": length.newsletter_name,
            "target_words": length.target_words,
            "cta_type": cta_type.value,
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
            "source_words": self.count_words(transcript),
            "ai_generated": True
        }
        
        # Calculate actual word count
        content_text = f"{newsletter_data['summary']} {' '.join(newsletter_data['key_points'])} {newsletter_data['cta']['text']}"
        newsletter_data["metadata"]["actual_words"] = self.count_words(content_text)
        
        return self._format_newsletter_content(newsletter_data, format_type)
    
    def _parse_ai_newsletter(
        self,
        response: str,
        style: NewsletterStyle,
        length: NewsletterLength,
        cta_type: CTAType,
        custom_cta_text: Optional[str]
    ) -> Dict[str, Any]:
        """Parse AI response into structured newsletter sections."""
        lines = response.strip().split('\n')
        
        # Initialize sections
        headline = ""
        summary = ""
        key_points = []
        cta_text = custom_cta_text or ""
        
        # Track current section
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if any(h in line.lower() for h in ['subject', 'headline', 'title']):
                current_section = 'headline'
                continue
            elif any(s in line.lower() for s in ['summary', 'overview', 'introduction']):
                current_section = 'summary'
                continue
            elif any(k in line.lower() for k in ['key points', 'key insights', 'takeaways', 'highlights']):
                current_section = 'key_points'
                continue
            elif any(c in line.lower() for c in ['call-to-action', 'cta', 'action', 'next steps']):
                current_section = 'cta'
                continue
            
            # Add content to appropriate section
            if current_section == 'headline' and line:
                headline = line.replace(':', '').strip()
            elif current_section == 'summary' and line:
                summary += line + " "
            elif current_section == 'key_points' and line:
                # Handle bullet points or numbered lists
                if line.startswith(('•', '-', '*', '1', '2', '3', '4', '5')):
                    # Clean up the bullet/number and add
                    clean_point = line.lstrip('•-*1234567890. ')
                    if clean_point:
                        key_points.append(clean_point)
                elif len(key_points) > 0 and not line.startswith(('•', '-', '*')):
                    # Continuation of previous point
                    key_points[-1] += " " + line
                elif line:
                    # New point without bullet
                    key_points.append(line)
            elif current_section == 'cta' and line and not custom_cta_text:
                cta_text += line + " "
        
        # Fallback if sections weren't clearly identified
        if not headline:
            # Try to extract first line as headline
            if lines:
                headline = lines[0].strip()
        
        if not summary and len(lines) > 1:
            # Extract paragraph after headline as summary
            for i, line in enumerate(lines[1:], 1):
                if line.strip() and len(line.strip()) > 50:
                    summary = line.strip()
                    break
        
        if not key_points:
            # Extract middle content as key points
            middle_start = len(lines) // 3
            middle_end = 2 * len(lines) // 3
            for line in lines[middle_start:middle_end]:
                if line.strip() and len(line.strip()) > 20:
                    key_points.append(line.strip())
                    if len(key_points) >= 4:
                        break
        
        if not cta_text:
            # Generate default CTA based on type
            cta_text = self._generate_default_cta(cta_type, style)
        
        # Ensure we have at least 3 key points
        while len(key_points) < 3:
            key_points.append(f"Additional insight from the video content")
        
        # Limit key points based on newsletter length
        max_points = {
            NewsletterLength.BRIEF: 3,
            NewsletterLength.STANDARD: 5,
            NewsletterLength.COMPREHENSIVE: 7
        }
        key_points = key_points[:max_points.get(length, 5)]
        
        return {
            "headline": headline or "Newsletter: Key Video Insights",
            "summary": summary.strip() or "This edition covers important topics from recent video content.",
            "key_points": key_points,
            "cta": {
                "type": cta_type.value,
                "text": cta_text.strip(),
                "button_text": self._get_cta_button_text(cta_type)
            }
        }
    
    def _generate_default_cta(self, cta_type: CTAType, style: NewsletterStyle) -> str:
        """Generate default CTA text based on type and style."""
        if cta_type == CTAType.WATCH_VIDEO:
            if style == NewsletterStyle.PROFESSIONAL:
                return "Watch the complete video for deeper insights and detailed analysis."
            else:
                return "Check out the full video for more great content!"
        elif cta_type == CTAType.SUBSCRIBE:
            if style == NewsletterStyle.PROFESSIONAL:
                return "Subscribe to our newsletter for weekly industry insights."
            else:
                return "Join our community! Subscribe for more awesome content."
        elif cta_type == CTAType.ENGAGE:
            return "Share your thoughts on social media and join the conversation."
        elif cta_type == CTAType.LEARN_MORE:
            return "Explore additional resources and continue your learning journey."
        else:
            return "Take action today and apply these insights."
    
    def _get_cta_button_text(self, cta_type: CTAType) -> str:
        """Get appropriate button text for CTA type."""
        button_map = {
            CTAType.WATCH_VIDEO: "Watch Now",
            CTAType.SUBSCRIBE: "Subscribe",
            CTAType.ENGAGE: "Join Discussion",
            CTAType.LEARN_MORE: "Learn More",
            CTAType.CUSTOM: "Take Action"
        }
        return button_map.get(cta_type, "Click Here")
    
    def _generate_placeholder_newsletter(
        self,
        transcript: str,
        style: NewsletterStyle,
        length: NewsletterLength,
        cta_type: CTAType,
        format_type: ContentFormat,
        custom_cta_text: Optional[str],
        custom_headline: Optional[str]
    ) -> str:
        """Generate placeholder newsletter content for Phase 1."""
        
        # Extract key info from transcript
        transcript_preview = transcript[:300] + "..." if len(transcript) > 300 else transcript
        key_topics = self._extract_newsletter_topics(transcript)
        
        # Create newsletter structure
        newsletter_data = {
            "headline": custom_headline or self._generate_headline(transcript_preview, style),
            "summary": self._generate_summary(transcript_preview, style, length),
            "key_points": self._generate_key_points(key_topics, style, length),
            "cta": self._generate_cta(cta_type, custom_cta_text, style),
            "metadata": {
                "style": style.value,
                "length": length.newsletter_name,
                "target_words": length.target_words,
                "cta_type": cta_type.value,
                "generated_date": datetime.now().strftime("%Y-%m-%d"),
                "source_words": self.count_words(transcript)
            }
        }
        
        # Calculate actual word count
        content_text = f"{newsletter_data['summary']} {' '.join(newsletter_data['key_points'])} {newsletter_data['cta']['text']}"
        newsletter_data["metadata"]["actual_words"] = self.count_words(content_text)
        
        return self._format_newsletter_content(newsletter_data, format_type)
    
    def _extract_newsletter_topics(self, transcript: str) -> List[str]:
        """Extract key topics for newsletter key points."""
        # Simple topic extraction for Phase 1
        sentences = transcript.split('.')[:5]  # First 5 sentences
        topics = []
        
        for sentence in sentences:
            if sentence.strip():
                # Take first meaningful part of each sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:
                    topics.append(clean_sentence[:100] + "..." if len(clean_sentence) > 100 else clean_sentence)
        
        return topics[:4]  # Limit to 4 topics
    
    def _generate_headline(self, content_preview: str, style: NewsletterStyle) -> str:
        """Generate style-appropriate headline."""
        if style == NewsletterStyle.PROFESSIONAL:
            return "Weekly Industry Insights: Key Takeaways from Recent Analysis"
        elif style == NewsletterStyle.CASUAL:
            return "🎯 This Week's Video Had Me Thinking..."
        elif style == NewsletterStyle.EDUCATIONAL:
            return "Learn This Week: Essential Concepts Explained"
        else:  # PROMOTIONAL
            return "🔥 Don't Miss These Game-Changing Insights"
    
    def _generate_summary(
        self,
        content_preview: str,
        style: NewsletterStyle,
        length: NewsletterLength
    ) -> str:
        """Generate style and length appropriate summary."""
        
        base_content = content_preview[:150] + "..." if len(content_preview) > 150 else content_preview
        
        if style == NewsletterStyle.PROFESSIONAL:
            if length == NewsletterLength.BRIEF:
                return f"This week's analysis focuses on {base_content.lower()}. Key insights and strategic implications are outlined below."
            elif length == NewsletterLength.STANDARD:
                return f"Our latest content review examines important developments in {base_content.lower()}. This edition provides actionable insights for industry professionals, covering strategic implications and practical applications relevant to current market conditions."
            else:  # COMPREHENSIVE
                return f"This comprehensive analysis explores the multifaceted aspects of {base_content.lower()}. Our detailed examination covers both immediate implications and long-term strategic considerations, providing executives and decision-makers with the context needed for informed planning. The following insights synthesize complex information into actionable intelligence."
        
        elif style == NewsletterStyle.CASUAL:
            if length == NewsletterLength.BRIEF:
                return f"Hey there! Quick update on something interesting: {base_content}. Here are the highlights!"
            elif length == NewsletterLength.STANDARD:
                return f"Hi friends! I came across some fascinating content about {base_content.lower()}, and I couldn't wait to share it with you. There's some really thought-provoking stuff here that I think you'll find valuable. Let me break down the key points that stood out to me."
            else:  # COMPREHENSIVE
                return f"Hello wonderful readers! This week I dove deep into some content that really got me thinking about {base_content.lower()}. I spent some time analyzing the key themes and wanted to share a comprehensive breakdown with all of you. There's a lot to unpack here, so grab your favorite beverage and let's explore these ideas together. I think you'll find some genuine insights that might change how you think about this topic."
        
        elif style == NewsletterStyle.EDUCATIONAL:
            if length == NewsletterLength.BRIEF:
                return f"Today's lesson focuses on understanding {base_content.lower()}. Let's explore the key concepts together."
            elif length == NewsletterLength.STANDARD:
                return f"Welcome to this week's learning digest! Today we're exploring the important concepts around {base_content.lower()}. By the end of this newsletter, you'll have a clear understanding of the fundamental principles and be able to apply these insights in practical contexts. Let's start with the foundational concepts."
            else:  # COMPREHENSIVE
                return f"Welcome to our comprehensive learning guide on {base_content.lower()}. This in-depth exploration is designed to take you from basic understanding to advanced application. We'll cover theoretical foundations, examine real-world examples, and provide practical exercises to reinforce your learning. Whether you're new to these concepts or looking to deepen your expertise, this guide offers structured insights that build upon each other progressively."
        
        else:  # PROMOTIONAL
            return f"🚀 Exciting insights await! We've discovered game-changing information about {base_content.lower()}. Don't miss out on these exclusive takeaways that could transform your approach."
    
    def _generate_key_points(
        self,
        topics: List[str],
        style: NewsletterStyle,
        length: NewsletterLength
    ) -> List[str]:
        """Generate key points based on extracted topics."""
        
        points = []
        num_points = {
            NewsletterLength.BRIEF: 3,
            NewsletterLength.STANDARD: 4,
            NewsletterLength.COMPREHENSIVE: 5
        }[length]
        
        for i, topic in enumerate(topics[:num_points], 1):
            if style == NewsletterStyle.PROFESSIONAL:
                point = f"Strategic consideration #{i}: Analysis indicates {topic.lower()}. This development presents both opportunities and challenges for industry stakeholders."
            elif style == NewsletterStyle.CASUAL:
                point = f"💡 Insight #{i}: {topic}. This really resonated with me because it connects to something we all experience."
            elif style == NewsletterStyle.EDUCATIONAL:
                point = f"Key Concept #{i}: {topic}. Understanding this principle helps explain the broader framework we're discussing."
            else:  # PROMOTIONAL
                point = f"🎯 Game-Changer #{i}: {topic}. This insight alone could revolutionize your approach!"
            
            points.append(point)
        
        # Add filler points if we don't have enough topics
        while len(points) < num_points:
            points.append("Additional insight extracted from the comprehensive content analysis.")
        
        return points
    
    def _generate_cta(
        self,
        cta_type: CTAType,
        custom_text: Optional[str],
        style: NewsletterStyle
    ) -> Dict[str, str]:
        """Generate call-to-action based on type and style."""
        
        if custom_text:
            return {
                "type": cta_type.value,
                "text": custom_text,
                "action": "custom"
            }
        
        cta_configs = {
            CTAType.WATCH_VIDEO: {
                NewsletterStyle.PROFESSIONAL: {
                    "text": "Access the complete video analysis for additional strategic insights and detailed context.",
                    "action": "View Full Analysis →"
                },
                NewsletterStyle.CASUAL: {
                    "text": "Want to see the full video? I highly recommend checking it out - there's so much more good stuff!",
                    "action": "Watch Now 🎥"
                },
                NewsletterStyle.EDUCATIONAL: {
                    "text": "Continue your learning journey by viewing the complete video content for comprehensive understanding.",
                    "action": "Continue Learning →"
                },
                NewsletterStyle.PROMOTIONAL: {
                    "text": "🔥 Don't miss the full video - it's packed with exclusive insights you won't find anywhere else!",
                    "action": "Get Exclusive Access →"
                }
            },
            CTAType.SUBSCRIBE: {
                NewsletterStyle.PROFESSIONAL: {
                    "text": "Stay informed with weekly industry insights delivered directly to your inbox.",
                    "action": "Subscribe to Updates →"
                },
                NewsletterStyle.CASUAL: {
                    "text": "Love these insights? Join our community and never miss an update!",
                    "action": "Join the Community 🤝"
                },
                NewsletterStyle.EDUCATIONAL: {
                    "text": "Subscribe for regular learning content and expand your knowledge with weekly insights.",
                    "action": "Subscribe for Learning →"
                },
                NewsletterStyle.PROMOTIONAL: {
                    "text": "🎯 Get exclusive insights delivered weekly - join thousands of smart subscribers!",
                    "action": "Get VIP Access →"
                }
            },
            CTAType.ENGAGE: {
                NewsletterStyle.PROFESSIONAL: {
                    "text": "Share your professional perspective and connect with industry peers in the discussion.",
                    "action": "Join the Discussion →"
                },
                NewsletterStyle.CASUAL: {
                    "text": "What did you think about these points? I'd love to hear your thoughts and experiences!",
                    "action": "Share Your Thoughts 💭"
                },
                NewsletterStyle.EDUCATIONAL: {
                    "text": "Test your understanding and engage with fellow learners in our study group.",
                    "action": "Join Study Group →"
                },
                NewsletterStyle.PROMOTIONAL: {
                    "text": "🚀 Share these insights with your network and start meaningful conversations!",
                    "action": "Share & Discuss →"
                }
            }
        }
        
        # Default to WATCH_VIDEO if type not found
        config = cta_configs.get(cta_type, cta_configs[CTAType.WATCH_VIDEO])
        style_config = config.get(style, config[NewsletterStyle.PROFESSIONAL])
        
        return {
            "type": cta_type.value,
            "text": style_config["text"],
            "action": style_config["action"]
        }
    
    def _format_newsletter_content(self, newsletter_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """Format newsletter content based on output type."""
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(newsletter_data, indent=2)
        elif format_type == ContentFormat.HTML:
            return self._format_html_newsletter(newsletter_data)
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_newsletter(newsletter_data)
        else:  # PLAIN_TEXT
            return self._format_text_newsletter(newsletter_data)
    
    def _format_html_newsletter(self, newsletter_data: Dict[str, Any]) -> str:
        """Format newsletter as HTML email."""
        html_parts = []
        
        # Email wrapper
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html>')
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append(f'<title>{newsletter_data["headline"]}</title>')
        html_parts.append('</head>')
        html_parts.append('<body style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 0 auto; padding: 20px;">')
        
        # Header
        html_parts.append(f'<h1 style="color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">{newsletter_data["headline"]}</h1>')
        
        # Summary
        html_parts.append(f'<p style="font-size: 16px; color: #555; margin: 20px 0;">{newsletter_data["summary"]}</p>')
        
        # Key Points
        html_parts.append('<h2 style="color: #333; margin-top: 30px;">Key Insights</h2>')
        html_parts.append('<ul style="list-style-type: none; padding: 0;">')
        
        for point in newsletter_data["key_points"]:
            html_parts.append(f'<li style="background: #f9f9f9; margin: 10px 0; padding: 15px; border-left: 4px solid #4CAF50;">{point}</li>')
        
        html_parts.append('</ul>')
        
        # CTA
        cta = newsletter_data["cta"]
        html_parts.append('<div style="background: #f0f0f0; padding: 20px; margin: 30px 0; text-align: center; border-radius: 5px;">')
        html_parts.append(f'<p style="margin: 0 0 15px 0;">{cta["text"]}</p>')
        html_parts.append(f'<a href="#" style="display: inline-block; background: #4CAF50; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">{cta["action"]}</a>')
        html_parts.append('</div>')
        
        # Footer
        metadata = newsletter_data["metadata"]
        html_parts.append('<hr style="margin: 40px 0 20px 0; border: none; border-top: 1px solid #ddd;">')
        html_parts.append(f'<p style="font-size: 12px; color: #888; text-align: center;">Generated on {metadata["generated_date"]} | {metadata["actual_words"]} words | {metadata["style"].title()} style</p>')
        
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        return '\n'.join(html_parts)
    
    def _format_markdown_newsletter(self, newsletter_data: Dict[str, Any]) -> str:
        """Format newsletter as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# {newsletter_data['headline']}")
        lines.append("")
        
        # Metadata
        metadata = newsletter_data["metadata"]
        lines.append(f"*{metadata['style'].title()} Newsletter • {metadata['actual_words']} words • {metadata['generated_date']}*")
        lines.append("")
        
        # Summary
        lines.append(newsletter_data["summary"])
        lines.append("")
        
        # Key Points
        lines.append("## Key Insights")
        lines.append("")
        
        for i, point in enumerate(newsletter_data["key_points"], 1):
            lines.append(f"{i}. {point}")
            lines.append("")
        
        # CTA
        cta = newsletter_data["cta"]
        lines.append("## Take Action")
        lines.append("")
        lines.append(cta["text"])
        lines.append("")
        lines.append(f"**[{cta['action']}](#)**")
        
        return "\n".join(lines)
    
    def _format_text_newsletter(self, newsletter_data: Dict[str, Any]) -> str:
        """Format newsletter as plain text."""
        lines = []
        
        # Header
        headline = newsletter_data["headline"]
        lines.append(headline.upper())
        lines.append("=" * len(headline))
        lines.append("")
        
        # Summary
        lines.append(newsletter_data["summary"])
        lines.append("")
        
        # Key Points
        lines.append("KEY INSIGHTS:")
        lines.append("-" * 15)
        lines.append("")
        
        for i, point in enumerate(newsletter_data["key_points"], 1):
            lines.append(f"{i}. {point}")
            lines.append("")
        
        # CTA
        cta = newsletter_data["cta"]
        lines.append("TAKE ACTION:")
        lines.append("-" * 12)
        lines.append("")
        lines.append(cta["text"])
        lines.append("")
        lines.append(f">>> {cta['action']} <<<")
        
        return "\n".join(lines)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate newsletter generation input."""
        if not super().validate_input(input_data):
            return False
        
        # Validate style if specified
        if "style" in input_data:
            try:
                self.get_newsletter_style(input_data["style"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate length if specified
        if "length" in input_data:
            try:
                self.get_newsletter_length(input_data["length"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate CTA type if specified
        if "cta_type" in input_data:
            try:
                self.get_cta_type(input_data["cta_type"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Ensure transcript is substantial enough for newsletter
        transcript = input_data["transcript"]
        if len(transcript.split()) < 30:
            self.log_with_context(
                "Transcript too short for meaningful newsletter (minimum 30 words)",
                level="ERROR"
            )
            return False
        
        return True
    
    def get_supported_configurations(self) -> Dict[str, Any]:
        """Get all supported newsletter configurations."""
        return {
            "styles": [style.value for style in NewsletterStyle],
            "lengths": {length.newsletter_name: length.target_words for length in NewsletterLength},
            "cta_types": [cta.value for cta in CTAType],
            "formats": [fmt.value for fmt in self.supported_formats]
        }