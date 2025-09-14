"""
Script generator for transforming video transcripts into various video script formats.

Creates scripts with duration targeting:
- 60 seconds: Short-form content (YouTube Shorts, TikTok)
- 180 seconds: Medium-form content (Instagram Reels, Twitter videos)  
- 600 seconds: Long-form content (YouTube videos, Podcasts)

Supports different script types optimized for various platforms and audiences.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import re

from .base_generator import BaseGenerator, ContentFormat, GenerationQuality


class ScriptType(Enum):
    """Types of video scripts for different platforms."""
    YOUTUBE_SHORTS = "youtube_shorts"    # Vertical, fast-paced, hook-heavy
    TIKTOK = "tiktok"                   # Trend-aware, engaging, viral-focused
    PODCAST = "podcast"                 # Audio-only, conversational, detailed
    INSTAGRAM_REEL = "instagram_reel"   # Visual, music-friendly, branded
    EXPLAINER = "explainer"             # Educational, clear structure, step-by-step
    TESTIMONIAL = "testimonial"         # Personal story, credible, emotional


class ScriptDuration(Enum):
    """Script duration targets with timing specifications."""
    SHORT = ("short", 60, 150, 180)      # 60s target, 150 words, 180 WPM
    MEDIUM = ("medium", 180, 450, 150)    # 180s target, 450 words, 150 WPM  
    LONG = ("long", 600, 1200, 120)      # 600s target, 1200 words, 120 WPM
    
    def __init__(self, name: str, seconds: int, target_words: int, wpm: int):
        self.duration_name = name
        self.seconds = seconds
        self.target_words = target_words
        self.words_per_minute = wpm


class ScriptStyle(Enum):
    """Script writing styles for different tones."""
    ENERGETIC = "energetic"        # High energy, excited delivery
    CONVERSATIONAL = "conversational"  # Natural, friendly dialogue
    AUTHORITATIVE = "authoritative"    # Expert, confident, informative
    STORYTELLING = "storytelling"      # Narrative-driven, engaging
    EDUCATIONAL = "educational"        # Clear, structured learning


class ScriptGenerator(BaseGenerator):
    """
    Generates video scripts from transcripts with duration targeting and platform optimization.
    
    Features:
    - Duration targeting (60s, 180s, 600s) with word count calculation
    - Platform-specific formatting and hooks
    - Multiple script types for different content styles
    - Timing annotations and delivery notes
    - Call-to-action integration
    - Scene/segment breakdown for longer scripts
    """
    
    def __init__(self, **kwargs):
        """Initialize the script generator."""
        super().__init__(
            name="ScriptGenerator",
            supported_formats=[
                ContentFormat.PLAIN_TEXT,
                ContentFormat.MARKDOWN,
                ContentFormat.JSON
            ],
            max_word_count=1500,  # Buffer over long script target
            min_word_count=100,   # Minimum for shortest scripts
            **kwargs
        )
        
        # Script-specific configuration
        self.supported_types = list(ScriptType)
        self.supported_durations = list(ScriptDuration)
        self.supported_styles = list(ScriptStyle)
        self.default_type = ScriptType.EXPLAINER
        self.default_duration = ScriptDuration.MEDIUM
        self.default_style = ScriptStyle.CONVERSATIONAL
    
    def _load_templates(self) -> None:
        """Load script-specific prompt templates."""
        self._prompt_templates = {
            "youtube_shorts_energetic": """Create an energetic YouTube Shorts script from this video content ({target_words} words, {seconds} seconds).

Transcript:
{transcript}

Format:
- HOOK (0-3s): Attention-grabbing opener
- MAIN CONTENT (3-{seconds_minus_10}s): Core message with energy
- CTA (last 5s): Subscribe/engage prompt
- Vertical video format considerations
- High energy throughout

Target: {target_words} words in {seconds} seconds""",

            "tiktok_storytelling": """Create a TikTok script from this video content ({target_words} words, {seconds} seconds).

Transcript:
{transcript}

Structure:
- HOOK (0-3s): Trend-relevant opener
- STORY ARC (3-{seconds_minus_5}s): Engaging narrative
- PAYOFF (last 3s): Satisfying conclusion
- Platform-specific elements
- Viral potential focus

Target: {target_words} words in {seconds} seconds""",

            "podcast_conversational": """Create a conversational podcast script from this video content ({target_words} words, {seconds} seconds).

Transcript:
{transcript}

Format:
- INTRO (0-30s): Topic introduction, guest intro if applicable
- MAIN DISCUSSION ({seconds_minus_60}s): Detailed conversation
- OUTRO (30s): Wrap-up, next episode tease
- Natural dialogue flow
- Audio-only considerations

Target: {target_words} words in {seconds} seconds"""
        }
    
    def get_script_type(self, type_str: str) -> ScriptType:
        """Get ScriptType enum from string."""
        try:
            return ScriptType(type_str.lower())
        except ValueError:
            available = [t.value for t in ScriptType]
            raise ValueError(f"Unknown script type '{type_str}'. Available: {available}")
    
    def get_script_duration(self, duration_str: str) -> ScriptDuration:
        """Get ScriptDuration enum from string."""
        for duration in ScriptDuration:
            if duration.duration_name == duration_str.lower():
                return duration
        
        available = [d.duration_name for d in ScriptDuration]
        raise ValueError(f"Unknown script duration '{duration_str}'. Available: {available}")
    
    def get_script_style(self, style_str: str) -> ScriptStyle:
        """Get ScriptStyle enum from string."""
        try:
            return ScriptStyle(style_str.lower())
        except ValueError:
            available = [s.value for s in ScriptStyle]
            raise ValueError(f"Unknown script style '{style_str}'. Available: {available}")
    
    def calculate_timing_breakdown(self, duration: ScriptDuration) -> Dict[str, int]:
        """Calculate timing breakdown for script sections."""
        total_seconds = duration.seconds
        
        if duration == ScriptDuration.SHORT:  # 60s
            return {
                "hook": 5,
                "main_content": 45,
                "cta": 10
            }
        elif duration == ScriptDuration.MEDIUM:  # 180s
            return {
                "intro": 15,
                "main_content": 140,
                "conclusion": 15,
                "cta": 10
            }
        else:  # LONG - 600s
            return {
                "intro": 60,
                "segment_1": 180,
                "segment_2": 180,
                "segment_3": 120,
                "conclusion": 45,
                "cta": 15
            }
    
    def generate_content(
        self,
        transcript: str,
        format_type: ContentFormat,
        script_type: str = "explainer",
        duration: str = "medium",
        style: str = "conversational",
        include_timing: bool = True,
        include_delivery_notes: bool = True,
        custom_cta: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate script content from transcript.
        
        Args:
            transcript: Input video transcript
            format_type: Desired output format
            script_type: Type of script ('youtube_shorts', 'tiktok', etc.)
            duration: Script duration ('short', 'medium', 'long')
            style: Writing style ('energetic', 'conversational', etc.)
            include_timing: Whether to include timing annotations
            include_delivery_notes: Whether to include delivery/direction notes
            custom_cta: Custom call-to-action text
            **kwargs: Additional parameters
            
        Returns:
            Generated script content
        """
        # Parse parameters
        try:
            script_type_enum = self.get_script_type(script_type)
        except ValueError:
            self.log_with_context(
                f"Invalid script type '{script_type}', using default 'explainer'",
                level="WARNING"
            )
            script_type_enum = self.default_type
        
        try:
            script_duration = self.get_script_duration(duration)
        except ValueError:
            self.log_with_context(
                f"Invalid script duration '{duration}', using default 'medium'",
                level="WARNING"
            )
            script_duration = self.default_duration
        
        try:
            script_style = self.get_script_style(style)
        except ValueError:
            self.log_with_context(
                f"Invalid script style '{style}', using default 'conversational'",
                level="WARNING"
            )
            script_style = self.default_style
        
        # Try AI generation first, fallback to placeholder
        try:
            from core.ai_backend import get_ai_backend
            ai_backend = get_ai_backend()
            
            if ai_backend and ai_backend.is_available():
                return self._generate_ai_script(
                    transcript=transcript,
                    script_type=script_type_enum,
                    duration=script_duration,
                    style=script_style,
                    format_type=format_type,
                    include_timing=include_timing,
                    include_delivery_notes=include_delivery_notes,
                    custom_cta=custom_cta
                )
        except Exception as e:
            self.log_with_context(
                f"AI generation failed, using placeholder: {e}",
                level="WARNING"
            )
        
        # Fallback to placeholder content
        return self._generate_placeholder_script(
            transcript=transcript,
            script_type=script_type_enum,
            duration=script_duration,
            style=script_style,
            format_type=format_type,
            include_timing=include_timing,
            include_delivery_notes=include_delivery_notes,
            custom_cta=custom_cta
        )
    
    def _generate_ai_script(
        self,
        transcript: str,
        script_type: ScriptType,
        duration: ScriptDuration,
        style: ScriptStyle,
        format_type: ContentFormat,
        include_timing: bool,
        include_delivery_notes: bool,
        custom_cta: Optional[str]
    ) -> str:
        """Generate real script content using AI backend."""
        from core.ai_backend import get_ai_backend
        ai_backend = get_ai_backend()
        
        # Select appropriate template
        template_key = f"{script_type.value}_{style.value}"
        
        # Build template mapping
        template_map = {
            "youtube_shorts_energetic": "youtube_shorts_energetic",
            "youtube_shorts_conversational": "youtube_shorts_energetic",
            "tiktok_storytelling": "tiktok_storytelling",
            "tiktok_energetic": "tiktok_storytelling",
            "podcast_conversational": "podcast_conversational",
            "podcast_authoritative": "podcast_conversational",
            "instagram_reel_energetic": "youtube_shorts_energetic",
            "explainer_educational": "podcast_conversational",
            "testimonial_storytelling": "tiktok_storytelling"
        }
        
        # Get template or use default
        template_name = template_map.get(template_key, "youtube_shorts_energetic")
        
        # Use template if available, otherwise create a custom prompt
        if template_name in self._prompt_templates:
            template = self.get_template(template_name)
            prompt = template.format(
                transcript=transcript[:5000],  # Limit transcript length
                target_words=duration.target_words,
                seconds=duration.seconds,
                seconds_minus_10=duration.seconds - 10,
                seconds_minus_5=duration.seconds - 5,
                seconds_minus_60=max(0, duration.seconds - 60)
            )
        else:
            # Build custom prompt
            prompt = self._build_custom_prompt(
                transcript, script_type, duration, style,
                include_timing, include_delivery_notes
            )
        
        # Add custom CTA if provided
        if custom_cta:
            prompt += f"\n\nInclude this call-to-action: {custom_cta}"
        
        # Generate content
        self.log_with_context(
            f"Generating {script_type.value} script ({duration.duration_name}, {style.value} style) with AI"
        )
        
        # Adjust token limit based on duration
        max_tokens = duration.target_words * 2
        
        response = ai_backend.generate_content(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8  # Slightly higher for creative scripts
        )
        
        if not response:
            raise ValueError("AI backend returned empty response")
        
        # Parse and structure the response
        script_data = self._parse_ai_script(
            response, script_type, duration, style,
            include_timing, include_delivery_notes, custom_cta
        )
        
        # Add metadata
        script_data["metadata"] = {
            "script_type": script_type.value,
            "duration": duration.duration_name,
            "target_seconds": duration.seconds,
            "target_words": duration.target_words,
            "wpm": duration.words_per_minute,
            "style": style.value,
            "has_timing": include_timing,
            "has_delivery_notes": include_delivery_notes,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_generated": True
        }
        
        # Calculate actual word count
        actual_words = self.count_words(script_data.get("full_script", ""))
        script_data["metadata"]["actual_words"] = actual_words
        script_data["metadata"]["estimated_duration"] = actual_words / duration.words_per_minute * 60
        
        return self._format_script_content(script_data, format_type)
    
    def _build_custom_prompt(
        self,
        transcript: str,
        script_type: ScriptType,
        duration: ScriptDuration,
        style: ScriptStyle,
        include_timing: bool,
        include_delivery_notes: bool
    ) -> str:
        """Build a custom prompt for scripts without specific templates."""
        timing_breakdown = self.calculate_timing_breakdown(duration)
        
        prompt = f"""Create a {style.value} {script_type.value} script from this video content.

Transcript:
{transcript[:5000]}

Requirements:
- Duration: {duration.seconds} seconds ({duration.target_words} words at {duration.words_per_minute} WPM)
- Style: {style.value} delivery
- Platform: {script_type.value}
"""
        
        if include_timing:
            prompt += "\n\nInclude timing annotations for each section:"
            for section, seconds in timing_breakdown.items():
                prompt += f"\n- {section.replace('_', ' ').title()}: {seconds} seconds"
        
        if include_delivery_notes:
            prompt += "\n\nInclude delivery notes (tone, pace, emphasis) in square brackets [like this]."
        
        prompt += f"\n\nTarget: {duration.target_words} words total"
        
        return prompt
    
    def _parse_ai_script(
        self,
        response: str,
        script_type: ScriptType,
        duration: ScriptDuration,
        style: ScriptStyle,
        include_timing: bool,
        include_delivery_notes: bool,
        custom_cta: Optional[str]
    ) -> Dict[str, Any]:
        """Parse AI response into structured script sections."""
        lines = response.strip().split('\n')
        
        # Initialize sections
        sections = []
        current_section = None
        current_content = []
        full_script = ""
        
        # Parse line by line
        for line in lines:
            # Check for timing annotations (e.g., "[0:00-0:05]", "HOOK (0-5s):")
            timing_match = re.match(r'[\[\(]?(\d+:?\d*)[s\-]?.*?[\]\)]?:?\s*(.*)', line)
            delivery_match = re.search(r'\[([^\]]+)\]', line)
            
            if timing_match and include_timing:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                timing = timing_match.group(1)
                content = timing_match.group(2) or ""
                current_section = {
                    "timing": timing,
                    "content": content.strip(),
                    "delivery_notes": []
                }
                current_content = [content] if content else []
            
            elif any(marker in line.upper() for marker in ['HOOK:', 'INTRO:', 'MAIN:', 'CTA:', 'OUTRO:', 'SEGMENT']):
                # Section header detected
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "title": line.split(':')[0].strip(),
                    "content": "",
                    "delivery_notes": []
                }
                current_content = []
            
            elif delivery_match and include_delivery_notes:
                # Extract delivery notes
                if current_section:
                    current_section["delivery_notes"].append(delivery_match.group(1))
                # Clean the line of delivery notes for content
                clean_line = re.sub(r'\[([^\]]+)\]', '', line).strip()
                if clean_line:
                    current_content.append(clean_line)
                    full_script += clean_line + " "
            
            elif line.strip():
                # Regular content line
                current_content.append(line.strip())
                full_script += line.strip() + " "
                if current_section:
                    current_section["content"] = " ".join(current_content)
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        # If no sections were parsed, create default structure
        if not sections:
            sections = self._create_default_sections(
                response, script_type, duration, include_timing
            )
            full_script = response.strip()
        
        # Ensure CTA is included
        if custom_cta and not any('cta' in s.get('title', '').lower() for s in sections):
            sections.append({
                "title": "CTA",
                "content": custom_cta,
                "timing": f"{duration.seconds - 10}-{duration.seconds}s" if include_timing else None,
                "delivery_notes": ["enthusiastic", "clear call to action"] if include_delivery_notes else []
            })
        
        return {
            "sections": sections,
            "full_script": full_script.strip(),
            "timing_breakdown": self.calculate_timing_breakdown(duration)
        }
    
    def _create_default_sections(
        self,
        content: str,
        script_type: ScriptType,
        duration: ScriptDuration,
        include_timing: bool
    ) -> List[Dict[str, Any]]:
        """Create default script sections when parsing fails."""
        words = content.split()
        total_words = len(words)
        timing_breakdown = self.calculate_timing_breakdown(duration)
        
        sections = []
        current_word = 0
        
        for section_name, section_seconds in timing_breakdown.items():
            # Calculate words for this section
            section_ratio = section_seconds / duration.seconds
            section_words = int(total_words * section_ratio)
            
            # Extract content for this section
            section_content = " ".join(words[current_word:current_word + section_words])
            
            sections.append({
                "title": section_name.replace('_', ' ').title(),
                "content": section_content,
                "timing": f"{sum(list(timing_breakdown.values())[:len(sections)-1])}-{sum(list(timing_breakdown.values())[:len(sections)])}s" if include_timing else None,
                "delivery_notes": []
            })
            
            current_word += section_words
        
        return sections
    
    def _format_script_content(self, script_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """Format script content based on output type."""
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(script_data, indent=2)
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_script(script_data)
        else:  # PLAIN_TEXT
            return self._format_text_script(script_data)
    
    def _format_markdown_script(self, script_data: Dict[str, Any]) -> str:
        """Format script as Markdown."""
        lines = []
        metadata = script_data.get("metadata", {})
        
        # Header
        script_type = metadata.get("script_type", "script").replace('_', ' ').title()
        lines.append(f"# {script_type}")
        lines.append("")
        
        # Metadata
        lines.append("## Script Details")
        lines.append(f"- **Duration:** {metadata.get('target_seconds', 0)} seconds")
        lines.append(f"- **Word Count:** {metadata.get('actual_words', 0)} / {metadata.get('target_words', 0)} target")
        lines.append(f"- **Pace:** {metadata.get('wpm', 0)} words per minute")
        lines.append(f"- **Style:** {metadata.get('style', 'conversational').title()}")
        lines.append("")
        
        # Sections
        lines.append("## Script")
        lines.append("")
        
        for section in script_data.get("sections", []):
            # Section header
            if section.get("timing"):
                lines.append(f"### {section.get('title', 'Section')} [{section['timing']}]")
            else:
                lines.append(f"### {section.get('title', 'Section')}")
            
            # Delivery notes
            if section.get("delivery_notes"):
                lines.append(f"*Delivery: {', '.join(section['delivery_notes'])}*")
            
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")
        
        # Full script (optional)
        if script_data.get("full_script"):
            lines.append("---")
            lines.append("")
            lines.append("## Full Script (Continuous)")
            lines.append("")
            lines.append(script_data["full_script"])
        
        return "\n".join(lines)
    
    def _format_text_script(self, script_data: Dict[str, Any]) -> str:
        """Format script as plain text."""
        lines = []
        metadata = script_data.get("metadata", {})
        
        # Header
        script_type = metadata.get("script_type", "script").upper()
        lines.append(f"{script_type} SCRIPT")
        lines.append("=" * 40)
        lines.append("")
        
        # Metadata
        lines.append(f"Duration: {metadata.get('target_seconds', 0)}s")
        lines.append(f"Words: {metadata.get('actual_words', 0)}/{metadata.get('target_words', 0)}")
        lines.append(f"Style: {metadata.get('style', 'conversational')}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
        
        # Sections
        for section in script_data.get("sections", []):
            if section.get("timing"):
                lines.append(f"[{section['timing']}] {section.get('title', 'SECTION').upper()}")
            else:
                lines.append(section.get('title', 'SECTION').upper())
            
            if section.get("delivery_notes"):
                lines.append(f"[{', '.join(section['delivery_notes'])}]")
            
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")
            lines.append("-" * 40)
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_placeholder_script(
        self,
        transcript: str,
        script_type: ScriptType,
        duration: ScriptDuration,
        style: ScriptStyle,
        format_type: ContentFormat,
        include_timing: bool,
        include_delivery_notes: bool,
        custom_cta: Optional[str]
    ) -> str:
        """Generate placeholder script content for Phase 1."""
        
        # Extract key content from transcript
        transcript_preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
        main_topics = self._extract_script_topics(transcript, script_type, duration)
        
        # Generate timing breakdown
        timing_breakdown = self.calculate_timing_breakdown(duration)
        
        # Create script structure
        script_data = {
            "title": self._generate_script_title(transcript_preview, script_type),
            "sections": self._generate_script_sections(
                main_topics, script_type, duration, style, timing_breakdown, custom_cta
            ),
            "metadata": {
                "type": script_type.value,
                "duration_target": duration.seconds,
                "target_words": duration.target_words,
                "words_per_minute": duration.words_per_minute,
                "style": style.value,
                "include_timing": include_timing,
                "include_delivery_notes": include_delivery_notes,
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_delivery_time": f"{duration.seconds // 60}m {duration.seconds % 60}s"
            },
            "timing_breakdown": timing_breakdown if include_timing else None,
            "delivery_notes": self._generate_delivery_notes(script_type, style) if include_delivery_notes else None
        }
        
        return self._format_script_content(script_data, format_type)
    
    def _extract_script_topics(self, transcript: str, script_type: ScriptType, duration: ScriptDuration) -> List[str]:
        """Extract main topics for script development."""
        # Simple topic extraction for Phase 1
        sentences = transcript.split('.')[:8]  # Get more sentences for longer scripts
        
        topics = []
        for sentence in sentences:
            if sentence.strip() and len(sentence.strip()) > 15:
                clean_topic = sentence.strip()
                topics.append(clean_topic[:80] + "..." if len(clean_topic) > 80 else clean_topic)
        
        # Limit topics based on script duration
        max_topics = {
            ScriptDuration.SHORT: 2,
            ScriptDuration.MEDIUM: 4,
            ScriptDuration.LONG: 6
        }[duration]
        
        return topics[:max_topics]
    
    def _generate_script_title(self, content_preview: str, script_type: ScriptType) -> str:
        """Generate appropriate script title."""
        if script_type == ScriptType.YOUTUBE_SHORTS:
            return "🔥 This Will Change Everything!"
        elif script_type == ScriptType.TIKTOK:
            return "POV: You Just Learned This Life-Changing Fact"
        elif script_type == ScriptType.PODCAST:
            return "Deep Dive: Understanding the Key Concepts"
        elif script_type == ScriptType.INSTAGRAM_REEL:
            return "✨ The Truth About This Topic"
        elif script_type == ScriptType.EXPLAINER:
            return "Complete Guide: Everything You Need to Know"
        else:  # TESTIMONIAL
            return "My Experience: What I Wish I Knew Earlier"
    
    def _generate_script_sections(
        self,
        topics: List[str],
        script_type: ScriptType,
        duration: ScriptDuration,
        style: ScriptStyle,
        timing_breakdown: Dict[str, int],
        custom_cta: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate script sections based on type and duration."""
        
        sections = []
        
        if duration == ScriptDuration.SHORT:
            sections = self._generate_short_script_sections(topics, script_type, style, timing_breakdown, custom_cta)
        elif duration == ScriptDuration.MEDIUM:
            sections = self._generate_medium_script_sections(topics, script_type, style, timing_breakdown, custom_cta)
        else:  # LONG
            sections = self._generate_long_script_sections(topics, script_type, style, timing_breakdown, custom_cta)
        
        return sections
    
    def _generate_short_script_sections(
        self, topics: List[str], script_type: ScriptType, style: ScriptStyle,
        timing: Dict[str, int], custom_cta: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate sections for 60-second scripts."""
        
        sections = []
        
        # Hook (5 seconds)
        hook_content = self._get_platform_hook(script_type, style, topics[0] if topics else "interesting content")
        sections.append({
            "name": "Hook",
            "duration": timing["hook"],
            "content": hook_content,
            "timing": "0:00 - 0:05",
            "delivery_notes": self._get_section_delivery_notes("hook", script_type, style)
        })
        
        # Main Content (45 seconds)
        main_content = f"Here's what you need to know: {topics[0] if topics else 'Key insight from the video content'}. This changes everything because it reveals {topics[1] if len(topics) > 1 else 'important principles that most people miss'}."
        sections.append({
            "name": "Main Content",
            "duration": timing["main_content"],
            "content": main_content,
            "timing": "0:05 - 0:50",
            "delivery_notes": self._get_section_delivery_notes("main", script_type, style)
        })
        
        # CTA (10 seconds)
        cta_content = custom_cta or self._get_platform_cta(script_type, style)
        sections.append({
            "name": "Call to Action",
            "duration": timing["cta"],
            "content": cta_content,
            "timing": "0:50 - 1:00",
            "delivery_notes": self._get_section_delivery_notes("cta", script_type, style)
        })
        
        return sections
    
    def _generate_medium_script_sections(
        self, topics: List[str], script_type: ScriptType, style: ScriptStyle,
        timing: Dict[str, int], custom_cta: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate sections for 180-second scripts."""
        
        sections = []
        
        # Intro (15 seconds)
        intro_content = f"Welcome! Today we're diving into {topics[0] if topics else 'an important topic'}. By the end of this video, you'll understand why this matters and how it applies to your situation."
        sections.append({
            "name": "Introduction",
            "duration": timing["intro"],
            "content": intro_content,
            "timing": "0:00 - 0:15",
            "delivery_notes": self._get_section_delivery_notes("intro", script_type, style)
        })
        
        # Main Content (140 seconds, split into parts)
        content_parts = [
            f"First, let's establish the foundation: {topics[0] if topics else 'Core concept explanation'}. This is crucial because it sets up everything else we'll discuss.",
            f"Next, consider this important aspect: {topics[1] if len(topics) > 1 else 'Secondary point that builds on the foundation'}. Here's why this matters in practical terms.",
            f"Finally, here's what most people miss: {topics[2] if len(topics) > 2 else 'Advanced insight that ties everything together'}. This is the key to real understanding."
        ]
        
        current_time = 15
        for i, content in enumerate(content_parts, 1):
            duration = 47  # ~140/3 seconds per part
            end_time = current_time + duration
            
            sections.append({
                "name": f"Main Point {i}",
                "duration": duration,
                "content": content,
                "timing": f"{current_time//60}:{current_time%60:02d} - {end_time//60}:{end_time%60:02d}",
                "delivery_notes": self._get_section_delivery_notes("main", script_type, style)
            })
            current_time = end_time
        
        # Conclusion (15 seconds)
        conclusion_content = "So to wrap up, the key takeaway is that understanding these concepts gives you the tools to approach this topic with confidence and clarity."
        sections.append({
            "name": "Conclusion",
            "duration": timing["conclusion"],
            "content": conclusion_content,
            "timing": "2:35 - 2:50",
            "delivery_notes": self._get_section_delivery_notes("conclusion", script_type, style)
        })
        
        # CTA (10 seconds)
        cta_content = custom_cta or self._get_platform_cta(script_type, style)
        sections.append({
            "name": "Call to Action",
            "duration": timing["cta"],
            "content": cta_content,
            "timing": "2:50 - 3:00",
            "delivery_notes": self._get_section_delivery_notes("cta", script_type, style)
        })
        
        return sections
    
    def _generate_long_script_sections(
        self, topics: List[str], script_type: ScriptType, style: ScriptStyle,
        timing: Dict[str, int], custom_cta: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate sections for 600-second (10-minute) scripts."""
        
        sections = []
        current_time = 0
        
        # Intro (60 seconds)
        intro_content = f"""Welcome to today's deep dive! I'm excited to explore {topics[0] if topics else 'this fascinating topic'} with you. 

        This is something I've been researching extensively, and I think you're going to find some genuinely surprising insights here. We'll cover the foundational concepts, examine some real-world applications, and by the end, you'll have a comprehensive understanding that you can immediately put to use.

        Let's jump right in."""
        
        sections.append({
            "name": "Introduction & Overview",
            "duration": timing["intro"],
            "content": intro_content,
            "timing": "0:00 - 1:00",
            "delivery_notes": self._get_section_delivery_notes("intro", script_type, style)
        })
        current_time += timing["intro"]
        
        # Three main segments
        segment_topics = topics[:3] if len(topics) >= 3 else topics + ["Additional analysis"] * (3 - len(topics))
        
        for i, (segment_key, topic) in enumerate(zip(["segment_1", "segment_2", "segment_3"], segment_topics), 1):
            duration = timing[segment_key]
            end_time = current_time + duration
            
            content = f"""Now let's dive into {topic}. 
            
            This is particularly interesting because it challenges some common assumptions. Here's what the research shows, and more importantly, here's what it means for you practically.
            
            [Detailed explanation with examples and supporting evidence would go here. In a real script, this section would be fully developed with specific data points, stories, and actionable insights.]
            
            The key insight here is that understanding this concept fundamentally changes how you approach related challenges."""
            
            sections.append({
                "name": f"Segment {i}: {topic[:30]}{'...' if len(topic) > 30 else ''}",
                "duration": duration,
                "content": content,
                "timing": f"{current_time//60}:{current_time%60:02d} - {end_time//60}:{end_time%60:02d}",
                "delivery_notes": self._get_section_delivery_notes("segment", script_type, style)
            })
            current_time = end_time
        
        # Conclusion (45 seconds)
        conclusion_content = """Let's tie this all together. The three main concepts we've covered today work synergistically to create a comprehensive framework for understanding this topic.

        The practical implications are significant, and I encourage you to start applying these insights immediately. Remember, knowledge without action is just entertainment."""
        
        end_time = current_time + timing["conclusion"]
        sections.append({
            "name": "Conclusion & Summary",
            "duration": timing["conclusion"],
            "content": conclusion_content,
            "timing": f"{current_time//60}:{current_time%60:02d} - {end_time//60}:{end_time%60:02d}",
            "delivery_notes": self._get_section_delivery_notes("conclusion", script_type, style)
        })
        current_time = end_time
        
        # CTA (15 seconds)
        cta_content = custom_cta or self._get_platform_cta(script_type, style)
        sections.append({
            "name": "Call to Action",
            "duration": timing["cta"],
            "content": cta_content,
            "timing": f"{current_time//60}:{current_time%60:02d} - 10:00",
            "delivery_notes": self._get_section_delivery_notes("cta", script_type, style)
        })
        
        return sections
    
    def _get_platform_hook(self, script_type: ScriptType, style: ScriptStyle, topic: str) -> str:
        """Generate platform-specific hooks."""
        if script_type == ScriptType.YOUTUBE_SHORTS:
            if style == ScriptStyle.ENERGETIC:
                return f"🔥 You WON'T believe what I just discovered about {topic}!"
            else:
                return f"Here's something interesting about {topic} that might surprise you..."
        elif script_type == ScriptType.TIKTOK:
            return f"POV: You just learned {topic} and everything makes sense now ✨"
        elif script_type == ScriptType.PODCAST:
            return f"Today we're exploring {topic}, and I think you'll find this conversation really valuable."
        else:
            return f"Let's talk about {topic} - this is something I've been thinking about a lot lately."
    
    def _get_platform_cta(self, script_type: ScriptType, style: ScriptStyle) -> str:
        """Generate platform-specific CTAs."""
        if script_type == ScriptType.YOUTUBE_SHORTS:
            return "If this helped you, smash that like button and follow for more insights like this!"
        elif script_type == ScriptType.TIKTOK:
            return "Follow for more content like this! What do you think about this? Comment below ⬇️"
        elif script_type == ScriptType.PODCAST:
            return "Thanks for listening! Subscribe for more episodes, and let me know your thoughts in the comments."
        elif script_type == ScriptType.INSTAGRAM_REEL:
            return "Save this for later and share it with someone who needs to see this! ✨"
        else:
            return "What did you think about this? Let me know in the comments, and subscribe for more content like this!"
    
    def _get_section_delivery_notes(self, section_type: str, script_type: ScriptType, style: ScriptStyle) -> str:
        """Generate delivery notes for script sections."""
        base_notes = {
            "hook": "High energy, direct eye contact, grab attention immediately",
            "intro": "Warm, welcoming tone, set expectations clearly",
            "main": "Steady pace, emphasize key points, maintain engagement",
            "segment": "Conversational but authoritative, use examples",
            "conclusion": "Confident, summarizing tone, reinforce key messages",
            "cta": "Enthusiastic but not pushy, clear action steps"
        }
        
        style_modifiers = {
            ScriptStyle.ENERGETIC: " - Keep energy high throughout, animated gestures",
            ScriptStyle.CONVERSATIONAL: " - Natural, friendly delivery, as if talking to a friend",
            ScriptStyle.AUTHORITATIVE: " - Confident, measured pace, establish credibility",
            ScriptStyle.STORYTELLING: " - Varied pace, emotional connection, narrative flow",
            ScriptStyle.EDUCATIONAL: " - Clear articulation, pause for emphasis, structured delivery"
        }
        
        base_note = base_notes.get(section_type, "Standard delivery")
        modifier = style_modifiers.get(style, "")
        
        return base_note + modifier
    
    def _generate_delivery_notes(self, script_type: ScriptType, style: ScriptStyle) -> Dict[str, str]:
        """Generate comprehensive delivery notes for the script."""
        return {
            "overall_tone": f"{style.value.title()} delivery throughout",
            "platform_notes": self._get_platform_delivery_notes(script_type),
            "pacing": "Maintain consistent energy, use natural pauses for emphasis",
            "visual_elements": self._get_visual_elements_notes(script_type),
            "audio_considerations": "Clear articulation, consistent volume, avoid filler words"
        }
    
    def _get_platform_delivery_notes(self, script_type: ScriptType) -> str:
        """Get platform-specific delivery notes."""
        notes = {
            ScriptType.YOUTUBE_SHORTS: "Vertical format, quick cuts, maintain visual interest",
            ScriptType.TIKTOK: "Trend-aware, authentic delivery, hook viewers in first 3 seconds",
            ScriptType.PODCAST: "Audio-only, paint pictures with words, conversational flow",
            ScriptType.INSTAGRAM_REEL: "Visual storytelling, music-friendly pacing, branded feel",
            ScriptType.EXPLAINER: "Educational clarity, logical progression, helpful tone",
            ScriptType.TESTIMONIAL: "Personal, credible, emotional authenticity"
        }
        return notes.get(script_type, "Standard video delivery")
    
    def _get_visual_elements_notes(self, script_type: ScriptType) -> str:
        """Get notes about visual elements for the script."""
        if script_type in [ScriptType.YOUTUBE_SHORTS, ScriptType.TIKTOK, ScriptType.INSTAGRAM_REEL]:
            return "Consider text overlays, quick cuts, engaging visuals to support narration"
        elif script_type == ScriptType.PODCAST:
            return "Audio-only content, no visual elements needed"
        else:
            return "Standard video elements, slides or graphics to support key points"
    
    def _format_script_content(self, script_data: Dict[str, Any], format_type: ContentFormat) -> str:
        """Format script content based on output type."""
        if format_type == ContentFormat.JSON:
            import json
            return json.dumps(script_data, indent=2)
        elif format_type == ContentFormat.MARKDOWN:
            return self._format_markdown_script(script_data)
        else:  # PLAIN_TEXT
            return self._format_text_script(script_data)
    
    def _format_markdown_script(self, script_data: Dict[str, Any]) -> str:
        """Format script as Markdown."""
        lines = []
        
        # Title and metadata
        lines.append(f"# {script_data['title']}")
        lines.append("")
        
        metadata = script_data['metadata']
        lines.append(f"**Type:** {metadata['type'].title().replace('_', ' ')}")
        lines.append(f"**Duration:** {metadata['estimated_delivery_time']}")
        lines.append(f"**Target Words:** {metadata['target_words']}")
        lines.append(f"**Style:** {metadata['style'].title()}")
        lines.append("")
        
        # Timing breakdown
        if script_data.get('timing_breakdown'):
            lines.append("## Timing Breakdown")
            lines.append("")
            for section, duration in script_data['timing_breakdown'].items():
                lines.append(f"- **{section.replace('_', ' ').title()}:** {duration}s")
            lines.append("")
        
        # Script sections
        lines.append("## Script")
        lines.append("")
        
        for section in script_data['sections']:
            lines.append(f"### {section['name']}")
            lines.append(f"**Timing:** {section['timing']} ({section['duration']}s)")
            lines.append("")
            lines.append(section['content'])
            lines.append("")
            
            if section.get('delivery_notes'):
                lines.append(f"*Delivery Notes: {section['delivery_notes']}*")
                lines.append("")
        
        # Overall delivery notes
        if script_data.get('delivery_notes'):
            lines.append("## Delivery Notes")
            lines.append("")
            for note_type, note in script_data['delivery_notes'].items():
                lines.append(f"**{note_type.replace('_', ' ').title()}:** {note}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_text_script(self, script_data: Dict[str, Any]) -> str:
        """Format script as plain text."""
        lines = []
        
        # Title
        title = script_data['title']
        lines.append(title.upper())
        lines.append("=" * len(title))
        lines.append("")
        
        # Metadata
        metadata = script_data['metadata']
        lines.append(f"Type: {metadata['type'].replace('_', ' ').title()}")
        lines.append(f"Duration: {metadata['estimated_delivery_time']}")
        lines.append(f"Target Words: {metadata['target_words']}")
        lines.append(f"Style: {metadata['style'].title()}")
        lines.append("")
        
        # Script sections
        lines.append("SCRIPT:")
        lines.append("-" * 50)
        lines.append("")
        
        for section in script_data['sections']:
            lines.append(f"[{section['name'].upper()}] ({section['timing']})")
            lines.append("")
            lines.append(section['content'])
            lines.append("")
            
            if section.get('delivery_notes'):
                lines.append(f">> {section['delivery_notes']}")
                lines.append("")
            
            lines.append("-" * 30)
            lines.append("")
        
        return "\n".join(lines)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate script generation input."""
        if not super().validate_input(input_data):
            return False
        
        # Validate script type if specified
        if "script_type" in input_data:
            try:
                self.get_script_type(input_data["script_type"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate duration if specified
        if "duration" in input_data:
            try:
                self.get_script_duration(input_data["duration"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        # Validate style if specified
        if "style" in input_data:
            try:
                self.get_script_style(input_data["style"])
            except ValueError as e:
                self.log_with_context(str(e), level="ERROR")
                return False
        
        return True
    
    def get_supported_configurations(self) -> Dict[str, Any]:
        """Get all supported script configurations."""
        return {
            "types": [t.value for t in ScriptType],
            "durations": {
                d.duration_name: {
                    "seconds": d.seconds,
                    "target_words": d.target_words,
                    "wpm": d.words_per_minute
                } for d in ScriptDuration
            },
            "styles": [s.value for s in ScriptStyle],
            "formats": [fmt.value for fmt in self.supported_formats]
        }