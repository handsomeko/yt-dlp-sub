# **Product Requirement Prompt (PRP)**
## YouTube Content Intelligence & Repurposing Platform

### **Project Context**
You are building a YouTube content intelligence and repurposing platform that evolves through three phases: CLI tool → API service → MicroSaaS product. The goal is to help users monitor YouTube channels, extract transcripts, and automatically transform them into multiple content formats for omnichannel distribution. This document defines the complete product vision and implementation strategy.

### **Problem Statement**
Content creators and businesses struggle with content repurposing and distribution:
- Manual transcript extraction is time-consuming
- Converting video content to other formats takes hours per video
- No automated pipeline from YouTube to multiple content channels
- Can't efficiently repurpose competitor or industry content
- Missing opportunities to maximize content ROI across platforms
- No tools for monitoring multiple channels and auto-generating content
- Existing tools only extract, don't transform or repurpose

### **Target Users & Use Cases**

**Primary Personas:**
1. **Content Creators** - Repurpose videos into blog posts, social media, newsletters
2. **Marketing Agencies** - Transform client videos into omnichannel campaigns
3. **Course Creators** - Convert lectures into study guides, summaries, quizzes
4. **Social Media Managers** - Generate posts, threads, carousels from videos
5. **Newsletter Writers** - Transform YouTube content into email content
6. **Businesses** - Repurpose webinars, tutorials into marketing materials

**Core Use Cases:**
- "Turn this YouTube video into a blog post, 5 tweets, and LinkedIn article"
- "Monitor competitors' channels and generate content ideas from their videos"
- "Create weekly newsletter from top videos in my industry"
- "Generate social media content calendar from my YouTube channel"
- "Transform educational videos into study guides and cliff notes"
- "Create infographics and quote cards from video transcripts"

### **Three-Phase Development Strategy**

#### **Phase 1: CLI Tool with Content Generation (Weeks 1-4)**
**Goal:** Build functional CLI that extracts and repurposes content

**MVP Features:**
- Add/remove YouTube channels to monitor
- Check channels for new videos (via RSS)
- Download transcripts using yt-dlp (fallback: youtube-transcript-api)
- Store transcripts in local SQLite database
- **Content Generation (NEW):**
  - Auto-generate summaries and key points
  - Create blog post drafts from videos
  - Generate social media posts (Twitter, LinkedIn)
  - Extract quotes and highlights
  - Create email newsletter content
- Basic text search across all transcripts
- Export in multiple formats (JSON, Markdown, HTML)
- Modular worker system for processing pipeline

**Technical Requirements:**
- Python 3.11+, Click for CLI
- SQLite with FTS5 for full-text search
- Environment-based deployment modes (LOCAL/MONOLITH/DISTRIBUTED)
- Must work on: Local machine, Seedbox, VPS with residential IP
- Handle YouTube rate limiting gracefully
- **LLM Integration:** OpenAI/Anthropic API for content generation
- **Worker Architecture:**
  - Orchestrator: Coordinates job flow
  - Monitor: RSS channel checking
  - Downloader: yt-dlp transcript extraction
  - Converter: SRT to plain text
  - Generator: AI content transformation
  - Quality: Output validation

**Validation Gate → Phase 2:**
- ✅ 50+ active users using the CLI weekly
- ✅ Successfully processing 1000+ videos
- ✅ Clear feature requests from users

#### **Phase 2: API Service with Advanced Generation (Weeks 5-8)**
**Goal:** API-first content generation platform

**Additional Features:**
- FastAPI REST endpoints for all content types
- API key authentication with usage tiers
- Rate limiting per API key
- Webhook notifications for new content
- Batch content generation
- **Advanced Content Types:**
  - Video scripts (short & long form)
  - Podcast outlines
  - Course materials
  - Infographic data
  - SEO-optimized articles
- Custom prompt templates
- Brand voice configuration
- Usage tracking and analytics
- **Background Processing:**
  - Supabase Edge Functions for async tasks
  - Celery workers for heavy processing
  - WebSocket support for real-time updates

**Technical Evolution:**
- Add FastAPI with async/await patterns
- Supabase for database + auth + storage
- Redis for caching and session management
- Deploy to seedbox with nginx reverse proxy
- SSL via Let's Encrypt
- Edge Functions for lightweight tasks

**Validation Gate → Phase 3:**
- ✅ 10+ users willing to pay $20+/month
- ✅ 100+ API calls daily
- ✅ Clear demand for web interface

#### **Phase 3: MicroSaaS Content Platform (Months 3-6)**
**Goal:** Full content repurposing platform with subscription model

**Additional Features:**
- Web dashboard (Next.js) with content editor
- User registration/authentication
- Stripe subscription billing
- Team collaboration and approval workflows
- **Content Management:**
  - Content calendar integration
  - Bulk generation and scheduling
  - Multi-channel publishing
  - A/B testing for content
  - Performance analytics
- Custom AI model fine-tuning
- White-label options for agencies

**Pricing Tiers:**
```
Starter ($49/mo): 10 channels, 50 videos/month, 5 content types
Professional ($149/mo): 50 channels, 500 videos/month, all content types, API
Business ($399/mo): Unlimited channels, 2000 videos/month, custom templates, team seats
Enterprise ($999/mo): White-label, custom AI training, dedicated support
```

### **Technical Architecture Requirements**

**Critical Constraint:** YouTube blocks cloud provider IPs (AWS, GCP, Azure)

**Solution:** Flexible deployment architecture
```python
DEPLOYMENT_MODE = 'LOCAL'      # Development on laptop
DEPLOYMENT_MODE = 'MONOLITH'   # Everything on seedbox/VPS  
DEPLOYMENT_MODE = 'DISTRIBUTED' # API on cloud, workers on seedbox
```

**Storage Strategy:**
- Phase 1: Local filesystem or seedbox storage
- Phase 2: Google Drive + Local hybrid (1TB available)
- Phase 3: S3/GCS for scale with CDN

**Database Evolution:**
- Phase 1: SQLite with FTS5 + Airtable for UI
- Phase 2: Supabase (PostgreSQL + Vector search) + Airtable
- Phase 3: Supabase with pgvector for semantic search

**Queue System:**
- Phase 1: SQLite-based job queue (simple, reliable)
- Phase 2: Supabase Queue (pgmq) or Redis
- Phase 3: Celery with Redis for distributed processing

**Audio Processing (NEW):**
- Phase 1: yt-dlp audio extraction (Opus format)
- Phase 2: Whisper transcription (95%+ accuracy)
- Phase 3: Real-time transcription with streaming

### **Content Generation Capabilities**

**Supported Content Formats:**

**Text Content:**
- Summaries (short, medium, detailed)
- Blog posts (500-2000 words)
- Social media posts (Twitter, LinkedIn, Facebook)
- Email newsletters
- Study guides and cliff notes
- SEO articles with keywords
- Quotes and key takeaways

**Visual Content (Data for):**
- Infographic outlines
- Quote cards
- Carousel posts
- Presentation slides
- Mind maps

**Scripts & Outlines:**
- YouTube Shorts scripts
- TikTok/Reels scripts
- Long-form video scripts
- Podcast episode outlines
- Webinar structures

**Educational Content:**
- Course modules
- Quiz questions
- Flashcards
- Lesson plans
- Workshop materials

### **Core Technical Decisions**

1. **Why FFmpeg af_whisper + Whisper fallbacks?**
   - FFmpeg 8.0 includes built-in Whisper (af_whisper filter)
   - Single command for audio extraction + transcription
   - 95%+ accuracy with lower resource usage
   - Enhanced fallback chain: ffmpeg-af_whisper → whisper-local → whisper-api → yt-dlp → youtube-transcript-api

2. **Why Supabase over raw PostgreSQL?**
   - Built-in auth, storage, and edge functions
   - Vector search with pgvector included
   - Real-time subscriptions for live updates
   - Scales from $0 to enterprise

3. **Why Celery for background tasks?**
   - Industry standard for Python async tasks
   - Robust retry mechanisms
   - Distributed task routing
   - Works with Redis/RabbitMQ

4. **Why job queue from Day 1?**
   - Enables deployment flexibility
   - Handles failures gracefully
   - Natural path to scaling

5. **Why SQLite FTS5 over Elasticsearch?**
   - Simpler, no separate service
   - Good enough for <1M transcripts
   - Easy migration path to PostgreSQL FTS

### **What We're NOT Building**

- ❌ Video downloader/player
- ❌ Video editing tools
- ❌ Real-time live stream transcription
- ❌ Translation service (YouTube provides this)
- ❌ Comment analysis
- ❌ Video recommendations engine
- ❌ Social media features

### **Success Metrics**

**Phase 1 Success:**
- 50+ weekly active users
- <5% transcript extraction failure rate
- 80% content generation satisfaction rate
- 10+ content pieces generated per video
- <60 second total processing time per video

**Phase 2 Success:**
- 20+ paying customers ($100+ average)
- 5000+ content pieces generated daily
- <2 second API response time
- 95% content quality score

**Phase 3 Success:**
- $10,000+ MRR within 3 months
- <5% monthly churn
- 100+ paying customers
- CAC < $200, LTV > $2000
- 50%+ revenue from enterprise tier

### **Development Principles**

1. **Lean First:** Build minimum viable solution, validate, then expand
2. **User-Driven:** Every feature must be requested by real users
3. **Cost-Conscious:** Keep infrastructure costs <$20/month until revenue
4. **Deployment Flexible:** Same code works locally, on seedbox, or distributed
5. **Data Ownership:** Users can export all their data anytime
6. **YouTube ToS Compliant:** Only store transcripts, not videos

### **Risk Mitigation**

- **YouTube API Changes:** Use multiple extraction methods
- **IP Blocking:** Seedbox/residential proxy strategy
- **Competition:** Focus on channel monitoring (unique differentiator)
- **Legal:** Clear ToS, only store text transcripts
- **Scaling:** Queue architecture handles growth naturally

### **Implementation Priorities**

**Week 1:**
- Core download functionality with yt-dlp
- SQLite database with schema
- Basic CLI commands

**Week 2:**
- Job queue system
- Channel monitoring via RSS
- Full-text search

**Week 3:**
- Export functionality
- Error handling and retries
- Rate limiting

**Week 4:**
- Testing and documentation
- Deploy to seedbox
- Share with test users

### **Budget Constraints**

- Phase 1: $20-50/month (seedbox + LLM API costs)
- Phase 2: $50-100/month (infrastructure + increased API usage)
- Phase 3: $200-500/month until profitable (scales with usage)

### **Key Technical Files**

```
# Core Infrastructure
config.py              # Environment-based configuration
queue.py               # Job queue abstraction (SQLite → Redis → Celery)
storage.py             # Storage backend abstraction (Local → GDrive → S3)

# Workers
workers/
  orchestrator.py      # Job coordination and flow control
  monitor.py           # RSS channel monitoring
  downloader.py        # Audio extraction with yt-dlp (Opus format)
  transcriber.py       # Whisper transcription with fallback chain
  generator.py         # Content generation organizer
  quality.py           # AI quality validation (transcript + content)

# Sub-Generators (Phase 2+)
generators/
  blog.py              # Blog post generator (1000-3000 words)
  social.py            # Social media content (Twitter, LinkedIn, FB)
  newsletter.py        # Email newsletter sections
  scripts.py           # Video/podcast scripts
  educational.py       # Course materials, quizzes

# Content Templates
templates/
  blog_post.py         # Blog post generation templates
  social_media.py      # Social media post templates
  educational.py       # Educational content templates
  scripts.py           # Video/podcast script templates

# API (Phase 2+)
api/
  main.py              # FastAPI application
  auth.py              # API key management
  webhooks.py          # Webhook handlers
  background.py        # Background task management

# Interfaces
cli.py                 # Click CLI commands
web/                   # Next.js dashboard (Phase 3)
```

### **Decision Points**

Before proceeding to each phase, evaluate:
1. Is there real user demand?
2. Will people pay for this?
3. Can we handle support at scale?
4. Is the unit economics sustainable?

**Remember:** It's better to have 10 users who love the product than 1000 who are indifferent. Build for the passionate few first.

---

**This PRP is your north star. Every feature, every technical decision, every prioritization should align with this document. When in doubt, choose simplicity and user value over complexity and features.**