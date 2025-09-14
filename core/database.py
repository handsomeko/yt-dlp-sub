"""
SQLite database schema with SQLAlchemy models and FTS5 support.
Implements the complete database design from the PRD specifications.
"""

import asyncio
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import aiosqlite
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, BigInteger, String, Text, JSON,
    Numeric, ForeignKey, UniqueConstraint, Index, CheckConstraint, create_engine, event
)
import enum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


# ========================================
# Database Enums for Data Integrity
# ========================================

class StorageVersion(enum.Enum):
    """Enum for storage structure versions."""
    V1 = "v1"
    V2 = "v2"


class TranscriptStatus(enum.Enum):
    """Enum for transcript processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationReviewStatus(enum.Enum):
    """Enum for generation review status."""
    PENDING_REVIEW = "pending_review"
    APPROVED_FOR_GENERATION = "approved_for_generation"
    GENERATION_SKIPPED = "generation_skipped"
    AUTO_APPROVED = "auto_approved"


class GenerationSelectionStatus(enum.Enum):
    """Enum for generation selection status."""
    PENDING_SELECTION = "pending_selection"
    FORMATS_SELECTED = "formats_selected"
    GENERATION_STARTED = "generation_started"


# ========================================
# SQLAlchemy Models
# ========================================

class Channel(Base):
    """YouTube channels being monitored."""
    __tablename__ = 'channels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(50), unique=True, nullable=False)
    channel_name = Column(String(255))
    channel_url = Column(String(500))
    description = Column(Text)
    subscriber_count = Column(Integer)
    video_count = Column(Integer)
    last_video_id = Column(String(50))
    last_checked = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_channels_active', 'is_active', 'last_checked'),
    )


class Video(Base):
    """Video metadata and processing status."""
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), unique=True, nullable=False)
    channel_id = Column(String(50), ForeignKey('channels.channel_id'), nullable=False)
    title = Column(String(500))
    description = Column(Text)
    duration = Column(Integer)  # seconds
    view_count = Column(Integer)
    like_count = Column(Integer)
    published_at = Column(DateTime)
    transcript_status = Column(String(20), default=TranscriptStatus.PENDING.value)  # pending, processing, completed, failed
    language = Column(String(10), default='en')
    is_auto_generated = Column(Boolean)
    
    # Review checkpoint fields
    generation_review_status = Column(String(30), default=GenerationReviewStatus.PENDING_REVIEW.value)  # pending_review, approved_for_generation, generation_skipped, auto_approved
    generation_approved_at = Column(DateTime)
    generation_review_notes = Column(Text)
    
    # AI Summary fields (optional step)
    ai_summary = Column(Text)
    extracted_topics = Column(Text)  # JSON array of topics
    summary_generated_at = Column(DateTime)
    summary_cost = Column(Numeric(10, 4))  # Track AI costs
    
    # Format selection fields
    generation_selection_status = Column(String(30), default=GenerationSelectionStatus.PENDING_SELECTION.value)  # pending_selection, formats_selected, generation_started
    selected_generators = Column(Text)  # JSON array of selected generator types
    generation_selected_at = Column(DateTime)
    generation_started_at = Column(DateTime)
    generation_completed_at = Column(DateTime)
    
    # Storage V2 fields for filename management
    video_title_snapshot = Column(String(500))  # Title at time of processing for consistent filenames
    title_sanitized = Column(String(200))  # Cached sanitized filename
    storage_version = Column(String(10), default=StorageVersion.V2.value)  # v1 or v2
    processing_completed_at = Column(DateTime)  # When all processing finished
    
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    channel = relationship("Channel", back_populates="videos")
    transcript = relationship("Transcript", back_populates="video", uselist=False)
    generated_content = relationship("GeneratedContent", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes and Constraints for Data Integrity
    __table_args__ = (
        # Performance indexes
        Index('idx_videos_channel', 'channel_id'),
        Index('idx_videos_status', 'transcript_status'),
        # V2 storage performance indexes
        Index('idx_videos_storage_version', 'storage_version'),
        Index('idx_videos_processing_completed', 'processing_completed_at'),
        Index('idx_videos_title_search', 'video_title_snapshot'),
        Index('idx_videos_channel_processing', 'channel_id', 'processing_completed_at'),
        # Common query patterns
        Index('idx_videos_channel_title', 'channel_id', 'title_sanitized'),
        
        # CRITICAL: Data integrity constraints to prevent invalid data
        CheckConstraint(
            "storage_version IN ('v1', 'v2')",
            name='chk_videos_storage_version'
        ),
        CheckConstraint(
            "transcript_status IN ('pending', 'processing', 'completed', 'failed')",
            name='chk_videos_transcript_status'
        ),
        CheckConstraint(
            "generation_review_status IN ('pending_review', 'approved_for_generation', 'generation_skipped', 'auto_approved')",
            name='chk_videos_generation_review_status'
        ),
        CheckConstraint(
            "generation_selection_status IN ('pending_selection', 'formats_selected', 'generation_started')",
            name='chk_videos_generation_selection_status'
        ),
        # Length and format validation
        CheckConstraint(
            "LENGTH(video_id) > 0 AND LENGTH(video_id) <= 50",
            name='chk_videos_video_id_length'
        ),
        CheckConstraint(
            "LENGTH(channel_id) > 0 AND LENGTH(channel_id) <= 50",
            name='chk_videos_channel_id_length'
        ),
        # Logical constraints
        CheckConstraint(
            "duration IS NULL OR duration >= 0",
            name='chk_videos_duration_positive'
        ),
        CheckConstraint(
            "view_count IS NULL OR view_count >= 0",
            name='chk_videos_view_count_positive'
        ),
        CheckConstraint(
            "like_count IS NULL OR like_count >= 0",
            name='chk_videos_like_count_positive'
        ),
    )


class Transcript(Base):
    """Audio transcripts with quality metrics and file paths."""
    __tablename__ = 'transcripts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), ForeignKey('videos.video_id'), nullable=False, unique=True)
    content_srt = Column(Text)
    content_text = Column(Text)
    word_count = Column(Integer)
    language = Column(String(10), default='en')
    extraction_method = Column(String(20))  # whisper-local, whisper-api, yt-dlp, youtube-transcript-api
    transcription_model = Column(String(50))  # whisper-base, whisper-large, etc
    quality_score = Column(Float)  # 0.0 to 1.0
    quality_details = Column(JSON)  # detailed quality metrics
    audio_path = Column(String(500))  # path to opus audio file
    srt_path = Column(String(500))  # path to SRT file
    transcript_path = Column(String(500))  # path to TXT file
    gdrive_audio_id = Column(String(100))  # Google Drive file ID
    gdrive_srt_id = Column(String(100))
    gdrive_transcript_id = Column(String(100))
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    video = relationship("Video", back_populates="transcript")


class Job(Base):
    """Job queue for async processing with retry tracking."""
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(50), nullable=False)  # download_transcript, process_channel
    target_id = Column(String(50), nullable=False)  # video_id or channel_id
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    priority = Column(Integer, default=5)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    error_message = Column(Text)
    worker_id = Column(String(50))
    created_at = Column(DateTime, default=func.current_timestamp())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_jobs_status', 'status', 'priority', 'created_at'),
    )


class QualityCheck(Base):
    """Quality validation results for transcripts and generated content."""
    __tablename__ = 'quality_checks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    target_id = Column(String(50), nullable=False)  # video_id or content_id
    target_type = Column(String(20), nullable=False)  # transcript, content
    check_type = Column(String(50), nullable=False)  # completeness, coherence, format, etc
    score = Column(Float)
    passed = Column(Boolean)
    details = Column(JSON)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Indexes
    __table_args__ = (
        Index('idx_quality_checks', 'target_id', 'target_type', 'passed'),
    )


class GeneratedContent(Base):
    """AI-generated content from transcripts."""
    __tablename__ = 'generated_content'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), ForeignKey('videos.video_id'), nullable=False)
    content_type = Column(String(50), nullable=False)  # summary, blog, twitter, linkedin, etc
    content = Column(Text)
    content_metadata = Column(JSON)  # word count, hashtags, etc
    quality_score = Column(Float)
    generation_model = Column(String(50))  # gpt-4, claude-3, etc
    prompt_template = Column(String(100))
    storage_path = Column(String(500))
    gdrive_file_id = Column(String(100))
    airtable_record_id = Column(String(50))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    video = relationship("Video", back_populates="generated_content")
    
    # Indexes
    __table_args__ = (
        Index('idx_generated_content', 'video_id', 'content_type'),
    )


class StorageSync(Base):
    """File synchronization status across storage backends."""
    __tablename__ = 'storage_sync'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_type = Column(String(50), nullable=False)  # audio, transcript, content
    local_path = Column(String(500))
    gdrive_file_id = Column(String(100))
    gdrive_url = Column(String(500))
    sync_status = Column(String(20), default='pending')  # pending, synced, failed
    last_synced = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Indexes
    __table_args__ = (
        Index('idx_storage_sync', 'sync_status', 'file_type'),
    )


# ========================================
# Prompt Management Tables
# ========================================

class Prompt(Base):
    """Prompt templates for content generation."""
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    content_type = Column(String(50), nullable=False)  # blog_post, summary, social_media, etc.
    description = Column(Text)
    current_version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)  # System default prompt
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    versions = relationship("PromptVersion", back_populates="prompt", cascade="all, delete-orphan")
    experiments = relationship("PromptExperiment", back_populates="prompt")
    analytics = relationship("PromptAnalytics", back_populates="prompt")
    
    # Indexes
    __table_args__ = (
        Index('idx_prompts_type_active', 'content_type', 'is_active'),
        UniqueConstraint('content_type', 'is_default', name='uq_one_default_per_type'),
    )


class PromptVersion(Base):
    """Version history for prompt templates."""
    __tablename__ = 'prompt_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    version = Column(Integer, nullable=False)
    template = Column(Text, nullable=False)  # The actual prompt template
    variables = Column(JSON)  # List of template variables used
    changelog = Column(Text)  # Description of changes
    performance_score = Column(Float)  # Average quality score for this version
    created_by = Column(String(100))  # User or system that created this version
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("Prompt", back_populates="versions")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('prompt_id', 'version', name='uq_prompt_version'),
        Index('idx_prompt_versions', 'prompt_id', 'version'),
    )


class PromptExperiment(Base):
    """A/B testing experiments for prompts."""
    __tablename__ = 'prompt_experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    content_type = Column(String(50), nullable=False)
    prompt_id = Column(Integer, ForeignKey('prompts.id'))  # Control prompt
    variants = Column(JSON, nullable=False)  # List of variant configurations
    traffic_split = Column(JSON)  # Traffic distribution percentages
    metrics = Column(JSON)  # Metrics to track (quality_score, tokens_used, etc.)
    winner_criteria = Column(JSON)  # Criteria for selecting winner
    status = Column(String(20), default='draft')  # draft, active, paused, completed
    winner_variant = Column(String(50))  # ID of winning variant
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    prompt = relationship("Prompt", back_populates="experiments")
    analytics = relationship("PromptAnalytics", back_populates="experiment")
    
    # Indexes
    __table_args__ = (
        Index('idx_experiments_status', 'status', 'content_type'),
        Index('idx_experiments_dates', 'start_date', 'end_date'),
    )


class PromptAnalytics(Base):
    """Analytics and performance tracking for prompts."""
    __tablename__ = 'prompt_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    prompt_version = Column(Integer)
    experiment_id = Column(Integer, ForeignKey('prompt_experiments.id'))
    variant_id = Column(String(50))  # Variant identifier if part of experiment
    generation_id = Column(Integer, ForeignKey('generated_content.id'))
    
    # Performance metrics
    quality_score = Column(Float)  # AI-evaluated quality score
    tokens_used = Column(Integer)  # Total tokens consumed
    execution_time = Column(Float)  # Generation time in seconds
    user_feedback = Column(Integer)  # User rating if provided (1-5)
    
    # Additional metadata
    model_used = Column(String(50))  # AI model used
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("Prompt", back_populates="analytics")
    experiment = relationship("PromptExperiment", back_populates="analytics")
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_prompt', 'prompt_id', 'created_at'),
        Index('idx_analytics_experiment', 'experiment_id', 'variant_id'),
        Index('idx_analytics_performance', 'quality_score', 'tokens_used'),
    )


# ========================================
# Transcript Quality Management Tables
# ========================================

class TranscriptQualityPrompt(Base):
    """Prompts specifically for transcript quality evaluation."""
    __tablename__ = 'transcript_quality_prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    strictness_level = Column(String(20), default='standard')  # standard, strict, lenient
    extraction_method = Column(String(50))  # whisper, ffmpeg, youtube-api, any
    current_version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    min_duration = Column(Integer)  # Minimum video duration for this prompt
    max_duration = Column(Integer)  # Maximum video duration for this prompt
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    versions = relationship("TranscriptQualityVersion", back_populates="prompt", cascade="all, delete-orphan")
    experiments = relationship("TranscriptQualityExperiment", back_populates="prompt")
    
    # Indexes
    __table_args__ = (
        Index('idx_tq_prompts_active', 'is_active', 'strictness_level'),
        UniqueConstraint('strictness_level', 'is_default', name='uq_one_default_per_strictness'),
    )


class TranscriptQualityVersion(Base):
    """Version history for transcript quality prompts."""
    __tablename__ = 'transcript_quality_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('transcript_quality_prompts.id'), nullable=False)
    version = Column(Integer, nullable=False)
    template = Column(Text, nullable=False)
    evaluation_criteria = Column(JSON)  # Specific criteria for this version
    changelog = Column(Text)
    accuracy_score = Column(Float)  # Historical accuracy of this version
    created_by = Column(String(100))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("TranscriptQualityPrompt", back_populates="versions")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('prompt_id', 'version', name='uq_tq_prompt_version'),
    )


class TranscriptQualityExperiment(Base):
    """A/B testing for transcript quality evaluation prompts."""
    __tablename__ = 'transcript_quality_experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    prompt_id = Column(Integer, ForeignKey('transcript_quality_prompts.id'))
    strictness_level = Column(String(20))
    variants = Column(JSON, nullable=False)
    traffic_split = Column(JSON)
    success_metrics = Column(JSON)  # accuracy, false_positive_rate, false_negative_rate
    status = Column(String(20), default='draft')
    winner_variant = Column(String(50))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("TranscriptQualityPrompt", back_populates="experiments")
    
    # Indexes
    __table_args__ = (
        Index('idx_tq_experiments_status', 'status', 'strictness_level'),
    )


# ========================================
# Content Quality Management Tables
# ========================================

class ContentQualityPrompt(Base):
    """Prompts specifically for generated content quality evaluation."""
    __tablename__ = 'content_quality_prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    content_type = Column(String(50), nullable=False)  # blog, social, summary, etc.
    platform = Column(String(50))  # twitter, linkedin, instagram, etc.
    description = Column(Text)
    evaluation_focus = Column(JSON)  # List of focus areas: engagement, seo, accuracy, etc.
    current_version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    min_word_count = Column(Integer)
    max_word_count = Column(Integer)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    versions = relationship("ContentQualityVersion", back_populates="prompt", cascade="all, delete-orphan")
    experiments = relationship("ContentQualityExperiment", back_populates="prompt")
    
    # Indexes
    __table_args__ = (
        Index('idx_cq_prompts_type_platform', 'content_type', 'platform', 'is_active'),
        UniqueConstraint('content_type', 'platform', 'is_default', name='uq_one_default_per_type_platform'),
    )


class ContentQualityVersion(Base):
    """Version history for content quality prompts."""
    __tablename__ = 'content_quality_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('content_quality_prompts.id'), nullable=False)
    version = Column(Integer, nullable=False)
    template = Column(Text, nullable=False)
    evaluation_rubric = Column(JSON)  # Detailed scoring rubric
    platform_requirements = Column(JSON)  # Platform-specific requirements
    changelog = Column(Text)
    avg_quality_score = Column(Float)  # Average quality score with this version
    created_by = Column(String(100))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("ContentQualityPrompt", back_populates="versions")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('prompt_id', 'version', name='uq_cq_prompt_version'),
    )


class ContentQualityExperiment(Base):
    """A/B testing for content quality evaluation prompts."""
    __tablename__ = 'content_quality_experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    content_type = Column(String(50), nullable=False)
    platform = Column(String(50))
    prompt_id = Column(Integer, ForeignKey('content_quality_prompts.id'))
    variants = Column(JSON, nullable=False)
    traffic_split = Column(JSON)
    success_metrics = Column(JSON)  # engagement_score, seo_score, accuracy_score
    status = Column(String(20), default='draft')
    winner_variant = Column(String(50))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    prompt = relationship("ContentQualityPrompt", back_populates="experiments")
    
    # Indexes
    __table_args__ = (
        Index('idx_cq_experiments_type_status', 'content_type', 'platform', 'status'),
    )


class TranscriptQualityAnalytics(Base):
    """Analytics for transcript quality evaluations."""
    __tablename__ = 'transcript_quality_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('transcript_quality_prompts.id'))
    prompt_version = Column(Integer)
    experiment_id = Column(Integer, ForeignKey('transcript_quality_experiments.id'))
    variant_id = Column(String(50))
    transcript_id = Column(Integer, ForeignKey('transcripts.id'))
    
    # Evaluation results
    quality_score = Column(Float)
    completeness_score = Column(Float)
    coherence_score = Column(Float)
    accuracy_score = Column(Float)
    
    # Detection metrics
    issues_detected = Column(Integer)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    
    # Performance metrics
    tokens_used = Column(Integer)
    execution_time = Column(Float)
    model_used = Column(String(50))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Indexes
    __table_args__ = (
        Index('idx_tq_analytics_prompt', 'prompt_id', 'created_at'),
        Index('idx_tq_analytics_experiment', 'experiment_id', 'variant_id'),
    )


class ContentQualityAnalytics(Base):
    """Analytics for content quality evaluations."""
    __tablename__ = 'content_quality_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('content_quality_prompts.id'))
    prompt_version = Column(Integer)
    experiment_id = Column(Integer, ForeignKey('content_quality_experiments.id'))
    variant_id = Column(String(50))
    content_id = Column(Integer, ForeignKey('generated_content.id'))
    
    # Evaluation results
    overall_score = Column(Float)
    engagement_score = Column(Float)
    seo_score = Column(Float)
    readability_score = Column(Float)
    accuracy_to_source = Column(Float)
    
    # Platform-specific scores
    platform_appropriateness = Column(Float)
    format_compliance = Column(Float)
    
    # Performance metrics
    tokens_used = Column(Integer)
    execution_time = Column(Float)
    model_used = Column(String(50))
    user_feedback = Column(Integer)  # 1-5 rating if provided
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Indexes
    __table_args__ = (
        Index('idx_cq_analytics_prompt', 'prompt_id', 'created_at'),
        Index('idx_cq_analytics_experiment', 'experiment_id', 'variant_id'),
        Index('idx_cq_analytics_content', 'content_id'),
    )


# ========================================
# AI Provider A/B Testing Tables
# ========================================

class AIProviderExperiment(Base):
    """Experiments comparing different AI providers."""
    __tablename__ = 'ai_provider_experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False)
    task_type = Column(String(50), nullable=False)  # transcript_quality, content_quality, content_generation
    status = Column(String(20), default='draft')  # draft, active, paused, completed, archived
    
    # Providers being tested
    providers = Column(JSON, nullable=False)  # List of provider IDs
    traffic_split = Column(JSON)  # Traffic distribution per provider
    
    # Experiment configuration
    min_samples = Column(Integer, default=100)
    confidence_level = Column(Float, default=0.95)
    primary_metric = Column(String(50), default='quality_score')  # quality_score, cost, latency
    
    # Dates
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Results
    winner_provider = Column(String(50))
    winner_metrics = Column(JSON)
    analysis_results = Column(JSON)
    
    # Relationships
    analytics = relationship("AIProviderAnalytics", back_populates="experiment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_provider_exp_status', 'status', 'task_type'),
        Index('idx_provider_exp_dates', 'start_date', 'end_date'),
    )


class AIProviderAnalytics(Base):
    """Analytics tracking for AI provider performance."""
    __tablename__ = 'ai_provider_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('ai_provider_experiments.id'))
    provider = Column(String(50), nullable=False)  # claude_cli, claude_api, openai_api, gemini_api
    task_type = Column(String(50), nullable=False)
    
    # Request details
    request_id = Column(String(100), unique=True)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    
    # Performance metrics
    latency_ms = Column(Float)  # Response time in milliseconds
    success = Column(Boolean, default=True)
    error_type = Column(String(100))
    error_message = Column(Text)
    
    # Quality metrics
    quality_score = Column(Float)  # Overall quality score
    accuracy_score = Column(Float)  # Accuracy of response
    coherence_score = Column(Float)  # Response coherence
    
    # Cost tracking
    cost_usd = Column(Float)  # Cost in USD
    model_used = Column(String(100))  # Specific model version
    
    # Content reference
    transcript_id = Column(Integer, ForeignKey('transcripts.id'))
    content_id = Column(Integer, ForeignKey('generated_content.id'))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    experiment = relationship("AIProviderExperiment", back_populates="analytics")
    
    # Indexes
    __table_args__ = (
        Index('idx_provider_analytics_exp', 'experiment_id', 'provider'),
        Index('idx_provider_analytics_perf', 'provider', 'task_type', 'success'),
        Index('idx_provider_analytics_cost', 'provider', 'cost_usd'),
        Index('idx_provider_analytics_quality', 'quality_score', 'provider'),
    )


class AIProviderCosts(Base):
    """Track cumulative costs per AI provider."""
    __tablename__ = 'ai_provider_costs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), unique=True, nullable=False)
    
    # Cumulative metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Token usage
    total_prompt_tokens = Column(BigInteger, default=0)
    total_completion_tokens = Column(BigInteger, default=0)
    total_tokens = Column(BigInteger, default=0)
    
    # Cost tracking
    total_cost_usd = Column(Float, default=0.0)
    average_cost_per_request = Column(Float, default=0.0)
    cost_per_1k_tokens = Column(Float)  # Current pricing
    
    # Performance
    average_latency_ms = Column(Float)
    average_quality_score = Column(Float)
    error_rate = Column(Float, default=0.0)
    
    # Limits and quotas
    daily_limit = Column(Integer)  # Daily request limit
    monthly_limit = Column(Integer)  # Monthly request limit
    rate_limit = Column(Integer)  # Requests per minute
    
    # Last updated
    last_request_at = Column(DateTime)
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Indexes
    __table_args__ = (
        Index('idx_provider_costs_provider', 'provider'),
        Index('idx_provider_costs_updated', 'updated_at'),
    )


# ========================================
# Database Connection Management
# ========================================

class DatabaseManager:
    """Async database connection and session management."""
    
    def __init__(self, database_url: str = "sqlite+aiosqlite:///data/yt-dl-sub.db"):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Initialize async database engine and session factory."""
        # Ensure data directory exists
        db_path = Path(self.database_url.replace("sqlite+aiosqlite:///", ""))
        db_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create async engine with performance optimizations
        engine_kwargs = {
            "echo": False,  # Set to True for SQL debugging
            "future": True,
        }
        
        # Add pooling only for non-SQLite databases
        if not self.database_url.startswith('sqlite'):
            engine_kwargs.update({
                "pool_size": 2,
                "max_overflow": 3,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
            })
        
        # Add SQLite-specific optimizations if using SQLite
        if self.database_url.startswith('sqlite'):
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30,  # Timeout for database operations
            }
        
        self.engine = create_async_engine(self.database_url, **engine_kwargs)
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Setup FTS5 virtual tables and triggers
        await self._setup_fts5()
    
    async def _setup_fts5(self):
        """Setup FTS5 virtual tables and triggers for full-text search."""
        # Get raw SQLite connection for FTS5 setup
        raw_db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            # Enable FTS5
            await db.execute("PRAGMA table_info=fts5")
            
            # Create FTS5 virtual table for transcripts
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                    video_id UNINDEXED,
                    content_text,
                    content='transcripts',
                    content_rowid='id'
                )
            """)
            
            # Create FTS5 virtual table for generated content
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    video_id,
                    content_type,
                    content,
                    content=generated_content,
                    content_rowid=id
                )
            """)
            
            # Create triggers to keep FTS5 tables in sync
            # Transcript FTS triggers
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_insert AFTER INSERT ON transcripts
                BEGIN
                    INSERT INTO transcripts_fts(rowid, video_id, content_text)
                    VALUES (NEW.id, NEW.video_id, COALESCE(NEW.content_text, ''));
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_update AFTER UPDATE ON transcripts
                BEGIN
                    UPDATE transcripts_fts SET 
                        video_id = NEW.video_id,
                        content_text = COALESCE(NEW.content_text, '')
                    WHERE rowid = NEW.id;
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS transcripts_fts_delete AFTER DELETE ON transcripts
                BEGIN
                    DELETE FROM transcripts_fts WHERE rowid = OLD.id;
                END
            """)
            
            # Generated content FTS triggers
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS content_fts_insert AFTER INSERT ON generated_content
                BEGIN
                    INSERT INTO content_fts(rowid, video_id, content_type, content)
                    VALUES (NEW.id, NEW.video_id, NEW.content_type, NEW.content);
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS content_fts_update AFTER UPDATE ON generated_content
                BEGIN
                    UPDATE content_fts SET 
                        video_id = NEW.video_id,
                        content_type = NEW.content_type,
                        content = NEW.content
                    WHERE rowid = NEW.id;
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS content_fts_delete AFTER DELETE ON generated_content
                BEGIN
                    DELETE FROM content_fts WHERE rowid = OLD.id;
                END
            """)
            
            await db.commit()
    
    @asynccontextmanager
    async def get_session(self):
        """Get an async database session."""
        if not self.session_factory:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            else:
                await session.commit()
    
    # ========================================
    # Bulk Operations for Performance
    # ========================================
    
    async def bulk_insert_videos(self, videos: List[Dict[str, Any]]) -> None:
        """Bulk insert multiple video records for performance."""
        if not videos:
            return
            
        async with self.session() as session:
            try:
                # Use bulk_insert_mappings for better performance
                await session.execute(Video.__table__.insert(), videos)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def bulk_update_videos(self, updates: List[Dict[str, Any]]) -> None:
        """Bulk update video records by video_id."""
        if not updates:
            return
            
        async with self.session() as session:
            try:
                # Prepare bulk update statement
                from sqlalchemy import update
                stmt = update(Video)
                
                for update_data in updates:
                    # Create a copy to avoid modifying original data
                    data_copy = update_data.copy()
                    video_id = data_copy.pop('video_id')  # Remove key for WHERE clause
                    update_stmt = stmt.where(Video.video_id == video_id).values(**data_copy)
                    await session.execute(update_stmt)
                    
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def bulk_insert_generated_content(self, content_list: List[Dict[str, Any]]) -> None:
        """Bulk insert generated content records."""
        if not content_list:
            return
            
        async with self.session() as session:
            try:
                await session.execute(GeneratedContent.__table__.insert(), content_list)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def get_videos_batch(self, video_ids: List[str]) -> List[Video]:
        """Fetch multiple videos in a single query for performance."""
        async with self.session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Video).where(Video.video_id.in_(video_ids))
            )
            return result.scalars().all()
    
    async def update_processing_status_batch(self, video_ids: List[str], status: str) -> None:
        """Update processing status for multiple videos at once."""
        if not video_ids:
            return
            
        async with self.session() as session:
            try:
                from sqlalchemy import update
                await session.execute(
                    update(Video)
                    .where(Video.video_id.in_(video_ids))
                    .values(
                        transcript_status=status,
                        updated_at=func.current_timestamp()
                    )
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    # ========================================
    # Connection Management & Caching
    # ========================================
    
    async def execute_with_retry(self, operation, max_retries: int = 3):
        """Execute database operation with automatic retry on connection errors."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                break
        
        raise last_exception
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()


# ========================================
# Search Functions
# ========================================

class SearchService:
    """Full-text search service using FTS5."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def search_transcripts(
        self, 
        query: str, 
        limit: int = 50,
        channel_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search transcripts using FTS5."""
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            sql = """
                SELECT 
                    fts.video_id,
                    fts.title,
                    v.channel_id,
                    c.channel_name,
                    snippet(transcripts_fts, 2, '<mark>', '</mark>', '...', 64) as snippet,
                    bm25(transcripts_fts) as rank
                FROM transcripts_fts fts
                JOIN videos v ON fts.video_id = v.video_id
                JOIN channels c ON v.channel_id = c.channel_id
                WHERE transcripts_fts MATCH ?
            """
            params = [query]
            
            if channel_id:
                sql += " AND v.channel_id = ?"
                params.append(channel_id)
            
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    async def search_content(
        self, 
        query: str, 
        content_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search generated content using FTS5."""
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            sql = """
                SELECT 
                    fts.video_id,
                    v.title,
                    fts.content_type,
                    snippet(content_fts, 2, '<mark>', '</mark>', '...', 64) as snippet,
                    bm25(content_fts) as rank
                FROM content_fts fts
                JOIN generated_content gc ON fts.rowid = gc.id
                JOIN videos v ON gc.video_id = v.video_id
                WHERE content_fts MATCH ?
            """
            params = [query]
            
            if content_type:
                sql += " AND fts.content_type = ?"
                params.append(content_type)
            
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]


# ========================================
# Migration and Setup
# ========================================

async def create_database(database_url: str = "sqlite+aiosqlite:///data/yt-dl-sub.db"):
    """Create database with all tables and FTS5 setup."""
    db_manager = DatabaseManager(database_url)
    await db_manager.initialize()
    return db_manager


async def reset_database(database_url: str = "sqlite+aiosqlite:///data/yt-dl-sub.db"):
    """Reset database by dropping and recreating all tables."""
    db_path = Path(database_url.replace("sqlite+aiosqlite:///", ""))
    
    # Remove existing database file
    if db_path.exists():
        db_path.unlink()
    
    # Recreate database
    return await create_database(database_url)


# ========================================
# Global instance
# ========================================

# Global database manager instance
from config.settings import get_settings
settings = get_settings()
db_manager = DatabaseManager(settings.database_url)


# ========================================
# CLI Setup Script
# ========================================

async def setup_database_cli():
    """CLI command to setup the database."""
    print("Setting up SQLite database with FTS5...")
    
    try:
        db_manager = await create_database()
        print("✅ Database created successfully")
        print("✅ Tables created")
        print("✅ FTS5 virtual tables setup")
        print("✅ Indexes created")
        print("✅ Triggers setup")
        
        # Test database connection
        async with db_manager.get_session() as session:
            # Test basic functionality
            from sqlalchemy import text
            result = await session.execute(text("SELECT COUNT(*) FROM channels"))
            count = result.scalar()
            print(f"✅ Database connection test passed (channels table has {count} records)")
        
        await db_manager.close()
        print("\n🎉 Database setup complete!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise


if __name__ == "__main__":
    # Run setup when executed directly
    asyncio.run(setup_database_cli())