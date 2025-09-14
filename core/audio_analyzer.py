"""
Audio Analysis Module for Whisper Timeout Prevention

This module provides pre-flight analysis of audio files to:
- Determine duration and size for timeout calculation
- Validate audio format and integrity
- Estimate resource requirements and processing time
- Recommend chunking strategy for long audio
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisResult:
    """Results from audio file analysis."""
    duration_seconds: float
    file_size_mb: float
    format: str
    channels: int
    sample_rate: int
    bitrate: Optional[int]
    is_valid: bool
    recommended_timeout: int
    requires_chunking: bool
    chunk_count: int
    estimated_memory_mb: int
    error: Optional[str] = None


class AudioAnalyzer:
    """Analyzes audio files for Whisper transcription planning."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_audio(
        self, 
        audio_path: str,
        whisper_timeout_base: int = 300,
        whisper_timeout_per_minute: float = 2.0,
        whisper_max_duration: int = 7200,
        whisper_chunk_duration: int = 1800,
        whisper_model: str = "base"
    ) -> AudioAnalysisResult:
        """
        Analyze audio file for Whisper transcription planning.
        
        Args:
            audio_path: Path to audio file
            whisper_timeout_base: Base timeout in seconds
            whisper_timeout_per_minute: Additional timeout per minute of audio
            whisper_max_duration: Maximum audio duration to process
            whisper_chunk_duration: Duration for chunking long audio
            whisper_model: Whisper model size for memory estimation
            
        Returns:
            AudioAnalysisResult with analysis and recommendations
        """
        try:
            file_path = Path(audio_path)
            
            # Check if file exists
            if not file_path.exists():
                return AudioAnalysisResult(
                    duration_seconds=0,
                    file_size_mb=0,
                    format="unknown",
                    channels=0,
                    sample_rate=0,
                    bitrate=None,
                    is_valid=False,
                    recommended_timeout=0,
                    requires_chunking=False,
                    chunk_count=0,
                    estimated_memory_mb=0,
                    error=f"Audio file not found: {audio_path}"
                )
            
            # Get file size
            file_size_bytes = file_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Use ffprobe to get audio metadata
            audio_info = self._get_audio_info_with_ffprobe(audio_path)
            
            if not audio_info:
                # Fallback to basic file analysis
                return self._basic_file_analysis(file_path, file_size_mb)
            
            # Extract key information
            duration_seconds = float(audio_info.get('duration', 0))
            format_name = audio_info.get('format_name', 'unknown')
            
            # Get stream information
            streams = audio_info.get('streams', [])
            audio_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            channels = int(audio_stream.get('channels', 2)) if audio_stream else 2
            sample_rate = int(audio_stream.get('sample_rate', 44100)) if audio_stream else 44100
            bitrate = int(audio_stream.get('bit_rate', 0)) if audio_stream and audio_stream.get('bit_rate') else None
            
            # Calculate timeout recommendation
            duration_minutes = duration_seconds / 60
            recommended_timeout = int(whisper_timeout_base + (duration_minutes * whisper_timeout_per_minute))
            
            # Determine if chunking is required
            requires_chunking = duration_seconds > whisper_chunk_duration
            chunk_count = max(1, int(duration_seconds / whisper_chunk_duration)) if requires_chunking else 1
            
            # Estimate memory requirements based on model and duration
            estimated_memory_mb = self._estimate_memory_requirements(whisper_model, duration_seconds, channels, sample_rate)
            
            # Validate audio is processable
            is_valid = (
                duration_seconds > 0 and
                duration_seconds <= whisper_max_duration and
                file_size_mb > 0.001  # At least 1KB
            )
            
            error = None
            if not is_valid:
                if duration_seconds <= 0:
                    error = "Audio duration is zero or invalid"
                elif duration_seconds > whisper_max_duration:
                    error = f"Audio too long ({duration_seconds:.1f}s > {whisper_max_duration}s max)"
                elif file_size_mb <= 0.001:
                    error = "Audio file too small or empty"
            
            return AudioAnalysisResult(
                duration_seconds=duration_seconds,
                file_size_mb=file_size_mb,
                format=format_name,
                channels=channels,
                sample_rate=sample_rate,
                bitrate=bitrate,
                is_valid=is_valid,
                recommended_timeout=recommended_timeout,
                requires_chunking=requires_chunking,
                chunk_count=chunk_count,
                estimated_memory_mb=estimated_memory_mb,
                error=error
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audio file {audio_path}: {e}")
            return AudioAnalysisResult(
                duration_seconds=0,
                file_size_mb=0,
                format="unknown",
                channels=0,
                sample_rate=0,
                bitrate=None,
                is_valid=False,
                recommended_timeout=0,
                requires_chunking=False,
                chunk_count=0,
                estimated_memory_mb=0,
                error=f"Analysis failed: {str(e)}"
            )
    
    def _get_audio_info_with_ffprobe(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Use ffprobe to get detailed audio information."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                audio_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout for ffprobe
            )
            
            if result.returncode != 0:
                self.logger.warning(f"ffprobe failed for {audio_path}: {result.stderr}")
                return None
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"ffprobe timeout for {audio_path}")
            return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse ffprobe output for {audio_path}: {e}")
            return None
        except FileNotFoundError:
            self.logger.warning("ffprobe not available - using basic analysis")
            return None
        except Exception as e:
            self.logger.warning(f"ffprobe error for {audio_path}: {e}")
            return None
    
    def _basic_file_analysis(self, file_path: Path, file_size_mb: float) -> AudioAnalysisResult:
        """Basic file analysis when ffprobe is not available."""
        # Rough estimation based on file size and format
        extension = file_path.suffix.lower()
        
        # Estimate duration based on file size and format (very rough)
        if extension in ['.opus', '.ogg']:
            # Opus typically 64kbps, so ~8KB per second
            estimated_duration = file_size_mb * 1024 * 1024 / (8 * 1024)  # seconds
        elif extension in ['.mp3']:
            # MP3 typically 128kbps, so ~16KB per second  
            estimated_duration = file_size_mb * 1024 * 1024 / (16 * 1024)
        elif extension in ['.wav']:
            # WAV uncompressed, typically much larger
            estimated_duration = file_size_mb * 1024 * 1024 / (176400 * 2)  # 44.1kHz stereo 16-bit
        else:
            # Generic estimation
            estimated_duration = file_size_mb * 60  # Assume 1MB per minute
        
        return AudioAnalysisResult(
            duration_seconds=max(1, estimated_duration),  # At least 1 second
            file_size_mb=file_size_mb,
            format=extension,
            channels=2,  # Assume stereo
            sample_rate=44100,  # Standard sample rate
            bitrate=None,
            is_valid=file_size_mb > 0.001,
            recommended_timeout=max(300, int(estimated_duration * 2 + 120)),  # 2x duration + 2 minutes
            requires_chunking=estimated_duration > 1800,  # 30 minutes
            chunk_count=max(1, int(estimated_duration / 1800)),
            estimated_memory_mb=self._estimate_memory_requirements("base", estimated_duration, 2, 44100),
            error="Basic analysis only - ffprobe not available" if file_size_mb > 0.001 else "File too small"
        )
    
    def _estimate_memory_requirements(
        self, 
        model_size: str, 
        duration_seconds: float, 
        channels: int, 
        sample_rate: int
    ) -> int:
        """Estimate memory requirements for Whisper transcription."""
        # Base memory requirements for different model sizes (MB)
        model_memory = {
            'tiny': 500,
            'base': 1000,
            'small': 2000,
            'medium': 4000,
            'large': 6000,
            'large-v2': 6000,
            'large-v3': 6000
        }
        
        base_memory = model_memory.get(model_size, 1000)
        
        # Additional memory based on audio length and quality
        # Roughly 1MB per minute of audio for processing
        audio_memory = int(duration_seconds / 60 * channels * (sample_rate / 44100))
        
        # Add buffer for processing overhead (50% more)
        total_memory = int((base_memory + audio_memory) * 1.5)
        
        return total_memory
    
    def validate_for_whisper(self, analysis: AudioAnalysisResult) -> Tuple[bool, Optional[str]]:
        """
        Validate if audio is suitable for Whisper transcription.
        
        Args:
            analysis: AudioAnalysisResult from analyze_audio()
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not analysis.is_valid:
            return False, analysis.error
        
        if analysis.duration_seconds < 0.1:
            return False, "Audio too short (less than 0.1 seconds)"
        
        if analysis.file_size_mb < 0.001:
            return False, "Audio file too small (less than 1KB)"
        
        if analysis.estimated_memory_mb > 16384:  # 16GB limit
            return False, f"Estimated memory requirement too high: {analysis.estimated_memory_mb}MB"
        
        return True, None
    
    def get_chunking_plan(
        self, 
        analysis: AudioAnalysisResult, 
        chunk_duration: int = 1800,
        overlap_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Generate chunking plan for long audio files.
        
        Args:
            analysis: AudioAnalysisResult from analyze_audio()
            chunk_duration: Duration of each chunk in seconds
            overlap_seconds: Overlap between chunks for accuracy
            
        Returns:
            Dict with chunking plan details
        """
        if not analysis.requires_chunking:
            return {
                'chunks_needed': False,
                'chunk_count': 1,
                'chunks': [{'start': 0, 'end': analysis.duration_seconds, 'index': 0}]
            }
        
        chunks = []
        current_start = 0
        chunk_index = 0
        
        while current_start < analysis.duration_seconds:
            chunk_end = min(current_start + chunk_duration, analysis.duration_seconds)
            
            chunks.append({
                'index': chunk_index,
                'start': current_start,
                'end': chunk_end,
                'duration': chunk_end - current_start,
                'overlap_start': max(0, current_start - overlap_seconds) if chunk_index > 0 else 0,
                'overlap_end': min(analysis.duration_seconds, chunk_end + overlap_seconds) if chunk_end < analysis.duration_seconds else chunk_end
            })
            
            current_start = chunk_end - overlap_seconds  # Overlap for accuracy
            chunk_index += 1
        
        return {
            'chunks_needed': True,
            'chunk_count': len(chunks),
            'chunks': chunks,
            'total_processing_time': sum(chunk['end'] - chunk['start'] for chunk in chunks),
            'overlap_seconds': overlap_seconds
        }