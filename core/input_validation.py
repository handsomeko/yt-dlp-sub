"""
Comprehensive Input Validation System
Provides centralized validation for all user inputs, external data, and system interfaces.
"""

import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse, parse_qs
from enum import Enum
import mimetypes

from pydantic import BaseModel, Field, field_validator, ValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """Result of input validation with detailed feedback."""
    
    def __init__(self, valid: bool = True, value: Any = None):
        self.valid = valid
        self.value = value
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.sanitized_value: Any = value
        self.metadata: Dict[str, Any] = {}
    
    def add_error(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add validation error."""
        if severity == ValidationSeverity.ERROR or severity == ValidationSeverity.CRITICAL:
            self.valid = False
            self.errors.append(message)
        else:
            self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "value": self.value,
            "sanitized_value": self.sanitized_value,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }
    
    def __bool__(self) -> bool:
        return self.valid


class URLValidator:
    """Comprehensive URL validation and sanitization."""
    
    YOUTUBE_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/channel/([a-zA-Z0-9_-]{24})',
        r'(?:https?://)?(?:www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/@([a-zA-Z0-9_.-]+)'
    ]
    
    DANGEROUS_URL_PATTERNS = [
        r'javascript:', r'data:', r'vbscript:', r'file:', r'ftp:',
        r'\.\./.*', r'[<>"\']', r'[\x00-\x1f\x7f-\x9f]'
    ]
    
    @classmethod
    def validate_url(cls, url: str, allow_youtube_only: bool = False) -> ValidationResult:
        """Validate URL with comprehensive security and format checking."""
        result = ValidationResult()
        
        if not url or not isinstance(url, str):
            result.add_error("URL cannot be empty or non-string", ValidationSeverity.ERROR)
            return result
        
        # Sanitize URL
        url = url.strip()
        result.sanitized_value = url
        
        # Length validation
        if len(url) > 2000:
            result.add_error("URL too long (max 2000 characters)", ValidationSeverity.ERROR)
            return result
        
        if len(url) < 7:  # Minimum for http://
            result.add_error("URL too short", ValidationSeverity.ERROR)
            return result
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                result.add_error(f"URL contains dangerous pattern: {pattern}", ValidationSeverity.CRITICAL)
                return result
        
        # Parse URL
        try:
            parsed = urlparse(url)
            
            # Validate scheme
            if not parsed.scheme:
                # Try adding https://
                url = f"https://{url}"
                parsed = urlparse(url)
                result.sanitized_value = url
                result.add_error("Added missing https:// scheme", ValidationSeverity.INFO)
            
            if parsed.scheme not in ['http', 'https']:
                result.add_error(f"Invalid URL scheme: {parsed.scheme}", ValidationSeverity.ERROR)
                return result
            
            # Validate hostname
            if not parsed.hostname:
                result.add_error("URL missing hostname", ValidationSeverity.ERROR)
                return result
            
            # YouTube-specific validation
            if allow_youtube_only:
                if not cls.is_youtube_url(url):
                    result.add_error("URL must be from YouTube", ValidationSeverity.ERROR)
                    return result
                
                youtube_info = cls.extract_youtube_info(url)
                result.metadata.update(youtube_info)
            
            result.metadata.update({
                'scheme': parsed.scheme,
                'hostname': parsed.hostname,
                'path': parsed.path,
                'is_youtube': cls.is_youtube_url(url)
            })
            
        except Exception as e:
            result.add_error(f"URL parsing failed: {e}", ValidationSeverity.ERROR)
            return result
        
        return result
    
    @classmethod
    def is_youtube_url(cls, url: str) -> bool:
        """Check if URL is from YouTube."""
        for pattern in cls.YOUTUBE_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def extract_youtube_info(cls, url: str) -> Dict[str, Any]:
        """Extract YouTube video/channel information from URL."""
        info = {'type': 'unknown', 'id': None}
        
        # Video ID patterns
        video_patterns = [
            (r'[?&]v=([a-zA-Z0-9_-]{11})', 'video'),
            (r'/embed/([a-zA-Z0-9_-]{11})', 'video'), 
            (r'/v/([a-zA-Z0-9_-]{11})', 'video'),
            (r'youtu\.be/([a-zA-Z0-9_-]{11})', 'video')
        ]
        
        for pattern, url_type in video_patterns:
            match = re.search(pattern, url)
            if match:
                info['type'] = url_type
                info['id'] = match.group(1)
                return info
        
        # Playlist patterns
        playlist_match = re.search(r'[?&]list=([a-zA-Z0-9_-]+)', url)
        if playlist_match:
            info['type'] = 'playlist'
            info['id'] = playlist_match.group(1)
            return info
        
        # Channel patterns
        channel_patterns = [
            (r'/channel/([a-zA-Z0-9_-]{24})', 'channel'),
            (r'/c/([a-zA-Z0-9_-]+)', 'channel'),
            (r'/@([a-zA-Z0-9_.-]+)', 'channel')
        ]
        
        for pattern, url_type in channel_patterns:
            match = re.search(pattern, url)
            if match:
                info['type'] = url_type
                info['id'] = match.group(1)
                return info
        
        return info


class FileValidator:
    """File path and content validation."""
    
    DANGEROUS_FILENAME_PATTERNS = [
        r'\.\./.*', r'~/', r'\$', r'`', r';', r'\|', r'<', r'>', r'"', r"'",
        r'[\x00-\x1f\x7f-\x9f]', r'^\.*$', r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$'
    ]
    
    ALLOWED_EXTENSIONS = {
        'video': ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'],
        'audio': ['.mp3', '.wav', '.opus', '.m4a', '.flac', '.ogg'],
        'transcript': ['.srt', '.vtt', '.txt', '.json'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp'],
        'document': ['.pdf', '.txt', '.md', '.json', '.yaml', '.yml']
    }
    
    @classmethod
    def validate_filename(cls, filename: str, allowed_types: List[str] = None) -> ValidationResult:
        """Validate filename for security and format compliance."""
        result = ValidationResult()
        
        if not filename or not isinstance(filename, str):
            result.add_error("Filename cannot be empty", ValidationSeverity.ERROR)
            return result
        
        # Sanitize filename
        filename = filename.strip()
        result.sanitized_value = cls.sanitize_filename(filename)
        
        # Length validation
        if len(filename) > 255:
            result.add_error("Filename too long (max 255 characters)", ValidationSeverity.ERROR)
            return result
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_FILENAME_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                result.add_error(f"Filename contains dangerous pattern: {pattern}", ValidationSeverity.ERROR)
                return result
        
        # Extension validation
        if allowed_types:
            file_ext = Path(filename).suffix.lower()
            allowed_extensions = []
            
            for file_type in allowed_types:
                allowed_extensions.extend(cls.ALLOWED_EXTENSIONS.get(file_type, []))
            
            if file_ext not in allowed_extensions:
                result.add_error(f"Invalid file extension. Allowed: {allowed_extensions}", ValidationSeverity.ERROR)
                return result
        
        result.metadata = {
            'extension': Path(filename).suffix.lower(),
            'name': Path(filename).stem,
            'mime_type': mimetypes.guess_type(filename)[0]
        }
        
        return result
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path], must_exist: bool = False) -> ValidationResult:
        """Validate file path for security and accessibility."""
        result = ValidationResult()
        
        if not file_path:
            result.add_error("File path cannot be empty", ValidationSeverity.ERROR)
            return result
        
        try:
            path = Path(file_path).expanduser().absolute()
            result.sanitized_value = path
            
            # Security validation
            path_str = str(path)
            if any(dangerous in path_str for dangerous in ['..', '$', '`', ';', '|']):
                result.add_error(f"File path contains dangerous characters", ValidationSeverity.ERROR)
                return result
            
            # Existence validation
            if must_exist and not path.exists():
                result.add_error(f"File does not exist: {path}", ValidationSeverity.ERROR)
                return result
            
            # Permission validation for existing files
            if path.exists():
                if not os.access(path, os.R_OK):
                    result.add_error(f"File not readable: {path}", ValidationSeverity.ERROR)
                    return result
                
                result.metadata = {
                    'exists': True,
                    'size': path.stat().st_size,
                    'readable': os.access(path, os.R_OK),
                    'writable': os.access(path, os.W_OK)
                }
            else:
                # Check if parent directory is writable for new files
                parent = path.parent
                if parent.exists() and not os.access(parent, os.W_OK):
                    result.add_error(f"Parent directory not writable: {parent}", ValidationSeverity.ERROR)
                    return result
                
                result.metadata = {
                    'exists': False,
                    'parent_exists': parent.exists(),
                    'parent_writable': os.access(parent, os.W_OK) if parent.exists() else False
                }
            
        except Exception as e:
            result.add_error(f"Path validation failed: {e}", ValidationSeverity.ERROR)
            return result
        
        return result
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', filename)
        
        # Remove leading/trailing periods and spaces
        sanitized = sanitized.strip('. ')
        
        # Handle Windows reserved names
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        name_part = Path(sanitized).stem.upper()
        if name_part in reserved_names:
            sanitized = f"_{sanitized}"
        
        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        return sanitized


class TextValidator:
    """Text content validation and sanitization."""
    
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
        r'eval\s*\(', r'exec\s*\(', r'system\s*\(', r'shell_exec\s*\(',
        r'DROP\s+TABLE', r'DELETE\s+FROM', r'INSERT\s+INTO', r'UPDATE\s+SET'
    ]
    
    @classmethod
    def validate_text(cls, text: str, max_length: int = None, allow_empty: bool = True) -> ValidationResult:
        """Validate text content for safety and format compliance."""
        result = ValidationResult()
        
        if text is None:
            if allow_empty:
                result.sanitized_value = ""
                return result
            else:
                result.add_error("Text cannot be None", ValidationSeverity.ERROR)
                return result
        
        if not isinstance(text, str):
            result.add_error("Text must be a string", ValidationSeverity.ERROR)
            return result
        
        # Length validation
        if max_length and len(text) > max_length:
            result.add_error(f"Text too long (max {max_length} characters)", ValidationSeverity.ERROR)
            return result
        
        if not text.strip() and not allow_empty:
            result.add_error("Text cannot be empty", ValidationSeverity.ERROR)
            return result
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result.add_error(f"Text contains suspicious pattern: {pattern}", ValidationSeverity.WARNING)
        
        # Sanitize text
        result.sanitized_value = cls.sanitize_text(text)
        
        result.metadata = {
            'length': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1,
            'has_html': bool(re.search(r'<[^>]+>', text)),
            'has_urls': bool(re.search(r'https?://[^\s]+', text))
        }
        
        return result
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Sanitize text content for safe usage."""
        if not text:
            return ""
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized


class NumericValidator:
    """Numeric value validation."""
    
    @classmethod
    def validate_integer(cls, value: Any, min_val: int = None, max_val: int = None) -> ValidationResult:
        """Validate integer value with range checking."""
        result = ValidationResult()
        
        try:
            if isinstance(value, str):
                # Try to convert string to int
                if not value.strip():
                    result.add_error("Integer value cannot be empty string", ValidationSeverity.ERROR)
                    return result
                
                int_val = int(value.strip())
            elif isinstance(value, (int, float)):
                int_val = int(value)
            else:
                result.add_error(f"Cannot convert {type(value)} to integer", ValidationSeverity.ERROR)
                return result
            
            result.sanitized_value = int_val
            
            # Range validation
            if min_val is not None and int_val < min_val:
                result.add_error(f"Value {int_val} below minimum {min_val}", ValidationSeverity.ERROR)
                return result
            
            if max_val is not None and int_val > max_val:
                result.add_error(f"Value {int_val} above maximum {max_val}", ValidationSeverity.ERROR)
                return result
            
            result.metadata = {
                'original_type': type(value).__name__,
                'converted_value': int_val
            }
            
        except ValueError as e:
            result.add_error(f"Invalid integer format: {e}", ValidationSeverity.ERROR)
            return result
        
        return result
    
    @classmethod
    def validate_float(cls, value: Any, min_val: float = None, max_val: float = None) -> ValidationResult:
        """Validate float value with range checking."""
        result = ValidationResult()
        
        try:
            if isinstance(value, str):
                if not value.strip():
                    result.add_error("Float value cannot be empty string", ValidationSeverity.ERROR)
                    return result
                
                float_val = float(value.strip())
            elif isinstance(value, (int, float)):
                float_val = float(value)
            else:
                result.add_error(f"Cannot convert {type(value)} to float", ValidationSeverity.ERROR)
                return result
            
            result.sanitized_value = float_val
            
            # Range validation
            if min_val is not None and float_val < min_val:
                result.add_error(f"Value {float_val} below minimum {min_val}", ValidationSeverity.ERROR)
                return result
            
            if max_val is not None and float_val > max_val:
                result.add_error(f"Value {float_val} above maximum {max_val}", ValidationSeverity.ERROR)
                return result
            
            result.metadata = {
                'original_type': type(value).__name__,
                'converted_value': float_val
            }
            
        except ValueError as e:
            result.add_error(f"Invalid float format: {e}", ValidationSeverity.ERROR)
            return result
        
        return result


class InputValidator:
    """Main input validation class that coordinates all validators."""
    
    def __init__(self):
        self.url_validator = URLValidator()
        self.file_validator = FileValidator()
        self.text_validator = TextValidator()
        self.numeric_validator = NumericValidator()
    
    def validate_youtube_url(self, url: str) -> ValidationResult:
        """Validate YouTube URL."""
        return self.url_validator.validate_url(url, allow_youtube_only=True)
    
    def validate_channel_id(self, channel_id: str) -> ValidationResult:
        """Validate YouTube channel ID format."""
        result = ValidationResult()
        
        if not channel_id or not isinstance(channel_id, str):
            result.add_error("Channel ID cannot be empty", ValidationSeverity.ERROR)
            return result
        
        channel_id = channel_id.strip()
        result.sanitized_value = channel_id
        
        # YouTube channel ID patterns
        patterns = [
            (r'^UC[a-zA-Z0-9_-]{22}$', 'channel_id'),  # Standard channel ID
            (r'^[a-zA-Z0-9_.-]{3,30}$', 'username'),    # Username format
            (r'^@[a-zA-Z0-9_.-]{1,30}$', 'handle')      # Handle format
        ]
        
        valid_format = False
        for pattern, id_type in patterns:
            if re.match(pattern, channel_id):
                valid_format = True
                result.metadata['type'] = id_type
                break
        
        if not valid_format:
            result.add_error("Invalid channel ID format", ValidationSeverity.ERROR)
            return result
        
        return result
    
    def validate_video_quality(self, quality: str) -> ValidationResult:
        """Validate video quality parameter."""
        result = ValidationResult()
        
        valid_qualities = ['144p', '240p', '360p', '480p', '720p', '1080p', '1440p', '2160p', 'best', 'worst']
        
        if not quality or quality not in valid_qualities:
            result.add_error(f"Invalid quality. Must be one of: {valid_qualities}", ValidationSeverity.ERROR)
            return result
        
        result.sanitized_value = quality
        result.metadata = {'valid_qualities': valid_qualities}
        
        return result
    
    def validate_file_format(self, file_format: str, format_type: str) -> ValidationResult:
        """Validate file format parameter."""
        result = ValidationResult()
        
        format_options = {
            'video': ['mp4', 'mkv', 'avi', 'mov', 'webm'],
            'audio': ['mp3', 'wav', 'opus', 'm4a', 'flac', 'ogg', 'best']
        }
        
        if format_type not in format_options:
            result.add_error(f"Invalid format type: {format_type}", ValidationSeverity.ERROR)
            return result
        
        valid_formats = format_options[format_type]
        
        if not file_format or file_format not in valid_formats:
            result.add_error(f"Invalid {format_type} format. Must be one of: {valid_formats}", ValidationSeverity.ERROR)
            return result
        
        result.sanitized_value = file_format
        result.metadata = {'valid_formats': valid_formats, 'type': format_type}
        
        return result
    
    def validate_batch_input(self, data: List[Any], max_items: int = None) -> ValidationResult:
        """Validate batch input data."""
        result = ValidationResult()
        
        if not isinstance(data, list):
            result.add_error("Batch data must be a list", ValidationSeverity.ERROR)
            return result
        
        if len(data) == 0:
            result.add_error("Batch data cannot be empty", ValidationSeverity.ERROR)
            return result
        
        if max_items and len(data) > max_items:
            result.add_error(f"Too many items in batch (max {max_items})", ValidationSeverity.ERROR)
            return result
        
        result.sanitized_value = data
        result.metadata = {
            'item_count': len(data),
            'max_items': max_items
        }
        
        return result


# Global validator instance
input_validator = InputValidator()


def validate_api_input(data: Dict[str, Any], schema: BaseModel) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate API input against Pydantic schema with enhanced error handling."""
    try:
        validated_data = schema(**data)
        return True, validated_data.model_dump(), []
    
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")
        
        return False, {}, errors
    
    except Exception as e:
        return False, {}, [f"Validation failed: {str(e)}"]


# Convenience functions for common validation tasks
def validate_youtube_url(url: str) -> ValidationResult:
    """Validate YouTube URL."""
    return input_validator.validate_youtube_url(url)


def validate_filename(filename: str, allowed_types: List[str] = None) -> ValidationResult:
    """Validate filename."""
    return input_validator.file_validator.validate_filename(filename, allowed_types)


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> ValidationResult:
    """Validate file path."""
    return input_validator.file_validator.validate_file_path(path, must_exist)


def validate_text_content(text: str, max_length: int = None) -> ValidationResult:
    """Validate text content."""
    return input_validator.text_validator.validate_text(text, max_length)


def validate_integer_input(value: Any, min_val: int = None, max_val: int = None) -> ValidationResult:
    """Validate integer input."""
    return input_validator.numeric_validator.validate_integer(value, min_val, max_val)