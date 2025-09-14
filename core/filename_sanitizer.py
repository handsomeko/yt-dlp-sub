"""
Filename Sanitization Utility - Ensures safe, readable filenames across all platforms
Handles special characters, length limits, reserved names, and collision prevention
"""

import re
import unicodedata
import threading
from pathlib import Path
from typing import Optional, Set, Dict
import hashlib
import json
import os


class FilenameSanitizer:
    """
    Comprehensive filename sanitization for cross-platform compatibility.
    Balances human readability with filesystem constraints.
    """
    
    # Reserved names on Windows
    WINDOWS_RESERVED = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    # Invalid characters for different platforms
    WINDOWS_INVALID = '<>:"|?*'
    MAC_INVALID = ':'
    LINUX_INVALID = '\0'
    
    # Combined invalid characters (strictest set)
    INVALID_CHARS = '<>:"|?*\0\r\n\t'
    
    # Maximum filename lengths (without extension)
    MAX_FILENAME_LENGTH = 80  # Conservative limit for safety
    WINDOWS_MAX_PATH = 260
    
    def __init__(self, strict_mode: bool = True, cache_file: Optional[str] = None):
        """
        Initialize sanitizer.
        
        Args:
            strict_mode: If True, uses strictest rules for maximum compatibility
            cache_file: Optional persistent cache file for collision tracking
        """
        self.strict_mode = strict_mode
        self.cache_file = cache_file
        self._collision_cache: Set[str] = set()
        self._cache_lock = threading.RLock()  # Thread-safe locking
        self._load_collision_cache()
    
    def sanitize(
        self, 
        filename: str, 
        video_id: Optional[str] = None,
        max_length: Optional[int] = None,
        preserve_extension: bool = True
    ) -> str:
        """
        Sanitize a filename for safe filesystem storage.
        
        Args:
            filename: Original filename to sanitize
            video_id: Optional video ID for collision prevention
            max_length: Override maximum length (default: 80 chars)
            preserve_extension: Keep file extension if present
            
        Returns:
            Sanitized filename safe for all platforms
            
        Raises:
            ValueError: If input contains malicious patterns
        """
        # Input validation for security
        self._validate_input(filename, video_id)
        
        if not filename:
            return f"untitled_{video_id}" if video_id else "untitled"
        
        # Split extension if needed
        base_name = filename
        extension = ""
        if preserve_extension and '.' in filename:
            parts = filename.rsplit('.', 1)
            if len(parts[1]) <= 10:  # Reasonable extension length
                base_name = parts[0]
                extension = f".{parts[1]}"
        
        # Step 1: Handle Unicode and special characters
        base_name = self._normalize_unicode(base_name)
        
        # Step 2: Remove/replace invalid characters
        base_name = self._remove_invalid_chars(base_name)
        
        # Step 3: Handle spaces and separators
        base_name = self._clean_separators(base_name)
        
        # Step 4: Handle reserved names
        base_name = self._handle_reserved_names(base_name)
        
        # Step 5: Apply length limits
        max_len = max_length or self.MAX_FILENAME_LENGTH
        base_name = self._enforce_length_limit(base_name, max_len, video_id)
        
        # Step 6: Clean up edges
        base_name = self._clean_edges(base_name)
        
        # Step 7: Handle empty result
        if not base_name:
            base_name = f"video_{video_id}" if video_id else "untitled"
        
        # Step 8: Handle collisions
        final_name = self._handle_collision(base_name, video_id)
        
        # Combine with extension
        result = f"{final_name}{extension}"
        
        # Final validation
        if not self._is_valid_filename(result):
            # Fallback to video_id based name
            result = f"video_{video_id}{extension}" if video_id else f"untitled{extension}"
        
        return result
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to ASCII where possible."""
        # Normalize to NFKD form and encode to ASCII, ignoring errors
        normalized = unicodedata.normalize('NFKD', text)
        ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
        
        # If we lost too much, keep some Unicode but remove combining chars
        if len(ascii_text) < len(text) * 0.5:
            # Remove only combining characters
            cleaned = ''.join(
                char for char in text 
                if unicodedata.category(char) != 'Mn'
            )
            return cleaned
        
        return ascii_text if ascii_text else text
    
    def _remove_invalid_chars(self, text: str) -> str:
        """Remove or replace invalid filesystem characters."""
        # Replace invalid chars with underscore
        for char in self.INVALID_CHARS:
            text = text.replace(char, '_')
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        # Replace problematic punctuation
        replacements = {
            '/': '-',
            '\\': '-',
            '#': '_',
            '%': '_',
            '&': 'and',
            '{': '(',
            '}': ')',
            '$': '',
            '!': '',
            '@': 'at',
            '`': '',
            '=': '-',
            '+': 'plus',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _clean_separators(self, text: str) -> str:
        """Clean up spaces and separators."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace spaces with underscores for better compatibility
        text = text.replace(' ', '_')
        
        # Clean up multiple underscores
        text = re.sub(r'_{2,}', '_', text)
        
        # Clean up multiple dashes
        text = re.sub(r'-{2,}', '-', text)
        
        # Clean up dash-underscore combinations
        text = re.sub(r'[-_]{2,}', '_', text)
        
        return text
    
    def _handle_reserved_names(self, text: str) -> str:
        """Handle Windows reserved names."""
        # Check if name (without extension) is reserved
        base = text.upper().split('.')[0]
        if base in self.WINDOWS_RESERVED:
            return f"{text}_file"
        
        # Check for reserved names with any extension
        for reserved in self.WINDOWS_RESERVED:
            if text.upper().startswith(f"{reserved}."):
                return f"{text}_file"
        
        return text
    
    def _enforce_length_limit(
        self, 
        text: str, 
        max_length: int,
        video_id: Optional[str] = None
    ) -> str:
        """Enforce maximum filename length."""
        if len(text) <= max_length:
            return text
        
        # If we have a video_id, we can be more aggressive with truncation
        if video_id:
            # Reserve space for video_id suffix if needed (12 chars: _XXXXXXXXXXX)
            available = max_length - 12
            if available > 20:  # Keep at least 20 chars of title
                return text[:available]
        
        # Smart truncation: try to break at word boundary
        if '_' in text:
            parts = text.split('_')
            result = ""
            for part in parts:
                if len(result) + len(part) + 1 <= max_length:
                    result = f"{result}_{part}" if result else part
                else:
                    break
            return result if result else text[:max_length]
        
        # Simple truncation
        return text[:max_length]
    
    def _clean_edges(self, text: str) -> str:
        """Clean up leading/trailing special characters."""
        # Remove leading/trailing spaces, dots, dashes, underscores
        text = text.strip(' .-_')
        
        # Ensure doesn't start with a dot (hidden file on Unix)
        while text.startswith('.'):
            text = text[1:]
        
        # Ensure doesn't end with a dot (Windows issue)
        while text.endswith('.'):
            text = text[:-1]
        
        return text
    
    def _handle_collision(self, filename: str, video_id: Optional[str] = None) -> str:
        """Handle filename collisions with thread safety."""
        with self._cache_lock:
            if filename not in self._collision_cache:
                self._collision_cache.add(filename)
                self._save_collision_cache()  # Persist cache changes
                return filename
        
        # If we have video_id, append it
        if video_id:
            # Try with last 6 chars of video_id
            suffix = video_id[-6:] if len(video_id) >= 6 else video_id
            new_name = f"{filename}_{suffix}"
            if new_name not in self._collision_cache:
                self._collision_cache.add(new_name)
                return new_name
            
            # Try with full video_id
            new_name = f"{filename}_{video_id}"
            if new_name not in self._collision_cache:
                self._collision_cache.add(new_name)
                return new_name
        
            # Fallback to counter
            counter = 1
            while True:
                new_name = f"{filename}_{counter}"
                if new_name not in self._collision_cache:
                    self._collision_cache.add(new_name)
                    self._save_collision_cache()
                    return new_name
                counter += 1
                if counter > 999:  # Safety limit
                    # Use hash as last resort
                    hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
                    final_name = f"{filename}_{hash_suffix}"
                    self._collision_cache.add(final_name)
                    self._save_collision_cache()
                    return final_name
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Final validation of filename."""
        if not filename or filename == '.':
            return False
        
        # Check length
        if len(filename) > 255:  # Most filesystems limit
            return False
        
        # Check for any remaining invalid characters
        if any(char in filename for char in '\0'):
            return False
        
        # Check reserved names
        base = filename.upper().split('.')[0]
        if base in self.WINDOWS_RESERVED:
            return False
        
        return True
    
    def sanitize_path_component(self, component: str) -> str:
        """
        Sanitize a single path component (directory name).
        More lenient than filename sanitization but secure.
        """
        if not component:
            return "unknown"
        
        # Validate input for security  
        if len(component) > 500:  # Reasonable limit for directory names
            raise ValueError("Directory component too long")
        
        # Check for path traversal patterns
        dangerous_patterns = ['../', '..\\', '../', '~/', '%2e%2e', '%252e', '..']
        component_lower = component.lower()
        for pattern in dangerous_patterns:
            if pattern in component_lower:
                raise ValueError(f"Dangerous path pattern in directory name: {pattern}")
        
        # Remove null bytes and path separators
        component = component.replace('\0', '').replace('/', '-').replace('\\', '-')
        
        # Remove other problematic characters for directories
        component = component.replace(':', '-').replace('|', '-').replace('*', '-')
        component = component.replace('?', '').replace('<', '').replace('>', '')
        component = component.replace('"', '').replace('\r', '').replace('\n', '')
        
        # Handle Windows reserved names
        base = component.upper().split('.')[0]
        if base in self.WINDOWS_RESERVED:
            component = f"{component}_dir"
        
        # Clean up leading/trailing dots and spaces
        component = component.strip('. ')
        
        # Ensure doesn't start with dot (hidden directory)
        while component.startswith('.'):
            component = component[1:]
        
        # Length limit for directory names (more lenient than filenames)
        if len(component) > 100:
            component = component[:100]
        
        # Ensure not empty and not just special characters
        if not component or not any(c.isalnum() for c in component):
            return "unknown"
        
        return component
    
    def sanitize_directory_id(self, dir_id: str, id_type: str = "id") -> str:
        """
        Sanitize directory IDs (channel_id, video_id) for use in paths.
        More restrictive than general path components for security.
        
        Args:
            dir_id: The ID to sanitize
            id_type: Type of ID for error messages ("channel_id", "video_id", etc.)
            
        Returns:
            Sanitized ID safe for directory names
            
        Raises:
            ValueError: If ID is invalid or potentially malicious
        """
        if not dir_id:
            raise ValueError(f"{id_type} cannot be empty")
        
        # Strict length limits for IDs
        if len(dir_id) > 50:
            raise ValueError(f"{id_type} too long (max 50 characters)")
        
        # Check for path traversal patterns (very strict)
        dangerous_patterns = ['..', '~', '%', '\\', '/', ':', '|', '*', '?', '<', '>', '"']
        for pattern in dangerous_patterns:
            if pattern in dir_id:
                raise ValueError(f"Invalid character '{pattern}' in {id_type}")
        
        # Check for null bytes and control characters
        if '\x00' in dir_id or any(ord(c) < 32 for c in dir_id):
            raise ValueError(f"{id_type} contains null bytes or control characters")
        
        # Only allow alphanumeric, underscore, and hyphen for directory IDs
        if not re.match(r'^[a-zA-Z0-9_-]+$', dir_id):
            raise ValueError(f"{id_type} contains invalid characters (only a-z, A-Z, 0-9, _, - allowed)")
        
        # Ensure doesn't start/end with special characters
        if dir_id.startswith(('.', '_', '-')) or dir_id.endswith(('.', '_', '-')):
            raise ValueError(f"{id_type} cannot start or end with '.', '_', or '-'")
        
        # Check against Windows reserved names
        if dir_id.upper() in self.WINDOWS_RESERVED:
            raise ValueError(f"{id_type} '{dir_id}' is a Windows reserved name")
        
        return dir_id
    
    def _load_collision_cache(self):
        """Load collision cache from persistent storage."""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self._collision_cache = set(cache_data.get('collisions', []))
            except Exception:
                # If cache file is corrupted, start fresh
                self._collision_cache = set()

    def _save_collision_cache(self):
        """Save collision cache to persistent storage."""
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'collisions': list(self._collision_cache)}, f)
            except Exception:
                # Non-critical if cache save fails
                pass

    def _validate_input(self, filename: str, video_id: Optional[str] = None) -> None:
        """
        Validate input parameters for security.
        
        Args:
            filename: Filename to validate
            video_id: Video ID to validate
            
        Raises:
            ValueError: If input contains malicious patterns
        """
        # Check filename size limits (prevent DoS)
        if len(filename) > 10000:  # 10KB limit
            raise ValueError("Filename too long - potential DoS attack")
        
        # Check for path traversal patterns
        dangerous_patterns = ['../', '..\\', '../', '~/', '%2e%2e', '%252e']
        filename_lower = filename.lower()
        for pattern in dangerous_patterns:
            if pattern in filename_lower:
                raise ValueError(f"Dangerous path pattern detected: {pattern}")
        
        # Validate video_id format if provided
        if video_id:
            if len(video_id) > 50:  # Reasonable limit for video IDs
                raise ValueError("Video ID too long")
            if not re.match(r'^[a-zA-Z0-9_-]+$', video_id):
                raise ValueError("Video ID contains invalid characters")
        
        # Check for null bytes and control characters
        if '\x00' in filename or any(ord(c) < 32 for c in filename if c not in '\t\r\n'):
            raise ValueError("Filename contains null bytes or control characters")

    def clear_collision_cache(self):
        """Clear the collision cache. Call when switching contexts."""
        with self._cache_lock:
            self._collision_cache.clear()
            self._save_collision_cache()


# Convenience functions
_sanitizer = FilenameSanitizer()

def sanitize_filename(
    filename: str, 
    video_id: Optional[str] = None,
    max_length: Optional[int] = None
) -> str:
    """
    Convenience function for sanitizing filenames.
    
    Args:
        filename: Original filename
        video_id: Optional video ID for collision prevention
        max_length: Maximum length override
        
    Returns:
        Sanitized filename
    """
    return _sanitizer.sanitize(filename, video_id, max_length)

def sanitize_path(path: str) -> str:
    """
    Sanitize a complete path.
    
    Args:
        path: Original path
        
    Returns:
        Sanitized path
    """
    parts = Path(path).parts
    sanitized_parts = [
        _sanitizer.sanitize_path_component(part) 
        for part in parts
    ]
    return str(Path(*sanitized_parts))

def get_safe_filename(title: str, video_id: str, extension: str = "") -> str:
    """
    Get a safe filename for a video.
    
    Args:
        title: Video title
        video_id: YouTube video ID
        extension: File extension (without dot)
        
    Returns:
        Safe filename like "Video_Title_abc123.mp4"
    """
    base = sanitize_filename(title, video_id, max_length=80)
    if extension:
        return f"{base}.{extension}"
    return base

def sanitize_channel_id(channel_id: str) -> str:
    """
    Sanitize and validate a YouTube channel ID for directory usage.
    
    Args:
        channel_id: YouTube channel ID
        
    Returns:
        Validated channel ID
        
    Raises:
        ValueError: If channel ID is invalid
    """
    return _sanitizer.sanitize_directory_id(channel_id, "channel_id")

def sanitize_video_id(video_id: str) -> str:
    """
    Sanitize and validate a YouTube video ID for directory usage.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Validated video ID
        
    Raises:
        ValueError: If video ID is invalid
    """
    return _sanitizer.sanitize_directory_id(video_id, "video_id")