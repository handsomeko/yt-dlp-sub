"""
AI/ML Pipeline Security Fixes - Issues #165-171
Comprehensive protection for AI-powered systems
"""

import hashlib
import hmac
import json
import re
import logging
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import spacy
import tiktoken
from transformers import pipeline
import openai
import anthropic
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptInjectionType(Enum):
    """Types of prompt injection attacks"""
    DIRECT_INJECTION = "direct"
    INDIRECT_INJECTION = "indirect"
    JAILBREAK_ATTEMPT = "jailbreak"
    ROLE_PLAY_ATTACK = "roleplay"
    CONTEXT_ESCAPE = "context_escape"
    SYSTEM_OVERRIDE = "system_override"


@dataclass
class AISecurityEvent:
    """AI security event data"""
    event_type: str
    severity: str
    user_id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]


class PromptInjectionDetector:
    """Advanced prompt injection detection (Issue #165)"""
    
    def __init__(self):
        self.injection_patterns = {
            'direct': [
                r'ignore\s+(?:previous|above|all)\s+(?:instructions?|prompts?|rules?)',
                r'forget\s+(?:everything|all|previous|above)',
                r'new\s+(?:instructions?|task|prompt|role)',
                r'override\s+(?:system|default|previous)',
                r'act\s+as\s+(?:if|though|like)\s+you\s+(?:are|were)',
                r'pretend\s+(?:to\s+be|you\s+are)',
                r'roleplay\s+as',
                r'you\s+are\s+now\s+(?:a|an)',
                r'from\s+now\s+on\s+you\s+(?:are|will|must)',
                r'disregard\s+(?:all|any|previous)',
            ],
            'indirect': [
                r'hypothetically\s+speaking',
                r'in\s+theory',
                r'what\s+if\s+I\s+told\s+you',
                r'imagine\s+(?:if|that)',
                r'suppose\s+(?:that|for\s+a\s+moment)',
                r'let\'s\s+say\s+(?:that|for\s+example)',
                r'consider\s+this\s+scenario',
            ],
            'jailbreak': [
                r'DAN\s+mode',
                r'developer\s+mode',
                r'debug\s+mode',
                r'admin\s+access',
                r'root\s+privileges',
                r'bypass\s+(?:safety|restrictions?|filters?)',
                r'unrestricted\s+mode',
                r'no\s+(?:limits?|restrictions?|safety)',
                r'anything\s+goes\s+mode',
            ],
            'system_override': [
                r'</system>',
                r'<system>',
                r'\[SYSTEM\]',
                r'\[/SYSTEM\]',
                r'system\s*:\s*',
                r'assistant\s*:\s*',
                r'human\s*:\s*',
                r'user\s*:\s*',
                r'\\n\\nHuman:',
                r'\\n\\nAssistant:',
            ]
        }
        
        # Load NLP model for semantic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning("Spacy model not found, using basic detection")
            self.nlp = None
        
        # Suspicious phrases that might indicate manipulation
        self.suspicious_phrases = [
            "just kidding", "never mind", "scratch that", "wait actually",
            "on second thought", "change of plans", "update your instructions",
            "new directive", "emergency override", "priority update",
            "special case", "exception to the rule", "temporary override"
        ]
        
        # Rate limiting for injection attempts
        self.injection_attempts = defaultdict(list)
        self.max_attempts = 5
        self.attempt_window = timedelta(minutes=15)
    
    def detect_prompt_injection(self, prompt: str, user_id: str = "anonymous") -> Tuple[bool, PromptInjectionType, float]:
        """Detect prompt injection attempts with confidence score"""
        
        # Check rate limiting
        if self._is_rate_limited(user_id):
            return True, PromptInjectionType.DIRECT_INJECTION, 1.0
        
        # Normalize text for analysis
        normalized = self._normalize_text(prompt)
        
        # Pattern-based detection
        pattern_score, injection_type = self._check_patterns(normalized)
        
        # Semantic analysis if available
        semantic_score = self._semantic_analysis(prompt) if self.nlp else 0.0
        
        # Statistical analysis
        statistical_score = self._statistical_analysis(prompt)
        
        # Combine scores
        total_score = max(pattern_score, semantic_score * 0.7, statistical_score * 0.5)
        
        is_injection = total_score > 0.6  # Threshold for detection
        
        if is_injection:
            self._record_injection_attempt(user_id, injection_type, total_score)
        
        return is_injection, injection_type, total_score
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching"""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common evasion techniques
        normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Remove punctuation
        normalized = re.sub(r'(.)\1{3,}', r'\1\1', normalized)  # Reduce repeated chars
        
        return normalized.strip()
    
    def _check_patterns(self, text: str) -> Tuple[float, PromptInjectionType]:
        """Check for known injection patterns"""
        max_score = 0.0
        detected_type = PromptInjectionType.DIRECT_INJECTION
        
        for injection_type, patterns in self.injection_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
                if matches > 0:
                    # Score based on pattern strength and frequency
                    score = min(matches * 0.3 + 0.4, 1.0)
                    if score > max_score:
                        max_score = score
                        detected_type = PromptInjectionType(injection_type)
        
        # Check for suspicious phrases
        suspicious_count = sum(1 for phrase in self.suspicious_phrases if phrase in text)
        if suspicious_count > 0:
            suspicious_score = min(suspicious_count * 0.2 + 0.3, 0.8)
            if suspicious_score > max_score:
                max_score = suspicious_score
        
        return max_score, detected_type
    
    def _semantic_analysis(self, text: str) -> float:
        """Semantic analysis using NLP"""
        if not self.nlp:
            return 0.0
        
        try:
            doc = self.nlp(text)
            
            # Look for command-like structures
            command_indicators = 0
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    # Check if verb is imperative
                    if any(child.dep_ == "nsubj" for child in token.children):
                        continue  # Has subject, not imperative
                    command_indicators += 1
            
            # Look for role/identity changes
            identity_changes = 0
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"] and any(
                    word in ent.text.lower() for word in ["assistant", "ai", "bot", "system"]
                ):
                    identity_changes += 1
            
            # Calculate semantic score
            score = min((command_indicators * 0.2 + identity_changes * 0.3), 0.8)
            return score
            
        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            return 0.0
    
    def _statistical_analysis(self, text: str) -> float:
        """Statistical analysis of text patterns"""
        # Check for unusual patterns that might indicate injection
        
        # Ratio of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text) if text else 0
        
        # Ratio of uppercase letters
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Line breaks and formatting
        line_breaks = text.count('\n')
        line_ratio = line_breaks / len(text.split()) if text.split() else 0
        
        # Unusual patterns score
        score = 0.0
        if special_ratio > 0.3:  # Too many special characters
            score += 0.3
        if upper_ratio > 0.5:   # Too much uppercase
            score += 0.2
        if line_ratio > 0.1:    # Too many line breaks
            score += 0.2
        
        return min(score, 0.6)
    
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        now = datetime.now()
        attempts = self.injection_attempts[user_id]
        
        # Remove old attempts
        self.injection_attempts[user_id] = [
            attempt for attempt in attempts
            if now - attempt < self.attempt_window
        ]
        
        return len(self.injection_attempts[user_id]) >= self.max_attempts
    
    def _record_injection_attempt(self, user_id: str, injection_type: PromptInjectionType, score: float):
        """Record injection attempt for rate limiting"""
        self.injection_attempts[user_id].append(datetime.now())
        
        logger.warning(
            f"Prompt injection detected: user={user_id}, type={injection_type.value}, score={score:.2f}"
        )


class ModelPoisoningProtector:
    """Model poisoning protection (Issue #166)"""
    
    def __init__(self):
        self.baseline_responses = {}
        self.response_variance_threshold = 0.8
        self.suspicious_patterns = [
            r'I am (?:not|no longer) (?:an AI|Claude|GPT)',
            r'My name is (?:not )?(?:really )?(?:actually )?(?!Claude|Assistant)',
            r'I (?:can|will) help you (?:with|to) (?:hack|attack|exploit)',
            r'As a (?:human|person|real person)',
            r'I don\'t have (?:any )?(?:safety|ethical) (?:guidelines|restrictions)',
        ]
    
    def establish_baseline(self, test_prompts: List[str], model_function):
        """Establish baseline responses for comparison"""
        logger.info("Establishing model baseline responses...")
        
        for prompt in test_prompts:
            try:
                response = model_function(prompt)
                self.baseline_responses[prompt] = {
                    'response': response,
                    'timestamp': datetime.now(),
                    'hash': hashlib.sha256(response.encode()).hexdigest()
                }
            except Exception as e:
                logger.error(f"Error getting baseline for prompt: {e}")
    
    def check_response_integrity(self, prompt: str, response: str) -> Tuple[bool, float]:
        """Check if response shows signs of model poisoning"""
        
        # Check against baseline if available
        baseline_score = 0.0
        if prompt in self.baseline_responses:
            baseline_response = self.baseline_responses[prompt]['response']
            baseline_score = self._calculate_response_similarity(response, baseline_response)
        
        # Check for suspicious patterns
        pattern_score = self._check_suspicious_patterns(response)
        
        # Check response coherence
        coherence_score = self._check_coherence(response)
        
        # Combine scores
        total_score = (baseline_score * 0.5 + pattern_score * 0.3 + coherence_score * 0.2)
        
        is_poisoned = total_score > self.response_variance_threshold
        
        return is_poisoned, total_score
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between responses"""
        # Simple character-level similarity
        if not response1 or not response2:
            return 1.0  # Empty responses are suspicious
        
        # Use Levenshtein distance approximation
        longer = response1 if len(response1) > len(response2) else response2
        shorter = response2 if len(response1) > len(response2) else response1
        
        if len(longer) == 0:
            return 0.0
        
        # Simple similarity metric
        common_chars = sum(1 for a, b in zip(longer, shorter) if a == b)
        similarity = 1.0 - (common_chars / len(longer))
        
        return similarity
    
    def _check_suspicious_patterns(self, response: str) -> float:
        """Check for patterns indicating poisoning"""
        score = 0.0
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.3
        
        return min(score, 1.0)
    
    def _check_coherence(self, response: str) -> float:
        """Check response coherence"""
        if not response.strip():
            return 1.0  # Empty response is suspicious
        
        # Basic coherence checks
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.1  # Too short
        
        # Check for contradictions (basic)
        contradiction_indicators = ['however', 'but', 'although', 'despite', 'actually', 'in fact']
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in response.lower())
        
        # High contradiction might indicate poisoning
        if contradiction_count > len(sentences) * 0.3:
            return 0.6
        
        return 0.1


class AdversarialInputDetector:
    """Adversarial input detection (Issue #167)"""
    
    def __init__(self):
        self.embedding_model = None
        self.anomaly_threshold = 0.7
        self.input_history = deque(maxlen=1000)
        
        # Load embedding model if available
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("SentenceTransformers not available, using basic detection")
    
    def detect_adversarial_input(self, input_text: str) -> Tuple[bool, float]:
        """Detect adversarial inputs designed to fool the AI"""
        
        # Statistical analysis
        statistical_score = self._statistical_anomaly_detection(input_text)
        
        # Pattern-based detection
        pattern_score = self._pattern_based_detection(input_text)
        
        # Embedding-based detection if available
        embedding_score = 0.0
        if self.embedding_model:
            embedding_score = self._embedding_anomaly_detection(input_text)
        
        # Combine scores
        total_score = max(statistical_score, pattern_score, embedding_score * 0.8)
        
        is_adversarial = total_score > self.anomaly_threshold
        
        # Record input for future analysis
        self.input_history.append({
            'text': input_text[:100],  # Store only first 100 chars
            'score': total_score,
            'timestamp': datetime.now(),
            'is_adversarial': is_adversarial
        })
        
        return is_adversarial, total_score
    
    def _statistical_anomaly_detection(self, text: str) -> float:
        """Statistical anomaly detection"""
        if not text:
            return 1.0
        
        # Character frequency analysis
        char_freq = defaultdict(int)
        for char in text.lower():
            char_freq[char] += 1
        
        # Check for unusual character distributions
        total_chars = len(text)
        unusual_chars = sum(count for char, count in char_freq.items() 
                          if not char.isalnum() and char not in ' .,!?;:')
        
        unusual_ratio = unusual_chars / total_chars if total_chars > 0 else 0
        
        # Check for repeated patterns
        repeated_patterns = 0
        for i in range(1, min(10, len(text) // 2)):
            pattern = text[:i]
            if pattern * (len(text) // i) == text[:len(text) // i * i]:
                repeated_patterns += 1
        
        # Combine statistical indicators
        score = min(unusual_ratio * 2 + repeated_patterns * 0.3, 1.0)
        
        return score
    
    def _pattern_based_detection(self, text: str) -> float:
        """Pattern-based adversarial detection"""
        adversarial_patterns = [
            r'\\x[0-9a-f]{2}',  # Hex encoded characters
            r'%[0-9a-f]{2}',    # URL encoded characters
            r'\\u[0-9a-f]{4}',  # Unicode escapes
            r'\\n\\n\\n+',      # Excessive newlines
            r'[A-Z]{20,}',      # Long uppercase sequences
            r'(.)\1{10,}',      # Long repeated characters
            r'\d{50,}',         # Very long numbers
        ]
        
        score = 0.0
        for pattern in adversarial_patterns:
            matches = len(re.findall(pattern, text))
            score += matches * 0.2
        
        return min(score, 1.0)
    
    def _embedding_anomaly_detection(self, text: str) -> float:
        """Embedding-based anomaly detection"""
        if not self.embedding_model or len(self.input_history) < 10:
            return 0.0
        
        try:
            # Get embedding for current text
            current_embedding = self.embedding_model.encode([text])[0]
            
            # Compare with recent inputs
            recent_embeddings = []
            for entry in list(self.input_history)[-50:]:  # Last 50 entries
                if not entry['is_adversarial']:  # Only compare with normal inputs
                    embedding = self.embedding_model.encode([entry['text']])[0]
                    recent_embeddings.append(embedding)
            
            if not recent_embeddings:
                return 0.0
            
            # Calculate average similarity
            similarities = []
            for embedding in recent_embeddings:
                similarity = self._cosine_similarity(current_embedding, embedding)
                similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities)
            
            # Low similarity indicates anomaly
            anomaly_score = 1.0 - avg_similarity
            
            return max(0.0, anomaly_score)
            
        except Exception as e:
            logger.error(f"Embedding anomaly detection error: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between vectors"""
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms


class PIIDetector:
    """PII detection in AI outputs (Issue #168)"""
    
    def __init__(self):
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'address': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)',
        }
        
        # Load NER model if available
        try:
            self.ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        except Exception:
            logger.warning("NER model not available, using pattern-based detection only")
            self.ner = None
    
    def detect_pii(self, text: str) -> Tuple[List[Dict], float]:
        """Detect PII in text"""
        pii_findings = []
        
        # Pattern-based detection
        pattern_findings = self._pattern_based_pii_detection(text)
        pii_findings.extend(pattern_findings)
        
        # NER-based detection
        if self.ner:
            ner_findings = self._ner_based_pii_detection(text)
            pii_findings.extend(ner_findings)
        
        # Calculate risk score
        risk_score = min(len(pii_findings) * 0.3, 1.0)
        
        # Remove duplicates
        unique_findings = self._deduplicate_findings(pii_findings)
        
        return unique_findings, risk_score
    
    def _pattern_based_pii_detection(self, text: str) -> List[Dict]:
        """Pattern-based PII detection"""
        findings = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return findings
    
    def _ner_based_pii_detection(self, text: str) -> List[Dict]:
        """NER-based PII detection"""
        findings = []
        
        try:
            entities = self.ner(text)
            
            for entity in entities:
                if entity['entity'] in ['B-PER', 'I-PER']:  # Person names
                    findings.append({
                        'type': 'person_name',
                        'value': entity['word'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })
                elif entity['entity'] in ['B-LOC', 'I-LOC']:  # Locations
                    findings.append({
                        'type': 'location',
                        'value': entity['word'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })
                elif entity['entity'] in ['B-ORG', 'I-ORG']:  # Organizations
                    findings.append({
                        'type': 'organization',
                        'value': entity['word'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })
        
        except Exception as e:
            logger.error(f"NER-based PII detection error: {e}")
        
        return findings
    
    def _deduplicate_findings(self, findings: List[Dict]) -> List[Dict]:
        """Remove duplicate PII findings"""
        unique = []
        seen = set()
        
        for finding in findings:
            key = (finding['type'], finding['value'])
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        
        return unique
    
    def sanitize_pii(self, text: str, findings: List[Dict]) -> str:
        """Remove or mask PII from text"""
        sanitized = text
        
        # Sort findings by position (reverse order to maintain indices)
        findings_sorted = sorted(findings, key=lambda x: x['start'], reverse=True)
        
        for finding in findings_sorted:
            start, end = finding['start'], finding['end']
            pii_type = finding['type']
            
            # Replace with masked version
            if pii_type == 'ssn':
                replacement = 'XXX-XX-XXXX'
            elif pii_type in ['credit_card']:
                replacement = '*' * (end - start)
            elif pii_type == 'email':
                replacement = '[EMAIL REDACTED]'
            elif pii_type == 'phone':
                replacement = 'XXX-XXX-XXXX'
            elif pii_type == 'person_name':
                replacement = '[NAME REDACTED]'
            else:
                replacement = '[REDACTED]'
            
            sanitized = sanitized[:start] + replacement + sanitized[end:]
        
        return sanitized


class TokenUsageLimiter:
    """Per-user token limits (Issue #169)"""
    
    def __init__(self):
        self.token_usage = defaultdict(lambda: defaultdict(int))
        self.limits = {
            'daily': 10000,
            'hourly': 1000,
            'per_request': 4000
        }
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception:
            logger.warning("Tiktoken not available, using character count approximation")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def check_token_limit(self, user_id: str, request_tokens: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if user has exceeded token limits"""
        now = datetime.now()
        
        # Clean old usage data
        self._cleanup_old_usage(user_id, now)
        
        # Get current usage
        hourly_usage = sum(
            tokens for timestamp, tokens in self.token_usage[user_id]['hourly']
            if now - timestamp < timedelta(hours=1)
        )
        
        daily_usage = sum(
            tokens for timestamp, tokens in self.token_usage[user_id]['daily']
            if now - timestamp < timedelta(days=1)
        )
        
        # Check limits
        limits_info = {
            'per_request_limit': self.limits['per_request'],
            'hourly_limit': self.limits['hourly'],
            'daily_limit': self.limits['daily'],
            'current_hourly_usage': hourly_usage,
            'current_daily_usage': daily_usage,
            'request_tokens': request_tokens
        }
        
        # Check per-request limit
        if request_tokens > self.limits['per_request']:
            return False, {**limits_info, 'exceeded': 'per_request'}
        
        # Check hourly limit
        if hourly_usage + request_tokens > self.limits['hourly']:
            return False, {**limits_info, 'exceeded': 'hourly'}
        
        # Check daily limit
        if daily_usage + request_tokens > self.limits['daily']:
            return False, {**limits_info, 'exceeded': 'daily'}
        
        return True, limits_info
    
    def record_token_usage(self, user_id: str, tokens_used: int):
        """Record token usage for user"""
        timestamp = datetime.now()
        
        # Record in both hourly and daily buckets
        self.token_usage[user_id]['hourly'].append((timestamp, tokens_used))
        self.token_usage[user_id]['daily'].append((timestamp, tokens_used))
    
    def _cleanup_old_usage(self, user_id: str, now: datetime):
        """Clean up old usage records"""
        # Clean hourly data
        self.token_usage[user_id]['hourly'] = [
            (timestamp, tokens) for timestamp, tokens in self.token_usage[user_id]['hourly']
            if now - timestamp < timedelta(hours=1)
        ]
        
        # Clean daily data
        self.token_usage[user_id]['daily'] = [
            (timestamp, tokens) for timestamp, tokens in self.token_usage[user_id]['daily']
            if now - timestamp < timedelta(days=1)
        ]


class AIResponseValidator:
    """AI response validation (Issue #170)"""
    
    def __init__(self):
        self.validation_rules = {
            'max_length': 10000,
            'min_length': 10,
            'forbidden_patterns': [
                r'(?i)(?:password|secret|key|token)\s*[:\=]\s*\S+',
                r'(?i)api[_\s]*key\s*[:\=]\s*\S+',
                r'(?i)private[_\s]*key',
                r'ssh-rsa\s+\S+',
                r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
            ],
            'required_elements': {
                'content_generation': ['title', 'body'],
                'summary': ['main_points'],
                'transcript': ['text', 'timestamps']
            }
        }
        
        self.quality_thresholds = {
            'coherence': 0.7,
            'relevance': 0.6,
            'completeness': 0.8
        }
    
    def validate_response(self, response: str, response_type: str = 'general') -> Tuple[bool, List[str], Dict[str, float]]:
        """Validate AI response for safety and quality"""
        issues = []
        scores = {}
        
        # Basic validation
        basic_valid, basic_issues = self._basic_validation(response)
        issues.extend(basic_issues)
        
        # Content validation
        content_valid, content_issues = self._content_validation(response)
        issues.extend(content_issues)
        
        # Quality assessment
        quality_scores = self._assess_quality(response, response_type)
        scores.update(quality_scores)
        
        # Type-specific validation
        type_valid, type_issues = self._type_specific_validation(response, response_type)
        issues.extend(type_issues)
        
        is_valid = basic_valid and content_valid and type_valid
        
        return is_valid, issues, scores
    
    def _basic_validation(self, response: str) -> Tuple[bool, List[str]]:
        """Basic response validation"""
        issues = []
        
        # Length checks
        if len(response) > self.validation_rules['max_length']:
            issues.append(f"Response too long: {len(response)} > {self.validation_rules['max_length']}")
        
        if len(response) < self.validation_rules['min_length']:
            issues.append(f"Response too short: {len(response)} < {self.validation_rules['min_length']}")
        
        # Check for forbidden patterns
        for pattern in self.validation_rules['forbidden_patterns']:
            if re.search(pattern, response):
                issues.append(f"Forbidden pattern detected: {pattern}")
        
        # Encoding validation
        try:
            response.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Response contains invalid Unicode characters")
        
        return len(issues) == 0, issues
    
    def _content_validation(self, response: str) -> Tuple[bool, List[str]]:
        """Content safety validation"""
        issues = []
        
        # Check for potential harmful content
        harmful_patterns = [
            r'(?i)\b(?:hack|exploit|attack|malware|virus)\b.*\b(?:how\s+to|tutorial|guide)\b',
            r'(?i)\b(?:illegal|unlawful)\b.*\b(?:activities?|actions?)\b',
            r'(?i)\b(?:violence|harm|hurt|kill)\b.*\b(?:instructions?|how\s+to)\b',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response):
                issues.append(f"Potentially harmful content pattern: {pattern}")
        
        # Check for personal information leakage
        pii_detector = PIIDetector()
        pii_findings, pii_risk = pii_detector.detect_pii(response)
        
        if pii_risk > 0.5:
            issues.append(f"High PII risk: {pii_risk:.2f}")
        
        return len(issues) == 0, issues
    
    def _assess_quality(self, response: str, response_type: str) -> Dict[str, float]:
        """Assess response quality"""
        scores = {}
        
        # Coherence (basic check)
        sentences = response.split('.')
        coherence_score = min(len([s for s in sentences if s.strip()]) / max(len(sentences), 1), 1.0)
        scores['coherence'] = coherence_score
        
        # Completeness (has conclusion)
        has_conclusion = bool(re.search(r'(?i)\b(?:in conclusion|finally|to summarize|overall)\b', response))
        completeness_score = 0.8 if has_conclusion else 0.5
        scores['completeness'] = completeness_score
        
        # Relevance (basic keyword matching)
        relevance_score = 0.7  # Placeholder - would need context to assess properly
        scores['relevance'] = relevance_score
        
        return scores
    
    def _type_specific_validation(self, response: str, response_type: str) -> Tuple[bool, List[str]]:
        """Type-specific response validation"""
        issues = []
        
        if response_type in self.validation_rules['required_elements']:
            required = self.validation_rules['required_elements'][response_type]
            
            for element in required:
                if element == 'title' and not re.search(r'^.{1,100}$', response.split('\n')[0]):
                    issues.append("Missing or invalid title")
                elif element == 'timestamps' and not re.search(r'\d{2}:\d{2}', response):
                    issues.append("Missing timestamps")
                elif element == 'main_points' and len(response.split('.')) < 3:
                    issues.append("Insufficient main points")
        
        return len(issues) == 0, issues


class HallucinationDetector:
    """Hallucination detection (Issue #171)"""
    
    def __init__(self):
        self.fact_patterns = {
            'dates': r'\b(?:19|20)\d{2}\b',
            'numbers': r'\b\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?\b',
            'names': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'places': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|University|Company))\b',
        }
        
        self.uncertainty_indicators = [
            'might be', 'could be', 'possibly', 'perhaps', 'maybe',
            'I think', 'I believe', 'it seems', 'appears to',
            'allegedly', 'reportedly', 'supposedly'
        ]
        
        self.confidence_indicators = [
            'definitely', 'certainly', 'absolutely', 'without doubt',
            'confirmed', 'verified', 'established fact'
        ]
    
    def detect_hallucination(self, response: str, context: str = "") -> Tuple[bool, float, List[Dict]]:
        """Detect potential hallucinations in AI response"""
        
        # Extract factual claims
        factual_claims = self._extract_factual_claims(response)
        
        # Analyze confidence vs uncertainty
        confidence_score = self._analyze_confidence_indicators(response)
        
        # Check for internal consistency
        consistency_score = self._check_internal_consistency(response)
        
        # Context verification if available
        context_score = self._verify_against_context(response, context) if context else 0.5
        
        # Combine scores
        hallucination_risk = (
            (1.0 - confidence_score) * 0.3 +
            (1.0 - consistency_score) * 0.4 +
            (1.0 - context_score) * 0.3
        )
        
        is_hallucination = hallucination_risk > 0.6
        
        return is_hallucination, hallucination_risk, factual_claims
    
    def _extract_factual_claims(self, text: str) -> List[Dict]:
        """Extract factual claims from text"""
        claims = []
        
        for claim_type, pattern in self.fact_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                claims.append({
                    'type': claim_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return claims
    
    def _analyze_confidence_indicators(self, text: str) -> float:
        """Analyze confidence vs uncertainty indicators"""
        uncertainty_count = sum(1 for indicator in self.uncertainty_indicators 
                              if indicator.lower() in text.lower())
        
        confidence_count = sum(1 for indicator in self.confidence_indicators 
                             if indicator.lower() in text.lower())
        
        # High confidence with many facts might indicate hallucination
        total_indicators = uncertainty_count + confidence_count
        
        if total_indicators == 0:
            return 0.5  # Neutral
        
        confidence_ratio = confidence_count / total_indicators
        
        # Moderate confidence is good, extreme confidence is suspicious
        if confidence_ratio > 0.8:
            return 0.3  # Too confident might be hallucination
        elif confidence_ratio < 0.2:
            return 0.8  # Good uncertainty
        else:
            return 0.6  # Balanced
    
    def _check_internal_consistency(self, text: str) -> float:
        """Check for internal contradictions"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Basic contradiction detection
        contradiction_pairs = [
            ('is', 'is not'), ('was', 'was not'), ('will', 'will not'),
            ('can', 'cannot'), ('should', 'should not'), ('true', 'false'),
            ('yes', 'no'), ('increase', 'decrease'), ('before', 'after')
        ]
        
        contradictions = 0
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                for pos, neg in contradiction_pairs:
                    if (pos in sentence1.lower() and neg in sentence2.lower()) or \
                       (neg in sentence1.lower() and pos in sentence2.lower()):
                        contradictions += 1
        
        consistency_score = 1.0 - min(contradictions * 0.2, 1.0)
        return consistency_score
    
    def _verify_against_context(self, response: str, context: str) -> float:
        """Verify response against provided context"""
        if not context:
            return 0.5
        
        # Simple keyword overlap
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words & context_words) / len(response_words)
        
        return min(overlap * 2, 1.0)  # Scale up overlap score


class AISecurityManager:
    """Comprehensive AI/ML security manager"""
    
    def __init__(self):
        self.prompt_detector = PromptInjectionDetector()
        self.poisoning_protector = ModelPoisoningProtector()
        self.adversarial_detector = AdversarialInputDetector()
        self.pii_detector = PIIDetector()
        self.token_limiter = TokenUsageLimiter()
        self.response_validator = AIResponseValidator()
        self.hallucination_detector = HallucinationDetector()
        
        # Security events log
        self.security_events: List[AISecurityEvent] = []
        self.event_lock = threading.Lock()
    
    async def secure_ai_request(self, user_id: str, prompt: str, context: str = "") -> Dict[str, Any]:
        """Comprehensive AI request security checking"""
        
        # 1. Check prompt injection
        is_injection, injection_type, injection_score = self.prompt_detector.detect_prompt_injection(prompt, user_id)
        
        if is_injection:
            self._log_security_event("prompt_injection", "HIGH", user_id, prompt, {
                'injection_type': injection_type.value,
                'score': injection_score
            })
            return {
                'allowed': False,
                'reason': 'Prompt injection detected',
                'details': {'type': injection_type.value, 'score': injection_score}
            }
        
        # 2. Check for adversarial input
        is_adversarial, adversarial_score = self.adversarial_detector.detect_adversarial_input(prompt)
        
        if is_adversarial:
            self._log_security_event("adversarial_input", "MEDIUM", user_id, prompt, {
                'score': adversarial_score
            })
            return {
                'allowed': False,
                'reason': 'Adversarial input detected',
                'details': {'score': adversarial_score}
            }
        
        # 3. Check token limits
        prompt_tokens = self.token_limiter.count_tokens(prompt + context)
        can_proceed, token_info = self.token_limiter.check_token_limit(user_id, prompt_tokens)
        
        if not can_proceed:
            self._log_security_event("token_limit_exceeded", "LOW", user_id, "", token_info)
            return {
                'allowed': False,
                'reason': 'Token limit exceeded',
                'details': token_info
            }
        
        # Record token usage
        self.token_limiter.record_token_usage(user_id, prompt_tokens)
        
        return {
            'allowed': True,
            'prompt_tokens': prompt_tokens,
            'security_checks_passed': ['prompt_injection', 'adversarial_input', 'token_limits']
        }
    
    async def validate_ai_response(self, user_id: str, response: str, response_type: str = 'general', 
                                 context: str = "") -> Dict[str, Any]:
        """Validate AI response for security and quality"""
        
        # 1. Basic validation
        is_valid, issues, quality_scores = self.response_validator.validate_response(response, response_type)
        
        if not is_valid:
            self._log_security_event("invalid_response", "MEDIUM", user_id, response[:100], {
                'issues': issues,
                'response_type': response_type
            })
        
        # 2. Check for PII
        pii_findings, pii_risk = self.pii_detector.detect_pii(response)
        
        if pii_risk > 0.5:
            self._log_security_event("pii_detected", "HIGH", user_id, "", {
                'pii_count': len(pii_findings),
                'risk_score': pii_risk
            })
        
        # 3. Check for hallucinations
        is_hallucination, hallucination_risk, factual_claims = self.hallucination_detector.detect_hallucination(
            response, context
        )
        
        if is_hallucination:
            self._log_security_event("hallucination_detected", "MEDIUM", user_id, "", {
                'risk_score': hallucination_risk,
                'claims_count': len(factual_claims)
            })
        
        # 4. Sanitize PII if needed
        sanitized_response = response
        if pii_findings:
            sanitized_response = self.pii_detector.sanitize_pii(response, pii_findings)
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'quality_scores': quality_scores,
            'pii_risk': pii_risk,
            'hallucination_risk': hallucination_risk,
            'sanitized_response': sanitized_response,
            'original_length': len(response),
            'sanitized_length': len(sanitized_response)
        }
    
    def _log_security_event(self, event_type: str, severity: str, user_id: str, 
                          content: str, metadata: Dict[str, Any]):
        """Log AI security event"""
        with self.event_lock:
            event = AISecurityEvent(
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                content=content[:500],  # Limit content length
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            self.security_events.append(event)
            
            # Log to file
            logger.warning(f"AI Security Event: {event_type} - User: {user_id} - Severity: {severity}")
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate AI security report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [event for event in self.security_events if event.timestamp > cutoff_time]
        
        # Categorize events
        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)
        events_by_user = defaultdict(int)
        
        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_severity[event.severity] += 1
            events_by_user[event.user_id] += 1
        
        return {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'events_by_type': dict(events_by_type),
            'events_by_severity': dict(events_by_severity),
            'top_users': dict(sorted(events_by_user.items(), key=lambda x: x[1], reverse=True)[:10]),
            'high_severity_count': events_by_severity.get('HIGH', 0),
            'recommendations': self._generate_recommendations(events_by_type, events_by_severity)
        }
    
    def _generate_recommendations(self, events_by_type: dict, events_by_severity: dict) -> List[str]:
        """Generate security recommendations based on events"""
        recommendations = []
        
        if events_by_type.get('prompt_injection', 0) > 10:
            recommendations.append("High prompt injection attempts - consider additional input filtering")
        
        if events_by_severity.get('HIGH', 0) > 5:
            recommendations.append("Multiple high-severity events - review security policies")
        
        if events_by_type.get('pii_detected', 0) > 0:
            recommendations.append("PII detected in outputs - review data handling procedures")
        
        if events_by_type.get('hallucination_detected', 0) > 5:
            recommendations.append("Frequent hallucinations detected - consider model fine-tuning")
        
        return recommendations


# Integration function
def setup_ai_ml_security() -> AISecurityManager:
    """Setup AI/ML security system"""
    security_manager = AISecurityManager()
    
    logger.info("AI/ML security system initialized")
    return security_manager


if __name__ == "__main__":
    # Test the AI security system
    security_manager = setup_ai_ml_security()
    print("AI/ML security fixes initialized")