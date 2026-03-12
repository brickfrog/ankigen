# Security enhancements for agent system

import time
import hashlib
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import asyncio

from ankigen.logging import logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    cooldown_period: int = 300  # seconds


@dataclass
class SecurityConfig:
    """Security configuration for agents"""

    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    enable_rate_limiting: bool = True
    max_input_length: int = 10000
    max_output_length: int = 50000
    blocked_patterns: List[str] = field(default_factory=list)
    allowed_file_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".json", ".yaml"]
    )

    def __post_init__(self):
        if not self.blocked_patterns:
            self.blocked_patterns = [
                r"(?i)(api[_\-]?key|secret|password|token|credential)",
                r"(?i)(sk-[a-zA-Z0-9]{48,})",  # OpenAI API key pattern
                r"(?i)(access[_\-]?token)",
                r"(?i)(private[_\-]?key)",
                r"(?i)(<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>)",  # Script tags
                r"(?i)(javascript:|data:|vbscript:)",  # URL schemes
            ]


class RateLimiter:
    """Rate limiter for API calls and agent executions"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        async with self._locks[identifier]:
            now = time.time()

            # Clean old requests
            self._requests[identifier] = [
                req_time
                for req_time in self._requests[identifier]
                if now - req_time < 3600  # Keep last hour
            ]

            recent_requests = self._requests[identifier]

            # Check burst limit (last minute)
            last_minute = [req for req in recent_requests if now - req < 60]
            if len(last_minute) >= self.config.burst_limit:
                logger.warning(f"Burst limit exceeded for {identifier}")
                return False

            # Check per-minute limit
            if len(last_minute) >= self.config.requests_per_minute:
                logger.warning(f"Per-minute rate limit exceeded for {identifier}")
                return False

            # Check per-hour limit
            if len(recent_requests) >= self.config.requests_per_hour:
                logger.warning(f"Per-hour rate limit exceeded for {identifier}")
                return False

            # Record this request
            self._requests[identifier].append(now)
            return True

    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get when rate limits will reset for identifier"""
        if identifier not in self._requests:
            return None

        now = time.time()
        recent_requests = [req for req in self._requests[identifier] if now - req < 60]

        if len(recent_requests) >= self.config.requests_per_minute:
            oldest_request = min(recent_requests)
            return datetime.fromtimestamp(oldest_request + 60)

        return None


class SecurityValidator:
    """Security validator for agent inputs and outputs"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._blocked_patterns = [
            re.compile(pattern) for pattern in config.blocked_patterns
        ]

    def validate_input(self, input_text: str, source: str = "unknown") -> bool:
        """Validate input for security issues"""
        if not self.config.enable_input_validation:
            return True

        try:
            # Check input length
            if len(input_text) > self.config.max_input_length:
                logger.warning(f"Input too long from {source}: {len(input_text)} chars")
                return False

            # Check for blocked patterns
            for pattern in self._blocked_patterns:
                if pattern.search(input_text):
                    logger.warning(f"Blocked pattern detected in input from {source}")
                    return False

            # Check for suspicious content
            if self._contains_suspicious_content(input_text):
                logger.warning(f"Suspicious content detected in input from {source}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating input from {source}: {e}")
            return False

    def validate_output(self, output_text: str, agent_name: str = "unknown") -> bool:
        """Validate output for security issues"""
        if not self.config.enable_output_filtering:
            return True

        try:
            # Check output length
            if len(output_text) > self.config.max_output_length:
                logger.warning(
                    f"Output too long from {agent_name}: {len(output_text)} chars"
                )
                return False

            # Check for leaked sensitive information
            for pattern in self._blocked_patterns:
                if pattern.search(output_text):
                    logger.warning(
                        f"Potential data leak detected in output from {agent_name}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating output from {agent_name}: {e}")
            return False

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize input by removing potentially dangerous content"""
        try:
            # Remove HTML/XML tags
            sanitized = re.sub(r"<[^>]+>", "", input_text)

            # Remove suspicious URLs
            sanitized = re.sub(
                r"(?i)(javascript:|data:|vbscript:)[^\s]*", "[URL_REMOVED]", sanitized
            )

            # Truncate if too long
            if len(sanitized) > self.config.max_input_length:
                sanitized = sanitized[: self.config.max_input_length] + "...[TRUNCATED]"

            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing input: {e}")
            return input_text[:1000]  # Return truncated original as fallback

    def sanitize_output(self, output_text: str) -> str:
        """Sanitize output by removing sensitive information"""
        try:
            sanitized = output_text

            # Replace potential API keys or secrets
            for pattern in self._blocked_patterns:
                sanitized = pattern.sub("[REDACTED]", sanitized)

            # Truncate if too long
            if len(sanitized) > self.config.max_output_length:
                sanitized = (
                    sanitized[: self.config.max_output_length] + "...[TRUNCATED]"
                )

            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing output: {e}")
            return output_text[:5000]  # Return truncated original as fallback

    def _contains_suspicious_content(self, text: str) -> bool:
        """Check for suspicious content patterns"""
        suspicious_patterns = [
            r"(?i)(\beval\s*\()",  # eval() calls
            r"(?i)(\bexec\s*\()",  # exec() calls
            r"(?i)(__import__)",  # Dynamic imports
            r"(?i)(subprocess|os\.system)",  # System commands
            r"(?i)(file://|ftp://)",  # File/FTP URLs
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",  # IP addresses
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                return True

        return False


class SecureAgentWrapper:
    """Secure wrapper for agent execution with rate limiting and validation"""

    def __init__(
        self, base_agent, rate_limiter: RateLimiter, validator: SecurityValidator
    ):
        self.base_agent = base_agent
        self.rate_limiter = rate_limiter
        self.validator = validator
        self._identifier = self._generate_identifier()

    def _generate_identifier(self) -> str:
        """Generate unique identifier for rate limiting"""
        agent_name = getattr(self.base_agent, "config", {}).get("name", "unknown")
        # Include agent name and some randomness for fairness
        return hashlib.md5(f"{agent_name}_{id(self.base_agent)}".encode()).hexdigest()[
            :16
        ]

    async def secure_execute(
        self, user_input: str, context: Dict[str, Any] = None
    ) -> Any:
        """Execute agent with security checks and rate limiting"""

        # Rate limiting check
        if not await self.rate_limiter.check_rate_limit(self._identifier):
            reset_time = self.rate_limiter.get_reset_time(self._identifier)
            raise SecurityError(f"Rate limit exceeded. Reset at: {reset_time}")

        # Input validation
        if not self.validator.validate_input(user_input, self._identifier):
            raise SecurityError("Input validation failed")

        # Sanitize input
        sanitized_input = self.validator.sanitize_input(user_input)

        try:
            # Execute the base agent
            result = await self.base_agent.execute(sanitized_input, context)

            # Validate output
            if isinstance(result, str):
                if not self.validator.validate_output(result, self._identifier):
                    raise SecurityError("Output validation failed")

                # Sanitize output
                result = self.validator.sanitize_output(result)

            return result

        except Exception as e:
            logger.error(f"Secure execution failed for {self._identifier}: {e}")
            raise


class SecurityError(Exception):
    """Custom exception for security-related errors"""

    pass


# Global security components
_global_rate_limiter: Optional[RateLimiter] = None
_global_validator: Optional[SecurityValidator] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(config or RateLimitConfig())
    return _global_rate_limiter


def get_security_validator(
    config: Optional[SecurityConfig] = None,
) -> SecurityValidator:
    """Get global security validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = SecurityValidator(config or SecurityConfig())
    return _global_validator


def create_secure_agent(
    base_agent,
    rate_config: Optional[RateLimitConfig] = None,
    security_config: Optional[SecurityConfig] = None,
) -> SecureAgentWrapper:
    """Create a secure wrapper for an agent"""
    rate_limiter = get_rate_limiter(rate_config)
    validator = get_security_validator(security_config)
    return SecureAgentWrapper(base_agent, rate_limiter, validator)


# Configuration file permissions utility
def set_secure_file_permissions(file_path: str):
    """Set secure permissions for configuration files"""
    try:
        import os
        import stat

        # Set read/write for owner only (0o600)
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        logger.info(f"Set secure permissions for {file_path}")

    except Exception as e:
        logger.warning(f"Could not set secure permissions for {file_path}: {e}")


# Input validation utilities
def strip_html_tags(text: str) -> str:
    """Strip HTML tags from text (improved version)"""
    import html

    # Decode HTML entities first
    text = html.unescape(text)

    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove remaining HTML entities
    text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_api_key_format(api_key: str) -> bool:
    """Validate OpenAI API key format without logging it"""
    if not api_key:
        return False

    # Check basic format (starts with sk- and has correct length)
    if not api_key.startswith("sk-"):
        return False

    if len(api_key) < 20:  # Minimum reasonable length
        return False

    # Check for obvious fake keys
    fake_patterns = ["test", "fake", "demo", "example", "placeholder"]
    lower_key = api_key.lower()
    if any(pattern in lower_key for pattern in fake_patterns):
        return False

    return True


# Logging security
def sanitize_for_logging(text: str, max_length: int = 100) -> str:
    """Sanitize text for safe logging"""
    if not text:
        return "[EMPTY]"

    # Remove potential secrets
    validator = get_security_validator()
    sanitized = validator.sanitize_output(text)

    # Truncate for logging
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "...[TRUNCATED]"

    return sanitized
