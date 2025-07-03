# Tests for ankigen_core/agents/security.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ankigen_core.agents.security import (
    RateLimitConfig,
    SecurityConfig,
    RateLimiter,
    SecurityValidator,
    SecureAgentWrapper,
    SecurityError,
    get_rate_limiter,
    get_security_validator,
    create_secure_agent,
    strip_html_tags,
    validate_api_key_format,
    sanitize_for_logging,
)


# Test RateLimitConfig
def test_rate_limit_config_defaults():
    """Test RateLimitConfig default values"""
    config = RateLimitConfig()

    assert config.requests_per_minute == 60
    assert config.requests_per_hour == 1000
    assert config.burst_limit == 10
    assert config.cooldown_period == 300


def test_rate_limit_config_custom():
    """Test RateLimitConfig with custom values"""
    config = RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        burst_limit=5,
        cooldown_period=600,
    )

    assert config.requests_per_minute == 30
    assert config.requests_per_hour == 500
    assert config.burst_limit == 5
    assert config.cooldown_period == 600


# Test SecurityConfig
def test_security_config_defaults():
    """Test SecurityConfig default values"""
    config = SecurityConfig()

    assert config.enable_input_validation is True
    assert config.enable_output_filtering is True
    assert config.enable_rate_limiting is True
    assert config.max_input_length == 10000
    assert config.max_output_length == 50000
    assert len(config.blocked_patterns) > 0
    assert ".txt" in config.allowed_file_extensions


def test_security_config_blocked_patterns():
    """Test SecurityConfig blocked patterns"""
    config = SecurityConfig()

    # Should have common sensitive patterns
    patterns = config.blocked_patterns
    assert any("api" in pattern.lower() for pattern in patterns)
    assert any("secret" in pattern.lower() for pattern in patterns)
    assert any("password" in pattern.lower() for pattern in patterns)


# Test RateLimiter
@pytest.fixture
def rate_limiter():
    """Rate limiter with test configuration"""
    config = RateLimitConfig(requests_per_minute=5, requests_per_hour=50, burst_limit=3)
    return RateLimiter(config)


async def test_rate_limiter_allows_requests_under_limit(rate_limiter):
    """Test rate limiter allows requests under limits"""
    identifier = "test_user"

    # Should allow first few requests
    assert await rate_limiter.check_rate_limit(identifier) is True
    assert await rate_limiter.check_rate_limit(identifier) is True
    assert await rate_limiter.check_rate_limit(identifier) is True


async def test_rate_limiter_blocks_burst_limit(rate_limiter):
    """Test rate limiter blocks requests exceeding burst limit"""
    identifier = "test_user"

    # Use up burst limit
    for _ in range(3):
        assert await rate_limiter.check_rate_limit(identifier) is True

    # Next request should be blocked
    assert await rate_limiter.check_rate_limit(identifier) is False


async def test_rate_limiter_per_minute_limit(rate_limiter):
    """Test rate limiter per-minute limit"""
    identifier = "test_user"

    # Mock time to control rate limiting
    with patch("time.time") as mock_time:
        current_time = 1000.0
        mock_time.return_value = current_time

        # Use up per-minute limit
        for _ in range(5):
            assert await rate_limiter.check_rate_limit(identifier) is True

        # Next request should be blocked
        assert await rate_limiter.check_rate_limit(identifier) is False


async def test_rate_limiter_different_identifiers(rate_limiter):
    """Test rate limiter handles different identifiers separately"""
    user1 = "user1"
    user2 = "user2"

    # Use up limit for user1
    for _ in range(3):
        assert await rate_limiter.check_rate_limit(user1) is True

    assert await rate_limiter.check_rate_limit(user1) is False

    # user2 should still be allowed
    assert await rate_limiter.check_rate_limit(user2) is True


async def test_rate_limiter_reset_time(rate_limiter):
    """Test rate limiter reset time calculation"""
    identifier = "test_user"

    # Use up burst limit
    for _ in range(3):
        await rate_limiter.check_rate_limit(identifier)

    # Should have reset time
    reset_time = rate_limiter.get_reset_time(identifier)
    assert reset_time is not None


# Test SecurityValidator
@pytest.fixture
def security_validator():
    """Security validator with test configuration"""
    config = SecurityConfig(max_input_length=100, max_output_length=200)
    return SecurityValidator(config)


def test_security_validator_valid_input(security_validator):
    """Test security validator allows valid input"""
    valid_input = "This is a normal, safe input text."
    assert security_validator.validate_input(valid_input, "test") is True


def test_security_validator_input_too_long(security_validator):
    """Test security validator rejects input that's too long"""
    long_input = "x" * 1000  # Exceeds max_input_length of 100
    assert security_validator.validate_input(long_input, "test") is False


def test_security_validator_blocked_patterns(security_validator):
    """Test security validator blocks dangerous patterns"""
    dangerous_inputs = [
        "Here is my API key: sk-1234567890abcdef",
        "My password is secret123",
        "The access_token is abc123",
        "<script>alert('xss')</script>",
    ]

    for dangerous_input in dangerous_inputs:
        assert security_validator.validate_input(dangerous_input, "test") is False


def test_security_validator_output_validation(security_validator):
    """Test security validator validates output"""
    safe_output = "This is a safe response with no sensitive information."
    assert security_validator.validate_output(safe_output, "test_agent") is True

    dangerous_output = "Here's your API key: sk-1234567890abcdef"
    assert security_validator.validate_output(dangerous_output, "test_agent") is False


def test_security_validator_sanitize_input(security_validator):
    """Test input sanitization"""
    dirty_input = "<script>alert('xss')</script>Normal text"
    sanitized = security_validator.sanitize_input(dirty_input)

    assert "<script>" not in sanitized
    assert "Normal text" in sanitized


def test_security_validator_sanitize_output(security_validator):
    """Test output sanitization"""
    output_with_secrets = "Response with API key sk-1234567890abcdef"
    sanitized = security_validator.sanitize_output(output_with_secrets)

    assert "sk-1234567890abcdef" not in sanitized
    assert "[REDACTED]" in sanitized


def test_security_validator_disabled_validation():
    """Test validator with validation disabled"""
    config = SecurityConfig(
        enable_input_validation=False, enable_output_filtering=False
    )
    validator = SecurityValidator(config)

    # Should allow anything when disabled
    assert validator.validate_input("api_key: sk-123", "test") is True
    assert validator.validate_output("secret: password", "test") is True


# Test SecureAgentWrapper
@pytest.fixture
def mock_base_agent():
    """Mock base agent for testing"""
    agent = MagicMock()
    agent.config = {"name": "test_agent"}
    agent.execute = AsyncMock(return_value="test response")
    return agent


@pytest.fixture
def secure_agent_wrapper(mock_base_agent):
    """Secure agent wrapper for testing"""
    rate_limiter = RateLimiter(RateLimitConfig(burst_limit=2))
    validator = SecurityValidator(SecurityConfig())
    return SecureAgentWrapper(mock_base_agent, rate_limiter, validator)


async def test_secure_agent_wrapper_successful_execution(
    secure_agent_wrapper, mock_base_agent
):
    """Test successful secure execution"""
    result = await secure_agent_wrapper.secure_execute("Safe input")

    assert result == "test response"
    mock_base_agent.execute.assert_called_once()


async def test_secure_agent_wrapper_rate_limit_exceeded(secure_agent_wrapper):
    """Test rate limit exceeded"""
    # Use up rate limit
    await secure_agent_wrapper.secure_execute("input1")
    await secure_agent_wrapper.secure_execute("input2")

    # Third request should be rate limited
    with pytest.raises(SecurityError, match="Rate limit exceeded"):
        await secure_agent_wrapper.secure_execute("input3")


async def test_secure_agent_wrapper_input_validation_failed():
    """Test input validation failure"""
    rate_limiter = RateLimiter(RateLimitConfig())
    validator = SecurityValidator(SecurityConfig())
    mock_agent = MagicMock()
    wrapper = SecureAgentWrapper(mock_agent, rate_limiter, validator)

    # Input with dangerous pattern
    with pytest.raises(SecurityError, match="Input validation failed"):
        await wrapper.secure_execute("API key: sk-1234567890abcdef")


async def test_secure_agent_wrapper_output_validation_failed():
    """Test output validation failure"""
    rate_limiter = RateLimiter(RateLimitConfig())
    validator = SecurityValidator(SecurityConfig())

    mock_agent = MagicMock()
    mock_agent.execute = AsyncMock(
        return_value="Response with API key: sk-1234567890abcdef"
    )

    wrapper = SecureAgentWrapper(mock_agent, rate_limiter, validator)

    with pytest.raises(SecurityError, match="Output validation failed"):
        await wrapper.secure_execute("Safe input")


# Test utility functions
def test_strip_html_tags():
    """Test HTML tag stripping"""
    html_text = "<p>Hello <b>World</b>!</p><script>alert('xss')</script>"
    clean_text = strip_html_tags(html_text)

    assert "<p>" not in clean_text
    assert "<b>" not in clean_text
    assert "<script>" not in clean_text
    assert "Hello World!" in clean_text


def test_validate_api_key_format():
    """Test API key format validation"""
    # Valid format
    assert validate_api_key_format("sk-1234567890abcdef1234567890abcdef") is True

    # Invalid formats
    assert validate_api_key_format("") is False
    assert validate_api_key_format("invalid") is False
    assert validate_api_key_format("sk-test") is False
    assert validate_api_key_format("sk-fake1234567890abcdef") is False


def test_sanitize_for_logging():
    """Test log sanitization"""
    sensitive_text = "User input with API key sk-1234567890abcdef"
    sanitized = sanitize_for_logging(sensitive_text, max_length=50)

    assert "sk-1234567890abcdef" not in sanitized
    assert len(sanitized) <= 50 + 20  # Account for truncation marker


# Test global instances
def test_get_rate_limiter():
    """Test global rate limiter getter"""
    limiter1 = get_rate_limiter()
    limiter2 = get_rate_limiter()

    # Should return same instance
    assert limiter1 is limiter2


def test_get_security_validator():
    """Test global security validator getter"""
    validator1 = get_security_validator()
    validator2 = get_security_validator()

    # Should return same instance
    assert validator1 is validator2


def test_create_secure_agent():
    """Test secure agent creation"""
    mock_agent = MagicMock()
    secure_agent = create_secure_agent(mock_agent)

    assert isinstance(secure_agent, SecureAgentWrapper)
    assert secure_agent.base_agent is mock_agent


# Integration tests
async def test_rate_limiter_cleanup():
    """Test rate limiter cleans up old requests"""
    config = RateLimitConfig(requests_per_minute=10, requests_per_hour=100)
    limiter = RateLimiter(config)

    identifier = "test_user"

    # Mock time progression
    with patch("time.time") as mock_time:
        # Start at time 1000
        mock_time.return_value = 1000.0

        # Make some requests
        for _ in range(5):
            await limiter.check_rate_limit(identifier)

        # Move forward in time (more than 1 hour)
        mock_time.return_value = 5000.0

        # Old requests should be cleaned up
        assert await limiter.check_rate_limit(identifier) is True

        # Verify cleanup happened
        assert len(limiter._requests[identifier]) == 1  # Only the new request


def test_security_config_file_permissions():
    """Test setting secure file permissions"""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        from ankigen_core.agents.security import set_secure_file_permissions

        # This should not raise an exception
        set_secure_file_permissions(tmp_path)

        # Check permissions (on Unix systems)
        if hasattr(os, "chmod"):
            stat_info = os.stat(tmp_path)
            # Should be readable/writable by owner only
            assert stat_info.st_mode & 0o077 == 0  # No permissions for group/other

    finally:
        os.unlink(tmp_path)


# Error handling tests
async def test_rate_limiter_concurrent_access():
    """Test rate limiter with concurrent access"""
    limiter = RateLimiter(RateLimitConfig(burst_limit=5))
    identifier = "concurrent_user"

    # Run multiple concurrent requests
    tasks = [limiter.check_rate_limit(identifier) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # Some should succeed, some should fail due to burst limit
    success_count = sum(1 for result in results if result)
    assert success_count <= 5  # Should not exceed burst limit


def test_security_validator_error_handling():
    """Test security validator error handling"""
    validator = SecurityValidator(SecurityConfig())

    # Test with None input
    assert validator.validate_input(None, "test") is False

    # Test with extremely large input that might cause issues
    huge_input = "x" * 1000000
    assert validator.validate_input(huge_input, "test") is False


async def test_secure_agent_wrapper_base_agent_error():
    """Test secure agent wrapper handles base agent errors"""
    rate_limiter = RateLimiter(RateLimitConfig())
    validator = SecurityValidator(SecurityConfig())

    mock_agent = MagicMock()
    mock_agent.config = {"name": "test_agent"}
    mock_agent.execute = AsyncMock(side_effect=Exception("Base agent failed"))

    wrapper = SecureAgentWrapper(mock_agent, rate_limiter, validator)

    with pytest.raises(Exception, match="Base agent failed"):
        await wrapper.secure_execute("Safe input")
