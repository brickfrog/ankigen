import time
from unittest.mock import MagicMock, AsyncMock
import pytest

from ankigen.llm_interface import OpenAIClientManager, OpenAIRateLimiter


# --- OpenAIClientManager Tests ---


def test_client_manager_init():
    """Test that OpenAIClientManager initializes with None values."""
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


def test_client_manager_get_client_raises_runtime_error():
    """Test that get_client() raises RuntimeError when client not initialized."""
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_client_manager_initialize_invalid_key_empty():
    """Test initialize_client() rejects empty API keys."""
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await manager.initialize_client("")


@pytest.mark.anyio
async def test_client_manager_initialize_invalid_key_prefix():
    """Test initialize_client() rejects keys missing 'sk-' prefix."""
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await manager.initialize_client("invalid-key")


@pytest.mark.anyio
async def test_client_manager_initialize_valid_key(mocker):
    """Test initialize_client() with a valid-format key using mock."""
    # Mock AsyncOpenAI constructor
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI", autospec=True)

    manager = OpenAIClientManager()
    await manager.initialize_client("sk-validkey123")

    assert manager._api_key == "sk-validkey123"
    assert manager._client is not None
    mock_openai.assert_called_once_with(api_key="sk-validkey123")
    assert manager.get_client() == mock_openai.return_value


def test_client_manager_close():
    """Test close() sets client to None and calls close() on client."""
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    manager.close()

    assert manager._client is None
    mock_client.close.assert_called_once()


@pytest.mark.anyio
async def test_client_manager_aclose():
    """Test aclose() sets client to None and calls aclose() on client."""
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client

    await manager.aclose()

    assert manager._client is None
    mock_client.aclose.assert_called_once()


def test_client_manager_context_manager():
    """Test __enter__/__exit__ context manager protocol."""
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    with manager as m:
        assert m == manager
        assert m._client == mock_client

    assert manager._client is None
    mock_client.close.assert_called_once()


@pytest.mark.anyio
async def test_client_manager_async_context_manager():
    """Test __aenter__/__aexit__ context manager protocol."""
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client

    async with manager as m:
        assert m == manager
        assert m._client == mock_client

    assert manager._client is None
    mock_client.aclose.assert_called_once()


# --- OpenAIRateLimiter Tests ---


def test_rate_limiter_init():
    """Test initial state (tokens_used_current_window=0)."""
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    assert limiter.tokens_per_minute_limit == 1000
    assert limiter.tokens_used_current_window == 0
    assert isinstance(limiter.current_window_start_time, float)


@pytest.mark.anyio
async def test_rate_limiter_wait_if_needed_tracks_usage():
    """Test wait_if_needed() tracks token usage correctly."""
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)

    await limiter.wait_if_needed(100)
    assert limiter.tokens_used_current_window == 100

    await limiter.wait_if_needed(250)
    assert limiter.tokens_used_current_window == 350


@pytest.mark.anyio
async def test_rate_limiter_window_reset(mocker):
    """Test window reset after 60 seconds (mock time.monotonic)."""
    start_time = 1000.0
    real_monotonic = time.monotonic

    def monotonic_gen():
        yield start_time  # Call in __init__
        yield start_time + 30.0  # Call 1 in first wait_if_needed
        yield start_time + 61.0  # Call 1 in second wait_if_needed
        while True:
            yield real_monotonic()

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=monotonic_gen())

    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    limiter.tokens_used_current_window = 500

    # Still within the same window (1000.0 + 30.0)
    await limiter.wait_if_needed(100)
    assert limiter.tokens_used_current_window == 600

    # 60 seconds passed (1000.0 + 61.0)
    await limiter.wait_if_needed(50)
    # Should have reset to 0 then added 50
    assert limiter.tokens_used_current_window == 50
    assert limiter.current_window_start_time == start_time + 61.0


@pytest.mark.anyio
async def test_rate_limiter_waits_when_approaching_limit(mocker):
    """Test that it waits when approaching the limit (mock asyncio.sleep)."""
    start_time = 1000.0

    # Use a generator that yields test values then falls back to real monotonic
    real_monotonic = time.monotonic

    def monotonic_gen():
        yield start_time  # Call in __init__
        yield start_time + 10.0  # Call 1 in wait_if_needed
        yield start_time + 60.0  # Call 2 in wait_if_needed (after sleep)
        while True:
            yield real_monotonic()

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=monotonic_gen())

    mock_sleep = mocker.patch(
        "ankigen.llm_interface.asyncio.sleep", new_callable=AsyncMock
    )

    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    limiter.tokens_used_current_window = 900

    await limiter.wait_if_needed(200)

    mock_sleep.assert_called_once_with(50.0)
    assert limiter.tokens_used_current_window == 200
    assert limiter.current_window_start_time == start_time + 60.0


@pytest.mark.anyio
async def test_client_manager_initialize_exception(mocker):
    """Test initialize_client() handles exceptions and sets client to None."""
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=Exception("Connection error")
    )

    manager = OpenAIClientManager()
    with pytest.raises(
        RuntimeError, match="Unexpected error initializing AsyncOpenAI client"
    ):
        await manager.initialize_client("sk-validkey")

    assert manager._client is None


def test_retryable_errors_constant():
    """Test that RETRYABLE_OPENAI_ERRORS contains the expected error types."""
    from ankigen.llm_interface import RETRYABLE_OPENAI_ERRORS
    from openai import APIConnectionError, RateLimitError, APIStatusError

    assert APIConnectionError in RETRYABLE_OPENAI_ERRORS
    assert RateLimitError in RETRYABLE_OPENAI_ERRORS
    assert APIStatusError in RETRYABLE_OPENAI_ERRORS
