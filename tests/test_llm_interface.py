import pytest
from unittest.mock import MagicMock, AsyncMock
from openai import OpenAIError
from ankigen.llm_interface import OpenAIClientManager, OpenAIRateLimiter


@pytest.fixture
def client_manager():
    return OpenAIClientManager()


@pytest.fixture
def rate_limiter():
    return OpenAIRateLimiter(tokens_per_minute=100)


# --- OpenAIClientManager Tests ---


def test_openai_client_manager_init(client_manager):
    """Test initial state of OpenAIClientManager."""
    assert client_manager._client is None
    assert client_manager._api_key is None


@pytest.mark.anyio
async def test_openai_client_manager_initialize_valid_key(client_manager, mocker):
    """Test initialize_client with a valid API key."""
    mock_async_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")

    await client_manager.initialize_client("sk-valid-key")

    assert client_manager._api_key == "sk-valid-key"
    assert client_manager._client is not None
    mock_async_openai.assert_called_once_with(api_key="sk-valid-key")


@pytest.mark.anyio
async def test_openai_client_manager_initialize_invalid_key(client_manager):
    """Test initialize_client with an invalid API key (doesn't start with sk-)."""
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await client_manager.initialize_client("invalid-key")
    assert client_manager._client is None


@pytest.mark.anyio
async def test_openai_client_manager_initialize_empty_key(client_manager):
    """Test initialize_client with an empty API key."""
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await client_manager.initialize_client("")
    assert client_manager._client is None


def test_openai_client_manager_get_client_uninitialized(client_manager):
    """Test get_client before initialization raises RuntimeError."""
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized"):
        client_manager.get_client()


@pytest.mark.anyio
async def test_openai_client_manager_get_client_success(client_manager, mocker):
    """Test get_client after initialization."""
    mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    await client_manager.initialize_client("sk-valid-key")
    client = client_manager.get_client()
    assert client is not None


def test_openai_client_manager_context_manager_sync(mocker):
    """Test synchronous context manager."""
    mock_close = mocker.patch.object(OpenAIClientManager, "close")
    with OpenAIClientManager() as manager:
        assert isinstance(manager, OpenAIClientManager)
    mock_close.assert_called_once()


@pytest.mark.anyio
async def test_openai_client_manager_context_manager_async(mocker):
    """Test asynchronous context manager."""
    mock_aclose = mocker.patch.object(
        OpenAIClientManager, "aclose", new_callable=AsyncMock
    )
    async with OpenAIClientManager() as manager:
        assert isinstance(manager, OpenAIClientManager)
    mock_aclose.assert_called_once()


def test_openai_client_manager_close(client_manager, mocker):
    """Test close method."""
    mock_client = MagicMock()
    client_manager._client = mock_client
    client_manager.close()
    mock_client.close.assert_called_once()
    assert client_manager._client is None


@pytest.mark.anyio
async def test_openai_client_manager_aclose(client_manager, mocker):
    """Test aclose method."""
    mock_client = AsyncMock()
    client_manager._client = mock_client
    await client_manager.aclose()
    mock_client.aclose.assert_called_once()
    assert client_manager._client is None


@pytest.mark.anyio
async def test_openai_client_manager_initialize_openai_error(client_manager, mocker):
    """Test initialize_client when AsyncOpenAI raises OpenAIError."""
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("API Error")
    )
    with pytest.raises(OpenAIError):
        await client_manager.initialize_client("sk-valid-key")
    assert client_manager._client is None


# --- OpenAIRateLimiter Tests ---


@pytest.mark.anyio
async def test_openai_rate_limiter_under_limit(rate_limiter):
    """Test wait_if_needed with tokens under limit."""
    # Limit is 100, we use 50
    await rate_limiter.wait_if_needed(50)
    assert rate_limiter.tokens_used_current_window == 50


@pytest.mark.anyio
async def test_openai_rate_limiter_over_limit(rate_limiter, mocker):
    """Test wait_if_needed with tokens exceeding limit triggers wait."""
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_time = mocker.patch("time.monotonic", return_value=100.0)

    # Ensure limiter's start time matches our mocked time
    rate_limiter.current_window_start_time = 100.0

    # First call uses 60 tokens (OK)
    await rate_limiter.wait_if_needed(60)
    assert rate_limiter.tokens_used_current_window == 60
    assert mock_sleep.call_count == 0

    # Second call uses 50 tokens (60 + 50 = 110 > 100) -> should trigger wait
    # We need to update time.monotonic for the second call's internal check
    # wait_if_needed calls time.monotonic() at the start.
    mock_time.return_value = 100.0

    await rate_limiter.wait_if_needed(50)

    # time_to_wait = (100.0 + 60.0) - 100.0 = 60.0
    mock_sleep.assert_called_once_with(pytest.approx(60.0))
    assert rate_limiter.tokens_used_current_window == 50


@pytest.mark.anyio
async def test_openai_rate_limiter_window_reset(rate_limiter, mocker):
    """Test window reset after 60s."""
    mock_time = mocker.patch("time.monotonic")
    mock_time.return_value = 100.0

    # Initialize limiter with current time
    rate_limiter.current_window_start_time = 100.0
    rate_limiter.tokens_used_current_window = 80

    # Advance time by 61 seconds
    mock_time.return_value = 161.0

    # Adding 10 tokens (80 + 10 = 90 < 100) but window should reset first
    await rate_limiter.wait_if_needed(10)

    assert rate_limiter.tokens_used_current_window == 10
    assert rate_limiter.current_window_start_time == 161.0


@pytest.mark.anyio
async def test_openai_rate_limiter_wait_exactly_limit(rate_limiter, mocker):
    """Test wait_if_needed when exactly reaching the limit."""
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Use 100 tokens (exactly the limit)
    await rate_limiter.wait_if_needed(100)
    assert rate_limiter.tokens_used_current_window == 100
    assert mock_sleep.call_count == 0

    # Next 1 token should trigger wait
    await rate_limiter.wait_if_needed(1)
    assert mock_sleep.call_count == 1
