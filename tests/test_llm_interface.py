import pytest
from unittest.mock import AsyncMock, MagicMock
from ankigen.llm_interface import OpenAIClientManager, OpenAIRateLimiter
from openai import OpenAIError

# --- OpenAIClientManager Tests ---


@pytest.mark.anyio
async def test_initialize_client_valid_key(mocker):
    manager = OpenAIClientManager()
    # Mock AsyncOpenAI class
    mock_openai_class = mocker.patch("ankigen.llm_interface.AsyncOpenAI", autospec=True)

    await manager.initialize_client("sk-valid-key")

    assert manager._api_key == "sk-valid-key"
    assert manager._client is not None
    mock_openai_class.assert_called_once_with(api_key="sk-valid-key")


@pytest.mark.anyio
async def test_initialize_client_invalid_key():
    manager = OpenAIClientManager()
    # Key not starting with sk-
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("invalid-key")
    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_empty_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("")
    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_openai_error(mocker):
    manager = OpenAIClientManager()
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("API Error")
    )

    with pytest.raises(OpenAIError):
        await manager.initialize_client("sk-error")

    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_unexpected_error(mocker):
    manager = OpenAIClientManager()
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=Exception("Unexpected")
    )

    with pytest.raises(
        RuntimeError, match="Unexpected error initializing AsyncOpenAI client."
    ):
        await manager.initialize_client("sk-unexpected")

    assert manager._client is None


def test_get_client_not_initialized():
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_get_client_initialized(mocker):
    manager = OpenAIClientManager()
    mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    await manager.initialize_client("sk-valid")

    client = manager.get_client()
    assert client is not None


def test_context_manager_sync(mocker):
    manager = OpenAIClientManager()
    mock_close = mocker.patch.object(manager, "close")

    with manager as m:
        assert m is manager

    mock_close.assert_called_once()


@pytest.mark.anyio
async def test_context_manager_async(mocker):
    manager = OpenAIClientManager()
    mock_aclose = mocker.patch.object(manager, "aclose", new_callable=AsyncMock)

    async with manager as m:
        assert m is manager

    mock_aclose.assert_called_once()


def test_close_method(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    # Ensure it has 'close' method
    mock_client.close = MagicMock()
    manager._client = mock_client

    manager.close()

    mock_client.close.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_aclose_method(mocker):
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    # Ensure it has 'aclose' method
    mock_client.aclose = AsyncMock()
    manager._client = mock_client

    await manager.aclose()

    mock_client.aclose.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_aclose_method_fallback_to_sync(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    # Only has 'close' method, not 'aclose'
    if hasattr(mock_client, "aclose"):
        del mock_client.aclose
    mock_client.close = MagicMock()
    manager._client = mock_client

    await manager.aclose()

    mock_client.close.assert_called_once()
    assert manager._client is None


# --- OpenAIRateLimiter Tests ---


@pytest.mark.anyio
async def test_rate_limiter_within_limits():
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    # Should not wait
    await limiter.wait_if_needed(50)
    assert limiter.tokens_used_current_window == 50

    await limiter.wait_if_needed(40)
    assert limiter.tokens_used_current_window == 90


@pytest.mark.anyio
async def test_rate_limiter_window_reset(mocker):
    # Mock monotonic to simulate time passing
    mock_monotonic = mocker.patch("time.monotonic")
    mock_monotonic.return_value = 0.0

    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    await limiter.wait_if_needed(80)
    assert limiter.tokens_used_current_window == 80

    # Advance time by 61 seconds
    mock_monotonic.return_value = 61.0

    await limiter.wait_if_needed(10)

    # Should have reset
    assert limiter.tokens_used_current_window == 10
    assert limiter.current_window_start_time == 61.0


@pytest.mark.anyio
async def test_rate_limiter_wait_needed(mocker):
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_monotonic = mocker.patch("time.monotonic")

    # Start at time 0
    mock_monotonic.return_value = 0.0
    limiter = OpenAIRateLimiter(tokens_per_minute=100)

    # Use 90 tokens
    await limiter.wait_if_needed(90)
    assert limiter.tokens_used_current_window == 90

    # Next token request of 20 would exceed 100 (90 + 20 = 110)
    # Current time is 0.0. window_start is 0.0.
    # time_to_wait = (0.0 + 60.0) - 0.0 = 60.0

    # Instead of side_effect list, we use a more robust approach
    # asyncio loop also calls time.monotonic(), so return_value is safer
    # we update it when sleep is called.
    async def side_effect_sleep(delay):
        mock_monotonic.return_value = 60.0  # Fast forward time

    mock_sleep.side_effect = side_effect_sleep

    await limiter.wait_if_needed(20)

    mock_sleep.assert_called_once_with(60.0)
    assert limiter.tokens_used_current_window == 20
    assert limiter.current_window_start_time == 60.0
