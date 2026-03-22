import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from openai import AsyncOpenAI

from ankigen.llm_interface import (
    OpenAIClientManager,
    OpenAIRateLimiter,
    structured_agent_call,
)
from ankigen.utils import ResponseCache

# --- OpenAIClientManager Tests ---


@pytest.mark.anyio
async def test_initialize_client_success(mocker):
    manager = OpenAIClientManager()
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")

    await manager.initialize_client("sk-valid-key")

    assert manager._api_key == "sk-valid-key"
    assert manager._client is not None
    mock_openai.assert_called_once_with(api_key="sk-valid-key")


@pytest.mark.anyio
async def test_initialize_client_invalid_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("invalid-key")
    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_empty_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("")
    assert manager._client is None


def test_get_client_before_init():
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_get_client_after_init(mocker):
    manager = OpenAIClientManager()
    mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    await manager.initialize_client("sk-valid-key")

    client = manager.get_client()
    assert client is not None


def test_context_manager_sync(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    with manager:
        pass

    mock_client.close.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_async_context_manager(mocker):
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client

    async with manager:
        pass

    mock_client.aclose.assert_awaited_once()
    assert manager._client is None


def test_close_clears_client(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    manager.close()

    mock_client.close.assert_called_once()
    assert manager._client is None


# --- OpenAIRateLimiter Tests ---


def test_init_default():
    limiter = OpenAIRateLimiter()
    assert limiter.tokens_per_minute_limit == 60000
    assert limiter.tokens_used_current_window == 0


@pytest.mark.anyio
async def test_wait_if_needed_no_wait(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep")

    await limiter.wait_if_needed(50)
    assert limiter.tokens_used_current_window == 50
    assert mock_sleep.call_count == 0


@pytest.mark.anyio
async def test_wait_if_needed_exceeds_limit(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 60

    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch("time.monotonic", return_value=100.0)

    # current_window_start_time defaults to time.monotonic() in __init__
    limiter.current_window_start_time = 100.0
    # 60 + 50 > 100, should wait
    # time_to_wait = (100.0 + 60.0) - 100.0 = 60.0
    await limiter.wait_if_needed(50)

    mock_sleep.assert_awaited_once_with(pytest.approx(60.0))
    assert limiter.tokens_used_current_window == 50


@pytest.mark.anyio
async def test_window_reset(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 80
    limiter.current_window_start_time = 100.0

    mocker.patch("time.monotonic", return_value=161.0)

    await limiter.wait_if_needed(10)

    # Window should have reset because 161 - 100 >= 60
    assert limiter.tokens_used_current_window == 10
    assert limiter.current_window_start_time == 161.0


# --- structured_agent_call Tests ---


class MockOutput(BaseModel):
    field: str


@pytest.mark.anyio
async def test_cache_hit(mocker):
    mock_cache = MagicMock(spec=ResponseCache)
    mock_cache.get.return_value = {"field": "cached_value"}

    mock_client = MagicMock(spec=AsyncOpenAI)

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        cache=mock_cache,
        cache_key="test_key",
    )

    assert result.field == "cached_value"
    mock_cache.get.assert_called_once_with("test_key", "gpt-4")


@pytest.mark.anyio
async def test_successful_call(mocker):
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="success")
    mock_runner.return_value = mock_result

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
    )

    assert result.field == "success"
    assert mock_runner.call_count == 1


@pytest.mark.anyio
async def test_retry_on_timeout(mocker):
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # First call times out, second call succeeds
    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="success")

    mock_runner.side_effect = [asyncio.TimeoutError(), mock_result]

    # We need to mock asyncio.wait_for as well because structured_agent_call wraps Runner.run in wait_for
    # Actually, Runner.run is what we're mocking, and wait_for will raise TimeoutError if we make it.
    # But wait_for is a builtin, so we can just let it happen or mock it.
    # Let's mock wait_for to be sure.
    mock_wait_for = mocker.patch("asyncio.wait_for")
    mock_wait_for.side_effect = [asyncio.TimeoutError(), mock_result]

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=2,
    )

    assert result.field == "success"
    assert mock_wait_for.call_count == 2
    mock_sleep.assert_awaited_once()


@pytest.mark.anyio
async def test_raises_after_max_retries(mocker):
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_wait_for = mocker.patch("asyncio.wait_for")
    mock_wait_for.side_effect = asyncio.TimeoutError()
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    with pytest.raises(asyncio.TimeoutError):
        await structured_agent_call(
            openai_client=mock_client,
            model="gpt-4",
            instructions="instr",
            user_input="input",
            output_type=MockOutput,
            retry_attempts=2,
        )

    assert mock_wait_for.call_count == 2
