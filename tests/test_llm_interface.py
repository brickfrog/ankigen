import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from openai import OpenAIError
from pydantic import BaseModel
from ankigen.llm_interface import (
    OpenAIClientManager,
    OpenAIRateLimiter,
    structured_agent_call,
    structured_output_completion,
    GenericJsonOutput,
)

# ... (rest of imports unchanged)

# --- structured_agent_call and structured_output_completion Tests ---


class MockOutputType(BaseModel):
    name: str


@pytest.mark.anyio
async def test_structured_agent_call_cache_hit(mocker):
    mock_cache = MagicMock()
    mock_cache.get.return_value = {"name": "test"}
    mock_client = AsyncMock()

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutputType,
        cache=mock_cache,
        cache_key="key",
    )

    assert isinstance(result, MockOutputType)
    assert result.name == "test"
    mock_cache.get.assert_called_once_with("key", "gpt-4")


@pytest.mark.anyio
async def test_structured_agent_call_success(mocker):
    mock_client = AsyncMock()
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mock_agent = mocker.patch("ankigen.llm_interface.Agent")
    mock_runner = mocker.patch("ankigen.llm_interface.Runner")

    mock_result = MagicMock()
    mock_result.final_output = MockOutputType(name="success")
    mock_runner.run = AsyncMock(return_value=mock_result)

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-5.2",
        instructions="instr",
        user_input="input",
        output_type=MockOutputType,
    )

    assert result.name == "success"
    mock_agent.assert_called_once()
    mock_runner.run.assert_called_once()


@pytest.mark.anyio
async def test_structured_output_completion_success(mocker):
    mock_client = AsyncMock()
    mock_cache = MagicMock()

    # Mock structured_agent_call instead of mocking everything inside it again
    mock_result = GenericJsonOutput()
    mock_result.name = "result"  # GenericJsonOutput allows extra fields

    mocker.patch(
        "ankigen.llm_interface.structured_agent_call",
        new_callable=AsyncMock,
        return_value=mock_result,
    )

    result = await structured_output_completion(
        openai_client=mock_client,
        model="gpt-4",
        response_format={},
        system_prompt="sys",
        user_prompt="user",
        cache=mock_cache,
    )

    assert result == {"name": "result"}


def test_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


@pytest.mark.anyio
async def test_client_manager_initialize_client_valid(mocker):
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    manager = OpenAIClientManager()
    api_key = "sk-valid-key"

    await manager.initialize_client(api_key)

    assert manager._api_key == api_key
    assert manager._client is not None
    mock_openai.assert_called_once_with(api_key=api_key)


@pytest.mark.anyio
async def test_client_manager_initialize_client_invalid():
    manager = OpenAIClientManager()

    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("invalid-key")

    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("")


@pytest.mark.anyio
async def test_client_manager_initialize_client_openai_error(mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("OpenAI error")
    )
    manager = OpenAIClientManager()

    with pytest.raises(OpenAIError):
        await manager.initialize_client("sk-key")

    assert manager._client is None


@pytest.mark.anyio
async def test_client_manager_initialize_client_unexpected_error(mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=Exception("Unexpected error")
    )
    manager = OpenAIClientManager()

    with pytest.raises(
        RuntimeError, match="Unexpected error initializing AsyncOpenAI client."
    ):
        await manager.initialize_client("sk-key")

    assert manager._client is None


def test_client_manager_get_client_success(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    assert manager.get_client() == mock_client


def test_client_manager_get_client_uninitialized():
    manager = OpenAIClientManager()

    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized."):
        manager.get_client()


def test_client_manager_sync_context_manager(mocker):
    mock_close = mocker.patch.object(OpenAIClientManager, "close")
    with OpenAIClientManager() as manager:
        assert isinstance(manager, OpenAIClientManager)

    mock_close.assert_called_once()


@pytest.mark.anyio
async def test_client_manager_async_context_manager(mocker):
    mock_aclose = mocker.patch.object(
        OpenAIClientManager, "aclose", new_callable=AsyncMock
    )
    async with OpenAIClientManager() as manager:
        assert isinstance(manager, OpenAIClientManager)

    mock_aclose.assert_called_once()


def test_client_manager_close(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    manager.close()

    mock_client.close.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_client_manager_aclose(mocker):
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client

    await manager.aclose()

    mock_client.aclose.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_client_manager_aclose_fallback_to_sync(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    del mock_client.aclose  # Ensure it doesn't have aclose
    manager._client = mock_client

    await manager.aclose()

    mock_client.close.assert_called_once()
    assert manager._client is None


# --- OpenAIRateLimiter Tests ---


def test_rate_limiter_init():
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    assert limiter.tokens_per_minute_limit == 1000
    assert limiter.tokens_used_current_window == 0
    assert isinstance(limiter.current_window_start_time, float)


def test_rate_limiter_init_default():
    limiter = OpenAIRateLimiter()
    assert limiter.tokens_per_minute_limit == 60000


@pytest.mark.anyio
async def test_rate_limiter_wait_if_needed_under_limit(mocker):
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    limiter = OpenAIRateLimiter(tokens_per_minute=100)

    await limiter.wait_if_needed(50)
    assert limiter.tokens_used_current_window == 50
    mock_sleep.assert_not_called()

    await limiter.wait_if_needed(40)
    assert limiter.tokens_used_current_window == 90
    mock_sleep.assert_not_called()


@pytest.mark.anyio
async def test_rate_limiter_wait_if_needed_at_limit(mocker):
    # Mock monotonic but fall back to real one to not break asyncio
    real_monotonic = time.monotonic

    def side_effect():
        yield 100.0  # init
        yield 100.0  # start of wait_if_needed
        yield 160.0  # after sleep/reset
        while True:
            yield real_monotonic()

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=side_effect())

    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 90

    # This should trigger a wait
    # time_to_wait = (100.0 + 60.0) - 100.0 = 60.0
    await limiter.wait_if_needed(20)

    mock_sleep.assert_called_once_with(pytest.approx(60.0))
    assert limiter.tokens_used_current_window == 20
    assert limiter.current_window_start_time == 160.0


@pytest.mark.anyio
async def test_rate_limiter_window_reset(mocker):
    real_monotonic = time.monotonic

    def side_effect():
        yield 100.0  # init
        yield 161.0  # start of wait_if_needed
        while True:
            yield real_monotonic()

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=side_effect())

    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 90

    # Window should reset because 161.0 - 100.0 > 60.0
    await limiter.wait_if_needed(20)

    assert limiter.tokens_used_current_window == 20
    assert limiter.current_window_start_time == 161.0
