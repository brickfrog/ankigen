import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from openai import OpenAIError, AsyncOpenAI
from ankigen.llm_interface import (
    OpenAIClientManager,
    OpenAIRateLimiter,
    structured_agent_call,
    structured_output_completion,
    GenericJsonOutput,
)
from pydantic import BaseModel
from ankigen.utils import ResponseCache


# --- OpenAIClientManager Tests ---


def test_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


@pytest.mark.anyio
async def test_initialize_client_valid(mocker):
    # Mock AsyncOpenAI to avoid real initialization
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    manager = OpenAIClientManager()
    await manager.initialize_client("sk-valid-key")
    assert manager._api_key == "sk-valid-key"
    mock_openai.assert_called_once_with(api_key="sk-valid-key")
    assert manager._client is not None


@pytest.mark.anyio
async def test_initialize_client_invalid():
    manager = OpenAIClientManager()
    # Key must start with "sk-"
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("invalid-key")
    assert manager._client is None

    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        await manager.initialize_client("")
    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_openai_error(mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("API Error")
    )
    manager = OpenAIClientManager()
    with pytest.raises(OpenAIError, match="API Error"):
        await manager.initialize_client("sk-key")
    assert manager._client is None


@pytest.mark.anyio
async def test_initialize_client_unexpected_error(mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=Exception("Unexpected")
    )
    manager = OpenAIClientManager()
    with pytest.raises(
        RuntimeError, match="Unexpected error initializing AsyncOpenAI client."
    ):
        await manager.initialize_client("sk-key")
    assert manager._client is None


def test_get_client_success():
    manager = OpenAIClientManager()
    mock_client = MagicMock(spec=AsyncOpenAI)
    manager._client = mock_client
    assert manager.get_client() == mock_client


def test_get_client_failure():
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized."):
        manager.get_client()


def test_sync_context_manager(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client

    with manager as m:
        assert m == manager

    mock_client.close.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_async_context_manager(mocker):
    manager = OpenAIClientManager()
    # Don't use spec=AsyncOpenAI because it might not have 'aclose' in this environment
    mock_client = AsyncMock()
    manager._client = mock_client

    async with manager as m:
        assert m == manager

    # It should try aclose first, then fallback to close
    assert mock_client.aclose.called or mock_client.close.called
    assert manager._client is None


def test_close_sync(mocker):
    manager = OpenAIClientManager()
    mock_client = MagicMock()
    manager._client = mock_client
    manager.close()
    mock_client.close.assert_called_once()
    assert manager._client is None


@pytest.mark.anyio
async def test_aclose_async(mocker):
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client
    await manager.aclose()
    # It should try aclose first, then fallback to close
    assert mock_client.aclose.called or mock_client.close.called
    assert manager._client is None


# --- OpenAIRateLimiter Tests ---


@pytest.mark.anyio
async def test_rate_limiter_init():
    limiter = OpenAIRateLimiter(tokens_per_minute=5000)
    assert limiter.tokens_per_minute_limit == 5000
    assert limiter.tokens_used_current_window == 0


@pytest.mark.anyio
async def test_rate_limiter_under_limit(mocker):
    mocker.patch("ankigen.llm_interface.time.monotonic", return_value=100.0)
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)

    await limiter.wait_if_needed(100)
    assert limiter.tokens_used_current_window == 100

    await limiter.wait_if_needed(200)
    assert limiter.tokens_used_current_window == 300


@pytest.mark.anyio
async def test_rate_limiter_window_reset(mocker):
    mock_monotonic = mocker.patch("ankigen.llm_interface.time.monotonic")
    # Start at T=100
    mock_monotonic.return_value = 100.0
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    limiter.tokens_used_current_window = 500

    # Move to T=161 (more than 60s later)
    mock_monotonic.return_value = 161.0
    await limiter.wait_if_needed(100)

    assert limiter.tokens_used_current_window == 100
    assert limiter.current_window_start_time == 161.0


@pytest.mark.anyio
async def test_rate_limiter_wait_needed(mocker):
    # Patch monotonic ONLY in the target module to avoid messing with asyncio/anyio loops
    mock_monotonic = mocker.patch("ankigen.llm_interface.time.monotonic")
    mock_sleep = mocker.patch(
        "ankigen.llm_interface.asyncio.sleep", new_callable=AsyncMock
    )

    # Return values for time.monotonic()
    # 1. 100.0 (in __init__)
    # 2. 100.0 (at start of wait_if_needed)
    # 3. 160.0 (after sleep in wait_if_needed)
    vals = [100.0, 100.0, 160.0]

    def side_effect():
        if len(vals) > 1:
            return vals.pop(0)
        return vals[0]

    mock_monotonic.side_effect = side_effect

    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    limiter.tokens_used_current_window = 950

    # Request 100 tokens -> 1050 > 1000. Must wait.
    # Window ends at 100 + 60 = 160.
    # Current time is 100. Sleep duration = 60.

    await limiter.wait_if_needed(100)

    mock_sleep.assert_called_once_with(pytest.approx(60.0))
    assert limiter.tokens_used_current_window == 100
    assert limiter.current_window_start_time == 160.0


# --- structured_agent_call Tests ---


class MockOutput(BaseModel):
    field: str


@pytest.mark.anyio
async def test_structured_agent_call_success(mocker):
    # Reset any global state that might have been affected by previous failures
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mock_agent = mocker.patch("ankigen.llm_interface.Agent")
    mock_runner = mocker.patch("ankigen.llm_interface.Runner")

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="value")
    mock_runner.run = AsyncMock(return_value=mock_result)

    client = MagicMock(spec=AsyncOpenAI)
    result = await structured_agent_call(
        openai_client=client,
        model="gpt-4",
        instructions="inst",
        user_input="input",
        output_type=MockOutput,
    )

    assert result.field == "value"
    # Verify mock_agent was called correctly
    mock_agent.assert_called_once()


@pytest.mark.anyio
async def test_structured_agent_call_cache_hit(mocker):
    mock_cache = MagicMock(spec=ResponseCache)
    mock_cache.get.return_value = {"field": "cached"}

    client = MagicMock(spec=AsyncOpenAI)
    result = await structured_agent_call(
        openai_client=client,
        model="gpt-4",
        instructions="inst",
        user_input="input",
        output_type=MockOutput,
        cache=mock_cache,
        cache_key="key",
    )

    assert result.field == "cached"
    mock_cache.get.assert_called_once_with("key", "gpt-4")


@pytest.mark.anyio
async def test_structured_agent_call_retry_on_timeout(mocker):
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mocker.patch("ankigen.llm_interface.Runner")
    mock_sleep = mocker.patch(
        "ankigen.llm_interface.asyncio.sleep", new_callable=AsyncMock
    )

    # First try fails with TimeoutError, second try succeeds
    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="success")

    # Mock asyncio.wait_for to raise TimeoutError then return mock_result
    # Use patch directly on ankigen.llm_interface.asyncio.wait_for
    mocker.patch(
        "ankigen.llm_interface.asyncio.wait_for",
        side_effect=[asyncio.TimeoutError(), mock_result],
    )

    client = MagicMock(spec=AsyncOpenAI)
    result = await structured_agent_call(
        openai_client=client,
        model="gpt-4",
        instructions="inst",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=2,
    )

    assert result.field == "success"
    assert mock_sleep.call_count == 1


@pytest.mark.anyio
async def test_structured_agent_call_fail_after_retries(mocker):
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mock_runner = mocker.patch("ankigen.llm_interface.Runner")
    mocker.patch("ankigen.llm_interface.asyncio.sleep", new_callable=AsyncMock)
    mock_runner.run.side_effect = Exception("Permanent Fail")

    client = MagicMock(spec=AsyncOpenAI)
    with pytest.raises(Exception, match="Permanent Fail"):
        await structured_agent_call(
            openai_client=client,
            model="gpt-4",
            instructions="inst",
            user_input="input",
            output_type=MockOutput,
            retry_attempts=2,
        )
    assert mock_runner.run.call_count == 2


@pytest.mark.anyio
async def test_structured_agent_call_gpt5_reasoning(mocker):
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mock_runner = mocker.patch("ankigen.llm_interface.Runner")

    # Reasoning is imported locally, patch the source module
    mock_reasoning = mocker.patch("openai.types.shared.Reasoning", create=True)
    # Return a dict to satisfy Pydantic validation in ModelSettings
    mock_reasoning.return_value = {"effort": "none"}

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="val")
    mock_runner.run = AsyncMock(return_value=mock_result)

    client = MagicMock(spec=AsyncOpenAI)
    await structured_agent_call(
        openai_client=client,
        model="gpt-5-model",
        instructions="inst",
        user_input="input",
        output_type=MockOutput,
    )

    mock_reasoning.assert_called_once_with(effort="none")


@pytest.mark.anyio
async def test_structured_agent_call_gpt5_chat_latest(mocker):
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mock_runner = mocker.patch("ankigen.llm_interface.Runner")
    # Mock Reasoning to check it's NOT called
    mock_reasoning = mocker.patch("openai.types.shared.Reasoning", create=True)

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="val")
    mock_runner.run = AsyncMock(return_value=mock_result)

    client = MagicMock(spec=AsyncOpenAI)
    await structured_agent_call(
        openai_client=client,
        model="gpt-5-chat-latest",
        instructions="inst",
        user_input="input",
        output_type=MockOutput,
    )

    assert mock_reasoning.call_count == 0


# --- structured_output_completion Tests ---


@pytest.mark.anyio
async def test_structured_output_completion_success(mocker):
    mock_call = mocker.patch(
        "ankigen.llm_interface.structured_agent_call", new_callable=AsyncMock
    )
    mock_call.return_value = GenericJsonOutput(data="test")

    client = MagicMock(spec=AsyncOpenAI)
    cache = MagicMock(spec=ResponseCache)
    result = await structured_output_completion(
        openai_client=client,
        model="gpt-4",
        response_format={},
        system_prompt="sys",
        user_prompt="user",
        cache=cache,
    )

    assert result == {"data": "test"}
    mock_call.assert_called_once()
    # Check if system prompt was augmented
    args, kwargs = mock_call.call_args
    assert "JSON object matching the specified schema" in kwargs["instructions"]
