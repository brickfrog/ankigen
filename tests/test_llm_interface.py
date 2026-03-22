import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAIError

from ankigen.llm_interface import (
    OpenAIClientManager,
    structured_agent_call,
    structured_output_completion,
    OpenAIRateLimiter,
    GenericJsonOutput,
)
from ankigen.utils import ResponseCache


# Test Pydantic model
class MockOutput(BaseModel):
    answer: str


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=AsyncOpenAI)


@pytest.fixture
def response_cache():
    return ResponseCache(maxsize=10)


# --- OpenAIClientManager Tests ---


@pytest.mark.anyio
async def test_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None
    with pytest.raises(RuntimeError, match="not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_client_manager_initialize_valid_key(mocker):
    mock_async_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    manager = OpenAIClientManager()
    await manager.initialize_client("sk-valid-key")
    assert manager._api_key == "sk-valid-key"
    assert manager._client is not None
    assert manager.get_client() == mock_async_openai.return_value


@pytest.mark.anyio
async def test_client_manager_initialize_invalid_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await manager.initialize_client("invalid-key")
    assert manager._client is None


@pytest.mark.anyio
async def test_client_manager_initialize_failure(mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("API Error")
    )
    manager = OpenAIClientManager()
    with pytest.raises(OpenAIError):
        await manager.initialize_client("sk-fail")
    assert manager._client is None


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


# --- structured_agent_call Tests ---


@pytest.mark.anyio
async def test_structured_agent_call_cache_hit(
    mock_openai_client, response_cache, mocker
):
    response_cache.set("cache_key", "gpt-4", {"answer": "cached answer"})
    mock_runner = mocker.patch("ankigen.llm_interface.Runner.run")

    result = await structured_agent_call(
        openai_client=mock_openai_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        cache=response_cache,
        cache_key="cache_key",
    )

    assert result.answer == "cached answer"
    mock_runner.assert_not_called()


@pytest.mark.anyio
async def test_structured_agent_call_cache_miss_success(
    mock_openai_client, response_cache, mocker
):
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mock_result = MagicMock()
    mock_result.final_output = MockOutput(answer="fresh answer")
    mock_runner.return_value = mock_result

    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")

    result = await structured_agent_call(
        openai_client=mock_openai_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        cache=response_cache,
        cache_key="new_key",
    )

    assert result.answer == "fresh answer"
    assert response_cache.get("new_key", "gpt-4") == {"answer": "fresh answer"}


@pytest.mark.anyio
async def test_structured_agent_call_retry_on_timeout(mock_openai_client, mocker):
    mock_wait_for = mocker.patch("asyncio.wait_for")
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(answer="retry success")

    # Fail twice with timeout, then succeed
    mock_wait_for.side_effect = [
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        mock_result,
    ]

    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mocker.patch("ankigen.llm_interface.Runner.run")

    result = await structured_agent_call(
        openai_client=mock_openai_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=3,
    )

    assert result.answer == "retry success"
    assert mock_wait_for.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.anyio
async def test_structured_agent_call_timeout_exhausted(mock_openai_client, mocker):
    mocker.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mocker.patch("ankigen.llm_interface.Runner.run")

    with pytest.raises(asyncio.TimeoutError):
        await structured_agent_call(
            openai_client=mock_openai_client,
            model="gpt-4",
            instructions="instr",
            user_input="input",
            output_type=MockOutput,
            retry_attempts=2,
        )


@pytest.mark.anyio
async def test_structured_agent_call_generic_exception_retry(
    mock_openai_client, mocker
):
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    mock_result = MagicMock()

    mock_result.final_output = MockOutput(answer="recovered")

    # Fail once with Exception, then succeed
    mock_runner.side_effect = [Exception("Random Error"), mock_result]

    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")

    result = await structured_agent_call(
        openai_client=mock_openai_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=2,
    )

    assert result.answer == "recovered"
    assert mock_runner.call_count == 2


@pytest.mark.anyio
async def test_structured_agent_call_reasoning_effort(mock_openai_client, mocker):
    mocker.patch("ankigen.llm_interface.Agent")
    mock_model_settings_cls = mocker.patch("ankigen.llm_interface.ModelSettings")
    mock_reasoning_cls = mocker.patch("openai.types.shared.Reasoning")

    mocker.patch("ankigen.llm_interface.Runner.run", new_callable=AsyncMock)
    mocker.patch("ankigen.llm_interface.set_default_openai_client")

    await structured_agent_call(
        openai_client=mock_openai_client,
        model="gpt-5.2",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
    )

    # Check that Reasoning was instantiated with effort="none"
    mock_reasoning_cls.assert_called_once_with(effort="none")
    # Check that ModelSettings was instantiated with reasoning
    args, kwargs = mock_model_settings_cls.call_args
    assert "reasoning" in kwargs


# --- structured_output_completion Tests ---


@pytest.mark.anyio
async def test_structured_output_completion_success(
    mock_openai_client, response_cache, mocker
):
    mock_agent_call = mocker.patch(
        "ankigen.llm_interface.structured_agent_call", new_callable=AsyncMock
    )
    mock_agent_call.return_value = GenericJsonOutput(data="some data")

    result = await structured_output_completion(
        openai_client=mock_openai_client,
        model="gpt-4",
        response_format={},
        system_prompt="system",
        user_prompt="user",
        cache=response_cache,
    )

    assert result == {"data": "some data"}
    mock_agent_call.assert_called_once()
    # Check instruction was appended
    instructions = mock_agent_call.call_args.kwargs["instructions"]
    assert "JSON object matching the specified schema" in instructions


@pytest.mark.anyio
async def test_structured_output_completion_no_instruction_append_if_present(
    mock_openai_client, response_cache, mocker
):
    mock_agent_call = mocker.patch(
        "ankigen.llm_interface.structured_agent_call", new_callable=AsyncMock
    )

    prompt = "Give me a JSON object matching the specified schema."
    await structured_output_completion(
        openai_client=mock_openai_client,
        model="gpt-4",
        response_format={},
        system_prompt=prompt,
        user_prompt="user",
        cache=response_cache,
    )

    instructions = mock_agent_call.call_args.kwargs["instructions"]
    # Should not be appended twice
    assert instructions.count("JSON object matching the specified schema") == 1


@pytest.mark.anyio
async def test_structured_output_completion_error_propagation(
    mock_openai_client, response_cache, mocker
):
    mocker.patch(
        "ankigen.llm_interface.structured_agent_call",
        side_effect=Exception("Agent failed"),
    )

    with pytest.raises(Exception, match="Agent failed"):
        await structured_output_completion(
            openai_client=mock_openai_client,
            model="gpt-4",
            response_format={},
            system_prompt="system",
            user_prompt="user",
            cache=response_cache,
        )


# --- OpenAIRateLimiter Tests ---


@pytest.mark.anyio
async def test_rate_limiter_window_reset(mocker):
    # Use a side_effect function to avoid StopIteration and only affect our module
    times = [100.0, 161.0, 161.0, 161.0, 161.0]

    def mock_time():
        if times:
            return times.pop(0)
        return 200.0

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=mock_time)

    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 50
    # Ensure limiter's start time is what we expect
    limiter.current_window_start_time = 100.0

    await limiter.wait_if_needed(10)

    assert limiter.tokens_used_current_window == 10
    assert limiter.current_window_start_time == 161.0


@pytest.mark.anyio
async def test_rate_limiter_wait_needed(mocker):
    times = [110.0, 110.0, 160.0, 160.0, 160.0, 160.0]

    def mock_time():
        if times:
            return times.pop(0)
        return 200.0

    mocker.patch("ankigen.llm_interface.time.monotonic", side_effect=mock_time)
    mock_sleep = mocker.patch(
        "ankigen.llm_interface.asyncio.sleep", new_callable=AsyncMock
    )

    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 95
    limiter.current_window_start_time = 100.0

    # Request 10 tokens, exceeds 100
    await limiter.wait_if_needed(10)

    # Should wait for remaining 50s (100 + 60 - 110)
    mock_sleep.assert_called_once_with(50.0)
    assert limiter.tokens_used_current_window == 10


@pytest.mark.anyio
async def test_rate_limiter_token_counting():
    limiter = OpenAIRateLimiter(tokens_per_minute=1000)
    await limiter.wait_if_needed(100)
    assert limiter.tokens_used_current_window == 100
    await limiter.wait_if_needed(200)
    assert limiter.tokens_used_current_window == 300
