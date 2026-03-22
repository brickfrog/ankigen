import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from ankigen.llm_interface import (
    OpenAIClientManager,
    structured_agent_call,
    structured_output_completion,
    OpenAIRateLimiter,
    GenericJsonOutput,
)


class MockOutput(BaseModel):
    field: str


@pytest.mark.anyio
async def test_openai_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    with pytest.raises(RuntimeError, match="not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_openai_client_manager_initialize_success():
    manager = OpenAIClientManager()
    with patch("ankigen.llm_interface.AsyncOpenAI") as mock_async_openai:
        await manager.initialize_client("sk-test-key")
        assert manager.get_client() is not None
        mock_async_openai.assert_called_once_with(api_key="sk-test-key")


@pytest.mark.anyio
async def test_openai_client_manager_initialize_invalid_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await manager.initialize_client("invalid-key")


@pytest.mark.anyio
async def test_openai_client_manager_aclose():
    manager = OpenAIClientManager()
    with patch("ankigen.llm_interface.AsyncOpenAI") as mock_async_openai:
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        mock_async_openai.return_value = mock_client

        await manager.initialize_client("sk-test-key")
        await manager.aclose()

        mock_client.aclose.assert_called_once()
        assert manager._client is None


@pytest.mark.anyio
async def test_structured_agent_call_cached():
    mock_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get.return_value = {"field": "cached_value"}

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        cache=mock_cache,
        cache_key="key",
    )

    assert result.field == "cached_value"
    mock_cache.get.assert_called_once_with("key", "gpt-4")


@pytest.mark.anyio
@pytest.mark.anyio
async def test_structured_agent_call_success():
    mock_client = MagicMock()
    with (
        patch("ankigen.llm_interface.Runner.run", new_callable=AsyncMock) as mock_run,
        patch("ankigen.llm_interface.Agent"),
        patch("ankigen.llm_interface.set_default_openai_client"),
    ):
        mock_result = MagicMock()
        mock_result.final_output = MockOutput(field="success")
        mock_run.return_value = mock_result

        result = await structured_agent_call(
            openai_client=mock_client,
            model="gpt-4",
            instructions="instr",
            user_input="input",
            output_type=MockOutput,
        )

        assert result.field == "success"
        mock_run.assert_called_once()


@pytest.mark.anyio
async def test_structured_agent_call_retry(mocker):
    mock_client = MagicMock()
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mocker.patch("asyncio.sleep", AsyncMock())

    mock_run = mocker.patch("ankigen.llm_interface.Runner.run", new_callable=AsyncMock)

    # Fail twice, then succeed
    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="retry_success")
    mock_run.side_effect = [Exception("Error 1"), Exception("Error 2"), mock_result]

    result = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-4",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=3,
    )

    assert result.field == "retry_success"
    assert mock_run.call_count == 3


@pytest.mark.anyio
async def test_structured_agent_call_timeout(mocker):
    mock_client = MagicMock()
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mocker.patch("ankigen.llm_interface.Agent")
    mocker.patch("asyncio.sleep", AsyncMock())

    mocker.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())

    with pytest.raises(asyncio.TimeoutError):
        await structured_agent_call(
            openai_client=mock_client,
            model="gpt-4",
            instructions="instr",
            user_input="input",
            output_type=MockOutput,
            retry_attempts=1,
            timeout=0.1,
        )


@pytest.mark.anyio
async def test_structured_output_completion_success(mocker):
    mock_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    mocker.patch(
        "ankigen.llm_interface.structured_agent_call",
        AsyncMock(return_value=GenericJsonOutput(test="data")),
    )

    result = await structured_output_completion(
        openai_client=mock_client,
        model="gpt-4",
        response_format={},
        system_prompt="sys",
        user_prompt="user",
        cache=mock_cache,
    )

    assert result == {"test": "data"}


@pytest.mark.anyio
async def test_openai_rate_limiter_no_wait(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep", AsyncMock())

    await limiter.wait_if_needed(50)
    assert limiter.tokens_used_current_window == 50
    mock_sleep.assert_not_called()


@pytest.mark.anyio
async def test_openai_rate_limiter_wait(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep", AsyncMock())
    mock_monotonic = mocker.patch("time.monotonic")

    # Start window at 1000.0
    mock_monotonic.return_value = 1000.0
    limiter.current_window_start_time = 1000.0

    # Use 90 tokens
    await limiter.wait_if_needed(90)

    # Next request of 20 tokens should wait because 90 + 20 > 100
    # Current time still 1000.0
    await limiter.wait_if_needed(20)

    mock_sleep.assert_called_once()
    # Wait time should be 1000.0 + 60.0 - 1000.0 = 60.0
    args, _ = mock_sleep.call_args
    assert args[0] == pytest.approx(60.0)
