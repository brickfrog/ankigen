import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from openai import AsyncOpenAI
from pydantic import BaseModel

from ankigen.llm_interface import (
    OpenAIClientManager,
    structured_agent_call,
    OpenAIRateLimiter,
    GenericJsonOutput,
)
from ankigen.utils import ResponseCache


class MockOutput(BaseModel):
    field: str


@pytest.fixture
def response_cache():
    return ResponseCache(maxsize=10)


@pytest.mark.anyio
async def test_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


@pytest.mark.anyio
async def test_initialize_client_success(mocker):
    manager = OpenAIClientManager()
    mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    await manager.initialize_client("sk-valid-key")
    assert manager._api_key == "sk-valid-key"
    assert manager._client is not None


@pytest.mark.anyio
async def test_initialize_client_invalid_key():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError):
        await manager.initialize_client("invalid-key")


@pytest.mark.anyio
async def test_get_client_not_initialized():
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError):
        manager.get_client()


@pytest.mark.anyio
async def test_get_client_success(mocker):
    manager = OpenAIClientManager()
    mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    await manager.initialize_client("sk-valid")
    assert manager.get_client() is not None


@pytest.mark.anyio
async def test_client_context_manager(mocker):
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    mock_instance = mock_openai.return_value
    with OpenAIClientManager() as manager:
        await manager.initialize_client("sk-valid")
        assert manager.get_client() is mock_instance
    assert manager._client is None


@pytest.mark.anyio
async def test_client_async_context_manager(mocker):
    mock_openai = mocker.patch("ankigen.llm_interface.AsyncOpenAI")
    mock_instance = mock_openai.return_value
    mock_instance.aclose = AsyncMock()
    async with OpenAIClientManager() as manager:
        await manager.initialize_client("sk-valid")
        assert manager.get_client() is mock_instance
    assert manager._client is None


@pytest.mark.anyio
async def test_structured_agent_call_success(mocker):
    openai_client = AsyncMock(spec=AsyncOpenAI)
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mock_result = MagicMock()
    mock_result.final_output = MockOutput(field="value")
    mock_runner.return_value = mock_result

    result = await structured_agent_call(
        openai_client=openai_client,
        model="gpt-5.2",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
    )
    assert result.field == "value"


@pytest.mark.anyio
async def test_structured_agent_call_cache_hit(mocker, response_cache):
    openai_client = AsyncMock(spec=AsyncOpenAI)
    response_cache.set("input", "gpt-5.2", {"field": "cached"})
    mock_runner = mocker.patch("ankigen.llm_interface.Runner.run")

    result = await structured_agent_call(
        openai_client=openai_client,
        model="gpt-5.2",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        cache=response_cache,
        cache_key="input",
    )
    assert result.field == "cached"
    mock_runner.assert_not_called()


@pytest.mark.anyio
async def test_structured_agent_call_retry_on_timeout(mocker):
    openai_client = AsyncMock(spec=AsyncOpenAI)
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mock_runner.side_effect = [
        asyncio.TimeoutError(),
        MagicMock(final_output=MockOutput(field="ok")),
    ]
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await structured_agent_call(
        openai_client=openai_client,
        model="gpt-5.2",
        instructions="instr",
        user_input="input",
        output_type=MockOutput,
        retry_attempts=2,
    )
    assert result.field == "ok"


@pytest.mark.anyio
async def test_structured_agent_call_failure_after_retries(mocker):
    openai_client = AsyncMock(spec=AsyncOpenAI)
    mocker.patch("ankigen.llm_interface.set_default_openai_client")
    mock_runner = mocker.patch(
        "ankigen.llm_interface.Runner.run", new_callable=AsyncMock
    )
    mock_runner.side_effect = Exception("Permanent failure")
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    with pytest.raises(Exception):
        await structured_agent_call(
            openai_client=openai_client,
            model="gpt-5.2",
            instructions="instr",
            user_input="input",
            output_type=MockOutput,
            retry_attempts=2,
        )


@pytest.mark.anyio
async def test_rate_limiter_wait_no_delay(mocker):
    # Use a side effect that won't exhaust
    mocker.patch("time.monotonic", side_effect=lambda: 100.0)
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    await limiter.wait_if_needed(50)
    assert limiter.tokens_used_current_window == 50
    mock_sleep.assert_not_called()


@pytest.mark.anyio
async def test_rate_limiter_wait_with_delay(mocker):
    # Controlled time jumps
    time_values = [100.0, 100.1, 100.2, 160.3, 160.4, 160.5, 160.6, 160.7]
    idx = 0

    def get_time():
        nonlocal idx
        val = time_values[min(idx, len(time_values) - 1)]
        idx += 1
        return val

    mocker.patch("time.monotonic", side_effect=get_time)
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    await limiter.wait_if_needed(90)
    await limiter.wait_if_needed(20)

    assert mock_sleep.called
    assert limiter.tokens_used_current_window == 20


def test_generic_json_output():
    obj = GenericJsonOutput(any_field="value")
    assert obj.model_extra["any_field"] == "value"
