import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

from ankigen.llm_interface import (
    OpenAIClientManager,
    OpenAIRateLimiter,
    structured_agent_call,
    GenericJsonOutput,
)

# --- OpenAIClientManager Tests ---


@pytest.mark.anyio
async def test_client_manager_init():
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


@pytest.mark.anyio
async def test_client_manager_initialize_valid():
    manager = OpenAIClientManager()
    with patch("ankigen.llm_interface.AsyncOpenAI") as mock_openai:
        await manager.initialize_client("sk-validkey")
        assert manager._api_key == "sk-validkey"
        assert manager._client is not None
        mock_openai.assert_called_once_with(api_key="sk-validkey")


@pytest.mark.anyio
async def test_client_manager_initialize_invalid():
    manager = OpenAIClientManager()
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await manager.initialize_client("invalid-key")


@pytest.mark.anyio
async def test_client_manager_get_client_fail():
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="AsyncOpenAI client is not initialized"):
        manager.get_client()


@pytest.mark.anyio
async def test_client_manager_aclose():
    manager = OpenAIClientManager()
    mock_client = AsyncMock()
    manager._client = mock_client
    await manager.aclose()
    mock_client.aclose.assert_awaited_once()
    assert manager._client is None


# --- OpenAIRateLimiter Tests ---


@pytest.mark.anyio
async def test_rate_limiter_init():
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    assert limiter.tokens_per_minute_limit == 100
    assert limiter.tokens_used_current_window == 0


@pytest.mark.anyio
async def test_rate_limiter_within_limit():
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    # Should not sleep
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.wait_if_needed(50)
        assert limiter.tokens_used_current_window == 50
        mock_sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_rate_limiter_exceed_limit():
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    limiter.tokens_used_current_window = 80

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with patch("time.monotonic", return_value=time.monotonic()):
            # Requesting 30 more tokens would exceed 100
            await limiter.wait_if_needed(30)
            mock_sleep.assert_awaited_once()
            assert limiter.tokens_used_current_window == 30  # Reset after wait


# --- GenericJsonOutput Tests ---


def test_generic_json_output():
    data = {"field1": "val1", "field2": 2}
    output = GenericJsonOutput(**data)
    assert output.field1 == "val1"
    assert output.field2 == 2


# --- structured_agent_call Mock Tests ---


@pytest.mark.anyio
async def test_structured_agent_call_simple():
    mock_client = AsyncMock()
    from pydantic import BaseModel

    class MockOutput(BaseModel):
        result: str

    mock_result = MagicMock()
    mock_result.final_output = MockOutput(result="success")

    with patch("ankigen.llm_interface.set_default_openai_client"):
        with patch("ankigen.llm_interface.Agent"):
            with patch(
                "ankigen.llm_interface.Runner.run",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                result = await structured_agent_call(
                    openai_client=mock_client,
                    model="gpt-5.2",
                    instructions="test",
                    user_input="hello",
                    output_type=MockOutput,
                )
                assert result.result == "success"


@pytest.mark.anyio
async def test_structured_agent_call_timeout():
    mock_client = AsyncMock()
    from pydantic import BaseModel

    class MockOutput(BaseModel):
        result: str

    with patch("ankigen.llm_interface.set_default_openai_client"):
        with patch("ankigen.llm_interface.Agent"):
            with patch(
                "ankigen.llm_interface.Runner.run", side_effect=asyncio.TimeoutError
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with pytest.raises(asyncio.TimeoutError):
                        await structured_agent_call(
                            openai_client=mock_client,
                            model="gpt-5.2",
                            instructions="test",
                            user_input="hello",
                            output_type=MockOutput,
                            retry_attempts=2,
                        )
