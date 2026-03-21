import pytest
from unittest.mock import MagicMock, AsyncMock
from ankigen.llm_interface import (
    OpenAIClientManager,
    OpenAIRateLimiter,
    structured_agent_call,
)
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel


@pytest.fixture
def client_manager():
    return OpenAIClientManager()


@pytest.mark.anyio
async def test_client_manager_initialize_success(client_manager, mocker):
    mocker.patch("ankigen.llm_interface.AsyncOpenAI", return_value=AsyncMock())
    await client_manager.initialize_client("sk-test-key")
    assert client_manager.get_client() is not None


@pytest.mark.anyio
async def test_client_manager_initialize_invalid_key(client_manager):
    with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
        await client_manager.initialize_client("invalid-key")


@pytest.mark.anyio
async def test_client_manager_initialize_failure(client_manager, mocker):
    mocker.patch(
        "ankigen.llm_interface.AsyncOpenAI", side_effect=OpenAIError("Connection error")
    )
    with pytest.raises(OpenAIError):
        await client_manager.initialize_client("sk-test-key")
    with pytest.raises(RuntimeError, match="client is not initialized"):
        client_manager.get_client()


@pytest.mark.anyio
async def test_rate_limiter_wait_if_needed(mocker):
    limiter = OpenAIRateLimiter(tokens_per_minute=100)
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # First request - should not wait
    await limiter.wait_if_needed(50)
    mock_sleep.assert_not_called()
    assert limiter.tokens_used_current_window == 50

    # Second request - should wait
    await limiter.wait_if_needed(60)
    mock_sleep.assert_called_once()
    # After wait and reset, it adds the tokens for the current request
    assert limiter.tokens_used_current_window == 60


class SampleOutputModel(BaseModel):
    name: str


@pytest.mark.anyio
async def test_structured_agent_call(mocker):
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_agent = MagicMock()
    mock_runner = mocker.patch("agents.Runner.run", new_callable=AsyncMock)

    mock_result = MagicMock()
    mock_result.final_output = SampleOutputModel(name="test")
    mock_runner.return_value = mock_result

    mocker.patch("agents.Agent", return_value=mock_agent)
    mocker.patch("agents.set_default_openai_client")

    output = await structured_agent_call(
        openai_client=mock_client,
        model="gpt-5.2",
        instructions="inst",
        user_input="input",
        output_type=SampleOutputModel,
    )

    assert output.name == "test"
    mock_runner.assert_called_once()
