# Tests for ankigen_core/agents/base.py

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from ankigen_core.agents.base import AgentConfig, BaseAgentWrapper, AgentResponse


# Test AgentConfig
def test_agent_config_creation():
    """Test basic AgentConfig creation"""
    config = AgentConfig(
        name="test_agent",
        instructions="Test instructions",
        model="gpt-4o",
        temperature=0.7,
    )

    assert config.name == "test_agent"
    assert config.instructions == "Test instructions"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.7
    assert config.custom_prompts == {}


def test_agent_config_defaults():
    """Test AgentConfig with default values"""
    config = AgentConfig(name="test_agent", instructions="Test instructions")

    assert config.model == "gpt-4o"
    assert config.temperature == 0.7
    assert config.max_tokens is None
    assert config.timeout == 30.0
    assert config.retry_attempts == 3
    assert config.enable_tracing is True
    assert config.custom_prompts == {}


def test_agent_config_custom_prompts():
    """Test AgentConfig with custom prompts"""
    custom_prompts = {"greeting": "Hello there", "farewell": "Goodbye"}
    config = AgentConfig(
        name="test_agent",
        instructions="Test instructions",
        custom_prompts=custom_prompts,
    )

    assert config.custom_prompts == custom_prompts


# Test BaseAgentWrapper
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    return MagicMock()


@pytest.fixture
def test_agent_config():
    """Sample agent config for testing"""
    return AgentConfig(
        name="test_agent",
        instructions="Test instructions",
        model="gpt-4o",
        temperature=0.7,
        timeout=10.0,
        retry_attempts=2,
    )


@pytest.fixture
def base_agent_wrapper(test_agent_config, mock_openai_client):
    """Base agent wrapper for testing"""
    return BaseAgentWrapper(test_agent_config, mock_openai_client)


def test_base_agent_wrapper_init(
    base_agent_wrapper, test_agent_config, mock_openai_client
):
    """Test BaseAgentWrapper initialization"""
    assert base_agent_wrapper.config == test_agent_config
    assert base_agent_wrapper.openai_client == mock_openai_client
    assert base_agent_wrapper.agent is None
    assert base_agent_wrapper.runner is None
    assert base_agent_wrapper._performance_metrics == {
        "total_calls": 0,
        "successful_calls": 0,
        "average_response_time": 0.0,
        "error_count": 0,
    }


@patch("ankigen_core.agents.base.Agent")
@patch("ankigen_core.agents.base.Runner")
async def test_base_agent_wrapper_initialize(
    mock_runner, mock_agent, base_agent_wrapper
):
    """Test agent initialization"""
    mock_agent_instance = MagicMock()
    mock_runner_instance = MagicMock()
    mock_agent.return_value = mock_agent_instance
    mock_runner.return_value = mock_runner_instance

    await base_agent_wrapper.initialize()

    mock_agent.assert_called_once_with(
        name="test_agent",
        instructions="Test instructions",
        model="gpt-4o",
        temperature=0.7,
    )
    mock_runner.assert_called_once_with(
        agent=mock_agent_instance, client=base_agent_wrapper.openai_client
    )
    assert base_agent_wrapper.agent == mock_agent_instance
    assert base_agent_wrapper.runner == mock_runner_instance


@patch("ankigen_core.agents.base.Agent")
@patch("ankigen_core.agents.base.Runner")
async def test_base_agent_wrapper_initialize_error(
    mock_runner, mock_agent, base_agent_wrapper
):
    """Test agent initialization with error"""
    mock_agent.side_effect = Exception("Agent creation failed")

    with pytest.raises(Exception, match="Agent creation failed"):
        await base_agent_wrapper.initialize()

    assert base_agent_wrapper.agent is None
    assert base_agent_wrapper.runner is None


async def test_base_agent_wrapper_execute_without_initialization(base_agent_wrapper):
    """Test execute method when agent isn't initialized"""
    with patch.object(base_agent_wrapper, "initialize") as mock_init:
        with patch.object(base_agent_wrapper, "_run_agent") as mock_run:
            mock_run.return_value = "test response"

            result = await base_agent_wrapper.execute("test input")

            mock_init.assert_called_once()
            mock_run.assert_called_once_with("test input")
            assert result == "test response"


async def test_base_agent_wrapper_execute_with_context(base_agent_wrapper):
    """Test execute method with context"""
    base_agent_wrapper.runner = MagicMock()

    with patch.object(base_agent_wrapper, "_run_agent") as mock_run:
        mock_run.return_value = "test response"

        context = {"key1": "value1", "key2": "value2"}
        result = await base_agent_wrapper.execute("test input", context)

        expected_input = "test input\n\nContext:\nkey1: value1\nkey2: value2"
        mock_run.assert_called_once_with(expected_input)
        assert result == "test response"


async def test_base_agent_wrapper_execute_timeout(base_agent_wrapper):
    """Test execute method with timeout"""
    base_agent_wrapper.runner = MagicMock()

    with patch.object(base_agent_wrapper, "_run_agent") as mock_run:
        mock_run.side_effect = asyncio.TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            await base_agent_wrapper.execute("test input")

        assert base_agent_wrapper._performance_metrics["error_count"] == 1


async def test_base_agent_wrapper_execute_exception(base_agent_wrapper):
    """Test execute method with exception"""
    base_agent_wrapper.runner = MagicMock()

    with patch.object(base_agent_wrapper, "_run_agent") as mock_run:
        mock_run.side_effect = Exception("Execution failed")

        with pytest.raises(Exception, match="Execution failed"):
            await base_agent_wrapper.execute("test input")

        assert base_agent_wrapper._performance_metrics["error_count"] == 1


async def test_base_agent_wrapper_run_agent_success(base_agent_wrapper):
    """Test _run_agent method with successful execution"""
    mock_runner = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "run_123"
    mock_run.status = "completed"
    mock_run.thread_id = "thread_456"

    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = "test response"

    mock_runner.create_run = AsyncMock(return_value=mock_run)
    mock_runner.get_run = AsyncMock(return_value=mock_run)
    mock_runner.get_messages = AsyncMock(return_value=[mock_message])

    base_agent_wrapper.runner = mock_runner

    result = await base_agent_wrapper._run_agent("test input")

    mock_runner.create_run.assert_called_once_with(
        messages=[{"role": "user", "content": "test input"}]
    )
    mock_runner.get_messages.assert_called_once_with("thread_456")
    assert result == "test response"


async def test_base_agent_wrapper_run_agent_retry(base_agent_wrapper):
    """Test _run_agent method with retry logic"""
    mock_runner = MagicMock()
    mock_runner.create_run = AsyncMock(
        side_effect=[
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
        ]
    )

    base_agent_wrapper.runner = mock_runner

    with pytest.raises(Exception, match="Second attempt failed"):
        await base_agent_wrapper._run_agent("test input")

    assert mock_runner.create_run.call_count == 2


async def test_base_agent_wrapper_run_agent_no_response(base_agent_wrapper):
    """Test _run_agent method when no assistant response is found"""
    mock_runner = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "run_123"
    mock_run.status = "completed"
    mock_run.thread_id = "thread_456"

    mock_message = MagicMock()
    mock_message.role = "user"  # No assistant response
    mock_message.content = "user message"

    mock_runner.create_run = AsyncMock(return_value=mock_run)
    mock_runner.get_run = AsyncMock(return_value=mock_run)
    mock_runner.get_messages = AsyncMock(return_value=[mock_message])

    base_agent_wrapper.runner = mock_runner

    with pytest.raises(ValueError, match="No assistant response found"):
        await base_agent_wrapper._run_agent("test input")


def test_base_agent_wrapper_update_performance_metrics(base_agent_wrapper):
    """Test performance metrics update"""
    base_agent_wrapper._update_performance_metrics(1.5, success=True)

    metrics = base_agent_wrapper._performance_metrics
    assert metrics["successful_calls"] == 1
    assert metrics["average_response_time"] == 1.5

    # Add another successful call
    base_agent_wrapper._update_performance_metrics(2.5, success=True)
    metrics = base_agent_wrapper._performance_metrics
    assert metrics["successful_calls"] == 2
    assert metrics["average_response_time"] == 2.0  # (1.5 + 2.5) / 2


def test_base_agent_wrapper_get_performance_metrics(base_agent_wrapper):
    """Test getting performance metrics"""
    base_agent_wrapper._performance_metrics = {
        "total_calls": 10,
        "successful_calls": 8,
        "average_response_time": 1.2,
        "error_count": 2,
    }

    metrics = base_agent_wrapper.get_performance_metrics()

    assert metrics["total_calls"] == 10
    assert metrics["successful_calls"] == 8
    assert metrics["average_response_time"] == 1.2
    assert metrics["error_count"] == 2
    assert metrics["success_rate"] == 0.8
    assert metrics["agent_name"] == "test_agent"


async def test_base_agent_wrapper_handoff_to(base_agent_wrapper):
    """Test handoff to another agent"""
    target_agent = MagicMock()
    target_agent.config.name = "target_agent"
    target_agent.execute = AsyncMock(return_value="handoff result")

    context = {
        "reason": "Test handoff",
        "user_input": "Continue with this",
        "additional_data": "some data",
    }

    result = await base_agent_wrapper.handoff_to(target_agent, context)

    expected_context = {
        "from_agent": "test_agent",
        "handoff_reason": "Test handoff",
        "user_input": "Continue with this",
        "additional_data": "some data",
    }

    target_agent.execute.assert_called_once_with("Continue with this", expected_context)
    assert result == "handoff result"


async def test_base_agent_wrapper_handoff_to_default_input(base_agent_wrapper):
    """Test handoff to another agent with default input"""
    target_agent = MagicMock()
    target_agent.config.name = "target_agent"
    target_agent.execute = AsyncMock(return_value="handoff result")

    context = {"reason": "Test handoff"}

    result = await base_agent_wrapper.handoff_to(target_agent, context)

    expected_context = {
        "from_agent": "test_agent",
        "handoff_reason": "Test handoff",
        "reason": "Test handoff",
    }

    target_agent.execute.assert_called_once_with(
        "Continue processing", expected_context
    )
    assert result == "handoff result"


# Test AgentResponse
def test_agent_response_creation():
    """Test AgentResponse creation"""
    response = AgentResponse(
        success=True,
        data={"cards": []},
        agent_name="test_agent",
        execution_time=1.5,
        metadata={"version": "1.0"},
        errors=["minor warning"],
    )

    assert response.success is True
    assert response.data == {"cards": []}
    assert response.agent_name == "test_agent"
    assert response.execution_time == 1.5
    assert response.metadata == {"version": "1.0"}
    assert response.errors == ["minor warning"]


def test_agent_response_defaults():
    """Test AgentResponse with default values"""
    response = AgentResponse(
        success=True,
        data={"result": "success"},
        agent_name="test_agent",
        execution_time=1.0,
    )

    assert response.metadata == {}
    assert response.errors == []
