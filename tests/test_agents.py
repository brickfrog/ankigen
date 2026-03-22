import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ankigen.agents.generators import (
    card_dict_to_card,
    SubjectExpertAgent,
    QualityReviewAgent,
)
from ankigen.agents.base import parse_agent_json_response, BaseAgentWrapper, AgentConfig
from ankigen.agents.token_tracker import TokenTracker
from ankigen.models import Card


@pytest.fixture
def mock_openai_client():
    return AsyncMock()


def test_card_dict_to_card_basic():
    data = {
        "card_type": "basic",
        "front": {"question": "q"},
        "back": {"answer": "a", "explanation": "e", "example": "x"},
    }
    card = card_dict_to_card(data, "t", "s")
    assert card.card_type == "basic"
    assert card.front.question == "q"


def test_card_dict_to_card_cloze():
    data = {
        "card_type": "cloze",
        "front": {"question": "q {{c1::a}}"},
        "back": {"answer": "a", "explanation": "e", "example": "x"},
    }
    card = card_dict_to_card(data, "t", "s")
    assert card.card_type == "cloze"


def test_card_dict_to_card_invalid_front():
    with pytest.raises(ValueError):
        card_dict_to_card({"front": {}}, "t", "s")


def test_card_dict_to_card_invalid_back():
    with pytest.raises(ValueError):
        card_dict_to_card({"front": {"question": "q"}, "back": {}}, "t", "s")


def test_parse_agent_json_response_raw():
    assert parse_agent_json_response({"k": "v"}) == {"k": "v"}


def test_parse_agent_json_response_markdown():
    md = '```json\n{"k": "v"}\n```'
    assert parse_agent_json_response(md) == {"k": "v"}


def test_base_agent_wrapper_init(mock_openai_client):
    config = AgentConfig(name="test", instructions="instr")
    wrapper = BaseAgentWrapper(config, mock_openai_client)
    assert wrapper.config.name == "test"


def test_base_agent_wrapper_enhance_input(mock_openai_client):
    wrapper = BaseAgentWrapper(
        AgentConfig(name="t", instructions="i"), mock_openai_client
    )
    enhanced = wrapper._enhance_input_with_context("in", {"k": "v"})
    assert "in" in enhanced and "k: v" in enhanced


def test_subject_expert_agent_init(mock_openai_client, mocker):
    mock_mgr = MagicMock()
    mock_mgr.get_agent_config.return_value = AgentConfig(
        name="subject_expert", instructions="i"
    )
    mocker.patch("ankigen.agents.generators.get_config_manager", return_value=mock_mgr)
    agent = SubjectExpertAgent(mock_openai_client)
    assert agent.openai_client == mock_openai_client


def test_subject_expert_agent_build_batch_prompt(mock_openai_client, mocker):
    mock_mgr = MagicMock()
    mock_mgr.get_agent_config.return_value = AgentConfig(
        name="subject_expert", instructions="i"
    )
    mocker.patch("ankigen.agents.generators.get_config_manager", return_value=mock_mgr)
    agent = SubjectExpertAgent(mock_openai_client)
    prompt = agent._build_batch_prompt("T", 5, 1, {"generate_cloze": True}, [])
    assert "cloze" in prompt


@pytest.mark.anyio
async def test_quality_review_agent_review_success(mock_openai_client, mocker):
    agent = QualityReviewAgent(mock_openai_client, model="gpt-4")
    card = Card(
        card_type="basic",
        front={"question": "q"},
        back={"answer": "a", "explanation": "e", "example": "x"},
    )
    # Use patch.object with AsyncMock
    with patch.object(agent, "execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = (
            {"approved": True, "reason": "ok", "revised_card": None},
            {},
        )
        revised, approved, reason = await agent.review_card(card)
        assert approved is True
        assert revised == card


def test_token_tracker_track_usage():
    tracker = TokenTracker()
    tracker.track_usage(10, 20, "m")
    assert tracker.total_tokens == 30


def test_token_tracker_count_tokens_for_text():
    tracker = TokenTracker()
    assert tracker.count_tokens_for_text("hello", "gpt-4") > 0


def test_token_tracker_summary():
    tracker = TokenTracker()
    tracker.track_usage(10, 20, "m", actual_cost=0.1)
    assert tracker.get_session_summary()["total_cost"] == 0.1
