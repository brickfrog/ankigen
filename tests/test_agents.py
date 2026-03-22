import pytest
from unittest.mock import MagicMock
from ankigen.agents.token_tracker import (
    TokenTracker,
    get_token_tracker,
    track_agent_usage,
    TokenUsage,
)
from ankigen.agents.generators import card_dict_to_card
from ankigen.models import Card

# --- TokenTracker Tests ---


def test_track_usage_basic():
    tracker = TokenTracker()
    usage = tracker.track_usage(10, 20, "gpt-4")

    assert isinstance(usage, TokenUsage)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30
    assert usage.model == "gpt-4"
    assert tracker.total_tokens == 30
    assert len(tracker.usage_history) == 1


def test_track_usage_with_cost():
    tracker = TokenTracker()
    cost = 0.05
    usage = tracker.track_usage(100, 200, "gpt-4", actual_cost=cost)

    assert usage.estimated_cost == cost
    assert tracker.total_cost == cost
    assert tracker.total_tokens == 300


def test_get_session_summary_empty():
    tracker = TokenTracker()
    summary = tracker.get_session_summary()

    assert summary["total_requests"] == 0
    assert summary["total_tokens"] == 0
    assert summary["total_cost"] == 0.0
    assert summary["by_model"] == {}


def test_get_session_summary_with_data():
    tracker = TokenTracker()
    tracker.track_usage(10, 20, "model-a", actual_cost=0.1)
    tracker.track_usage(30, 40, "model-a", actual_cost=0.2)
    tracker.track_usage(5, 5, "model-b", actual_cost=0.05)

    summary = tracker.get_session_summary()

    assert summary["total_requests"] == 3
    assert summary["total_tokens"] == 110
    assert pytest.approx(summary["total_cost"]) == 0.35

    assert "model-a" in summary["by_model"]
    assert summary["by_model"]["model-a"]["requests"] == 2
    assert summary["by_model"]["model-a"]["tokens"] == 100
    assert pytest.approx(summary["by_model"]["model-a"]["cost"]) == 0.3

    assert "model-b" in summary["by_model"]
    assert summary["by_model"]["model-b"]["requests"] == 1
    assert summary["by_model"]["model-b"]["tokens"] == 10
    assert pytest.approx(summary["by_model"]["model-b"]["cost"]) == 0.05


def test_reset_session():
    tracker = TokenTracker()
    tracker.track_usage(10, 20, "gpt-4", actual_cost=0.1)
    tracker.reset_session()

    assert tracker.total_tokens == 0
    assert tracker.total_cost == 0.0
    assert len(tracker.usage_history) == 0


def test_count_tokens_for_text():
    tracker = TokenTracker()
    # Non-empty text should return positive int
    count = tracker.count_tokens_for_text("Hello world", "gpt-4")
    assert isinstance(count, int)
    assert count > 0

    # Unknown model should fallback and still work
    count_unknown = tracker.count_tokens_for_text("Hello world", "unknown-model")
    assert isinstance(count_unknown, int)
    assert count_unknown > 0


def test_count_tokens_for_messages():
    tracker = TokenTracker()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    count = tracker.count_tokens_for_messages(messages, "gpt-4")
    assert isinstance(count, int)
    assert count > 0


def test_track_usage_from_agents_sdk():
    tracker = TokenTracker()
    usage_dict = {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75}
    usage = tracker.track_usage_from_agents_sdk(usage_dict, "gpt-4")

    assert usage is not None
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 75
    assert tracker.total_tokens == 75


def test_track_usage_from_agents_sdk_empty():
    tracker = TokenTracker()

    # Empty dict
    assert tracker.track_usage_from_agents_sdk({}, "gpt-4") is None

    # Zero tokens
    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert tracker.track_usage_from_agents_sdk(usage_dict, "gpt-4") is None


def test_track_usage_from_response():
    tracker = TokenTracker()

    # Mock response object with .usage
    response = MagicMock()
    response.usage.prompt_tokens = 40
    response.usage.completion_tokens = 60
    response.usage.total_cost = 0.02

    usage = tracker.track_usage_from_response(response, "gpt-4")

    assert usage.prompt_tokens == 40
    assert usage.completion_tokens == 60
    assert usage.estimated_cost == 0.02
    assert tracker.total_tokens == 100
    assert tracker.total_cost == 0.02


def test_track_usage_from_response_different_cost_attr():
    tracker = TokenTracker()

    # Mock response object with .usage.cost instead of .total_cost
    response = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    del response.usage.total_cost
    response.usage.cost = 0.001

    usage = tracker.track_usage_from_response(response, "gpt-4")
    assert usage.estimated_cost == 0.001


def test_track_usage_from_response_no_usage():
    tracker = TokenTracker()

    # Object without .usage
    response = object()
    assert tracker.track_usage_from_response(response, "gpt-4") is None


# --- card_dict_to_card Tests ---


def test_card_dict_to_card_basic():
    card_data = {
        "card_type": "basic",
        "front": {"question": "What is Python?"},
        "back": {
            "answer": "A programming language.",
            "explanation": "High-level, interpreted language.",
            "example": "print('Hello')",
        },
        "metadata": {"subject": "CS", "topic": "Languages"},
    }

    card = card_dict_to_card(card_data, "Default Topic", "Default Subject")

    assert isinstance(card, Card)
    assert card.card_type == "basic"
    assert card.front.question == "What is Python?"
    assert card.back.answer == "A programming language."
    assert card.metadata["subject"] == "CS"
    assert card.metadata["topic"] == "Languages"


def test_card_dict_to_card_cloze():
    card_data = {
        "card_type": "cloze",
        "front": {"question": "Python is a {{c1::programming language}}."},
        "back": {
            "answer": "programming language",
            "explanation": "Fill in the blank.",
            "example": "",
        },
    }

    card = card_dict_to_card(card_data, "Topic", "Subject")
    assert card.card_type == "cloze"


def test_card_dict_to_card_missing_front():
    # Missing 'front' key
    card_data = {"back": {"answer": "A"}}
    with pytest.raises(ValueError, match="Card front must include a question field"):
        card_dict_to_card(card_data, "T", "S")

    # 'front' is not a dict
    card_data = {"front": "What is Python?", "back": {"answer": "A"}}
    with pytest.raises(ValueError, match="Card front must include a question field"):
        card_dict_to_card(card_data, "T", "S")

    # 'front' missing 'question'
    card_data = {"front": {}, "back": {"answer": "A"}}
    with pytest.raises(ValueError, match="Card front must include a question field"):
        card_dict_to_card(card_data, "T", "S")


def test_card_dict_to_card_missing_back():
    card_data = {"front": {"question": "Q"}}
    with pytest.raises(ValueError, match="Card back must include an answer field"):
        card_dict_to_card(card_data, "T", "S")

    card_data = {"front": {"question": "Q"}, "back": {}}
    with pytest.raises(ValueError, match="Card back must include an answer field"):
        card_dict_to_card(card_data, "T", "S")


def test_card_dict_to_card_not_dict():
    with pytest.raises(ValueError, match="Card payload must be a dictionary"):
        card_dict_to_card("not a dict", "T", "S")


def test_card_dict_to_card_default_metadata():
    card_data = {"front": {"question": "Q"}, "back": {"answer": "A"}}

    # Test fallback to defaults
    card = card_dict_to_card(card_data, "Default Topic", "Default Subject")
    assert card.metadata["subject"] == "Default Subject"
    assert card.metadata["topic"] == "Default Topic"

    # Test partial metadata
    card_data["metadata"] = {"subject": "Explicit Subject"}
    card = card_dict_to_card(card_data, "Default Topic", "Default Subject")
    assert card.metadata["subject"] == "Explicit Subject"
    assert card.metadata["topic"] == "Default Topic"


# --- Module-level functions Tests ---


def test_get_token_tracker_singleton():
    t1 = get_token_tracker()
    t2 = get_token_tracker()
    assert t1 is t2
    assert isinstance(t1, TokenTracker)


def test_track_agent_usage():
    tracker = get_token_tracker()
    tracker.reset_session()

    usage = track_agent_usage("Hello", "Hi there!", "gpt-4", actual_cost=0.01)

    assert isinstance(usage, TokenUsage)
    assert usage.estimated_cost == 0.01
    assert tracker.total_cost == 0.01
    assert len(tracker.usage_history) == 1
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


def test_track_usage_from_openai_response_module():
    from ankigen.agents.token_tracker import track_usage_from_openai_response

    tracker = get_token_tracker()
    tracker.reset_session()

    # Mock response object with .usage
    response = MagicMock()
    response.usage.prompt_tokens = 5
    response.usage.completion_tokens = 5
    response.usage.total_cost = 0.005

    usage = track_usage_from_openai_response(response, "gpt-4")

    assert usage is not None
    assert usage.prompt_tokens == 5
    assert tracker.total_tokens == 10
    assert tracker.total_cost == 0.005


def test_track_usage_from_agents_sdk_module():
    from ankigen.agents.token_tracker import track_usage_from_agents_sdk

    tracker = get_token_tracker()
    tracker.reset_session()

    usage_dict = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    usage = track_usage_from_agents_sdk(usage_dict, "gpt-4")

    assert usage is not None
    assert usage.prompt_tokens == 10
    assert tracker.total_tokens == 30
