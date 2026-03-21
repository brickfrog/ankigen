from unittest.mock import MagicMock
from ankigen_core.card_generator import (
    _parse_model_selection,
    _map_generation_mode_to_subject,
    _build_generation_context,
    _get_token_usage_html,
    format_cards_for_dataframe,
    get_dataframe_columns,
    generate_token_usage_html,
)
from ankigen_core.models import Card, CardFront, CardBack

# --- Helper Tests ---


def test_parse_model_selection():
    assert _parse_model_selection("gpt-5.2-auto") == ("gpt-5.2", None)
    assert _parse_model_selection("gpt-5.2-instant") == ("gpt-5.2", "none")
    assert _parse_model_selection("gpt-5.2-thinking") == ("gpt-5.2", "high")
    assert _parse_model_selection("other-model") == ("other-model", None)
    assert _parse_model_selection("") == ("gpt-5.2", None)


def test_map_generation_mode_to_subject():
    assert _map_generation_mode_to_subject("subject", "Python") == "Python"
    assert _map_generation_mode_to_subject("subject", "") == "general"
    assert _map_generation_mode_to_subject("path", "") == "curriculum_design"
    assert _map_generation_mode_to_subject("text", "") == "content_analysis"
    assert _map_generation_mode_to_subject("unknown", "") == "general"


def test_build_generation_context():
    assert _build_generation_context("text", "some source") == {
        "source_text": "some source"
    }
    assert _build_generation_context("subject", "some source") == {}
    assert _build_generation_context("text", "") == {}


def test_get_token_usage_html():
    # Use spec to ensure hasattr(mock, "get_session_summary") is true
    mock_tracker = MagicMock(spec=["get_session_summary"])
    mock_tracker.get_session_summary.return_value = {"total_tokens": 100}

    html = _get_token_usage_html(mock_tracker)
    assert "100 tokens" in html

    # Test fallback: Use spec to ensure hasattr(mock, "get_session_summary") is false
    mock_tracker_legacy = MagicMock(spec=["get_session_usage"])
    mock_tracker_legacy.get_session_usage.return_value = {"total_tokens": 200}

    html = _get_token_usage_html(mock_tracker_legacy)
    assert "200 tokens" in html


def test_get_token_usage_html_error():
    mock_tracker = MagicMock()
    mock_tracker.get_session_summary.side_effect = Exception("error")

    html = _get_token_usage_html(mock_tracker)
    assert "No usage data" in html


# --- Formatting Tests ---


def test_format_cards_for_dataframe():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"difficulty": "beginner", "source_url": "http://x.com"},
        )
    ]
    formatted = format_cards_for_dataframe(cards, "Python")
    assert len(formatted) == 1
    assert formatted[0]["Topic"] == "Python"
    assert formatted[0]["Question"] == "Q1"
    assert formatted[0]["Difficulty"] == "beginner"
    assert formatted[0]["Source_URL"] == "http://x.com"


def test_format_cards_for_dataframe_missing_metadata():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]
    formatted = format_cards_for_dataframe(cards, "Python")
    assert formatted[0]["Difficulty"] == "N/A"
    assert formatted[0]["Source_URL"] == ""


def test_get_dataframe_columns():
    cols = get_dataframe_columns()
    assert "Question" in cols
    assert "Answer" in cols
    assert "Index" in cols


def test_generate_token_usage_html():
    assert "150 tokens" in generate_token_usage_html({"total_tokens": 150})
    assert "No usage data" in generate_token_usage_html(None)
    assert "No usage data" in generate_token_usage_html("not a dict")
