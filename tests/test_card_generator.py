import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from ankigen_core.card_generator import (
    _parse_model_selection,
    _map_generation_mode_to_subject,
    _build_generation_context,
    _get_token_usage_html,
    _format_cards_to_dataframe,
    format_cards_for_dataframe,
    get_dataframe_columns,
    generate_token_usage_html,
    orchestrate_card_generation,
)
from ankigen_core.models import Card, CardFront, CardBack


def test_parse_model_selection():
    # Test defaults
    assert _parse_model_selection("") == ("gpt-5.2", None)
    assert _parse_model_selection(None) == ("gpt-5.2", None)

    # Test explicit UI values
    assert _parse_model_selection("gpt-5.2-auto") == ("gpt-5.2", None)
    assert _parse_model_selection("gpt-5.2-instant") == ("gpt-5.2", "none")
    assert _parse_model_selection("gpt-5.2-thinking") == ("gpt-5.2", "high")

    # Test variations
    assert _parse_model_selection("GPT-5.2-INSTANT") == ("gpt-5.2", "none")
    assert _parse_model_selection("gpt-5.2-some-other") == ("gpt-5.2", None)

    # Test direct model names
    assert _parse_model_selection("gpt-4o") == ("gpt-4o", None)


def test_map_generation_mode_to_subject():
    assert _map_generation_mode_to_subject("subject", "Python") == "Python"
    assert _map_generation_mode_to_subject("subject", "") == "general"
    assert _map_generation_mode_to_subject("path", "Any") == "curriculum_design"
    assert _map_generation_mode_to_subject("text", "Any") == "content_analysis"
    assert _map_generation_mode_to_subject("unknown", "Any") == "general"


def test_build_generation_context():
    # mode text with source_text
    context = _build_generation_context("text", "Some source text")
    assert context == {"source_text": "Some source text"}

    # mode text without source_text
    context = _build_generation_context("text", "")
    assert context == {}

    # other mode
    context = _build_generation_context("subject", "Some source text")
    assert context == {}


def test_get_dataframe_columns():
    cols = get_dataframe_columns()
    assert isinstance(cols, list)
    assert "Question" in cols
    assert "Answer" in cols
    assert len(cols) == 11


def test_format_cards_for_dataframe():
    front = CardFront(question="Q1")
    back = CardBack(answer="A1", explanation="E1", example="Ex1")
    card1 = Card(
        front=front,
        back=back,
        metadata={"difficulty": "easy", "source_url": "http://test.com"},
    )

    front2 = CardFront(question="Q2")
    back2 = CardBack(answer="A2", explanation="E2", example="Ex2")
    card2 = Card(
        front=front2,
        back=back2,
        card_type="cloze",
        metadata={"prerequisites": ["P1", "P2"]},
    )

    cards = [card1, card2]
    formatted = format_cards_for_dataframe(cards, "Test Topic")

    assert len(formatted) == 2
    assert formatted[0]["Index"] == "1"
    assert formatted[0]["Topic"] == "Test Topic"
    assert formatted[0]["Question"] == "Q1"
    assert formatted[0]["Difficulty"] == "easy"
    assert formatted[0]["Source_URL"] == "http://test.com"

    assert formatted[1]["Index"] == "2"
    assert formatted[1]["Card_Type"] == "cloze"
    assert formatted[1]["Prerequisites"] == "P1, P2"


def test_generate_token_usage_html():
    # Valid usage
    usage = {"total_tokens": 100}
    html = generate_token_usage_html(usage)
    assert "100 tokens" in html

    # None usage
    html = generate_token_usage_html(None)
    assert "No usage data" in html

    # Invalid usage type
    html = generate_token_usage_html("invalid")
    assert "No usage data" in html


def test_get_token_usage_html_mock():
    # Test with object having get_session_summary
    tracker_summary = MagicMock()
    tracker_summary.get_session_summary.return_value = {"total_tokens": 50}
    assert "50 tokens" in _get_token_usage_html(tracker_summary)

    # Test with object having get_session_usage
    tracker_usage = MagicMock()
    del tracker_usage.get_session_summary
    tracker_usage.get_session_usage.return_value = {"total_tokens": 75}
    assert "75 tokens" in _get_token_usage_html(tracker_usage)

    # Test failure
    tracker_fail = MagicMock()
    del tracker_fail.get_session_summary
    del tracker_fail.get_session_usage
    assert "No usage data" in _get_token_usage_html(tracker_fail)


def test_format_cards_to_dataframe():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
    )
    df, msg = _format_cards_to_dataframe([card], "Python")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Cards Generated" in msg
    assert "1" in msg


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@patch("ankigen_core.card_generator.AgentOrchestrator")
@patch("ankigen_core.agents.token_tracker.get_token_tracker")
async def test_orchestrate_card_generation_success(
    mock_get_tracker, mock_orchestrator_class, anyio_backend
):
    # Mock token tracker
    mock_tracker = MagicMock()
    mock_tracker.get_session_summary.return_value = {"total_tokens": 123}
    mock_get_tracker.return_value = mock_tracker

    # Mock orchestrator
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.initialize = AsyncMock()

    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
    )
    mock_orchestrator.generate_cards_with_agents = AsyncMock(return_value=([card], {}))

    client_manager = MagicMock()
    cache = MagicMock()

    df, msg, usage_html = await orchestrate_card_generation(
        client_manager,
        cache,
        "api-key",
        "Python",
        "subject",
        "",
        "",
        "gpt-5.2-auto",
        1,
        1,
        "",
        False,
    )

    assert len(df) == 1
    assert "1" in msg
    assert "123 tokens" in usage_html
    mock_orchestrator.initialize.assert_called_once()
    mock_orchestrator.generate_cards_with_agents.assert_called_once()


@pytest.mark.anyio
@patch("ankigen_core.card_generator.AgentOrchestrator")
@patch("ankigen_core.agents.token_tracker.get_token_tracker")
async def test_orchestrate_card_generation_no_cards(
    mock_get_tracker, mock_orchestrator_class, anyio_backend
):
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.initialize = AsyncMock()
    mock_orchestrator.generate_cards_with_agents = AsyncMock(return_value=([], {}))

    df, msg, usage_html = await orchestrate_card_generation(
        MagicMock(),
        MagicMock(),
        "api-key",
        "Python",
        "subject",
        "",
        "",
        "gpt-5.2-auto",
        1,
        1,
        "",
        False,
    )

    assert len(df) == 0
    assert "returned no cards" in msg


@pytest.mark.anyio
@patch("ankigen_core.card_generator.AgentOrchestrator")
async def test_orchestrate_card_generation_error(
    mock_orchestrator_class, anyio_backend
):
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.initialize = AsyncMock(side_effect=Exception("Test Error"))

    df, msg, usage_html = await orchestrate_card_generation(
        MagicMock(),
        MagicMock(),
        "api-key",
        "Python",
        "subject",
        "",
        "",
        "gpt-5.2-auto",
        1,
        1,
        "",
        False,
    )

    assert len(df) == 0
    assert "Test Error" in msg
