import pandas as pd
from unittest.mock import MagicMock
from ankigen.card_generator import (
    _get_token_usage_html,
    _format_cards_to_dataframe,
    get_dataframe_columns,
)
from ankigen.models import Card, CardFront, CardBack

# --- _get_token_usage_html Tests ---


def test_get_token_usage_html_with_session_summary():
    """Test with a mock token_tracker that has get_session_summary returning dict with total_tokens."""
    mock_tracker = MagicMock()
    mock_tracker.get_session_summary.return_value = {"total_tokens": 123}
    # Ensure it doesn't have get_session_usage to specifically test the first branch
    if hasattr(mock_tracker, "get_session_usage"):
        del mock_tracker.get_session_usage

    result = _get_token_usage_html(mock_tracker)
    assert "123 tokens" in result
    assert "<b>Token Usage:</b>" in result
    mock_tracker.get_session_summary.assert_called_once()


def test_get_token_usage_html_with_session_usage_fallback():
    """Test with get_session_usage fallback when get_session_summary is missing."""
    # Using MagicMock with spec to ensure only certain attributes exist
    mock_tracker = MagicMock(spec=["get_session_usage"])
    mock_tracker.get_session_usage.return_value = {"total_tokens": 456}

    result = _get_token_usage_html(mock_tracker)
    assert "456 tokens" in result
    mock_tracker.get_session_usage.assert_called_once()
    assert not hasattr(mock_tracker, "get_session_summary")


def test_get_token_usage_html_with_neither_method():
    """Test with tracker that has neither method."""
    mock_tracker = MagicMock(spec=[])

    result = _get_token_usage_html(mock_tracker)
    assert "No usage data" in result


def test_get_token_usage_html_method_raises_exception():
    """Test when the tracker method raises an exception."""
    mock_tracker = MagicMock()
    mock_tracker.get_session_summary.side_effect = Exception("Mock Error")

    result = _get_token_usage_html(mock_tracker)
    assert "No usage data" in result


def test_get_token_usage_html_missing_total_tokens_key():
    """Test when the method returns a dict but total_tokens key is missing."""
    mock_tracker = MagicMock()
    mock_tracker.get_session_summary.return_value = {"some_other_key": 0}

    result = _get_token_usage_html(mock_tracker)
    assert "No usage data" in result


def test_get_token_usage_html_none_tracker():
    """Test with None as token_tracker."""
    result = _get_token_usage_html(None)
    assert "No usage data" in result


# --- _format_cards_to_dataframe Tests ---


def test_format_cards_to_dataframe_with_cards():
    """Test with a list of Card objects and verify it returns a tuple of (DataFrame, str message)."""
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="X1"),
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="X2"),
        ),
    ]
    df, message = _format_cards_to_dataframe(cards, "Test Subject")

    assert isinstance(df, pd.DataFrame)
    assert isinstance(message, str)
    assert len(df) == 2
    assert "2" in message
    assert "Cards Generated:" in message
    assert df.iloc[0]["Question"] == "Q1"
    assert df.iloc[1]["Question"] == "Q2"
    assert df.iloc[0]["Topic"] == "Test Subject"


def test_format_cards_to_dataframe_empty_list():
    """Test with an empty card list."""
    df, message = _format_cards_to_dataframe([], "Empty")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "0" in message


def test_format_cards_to_dataframe_message_contains_correct_count():
    """Test that the message contains the correct count and HTML elements."""
    cards = [
        Card(front=CardFront(question="Q"), back=CardBack(explanation="E", example="X"))
    ] * 5
    _, message = _format_cards_to_dataframe(cards, "Multiple")

    assert "<span id='total-cards-count'>5</span>" in message


def test_format_cards_to_dataframe_none_subject():
    """Test that subject defaults to 'General' when subject is None or empty."""
    cards = [
        Card(front=CardFront(question="Q"), back=CardBack(explanation="E", example="X"))
    ]
    df, _ = _format_cards_to_dataframe(cards, "")
    assert df.iloc[0]["Topic"] == "General"

    df2, _ = _format_cards_to_dataframe(cards, None)
    assert df2.iloc[0]["Topic"] == "General"


def test_format_cards_to_dataframe_columns():
    """Verify the DataFrame columns match get_dataframe_columns."""
    cards = [
        Card(front=CardFront(question="Q"), back=CardBack(explanation="E", example="X"))
    ]
    df, _ = _format_cards_to_dataframe(cards, "Test")

    expected_cols = get_dataframe_columns()
    assert list(df.columns) == expected_cols


def test_format_cards_to_dataframe_internal_calls(mocker):
    """Verify that _format_cards_to_dataframe calls format_cards_for_dataframe with correct args."""
    # We patch the function in the card_generator module
    mock_format = mocker.patch("ankigen.card_generator.format_cards_for_dataframe")
    mock_format.return_value = [{"Index": "1", "Topic": "Mocked"}]

    cards = [
        Card(front=CardFront(question="Q"), back=CardBack(explanation="E", example="X"))
    ]
    _format_cards_to_dataframe(cards, "Special Subject")

    mock_format.assert_called_once_with(
        cards, topic_name="Special Subject", start_index=1
    )
