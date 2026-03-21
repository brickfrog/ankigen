import pytest
import pandas as pd
from ankigen.models import Card, CardFront, CardBack
from ankigen.ui_logic import (
    cards_to_dataframe,
    dataframe_to_cards,
    update_mode_visibility,
)


def test_cards_to_dataframe_empty():
    """Test cards_to_dataframe with an empty list of cards."""
    df = cards_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == [
        "ID",
        "Topic",
        "Front",
        "Back",
        "Tags",
        "Card Type",
        "Explanation",
        "Example",
        "Source_URL",
    ]


def test_cards_to_dataframe_single():
    """Test cards_to_dataframe with a single card."""
    card = Card(
        front=CardFront(question="What is Python?"),
        back=CardBack(answer="A programming language", explanation="E", example="Ex"),
        metadata={
            "topic": "Programming",
            "tags": ["python", "coding"],
            "source_url": "http://python.org",
        },
        card_type="Basic",
    )
    df = cards_to_dataframe([card])
    assert len(df) == 1
    assert df.iloc[0]["ID"] == 1
    assert df.iloc[0]["Topic"] == "Programming"
    assert df.iloc[0]["Front"] == "What is Python?"
    assert df.iloc[0]["Back"] == "A programming language"
    assert df.iloc[0]["Tags"] == "python, coding"
    assert df.iloc[0]["Card Type"] == "Basic"
    assert df.iloc[0]["Source_URL"] == "http://python.org"


def test_cards_to_dataframe_multiple():
    """Test cards_to_dataframe with multiple cards having varying metadata/tags."""
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"topic": "T1", "tags": ["tag1"]},
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="Ex2"),
            metadata={"tags": ["tag2", "tag3"]},
        ),
        Card(
            front=CardFront(question="Q3"),
            back=CardBack(answer="A3", explanation="E3", example="Ex3"),
            metadata={},
        ),
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 3
    assert df.iloc[0]["Topic"] == "T1"
    assert df.iloc[0]["Tags"] == "tag1"
    assert df.iloc[1]["Topic"] == "N/A"
    assert df.iloc[1]["Tags"] == "tag2, tag3"
    assert df.iloc[2]["Topic"] == "N/A"
    assert df.iloc[2]["Tags"] == ""


def test_cards_to_dataframe_no_metadata():
    """Test cards_to_dataframe with cards that have no metadata."""
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        metadata=None,
    )
    df = cards_to_dataframe([card])
    assert len(df) == 1
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""
    assert df.iloc[0]["Source_URL"] == ""


def test_dataframe_to_cards_empty():
    """Test dataframe_to_cards with empty dataframe and/or empty cards."""
    # Both empty
    assert dataframe_to_cards(pd.DataFrame(), []) == []

    # Empty DF, non-empty cards
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    assert dataframe_to_cards(pd.DataFrame(), original_cards) == []


def test_dataframe_to_cards_round_trip():
    """Test a normal round-trip from cards to dataframe and back with edits."""
    original_cards = [
        Card(
            front=CardFront(question="Original Q"),
            back=CardBack(answer="Original A", explanation="E", example="Ex"),
            metadata={"topic": "Original T", "tags": ["tag1"]},
        )
    ]
    df = cards_to_dataframe(original_cards)

    # Edit the dataframe
    df.at[0, "Front"] = "Updated Q"
    df.at[0, "Back"] = "Updated A"
    df.at[0, "Tags"] = "tag1, tag2"
    df.at[0, "Topic"] = "Updated T"
    df.at[0, "Card Type"] = "Cloze"

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "Updated Q"
    assert updated_cards[0].back.answer == "Updated A"
    assert updated_cards[0].metadata["tags"] == ["tag1", "tag2"]
    assert updated_cards[0].metadata["topic"] == "Updated T"
    assert updated_cards[0].card_type == "Cloze"


def test_dataframe_to_cards_out_of_bounds_id(mocker):
    """Test handling of IDs that are out of bounds for the original cards list."""
    mock_logger = mocker.patch("ankigen.ui_logic.logger")
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    # ID 2 is out of bounds (only 1 card exists)
    df = pd.DataFrame({"ID": [2], "Front": ["Q2"]})

    result = dataframe_to_cards(df, original_cards)
    assert result == []
    mock_logger.warning.assert_called_with(
        "Card ID 2 from DataFrame is out of bounds for original_cards list."
    )


def test_dataframe_to_cards_error_handling(mocker):
    """Test handling of rows that cause errors during processing."""
    mock_logger = mocker.patch("ankigen.ui_logic.logger")
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    # Case 1: Missing ID column (KeyError)
    # This triggers an UnboundLocalError in ui_logic.py because original_card_index is not defined
    # but used in the except block. We test that it happens (documenting the bug).
    df_missing_id = pd.DataFrame({"Front": ["Q1"]})

    with pytest.raises(UnboundLocalError):
        dataframe_to_cards(df_missing_id, original_cards)

    mock_logger.error.assert_called()


def test_dataframe_to_cards_error_handling_recovery(mocker):
    """Test that it recovers gracefully if an error occurs after ID is parsed."""
    mock_logger = mocker.patch("ankigen.ui_logic.logger")

    # Patch the class method instead of instance because Pydantic blocks instance patching of methods
    mocker.patch(
        "ankigen.models.CardFront.copy", side_effect=AttributeError("Mock Error")
    )

    card = Card(
        front=CardFront(question="Q1"),
        back=CardBack(answer="A1", explanation="E1", example="Ex1"),
    )
    original_cards = [card]
    df = pd.DataFrame({"ID": [1], "Front": ["New Q"]})

    result = dataframe_to_cards(df, original_cards)

    # It should catch the error, log it, and use the original card
    assert len(result) == 1
    assert result[0] is card
    assert mock_logger.error.call_count >= 1


def test_update_mode_visibility(mocker):
    """Verify update_mode_visibility returns the expected 5-tuple of gr.update() calls."""
    # Mock gr.update to return its arguments for easy verification
    mocker.patch("gradio.update", side_effect=lambda **kwargs: kwargs)

    result = update_mode_visibility("subject", "Biology")
    assert isinstance(result, tuple)
    assert len(result) == 5

    # Verify each update call
    assert result[0] == {"visible": True}
    assert result[1] == {"visible": True}
    assert result[2] == {"value": "Biology"}

    # result[3] should have a value that is an empty DataFrame with specific columns
    assert isinstance(result[3]["value"], pd.DataFrame)
    assert list(result[3]["value"].columns) == [
        "Index",
        "Topic",
        "Card_Type",
        "Question",
        "Answer",
        "Explanation",
        "Example",
        "Prerequisites",
        "Learning_Outcomes",
        "Difficulty",
    ]

    # result[4] is total_cards_html
    assert "total-cards-count" in result[4]["value"]
    assert result[4]["visible"] is False
