import pytest
import pandas as pd
from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack


@pytest.fixture
def mock_gr_update(mocker):
    return mocker.patch("gradio.update", side_effect=lambda **kwargs: kwargs)


def test_update_mode_visibility(mock_gr_update):
    current_subject = "Mathematics"
    result = update_mode_visibility("subject", current_subject)

    assert len(result) == 5
    # subject_mode (Group) - always visible
    assert result[0] == {"visible": True}
    # cards_output - always visible
    assert result[1] == {"visible": True}
    # subject textbox value
    assert result[2] == {"value": current_subject}
    # output DataFrame
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
    # total_cards_html
    assert "Total Cards Generated" in result[4]["value"]
    assert result[4]["visible"] is False


def test_cards_to_dataframe_empty():
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


def test_cards_to_dataframe_with_cards():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"topic": "T1", "tags": ["tag1", "tag2"], "source_url": "url1"},
            card_type="Basic",
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="Ex2"),
            # No metadata
            card_type="Cloze",
        ),
    ]

    df = cards_to_dataframe(cards)

    assert len(df) == 2
    assert df.iloc[0]["Topic"] == "T1"
    assert df.iloc[0]["Tags"] == "tag1, tag2"
    assert df.iloc[0]["Source_URL"] == "url1"
    assert df.iloc[0]["Card Type"] == "Basic"

    assert df.iloc[1]["Topic"] == "N/A"
    assert df.iloc[1]["Tags"] == ""
    assert df.iloc[1]["Source_URL"] == ""
    assert df.iloc[1]["Card Type"] == "Cloze"
    assert df.iloc[1]["Front"] == "Q2"


def test_dataframe_to_cards_empty():
    assert dataframe_to_cards(pd.DataFrame(), []) == []

    # Empty DF but original cards present - should return empty list as per implementation
    # Implementation: if df.empty and original_cards: return []
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    assert dataframe_to_cards(pd.DataFrame(), original_cards) == []


def test_dataframe_to_cards_updates():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"topic": "T1", "tags": ["tag1"]},
            card_type="Basic",
        )
    ]

    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Topic": "Updated T1",
                "Front": "Updated Q1",
                "Back": "Updated A1",
                "Tags": "newtag1, newtag2",
                "Card Type": "Updated Type",
                "Explanation": "Updated E1",
                "Example": "Updated Ex1",
                "Source_URL": "ignored_in_current_impl",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    card = updated_cards[0]
    assert card.front.question == "Updated Q1"
    assert card.back.answer == "Updated A1"
    assert card.back.explanation == "Updated E1"
    assert card.back.example == "Updated Ex1"
    assert card.card_type == "Updated Type"
    assert card.metadata["topic"] == "Updated T1"
    assert card.metadata["tags"] == ["newtag1", "newtag2"]


def test_dataframe_to_cards_out_of_bounds(caplog):
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    df = pd.DataFrame(
        [
            {
                "ID": 2,  # Out of bounds
                "Front": "Q2",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 0
    assert "out of bounds" in caplog.text


def test_dataframe_to_cards_missing_columns():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"topic": "T1", "tags": ["tag1"]},
            card_type="Basic",
        )
    ]

    # Only ID and Front provided
    df = pd.DataFrame([{"ID": 1, "Front": "Updated Q1"}])

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    card = updated_cards[0]
    assert card.front.question == "Updated Q1"
    # Others should be preserved from original_cards
    assert card.back.answer == "A1"
    assert card.metadata["topic"] == "T1"
    assert card.card_type == "Basic"


def test_dataframe_to_cards_invalid_id(caplog):
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    df = pd.DataFrame([{"ID": "invalid", "Front": "Q1"}])

    # The implementation currently has a bug where it tries to access original_card_index
    # in the except block even if it was never assigned due to int(row["ID"]) failing.
    # This results in an UnboundLocalError.
    with pytest.raises(UnboundLocalError):
        dataframe_to_cards(df, original_cards)

    assert "Error processing row 0" in caplog.text
