import pandas as pd
from unittest.mock import patch

from ankigen.models import Card, CardFront, CardBack
from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)


def test_update_mode_visibility():
    mode = "subject"
    current_subject = "Mathematics"

    with patch("gradio.update") as mock_update:
        # We want to return something distinct for each call to verify they are all returned
        mock_update.side_effect = lambda **kwargs: kwargs

        result = update_mode_visibility(mode, current_subject)

        assert len(result) == 5
        assert mock_update.call_count == 5

        # Verify the arguments of each gr.update call
        # gr.update(visible=True) - subject_mode
        assert result[0] == {"visible": True}
        # gr.update(visible=True) - cards_output
        assert result[1] == {"visible": True}
        # gr.update(value=current_subject)
        assert result[2] == {"value": current_subject}
        # gr.update(value=pd.DataFrame(columns=...))
        assert isinstance(result[3]["value"], pd.DataFrame)
        # gr.update(value=..., visible=False)
        assert "Total Cards Generated" in result[4]["value"]
        assert result[4]["visible"] is False


def test_cards_to_dataframe_with_metadata():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={
                "tags": ["tag1", "tag2"],
                "topic": "Topic1",
                "source_url": "http://example.com",
            },
            card_type="Basic",
        )
    ]
    df = cards_to_dataframe(cards)

    assert len(df) == 1
    assert df.iloc[0]["ID"] == 1
    assert df.iloc[0]["Topic"] == "Topic1"
    assert df.iloc[0]["Front"] == "Q1"
    assert df.iloc[0]["Back"] == "A1"
    assert df.iloc[0]["Tags"] == "tag1, tag2"
    assert df.iloc[0]["Card Type"] == "Basic"
    assert df.iloc[0]["Explanation"] == "E1"
    assert df.iloc[0]["Example"] == "Ex1"
    assert df.iloc[0]["Source_URL"] == "http://example.com"


def test_cards_to_dataframe_no_metadata():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata=None,
            card_type="Basic",
        )
    ]
    df = cards_to_dataframe(cards)

    assert len(df) == 1
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""
    assert df.iloc[0]["Source_URL"] == ""


def test_cards_to_dataframe_empty():
    df = cards_to_dataframe([])
    assert len(df) == 0
    # ID Topic Front Back Tags Card Type Explanation Example Source_URL
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


def test_dataframe_to_cards_roundtrip():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={"tags": ["tag1"], "topic": "Topic1"},
            card_type="Basic",
        )
    ]

    df = cards_to_dataframe(original_cards)

    # Modify the dataframe
    df.at[0, "Front"] = "Updated Q1"
    df.at[0, "Tags"] = "tag1, tag2"

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "Updated Q1"
    assert updated_cards[0].metadata["tags"] == ["tag1", "tag2"]
    # Check that other fields are preserved
    assert updated_cards[0].back.answer == "A1"
    assert updated_cards[0].metadata["topic"] == "Topic1"


def test_dataframe_to_cards_empty_df():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]
    # If df is empty, it should return empty list (based on implementation)
    updated_cards = dataframe_to_cards(pd.DataFrame(), original_cards)
    assert updated_cards == []

    # If both are empty
    updated_cards = dataframe_to_cards(pd.DataFrame(), [])
    assert updated_cards == []


def test_dataframe_to_cards_out_of_bounds():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    # Create DF with ID that doesn't exist (ID is 1-indexed)
    df = pd.DataFrame(
        [
            {
                "ID": 5,
                "Topic": "T",
                "Front": "Q",
                "Back": "A",
                "Tags": "tag",
                "Card Type": "Basic",
                "Explanation": "E",
                "Example": "Ex",
                "Source_URL": "",
            }
        ]
    )

    # We patch the logger in ui_logic to verify warning
    with patch("ankigen.ui_logic.logger") as mock_logger:
        updated_cards = dataframe_to_cards(df, original_cards)
        assert len(updated_cards) == 0
        mock_logger.warning.assert_called()
        # Find the call that contains "out of bounds"
        args, kwargs = mock_logger.warning.call_args
        assert "out of bounds" in args[0]
