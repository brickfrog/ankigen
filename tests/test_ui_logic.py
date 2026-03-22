import pytest
import pandas as pd
from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack


# Mock gradio.update since we don't want to depend on a running gradio instance
@pytest.fixture(autouse=True)
def mock_gr_update(mocker):
    return mocker.patch("gradio.update", side_effect=lambda **kwargs: kwargs)


def test_update_mode_visibility_subject_mode():
    """Test update_mode_visibility returns correct updates for 'subject' mode."""
    subject = "Python Programming"
    updates = update_mode_visibility("subject", subject)

    assert len(updates) == 5
    # subject_mode (Group)
    assert updates[0] == {"visible": True}
    # cards_output
    assert updates[1] == {"visible": True}
    # subject textbox value
    assert updates[2] == {"value": subject}
    # output DataFrame
    assert isinstance(updates[3]["value"], pd.DataFrame)
    assert list(updates[3]["value"].columns) == [
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
    assert "Total Cards Generated:" in updates[4]["value"]
    assert updates[4]["visible"] is False


def test_cards_to_dataframe_empty():
    """Test cards_to_dataframe with an empty list of cards."""
    df = cards_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
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


def test_cards_to_dataframe_with_metadata():
    """Test cards_to_dataframe with cards having metadata and tags."""
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={
                "tags": ["tag1", "tag2"],
                "topic": "Topic1",
                "source_url": "http://url1",
            },
            card_type="cloze",
        )
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 1
    assert df.iloc[0]["ID"] == 1
    assert df.iloc[0]["Topic"] == "Topic1"
    assert df.iloc[0]["Front"] == "Q1"
    assert df.iloc[0]["Back"] == "A1"
    assert df.iloc[0]["Tags"] == "tag1, tag2"
    assert df.iloc[0]["Card Type"] == "cloze"
    assert df.iloc[0]["Explanation"] == "E1"
    assert df.iloc[0]["Example"] == "Ex1"
    assert df.iloc[0]["Source_URL"] == "http://url1"


def test_cards_to_dataframe_without_metadata():
    """Test cards_to_dataframe with cards missing metadata."""
    cards = [
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="Ex2"),
            metadata=None,
            # card_type defaults to "basic"
        )
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 1
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""
    assert df.iloc[0]["Card Type"] == "basic"
    assert df.iloc[0]["Source_URL"] == ""


def test_cards_to_dataframe_multiple_cards():
    """Test cards_to_dataframe with multiple cards."""
    cards = [
        Card(
            front=CardFront(question=f"Q{i}"),
            back=CardBack(answer=f"A{i}", explanation=f"E{i}", example=f"Ex{i}"),
        )
        for i in range(3)
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 3
    assert list(df["ID"]) == [1, 2, 3]
    assert list(df["Front"]) == ["Q0", "Q1", "Q2"]


def test_dataframe_to_cards_round_trip():
    """Test round-trip conversion from cards to dataframe and back."""
    original_cards = [
        Card(
            front=CardFront(question="Original Q"),
            back=CardBack(answer="Original A", explanation="E", example="Ex"),
            metadata={"tags": ["tag1"], "topic": "Original Topic"},
        )
    ]
    df = cards_to_dataframe(original_cards)

    # Modify the dataframe
    df.at[0, "Front"] = "Updated Q"
    df.at[0, "Tags"] = "tag1, tag3"

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "Updated Q"
    assert updated_cards[0].back.answer == "Original A"
    assert updated_cards[0].metadata["tags"] == ["tag1", "tag3"]
    assert updated_cards[0].metadata["topic"] == "Original Topic"


def test_dataframe_to_cards_empty():
    """Test dataframe_to_cards with empty inputs."""
    assert dataframe_to_cards(pd.DataFrame(), []) == []
    assert (
        dataframe_to_cards(
            pd.DataFrame(),
            [Card(front=CardFront(), back=CardBack(explanation="", example=""))],
        )
        == []
    )


def test_dataframe_to_cards_out_of_bounds_id(mocker):
    """Test dataframe_to_cards handles out of bounds IDs by logging a warning."""
    mock_logger = mocker.patch("ankigen.ui_logic.logger")
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="", example=""),
        )
    ]
    df = pd.DataFrame(
        [
            {
                "ID": 5,  # Out of bounds
                "Front": "New Q",
                "Back": "New A",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 0
    mock_logger.warning.assert_called_once()


def test_dataframe_to_cards_malformed_id(mocker):
    """Test dataframe_to_cards handles malformed IDs."""
    mock_logger = mocker.patch("ankigen.ui_logic.logger")
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="", example=""),
        )
    ]
    df = pd.DataFrame([{"ID": "not-a-number", "Front": "New Q"}])

    updated_cards = dataframe_to_cards(df, original_cards)

    # The current implementation catches the error and might try to append the original card if card_id was somehow set?
    # Looking at the code:
    # except (ValueError, KeyError, AttributeError) as e:
    #     logger.error(...)
    #     if 0 <= original_card_index < len(original_cards):
    #         updated_cards.append(original_cards[original_card_index])
    # In this case original_card_index is not defined before the exception if card_id = int(row["ID"]) fails.

    assert len(updated_cards) == 0
    mock_logger.error.assert_called_once()


def test_dataframe_to_cards_partial_updates():
    """Test dataframe_to_cards with missing columns in DataFrame."""
    original_cards = [
        Card(
            front=CardFront(question="Original Q"),
            back=CardBack(answer="Original A", explanation="Orig E", example="Orig Ex"),
            metadata={"topic": "Orig Topic", "tags": ["tag1"]},
        )
    ]
    # DataFrame missing many columns
    df = pd.DataFrame([{"ID": 1, "Front": "New Q"}])

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "New Q"
    assert updated_cards[0].back.answer == "Original A"  # Kept from original
    assert updated_cards[0].back.explanation == "Orig E"  # Kept from original
    assert updated_cards[0].metadata["topic"] == "Orig Topic"


def test_dataframe_to_cards_tags_processing():
    """Test complex tags processing in dataframe_to_cards."""
    original_cards = [
        Card(front=CardFront(), back=CardBack(explanation="", example=""))
    ]
    df = pd.DataFrame([{"ID": 1, "Tags": " tag1 , tag2,, tag3 "}])

    updated_cards = dataframe_to_cards(df, original_cards)
    assert updated_cards[0].metadata["tags"] == ["tag1", "tag2", "tag3"]


def test_dataframe_to_cards_topic_update():
    """Test topic update in dataframe_to_cards."""
    original_cards = [
        Card(
            front=CardFront(),
            back=CardBack(explanation="", example=""),
            metadata={"topic": "Old Topic"},
        )
    ]
    df = pd.DataFrame([{"ID": 1, "Topic": "New Topic"}])

    updated_cards = dataframe_to_cards(df, original_cards)
    assert updated_cards[0].metadata["topic"] == "New Topic"
