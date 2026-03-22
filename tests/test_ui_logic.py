import pytest
import pandas as pd

from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack

# --- update_mode_visibility Tests ---


def test_update_mode_visibility_structure():
    """Test that update_mode_visibility returns the correct structure of 5 gr.update() calls."""
    result = update_mode_visibility("subject", "Math")
    assert isinstance(result, tuple)
    assert len(result) == 5
    for item in result:
        assert isinstance(item, dict)
        assert item.get("__type__") == "update"


def test_update_mode_visibility_values():
    """Test the values returned by update_mode_visibility."""
    subject = "History"
    result = update_mode_visibility("subject", subject)

    # 1. subject_mode visibility
    assert result[0]["visible"] is True
    # 2. cards_output visibility
    assert result[1]["visible"] is True
    # 3. subject value
    assert result[2]["value"] == subject
    # 4. output DataFrame columns
    df = result[3]["value"]
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
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
    assert list(df.columns) == expected_cols
    # 5. total_cards_html
    assert "Total Cards Generated" in result[4]["value"]
    assert result[4]["visible"] is False


# --- cards_to_dataframe Tests ---


def test_cards_to_dataframe_empty():
    """Test cards_to_dataframe with an empty list."""
    df = cards_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "ID" in df.columns


def test_cards_to_dataframe_single_card():
    """Test cards_to_dataframe with a single card."""
    card = Card(
        front=CardFront(question="What is 2+2?"),
        back=CardBack(
            answer="4", explanation="Basic math", example="2 apples + 2 apples"
        ),
        metadata={
            "topic": "Math",
            "tags": ["arithmetic"],
            "source_url": "http://math.com",
        },
        card_type="Basic",
    )
    df = cards_to_dataframe([card])
    assert len(df) == 1
    assert df.iloc[0]["ID"] == 1
    assert df.iloc[0]["Topic"] == "Math"
    assert df.iloc[0]["Front"] == "What is 2+2?"
    assert df.iloc[0]["Back"] == "4"
    assert df.iloc[0]["Tags"] == "arithmetic"
    assert df.iloc[0]["Card Type"] == "Basic"
    assert df.iloc[0]["Source_URL"] == "http://math.com"


def test_cards_to_dataframe_multiple_cards():
    """Test cards_to_dataframe with multiple cards."""
    cards = [
        Card(
            front=CardFront(question=f"Q{i}"),
            back=CardBack(answer=f"A{i}", explanation=f"E{i}", example=f"X{i}"),
        )
        for i in range(3)
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 3
    assert list(df["ID"]) == [1, 2, 3]
    assert df.iloc[1]["Front"] == "Q1"


def test_cards_to_dataframe_missing_metadata():
    """Test cards_to_dataframe with cards missing metadata."""
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
        metadata=None,
    )
    df = cards_to_dataframe([card])
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""
    assert df.iloc[0]["Source_URL"] == ""


def test_cards_to_dataframe_tags_list():
    """Test cards_to_dataframe with multiple tags."""
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
        metadata={"tags": ["tag1", "tag2", "tag3"]},
    )
    df = cards_to_dataframe([card])
    assert df.iloc[0]["Tags"] == "tag1, tag2, tag3"


# --- dataframe_to_cards Tests ---


def test_dataframe_to_cards_empty():
    """Test dataframe_to_cards with empty DataFrame and empty cards."""
    df = pd.DataFrame()
    updated = dataframe_to_cards(df, [])
    assert updated == []


def test_dataframe_to_cards_empty_df_existing_cards():
    """Test dataframe_to_cards with empty DataFrame but existing cards."""
    cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="X"),
        )
    ]
    df = pd.DataFrame()
    updated = dataframe_to_cards(df, cards)
    assert updated == []


def test_dataframe_to_cards_normal_update():
    """Test normal update flow for dataframe_to_cards."""
    original_cards = [
        Card(
            front=CardFront(question="Original Q"),
            back=CardBack(answer="Original A", explanation="Orig E", example="Orig X"),
            metadata={"topic": "Original Topic", "tags": ["tag1"]},
            card_type="Basic",
        )
    ]
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Topic": "New Topic",
                "Front": "New Q",
                "Back": "New A",
                "Tags": "tag1, tag2",
                "Card Type": "Cloze",
                "Explanation": "New E",
                "Example": "New X",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 1
    updated = updated_cards[0]
    assert updated.front.question == "New Q"
    assert updated.back.answer == "New A"
    assert updated.back.explanation == "New E"
    assert updated.back.example == "New X"
    assert updated.metadata["topic"] == "New Topic"
    assert updated.metadata["tags"] == ["tag1", "tag2"]
    assert updated.card_type == "Cloze"


def test_dataframe_to_cards_out_of_bounds():
    """Test dataframe_to_cards with out-of-bounds IDs."""
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="X"),
        )
    ]
    df = pd.DataFrame([{"ID": 5, "Front": "New Q"}])  # ID 5 is out of bounds

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 0


def test_dataframe_to_cards_invalid_data():
    """Test dataframe_to_cards with invalid data (e.g., non-integer ID).
    Note: Currently triggers UnboundLocalError in ui_logic.py due to a bug
    where original_card_index is used in the except block before definition.
    """
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="X"),
        )
    ]
    df = pd.DataFrame([{"ID": "invalid", "Front": "New Q"}])

    with pytest.raises(UnboundLocalError):
        dataframe_to_cards(df, original_cards)


def test_dataframe_to_cards_partial_row_data():
    """Test dataframe_to_cards when some columns are missing from the row."""
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="X"),
        )
    ]
    # Only ID and Front are provided
    df = pd.DataFrame([{"ID": 1, "Front": "Updated Q"}])

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "Updated Q"
    assert updated_cards[0].back.answer == "A"  # Should remain original
