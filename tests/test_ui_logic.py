import sys
from unittest.mock import MagicMock

# Mock gradio and pandas before importing ui_logic
mock_gr = MagicMock()
sys.modules["gradio"] = mock_gr

mock_pd = MagicMock()
sys.modules["pandas"] = mock_pd


# Minimal DataFrame mock to support basic operations in ui_logic
class MockDataFrame:
    def __init__(self, data=None, columns=None):
        self.data = data or []
        self.columns = columns or []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            self.columns = list(data[0].keys())

    @property
    def empty(self):
        return len(self.data) == 0

    def iterrows(self):
        for i, row in enumerate(self.data):
            yield i, row


mock_pd.DataFrame = MockDataFrame

from ankigen.ui_logic import (  # noqa: E402
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack  # noqa: E402


def test_update_mode_visibility():
    # Arrange
    mock_gr.update = MagicMock(side_effect=lambda **kwargs: kwargs)
    mode = "subject"
    current_subject = "Physics"

    # Act
    results = update_mode_visibility(mode, current_subject)

    # Assert
    assert len(results) == 5
    assert results[0] == {"visible": True}  # subject_mode (Group)
    assert results[1] == {"visible": True}  # cards_output
    assert results[2] == {"value": current_subject}  # subject textbox value

    # Check DataFrame update
    df_update = results[3]
    assert "value" in df_update
    assert isinstance(df_update["value"], MockDataFrame)
    expected_columns = [
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
    assert list(df_update["value"].columns) == expected_columns

    # Check total_cards_html update
    assert results[4]["visible"] is False
    assert "Total Cards Generated:" in results[4]["value"]


def test_cards_to_dataframe():
    # Arrange
    card1 = Card(
        front=CardFront(question="Q1"),
        back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        metadata={
            "tags": ["tag1", "tag2"],
            "topic": "Topic1",
            "source_url": "http://url1",
        },
        card_type="Basic",
    )
    card2 = Card(
        front=CardFront(question="Q2"),
        back=CardBack(answer="A2", explanation="E2", example="Ex2"),
        # Missing metadata
        card_type="Cloze",
    )
    cards = [card1, card2]

    # Act
    df = cards_to_dataframe(cards)

    # Assert
    assert isinstance(df, MockDataFrame)
    assert len(df.data) == 2

    # Row 1
    row1 = df.data[0]
    assert row1["ID"] == 1
    assert row1["Topic"] == "Topic1"
    assert row1["Front"] == "Q1"
    assert row1["Back"] == "A1"
    assert row1["Tags"] == "tag1, tag2"
    assert row1["Card Type"] == "Basic"
    assert row1["Explanation"] == "E1"
    assert row1["Example"] == "Ex1"
    assert row1["Source_URL"] == "http://url1"

    # Row 2 (missing metadata)
    row2 = df.data[1]
    assert row2["ID"] == 2
    assert row2["Topic"] == "N/A"
    assert row2["Front"] == "Q2"
    assert row2["Back"] == "A2"
    assert row2["Tags"] == ""
    assert row2["Card Type"] == "Cloze"
    assert row2["Explanation"] == "E2"
    assert row2["Example"] == "Ex2"
    assert row2["Source_URL"] == ""


def test_dataframe_to_cards_success():
    # Arrange
    original_card1 = Card(
        front=CardFront(question="Q1"),
        back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        metadata={"tags": ["old_tag"], "topic": "Old Topic"},
        card_type="Basic",
    )
    original_cards = [original_card1]

    df_data = [
        {
            "ID": 1,
            "Topic": "New Topic",
            "Front": "New Q1",
            "Back": "New A1",
            "Tags": "tag1, tag2",
            "Card Type": "Cloze",
            "Explanation": "New E1",
            "Example": "New Ex1",
        }
    ]
    df = MockDataFrame(df_data)

    # Act
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert len(updated_cards) == 1
    updated = updated_cards[0]
    assert updated.front.question == "New Q1"
    assert updated.back.answer == "New A1"
    assert updated.back.explanation == "New E1"
    assert updated.back.example == "New Ex1"
    assert updated.metadata["topic"] == "New Topic"
    assert updated.metadata["tags"] == ["tag1", "tag2"]
    assert updated.card_type == "Cloze"


def test_dataframe_to_cards_empty_df():
    # Arrange
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    df = MockDataFrame()

    # Act
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert updated_cards == []

    # Test both empty
    assert dataframe_to_cards(MockDataFrame(), []) == []


def test_dataframe_to_cards_out_of_bounds():
    # Arrange
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    df = MockDataFrame([{"ID": 1, "Front": "New Q1"}, {"ID": 2, "Front": "New Q2"}])

    # Act
    # This should log a warning but only return the card that matches ID 1
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert len(updated_cards) == 1
    assert updated_cards[0].front.question == "New Q1"


def test_dataframe_to_cards_missing_columns():
    # Arrange
    original_card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        metadata={"tags": ["tag"]},
        card_type="Basic",
    )
    original_cards = [original_card]

    # Only ID and Front provided
    df = MockDataFrame([{"ID": 1, "Front": "Updated Q"}])

    # Act
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert len(updated_cards) == 1
    updated = updated_cards[0]
    assert updated.front.question == "Updated Q"
    # Other values should remain original
    assert updated.back.answer == "A"
    assert updated.metadata["tags"] == ["tag"]


def test_dataframe_to_cards_tags_handling():
    # Arrange
    original_card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        metadata={"tags": ["tag1"]},
    )
    original_cards = [original_card]

    # Test different tag formats
    df = MockDataFrame([{"ID": 1, "Tags": "  tag2, tag3 , tag1  "}])

    # Act
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert updated_cards[0].metadata["tags"] == ["tag2", "tag3", "tag1"]


def test_dataframe_to_cards_missing_topic():
    # Arrange
    original_card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        metadata={"topic": "Original Topic"},
    )
    original_cards = [original_card]

    # Topic not in DF
    df = MockDataFrame([{"ID": 1}])

    # Act
    updated_cards = dataframe_to_cards(df, original_cards)

    # Assert
    assert updated_cards[0].metadata["topic"] == "Original Topic"

    # Topic is in DF but empty
    df2 = MockDataFrame([{"ID": 1, "Topic": ""}])
    updated_cards2 = dataframe_to_cards(df2, original_cards)
    assert updated_cards2[0].metadata["topic"] == ""
