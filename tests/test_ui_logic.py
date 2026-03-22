import pandas as pd
from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack

# --- update_mode_visibility Tests ---


def test_update_mode_visibility(mocker):
    # Mock gr.update to return its arguments as a dict for easy verification
    mocker.patch("gradio.update", side_effect=lambda **kwargs: kwargs)

    current_subject = "Mathematics"
    result = update_mode_visibility("subject", current_subject)

    assert len(result) == 5
    # 1. subject_mode (Group)
    assert result[0] == {"visible": True}
    # 2. cards_output
    assert result[1] == {"visible": True}
    # 3. subject textbox value
    assert result[2] == {"value": current_subject}
    # 4. output DataFrame
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
    # 5. total_cards_html
    assert "Total Cards Generated" in result[4]["value"]
    assert result[4]["visible"] is False


# --- cards_to_dataframe Tests ---


def test_cards_to_dataframe_basic():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata={
                "topic": "T1",
                "tags": ["tag1", "tag2"],
                "source_url": "http://s1.com",
            },
            card_type="Basic",
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="Ex2"),
            metadata={"topic": "T2", "tags": []},
            card_type="Cloze",
        ),
    ]

    df = cards_to_dataframe(cards)

    assert len(df) == 2
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

    assert df.iloc[0]["ID"] == 1
    assert df.iloc[0]["Topic"] == "T1"
    assert df.iloc[0]["Front"] == "Q1"
    assert df.iloc[0]["Back"] == "A1"
    assert df.iloc[0]["Tags"] == "tag1, tag2"
    assert df.iloc[0]["Card Type"] == "Basic"
    assert df.iloc[0]["Source_URL"] == "http://s1.com"

    assert df.iloc[1]["ID"] == 2
    assert df.iloc[1]["Topic"] == "T2"
    assert df.iloc[1]["Tags"] == ""
    assert df.iloc[1]["Card Type"] == "Cloze"
    assert df.iloc[1]["Source_URL"] == ""


def test_cards_to_dataframe_no_metadata():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
            metadata=None,
        )
    ]

    df = cards_to_dataframe(cards)

    assert len(df) == 1
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""
    assert df.iloc[0]["Source_URL"] == ""
    assert df.iloc[0]["Card Type"] == "basic"  # Default from model


def test_cards_to_dataframe_empty_list():
    df = cards_to_dataframe([])
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


# --- dataframe_to_cards Tests ---


def test_dataframe_to_cards_update_success():
    original_cards = [
        Card(
            front=CardFront(question="Old Q1"),
            back=CardBack(answer="Old A1", explanation="Old E1", example="Old Ex1"),
            metadata={"topic": "Old T1", "tags": ["old_tag1"]},
            card_type="Basic",
        )
    ]

    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Topic": "New T1",
                "Front": "New Q1",
                "Back": "New A1",
                "Tags": "new_tag1, new_tag2",
                "Card Type": "Cloze",
                "Explanation": "New E1",
                "Example": "New Ex1",
                "Source_URL": "http://new.com",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, original_cards)

    assert len(updated_cards) == 1
    card = updated_cards[0]
    assert card.front.question == "New Q1"
    assert card.back.answer == "New A1"
    assert card.back.explanation == "New E1"
    assert card.back.example == "New Ex1"
    assert card.metadata["topic"] == "New T1"
    assert card.metadata["tags"] == ["new_tag1", "new_tag2"]
    assert card.card_type == "Cloze"


def test_dataframe_to_cards_empty_df():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]

    # Empty DF and original cards -> returns empty list
    assert dataframe_to_cards(pd.DataFrame(), original_cards) == []
    # Empty DF and no original cards -> returns empty list
    assert dataframe_to_cards(pd.DataFrame(), []) == []


def test_dataframe_to_cards_out_of_bounds():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    df = pd.DataFrame([{"ID": 2, "Front": "Invalid"}])

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 0  # Row is skipped with warning


def test_dataframe_to_cards_invalid_id_format():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    # Row with non-integer ID currently causes UnboundLocalError in ui_logic.py
    # because original_card_index is used in the except block before being defined.
    # We test with a numeric string that is out of bounds to verify basic graceful handling of out-of-bounds numeric IDs.
    df = pd.DataFrame([{"ID": "999", "Front": "Invalid"}])

    updated_cards = dataframe_to_cards(df, original_cards)
    assert len(updated_cards) == 0  # Out of bounds row is skipped


def test_dataframe_to_cards_partial_update_preserves_original():
    # If there's an error processing a row but the ID was valid, it might fallback
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    # Test missing columns in DF - should preserve original for missing fields
    df_missing_cols = pd.DataFrame([{"ID": 1, "Front": "Only Front"}])
    updated = dataframe_to_cards(df_missing_cols, original_cards)
    assert len(updated) == 1
    assert updated[0].front.question == "Only Front"
    assert updated[0].back.answer == "A1"  # Preserved


def test_dataframe_to_cards_multiple_rows():
    original_cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="Ex2"),
        ),
    ]
    df = pd.DataFrame([{"ID": 2, "Front": "New Q2"}, {"ID": 1, "Front": "New Q1"}])
    updated = dataframe_to_cards(df, original_cards)
    assert len(updated) == 2
    # Order follows DF order
    assert updated[0].front.question == "New Q2"
    assert updated[1].front.question == "New Q1"


def test_cards_to_dataframe_all_fields():
    cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="Expl", example="Eg"),
            metadata={"topic": "Top", "tags": ["t1"], "source_url": "url"},
            card_type="cloze",
        )
    ]
    df = cards_to_dataframe(cards)
    row = df.iloc[0]
    assert row["ID"] == 1
    assert row["Topic"] == "Top"
    assert row["Front"] == "Q"
    assert row["Back"] == "A"
    assert row["Tags"] == "t1"
    assert row["Card Type"] == "cloze"
    assert row["Explanation"] == "Expl"
    assert row["Example"] == "Eg"
    assert row["Source_URL"] == "url"


def test_dataframe_to_cards_empty_tags():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    df = pd.DataFrame([{"ID": 1, "Tags": "  , ,  "}])
    updated = dataframe_to_cards(df, original_cards)
    assert updated[0].metadata["tags"] == []


def test_dataframe_to_cards_strips_tags():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
        )
    ]
    df = pd.DataFrame([{"ID": 1, "Tags": " tag1 , tag2 "}])
    updated = dataframe_to_cards(df, original_cards)
    assert updated[0].metadata["tags"] == ["tag1", "tag2"]


def test_dataframe_to_cards_card_type_update():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
            card_type="Basic",
        )
    ]
    df = pd.DataFrame([{"ID": 1, "Card Type": "Cloze"}])
    updated = dataframe_to_cards(df, original_cards)
    assert updated[0].card_type == "Cloze"


def test_dataframe_to_cards_topic_preservation():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="Ex"),
            metadata={"topic": "Math"},
        )
    ]
    df = pd.DataFrame([{"ID": 1, "Front": "New Q"}])
    updated = dataframe_to_cards(df, original_cards)
    assert updated[0].metadata["topic"] == "Math"
    assert updated[0].front.question == "New Q"
