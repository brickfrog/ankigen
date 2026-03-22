import pandas as pd
import gradio as gr
from ankigen.ui_logic import (
    update_mode_visibility,
    cards_to_dataframe,
    dataframe_to_cards,
)
from ankigen.models import Card, CardFront, CardBack


def test_update_mode_visibility():
    result = update_mode_visibility("subject", "Math")

    # 1. subject_mode (Group) - always visible
    assert result[0] == gr.update(visible=True)
    # 2. cards_output - always visible
    assert result[1] == gr.update(visible=True)
    # 3. subject textbox value
    assert result[2] == gr.update(value="Math")
    # 4. output DataFrame
    assert isinstance(result[3], dict)
    assert "value" in result[3]
    assert isinstance(result[3]["value"], pd.DataFrame)
    assert "Topic" in result[3]["value"].columns
    # 5. total_cards_html
    assert result[4] == gr.update(
        value="<div><b>Total Cards Generated:</b> <span id='total-cards-count'>0</span></div>",
        visible=False,
    )


def test_cards_to_dataframe():
    card = Card(
        front=CardFront(question="Front 1"),
        back=CardBack(answer="Back 1", explanation="Expl", example="Ex"),
        metadata={"tags": ["tag1", "tag2"], "topic": "Math", "source_url": "url"},
        card_type="Cloze",
    )

    df = cards_to_dataframe([card])

    assert len(df) == 1
    assert df.iloc[0]["Front"] == "Front 1"
    assert df.iloc[0]["Back"] == "Back 1"
    assert df.iloc[0]["Tags"] == "tag1, tag2"
    assert df.iloc[0]["Topic"] == "Math"
    assert df.iloc[0]["Card Type"] == "Cloze"
    assert df.iloc[0]["Source_URL"] == "url"


def test_cards_to_dataframe_empty():
    df = cards_to_dataframe([])
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


def test_dataframe_to_cards():
    original_card = Card(
        front=CardFront(question="Old Q"),
        back=CardBack(answer="Old A", explanation="Old E", example="Old X"),
        metadata={"tags": ["old"], "topic": "Old Topic"},
        card_type="basic",
    )

    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Topic": "New Topic",
                "Front": "New Q",
                "Back": "New A",
                "Tags": "new1, new2",
                "Card Type": "cloze",
                "Explanation": "New E",
                "Example": "New X",
            }
        ]
    )

    updated_cards = dataframe_to_cards(df, [original_card])

    assert len(updated_cards) == 1
    new_card = updated_cards[0]
    assert new_card.front.question == "New Q"
    assert new_card.back.answer == "New A"
    assert new_card.back.explanation == "New E"
    assert new_card.metadata["topic"] == "New Topic"
    assert new_card.metadata["tags"] == ["new1", "new2"]
    assert new_card.card_type == "cloze"


def test_dataframe_to_cards_empty_df():
    original_cards = [
        Card(
            front=CardFront(question="Q"),
            back=CardBack(answer="A", explanation="E", example="X"),
        )
    ]
    assert dataframe_to_cards(pd.DataFrame(), original_cards) == []


def test_dataframe_to_cards_out_of_bounds():
    df = pd.DataFrame([{"ID": 5, "Front": "Invalid"}])
    assert dataframe_to_cards(df, []) == []


def test_cards_to_dataframe_various_types():
    cards = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="X1"),
            card_type="Basic",
        ),
        Card(
            front=CardFront(question="Q2"),
            back=CardBack(answer="A2", explanation="E2", example="X2"),
            card_type="Cloze",
        ),
    ]
    df = cards_to_dataframe(cards)
    assert len(df) == 2
    assert df.iloc[0]["Card Type"] == "Basic"
    assert df.iloc[1]["Card Type"] == "Cloze"


def test_cards_to_dataframe_no_metadata():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
    )
    df = cards_to_dataframe([card])
    assert df.iloc[0]["Topic"] == "N/A"
    assert df.iloc[0]["Tags"] == ""


def test_dataframe_to_cards_missing_columns():
    original_card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="X"),
    )
    df = pd.DataFrame([{"ID": 1}])  # Only ID
    updated_cards = dataframe_to_cards(df, [original_card])
    assert len(updated_cards) == 1
    # Should use defaults from original_card
    assert updated_cards[0].front.question == "Q"


def test_dataframe_to_cards_multiple():
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
    df = pd.DataFrame(
        [
            {"ID": 1, "Front": "U1"},
            {"ID": 2, "Front": "U2"},
        ]
    )
    updated = dataframe_to_cards(df, cards)
    assert len(updated) == 2
    assert updated[0].front.question == "U1"
    assert updated[1].front.question == "U2"
