from ankigen_core.card_generator import (
    _parse_model_selection,
    _map_generation_mode_to_subject,
    _build_generation_context,
    format_cards_for_dataframe,
    get_dataframe_columns,
    generate_token_usage_html,
)
from ankigen_core.models import Card, CardFront, CardBack

# --- _parse_model_selection Tests ---


def test_parse_model_selection():
    assert _parse_model_selection("gpt-5.2-auto") == ("gpt-5.2", None)
    assert _parse_model_selection("gpt-5.2-instant") == ("gpt-5.2", "none")
    assert _parse_model_selection("gpt-5.2-thinking") == ("gpt-5.2", "high")
    assert _parse_model_selection("custom-model") == ("custom-model", None)
    assert _parse_model_selection("") == ("gpt-5.2", None)


# --- _map_generation_mode_to_subject Tests ---


def test_map_generation_mode_to_subject():
    assert _map_generation_mode_to_subject("subject", "Math") == "Math"
    assert _map_generation_mode_to_subject("subject", "") == "general"
    assert _map_generation_mode_to_subject("path", "") == "curriculum_design"
    assert _map_generation_mode_to_subject("text", "") == "content_analysis"
    assert _map_generation_mode_to_subject("unknown", "") == "general"


# --- _build_generation_context Tests ---


def test_build_generation_context():
    assert _build_generation_context("text", "some text") == {
        "source_text": "some text"
    }
    assert _build_generation_context("subject", "ignored") == {}
    assert _build_generation_context("text", "") == {}


# --- format_cards_for_dataframe Tests ---


def test_format_cards_for_dataframe():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        metadata={
            "prerequisites": ["P1", "P2"],
            "learning_outcomes": ["L1"],
            "difficulty": "beginner",
            "source_url": "http://example.com",
        },
        card_type="cloze",
    )

    formatted = format_cards_for_dataframe([card], "Test Topic")
    assert len(formatted) == 1
    f_card = formatted[0]
    assert f_card["Index"] == "1"
    assert f_card["Topic"] == "Test Topic"
    assert f_card["Card_Type"] == "cloze"
    assert f_card["Question"] == "Q"
    assert f_card["Prerequisites"] == "P1, P2"
    assert f_card["Learning_Outcomes"] == "L1"
    assert f_card["Difficulty"] == "beginner"
    assert f_card["Source_URL"] == "http://example.com"


def test_format_cards_for_dataframe_missing_metadata():
    card = Card(
        front=CardFront(question="Q"),
        back=CardBack(answer="A", explanation="E", example="Ex"),
        card_type="basic",
    )
    formatted = format_cards_for_dataframe([card], "No Metadata")
    f_card = formatted[0]
    assert f_card["Prerequisites"] == ""
    assert f_card["Difficulty"] == "N/A"


# --- get_dataframe_columns Tests ---


def test_get_dataframe_columns():
    cols = get_dataframe_columns()
    assert isinstance(cols, list)
    assert "Question" in cols
    assert "Answer" in cols
    assert "Topic" in cols


# --- generate_token_usage_html Tests ---


def test_generate_token_usage_html():
    usage = {"total_tokens": 100}
    html = generate_token_usage_html(usage)
    assert "100 tokens" in html

    html_none = generate_token_usage_html(None)
    assert "No usage data" in html_none

    html_invalid = generate_token_usage_html("invalid")
    assert "No usage data" in html_invalid


# --- Additional Tests to meet 10+ requirement ---


def test_available_models_constant():
    from ankigen_core.card_generator import AVAILABLE_MODELS

    assert len(AVAILABLE_MODELS) >= 3
    assert AVAILABLE_MODELS[0]["value"] == "gpt-5.2-auto"


def test_generation_modes_constant():
    from ankigen_core.card_generator import GENERATION_MODES

    assert len(GENERATION_MODES) >= 1
    assert GENERATION_MODES[0]["value"] == "subject"


def test_format_cards_for_dataframe_empty():
    assert format_cards_for_dataframe([], "Empty") == []


def test_format_cards_for_dataframe_multiple_cards():
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
    formatted = format_cards_for_dataframe(cards, "Multiple", start_index=10)
    assert len(formatted) == 2
    assert formatted[0]["Index"] == "10"
    assert formatted[1]["Index"] == "11"
