import pytest
import os
import pandas as pd
from unittest.mock import patch
from ankigen_core.exporters import (
    _format_field_as_string,
    _generate_timestamped_filename,
    _validate_non_empty_data,
    export_cards_to_csv,
    export_cards_to_apkg,
    export_cards_from_crawled_content,
    export_dataframe_to_csv,
    export_dataframe_to_apkg,
)

# --- Helper Tests ---


def test_format_field_as_string():
    assert _format_field_as_string("test") == "test"
    assert _format_field_as_string(["a", "b"]) == "a, b"
    assert _format_field_as_string((1, 2)) == "1, 2"
    assert _format_field_as_string(None) == ""
    assert _format_field_as_string(float("nan")) == ""
    assert _format_field_as_string("  trimmed  ") == "trimmed"


def test_generate_timestamped_filename():
    filename = _generate_timestamped_filename("base", "csv", include_timestamp=False)
    assert filename == "base.csv"

    filename_ts = _generate_timestamped_filename("base", "csv", include_timestamp=True)
    assert filename_ts.startswith("base_")
    assert filename_ts.endswith(".csv")


def test_validate_non_empty_data():
    with pytest.raises(ValueError, match="No data provided to export"):
        _validate_non_empty_data(None, "data")
    with pytest.raises(ValueError, match="No data provided to export"):
        _validate_non_empty_data([], "data")
    with pytest.raises(ValueError, match="No data available to export"):
        _validate_non_empty_data(pd.DataFrame(), "data")

    # Should not raise
    _validate_non_empty_data(["item"], "data")
    _validate_non_empty_data(pd.DataFrame({"a": [1]}), "data")


# --- Export Tests ---


def test_export_cards_to_csv(tmp_path):
    cards = [{"front": "Q1", "back": "A1", "tags": "t1"}, {"front": "Q2", "back": "A2"}]
    filename = str(tmp_path / "test.csv")

    path = export_cards_to_csv(cards, filename)
    assert path == filename
    assert os.path.exists(path)

    df = pd.read_csv(path).fillna("")
    assert len(df) == 2
    assert df.iloc[0]["front"] == "Q1"
    assert df.iloc[1]["tags"] == ""  # Default empty string for missing tags


def test_export_cards_to_csv_missing_keys(tmp_path):
    # Test empty cards list already covered by test_validate_non_empty_data which is called first
    with pytest.raises(ValueError, match="No cards provided to export"):
        export_cards_to_csv([])

    # Test cards with missing mandatory keys (they should be skipped)
    cards = [
        {"front": "Q1", "back": "A1"},  # Valid
        {"front": "Q2"},  # Missing "back"
        {"back": "A3"},  # Missing "front"
    ]
    filename = str(tmp_path / "mixed.csv")
    path = export_cards_to_csv(cards, filename)

    assert os.path.exists(path)
    df = pd.read_csv(path).fillna("")
    assert len(df) == 1
    assert df.iloc[0]["front"] == "Q1"


@patch("genanki.Package")
@patch("genanki.Deck")
@patch("genanki.Note")
def test_export_cards_to_apkg(mock_note, mock_deck, mock_package, tmp_path):
    cards = [{"Question": "Q1", "Answer": "A1", "note_type": "Basic"}]
    filename = str(tmp_path / "test.apkg")

    # Mock behavior
    mock_pkg_instance = mock_package.return_value
    mock_pkg_instance.write_to_file.return_value = None

    path = export_cards_to_apkg(cards, filename)
    assert path == filename
    mock_deck.return_value.add_note.assert_called_once()
    mock_pkg_instance.write_to_file.assert_called_once_with(filename)


def test_export_cards_to_apkg_empty_question(tmp_path):
    # If question is empty, it skips the card. If all skipped, it raises gr.Error.
    cards = [{"Question": "", "Answer": "A1"}]
    import gradio as gr

    with pytest.raises(gr.Error):
        export_cards_to_apkg(cards, str(tmp_path / "fail.apkg"))


def test_export_cards_from_crawled_content_dispatch():
    cards = [{"front": "Q1", "back": "A1"}]

    with patch("ankigen_core.exporters.export_cards_to_csv") as mock_csv:
        export_cards_from_crawled_content(cards, export_format="csv")
        mock_csv.assert_called_once()

    with patch("ankigen_core.exporters.export_cards_to_apkg") as mock_apkg:
        export_cards_from_crawled_content(cards, export_format="apkg")
        mock_apkg.assert_called_once()


def test_export_cards_from_crawled_content_unsupported():
    cards = [{"front": "Q1", "back": "A1"}]
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_cards_from_crawled_content(cards, export_format="pdf")


def test_export_dataframe_to_csv(tmp_path):
    df = pd.DataFrame({"Question": ["Q1"], "Answer": ["A1"]})
    # Gradio Info/Error are called, we should mock them
    with patch("gradio.Info"), patch("gradio.Error"):
        # We need to ensure the directory for final_filename exists or let it use default
        # The implementation uses _generate_timestamped_filename which might put it in root
        # Let's provide a path
        filename = str(tmp_path / "df_test.csv")
        # Wait, export_dataframe_to_csv doesn't take filename, it takes filename_suggestion
        # and generates a timestamped one in the current directory or relative to it.
        # It's better to patch _generate_timestamped_filename to return our temp path
        with patch(
            "ankigen_core.exporters._generate_timestamped_filename",
            return_value=filename,
        ):
            path = export_dataframe_to_csv(df)
            assert path == filename
            assert os.path.exists(path)


@patch("ankigen_core.exporters.export_cards_to_apkg")
def test_export_dataframe_to_apkg(mock_export_apkg):
    df = pd.DataFrame(
        {"Question": ["Q1"], "Answer": ["A1"], "Topic": ["T1"], "Difficulty": ["Easy"]}
    )
    export_dataframe_to_apkg(df, "output.apkg", "My Deck")
    mock_export_apkg.assert_called_once()
    # Check if cards were correctly prepared
    args, kwargs = mock_export_apkg.call_args
    prepared_cards = args[0]
    assert len(prepared_cards) == 1
    assert prepared_cards[0]["Question"] == "Q1"
    assert prepared_cards[0]["TagsStr"] == "T1 Easy"
