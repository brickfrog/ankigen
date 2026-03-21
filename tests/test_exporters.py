import pytest
import pandas as pd
import os
import gradio as gr
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

# --- _format_field_as_string Tests ---


def test_format_field_as_string():
    assert _format_field_as_string("test") == "test"
    assert _format_field_as_string(["a", "b"]) == "a, b"
    assert _format_field_as_string(("c", "d")) == "c, d"
    assert _format_field_as_string(None) == ""
    assert _format_field_as_string(float("nan")) == ""
    assert _format_field_as_string(123) == "123"


# --- _generate_timestamped_filename Tests ---


def test_generate_timestamped_filename():
    filename = _generate_timestamped_filename("test", "csv", include_timestamp=False)
    assert filename == "test.csv"

    filename_ts = _generate_timestamped_filename("test", "csv", include_timestamp=True)
    assert filename_ts.startswith("test_")
    assert filename_ts.endswith(".csv")
    assert len(filename_ts) > len("test.csv")


# --- _validate_non_empty_data Tests ---


def test_validate_non_empty_data():
    _validate_non_empty_data([1], "test")  # Should not raise
    _validate_non_empty_data(pd.DataFrame({"a": [1]}), "test")  # Should not raise

    with pytest.raises(ValueError, match="No test provided to export."):
        _validate_non_empty_data(None, "test")

    with pytest.raises(ValueError, match="No test provided to export."):
        _validate_non_empty_data([], "test")

    with pytest.raises(ValueError, match="No test available to export."):
        _validate_non_empty_data(pd.DataFrame(), "test")


# --- export_cards_to_csv Tests ---


def test_export_cards_to_csv_success(tmp_path):
    cards = [
        {"front": "Q1", "back": "A1", "tags": "tag1", "note_type": "Basic"},
        {"front": "Q2", "back": "A2"},
    ]
    filename = str(tmp_path / "test.csv")
    result = export_cards_to_csv(cards, filename=filename)

    assert result == filename
    assert os.path.exists(filename)
    df = pd.read_csv(filename)
    assert len(df) == 2
    assert df.iloc[0]["front"] == "Q1"
    assert df.iloc[1]["note_type"] == "Basic"


def test_export_cards_to_csv_missing_keys(tmp_path):
    cards = [
        {"front": "Q1", "back": "A1"},
        {"only_front": "Q2"},  # Missing 'back', should be skipped
    ]
    filename = str(tmp_path / "test_missing.csv")
    export_cards_to_csv(cards, filename=filename)

    df = pd.read_csv(filename)
    assert len(df) == 1  # Only one card should be exported


# --- export_cards_to_apkg Tests ---


@patch("genanki.Package.write_to_file")
def test_export_cards_to_apkg_success(mock_write, tmp_path):
    cards = [
        {"Question": "Q1", "Answer": "A1", "note_type": "Basic"},
        {"Question": "Q2", "Answer": "A2", "note_type": "Cloze"},
    ]
    filename = str(tmp_path / "test.apkg")
    result = export_cards_to_apkg(cards, filename=filename)

    assert result == filename
    assert mock_write.called


@patch("genanki.Package.write_to_file")
def test_export_cards_to_apkg_skip_empty_question(mock_write, tmp_path):
    cards = [
        {"Question": "Q1", "Answer": "A1"},
        {"Question": "", "Answer": "A2"},  # Empty question, should skip
    ]
    filename = str(tmp_path / "test_skip.apkg")
    export_cards_to_apkg(cards, filename=filename)
    # genanki objects are a bit hard to inspect, but we mainly check it doesn't crash


def test_export_cards_to_apkg_zero_valid_notes():
    cards = [{"Question": ""}]  # No valid notes
    with pytest.raises(gr.Error, match="Failed to create any valid Anki notes"):
        export_cards_to_apkg(cards)


# --- export_cards_from_crawled_content Tests ---


@patch("ankigen_core.exporters.export_cards_to_csv")
def test_export_cards_from_crawled_content_csv(mock_csv):
    cards = [{"front": "Q", "back": "A"}]
    export_cards_from_crawled_content(cards, export_format="csv")
    mock_csv.assert_called_once()


@patch("ankigen_core.exporters.export_cards_to_apkg")
def test_export_cards_from_crawled_content_apkg(mock_apkg):
    cards = [{"front": "Q", "back": "A"}]
    export_cards_from_crawled_content(cards, export_format="apkg")
    mock_apkg.assert_called_once()


def test_export_cards_from_crawled_content_unsupported():
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_cards_from_crawled_content([{"f": "b"}], export_format="pdf")


# --- DataFrame Export Tests ---


def test_export_dataframe_to_csv_success(tmp_path):
    df = pd.DataFrame({"Question": ["Q1"], "Answer": ["A1"]})
    # We need to mock gr.Info to avoid issues if it requires a running Gradio app
    with patch("gradio.Info"):
        filename = export_dataframe_to_csv(
            df, filename_suggestion=str(tmp_path / "suggested.csv")
        )
        assert filename is not None
        assert os.path.exists(filename)


def test_export_dataframe_to_csv_empty():
    with pytest.raises(gr.Error, match="No card data available"):
        export_dataframe_to_csv(None)


@patch("ankigen_core.exporters.export_cards_to_apkg")
def test_export_dataframe_to_apkg_success(mock_apkg):
    df = pd.DataFrame({"Question": ["Q1"], "Answer": ["A1"], "Card_Type": ["Basic"]})
    export_dataframe_to_apkg(df, output_path="test.apkg", deck_name="Test Deck")
    mock_apkg.assert_called_once()
    # Check that cards were processed correctly
    args, kwargs = mock_apkg.call_args
    processed_cards = args[0]
    assert len(processed_cards) == 1
    assert processed_cards[0]["Question"] == "Q1"
    assert processed_cards[0]["note_type"] == "Basic"
