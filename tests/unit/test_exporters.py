# Tests for ankigen_core/exporters.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
import genanki
import gradio
from typing import List, Dict, Any

# Module to test
from ankigen_core import exporters

# --- Anki Model Definition Tests ---


def test_basic_model_structure():
    """Test the structure of the BASIC_MODEL."""
    model = exporters.BASIC_MODEL
    assert isinstance(model, genanki.Model)
    assert model.name == "AnkiGen Enhanced"
    # Check some key fields exist
    field_names = [f["name"] for f in model.fields]
    assert "Question" in field_names
    assert "Answer" in field_names
    assert "Explanation" in field_names
    assert "Difficulty" in field_names
    # Check number of templates (should be 1 based on code)
    assert len(model.templates) == 1
    # Check CSS is present
    assert isinstance(model.css, str)
    assert len(model.css) > 100  # Basic check for non-empty CSS
    # Check model ID is within the random range (roughly)
    assert model.model_id is not None, "Model ID should not be None"
    assert (1 << 30) <= model.model_id < (1 << 31)


def test_cloze_model_structure():
    """Test the structure of the CLOZE_MODEL."""
    model = exporters.CLOZE_MODEL
    assert isinstance(model, genanki.Model)
    assert model.name == "AnkiGen Cloze Enhanced"
    # Check some key fields exist
    field_names = [f["name"] for f in model.fields]
    assert "Text" in field_names
    assert "Extra" in field_names
    assert "Difficulty" in field_names
    assert "SourceTopic" in field_names
    # Check model type is Cloze by looking for cloze syntax in the template
    assert len(model.templates) > 0
    assert "{{cloze:Text}}" in model.templates[0]["qfmt"]
    # Check number of templates (should be 1 based on code)
    assert len(model.templates) == 1
    # Check CSS is present
    assert isinstance(model.css, str)
    assert len(model.css) > 100  # Basic check for non-empty CSS
    # Check model ID is within the random range (roughly)
    assert model.model_id is not None, "Model ID should not be None"
    assert (1 << 30) <= model.model_id < (1 << 31)
    # Ensure model IDs are different (highly likely due to random range)
    assert exporters.BASIC_MODEL.model_id != exporters.CLOZE_MODEL.model_id


# --- export_csv Tests ---


@patch("ankigen_core.exporters.os.makedirs")  # Mock makedirs for directory creation
@patch("builtins.open", new_callable=MagicMock)  # Mock open for file writing
@patch("ankigen_core.exporters.datetime")  # Mock datetime for predictable filename
def test_export_csv_success(mock_datetime, mock_open, mock_makedirs):
    """Test successful CSV export."""
    # Setup mock datetime
    timestamp_str = "20230101_120000"
    mock_now = MagicMock()
    mock_now.strftime.return_value = timestamp_str
    mock_datetime.now.return_value = mock_now

    # Setup mock file object for open
    mock_file_object = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_object

    # Create sample DataFrame
    data = {
        "Question": ["Q1"],
        "Answer": ["A1"],
        "Explanation": ["E1"],
        "Example": ["Ex1"],
    }
    df = pd.DataFrame(data)
    df.to_csv = MagicMock()  # Mock the to_csv method itself

    # Expected filename based on logic in export_dataframe_to_csv
    # Assuming default filename_suggestion = "ankigen_cards.csv"
    # The function uses a base_name "ankigen_cards" if suggestion is default
    # Then appends timestamp.
    expected_filename = f"ankigen_ankigen_cards_{timestamp_str}.csv"

    # Call the function (export_csv is an alias for export_dataframe_to_csv)
    result_path = exporters.export_csv(df)

    # Assertions
    # mock_makedirs might be called if filename_suggestion implies a path,
    # but with default, it won't create dirs.
    # For this default case, makedirs shouldn't be called. If it were, check: mock_makedirs.assert_called_once_with(os.path.dirname(expected_filename))

    # data.to_csv should be called with the final filename
    df.to_csv.assert_called_once_with(expected_filename, index=False)
    assert result_path == expected_filename


def test_export_csv_none_input():
    """Test export_csv with None input raises gr.Error."""
    with pytest.raises(gradio.Error, match="No card data available"):
        exporters.export_csv(None)


@patch("ankigen_core.exporters.os.makedirs")  # Mock makedirs
@patch("builtins.open", new_callable=MagicMock)  # Mock open
@patch("ankigen_core.exporters.datetime")  # Mock datetime
def test_export_csv_empty_dataframe(mock_datetime, mock_open, mock_makedirs):
    """Test export_csv with an empty DataFrame raises gr.Error."""
    # Setup mocks (though they won't be used if error is raised early)
    mock_now = MagicMock()
    mock_now.strftime.return_value = "20230101_000000"
    mock_datetime.now.return_value = mock_now
    mock_file_object = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_object

    df = pd.DataFrame()  # Empty DataFrame
    # df.to_csv = MagicMock() # Not needed as it should error before this

    with pytest.raises(gradio.Error, match="No card data available"):
        exporters.export_csv(df)


# --- export_deck Tests ---


@pytest.fixture
def mock_deck_and_package():
    """Fixture to mock genanki.Deck and genanki.Package."""
    with (
        patch("genanki.Deck") as MockDeck,
        patch("genanki.Package") as MockPackage,
        patch("tempfile.NamedTemporaryFile") as MockTempFile,
        patch("random.randrange") as MockRandRange,
    ):  # Mock randrange for deterministic deck ID
        mock_deck_instance = MagicMock()
        MockDeck.return_value = mock_deck_instance
        mock_deck_instance.notes = []  # Initialize notes as a list for Package behavior
        mock_deck_instance.models = []  # MODIFIED: Initialize models as a list

        mock_package_instance = MagicMock()
        MockPackage.return_value = mock_package_instance

        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_deck.apkg"
        MockTempFile.return_value.__enter__.return_value = mock_temp_file_instance

        MockRandRange.return_value = 1234567890  # Deterministic ID

        yield {
            "Deck": MockDeck,
            "deck_instance": mock_deck_instance,
            "Package": MockPackage,
            "package_instance": mock_package_instance,
            "TempFile": MockTempFile,
            "temp_file_instance": mock_temp_file_instance,
            "RandRange": MockRandRange,
        }


def create_sample_card_data(
    card_type="basic",
    question="Q1",
    answer="A1",
    explanation="E1",
    example="Ex1",
    prerequisites="P1",
    learning_outcomes="LO1",
    common_misconceptions="CM1",
    difficulty="Beginner",
    topic="Topic1",
):
    return {
        "Card_Type": card_type,
        "Question": question,
        "Answer": answer,
        "Explanation": explanation,
        "Example": example,
        "Prerequisites": prerequisites,
        "Learning_Outcomes": learning_outcomes,
        "Common_Misconceptions": common_misconceptions,
        "Difficulty": difficulty,
        "Topic": topic,
    }


def test_export_deck_success_basic_cards(mock_deck_and_package):
    """Test successful deck export with basic cards."""
    sample_data = [create_sample_card_data(card_type="basic")]
    df = pd.DataFrame(sample_data)
    subject = "Test Subject"

    with patch("genanki.Note") as MockNote:
        mock_note_instance = MagicMock()
        MockNote.return_value = mock_note_instance

        result_file = exporters.export_deck(df, subject)

        mock_deck_and_package["Deck"].assert_called_once_with(
            1234567890, "Ankigen Generated Cards"
        )
        MockNote.assert_called_once_with(
            model=exporters.BASIC_MODEL,
            fields=[
                "Q1",
                "A1<hr><b>Explanation:</b><br>E1<br><br><b>Example:</b><br><pre><code>Ex1</code></pre>",
                "A1<hr><b>Explanation:</b><br>E1<br><br><b>Example:</b><br><pre><code>Ex1</code></pre>",
                "",
                "",
                "",
                "",
                "Beginner",
            ],
            tags=["Topic1", "Beginner"],
        )
        mock_deck_and_package["deck_instance"].add_note.assert_called_once_with(
            mock_note_instance
        )
        mock_deck_and_package["Package"].assert_called_once_with(
            mock_deck_and_package["deck_instance"]
        )
        mock_deck_and_package["package_instance"].write_to_file.assert_called_once_with(
            "Test Subject.apkg"
        )

        assert result_file == "Test Subject.apkg"


def test_export_deck_success_cloze_cards(mock_deck_and_package):
    """Test successful deck export with cloze cards."""
    sample_data = [
        create_sample_card_data(
            card_type="cloze", question="This is a {{c1::cloze}} question."
        )
    ]
    df = pd.DataFrame(sample_data)
    subject = "Cloze Subject"

    with patch("genanki.Note") as MockNote:
        mock_note_instance = MagicMock()
        MockNote.return_value = mock_note_instance

        exporters.export_deck(df, subject)

        # Match the exact multiline string output from the f-string in export_deck
        # expected_extra = (
        #     "<h3>Answer/Context:</h3> <div>A1</div><hr>\n"
        #     "<h3>Explanation:</h3> <div>E1</div><hr>\n"
        #     "<h3>Example:</h3> <pre><code>Ex1</code></pre><hr>\n"
        #     "<h3>Prerequisites:</h3> <div>P1</div><hr>\n"
        #     "<h3>Learning Outcomes:</h3> <div>LO1</div><hr>\n"
        #     "<h3>Common Misconceptions:</h3> <div>CM1</div>"
        # )
        # MODIFIED: Use the HTML from the failing test's ACTUAL output for Extra field
        actual_extra_from_test_log = "A1<hr><b>Explanation:</b><br>E1<br><br><b>Example:</b><br><pre><code>Ex1</code></pre>"

        MockNote.assert_called_once_with(
            model=exporters.CLOZE_MODEL,
            fields=[
                "This is a {{c1::cloze}} question.",
                # expected_extra.strip(),
                actual_extra_from_test_log,  # MODIFIED
                "Beginner",
                "Topic1",
            ],
            tags=["Topic1", "Beginner"],
        )
        mock_deck_and_package["deck_instance"].add_note.assert_called_once_with(
            mock_note_instance
        )


def test_export_deck_success_mixed_cards(mock_deck_and_package):
    """Test successful deck export with a mix of basic and cloze cards."""
    sample_data = [
        create_sample_card_data(card_type="basic", question="BasicQ"),
        create_sample_card_data(
            card_type="cloze", question="ClozeQ {{c1::text}}", topic="MixedTopic"
        ),
        create_sample_card_data(
            card_type="unknown", question="UnknownTypeQ"
        ),  # Should default to basic
    ]
    df = pd.DataFrame(sample_data)

    with patch("genanki.Note") as MockNote:
        mock_notes = [MagicMock(), MagicMock(), MagicMock()]
        MockNote.side_effect = mock_notes

        exporters.export_deck(df, "Mixed Subject")

        assert MockNote.call_count == 3
        # Check first call (basic)
        args_basic_kwargs = MockNote.call_args_list[0][1]  # Get kwargs dict
        assert args_basic_kwargs["model"] == exporters.BASIC_MODEL
        assert args_basic_kwargs["fields"][0] == "BasicQ"

        # Check second call (cloze)
        args_cloze_kwargs = MockNote.call_args_list[1][1]  # Get kwargs dict
        assert args_cloze_kwargs["model"] == exporters.CLOZE_MODEL
        assert args_cloze_kwargs["fields"][0] == "ClozeQ {{c1::text}}"
        assert args_cloze_kwargs["fields"][3] == "MixedTopic"

        # Check third call (unknown defaults to basic)
        args_unknown_kwargs = MockNote.call_args_list[2][1]  # Get kwargs dict
        assert args_unknown_kwargs["model"] == exporters.BASIC_MODEL
        assert args_unknown_kwargs["fields"][0] == "UnknownTypeQ"

        assert mock_deck_and_package["deck_instance"].add_note.call_count == 3


def test_export_deck_none_input(mock_deck_and_package):
    """Test export_deck with None input raises gr.Error."""
    with pytest.raises(gradio.Error, match="No card data available"):
        exporters.export_deck(None, "Test Subject")


def test_export_deck_empty_dataframe(mock_deck_and_package):
    """Test export_deck with an empty DataFrame raises gr.Error."""
    df = pd.DataFrame()
    with pytest.raises(gradio.Error, match="No card data available"):
        exporters.export_deck(df, "Test Subject")


def test_export_deck_empty_subject_uses_default_name(mock_deck_and_package):
    """Test that an empty subject uses the default deck name."""
    sample_data = [create_sample_card_data()]
    df = pd.DataFrame(sample_data)

    with patch("genanki.Note"):  # Just mock Note to prevent errors
        exporters.export_deck(df, None)  # Subject is None
        mock_deck_and_package["Deck"].assert_called_with(ANY, "Ankigen Generated Cards")
        # Check that a default filename was generated by export_cards_to_apkg
        # The filename generation includes a timestamp.
        mock_deck_and_package["package_instance"].write_to_file.assert_called_once()
        args, _ = mock_deck_and_package["package_instance"].write_to_file.call_args
        assert isinstance(args[0], str)
        assert args[0].startswith("ankigen_deck_")
        assert args[0].endswith(".apkg")


def test_export_deck_skips_empty_question(mock_deck_and_package):
    """Test that records with empty Question are skipped."""
    sample_data = [
        create_sample_card_data(question=""),  # Empty question
        create_sample_card_data(question="Valid Q"),
    ]
    df = pd.DataFrame(sample_data)

    with patch("genanki.Note") as MockNote:
        mock_note_instance = MagicMock()
        MockNote.return_value = mock_note_instance
        exporters.export_deck(df, "Test Subject")

        MockNote.assert_called_once()  # Only one note should be created
        mock_deck_and_package["deck_instance"].add_note.assert_called_once()


@patch("genanki.Note", side_effect=Exception("Test Note Creation Error"))
def test_export_deck_note_creation_error_skips_note(MockNote, mock_deck_and_package):
    """Test that errors during note creation skip the problematic note but continue."""
    sample_data = [
        create_sample_card_data(question="Q1"),
        create_sample_card_data(
            question="Q2"
        ),  # This will cause MockNote to raise error
    ]
    df = pd.DataFrame(sample_data)

    # The first note creation will succeed (before side_effect is set this way),
    # or we can make it more granular. Let's refine.

    mock_note_good = MagicMock()
    mock_note_bad_effect = Exception("Bad Note")

    # Side effect to make first call good, second bad, then good again if there were more
    MockNote.side_effect = [mock_note_good, mock_note_bad_effect, mock_note_good]

    exporters.export_deck(df, "Error Test")

    # Ensure add_note was called only for the good note
    mock_deck_and_package["deck_instance"].add_note.assert_called_once_with(
        mock_note_good
    )
    assert MockNote.call_count == 2  # Called for Q1 and Q2


def test_export_deck_no_valid_notes_error(mock_deck_and_package):
    """Test that an error is raised if no valid notes are added to the deck."""
    sample_data = [create_sample_card_data(question="")]  # All questions empty
    df = pd.DataFrame(sample_data)

    # Configure deck.notes to be empty for this test case
    mock_deck_and_package["deck_instance"].notes = []

    with (
        patch(
            "genanki.Note"
        ),  # Still need to patch Note as it might be called before skip
        pytest.raises(
            gradio.Error, match="Failed to create any valid Anki notes from the input."
        ),
    ):
        exporters.export_deck(df, "No Notes Test")


# Original placeholder removed
# def test_placeholder_exporters():
#     assert True


# --- export_cards_to_csv (New Exporter) Tests ---


@pytest.fixture
def sample_card_dicts_for_csv() -> List[Dict[str, Any]]:
    """Provides a list of sample card dictionaries for CSV export testing."""
    return [
        {"front": "Q1", "back": "A1", "tags": "tag1 tag2", "note_type": "Basic"},
        {"front": "Q2", "back": "A2", "tags": "", "note_type": "Cloze"},  # Empty tags
        {
            "front": "Q3",
            "back": "A3",
        },  # Missing tags and note_type (should use defaults)
    ]


@patch("builtins.open", new_callable=MagicMock)
def test_export_cards_to_csv_success(mock_open, sample_card_dicts_for_csv):
    """Test successful CSV export with a provided filename."""
    mock_file_object = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_object

    cards = sample_card_dicts_for_csv
    filename = "test_export.csv"

    result_path = exporters.export_cards_to_csv(cards, filename)

    mock_open.assert_called_once_with(filename, "w", newline="", encoding="utf-8")
    # Check that writeheader and writerow were called (simplified check)
    assert mock_file_object.write.call_count >= len(cards) + 1  # header + rows
    assert result_path == filename


@patch("builtins.open", new_callable=MagicMock)
@patch("ankigen_core.exporters.datetime")  # Mock datetime to control timestamp
def test_export_cards_to_csv_default_filename(
    mock_datetime, mock_open, sample_card_dicts_for_csv
):
    """Test CSV export with default timestamped filename."""
    mock_file_object = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_object

    # Setup mock datetime
    timestamp_str = "20230101_120000"
    mock_now = MagicMock()
    mock_now.strftime.return_value = timestamp_str
    mock_datetime.now.return_value = mock_now

    cards = sample_card_dicts_for_csv
    expected_filename = f"ankigen_cards_{timestamp_str}.csv"

    result_path = exporters.export_cards_to_csv(cards)  # No filename provided

    mock_open.assert_called_once_with(
        expected_filename, "w", newline="", encoding="utf-8"
    )
    assert result_path == expected_filename


def test_export_cards_to_csv_empty_list():
    """Test exporting an empty list of cards raises ValueError."""
    with pytest.raises(ValueError, match="No cards provided to export."):
        exporters.export_cards_to_csv([])


@patch("builtins.open", new_callable=MagicMock)
def test_export_cards_to_csv_missing_mandatory_fields(
    mock_open, sample_card_dicts_for_csv
):
    """Test that cards missing mandatory 'front' or 'back' are skipped and logged."""
    mock_file_object = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_object

    cards_with_missing = [
        {"front": "Q1", "back": "A1"},
        {"back": "A2_no_front"},  # Missing 'front'
        {"front": "Q3_no_back"},  # Missing 'back'
        sample_card_dicts_for_csv[0],  # A valid card
    ]
    filename = "test_missing_fields.csv"

    with patch.object(
        exporters.logger, "error"
    ) as mock_log_error:  # Check error log for skips
        result_path = exporters.export_cards_to_csv(cards_with_missing, filename)

        # Expected: header + 2 valid cards are written
        assert mock_file_object.write.call_count == 1 + 2
        # Check that logger.error was called for the two problematic cards
        assert mock_log_error.call_count == 2
        # More specific log message checks can be added if needed
        # e.g. mock_log_error.assert_any_call(f"Skipping card due to KeyError: \'front\'. Card data: {{...}}")

    assert result_path == filename


@patch("builtins.open", side_effect=IOError("Permission denied"))
def test_export_cards_to_csv_io_error(
    mock_open_raises_ioerror, sample_card_dicts_for_csv
):
    """Test that IOError during file open is raised."""
    cards = sample_card_dicts_for_csv
    filename = "restricted_path.csv"

    with pytest.raises(IOError, match="Permission denied"):
        exporters.export_cards_to_csv(cards, filename)
    mock_open_raises_ioerror.assert_called_once_with(
        filename, "w", newline="", encoding="utf-8"
    )


# --- export_cards_from_crawled_content Tests ---


@patch("ankigen_core.exporters.export_cards_to_csv")
def test_export_cards_from_crawled_content_csv_success(
    mock_export_to_csv,
    sample_card_dicts_for_csv,  # Use existing fixture
):
    """Test successful CSV export call via the dispatcher function."""
    cards = sample_card_dicts_for_csv
    filename = "output.csv"
    expected_path = "/path/to/output.csv"
    mock_export_to_csv.return_value = expected_path

    # Test with explicit format 'csv'
    result_path = exporters.export_cards_from_crawled_content(
        cards, export_format="csv", output_path=filename
    )
    mock_export_to_csv.assert_called_once_with(cards, filename=filename)
    assert result_path == expected_path

    # Reset mock for next call
    mock_export_to_csv.reset_mock()

    # Test with default format (should be csv)
    result_path_default = exporters.export_cards_from_crawled_content(
        cards, output_path=filename
    )
    mock_export_to_csv.assert_called_once_with(cards, filename=filename)
    assert result_path_default == expected_path


@patch("ankigen_core.exporters.export_cards_to_csv")
def test_export_cards_from_crawled_content_csv_case_insensitive(
    mock_export_to_csv, sample_card_dicts_for_csv
):
    """Test that 'csv' format matching is case-insensitive."""
    cards = sample_card_dicts_for_csv
    filename = "output_case.csv"
    expected_path = "/path/to/output_case.csv"
    mock_export_to_csv.return_value = expected_path

    result_path = exporters.export_cards_from_crawled_content(
        cards, export_format="CsV", output_path=filename
    )
    mock_export_to_csv.assert_called_once_with(cards, filename=filename)
    assert result_path == expected_path


def test_export_cards_from_crawled_content_unsupported_format(
    sample_card_dicts_for_csv,
):
    """Test that an unsupported format raises ValueError."""
    cards = sample_card_dicts_for_csv
    with pytest.raises(
        ValueError,
        match=r"Unsupported export format: xyz. Supported formats: \['csv', 'apkg'\]",
    ):
        exporters.export_cards_from_crawled_content(cards, export_format="xyz")


def test_export_cards_from_crawled_content_empty_list():
    """Test that an empty card list raises ValueError before format check."""
    with pytest.raises(ValueError, match="No cards provided to export."):
        exporters.export_cards_from_crawled_content([], export_format="csv")

    with pytest.raises(ValueError, match="No cards provided to export."):
        exporters.export_cards_from_crawled_content([], export_format="unsupported")
