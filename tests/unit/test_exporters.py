# Tests for ankigen_core/exporters.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
import genanki
import gradio

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
    assert (1 << 30) <= model.model_id < (1 << 31)
    # Ensure model IDs are different (highly likely due to random range)
    assert exporters.BASIC_MODEL.model_id != exporters.CLOZE_MODEL.model_id


# --- export_csv Tests ---


@patch("tempfile.NamedTemporaryFile")
def test_export_csv_success(mock_named_temp_file):
    """Test successful CSV export."""
    # Setup mock temp file
    mock_file = MagicMock()
    mock_file.name = "/tmp/test_anki_cards.csv"
    mock_named_temp_file.return_value.__enter__.return_value = mock_file

    # Create sample DataFrame
    data = {
        "Question": ["Q1"],
        "Answer": ["A1"],
        "Explanation": ["E1"],
        "Example": ["Ex1"],
    }
    df = pd.DataFrame(data)

    # Mock the to_csv method to return a dummy string
    dummy_csv_string = "Question,Answer,Explanation,Example\\nQ1,A1,E1,Ex1"
    df.to_csv = MagicMock(return_value=dummy_csv_string)

    # Call the function
    result_path = exporters.export_csv(df)

    # Assertions
    mock_named_temp_file.assert_called_once_with(
        mode="w+", delete=False, suffix=".csv", encoding="utf-8"
    )
    df.to_csv.assert_called_once_with(index=False)
    mock_file.write.assert_called_once_with(dummy_csv_string)
    assert result_path == mock_file.name


def test_export_csv_none_input():
    """Test export_csv with None input raises gr.Error."""
    with pytest.raises(gradio.Error, match="No card data available"):
        exporters.export_csv(None)


@patch("tempfile.NamedTemporaryFile")
def test_export_csv_empty_dataframe(mock_named_temp_file):
    """Test export_csv with an empty DataFrame raises gr.Error."""
    mock_file = MagicMock()
    mock_file.name = "/tmp/empty_anki_cards.csv"
    mock_named_temp_file.return_value.__enter__.return_value = mock_file

    df = pd.DataFrame()  # Empty DataFrame
    df.to_csv = MagicMock()

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
            1234567890, f"AnkiGen - {subject}"
        )
        mock_deck_and_package["deck_instance"].add_model.assert_any_call(
            exporters.BASIC_MODEL
        )
        mock_deck_and_package["deck_instance"].add_model.assert_any_call(
            exporters.CLOZE_MODEL
        )
        MockNote.assert_called_once_with(
            model=exporters.BASIC_MODEL,
            fields=["Q1", "A1", "E1", "Ex1", "P1", "LO1", "CM1", "Beginner"],
        )
        mock_deck_and_package["deck_instance"].add_note.assert_called_once_with(
            mock_note_instance
        )
        mock_deck_and_package["Package"].assert_called_once_with(
            mock_deck_and_package["deck_instance"]
        )
        mock_deck_and_package["package_instance"].write_to_file.assert_called_once_with(
            "/tmp/test_deck.apkg"
        )

        assert result_file == "/tmp/test_deck.apkg"


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
        expected_extra = (
            "<h3>Answer/Context:</h3> <div>A1</div><hr>\n"
            "<h3>Explanation:</h3> <div>E1</div><hr>\n"
            "<h3>Example:</h3> <pre><code>Ex1</code></pre><hr>\n"
            "<h3>Prerequisites:</h3> <div>P1</div><hr>\n"
            "<h3>Learning Outcomes:</h3> <div>LO1</div><hr>\n"
            "<h3>Common Misconceptions:</h3> <div>CM1</div>"
        )
        MockNote.assert_called_once_with(
            model=exporters.CLOZE_MODEL,
            fields=[
                "This is a {{c1::cloze}} question.",
                expected_extra.strip(),
                "Beginner",
                "Topic1",
            ],
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
        mock_deck_and_package["Deck"].assert_called_with(ANY, "AnkiGen Deck")

        exporters.export_deck(df, "   ")  # Subject is whitespace
        mock_deck_and_package["Deck"].assert_called_with(ANY, "AnkiGen Deck")


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
        pytest.raises(gradio.Error, match="Failed to create any valid Anki notes"),
    ):
        exporters.export_deck(df, "No Notes Test")


# Original placeholder removed
# def test_placeholder_exporters():
#     assert True
