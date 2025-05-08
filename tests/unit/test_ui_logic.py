# Tests for ankigen_core/ui_logic.py
import pytest
import pandas as pd
import gradio as gr
from unittest.mock import patch

# Module to test
from ankigen_core import ui_logic

# --- update_mode_visibility Tests ---


@pytest.mark.parametrize(
    "mode, expected_visibility",
    [
        (
            "subject",
            {
                "subject": True,
                "path": False,
                "text": False,
                "web": False,
                "cards": True,
                "path_res": False,
            },
        ),
        (
            "path",
            {
                "subject": False,
                "path": True,
                "text": False,
                "web": False,
                "cards": False,
                "path_res": True,
            },
        ),
        (
            "text",
            {
                "subject": False,
                "path": False,
                "text": True,
                "web": False,
                "cards": True,
                "path_res": False,
            },
        ),
        (
            "web",
            {
                "subject": False,
                "path": False,
                "text": False,
                "web": True,
                "cards": True,
                "path_res": False,
            },
        ),
        (
            "invalid",
            {
                "subject": False,
                "path": False,
                "text": False,
                "web": False,
                "cards": False,
                "path_res": False,
            },
        ),
    ],
)
def test_update_mode_visibility_group_visibility(mode, expected_visibility):
    """Test visibility updates for different modes."""
    result = ui_logic.update_mode_visibility(mode, "s", "d", "t", "u")

    # Check visibility of mode-specific input groups
    assert result["subject_mode_group"]["visible"] == expected_visibility["subject"]
    assert result["path_mode_group"]["visible"] == expected_visibility["path"]
    assert result["text_mode_group"]["visible"] == expected_visibility["text"]
    assert result["web_mode_group"]["visible"] == expected_visibility["web"]

    # Check visibility of output groups
    assert result["cards_output_group"]["visible"] == expected_visibility["cards"]
    assert result["path_results_group"]["visible"] == expected_visibility["path_res"]


def test_update_mode_visibility_value_persistence():
    """Test that input values are preserved for the selected mode and cleared otherwise."""
    subject_val = "Test Subject"
    desc_val = "Test Description"
    text_val = "Test Text"
    url_val = "http://test.com"

    # Subject mode - Subject should persist, others clear
    result = ui_logic.update_mode_visibility(
        "subject", subject_val, desc_val, text_val, url_val
    )
    assert result["subject_textbox"]["value"] == subject_val
    assert result["description_textbox"]["value"] == ""
    assert result["source_text_textbox"]["value"] == ""
    assert result["url_textbox"]["value"] == ""

    # Path mode - Description should persist, others clear
    result = ui_logic.update_mode_visibility(
        "path", subject_val, desc_val, text_val, url_val
    )
    assert result["subject_textbox"]["value"] == ""
    assert result["description_textbox"]["value"] == desc_val
    assert result["source_text_textbox"]["value"] == ""
    assert result["url_textbox"]["value"] == ""

    # Text mode - Text should persist, others clear
    result = ui_logic.update_mode_visibility(
        "text", subject_val, desc_val, text_val, url_val
    )
    assert result["subject_textbox"]["value"] == ""
    assert result["description_textbox"]["value"] == ""
    assert result["source_text_textbox"]["value"] == text_val
    assert result["url_textbox"]["value"] == ""

    # Web mode - URL should persist, others clear
    result = ui_logic.update_mode_visibility(
        "web", subject_val, desc_val, text_val, url_val
    )
    assert result["subject_textbox"]["value"] == ""
    assert result["description_textbox"]["value"] == ""
    assert result["source_text_textbox"]["value"] == ""
    assert result["url_textbox"]["value"] == url_val


def test_update_mode_visibility_clears_outputs():
    """Test that changing mode always clears output components."""
    result = ui_logic.update_mode_visibility("subject", "s", "d", "t", "u")
    assert result["output_dataframe"]["value"] is None
    assert result["subjects_dataframe"]["value"] is None
    assert result["learning_order_markdown"]["value"] == ""
    assert result["projects_markdown"]["value"] == ""
    assert result["progress_html"]["value"] == ""
    assert result["progress_html"]["visible"] is False
    assert result["total_cards_number"]["value"] == 0
    assert result["total_cards_number"]["visible"] is False


# --- use_selected_subjects Tests ---


def test_use_selected_subjects_success():
    """Test successful transition using subjects DataFrame."""
    data = {
        "Subject": ["Subj A", "Subj B"],
        "Prerequisites": ["P1", "P2"],
        "Time Estimate": ["T1", "T2"],
    }
    df = pd.DataFrame(data)

    result = ui_logic.use_selected_subjects(df)

    # Check mode switch
    assert result["generation_mode_radio"] == "subject"
    assert result["subject_mode_group"]["visible"] is True
    assert result["path_mode_group"]["visible"] is False
    assert result["text_mode_group"]["visible"] is False
    assert result["web_mode_group"]["visible"] is False
    assert result["path_results_group"]["visible"] is False  # Path results hidden
    assert result["cards_output_group"]["visible"] is True  # Card output shown

    # Check input population
    assert result["subject_textbox"] == "Subj A, Subj B"
    assert result["topic_number_slider"] == 3  # len(subjects) + 1
    assert (
        "connections between these subjects" in result["preference_prompt_textbox"]
    )  # Check suggested prompt

    # Check clearing of other inputs/outputs
    assert result["description_textbox"] == ""
    assert result["source_text_textbox"] == ""
    assert result["url_textbox"] == ""
    assert result["output_dataframe"]["value"] is None
    assert result["subjects_dataframe"] is df  # Check if it returns the df directly


@patch("gradio.Warning")
def test_use_selected_subjects_none_input(mock_gr_warning):
    """Test behavior with None input."""
    result = ui_logic.use_selected_subjects(None)

    mock_gr_warning.assert_called_once_with(
        "No subjects available to copy from Learning Path analysis."
    )
    # Check that it returns updates, but they are likely no-op (default gr.update())
    assert isinstance(result, dict)
    assert "generation_mode_radio" in result
    assert (
        result["generation_mode_radio"] == gr.update()
    )  # Default update means no change


@patch("gradio.Warning")
def test_use_selected_subjects_empty_dataframe(mock_gr_warning):
    """Test behavior with an empty DataFrame."""
    df = pd.DataFrame()
    result = ui_logic.use_selected_subjects(df)

    mock_gr_warning.assert_called_once_with(
        "No subjects available to copy from Learning Path analysis."
    )
    assert isinstance(result, dict)
    assert result["generation_mode_radio"] == gr.update()


@patch("gradio.Error")
def test_use_selected_subjects_missing_column(mock_gr_error):
    """Test behavior when DataFrame is missing the 'Subject' column."""
    df = pd.DataFrame({"WrongColumn": ["Data"]})
    result = ui_logic.use_selected_subjects(df)

    mock_gr_error.assert_called_once_with(
        "Learning path analysis result is missing the 'Subject' column."
    )
    assert isinstance(result, dict)
    assert result["generation_mode_radio"] == gr.update()
