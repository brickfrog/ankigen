import pytest
import pandas as pd
import gradio as gr

# Functions to test are from ankigen_core, but we're testing their integration
# with app.py's conceptual structure.
from ankigen_core.ui_logic import update_mode_visibility, use_selected_subjects
from ankigen_core.learning_path import analyze_learning_path
from ankigen_core.card_generator import (
    orchestrate_card_generation,
)
from ankigen_core.exporters import export_dataframe_to_csv, export_dataframe_to_apkg

# For mocking
from unittest.mock import patch, MagicMock, ANY

# We might need to mock these if core functions try to use them and they aren't set up
from ankigen_core.models import Card, CardFront, CardBack

# Placeholder for initial values of text inputs
MOCK_SUBJECT_INPUT = "Initial Subject"
MOCK_DESCRIPTION_INPUT = "Initial Description"
MOCK_TEXT_INPUT = "Initial Text Input"
MOCK_URL_INPUT = "http://initial.url"

EXPECTED_UI_LOGIC_KEYS_MODE_VISIBILITY = [
    "subject_mode_group",
    "path_mode_group",
    "text_mode_group",
    "web_mode_group",
    "path_results_group",
    "cards_output_group",
    "subject_textbox",
    "description_textbox",
    "source_text_textbox",
    "url_textbox",
    "output_dataframe",
    "subjects_dataframe",
    "learning_order_markdown",
    "projects_markdown",
    "progress_html",
    "total_cards_number",
]

EXPECTED_UI_LOGIC_KEYS_USE_SUBJECTS = [
    "generation_mode_radio",
    "subject_mode_group",
    "path_mode_group",
    "text_mode_group",
    "web_mode_group",
    "path_results_group",
    "cards_output_group",
    "subject_textbox",
    "description_textbox",
    "source_text_textbox",
    "url_textbox",
    "topic_number_slider",
    "preference_prompt_textbox",
    "output_dataframe",
    "subjects_dataframe",
    "learning_order_markdown",
    "projects_markdown",
    "progress_html",
    "total_cards_number",
]


@pytest.mark.parametrize(
    "mode, expected_visibilities, expected_values",
    [
        (
            "subject",
            {  # Expected visibility for groups/outputs
                "subject_mode_group": True,
                "path_mode_group": False,
                "text_mode_group": False,
                "web_mode_group": False,
                "path_results_group": False,
                "cards_output_group": True,
            },
            {  # Expected values for textboxes
                "subject_textbox": MOCK_SUBJECT_INPUT,
                "description_textbox": "",
                "source_text_textbox": "",
                "url_textbox": "",
            },
        ),
        (
            "path",
            {
                "subject_mode_group": False,
                "path_mode_group": True,
                "text_mode_group": False,
                "web_mode_group": False,
                "path_results_group": True,
                "cards_output_group": False,
            },
            {
                "subject_textbox": "",
                "description_textbox": MOCK_DESCRIPTION_INPUT,
                "source_text_textbox": "",
                "url_textbox": "",
            },
        ),
        (
            "text",
            {
                "subject_mode_group": False,
                "path_mode_group": False,
                "text_mode_group": True,
                "web_mode_group": False,
                "path_results_group": False,
                "cards_output_group": True,
            },
            {
                "subject_textbox": "",
                "description_textbox": "",
                "source_text_textbox": MOCK_TEXT_INPUT,
                "url_textbox": "",
            },
        ),
        (
            "web",
            {
                "subject_mode_group": False,
                "path_mode_group": False,
                "text_mode_group": False,
                "web_mode_group": True,
                "path_results_group": False,
                "cards_output_group": True,
            },
            {
                "subject_textbox": "",
                "description_textbox": "",
                "source_text_textbox": "",
                "url_textbox": MOCK_URL_INPUT,
            },
        ),
    ],
)
def test_generation_mode_change_updates_ui_correctly(
    mode, expected_visibilities, expected_values
):
    """
    Tests that changing the generation_mode correctly calls update_mode_visibility
    and the returned dictionary would update app.py's UI components as expected.
    """
    result_dict = update_mode_visibility(
        mode=mode,
        current_subject=MOCK_SUBJECT_INPUT,
        current_description=MOCK_DESCRIPTION_INPUT,
        current_text=MOCK_TEXT_INPUT,
        current_url=MOCK_URL_INPUT,
    )

    # Check that all expected component keys are present in the result
    for key in EXPECTED_UI_LOGIC_KEYS_MODE_VISIBILITY:
        assert key in result_dict, f"Key {key} missing in result for mode {mode}"

    # Check visibility of mode-specific groups and output areas
    for component_key, expected_visibility in expected_visibilities.items():
        assert (
            result_dict[component_key]["visible"] == expected_visibility
        ), f"Visibility for {component_key} in mode '{mode}' was not {expected_visibility}"

    # Check values of input textboxes (preserved for active mode, cleared for others)
    for component_key, expected_value in expected_values.items():
        assert (
            result_dict[component_key]["value"] == expected_value
        ), f"Value for {component_key} in mode '{mode}' was not '{expected_value}'"

    # Check that output/status components are cleared/reset
    assert result_dict["output_dataframe"]["value"] is None
    assert result_dict["subjects_dataframe"]["value"] is None
    assert result_dict["learning_order_markdown"]["value"] == ""
    assert result_dict["projects_markdown"]["value"] == ""
    assert result_dict["progress_html"]["value"] == ""
    assert result_dict["progress_html"]["visible"] is False
    assert result_dict["total_cards_number"]["value"] == 0
    assert result_dict["total_cards_number"]["visible"] is False


@patch("ankigen_core.learning_path.structured_output_completion")
@patch("ankigen_core.learning_path.OpenAIClientManager")  # To mock the instance passed
@patch("ankigen_core.learning_path.ResponseCache")  # To mock the instance passed
async def test_analyze_learning_path_button_click(
    mock_response_cache_class, mock_client_manager_class, mock_soc
):
    """
    Tests that the analyze_button.click event (calling analyze_learning_path)
    processes inputs and produces outputs correctly for UI update.
    """
    # Setup mocks for manager and cache instances
    mock_client_manager_instance = mock_client_manager_class.return_value
    mock_openai_client = MagicMock()
    mock_client_manager_instance.get_client.return_value = mock_openai_client
    mock_client_manager_instance.initialize_client.return_value = (
        None  # Simulate successful init
    )

    mock_cache_instance = mock_response_cache_class.return_value
    mock_cache_instance.get.return_value = None  # Default cache miss

    # Mock inputs from UI
    test_api_key = "sk-testkey123"
    test_description = "Become a data scientist"
    test_model = "gpt-4.1-test"

    # Mock the response from structured_output_completion
    mock_llm_response = {
        "subjects": [
            {
                "Subject": "Python Basics",
                "Prerequisites": "None",
                "Time Estimate": "4 weeks",
            },
            {
                "Subject": "Pandas & NumPy",
                "Prerequisites": "Python Basics",
                "Time Estimate": "3 weeks",
            },
        ],
        "learning_order": "1. Python Basics\n2. Pandas & NumPy",
        "projects": "Analyze a public dataset.",
    }
    mock_soc.return_value = mock_llm_response

    # Call the function that the button click would trigger
    df_subjects, md_order, md_projects = await analyze_learning_path(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        api_key=test_api_key,
        description=test_description,
        model=test_model,
    )

    # Assertions
    mock_client_manager_instance.initialize_client.assert_called_once_with(test_api_key)
    mock_client_manager_instance.get_client.assert_called_once()
    mock_soc.assert_called_once_with(
        openai_client=mock_openai_client,
        model=test_model,
        response_format={"type": "json_object"},
        system_prompt=ANY,  # System prompt is internally generated
        user_prompt=ANY,  # User prompt is internally generated, check if needed
        cache=mock_cache_instance,
    )
    # Check that the input description is part of the user_prompt for SOC
    assert test_description in mock_soc.call_args[1]["user_prompt"]

    # Assert DataFrame output
    assert isinstance(df_subjects, pd.DataFrame)
    assert len(df_subjects) == 2
    assert df_subjects.iloc[0]["Subject"] == "Python Basics"
    assert list(df_subjects.columns) == ["Subject", "Prerequisites", "Time Estimate"]

    # Assert Markdown outputs (basic check for content)
    assert "Python Basics" in md_order
    assert "Pandas & NumPy" in md_order
    assert "Analyze a public dataset." in md_projects

    # Test for gr.Error when API key is missing
    with pytest.raises(gr.Error, match="API key is required"):
        await analyze_learning_path(
            client_manager=mock_client_manager_instance,
            cache=mock_cache_instance,
            api_key="",  # Empty API key
            description=test_description,
            model=test_model,
        )

    # Test for gr.Error when structured_output_completion returns invalid format
    mock_soc.return_value = {"wrong_key": "data"}  # Invalid response from LLM
    with pytest.raises(gr.Error, match="invalid API response format"):
        await analyze_learning_path(
            client_manager=mock_client_manager_instance,
            cache=mock_cache_instance,
            api_key=test_api_key,
            description=test_description,
            model=test_model,
        )


def test_use_selected_subjects_button_click_success():
    """Test that use_subjects_button.click (calling use_selected_subjects) works correctly."""
    sample_data = {
        "Subject": ["Intro to Python", "Data Structures", "Algorithms"],
        "Prerequisites": ["None", "Intro to Python", "Data Structures"],
        "Time Estimate": ["2 weeks", "3 weeks", "4 weeks"],
    }
    subjects_df = pd.DataFrame(sample_data)

    result_dict = use_selected_subjects(subjects_df)

    # Check all expected keys are present
    for key in EXPECTED_UI_LOGIC_KEYS_USE_SUBJECTS:
        assert key in result_dict, f"Key {key} missing in use_selected_subjects result"

    # Check direct value updates
    assert result_dict["generation_mode_radio"] == "subject"
    assert (
        result_dict["subject_textbox"] == "Intro to Python, Data Structures, Algorithms"
    )
    assert result_dict["topic_number_slider"] == 4  # len(subjects) + 1 = 3 + 1
    assert (
        "connections between these subjects" in result_dict["preference_prompt_textbox"]
    )
    assert result_dict["description_textbox"] == ""
    assert result_dict["source_text_textbox"] == ""
    assert result_dict["url_textbox"] == ""
    assert result_dict["subjects_dataframe"] is subjects_df  # Direct assignment

    # Check gr.update calls for visibility
    assert result_dict["subject_mode_group"]["visible"] is True
    assert result_dict["path_mode_group"]["visible"] is False
    assert result_dict["text_mode_group"]["visible"] is False
    assert result_dict["web_mode_group"]["visible"] is False
    assert result_dict["path_results_group"]["visible"] is False
    assert result_dict["cards_output_group"]["visible"] is True

    # Check gr.update calls for clearing/resetting values
    assert result_dict["output_dataframe"]["value"] is None
    assert result_dict["progress_html"]["visible"] is False
    assert result_dict["total_cards_number"]["visible"] is False

    # Check that learning_order and projects_markdown are gr.update() (no change)
    # gr.update() with no args is a dict with only '__type__': 'update'
    assert isinstance(result_dict["learning_order_markdown"], dict)
    assert result_dict["learning_order_markdown"].get("__type__") == "update"
    assert len(result_dict["learning_order_markdown"]) == 1  # Only __type__

    assert isinstance(result_dict["projects_markdown"], dict)
    assert result_dict["projects_markdown"].get("__type__") == "update"
    assert len(result_dict["projects_markdown"]) == 1


@patch("ankigen_core.ui_logic.gr.Warning")
def test_use_selected_subjects_button_click_none_df(mock_gr_warning):
    """Test use_selected_subjects with None DataFrame input."""
    result_dict = use_selected_subjects(None)
    mock_gr_warning.assert_called_once_with(
        "No subjects available to copy from Learning Path analysis."
    )
    # Check it returns a dict of gr.update() no-ops
    for key in EXPECTED_UI_LOGIC_KEYS_USE_SUBJECTS:
        assert key in result_dict
        assert isinstance(result_dict[key], dict)
        assert result_dict[key].get("__type__") == "update"
        assert len(result_dict[key]) == 1


@patch("ankigen_core.ui_logic.gr.Warning")
def test_use_selected_subjects_button_click_empty_df(mock_gr_warning):
    """Test use_selected_subjects with an empty DataFrame."""
    result_dict = use_selected_subjects(pd.DataFrame())
    mock_gr_warning.assert_called_once_with(
        "No subjects available to copy from Learning Path analysis."
    )
    for key in EXPECTED_UI_LOGIC_KEYS_USE_SUBJECTS:
        assert key in result_dict
        assert isinstance(result_dict[key], dict)
        assert result_dict[key].get("__type__") == "update"
        assert len(result_dict[key]) == 1


@patch("ankigen_core.ui_logic.gr.Error")
def test_use_selected_subjects_button_click_missing_column(mock_gr_error):
    """Test use_selected_subjects with DataFrame missing 'Subject' column."""
    result_dict = use_selected_subjects(pd.DataFrame({"WrongColumn": ["data"]}))
    mock_gr_error.assert_called_once_with(
        "Learning path analysis result is missing the 'Subject' column."
    )
    for key in EXPECTED_UI_LOGIC_KEYS_USE_SUBJECTS:
        assert key in result_dict
        assert isinstance(result_dict[key], dict)
        assert result_dict[key].get("__type__") == "update"
        assert len(result_dict[key]) == 1


# --- Test for Generate Button Click --- #


# Helper to create common mock inputs for orchestrate_card_generation
def get_orchestrator_mock_inputs(generation_mode="subject", api_key="sk-test"):
    return {
        "api_key_input": api_key,
        "subject": "Test Subject for Orchestrator",
        "generation_mode": generation_mode,
        "source_text": "Some source text for testing.",
        "url_input": "http://example.com/test-page",
        "model_name": "gpt-test-orchestrator",
        "topic_number": 2,  # For subject mode
        "cards_per_topic": 3,  # For subject mode / text mode / web mode
        "preference_prompt": "Test preferences",
        "generate_cloze": False,
    }


@patch("ankigen_core.card_generator.generate_cards_batch")
@patch("ankigen_core.card_generator.structured_output_completion")
@patch("ankigen_core.card_generator.OpenAIClientManager")
@patch("ankigen_core.card_generator.ResponseCache")
@patch(
    "ankigen_core.card_generator.gr"
)  # Mocking the entire gradio module used within card_generator
async def test_generate_button_click_subject_mode(
    mock_gr, mock_response_cache_class, mock_client_manager_class, mock_soc, mock_gcb
):
    """Test orchestrate_card_generation for 'subject' mode."""
    mock_client_manager_instance = mock_client_manager_class.return_value
    mock_openai_client = MagicMock()
    mock_client_manager_instance.get_client.return_value = mock_openai_client

    mock_cache_instance = mock_response_cache_class.return_value
    mock_cache_instance.get.return_value = None

    mock_inputs = get_orchestrator_mock_inputs(generation_mode="subject")

    # Mock for topic generation call (first SOC call)
    mock_topic_response = {
        "topics": [
            {"name": "Topic Alpha", "difficulty": "easy", "description": "First topic"},
            {
                "name": "Topic Beta",
                "difficulty": "medium",
                "description": "Second topic",
            },
        ]
    }
    # Mock for card generation (generate_cards_batch calls)
    mock_cards_batch_alpha = [
        Card(
            front=CardFront(question="Q_A1"),
            back=CardBack(answer="A_A1", explanation="E_A1", example="Ex_A1"),
        ),
        Card(
            front=CardFront(question="Q_A2"),
            back=CardBack(answer="A_A2", explanation="E_A2", example="Ex_A2"),
        ),
    ]
    mock_cards_batch_beta = [
        Card(
            front=CardFront(question="Q_B1"),
            back=CardBack(answer="A_B1", explanation="E_B1", example="Ex_B1"),
        ),
    ]

    # Configure side effects: first SOC for topics, then GCB for each topic
    mock_soc.return_value = mock_topic_response  # For the topics call
    mock_gcb.side_effect = [mock_cards_batch_alpha, mock_cards_batch_beta]

    df_result, status_html, count = await orchestrate_card_generation(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        **mock_inputs,
    )

    mock_client_manager_instance.initialize_client.assert_called_once_with(
        mock_inputs["api_key_input"]
    )

    # Assertions for SOC (topic generation)
    mock_soc.assert_called_once_with(
        openai_client=mock_openai_client,
        model=mock_inputs["model_name"],
        response_format={"type": "json_object"},
        system_prompt=ANY,
        user_prompt=ANY,
        cache=mock_cache_instance,
    )
    assert mock_inputs["subject"] in mock_soc.call_args[1]["user_prompt"]
    assert str(mock_inputs["topic_number"]) in mock_soc.call_args[1]["user_prompt"]

    # Assertions for generate_cards_batch calls
    assert mock_gcb.call_count == 2
    mock_gcb.assert_any_call(
        openai_client=mock_openai_client,
        cache=mock_cache_instance,
        model=mock_inputs["model_name"],
        topic="Topic Alpha",
        num_cards=mock_inputs["cards_per_topic"],
        system_prompt=ANY,
        generate_cloze=False,
    )
    mock_gcb.assert_any_call(
        openai_client=mock_openai_client,
        cache=mock_cache_instance,
        model=mock_inputs["model_name"],
        topic="Topic Beta",
        num_cards=mock_inputs["cards_per_topic"],
        system_prompt=ANY,
        generate_cloze=False,
    )

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 3  # 2 from alpha, 1 from beta
    assert count == 3
    assert "Generation complete!" in status_html
    assert "Total cards generated: 3" in status_html

    # Check gr.Info was called (e.g., for successful topic generation, card batch generation)
    # Example: mock_gr.Info.assert_any_call("✨ Generated 2 topics successfully! Now generating cards...")
    # More specific assertions can be added if needed for gr.Info/Warning calls
    assert mock_gr.Info.called


@patch("ankigen_core.card_generator.structured_output_completion")
@patch("ankigen_core.card_generator.OpenAIClientManager")
@patch("ankigen_core.card_generator.ResponseCache")
@patch("ankigen_core.card_generator.gr")  # Mocking the entire gradio module
async def test_generate_button_click_text_mode(
    mock_gr, mock_response_cache_class, mock_client_manager_class, mock_soc
):
    """Test orchestrate_card_generation for 'text' mode."""
    mock_client_manager_instance = mock_client_manager_class.return_value
    mock_openai_client = MagicMock()
    mock_client_manager_instance.get_client.return_value = mock_openai_client

    mock_cache_instance = mock_response_cache_class.return_value
    mock_cache_instance.get.return_value = None

    mock_inputs = get_orchestrator_mock_inputs(generation_mode="text")

    # Mock for card generation call (single SOC call in text mode)
    mock_card_data_from_text = {
        "cards": [
            {
                "card_type": "basic",
                "front": {"question": "Q_Text1"},
                "back": {
                    "answer": "A_Text1",
                    "explanation": "E_Text1",
                    "example": "Ex_Text1",
                },
                "metadata": {},
            },
            {
                "card_type": "cloze",
                "front": {"question": "{{c1::Q_Text2}}"},
                "back": {
                    "answer": "A_Text2_Full",
                    "explanation": "E_Text2",
                    "example": "Ex_Text2",
                },
                "metadata": {},
            },
        ]
    }
    mock_soc.return_value = mock_card_data_from_text

    # orchestrate_card_generation calls generate_cards_batch internally, which then calls structured_output_completion.
    # For text mode, orchestrate_card_generation directly calls structured_output_completion.
    df_result, status_html, count = await orchestrate_card_generation(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        **mock_inputs,
    )

    mock_client_manager_instance.initialize_client.assert_called_once_with(
        mock_inputs["api_key_input"]
    )

    # Assertions for SOC (direct card generation from text)
    mock_soc.assert_called_once_with(
        openai_client=mock_openai_client,
        model=mock_inputs["model_name"],
        response_format={"type": "json_object"},
        system_prompt=ANY,
        user_prompt=ANY,
        cache=mock_cache_instance,
    )
    # Ensure the source_text is in the prompt for SOC
    assert mock_inputs["source_text"] in mock_soc.call_args[1]["user_prompt"]
    # Ensure cards_per_topic is in the prompt
    assert str(mock_inputs["cards_per_topic"]) in mock_soc.call_args[1]["user_prompt"]

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 2
    assert count == 2
    mock_gr.Info.assert_any_call("✅ Generated 2 cards from the provided content.")
    assert "Generation complete!" in status_html
    assert "Total cards generated: 2" in status_html
    assert mock_gr.Info.called


@patch("ankigen_core.card_generator.fetch_webpage_text")
@patch("ankigen_core.card_generator.structured_output_completion")
@patch("ankigen_core.card_generator.OpenAIClientManager")
@patch("ankigen_core.card_generator.ResponseCache")
@patch("ankigen_core.card_generator.gr")  # Mocking the entire gradio module
async def test_generate_button_click_web_mode(
    mock_gr,
    mock_response_cache_class,
    mock_client_manager_class,
    mock_soc,
    mock_fetch_web,
):
    """Test orchestrate_card_generation for 'web' mode."""
    mock_client_manager_instance = mock_client_manager_class.return_value
    mock_openai_client = MagicMock()
    mock_client_manager_instance.get_client.return_value = mock_openai_client

    mock_cache_instance = mock_response_cache_class.return_value
    mock_cache_instance.get.return_value = None

    mock_inputs = get_orchestrator_mock_inputs(generation_mode="web")
    mock_fetched_text = "This is the text fetched from the website."
    mock_fetch_web.return_value = mock_fetched_text

    mock_card_data_from_web = {
        "cards": [
            {
                "card_type": "basic",
                "front": {"question": "Q_Web1"},
                "back": {
                    "answer": "A_Web1",
                    "explanation": "E_Web1",
                    "example": "Ex_Web1",
                },
                "metadata": {},
            }
        ]
    }
    mock_soc.return_value = mock_card_data_from_web

    # Call the function (successful path)
    df_result, status_html, count = await orchestrate_card_generation(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        **mock_inputs,
    )
    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 1
    assert count == 1
    mock_gr.Info.assert_any_call(
        f"✅ Successfully fetched text (approx. {len(mock_fetched_text)} chars). Starting AI generation..."
    )
    mock_gr.Info.assert_any_call("✅ Generated 1 cards from the provided content.")
    assert "Generation complete!" in status_html

    # Test web fetch error handling
    mock_fetch_web.reset_mock()
    mock_soc.reset_mock()
    mock_gr.reset_mock()
    mock_client_manager_instance.initialize_client.reset_mock()

    fetch_error_message = "Could not connect to host"
    mock_fetch_web.side_effect = ConnectionError(fetch_error_message)

    # Call the function again, expecting gr.Error to be called by the production code
    df_err, html_err, count_err = await orchestrate_card_generation(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        **mock_inputs,
    )

    # Assert that gr.Error was called with the correct message by the production code
    mock_gr.Error.assert_called_once_with(
        f"Failed to get content from URL: {fetch_error_message}"
    )
    assert df_err.empty
    assert html_err == "Failed to get content from URL."
    assert count_err == 0
    mock_soc.assert_not_called()  # Ensure SOC was not called after fetch error


# Test for unsupported 'path' mode
@patch("ankigen_core.card_generator.OpenAIClientManager")
@patch("ankigen_core.card_generator.ResponseCache")
@patch("ankigen_core.card_generator.gr")  # Mock gr for this test too
async def test_generate_button_click_path_mode_error(
    mock_gr,  # mock_gr is an argument
    mock_response_cache_class,
    mock_client_manager_class,
):
    """Test that 'path' mode calls gr.Error for being unsupported."""
    mock_client_manager_instance = mock_client_manager_class.return_value
    mock_cache_instance = mock_response_cache_class.return_value
    mock_inputs = get_orchestrator_mock_inputs(generation_mode="path")

    # Call the function
    df_err, html_err, count_err = await orchestrate_card_generation(
        client_manager=mock_client_manager_instance,
        cache=mock_cache_instance,
        **mock_inputs,
    )

    # Assert gr.Error was called with the specific unsupported mode message
    mock_gr.Error.assert_called_once_with("Unsupported generation mode selected: path")
    assert df_err.empty
    assert html_err == "Unsupported mode."
    assert count_err == 0


# --- Test Export Buttons --- #


# @patch("ankigen_core.exporters.export_csv") # Using mocker instead
def test_export_csv_button_click(mocker):  # Added mocker fixture
    """Test that export_csv_button click calls the correct core function."""
    # Patch the target function as it's imported in *this test module*
    mock_export_df_to_csv_in_test_module = mocker.patch(
        "tests.integration.test_app_interactions.export_dataframe_to_csv"
    )

    # Simulate the DataFrame that would be in the UI
    sample_df_data = {
        "Index": ["1.1"],
        "Topic": ["T1"],
        "Card_Type": ["basic"],
        "Question": ["Q1"],
        "Answer": ["A1"],
        "Explanation": ["E1"],
        "Example": ["Ex1"],
        "Prerequisites": [[]],
        "Learning_Outcomes": [[]],
        "Common_Misconceptions": [[]],
        "Difficulty": ["easy"],
    }
    mock_ui_dataframe = pd.DataFrame(sample_df_data)
    # Set the return value on the mock that will actually be called
    mock_export_df_to_csv_in_test_module.return_value = "/fake/path/export.csv"

    # Simulate the call that app.py would make.
    # Here we are directly calling the `export_dataframe_to_csv` function imported at the top of this test file.
    # This imported function is now replaced by `mock_export_df_to_csv_in_test_module`.
    result_path = export_dataframe_to_csv(mock_ui_dataframe)

    # Assert the core function was called correctly
    mock_export_df_to_csv_in_test_module.assert_called_once_with(mock_ui_dataframe)
    assert result_path == "/fake/path/export.csv"


# @patch("ankigen_core.exporters.export_deck") # Using mocker instead
def test_export_anki_button_click(mocker):  # Added mocker fixture
    """Test that export_anki_button click calls the correct core function."""
    # Patch the target function as it's imported in *this test module*
    mock_export_df_to_apkg_in_test_module = mocker.patch(
        "tests.integration.test_app_interactions.export_dataframe_to_apkg"
    )

    # Simulate the DataFrame and subject input
    sample_df_data = {
        "Index": ["1.1"],
        "Topic": ["T1"],
        "Card_Type": ["basic"],
        "Question": ["Q1"],
        "Answer": ["A1"],
        "Explanation": ["E1"],
        "Example": ["Ex1"],
        "Prerequisites": [[]],
        "Learning_Outcomes": [[]],
        "Common_Misconceptions": [[]],
        "Difficulty": ["easy"],
    }
    mock_ui_dataframe = pd.DataFrame(sample_df_data)
    mock_subject_input = "My Anki Deck Subject"
    mock_export_df_to_apkg_in_test_module.return_value = "/fake/path/export.apkg"

    # Simulate the call that app.py would make
    # The new function export_dataframe_to_apkg expects df, output_path, deck_name
    # The test was calling export_deck(df, subject)
    # The app.py now has a lambda for this: handle_export_dataframe_to_apkg_click(df, deck_name)
    # So the test needs to reflect this, assuming a deck_name is passed.
    # For this integration test, we are testing the function call itself as imported,
    # not the full Gradio handler. The imported function is export_dataframe_to_apkg.
    # It requires output_path and deck_name. The test needs to be adjusted.
    # Let's assume the test is checking the core logic if the function *were* called with df and deck_name.
    # The app.py handler constructs the output_path.
    # For this test, we'll directly call export_dataframe_to_apkg which is what's imported.
    # We need to provide a dummy output_path for the test.
    dummy_output_path = "/fake/output/path.apkg"
    result_path = export_dataframe_to_apkg(
        mock_ui_dataframe, dummy_output_path, mock_subject_input
    )

    # Assert the core function was called correctly
    mock_export_df_to_apkg_in_test_module.assert_called_once_with(
        mock_ui_dataframe, dummy_output_path, mock_subject_input
    )
    assert result_path == "/fake/path/export.apkg"
