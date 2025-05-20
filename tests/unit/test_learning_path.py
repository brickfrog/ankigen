# Tests for ankigen_core/learning_path.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
import gradio as gr
from openai import OpenAIError

# Module to test
from ankigen_core.learning_path import analyze_learning_path
from ankigen_core.llm_interface import OpenAIClientManager
from ankigen_core.utils import ResponseCache


@pytest.fixture
def mock_client_manager_learning_path():
    """Provides a mock OpenAIClientManager for learning path tests."""
    manager = MagicMock(spec=OpenAIClientManager)
    mock_client = MagicMock()
    manager.get_client.return_value = mock_client
    manager.initialize_client.return_value = None
    return manager, mock_client


@pytest.fixture
def mock_response_cache_learning_path():
    """Provides a mock ResponseCache for learning path tests."""
    cache = MagicMock(spec=ResponseCache)
    cache.get.return_value = None  # Default to cache miss
    return cache


@patch("ankigen_core.learning_path.structured_output_completion")
async def test_analyze_learning_path_success(
    mock_soc, mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test successful learning path analysis."""
    manager, client = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path
    api_key = "valid_key"
    description = "Learn Python for data science"
    model = "gpt-test"

    # Mock the successful response from structured_output_completion
    mock_response = {
        "subjects": [
            {
                "Subject": "Python Basics",
                "Prerequisites": "None",
                "Time Estimate": "2 weeks",
            },
            {
                "Subject": "Pandas",
                "Prerequisites": "Python Basics",
                "Time Estimate": "1 week",
            },
        ],
        "learning_order": "Start with Basics, then move to Pandas.",
        "projects": "Analyze a sample dataset.",
    }
    mock_soc.return_value = mock_response

    df_result, order_text, projects_text = await analyze_learning_path(
        client_manager=manager,
        cache=cache,
        api_key=api_key,
        description=description,
        model=model,
    )

    # Assertions
    manager.initialize_client.assert_called_once_with(api_key)
    manager.get_client.assert_called_once()
    mock_soc.assert_called_once_with(
        openai_client=client,
        model=model,
        response_format={"type": "json_object"},
        system_prompt=ANY,
        user_prompt=ANY,  # Could assert description is in here if needed
        cache=cache,
    )

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 2
    assert list(df_result.columns) == ["Subject", "Prerequisites", "Time Estimate"]
    assert df_result.iloc[0]["Subject"] == "Python Basics"
    assert df_result.iloc[1]["Subject"] == "Pandas"

    assert "Recommended Learning Order" in order_text
    assert "Start with Basics, then move to Pandas." in order_text

    assert "Suggested Projects" in projects_text
    assert "Analyze a sample dataset." in projects_text

    assert projects_text == mock_response["projects"]


async def test_analyze_learning_path_no_api_key(
    mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test that gr.Error is raised if API key is missing."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path

    with pytest.raises(gr.Error, match="API key is required"):
        await analyze_learning_path(
            client_manager=manager,
            cache=cache,
            api_key="",  # Empty API key
            description="Test",
            model="gpt-test",
        )


async def test_analyze_learning_path_client_init_error(
    mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test that gr.Error is raised if client initialization fails."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path
    error_msg = "Invalid Key"
    manager.initialize_client.side_effect = ValueError(error_msg)

    with pytest.raises(gr.Error, match=f"OpenAI Client Error: {error_msg}"):
        await analyze_learning_path(
            client_manager=manager,
            cache=cache,
            api_key="invalid_key",
            description="Test",
            model="gpt-test",
        )


@patch("ankigen_core.learning_path.structured_output_completion")
async def test_analyze_learning_path_api_error(
    mock_soc, mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test that errors from structured_output_completion are handled."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path
    error_msg = "API connection failed"
    mock_soc.side_effect = OpenAIError(error_msg)

    with pytest.raises(gr.Error, match=f"Failed to analyze learning path: {error_msg}"):
        await analyze_learning_path(
            client_manager=manager,
            cache=cache,
            api_key="valid_key",
            description="Test",
            model="gpt-test",
        )


@patch("ankigen_core.learning_path.structured_output_completion")
async def test_analyze_learning_path_invalid_response_format(
    mock_soc, mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test handling of invalid response format from API."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path

    # Simulate various invalid responses (excluding cases where subjects list is present but items are invalid)
    invalid_responses = [
        None,
        "just a string",
        {},
        {"subjects": "not a list"},
        {"subjects": [], "learning_order": "Order"},  # Missing projects
        # Removed cases handled by test_analyze_learning_path_invalid_subject_structure
        # {
        #     "subjects": [{"Subject": "S1"}],
        #     "learning_order": "O",
        #     "projects": "P",
        # }, # Missing fields in subject
        # {
        #     "subjects": [
        #         {"Subject": "S1", "Prerequisites": "P1", "Time Estimate": "T1"},
        #         "invalid_entry",
        #     ],
        #     "learning_order": "O",
        #     "projects": "P",
        # }, # Invalid entry in subjects list
    ]

    for mock_response in invalid_responses:
        mock_soc.reset_mock()
        mock_soc.return_value = mock_response
        with pytest.raises(gr.Error, match="invalid API response format"):
            await analyze_learning_path(
                client_manager=manager,
                cache=cache,
                api_key="valid_key",
                description="Test Invalid",
                model="gpt-test",
            )


@patch("ankigen_core.learning_path.structured_output_completion")
async def test_analyze_learning_path_no_valid_subjects(
    mock_soc, mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test handling when API returns subjects but none are valid."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path

    mock_response = {
        "subjects": [{"wrong_key": "value"}, {}],  # No valid subjects
        "learning_order": "Order",
        "projects": "Projects",
    }
    mock_soc.return_value = mock_response

    with pytest.raises(gr.Error, match="API returned no valid subjects"):
        await analyze_learning_path(
            client_manager=manager,
            cache=cache,
            api_key="valid_key",
            description="Test No Valid Subjects",
            model="gpt-test",
        )


@patch("ankigen_core.learning_path.structured_output_completion")
async def test_analyze_learning_path_invalid_subject_structure(
    mock_soc, mock_client_manager_learning_path, mock_response_cache_learning_path
):
    """Test handling when subjects list contains ONLY invalid/incomplete dicts."""
    manager, _ = mock_client_manager_learning_path
    cache = mock_response_cache_learning_path

    # Simulate responses where subjects list is present but ALL items are invalid
    invalid_subject_responses = [
        {
            "subjects": [{"Subject": "S1"}],
            "learning_order": "O",
            "projects": "P",
        },  # Missing fields
        {
            "subjects": ["invalid_string"],
            "learning_order": "O",
            "projects": "P",
        },  # String entry only
        {
            "subjects": [{"wrong_key": "value"}],
            "learning_order": "O",
            "projects": "P",
        },  # Wrong keys only
    ]

    for mock_response in invalid_subject_responses:
        mock_soc.reset_mock()
        mock_soc.return_value = mock_response
        with pytest.raises(gr.Error, match="API returned no valid subjects"):
            await analyze_learning_path(
                client_manager=manager,
                cache=cache,
                api_key="valid_key",
                description="Test Invalid Subject Structure",
                model="gpt-test",
            )
