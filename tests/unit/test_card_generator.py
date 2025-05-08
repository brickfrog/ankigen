# Tests for ankigen_core/card_generator.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd

# Assuming Pydantic models, ResponseCache etc. are needed
from ankigen_core.models import Card, CardFront, CardBack
from ankigen_core.utils import ResponseCache
from ankigen_core.llm_interface import OpenAIClientManager  # Needed for type hints

# Module to test
from ankigen_core import card_generator
from ankigen_core.card_generator import (
    get_dataframe_columns,
)  # Import for use in error returns

# --- Constants Tests (Optional but good practice) ---


def test_constants_exist_and_have_expected_type():
    """Test that constants exist and are lists."""
    assert isinstance(card_generator.AVAILABLE_MODELS, list)
    assert isinstance(card_generator.GENERATION_MODES, list)
    assert len(card_generator.AVAILABLE_MODELS) > 0
    assert len(card_generator.GENERATION_MODES) > 0


# --- generate_cards_batch Tests ---


@pytest.fixture
def mock_openai_client_fixture():  # Renamed to avoid conflict with llm_interface tests fixture
    """Provides a MagicMock OpenAI client."""
    return MagicMock()


@pytest.fixture
def mock_response_cache_fixture():
    """Provides a MagicMock ResponseCache."""
    cache = MagicMock(spec=ResponseCache)
    cache.get.return_value = None  # Default to cache miss
    return cache


@patch("ankigen_core.card_generator.structured_output_completion")
def test_generate_cards_batch_success(
    mock_soc, mock_openai_client_fixture, mock_response_cache_fixture
):
    """Test successful card generation using generate_cards_batch."""
    mock_openai_client = mock_openai_client_fixture
    mock_response_cache = mock_response_cache_fixture
    model = "gpt-test"
    topic = "Test Topic"
    num_cards = 2
    system_prompt = "System prompt"
    generate_cloze = False

    # Mock the response from structured_output_completion
    mock_soc.return_value = {
        "cards": [
            {
                "card_type": "basic",
                "front": {"question": "Q1"},
                "back": {"answer": "A1", "explanation": "E1", "example": "Ex1"},
                "metadata": {"difficulty": "beginner"},
            },
            {
                "card_type": "cloze",
                "front": {"question": "{{c1::Q2}}"},
                "back": {"answer": "A2_full", "explanation": "E2", "example": "Ex2"},
                "metadata": {"difficulty": "intermediate"},
            },
        ]
    }

    result_cards = card_generator.generate_cards_batch(
        openai_client=mock_openai_client,
        cache=mock_response_cache,
        model=model,
        topic=topic,
        num_cards=num_cards,
        system_prompt=system_prompt,
        generate_cloze=generate_cloze,
    )

    assert len(result_cards) == 2
    assert isinstance(result_cards[0], Card)
    assert result_cards[0].card_type == "basic"
    assert result_cards[0].front.question == "Q1"
    assert result_cards[1].card_type == "cloze"
    assert result_cards[1].front.question == "{{c1::Q2}}"
    assert result_cards[1].metadata["difficulty"] == "intermediate"

    mock_soc.assert_called_once()
    call_args = mock_soc.call_args[1]  # Get keyword args
    assert call_args["openai_client"] == mock_openai_client
    assert call_args["cache"] == mock_response_cache
    assert call_args["model"] == model
    assert call_args["system_prompt"] == system_prompt
    assert topic in call_args["user_prompt"]
    assert str(num_cards) in call_args["user_prompt"]
    # Check cloze instruction is NOT present
    assert "generate Cloze deletion cards" not in call_args["user_prompt"]


@patch("ankigen_core.card_generator.structured_output_completion")
def test_generate_cards_batch_cloze_prompt(
    mock_soc, mock_openai_client_fixture, mock_response_cache_fixture
):
    """Test generate_cards_batch includes cloze instructions when requested."""
    mock_openai_client = mock_openai_client_fixture
    mock_response_cache = mock_response_cache_fixture
    mock_soc.return_value = {"cards": []}  # Return empty for simplicity

    card_generator.generate_cards_batch(
        openai_client=mock_openai_client,
        cache=mock_response_cache,
        model="gpt-test",
        topic="Cloze Topic",
        num_cards=1,
        system_prompt="System",
        generate_cloze=True,
    )

    mock_soc.assert_called_once()
    call_args = mock_soc.call_args[1]
    # Check that specific cloze instructions are present
    assert "generate Cloze deletion cards" in call_args["user_prompt"]
    # Corrected check: Look for instruction text, not the JSON example syntax
    assert (
        "Format the question field using Anki's cloze syntax"
        in call_args["user_prompt"]
    )


@patch("ankigen_core.card_generator.structured_output_completion")
def test_generate_cards_batch_api_error(
    mock_soc, mock_openai_client_fixture, mock_response_cache_fixture
):
    """Test generate_cards_batch handles API errors by re-raising."""
    mock_openai_client = mock_openai_client_fixture
    mock_response_cache = mock_response_cache_fixture
    error_message = "API Error"
    mock_soc.side_effect = ValueError(error_message)  # Simulate error from SOC

    with pytest.raises(ValueError, match=error_message):
        card_generator.generate_cards_batch(
            openai_client=mock_openai_client,
            cache=mock_response_cache,
            model="gpt-test",
            topic="Error Topic",
            num_cards=1,
            system_prompt="System",
            generate_cloze=False,
        )


@patch("ankigen_core.card_generator.structured_output_completion")
def test_generate_cards_batch_invalid_response(
    mock_soc, mock_openai_client_fixture, mock_response_cache_fixture
):
    """Test generate_cards_batch handles invalid JSON or missing keys."""
    mock_openai_client = mock_openai_client_fixture
    mock_response_cache = mock_response_cache_fixture
    mock_soc.return_value = {"wrong_key": []}  # Missing 'cards' key

    with pytest.raises(ValueError, match="Failed to generate cards"):
        card_generator.generate_cards_batch(
            openai_client=mock_openai_client,
            cache=mock_response_cache,
            model="gpt-test",
            topic="Invalid Response Topic",
            num_cards=1,
            system_prompt="System",
            generate_cloze=False,
        )


# --- orchestrate_card_generation Tests ---


@pytest.fixture
def mock_client_manager_fixture():
    """Provides a MagicMock OpenAIClientManager."""
    manager = MagicMock(spec=OpenAIClientManager)
    mock_client = MagicMock()  # Mock the client instance it returns
    manager.get_client.return_value = mock_client
    # Simulate successful initialization by default
    manager.initialize_client.return_value = None
    return manager, mock_client


def base_orchestrator_args(api_key="valid_key", **kwargs):
    """Base arguments for orchestrate_card_generation."""
    base_args = {
        "api_key_input": api_key,
        "subject": "Subject",
        "generation_mode": "subject",  # Default mode
        "source_text": "Source text",
        "url_input": "http://example.com",
        "model_name": "gpt-test",
        "topic_number": 1,  # Corresponds to num_cards in generate_cards_batch
        "cards_per_topic": 5,  # Corresponds to num_cards in generate_cards_batch
        "preference_prompt": "Pref prompt",  # Corresponds to system_prompt
        "generate_cloze": False,
    }
    base_args.update(kwargs)  # Update with any provided kwargs
    return base_args


@patch("ankigen_core.card_generator.structured_output_completion")
@patch("ankigen_core.card_generator.generate_cards_batch")
def test_orchestrate_subject_mode(
    mock_gcb, mock_soc, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrate_card_generation in 'subject' mode."""
    manager, client = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="subject")

    # Mock the first SOC call (for topics)
    mock_soc.return_value = {
        "topics": [
            {"name": "Topic 1", "difficulty": "beginner", "description": "Desc 1"}
        ]
    }

    # Mock return value from generate_cards_batch (called inside loop)
    mock_gcb.return_value = [
        Card(
            front=CardFront(question="Q1"),
            back=CardBack(answer="A1", explanation="E1", example="Ex1"),
        )
    ]

    # Patch gr.Info/Warning
    with patch("gradio.Info"), patch("gradio.Warning"):
        df_result, status, count = card_generator.orchestrate_card_generation(
            client_manager=manager, cache=cache, **args
        )

    manager.initialize_client.assert_called_once_with(args["api_key_input"])
    manager.get_client.assert_called_once()

    # Check SOC call for topics
    mock_soc.assert_called_once()
    soc_call_args = mock_soc.call_args[1]
    assert soc_call_args["openai_client"] == client
    assert "Generate the top" in soc_call_args["user_prompt"]
    assert args["subject"] in soc_call_args["user_prompt"]

    # Check GCB call for the generated topic
    mock_gcb.assert_called_once_with(
        openai_client=client,
        cache=cache,
        model=args["model_name"],
        topic="Topic 1",  # Topic name from mock_soc response
        num_cards=args["cards_per_topic"],
        system_prompt=ANY,  # System prompt is constructed internally
        generate_cloze=args["generate_cloze"],
    )
    assert count == 1
    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 1
    assert df_result.iloc[0]["Question"] == "Q1"
    # Correct assertion to check for the returned HTML string (ignoring precise whitespace)
    assert "Generation complete!" in status
    assert "Total cards generated: 1" in status
    assert "<div" in status  # Basic check for HTML structure
    # expected_html_status = '''
    # <div style="text-align: center">
    #     <p>✅ Generation complete!</p>
    #     <p>Total cards generated: 1</p>
    # </div>
    # '''
    # assert status.strip() == expected_html_status.strip()


@patch("ankigen_core.card_generator.structured_output_completion")
@patch("ankigen_core.card_generator.generate_cards_batch")
def test_orchestrate_text_mode(
    mock_gcb, mock_soc, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrate_card_generation in 'text' mode."""
    manager, client = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="text")
    mock_soc.return_value = {"cards": []}

    card_generator.orchestrate_card_generation(
        client_manager=manager, cache=cache, **args
    )

    mock_soc.assert_called_once()
    call_args = mock_soc.call_args[1]
    assert args["source_text"] in call_args["user_prompt"]


@patch("ankigen_core.card_generator.fetch_webpage_text")
@patch("ankigen_core.card_generator.structured_output_completion")
def test_orchestrate_web_mode(
    mock_soc, mock_fetch, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrate_card_generation in 'web' mode."""
    manager, client = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="web")

    fetched_text = "This is the fetched web page text."
    mock_fetch.return_value = fetched_text
    mock_soc.return_value = {
        "cards": []
    }  # Mock successful SOC call returning empty cards

    # Mock gr.Info and gr.Warning to avoid Gradio UI calls during test
    # Removed the incorrect pytest.raises and mock_gr_warning patch from here
    with patch("gradio.Info"), patch("gradio.Warning"):
        card_generator.orchestrate_card_generation(
            client_manager=manager, cache=cache, **args
        )

    mock_fetch.assert_called_once_with(args["url_input"])
    mock_soc.assert_called_once()
    call_args = mock_soc.call_args[1]
    assert fetched_text in call_args["user_prompt"]


@patch("ankigen_core.card_generator.fetch_webpage_text")
@patch(
    "ankigen_core.card_generator.gr.Error"
)  # Mock gr.Error used by orchestrate_card_generation
def test_orchestrate_web_mode_fetch_error(
    mock_gr_error, mock_fetch, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test 'web' mode handles errors during webpage fetching by calling gr.Error."""
    manager, _ = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="web")
    error_msg = "Connection timed out"
    mock_fetch.side_effect = ConnectionError(error_msg)

    with patch("gradio.Info"), patch("gradio.Warning"):
        df, status_msg, count = card_generator.orchestrate_card_generation(
            client_manager=manager, cache=cache, **args
        )

    mock_gr_error.assert_called_once_with(
        f"Failed to get content from URL: {error_msg}"
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.columns.tolist() == get_dataframe_columns()
    assert status_msg == "Failed to get content from URL."
    assert count == 0


@patch("ankigen_core.card_generator.structured_output_completion")  # Patch SOC
@patch("ankigen_core.card_generator.generate_cards_batch")
def test_orchestrate_generation_batch_error(
    mock_gcb, mock_soc, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrator handles errors from generate_cards_batch."""
    manager, client = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="subject")
    error_msg = "LLM generation failed"  # Define error_msg here

    # Mock the first SOC call (for topics) - needs to succeed
    mock_soc.return_value = {
        "topics": [
            {"name": "Topic 1", "difficulty": "beginner", "description": "Desc 1"}
        ]
    }

    # Configure GCB to raise an error
    mock_gcb.side_effect = ValueError(error_msg)

    # Patch gr.Info/Warning and assert Warning is called
    # Removed pytest.raises
    with patch("gradio.Info"), patch("gradio.Warning") as mock_gr_warning:
        # Add the call to the function back in
        card_generator.orchestrate_card_generation(
            client_manager=manager, cache=cache, **args
        )

    # Assert that the warning was called due to the GCB error
    mock_gr_warning.assert_called_with(
        "Failed to generate cards for 'Topic 1'. Skipping."
    )

    mock_soc.assert_called_once()  # Ensure topic generation was attempted
    mock_gcb.assert_called_once()  # Ensure card generation was attempted


@patch("ankigen_core.card_generator.gr.Error")
def test_orchestrate_path_mode_raises_not_implemented(
    mock_gr_error, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test 'path' mode calls gr.Error for being unsupported."""
    manager, _ = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="path")

    df, status_msg, count = card_generator.orchestrate_card_generation(
        client_manager=manager, cache=cache, **args
    )

    mock_gr_error.assert_called_once_with("Unsupported generation mode selected: path")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.columns.tolist() == get_dataframe_columns()
    assert status_msg == "Unsupported mode."
    assert count == 0


@patch("ankigen_core.card_generator.gr.Error")
def test_orchestrate_invalid_mode_raises_value_error(
    mock_gr_error, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test invalid mode calls gr.Error."""
    manager, _ = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(generation_mode="invalid_mode")

    df, status_msg, count = card_generator.orchestrate_card_generation(
        client_manager=manager, cache=cache, **args
    )

    mock_gr_error.assert_called_once_with(
        "Unsupported generation mode selected: invalid_mode"
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.columns.tolist() == get_dataframe_columns()
    assert status_msg == "Unsupported mode."
    assert count == 0


@patch("ankigen_core.card_generator.gr.Error")
def test_orchestrate_no_api_key_raises_error(
    mock_gr_error, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrator calls gr.Error if API key is missing."""
    manager, _ = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args(api_key="")  # Empty API key

    df, status_msg, count = card_generator.orchestrate_card_generation(
        client_manager=manager, cache=cache, **args
    )

    mock_gr_error.assert_called_once_with("OpenAI API key is required")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.columns.tolist() == get_dataframe_columns()
    assert status_msg == "API key is required."
    assert count == 0


@patch("ankigen_core.card_generator.gr.Error")
def test_orchestrate_client_init_error_raises_error(
    mock_gr_error, mock_client_manager_fixture, mock_response_cache_fixture
):
    """Test orchestrator calls gr.Error if client initialization fails."""
    manager, _ = mock_client_manager_fixture
    cache = mock_response_cache_fixture
    args = base_orchestrator_args()
    error_msg = "Invalid API Key"
    manager.initialize_client.side_effect = ValueError(error_msg)

    df, status_msg, count = card_generator.orchestrate_card_generation(
        client_manager=manager, cache=cache, **args
    )

    mock_gr_error.assert_called_once_with(f"OpenAI Client Error: {error_msg}")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.columns.tolist() == get_dataframe_columns()
    assert status_msg == f"OpenAI Client Error: {error_msg}"
    assert count == 0
