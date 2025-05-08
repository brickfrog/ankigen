# Tests for ankigen_core/llm_interface.py
import pytest
from unittest.mock import patch, MagicMock, ANY
from openai import OpenAIError
import json
import tenacity

# Modules to test
from ankigen_core.llm_interface import OpenAIClientManager, structured_output_completion
from ankigen_core.utils import (
    ResponseCache,
)  # Need ResponseCache for testing structured_output_completion

# --- OpenAIClientManager Tests ---


def test_client_manager_init():
    """Test initial state of the client manager."""
    manager = OpenAIClientManager()
    assert manager._client is None
    assert manager._api_key is None


def test_client_manager_initialize_success():
    """Test successful client initialization."""
    manager = OpenAIClientManager()
    valid_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # We don't need to actually connect, so patch the OpenAI constructor
    with patch("ankigen_core.llm_interface.OpenAI") as mock_openai_constructor:
        mock_client_instance = MagicMock()
        mock_openai_constructor.return_value = mock_client_instance

        manager.initialize_client(valid_key)

        mock_openai_constructor.assert_called_once_with(api_key=valid_key)
        assert manager._api_key == valid_key
        assert manager._client is mock_client_instance


def test_client_manager_initialize_invalid_key_format():
    """Test initialization failure with invalid API key format."""
    manager = OpenAIClientManager()
    invalid_key = "invalid-key-format"
    with pytest.raises(ValueError, match="Invalid OpenAI API key format."):
        manager.initialize_client(invalid_key)
    assert manager._client is None
    assert manager._api_key is None  # Should remain None


def test_client_manager_initialize_openai_error():
    """Test handling of OpenAIError during client initialization."""
    manager = OpenAIClientManager()
    valid_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    error_message = "Test OpenAI Init Error"

    with patch(
        "ankigen_core.llm_interface.OpenAI", side_effect=OpenAIError(error_message)
    ) as mock_openai_constructor:
        with pytest.raises(OpenAIError, match=error_message):
            manager.initialize_client(valid_key)

        mock_openai_constructor.assert_called_once_with(api_key=valid_key)
        assert manager._client is None  # Ensure client is None after failure
        assert (
            manager._api_key == valid_key
        )  # API key is set before client creation attempt


def test_client_manager_get_client_success():
    """Test getting the client after successful initialization."""
    manager = OpenAIClientManager()
    valid_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    with patch("ankigen_core.llm_interface.OpenAI") as mock_openai_constructor:
        mock_client_instance = MagicMock()
        mock_openai_constructor.return_value = mock_client_instance
        manager.initialize_client(valid_key)

        client = manager.get_client()
        assert client is mock_client_instance


def test_client_manager_get_client_not_initialized():
    """Test getting the client before initialization."""
    manager = OpenAIClientManager()
    with pytest.raises(RuntimeError, match="OpenAI client is not initialized."):
        manager.get_client()


# --- structured_output_completion Tests ---


# Fixture for mock OpenAI client
@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    # Mock the specific method used by the function
    client.chat.completions.create = MagicMock()
    return client


# Fixture for mock ResponseCache
@pytest.fixture
def mock_response_cache():
    cache = MagicMock(spec=ResponseCache)
    return cache


def test_structured_output_completion_cache_hit(
    mock_openai_client, mock_response_cache
):
    """Test behavior when the response is found in the cache."""
    system_prompt = "System prompt"
    user_prompt = "User prompt"
    model = "test-model"
    cached_result = {"data": "cached result"}

    # Configure mock cache to return the cached result
    mock_response_cache.get.return_value = cached_result

    result = structured_output_completion(
        openai_client=mock_openai_client,
        model=model,
        response_format={"type": "json_object"},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache=mock_response_cache,
    )

    # Assertions
    mock_response_cache.get.assert_called_once_with(
        f"{system_prompt}:{user_prompt}", model
    )
    mock_openai_client.chat.completions.create.assert_not_called()  # API should not be called
    mock_response_cache.set.assert_not_called()  # Cache should not be set again
    assert result == cached_result


def test_structured_output_completion_cache_miss_success(
    mock_openai_client, mock_response_cache
):
    """Test behavior on cache miss with a successful API call."""
    system_prompt = "System prompt for success"
    user_prompt = "User prompt for success"
    model = "test-model-success"
    expected_result = {"data": "successful API result"}

    # Configure mock cache to return None (cache miss)
    mock_response_cache.get.return_value = None

    # Configure mock API response
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(expected_result)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    result = structured_output_completion(
        openai_client=mock_openai_client,
        model=model,
        response_format={"type": "json_object"},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache=mock_response_cache,
    )

    # Assertions
    mock_response_cache.get.assert_called_once_with(
        f"{system_prompt}:{user_prompt}", model
    )
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=[
            {
                "role": "system",
                "content": ANY,
            },  # Check prompt structure later if needed
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    mock_response_cache.set.assert_called_once_with(
        f"{system_prompt}:{user_prompt}", model, expected_result
    )
    assert result == expected_result


def test_structured_output_completion_api_error(
    mock_openai_client, mock_response_cache
):
    """Test behavior when the OpenAI API call raises an error."""
    system_prompt = "System prompt for error"
    user_prompt = "User prompt for error"
    model = "test-model-error"
    error_message = "Test API Error"

    # Configure mock cache for cache miss
    mock_response_cache.get.return_value = None

    # Configure mock API call to raise an error (after potential retries)
    # The @retry decorator is hard to mock precisely without tenacity knowledge.
    # We assume it eventually raises the error if all retries fail.
    mock_openai_client.chat.completions.create.side_effect = OpenAIError(error_message)

    with pytest.raises(tenacity.RetryError):
        structured_output_completion(
            openai_client=mock_openai_client,
            model=model,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cache=mock_response_cache,
        )

    # Optionally, check the underlying exception type if needed:
    # assert isinstance(excinfo.value.last_attempt.exception(), OpenAIError)
    # assert str(excinfo.value.last_attempt.exception()) == error_message

    # Assertions
    # cache.get is called on each retry attempt
    assert (
        mock_response_cache.get.call_count == 3
    ), f"Expected cache.get to be called 3 times due to retries, but was {mock_response_cache.get.call_count}"
    # Check that create was called 3 times due to retry
    assert (
        mock_openai_client.chat.completions.create.call_count == 3
    ), f"Expected create to be called 3 times due to retries, but was {mock_openai_client.chat.completions.create.call_count}"
    mock_response_cache.set.assert_not_called()  # Cache should not be set on error


def test_structured_output_completion_invalid_json(
    mock_openai_client, mock_response_cache
):
    """Test behavior when the API returns invalid JSON."""
    system_prompt = "System prompt for invalid json"
    user_prompt = "User prompt for invalid json"
    model = "test-model-invalid-json"
    invalid_json_content = "this is not json"

    # Configure mock cache for cache miss
    mock_response_cache.get.return_value = None

    # Configure mock API response with invalid JSON
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = invalid_json_content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    with pytest.raises(tenacity.RetryError):
        structured_output_completion(
            openai_client=mock_openai_client,
            model=model,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cache=mock_response_cache,
        )

    # Assertions
    # cache.get is called on each retry attempt
    assert (
        mock_response_cache.get.call_count == 3
    ), f"Expected cache.get to be called 3 times due to retries, but was {mock_response_cache.get.call_count}"
    # create is also called on each retry attempt
    assert (
        mock_openai_client.chat.completions.create.call_count == 3
    ), f"Expected create to be called 3 times due to retries, but was {mock_openai_client.chat.completions.create.call_count}"
    mock_response_cache.set.assert_not_called()  # Cache should not be set on error


def test_structured_output_completion_no_choices(
    mock_openai_client, mock_response_cache
):
    """Test behavior when API completion has no choices."""
    system_prompt = "System prompt no choices"
    user_prompt = "User prompt no choices"
    model = "test-model-no-choices"

    mock_response_cache.get.return_value = None
    mock_completion = MagicMock()
    mock_completion.choices = []  # No choices
    mock_openai_client.chat.completions.create.return_value = mock_completion

    # Currently function logs warning and returns None. We test for None.
    result = structured_output_completion(
        openai_client=mock_openai_client,
        model=model,
        response_format={"type": "json_object"},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache=mock_response_cache,
    )
    assert result is None
    mock_response_cache.set.assert_not_called()


def test_structured_output_completion_no_message_content(
    mock_openai_client, mock_response_cache
):
    """Test behavior when API choice has no message content."""
    system_prompt = "System prompt no content"
    user_prompt = "User prompt no content"
    model = "test-model-no-content"

    mock_response_cache.get.return_value = None
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = None  # No content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    # Currently function logs warning and returns None. We test for None.
    result = structured_output_completion(
        openai_client=mock_openai_client,
        model=model,
        response_format={"type": "json_object"},
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache=mock_response_cache,
    )
    assert result is None
    mock_response_cache.set.assert_not_called()


# Remove original placeholder
# def test_placeholder_llm_interface():
#     assert True
