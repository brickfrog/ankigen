import pytest
from unittest.mock import MagicMock
from ankigen.exceptions import (
    AnkigenError,
    ValidationError,
    SecurityError,
    APIError,
    OpenAIAPIError,
    Context7APIError,
    ExportError,
    CardGenerationError,
    ConfigurationError,
    handle_exception,
)


def test_exception_hierarchy():
    """Test that all exceptions inherit correctly from AnkigenError."""
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(OpenAIAPIError, AnkigenError)
    assert issubclass(Context7APIError, APIError)
    assert issubclass(Context7APIError, AnkigenError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)


def test_exception_catching():
    """Test that exceptions can be caught by their parent classes."""
    with pytest.raises(AnkigenError):
        raise ValidationError("Validation failed")

    with pytest.raises(APIError):
        raise OpenAIAPIError("OpenAI API error")

    with pytest.raises(AnkigenError):
        raise OpenAIAPIError("OpenAI API error")


def test_handle_exception_logging():
    """Test that handle_exception logs the error."""
    mock_logger = MagicMock()
    exc = ValueError("Test error")
    message = "An error occurred"

    try:
        handle_exception(exc, mock_logger, message, reraise=False)
    except Exception:
        pytest.fail("handle_exception raised an error when reraise=False")

    mock_logger.error.assert_called_once_with(f"{message}: {exc}", exc_info=True)


def test_handle_exception_reraise_true():
    """Test that handle_exception re-raises the original exception when reraise=True."""
    mock_logger = MagicMock()

    try:
        raise ValueError("Test error")
    except ValueError as exc:
        with pytest.raises(ValueError) as excinfo:
            handle_exception(exc, mock_logger, "Error", reraise=True)
        assert str(excinfo.value) == "Test error"


def test_handle_exception_reraise_false():
    """Test that handle_exception does not re-raise when reraise=False."""
    mock_logger = MagicMock()
    exc = ValueError("Test error")

    handle_exception(exc, mock_logger, "Error", reraise=False)
    # Should not raise anything


def test_handle_exception_reraise_as():
    """Test that handle_exception wraps the exception in reraise_as type."""
    mock_logger = MagicMock()
    exc = ValueError("Original error")
    message = "Wrapped error"

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(
            exc, mock_logger, message, reraise=True, reraise_as=AnkigenError
        )

    assert isinstance(excinfo.value, AnkigenError)
    assert str(excinfo.value) == f"{message}: {exc}"
    assert excinfo.value.__cause__ is exc
