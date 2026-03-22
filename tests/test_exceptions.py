import pytest
from unittest.mock import Mock
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


def test_ankigen_error_inheritance():
    """Verify AnkigenError is a subclass of Exception."""
    assert issubclass(AnkigenError, Exception)
    assert isinstance(AnkigenError("test"), Exception)


def test_exception_hierarchy():
    """Verify the custom exception hierarchy."""
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)

    # API sub-exceptions
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)


def test_exception_catching_by_parent():
    """Test that each exception can be raised and caught by its parent class."""
    with pytest.raises(AnkigenError):
        raise ValidationError("Validation failed")

    with pytest.raises(AnkigenError):
        raise SecurityError("Security breach")

    with pytest.raises(APIError):
        raise OpenAIAPIError("OpenAI error")

    with pytest.raises(AnkigenError):
        raise Context7APIError("Context7 error")


def test_exception_messages_preserved():
    """Test exception messages are preserved."""
    msg = "Custom error message"

    assert str(AnkigenError(msg)) == msg
    assert str(ValidationError(msg)) == msg
    assert str(SecurityError(msg)) == msg
    assert str(APIError(msg)) == msg
    assert str(OpenAIAPIError(msg)) == msg
    assert str(Context7APIError(msg)) == msg
    assert str(ExportError(msg)) == msg
    assert str(CardGenerationError(msg)) == msg
    assert str(ConfigurationError(msg)) == msg


def test_handle_exception_logging():
    """Test that it logs the error message with exc_info=True."""
    mock_logger = Mock()
    exc = ValueError("Original error")
    message = "Context message"

    # We use reraise=False to avoid having to catch it here
    handle_exception(exc, mock_logger, message, reraise=False)

    mock_logger.error.assert_called_once_with(f"{message}: {exc}", exc_info=True)


def test_handle_exception_reraise_default():
    """Test that it re-raises the original exception when reraise=True (default)."""
    mock_logger = Mock()

    try:
        raise ValueError("Original error")
    except ValueError as e:
        with pytest.raises(ValueError) as excinfo:
            handle_exception(e, mock_logger, "Error")
        assert str(excinfo.value) == "Original error"


def test_handle_exception_no_reraise():
    """Test that it does NOT re-raise when reraise=False."""
    mock_logger = Mock()
    exc = ValueError("Original error")

    # Should not raise
    handle_exception(exc, mock_logger, "Error", reraise=False)
    assert mock_logger.error.called


def test_handle_exception_reraise_as():
    """Test that it wraps and re-raises as reraise_as type when specified."""
    mock_logger = Mock()
    exc = ValueError("Original error")
    message = "Wrapped message"

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(exc, mock_logger, message, reraise_as=AnkigenError)

    assert isinstance(excinfo.value, AnkigenError)
    assert message in str(excinfo.value)
    assert "Original error" in str(excinfo.value)


def test_handle_exception_chaining():
    """Test the 'from exc' chaining (check __cause__ on the wrapped exception)."""
    mock_logger = Mock()
    exc = ValueError("Original error")

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(exc, mock_logger, "Error", reraise_as=AnkigenError)

    assert excinfo.value.__cause__ is exc


def test_specific_exception_types():
    """Verify specific exception types can be instantiated and have correct class."""
    exceptions = [
        (AnkigenError, "AnkigenError"),
        (ValidationError, "ValidationError"),
        (SecurityError, "SecurityError"),
        (APIError, "APIError"),
        (OpenAIAPIError, "OpenAIAPIError"),
        (Context7APIError, "Context7APIError"),
        (ExportError, "ExportError"),
        (CardGenerationError, "CardGenerationError"),
        (ConfigurationError, "ConfigurationError"),
    ]

    for exc_class, name in exceptions:
        e = exc_class("test")
        assert e.__class__.__name__ == name
        assert isinstance(e, exc_class)


def test_handle_exception_with_custom_exception_instance():
    """Test handle_exception with a custom AnkigenError."""
    mock_logger = Mock()

    try:
        raise ValidationError("Invalid input")
    except ValidationError as e:
        with pytest.raises(ValidationError):
            handle_exception(e, mock_logger, "Validation failed")

    mock_logger.error.assert_called_once()


def test_handle_exception_with_reraise_false_and_reraise_as():
    """Test handle_exception with reraise=False but reraise_as specified (should NOT reraise)."""
    mock_logger = Mock()
    exc = ValueError("error")

    # Should not raise because reraise=False
    handle_exception(
        exc, mock_logger, "message", reraise=False, reraise_as=AnkigenError
    )
    assert mock_logger.error.called
