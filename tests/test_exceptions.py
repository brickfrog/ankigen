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


def test_exception_hierarchy():
    """Test that all custom exceptions inherit correctly."""
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)
    assert issubclass(OpenAIAPIError, AnkigenError)
    assert issubclass(Context7APIError, AnkigenError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)


def test_exceptions_can_be_raised():
    """Test that all exceptions can be raised and caught by their parent classes."""
    with pytest.raises(AnkigenError):
        raise ValidationError("test validation error")

    with pytest.raises(AnkigenError):
        raise SecurityError("test security error")

    with pytest.raises(APIError):
        raise OpenAIAPIError("test openai error")

    with pytest.raises(AnkigenError):
        raise OpenAIAPIError("test openai error")

    with pytest.raises(APIError):
        raise Context7APIError("test context7 error")

    with pytest.raises(AnkigenError):
        raise ExportError("test export error")

    with pytest.raises(AnkigenError):
        raise CardGenerationError("test card generation error")

    with pytest.raises(AnkigenError):
        raise ConfigurationError("test configuration error")


def test_handle_exception_logging():
    """Test that handle_exception logs the error correctly."""
    mock_logger = Mock()
    exc = ValueError("original error")
    message = "Context message"

    handle_exception(exc, mock_logger, message, reraise=False)

    mock_logger.error.assert_called_once_with(f"{message}: {exc}", exc_info=True)


def test_handle_exception_reraise_default():
    """Test that handle_exception re-raises the original exception by default."""
    mock_logger = Mock()
    exc = ValueError("original error")

    with pytest.raises(ValueError) as excinfo:
        try:
            raise exc
        except ValueError as e:
            handle_exception(e, mock_logger, "msg")

    assert excinfo.value is exc


def test_handle_exception_no_reraise():
    """Test that handle_exception does not re-raise when reraise=False."""
    mock_logger = Mock()
    exc = ValueError("original error")

    # Should not raise
    try:
        raise exc
    except ValueError as e:
        handle_exception(e, mock_logger, "msg", reraise=False)

    mock_logger.error.assert_called_once()


def test_handle_exception_reraise_as():
    """Test that handle_exception wraps the exception when reraise_as is provided."""
    mock_logger = Mock()
    exc = ValueError("original error")
    message = "Context message"

    with pytest.raises(AnkigenError) as excinfo:
        try:
            raise exc
        except ValueError as e:
            handle_exception(e, mock_logger, message, reraise_as=AnkigenError)

    assert isinstance(excinfo.value, AnkigenError)
    assert f"{message}: {exc}" in str(excinfo.value)
    assert excinfo.value.__cause__ is exc


def test_handle_exception_reraise_as_specific():
    """Test wrapping in a specific subclass of AnkigenError."""
    mock_logger = Mock()
    exc = RuntimeError("runtime")
    message = "Generation failed"

    with pytest.raises(CardGenerationError) as excinfo:
        try:
            raise exc
        except RuntimeError as e:
            handle_exception(e, mock_logger, message, reraise_as=CardGenerationError)

    assert isinstance(excinfo.value, CardGenerationError)
    assert f"{message}: {exc}" in str(excinfo.value)
    assert excinfo.value.__cause__ is exc
