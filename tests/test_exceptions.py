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


def test_exception_inheritance():
    """Test that all exception classes have the correct inheritance hierarchy."""
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)
    assert issubclass(AnkigenError, Exception)


def test_exception_instantiation():
    """Test that all exception classes can be instantiated with a message."""
    msg = "test error"
    for exc_class in [
        AnkigenError,
        ValidationError,
        SecurityError,
        APIError,
        OpenAIAPIError,
        Context7APIError,
        ExportError,
        CardGenerationError,
        ConfigurationError,
    ]:
        exc = exc_class(msg)
        assert str(exc) == msg


def test_handle_exception_logging():
    """Test that handle_exception logs with exc_info=True."""
    mock_logger = MagicMock()
    exc = ValueError("original error")
    message = "context message"

    try:
        handle_exception(exc, mock_logger, message, reraise=False)
    except Exception:
        pytest.fail("handle_exception raised an exception when reraise=False")

    mock_logger.error.assert_called_once_with(f"{message}: {exc}", exc_info=True)


def test_handle_exception_reraise_true():
    """Test that handle_exception with reraise=True re-raises original exception."""
    mock_logger = MagicMock()
    exc = ValueError("original error")
    message = "context message"

    with pytest.raises(ValueError) as excinfo:
        try:
            raise exc
        except ValueError as e:
            handle_exception(e, mock_logger, message, reraise=True)

    assert excinfo.value is exc


def test_handle_exception_reraise_false():
    """Test that handle_exception with reraise=False does not raise."""
    mock_logger = MagicMock()
    exc = ValueError("original error")
    message = "context message"

    # Should not raise
    handle_exception(exc, mock_logger, message, reraise=False)


def test_handle_exception_reraise_as():
    """Test that handle_exception with reraise_as wraps and re-raises as specified type."""
    mock_logger = MagicMock()
    exc = ValueError("original error")
    message = "context message"

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(exc, mock_logger, message, reraise_as=AnkigenError)

    assert isinstance(excinfo.value, AnkigenError)
    assert f"{message}: {exc}" in str(excinfo.value)
    assert excinfo.value.__cause__ is exc


def test_handle_exception_reraise_as_with_reraise_false():
    """Test that handle_exception with reraise=False and reraise_as does NOT raise."""
    mock_logger = MagicMock()
    exc = ValueError("original error")
    message = "context message"

    # Should not raise even if reraise_as is provided, if reraise is False
    handle_exception(exc, mock_logger, message, reraise=False, reraise_as=AnkigenError)
    mock_logger.error.assert_called_once()
