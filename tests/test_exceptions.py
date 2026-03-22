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
    """Test that all custom exceptions inherit correctly."""
    assert issubclass(AnkigenError, Exception)
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)


def test_handle_exception_logging():
    """Test that handle_exception logs the error correctly."""
    logger = MagicMock()
    exc = ValueError("test error")
    message = "An error occurred"

    # We use reraise=False to avoid catching exception here
    handle_exception(exc, logger, message, reraise=False)

    logger.error.assert_called_once_with(f"{message}: {exc}", exc_info=True)


def test_handle_exception_default_reraise():
    """Test that handle_exception re-raises the original exception by default."""
    logger = MagicMock()
    message = "An error occurred"

    try:
        raise ValueError("test error")
    except ValueError as exc:
        with pytest.raises(ValueError) as excinfo:
            handle_exception(exc, logger, message)
        assert excinfo.value is exc


def test_handle_exception_no_reraise():
    """Test that handle_exception does not re-raise when reraise=False."""
    logger = MagicMock()
    exc = ValueError("test error")
    message = "An error occurred"

    # Should not raise
    handle_exception(exc, logger, message, reraise=False)


def test_handle_exception_reraise_as():
    """Test that handle_exception wraps the exception in reraise_as when specified."""
    logger = MagicMock()
    exc = ValueError("original error")
    message = "Custom message"

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(exc, logger, message, reraise_as=AnkigenError)

    assert isinstance(excinfo.value, AnkigenError)
    assert f"{message}: {exc}" in str(excinfo.value)
    assert excinfo.value.__cause__ is exc


def test_handle_exception_reraise_as_specific():
    """Test handle_exception with a specific subclass in reraise_as."""
    logger = MagicMock()
    exc = RuntimeError("runtime error")
    message = "Validation failed"

    with pytest.raises(ValidationError) as excinfo:
        handle_exception(exc, logger, message, reraise_as=ValidationError)

    assert isinstance(excinfo.value, ValidationError)
    assert "Validation failed: runtime error" in str(excinfo.value)
    assert excinfo.value.__cause__ is exc
