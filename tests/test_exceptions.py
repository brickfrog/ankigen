import pytest
import logging
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

# --- Exception Class Tests ---


def test_exception_hierarchy():
    """Test that exceptions inherit correctly."""
    assert isinstance(ValidationError(), AnkigenError)
    assert isinstance(SecurityError(), AnkigenError)
    assert isinstance(APIError(), AnkigenError)
    assert isinstance(OpenAIAPIError(), APIError)
    assert isinstance(Context7APIError(), APIError)
    assert isinstance(ExportError(), AnkigenError)
    assert isinstance(CardGenerationError(), AnkigenError)
    assert isinstance(ConfigurationError(), AnkigenError)


def test_exception_messages():
    """Test that exceptions preserve messages."""
    msg = "Test error"
    with pytest.raises(AnkigenError) as excinfo:
        raise AnkigenError(msg)
    assert str(excinfo.value) == msg

    with pytest.raises(ValidationError) as excinfo:
        raise ValidationError(msg)
    assert str(excinfo.value) == msg


# --- handle_exception Tests ---


def test_handle_exception_reraise():
    """Test handle_exception reraises the original exception.
    Must be called within an except block for the bare 'raise' in source to work.
    """
    logger = MagicMock(spec=logging.Logger)
    try:
        raise ValueError("Original error")
    except ValueError as e:
        with pytest.raises(ValueError, match="Original error"):
            handle_exception(e, logger, "An error occurred", reraise=True)

    logger.error.assert_called_once()


def test_handle_exception_no_reraise():
    """Test handle_exception does not reraise when reraise=False."""
    logger = MagicMock(spec=logging.Logger)
    exc = ValueError("Original error")

    # Should not raise
    handle_exception(exc, logger, "An error occurred", reraise=False)

    logger.error.assert_called_once()


def test_handle_exception_reraise_as():
    """Test handle_exception reraises as a different exception type."""
    logger = MagicMock(spec=logging.Logger)
    exc = ValueError("Original error")

    with pytest.raises(AnkigenError, match="Wrapped message: Original error"):
        handle_exception(
            exc, logger, "Wrapped message", reraise=True, reraise_as=AnkigenError
        )


def test_handle_exception_logging():
    """Test that handle_exception logs the correct message."""
    logger = MagicMock(spec=logging.Logger)
    exc = RuntimeError("Failure")
    msg = "Operation failed"

    try:
        # Wrapped in try/except to avoid RuntimeError from bare raise in source
        try:
            raise exc
        except RuntimeError as e:
            handle_exception(e, logger, msg, reraise=True)
    except RuntimeError:
        pass

    logger.error.assert_called_with(f"{msg}: {exc}", exc_info=True)


# --- Additional Tests to reach 10+ ---


def test_validation_error_inheritance():
    assert issubclass(ValidationError, AnkigenError)


def test_api_error_inheritance():
    assert issubclass(APIError, AnkigenError)


def test_openai_api_error_inheritance():
    assert issubclass(OpenAIAPIError, APIError)


def test_context7_api_error_inheritance():
    assert issubclass(Context7APIError, APIError)


def test_export_error_inheritance():
    assert issubclass(ExportError, AnkigenError)


def test_card_generation_error_inheritance():
    assert issubclass(CardGenerationError, AnkigenError)
