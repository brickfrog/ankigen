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

# --- Exception Hierarchy Tests ---


def test_exception_hierarchy():
    """Verify the inheritance hierarchy of custom exceptions."""
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)

    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)

    # Verify grand-parent inheritance
    assert issubclass(OpenAIAPIError, AnkigenError)
    assert issubclass(Context7APIError, AnkigenError)


def test_catch_by_parent():
    """Verify exceptions can be caught by their parent classes."""
    try:
        raise ValidationError("test")
    except AnkigenError:
        pass
    else:
        pytest.fail("ValidationError not caught by AnkigenError")

    try:
        raise OpenAIAPIError("test")
    except APIError:
        pass
    except AnkigenError:
        pytest.fail(
            "OpenAIAPIError caught by AnkigenError but should have been caught by APIError first"
        )
    else:
        pytest.fail("OpenAIAPIError not caught by APIError")

    try:
        raise OpenAIAPIError("test")
    except AnkigenError:
        pass
    else:
        pytest.fail("OpenAIAPIError not caught by AnkigenError")


# --- handle_exception Tests ---


def test_handle_exception_logs_error():
    """Verify handle_exception calls logger.error with correct message and exc_info=True."""
    mock_logger = MagicMock()
    exc = ValueError("test error")
    message = "Context message"

    try:
        raise exc
    except ValueError as e:
        with pytest.raises(ValueError):
            handle_exception(e, mock_logger, message, reraise=True)

    mock_logger.error.assert_called_once_with(
        "Context message: test error", exc_info=True
    )


def test_handle_exception_reraise_true():
    """Verify handle_exception re-raises the original exception when reraise=True."""
    mock_logger = MagicMock()
    exc = ValueError("original error")

    try:
        raise exc
    except ValueError as e:
        with pytest.raises(ValueError) as excinfo:
            handle_exception(e, mock_logger, "msg", reraise=True)
        assert excinfo.value is e


def test_handle_exception_reraise_false():
    """Verify handle_exception does not re-raise when reraise=False."""
    mock_logger = MagicMock()
    exc = ValueError("original error")

    # Should not raise any exception
    handle_exception(exc, mock_logger, "msg", reraise=False)

    mock_logger.error.assert_called_once()


def test_handle_exception_reraise_as():
    """Verify handle_exception wraps the exception in reraise_as type."""
    mock_logger = MagicMock()
    original_exc = ValueError("original")

    with pytest.raises(AnkigenError) as excinfo:
        handle_exception(
            original_exc,
            mock_logger,
            "New context",
            reraise=True,
            reraise_as=AnkigenError,
        )

    assert isinstance(excinfo.value, AnkigenError)
    assert "New context: original" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_exc


def test_handle_exception_message_formatting():
    """Verify the message formatting in handle_exception."""
    mock_logger = MagicMock()
    exc = RuntimeError("runtime failure")

    # Test reraise=False logs correctly
    handle_exception(exc, mock_logger, "Operation failed", reraise=False)
    mock_logger.error.assert_called_with(
        "Operation failed: runtime failure", exc_info=True
    )

    # Test reraise_as includes both messages
    with pytest.raises(ExportError) as excinfo:
        handle_exception(
            exc, mock_logger, "Export failed", reraise=True, reraise_as=ExportError
        )

    assert str(excinfo.value) == "Export failed: runtime failure"


def test_api_error_subclasses():
    """Verify specific API error subclasses."""
    try:
        raise OpenAIAPIError("openai error")
    except OpenAIAPIError as e:
        assert str(e) == "openai error"
        assert isinstance(e, APIError)
        assert isinstance(e, AnkigenError)

    try:
        raise Context7APIError("context7 error")
    except Context7APIError as e:
        assert str(e) == "context7 error"
        assert isinstance(e, APIError)
        assert isinstance(e, AnkigenError)


def test_all_exceptions_instantiation():
    """Verify all custom exceptions can be instantiated."""
    exceptions = [
        AnkigenError,
        ValidationError,
        SecurityError,
        APIError,
        OpenAIAPIError,
        Context7APIError,
        ExportError,
        CardGenerationError,
        ConfigurationError,
    ]
    for exc_class in exceptions:
        msg = f"Test message for {exc_class.__name__}"
        instance = exc_class(msg)
        assert str(instance) == msg


def test_handle_exception_with_standard_logger(caplog):
    """Verify handle_exception works with a real standard logger."""
    logger = logging.getLogger("test_logger")
    exc = Exception("real error")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception):
            handle_exception(exc, logger, "Testing standard logger", reraise=True)

    assert "Testing standard logger: real error" in caplog.text
    # Check that exc_info was used (it should include the traceback in caplog)
    # Note: caplog.text might not show full traceback depending on config,
    # but the log record should have exc_info.
    for record in caplog.records:
        if record.message == "Testing standard logger: real error":
            assert record.exc_info is not None


def test_handle_exception_custom_message_none():
    """Test handle_exception with an empty message or unusual string."""
    mock_logger = MagicMock()
    exc = Exception("error")

    handle_exception(exc, mock_logger, "", reraise=False)
    mock_logger.error.assert_called_with(": error", exc_info=True)
