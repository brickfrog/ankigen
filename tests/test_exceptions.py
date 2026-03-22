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
    assert issubclass(ValidationError, AnkigenError)
    assert issubclass(SecurityError, AnkigenError)
    assert issubclass(APIError, AnkigenError)
    assert issubclass(OpenAIAPIError, APIError)
    assert issubclass(Context7APIError, APIError)
    assert issubclass(ExportError, AnkigenError)
    assert issubclass(CardGenerationError, AnkigenError)
    assert issubclass(ConfigurationError, AnkigenError)


def test_handle_exception_reraise():
    mock_logger = MagicMock()
    try:
        raise ValueError("Test error")
    except ValueError as e:
        with pytest.raises(ValueError, match="Test error"):
            handle_exception(e, mock_logger, "Context message")

    mock_logger.error.assert_called_once()
    args, _ = mock_logger.error.call_args
    assert "Context message: Test error" in args[0]


def test_handle_exception_no_reraise():
    mock_logger = MagicMock()
    exc = ValueError("Test error")

    handle_exception(exc, mock_logger, "Context message", reraise=False)

    mock_logger.error.assert_called_once()


def test_handle_exception_reraise_as():
    mock_logger = MagicMock()
    exc = ValueError("Original error")

    with pytest.raises(AnkigenError, match="Context message: Original error"):
        handle_exception(exc, mock_logger, "Context message", reraise_as=AnkigenError)


def test_validation_error():
    with pytest.raises(ValidationError):
        raise ValidationError("invalid")


def test_security_error():
    with pytest.raises(SecurityError):
        raise SecurityError("ssrf")


def test_api_error():
    with pytest.raises(APIError):
        raise APIError("api fail")


def test_export_error():
    with pytest.raises(ExportError):
        raise ExportError("export fail")


def test_card_generation_error():
    with pytest.raises(CardGenerationError):
        raise CardGenerationError("gen fail")


def test_configuration_error():
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("config missing")
