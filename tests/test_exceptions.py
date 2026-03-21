import pytest
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


def test_handle_exception_reraise(mocker):
    logger = mocker.Mock()
    exc = ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        try:
            raise exc
        except ValueError as e:
            handle_exception(e, logger, "An error occurred", reraise=True)

    logger.error.assert_called_once()


def test_handle_exception_no_reraise(mocker):
    logger = mocker.Mock()
    exc = ValueError("test error")

    handle_exception(exc, logger, "An error occurred", reraise=False)

    logger.error.assert_called_once()


def test_handle_exception_reraise_as(mocker):
    logger = mocker.Mock()
    exc = ValueError("test error")

    with pytest.raises(AnkigenError, match="An error occurred: test error"):
        handle_exception(
            exc, logger, "An error occurred", reraise=True, reraise_as=AnkigenError
        )

    logger.error.assert_called_once()
