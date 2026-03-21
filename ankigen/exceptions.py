"""Custom exceptions for AnkiGen application.

This module provides a hierarchy of custom exceptions to standardize
error handling across the codebase.
"""


class AnkigenError(Exception):
    """Base exception for all AnkiGen errors."""

    pass


class ValidationError(AnkigenError):
    """Raised when input validation fails."""

    pass


class SecurityError(AnkigenError):
    """Raised when a security check fails (SSRF, command injection, etc.)."""

    pass


class APIError(AnkigenError):
    """Base exception for API-related errors."""

    pass


class OpenAIAPIError(APIError):
    """Raised when OpenAI API calls fail."""

    pass


class Context7APIError(APIError):
    """Raised when Context7 API calls fail."""

    pass


class ExportError(AnkigenError):
    """Base exception for export-related errors."""

    pass


class CardGenerationError(AnkigenError):
    """Raised when card generation fails."""

    pass


class ConfigurationError(AnkigenError):
    """Raised when configuration is invalid or missing."""

    pass


def handle_exception(
    exc: Exception,
    logger,
    message: str,
    reraise: bool = True,
    reraise_as: type[Exception] | None = None,
) -> None:
    """Standardized exception handler.

    Args:
        exc: The exception to handle
        logger: Logger instance to use
        message: Error message to log
        reraise: Whether to re-raise the exception
        reraise_as: Optional exception type to wrap and re-raise as

    Raises:
        The original exception or wrapped exception if reraise is True
    """
    logger.error(f"{message}: {exc}", exc_info=True)

    if reraise:
        if reraise_as:
            raise reraise_as(f"{message}: {exc}") from exc
        raise exc
