# Module for utility functions (logging, caching, web fetching)

import logging
from logging.handlers import RotatingFileHandler
import sys
import hashlib
import requests
from bs4 import BeautifulSoup
from functools import lru_cache
from typing import Any, Optional
import time

# --- Logging Setup ---
_logger_instance = None


def setup_logging():
    """Configure logging to both file and console"""
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    logger = logging.getLogger("ankigen")
    logger.setLevel(logging.DEBUG)  # Keep debug level for the root logger

    # Prevent duplicate handlers if called multiple times (though get_logger should prevent this)
    if logger.hasHandlers():
        logger.handlers.clear()

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    file_handler = RotatingFileHandler(
        "ankigen.log", maxBytes=1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # File handler logs everything from DEBUG up
    file_handler.setFormatter(detailed_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console handler logs INFO and above
    console_handler.setFormatter(simple_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _logger_instance = logger
    return logger


def get_logger():
    """Returns the initialized logger instance."""
    if _logger_instance is None:
        return setup_logging()
    return _logger_instance


# Initialize logger when module is loaded
logger = get_logger()


# --- Caching ---
class ResponseCache:
    """A simple cache for API responses using LRU for get operations."""

    def __init__(self, maxsize=128):
        # This internal method will be decorated by lru_cache
        self._internal_get_from_dict = self._get_from_dict_actual
        self._lru_cached_get = lru_cache(maxsize=maxsize)(self._internal_get_from_dict)
        self._dict_cache = {}  # Main store for set operations

    def _get_from_dict_actual(self, cache_key: str):
        """Actual dictionary lookup, intended to be wrapped by lru_cache."""
        logger.debug(f"Cache DICT GET: key={cache_key}")
        return self._dict_cache.get(cache_key)

    def get(self, prompt: str, model: str) -> Optional[Any]:
        """Retrieves an item from the cache. Uses LRU for this get path."""
        cache_key = self._create_key(prompt, model)
        # Use the LRU cached getter which looks up in _dict_cache
        return self._lru_cached_get(cache_key)

    def set(self, prompt: str, model: str, response: Any):
        """Sets an item in the cache."""
        cache_key = self._create_key(prompt, model)
        logger.debug(f"Cache SET: key={cache_key}, type={type(response)}")
        self._dict_cache[cache_key] = response
        # To make the LRU cache aware of this new item for subsequent gets:
        # We can call the LRU getter so it caches it, or clear specific lru entry if updating.
        # For simplicity, if a new item is set, a subsequent get will fetch and cache it via LRU.
        # Or, we can "prime" the lru_cache, but that's more complex.
        # Current approach: set updates _dict_cache. Next get for this key will use _lru_cached_get,
        # which will fetch from _dict_cache and then be LRU-managed.

    def _create_key(self, prompt: str, model: str) -> str:
        """Creates a unique MD5 hash key for caching."""
        return hashlib.md5(f"{model}:{prompt}".encode("utf-8")).hexdigest()


# --- Web Content Fetching ---
def fetch_webpage_text(url: str) -> str:
    """Fetches and extracts main text content from a URL."""
    logger_util = get_logger()  # Use the logger from this module
    try:
        logger_util.info(f"Fetching content from URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        logger_util.debug(f"Parsing HTML content for {url}")
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except ImportError:  # Keep existing fallback
            logger_util.warning("lxml not found, using html.parser instead.")
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:  # Catch other BeautifulSoup init errors
            logger_util.error(
                f"BeautifulSoup initialization failed for {url}: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to parse HTML content for {url}.")

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        main_content = soup.find("main")
        if not main_content:
            main_content = soup.find("article")

        if main_content:
            text = main_content.get_text()
            logger_util.debug(f"Extracted text from <{main_content.name}> tag.")
        else:
            body = soup.find("body")
            if body:
                text = body.get_text()
                logger_util.debug("Extracted text from <body> tag (fallback).")
            else:
                text = ""
                logger_util.warning(f"Could not find <body> tag in {url}")

        # Simpler text cleaning: join stripped lines
        lines = (line.strip() for line in text.splitlines())
        cleaned_text = "\n".join(line for line in lines if line)

        if not cleaned_text:
            logger_util.warning(f"Could not extract meaningful text from {url}")
            return ""

        logger_util.info(
            f"Successfully extracted text from {url} (Length: {len(cleaned_text)} chars)"
        )
        return cleaned_text

    except requests.exceptions.RequestException as e:
        logger_util.error(f"Network error fetching URL {url}: {e}", exc_info=True)
        raise ConnectionError(f"Could not fetch URL: {e}")
    except Exception as e:
        logger_util.error(f"Error processing URL {url}: {e}", exc_info=True)
        if isinstance(e, (ValueError, ConnectionError, RuntimeError)):
            raise e
        else:
            raise RuntimeError(
                f"An unexpected error occurred while processing the URL: {e}"
            )


# --- New Synchronous RateLimiter Class ---
class RateLimiter:
    """A simple synchronous rate limiter."""

    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("Requests per second must be positive.")
        self.min_interval_seconds: float = 1.0 / requests_per_second
        self.last_request_timestamp: float = 0.0
        # Use a lock if this were to be used by multiple threads, but for now assuming single thread access per instance

    def wait(self):
        """Blocks until it's safe to make the next request."""
        current_time = time.monotonic()  # Use monotonic clock for intervals
        time_since_last_request = current_time - self.last_request_timestamp

        if time_since_last_request < self.min_interval_seconds:
            wait_duration = self.min_interval_seconds - time_since_last_request
            # logger.debug(f"RateLimiter waiting for {wait_duration:.3f} seconds.") # Optional: add logging
            time.sleep(wait_duration)

        self.last_request_timestamp = time.monotonic()


# --- Existing Utility Functions (if any) ---
# def some_other_util_function():
#     pass


def strip_html_tags(text: str) -> str:
    """Removes HTML tags from a string using a safe, non-regex approach."""
    if not isinstance(text, str):
        return str(text)  # Ensure it's a string, or return as is if not coercible

    # Use BeautifulSoup for safe HTML parsing
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text().strip()
