# Module for utility functions (logging, caching, web fetching)

import logging
from logging.handlers import RotatingFileHandler
import sys
import hashlib
import requests
from bs4 import BeautifulSoup
from typing import Any, Optional
import time

# --- Logging Setup ---
_logger_instance = None


def setup_logging() -> logging.Logger:
    """Configure logging to both file and console"""
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    logger = logging.getLogger("ankigen")
    logger.setLevel(logging.DEBUG)  # Keep debug level for the root logger

    # Prevent duplicate handlers if called multiple times (though get_logger should prevent this)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

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


def get_logger() -> logging.Logger:
    """Returns the initialized logger instance."""
    if _logger_instance is None:
        return setup_logging()
    return _logger_instance


# Initialize logger when module is loaded
logger = get_logger()


# --- Caching ---
class ResponseCache:
    """Simple and efficient LRU cache for API responses with proper eviction."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache = {}  # {key: response}
        self._access_order = []  # Track access order for LRU eviction
        self.hits = 0
        self.misses = 0

    def get(self, prompt: str, model: str) -> Optional[Any]:
        """Retrieve item from cache, updating LRU order."""
        cache_key = self._create_key(prompt, model)

        if cache_key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            self.hits += 1
            logger.debug(
                f"Cache HIT: {cache_key[:16]}... (hits={self.hits}, misses={self.misses})"
            )
            return self._cache[cache_key]

        self.misses += 1
        logger.debug(
            f"Cache MISS: {cache_key[:16]}... (hits={self.hits}, misses={self.misses})"
        )
        return None

    def set(self, prompt: str, model: str, response: Any):
        """Store item in cache with LRU eviction when full."""
        cache_key = self._create_key(prompt, model)

        # If key exists, update and move to end
        if cache_key in self._cache:
            self._access_order.remove(cache_key)
        # If cache is full, evict least recently used
        elif len(self._cache) >= self.maxsize:
            evicted_key = self._access_order.pop(0)
            del self._cache[evicted_key]
            logger.debug(
                f"Cache EVICT: {evicted_key[:16]}... (size={len(self._cache)})"
            )

        self._cache[cache_key] = response
        self._access_order.append(cache_key)
        logger.debug(f"Cache SET: {cache_key[:16]}... (size={len(self._cache)})")

    def clear(self) -> None:
        """Clear all cache entries and statistics."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache CLEARED")

    def _create_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model (MD5 hash for size efficiency)."""
        # Hash to keep keys manageable size while maintaining uniqueness
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

    def wait(self) -> None:
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

    if "<" not in text and ">" not in text:
        return text.strip()

    # Use BeautifulSoup for safe HTML parsing
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text().strip()
