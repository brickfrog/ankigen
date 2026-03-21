import pytest
import hashlib
from unittest.mock import MagicMock, patch
from ankigen_core.utils import (
    ResponseCache,
    RateLimiter,
    strip_html_tags,
    fetch_webpage_text,
    setup_logging,
    get_logger,
)
import requests

# --- ResponseCache Tests ---


def test_response_cache_get_set():
    cache = ResponseCache(maxsize=2)
    cache.set("prompt1", "gpt-4", "response1")
    assert cache.get("prompt1", "gpt-4") == "response1"
    assert cache.hits == 1
    assert cache.misses == 0


def test_response_cache_miss():
    cache = ResponseCache(maxsize=2)
    assert cache.get("nonexistent", "gpt-4") is None
    assert cache.hits == 0
    assert cache.misses == 1


def test_response_cache_eviction():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.set("p2", "m1", "r2")
    cache.set("p3", "m1", "r3")  # Should evict p1

    assert cache.get("p1", "m1") is None
    assert cache.get("p2", "m1") == "r2"
    assert cache.get("p3", "m1") == "r3"


def test_response_cache_lru_behavior():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.set("p2", "m1", "r2")
    cache.get("p1", "m1")  # p1 is now MRU, p2 is LRU
    cache.set("p3", "m1", "r3")  # Should evict p2

    assert cache.get("p2", "m1") is None
    assert cache.get("p1", "m1") == "r1"


def test_response_cache_clear():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.get("p1", "m1")
    cache.clear()
    assert cache.get("p1", "m1") is None
    assert cache.hits == 0
    assert cache.misses == 1  # Clear doesn't prevent future misses


def test_response_cache_create_key():
    cache = ResponseCache()
    prompt = "test prompt"
    model = "test model"
    expected_hash = hashlib.md5(f"{model}:{prompt}".encode("utf-8")).hexdigest()
    assert cache._create_key(prompt, model) == expected_hash


# --- RateLimiter Tests ---


def test_rate_limiter_init():
    rl = RateLimiter(2.0)
    assert rl.min_interval_seconds == 0.5


def test_rate_limiter_invalid_input():
    with pytest.raises(ValueError, match="Requests per second must be positive."):
        RateLimiter(0)
    with pytest.raises(ValueError, match="Requests per second must be positive."):
        RateLimiter(-1.0)


@patch("time.monotonic")
@patch("time.sleep")
def test_rate_limiter_wait(mock_sleep, mock_monotonic):
    # Setup: 1 request per second (1.0s interval)
    rl = RateLimiter(1.0)

    # First request
    mock_monotonic.return_value = 100.0
    rl.wait()
    assert rl.last_request_timestamp == 100.0
    mock_sleep.assert_not_called()

    # Second request immediately after (0.1s later)
    mock_monotonic.return_value = 100.1
    rl.wait()
    # Should wait for 0.9s (1.0 - 0.1)
    mock_sleep.assert_called_once_with(pytest.approx(0.9))


# --- strip_html_tags Tests ---


def test_strip_html_tags_basic():
    html = "<div>Hello <b>World</b></div>"
    assert strip_html_tags(html) == "Hello World"


def test_strip_html_tags_empty():
    assert strip_html_tags("") == ""


def test_strip_html_tags_non_string():
    assert strip_html_tags(None) == "None"
    assert strip_html_tags(123) == "123"


# --- fetch_webpage_text Tests ---


@patch("requests.get")
def test_fetch_webpage_text_success(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html><body><main>Main content</main></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    content = fetch_webpage_text("http://example.com")
    assert content == "Main content"


@patch("requests.get")
def test_fetch_webpage_text_fallback_to_body(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html><body>Just body content</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    content = fetch_webpage_text("http://example.com")
    assert content == "Just body content"


@patch("requests.get")
def test_fetch_webpage_text_network_error(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

    with pytest.raises(ConnectionError):
        fetch_webpage_text("http://example.com")


@patch("requests.get")
def test_fetch_webpage_text_missing_content(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    content = fetch_webpage_text("http://example.com")
    assert content == ""


# --- Logging Tests ---


def test_get_logger_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2
    assert logger1.name == "ankigen"


def test_setup_logging_behavior():
    # Calling setup_logging multiple times should return the same logger
    logger1 = setup_logging()
    logger2 = setup_logging()
    assert logger1 is logger2
    assert len(logger1.handlers) >= 2  # Should have at least File and Stream handlers
