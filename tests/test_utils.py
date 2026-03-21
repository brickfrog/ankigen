import pytest
import hashlib
from unittest.mock import MagicMock
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


def test_cache_set_get():
    cache = ResponseCache(maxsize=2)
    cache.set("prompt1", "model1", "response1")
    assert cache.get("prompt1", "model1") == "response1"
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss():
    cache = ResponseCache(maxsize=2)
    assert cache.get("nonexistent", "model1") is None
    assert cache.hits == 0
    assert cache.misses == 1


def test_cache_eviction():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.set("p2", "m2", "r2")
    cache.set("p3", "m3", "r3")  # Should evict p1

    assert cache.get("p1", "m1") is None
    assert cache.get("p2", "m2") == "r2"
    assert cache.get("p3", "m3") == "r3"


def test_cache_lru_update():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.set("p2", "m2", "r2")
    cache.get("p1", "m1")  # p1 is now MRU
    cache.set("p3", "m3", "r3")  # Should evict p2

    assert cache.get("p2", "m2") is None
    assert cache.get("p1", "m1") == "r1"
    assert cache.get("p3", "m3") == "r3"


def test_cache_clear():
    cache = ResponseCache(maxsize=2)
    cache.set("p1", "m1", "r1")
    cache.hits = 5
    cache.misses = 2
    cache.clear()

    assert cache.get("p1", "m1") is None
    assert cache.hits == 0
    assert cache.misses == 1  # Miss from the get() call above


def test_cache_key_hashing():
    cache = ResponseCache()
    prompt = "test prompt"
    model = "gpt-4"
    expected_key = hashlib.md5(f"{model}:{prompt}".encode("utf-8")).hexdigest()
    assert cache._create_key(prompt, model) == expected_key


# --- RateLimiter Tests ---


def test_rate_limiter_init_invalid():
    with pytest.raises(ValueError, match="Requests per second must be positive."):
        RateLimiter(0)
    with pytest.raises(ValueError, match="Requests per second must be positive."):
        RateLimiter(-1)


def test_rate_limiter_wait(mocker):
    # Mock time.monotonic and time.sleep
    mock_monotonic = mocker.patch("time.monotonic")
    mock_sleep = mocker.patch("time.sleep")

    # First call: current_time = 10.0, last_request = 0.0
    # Interval = 1.0 (for 1 req/sec)
    # diff = 10.0 >= 1.0, no sleep
    mock_monotonic.side_effect = [10.0, 10.1, 10.2, 10.3]

    limiter = RateLimiter(1.0)
    limiter.wait()
    assert mock_sleep.call_count == 0

    # Second call: current_time = 10.2, last_request = 10.1 (from end of first wait)
    # diff = 0.1 < 1.0, sleep for 0.9
    limiter.wait()
    mock_sleep.assert_called_once_with(pytest.approx(0.9))


# --- strip_html_tags Tests ---


def test_strip_html_tags_normal():
    html = "<div>Hello <b>World</b></div>"
    assert strip_html_tags(html) == "Hello World"


def test_strip_html_tags_empty():
    assert strip_html_tags("") == ""


def test_strip_html_tags_non_string():
    assert strip_html_tags(None) == "None"
    assert strip_html_tags(123) == "123"


# --- fetch_webpage_text Tests ---


def test_fetch_webpage_text_success(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = MagicMock()
    mock_response.text = "<html><body><main>Relevant content</main></body></html>"
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    result = fetch_webpage_text("http://example.com")
    assert result == "Relevant content"


def test_fetch_webpage_text_article_fallback(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = MagicMock()
    mock_response.text = "<html><body><article>Article content</article></body></html>"
    mock_get.return_value = mock_response

    result = fetch_webpage_text("http://example.com")
    assert result == "Article content"


def test_fetch_webpage_text_body_fallback(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = MagicMock()
    mock_response.text = "<html><body>Body content</body></html>"
    mock_get.return_value = mock_response

    result = fetch_webpage_text("http://example.com")
    assert result == "Body content"


def test_fetch_webpage_text_network_error(mocker):
    mock_get = mocker.patch("requests.get")
    mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

    with pytest.raises(ConnectionError, match="Could not fetch URL"):
        fetch_webpage_text("http://example.com")


def test_fetch_webpage_text_empty_content(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = MagicMock()
    mock_response.text = "<html><body></body></html>"
    mock_get.return_value = mock_response

    assert fetch_webpage_text("http://example.com") == ""


# --- Logging Tests ---


def test_get_logger_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_setup_logging_idempotent():
    # Calling setup_logging twice should return same logger
    logger1 = setup_logging()
    logger2 = setup_logging()
    assert logger1 is logger2
