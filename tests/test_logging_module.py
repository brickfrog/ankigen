import logging
import os
import sys
from datetime import datetime
from ankigen.logging import setup_logger


def test_setup_logger_returns_logger():
    """1) setup_logger returns a logging.Logger"""
    logger = setup_logger("test_returns_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_returns_logger"


def test_setup_logger_handlers(tmp_path, monkeypatch):
    """2) Logger has exactly 2 handlers (StreamHandler + FileHandler)"""
    # Redirect home directory to tmp_path to avoid creating files in actual home
    monkeypatch.setattr(
        "ankigen.logging.os.path.expanduser", lambda x: str(tmp_path) if x == "~" else x
    )

    logger = setup_logger("test_handlers")

    # Check number of handlers
    assert len(logger.handlers) == 2

    handler_types = [type(h) for h in logger.handlers]
    # One should be StreamHandler (for console)
    assert any(
        issubclass(t, logging.StreamHandler) and not issubclass(t, logging.FileHandler)
        for t in handler_types
    )
    # One should be FileHandler
    assert any(issubclass(t, logging.FileHandler) for t in handler_types)

    # Verify StreamHandler uses stdout
    console_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    )
    assert console_handler.stream == sys.stdout


def test_file_handler_path_pattern(tmp_path, monkeypatch):
    """3) File handler writes to the correct path pattern"""
    monkeypatch.setattr(
        "ankigen.logging.os.path.expanduser", lambda x: str(tmp_path) if x == "~" else x
    )

    name = "test_path"
    logger = setup_logger(name)

    file_handler = next(
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    log_file_path = file_handler.baseFilename

    timestamp = datetime.now().strftime("%Y%m%d")
    # Expected pattern: ~/.ankigen/logs/{name}_{YYYYMMDD}.log
    expected_path = tmp_path / ".ankigen" / "logs" / f"{name}_{timestamp}.log"

    assert os.path.abspath(log_file_path) == os.path.abspath(str(expected_path))
    # Check if the directory was created
    assert os.path.exists(os.path.dirname(log_file_path))


def test_setup_logger_clears_old_handlers(tmp_path, monkeypatch):
    """4) Calling setup_logger twice with same name clears old handlers (no duplicates)"""
    monkeypatch.setattr(
        "ankigen.logging.os.path.expanduser", lambda x: str(tmp_path) if x == "~" else x
    )

    name = "test_clear_handlers"
    logger = setup_logger(name)
    assert len(logger.handlers) == 2
    initial_handlers = list(logger.handlers)

    # Call again with same name
    logger2 = setup_logger(name)
    assert logger is logger2
    assert len(logger2.handlers) == 2

    # Ensure handlers were replaced, not accumulated
    for handler in initial_handlers:
        assert handler not in logger2.handlers


def test_custom_log_level_respected():
    """5) Custom log_level is respected"""
    name_debug = "test_log_level_debug"
    logger_debug = setup_logger(name_debug, log_level=logging.DEBUG)
    assert logger_debug.level == logging.DEBUG

    name_info = "test_log_level_info"
    logger_info = setup_logger(name_info, log_level=logging.INFO)
    assert logger_info.level == logging.INFO


def test_custom_name_creates_distinct_logger():
    """6) Custom name creates a distinct logger"""
    logger1 = setup_logger("logger1")
    logger2 = setup_logger("logger2")
    assert logger1.name == "logger1"
    assert logger2.name == "logger2"
    assert logger1 is not logger2
