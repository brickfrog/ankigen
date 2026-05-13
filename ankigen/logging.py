import logging
import os
import sys
from datetime import datetime


def setup_logger(name="ankigen", log_level=logging.INFO):
    """Set up and return a logger with file and console handlers"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers if any
    # This ensures that if setup_logger is called multiple times for the same logger name,
    # it doesn't accumulate handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    # Logs will be stored in ~/.ankigen/logs/
    # A new log file is created each day (e.g., ankigen_20231027.log)
    log_dir = os.path.join(os.path.expanduser("~"), ".ankigen", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create a default logger instance for easy import and use.
# Projects can also create their own named loggers using setup_logger(name="my_module_logger")
logger = setup_logger()
