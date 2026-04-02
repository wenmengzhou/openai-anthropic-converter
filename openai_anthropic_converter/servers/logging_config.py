"""
Shared logging configuration for proxy servers.

Sets up logging with:
- Console handler (stdout) with colored level
- File handler (rotating) with full details
- Format: timestamp, level, filename:lineno, message
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    server_name: str,
    level: str = "info",
    log_dir: str | None = None,
) -> None:
    """
    Configure logging for a server with both console and file output.

    Args:
        server_name: Name used for log file (e.g. "openai_server" -> logs/openai_server.log)
        level: Log level string ("debug", "info", "warning", "error")
        log_dir: Directory for log files (default: logs/)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_dir = log_dir or os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler → stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # File handler → rotating log file
    log_file = os.path.join(log_dir, f"{server_name}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Remove any existing handlers (e.g. from basicConfig)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger(server_name).info(
        "Logging initialized: level=%s, file=%s", level, log_file
    )
