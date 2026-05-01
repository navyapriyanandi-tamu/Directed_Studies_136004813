"""
Logger Module
Provides logging functionality for the application
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = 'task_manager', level: int = logging.INFO):
    """
    Set up and configure logger

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str = 'task_manager'):
    """Get existing logger instance"""
    return logging.getLogger(name)


# DEAD CODE - Never used
def log_to_file(message: str, filename: str = 'app.log'):
    """
    Log message to file
    This function was created but never integrated
    """
    with open(filename, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - {message}\n")
