"""
日志模块.
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger

_logger_initialized = False


def setup_logger(
    log_level: str = "INFO",
    log_path: str = "./logs/app.log",
    rotation: str = "10 MB",
    retention: int = 5,
):
    global _logger_initialized
    if _logger_initialized:
        return
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )
    _logger_initialized = True
    loguru_logger.info("Logger initialized.")


def get_logger(name: Optional[str] = None):
    if not _logger_initialized:
        setup_logger()
    return loguru_logger.bind(name=name or "root")
