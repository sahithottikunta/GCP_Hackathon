"""
SENTINEL — Structured Logging
"""

import logging
import sys
from datetime import datetime


class SentinelFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m",# Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        module = record.name.replace("sentinel.", "")
        return (
            f" {color}{'▸'} {record.levelname:<8}{self.RESET} "
            f"\033[90m{ts}\033[0m "
            f"\033[94m[{module}]\033[0m "
            f"{record.getMessage()}"
        )


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"sentinel.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SentinelFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
