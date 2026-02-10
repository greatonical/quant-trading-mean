# utils/logger.py
import logging
from logging import Logger

def get_logger(name: str = "quant", level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


