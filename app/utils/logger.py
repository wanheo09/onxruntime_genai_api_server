import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 포매터 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """이름으로 로거 가져오기"""
    return logging.getLogger(name)
