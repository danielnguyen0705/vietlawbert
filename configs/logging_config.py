"""
configs/logging_config.py - Trung tâm điều phối hệ thống nhật ký phân cấp VietLawBERT.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional
from configs.paths import DAILY_LOGS_DIR, ensure_dirs

LOG_FORMAT = "%(asctime)s | [%(levelname)s] | %(name)s [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_hierarchical_logging():
    """Thiết lập cấu hình phân cấp cho Root Logger 'vietlawbert'."""
    ensure_dirs()
    root_logger = logging.getLogger("vietlawbert")
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Handler: Chỉ xuất INFO ra terminal để tránh nghẽn I/O
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Centralized Error Sink: Thu gom toàn bộ WARNING và ERROR từ mọi phân hệ
    error_file = DAILY_LOGS_DIR / "system_errors.log"
    error_handler = logging.FileHandler(error_file, encoding="utf-8", mode="a")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)


def get_subsystem_logger(subsystem_name: str, log_filename: Optional[str] = None) -> logging.Logger:
    """
    Tạo hoặc lấy Logger con thuộc namespace 'vietlawbert.<subsystem_name>'.
    Gắn FileHandler chuyên biệt và bật propagate để đẩy lỗi về system_errors.log.
    """
    logger_name = f"vietlawbert.{subsystem_name}"
    logger = logging.getLogger(logger_name)

    target_file = DAILY_LOGS_DIR / f"{log_filename or subsystem_name}.log"
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target_file
        for h in logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(target_file, encoding="utf-8", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    logger.propagate = True
    return logger