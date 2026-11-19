"""
configs/logging_config.py - Trung tâm điều phối hệ thống nhật ký phân cấp VietLawBERT.
Khắc phục triệt để lỗi nhân bản FileHandler qua resolve() và bao quát toàn bộ namespace logger.
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
    """Thiết lập cấu hình phân cấp cho Root Logger điều phối tập trung."""
    ensure_dirs()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Dọn dẹp handlers cũ để tránh duplicate log khi reload module
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Handler: Xuất INFO ra console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Centralized Error Sink: Thu gom toàn bộ WARNING và ERROR
    error_file = DAILY_LOGS_DIR / "system_errors.log"
    error_file.parent.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(str(error_file.resolve()), encoding="utf-8", mode="a")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)


def get_subsystem_logger(subsystem_name: str, log_filename: Optional[str] = None) -> logging.Logger:
    """
    Tạo hoặc lấy Logger thuộc namespace 'vietlawbert.<subsystem_name>'.
    Sử dụng resolve() đường dẫn tuyệt đối để triệt tiêu lỗi gắn trùng lặp FileHandler.
    """
    ensure_dirs()
    logger_name = f"vietlawbert.{subsystem_name}"
    sub_logger = logging.getLogger(logger_name)

    target_file = (DAILY_LOGS_DIR / f"{log_filename or subsystem_name}.log").resolve()
    target_file.parent.mkdir(parents=True, exist_ok=True)

    has_file_handler = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == target_file
        for h in sub_logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(str(target_file), encoding="utf-8", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        sub_logger.addHandler(file_handler)

    sub_logger.propagate = True
    return sub_logger