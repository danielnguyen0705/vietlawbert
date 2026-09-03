"""
helpers.py - Bộ hàm tiện ích cốt lõi cho tiền xử lý, thao tác tệp và profiling.
Đảm bảo an toàn hệ điều hành trên cả Windows lẫn Linux POSIX.
"""

from __future__ import annotations

import re
import time
import logging
from functools import wraps
from typing import Iterator, List, Any, Callable

logger = logging.getLogger("VietLawBERT_Helpers")


def slugify_doc_number(doc_number: str) -> str:
    """
    Chuyển đổi số hiệu văn bản thành chuỗi an toàn cho hệ thống tệp.
    Ví dụ: '100/2019/NĐ-CP' -> '100_2019_ND-CP'.
    """
    if not doc_number:
        return "Unknown_Doc"
    # Thay thế các ký tự cấm trên Windows và Linux
    clean = re.sub(r'[\\/*?:"<>|]', "_", str(doc_number).strip())
    # Rút gọn các dấu gạch dưới liên tiếp
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_")


def get_clean_filename(doc_number: Any, item_id: Any) -> str:
    """
    Tạo tên tệp định danh duy nhất và an toàn cho hệ thống tệp.
    Ưu tiên: '{item_id}_{slug_doc_number}'. Nếu thiếu, sử dụng item_id.
    """
    clean_id = str(item_id or "").strip()
    clean_num = slugify_doc_number(str(doc_number or ""))

    if clean_id and clean_num and clean_num != "Unknown_Doc":
        return f"{clean_id}_{clean_num}"
    if clean_id:
        return clean_id
    if clean_num and clean_num != "Unknown_Doc":
        return clean_num
    return "Unknown_Document"


def batch_iterator(iterable: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Tạo bộ lặp phân tách danh sách thành các lô có kích thước cố định."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def safe_truncate_text(text: str, max_chars: int = 1500) -> str:
    """Cắt ngắn chuỗi văn bản ở biên từ tự nhiên, không làm vỡ từ."""
    if not text or len(text) <= max_chars:
        return text or ""
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + "..."
    return truncated + "..."


def timing_decorator(func: Callable) -> Callable:
    """Decorator đo đạc thời gian thực thi (Latency Profiling) của hàm."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"[PROFILER] Hàm '{func.__name__}' thực thi trong {elapsed_time:.4f} giây.")
        return result
    return wrapper