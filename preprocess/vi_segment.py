"""
preprocess/vi_segment.py — Trợ thủ tách từ tiếng Việt (Vietnamese Word Segmentation).

Hỗ trợ các thư viện:
- pyvi: ViTokenizer.tokenize (nhanh, quy tắc + CRF nhẹ)
- underthesea: word_tokenize(..., format="text")
"""
from __future__ import annotations

import warnings
from typing import Callable
from importlib.metadata import version, PackageNotFoundError


def get_segmenter(name: str = "pyvi") -> Callable[[str], str]:
    """
    Trả về hàm tách từ (Callable[[str], str]) theo tên thư viện.
    Lazy import bên trong để tránh tải chậm khi không dùng.
    """
    norm_name = (name or "").strip().lower()
    if norm_name == "pyvi":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyvi import ViTokenizer
        return lambda text: ViTokenizer.tokenize(text)
    elif norm_name == "underthesea":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from underthesea import word_tokenize
        return lambda text: word_tokenize(text, format="text")
    else:
        raise ValueError(f"Không hỗ trợ segmenter: {name!r}. Chỉ hỗ trợ 'pyvi' hoặc 'underthesea'.")


def segment_text(text: str, seg_fn: Callable[[str], str]) -> str:
    """
    Tách từ cho chuỗi văn bản:
    - Trả về "" nếu chuỗi rỗng hoặc chỉ toàn khoảng trắng.
    - Chuẩn hóa khoảng trắng dư thừa thành dấu cách đơn.
    - Loại bỏ khoảng trắng đầu/cuối (strip).
    """
    if not text or not text.strip():
        return ""
    segmented = seg_fn(text)
    return " ".join(segmented.split())


def segmenter_version(name: str) -> str:
    """Trả về chuỗi phiên bản của thư viện tách từ."""
    norm_name = (name or "").strip().lower()
    try:
        return version(norm_name)
    except PackageNotFoundError:
        return "unknown"


_worker_seg_fn = None


def init_worker_segmenter(name: str = "pyvi") -> None:
    """Khởi tạo segmenter cho worker process trong multiprocessing."""
    global _worker_seg_fn
    _worker_seg_fn = get_segmenter(name)


def worker_segment_tuple(item: tuple[str, str]) -> tuple[str, str]:
    """Worker task nhận (chunk_id, content), trả về (chunk_id, content_seg)."""
    global _worker_seg_fn
    if _worker_seg_fn is None:
        _worker_seg_fn = get_segmenter("pyvi")
    chunk_id, text = item
    return chunk_id, segment_text(text, _worker_seg_fn)
