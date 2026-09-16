"""
vietlawbert.utils
~~~~~~~~~~~~~~~~~
Phân hệ tiện ích bổ trợ cho hệ sinh thái VietLawBERT:
- api_client: Client giao tiếp với cổng API dữ liệu VBPL có Session Pooling và Retry.
- helpers: Bộ tiện ích xử lý tên tệp, chuỗi pháp lý, phân lô và đo đạc thời gian.
- model_router: Bộ định tuyến mô hình AI theo tác vụ (Text, OCR, Embeddings, Reranker).
"""

from .api_client import VBPLApiClient
from .helpers import (
    get_clean_filename,
    slugify_doc_number,
    batch_iterator,
    timing_decorator,
    safe_truncate_text,
)
from .model_router import ModelRouter

__all__ = [
    "VBPLApiClient",
    "ModelRouter",
    "get_clean_filename",
    "slugify_doc_number",
    "batch_iterator",
    "timing_decorator",
    "safe_truncate_text",
]