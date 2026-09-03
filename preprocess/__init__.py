"""
vietlawbert.preprocess
~~~~~~~~~~~~~~~~~~~~~~
Phân hệ tiền xử lý văn bản quy phạm pháp luật Việt Nam:
- Structure-Aware Legal Chunker: Phân tích cú pháp AST bảo toàn Điều/Khoản/Điểm.
- Legal Text Cleaner: Khử nhiễu tiêu đề hành chính, chữ ký, chuẩn hóa ngày tháng.
- HTML to Markdown Converter: Chuẩn hóa HTML sang Markdown giữ cấu trúc cấp bậc.
- Contextualizer: Làm giàu ngữ cảnh pháp lý qua LLM Engine.
"""

from .text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
    extract_preamble,
)
from .legal_chunker import (
    LegalLevel,
    HierarchyPath,
    LegalChunk,
    chunk_legal_document,
    chunk_by_dieu,
    chunk_by_khoan,
    extract_cross_references,
)
from .html_to_md import HTMLConverter
from .contextualizer import (
    build_prompt,
    call_ollama,
    build_kafka_producer,
)

__all__ = [
    "clean_boilerplate",
    "extract_doc_type",
    "extract_doc_number",
    "extract_effective_date",
    "extract_preamble",
    "LegalLevel",
    "HierarchyPath",
    "LegalChunk",
    "chunk_legal_document",
    "chunk_by_dieu",
    "chunk_by_khoan",
    "extract_cross_references",
    "HTMLConverter",
    "build_prompt",
    "call_ollama",
    "build_kafka_producer",
]