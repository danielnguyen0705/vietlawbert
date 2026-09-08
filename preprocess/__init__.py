"""
vietlawbert.preprocess
~~~~~~~~~~~~~~~~~~~~~~
Phân hệ tiền xử lý văn bản quy phạm pháp luật Việt Nam (Kiến trúc v3):
- Hybrid Legal AST Parser: Bóc tách cây cú pháp EBNF và tiêm Metadata/Hierarchy.
- Legal Text Cleaner: Khử nhiễu tiêu ngữ, chữ ký, chuẩn hóa ngày tháng ISO 8601.
- Structure-Aware Legal Chunker: Cắt đoạn dự phòng theo Điều/Khoản/Điểm.
- HTML to Markdown Converter: Chuyển đổi định dạng HTML sang Markdown giữ cấu trúc.
"""

from .ast_parser import HybridASTParser, LegalHierarchyTracker
from .text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
    extract_preamble,
    normalize_unicode,
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

__all__ = [
    # Cốt lõi AST Parser 2026
    "HybridASTParser",
    "LegalHierarchyTracker",
    # Làm sạch & Trích xuất Metadata
    "clean_boilerplate",
    "extract_doc_type",
    "extract_doc_number",
    "extract_effective_date",
    "extract_preamble",
    "normalize_unicode",
    # Baseline Chunker
    "LegalLevel",
    "HierarchyPath",
    "LegalChunk",
    "chunk_legal_document",
    "chunk_by_dieu",
    "chunk_by_khoan",
    "extract_cross_references",
    # Tiện ích HTML
    "HTMLConverter",
]