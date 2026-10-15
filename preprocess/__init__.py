"""
vietlawbert.preprocess
~~~~~~~~~~~~~~~~~~~~~~
Phân hệ tiền xử lý văn bản quy phạm pháp luật Việt Nam (Kiến trúc v3):
- Hybrid Legal AST Parser: Bóc tách cây cú pháp EBNF và tiêm Metadata/Hierarchy tất định.
- Legal Text Cleaner: Khử nhiễu tiêu ngữ, chữ ký, trích xuất ngày hiệu lực ISO 8601.
- HTML to Markdown Converter: Chuyển đổi định dạng HTML sang Markdown giữ nguyên cấu trúc.
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
from .html_to_md import HTMLConverter

__all__ = [
    # AST Parser & Phân cấp
    "HybridASTParser",
    "LegalHierarchyTracker",
    # Làm sạch & Trích xuất Metadata
    "clean_boilerplate",
    "extract_doc_type",
    "extract_doc_number",
    "extract_effective_date",
    "extract_preamble",
    "normalize_unicode",
    # Chuyển đổi HTML
    "HTMLConverter",
]