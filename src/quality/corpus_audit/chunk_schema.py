# -*- coding: utf-8 -*-
"""
quality/corpus_audit/chunk_schema.py
====================================
Mô hình Pydantic chuẩn hóa cho Lược đồ Phân đoạn Pháp lý (Legal Chunk Schema)
theo đúng đặc tả tại CLAUDE.md §E & §H.2 và Đề cương v3.1 mục 5.1.

Bảo toàn tuyệt đối các trường cốt lõi của pipeline hiện hành:
  - doc_id, doc_number, doc_type, hierarchy_path, chunk_id, content (chuỗi nhúng
    [META] ... [HIERARCHY] ... [CONTENT] ...)
Và bổ sung các trường phục vụ kiểm định chất lượng (CLAUDE.md §E):
  - char_span        : (start_char, end_char) trong văn bản nguồn (nếu xác định được)
  - structure_method : 'ast' | 'llm_fallback'
  - source_format    : 'detail_html' | 'fallback_pdf' | 'fallback_html' | 'fallback_docx' | 'unknown'
  - quality          : cờ chất lượng từ 7 detector (audit_chunk)
LƯU Ý: Trạng thái hiệu lực pháp lý chỉ truy vấn từ Đồ thị HIN (Neo4j), KHÔNG nhúng cứng
thành thuộc tính tĩnh thay thế đồ thị.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

_AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AUDIT_DIR not in sys.path:
    sys.path.insert(0, _AUDIT_DIR)

from detectors import Chunk, audit_chunk  # noqa: E402


class ChunkQualityFlags(BaseModel):
    """Kết quả kiểm định 7 detector lỗi hệ thống trên một phân đoạn (CLAUDE.md §E)."""
    model_config = ConfigDict(extra="ignore")

    flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Bản đồ 7 cờ lỗi: ocr_corruption, execution_tail, empty_node, "
                    "flattened_table, doctype_mismatch, hierarchy_anomaly, identity_collision.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Thông tin chi tiết kèm theo từng detector.",
    )

    @property
    def is_clean(self) -> bool:
        return not any(self.flags.values())


class LegalChunkSchema(BaseModel):
    """Lược đồ chuẩn hóa cho một phân đoạn pháp lý (Điều hoặc Khoản) trong VietLawBERT."""
    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(..., min_length=1, description="Định danh toàn cục: {doc_id}[_bN]_art_{M}[_cl_{K}]")
    doc_id: str = Field(..., min_length=1, description="Khóa định danh văn bản toàn cục (canonical_doc_id)")
    doc_number: str = Field(..., min_length=1, description="Số ký hiệu văn bản quy phạm pháp luật")
    doc_type: str = Field(default="Khác", description="Loại văn bản theo cấp ban hành / hình thức")
    hierarchy_path: str = Field(..., min_length=1, description="Đường dẫn phân cấp đầy đủ: Chương > Mục > Điều > Khoản")
    content: str = Field(..., min_length=10, description="Chuỗi nhúng [META] ... [HIERARCHY] ... [CONTENT] ...")
    article_id: str = Field(default="", description="Định danh cấp Điều phục vụ liên kết Neo4j HIN")
    article_num: str = Field(default="", description="Số thứ tự Điều (hỗ trợ cả 15a, 28b)")
    macro_label: str = Field(default="CHUNG", max_length=200, description="Nhãn gom cụm vĩ mô cho MRL d=64 (<=200 ký tự)")

    # Các trường bổ sung phục vụ kiểm định chất lượng (CLAUDE.md §E, §H.2)
    char_span: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Vị trí (start_char, end_char) của thân phân đoạn trong văn bản nguồn",
    )
    structure_method: Literal["ast", "llm_fallback"] = Field(
        default="ast",
        description="Phương pháp bóc tách cấu trúc: AST là bộ cắt chính; llm_fallback chỉ cho ngoại lệ",
    )
    source_format: str = Field(
        default="detail_html",
        description="Định dạng nguồn trích xuất (detail_html, fallback_pdf, fallback_html, fallback_docx)",
    )
    quality: ChunkQualityFlags = Field(
        default_factory=ChunkQualityFlags,
        description="Kết quả 7 detector kiểm định chất lượng phân đoạn",
    )

    @field_validator("content")
    @classmethod
    def _validate_embedding_format(cls, v: str) -> str:
        if "[CONTENT]" not in v:
            raise ValueError("Chuỗi nhúng phải chứa thẻ [CONTENT] theo chuẩn Metadata Text Injection.")
        body = v.split("[CONTENT]", 1)[-1].strip()
        if len(body) < 10:
            raise ValueError("Phần thân sau [CONTENT] phải có ít nhất 10 ký tự thực chất (chống node cụt).")
        return v

    @field_validator("char_span")
    @classmethod
    def _validate_char_span(cls, v: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if v is not None:
            start, end = v
            if start < 0 or end < start:
                raise ValueError(f"char_span không hợp lệ: ({start}, {end})")
        return v

    @classmethod
    def from_parser_chunk(
        cls,
        raw_chunk: Dict[str, Any],
        doc_type: str = "Khác",
        source_format: str = "detail_html",
        structure_method: Literal["ast", "llm_fallback"] = "ast",
        source_text: Optional[str] = None,
    ) -> "LegalChunkSchema":
        """Chuyển đổi một dict phân đoạn từ HybridASTParser sang LegalChunkSchema có kiểm định 7 detector."""
        cid = str(raw_chunk.get("chunk_id") or "")
        did = str(raw_chunk.get("doc_id") or "")
        dn = str(raw_chunk.get("doc_number") or "N/A")
        h_path = str(raw_chunk.get("hierarchy_path") or "Toàn văn")
        content = str(raw_chunk.get("content") or raw_chunk.get("text") or "")

        det_chunk = Chunk.from_flat(cid, dn, doc_type, content)
        audit_res = audit_chunk(det_chunk)
        flags = {k: bool(v) for k, v in audit_res.items() if isinstance(v, bool)}
        details = {k[:-5]: v for k, v in audit_res.items() if k.endswith("_info") and v}

        span: Optional[Tuple[int, int]] = raw_chunk.get("char_span")
        if span is None and source_text and det_chunk.body:
            idx = source_text.find(det_chunk.body)
            if idx >= 0:
                span = (idx, idx + len(det_chunk.body))

        return cls(
            chunk_id=cid,
            doc_id=did,
            doc_number=dn,
            doc_type=doc_type,
            hierarchy_path=h_path,
            content=content,
            article_id=str(raw_chunk.get("article_id") or ""),
            article_num=str(raw_chunk.get("article_num") or ""),
            macro_label=str(raw_chunk.get("macro_label") or "CHUNG")[:200],
            char_span=span,
            structure_method=structure_method,
            source_format=source_format,
            quality=ChunkQualityFlags(flags=flags, details=details),
        )
