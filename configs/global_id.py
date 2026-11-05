"""
configs/global_id.py — KHÓA ĐỊNH DANH TOÀN CỤC (global ID) cho VietLawBERT.

Vấn đề gốc (phát hiện 04/11): doc_id lấy từ crawler (item.doc_id/item_id) KHÔNG ổn định
→ mỗi lần crawl/ingest sinh id khác cho cùng một văn bản → MERGE{doc_id} tạo node trùng (62×),
và doc_id ở ES ≠ Neo4j (lệch 78%). doc_number thì nhất quán 100% xuyên kho.

Giải pháp: doc_id = HÀM TẤT ĐỊNH của doc_number (khóa nghiệp vụ). Idempotent qua mọi lần ingest,
đồng nhất MongoDB/Neo4j/Qdrant/ES. chunk_id giữ nguyên lược đồ "{doc_id}_art_N[_cl_M...]" của parser
(chỉ cần đặt doc_id canonical vào metadata TRƯỚC khi parse là chunk_id tự canonical theo).
"""
from __future__ import annotations
import re
import hashlib
import unicodedata

_INVALID = {"", "N/A", "NA", "UNKNOWN", "NONE", "NULL"}


def normalize_doc_number(doc_number: str) -> str:
    """Chuẩn hóa doc_number để so khớp/định danh: bỏ dấu, Đ→D, hoa, bỏ mọi khoảng trắng."""
    s = unicodedata.normalize("NFKD", str(doc_number or "")).strip()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("Đ", "D").replace("đ", "d")
    s = re.sub(r"\s+", "", s).upper()
    return s


def canonical_doc_id(doc_number: str) -> str:
    """
    doc_id tất định từ doc_number. Trả "" nếu doc_number không hợp lệ (để caller xử lý riêng).
    Dạng: <slug>-<hash8>, chỉ dùng [A-Z0-9-] (KHÔNG có '_' để không phá lược đồ '_art_' của chunk_id).
    Vd: "72/2026/QĐ-UBND" -> "72-2026-QD-UBND-5d41402a"
    """
    norm = normalize_doc_number(doc_number)
    if norm in _INVALID:
        return ""
    slug = re.sub(r"[^A-Z0-9]+", "-", norm).strip("-")
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}" if slug else h


def resolve_target_id(raw_target: str, crawlerid_to_docnum: dict) -> str:
    """
    Canonical hóa đầu ĐÍCH của một quan hệ. raw_target có thể là crawler-id hoặc doc_number.
    Ưu tiên tra map crawler_id→doc_number; nếu không có, thử coi raw_target chính là doc_number.
    Trả "" nếu không giải được (đích ngoài corpus) để caller quyết định giữ raw hay bỏ.
    """
    raw = str(raw_target or "").strip()
    if not raw:
        return ""
    dn = crawlerid_to_docnum.get(raw)
    if dn:
        return canonical_doc_id(dn)
    # raw_target có thể vốn đã là doc_number
    cid = canonical_doc_id(raw)
    return cid
