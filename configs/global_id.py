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
    s = s.strip("'\"`.,;:()[]{}-")  # bỏ nháy/ký tự rác đầu-cuối (doc_number nguồn hay dính)
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
    if not slug:
        return ""
    h = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]  # hash SLUG → quote/rác không tách id
    return f"{slug}-{h}"


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


# Bậc hiệu lực pháp lý theo Luật Ban hành VBQPPL 2025 (1=cao nhất). Suy tất định từ ký hiệu doc_number.
_RANK = {"Hiến pháp":1,"Luật/Nghị quyết Quốc hội":2,"Pháp lệnh/Nghị quyết UBTVQH":3,
         "Lệnh/Quyết định Chủ tịch nước":4,"Nghị định/Nghị quyết Chính phủ":5,
         "Quyết định/Chỉ thị Thủ tướng":6,"Thông tư":7,"Văn bản địa phương":8,"Khác":9}

def doc_type_and_rank(doc_number: str):
    """Trả (doc_type, legal_rank) từ ký hiệu doc_number. rank nhỏ = hiệu lực cao."""
    norm = normalize_doc_number(doc_number)
    sym = norm.split("/")[-1] if "/" in norm else norm   # vd 80/2015/QH13 -> QH13 ; 14/2022/TT-NHNN -> TT-NHNN
    def R(t): return (t, _RANK[t])
    if not sym or norm in _INVALID: return R("Khác")
    if "UBTVQH" in sym:                      return R("Pháp lệnh/Nghị quyết UBTVQH")
    if sym.startswith("QH") or re.search(r"QH\d+$", sym): return R("Luật/Nghị quyết Quốc hội")
    if "HDND" in sym or sym.endswith("UBND") or "QD-UBND" in sym or "NQ-HDND" in sym:
        return R("Văn bản địa phương")
    if "TTLT" in sym or sym.startswith("TT"): return R("Thông tư")
    if "QD-TTG" in sym or "CT-TTG" in sym or sym.endswith("TTG"): return R("Quyết định/Chỉ thị Thủ tướng")
    if "ND-CP" in sym or "NQ-CP" in sym:      return R("Nghị định/Nghị quyết Chính phủ")
    if "CTN" in sym or sym.startswith("L-") or sym.startswith("LCT"): return R("Lệnh/Quyết định Chủ tịch nước")
    return R("Khác")
