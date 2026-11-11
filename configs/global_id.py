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

# --- Làm sạch doc_number bẩn (nhiễm tiền tố/hậu tố, "Không số", rò rỉ tiêu đề) ---
_LABEL_PREFIX = re.compile(r"^(?:số|sô|so|văn\s*bản|vb(?:\s*tạm\s*nhập)?|bản\s*nháp|ban\s*nhap|dự\s*thảo|du\s*thao)\b[\s:.\-]*", re.I)
_TYPE_PREFIX  = re.compile(r"^(?:TT|NĐ|ND|QĐ|QD|NQ|CT|PL)\s+(?=\d)", re.I)
_DRAFT_SUFFIX = re.compile(r"[\s.]*(?:\.sua|\.sửa|\.draft)\s*$", re.I)
_NO_NUMBER    = re.compile(r"không\s*số|khong\s*so", re.I)
_DESC_MARKERS = re.compile(r"(kỳ\s*họp|về\s*việc|khóa\s*\d|bất\s*thường|phiên\s*họp)", re.I)


def _preclean(raw: str) -> str:
    """Bóc tiền tố nhãn ('Số:', 'số', 'VB tạm nhập', 'bản nháp'...), hậu tố nháp ('.sua'),
    và tiền tố loại-văn-bản dư ('TT 14/...' -> '14/...'). KHÔNG đổi phần số hiệu thật."""
    s = str(raw or "").strip()
    s = _DRAFT_SUFFIX.sub("", s)
    prev = None
    while prev != s:                       # bóc lặp phòng nhiều tiền tố chồng
        prev = s
        s = _LABEL_PREFIX.sub("", s).strip()
    s = _TYPE_PREFIX.sub("", s).strip()
    return s


def classify_doc_number(raw: str):
    """Phân loại doc_number thô → (status, cleaned).
    - VALID       : số hiệu hợp lệ (đã làm sạch) → dùng canonical.
    - NO_NUMBER   : văn bản THẬT nhưng không có số ('Không số') → KHÔNG gộp chung; caller cấp khóa thay thế.
    - CONTAMINATED: rò rỉ tiêu đề/mô tả kỳ họp hoặc không còn chữ số → cách ly, caller fallback.
    """
    s = str(raw or "").strip()
    if not s or s.upper() in _INVALID:
        return ("CONTAMINATED", "")
    if _NO_NUMBER.search(s):
        return ("NO_NUMBER", "")
    cleaned = _preclean(s)
    if (not re.search(r"\d", cleaned)) or _DESC_MARKERS.search(cleaned) or len(cleaned) > 40:
        return ("CONTAMINATED", cleaned)
    return ("VALID", cleaned)



def normalize_doc_number(doc_number: str) -> str:
    """Chuẩn hóa doc_number để so khớp/định danh: bỏ dấu, Đ→D, hoa, bỏ mọi khoảng trắng."""
    s = _preclean(doc_number)
    s = unicodedata.normalize("NFKD", s).strip()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("Đ", "D").replace("đ", "d")
    s = re.sub(r"\s+", "", s).upper()
    s = s.strip("'\"`.,;:()[]{}-")  # bỏ nháy/ký tự rác đầu-cuối (doc_number nguồn hay dính)
    return s


def canonical_doc_id(doc_number: str, sign_date: str = "", issuer: str = "") -> str:
    """
    doc_id tất định từ doc_number. Trả "" nếu doc_number không hợp lệ (để caller xử lý riêng).
    Dạng: <slug>-<hash8>, chỉ dùng [A-Z0-9-] (KHÔNG có '_' để không phá lược đồ '_art_' của chunk_id).
    Vd: "72/2026/QĐ-UBND" -> "72-2026-QD-UBND-5d41402a"

    sign_date/issuer (TÙY CHỌN): khóa phân biệt cho số hiệu LẶP (vd VBHN đánh lại theo năm —
    "30/VBHN-NHNN" có nhiều bản khác ngày/cơ quan). KHI được cấp, chúng được trộn vào hash để
    hai bản cùng số hiệu KHÔNG gộp về một node. Mặc định rỗng → hành vi CŨ (tương thích ngược:
    hash chỉ theo slug) để không phá khả năng khớp cạnh (relation target chỉ có số hiệu).
    CẢNH BÁO tích hợp: chỉ bật disambiguator khi ĐỒNG THỜI có cơ chế resolve cạnh về đúng bản
    (xem detect_collisions + ghi chú bên dưới), nếu không cạnh trỏ số hiệu trần sẽ lệch node.
    """
    status, cleaned = classify_doc_number(doc_number)
    if status != "VALID":
        return ""   # "Không số" & rác tiêu đề KHÔNG gộp chung — caller dùng khóa thay thế/crawler_id
    norm = normalize_doc_number(cleaned)
    if norm in _INVALID:
        return ""
    slug = re.sub(r"[^A-Z0-9]+", "-", norm).strip("-")
    if not slug:
        return ""
    key = slug
    disamb = "|".join(str(x).strip() for x in (sign_date, issuer) if x and str(x).strip() not in _INVALID)
    if disamb:
        key = f"{slug}|{disamb}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]  # hash KEY → quote/rác không tách id
    return f"{slug}-{h}"


def detect_collisions(records):
    """Phát hiện số hiệu LẶP mang nội dung/ngày/cơ quan khác nhau (vd VBHN theo năm).
    records: iterable dict có 'doc_number' và (tùy) 'sign_date'/'effective_date'/'co_quan'.
    Trả {slug: [bản ghi...]} cho các slug xuất hiện > 1 biến thể phân biệt → caller quyết định
    gắn thuộc tính version hoặc resolve cạnh về bản mới nhất."""
    from collections import defaultdict
    buckets = defaultdict(set)
    rows = defaultdict(list)
    for r in records:
        cid = canonical_doc_id(r.get("doc_number", ""))
        if not cid:
            continue
        slug = cid.rsplit("-", 1)[0]
        variant = (str(r.get("sign_date") or r.get("effective_date") or "").strip(),
                   str(r.get("co_quan") or r.get("issuer") or "").strip())
        buckets[slug].add(variant)
        rows[slug].append(r)
    return {slug: rows[slug] for slug, vs in buckets.items() if len(vs) > 1}


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
         "Quyết định/Chỉ thị Thủ tướng":6,"Thông tư":7,"Văn bản hợp nhất":7,
         "Văn bản địa phương":8,"Khác":9}

# Tiêu đề (HÌNH THỨC) → bucket taxonomy, chỉ cho loại KHÔNG mơ hồ; khớp ở ĐẦU tiêu đề để
# tránh bắt nhầm 'Luật' trong 'Nghị định quy định chi tiết Luật ...'.
_RE_TITLE_TYPE = re.compile(
    r"^\s*(hiến\s*pháp|bộ\s*luật|luật|pháp\s*lệnh|văn\s*bản\s*hợp\s*nhất|nghị\s*định|thông\s*tư\s*liên\s*tịch|thông\s*tư)\b",
    re.IGNORECASE)
_TITLE_BUCKET = {"hiến pháp":"Hiến pháp","bộ luật":"Luật/Nghị quyết Quốc hội","luật":"Luật/Nghị quyết Quốc hội",
                 "pháp lệnh":"Pháp lệnh/Nghị quyết UBTVQH","văn bản hợp nhất":"Văn bản hợp nhất",
                 "nghị định":"Nghị định/Nghị quyết Chính phủ","thông tư liên tịch":"Thông tư","thông tư":"Thông tư"}


def _type_from_title(title: str):
    m = _RE_TITLE_TYPE.match(unicodedata.normalize("NFKC", str(title or "")).strip())
    if not m:
        return None
    return _TITLE_BUCKET[re.sub(r"\s+", " ", m.group(1).lower())]


def _classify_symbol(norm: str) -> str:
    """Phân loại cấp ban hành từ TOÀN BỘ ký hiệu đã chuẩn hóa (KHÔNG chỉ đoạn sau '/')."""
    s = (norm or "").upper()
    if not s:
        return "Khác"
    if "UBTVQH" in s:                                           return "Pháp lệnh/Nghị quyết UBTVQH"
    if re.search(r"QH\d+", s):                                  return "Luật/Nghị quyết Quốc hội"
    if "UBND" in s or "HDND" in s or "-UB-" in s or re.search(r"[-/]UB$", s):
                                                                return "Văn bản địa phương"
    if "TTG" in s:                                              return "Quyết định/Chỉ thị Thủ tướng"
    if "ND-CP" in s or "NQ-CP" in s or "HDBT" in s:             return "Nghị định/Nghị quyết Chính phủ"
    if "TTLT" in s or re.search(r"(^|[-/])TT([-/]|$)", s):      return "Thông tư"
    if re.search(r"^\d+-[A-ZĐ]{2,5}/[A-ZĐ]", s):                return "Thông tư"   # thông tư bộ cũ: 60-TC/CDKT, 05-BXD/KTQH
    if "CTN" in s or s.startswith("L-") or "LCT" in s:          return "Lệnh/Quyết định Chủ tịch nước"
    return "Khác"


def doc_type_and_rank(doc_number: str, title: str = ""):
    """Trả (doc_type, legal_rank). rank nhỏ = hiệu lực cao.
    Thứ tự suy luận: (1) VBHN ưu tiên tuyệt đối; (2) HÌNH THỨC từ tiêu đề cho loại form-xác định
    (Hiến pháp/Luật/Pháp lệnh/Nghị định/Thông tư) — sửa các số hiệu cũ 58-L/CTN, 60-TC/...;
    (3) còn lại suy theo CẤP BAN HÀNH từ ký hiệu (địa phương/Thủ tướng/Chủ tịch nước/Chính phủ)."""
    norm = normalize_doc_number(doc_number)
    if "VBHN" in norm.upper() or "hợp nhất" in (title or "").lower():
        emb = _classify_symbol(norm.upper().replace("VBHN-", "").replace("VBHN", ""))
        return ("Văn bản hợp nhất", _RANK[emb] if emb != "Khác" else _RANK["Văn bản hợp nhất"])
    bt = _type_from_title(title)
    if bt:
        return (bt, _RANK[bt])
    base = _classify_symbol(norm)
    return (base, _RANK[base])
