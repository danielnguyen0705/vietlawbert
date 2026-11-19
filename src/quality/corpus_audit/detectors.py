# -*- coding: utf-8 -*-
"""
corpus_audit/detectors.py
==========================
Bộ dò (detector) 7 lỗi hệ thống của ngữ liệu VietLawBERT, viết dưới dạng HÀM THUẦN
(pure function) để vừa unit-test được ở local, vừa quét được toàn corpus trên cloud.

Mỗi detector nhận một "bản ghi chunk" đã tách sẵn 3 trường:
    meta       : dòng [META] ...  (tiêu đề, cơ quan, hiệu lực, tình trạng)
    hierarchy  : dòng [HIERARCHY] ... (đường dẫn cấp bậc)
    body       : phần [CONTENT] ... (nội dung thực)
và trả về (flag: bool, score_or_reason).

Không phụ thuộc hạ tầng — chạy đứng một mình. Trong repo, driver sẽ bọc thêm
get_subsystem_logger("evaluation","corpus_audit") theo quy ước dự án.

Căn cứ các ca đã xác minh tay trên mẫu Cochran 382 (xem tests/test_detectors.py).
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Tiện ích tách [META]/[HIERARCHY]/[CONTENT] từ content phẳng
# ---------------------------------------------------------------------------
_RE_META = re.compile(r"\[META\](.*?)(?=\[HIERARCHY\]|\[CONTENT\]|$)", re.S)
_RE_HIER = re.compile(r"\[HIERARCHY\](.*?)(?=\[CONTENT\]|$)", re.S)
_RE_BODY = re.compile(r"\[CONTENT\](.*)$", re.S)


@dataclass
class Chunk:
    chunk_id: str
    doc_number: str
    doc_type: str
    meta: str
    hierarchy: str
    body: str
    raw: str = ""

    @classmethod
    def from_flat(cls, chunk_id: str, doc_number: str, doc_type: str, content: str) -> "Chunk":
        meta = (_RE_META.search(content) or _blank()).group(1).strip() if "[META]" in content else ""
        hier = (_RE_HIER.search(content) or _blank()).group(1).strip() if "[HIERARCHY]" in content else ""
        body = (_RE_BODY.search(content) or _blank()).group(1).strip() if "[CONTENT]" in content else content.strip()
        return cls(chunk_id, doc_number, doc_type, meta, hier, body, raw=content)


class _blank:
    def group(self, _):  # cho phép .group(1) trả "" khi không match
        return ""


# ---------------------------------------------------------------------------
# Bảng ký tự/nguyên âm tiếng Việt (để nhận token "không nguyên âm" = nghi OCR vỡ)
# ---------------------------------------------------------------------------
_VI_VOWELS = set("aeiouyAEIOUY" "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                 "ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
                 "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆ"
                 "ÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ")
_GARBAGE_SYMBOLS = set("€£¥©®•‡†¢¤§¶<>|\\~^`")
_RE_TOKEN = re.compile(r"\S+")
_RE_DIGIT_INTERIOR = re.compile(r"[A-Za-zÀ-ỹ]\d[A-Za-zÀ-ỹ]")   # chữ-số-chữ: O6i, Ci6y
_RE_LEGAL_REF = re.compile(r"\d+\s*[/\-]")                      # 15/2020, 60-TC → KHÔNG tính là rác
_RE_ROMAN = re.compile(r"^[IVXLCDM]+$", re.I)


def _strip_punct(tok: str) -> str:
    return tok.strip(".,;:()[]{}\"'…–—-/")


def _is_broken_token(tok: str) -> bool:
    """Một token bị coi là 'vỡ' (dấu hiệu OCR hỏng) nếu:
    - chứa ký tự rác (€, <, >, |...), HOẶC
    - có mẫu chữ-số-chữ ở giữa (O6i, Ci6y), HOẶC
    - là token thường (không viết HOA toàn bộ, không số La Mã) dài >=3 mà KHÔNG có
      một nguyên âm tiếng Việt nào (sd, phdp, lflp).
    Loại trừ: số, tham chiếu luật (15/2020), viết tắt HOA (UBND, BHXH, TT, NĐ, CP).
    """
    if any(c in _GARBAGE_SYMBOLS for c in tok):
        return True
    core = _strip_punct(tok)
    if not core:
        return False
    # tham chiếu luật / số hiệu: bỏ qua
    if _RE_LEGAL_REF.search(core) or core.replace(",", "").replace(".", "").isdigit():
        return False
    # chữ-số-chữ ở giữa một token chữ
    if _RE_DIGIT_INTERIOR.search(core):
        return True
    alpha = [c for c in core if c.isalpha()]
    if len(alpha) >= 3:
        if core.isupper() or _RE_ROMAN.match(core):   # viết tắt / số La Mã hợp lệ
            return False
        if not any(c in _VI_VOWELS for c in core):     # không có nguyên âm nào
            return True
    return False


def detect_ocr_corruption(body: str, threshold: float = 0.12) -> tuple[bool, float]:
    """Lỗi #1 — OCR/encoding vỡ. Trả (flag, tỉ lệ token vỡ)."""
    toks = _RE_TOKEN.findall(body or "")
    toks = [t for t in toks if any(c.isalpha() for c in t)]
    if not toks:
        return False, 0.0
    broken = sum(1 for t in toks if _is_broken_token(t))
    score = broken / len(toks)
    return score >= threshold, round(score, 4)


# ---------------------------------------------------------------------------
# Lỗi #2 — giữ đuôi chữ ký / điều khoản thi hành (không có nội dung quy phạm)
# ---------------------------------------------------------------------------
_RE_EXEC = re.compile(
    r"(chịu trách nhiệm thi hành"
    r"|\bnơi nhận\s*:"
    r"|\bT/M\b|\bTM\.|\bK/T\b|\bKT\."
    r"|\(đã ký\)|\(Đã ký\)"
    r"|chủ tịch|phó chủ tịch|thứ trưởng|bộ trưởng"
    r"|thừa lệnh|quyền hạn và trách nhiệm)",
    re.I,
)
_RE_SIGN_CLOSE = re.compile(r"\./\.")   # kết "./.", dấu đóng quyết định


def detect_execution_tail(body: str) -> tuple[bool, str]:
    """Lỗi #2. Cờ bật khi chunk là khối thi hành/chữ ký thay vì nội dung quy phạm.
    Luật: có 'chịu trách nhiệm thi hành' HOẶC (có khối ký "./." + chức danh ký)."""
    b = body or ""
    has_resp = "chịu trách nhiệm thi hành" in b.lower()
    has_close = bool(_RE_SIGN_CLOSE.search(b))
    has_title = bool(re.search(r"(phó chủ tịch|chủ tịch|thứ trưởng|bộ trưởng|TM\.|T/M|KT\.|K/T)", b, re.I))
    if has_resp:
        return True, "co_cum_'chiu trach nhiem thi hanh'"
    if has_close and has_title:
        return True, "khoi_ky_'./.'_+_chuc_danh"
    return False, ""


# ---------------------------------------------------------------------------
# Lỗi #3 — node cụt: body chỉ còn tiêu đề (Chương/Điều), không có thân
# ---------------------------------------------------------------------------
_RE_HEADING_LINE = re.compile(
    r"^\s*(PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục|Tiểu mục|ĐIỀU|Điều)\b.*$", re.M)
_RE_ARTICLE_HEAD = re.compile(r"^\s*Điều\s+\d+\s*[\.:]", re.M)


_RE_SECTION_LINE = re.compile(r"^\s*(PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục|Tiểu mục)\b[^\n]*$", re.M)
_RE_ARTICLE_LABEL = re.compile(r"\bĐiều\s+\d+\s*\.")


def detect_empty_node(body: str, min_body_chars: int = 50) -> tuple[bool, int]:
    """Lỗi #3 — node cụt. Chỉ bỏ DÒNG tiêu đề mục (Phần/Chương/Mục) và NHÃN 'Điều N.'
    (giữ lại tên điều + thân); phần còn lại ngắn hơn ngưỡng → cụt.
    LƯU Ý diễn giải: chunk có đuôi _root/_preamble có thể là node CHA cấu trúc (thân nằm
    ở chunk khoản con) — driver tách riêng loại này, chỉ node LÁ rỗng mới là mất nội dung."""
    b = (body or "").strip()
    b = _RE_SECTION_LINE.sub("", b)          # bỏ dòng tiêu đề mục/chương/phần
    b = _RE_ARTICLE_LABEL.sub("", b)          # bỏ nhãn 'Điều 5.' nhưng GIỮ phần sau
    substantive = re.sub(r"\s+", " ", b).strip(" .-")
    return len(substantive) < min_body_chars, len(substantive)


def is_structural_node(chunk_id: str) -> bool:
    """Chunk cha cấu trúc (node rỗng ở đây có thể bình thường nếu có chunk con)."""
    cid = (chunk_id or "").lower()
    return cid.endswith("_root") or "_preamble" in cid or cid.endswith("_root")


# ---------------------------------------------------------------------------
# Lỗi #4 — bảng bị dẹt thành nhiều dòng token rời (mất cấu trúc bảng)
# ---------------------------------------------------------------------------
def detect_flattened_table(body: str, run_threshold: int = 6) -> tuple[bool, str]:
    """Lỗi #4. Nhận diện bảng bị dẹt: có header 'STT' đứng riêng một dòng, hoặc
    một chuỗi >= run_threshold dòng ngắn (<=3 token) liên tiếp (số/nhãn lìa nhau)."""
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    if not lines:
        return False, ""
    if any(ln.upper() == "STT" for ln in lines):
        return True, "co_dong_'STT'_rieng"
    run = best = 0
    for ln in lines:
        if len(ln.split()) <= 3:
            run += 1
            best = max(best, run)
        else:
            run = 0
    if best >= run_threshold:
        return True, f"chuoi_{best}_dong_ngan_lien_tiep"
    return False, ""


# ---------------------------------------------------------------------------
# Lỗi #5 — doc_type sai: tiêu đề META khai loại rõ ràng nhưng gán "Khác"/lệch
# ---------------------------------------------------------------------------
_TYPE_KEYWORDS = [
    ("Văn bản hợp nhất", "Văn bản hợp nhất"),
    ("Thông tư liên tịch", "Thông tư"),
    ("Thông tư", "Thông tư"),
    ("Nghị định", "Nghị định"),
    ("Nghị quyết", "Nghị quyết"),
    ("Quyết định", "Quyết định"),
    ("Pháp lệnh", "Pháp lệnh"),
    ("Chỉ thị", "Chỉ thị"),
    ("Luật", "Luật"),
    ("Hiến pháp", "Hiến pháp"),
]
_RE_META_TITLE = re.compile(r"Tiêu đề\s*:\s*(.*?)\s*(\||$)")


def _declared_type(meta: str) -> Optional[str]:
    m = _RE_META_TITLE.search(meta or "")
    title = m.group(1) if m else (meta or "")
    for kw, canon in _TYPE_KEYWORDS:
        if kw.lower() in title.lower():
            return canon
    return None


_RE_LOCAL = re.compile(r"UBND|HĐND|HDND", re.I)


def detect_doctype_mismatch(meta: str, doc_type: str, doc_number: str = "") -> tuple[bool, str]:
    """Lỗi #5 — doc_type sai. LƯU Ý: taxonomy dự án phân theo CẤP BAN HÀNH, không theo
    hình thức; nên 'Văn bản địa phương' cho QĐ-UBND/NQ-HĐND là ĐÚNG. Chỉ bắt các ca
    không cãi được: loại trung ương chuẩn (Thông tư/Nghị định/Luật/Pháp lệnh/Nghị quyết)
    hoặc văn bản địa phương mà bị dồn vào 'Khác'/để rỗng.
    VBHN không có bucket riêng → không phạt VBHN→Khác (tránh dương tính giả)."""
    declared = _declared_type(meta)
    if declared is None:
        return False, ""
    dt = (doc_type or "").strip()
    # địa phương: đúng nếu nằm bucket địa phương; chỉ sai khi rơi vào Khác/rỗng
    if _RE_LOCAL.search(doc_number or ""):
        if "địa phương" in dt.lower():
            return False, ""
        if dt in ("", "Khác"):
            return True, f"dia_phuong_nhung_doc_type='{dt or 'rong'}'"
        return False, ""
    # VBHN: taxonomy không có bucket riêng → bỏ qua
    if declared == "Văn bản hợp nhất":
        return False, ""
    # loại trung ương chuẩn mà rơi vào 'Khác'/rỗng = lỗi rõ
    if dt in ("", "Khác"):
        return True, f"tieu_de='{declared}'_nhung_doc_type='{dt or 'rong'}'"
    # xếp sai sang bucket trung ương khác → borderline, không phạt ở bản thận trọng này
    return False, ""


# ---------------------------------------------------------------------------
# Lỗi #6 — hierarchy tùy hứng: cấp không thuộc từ vựng cấu trúc, hoặc Điều lệch
# ---------------------------------------------------------------------------
_STRUCT_KW = ("Phần", "Chương", "Mục", "Tiểu mục", "Điều", "Khoản", "Điểm")
_RE_HIER_ARTICLE = re.compile(r"Điều\s+(\d+)")
_RE_BODY_ARTICLE = re.compile(r"Điều\s+(\d+)")


def detect_hierarchy_anomaly(hierarchy: str, body: str, chunk_id: str = "") -> tuple[bool, str]:
    """Lỗi #6 — hierarchy tùy hứng. Hai dạng bất thường:
    (a) một cấp trong đường dẫn KHÔNG bắt đầu bằng từ khóa cấu trúc, trên chunk KHÔNG phải
        preamble (chunk preamble mang nhãn 'Căn cứ pháp lý' là nhất quán, không phạt);
    (b) số Điều ở hierarchy khác số Điều xuất hiện ở body."""
    h = (hierarchy or "").strip()
    if not h:
        return False, ""
    is_preamble = "preamble" in (chunk_id or "").lower()
    levels = [lv.strip() for lv in h.split(">") if lv.strip()]
    for lv in levels:
        if not any(lv.startswith(kw) for kw in _STRUCT_KW):
            if is_preamble and lv.lower().startswith("căn cứ"):
                continue  # nhãn preamble hợp lệ, bỏ qua
            return True, f"cap_phi_cau_truc='{lv}'"
    mh = _RE_HIER_ARTICLE.search(h)
    mb = _RE_BODY_ARTICLE.search(body or "")
    if mh and mb and mh.group(1) != mb.group(1):
        return True, f"Dieu_hierarchy={mh.group(1)}_!=_body={mb.group(1)}"
    return False, ""


# ---------------------------------------------------------------------------
# Lỗi #7 — rủi ro trùng khóa định danh (số hiệu lặp qua năm/cơ quan)
# ---------------------------------------------------------------------------
_RE_VBHN = re.compile(r"VBHN", re.I)


def detect_identity_collision_risk(doc_number: str) -> tuple[bool, str]:
    """Lỗi #5 (nội bộ #7) — SÀNG LỌC. Số hiệu kiểu VBHN (và các dạng đánh lại theo năm)
    có nguy cơ trùng → canonical_doc_id(số hiệu) tất định sẽ gộp nhầm.
    Đây chỉ là cờ sàng lọc; đếm trùng THẬT cần quét raw-crawl (ngày ký + cơ quan)."""
    dn = doc_number or ""
    if _RE_VBHN.search(dn):
        return True, "so_hieu_VBHN_co_the_lap_theo_nam"
    return False, ""


# ---------------------------------------------------------------------------
# Gộp: chạy toàn bộ detector trên 1 chunk
# ---------------------------------------------------------------------------
DETECTORS = [
    ("ocr_corruption",      lambda c: detect_ocr_corruption(c.body)),
    ("execution_tail",      lambda c: detect_execution_tail(c.body)),
    ("empty_node",          lambda c: detect_empty_node(c.body)),
    ("flattened_table",     lambda c: detect_flattened_table(c.body)),
    ("doctype_mismatch",    lambda c: detect_doctype_mismatch(c.meta, c.doc_type, c.doc_number)),
    ("hierarchy_anomaly",   lambda c: detect_hierarchy_anomaly(c.hierarchy, c.body, c.chunk_id)),
    ("identity_collision",  lambda c: detect_identity_collision_risk(c.doc_number)),
]


def audit_chunk(c: Chunk) -> dict:
    out = {"chunk_id": c.chunk_id, "doc_number": c.doc_number}
    for name, fn in DETECTORS:
        flag, info = fn(c)
        out[name] = bool(flag)
        out[name + "_info"] = info
    return out
