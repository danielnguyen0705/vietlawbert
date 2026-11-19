# -*- coding: utf-8 -*-
"""
preprocess/ocr_clean.py
=======================
LÀM SẠCH HẬU-OCR (xác định, KHÔNG LLM, không viết lại nội dung — tuân thủ CLAUDE.md §E).
Chỉ bỏ "râu ria" và nối dòng gãy; TUYỆT ĐỐI không sửa chính tả / không đổi số hiệu, con số.

Dựa trên quan sát OCR thực (Nghị định/Thông tư scan VBPL):
  1) Dòng chữ ký số / metadata phát hành chèn ở đầu-cuối trang
     ("Cơ quan phát hành:", "Ký bởi:", "Thời gian ký:", "Email:", "Chức vụ:"…).
  2) Dòng trang trí / rác ký tự ("«", "———", "sÀ + Lễ ` Ká ` Ã - vá ~ -").
  3) Dòng gãy mềm: một đoạn văn bị OCR cắt thành nhiều dòng → nối lại thành đoạn,
     GIỮ tiêu đề (Phần/Chương/Mục/Điều) và đầu mục (khoản "1.", điểm "a)") trên dòng riêng
     để AST parser vẫn tách cấu trúc được.

KHÔNG xử lý được (giới hạn trung thực): lỗi OCR cấp ký tự (vd "Tùi"→"Tài", "BVN"→"EVN",
"chỉ phí"→"chi phí"). Sửa tự động = nguy cơ đổi nghĩa/số hiệu → vi phạm §E. Thay vào đó ĐO và
gắn cờ needs_human_review khi mật độ nghi ngờ cao, đẩy sang soi tay (Lớp 2 §E).
"""
from __future__ import annotations
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quality", "corpus_audit"))
from corpus_gate import compute_vi_ratio  # noqa: E402

# 1) Chữ ký số / metadata phát hành (bắt cả khi lẫn ký tự rác đầu dòng như "kau Cơ quan…")
_SIG_RE = re.compile(
    r"(cơ\s*quan\s*phát\s*hành|ký\s*bởi|thời\s*gian\s*ký|chức\s*vụ|chữ\s*ký\s*số"
    r"|digitally\s*signed|email\s*:|signature\s*valid|người\s*ký\s*:)",
    re.IGNORECASE,
)
# 3) đầu mục / tiêu đề cấu trúc pháp lý
# dung sai OCR cho nguyên âm ê/ề/e trong từ khóa cấu trúc (chỉ để NHẬN DẠNG, không sửa nội dung)
_HEADING_RE = re.compile(r"^(ph[âầa]n|ch[ươu][ơo]?ng|m[ụu]c|ti[êềe]u\s*m[ụu]c)\s+[IVXLCDM0-9]", re.IGNORECASE)
_DIEU_RE = re.compile(r"^đi[êềe]u\s+\d+", re.IGNORECASE)
_LIST_RE = re.compile(r"^(\d{1,3}\.\s|\d{1,3}\.\d|[a-zđ]\)\s|[-•]\s)")
# tiêu đề IN HOA (NGHỊ ĐỊNH, QUYẾT ĐỊNH, QUY ĐỊNH CHUNG) — 2+ từ, không có chữ thường
_CAPS_RE = re.compile(r"^[^a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]*$")


def _is_signature(line: str) -> bool:
    return bool(_SIG_RE.search(line))


def _is_decorative_or_garbage(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Trang trí: chỉ gồm ký tự không phải chữ/số ("«", "———", "- -")
    if re.fullmatch(r"[\W_]+", s):
        return True
    # Rác OCR (stamp/vệt bẩn): dòng ngắn, ≥4 token nhưng hầu như không có TỪ thực (≥3 chữ cái).
    # Không dựa vào vi_ratio vì rác có dấu ngẫu nhiên (vd "sÀ + Lễ ` Ká ` Ã - vá ~ -") vẫn vi cao.
    toks = s.split()
    if len(s) <= 45 and len(toks) >= 4:
        real_words = sum(1 for t in toks if sum(c.isalpha() for c in t) >= 3)
        if real_words <= 1:
            return True
    return False


def _is_heading(line: str) -> bool:
    s = line.strip()
    if _HEADING_RE.match(s) or _DIEU_RE.match(s):
        return True
    # Tiêu đề IN HOA ngắn (<= 12 từ) và có chữ cái
    if len(s.split()) <= 12 and any(c.isalpha() for c in s) and _CAPS_RE.match(s):
        return True
    return False


def _ends_paragraph(line: str) -> bool:
    return line.rstrip().endswith((".", ";", ":", "!", "?", "…"))


def clean_ocr_text(text: str) -> Tuple[str, Dict]:
    """Trả (clean_text, stats). stats gồm số dòng bỏ, cờ needs_human_review + lý do."""
    raw_lines = [ln.rstrip() for ln in (text or "").splitlines()]
    kept: List[str] = []
    dropped_sig = dropped_junk = 0
    for ln in raw_lines:
        if not ln.strip():
            continue
        if _is_signature(ln):
            dropped_sig += 1
            continue
        if _is_decorative_or_garbage(ln):
            dropped_junk += 1
            continue
        kept.append(ln.strip())

    # Nối dòng gãy mềm; giữ tiêu đề + đầu mục trên dòng riêng
    out: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            out.append(re.sub(r"\s{2,}", " ", " ".join(buf)).strip())
            buf.clear()

    for ln in kept:
        if _is_heading(ln):
            flush()
            out.append(ln)
        elif _LIST_RE.match(ln):
            flush()
            buf.append(ln)
        else:
            if buf and buf[-1].endswith("-"):   # nối từ bị gạch nối cuối dòng (hiếm)
                buf[-1] = buf[-1][:-1] + ln
            else:
                buf.append(ln)
            if _ends_paragraph(ln) and not out:
                pass  # vẫn giữ trong buf, flush tự nhiên ở heading/list kế
    flush()

    clean = "\n".join(out).strip()
    # Đo nghi ngờ lỗi ký tự để gắn cờ soi tay (không sửa): token có chữ+số lẫn lộn, hoặc vi tụt
    vr = compute_vi_ratio(clean) or 0.0
    mixed = len(re.findall(r"\b[A-Za-zÀ-ỹ]*\d[A-Za-zÀ-ỹ]+\b", clean))
    words = max(1, len(clean.split()))
    suspicious_ratio = round(mixed / words, 4)
    needs_review = vr < 0.18 or suspicious_ratio > 0.02
    stats = {
        "lines_in": len(raw_lines),
        "lines_out": len(out),
        "dropped_signature": dropped_sig,
        "dropped_junk": dropped_junk,
        "vi_ratio": vr,
        "suspicious_token_ratio": suspicious_ratio,
        "needs_human_review": needs_review,
    }
    return clean, stats
