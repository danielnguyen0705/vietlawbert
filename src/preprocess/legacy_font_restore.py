# -*- coding: utf-8 -*-
"""
preprocess/legacy_font_restore.py
=================================
PHỤC HỒI VĂN BẢN VỠ FONT LEGACY (TCVN3/VNI) cho nhóm fallback_pdf dưới ngưỡng tiếng
Việt — nhánh mà corpus_gate đã cách ly (reject_reason=fallback_pdf_low_vi). CHẨN ĐOÁN
THỰC: đây KHÔNG phải PDF scan (chữ vẫn đủ, chỉ sai glyph dấu, vd "HOIDdNG NHAN DAN"
= "HỘI ĐỒNG NHÂN DÂN"), và nguồn ảnh PDF không còn → dùng phục hồi text→text qua
hhtechapi (configs CONTEXTUALIZER_*, Qwen2.5-14B), RẺ hơn Vision OCR nhiều.

TUÂN THỦ CLAUDE.md §E (LLM KHÔNG được viết lại/diễn giải văn bản pháp luật). Mọi kết
quả qua HÀNG RÀO CHỐNG BỊA:
  - giữ nguyên mọi cụm số (số hiệu, con số) có trong bản gốc,
  - độ dài trong khoảng [0.7, 1.3]× bản gốc (chặn phình nội dung = bịa),
  - tỉ lệ tiếng Việt sau phục hồi ≥ 0.05 VÀ cao hơn bản gốc.
Thất bại bất kỳ chốt nào → GIỮ bản gốc, đánh dấu unresolved + cần soi tay (Lớp 2).

CHỈ GHI MANIFEST (reports/*.jsonl) để ingest hợp nhất; KHÔNG ghi MongoDB/Qdrant/Neo4j/ES.
Tái sử dụng: corpus_gate.compute_vi_ratio, crawl_audit.safe_read_jsonl, configs.config.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from datetime import date
from typing import Callable, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "quality", "corpus_audit"))
from corpus_gate import compute_vi_ratio  # noqa: E402 (nhẹ, không kéo torch)

_VN_GATE = 0.05
_LEN_LO, _LEN_HI = 0.7, 1.3
_RE_DIGIT_RUN = re.compile(r"\d{2,}")

_SYSTEM = (
    "Bạn là bộ SỬA MÃ HÓA font tiếng Việt legacy (TCVN3/VNI) về Unicode chuẩn. "
    "Nhiệm vụ DUY NHẤT: phục hồi đúng dấu thanh/nguyên âm tiếng Việt cho văn bản bị vỡ font. "
    "TUYỆT ĐỐI KHÔNG thêm, bớt, tóm tắt, diễn giải hay định dạng lại; KHÔNG đổi số hiệu, con số, "
    "ngày tháng, thứ tự dòng. Nếu một đoạn không chắc, giữ nguyên. Trả về DUY NHẤT văn bản đã sửa."
)


def _digit_runs(text: str) -> set[str]:
    return set(_RE_DIGIT_RUN.findall(text or ""))


def restore_guard(original: str, restored: str) -> tuple[bool, str]:
    """HÀNG RÀO §E. Trả (accepted, reason). Chấp nhận chỉ khi phục hồi an toàn, không bịa."""
    if not restored or not restored.strip():
        return False, "empty_output"
    lo = _digit_runs(original)
    if not _digit_runs(restored).issuperset(lo):
        return False, "digits_lost"            # mất số hiệu/con số → nghi bịa/cắt
    ratio = len(restored) / max(1, len(original))
    if not (_LEN_LO <= ratio <= _LEN_HI):
        return False, f"length_ratio_{ratio:.2f}"
    vi_new = compute_vi_ratio(restored) or 0.0
    vi_old = compute_vi_ratio(original) or 0.0
    if vi_new < _VN_GATE or vi_new <= vi_old:
        return False, f"vi_not_improved_{vi_old:.3f}->{vi_new:.3f}"
    return True, "ok"


def hhtech_complete(text: str, max_chars: int = 12000) -> str:
    """Gọi hhtechapi (OpenAI-compatible, CONTEXTUALIZER_*). Lazy import để test không cần config/torng."""
    import urllib.request
    from configs.config import config
    key = config.CONTEXTUALIZER_API_KEY
    if not key:
        raise RuntimeError("CONTEXTUALIZER_API_KEY rỗng — set hhtechapi key trong .env trước khi chạy thật.")
    payload = json.dumps({
        "model": config.CONTEXTUALIZER_MODEL, "temperature": 0,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": text[:max_chars]}],
    }).encode("utf-8")
    req = urllib.request.Request(
        config.CONTEXTUALIZER_API_BASE.rstrip("/") + "/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def restore_record(raw: dict, complete_fn: Callable[[str], str]) -> dict:
    """Phục hồi MỘT bản ghi. Trả dict trạng thái (không ghi DB)."""
    doc_id = raw.get("doc_id") or raw.get("item_id")
    original = (raw.get("full_text") or raw.get("text") or "").strip()
    vi_old = compute_vi_ratio(original) or 0.0
    out = {"doc_id": doc_id, "doc_number": raw.get("doc_number"),
           "vi_before": vi_old, "status": "unresolved", "reason": "", "restored_text": None}
    if not original:
        out["reason"] = "no_text"
        return out
    try:
        restored = complete_fn(original)
    except Exception as e:
        out["reason"] = f"api_error:{type(e).__name__}"
        return out
    ok, reason = restore_guard(original, restored)
    out["reason"] = reason
    if ok:
        out["status"] = "restored"
        out["vi_after"] = compute_vi_ratio(restored)
        out["restored_text"] = restored
        out["needs_human_review"] = True       # §E Lớp 2: vẫn phải soi tay tập vàng
    return out


def _iter_targets(shards_dir: str, limit: Optional[int]):
    """Các bản ghi fallback_pdf dưới ngưỡng tiếng Việt (nhánh corpus_gate cách ly)."""
    from quality.crawl_audit import safe_read_jsonl
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    seen = 0
    for sh in shards:
        for raw in safe_read_jsonl(sh):
            if raw.get("content_source") != "fallback_pdf":
                continue
            text = (raw.get("full_text") or raw.get("text") or "").strip()
            vi = compute_vi_ratio(text) if text else None
            if vi is not None and vi < _VN_GATE:
                yield raw
                seen += 1
                if limit and seen >= limit:
                    return


def run(shards_dir: str, out_dir: str, limit: Optional[int] = None,
        dry_run: bool = False, complete_fn: Optional[Callable[[str], str]] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    fn = complete_fn or hhtech_complete
    stamp = date.today().strftime("%Y%m%d")
    manifest = os.path.join(out_dir, f"legacy_restore_{stamp}.jsonl")
    n = restored = 0
    reasons: dict[str, int] = {}
    with open(manifest, "w", encoding="utf-8") as mf:
        for raw in _iter_targets(shards_dir, limit):
            n += 1
            if dry_run:
                reasons["dry_run"] = reasons.get("dry_run", 0) + 1
                continue
            r = restore_record(raw, fn)
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            if r["status"] == "restored":
                restored += 1
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"targets": n, "restored": restored, "reasons": reasons,
               "manifest": manifest, "dry_run": dry_run}
    with open(os.path.join(out_dir, f"legacy_restore_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    from configs.paths import RAW_SHARDS_DIR, REPORTS_DIR
    ap = argparse.ArgumentParser(description="Phục hồi văn bản vỡ font legacy qua hhtechapi (§E-guarded).")
    ap.add_argument("--shards-dir", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--limit", type=int, default=None, help="số văn bản tối đa (bắt đầu nên để 2–5 smoke)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm mục tiêu, KHÔNG gọi API")
    a = ap.parse_args()
    s = run(a.shards_dir, a.out, a.limit, a.dry_run)
    print(f"Mục tiêu: {s['targets']} | phục hồi ĐẠT: {s['restored']} | lý do: {s['reasons']}")
    print(f"Manifest: {s['manifest']}  (ingest hợp nhất; content_source→llm_restored, cần soi tay §E Lớp 2)")
