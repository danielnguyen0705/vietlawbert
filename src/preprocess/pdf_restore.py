# -*- coding: utf-8 -*-
"""
preprocess/pdf_restore.py
=========================
KHÔI PHỤC NỘI DUNG PDF cục bộ cho nhóm fallback_pdf (509 văn bản VBPL tải bản gốc).
Chẩn đoán thực tế trên mẫu 28 PDF: ba loại cùng tồn tại —
  (A) PDF SCAN ảnh (không có lớp text),
  (B) SCAN kèm lớp chữ ký số mỏng ("Ký bởi: BỘ TÀI CHÍNH"),
  (C) PDF có lớp text SẠCH ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", vi≈0.3).

Định tuyến TỐI GIẢN (đã kiểm chứng OCR cho vi≈0.31, broken≈0.00 trên cả 3 loại):
  - Lớp text gốc đủ dài & đạt ngưỡng tiếng Việt  → DÙNG TRỰC TIẾP (rẻ nhất, trung thực nhất).
  - Ngược lại (scan / vỡ font / chỉ có chữ ký số) → render 300dpi + Tesseract 'vie'. OCR đọc
    theo ẢNH nên xử lý được CẢ scan LẪN vỡ font TCVN3/VNI, KHÔNG cần gọi LLM (miễn phí, chạy CPU).

Tái sử dụng (không viết lại OCR): crawler.ocr_runner.perform_tesseract_ocr + ocr_broken_ratio,
corpus_gate.compute_vi_ratio. CHỈ ghi manifest reports/*.jsonl để ingest hợp nhất; KHÔNG ghi DB.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from datetime import date
from typing import Any, Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "quality", "corpus_audit"))

from crawler.ocr_runner import perform_tesseract_ocr, ocr_broken_ratio, OCR_BROKEN_THRESHOLD  # noqa: E402
from corpus_gate import compute_vi_ratio  # noqa: E402
from preprocess.ocr_clean import clean_ocr_text  # noqa: E402

_MIN_VI = 0.05
_MIN_NATIVE_CHARS = 200   # lớp text gốc ngắn hơn ngần này coi như chỉ còn chữ ký số/scan → đẩy sang OCR
_MIN_OCR_CHARS = 100


def extract_native_text(pdf_bytes: bytes, max_pages: int) -> str:
    """Bóc lớp text có sẵn trong PDF (không render). Rỗng nếu là scan ảnh."""
    import fitz  # PyMuPDF
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as d:
        for i in range(min(len(d), max_pages)):
            t = d[i].get_text()
            if t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts)


def restore_pdf(pdf_bytes: bytes, max_pages: int = 50) -> Dict[str, Any]:
    """Trả về {method, status, text, vi_ratio, broken_ratio, pages, chars}. KHÔNG ghi DB.
    status: 'accepted' | 'low_quality' | 'empty'. method: 'native_text' | 'ocr'."""
    if not pdf_bytes or len(pdf_bytes) < 100:
        return {"method": "none", "status": "empty", "text": "", "vi_ratio": 0.0,
                "broken_ratio": 0.0, "pages": 0, "chars": 0}
    try:
        import fitz
        with fitz.open(stream=pdf_bytes, filetype="pdf") as d:
            pages = len(d)
    except Exception:
        pages = 0

    native = extract_native_text(pdf_bytes, max_pages)
    nvi = compute_vi_ratio(native) or 0.0
    # Nhánh C: lớp text gốc sạch & đủ dài → dùng luôn, không OCR.
    if len(native.strip()) >= _MIN_NATIVE_CHARS and nvi >= _MIN_VI:
        return {"method": "native_text", "status": "accepted", "text": native,
                "vi_ratio": nvi, "broken_ratio": 0.0, "pages": pages, "chars": len(native)}

    # Nhánh A/B: scan hoặc vỡ font → render + Tesseract → làm sạch hậu-OCR (bỏ râu ria, nối dòng).
    ocr_raw = perform_tesseract_ocr(pdf_bytes, max_pages=max_pages)
    ocr, clean_stats = clean_ocr_text(ocr_raw)
    ovi = compute_vi_ratio(ocr) or 0.0
    broken = ocr_broken_ratio(ocr)
    if len(ocr.strip()) < _MIN_OCR_CHARS:
        status = "empty"
    elif ovi < _MIN_VI or broken > OCR_BROKEN_THRESHOLD:
        status = "low_quality"       # cách ly soi tay / tái-OCR engine mạnh hơn
    else:
        status = "accepted"
    return {"method": "ocr", "status": status, "text": ocr, "vi_ratio": ovi,
            "broken_ratio": round(broken, 4), "pages": pages, "chars": len(ocr),
            "needs_human_review": clean_stats["needs_human_review"],
            "dropped_signature": clean_stats["dropped_signature"],
            "dropped_junk": clean_stats["dropped_junk"]}


def process_folder(pdf_dir: str, out_dir: str, max_pages: int = 50,
                   keep_text: bool = False) -> Dict[str, Any]:
    """Duyệt mọi *.pdf, định tuyến + khôi phục, ghi manifest. Trả thống kê tổng hợp."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    manifest = os.path.join(out_dir, f"pdf_restore_{stamp}.jsonl")
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    by_method: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    with open(manifest, "w", encoding="utf-8") as mf:
        for p in pdfs:
            try:
                with open(p, "rb") as fh:
                    r = restore_pdf(fh.read(), max_pages=max_pages)
            except Exception as e:
                r = {"method": "error", "status": "error", "text": "", "vi_ratio": 0.0,
                     "broken_ratio": 0.0, "pages": 0, "chars": 0, "error": f"{type(e).__name__}:{e}"}
            by_method[r["method"]] = by_method.get(r["method"], 0) + 1
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            rec = {"file": os.path.basename(p), **{k: v for k, v in r.items() if k != "text"}}
            if keep_text:
                rec["text"] = r["text"]
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"pdf_dir": pdf_dir, "total": len(pdfs), "by_method": by_method,
               "by_status": by_status, "manifest": manifest, "max_pages": max_pages}
    with open(os.path.join(out_dir, f"pdf_restore_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    from configs.paths import REPORTS_DIR
    ap = argparse.ArgumentParser(description="Khôi phục nội dung PDF cục bộ (native text / Tesseract OCR), ghi manifest.")
    ap.add_argument("--pdf-dir", required=True, help="Thư mục chứa *.pdf (mẫu đại diện fallback_pdf)")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--max-pages", type=int, default=50, help="Số trang tối đa xử lý/OCR mỗi PDF")
    ap.add_argument("--keep-text", action="store_true", help="Lưu cả text đã khôi phục vào manifest")
    a = ap.parse_args()
    s = process_folder(a.pdf_dir, a.out, a.max_pages, a.keep_text)
    print(f"PDF: {s['total']} | method={s['by_method']} | status={s['by_status']}")
    print(f"Manifest: {s['manifest']}")
