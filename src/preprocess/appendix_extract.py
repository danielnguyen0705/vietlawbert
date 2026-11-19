# -*- coding: utf-8 -*-
"""
preprocess/appendix_extract.py
==============================
BÓC NỘI DUNG PHỤ LỤC từ file ZIP đính kèm văn bản (tab "tải về" của VBPL). Phụ lục KHÔNG
có cấu trúc đồng nhất: có thể là .docx / .pdf / .doc cũ / .xlsx, mỗi văn bản một kiểu. Module
điều phối theo ĐUÔI FILE, trả text + trạng thái cho từng thành phần, gắn vào văn bản CHA
(quan hệ cha–con: phụ lục là nút con "Phụ lục" dưới văn bản gốc). CHỈ ghi manifest, KHÔNG ghi DB.

Định tuyến:
  .docx → python-docx (bóc đoạn trực tiếp, không OCR).
  .pdf  → preprocess.pdf_restore.restore_pdf (native text / OCR + làm sạch).
  .doc (nhị phân cũ), .xls/.xlsx → GẮN CỜ unsupported (cần libreoffice/openpyxl) — không đoán bừa.
Tái sử dụng restore_pdf + compute_vi_ratio; không viết lại trích xuất.
"""
from __future__ import annotations
import argparse
import io
import json
import hashlib
import os
import re
import sys
import zipfile
from datetime import date
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "quality", "corpus_audit"))

from preprocess.pdf_restore import restore_pdf  # noqa: E402
from corpus_gate import compute_vi_ratio  # noqa: E402

_MIN_VI = 0.05


def extract_docx_text(data: bytes) -> str:
    """Bóc text .docx bằng python-docx (đoạn + bảng)."""
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_member(name: str, data: bytes, max_pages: int = 50) -> Dict[str, Any]:
    """Bóc một thành phần trong zip theo đuôi file. Trả dict trạng thái (không text nặng nếu rỗng)."""
    ext = os.path.splitext(name)[1].lower()
    rec: Dict[str, Any] = {"member": name, "ext": ext, "method": "", "status": "",
                           "chars": 0, "vi_ratio": 0.0, "text": ""}
    try:
        if ext == ".docx":
            txt = extract_docx_text(data)
            rec["method"] = "docx"
        elif ext == ".pdf":
            r = restore_pdf(data, max_pages=max_pages)
            txt = r["text"]
            rec["method"] = f"pdf_{r['method']}"
        elif ext in (".doc", ".xls", ".xlsx", ".rtf"):
            rec["method"] = ext.lstrip(".")
            rec["status"] = "unsupported_format"   # cần libreoffice/openpyxl — gắn cờ, không đoán
            return rec
        else:
            rec["status"] = "skipped_non_content"
            return rec
    except Exception as e:
        rec["status"] = f"error:{type(e).__name__}"
        return rec

    vr = compute_vi_ratio(txt) or 0.0
    rec["chars"] = len(txt)
    rec["vi_ratio"] = vr
    rec["text"] = txt
    rec["status"] = "accepted" if (len(txt.strip()) >= 100 and vr >= _MIN_VI) else "low_content"
    return rec


def extract_zip(zip_path: str, max_pages: int = 50) -> List[Dict[str, Any]]:
    """Bóc mọi thành phần nội dung trong zip (bỏ thư mục rỗng)."""
    out: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or info.file_size == 0:
                continue
            out.append(extract_member(info.filename, z.read(info), max_pages=max_pages))
    return out


_WS = re.compile(r"\s+")


def _norm_para(p: str) -> str:
    return _WS.sub(" ", p.strip()).lower()


def dedup_paragraphs(members: List[Dict[str, Any]], min_len: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Khử trùng ĐOẠN across mọi thành phần phụ lục (giữ lần xuất hiện ĐẦU TIÊN). Các mẫu HSMT
    lặp nhau rất nhiều → chỉ nhúng đoạn duy nhất, cắt mạnh chi phí. Trả (unique, stats).
    members: list[{member, text, ...}]. unique: [{text, source}] theo thứ tự gặp."""
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    total = 0
    for m in members:
        for para in str(m.get("text") or "").split("\n"):
            s = para.strip()
            if len(s) < min_len:
                continue
            total += 1
            h = hashlib.md5(_norm_para(s).encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            unique.append({"text": s, "source": m.get("member")})
    uniq_chars = sum(len(u["text"]) for u in unique)
    stats = {
        "total_paragraphs": total,
        "unique_paragraphs": len(unique),
        "dup_removed": total - len(unique),
        "unique_ratio": round(len(unique) / total, 4) if total else 0.0,
        "unique_chars": uniq_chars,
        "est_chunks_after_dedup": round(uniq_chars / 1500),
    }
    return unique, stats


def build_appendix_chunks(zip_path: str, doc_id: str, parent_metadata: Dict[str, Any],
                          max_pages: int = 50, target_chars: int = 1500) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Biến zip phụ lục thành các CHUNK CON của văn bản cha (đã khử trùng đoạn, đóng gói ~1500 ký tự).
    Chunk kế thừa status của cha (→ is_effective) và mang parent_doc_id (quan hệ cha–con).
    Trả (chunks hợp khuôn ingest, dedup_stats). KHÔNG ghi DB."""
    members = [r for r in extract_zip(zip_path, max_pages=max_pages) if r["status"] == "accepted"]
    unique, stats = dedup_paragraphs(members)
    pm = parent_metadata or {}
    status = pm.get("status", "")
    doc_number = pm.get("doc_number", "")
    chunks: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    blen = 0

    def flush():
        nonlocal buf, blen
        if not buf:
            return
        text = "\n".join(p["text"] for p in buf)
        chunks.append({
            "chunk_id": f"{doc_id}_appendix_{len(chunks)}",
            "doc_id": doc_id,
            "doc_number": doc_number,
            "text": text,
            "content": text,
            "metadata": {
                "status": status,
                "is_appendix": True,
                "parent_doc_id": doc_id,
                "appendix_sources": sorted({p["source"] for p in buf if p.get("source")}),
            },
        })
        buf = []
        blen = 0

    for p in unique:
        buf.append(p)
        blen += len(p["text"])
        if blen >= target_chars:
            flush()
    flush()
    return chunks, stats


def process(zip_path: str, out_dir: str, parent_doc_number: str = "",
            max_pages: int = 50, keep_text: bool = False) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    recs = extract_zip(zip_path, max_pages=max_pages)
    stamp = date.today().strftime("%Y%m%d")
    base = os.path.splitext(os.path.basename(zip_path))[0]
    manifest = os.path.join(out_dir, f"appendix_{stamp}.jsonl")
    by_status: Dict[str, int] = {}
    by_method: Dict[str, int] = {}
    total_chars = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for r in recs:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            by_method[r["method"] or "none"] = by_method.get(r["method"] or "none", 0) + 1
            total_chars += r["chars"]
            row = {"parent_doc_number": parent_doc_number, "appendix_zip": base,
                   **{k: v for k, v in r.items() if k != "text"}}
            if keep_text:
                row["text"] = r["text"]
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Khử trùng đoạn trên các thành phần ĐẠT → ước lượng chi phí nhúng thực tế (cha–con).
    _, dedup_stats = dedup_paragraphs([r for r in recs if r["status"] == "accepted"])
    summary = {"zip": zip_path, "parent_doc_number": parent_doc_number, "members": len(recs),
               "by_status": by_status, "by_method": by_method, "total_chars": total_chars,
               "est_chunks_if_embed_all": round(total_chars / 1500),
               "dedup": dedup_stats, "manifest": manifest}
    with open(os.path.join(out_dir, f"appendix_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    from configs.paths import REPORTS_DIR
    ap = argparse.ArgumentParser(description="Bóc nội dung phụ lục từ zip đính kèm văn bản, ghi manifest.")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--parent", default="", help="Số hiệu văn bản cha (vd 168/2026/TT-BTC)")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--keep-text", action="store_true")
    a = ap.parse_args()
    s = process(a.zip, a.out, a.parent, a.max_pages, a.keep_text)
    print(f"Phụ lục: {s['members']} thành phần | method={s['by_method']} | status={s['by_status']}")
    print(f"Tổng {s['total_chars']:,} ký tự → ~{s['est_chunks_if_embed_all']:,} chunks nếu nhúng TẤT CẢ")
    print(f"Manifest: {s['manifest']}")
