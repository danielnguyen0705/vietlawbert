"""
ocr_runner.py - Bộ xử lý văn bản quét (PDF Scan Quarantine Batch Runner).
Thực thi Tesseract OCR thực thụ theo lô (Batch Processing) đa luồng an toàn.
Sử dụng đường dẫn tuyệt đối chuẩn hóa chống lỗi cross-mount point và hợp nhất nguyên tử.
"""

from __future__ import annotations

import os
import sys
import io
import json
import argparse
import logging
import urllib.request
import html as html_lib
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs.paths import ROOT_DIR, RAW_SHARDS_DIR
from configs.config import config
from artifacts.canonical import read_jsonl, write_jsonl
from artifacts.merge import merge_records_streaming
from crawler.shard_runner import write_json_atomic

logger = logging.getLogger("VietLawBERT_OCRRunner")


def pending_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        record
        for record in read_jsonl(path, require_item_id=False)
        if str(record.get("ocr_status") or "") == "OCR_PENDING"
        or (record.get("html_status") != "VALID" and record.get("rescue_file"))
    ]


def base_artifact_for(quarantine: Path) -> Path:
    base_name = quarantine.name.replace(".quarantine.jsonl", ".jsonl.gz")
    return quarantine.with_name(base_name)


def rescued_artifact_for(quarantine: Path) -> Path:
    rescued_name = quarantine.name.replace(".quarantine.jsonl", ".rescued.jsonl.gz")
    return quarantine.with_name(rescued_name)


def perform_tesseract_ocr(pdf_bytes: bytes, max_pages: int = 50) -> str:
    """Chuyển đổi các trang PDF scan thành hình ảnh và nhận diện chữ bằng Tesseract tiếng Việt."""
    if not pdf_bytes or len(pdf_bytes) < 100:
        return ""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []

        total_pages = min(len(doc), max_pages)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="vie")
            if text.strip():
                text_parts.append(text.strip())

        doc.close()
        return "\n\n".join(text_parts)
    except Exception as exc:
        logger.debug("Lỗi trong quá trình nhận dạng Tesseract: %s", exc)
        return ""


def process_quarantine_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Tải và OCR trực tiếp tệp PDF từ presignedUrl được lưu trong rescue_file."""
    rec = dict(record)
    rescue_file = rec.get("rescue_file") or {}
    url = rescue_file.get("presignedUrl") or rec.get("pdf_url") or rec.get("url")

    if not url or not str(url).startswith("http"):
        rec["ocr_status"] = "OCR_NO_URL"
        return rec

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            pdf_data = response.read()

        extracted_text = perform_tesseract_ocr(pdf_data)
        if len(extracted_text.strip()) >= 100:
            rec["html_status"] = "VALID"
            rec["ocr_status"] = "OCR_COMPLETED"
            rec["rescue_status"] = "OCR_RECOVERED"
            rec["content_source"] = "tesseract_ocr"
            clean_html = "<html><body><pre>" + html_lib.escape(extracted_text) + "</pre></body></html>"
            rec["html_raw"] = clean_html
            rec["full_text"] = extracted_text
            rec["text"] = extracted_text
        else:
            rec["ocr_status"] = "OCR_EMPTY_RESULT"
    except Exception as exc:
        logger.warning("Không thể OCR tài liệu %s: %s", rec.get("item_id"), exc)
        rec["ocr_status"] = "OCR_NETWORK_ERROR"

    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Bộ điều phối OCR thực thụ cho danh mục văn bản cách ly")
    parser.add_argument("--input-dir", type=Path, default=RAW_SHARDS_DIR, help="Thư mục chứa Shards")
    parser.add_argument("--max-shards", type=int, help="Giới hạn số Shard cần xử lý")
    parser.add_argument("--concurrency", type=int, default=getattr(config, "OCR_CONCURRENCY", 2), help="Số luồng tải và OCR song song")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = input_dir / "ocr_retries"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "ocr_state.json"

    state = []
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")).get("shards", [])
        except Exception:
            state = []

    completed_paths = {entry.get("quarantine") for entry in state}
    quarantines = [
        path
        for path in sorted(input_dir.glob("*.quarantine.jsonl"))
        if str(path) not in completed_paths
    ]

    if args.max_shards is not None:
        quarantines = quarantines[: args.max_shards]

    from quality.crawl_audit import audit_crawl

    for quarantine in quarantines:
        records = pending_records(quarantine)
        if not records:
            continue

        stem = quarantine.name.replace(".quarantine.jsonl", "")
        print(f"[TIẾN HÀNH OCR] Shard: {stem} | Số lượng tài liệu: {len(records)} (Đa luồng: {args.concurrency})...", flush=True)

        recovered = []
        unresolved = []

        # Xử lý đa luồng an toàn bộ nhớ bằng ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_rec = {executor.submit(process_quarantine_record, r): r for r in records}
            for future in as_completed(future_to_rec):
                try:
                    processed = future.result()
                    if processed.get("html_status") == "VALID":
                        recovered.append(processed)
                    else:
                        unresolved.append(processed)
                except Exception as ex:
                    rec_err = future_to_rec[future]
                    rec_err["ocr_status"] = f"OCR_WORKER_ERROR:{ex}"
                    unresolved.append(rec_err)

        recovered_path = output_dir / f"{stem}.ocr_recovered.jsonl"
        unresolved_path = output_dir / f"{stem}.ocr_unresolved.jsonl"
        write_jsonl(recovered_path, recovered)
        write_jsonl(unresolved_path, unresolved)

        base = base_artifact_for(quarantine)
        rescued = rescued_artifact_for(quarantine)

        gate = None
        if base.exists():
            merge_records_streaming(base_path=base, overlay_paths=[recovered_path], output_path=rescued)
            gate = audit_crawl(rescued, allow_upstream_missing=True, allow_ocr_pending=True)
            logger.info("✓ Đã hợp nhất dữ liệu OCR thành công vào %s", rescued.name)
        else:
            logger.warning("Không tìm thấy base artifact tương ứng: %s", base)

        entry = {
            "quarantine": str(quarantine),
            "base": str(base),
            "rescued": str(rescued),
            "pending": len(records),
            "recovered": len(recovered),
            "unresolved": len(unresolved),
            "gate": gate,
        }
        state.append(entry)
        write_json_atomic(state_path, {"shards": state})

    print("\n✓ HOÀN TẤT TIẾN TRÌNH XỬ LÝ OCR CHO TOÀN BỘ CÁC SHARD CÁCH LY.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())