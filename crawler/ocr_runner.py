"""
ocr_runner.py - Bộ xử lý văn bản quét (PDF Scan Quarantine Batch Runner).
Thực thi OCR hàng loạt theo lô (Mini-batch) và hợp nhất kết quả nguyên tử vào Shards.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List

from configs.paths import ROOT_DIR, ARTIFACTS_DIR
from artifacts.canonical import read_jsonl, write_jsonl
from artifacts.merge import merge_records_streaming
from crawler.shard_runner import write_json_atomic

logger = logging.getLogger("VietLawBERT_OCRRunner")


def pending_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        str(record["item_id"])
        for record in read_jsonl(path)
        if str(record.get("ocr_status") or "") == "OCR_PENDING"
    ]


def base_artifact_for(quarantine: Path) -> Path:
    base_name = quarantine.name.replace(".quarantine.jsonl", ".jsonl.gz")
    return quarantine.with_name(base_name)


def rescued_artifact_for(quarantine: Path) -> Path:
    rescued_name = quarantine.name.replace(".quarantine.jsonl", ".rescued.jsonl.gz")
    return quarantine.with_name(rescued_name)


def chunks(values: List[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_batch(root_dir: Path, ids: List[str], output: Path, log: Path, env: dict) -> bool:
    command = [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "law_spider",
        "-a",
        f"doc_ids={','.join(ids)}",
        "-s",
        f"LOG_FILE={os.path.relpath(log, root_dir)}",
        "-O",
        os.path.relpath(output, root_dir),
    ]
    try:
        subprocess.run(command, cwd=root_dir, env=env, check=True)
        return True
    except subprocess.CalledProcessError as err:
        logger.error(f"[LỖI LÔ OCR] Tiến trình con Scrapy thất bại cho lô {output.name}: {err}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Bộ điều phối OCR cho danh mục văn bản cách ly")
    parser.add_argument("--input-dir", type=Path, default=ARTIFACTS_DIR / "full_crawl", help="Thư mục chứa Shards")
    parser.add_argument("--batch-size", type=int, default=10, help="Số lượng PDF xử lý trong 1 lượt")
    parser.add_argument("--max-shards", type=int, help="Giới hạn số Shard cần xử lý")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = input_dir / "ocr_retries"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "ocr_state.json"

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT_DIR),
        "KAFKA_ENABLED": "0",
        "OCR_ENABLED": "1",
        "OCR_INLINE_ENABLED": "1",
    })

    from quality.crawl_audit import audit_crawl

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
        if pending_ids(path) and str(path) not in completed_paths
    ]

    if args.max_shards is not None:
        quarantines = quarantines[: args.max_shards]

    for quarantine in quarantines:
        ids = pending_ids(quarantine)
        if not ids:
            continue

        stem = quarantine.name.replace(".quarantine.jsonl", "")
        batch_outputs = []

        for index, batch in enumerate(chunks(ids, args.batch_size), 1):
            output = output_dir / f"{stem}.ocr_{index:04d}.jsonl"
            log = output.with_suffix(".log")

            if not output.exists():
                print(f"[TIẾN HÀNH OCR] {stem} lô {index}: {len(batch)} tài liệu...", flush=True)
                success = run_batch(ROOT_DIR, batch, output, log, env)
                if not success and not output.exists():
                    output.touch()
            else:
                print(f"[BỎ QUA LÔ OCR] {output.name} đã tồn tại.", flush=True)

            batch_outputs.append(output)

        # Tổng hợp kết quả từ các lô
        results = [record for path in batch_outputs if path.exists() for record in read_jsonl(path)]
        recovered = [
            record
            for record in results
            if record.get("html_status") == "VALID" and len(str(record.get("html_raw") or "").strip()) >= 100
        ]
        recovered_ids = {str(r.get("item_id")) for r in recovered}

        # Bảo toàn tất cả ID chưa xử lý thành công (kể cả khi scrapy bị rơi rớt record)
        all_ids_set = set(ids)
        unresolved_ids = all_ids_set - recovered_ids

        raw_unresolved_map = {str(r.get("item_id")): r for r in results if str(r.get("item_id")) in unresolved_ids}
        unresolved = []
        for missing_id in sorted(unresolved_ids):
            if missing_id in raw_unresolved_map:
                unresolved.append(raw_unresolved_map[missing_id])
            else:
                unresolved.append({
                    "item_id": missing_id,
                    "ocr_status": "OCR_FAILED_CRASH",
                    "html_status": "CORRUPTED",
                    "html_raw": "",
                })

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
        else:
            logger.warning(f"Không tìm thấy base artifact tương ứng: {base}")

        entry = {
            "quarantine": str(quarantine),
            "base": str(base),
            "rescued": str(rescued),
            "pending": len(ids),
            "recovered": len(recovered),
            "unresolved": len(unresolved),
            "gate": gate,
        }
        state.append(entry)
        write_json_atomic(state_path, {"shards": state})
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
