"""
shard_runner.py - Điều phối chiến dịch cào phân đoạn (Sharding Crawl Orchestrator).
Tích hợp checkpoint tự phục hồi, cách ly tiến trình con giải phóng RAM và ghi file nguyên tử.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import math
import shutil
import argparse
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from configs.paths import ROOT_DIR, RAW_SHARDS_DIR
from artifacts.canonical import read_jsonl


@dataclass(frozen=True)
class Shard:
    start_page: int
    pages: int
    expected_documents: int

    @property
    def end_page(self) -> int:
        return self.start_page + self.pages - 1


def build_shards(total_documents: int, page_size: int, pages_per_shard: int) -> List[Shard]:
    total_pages = math.ceil(total_documents / page_size)
    shards = []
    for start_page in range(1, total_pages + 1, pages_per_shard):
        pages = min(pages_per_shard, total_pages - start_page + 1)
        consumed_before = (start_page - 1) * page_size
        expected = min(pages * page_size, total_documents - consumed_before)
        shards.append(Shard(start_page, pages, expected))
    return shards


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def artifact_path(output_dir: Path, shard: Shard) -> Path:
    return output_dir / f"crawl_pages_{shard.start_page:05d}_{shard.end_page:05d}.jsonl.gz"


def write_content_quarantine(artifact: Path) -> int:
    quarantine_path = artifact.with_name(artifact.name.replace(".jsonl.gz", ".quarantine.jsonl"))
    count = 0
    records = []

    if artifact.suffix == ".gz":
        with gzip.open(artifact, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = list(read_jsonl(artifact))

    with quarantine_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if record.get("html_status") == "VALID" and len(str(record.get("html_raw") or "").strip()) >= 100:
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if count == 0 and quarantine_path.exists():
        quarantine_path.unlink()
    return count


def run_shard(
    root_dir: Path,
    output_dir: Path,
    shard: Shard,
    env: dict,
    allow_upstream_missing: bool = False,
    allow_ocr_pending: bool = False,
) -> dict:
    from quality.crawl_audit import audit_crawl

    artifact = artifact_path(output_dir, shard)
    raw_artifact = artifact.with_name(artifact.name.replace(".jsonl.gz", ".jsonl"))
    audit_path = artifact.with_name(artifact.name.replace(".jsonl.gz", ".audit.json"))
    log_file = artifact.with_name(artifact.name.replace(".jsonl.gz", ".log"))

    # Sử dụng đường dẫn tuyệt đối chuẩn hóa tránh lỗi relpath qua các mount point khác nhau
    command = [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "law_spider",
        "-a",
        f"start_page={shard.start_page}",
        "-a",
        f"pages={shard.pages}",
        "-a",
        f"page_size={env['CRAWL_PAGE_SIZE']}",
        "-a",
        f"limit={shard.expected_documents}",
        "-s",
        f"LOG_FILE={str(log_file.resolve())}",
        "-O",
        str(raw_artifact.resolve()),
    ]

    result = None
    if raw_artifact.exists():
        candidate = audit_crawl(raw_artifact, shard.expected_documents, allow_upstream_missing, allow_ocr_pending)
        if candidate["passed"]:
            print(f"[KHÔI PHỤC TỆP THÔ THÀNH CÔNG] {raw_artifact.name}", flush=True)
            result = candidate

    if result is None:
        subprocess.run(command, cwd=root_dir, env=env, check=True)
        result = audit_crawl(raw_artifact, shard.expected_documents, allow_upstream_missing, allow_ocr_pending)

    if not result["passed"]:
        write_json_atomic(audit_path, result)
        raise RuntimeError(f"Shard {shard.start_page}-{shard.end_page} không vượt qua Quality Gate: {result['failures']}")

    # Nén Gzip bảo toàn dung lượng đĩa
    with raw_artifact.open("rb") as source, gzip.open(artifact, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    compressed_result = audit_crawl(artifact, shard.expected_documents, allow_upstream_missing, allow_ocr_pending)
    if not compressed_result["passed"]:
        raise RuntimeError(f"Tệp nén Shard {shard.start_page}-{shard.end_page} bị lỗi sau khi nén Gzip!")

    raw_artifact.unlink(missing_ok=True)
    result = compressed_result
    result["quarantined_content_records"] = write_content_quarantine(artifact)
    write_json_atomic(audit_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Chương trình cào phân đoạn chống OOM cho VietLawBERT")
    parser.add_argument("--total-documents", type=int, default=160660, help="Tổng số lượng văn bản cần cào")
    parser.add_argument("--page-size", type=int, default=100, help="Số văn bản trên 1 trang API")
    parser.add_argument("--pages-per-shard", type=int, default=10, help="Số trang gom vào 1 Shard (1.000 docs)")
    parser.add_argument("--output-dir", type=Path, default=RAW_SHARDS_DIR, help="Thư mục xuất Shards")
    parser.add_argument("--concurrency", type=int, default=4, help="Số luồng cào song song")
    parser.add_argument("--download-delay", type=float, default=0.1, help="Độ trễ giữa các request")
    parser.add_argument("--allow-upstream-missing", action="store_true", default=True, help="Chấp nhận template rỗng")
    parser.add_argument("--defer-ocr", action="store_true", default=True, help="Trì hoãn OCR vào phân vùng cách ly")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "crawl_state.json"

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT_DIR),
        "CRAWLER_CONCURRENCY": str(args.concurrency),
        "CRAWLER_DOWNLOAD_DELAY": str(args.download_delay),
        "CRAWL_PAGE_SIZE": str(args.page_size),
        "OCR_INLINE_ENABLED": "0" if args.defer_ocr else "1",
    })

    from quality.crawl_audit import audit_crawl

    shards = build_shards(args.total_documents, args.page_size, args.pages_per_shard)
    completed = []

    for index, shard in enumerate(shards, 1):
        artifact = artifact_path(output_dir, shard)
        if artifact.exists():
            existing = audit_crawl(artifact, shard.expected_documents, args.allow_upstream_missing, args.defer_ocr)
            if not existing["passed"]:
                raise RuntimeError(f"Tệp Shard {artifact.name} không đạt chuẩn Quality Gate.")
            print(f"[BỎ QUA VÌ ĐÃ CÀO] {index}/{len(shards)}: {artifact.name}", flush=True)
            result = existing
        else:
            print(f"[TIẾN HÀNH CÀO] {index}/{len(shards)}: Trang {shard.start_page} - {shard.end_page}", flush=True)
            result = run_shard(
                ROOT_DIR,
                output_dir,
                shard,
                env,
                allow_upstream_missing=args.allow_upstream_missing,
                allow_ocr_pending=args.defer_ocr,
            )

        completed.append({**asdict(shard), "artifact": str(artifact), "audit": result})
        write_json_atomic(
            state_path,
            {
                "total_documents": args.total_documents,
                "page_size": args.page_size,
                "pages_per_shard": args.pages_per_shard,
                "completed_shards": completed,
                "complete": len(completed) == len(shards),
            },
        )

    print("\n✓ HOÀN THÀNH CÀO PHÂN ĐOẠN VĂN BẢN PHÁP LUẬT.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())