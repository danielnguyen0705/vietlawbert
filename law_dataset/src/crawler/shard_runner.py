"""Chạy full crawl theo shard có resume và quality gate từng checkpoint."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from quality.crawl_audit import audit_crawl, read_jsonl


@dataclass(frozen=True)
class Shard:
    start_page: int
    pages: int
    expected_documents: int

    @property
    def end_page(self) -> int:
        return self.start_page + self.pages - 1


def build_shards(total_documents: int, page_size: int, pages_per_shard: int) -> list[Shard]:
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_path(output_dir: Path, shard: Shard) -> Path:
    return output_dir / f"crawl_pages_{shard.start_page:05d}_{shard.end_page:05d}.jsonl.gz"


def write_content_quarantine(artifact: Path) -> int:
    quarantine_path = artifact.with_suffix(".quarantine.jsonl")
    count = 0
    with quarantine_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in read_jsonl(artifact):
            if record.get("html_status") == "VALID" and len(str(record.get("html_raw") or "").strip()) >= 100:
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        quarantine_path.unlink()
    return count


def run_shard(
    src_dir: Path,
    output_dir: Path,
    shard: Shard,
    env: dict[str, str],
    allow_upstream_missing: bool = False,
    allow_ocr_pending: bool = False,
) -> dict:
    artifact = artifact_path(output_dir, shard)
    raw_artifact = artifact.with_suffix("")
    audit_path = artifact.with_suffix(".audit.json")
    relative_artifact = os.path.relpath(raw_artifact, src_dir)
    relative_log = os.path.relpath(artifact.with_suffix(".log"), src_dir)
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
        f"LOG_FILE={relative_log}",
        "-O",
        relative_artifact,
    ]
    result = None
    if raw_artifact.exists():
        candidate = audit_crawl(
            raw_artifact,
            shard.expected_documents,
            allow_upstream_missing,
            allow_ocr_pending,
        )
        if candidate["passed"]:
            print(f"[RAW RESUME PASS] {raw_artifact.name}", flush=True)
            result = candidate
    if result is None:
        subprocess.run(command, cwd=src_dir, env=env, check=True)
        result = audit_crawl(
            raw_artifact,
            shard.expected_documents,
            allow_upstream_missing,
            allow_ocr_pending,
        )
    if not result["passed"]:
        write_json_atomic(audit_path, result)
        raise RuntimeError(f"shard {shard.start_page}-{shard.end_page} không qua gate: {result['failures']}")

    # Scrapy CLI không nhận `.jsonl.gz` là format. Chỉ nén và xóa file tạm
    # sau khi JSONL thô đã qua gate, rồi audit lại chính artifact nén.
    with raw_artifact.open("rb") as source, gzip.open(artifact, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    compressed_result = audit_crawl(
        artifact,
        shard.expected_documents,
        allow_upstream_missing,
        allow_ocr_pending,
    )
    if not compressed_result["passed"]:
        raise RuntimeError(f"artifact nén của shard {shard.start_page}-{shard.end_page} không qua gate")
    raw_artifact.unlink()
    result = compressed_result
    result["quarantined_content_records"] = write_content_quarantine(artifact)
    write_json_atomic(audit_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Full crawl có checkpoint, resume và audit")
    parser.add_argument("--total-documents", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--pages-per-shard", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("../artifacts/full_crawl"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--download-delay", type=float, default=0)
    parser.add_argument(
        "--allow-upstream-missing",
        action="store_true",
        help="quarantine hasContent=false + Template.pdf, nhưng vẫn dừng với content lỗi không giải thích được",
    )
    parser.add_argument(
        "--defer-ocr",
        action="store_true",
        help="không OCR inline; đưa PDF scan vào quarantine để xử lý ở pha riêng",
    )
    args = parser.parse_args()
    if min(args.total_documents, args.page_size, args.pages_per_shard, args.concurrency) <= 0:
        parser.error("các tham số số lượng phải lớn hơn 0")

    src_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "crawl_state.json"
    env = os.environ.copy()
    env.update(
        {
            "KAFKA_ENABLED": "0",
            "CRAWLER_CONCURRENCY": str(args.concurrency),
            "CRAWLER_DOWNLOAD_DELAY": str(args.download_delay),
            "CRAWL_PAGE_SIZE": str(args.page_size),
            "OCR_INLINE_ENABLED": "0" if args.defer_ocr else "1",
        }
    )

    shards = build_shards(args.total_documents, args.page_size, args.pages_per_shard)
    completed = []
    for index, shard in enumerate(shards, 1):
        artifact = artifact_path(output_dir, shard)
        if artifact.exists():
            existing = audit_crawl(
                artifact,
                shard.expected_documents,
                args.allow_upstream_missing,
                args.defer_ocr,
            )
            if not existing["passed"]:
                raise RuntimeError(
                    f"checkpoint có sẵn {artifact} không pass; di chuyển/xóa riêng file này rồi resume"
                )
            print(f"[SKIP PASS] {index}/{len(shards)} {artifact.name}", flush=True)
            result = existing
        else:
            print(f"[CRAWL] {index}/{len(shards)} pages {shard.start_page}-{shard.end_page}", flush=True)
            result = run_shard(
                src_dir,
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
                "allow_upstream_missing": args.allow_upstream_missing,
                "defer_ocr": args.defer_ocr,
                "completed_shards": completed,
                "complete": len(completed) == len(shards),
            },
        )

    # Kiểm tra duplicate xuyên shard mà không giữ payload HTML trong RAM.
    seen: set[str] = set()
    duplicates: set[str] = set()
    for shard in shards:
        for record in read_jsonl(artifact_path(output_dir, shard)):
            item_id = str(record.get("item_id") or "").strip()
            if item_id in seen:
                duplicates.add(item_id)
            seen.add(item_id)
    final = {
        "expected_documents": args.total_documents,
        "unique_documents": len(seen),
        "cross_shard_duplicates": len(duplicates),
        "content_valid": sum(entry["audit"]["html_valid"] for entry in completed),
        "upstream_content_unavailable": sum(
            entry["audit"]["upstream_content_unavailable"] for entry in completed
        ),
        "ocr_pending": sum(entry["audit"]["ocr_pending"] for entry in completed),
        "passed": len(seen) == args.total_documents and not duplicates,
    }
    final["content_complete"] = final["content_valid"] == args.total_documents
    write_json_atomic(output_dir / "full_crawl_gate.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if final["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
