"""Xử lý OCR_PENDING theo batch và overlay kết quả hợp lệ vào từng shard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from quality.crawl_audit import audit_crawl, read_jsonl
from artifacts.merge import merge_records, write_jsonl
from run_crawl_shards import write_json_atomic


def pending_ids(path: Path) -> list[str]:
    return [
        str(record["item_id"])
        for record in read_jsonl(path)
        if str(record.get("ocr_status") or "") == "OCR_PENDING"
    ]


def base_artifact_for(quarantine: Path) -> Path:
    suffix = ".quarantine.jsonl"
    if not quarantine.name.endswith(suffix):
        raise ValueError(f"Tên quarantine không hợp lệ: {quarantine}")
    return quarantine.with_name(quarantine.name[: -len(suffix)] + ".gz")


def rescued_artifact_for(quarantine: Path) -> Path:
    base = base_artifact_for(quarantine)
    return base.with_name(base.name.replace(".jsonl.gz", ".rescued.jsonl.gz"))


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def result_has_ids(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    actual = [str(record.get("item_id") or "") for record in read_jsonl(path)]
    return len(actual) == len(expected_ids) and set(actual) == set(expected_ids)


def run_batch(src_dir: Path, ids: list[str], output: Path, log: Path, env: dict[str, str]) -> None:
    command = [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        "law_spider",
        "-a",
        f"doc_ids={','.join(ids)}",
        "-s",
        f"LOG_FILE={os.path.relpath(log, src_dir)}",
        "-O",
        os.path.relpath(output, src_dir),
    ]
    subprocess.run(command, cwd=src_dir, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR targeted cho quarantine của full crawl")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")

    src_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir.resolve()
    output_dir = input_dir / "ocr_retries"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "ocr_state.json"
    env = os.environ.copy()
    env.update({"KAFKA_ENABLED": "0", "OCR_ENABLED": "1", "OCR_INLINE_ENABLED": "1"})

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8")).get("shards", [])
    else:
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
        stem = quarantine.name.removesuffix(".jsonl.quarantine.jsonl")
        batch_outputs = []
        for index, batch in enumerate(chunks(ids, args.batch_size), 1):
            output = output_dir / f"{stem}.ocr_{index:04d}.jsonl"
            log = output.with_suffix(".log")
            if not result_has_ids(output, batch):
                print(f"[OCR] {stem} batch {index}: {len(batch)} IDs", flush=True)
                run_batch(src_dir, batch, output, log, env)
            else:
                print(f"[OCR SKIP] {output.name}", flush=True)
            if not result_has_ids(output, batch):
                raise RuntimeError(f"OCR batch không trả đủ ID: {output}")
            batch_outputs.append(output)

        results = [record for path in batch_outputs for record in read_jsonl(path)]
        recovered = [
            record
            for record in results
            if record.get("html_status") == "VALID"
            and len(str(record.get("html_raw") or "").strip()) >= 100
        ]
        unresolved = [record for record in results if record not in recovered]
        recovered_path = output_dir / f"{stem}.ocr_recovered.jsonl"
        unresolved_path = output_dir / f"{stem}.ocr_unresolved.jsonl"
        write_jsonl(recovered_path, recovered)
        write_jsonl(unresolved_path, unresolved)

        base = base_artifact_for(quarantine)
        rescued = rescued_artifact_for(quarantine)
        merged = merge_records(read_jsonl(base), [recovered])
        write_jsonl(rescued, merged)
        gate = audit_crawl(
            rescued,
            expected_documents=len(merged),
            allow_upstream_missing=True,
            allow_ocr_pending=True,
        )
        if not gate["passed"]:
            raise RuntimeError(f"Artifact sau OCR không qua gate: {rescued}: {gate['failures']}")
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
    raise SystemExit(main())
