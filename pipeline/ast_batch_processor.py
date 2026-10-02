"""
ast_batch_processor.py - Phân rã cấu trúc văn bản pháp luật bằng Hybrid AST Parser.
Đọc song song các Shards .jsonl.gz và xuất ra định dạng Apache Parquet chuẩn hóa schema V3.
"""

from __future__ import annotations

import os
import gzip
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from configs.paths import RAW_SHARDS_DIR, PROCESSED_DIR
from preprocess.ast_parser import HybridASTParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_ASTBatchProcessor")

OUTPUT_PARQUET_DIR = PROCESSED_DIR / "parquet"
OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("doc_number", pa.string()),
    ("doc_title", pa.string()),
    ("article_id", pa.string()),
    ("article_num", pa.string()),
    ("hierarchy_path", pa.string()),
    ("macro_label", pa.string()),
    ("content", pa.string()),
    ("full_context_text", pa.string()),
    ("issue_date", pa.string()),
    ("effective_date", pa.string()),
    ("status", pa.string()),
    ("is_administrative", pa.bool_()),
])


def process_single_shard(shard_path_str: str) -> Dict[str, Any]:
    shard_path = Path(shard_path_str)
    parquet_path = OUTPUT_PARQUET_DIR / shard_path.name.replace(".jsonl.gz", ".parquet").replace(".jsonl", ".parquet")

    if parquet_path.exists():
        return {"shard": shard_path.name, "chunks": 0, "status": "ALREADY_EXISTS"}

    parser = HybridASTParser()
    all_chunks = []

    opener = gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") if shard_path.suffix == ".gz" else open(shard_path, "r", encoding="utf-8", errors="replace")
    with opener as stream:
        for line in stream:
            clean_line = line.strip()
            if not clean_line:
                continue
            try:
                record = json.loads(clean_line)
            except Exception:
                continue

            raw_text = record.get("full_text") or record.get("text") or record.get("html_raw") or ""
            if len(raw_text.strip()) < 50:
                continue

            meta_detail = record.get("metadata_detail") or {}
            meta_api = record.get("metadata_api") or {}

            doc_id = str(record.get("doc_id") or record.get("item_id") or "")
            doc_number = str(record.get("doc_number") or meta_detail.get("docNum") or meta_api.get("docNum") or "N/A")
            title = str(record.get("title") or meta_detail.get("title") or meta_api.get("title") or "")
            issue_date = str(meta_detail.get("issueDate") or meta_api.get("issueDate") or "")
            effective_date = str(record.get("effective_date") or meta_detail.get("effFrom") or meta_api.get("effFrom") or "")
            status_val = str(record.get("status") or (meta_detail.get("effStatus") or {}).get("name") or "Còn hiệu lực")
            is_admin = bool(meta_detail.get("isAdministrativeDocument"))

            metadata = {
                "doc_id": doc_id,
                "doc_number": doc_number,
                "title": title,
                "effective_date": effective_date,
                "status": status_val,
                "co_quan": str(record.get("co_quan") or meta_detail.get("agencyName") or "N/A"),
            }

            try:
                chunks = parser.parse_document(raw_text, metadata)
                for c in chunks:
                    all_chunks.append({
                        "chunk_id": str(c.get("chunk_id") or ""),
                        "doc_id": doc_id,
                        "doc_number": doc_number,
                        "doc_title": title,
                        "article_id": str(c.get("article_id") or ""),
                        "article_num": str(c.get("article_num") or ""),
                        "hierarchy_path": str(c.get("hierarchy_path") or ""),
                        "macro_label": str(c.get("macro_label") or "CHUNG"),
                        "content": str(c.get("content") or c.get("text") or ""),
                        "full_context_text": str(c.get("content") or c.get("text") or ""),
                        "issue_date": issue_date,
                        "effective_date": effective_date,
                        "status": status_val,
                        "is_administrative": is_admin,
                    })
            except Exception:
                continue

    if all_chunks:
        table = pa.Table.from_pylist(all_chunks, schema=PARQUET_SCHEMA)
        pq.write_table(table, parquet_path, compression="snappy")

    return {"shard": shard_path.name, "chunks": len(all_chunks), "status": "SUCCESS"}


def main():
    shard_files = sorted([
        str(f) for f in RAW_SHARDS_DIR.glob("*.jsonl*")
        if not f.name.endswith(".quarantine.jsonl") and not f.name.endswith(".corrupted")
    ])
    logger.info("Bắt đầu bóc tách AST phân cấp chuẩn hóa cho %d Shards...", len(shard_files))

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    total_extracted_chunks = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(process_single_shard, shard_files):
            total_extracted_chunks += res["chunks"]
            if res["status"] == "SUCCESS":
                logger.info("[AST PARSED] %s: Trích xuất thành công %d Chunks", res["shard"], res["chunks"])

    logger.info("=================================================================")
    logger.info("✓ HOÀN THÀNH PHA 2: Tổng số Chunks phân cấp Parquet: %d", total_extracted_chunks)
    logger.info("Thư mục lưu trữ: %s", OUTPUT_PARQUET_DIR)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()