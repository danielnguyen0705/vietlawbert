"""
neo4j_ingest.py - CLI tool nạp dữ liệu phân cấp và quan hệ pháp lý vào Neo4j.
Hỗ trợ đọc trực tiếp tệp JSONL hoặc Shard nén Gzip (.jsonl.gz).
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from configs.logging_config import get_subsystem_logger
from database.neo4j_client import Neo4jClient
from database.cypher_templates import (
    get_trace_hierarchical_root,
    get_find_cascade_impact,
    get_neighbors,
    get_gg_slm_triplets,
)

logger = get_subsystem_logger("database", "database")


def stream_records(file_path: Path):
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                try:
                    yield json.loads(clean_line)
                except json.JSONDecodeError:
                    continue


def ingest_file_to_graph(neo: Neo4jClient, file_path: Path, batch_size: int = 500):
    logger.info(f"Bắt đầu nạp dữ liệu từ: {file_path.name}")
    struct_batch = []
    semantic_batch = []
    total_struct = 0
    total_semantic = 0

    for record in stream_records(file_path):
        doc_id = str(record.get("item_id") or record.get("doc_id") or "")
        meta = record.get("metadata", {}) or record.get("metadata_detail", {})
        hierarchy = meta.get("hierarchy_path", {})

        if "chunk_id" in record:
            struct_batch.append({
                "chunk_id": record.get("chunk_id", ""),
                "original_text": record.get("original_text", ""),
                "doc_id": doc_id,
                "doc_number": meta.get("doc_number") or record.get("doc_number", "N/A"),
                "effective_date": meta.get("effective_date", "Chưa xác định"),
                "source_doc": meta.get("doc_type", "") + " " + meta.get("doc_number", ""),
                "chuong": hierarchy.get("chương") or "Chương N/A",
                "dieu": hierarchy.get("điều") or "Điều N/A",
            })

        if "relationships" in record:
            semantic_batch.append(record)

        if len(struct_batch) >= batch_size:
            neo.insert_structural_batch(struct_batch)
            total_struct += len(struct_batch)
            struct_batch.clear()

        if len(semantic_batch) >= batch_size:
            inserted = neo.insert_semantic_relations_batch(semantic_batch)
            total_semantic += inserted
            semantic_batch.clear()

    if struct_batch:
        neo.insert_structural_batch(struct_batch)
        total_struct += len(struct_batch)

    if semantic_batch:
        inserted = neo.insert_semantic_relations_batch(semantic_batch)
        total_semantic += inserted

    logger.info(f"✓ Hoàn tất {file_path.name}: {total_struct} chunks phân cấp, {total_semantic} quan hệ ngữ nghĩa.")


def main():
    parser = argparse.ArgumentParser(description="Bộ điều phối nạp đồ thị tri thức Neo4j")
    parser.add_argument("--drop", action="store_true", help="Xóa sạch dữ liệu cũ trước khi nạp")
    parser.add_argument("--input", type=Path, help="Đường dẫn tới tệp .jsonl hoặc .jsonl.gz")
    parser.add_argument("--batch-size", type=int, default=500, help="Số lượng phần tử nạp trong 1 transaction")
    parser.add_argument(
        "--query",
        choices=["hierarchical", "cascade", "neighbors", "gg_slm"],
        help="Chạy thử nghiệm câu truy vấn Cypher mẫu",
    )
    parser.add_argument("--doc-id", help="Định danh văn bản (doc_id) phục vụ truy vấn")
    args = parser.parse_args()

    neo = Neo4jClient(drop_existing=args.drop)

    if args.query:
        if args.query in ("hierarchical", "neighbors") and not args.doc_id:
            logger.error(f"Lỗi: Kịch bản '{args.query}' bắt buộc phải cung cấp tham số --doc-id.")
            sys.exit(1)

        if args.query == "hierarchical":
            query, params = get_trace_hierarchical_root(args.doc_id)
        elif args.query == "cascade":
            query, params = get_find_cascade_impact()
        elif args.query == "gg_slm":
            query, params = get_gg_slm_triplets(limit=5)
        else:
            query, params = get_neighbors(args.doc_id)

        logger.info(f"Thực thi truy vấn Cypher: {args.query}")
        with neo.driver.session() as session:
            result = session.run(query, **params)
            for rec in result:
                print(dict(rec))
    elif args.input:
        if not args.input.exists():
            logger.error(f"Không tìm thấy tệp đầu vào: {args.input}")
            sys.exit(1)
        ingest_file_to_graph(neo, args.input, batch_size=args.batch_size)
        neo.build_cung_van_ban_relations()
    else:
        logger.info("Không có file đầu vào được chỉ định. Đã khởi tạo xong các ràng buộc cơ sở dữ liệu.")

    neo.close()


if __name__ == "__main__":
    main()