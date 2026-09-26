"""
neo4j_batch_loader.py - Điều phối nạp dữ liệu phân cấp và 22 quan hệ pháp lý vào Neo4j.
Tối ưu hóa giao dịch vi mô (UNWIND Batching) chống tràn Heap JVM và gián đoạn kết nối Bolt.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Set
import pyarrow.parquet as pq
from neo4j import GraphDatabase

PARQUET_DIR = Path("/mnt/data/vietlawbert_data/processed_parquet")
RAW_SHARDS_DIR = Path("/mnt/data/vietlawbert_data/raw_shards")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vietlawbert")

CYPHER_DOC_UPSERT = """
UNWIND $docs AS row
MERGE (d:LawDocument {doc_id: row.doc_id})
ON CREATE SET 
    d.doc_number = row.doc_number,
    d.doc_title = row.doc_title,
    d.issue_date = row.issue_date,
    d.effective_date = row.effective_date,
    d.is_administrative = row.is_administrative
ON MATCH SET
    d.doc_number = coalesce(d.doc_number, row.doc_number),
    d.doc_title = coalesce(d.doc_title, row.doc_title)
"""

CYPHER_CHUNK_UPSERT = """
UNWIND $chunks AS row
MATCH (d:LawDocument {doc_id: row.doc_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
ON CREATE SET 
    c.hierarchy_path = row.hierarchy_path,
    c.level = row.level,
    c.content = row.content
MERGE (d)-[:HAS_CHUNK]->(c)
"""

CYPHER_RELATION_UPSERT = """
UNWIND $relations AS rel
MATCH (src:LawDocument {doc_id: rel.source_id})
MATCH (tgt:LawDocument {doc_id: rel.target_id})
MERGE (src)-[r:LEGAL_RELATION {type: rel.type}]->(tgt)
ON CREATE SET r.created_at = timestamp()
"""


def load_parquet_stage(driver, parquet_files: List[str]):
    print(f"\n--- [GIAI ĐOẠN 1] Nạp cấu trúc Văn bản & Chunks từ {len(parquet_files)} tệp Parquet ---")
    total_docs_seen = 0
    total_chunks_loaded = 0
    start_time = time.time()

    with driver.session() as session:
        for idx, file_path in enumerate(parquet_files, 1):
            table = pq.read_table(file_path).to_pydict()
            num_rows = len(table["chunk_id"])

            docs_map: Dict[str, Dict[str, Any]] = {}
            chunks_list: List[Dict[str, Any]] = []

            for i in range(num_rows):
                doc_id = table["doc_id"][i]
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        "doc_id": doc_id,
                        "doc_number": table["doc_number"][i],
                        "doc_title": table["doc_title"][i],
                        "issue_date": table["issue_date"][i],
                        "effective_date": table["effective_date"][i],
                        "is_administrative": table["is_administrative"][i],
                    }

                chunks_list.append({
                    "chunk_id": table["chunk_id"][i],
                    "doc_id": doc_id,
                    "hierarchy_path": table["hierarchy_path"][i],
                    "level": table["level"][i],
                    "content": table["content"][i],
                })

            docs_payload = list(docs_map.values())
            session.run(CYPHER_DOC_UPSERT, docs=docs_payload)
            total_docs_seen += len(docs_payload)

            batch_size = 2000
            for b_idx in range(0, len(chunks_list), batch_size):
                sub_batch = chunks_list[b_idx : b_idx + batch_size]
                session.run(CYPHER_CHUNK_UPSERT, chunks=sub_batch)

            total_chunks_loaded += num_rows
            if idx % 10 == 0 or idx == len(parquet_files):
                elapsed = time.time() - start_time
                speed = total_chunks_loaded / max(1, elapsed)
                print(f"[{idx:03d}/{len(parquet_files)}] Đã nạp: {total_chunks_loaded:,} Chunks (~{speed:.1f} chunks/s)")

    print(f"✓ Hoàn tất Giai đoạn 1: {total_chunks_loaded:,} Chunks liên kết vào {total_docs_seen:,} Văn bản.")


def extract_relationships_from_raw_record(record: dict) -> List[Dict[str, str]]:
    source_id = str(record.get("item_id") or record.get("doc_id") or "").strip()
    if not source_id:
        return []

    results = []

    # 1. Trích xuất từ mảng relationships trực tiếp
    raw_rels = record.get("relationships") or []
    for r in raw_rels:
        target_id = str(r.get("target_id") or r.get("id") or "").strip()
        rel_type = str(r.get("type") or r.get("relation_type") or "DAN_CHIEU").strip().upper()
        if target_id and source_id != target_id:
            results.append({"source_id": source_id, "target_id": target_id, "type": rel_type})

    # 2. Trích xuất từ sơ đồ diagram_json (nếu có)
    diagram = record.get("diagram_json") or {}
    edges = diagram.get("edges") or diagram.get("relations") or []
    for edge in edges:
        target_id = str(edge.get("target") or edge.get("target_id") or edge.get("to") or "").strip()
        rel_type = str(edge.get("label") or edge.get("type") or "DAN_CHIEU").strip().upper()
        if target_id and source_id != target_id:
            results.append({"source_id": source_id, "target_id": target_id, "type": rel_type})

    return results


def load_relations_stage(driver, raw_shard_files: List[str]):
    print(f"\n--- [GIAI ĐOẠN 2] Trích xuất & Nạp 22 quan hệ pháp lý từ {len(raw_shard_files)} Raw Shards ---")
    total_relations_loaded = 0
    start_time = time.time()

    with driver.session() as session:
        for idx, shard_path in enumerate(raw_shard_files, 1):
            relations_batch: List[Dict[str, str]] = []
            with gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") as gz:
                for line in gz:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        rels = extract_relationships_from_raw_record(record)
                        relations_batch.extend(rels)
                    except Exception:
                        continue

            if relations_batch:
                # Deduplicate nhẹ trong cùng 1 shard
                unique_rels = {
                    (r["source_id"], r["target_id"], r["type"]): r for r in relations_batch
                }.values()

                session.run(CYPHER_RELATION_UPSERT, relations=list(unique_rels))
                total_relations_loaded += len(unique_rels)

            if idx % 10 == 0 or idx == len(raw_shard_files):
                elapsed = time.time() - start_time
                print(f"[{idx:03d}/{len(raw_shard_files)}] Đã liên kết: {total_relations_loaded:,} Cạnh quan hệ pháp lý")

    print(f"✓ Hoàn tất Giai đoạn 2: {total_relations_loaded:,} Quan hệ pháp lý đã được nạp.")


def main():
    parquet_files = sorted(glob.glob(str(PARQUET_DIR / "crawl_pages_*_*.parquet")))
    raw_shard_files = sorted(glob.glob(str(RAW_SHARDS_DIR / "crawl_pages_*_*.jsonl.gz")))

    if not parquet_files:
        print(f"[-] Không tìm thấy tệp Parquet nào tại: {PARQUET_DIR}")
        return 1

    print("=" * 65)
    print("KHỞI ĐỘNG TIẾN TRÌNH NẠP ĐỒ THỊ TRI THỨC PHÁP LUẬT VIỆT NAM (NEO4J)")
    print(f"Neo4j URI: {NEO4J_URI} | User: {NEO4J_USER}")
    print(f"Tổng số Parquet Shards : {len(parquet_files)}")
    print(f"Tổng số Raw Shards     : {len(raw_shard_files)}")
    print("=" * 65)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # Giai đoạn 1: Nạp Nodes & AST
        load_parquet_stage(driver, parquet_files)

        # Giai đoạn 2: Nạp Cạnh quan hệ
        if raw_shard_files:
            load_relations_stage(driver, raw_shard_files)
        else:
            print("[CẢNH BÁO] Không tìm thấy raw_shard_files để nạp quan hệ liên văn bản.")

    finally:
        driver.close()

    print("\n" + "=" * 65)
    print("✓ HOÀN THÀNH TOÀN DIỆN VIỆC XÂY DỰNG ĐỒ THỊ NEO4J!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())