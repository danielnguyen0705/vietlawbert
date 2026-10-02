"""
neo4j_batch_loader.py - Điều phối nạp dữ liệu phân cấp và 22 quan hệ pháp lý vào Neo4j từ Parquet.
Đồng bộ tuyệt đối cấu trúc Schema 3 tầng: (LawDocument) -> (Article) -> (Chunk) và các cạnh LEGAL_RELATION.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import pyarrow.parquet as pq

from configs.config import config
from configs.paths import RAW_SHARDS_DIR, PROCESSED_DIR
from database.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_Neo4jBatchLoader")

PARQUET_DIR = PROCESSED_DIR / "parquet"

CYPHER_DOC_UPSERT = """
UNWIND $docs AS row
MERGE (d:LawDocument {doc_id: row.doc_id})
ON CREATE SET 
    d.doc_number = row.doc_number,
    d.title = row.doc_title,
    d.issue_date = row.issue_date,
    d.effective_date = row.effective_date,
    d.status = row.status
ON MATCH SET
    d.doc_number = coalesce(d.doc_number, row.doc_number),
    d.title = coalesce(d.title, row.doc_title)
"""

CYPHER_CHUNK_UPSERT = """
UNWIND $chunks AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.hierarchy_path = row.hierarchy_path,
    c.macro_label = row.macro_label,
    c.content = row.content,
    c.doc_id = row.doc_id

WITH c, row
MERGE (d:LawDocument {doc_id: row.doc_id})
MERGE (d)-[:HAS_CHUNK]->(c)

WITH c, row, d
WHERE row.article_id IS NOT NULL AND row.article_id <> ''
MERGE (a:Article {article_id: row.article_id})
SET a.doc_id = row.doc_id,
    a.article_num = row.article_num,
    a.macro_label = row.macro_label
MERGE (d)-[:HAS_ARTICLE]->(a)
MERGE (a)-[:HAS_CHUNK]->(c)
"""

CYPHER_RELATION_UPSERT = """
UNWIND $relations AS rel
MERGE (src:LawDocument {doc_id: rel.source_id})
MERGE (tgt:LawDocument {doc_id: rel.target_id})
MERGE (src)-[r:LEGAL_RELATION {type: rel.type}]->(tgt)
ON CREATE SET r.created_at = timestamp()
"""


def load_parquet_stage(client: Neo4jClient, parquet_files: List[Path]):
    logger.info("--- [GIAI ĐOẠN 1] Nạp cấu trúc Văn bản & Chunks từ %d tệp Parquet ---", len(parquet_files))
    total_docs_seen = 0
    total_chunks_loaded = 0
    start_time = time.time()

    for idx, file_path in enumerate(parquet_files, 1):
        table = pq.read_table(str(file_path)).to_pydict()
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
                    "status": table.get("status", ["Còn hiệu lực"] * num_rows)[i],
                }

            chunks_list.append({
                "chunk_id": table["chunk_id"][i],
                "doc_id": doc_id,
                "article_id": table.get("article_id", [""] * num_rows)[i],
                "article_num": table.get("article_num", [""] * num_rows)[i],
                "hierarchy_path": table["hierarchy_path"][i],
                "macro_label": table.get("macro_label", ["CHUNG"] * num_rows)[i],
                "content": table["content"][i],
            })

        docs_payload = list(docs_map.values())
        client.execute_batch(CYPHER_DOC_UPSERT, docs_payload, batch_size=500)
        total_docs_seen += len(docs_payload)

        client.execute_batch(CYPHER_CHUNK_UPSERT, chunks_list, batch_size=1000)
        total_chunks_loaded += num_rows

        if idx % 10 == 0 or idx == len(parquet_files):
            elapsed = time.time() - start_time
            speed = total_chunks_loaded / max(1.0, elapsed)
            logger.info("[%d/%d] Đã nạp: %d Chunks (~%.1f chunks/s)", idx, len(parquet_files), total_chunks_loaded, speed)

    logger.info("✓ Hoàn tất Giai đoạn 1: %d Chunks liên kết vào %d Văn bản.", total_chunks_loaded, total_docs_seen)


def load_relations_stage(client: Neo4jClient, raw_shard_files: List[Path]):
    logger.info("--- [GIAI ĐOẠN 2] Trích xuất & Nạp quan hệ pháp lý từ %d Raw Shards ---", len(raw_shard_files))
    total_relations_loaded = 0

    for idx, shard_path in enumerate(raw_shard_files, 1):
        relations_batch = []
        opener = gzip.open(shard_path, "rt", encoding="utf-8", errors="replace") if shard_path.suffix == ".gz" else open(shard_path, "r", encoding="utf-8")
        with opener as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    src_id = str(record.get("doc_id") or record.get("item_id") or "").strip()
                    if not src_id:
                        continue
                    rels = record.get("relationships") or []
                    for r in rels:
                        tgt_id = str(r.get("target_id") or "").strip()
                        r_type = str(r.get("edge_type") or "CAN_CU_BAN_HANH").upper().replace(" ", "_")
                        if tgt_id and tgt_id != src_id:
                            relations_batch.append({"source_id": src_id, "target_id": tgt_id, "type": r_type})
                except Exception:
                    continue

        if relations_batch:
            unique_rels = list({(r["source_id"], r["target_id"], r["type"]): r for r in relations_batch}.values())
            client.execute_batch(CYPHER_RELATION_UPSERT, unique_rels, batch_size=1000)
            total_relations_loaded += len(unique_rels)

    logger.info("✓ Hoàn tất Giai đoạn 2: %d Cạnh quan hệ pháp lý đã được nạp.", total_relations_loaded)


def main():
    parquet_files = sorted(list(PARQUET_DIR.glob("*.parquet")))
    raw_shard_files = sorted([
        f for f in RAW_SHARDS_DIR.glob("*.jsonl*")
        if not f.name.endswith(".quarantine.jsonl") and not f.name.endswith(".corrupted")
    ])

    if not parquet_files:
        logger.error("Không tìm thấy tệp Parquet nào tại: %s. Hãy chạy ast_batch_processor trước.", PARQUET_DIR)
        return 1

    client = Neo4jClient(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
    try:
        load_parquet_stage(client, parquet_files)
        if raw_shard_files:
            load_relations_stage(client, raw_shard_files)
    finally:
        client.close()

    logger.info("✓ XÂY DỰNG MẠNG THÔNG TIN DỊ THỂ HIN TRÊN NEO4J HOÀN TẤT!")
    return 0


if __name__ == "__main__":
    sys.exit(main())