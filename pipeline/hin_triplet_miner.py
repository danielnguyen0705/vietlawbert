"""
hin_triplet_miner.py - Khai phá bộ ba mẫu khó đối lập (HIN-Guided Hard Negative Mining).
Sử dụng cấu trúc tô-pô Neo4j và kho tệp Parquet để sinh tập huấn luyện VietLawBERT-MRL.
"""

from __future__ import annotations

import os
import sys
import glob
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_TripletMiner")

PARQUET_DIR = Path("/mnt/data/vietlawbert_data/processed_parquet")
OUTPUT_DIR = Path("/mnt/data/vietlawbert_data/training_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vietlawbert")

TRIPLET_SCHEMA = pa.schema([
    ("anchor_id", pa.string()),
    ("anchor_text", pa.string()),
    ("positive_id", pa.string()),
    ("positive_text", pa.string()),
    ("negative_id", pa.string()),
    ("negative_text", pa.string()),
    ("mining_strategy", pa.string()),
])


def load_chunk_lookup(parquet_files: List[str], max_chunks: int = 400000) -> Dict[str, str]:
    """Tải bảng băm (chunk_id -> full_context_text) vào RAM (O(1) lookup)."""
    logger.info("Đang nạp bảng băm ngữ liệu Chunks từ Parquet vào bộ nhớ...")
    lookup = {}
    for f in parquet_files:
        table = pq.read_table(f, columns=["chunk_id", "full_context_text"]).to_pydict()
        for cid, text in zip(table["chunk_id"], table["full_context_text"]):
            if text and len(text) >= 50:
                lookup[cid] = text
        if len(lookup) >= max_chunks:
            break
    logger.info(f"✓ Đã nạp {len(lookup):,} Chunks hợp lệ vào RAM để đối soát nhanh.")
    return lookup


def mine_triplets_from_graph(
    driver,
    chunk_lookup: Dict[str, str],
    target_triplets: int = 100000,
) -> List[Dict[str, Any]]:
    logger.info(f"Bắt đầu khai phá {target_triplets:,} Triplets dựa trên mạng HIN Neo4j...")
    triplets = []
    seen_pairs: Set[tuple] = set()

    # Chiến lược 1: Cặp dương từ quan hệ pháp lý liên văn bản (Legal Relations)
    # Mẫu âm: Lấy điều luật khác thuộc cùng văn bản nguồn nhưng không liên quan
    cypher_inter_doc = """
    MATCH (d_src:LawDocument)-[r:LEGAL_RELATION]->(d_tgt:LawDocument)
    WHERE r.type IN ["CAN_CU", "SUA_DOI_BO_SUNG", "DAN_CHIEU", "HUONG_DAN"]
    MATCH (d_src)-[:HAS_CHUNK]->(c_anchor:Chunk)
    MATCH (d_tgt)-[:HAS_CHUNK]->(c_pos:Chunk)
    WITH d_src, c_anchor, c_pos
    LIMIT $limit
    MATCH (d_src)-[:HAS_CHUNK]->(c_neg:Chunk)
    WHERE c_neg.chunk_id <> c_anchor.chunk_id
    RETURN c_anchor.chunk_id AS anchor_id,
           c_pos.chunk_id AS positive_id,
           c_neg.chunk_id AS negative_id
    LIMIT $limit
    """

    # Chiến lược 2: Cặp dương nội bộ văn bản (Intra-Document Context)
    # Mẫu âm: Lấy điều luật từ văn bản khác cùng năm ban hành nhưng không có quan hệ dẫn chiếu
    cypher_intra_doc = """
    MATCH (d:LawDocument)-[:HAS_CHUNK]->(c1:Chunk)
    MATCH (d)-[:HAS_CHUNK]->(c2:Chunk)
    WHERE c1.chunk_id < c2.chunk_id
    WITH d, c1, c2
    LIMIT $limit
    MATCH (d_other:LawDocument)-[:HAS_CHUNK]->(c_neg:Chunk)
    WHERE d_other.doc_id <> d.doc_id 
      AND NOT (d)-[:LEGAL_RELATION]-(d_other)
    RETURN c1.chunk_id AS anchor_id,
           c2.chunk_id AS positive_id,
           c_neg.chunk_id AS negative_id
    LIMIT $limit
    """

    with driver.session() as session:
        # 1. Khai phá liên văn bản
        logger.info("1/2 Đang khai phá bộ ba liên văn bản (Inter-Document Citation Mining)...")
        results_inter = session.run(cypher_inter_doc, limit=target_triplets // 2)
        for row in results_inter:
            a_id, p_id, n_id = row["anchor_id"], row["positive_id"], row["negative_id"]
            if a_id in chunk_lookup and p_id in chunk_lookup and n_id in chunk_lookup:
                pair_key = (a_id, p_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    triplets.append({
                        "anchor_id": a_id,
                        "anchor_text": chunk_lookup[a_id],
                        "positive_id": p_id,
                        "positive_text": chunk_lookup[p_id],
                        "negative_id": n_id,
                        "negative_text": chunk_lookup[n_id],
                        "mining_strategy": "HIN_INTER_DOCUMENT",
                    })

        logger.info(f"-> Thu được {len(triplets):,} mẫu từ liên kết văn bản.")

        # 2. Khai phá nội văn bản
        remaining = target_triplets - len(triplets)
        if remaining > 0:
            logger.info("2/2 Đang khai phá bộ ba nội văn bản (Intra-Document Hard Negatives)...")
            results_intra = session.run(cypher_intra_doc, limit=remaining * 2)
            for row in results_intra:
                if len(triplets) >= target_triplets:
                    break
                a_id, p_id, n_id = row["anchor_id"], row["positive_id"], row["negative_id"]
                if a_id in chunk_lookup and p_id in chunk_lookup and n_id in chunk_lookup:
                    pair_key = (a_id, p_id)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        triplets.append({
                            "anchor_id": a_id,
                            "anchor_text": chunk_lookup[a_id],
                            "positive_id": p_id,
                            "positive_text": chunk_lookup[p_id],
                            "negative_id": n_id,
                            "negative_text": chunk_lookup[n_id],
                            "mining_strategy": "HIN_INTRA_DOCUMENT",
                        })

    logger.info(f"✓ Tổng số Triplets khai phá thành công: {len(triplets):,} mẫu.")
    return triplets


def main():
    parquet_files = sorted(glob.glob(str(PARQUET_DIR / "crawl_pages_*_*.parquet")))
    if not parquet_files:
        logger.error(f"Không tìm thấy Parquet files tại {PARQUET_DIR}")
        return 1

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        chunk_lookup = load_chunk_lookup(parquet_files)
        triplets_data = mine_triplets_from_graph(driver, chunk_lookup, target_triplets=100000)

        if not triplets_data:
            logger.warning("Không trích xuất được triplet nào. Vui lòng kiểm tra lại đồ thị.")
            return 1

        output_file = OUTPUT_DIR / "hin_triplets_100k.parquet"
        logger.info(f"Đang lưu tập huấn luyện vào: {output_file}...")
        table = pa.Table.from_pylist(triplets_data, schema=TRIPLET_SCHEMA)
        pq.write_table(table, output_file, compression="snappy")

        logger.info("=" * 65)
        logger.info(f"✓ HOÀN TẤT KHAI PHÁ MẪU KHÓ: {len(triplets_data):,} TRIPLETS ĐÃ ĐƯỢC ĐÓNG GÓI!")
        logger.info(f"Đường dẫn artifact: {output_file}")
        logger.info("=" * 65)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
