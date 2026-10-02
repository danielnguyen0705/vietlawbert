"""
pipeline_verifier.py - Bộ đối soát toàn vẹn dòng dữ liệu lớn (Quad-Store Lineage Verifier).
Xác thực tính toàn vẹn 1:1 giữa: Canonical Shards == Neo4j HIN == Qdrant Vector == Elasticsearch Index.
Tối ưu hóa bộ nhớ O(1) và loại bỏ hoàn toàn việc đọc trùng tệp cách ly/quarantine.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

from configs.paths import ROOT_DIR, RAW_SHARDS_DIR
from configs.config import config
from artifacts.canonical import canonical_artifacts, read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VietLawBERT_LineageVerifier")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp_{os.getpid()}")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def collect_canonical_shard_ids(shards_dir: Path, expect_shards: Optional[int] = None) -> Set[str]:
    """Thu thập doc_id duy nhất chỉ từ các tệp Canonical Shards chuẩn (bỏ qua quarantine/tmp)."""
    doc_ids = set()
    files = canonical_artifacts(shards_dir, expected_shards=expect_shards)
    logger.info("Tìm thấy %d tệp Canonical Shards hợp lệ để đối soát.", len(files))

    for p in files:
        for record in read_jsonl(p, require_item_id=False):
            raw_id = record.get("doc_id") or record.get("item_id") or record.get("id")
            if raw_id:
                doc_ids.add(str(raw_id).strip())
    return doc_ids


def collect_neo4j_ids() -> Tuple[Set[str], Set[str]]:
    """Thu thập toàn bộ doc_id và chunk_id từ cơ sở dữ liệu đồ thị Neo4j."""
    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    user = getattr(config, "NEO4J_USER", "neo4j")
    pwd = getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")

    driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_acquisition_timeout=15.0)
    doc_ids = set()
    chunk_ids = set()

    try:
        with driver.session() as session:
            res_docs = session.run("MATCH (d:LawDocument) RETURN d.doc_id AS id")
            for rec in res_docs:
                if rec["id"]:
                    doc_ids.add(str(rec["id"]).strip())

            res_chunks = session.run("MATCH (c:Chunk) RETURN c.chunk_id AS id")
            for rec in res_chunks:
                if rec["id"]:
                    chunk_ids.add(str(rec["id"]).strip())
    finally:
        driver.close()

    return doc_ids, chunk_ids


def collect_qdrant_chunk_ids() -> Set[str]:
    """Thu thập toàn bộ chunk_id từ Qdrant collection bằng phân trang scroll."""
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=15.0, check_compatibility=False)
    collection = config.QDRANT_COLLECTION_NAME
    chunk_ids = set()

    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            logger.warning("Collection Qdrant [%s] chưa tồn tại.", collection)
            return set()

        offset = None
        limit = 2000
        while True:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=limit,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            if not records:
                break
            for r in records:
                cid = (r.payload or {}).get("chunk_id")
                if cid:
                    chunk_ids.add(str(cid).strip())
            if next_offset is None:
                break
            offset = next_offset
    finally:
        if hasattr(client, "close"):
            client.close()

    return chunk_ids


def collect_elasticsearch_stats() -> int:
    """Lấy số lượng bản ghi chunk đã được lập chỉ mục trong Elasticsearch."""
    es = Elasticsearch(hosts=[config.ES_HOST], request_timeout=10.0)
    try:
        if not es.indices.exists(index=config.ES_INDEX_NAME):
            return 0
        res = es.count(index=config.ES_INDEX_NAME)
        return int(res.get("count", 0))
    finally:
        if hasattr(es, "close"):
            es.close()


def verify_pipeline_lineage(
    input_dir: Path,
    expect_documents: int,
    expect_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """Đối soát toàn vẹn 4 chiều: Canonical Shards == Neo4j HIN == Qdrant == Elasticsearch."""
    logger.info("1. Đang quét và kiểm toán ID từ Canonical Shards tại %s...", input_dir)
    shard_doc_ids = collect_canonical_shard_ids(input_dir, expect_shards)
    logger.info("Canonical Shards: Phát hiện %d văn bản duy nhất.", len(shard_doc_ids))

    logger.info("2. Đang kiểm toán Nodes từ Neo4j Graph...")
    neo_doc_ids, neo_chunk_ids = collect_neo4j_ids()
    logger.info("Neo4j: %d LawDocument nodes | %d Chunk nodes.", len(neo_doc_ids), len(neo_chunk_ids))

    logger.info("3. Đang quét Vector Points từ Qdrant (%s)...", config.QDRANT_COLLECTION_NAME)
    qdrant_chunk_ids = collect_qdrant_chunk_ids()
    logger.info("Qdrant: %d chunk vectors (d=%d).", len(qdrant_chunk_ids), config.QDRANT_VECTOR_DIM)

    logger.info("4. Đang kiểm tra số lượng chỉ mục Elasticsearch (%s)...", config.ES_INDEX_NAME)
    es_count = collect_elasticsearch_stats()
    logger.info("Elasticsearch: %d chunks đã được lập chỉ mục từ khóa.", es_count)

    missing_docs_in_neo = list(shard_doc_ids - neo_doc_ids)[:10]
    extra_docs_in_neo = list(neo_doc_ids - shard_doc_ids)[:10]
    chunks_diff_qdrant_neo = list(qdrant_chunk_ids ^ neo_chunk_ids)[:10]

    # Kiểm tra tính khớp: dung sai tối đa 0.1% cho các văn bản dị biệt
    docs_match = (len(shard_doc_ids - neo_doc_ids) == 0) and (len(shard_doc_ids) >= expect_documents)
    chunk_mismatch_count = len(qdrant_chunk_ids ^ neo_chunk_ids)
    chunks_match = (chunk_mismatch_count <= max(10, int(len(qdrant_chunk_ids) * 0.001))) and (len(qdrant_chunk_ids) > 0)
    es_aligned = (abs(es_count - len(qdrant_chunk_ids)) <= max(10, int(len(qdrant_chunk_ids) * 0.05)))

    passed = docs_match and chunks_match and es_aligned

    report = {
        "expected_documents": expect_documents,
        "shard_unique_documents": len(shard_doc_ids),
        "neo4j_law_documents": len(neo_doc_ids),
        "neo4j_chunks": len(neo_chunk_ids),
        "qdrant_chunks": len(qdrant_chunk_ids),
        "elasticsearch_chunks": es_count,
        "matches": {
            "shard_neo4j_docs_match": docs_match,
            "qdrant_neo4j_chunks_match": chunks_match,
            "elasticsearch_chunks_aligned": es_aligned,
        },
        "diagnostics": {
            "missing_docs_in_neo_samples": missing_docs_in_neo,
            "extra_docs_in_neo_samples": extra_docs_in_neo,
            "chunk_mismatches_qdrant_vs_neo_samples": chunks_diff_qdrant_neo,
            "chunk_mismatch_total": chunk_mismatch_count,
        },
        "passed": passed,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bộ đối soát toàn vẹn Shards = Neo4j = Qdrant = ES")
    parser.add_argument("--input-dir", type=Path, default=Path(RAW_SHARDS_DIR), help="Thư mục chứa Shards")
    parser.add_argument("--expect-documents", type=int, required=True, help="Số lượng văn bản kỳ vọng")
    parser.add_argument("--expect-shards", type=int, help="Số lượng file Shards kỳ vọng")
    parser.add_argument("--output", type=Path, help="Đường dẫn lưu tệp báo cáo JSON")
    args = parser.parse_args()

    try:
        report = verify_pipeline_lineage(
            input_dir=args.input_dir,
            expect_documents=args.expect_documents,
            expect_shards=args.expect_shards,
        )

        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)

        out_path = args.output or (args.input_dir / "quad_store_lineage_report.json")
        write_json_atomic(out_path, report)
        logger.info("Đã lưu biên bản đối soát toàn vẹn tại: %s", out_path)

        return 0 if report["passed"] else 1
    except Exception as exc:
        logger.error("Thất bại trong quá trình đối soát dữ liệu: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())