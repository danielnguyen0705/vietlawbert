"""
inspect_database.py - CLI kiểm tra trạng thái sức khỏe toàn diện của 5 dịch vụ CSDL lai.
Bao gồm: MongoDB, Neo4j (HIN Graph 22 quan hệ), Qdrant (Dense 256d), Elasticsearch và Redis Cache.
Hỗ trợ xuất định dạng bảng terminal hoặc JSON phục vụ tích hợp CI/CD.
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from configs.config import config

logger = logging.getLogger("VietLawBERT_DBInspector")


def inspect_mongodb() -> Dict[str, Any]:
    try:
        from pymongo import MongoClient
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client[config.MONGO_DB_NAME]
        collections = db.list_collection_names()
        doc_count = sum(db[col].count_documents({}) for col in collections)
        client.close()
        return {"status": "ONLINE", "collections": collections, "total_docs": doc_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_neo4j() -> Dict[str, Any]:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            connection_acquisition_timeout=5.0
        )
        driver.verify_connectivity()
        with driver.session() as session:
            doc_nodes = session.run("MATCH (d:LawDocument) RETURN count(d) AS c").single()["c"]
            article_nodes = session.run("MATCH (a:Article) RETURN count(a) AS c").single()["c"]
            chunk_nodes = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            
            # Đếm chuẩn xác toàn bộ quan hệ pháp lý liên văn bản (loại trừ quan hệ cấu trúc cây nội bộ)
            rel_query = """
            MATCH ()-[r]->() 
            WHERE NOT type(r) IN ['HAS_ARTICLE', 'HAS_CHUNK', 'PART_OF']
            RETURN count(r) AS c
            """
            rel_count = session.run(rel_query).single()["c"]
        driver.close()
        return {
            "status": "ONLINE",
            "law_documents": doc_nodes,
            "articles": article_nodes,
            "chunks": chunk_nodes,
            "legal_relations": rel_count,
        }
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_qdrant() -> Dict[str, Any]:
    try:
        from qdrant_client import QdrantClient
        # check_compatibility=False ngăn chặn cảnh báo lệch version giữa client 1.19 và server 1.9
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=5.0, check_compatibility=False)
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        points_count = 0
        if config.QDRANT_COLLECTION_NAME in col_names:
            points_count = client.count(collection_name=config.QDRANT_COLLECTION_NAME, exact=True).count
        if hasattr(client, "close"):
            client.close()
        return {
            "status": "ONLINE",
            "collections": col_names,
            "active_points": points_count,
        }
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_elasticsearch() -> Dict[str, Any]:
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(hosts=[config.ES_HOST], request_timeout=5.0)
        if not es.ping():
            return {"status": "OFFLINE", "error": "Ping thất bại"}
        doc_count = 0
        if es.indices.exists(index=config.ES_INDEX_NAME):
            res = es.count(index=config.ES_INDEX_NAME)
            doc_count = res.get("count", 0)
        if hasattr(es, "close"):
            es.close()
        return {"status": "ONLINE", "index": config.ES_INDEX_NAME, "docs_indexed": doc_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_redis() -> Dict[str, Any]:
    try:
        import redis
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, socket_timeout=3.0)
        r.ping()
        keys_count = r.dbsize()
        r.close()
        return {"status": "ONLINE", "cached_keys": keys_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra trạng thái hệ thống lưu trữ phân tán VietLawBERT")
    parser.add_argument("--json", action="store_true", help="Xuất báo cáo dưới dạng JSON")
    parser.add_argument("--verbose", action="store_true", help="Hiển thị chi tiết cấu hình và lỗi kết nối")
    args = parser.parse_args()

    results = {
        "mongodb": inspect_mongodb(),
        "neo4j": inspect_neo4j(),
        "qdrant": inspect_qdrant(),
        "elasticsearch": inspect_elasticsearch(),
        "redis": inspect_redis(),
    }

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0 if all(s["status"] == "ONLINE" for s in results.values()) else 1

    print("\n" + "=" * 80)
    print(f"{'DỊCH VỤ CƠ SỞ DỮ LIỆU':<24} | {'TRẠNG THÁI':<10} | {'CHI TIẾT CHỈ SỐ HOẠT ĐỘNG'}")
    print("=" * 80)
    print(f"{'MongoDB (Document Store)':<24} | {results['mongodb']['status']:<10} | {results['mongodb'].get('total_docs', 0):,} bản ghi")
    
    neo = results["neo4j"]
    neo_detail = f"Docs: {neo.get('law_documents', 0):,}, Articles: {neo.get('articles', 0):,}, Chunks: {neo.get('chunks', 0):,}, HIN Rels: {neo.get('legal_relations', 0):,}" if neo["status"] == "ONLINE" else neo.get("error", "")
    print(f"{'Neo4j (HIN Graph)':<24} | {neo['status']:<10} | {neo_detail}")
    
    qdr = results["qdrant"]
    qdr_detail = f"{qdr.get('active_points', 0):,} vector points ({config.QDRANT_COLLECTION_NAME})" if qdr["status"] == "ONLINE" else qdr.get("error", "")
    print(f"{'Qdrant (Dense Vector)':<24} | {qdr['status']:<10} | {qdr_detail}")
    
    es = results["elasticsearch"]
    es_detail = f"{es.get('docs_indexed', 0):,} documents ({config.ES_INDEX_NAME})" if es["status"] == "ONLINE" else es.get("error", "")
    print(f"{'Elasticsearch (Sparse)':<24} | {es['status']:<10} | {es_detail}")
    
    red = results["redis"]
    red_detail = f"{red.get('cached_keys', 0):,} cached embeddings" if red["status"] == "ONLINE" else red.get("error", "")
    print(f"{'Redis (Graph Cache)':<24} | {red['status']:<10} | {red_detail}")
    print("=" * 80 + "\n")

    offline_svcs = [name for name, val in results.items() if val["status"] == "OFFLINE"]
    if offline_svcs:
        logger.warning("Phát hiện %d dịch vụ OFFLINE: %s. Hãy kiểm tra 'docker compose ps'.", len(offline_svcs), ", ".join(offline_svcs))
        return 1

    logger.info("✓ Toàn bộ cụm CSDL phân tán hoạt động đồng bộ và sẵn sàng phục vụ.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)