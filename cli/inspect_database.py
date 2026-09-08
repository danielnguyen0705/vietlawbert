"""
inspect_database.py - CLI kiểm tra trạng thái sức khỏe toàn diện của 5 dịch vụ CSDL.
Bao gồm: MongoDB, Neo4j (HIN), Qdrant (Dense 256d), Elasticsearch (Sparse) và Redis Cache.
"""

from __future__ import annotations

import sys
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
        return {"status": "ONLINE", "collections": collections, "total_docs": doc_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_neo4j() -> Dict[str, Any]:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        driver.verify_connectivity()
        with driver.session() as session:
            doc_nodes = session.run("MATCH (d:LawDocument) RETURN count(d) AS c").single()["c"]
            chunk_nodes = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r:LEGAL_RELATION]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        return {
            "status": "ONLINE",
            "law_documents": doc_nodes,
            "chunks": chunk_nodes,
            "legal_relations": rel_count,
        }
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_qdrant() -> Dict[str, Any]:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=3.0)
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        points_count = 0
        if config.QDRANT_COLLECTION_NAME in col_names:
            info = client.get_collection(config.QDRANT_COLLECTION_NAME)
            points_count = info.points_count
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
        es = Elasticsearch([config.ES_HOST], request_timeout=3)
        if not es.ping():
            return {"status": "OFFLINE", "error": "Ping thất bại"}
        doc_count = 0
        if es.indices.exists(index=config.ES_INDEX_NAME):
            res = es.count(index=config.ES_INDEX_NAME)
            doc_count = res.get("count", 0)
        return {"status": "ONLINE", "index": config.ES_INDEX_NAME, "docs_indexed": doc_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def inspect_redis() -> Dict[str, Any]:
    try:
        import redis
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, socket_timeout=2)
        r.ping()
        keys_count = r.dbsize()
        return {"status": "ONLINE", "cached_keys": keys_count}
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def main() -> int:
    logger.info("=== BẮT ĐẦU KIỂM TOÁN CƠ SỞ DỮ LIỆU LAI VIETLAWBERT (2026 STACK) ===")
    
    mongo_res = inspect_mongodb()
    neo4j_res = inspect_neo4j()
    qdrant_res = inspect_qdrant()
    es_res = inspect_elasticsearch()
    redis_res = inspect_redis()

    print("\n" + "=" * 70)
    print(f"{'DỊCH VỤ CƠ SỞ DỮ LIỆU':<25} | {'TRẠNG THÁI':<10} | {'CHI TIẾT CHỈ SỐ'}")
    print("=" * 70)
    print(f"{'MongoDB (Document)':<25} | {mongo_res['status']:<10} | {mongo_res.get('total_docs', 0):,} bản ghi")
    print(f"{'Neo4j (HIN Graph)':<25} | {neo4j_res['status']:<10} | Docs: {neo4j_res.get('law_documents', 0):,}, Chunks: {neo4j_res.get('chunks', 0):,}, Rels: {neo4j_res.get('legal_relations', 0):,}")
    print(f"{'Qdrant (Dense 256d)':<25} | {qdrant_res['status']:<10} | {qdrant_res.get('active_points', 0):,} vector points")
    print(f"{'Elasticsearch (Sparse)':<25} | {es_res['status']:<10} | {es_res.get('docs_indexed', 0):,} chunks indexed")
    print(f"{'Redis (Graph Cache)':<25} | {redis_res['status']:<10} | {redis_res.get('cached_keys', 0):,} cached embeddings")
    print("=" * 70 + "\n")

    services = [mongo_res, neo4j_res, qdrant_res, es_res, redis_res]
    offline_count = sum(1 for s in services if s["status"] == "OFFLINE")

    if offline_count > 0:
        logger.warning(f"Phát hiện {offline_count}/5 dịch vụ đang ngắt kết nối. Hãy kiểm tra 'docker compose ps'.")
        return 1

    logger.info("✓ Toàn bộ cụm CSDL phân tán hoạt động đồng bộ và sẵn sàng phục vụ.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)