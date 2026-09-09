"""
database_inspection.py - Công cụ kiểm tra nhanh tình trạng và tính sẵn sàng của cụm CSDL lai.
Kiểm định toàn diện: MongoDB, Neo4j (HIN), Qdrant (Dense 256d), Elasticsearch (Sparse) và Redis Cache.
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
from typing import Dict, Any

from configs.config import config

logger = logging.getLogger("VietLawBERT_DBInspect")


def inspect_mongodb() -> Dict[str, Any]:
    """Kiểm tra kết nối và số lượng tài liệu thô trong MongoDB."""
    from pymongo import MongoClient

    uri = getattr(config, "MONGO_URI", "mongodb://localhost:27017/")
    db_name = getattr(config, "MONGO_DB_NAME", "vietlawbert_db")
    result: Dict[str, Any] = {"status": "ERROR", "uri": uri, "database": db_name}

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client[db_name]
        collections = db.list_collection_names()
        coll_stats = {coll: db[coll].count_documents({}) for coll in collections}
        total_docs = sum(coll_stats.values())

        result.update({
            "status": "HEALTHY",
            "collections": collections,
            "document_counts": coll_stats,
            "total_documents": total_docs,
        })
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if client:
            client.close()

    return result


def inspect_neo4j() -> Dict[str, Any]:
    """Kiểm tra số lượng nút LawDocument/Chunk, 22 loại quan hệ và chỉ mục trên Neo4j."""
    from neo4j import GraphDatabase

    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    user = getattr(config, "NEO4J_USER", "neo4j")
    pwd = getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")
    result: Dict[str, Any] = {"status": "ERROR", "uri": uri}

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_acquisition_timeout=5.0)
        driver.verify_connectivity()

        with driver.session() as session:
            node_counts = {}
            for record in session.run("MATCH (n) RETURN labels(n) AS labels, count(*) AS total"):
                lbl = ":".join(record["labels"]) if record["labels"] else "Unlabeled"
                node_counts[lbl] = record["total"]

            edge_counts = {}
            for record in session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS total"):
                edge_counts[record["rel_type"]] = record["total"]

            total_nodes = session.run("MATCH (n) RETURN count(n) AS total").single()["total"]
            total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS total").single()["total"]
            constraints = [rec["name"] for rec in session.run("SHOW CONSTRAINTS")]

            result.update({
                "status": "HEALTHY",
                "total_nodes": total_nodes,
                "total_relationships": total_edges,
                "node_distribution": node_counts,
                "edge_distribution": edge_counts,
                "constraints": constraints,
            })
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if driver:
            driver.close()

    return result


def inspect_qdrant() -> Dict[str, Any]:
    """Kiểm tra Collection vector lát cắt Matryoshka d=256 và Payload Indices trên Qdrant."""
    from qdrant_client import QdrantClient

    host = getattr(config, "QDRANT_HOST", "localhost")
    port = getattr(config, "QDRANT_PORT", 6333)
    collection = getattr(config, "QDRANT_COLLECTION_NAME", "vietlawbert_chunks")
    result: Dict[str, Any] = {"status": "ERROR", "host": host, "port": port, "collection": collection}

    client = None
    try:
        client = QdrantClient(host=host, port=port, timeout=5.0)
        collections = [c.name for c in client.get_collections().collections]
        if collection in collections:
            info = client.get_collection(collection_name=collection)
            points_cnt = info.points_count
            if points_cnt is None:
                points_cnt = client.count(collection_name=collection, exact=True).count

            vectors_param = info.config.params.vectors
            dim = getattr(vectors_param, "size", config.QDRANT_VECTOR_DIM)
            dist = getattr(vectors_param, "distance", "Cosine")

            result.update({
                "status": "HEALTHY",
                "exists": True,
                "vector_dim": dim,
                "distance_metric": str(dist),
                "points_count": points_cnt,
                "payload_schema": list(info.payload_schema.keys()) if info.payload_schema else [],
            })
        else:
            result.update({"status": "WARNING", "exists": False, "message": f"Collection '{collection}' chưa được tạo"})
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if client and hasattr(client, "close"):
            client.close()

    return result


def inspect_elasticsearch() -> Dict[str, Any]:
    """Kiểm tra chỉ mục Sparse Retrieval và Custom Vietnamese Analyzer trên Elasticsearch."""
    from elasticsearch import Elasticsearch

    es_host = getattr(config, "ES_HOST", "http://localhost:9200")
    index_name = getattr(config, "ES_INDEX_NAME", "vietlaw_sparse_idx")
    result: Dict[str, Any] = {"status": "ERROR", "host": es_host, "index": index_name}

    es = None
    try:
        es = Elasticsearch([es_host], request_timeout=5)
        if not es.ping():
            result["error"] = "Ping Elasticsearch thất bại"
            return result

        exists = es.indices.exists(index=index_name)
        doc_count = 0
        if exists:
            res = es.count(index=index_name)
            doc_count = res.get("count", 0)

        result.update({
            "status": "HEALTHY",
            "exists": bool(exists),
            "docs_indexed": doc_count,
        })
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if es and hasattr(es, "close"):
            es.close()

    return result


def inspect_redis() -> Dict[str, Any]:
    """Kiểm tra In-Memory Cache lưu trữ Vector Đồ thị 128 chiều trên Redis."""
    import redis

    host = getattr(config, "REDIS_HOST", "localhost")
    port = getattr(config, "REDIS_PORT", 6379)
    db = getattr(config, "REDIS_DB", 0)
    result: Dict[str, Any] = {"status": "ERROR", "host": host, "port": port}

    r = None
    try:
        r = redis.Redis(host=host, port=port, db=db, socket_timeout=3)
        r.ping()
        result.update({
            "status": "HEALTHY",
            "cached_keys": r.dbsize(),
        })
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if r and hasattr(r, "close"):
            r.close()

    return result


def inspect_database(json_format: bool = False, verbose: bool = False) -> int:
    """Điều phối kiểm tra toàn bộ 5 tầng lưu trữ và xuất báo cáo chuẩn xác."""
    report = {
        "mongodb": inspect_mongodb(),
        "neo4j": inspect_neo4j(),
        "qdrant": inspect_qdrant(),
        "elasticsearch": inspect_elasticsearch(),
        "redis": inspect_redis(),
    }

    all_healthy = all(sec.get("status") == "HEALTHY" for sec in report.values())
    report["all_services_healthy"] = all_healthy

    if json_format:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all_healthy else 1

    print("\n" + "=" * 72)
    print("      BÁO CÁO KIỂM TRA HẠ TẦNG CƠ SỞ DỮ LIỆU LAI VIETLAWBERT (v3)")
    print("=" * 72)

    mg = report["mongodb"]
    print(f"\n[{'✓' if mg['status'] == 'HEALTHY' else '✗'}] MONGODB DOCUMENT STORE: {mg['status']}")
    if mg["status"] == "HEALTHY":
        print(f"  * Tổng số tài liệu: {mg.get('total_documents', 0):,}")
    else:
        print(f"  * Lỗi: {mg.get('error')}")

    n = report["neo4j"]
    print(f"\n[{'✓' if n['status'] == 'HEALTHY' else '✗'}] NEO4J HIN GRAPH STORE: {n['status']}")
    if n["status"] == "HEALTHY":
        print(f"  * Tổng số Nodes: {n['total_nodes']:,} | Cạnh quan hệ: {n['total_relationships']:,}")
        if verbose:
            print("  * Chi tiết Nodes:", json.dumps(n.get("node_distribution", {}), ensure_ascii=False))
            print("  * Chi tiết Cạnh:", json.dumps(n.get("edge_distribution", {}), ensure_ascii=False))
    else:
        print(f"  * Lỗi: {n.get('error')}")

    q = report["qdrant"]
    print(f"\n[{'✓' if q['status'] == 'HEALTHY' else '✗'}] QDRANT DENSE VECTOR STORE (d={q.get('vector_dim', 256)}): {q['status']}")
    if q["status"] == "HEALTHY":
        print(f"  * Collection: {q['collection']} | Active Points: {q.get('points_count', 0):,}")
    else:
        print(f"  * Lỗi: {q.get('error') or q.get('message')}")

    es = report["elasticsearch"]
    print(f"\n[{'✓' if es['status'] == 'HEALTHY' else '✗'}] ELASTICSEARCH SPARSE RETRIEVER: {es['status']}")
    if es["status"] == "HEALTHY":
        print(f"  * Index: {es['index']} | Chunks đã lập chỉ mục: {es.get('docs_indexed', 0):,}")
    else:
        print(f"  * Lỗi: {es.get('error')}")

    rd = report["redis"]
    print(f"\n[{'✓' if rd['status'] == 'HEALTHY' else '✗'}] REDIS COMPILE-TIME GRAPH CACHE: {rd['status']}")
    if rd["status"] == "HEALTHY":
        print(f"  * Số lượng Graph Vector 128d trong RAM Cache: {rd.get('cached_keys', 0):,}")
    else:
        print(f"  * Lỗi: {rd.get('error')}")

    print("\n" + "-" * 72)
    print(f"KẾT LUẬN: {'HẠ TẦNG HOÀN TOÀN SẴN SÀNG' if all_healthy else 'CẦN KHẮC PHỤC CÁC DỊCH VỤ OFFLINE'}")
    print("=" * 72 + "\n")

    return 0 if all_healthy else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra trạng thái cụm CSDL lai VietLawBERT")
    parser.add_argument("--json", action="store_true", help="Xuất báo cáo định dạng JSON")
    parser.add_argument("--verbose", action="store_true", help="Hiển thị chi tiết phân phối nhãn")
    args = parser.parse_args()
    return inspect_database(json_format=args.json, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())