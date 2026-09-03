"""
database_inspection.py - Công cụ kiểm tra nhanh tình trạng và tính sẵn sàng của cụm CSDL.
Kiểm định toàn diện Milvus (Vector), Neo4j (Knowledge Graph) và MongoDB (Document Store).
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
from typing import Dict, Any

from configs.config import config

logger = logging.getLogger("VietLawBERT_DBInspect")


def inspect_milvus() -> Dict[str, Any]:
    """Kiểm tra chỉ mục HNSW, cấu trúc Schema và số lượng vector trong Milvus."""
    from pymilvus import MilvusClient

    uri = getattr(config, "MILVUS_URI", "http://localhost:19530")
    collection = getattr(config, "MILVUS_COLLECTION_NAME", "vietlawbert_chunks")
    result: Dict[str, Any] = {"status": "ERROR", "uri": uri, "collection": collection}

    try:
        client = MilvusClient(uri=uri)
        if client.has_collection(collection_name=collection):
            client.flush(collection_name=collection)
            stats = client.get_collection_stats(collection_name=collection)
            row_count = int(stats.get("row_count") or 0)
            
            # Trích xuất thông tin chỉ mục
            index_info = client.list_indexes(collection_name=collection)
            result.update({
                "status": "HEALTHY",
                "exists": True,
                "row_count": row_count,
                "indexes": index_info,
            })
        else:
            result.update({"status": "WARNING", "exists": False, "message": "Collection chưa được tạo"})
        client.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def inspect_neo4j() -> Dict[str, Any]:
    """Kiểm tra số lượng nút phân cấp, cạnh quan hệ ngữ nghĩa và ràng buộc duy nhất trên Neo4j."""
    from neo4j import GraphDatabase

    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    user = getattr(config, "NEO4J_USER", "neo4j")
    pwd = getattr(config, "NEO4J_PASSWORD", "vietlawbert")
    result: Dict[str, Any] = {"status": "ERROR", "uri": uri}

    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()

        with driver.session() as session:
            # Đếm số node theo Label
            node_counts = {}
            for record in session.run("MATCH (n) RETURN labels(n) AS labels, count(*) AS total"):
                lbl = ":".join(record["labels"]) if record["labels"] else "Unlabeled"
                node_counts[lbl] = record["total"]

            # Đếm số edge theo Type
            edge_counts = {}
            for record in session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS total"):
                edge_counts[record["rel_type"]] = record["total"]

            # Đếm tổng quan
            total_nodes = session.run("MATCH (n) RETURN count(n) AS total").single()["total"]
            total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS total").single()["total"]

            # Kiểm tra ràng buộc
            constraints = [rec["name"] for rec in session.run("SHOW CONSTRAINTS")]

            result.update({
                "status": "HEALTHY",
                "total_nodes": total_nodes,
                "total_relationships": total_edges,
                "node_distribution": node_counts,
                "edge_distribution": edge_counts,
                "constraints": constraints,
            })
        driver.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def inspect_mongodb() -> Dict[str, Any]:
    """Kiểm tra kết nối và số lượng tài liệu thô trong MongoDB."""
    from pymongo import MongoClient

    uri = getattr(config, "MONGO_URI", "mongodb://localhost:27017/")
    db_name = getattr(config, "MONGO_DB_NAME", "vietlawbert_db")
    result: Dict[str, Any] = {"status": "ERROR", "uri": uri, "database": db_name}

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client[db_name]
        collections = db.list_collection_names()
        coll_stats = {coll: db[coll].count_documents({}) for coll in collections}

        result.update({
            "status": "HEALTHY",
            "collections": collections,
            "document_counts": coll_stats,
        })
        client.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def inspect_database(json_format: bool = False, verbose: bool = False) -> int:
    """Điều phối kiểm tra toàn bộ 3 tầng lưu trữ và in báo cáo."""
    report = {
        "milvus": inspect_milvus(),
        "neo4j": inspect_neo4j(),
        "mongodb": inspect_mongodb(),
    }

    all_healthy = all(sec.get("status") == "HEALTHY" for sec in report.values())
    report["all_services_healthy"] = all_healthy

    if json_format:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all_healthy else 1

    print("\n=======================================================")
    print("      BÁO CÁO KIỂM TRA HẠ TẦNG CƠ SỞ DỮ LIỆU VIETLAWBERT")
    print("=======================================================")

    # In kết quả Milvus
    m = report["milvus"]
    m_icon = "✓" if m["status"] == "HEALTHY" else "✗"
    print(f"\n[{m_icon}] MILVUS VECTOR STORE: {m['status']}")
    if m["status"] == "HEALTHY":
        print(f"  * Collection: {m['collection']}")
        print(f"  * Tổng số Vector Chunks: {m['row_count']:,}")
    else:
        print(f"  * Chi tiết lỗi: {m.get('error') or m.get('message')}")

    # In kết quả Neo4j
    n = report["neo4j"]
    n_icon = "✓" if n["status"] == "HEALTHY" else "✗"
    print(f"\n[{n_icon}] NEO4J GRAPH STORE: {n['status']}")
    if n["status"] == "HEALTHY":
        print(f"  * Tổng số Nodes: {n['total_nodes']:,}")
        print(f"  * Tổng số Cạnh quan hệ: {n['total_relationships']:,}")
        if verbose:
            print("  * Phân phối Nodes:", json.dumps(n["node_distribution"], ensure_ascii=False))
            print("  * Phân phối Cạnh:", json.dumps(n["edge_distribution"], ensure_ascii=False))
    else:
        print(f"  * Chi tiết lỗi: {n.get('error')}")

    # In kết quả MongoDB
    mg = report["mongodb"]
    mg_icon = "✓" if mg["status"] == "HEALTHY" else "✗"
    print(f"\n[{mg_icon}] MONGODB DOCUMENT STORE: {mg['status']}")
    if mg["status"] == "HEALTHY":
        print(f"  * Database: {mg['database']}")
        print(f"  * Thống kê Collections:", json.dumps(mg["document_counts"], ensure_ascii=False))
    else:
        print(f"  * Chi tiết lỗi: {mg.get('error')}")

    print("\n-------------------------------------------------------")
    print(f"KẾT LUẬN CHUNG: {'HẠ TẦNG ĐÃ SẴN SÀNG 100%' if all_healthy else 'CẦN KHẮC PHỤC DỊCH VỤ LỖI'}")
    print("=======================================================\n")

    return 0 if all_healthy else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra trạng thái cụm CSDL VietLawBERT")
    parser.add_argument("--json", action="store_true", help="Xuất báo cáo định dạng JSON")
    parser.add_argument("--verbose", action="store_true", help="Hiển thị chi tiết thống kê schema và phân phối nhãn")
    args = parser.parse_args()
    return inspect_database(json_format=args.json, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())